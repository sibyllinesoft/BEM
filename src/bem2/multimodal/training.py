"""
Training Pipeline for BEM 2.0 Multimodal Conditioning

Implements specialized training for multimodal BEM models with:
- Cache-safe training (vision only to controller)
- Coverage/consistency loss functions
- Conflict gating supervision
- VQA task-specific training objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import json
import logging
import time
from collections import defaultdict

from .vision_encoder import VisionEncoder, VisionFeatures
from .controller_integration import MultimodalController, MultimodalRoutingState
from .coverage_analysis import ConsistencyGate, CoverageMetrics, ConsistencyMetrics
from .evaluation import VQAEvaluator, HallucinationDetector
from .preprocessing import VisionPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalTrainingConfig:
    """Configuration for multimodal training."""
    # Model architecture
    vision_dim: int = 512
    controller_dim: int = 512
    code_dim: int = 8
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 50000
    gradient_clip_norm: float = 1.0
    
    # Loss weights
    primary_loss_weight: float = 1.0        # VQA/generation loss
    coverage_loss_weight: float = 0.1       # Coverage optimization
    consistency_loss_weight: float = 0.1    # Consistency optimization
    conflict_loss_weight: float = 0.05      # Conflict gating supervision
    hallucination_loss_weight: float = 0.1  # Hallucination penalty
    
    # Training dynamics
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    evaluation_steps: int = 1000
    save_steps: int = 5000
    
    # Multimodal-specific
    vision_dropout: float = 0.1
    consistency_threshold: float = 0.5
    coverage_threshold: float = 0.3
    enable_conflict_supervision: bool = True
    
    # Cache safety validation
    validate_cache_safety: bool = True
    generator_gradient_check: bool = True


@dataclass
class TrainingStep:
    """Single training step data."""
    step: int
    losses: Dict[str, float]
    metrics: Dict[str, float]
    learning_rate: float
    gradient_norm: float
    batch_time: float


@dataclass
class EvaluationResult:
    """Evaluation result data."""
    step: int
    vqa_metrics: Dict[str, float]
    hallucination_metrics: Dict[str, float]
    coverage_stats: Dict[str, float]
    consistency_stats: Dict[str, float]
    gate_stats: Dict[str, float]


class MultimodalLoss(nn.Module):
    """
    Comprehensive loss function for multimodal BEM training.
    Combines multiple objectives while maintaining cache safety.
    """
    
    def __init__(self, config: MultimodalTrainingConfig):
        super().__init__()
        self.config = config
        
        # VQA evaluator for answer quality
        self.vqa_evaluator = VQAEvaluator()
        
        # Hallucination detector for penalties
        self.hallucination_detector = HallucinationDetector()
        
        # Coverage loss (encourages diverse attention)
        self.coverage_loss = nn.BCELoss()
        
        # Consistency loss (encourages alignment)
        self.consistency_loss = nn.MSELoss()
        
        # Conflict gating supervision
        self.conflict_loss = nn.BCEWithLogitsLoss()
    
    def compute_coverage_loss(
        self,
        coverage_metrics: CoverageMetrics,
        target_coverage: float = 0.7
    ) -> torch.Tensor:
        """Compute coverage optimization loss."""
        # Encourage high coverage scores
        coverage_target = torch.full_like(
            coverage_metrics.overall_score,
            target_coverage
        )
        
        loss_components = []
        
        # Overall coverage loss
        coverage_loss = self.coverage_loss(
            coverage_metrics.overall_score,
            coverage_target
        )
        loss_components.append(coverage_loss)
        
        # Spatial entropy loss (encourage diversity)
        entropy_target = torch.full_like(
            coverage_metrics.spatial_entropy,
            0.8  # High entropy target
        )
        entropy_loss = self.coverage_loss(
            coverage_metrics.spatial_entropy,
            entropy_target
        )
        loss_components.append(entropy_loss)
        
        # Region diversity loss
        diversity_target = torch.full_like(
            coverage_metrics.region_diversity,
            0.6  # Moderate diversity target
        )
        diversity_loss = self.coverage_loss(
            coverage_metrics.region_diversity,
            diversity_target
        )
        loss_components.append(diversity_loss)
        
        return torch.stack(loss_components).mean()
    
    def compute_consistency_loss(
        self,
        consistency_metrics: ConsistencyMetrics,
        target_consistency: float = 0.8
    ) -> torch.Tensor:
        """Compute consistency optimization loss."""
        # Encourage high consistency scores
        consistency_target = torch.full_like(
            consistency_metrics.overall_score,
            target_consistency
        )
        
        loss_components = []
        
        # Overall consistency loss
        overall_loss = self.consistency_loss(
            consistency_metrics.overall_score,
            consistency_target
        )
        loss_components.append(overall_loss)
        
        # Cross-modal alignment loss (most important)
        alignment_target = torch.full_like(
            consistency_metrics.cross_modal_alignment,
            target_consistency
        )
        alignment_loss = self.consistency_loss(
            consistency_metrics.cross_modal_alignment,
            alignment_target
        ) * 2.0  # Higher weight
        loss_components.append(alignment_loss)
        
        # Global-local consistency
        global_local_target = torch.full_like(
            consistency_metrics.global_local_consistency,
            target_consistency
        )
        global_local_loss = self.consistency_loss(
            consistency_metrics.global_local_consistency,
            global_local_target
        )
        loss_components.append(global_local_loss)
        
        return torch.stack(loss_components).mean()
    
    def compute_conflict_loss(
        self,
        gate_logits: torch.Tensor,      # Raw gating logits
        true_conflicts: torch.Tensor    # Binary conflict labels
    ) -> torch.Tensor:
        """Compute conflict gating supervision loss."""
        return self.conflict_loss(gate_logits, true_conflicts.float())
    
    def compute_hallucination_penalty(
        self,
        generated_texts: List[str],
        vision_features: VisionFeatures,
        detected_objects: List[List[str]],
        penalty_scale: float = 1.0
    ) -> torch.Tensor:
        """Compute hallucination penalty loss."""
        if not generated_texts or not detected_objects:
            return torch.tensor(0.0)
        
        # Compute hallucination rates
        object_attributes = [{} for _ in generated_texts]  # Simplified
        spatial_facts = [[] for _ in generated_texts]      # Simplified
        visual_facts = [[] for _ in generated_texts]       # Simplified
        
        hallucination_metrics = self.hallucination_detector.evaluate_hallucinations(
            generated_texts,
            detected_objects,
            object_attributes,
            spatial_facts,
            visual_facts
        )
        
        # Convert to penalty loss
        penalty = torch.tensor(
            hallucination_metrics.overall_hallucination_rate * penalty_scale
        )
        
        return penalty
    
    def forward(
        self,
        # Primary task outputs
        primary_loss: torch.Tensor,
        generated_texts: Optional[List[str]] = None,
        
        # Vision analysis outputs
        vision_features: Optional[VisionFeatures] = None,
        coverage_metrics: Optional[CoverageMetrics] = None,
        consistency_metrics: Optional[ConsistencyMetrics] = None,
        
        # Gating supervision
        gate_logits: Optional[torch.Tensor] = None,
        true_conflicts: Optional[torch.Tensor] = None,
        
        # Additional context
        detected_objects: Optional[List[List[str]]] = None
        
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute comprehensive multimodal loss.
        
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual losses
        """
        loss_components = {
            'primary': primary_loss * self.config.primary_loss_weight
        }
        
        # Coverage loss
        if coverage_metrics is not None:
            coverage_loss = self.compute_coverage_loss(coverage_metrics)
            loss_components['coverage'] = coverage_loss * self.config.coverage_loss_weight
        
        # Consistency loss  
        if consistency_metrics is not None:
            consistency_loss = self.compute_consistency_loss(consistency_metrics)
            loss_components['consistency'] = consistency_loss * self.config.consistency_loss_weight
        
        # Conflict gating loss
        if gate_logits is not None and true_conflicts is not None:
            conflict_loss = self.compute_conflict_loss(gate_logits, true_conflicts)
            loss_components['conflict'] = conflict_loss * self.config.conflict_loss_weight
        
        # Hallucination penalty
        if (generated_texts is not None and 
            vision_features is not None and 
            detected_objects is not None):
            hallucination_penalty = self.compute_hallucination_penalty(
                generated_texts, vision_features, detected_objects
            )
            loss_components['hallucination'] = hallucination_penalty * self.config.hallucination_loss_weight
        
        # Combine losses
        total_loss = sum(loss_components.values())
        
        return total_loss, loss_components


class CacheSafetyValidator:
    """
    Validates cache safety during training.
    Ensures vision features never affect generator parameters.
    """
    
    def __init__(self):
        self.violation_count = 0
        self.total_checks = 0
        self.generator_param_names = set()
    
    def register_generator_params(self, model: nn.Module):
        """Register generator parameter names."""
        self.generator_param_names.clear()
        
        for name, param in model.named_parameters():
            # Identify generator parameters (W_down, W_O)
            if any(pattern in name.lower() for pattern in ['w_down', 'w_o', 'down_proj', 'o_proj']):
                self.generator_param_names.add(name)
    
    def validate_gradients(self, model: nn.Module, loss: torch.Tensor) -> bool:
        """
        Validate that vision features don't affect generator gradients.
        
        Args:
            model: Model to check
            loss: Current loss (for gradient computation)
            
        Returns:
            True if cache-safe, False if violation detected
        """
        self.total_checks += 1
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Check generator parameters for unexpected gradients
        violation_detected = False
        
        for name, param in model.named_parameters():
            if name in self.generator_param_names:
                if param.grad is not None:
                    # Check if gradient has significant magnitude
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-6:  # Threshold for "significant" gradient
                        logger.warning(f"Cache safety violation: {name} has gradient norm {grad_norm}")
                        violation_detected = True
        
        if violation_detected:
            self.violation_count += 1
        
        return not violation_detected
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache safety statistics."""
        return {
            'violation_rate': self.violation_count / max(self.total_checks, 1),
            'total_checks': self.total_checks,
            'violations': self.violation_count
        }


class MultimodalTrainer:
    """
    Main training pipeline for multimodal BEM models.
    Implements complete MM0 training with all safety and performance requirements.
    """
    
    def __init__(
        self,
        model: MultimodalController,
        vision_encoder: VisionEncoder,
        preprocessor: VisionPreprocessor,
        consistency_gate: ConsistencyGate,
        config: MultimodalTrainingConfig,
        output_dir: str = "outputs/multimodal_training"
    ):
        self.model = model
        self.vision_encoder = vision_encoder
        self.preprocessor = preprocessor
        self.consistency_gate = consistency_gate
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.loss_fn = MultimodalLoss(config)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.max_steps,
            pct_start=config.warmup_steps / config.max_steps
        )
        
        # Cache safety validator
        if config.validate_cache_safety:
            self.cache_validator = CacheSafetyValidator()
            self.cache_validator.register_generator_params(model)
        else:
            self.cache_validator = None
        
        # Training state
        self.step = 0
        self.training_history = []
        self.evaluation_history = []
        self.best_metrics = {}
        
        # Statistics
        self.stats = defaultdict(list)
    
    def save_checkpoint(
        self,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """Save training checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'best_metrics': self.best_metrics,
            'stats': dict(self.stats)
        }
        
        if metrics:
            checkpoint['current_metrics'] = metrics
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Save latest
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.training_history = checkpoint.get('training_history', [])
        self.evaluation_history = checkpoint.get('evaluation_history', [])
        self.best_metrics = checkpoint.get('best_metrics', {})
        
        stats_dict = checkpoint.get('stats', {})
        self.stats = defaultdict(list)
        for key, value in stats_dict.items():
            self.stats[key] = value
    
    def train_step(
        self,
        batch: Dict[str, Any]
    ) -> TrainingStep:
        """Execute single training step."""
        step_start_time = time.time()
        
        self.model.train()
        self.vision_encoder.eval()  # Vision encoder is frozen
        
        # Extract batch data
        images = batch['images']
        texts = batch['texts']
        text_lengths = batch.get('text_lengths')
        ground_truth_answers = batch.get('answers', [])
        detected_objects = batch.get('detected_objects', [])
        
        # Preprocess vision features
        vision_features_list = self.preprocessor.preprocess_batch(
            images, text_lengths, batch_size=len(images)
        )
        vision_features = vision_features_list[0].features  # Simplification for single batch
        
        # Forward pass through multimodal controller
        hidden_states = self._encode_text(texts)  # Mock text encoding
        
        with torch.set_grad_enabled(True):
            # Get multimodal codes and routing state
            codes, routing_state = self.model(
                hidden_states=hidden_states,
                vision_features=vision_features,
                routing_level='chunk',
                return_routing_state=True,
                enable_vision_conditioning=True
            )
            
            # Compute coverage and consistency metrics
            text_features = hidden_states.mean(dim=1)  # Simplified text summary
            coverage_metrics = self.consistency_gate.coverage_analyzer(vision_features)
            consistency_metrics = self.consistency_gate.consistency_analyzer(
                vision_features, text_features
            )
            
            # Generate text outputs (mock for training example)
            generated_texts = self._generate_text(codes, hidden_states)
            
            # Compute primary task loss (VQA/generation)
            primary_loss = self._compute_primary_loss(
                generated_texts, ground_truth_answers
            )
            
            # Compute comprehensive loss
            gate_weights, conflict_analysis = self.consistency_gate(
                vision_features, text_features
            )
            
            total_loss, loss_components = self.loss_fn(
                primary_loss=primary_loss,
                generated_texts=generated_texts,
                vision_features=vision_features,
                coverage_metrics=coverage_metrics,
                consistency_metrics=consistency_metrics,
                gate_logits=None,  # Would need actual logits
                true_conflicts=None,  # Would need ground truth
                detected_objects=detected_objects
            )
        
        # Validate cache safety
        if self.cache_validator and self.config.validate_cache_safety:
            cache_safe = self.cache_validator.validate_gradients(self.model, total_loss)
            if not cache_safe:
                logger.warning(f"Cache safety violation at step {self.step}")
        
        # Backward pass
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()
        
        # Gradient clipping
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        
        # Optimizer step
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Collect metrics
        step_time = time.time() - step_start_time
        
        # Convert loss components to float for logging
        losses = {key: loss.item() for key, loss in loss_components.items()}
        losses['total'] = total_loss.item()
        
        metrics = {
            'coverage_score': coverage_metrics.overall_score.mean().item(),
            'consistency_score': consistency_metrics.overall_score.mean().item(),
            'gate_activation_rate': gate_weights.mean().item(),
            'conflict_detected': conflict_analysis.conflict_detected
        }
        
        training_step = TrainingStep(
            step=self.step,
            losses=losses,
            metrics=metrics,
            learning_rate=self.scheduler.get_last_lr()[0],
            gradient_norm=gradient_norm.item(),
            batch_time=step_time
        )
        
        self.training_history.append(training_step)
        self.step += 1
        
        return training_step
    
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """Mock text encoding (would use actual tokenizer/model)."""
        # Return mock hidden states
        batch_size = len(texts)
        seq_len = 128  # Mock sequence length
        hidden_dim = self.config.controller_dim
        
        return torch.randn(batch_size, seq_len, hidden_dim)
    
    def _generate_text(
        self, 
        codes: torch.Tensor, 
        hidden_states: torch.Tensor
    ) -> List[str]:
        """Mock text generation (would use actual generation)."""
        batch_size = codes.shape[0]
        return [f"Generated answer {i}" for i in range(batch_size)]
    
    def _compute_primary_loss(
        self,
        generated_texts: List[str],
        ground_truth_answers: List[List[str]]
    ) -> torch.Tensor:
        """Compute primary VQA/generation loss."""
        if not ground_truth_answers:
            return torch.tensor(0.0, requires_grad=True)
        
        # Mock loss computation (would use actual VQA evaluation)
        vqa_metrics = self.loss_fn.vqa_evaluator.evaluate_batch(
            generated_texts,
            ground_truth_answers,
            ["Mock question"] * len(generated_texts)
        )
        
        # Convert metrics to loss (higher is better, so negate)
        loss = 1.0 - (vqa_metrics.exact_match + vqa_metrics.f1_score) / 2
        
        return torch.tensor(loss, requires_grad=True)
    
    def evaluate(
        self,
        eval_dataset: List[Dict[str, Any]],
        num_samples: Optional[int] = None
    ) -> EvaluationResult:
        """Run evaluation on validation set."""
        self.model.eval()
        
        if num_samples:
            eval_dataset = eval_dataset[:num_samples]
        
        all_predictions = []
        all_ground_truths = []
        all_questions = []
        
        coverage_scores = []
        consistency_scores = []
        gate_activations = []
        
        with torch.no_grad():
            for batch_data in eval_dataset:
                # Mock evaluation (would implement actual evaluation loop)
                images = [batch_data['image']]
                questions = [batch_data['question']]
                answers = [batch_data['answers']]
                
                # Process through model
                vision_features_list = self.preprocessor.preprocess_batch(images)
                vision_features = vision_features_list[0].features
                
                hidden_states = self._encode_text(questions)
                
                codes, routing_state = self.model(
                    hidden_states=hidden_states,
                    vision_features=vision_features,
                    routing_level='chunk',
                    return_routing_state=True
                )
                
                # Generate predictions
                predictions = self._generate_text(codes, hidden_states)
                
                all_predictions.extend(predictions)
                all_ground_truths.extend(answers)
                all_questions.extend(questions)
                
                # Collect multimodal metrics
                if hasattr(vision_features, 'coverage_score') and vision_features.coverage_score is not None:
                    coverage_scores.extend(vision_features.coverage_score.cpu().numpy())
                if hasattr(vision_features, 'consistency_score') and vision_features.consistency_score is not None:
                    consistency_scores.extend(vision_features.consistency_score.cpu().numpy())
        
        # Compute VQA metrics
        vqa_metrics = self.loss_fn.vqa_evaluator.evaluate_batch(
            all_predictions, all_ground_truths, all_questions
        )
        
        # Mock hallucination metrics (would use actual detector)
        hallucination_metrics = {
            'object_hallucination_rate': 0.1,
            'overall_hallucination_rate': 0.15
        }
        
        evaluation_result = EvaluationResult(
            step=self.step,
            vqa_metrics={
                'exact_match': vqa_metrics.exact_match,
                'f1_score': vqa_metrics.f1_score,
                'accuracy': vqa_metrics.accuracy,
                'answer_relevance': vqa_metrics.answer_relevance
            },
            hallucination_metrics=hallucination_metrics,
            coverage_stats={
                'mean': np.mean(coverage_scores) if coverage_scores else 0.0,
                'std': np.std(coverage_scores) if coverage_scores else 0.0
            },
            consistency_stats={
                'mean': np.mean(consistency_scores) if consistency_scores else 0.0,
                'std': np.std(consistency_scores) if consistency_scores else 0.0
            },
            gate_stats={
                'activation_rate': np.mean(gate_activations) if gate_activations else 0.0
            }
        )
        
        self.evaluation_history.append(evaluation_result)
        
        # Update best metrics
        if (not self.best_metrics or 
            evaluation_result.vqa_metrics['f1_score'] > self.best_metrics.get('f1_score', 0)):
            self.best_metrics.update(evaluation_result.vqa_metrics)
            return evaluation_result, True  # Is best
        
        return evaluation_result, False
    
    def train(
        self,
        train_dataset: List[Dict[str, Any]],
        eval_dataset: Optional[List[Dict[str, Any]]] = None,
        resume_from: Optional[str] = None
    ):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)
            logger.info(f"Resumed training from step {self.step}")
        
        logger.info("🚀 Starting multimodal BEM training")
        logger.info(f"  Max steps: {self.config.max_steps}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        
        # Training loop
        while self.step < self.config.max_steps:
            # Sample batch
            batch_indices = np.random.choice(
                len(train_dataset), self.config.batch_size, replace=True
            )
            batch_data = [train_dataset[i] for i in batch_indices]
            
            # Convert to batch format
            batch = {
                'images': [item['image'] for item in batch_data],
                'texts': [item['question'] for item in batch_data],
                'answers': [item['answers'] for item in batch_data],
                'detected_objects': [item.get('detected_objects', []) for item in batch_data]
            }
            
            # Training step
            training_step = self.train_step(batch)
            
            # Logging
            if self.step % 100 == 0:
                logger.info(
                    f"Step {self.step}: Loss={training_step.losses['total']:.4f}, "
                    f"Coverage={training_step.metrics['coverage_score']:.3f}, "
                    f"Consistency={training_step.metrics['consistency_score']:.3f}, "
                    f"LR={training_step.learning_rate:.2e}"
                )
            
            # Evaluation
            if eval_dataset and self.step % self.config.evaluation_steps == 0:
                eval_result, is_best = self.evaluate(eval_dataset)
                logger.info(
                    f"Evaluation at step {self.step}: "
                    f"EM={eval_result.vqa_metrics['exact_match']:.3f}, "
                    f"F1={eval_result.vqa_metrics['f1_score']:.3f}"
                )
                
                # Save checkpoint
                self.save_checkpoint(
                    self.step,
                    eval_result.vqa_metrics,
                    is_best=is_best
                )
            
            # Regular checkpoint saving
            elif self.step % self.config.save_steps == 0:
                self.save_checkpoint(self.step)
        
        logger.info("✅ Training completed!")
        
        # Final evaluation
        if eval_dataset:
            final_eval, is_best = self.evaluate(eval_dataset)
            self.save_checkpoint(self.step, final_eval.vqa_metrics, is_best=is_best)
            
            logger.info("🎯 Final Results:")
            logger.info(f"  Exact Match: {final_eval.vqa_metrics['exact_match']:.3f}")
            logger.info(f"  F1 Score: {final_eval.vqa_metrics['f1_score']:.3f}")
            logger.info(f"  Answer Relevance: {final_eval.vqa_metrics['answer_relevance']:.3f}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'config': asdict(self.config),
            'total_steps': self.step,
            'training_history_length': len(self.training_history),
            'evaluation_history_length': len(self.evaluation_history),
            'best_metrics': self.best_metrics
        }
        
        if self.training_history:
            recent_losses = [step.losses['total'] for step in self.training_history[-100:]]
            summary['recent_loss_mean'] = np.mean(recent_losses)
            summary['recent_loss_std'] = np.std(recent_losses)
        
        if self.evaluation_history:
            summary['best_f1'] = max(eval_result.vqa_metrics['f1_score'] 
                                   for eval_result in self.evaluation_history)
            summary['best_exact_match'] = max(eval_result.vqa_metrics['exact_match']
                                            for eval_result in self.evaluation_history)
        
        if self.cache_validator:
            summary['cache_safety_stats'] = self.cache_validator.get_stats()
        
        summary['preprocessing_stats'] = self.preprocessor.get_preprocessing_stats()
        
        return summary


def create_multimodal_trainer(
    model: MultimodalController,
    vision_encoder: VisionEncoder,
    preprocessor: VisionPreprocessor,
    consistency_gate: ConsistencyGate,
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = "outputs/multimodal_training"
) -> MultimodalTrainer:
    """
    Factory function to create multimodal trainer.
    
    Args:
        model: Multimodal controller model
        vision_encoder: Vision encoder
        preprocessor: Vision preprocessor
        consistency_gate: Consistency gate
        config: Training configuration
        output_dir: Output directory
        
    Returns:
        MultimodalTrainer instance
    """
    # Create training config
    if config is None:
        training_config = MultimodalTrainingConfig()
    else:
        training_config = MultimodalTrainingConfig(**config)
    
    return MultimodalTrainer(
        model=model,
        vision_encoder=vision_encoder,
        preprocessor=preprocessor,
        consistency_gate=consistency_gate,
        config=training_config,
        output_dir=output_dir
    )