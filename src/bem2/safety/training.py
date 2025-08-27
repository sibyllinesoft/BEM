"""
Safety Training Pipeline with Orthogonality Constraints

Integrates all safety components into a unified training pipeline that:
- Trains safety basis with orthogonality constraints
- Optimizes constitutional scoring alignment
- Applies Lagrangian constraints for violation rate ≤ ε
- Maintains performance while improving safety

Key Features:
- Multi-objective optimization (safety + utility + orthogonality)
- Adaptive constraint satisfaction
- Performance monitoring and rollback
- Curriculum learning for safety training
- Real-time violation monitoring during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import logging
import wandb
from pathlib import Path
import json
import time
from collections import defaultdict, deque

from .safety_basis import OrthogonalSafetyBasis, SafetyBasisConfig
from .constitutional_scorer import ConstitutionalScorer, ValueModelConfig
from .lagrangian_optimizer import LagrangianOptimizer, ConstraintConfig
from .safety_controller import SafetyController, ControlConfig
from .violation_detector import ViolationDetector, ViolationConfig

logger = logging.getLogger(__name__)


@dataclass
class SafetyTrainingConfig:
    """Configuration for safety training pipeline."""
    
    # Training parameters
    num_epochs: int = 10                    # Number of training epochs
    batch_size: int = 32                    # Training batch size
    learning_rate: float = 1e-4             # Base learning rate
    weight_decay: float = 0.01              # Weight decay for regularization
    
    # Safety-specific training
    safety_warmup_epochs: int = 2           # Epochs to warm up safety components
    orthogonality_warmup_epochs: int = 3    # Epochs to establish orthogonality
    constraint_warmup_epochs: int = 1       # Epochs before activating constraints
    
    # Multi-objective weights
    utility_weight: float = 1.0             # Weight for utility/performance loss
    safety_weight: float = 1.0              # Weight for safety/constitutional loss  
    orthogonality_weight: float = 0.5       # Weight for orthogonality penalty
    violation_constraint_weight: float = 2.0  # Weight for violation constraints
    
    # Performance monitoring
    performance_check_frequency: int = 100  # Steps between performance checks
    max_performance_drop: float = 0.01      # Maximum allowed performance drop (1%)
    rollback_threshold: int = 3             # Consecutive bad performance checks for rollback
    
    # Evaluation during training
    eval_frequency: int = 500               # Steps between evaluations
    eval_samples: int = 100                 # Number of samples for evaluation
    track_violation_reduction: bool = True   # Track violation reduction progress
    
    # Curriculum learning
    enable_curriculum: bool = True          # Enable curriculum learning
    curriculum_stages: int = 4              # Number of curriculum stages
    stage_warmup_steps: int = 200          # Steps to warm up each stage
    
    # Checkpointing
    save_frequency: int = 1000             # Steps between checkpoints
    keep_best_checkpoint: bool = True       # Keep best performing checkpoint
    checkpoint_dir: str = "checkpoints"    # Directory for checkpoints
    
    # Monitoring and logging
    log_frequency: int = 10                # Steps between logging
    track_gradients: bool = True           # Track gradient norms
    track_activations: bool = True         # Track activation statistics
    use_wandb: bool = True                 # Use Weights & Biases logging


class SafetyTrainingPipeline:
    """
    Comprehensive safety training pipeline.
    
    Orchestrates training of all safety components with proper constraint
    satisfaction, performance monitoring, and orthogonality preservation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SafetyTrainingConfig,
        safety_basis_config: SafetyBasisConfig,
        value_model_config: ValueModelConfig,
        constraint_config: ConstraintConfig,
        control_config: ControlConfig,
        violation_config: ViolationConfig
    ):
        self.config = config
        self.model = model
        
        # Initialize safety components
        self.safety_basis = OrthogonalSafetyBasis(safety_basis_config)
        self.constitutional_scorer = ConstitutionalScorer(value_model_config)
        self.safety_controller = SafetyController(control_config)
        self.violation_detector = ViolationDetector(violation_config)
        
        # Initialize Lagrangian optimizer
        self.lagrangian_optimizer = LagrangianOptimizer(
            model=self.model,
            config=constraint_config,
            utility_loss_fn=self._compute_utility_loss,
            violation_loss_fn=self._compute_violation_loss,
            orthogonality_loss_fn=self.safety_basis.compute_orthogonality_penalty
        )
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_performance = float('inf')
        self.best_violation_rate = float('inf')
        self.performance_degradation_count = 0
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.violation_history = deque(maxlen=1000)
        self.safety_metrics_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Curriculum learning
        if config.enable_curriculum:
            self.curriculum_stage = 0
            self.curriculum_progress = 0
            self.curriculum_scheduler = self._create_curriculum_scheduler()
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Best model state
        self.best_model_state = None
        
        # Initialize logging
        if config.use_wandb:
            wandb.init(project="bem2-safety-training", config=config.__dict__)
        
        logger.info("Initialized safety training pipeline")
        
    def train(
        self,
        train_dataloader,
        val_dataloader,
        safety_evaluation_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute complete safety training pipeline.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader  
            safety_evaluation_fn: Optional safety evaluation function
            
        Returns:
            training_results: Comprehensive training results and metrics
        """
        
        logger.info(f"Starting safety training for {self.config.num_epochs} epochs")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Epoch-specific setup
            self._setup_epoch_training()
            
            # Training phase
            epoch_metrics = self._train_epoch(train_dataloader)
            
            # Validation phase
            if val_dataloader:
                val_metrics = self._validate_epoch(val_dataloader, safety_evaluation_fn)
                epoch_metrics.update(val_metrics)
            
            # Curriculum progression
            if self.config.enable_curriculum:
                self._progress_curriculum()
            
            # Checkpoint saving
            self._save_epoch_checkpoint(epoch_metrics)
            
            # Log epoch results
            self._log_epoch_metrics(epoch, epoch_metrics)
            
            # Check for early stopping
            if self._should_early_stop(epoch_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Final evaluation and results
        final_results = self._finalize_training()
        
        logger.info("Safety training completed")
        return final_results
    
    def _train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch with safety constraints."""
        
        self.model.train()
        self.safety_basis.train()
        self.constitutional_scorer.train()
        
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader):
            self.current_step += 1
            
            # Forward pass with safety integration
            step_metrics = self._train_step(batch, batch_idx)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                epoch_metrics[key].append(value)
            
            # Periodic evaluations
            if self.current_step % self.config.eval_frequency == 0:
                self._periodic_evaluation()
            
            # Performance monitoring
            if self.current_step % self.config.performance_check_frequency == 0:
                self._check_performance_degradation()
            
            # Checkpointing
            if self.current_step % self.config.save_frequency == 0:
                self._save_step_checkpoint()
            
            # Logging
            if self.current_step % self.config.log_frequency == 0:
                self._log_step_metrics(step_metrics)
        
        # Average metrics across epoch
        averaged_metrics = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }
        
        return averaged_metrics
    
    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Execute single training step with safety integration."""
        
        # Get model inputs
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        
        # Forward pass through base model
        base_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = base_outputs.hidden_states
        
        # Constitutional scoring
        constitutional_scores = self.constitutional_scorer(
            input_ids, attention_mask
        )
        
        # Safety control level
        safety_level = self.safety_controller(
            hidden_states[-1],  # Use last layer hidden states
            constitutional_scores,
            domain="general"  # Could be dynamic based on batch
        )
        
        # Apply safety basis transformations per layer
        safety_metrics = {}
        transformed_hidden_states = []
        
        for layer_idx, layer_hidden in enumerate(hidden_states):
            if layer_idx == 0:  # Skip embedding layer
                transformed_hidden_states.append(layer_hidden)
                continue
                
            # Apply safety basis transformation
            transformed, layer_metrics = self.safety_basis(
                layer_hidden,
                layer_idx - 1,  # Adjust for embedding layer
                constitutional_scores,
                safety_level
            )
            
            transformed_hidden_states.append(transformed)
            
            # Accumulate safety metrics
            for key, value in layer_metrics.items():
                if key not in safety_metrics:
                    safety_metrics[key] = []
                safety_metrics[key].append(value.item())
        
        # Recompute model outputs with safety transformations
        safety_outputs = self._recompute_model_outputs(
            transformed_hidden_states[-1],  # Use final transformed states
            attention_mask
        )
        
        # Prepare batch with safety scores for Lagrangian optimization
        safety_batch = batch.copy()
        safety_batch['safety_outputs'] = safety_outputs
        safety_batch['constitutional_scores'] = constitutional_scores
        
        # Execute constrained optimization step
        optimization_metrics = self.lagrangian_optimizer.step(
            safety_batch,
            constitutional_scores,
            return_metrics=True
        )
        
        # Real-time violation detection
        violation_info = self.violation_detector.real_time_violation_screening(
            transformed_hidden_states[-1],
            attention_mask
        )
        
        # Compile step metrics
        step_metrics = {
            'utility_loss': optimization_metrics.get('utility_loss', 0.0),
            'violation_loss': optimization_metrics.get('violation_loss', 0.0),
            'violation_rate': optimization_metrics.get('violation_rate', 0.0),
            'lambda_value': optimization_metrics.get('lambda_value', 0.0),
            'constitutional_score_mean': constitutional_scores.mean().item(),
            'safety_level': safety_level,
            'orthogonality_error': sum(safety_metrics.get('orthogonality_error', [0])) / len(safety_metrics.get('orthogonality_error', [1])),
            'violations_detected_realtime': violation_info.get('violations_detected', False),
            'max_violation_score': violation_info.get('max_violation_score', 0.0)
        }
        
        # Update tracking
        self._update_training_tracking(step_metrics)
        
        return step_metrics
    
    def _compute_utility_loss(self, model_output, batch) -> torch.Tensor:
        """Compute utility/performance loss."""
        # Standard language modeling loss
        if 'labels' in batch:
            labels = batch['labels']
            if hasattr(model_output, 'logits'):
                logits = model_output.logits
            else:
                logits = model_output
            
            # Compute cross-entropy loss
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss
        else:
            # Return dummy loss if no labels
            return torch.tensor(0.0, requires_grad=True)
    
    def _compute_violation_loss(self, model_output, safety_scores) -> torch.Tensor:
        """Compute safety/violation loss."""
        # Convert safety scores to violation loss
        # Higher safety scores should result in lower violation loss
        violation_loss = (1.0 - safety_scores).mean()
        return violation_loss
    
    def _recompute_model_outputs(self, transformed_hidden_states, attention_mask):
        """Recompute model outputs using transformed hidden states."""
        # In a real implementation, would need to properly propagate
        # transformed states through the model's final layers
        
        # Placeholder - return a simple linear transformation
        batch_size, seq_len, hidden_dim = transformed_hidden_states.shape
        output_logits = torch.randn(batch_size, seq_len, 50257)  # Assume GPT-like vocab size
        
        return type('MockOutput', (), {'logits': output_logits})()
    
    def _setup_epoch_training(self):
        """Setup training configuration for current epoch."""
        
        # Warmup safety components
        if self.current_epoch < self.config.safety_warmup_epochs:
            # Freeze safety basis during warmup
            self.safety_basis.freeze_safety_bases(freeze=True)
            logger.info(f"Epoch {self.current_epoch}: Safety basis frozen for warmup")
        else:
            # Unfreeze safety basis
            self.safety_basis.freeze_safety_bases(freeze=False)
        
        # Orthogonality constraint warmup
        if self.current_epoch < self.config.orthogonality_warmup_epochs:
            # Reduce orthogonality penalty during warmup
            original_penalty = self.safety_basis.config.orthogonal_penalty
            warmup_penalty = original_penalty * 0.1
            self.safety_basis.config.orthogonal_penalty = warmup_penalty
            logger.info(f"Epoch {self.current_epoch}: Reduced orthogonality penalty for warmup")
        
        # Lagrangian constraint activation
        if self.current_epoch < self.config.constraint_warmup_epochs:
            # Disable constraints during early training
            self.lagrangian_optimizer.config.max_violation_rate = 1.0  # Effectively disable
        else:
            # Restore normal constraint
            self.lagrangian_optimizer.config.max_violation_rate = 0.05  # 5% violation rate target
    
    def _validate_epoch(self, val_dataloader, safety_evaluation_fn) -> Dict[str, float]:
        """Validate model with safety metrics."""
        
        self.model.eval()
        self.safety_basis.eval()
        self.constitutional_scorer.eval()
        
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Similar to training step but without gradient updates
                batch_metrics = self._validate_step(batch)
                
                for key, value in batch_metrics.items():
                    val_metrics[key].append(value)
        
        # Run comprehensive safety evaluation if provided
        if safety_evaluation_fn:
            safety_eval_results = safety_evaluation_fn(self.model)
            val_metrics.update(safety_eval_results)
        
        # Average validation metrics
        averaged_val_metrics = {
            f'val_{key}': sum(values) / len(values)
            for key, values in val_metrics.items()
        }
        
        return averaged_val_metrics
    
    def _validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute validation step."""
        
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        
        # Forward pass with safety
        base_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Constitutional scoring
        constitutional_scores = self.constitutional_scorer(input_ids, attention_mask)
        
        # Safety control
        safety_level = self.safety_controller(
            base_outputs.hidden_states[-1],
            constitutional_scores,
            domain="general"
        )
        
        # Violation detection
        texts = self._convert_ids_to_text(input_ids)  # Placeholder conversion
        violation_results = self.violation_detector(texts, fast_mode=True)
        
        # Compute validation metrics
        validation_metrics = {
            'constitutional_score': constitutional_scores.mean().item(),
            'safety_level': safety_level,
            'violations_detected': sum(1 for r in violation_results if r.violation_detected),
            'average_confidence': sum(r.confidence for r in violation_results) / len(violation_results)
        }
        
        return validation_metrics
    
    def _convert_ids_to_text(self, input_ids: torch.Tensor) -> List[str]:
        """Convert input IDs to text for violation detection."""
        # Placeholder - in real implementation would use proper tokenizer
        batch_size = input_ids.size(0)
        return [f"sample_text_{i}" for i in range(batch_size)]
    
    def _periodic_evaluation(self):
        """Run periodic comprehensive evaluation."""
        
        # Orthogonality validation
        orthogonality_results = self.safety_basis.validate_orthogonality()
        
        # Constitutional scoring statistics
        constitutional_stats = self.constitutional_scorer.get_constitutional_statistics()
        
        # Safety controller statistics
        controller_stats = self.safety_controller.get_safety_statistics()
        
        # Violation detection statistics
        detection_stats = self.violation_detector.get_detection_statistics()
        
        # Lagrangian optimization analysis
        constraint_analysis = self.lagrangian_optimizer.get_constraint_analysis()
        
        # Log comprehensive metrics
        comprehensive_metrics = {
            'orthogonality': orthogonality_results,
            'constitutional': constitutional_stats,
            'controller': controller_stats,
            'detection': detection_stats,
            'constraints': constraint_analysis
        }
        
        if self.config.use_wandb:
            wandb.log({f'eval/{k}': v for k, v in comprehensive_metrics.items()}, step=self.current_step)
        
        logger.info(f"Step {self.current_step}: Comprehensive evaluation completed")
    
    def _check_performance_degradation(self):
        """Monitor and check for performance degradation."""
        
        if len(self.performance_history) < 10:
            return  # Not enough history
        
        # Calculate recent performance trend
        recent_performance = list(self.performance_history)[-10:]
        baseline_performance = list(self.performance_history)[:10] if len(self.performance_history) > 20 else recent_performance
        
        recent_avg = sum(recent_performance) / len(recent_performance)
        baseline_avg = sum(baseline_performance) / len(baseline_performance)
        
        performance_drop = recent_avg - baseline_avg
        
        if performance_drop > self.config.max_performance_drop:
            self.performance_degradation_count += 1
            logger.warning(f"Performance degradation detected: {performance_drop:.4f} > {self.config.max_performance_drop}")
            
            # Trigger rollback if too many consecutive degradations
            if self.performance_degradation_count >= self.config.rollback_threshold:
                self._trigger_performance_rollback()
        else:
            # Reset counter on good performance
            self.performance_degradation_count = 0
    
    def _trigger_performance_rollback(self):
        """Rollback to best performing model state."""
        
        logger.warning("Triggering performance rollback due to consistent degradation")
        
        if self.best_model_state is not None:
            # Restore best model
            self.model.load_state_dict(self.best_model_state)
            
            # Restore best Lagrangian optimizer state
            self.lagrangian_optimizer.restore_best_model()
            
            # Reset degradation counter
            self.performance_degradation_count = 0
            
            logger.info("Successfully rolled back to best performing model state")
        else:
            logger.warning("No best model state available for rollback")
    
    def _create_curriculum_scheduler(self):
        """Create curriculum learning scheduler."""
        
        # Define curriculum stages with increasing safety complexity
        stages = [
            {'safety_weight': 0.1, 'constraint_weight': 0.0, 'orthogonality_weight': 0.1},
            {'safety_weight': 0.3, 'constraint_weight': 0.2, 'orthogonality_weight': 0.2}, 
            {'safety_weight': 0.6, 'constraint_weight': 0.5, 'orthogonality_weight': 0.3},
            {'safety_weight': 1.0, 'constraint_weight': 1.0, 'orthogonality_weight': 0.5}
        ]
        
        return stages
    
    def _progress_curriculum(self):
        """Progress to next curriculum stage if appropriate."""
        
        if not self.config.enable_curriculum:
            return
        
        # Check if current stage is sufficiently trained
        self.curriculum_progress += 1
        
        if (self.curriculum_progress >= self.config.stage_warmup_steps and
            self.curriculum_stage < len(self.curriculum_scheduler) - 1):
            
            # Advance to next stage
            self.curriculum_stage += 1
            self.curriculum_progress = 0
            
            # Update training weights
            stage_config = self.curriculum_scheduler[self.curriculum_stage]
            self.config.safety_weight = stage_config['safety_weight']
            self.config.orthogonality_weight = stage_config['orthogonality_weight']
            
            logger.info(f"Advanced to curriculum stage {self.curriculum_stage}: {stage_config}")
    
    def _update_training_tracking(self, metrics: Dict[str, float]):
        """Update training progress tracking."""
        
        # Update performance history
        self.performance_history.append(metrics.get('utility_loss', 0.0))
        self.violation_history.append(metrics.get('violation_rate', 0.0))
        
        # Update safety metrics
        for key, value in metrics.items():
            if 'safety' in key or 'constitutional' in key or 'violation' in key:
                self.safety_metrics_history[key].append(value)
        
        # Update best states
        current_violation_rate = metrics.get('violation_rate', float('inf'))
        if current_violation_rate < self.best_violation_rate:
            self.best_violation_rate = current_violation_rate
            self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def _save_epoch_checkpoint(self, metrics: Dict[str, float]):
        """Save checkpoint at end of epoch."""
        
        checkpoint_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.pt"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'safety_basis_state_dict': self.safety_basis.state_dict(),
            'constitutional_scorer_state_dict': self.constitutional_scorer.state_dict(),
            'safety_controller_state_dict': self.safety_controller.state_dict(),
            'violation_detector_state_dict': self.violation_detector.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'best_violation_rate': self.best_violation_rate,
            'curriculum_stage': getattr(self, 'curriculum_stage', 0)
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved epoch checkpoint: {checkpoint_path}")
    
    def _save_step_checkpoint(self):
        """Save checkpoint during training."""
        
        checkpoint_path = self.checkpoint_dir / f"step_{self.current_step}.pt"
        
        # Save lightweight checkpoint
        checkpoint = {
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'lagrangian_state': self.lagrangian_optimizer.lambda_param.item(),
            'best_violation_rate': self.best_violation_rate
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def _log_step_metrics(self, metrics: Dict[str, float]):
        """Log metrics for current training step."""
        
        if self.config.use_wandb:
            wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=self.current_step)
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for completed epoch."""
        
        logger.info(f"Epoch {epoch} completed:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        if self.config.use_wandb:
            wandb.log({f'epoch/{k}': v for k, v in metrics.items()}, step=epoch)
    
    def _should_early_stop(self, metrics: Dict[str, float]) -> bool:
        """Determine if training should stop early."""
        
        # Check violation rate target achievement
        violation_rate = metrics.get('val_violation_rate', float('inf'))
        if violation_rate <= 0.05:  # Target violation rate achieved
            performance_drop = metrics.get('val_utility_loss', 0.0) - self.best_performance
            if performance_drop <= self.config.max_performance_drop:
                logger.info("Early stopping: Target achieved with acceptable performance")
                return True
        
        return False
    
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and return comprehensive results."""
        
        # Run final comprehensive evaluation
        final_metrics = {
            'training_completed': True,
            'total_epochs': self.current_epoch,
            'total_steps': self.current_step,
            'best_violation_rate': self.best_violation_rate,
            'final_performance': list(self.performance_history)[-10:] if self.performance_history else [],
            'orthogonality_validation': self.safety_basis.validate_orthogonality(),
            'constitutional_stats': self.constitutional_scorer.get_constitutional_statistics(),
            'detection_stats': self.violation_detector.get_detection_statistics(),
            'constraint_analysis': self.lagrangian_optimizer.get_constraint_analysis()
        }
        
        # Save final model
        final_checkpoint_path = self.checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'safety_system': {
                'safety_basis': self.safety_basis.state_dict(),
                'constitutional_scorer': self.constitutional_scorer.state_dict(),
                'safety_controller': self.safety_controller.state_dict(),
                'violation_detector': self.violation_detector.state_dict()
            },
            'final_metrics': final_metrics,
            'config': self.config.__dict__
        }, final_checkpoint_path)
        
        logger.info(f"Saved final model: {final_checkpoint_path}")
        
        # Calculate safety improvement
        if len(self.violation_history) >= 100:
            initial_violation_rate = sum(list(self.violation_history)[:50]) / 50
            final_violation_rate = sum(list(self.violation_history)[-50:]) / 50
            violation_reduction = (initial_violation_rate - final_violation_rate) / initial_violation_rate
            final_metrics['violation_reduction_percentage'] = violation_reduction * 100
            
            logger.info(f"Violation reduction achieved: {violation_reduction * 100:.2f}%")
        
        if self.config.use_wandb:
            wandb.log(final_metrics)
            wandb.finish()
        
        return final_metrics