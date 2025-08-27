"""Unified multimodal trainer using bem_core infrastructure.

Demonstrates migration of the multimodal vision-text system to use unified
core infrastructure while preserving multimodal-specific functionality.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from PIL import Image

from ...bem_core.training.base_trainer import BaseTrainer, TrainingConfig
from ...bem_core.evaluation.base_evaluator import BaseEvaluator, EvaluationConfig
from ...bem_core.config.base_config import BaseConfig, ExperimentConfig
from .vision_encoder import VisionEncoder
from .controller_integration import MultimodalController
from .coverage_analysis import CoverageAnalyzer


@dataclass
class MultimodalConfig(BaseConfig):
    """Configuration for multimodal-specific parameters."""
    
    # Vision encoder settings
    vision_encoder_name: str = "openai/clip-vit-base-patch32"
    vision_encoder_dim: int = 768
    freeze_vision_encoder: bool = False
    vision_dropout: float = 0.1
    
    # Multimodal fusion
    fusion_strategy: str = "concat"  # "concat", "attention", "gated"
    fusion_hidden_dim: int = 1024
    cross_attention_heads: int = 8
    cross_attention_layers: int = 2
    
    # Text-image alignment
    alignment_loss_weight: float = 0.1
    contrastive_temperature: float = 0.07
    negative_sampling_ratio: float = 1.0
    
    # Visual grounding
    grounding_enabled: bool = True
    attention_visualization: bool = True
    spatial_attention_heads: int = 4
    
    # Multimodal data handling
    image_size: Tuple[int, int] = (224, 224)
    max_image_tokens: int = 196  # 14x14 patches for ViT
    image_preprocessing: str = "clip"  # "clip", "imagenet"
    
    # Coverage and conflict resolution
    coverage_analysis_enabled: bool = True
    conflict_resolution_strategy: str = "weighted_fusion"  # "weighted_fusion", "gating"
    modality_balance_weight: float = 0.5  # 0.0 = text only, 1.0 = image only
    
    # Performance optimization
    gradient_checkpointing_vision: bool = False
    mixed_precision_vision: bool = True
    vision_batch_processing: bool = True
    

class MultimodalTrainer(BaseTrainer):
    """Unified trainer for multimodal vision-text system.
    
    Inherits from BaseTrainer to provide standardized training infrastructure
    while implementing multimodal-specific training with vision-text fusion,
    cross-modal alignment, and conflict resolution.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        multimodal_config: Optional[MultimodalConfig] = None,
        output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Initialize multimodal trainer.
        
        Args:
            config: Complete experiment configuration
            multimodal_config: Multimodal-specific configuration
            output_dir: Output directory override
            experiment_name: Experiment name override
        """
        # Extract training config from experiment config
        training_config = TrainingConfig(
            learning_rate=config.model.custom_params.get('learning_rate', 5e-5),
            batch_size=config.data.max_samples or 16,
            max_steps=config.model.custom_params.get('max_steps', 1000),
            warmup_steps=config.model.custom_params.get('warmup_steps', 100),
            weight_decay=config.model.custom_params.get('weight_decay', 0.01),
            gradient_checkpointing=config.hardware.gradient_checkpointing,
            fp16=config.hardware.mixed_precision == "fp16",
            bf16=config.hardware.mixed_precision == "bf16",
            device=config.hardware.device,
            seed=config.seed,
            deterministic=config.deterministic,
        )
        
        super().__init__(
            config=training_config,
            output_dir=output_dir or config.output_dir,
            experiment_name=experiment_name or config.name,
        )
        
        # Store configurations
        self.experiment_config = config
        self.multimodal_config = multimodal_config or MultimodalConfig()
        
        # Multimodal components (initialized in _setup_model)
        self.text_encoder = None
        self.vision_encoder = None
        self.multimodal_controller = None
        self.coverage_analyzer = None
        
        # Training state
        self.alignment_history = []
        self.coverage_metrics = {}
        self.conflict_resolution_stats = {}
        
        self.logger.info(f"Initialized MultimodalTrainer with {self.multimodal_config.fusion_strategy} fusion")
    
    def _setup_model(self) -> nn.Module:
        """Set up the multimodal vision-text system."""
        # Initialize text encoder (base model)
        base_model_name = self.experiment_config.model.base_model
        
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=getattr(torch, self.experiment_config.model.torch_dtype),
        )
        
        # Initialize vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=self.multimodal_config.vision_encoder_name,
            output_dim=self.multimodal_config.vision_encoder_dim,
            image_size=self.multimodal_config.image_size,
            freeze_encoder=self.multimodal_config.freeze_vision_encoder,
            dropout=self.multimodal_config.vision_dropout,
        )
        
        # Initialize coverage analyzer
        if self.multimodal_config.coverage_analysis_enabled:
            self.coverage_analyzer = CoverageAnalyzer(
                text_dim=self.experiment_config.model.hidden_size,
                vision_dim=self.multimodal_config.vision_encoder_dim,
                analysis_dim=512,
            )
        
        # Initialize multimodal controller (combines all components)
        self.multimodal_controller = MultimodalController(
            text_encoder=self.text_encoder,
            vision_encoder=self.vision_encoder,
            fusion_strategy=self.multimodal_config.fusion_strategy,
            fusion_hidden_dim=self.multimodal_config.fusion_hidden_dim,
            cross_attention_heads=self.multimodal_config.cross_attention_heads,
            cross_attention_layers=self.multimodal_config.cross_attention_layers,
            coverage_analyzer=self.coverage_analyzer,
            grounding_enabled=self.multimodal_config.grounding_enabled,
        )
        
        # Enable gradient checkpointing for vision if requested
        if self.multimodal_config.gradient_checkpointing_vision:
            if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
                self.vision_encoder.gradient_checkpointing_enable()
        
        self.logger.info("Initialized multimodal vision-text system")
        self.logger.info(f"Vision encoder: {self.multimodal_config.vision_encoder_name}")
        self.logger.info(f"Fusion strategy: {self.multimodal_config.fusion_strategy}")
        self.logger.info(f"Cross-attention heads: {self.multimodal_config.cross_attention_heads}")
        
        return self.multimodal_controller
    
    def _compute_loss(self, batch: Dict[str, Any], model_outputs: Any) -> Dict[str, torch.Tensor]:
        """Compute multimodal loss function with cross-modal alignment."""
        # Extract outputs from multimodal controller
        text_outputs = model_outputs.get('text_outputs')
        vision_outputs = model_outputs.get('vision_outputs')
        fused_outputs = model_outputs.get('fused_outputs')
        alignment_scores = model_outputs.get('alignment_scores')
        coverage_metrics = model_outputs.get('coverage_metrics')
        
        # Compute primary task loss (e.g., language modeling or VQA)
        task_loss = self._compute_task_loss(batch, fused_outputs)
        
        # Compute cross-modal alignment loss
        alignment_loss = self._compute_alignment_loss(alignment_scores, batch)
        
        # Compute coverage loss (ensure both modalities contribute)
        coverage_loss = self._compute_coverage_loss(coverage_metrics, batch)
        
        # Compute contrastive loss (text-image pairs)
        contrastive_loss = self._compute_contrastive_loss(
            text_outputs, vision_outputs, batch
        )
        
        # Combine losses with multimodal weighting
        task_weight = 1.0
        alignment_weight = self.multimodal_config.alignment_loss_weight
        coverage_weight = 0.1  # Encourage balanced modality usage
        contrastive_weight = 0.05  # Weak supervision from image-text pairs
        
        total_loss = (
            task_weight * task_loss +
            alignment_weight * alignment_loss +
            coverage_weight * coverage_loss +
            contrastive_weight * contrastive_loss
        )
        
        return {
            'loss': total_loss,
            'task_loss': task_loss,
            'alignment_loss': alignment_loss,
            'coverage_loss': coverage_loss,
            'contrastive_loss': contrastive_loss,
        }
    
    def _compute_task_loss(self, batch: Dict[str, Any], outputs: Any) -> torch.Tensor:
        """Compute primary task loss (VQA, captioning, etc.)."""
        if hasattr(outputs, 'loss'):
            return outputs.loss
        
        # Fallback: compute cross-entropy loss
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        labels = batch.get('labels')
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    def _compute_alignment_loss(
        self, 
        alignment_scores: Optional[torch.Tensor], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute cross-modal alignment loss."""
        if alignment_scores is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Alignment scores should be high for matching text-image pairs
        # Use InfoNCE-style contrastive loss
        batch_size = alignment_scores.size(0)
        
        # Create positive/negative mask (diagonal should be positive)
        positive_mask = torch.eye(batch_size, device=alignment_scores.device)
        
        # Compute contrastive loss
        exp_scores = torch.exp(alignment_scores / self.multimodal_config.contrastive_temperature)
        pos_exp = exp_scores * positive_mask
        neg_exp = exp_scores * (1 - positive_mask)
        
        # InfoNCE loss
        pos_sum = pos_exp.sum(dim=1)
        total_sum = exp_scores.sum(dim=1)
        
        alignment_loss = -torch.log(pos_sum / total_sum).mean()
        
        return alignment_loss
    
    def _compute_coverage_loss(
        self, 
        coverage_metrics: Optional[Dict[str, torch.Tensor]], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute coverage loss to ensure both modalities contribute."""
        if coverage_metrics is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        text_coverage = coverage_metrics.get('text_coverage')
        vision_coverage = coverage_metrics.get('vision_coverage')
        
        if text_coverage is None or vision_coverage is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Encourage balanced usage of both modalities
        target_balance = self.multimodal_config.modality_balance_weight
        current_balance = vision_coverage / (text_coverage + vision_coverage + 1e-8)
        
        balance_loss = torch.abs(current_balance - target_balance)
        
        # Encourage non-zero coverage for both modalities
        min_coverage_loss = torch.relu(0.1 - text_coverage) + torch.relu(0.1 - vision_coverage)
        
        return balance_loss.mean() + min_coverage_loss.mean()
    
    def _compute_contrastive_loss(
        self, 
        text_outputs: Optional[torch.Tensor],
        vision_outputs: Optional[torch.Tensor], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute contrastive loss between text and vision representations."""
        if text_outputs is None or vision_outputs is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Extract pooled representations
        if hasattr(text_outputs, 'pooler_output'):
            text_repr = text_outputs.pooler_output
        else:
            text_repr = text_outputs.last_hidden_state.mean(dim=1)
        
        vision_repr = vision_outputs  # Assume already pooled
        
        # Normalize representations
        text_repr = nn.functional.normalize(text_repr, p=2, dim=1)
        vision_repr = nn.functional.normalize(vision_repr, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_repr, vision_repr.T)
        
        # Contrastive loss (similar to CLIP)
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Cross-entropy loss in both directions
        loss_t2v = nn.functional.cross_entropy(
            similarity_matrix / self.multimodal_config.contrastive_temperature, 
            labels
        )
        loss_v2t = nn.functional.cross_entropy(
            similarity_matrix.T / self.multimodal_config.contrastive_temperature, 
            labels
        )
        
        return (loss_t2v + loss_v2t) / 2
    
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run multimodal-specific evaluation."""
        # Create multimodal evaluator
        eval_config = EvaluationConfig(
            batch_size=self.config.batch_size,
            metrics=[
                'cross_modal_alignment', 'vision_text_consistency', 
                'modality_balance', 'grounding_accuracy'
            ],
            measure_latency=True,
            measure_memory=True,
        )
        
        evaluator = MultimodalEvaluator(
            model=self.model,
            config=eval_config,
            multimodal_config=self.multimodal_config,
            device=self.device,
            logger=self.logger,
        )
        
        result = evaluator.evaluate(dataloader)
        
        # Track alignment history
        if 'cross_modal_alignment' in result.metrics:
            self.alignment_history.append(result.metrics['cross_modal_alignment'])
        
        # Update coverage metrics
        if 'modality_balance' in result.metrics:
            self.coverage_metrics['latest_balance'] = result.metrics['modality_balance']
        
        return result.metrics
    
    def get_multimodal_metrics(self) -> Dict[str, Any]:
        """Get current multimodal training metrics."""
        return {
            'alignment_history_length': len(self.alignment_history),
            'avg_recent_alignment': (
                sum(self.alignment_history[-10:]) / min(10, len(self.alignment_history))
                if self.alignment_history else 0.0
            ),
            'coverage_metrics': dict(self.coverage_metrics),
            'conflict_resolution_stats': dict(self.conflict_resolution_stats),
            'fusion_strategy': self.multimodal_config.fusion_strategy,
            'modality_balance_target': self.multimodal_config.modality_balance_weight,
        }


class MultimodalEvaluator(BaseEvaluator):
    """Evaluator for multimodal vision-text system.
    
    Inherits from BaseEvaluator to provide standardized evaluation infrastructure
    while implementing multimodal-specific metrics and inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig,
        multimodal_config: MultimodalConfig,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None,
    ):
        """Initialize multimodal evaluator."""
        super().__init__(model, config, device, logger)
        self.multimodal_config = multimodal_config
    
    def _run_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run multimodal inference on a batch."""
        # Forward pass through multimodal controller
        outputs = self.model(**batch)
        
        # Extract multimodal-specific outputs
        predictions = []
        alignment_scores = []
        coverage_stats = []
        attention_maps = []
        
        if hasattr(outputs, 'logits'):
            # Primary task predictions
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predictions = predicted_ids.cpu().tolist()
        
        if outputs.get('alignment_scores') is not None:
            alignment_scores = outputs['alignment_scores'].cpu().numpy()
        
        if outputs.get('coverage_metrics') is not None:
            coverage_stats = outputs['coverage_metrics']
        
        if outputs.get('attention_maps') is not None:
            attention_maps = outputs['attention_maps'].cpu().numpy()
        
        return {
            'predictions': predictions,
            'alignment_scores': alignment_scores,
            'coverage_stats': coverage_stats,
            'attention_maps': attention_maps,
            'model_outputs': outputs,
        }
    
    def _compute_component_metrics(
        self, 
        predictions: List[Any], 
        targets: List[Any]
    ) -> Dict[str, float]:
        """Compute multimodal-specific metrics."""
        metrics = {}
        
        # Extract multimodal information from predictions
        all_alignment_scores = []
        all_coverage_stats = []
        
        for pred in predictions:
            if isinstance(pred, dict):
                if 'alignment_scores' in pred:
                    all_alignment_scores.append(pred['alignment_scores'])
                if 'coverage_stats' in pred:
                    all_coverage_stats.append(pred['coverage_stats'])
        
        if all_alignment_scores:
            import numpy as np
            
            # Cross-modal alignment score
            alignment_matrix = np.concatenate(all_alignment_scores, axis=0)
            diagonal_scores = np.diag(alignment_matrix)
            off_diagonal_scores = alignment_matrix[~np.eye(alignment_matrix.shape[0], dtype=bool)]
            
            # Good alignment: high diagonal, low off-diagonal
            metrics['cross_modal_alignment'] = float(np.mean(diagonal_scores))
            metrics['alignment_contrast'] = float(np.mean(diagonal_scores) - np.mean(off_diagonal_scores))
            metrics['alignment_consistency'] = float(1.0 / (1.0 + np.std(diagonal_scores)))
        
        if all_coverage_stats:
            # Modality balance metrics
            text_coverage_values = []
            vision_coverage_values = []
            
            for stats in all_coverage_stats:
                if isinstance(stats, dict):
                    if 'text_coverage' in stats:
                        text_coverage_values.append(float(stats['text_coverage']))
                    if 'vision_coverage' in stats:
                        vision_coverage_values.append(float(stats['vision_coverage']))
            
            if text_coverage_values and vision_coverage_values:
                import numpy as np
                
                text_mean = np.mean(text_coverage_values)
                vision_mean = np.mean(vision_coverage_values)
                total_mean = text_mean + vision_mean
                
                if total_mean > 0:
                    vision_ratio = vision_mean / total_mean
                    target_ratio = self.multimodal_config.modality_balance_weight
                    
                    metrics['modality_balance'] = float(1.0 - abs(vision_ratio - target_ratio))
                    metrics['text_coverage'] = float(text_mean)
                    metrics['vision_coverage'] = float(vision_mean)
                    metrics['total_coverage'] = float(total_mean)
        
        # Visual grounding accuracy (if attention maps are available)
        # This would require ground truth attention annotations
        
        return metrics


# Migration helper function
def migrate_from_legacy_multimodal_trainer(
    legacy_config_path: str,
    output_dir: str,
    vision_encoder: str = "openai/clip-vit-base-patch32",
) -> MultimodalTrainer:
    """Helper function to migrate from legacy multimodal trainer.
    
    Args:
        legacy_config_path: Path to legacy multimodal configuration
        output_dir: Output directory for new trainer
        vision_encoder: Vision encoder model name
        
    Returns:
        New unified multimodal trainer
    """
    from ...bem_core.config.config_loader import load_experiment_config
    
    # Load legacy config and convert to new format
    config = load_experiment_config(legacy_config_path)
    
    # Create multimodal config with custom vision encoder
    multimodal_config = MultimodalConfig(
        vision_encoder_name=vision_encoder,
        coverage_analysis_enabled=True,
        grounding_enabled=True,
    )
    
    # Create new trainer with unified infrastructure
    trainer = MultimodalTrainer(
        config=config,
        multimodal_config=multimodal_config,
        output_dir=output_dir,
        experiment_name=config.name,
    )
    
    return trainer