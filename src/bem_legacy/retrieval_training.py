"""
Training pipeline enhancements for retrieval-aware BEM.

This module extends the hierarchical training pipeline to include:
1. Retrieval feature integration in training loops
2. Retrieval effectiveness metrics and logging
3. Coverage/consistency loss terms for better feature learning
4. Index-swap evaluation during training
5. Retrieval-aware validation and testing

Based on TODO.md Phase 3 training requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import json
from tqdm import tqdm

from .retrieval_bem import FullRetrievalAwareBEM, RetrievalBEMConfig
from .hierarchical_training import HierarchicalBEMTrainer, HierarchicalTrainingConfig, create_hierarchical_trainer
from .retrieval import MicroRetriever, RetrievalConfig
from .retrieval_features import RetrievalFeaturesConfig
from .telemetry import TelemetryCollector as BEMTelemetry

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTrainingConfig(HierarchicalTrainingConfig):
    """Extended training configuration for retrieval-aware BEM."""
    
    # Retrieval training settings
    include_retrieval_loss: bool = True
    retrieval_loss_weight: float = 0.1  # Weight for retrieval-based losses
    coverage_loss_weight: float = 0.05
    consistency_loss_weight: float = 0.05
    
    # Training phases
    warmup_without_retrieval: int = 100  # Warmup steps without retrieval loss
    retrieval_curriculum: bool = True  # Gradually increase retrieval influence
    
    # Evaluation settings
    run_index_swap_eval: bool = True
    index_swap_eval_frequency: int = 500  # Every N steps
    index_swap_test_queries: int = 50
    
    # Retrieval effectiveness tracking
    track_retrieval_effectiveness: bool = True
    log_retrieval_stats_frequency: int = 100
    
    # Feature learning
    adaptive_feature_weighting: bool = True
    feature_weight_lr: float = 0.01
    
    # Cache management during training
    clear_retrieval_cache_frequency: int = 1000  # Clear cache every N steps
    max_cache_size_during_training: int = 500


class RetrievalLossFunction:
    """
    Custom loss functions for training retrieval-aware BEM.
    Includes coverage and consistency losses to improve feature quality.
    """
    
    def __init__(
        self,
        coverage_weight: float = 0.05,
        consistency_weight: float = 0.05,
        margin: float = 0.1
    ):
        self.coverage_weight = coverage_weight
        self.consistency_weight = consistency_weight
        self.margin = margin
    
    def compute_coverage_loss(
        self,
        coverage_scores: torch.Tensor,  # [batch]
        target_relevance: Optional[torch.Tensor] = None  # [batch] - ground truth relevance
    ) -> torch.Tensor:
        """
        Compute coverage loss that encourages high coverage for relevant queries.
        
        Args:
            coverage_scores: Coverage scores from retrieval features
            target_relevance: Optional ground truth relevance (0-1 scores)
            
        Returns:
            coverage_loss: Loss encouraging better coverage
        """
        if target_relevance is not None:
            # Supervised coverage loss - encourage high coverage for relevant queries
            coverage_loss = F.mse_loss(coverage_scores, target_relevance)
        else:
            # Unsupervised coverage loss - encourage reasonable coverage values
            # Penalize very low coverage (suggests poor retrieval)
            low_coverage_penalty = F.relu(0.3 - coverage_scores).mean()
            
            # Penalize extreme high coverage (suggests over-fitting to retrieval)
            high_coverage_penalty = F.relu(coverage_scores - 0.9).mean()
            
            coverage_loss = low_coverage_penalty + high_coverage_penalty
        
        return coverage_loss
    
    def compute_consistency_loss(
        self,
        consistency_scores: torch.Tensor,  # [batch]
        coverage_scores: torch.Tensor  # [batch]
    ) -> torch.Tensor:
        """
        Compute consistency loss that encourages consistency when coverage is high.
        
        Args:
            consistency_scores: Consistency scores from retrieval features
            coverage_scores: Coverage scores for weighting
            
        Returns:
            consistency_loss: Loss encouraging consistency for high-coverage queries
        """
        # Weight consistency loss by coverage - only care about consistency when coverage is good
        coverage_weights = torch.sigmoid(coverage_scores * 5.0)  # Sharpen the weighting
        
        # Encourage high consistency for high-coverage queries
        consistency_targets = torch.ones_like(consistency_scores) * 0.7  # Target consistency of 0.7
        weighted_mse = (consistency_scores - consistency_targets).pow(2) * coverage_weights
        
        consistency_loss = weighted_mse.mean()
        
        return consistency_loss
    
    def compute_feature_alignment_loss(
        self,
        features: Dict[str, torch.Tensor],
        task_performance: torch.Tensor  # [batch] - task performance scores
    ) -> torch.Tensor:
        """
        Compute loss that aligns retrieval features with task performance.
        
        Args:
            features: Dictionary of retrieval features
            task_performance: Task performance scores (higher = better)
            
        Returns:
            alignment_loss: Loss encouraging feature-performance alignment
        """
        # Combine features into a single score
        combined_score = (
            features.get('coverage', torch.zeros_like(task_performance)) * 0.4 +
            features.get('consistency', torch.zeros_like(task_performance)) * 0.3 +
            features.get('coherence', torch.zeros_like(task_performance)) * 0.3
        )
        
        # Encourage positive correlation between features and performance
        # Using ranking loss to maintain relative ordering
        batch_size = task_performance.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=task_performance.device)
        
        alignment_loss = torch.tensor(0.0, device=task_performance.device)
        count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # If task_performance[i] > task_performance[j], then combined_score[i] should > combined_score[j]
                if task_performance[i] > task_performance[j]:
                    # Margin-based ranking loss
                    loss = F.relu(self.margin - (combined_score[i] - combined_score[j]))
                    alignment_loss += loss
                    count += 1
                elif task_performance[j] > task_performance[i]:
                    loss = F.relu(self.margin - (combined_score[j] - combined_score[i]))
                    alignment_loss += loss
                    count += 1
        
        if count > 0:
            alignment_loss /= count
        
        return alignment_loss
    
    def __call__(
        self,
        features: Dict[str, torch.Tensor],
        task_performance: Optional[torch.Tensor] = None,
        target_relevance: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all retrieval losses.
        
        Args:
            features: Dictionary of retrieval features
            task_performance: Optional task performance scores
            target_relevance: Optional ground truth relevance
            
        Returns:
            losses: Dictionary of computed losses
        """
        losses = {}
        
        # Coverage loss
        if 'coverage' in features:
            coverage_loss = self.compute_coverage_loss(
                features['coverage'], target_relevance
            )
            losses['coverage_loss'] = coverage_loss * self.coverage_weight
        
        # Consistency loss  
        if 'consistency' in features and 'coverage' in features:
            consistency_loss = self.compute_consistency_loss(
                features['consistency'], features['coverage']
            )
            losses['consistency_loss'] = consistency_loss * self.consistency_weight
        
        # Feature alignment loss
        if task_performance is not None:
            alignment_loss = self.compute_feature_alignment_loss(
                features, task_performance
            )
            losses['alignment_loss'] = alignment_loss * 0.02  # Small weight
        
        # Total retrieval loss
        total_retrieval_loss = sum(losses.values())
        losses['total_retrieval_loss'] = total_retrieval_loss
        
        return losses


class RetrievalTraining(HierarchicalBEMTrainer):
    """
    Extended training class for retrieval-aware BEM with retrieval loss integration.
    """
    
    def __init__(
        self,
        model: FullRetrievalAwareBEM,
        config: RetrievalTrainingConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        telemetry: Optional[BEMTelemetry] = None
    ):
        # Initialize parent hierarchical training
        super().__init__(model, config, optimizer, scheduler, telemetry)
        
        self.retrieval_config = config
        
        # Retrieval loss function
        self.retrieval_loss_fn = RetrievalLossFunction(
            coverage_weight=config.coverage_loss_weight,
            consistency_weight=config.consistency_loss_weight
        )
        
        # Adaptive feature weights
        if config.adaptive_feature_weighting:
            feature_names = ['coverage', 'consistency', 'coherence', 'diversity_coverage']
            self.feature_weights = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(1.0))
                for name in feature_names
            })
            self.feature_optimizer = torch.optim.Adam(
                self.feature_weights.parameters(), 
                lr=config.feature_weight_lr
            )
        else:
            self.feature_weights = None
            self.feature_optimizer = None
        
        # Retrieval effectiveness tracking
        self.retrieval_stats = {
            'total_retrievals': 0,
            'avg_coverage': 0.0,
            'avg_consistency': 0.0,
            'cache_hit_rate': 0.0,
            'retrieval_timeouts': 0
        }
        
        # Index swap test queries (loaded lazily)
        self.index_swap_queries = None
    
    def _compute_retrieval_curriculum_weight(self, step: int) -> float:
        """Compute curriculum weight for retrieval loss."""
        if not self.retrieval_config.retrieval_curriculum:
            return 1.0
        
        warmup_steps = self.retrieval_config.warmup_without_retrieval
        if step < warmup_steps:
            return 0.0
        
        # Gradual ramp-up over next 500 steps
        rampup_steps = 500
        if step < warmup_steps + rampup_steps:
            return (step - warmup_steps) / rampup_steps
        
        return 1.0
    
    def _extract_retrieval_features_from_outputs(
        self,
        outputs: Union[torch.Tensor, Tuple],
        routing_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Extract retrieval features from model outputs."""
        if routing_info is None:
            return None
        
        features = {}
        
        # Aggregate retrieval features across layers
        for layer_name, layer_info in routing_info.get('layers', {}).items():
            retrieval_info = layer_info.get('retrieval_info', {})
            
            if retrieval_info.get('retrieval_success', False):
                coverage = retrieval_info.get('coverage', 0.0)
                consistency = retrieval_info.get('consistency', 0.0)
                
                if 'coverage' not in features:
                    features['coverage'] = []
                if 'consistency' not in features:
                    features['consistency'] = []
                
                features['coverage'].append(coverage)
                features['consistency'].append(consistency)
        
        # Convert to tensors
        for key, values in features.items():
            if values:
                features[key] = torch.tensor(values, device=outputs.device if hasattr(outputs, 'device') else 'cpu')
            else:
                features[key] = torch.tensor(0.0, device=outputs.device if hasattr(outputs, 'device') else 'cpu')
        
        return features if features else None
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced training step with retrieval loss integration.
        
        Args:
            batch: Training batch
            step: Current training step
            
        Returns:
            losses: Dictionary of computed losses
        """
        # Run standard hierarchical training step
        hierarchical_losses = super().training_step(batch, step)
        
        # Add retrieval losses if enabled
        if (self.retrieval_config.include_retrieval_loss and 
            hasattr(self.model, 'retrieval_config') and 
            self.model.retrieval_config.retrieval_enabled):
            
            try:
                # Get routing info from last forward pass
                routing_info = getattr(self, '_last_routing_info', None)
                
                if routing_info is not None:
                    # Extract retrieval features
                    retrieval_features = self._extract_retrieval_features_from_outputs(
                        hierarchical_losses.get('logits'), routing_info
                    )
                    
                    if retrieval_features is not None:
                        # Compute task performance proxy (negative loss)
                        task_performance = -hierarchical_losses.get('total_loss', torch.tensor(0.0))
                        if task_performance.dim() == 0:
                            task_performance = task_performance.unsqueeze(0)
                        
                        # Compute retrieval losses
                        retrieval_losses = self.retrieval_loss_fn(
                            retrieval_features,
                            task_performance
                        )
                        
                        # Apply curriculum weighting
                        curriculum_weight = self._compute_retrieval_curriculum_weight(step)
                        
                        for key, loss in retrieval_losses.items():
                            retrieval_losses[key] = loss * curriculum_weight * self.retrieval_config.retrieval_loss_weight
                        
                        # Add to total loss
                        hierarchical_losses.update(retrieval_losses)
                        hierarchical_losses['total_loss'] += retrieval_losses['total_retrieval_loss']
                        
                        # Update retrieval statistics
                        self._update_retrieval_stats(retrieval_features, step)
            
            except Exception as e:
                logger.warning(f"Error computing retrieval losses: {e}")
        
        # Clear retrieval cache periodically
        if (step > 0 and 
            step % self.retrieval_config.clear_retrieval_cache_frequency == 0):
            self._clear_retrieval_caches()
        
        return hierarchical_losses
    
    def _update_retrieval_stats(
        self,
        features: Dict[str, torch.Tensor],
        step: int
    ):
        """Update retrieval effectiveness statistics."""
        coverage = features.get('coverage', torch.tensor(0.0)).mean().item()
        consistency = features.get('consistency', torch.tensor(0.0)).mean().item()
        
        # Exponential moving average
        alpha = 0.01
        self.retrieval_stats['avg_coverage'] = (
            alpha * coverage + (1 - alpha) * self.retrieval_stats['avg_coverage']
        )
        self.retrieval_stats['avg_consistency'] = (
            alpha * consistency + (1 - alpha) * self.retrieval_stats['avg_consistency']
        )
        
        # Log statistics
        if step % self.retrieval_config.log_retrieval_stats_frequency == 0:
            self._log_retrieval_stats(step)
    
    def _log_retrieval_stats(self, step: int):
        """Log retrieval effectiveness statistics."""
        if self.telemetry is not None:
            self.telemetry.log({
                'retrieval/avg_coverage': self.retrieval_stats['avg_coverage'],
                'retrieval/avg_consistency': self.retrieval_stats['avg_consistency'],
                'retrieval/cache_hit_rate': self.retrieval_stats['cache_hit_rate']
            }, step)
        
        # Get model-level retrieval stats
        if hasattr(self.model, 'get_comprehensive_statistics'):
            model_stats = self.model.get_comprehensive_statistics()
            retrieval_stats = model_stats.get('retrieval', {})
            
            if self.telemetry is not None:
                for key, value in retrieval_stats.get('global_retriever_stats', {}).items():
                    self.telemetry.log({f'retrieval_model/{key}': value}, step)
    
    def _clear_retrieval_caches(self):
        """Clear retrieval caches across all BEM modules."""
        if hasattr(self.model, 'bem_modules'):
            for bem_module in self.model.bem_modules.values():
                if hasattr(bem_module, 'retrieval_cache') and bem_module.retrieval_cache:
                    bem_module.retrieval_cache.clear()
    
    def run_index_swap_evaluation(
        self,
        step: int,
        alternative_index_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run index-swap evaluation to test policy vs memory.
        
        Args:
            step: Current training step
            alternative_index_path: Path to alternative index
            
        Returns:
            results: Index-swap evaluation results
        """
        if not self.retrieval_config.run_index_swap_eval:
            return {}
        
        if alternative_index_path is None:
            logger.warning("No alternative index path provided for index-swap evaluation")
            return {}
        
        # Load test queries (create simple ones if not available)
        if self.index_swap_queries is None:
            self.index_swap_queries = [
                f"test query {i}" for i in range(self.retrieval_config.index_swap_test_queries)
            ]
        
        try:
            results = self.model.run_index_swap_evaluation(
                self.index_swap_queries[:self.retrieval_config.index_swap_test_queries],
                alternative_index_path
            )
            
            # Log results
            if self.telemetry is not None:
                for key, value in results.items():
                    self.telemetry.log({f'index_swap/{key}': value}, step)
            
            logger.info(f"Index-swap evaluation at step {step}: "
                       f"policy_over_memory={results.get('policy_over_memory', False)}, "
                       f"behavior_change={results.get('mean_behavior_change', 0.0):.3f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Index-swap evaluation failed: {e}")
            return {'error': str(e)}
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, torch.Tensor]:
        """Enhanced validation with retrieval metrics."""
        # Run standard hierarchical validation
        val_losses = super().validation_step(batch, step)
        
        # Add retrieval-specific validation
        if hasattr(self.model, 'get_comprehensive_statistics'):
            stats = self.model.get_comprehensive_statistics()
            retrieval_stats = stats.get('retrieval', {})
            
            # Add retrieval metrics to validation output
            for layer_name, layer_stats in retrieval_stats.get('layer_retrieval_stats', {}).items():
                val_losses[f'retrieval_{layer_name}_hit_rate'] = torch.tensor(
                    layer_stats.get('cache_hit_rate', 0.0)
                )
        
        # Run index-swap evaluation periodically during validation
        if (step > 0 and 
            step % self.retrieval_config.index_swap_eval_frequency == 0):
            # This would need an alternative index path - placeholder for now
            # index_swap_results = self.run_index_swap_evaluation(step, "alt_index.faiss")
            pass
        
        return val_losses
    
    def save_checkpoint(
        self,
        path: str,
        step: int,
        include_retrieval_index: bool = True
    ):
        """Save checkpoint with retrieval components."""
        # Save standard checkpoint
        super().save_checkpoint(path, step)
        
        # Save retrieval-specific components
        if include_retrieval_index and hasattr(self.model, 'micro_retriever'):
            index_path = Path(path).parent / f"retrieval_index_step_{step}.faiss"
            try:
                self.model.save_retrieval_index(str(index_path))
                logger.info(f"Saved retrieval index to {index_path}")
            except Exception as e:
                logger.warning(f"Failed to save retrieval index: {e}")
        
        # Save adaptive feature weights
        if self.feature_weights is not None:
            weights_path = Path(path).parent / f"feature_weights_step_{step}.pt"
            torch.save({
                'feature_weights': {k: v.item() for k, v in self.feature_weights.items()},
                'step': step
            }, weights_path)
    
    def load_checkpoint(
        self,
        path: str,
        load_retrieval_index: bool = True
    ):
        """Load checkpoint with retrieval components."""
        # Load standard checkpoint
        super().load_checkpoint(path)
        
        # Load retrieval index if available
        if load_retrieval_index and hasattr(self.model, 'micro_retriever'):
            checkpoint_dir = Path(path).parent
            index_files = list(checkpoint_dir.glob("retrieval_index_step_*.faiss"))
            
            if index_files:
                # Load the most recent index
                latest_index = max(index_files, key=lambda p: p.stat().st_mtime)
                try:
                    self.model.load_retrieval_index(str(latest_index))
                    logger.info(f"Loaded retrieval index from {latest_index}")
                except Exception as e:
                    logger.warning(f"Failed to load retrieval index: {e}")
        
        # Load feature weights if available
        if self.feature_weights is not None:
            checkpoint_dir = Path(path).parent
            weight_files = list(checkpoint_dir.glob("feature_weights_step_*.pt"))
            
            if weight_files:
                latest_weights = max(weight_files, key=lambda p: p.stat().st_mtime)
                try:
                    weights_data = torch.load(latest_weights)
                    for name, weight in weights_data['feature_weights'].items():
                        if name in self.feature_weights:
                            self.feature_weights[name].data.fill_(weight)
                    logger.info(f"Loaded feature weights from {latest_weights}")
                except Exception as e:
                    logger.warning(f"Failed to load feature weights: {e}")


def create_retrieval_trainer(
    model: FullRetrievalAwareBEM,
    config: Optional[RetrievalTrainingConfig] = None,
    **config_kwargs
) -> RetrievalTraining:
    """
    Factory function to create a retrieval-aware trainer.
    
    Args:
        model: FullRetrievalAwareBEM instance
        config: RetrievalTrainingConfig instance
        **config_kwargs: Config overrides
        
    Returns:
        RetrievalTraining instance
    """
    if config is None:
        config = RetrievalTrainingConfig(**config_kwargs)
    
    # Create optimizer for BEM parameters only
    bem_params = model.get_bem_parameters()
    optimizer = torch.optim.AdamW(
        bem_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler if requested
    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler_warmup_steps,
            eta_min=config.learning_rate * 0.1
        )
    
    # Create telemetry
    telemetry = BEMTelemetry() if config.track_retrieval_effectiveness else None
    
    return RetrievalTraining(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        telemetry=telemetry
    )