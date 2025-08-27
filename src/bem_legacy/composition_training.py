"""
BEM Phase 4: Composition Training Module

This module implements training pipeline for multi-BEM systems with orthogonality
constraints, budget enforcement, and performance balancing across component tasks.

Key Features:
- Joint Training: Train multiple BEMs simultaneously with constraints
- Orthogonality Loss: Penalize violations of subspace reservations
- Budget Enforcement: Apply trust region projection during training
- Performance Balancing: Maintain performance across all component tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Any, Callable
from dataclasses import dataclass
import logging
from collections import defaultdict
import json
import wandb

from .multi_bem import MultiBEMComposer, MultiBEMConfig
from .trust_region import TrustRegionProjector
from .subspace import OrthogonalityEnforcer
from .interference_testing import InterferenceTester, CanaryTask

logger = logging.getLogger(__name__)


@dataclass
class CompositionTrainingConfig:
    """Configuration for composition training."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    batch_size: int = 32
    
    # Loss component weights
    task_loss_weight: float = 1.0
    orthogonality_loss_weight: float = 0.1
    trust_region_loss_weight: float = 0.05
    interference_loss_weight: float = 0.1
    
    # Training schedule
    warmup_epochs: int = 10
    orthogonality_enforcement_start: int = 5
    trust_region_enforcement_start: int = 3
    
    # Validation and monitoring
    validation_frequency: int = 5
    save_frequency: int = 10
    early_stopping_patience: int = 20
    
    # Numerical stability
    gradient_clip_norm: float = 1.0
    use_fp32_for_loss: bool = True
    
    # Logging
    log_to_wandb: bool = False
    wandb_project: str = "bem-phase4"


class TaskBatch(NamedTuple):
    """Batch data for a specific task."""
    task_id: str
    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: Dict[str, Any]


class TrainingStep(NamedTuple):
    """Information about a training step."""
    step: int
    epoch: int
    task_losses: Dict[str, float]
    orthogonality_loss: float
    trust_region_loss: float
    interference_loss: float
    total_loss: float
    learning_rate: float


class CompositionLossFunction(nn.Module):
    """
    Composite loss function for multi-BEM training.
    
    Combines task-specific losses with orthogonality, trust region,
    and interference penalties.
    """
    
    def __init__(
        self,
        config: CompositionTrainingConfig,
        composer: MultiBEMComposer,
        task_loss_functions: Dict[str, Callable]
    ):
        """
        Initialize the composite loss function.
        
        Args:
            config: Training configuration
            composer: Multi-BEM composer
            task_loss_functions: Dict mapping task IDs to loss functions
        """
        super().__init__()
        
        self.config = config
        self.composer = composer
        self.task_loss_functions = task_loss_functions
        self.orthogonality_enforcer = OrthogonalityEnforcer()
        
        # Track loss components
        self.loss_history = defaultdict(list)
        
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        task_batches: List[TaskBatch],
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss for multi-BEM training.
        
        Args:
            model_outputs: Model outputs for each task
            task_batches: List of task batches
            epoch: Current epoch (affects loss weighting)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        device = next(iter(model_outputs.values())).device
        
        # Compute task-specific losses
        task_losses = {}
        total_task_loss = torch.tensor(0.0, device=device)
        
        for task_batch in task_batches:
            task_id = task_batch.task_id
            
            if task_id in model_outputs and task_id in self.task_loss_functions:
                output = model_outputs[task_id]
                target = task_batch.targets.to(device)
                
                loss_fn = self.task_loss_functions[task_id]
                task_loss = loss_fn(output, target)
                
                task_losses[task_id] = task_loss.item()
                total_task_loss = total_task_loss + task_loss
            else:
                task_losses[task_id] = 0.0
        
        # Orthogonality loss
        orthogonality_loss = self._compute_orthogonality_loss(epoch)
        
        # Trust region loss  
        trust_region_loss = self._compute_trust_region_loss(epoch)
        
        # Interference loss (penalty for canary task degradation)
        interference_loss = self._compute_interference_loss(epoch)
        
        # Combine losses with epoch-dependent weighting
        task_weight = self.config.task_loss_weight
        ortho_weight = self.config.orthogonality_loss_weight if epoch >= self.config.orthogonality_enforcement_start else 0.0
        tr_weight = self.config.trust_region_loss_weight if epoch >= self.config.trust_region_enforcement_start else 0.0
        interference_weight = self.config.interference_loss_weight
        
        total_loss = (
            task_weight * total_task_loss +
            ortho_weight * orthogonality_loss +
            tr_weight * trust_region_loss +
            interference_weight * interference_loss
        )
        
        # Record loss components
        loss_components = {
            'task_loss': total_task_loss.item(),
            'orthogonality_loss': orthogonality_loss.item(),
            'trust_region_loss': trust_region_loss.item(), 
            'interference_loss': interference_loss.item(),
            'total_loss': total_loss.item()
        }
        loss_components.update(task_losses)
        
        # Update history
        for component, value in loss_components.items():
            self.loss_history[component].append(value)
        
        return total_loss, loss_components
    
    def _compute_orthogonality_loss(self, epoch: int) -> torch.Tensor:
        """
        Compute orthogonality penalty loss.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Orthogonality loss tensor
        """
        if epoch < self.config.orthogonality_enforcement_start:
            return torch.tensor(0.0)
        
        try:
            # Get BEM registry
            registry = self.composer.get_bem_registry()
            
            if len(registry) < 2:
                return torch.tensor(0.0)  # No orthogonality needed for single BEM
            
            # Collect U and V matrices
            u_matrices = {}
            v_matrices = {}
            allocations = {}
            
            for bem_id, entry in registry.items():
                if hasattr(entry.bem_module, 'lora_U') and hasattr(entry.bem_module, 'lora_V'):
                    u_matrices[bem_id] = entry.bem_module.lora_U
                    v_matrices[bem_id] = entry.bem_module.lora_V
                    allocations[bem_id] = entry.allocation
            
            if len(u_matrices) < 2:
                return torch.tensor(0.0)
            
            # Compute orthogonality violations
            ortho_loss = torch.tensor(0.0, device=next(iter(u_matrices.values())).device)
            
            bem_ids = list(u_matrices.keys())
            for i, bem_id_1 in enumerate(bem_ids):
                for bem_id_2 in bem_ids[i+1:]:
                    alloc_1 = allocations[bem_id_1]
                    alloc_2 = allocations[bem_id_2]
                    
                    # Extract subspaces
                    u1 = u_matrices[bem_id_1][:, alloc_1.u_start_idx:alloc_1.u_end_idx]
                    u2 = u_matrices[bem_id_2][:, alloc_2.u_start_idx:alloc_2.u_end_idx]
                    v1 = v_matrices[bem_id_1][:, alloc_1.v_start_idx:alloc_1.v_end_idx]
                    v2 = v_matrices[bem_id_2][:, alloc_2.v_start_idx:alloc_2.v_end_idx]
                    
                    # Compute cross-correlations (should be zero for orthogonal subspaces)
                    u_cross_corr = torch.matmul(u1.t(), u2)
                    v_cross_corr = torch.matmul(v1.t(), v2)
                    
                    # L2 penalty on cross-correlations
                    ortho_loss = ortho_loss + torch.sum(u_cross_corr ** 2) + torch.sum(v_cross_corr ** 2)
            
            return ortho_loss
            
        except Exception as e:
            logger.error(f"Error computing orthogonality loss: {e}")
            return torch.tensor(0.0)
    
    def _compute_trust_region_loss(self, epoch: int) -> torch.Tensor:
        """
        Compute trust region violation penalty.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Trust region loss tensor
        """
        if epoch < self.config.trust_region_enforcement_start:
            return torch.tensor(0.0)
        
        try:
            # This is a simplified implementation
            # In practice, you would accumulate trust region violations during forward passes
            # and penalize them here
            
            # Placeholder: penalty for budget violations
            # You would track these violations in the composer
            tr_loss = torch.tensor(0.0)
            
            return tr_loss
            
        except Exception as e:
            logger.error(f"Error computing trust region loss: {e}")
            return torch.tensor(0.0)
    
    def _compute_interference_loss(self, epoch: int) -> torch.Tensor:
        """
        Compute interference penalty based on canary task performance.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Interference loss tensor
        """
        try:
            # This is a placeholder implementation
            # In practice, you would periodically evaluate canary tasks
            # and penalize performance degradation
            
            interference_loss = torch.tensor(0.0)
            
            return interference_loss
            
        except Exception as e:
            logger.error(f"Error computing interference loss: {e}")
            return torch.tensor(0.0)


class CompositionTrainer:
    """
    Main trainer for multi-BEM composition systems.
    
    This class orchestrates the training process, handling joint optimization
    of multiple BEMs with orthogonality and trust region constraints.
    """
    
    def __init__(
        self,
        config: CompositionTrainingConfig,
        composer: MultiBEMComposer,
        task_datasets: Dict[str, Dataset],
        task_loss_functions: Dict[str, Callable],
        canary_tasks: Optional[List[CanaryTask]] = None
    ):
        """
        Initialize the composition trainer.
        
        Args:
            config: Training configuration
            composer: Multi-BEM composer to train
            task_datasets: Dict mapping task IDs to datasets
            task_loss_functions: Dict mapping task IDs to loss functions
            canary_tasks: Optional canary tasks for interference testing
        """
        self.config = config
        self.composer = composer
        self.task_datasets = task_datasets
        self.task_loss_functions = task_loss_functions
        
        # Create data loaders
        self.task_loaders = {
            task_id: DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0  # Simplified for example
            )
            for task_id, dataset in task_datasets.items()
        }
        
        # Initialize loss function
        self.loss_function = CompositionLossFunction(
            config, composer, task_loss_functions
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            composer.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs
        )
        
        # Interference tester
        if canary_tasks:
            from .interference_testing import InterferenceTester
            self.interference_tester = InterferenceTester(composer, canary_tasks)
        else:
            self.interference_tester = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Logging
        if config.log_to_wandb:
            wandb.init(project=config.wandb_project, config=config.__dict__)
        
        logger.info(f"Initialized CompositionTrainer with {len(task_datasets)} tasks")
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """
        Run the complete training process.
        
        Args:
            model: Base model to train with BEMs
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting multi-BEM composition training")
        
        # Establish interference baselines if tester available
        if self.interference_tester:
            self.interference_tester.establish_baselines(model)
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            epoch_metrics = self._train_epoch(model)
            
            # Validation
            if epoch % self.config.validation_frequency == 0:
                val_metrics = self._validate_epoch(model)
                epoch_metrics.update(val_metrics)
            
            # Learning rate step
            self.scheduler.step()
            
            # Early stopping check
            current_loss = epoch_metrics.get('total_loss', float('inf'))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            self._log_epoch(epoch, epoch_metrics)
            
            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self._save_checkpoint(epoch)
            
            # Enforce orthogonality periodically
            if epoch >= self.config.orthogonality_enforcement_start and epoch % 5 == 0:
                success = self.composer.enforce_orthogonality()
                if not success:
                    logger.warning(f"Orthogonality enforcement failed at epoch {epoch}")
        
        # Final validation and interference testing
        final_results = self._finalize_training(model)
        
        logger.info("Training completed")
        
        return final_results
    
    def _train_epoch(self, model: nn.Module) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            
        Returns:
            Dictionary with epoch metrics
        """
        model.train()
        self.composer.train()
        
        epoch_losses = defaultdict(list)
        
        # Create iterator that cycles through all tasks
        task_iterators = {
            task_id: iter(loader) for task_id, loader in self.task_loaders.items()
        }
        
        # Determine steps per epoch (use the largest dataset)
        steps_per_epoch = max(len(loader) for loader in self.task_loaders.values())
        
        for step in range(steps_per_epoch):
            self.optimizer.zero_grad()
            
            # Collect batches from all tasks
            task_batches = []
            model_outputs = {}
            
            for task_id, iterator in task_iterators.items():
                try:
                    batch_data = next(iterator)
                except StopIteration:
                    # Reset iterator if exhausted
                    task_iterators[task_id] = iter(self.task_loaders[task_id])
                    batch_data = next(task_iterators[task_id])
                
                # Create TaskBatch
                if isinstance(batch_data, (list, tuple)):
                    inputs, targets = batch_data[0], batch_data[1]
                    metadata = {}
                else:
                    # Assume dict-like batch
                    inputs = batch_data['inputs']
                    targets = batch_data['targets']
                    metadata = {k: v for k, v in batch_data.items() if k not in ['inputs', 'targets']}
                
                task_batch = TaskBatch(
                    task_id=task_id,
                    inputs=inputs,
                    targets=targets,
                    metadata=metadata
                )
                task_batches.append(task_batch)
                
                # Forward pass through model + composer
                with torch.no_grad():
                    # This is simplified - in practice you'd route through appropriate layers
                    output = model(inputs.to(next(model.parameters()).device))
                    model_outputs[task_id] = output
            
            # Compute loss
            total_loss, loss_components = self.loss_function(
                model_outputs, task_batches, self.current_epoch
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.composer.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Record losses
            for component, value in loss_components.items():
                epoch_losses[component].append(value)
            
            self.current_step += 1
        
        # Aggregate epoch losses
        return {component: np.mean(values) for component, values in epoch_losses.items()}
    
    def _validate_epoch(self, model: nn.Module) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            model: Model to validate
            
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        self.composer.eval()
        
        val_metrics = {}
        
        # Run interference testing if available
        if self.interference_tester:
            try:
                # Create a simple BEM configuration (all BEMs enabled)
                from .interference_testing import BEMConfiguration
                bem_configs = [
                    BEMConfiguration(bem_id=bem_id, enabled=True)
                    for bem_id in self.composer.get_bem_registry().keys()
                ]
                
                if bem_configs:
                    interference_result = self.interference_tester.run_interference_test(
                        model=model,
                        bem_configs=bem_configs,
                        config_id=f"epoch_{self.current_epoch}",
                        num_trials=1
                    )
                    
                    val_metrics.update({
                        'interference_score': interference_result.overall_interference_score,
                        'num_violations': len(interference_result.violations),
                        'violation_tasks': interference_result.violations
                    })
                
            except Exception as e:
                logger.error(f"Interference testing failed during validation: {e}")
        
        # Add composition statistics
        composition_stats = self.composer.get_composition_stats()
        val_metrics.update({
            'orthogonality_valid': composition_stats['orthogonality_valid'],
            'num_active_bems': composition_stats['num_bems']
        })
        
        return val_metrics
    
    def _finalize_training(self, model: nn.Module) -> Dict[str, Any]:
        """
        Finalize training with comprehensive evaluation.
        
        Args:
            model: Trained model
            
        Returns:
            Final training results
        """
        # Run comprehensive interference testing
        final_results = {
            'training_config': self.config.__dict__,
            'total_epochs': self.current_epoch,
            'total_steps': self.current_step,
            'best_loss': self.best_loss,
            'composition_stats': self.composer.get_composition_stats()
        }
        
        if self.interference_tester:
            try:
                # Test all BEM combinations
                combination_results = self.interference_tester.test_bem_combinations(
                    model, max_combination_size=3, num_trials_per_config=2
                )
                
                # Generate report
                report = self.interference_tester.generate_interference_report()
                
                final_results.update({
                    'interference_report': report,
                    'combination_results': len(combination_results)
                })
                
            except Exception as e:
                logger.error(f"Final interference testing failed: {e}")
        
        return final_results
    
    def _log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log epoch metrics.
        
        Args:
            epoch: Current epoch
            metrics: Metrics to log
        """
        # Log to console
        logger.info(f"Epoch {epoch}: " + 
                   ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        
        # Log to wandb if enabled
        if self.config.log_to_wandb:
            wandb.log(metrics, step=epoch)
        
        # Store in history
        self.training_history.append({
            'epoch': epoch,
            'step': self.current_step,
            **metrics
        })
    
    def _save_checkpoint(self, epoch: int):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
        """
        checkpoint = {
            'epoch': epoch,
            'step': self.current_step,
            'composer_state_dict': self.composer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }
        
        checkpoint_path = f"composition_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")


def create_composition_trainer(
    composer: MultiBEMComposer,
    task_datasets: Dict[str, Dataset],
    task_loss_functions: Dict[str, Callable],
    config: Optional[CompositionTrainingConfig] = None,
    canary_tasks: Optional[List[CanaryTask]] = None
) -> CompositionTrainer:
    """
    Factory function to create a CompositionTrainer.
    
    Args:
        composer: Multi-BEM composer to train
        task_datasets: Task datasets
        task_loss_functions: Task loss functions
        config: Optional training configuration
        canary_tasks: Optional canary tasks
        
    Returns:
        Configured CompositionTrainer instance
    """
    if config is None:
        config = CompositionTrainingConfig()
    
    return CompositionTrainer(
        config=config,
        composer=composer,
        task_datasets=task_datasets,
        task_loss_functions=task_loss_functions,
        canary_tasks=canary_tasks
    )


def create_default_composition_training_config() -> CompositionTrainingConfig:
    """
    Create a default composition training configuration.
    
    Returns:
        Default CompositionTrainingConfig
    """
    return CompositionTrainingConfig(
        learning_rate=1e-4,
        max_epochs=100,
        batch_size=32,
        orthogonality_loss_weight=0.1,
        trust_region_loss_weight=0.05,
        interference_loss_weight=0.1
    )