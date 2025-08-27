"""
Performance Track Training Infrastructure.

Training framework for PT1-PT4 variants with specialized optimizers,
regularization, and monitoring for each performance optimization approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
import logging
import time
import json
from pathlib import Path

# Import PT variant modules
from .pt1_head_gating import HeadGroupGatingModule, HeadGroupGatingConfig
from .pt2_dynamic_mask import DynamicRankMaskModule, DynamicRankMaskConfig  
from .pt3_kronecker import KroneckerModule, KroneckerConfig
from .pt4_residual_film import ResidualFiLMModule, ResidualFiLMConfig
from .evaluation import BudgetValidator, BudgetConstraints, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class PTTrainingConfig:
    """Training configuration for Performance Track variants."""
    
    # Basic training parameters
    learning_rate: float = 5e-4
    batch_size: int = 32
    max_steps: int = 1000
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    
    # Variant-specific learning rates
    variant_lr_multipliers: Dict[str, float] = None
    
    # Regularization
    gradient_clip_norm: float = 1.0
    spectral_penalty_weight: float = 0.1
    budget_penalty_weight: float = 1.0
    
    # Monitoring and validation
    eval_steps: int = 100
    save_steps: int = 500  
    logging_steps: int = 50
    early_stopping_patience: int = 5
    
    # Budget constraints
    budget_constraints: BudgetConstraints = None
    enforce_budget_during_training: bool = True
    
    # Optimization strategy
    optimizer_type: str = "adamw"  # adamw, adam, sgd
    scheduler_type: str = "cosine"  # cosine, linear, constant
    
    # Variant-specific training
    variant_specific_config: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.variant_lr_multipliers is None:
            self.variant_lr_multipliers = {
                'PT1': 1.0,  # Standard LR for head gating
                'PT2': 0.5,  # Lower LR for dynamic masks (sensitive)
                'PT3': 1.5,  # Higher LR for Kronecker (needs more aggressive updates)
                'PT4': 0.1   # Very low LR for FiLM (stability critical)
            }
        
        if self.budget_constraints is None:
            self.budget_constraints = BudgetConstraints()
        
        if self.variant_specific_config is None:
            self.variant_specific_config = {
                'PT1': {
                    'decorrelation_annealing': True,
                    'gate_entropy_weight': 0.01
                },
                'PT2': {
                    'sparsity_annealing': True,
                    'mask_consistency_weight': 0.1
                },
                'PT3': {
                    'orthogonality_weight': 0.01,
                    'diversity_weight': 0.001
                },
                'PT4': {
                    'stability_monitoring': True,
                    'emergency_reset_threshold': 10.0
                }
            }


class PTOptimizer:
    """Specialized optimizer for Performance Track variants."""
    
    def __init__(self, config: PTTrainingConfig):
        self.config = config
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        
    def setup_optimizers(self, model: nn.Module, variant_type: str):
        """Setup variant-specific optimizers and schedulers."""
        
        # Get variant-specific learning rate
        base_lr = self.config.learning_rate
        variant_lr = base_lr * self.config.variant_lr_multipliers.get(variant_type, 1.0)
        
        # Separate parameters for different components
        param_groups = self._create_parameter_groups(model, variant_type, variant_lr)
        
        # Create optimizer
        if self.config.optimizer_type == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=variant_lr,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer_type == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=variant_lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            optimizer = optim.SGD(
                param_groups,
                lr=variant_lr,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
        
        self.optimizers[variant_type] = optimizer
        
        # Create scheduler
        if self.config.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_steps,
                eta_min=variant_lr * 0.01
            )
        elif self.config.scheduler_type == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        else:
            scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        
        self.schedulers[variant_type] = scheduler
        
        logger.info(f"Setup optimizer for {variant_type}: LR={variant_lr:.2e}")
        
    def _create_parameter_groups(
        self, 
        model: nn.Module, 
        variant_type: str, 
        base_lr: float
    ) -> List[Dict[str, Any]]:
        """Create parameter groups with variant-specific learning rates."""
        
        param_groups = []
        
        if variant_type == "PT1":
            # Head gating: separate rates for controller vs projections
            gate_controller_params = []
            group_projection_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if 'gate_controller' in name:
                    gate_controller_params.append(param)
                elif 'group_projections' in name:
                    group_projection_params.append(param)
                else:
                    other_params.append(param)
            
            param_groups.extend([
                {'params': gate_controller_params, 'lr': base_lr * 0.5},  # Lower LR for controller
                {'params': group_projection_params, 'lr': base_lr},
                {'params': other_params, 'lr': base_lr}
            ])
            
        elif variant_type == "PT2":
            # Dynamic masking: very low LR for mask parameters
            mask_params = []
            projection_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if 'mask' in name or 'instance_controller' in name:
                    mask_params.append(param)
                elif any(proj in name for proj in ['U', 'V']):
                    projection_params.append(param)
                else:
                    other_params.append(param)
            
            param_groups.extend([
                {'params': mask_params, 'lr': base_lr * 0.1},  # Very low for masks
                {'params': projection_params, 'lr': base_lr},
                {'params': other_params, 'lr': base_lr}
            ])
            
        elif variant_type == "PT3":
            # Kronecker: different rates for U, V factors vs projections
            kronecker_u_params = []
            kronecker_v_params = []
            projection_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if 'kronecker.U' in name:
                    kronecker_u_params.append(param)
                elif 'kronecker.V' in name:
                    kronecker_v_params.append(param)
                elif 'output_projection' in name:
                    projection_params.append(param)
                else:
                    other_params.append(param)
            
            param_groups.extend([
                {'params': kronecker_u_params, 'lr': base_lr * 1.5},  # Higher for factors
                {'params': kronecker_v_params, 'lr': base_lr * 1.5},
                {'params': projection_params, 'lr': base_lr},
                {'params': other_params, 'lr': base_lr}
            ])
            
        elif variant_type == "PT4":
            # FiLM: extremely low rates for stability
            controller_params = []
            modulation_params = []
            residual_scale_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if 'controller' in name:
                    controller_params.append(param)
                elif 'modulation' in name:
                    modulation_params.append(param)
                elif 'residual_scale' in name:
                    residual_scale_params.append(param)
                else:
                    other_params.append(param)
            
            param_groups.extend([
                {'params': controller_params, 'lr': base_lr * 0.05},
                {'params': modulation_params, 'lr': base_lr * 0.02},
                {'params': residual_scale_params, 'lr': base_lr * 0.01},
                {'params': other_params, 'lr': base_lr}
            ])
        else:
            # Default: single parameter group
            param_groups = [{'params': model.parameters(), 'lr': base_lr}]
        
        return param_groups
    
    def step(self, variant_type: str):
        """Step optimizer and scheduler for variant."""
        if variant_type in self.optimizers:
            self.optimizers[variant_type].step()
        if variant_type in self.schedulers:
            self.schedulers[variant_type].step()
    
    def zero_grad(self, variant_type: str):
        """Zero gradients for variant optimizer."""
        if variant_type in self.optimizers:
            self.optimizers[variant_type].zero_grad()
    
    def get_lr(self, variant_type: str) -> float:
        """Get current learning rate for variant."""
        if variant_type in self.optimizers:
            return self.optimizers[variant_type].param_groups[0]['lr']
        return 0.0


class PTLossFunction:
    """Composite loss function for Performance Track training."""
    
    def __init__(self, config: PTTrainingConfig):
        self.config = config
        self.budget_validator = BudgetValidator(config.budget_constraints)
        
    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        variant_type: str,
        attention_info: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute composite loss with variant-specific regularization."""
        
        # Base task loss (cross-entropy, MSE, etc.)
        base_loss = nn.functional.cross_entropy(outputs, targets)
        
        total_loss = base_loss
        loss_components = {'base_loss': base_loss}
        
        # Variant-specific regularization
        if attention_info:
            if variant_type == "PT1":
                reg_loss = self._compute_pt1_regularization(attention_info)
            elif variant_type == "PT2":
                reg_loss = self._compute_pt2_regularization(attention_info)
            elif variant_type == "PT3":
                reg_loss = self._compute_pt3_regularization(attention_info)
            elif variant_type == "PT4":
                reg_loss = self._compute_pt4_regularization(attention_info)
            else:
                reg_loss = {}
            
            for name, value in reg_loss.items():
                loss_components[name] = value
                total_loss = total_loss + value
        
        # Spectral penalty
        spectral_loss = self._compute_spectral_penalty(model)
        if spectral_loss > 0:
            spectral_weighted = self.config.spectral_penalty_weight * spectral_loss
            loss_components['spectral_penalty'] = spectral_weighted
            total_loss = total_loss + spectral_weighted
        
        # Budget penalty (if enforced during training)
        if self.config.enforce_budget_during_training:
            budget_loss = self._compute_budget_penalty(model)
            if budget_loss > 0:
                budget_weighted = self.config.budget_penalty_weight * budget_loss
                loss_components['budget_penalty'] = budget_weighted
                total_loss = total_loss + budget_weighted
        
        loss_components['total_loss'] = total_loss
        return loss_components
    
    def _compute_pt1_regularization(self, attention_info: Dict) -> Dict[str, torch.Tensor]:
        """PT1-specific regularization terms."""
        reg_losses = {}
        
        # Decorrelation penalty
        if 'decorrelation_loss' in attention_info:
            reg_losses['decorrelation'] = attention_info['decorrelation_loss']
        
        # Gate entropy regularization (encourage diversity)
        if 'gate_entropy' in attention_info:
            entropy_target = torch.log(torch.tensor(4.0))  # For 4 groups
            entropy_loss = torch.abs(attention_info['gate_entropy'] - entropy_target)
            reg_losses['gate_entropy'] = 0.01 * entropy_loss
        
        return reg_losses
    
    def _compute_pt2_regularization(self, attention_info: Dict) -> Dict[str, torch.Tensor]:
        """PT2-specific regularization terms."""
        reg_losses = {}
        
        # Sparsity regularization (encourage target sparsity)
        if 'sparsity' in attention_info:
            target_sparsity = 0.5
            sparsity_loss = torch.abs(attention_info['sparsity'] - target_sparsity)
            reg_losses['sparsity'] = 0.1 * sparsity_loss
        
        # Mask entropy (encourage diversity in selection)
        if 'mask_entropy' in attention_info:
            reg_losses['mask_entropy'] = -0.01 * attention_info['mask_entropy']
        
        return reg_losses
    
    def _compute_pt3_regularization(self, attention_info: Dict) -> Dict[str, torch.Tensor]:
        """PT3-specific regularization terms."""
        reg_losses = {}
        
        # Orthogonality regularization
        if 'orthogonal_reg' in attention_info:
            reg_losses['orthogonal'] = attention_info['orthogonal_reg']
        
        # Diversity regularization  
        if 'diversity_reg' in attention_info:
            reg_losses['diversity'] = attention_info['diversity_reg']
        
        return reg_losses
    
    def _compute_pt4_regularization(self, attention_info: Dict) -> Dict[str, torch.Tensor]:
        """PT4-specific regularization terms."""
        reg_losses = {}
        
        # Stability penalty (penalize large modulations)
        if 'gamma' in attention_info and 'beta' in attention_info:
            gamma_deviation = torch.abs(attention_info['gamma'] - 1.0).mean()
            beta_magnitude = torch.abs(attention_info['beta']).mean()
            reg_losses['stability'] = 0.1 * (gamma_deviation + beta_magnitude)
        
        return reg_losses
    
    def _compute_spectral_penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute spectral penalty for all model parameters."""
        spectral_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if len(param.shape) == 2 and param.numel() > 100:  # Only for matrices
                try:
                    # Compute largest singular value
                    u, s, v = torch.svd(param)
                    max_sv = s[0]
                    
                    # Penalty for exceeding threshold
                    if max_sv > 1.0:
                        spectral_loss = spectral_loss + (max_sv - 1.0) ** 2
                        
                except RuntimeError:
                    # Skip if SVD fails
                    continue
        
        return spectral_loss
    
    def _compute_budget_penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute penalty for budget constraint violations."""
        param_count = sum(p.numel() for p in model.parameters())
        
        # Parameter count penalty
        param_validation = self.budget_validator.validate_parameters(param_count)
        if not param_validation['valid']:
            deviation = abs(param_validation['deviation_pct']) / 100.0
            return torch.tensor(deviation ** 2)
        
        return torch.tensor(0.0)


class PTTrainer:
    """Main trainer for Performance Track variants."""
    
    def __init__(self, config: PTTrainingConfig):
        self.config = config
        self.optimizer = PTOptimizer(config)
        self.loss_function = PTLossFunction(config)
        
        # Training state
        self.global_step = 0
        self.best_metrics = {}
        self.early_stopping_counter = 0
        
        # Logging
        self.train_metrics = []
        self.eval_metrics = []
        
    def train_variant(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        variant_type: str,
        save_dir: Path
    ) -> Dict[str, Any]:
        """Train a specific PT variant."""
        
        logger.info(f"Starting training for {variant_type}")
        
        # Setup optimizer for this variant
        self.optimizer.setup_optimizers(model, variant_type)
        
        # Training loop
        model.train()
        start_time = time.time()
        
        for step in range(self.config.max_steps):
            self.global_step = step
            
            # Training step
            train_metrics = self._training_step(model, train_loader, variant_type)
            self.train_metrics.append(train_metrics)
            
            # Logging
            if step % self.config.logging_steps == 0:
                self._log_metrics(train_metrics, step, 'train')
            
            # Evaluation
            if step % self.config.eval_steps == 0:
                eval_metrics = self._evaluation_step(model, eval_loader, variant_type)
                self.eval_metrics.append(eval_metrics)
                self._log_metrics(eval_metrics, step, 'eval')
                
                # Early stopping check
                if self._check_early_stopping(eval_metrics, variant_type):
                    logger.info(f"Early stopping triggered at step {step}")
                    break
            
            # Saving
            if step % self.config.save_steps == 0 and step > 0:
                self._save_checkpoint(model, save_dir, variant_type, step)
        
        # Final evaluation and save
        final_metrics = self._evaluation_step(model, eval_loader, variant_type)
        self._save_checkpoint(model, save_dir, variant_type, 'final')
        
        training_time = time.time() - start_time
        
        # Training summary
        training_summary = {
            'variant_type': variant_type,
            'total_steps': self.global_step + 1,
            'training_time_seconds': training_time,
            'final_metrics': final_metrics,
            'best_metrics': self.best_metrics.get(variant_type, {}),
            'train_metrics_history': self.train_metrics,
            'eval_metrics_history': self.eval_metrics,
            'config': asdict(self.config)
        }
        
        # Save training summary
        summary_path = save_dir / f"{variant_type}_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        logger.info(f"Training completed for {variant_type} in {training_time:.2f}s")
        return training_summary
    
    def _training_step(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        variant_type: str
    ) -> Dict[str, float]:
        """Single training step."""
        
        model.train()
        self.optimizer.zero_grad(variant_type)
        
        # Get batch
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            train_loader = DataLoader(train_loader.dataset, 
                                    batch_size=self.config.batch_size, 
                                    shuffle=True)
            batch = next(iter(train_loader))
        
        inputs, targets = batch
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # Forward pass
        outputs = model(inputs, output_attentions=True)
        if isinstance(outputs, tuple):
            logits, attention_info = outputs
        else:
            logits, attention_info = outputs, None
        
        # Compute loss
        loss_dict = self.loss_function.compute_loss(
            model, logits, targets, variant_type, attention_info
        )
        
        # Backward pass
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.config.gradient_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step(variant_type)
        
        # Convert to float for logging
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in loss_dict.items()}
        metrics['learning_rate'] = self.optimizer.get_lr(variant_type)
        metrics['step'] = self.global_step
        
        return metrics
    
    def _evaluation_step(
        self,
        model: nn.Module,
        eval_loader: DataLoader,
        variant_type: str
    ) -> Dict[str, float]:
        """Evaluation step."""
        
        model.eval()
        eval_losses = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs, targets = batch
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                # Forward pass
                outputs = model(inputs, output_attentions=True)
                if isinstance(outputs, tuple):
                    logits, attention_info = outputs
                else:
                    logits, attention_info = outputs, None
                
                # Compute loss
                loss_dict = self.loss_function.compute_loss(
                    model, logits, targets, variant_type, attention_info
                )
                eval_losses.append(loss_dict['total_loss'].item())
                
                # Accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.numel()
        
        # Average metrics
        avg_loss = np.mean(eval_losses)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'step': self.global_step
        }
        
        return metrics
    
    def _check_early_stopping(
        self,
        eval_metrics: Dict[str, float],
        variant_type: str
    ) -> bool:
        """Check early stopping condition."""
        
        if variant_type not in self.best_metrics:
            self.best_metrics[variant_type] = eval_metrics
            self.early_stopping_counter = 0
            return False
        
        # Check if current metrics are better (lower loss)
        current_loss = eval_metrics['eval_loss']
        best_loss = self.best_metrics[variant_type]['eval_loss']
        
        if current_loss < best_loss:
            self.best_metrics[variant_type] = eval_metrics
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        mode: str
    ):
        """Log training/evaluation metrics."""
        
        log_str = f"[{mode.upper()}] Step {step}: "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f"{key}={value:.4f} "
        
        logger.info(log_str)
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        save_dir: Path,
        variant_type: str,
        step: Union[int, str]
    ):
        """Save model checkpoint."""
        
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"{variant_type}_checkpoint_{step}.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dicts': {
                k: v.state_dict() for k, v in self.optimizer.optimizers.items()
            },
            'scheduler_state_dicts': {
                k: v.state_dict() for k, v in self.optimizer.schedulers.items()
            },
            'global_step': self.global_step,
            'config': asdict(self.config),
            'best_metrics': self.best_metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def run_pt_variant_training(
    variant_type: str,
    config: PTTrainingConfig,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    save_dir: Path,
    model_factory_fn: Callable[[], nn.Module]
) -> Dict[str, Any]:
    """
    Run training for a specific PT variant.
    
    Args:
        variant_type: One of PT1, PT2, PT3, PT4
        config: Training configuration
        train_loader: Training data loader
        eval_loader: Evaluation data loader  
        save_dir: Directory to save results
        model_factory_fn: Function that creates the model instance
        
    Returns:
        Training results dictionary
    """
    
    logger.info(f"Initializing training for {variant_type}")
    
    # Create model
    model = model_factory_fn()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create trainer
    trainer = PTTrainer(config)
    
    # Run training
    training_results = trainer.train_variant(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        variant_type=variant_type,
        save_dir=save_dir
    )
    
    return training_results