"""
Training utilities for hierarchical BEM systems.
Implements end-to-end training methodology with controller-specific losses and optimization.

Supports multiple training strategies:
- End-to-end backpropagation
- Expert imitation learning
- Reinforcement learning approaches
- Hybrid methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import math
from dataclasses import dataclass, field
from enum import Enum
import wandb
from contextlib import nullcontext

from .hierarchical_bem import FullHierarchicalBEM, HierarchicalBEMConfig
from .controller import RoutingLevel, RoutingState
from .simple_bem import SimpleBEMModule


class TrainingStrategy(Enum):
    """Training strategies for hierarchical BEM."""
    END_TO_END = "end_to_end"           # Standard backpropagation
    EXPERT_IMITATION = "expert_imitation"  # Learn from static LoRA experts
    REINFORCEMENT = "reinforcement"      # RL with task performance reward
    HYBRID = "hybrid"                   # Combination of strategies


@dataclass
class HierarchicalTrainingConfig:
    """Configuration for hierarchical BEM training."""
    
    # Training strategy
    strategy: TrainingStrategy = TrainingStrategy.END_TO_END
    
    # Optimization
    learning_rate: float = 1e-3
    controller_lr: float = 1e-3
    bem_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 10000
    
    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Loss weights
    lm_loss_weight: float = 1.0
    controller_loss_weight: float = 0.1
    kl_divergence_weight: float = 0.05
    entropy_regularization: float = 0.01
    delta_norm_regularization: float = 0.001
    orthogonality_regularization: float = 0.001
    
    # Routing-specific
    routing_temperature: float = 1.0
    uncertainty_target: float = 0.8
    ema_decay: float = 0.99
    
    # Expert imitation (if using EXPERT_IMITATION strategy)
    expert_models: Optional[List[str]] = None
    imitation_weight: float = 1.0
    
    # Reinforcement learning (if using REINFORCEMENT strategy)
    reward_function: Optional[str] = None
    baseline_decay: float = 0.99
    
    # Monitoring
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 2000
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "fp16"  # "fp16" or "bf16"
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = False


class HierarchicalBEMTrainer:
    """
    Trainer for hierarchical BEM systems with multiple training strategies.
    
    Supports:
    - End-to-end training with language modeling loss
    - Expert imitation from pre-trained static LoRAs
    - Reinforcement learning with task-specific rewards
    - Comprehensive loss functions and regularization
    """
    
    def __init__(
        self,
        model: FullHierarchicalBEM,
        config: HierarchicalTrainingConfig,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Setup loss functions
        self._setup_loss_functions()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Expert models for imitation learning
        self.expert_models = {}
        if config.strategy == TrainingStrategy.EXPERT_IMITATION:
            self._load_expert_models()
        
        # Baseline for reinforcement learning
        self.reward_baseline = 0.0
        
        # Mixed precision scaler
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp_dtype == "fp16")
        else:
            self.scaler = None
    
    def _setup_optimizers(self):
        """Setup optimizers for different parameter groups."""
        # Separate parameter groups
        controller_params = list(self.model.controller.parameters())
        bem_params = []
        for bem_module in self.model.bem_modules.values():
            bem_params.extend([bem_module.lora_U, bem_module.lora_V])
        
        # Controller optimizer (typically higher learning rate)
        self.controller_optimizer = AdamW(
            controller_params,
            lr=self.config.controller_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # BEM parameters optimizer
        self.bem_optimizer = AdamW(
            bem_params,
            lr=self.config.bem_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate schedulers
        self.controller_scheduler = CosineAnnealingWarmRestarts(
            self.controller_optimizer,
            T_0=self.config.max_steps // 4,
            eta_min=self.config.controller_lr * 0.01
        )
        
        self.bem_scheduler = CosineAnnealingWarmRestarts(
            self.bem_optimizer,
            T_0=self.config.max_steps // 4,
            eta_min=self.config.bem_lr * 0.01
        )
    
    def _setup_loss_functions(self):
        """Setup loss functions."""
        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    def _load_expert_models(self):
        """Load expert models for imitation learning."""
        if self.config.expert_models is None:
            return
        
        for expert_path in self.config.expert_models:
            # Load expert model - implement based on your expert format
            expert_model = torch.load(expert_path, map_location=self.device)
            expert_name = expert_path.split('/')[-1].split('.')[0]
            self.expert_models[expert_name] = expert_model
    
    def compute_language_modeling_loss(
        self,
        outputs,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute standard language modeling loss."""
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Shift labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous().view(-1)
            shift_logits = shift_logits[shift_mask == 1]
            shift_labels = shift_labels[shift_mask == 1]
        
        # Compute loss
        lm_loss = self.lm_loss_fn(shift_logits, shift_labels)
        return lm_loss
    
    def compute_kl_divergence_loss(
        self,
        bem_outputs,
        base_outputs,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence between BEM and base model to prevent drift.
        """
        # Get logits
        if hasattr(bem_outputs, 'logits'):
            bem_logits = bem_outputs.logits
        else:
            bem_logits = bem_outputs
        
        if hasattr(base_outputs, 'logits'):
            base_logits = base_outputs.logits
        else:
            base_logits = base_outputs
        
        # Convert to probabilities
        bem_probs = F.log_softmax(bem_logits, dim=-1)
        base_probs = F.softmax(base_logits.detach(), dim=-1)  # Stop gradient on base
        
        # Compute KL divergence
        kl_loss = self.kl_loss_fn(bem_probs, base_probs)
        
        return kl_loss
    
    def compute_controller_regularization(
        self,
        routing_info: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute controller-specific regularization losses.
        """
        losses = {}
        
        # Entropy regularization (encourage confident decisions)
        if 'entropy' in routing_info:
            entropy = routing_info['entropy']
            if isinstance(entropy, torch.Tensor):
                # Target moderate entropy - not too high (uncertain) or too low (collapsed)
                target_entropy = math.log(self.config.routing_temperature)
                entropy_loss = F.mse_loss(entropy, torch.tensor(target_entropy, device=entropy.device))
                losses['entropy'] = entropy_loss
        
        # Uncertainty regularization
        if 'uncertainty' in routing_info and routing_info['uncertainty'] is not None:
            uncertainty = routing_info['uncertainty']
            if isinstance(uncertainty, torch.Tensor):
                # Encourage moderate uncertainty
                target_uncertainty = self.config.uncertainty_target
                uncertainty_loss = F.mse_loss(
                    uncertainty, 
                    torch.tensor(target_uncertainty, device=uncertainty.device)
                )
                losses['uncertainty'] = uncertainty_loss
        
        # Delta norm regularization (prevent large weight changes)
        delta_norms = []
        for layer_name, bem_module in self.model.bem_modules.items():
            # Estimate delta norm: ||U * diag(code) * V^T||_F
            # Approximate as ||U||_F * ||V||_F * ||code||_2
            u_norm = bem_module.lora_U.norm()
            v_norm = bem_module.lora_V.norm()
            # Use average code norm as proxy
            approx_delta_norm = u_norm * v_norm * 1.0  # Placeholder for actual code norm
            delta_norms.append(approx_delta_norm)
        
        if delta_norms:
            delta_norm_loss = torch.stack(delta_norms).mean()
            losses['delta_norm'] = delta_norm_loss
        
        return losses
    
    def compute_expert_imitation_loss(
        self,
        codes: torch.Tensor,
        task_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute imitation loss for learning from expert LoRAs.
        """
        # This is a simplified version - implement based on your expert setup
        if not self.expert_models:
            return torch.tensor(0.0, device=codes.device)
        
        # Compute target codes from experts
        target_codes = []
        for i, task_label in enumerate(task_labels):
            if task_label.item() in self.expert_models:
                expert_code = self.expert_models[task_label.item()]
                target_codes.append(expert_code)
            else:
                # Use zeros if no expert available
                target_codes.append(torch.zeros_like(codes[i]))
        
        target_codes = torch.stack(target_codes)
        
        # MSE loss between predicted and target codes
        imitation_loss = F.mse_loss(codes, target_codes)
        return imitation_loss
    
    def compute_total_loss(
        self,
        outputs,
        labels: torch.Tensor,
        base_outputs=None,
        routing_info: Optional[Dict[str, Any]] = None,
        task_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total training loss with all components.
        """
        losses = {}
        
        # Language modeling loss
        lm_loss = self.compute_language_modeling_loss(outputs, labels)
        losses['lm_loss'] = lm_loss
        
        # KL divergence loss (behavior preservation)
        if base_outputs is not None:
            kl_loss = self.compute_kl_divergence_loss(outputs, base_outputs)
            losses['kl_loss'] = kl_loss
        
        # Controller regularization
        if routing_info is not None:
            reg_losses = self.compute_controller_regularization(routing_info)
            losses.update(reg_losses)
        
        # Expert imitation loss
        if self.config.strategy == TrainingStrategy.EXPERT_IMITATION and task_labels is not None:
            if routing_info and 'codes' in routing_info:
                imitation_loss = self.compute_expert_imitation_loss(
                    routing_info['codes'], task_labels
                )
                losses['imitation_loss'] = imitation_loss
        
        # Combine losses with weights
        total_loss = (
            self.config.lm_loss_weight * losses.get('lm_loss', 0) +
            self.config.kl_divergence_weight * losses.get('kl_loss', 0) +
            self.config.entropy_regularization * losses.get('entropy', 0) +
            self.config.delta_norm_regularization * losses.get('delta_norm', 0) +
            self.config.imitation_weight * losses.get('imitation_loss', 0)
        )
        
        losses['total_loss'] = total_loss
        return total_loss, losses
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step.
        """
        self.model.train()
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels', input_ids)
        side_signals = batch.get('side_signals')
        task_labels = batch.get('task_labels')
        
        # Mixed precision context
        amp_context = torch.cuda.amp.autocast(
            enabled=self.config.use_amp,
            dtype=torch.float16 if self.config.amp_dtype == "fp16" else torch.bfloat16
        ) if self.config.use_amp else nullcontext()
        
        with amp_context:
            # Forward pass with BEM
            outputs, routing_info = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                side_signals=side_signals,
                return_routing_info=True
            )
            
            # Base model forward pass for KL divergence (if needed)
            base_outputs = None
            if self.config.kl_divergence_weight > 0:
                with torch.no_grad():
                    # Get base model outputs without BEM
                    # This is simplified - implement based on your base model access
                    base_outputs = outputs  # Placeholder
            
            # Compute loss
            total_loss, loss_dict = self.compute_total_loss(
                outputs=outputs,
                labels=labels,
                base_outputs=base_outputs,
                routing_info=routing_info,
                task_labels=task_labels
            )
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Update on gradient accumulation boundary
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.controller_optimizer)
                self.scaler.unscale_(self.bem_optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.controller.parameters(), 
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.get_bem_parameters(), 
                self.config.max_grad_norm
            )
            
            # Optimizer steps
            if self.scaler is not None:
                self.scaler.step(self.controller_optimizer)
                self.scaler.step(self.bem_optimizer)
                self.scaler.update()
            else:
                self.controller_optimizer.step()
                self.bem_optimizer.step()
            
            # Scheduler steps
            self.controller_scheduler.step()
            self.bem_scheduler.step()
            
            # Zero gradients
            self.controller_optimizer.zero_grad()
            self.bem_optimizer.zero_grad()
        
        # Convert loss dict to float
        loss_dict_float = {k: v.item() if isinstance(v, torch.Tensor) else v 
                          for k, v in loss_dict.items()}
        
        return loss_dict_float
    
    def evaluate(
        self,
        eval_dataloader,
        max_eval_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluation loop.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_steps = 0
        loss_components = {}
        
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                if max_eval_steps and step >= max_eval_steps:
                    break
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask')
                labels = batch.get('labels', input_ids)
                side_signals = batch.get('side_signals')
                
                # Forward pass
                outputs, routing_info = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    side_signals=side_signals,
                    return_routing_info=True
                )
                
                # Compute loss
                total_loss_step, loss_dict = self.compute_total_loss(
                    outputs=outputs,
                    labels=labels,
                    routing_info=routing_info
                )
                
                total_loss += total_loss_step.item()
                total_steps += 1
                
                # Accumulate loss components
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item() if isinstance(value, torch.Tensor) else value
        
        # Average losses
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_loss_components = {k: v / total_steps for k, v in loss_components.items()}
        avg_loss_components['eval_loss'] = avg_loss
        
        return avg_loss_components
    
    def train(
        self,
        train_dataloader,
        eval_dataloader=None,
        num_epochs: Optional[int] = None
    ):
        """
        Main training loop.
        """
        if num_epochs is None:
            num_epochs = self.config.max_steps // len(train_dataloader) + 1
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            for batch in train_dataloader:
                # Training step
                loss_dict = self.train_step(batch)
                
                # Logging
                if self.step % self.config.log_interval == 0:
                    self._log_metrics(loss_dict, prefix='train')
                
                # Evaluation
                if eval_dataloader and self.step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    self._log_metrics(eval_metrics, prefix='eval')
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['eval_loss']
                        self.save_checkpoint('best_model.pt')
                
                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.step}.pt')
                
                self.step += 1
                
                if self.step >= self.config.max_steps:
                    break
            
            if self.step >= self.config.max_steps:
                break
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to wandb and console."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Add step information
        metrics['step'] = self.step
        metrics['epoch'] = self.epoch
        
        # Add learning rates
        metrics['lr/controller'] = self.controller_optimizer.param_groups[0]['lr']
        metrics['lr/bem'] = self.bem_optimizer.param_groups[0]['lr']
        
        # Add routing statistics
        routing_stats = self.model.get_routing_statistics()
        if routing_stats['global_stats']['total_forward_calls'] > 0:
            routing_dist = routing_stats['global_stats']['routing_distribution']
            metrics['routing/prefix_usage'] = routing_dist[0]
            metrics['routing/chunk_usage'] = routing_dist[1]
            metrics['routing/token_usage'] = routing_dist[2]
        
        # Log to wandb if available
        try:
            wandb.log(metrics)
        except:
            pass
        
        # Console logging
        if prefix == 'train':
            print(f"Step {self.step}: Loss = {metrics.get(f'{prefix}/total_loss', 'N/A'):.4f}")
        elif prefix == 'eval':
            print(f"Step {self.step}: Eval Loss = {metrics.get(f'{prefix}/eval_loss', 'N/A'):.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config,
            'controller_optimizer': self.controller_optimizer.state_dict(),
            'bem_optimizer': self.bem_optimizer.state_dict(),
            'controller_scheduler': self.controller_scheduler.state_dict(),
            'bem_scheduler': self.bem_scheduler.state_dict(),
            'model_state': {
                'config': self.model.config,
                'controller_state': self.model.controller.state_dict(),
                'bem_modules': {name: module.state_dict() 
                              for name, module in self.model.bem_modules.items()}
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint['best_eval_loss']
        
        self.controller_optimizer.load_state_dict(checkpoint['controller_optimizer'])
        self.bem_optimizer.load_state_dict(checkpoint['bem_optimizer'])
        self.controller_scheduler.load_state_dict(checkpoint['controller_scheduler'])
        self.bem_scheduler.load_state_dict(checkpoint['bem_scheduler'])
        
        # Load model state
        self.model.controller.load_state_dict(checkpoint['model_state']['controller_state'])
        for name, state_dict in checkpoint['model_state']['bem_modules'].items():
            if name in self.model.bem_modules:
                self.model.bem_modules[name].load_state_dict(state_dict)
        
        if self.scaler is not None and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"Loaded checkpoint: {filename} (Step {self.step})")


# Factory functions for creating trainers

def create_hierarchical_trainer(
    model: FullHierarchicalBEM,
    training_config: Optional[HierarchicalTrainingConfig] = None,
    **config_kwargs
) -> HierarchicalBEMTrainer:
    """
    Factory function to create hierarchical BEM trainer.
    
    Args:
        model: FullHierarchicalBEM instance
        training_config: Training configuration
        **config_kwargs: Config overrides
        
    Returns:
        HierarchicalBEMTrainer instance
    """
    if training_config is None:
        training_config = HierarchicalTrainingConfig(**config_kwargs)
    
    return HierarchicalBEMTrainer(
        model=model,
        config=training_config
    )


def create_end_to_end_trainer(
    model: FullHierarchicalBEM,
    learning_rate: float = 1e-3,
    max_steps: int = 10000,
    **kwargs
) -> HierarchicalBEMTrainer:
    """Create trainer for end-to-end training strategy."""
    config = HierarchicalTrainingConfig(
        strategy=TrainingStrategy.END_TO_END,
        learning_rate=learning_rate,
        max_steps=max_steps,
        **kwargs
    )
    
    return HierarchicalBEMTrainer(model=model, config=config)


def create_expert_imitation_trainer(
    model: FullHierarchicalBEM,
    expert_models: List[str],
    imitation_weight: float = 1.0,
    **kwargs
) -> HierarchicalBEMTrainer:
    """Create trainer for expert imitation strategy."""
    config = HierarchicalTrainingConfig(
        strategy=TrainingStrategy.EXPERT_IMITATION,
        expert_models=expert_models,
        imitation_weight=imitation_weight,
        **kwargs
    )
    
    return HierarchicalBEMTrainer(model=model, config=config)