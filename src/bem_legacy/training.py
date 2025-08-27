"""
Training utilities for BEM validation experiment.
Includes trainers for static LoRAs and the interpolation controller.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from tqdm import tqdm
import json
import os
from dataclasses import dataclass
import time
import math

from .interpolation_bem import StaticLoRA, InterpolationBEM
from .simple_bem import analyze_code_distribution, compute_effective_rank


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    
    # BEM-specific
    kl_divergence_weight: float = 0.1
    entropy_regularization: float = 0.01
    orthogonality_weight: float = 0.001


class LoRATrainer:
    """Trainer for static LoRA modules."""
    
    def __init__(
        self,
        model: nn.Module,
        lora_modules: Dict[str, StaticLoRA],
        config: TrainingConfig,
        device: torch.device,
        output_dir: str
    ):
        self.model = model
        self.lora_modules = lora_modules
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Setup optimizer
        lora_params = []
        for lora in lora_modules.values():
            lora_params.extend(list(lora.parameters()))
        
        self.optimizer = optim.AdamW(
            lora_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        task_name: str = "lora_training"
    ) -> Dict[str, Any]:
        """Train the LoRA modules."""
        
        self.logger.info(f"Starting {task_name} training...")
        self.logger.info(f"Training steps per epoch: {len(train_dataloader)}")
        
        training_stats = {
            'train_losses': [],
            'eval_losses': [],
            'perplexities': [],
            'learning_rates': []
        }
        
        # Setup scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training loop
            train_loss = self._train_epoch(train_dataloader, scheduler)
            training_stats['train_losses'].append(train_loss)
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self._evaluate(eval_dataloader)
                training_stats['eval_losses'].append(eval_metrics['loss'])
                training_stats['perplexities'].append(eval_metrics['perplexity'])
                
                self.logger.info(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                    f"eval_loss={eval_metrics['loss']:.4f}, "
                    f"perplexity={eval_metrics['perplexity']:.4f}"
                )
            else:
                self.logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")
        
        # Save final model
        self._save_checkpoint(f"{task_name}_final")
        
        # Save training stats
        with open(os.path.join(self.output_dir, f"{task_name}_stats.json"), 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        return training_stats
    
    def _train_epoch(self, dataloader: DataLoader, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for lora in self.lora_modules.values() for p in lora.parameters()],
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / num_batches
    
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save LoRA checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for name, lora in self.lora_modules.items():
            torch.save(
                lora.state_dict(),
                os.path.join(checkpoint_dir, f"{name}_lora.pt")
            )
        
        # Save training state
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
        }, os.path.join(checkpoint_dir, "training_state.pt"))


class BEMTrainer:
    """Trainer for BEM interpolation controller."""
    
    def __init__(
        self,
        model: nn.Module,
        bem_modules: Dict[str, InterpolationBEM],
        config: TrainingConfig,
        device: torch.device,
        output_dir: str
    ):
        self.model = model
        self.bem_modules = bem_modules
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Setup optimizer for controller parameters only
        controller_params = []
        for bem in bem_modules.values():
            controller_params.extend(list(bem.controller.parameters()))
        
        self.optimizer = optim.AdamW(
            controller_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        feature_extractor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train the BEM controllers."""
        
        self.logger.info("Starting BEM controller training...")
        self.logger.info(f"Training steps per epoch: {len(train_dataloader)}")
        
        training_stats = {
            'train_losses': [],
            'eval_losses': [],
            'specialization_scores': [],
            'interpolation_entropies': [],
            'learning_rates': []
        }
        
        # Setup scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training loop
            epoch_stats = self._train_epoch(train_dataloader, scheduler, feature_extractor)
            training_stats['train_losses'].append(epoch_stats['train_loss'])
            training_stats['specialization_scores'].append(epoch_stats['specialization_score'])
            training_stats['interpolation_entropies'].append(epoch_stats['interpolation_entropy'])
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self._evaluate(eval_dataloader, feature_extractor)
                training_stats['eval_losses'].append(eval_metrics['loss'])
                
                self.logger.info(
                    f"Epoch {epoch + 1}: train_loss={epoch_stats['train_loss']:.4f}, "
                    f"eval_loss={eval_metrics['loss']:.4f}, "
                    f"specialization={epoch_stats['specialization_score']:.4f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}: train_loss={epoch_stats['train_loss']:.4f}, "
                    f"specialization={epoch_stats['specialization_score']:.4f}"
                )
        
        # Save final model
        self._save_checkpoint("bem_final")
        
        # Save training stats
        with open(os.path.join(self.output_dir, "bem_training_stats.json"), 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        return training_stats
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        scheduler,
        feature_extractor: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_specialization = 0.0
        total_entropy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training BEM")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            task_type = batch.get('task_type', 'unknown')
            
            # Extract features for controller
            if feature_extractor is not None:
                features = feature_extractor(input_ids, attention_mask)
            else:
                # Use mean pooled embeddings as features
                with torch.no_grad():
                    embeddings = self.model.get_input_embeddings()(input_ids)
                    features = embeddings.mean(dim=1)  # [batch_size, hidden_dim]
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Base loss
            base_loss = outputs.loss
            
            # Compute BEM-specific losses
            bem_losses = self._compute_bem_losses(features, task_type)
            
            # Total loss
            loss = (
                base_loss +
                self.config.entropy_regularization * bem_losses['entropy_loss'] +
                self.config.orthogonality_weight * bem_losses['orthogonality_loss']
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for bem in self.bem_modules.values() for p in bem.controller.parameters()],
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_specialization += bem_losses['specialization_score']
            total_entropy += bem_losses['avg_entropy']
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'spec': f'{bem_losses["specialization_score"]:.3f}',
                    'ent': f'{bem_losses["avg_entropy"]:.3f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(f"bem_step_{self.global_step}")
        
        return {
            'train_loss': total_loss / num_batches,
            'specialization_score': total_specialization / num_batches,
            'interpolation_entropy': total_entropy / num_batches
        }
    
    def _compute_bem_losses(self, features: torch.Tensor, task_types: List[str]) -> Dict[str, float]:
        """Compute BEM-specific losses."""
        losses = {
            'entropy_loss': 0.0,
            'orthogonality_loss': 0.0,
            'specialization_score': 0.0,
            'avg_entropy': 0.0
        }
        
        all_weights = []
        task_specializations = []
        
        for name, bem in self.bem_modules.items():
            # Get interpolation weights
            weights = bem.get_interpolation_weights(features)
            all_weights.append(weights)
            
            # Compute entropy (we want moderate entropy, not too low or too high)
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            losses['entropy_loss'] += torch.abs(entropy - 0.5)  # Target entropy of 0.5
            losses['avg_entropy'] += entropy.item()
            
            # Compute task specialization
            json_mask = torch.tensor([t == 'json' for t in task_types], device=features.device)
            summary_mask = torch.tensor([t == 'summary' for t in task_types], device=features.device)
            
            if json_mask.sum() > 0:
                json_specialization = weights[json_mask, 0].mean()  # Should prefer JSON LoRA
                task_specializations.append(json_specialization.item())
            
            if summary_mask.sum() > 0:
                summary_specialization = weights[summary_mask, 1].mean()  # Should prefer summary LoRA
                task_specializations.append(summary_specialization.item())
        
        # Average across modules
        num_modules = len(self.bem_modules)
        losses['entropy_loss'] /= num_modules
        losses['avg_entropy'] /= num_modules
        
        # Specialization score
        if task_specializations:
            losses['specialization_score'] = sum(task_specializations) / len(task_specializations)
        
        # Orthogonality loss between different BEM modules (if multiple)
        if len(all_weights) > 1:
            ortho_loss = 0.0
            count = 0
            for i in range(len(all_weights)):
                for j in range(i + 1, len(all_weights)):
                    # Encourage different modules to make different decisions
                    similarity = F.cosine_similarity(all_weights[i], all_weights[j], dim=1).mean()
                    ortho_loss += similarity ** 2
                    count += 1
            
            if count > 0:
                losses['orthogonality_loss'] = ortho_loss / count
        
        return losses
    
    def _evaluate(
        self,
        dataloader: DataLoader,
        feature_extractor: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Evaluate the BEM controllers."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating BEM"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save BEM checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for name, bem in self.bem_modules.items():
            torch.save(
                bem.state_dict(),
                os.path.join(checkpoint_dir, f"{name}_bem.pt")
            )
        
        # Save training state
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
        }, os.path.join(checkpoint_dir, "bem_training_state.pt"))