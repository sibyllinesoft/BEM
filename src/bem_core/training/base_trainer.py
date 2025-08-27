"""Base trainer class providing unified training infrastructure.

This module defines the abstract base trainer that all BEM components inherit from,
standardizing training loops, logging, checkpointing, and evaluation across variants.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler

from ..utils.logging_utils import setup_logger
from ..utils.checkpoint_utils import save_checkpoint, load_checkpoint
from ..config.base_config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training parameters."""
    
    # Core training parameters
    learning_rate: float = 5e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    max_epochs: Optional[int] = None
    warmup_steps: int = 100
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    scheduler_type: str = "linear"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation and logging
    eval_strategy: str = "steps"  # "steps", "epoch", or "no"
    eval_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 500
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    # Mixed precision and optimization
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    
    # Device and distributed training
    device: Optional[str] = None
    local_rank: int = -1
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


class BaseTrainer(ABC):
    """Abstract base trainer for all BEM components.
    
    Provides standardized training infrastructure including:
    - Configuration loading and validation
    - Model and optimizer setup
    - Training loop with hooks for customization
    - Evaluation and metrics tracking
    - Checkpointing and resumption
    - Logging and monitoring
    
    Subclasses must implement:
    - _setup_model(): Initialize the specific model architecture
    - _compute_loss(): Compute component-specific loss function
    - _evaluate(): Run component-specific evaluation
    """
    
    def __init__(
        self,
        config: Union[TrainingConfig, Dict[str, Any], str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
    ):
        """Initialize the base trainer.
        
        Args:
            config: Training configuration (object, dict, or path to config file)
            output_dir: Output directory for logs and checkpoints
            experiment_name: Name for the experiment (used in logging)
        """
        # Load and validate configuration
        self.config = self._load_config(config)
        self.output_dir = Path(output_dir or f"logs/{experiment_name or 'experiment'}")
        self.experiment_name = experiment_name or "experiment"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger(
            name=f"bem_trainer_{self.experiment_name}",
            log_file=self.output_dir / "training.log",
            level=logging.INFO
        )
        
        # Set device
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.config.device)
        
        # Set random seeds for reproducibility
        if self.config.seed is not None:
            self._set_seed(self.config.seed)
        
        # Initialize training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        
        # Training tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.metrics_history = []
        self.training_start_time = None
        
        self.logger.info(f"Initialized {self.__class__.__name__} for {self.experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
    
    def _load_config(self, config: Union[TrainingConfig, Dict, str, Path]) -> TrainingConfig:
        """Load and validate training configuration."""
        if isinstance(config, TrainingConfig):
            return config
        elif isinstance(config, dict):
            return TrainingConfig(**config)
        elif isinstance(config, (str, Path)):
            # Load from file (implementation depends on config system)
            from ..config.config_loader import load_training_config
            return load_training_config(config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @abstractmethod
    def _setup_model(self) -> nn.Module:
        """Set up the model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod 
    def _compute_loss(self, batch: Dict[str, Any], model_outputs: Any) -> Dict[str, torch.Tensor]:
        """Compute the loss for a batch. Must be implemented by subclasses.
        
        Args:
            batch: Input batch data
            model_outputs: Model forward pass outputs
            
        Returns:
            Dictionary with 'loss' key and optional auxiliary losses
        """
        pass
    
    @abstractmethod
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation on the given dataloader. Must be implemented by subclasses.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def setup_training(
        self, 
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ) -> None:
        """Set up training components.
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
        """
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Set up model
        self.model = self._setup_model()
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Set up optimizer
        self.optimizer = self._setup_optimizer()
        
        # Set up learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.logger.info("Training setup completed")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Set up the optimizer."""
        # Group parameters by weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Set up the learning rate scheduler."""
        if self.config.scheduler_type == "none":
            return None
        
        num_training_steps = self._get_num_training_steps()
        
        return get_scheduler(
            self.config.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
            **self.config.lr_scheduler_kwargs
        )
    
    def _get_num_training_steps(self) -> int:
        """Calculate the total number of training steps."""
        if self.config.max_steps > 0:
            return self.config.max_steps
        
        if self.config.max_epochs and self.train_dataloader:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            return steps_per_epoch * self.config.max_epochs
        
        return 1000  # Default fallback
    
    def train(self) -> Dict[str, Any]:
        """Main training loop.
        
        Returns:
            Training results and metrics
        """
        if not self.model or not self.train_dataloader:
            raise RuntimeError("Must call setup_training() before train()")
        
        self.logger.info("Starting training")
        self.training_start_time = time.time()
        
        # Training loop
        self.model.train()
        early_stopping_counter = 0
        
        for epoch in range(self.config.max_epochs or float('inf')):
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
                
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch()
            
            # Evaluation
            if self._should_evaluate():
                eval_metrics = self._run_evaluation()
                epoch_metrics.update(eval_metrics)
                
                # Check for improvement
                current_metric = eval_metrics.get("eval_loss", float('inf'))
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    early_stopping_counter = 0
                    
                    # Save best model
                    self._save_checkpoint(is_best=True)
                else:
                    early_stopping_counter += 1
            
            # Log epoch metrics
            self._log_metrics(epoch_metrics, step=self.global_step)
            self.metrics_history.append(epoch_metrics)
            
            # Early stopping
            if (self.config.early_stopping_patience and 
                early_stopping_counter >= self.config.early_stopping_patience):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()
        
        # Training completion
        total_time = time.time() - self.training_start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        return {
            "total_steps": self.global_step,
            "total_epochs": self.current_epoch + 1,
            "total_time": total_time,
            "best_metric": self.best_metric,
            "metrics_history": self.metrics_history,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = []
        
        for step, batch in enumerate(self.train_dataloader):
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            model_outputs = self.model(**batch)
            loss_dict = self._compute_loss(batch, model_outputs)
            loss = loss_dict["loss"] / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.fp16:
                # Use automatic mixed precision
                from torch.cuda.amp import autocast, GradScaler
                with autocast():
                    loss.backward()
            else:
                loss.backward()
            
            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                    self.logger.info(
                        f"Step {self.global_step}: loss={loss.item():.4f}, lr={lr:.2e}"
                    )
        
        return {
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "epoch": self.current_epoch,
        }
    
    def _should_evaluate(self) -> bool:
        """Check if evaluation should be run."""
        if not self.eval_dataloader or self.config.eval_strategy == "no":
            return False
        
        if self.config.eval_strategy == "steps":
            return self.global_step % self.config.eval_steps == 0
        elif self.config.eval_strategy == "epoch":
            return True
        
        return False
    
    def _run_evaluation(self) -> Dict[str, float]:
        """Run evaluation and return metrics."""
        self.logger.info("Running evaluation...")
        
        self.model.eval()
        with torch.no_grad():
            eval_metrics = self._evaluate(self.eval_dataloader)
        self.model.train()
        
        # Add eval_ prefix to metrics
        eval_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
        
        return eval_metrics
    
    def _log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to console and files."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{key}: {value}")
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / (
            "best_model.pt" if is_best else f"checkpoint-{self.global_step}.pt"
        )
        
        save_checkpoint(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "best_metric": self.best_metric,
                "config": self.config.__dict__,
            },
            checkpoint_path
        )
        
        self.logger.info(f"Saved {'best model' if is_best else 'checkpoint'} to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_metric = checkpoint["best_metric"]
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (step {self.global_step})")
