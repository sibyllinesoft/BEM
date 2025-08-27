"""
BEM v1.1 Trainer Implementation

Training pipeline for BEM-v1.1-stable with full governance, mixed precision,
and cache-aware metrics according to TODO.md specifications.

Key features:
- Spectral + Frobenius governance
- Mixed precision training (bf16/fp16)
- Cache metrics logging (KV hit%, flips/token, etc.)
- PEFT framework compatibility
- 24GB VRAM budget awareness
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from typing import Dict, List, Optional, Any, Union
import os
import json
import time
import math
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

from ..models import BEMv11Model
from .cache_metrics import CacheMetricsCollector


@dataclass
class BEMv11TrainingConfig:
    """Training configuration for BEM v1.1 with governance."""
    
    # Base training parameters
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # Mixed precision settings
    fp16: bool = False
    bf16: bool = True  # Preferred for stability
    dataloader_pin_memory: bool = True
    
    # BEM-specific parameters
    governance_weight: float = 0.1
    entropy_regularization: float = 0.01
    flip_penalty_weight: float = 0.1
    decorrelation_weight: float = 0.01
    
    # Cache and memory management
    chunk_size: int = 128
    max_context_length: int = 4096
    vram_budget_gb: float = 24.0
    
    # Logging and checkpointing  
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Output settings
    output_dir: str = "./outputs/bem_v11_training"
    run_name: Optional[str] = None
    project_name: str = "bem-v11-stable"


class BEMv11Trainer(Trainer):
    """
    Custom trainer for BEM v1.1 with governance and cache metrics.
    
    Extends HuggingFace Trainer with BEM-specific functionality:
    - Governance penalty integration
    - Cache metrics collection
    - Memory-aware training
    """
    
    def __init__(
        self,
        model: BEMv11Model,
        args: TrainingArguments,
        bem_config: BEMv11TrainingConfig,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs
        )
        
        self.bem_config = bem_config
        self.cache_metrics = CacheMetricsCollector()
        
        # Initialize governance tracking
        self.governance_stats = {
            'total_steps': 0,
            'governance_penalties': [],
            'cache_metrics': [],
            'flip_rates': [],
            'expert_utilizations': []
        }
        
        # Memory monitoring
        self.peak_memory_mb = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute BEM loss with governance penalties.
        
        Integrates base language modeling loss with BEM governance terms.
        """
        # Extract retrieval context if available
        retrieval_context = inputs.pop('retrieval_context', None)
        
        # Forward pass with auxiliary info for governance
        outputs = model(
            **inputs,
            retrieval_context=retrieval_context,
            return_aux_info=True
        )
        
        # Base language modeling loss
        base_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)
        
        # Governance penalties
        governance_penalty = getattr(outputs, 'total_governance_penalty', torch.tensor(0.0))
        
        # Total loss
        total_loss = base_loss + self.bem_config.governance_weight * governance_penalty
        
        # Collect metrics for logging
        if self.state.global_step % self.bem_config.logging_steps == 0:
            self._collect_training_metrics(outputs)
        
        # Memory monitoring
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _collect_training_metrics(self, outputs):
        """Collect BEM-specific training metrics."""
        metrics = {
            'step': self.state.global_step,
            'base_loss': outputs.loss.item() if outputs.loss is not None else 0.0,
            'governance_penalty': getattr(outputs, 'total_governance_penalty', torch.tensor(0.0)).item(),
            'peak_memory_mb': self.peak_memory_mb
        }
        
        # Governance statistics
        if hasattr(outputs, 'governance_summary'):
            gov_stats = outputs.governance_summary
            metrics.update({
                f'gov_{k}': v for k, v in gov_stats.items()
                if isinstance(v, (int, float))
            })
        
        # Cache metrics from BEM auxiliary info
        if hasattr(outputs, 'bem_aux_info'):
            cache_stats = self.cache_metrics.collect_from_bem_output(outputs.bem_aux_info)
            metrics.update(cache_stats)
        
        # Store metrics
        self.governance_stats['governance_penalties'].append(metrics.get('governance_penalty', 0.0))
        if 'flip_rate' in metrics:
            self.governance_stats['flip_rates'].append(metrics['flip_rate'])
        
        # Log to console and file
        if self.state.global_step % (self.bem_config.logging_steps * 5) == 0:
            self._log_metrics_summary(metrics)
    
    def _log_metrics_summary(self, metrics):
        """Log comprehensive metrics summary."""
        print(f"\n📊 BEM v1.1 Training Metrics (Step {metrics['step']}):")
        print(f"  Base Loss: {metrics['base_loss']:.4f}")
        print(f"  Governance Penalty: {metrics['governance_penalty']:.6f}")
        
        if 'flip_rate' in metrics:
            print(f"  Routing Flip Rate: {metrics['flip_rate']:.3f}")
        if 'gate_entropy' in metrics:
            print(f"  Gate Entropy: {metrics['gate_entropy']:.3f}")
        if 'expert_utilization_std' in metrics:
            print(f"  Expert Util. Std: {metrics['expert_utilization_std']:.3f}")
        
        print(f"  Memory Usage: {metrics['peak_memory_mb']:.1f} MB")
        
        # Cache safety check
        cache_violation_rate = metrics.get('cache_violation_rate', 0.0)
        if cache_violation_rate > 0.01:  # >1% violations
            print(f"  ⚠️  Cache violations: {cache_violation_rate:.2%}")
        else:
            print(f"  ✅ Cache safety: {1-cache_violation_rate:.2%}")
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Enhanced evaluation loop with BEM metrics.
        """
        # Call parent evaluation
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Add BEM-specific evaluation metrics
        bem_eval_metrics = self._compute_bem_eval_metrics(dataloader)
        
        # Merge metrics
        if hasattr(output, 'metrics'):
            output.metrics.update({f"{metric_key_prefix}_{k}": v for k, v in bem_eval_metrics.items()})
        
        return output
    
    def _compute_bem_eval_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """Compute BEM-specific evaluation metrics."""
        self.model.eval()
        
        all_cache_metrics = []
        all_governance_stats = []
        total_tokens = 0
        total_inference_time = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="BEM Evaluation"):
                batch = self._prepare_inputs(batch)
                batch_size, seq_len = batch['input_ids'].shape
                
                # Time inference
                start_time = time.time()
                
                outputs = self.model(
                    **batch,
                    retrieval_context=batch.get('retrieval_context'),
                    return_aux_info=True
                )
                
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                total_tokens += batch_size * seq_len
                
                # Collect cache metrics
                if hasattr(outputs, 'bem_aux_info'):
                    cache_stats = self.cache_metrics.collect_from_bem_output(outputs.bem_aux_info)
                    all_cache_metrics.append(cache_stats)
                
                # Collect governance stats
                if hasattr(outputs, 'governance_summary'):
                    all_governance_stats.append(outputs.governance_summary)
        
        # Aggregate metrics
        eval_metrics = {}
        
        # Cache metrics
        if all_cache_metrics:
            cache_keys = all_cache_metrics[0].keys()
            for key in cache_keys:
                values = [m[key] for m in all_cache_metrics if key in m]
                if values:
                    eval_metrics[f'cache_{key}'] = np.mean(values)
        
        # Governance metrics  
        if all_governance_stats:
            gov_keys = all_governance_stats[0].keys()
            for key in gov_keys:
                values = [g[key] for g in all_governance_stats if key in g]
                if values:
                    eval_metrics[f'gov_{key}'] = np.mean(values)
        
        # Performance metrics
        if total_tokens > 0:
            eval_metrics['tokens_per_second'] = total_tokens / total_inference_time
            eval_metrics['latency_per_token_ms'] = (total_inference_time / total_tokens) * 1000
        
        return eval_metrics
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save BEM model with comprehensive metadata.
        """
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Save BEM-specific information
        bem_info = {
            'bem_config': self.bem_config.__dict__,
            'architecture': 'BEM-v1.1-stable',
            'governance_stats': self.governance_stats,
            'cache_safety_report': self.model.get_cache_safety_report(),
            'training_completed': True
        }
        
        with open(os.path.join(output_dir, 'bem_training_info.json'), 'w') as f:
            json.dump(bem_info, f, indent=2, default=str)
        
        print(f"💾 BEM v1.1 model saved to {output_dir}")
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader with BEM-aware settings."""
        dataloader = super().get_train_dataloader()
        
        # Monitor memory usage of first batch
        if hasattr(self, '_memory_checked') and not self._memory_checked:
            self._check_memory_usage(dataloader)
            self._memory_checked = True
        
        return dataloader
    
    def _check_memory_usage(self, dataloader):
        """Check if training fits within VRAM budget."""
        if not torch.cuda.is_available():
            return
        
        # Test forward pass with one batch
        try:
            batch = next(iter(dataloader))
            batch = self._prepare_inputs(batch)
            
            torch.cuda.reset_peak_memory_stats()
            
            with torch.cuda.amp.autocast(enabled=self.bem_config.fp16 or self.bem_config.bf16):
                outputs = self.model(**batch, return_aux_info=True)
                loss = self.compute_loss(self.model, batch)
                loss.backward()
            
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            
            print(f"\n🔍 Memory Usage Check:")
            print(f"  Peak VRAM: {peak_memory_gb:.2f} GB")
            print(f"  Budget: {self.bem_config.vram_budget_gb:.2f} GB")
            
            if peak_memory_gb > self.bem_config.vram_budget_gb:
                print(f"  ⚠️  Memory usage exceeds budget!")
                print(f"  💡 Consider: reduce batch_size, use gradient checkpointing, or enable mixed precision")
            else:
                print(f"  ✅ Within memory budget ({peak_memory_gb/self.bem_config.vram_budget_gb:.1%} used)")
                
        except Exception as e:
            print(f"  ❌ Memory check failed: {e}")


def create_bem_v11_trainer(
    model: BEMv11Model,
    train_dataset,
    eval_dataset=None,
    bem_config: Optional[BEMv11TrainingConfig] = None,
    **kwargs
) -> BEMv11Trainer:
    """
    Factory function to create BEM v1.1 trainer with proper configuration.
    
    Args:
        model: BEM v1.1 model
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        bem_config: BEM training configuration
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured BEM v1.1 trainer
    """
    if bem_config is None:
        bem_config = BEMv11TrainingConfig()
    
    # Create HuggingFace TrainingArguments
    training_args = TrainingArguments(
        output_dir=bem_config.output_dir,
        learning_rate=bem_config.learning_rate,
        weight_decay=bem_config.weight_decay,
        warmup_ratio=bem_config.warmup_ratio,
        num_train_epochs=bem_config.num_epochs,
        per_device_train_batch_size=bem_config.batch_size,
        gradient_accumulation_steps=bem_config.gradient_accumulation_steps,
        max_grad_norm=bem_config.max_grad_norm,
        fp16=bem_config.fp16,
        bf16=bem_config.bf16,
        dataloader_pin_memory=bem_config.dataloader_pin_memory,
        logging_steps=bem_config.logging_steps,
        eval_steps=bem_config.eval_steps,
        save_steps=bem_config.save_steps,
        save_total_limit=bem_config.save_total_limit,
        run_name=bem_config.run_name or "bem-v11-training",
        report_to=["wandb"] if bem_config.run_name else [],
        **kwargs
    )
    
    trainer = BEMv11Trainer(
        model=model,
        args=training_args,
        bem_config=bem_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    return trainer