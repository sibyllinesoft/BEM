"""Unified router trainer using bem_core infrastructure.

Demonstrates how to migrate existing BEM components to use the unified
core infrastructure, reducing code duplication while preserving functionality.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader

from ...bem_core.training.base_trainer import BaseTrainer, TrainingConfig
from ...bem_core.evaluation.base_evaluator import BaseEvaluator, EvaluationConfig
from ...bem_core.config.base_config import ExperimentConfig
from .agentic_router import AgenticRouter
from .composition_engine import CompositionEngine


class RouterTrainer(BaseTrainer):
    """Unified trainer for agentic router components.
    
    Inherits from BaseTrainer to provide standardized training infrastructure
    while implementing router-specific model setup and loss computation.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Initialize router trainer.
        
        Args:
            config: Complete experiment configuration
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
        
        # Store experiment config for router-specific setup
        self.experiment_config = config
        
        # Router-specific components
        self.composition_engine = None
        self.router_metrics = {}
    
    def _setup_model(self) -> nn.Module:
        """Set up the agentic router model."""
        router_config = self.experiment_config.model.custom_params
        
        # Initialize base model
        base_model_name = self.experiment_config.model.base_model
        
        # Create composition engine
        self.composition_engine = CompositionEngine(
            base_model_name=base_model_name,
            num_experts=router_config.get('num_experts', 8),
            expert_dim=router_config.get('expert_dim', 768),
            **router_config.get('composition_params', {})
        )
        
        # Create agentic router
        router = AgenticRouter(
            composition_engine=self.composition_engine,
            policy_type=router_config.get('policy_type', 'pg'),
            temperature=router_config.get('temperature', 1.0),
            **router_config.get('router_params', {})
        )
        
        self.logger.info(f"Initialized AgenticRouter with {router_config.get('num_experts', 8)} experts")
        
        return router
    
    def _compute_loss(self, batch: Dict[str, Any], model_outputs: Any) -> Dict[str, torch.Tensor]:
        """Compute router-specific loss function."""
        # Extract outputs from agentic router
        routing_logits = model_outputs.get('routing_logits')
        composition_outputs = model_outputs.get('composition_outputs')
        policy_outputs = model_outputs.get('policy_outputs')
        
        # Compute primary task loss (e.g., language modeling)
        task_loss = self._compute_task_loss(batch, composition_outputs)
        
        # Compute routing loss (expert selection)
        routing_loss = self._compute_routing_loss(routing_logits, batch)
        
        # Compute policy loss (for reinforcement learning)
        policy_loss = self._compute_policy_loss(policy_outputs, batch)
        
        # Combine losses with weighting
        router_config = self.experiment_config.model.custom_params
        task_weight = router_config.get('task_loss_weight', 1.0)
        routing_weight = router_config.get('routing_loss_weight', 0.1)
        policy_weight = router_config.get('policy_loss_weight', 0.1)
        
        total_loss = (
            task_weight * task_loss +
            routing_weight * routing_loss +
            policy_weight * policy_loss
        )
        
        return {
            'loss': total_loss,
            'task_loss': task_loss,
            'routing_loss': routing_loss,
            'policy_loss': policy_loss,
        }
    
    def _compute_task_loss(self, batch: Dict[str, Any], outputs: Any) -> torch.Tensor:
        """Compute primary task loss (e.g., language modeling)."""
        if hasattr(outputs, 'loss'):
            return outputs.loss
        
        # Fallback: compute cross-entropy loss
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        labels = batch.get('labels')
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    def _compute_routing_loss(self, routing_logits: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute routing loss for expert selection."""
        if routing_logits is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Load balancing loss - encourage uniform expert usage
        routing_probs = torch.softmax(routing_logits, dim=-1)
        expert_usage = routing_probs.mean(dim=0)
        num_experts = routing_probs.size(-1)
        
        # Compute load balancing loss (encourage uniform distribution)
        uniform_dist = torch.ones_like(expert_usage) / num_experts
        load_balance_loss = torch.nn.functional.kl_div(
            expert_usage.log(), uniform_dist, reduction='batchmean'
        )
        
        return load_balance_loss
    
    def _compute_policy_loss(self, policy_outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute policy gradient loss for reinforcement learning."""
        if policy_outputs is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Extract policy components
        log_probs = policy_outputs.get('log_probs')
        rewards = policy_outputs.get('rewards')
        baselines = policy_outputs.get('baselines')
        
        if log_probs is None or rewards is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute advantage (reward - baseline)
        advantages = rewards
        if baselines is not None:
            advantages = rewards - baselines
        
        # Policy gradient loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        return policy_loss
    
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run router-specific evaluation."""
        # Create router evaluator
        eval_config = EvaluationConfig(
            batch_size=self.config.batch_size,
            metrics=['routing_accuracy', 'expert_utilization', 'composition_quality'],
            measure_latency=True,
            measure_memory=True,
        )
        
        evaluator = RouterEvaluator(
            model=self.model,
            config=eval_config,
            device=self.device,
            logger=self.logger,
        )
        
        result = evaluator.evaluate(dataloader)
        return result.metrics


class RouterEvaluator(BaseEvaluator):
    """Evaluator for agentic router components.
    
    Inherits from BaseEvaluator to provide standardized evaluation infrastructure
    while implementing router-specific metrics and inference.
    """
    
    def _run_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run router inference on a batch."""
        # Forward pass through router
        outputs = self.model(**batch)
        
        # Extract predictions and routing decisions
        predictions = []
        routing_decisions = []
        
        if hasattr(outputs, 'logits'):
            # For language modeling tasks
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predictions = predicted_ids.cpu().tolist()
        
        if hasattr(outputs, 'routing_logits'):
            # Extract expert routing decisions
            routing_logits = outputs.routing_logits
            expert_choices = torch.argmax(routing_logits, dim=-1)
            routing_decisions = expert_choices.cpu().tolist()
        
        return {
            'predictions': predictions,
            'routing_decisions': routing_decisions,
            'model_outputs': outputs,
        }
    
    def _compute_component_metrics(
        self, 
        predictions: List[Any], 
        targets: List[Any]
    ) -> Dict[str, float]:
        """Compute router-specific metrics."""
        metrics = {}
        
        # Extract routing decisions if available
        routing_decisions = []
        for pred in predictions:
            if isinstance(pred, dict) and 'routing_decisions' in pred:
                routing_decisions.extend(pred['routing_decisions'])
        
        if routing_decisions:
            # Compute expert utilization
            import numpy as np
            routing_array = np.array(routing_decisions)
            num_experts = len(np.unique(routing_array))
            
            # Utilization entropy (higher is better for load balancing)
            expert_counts = np.bincount(routing_array)
            expert_probs = expert_counts / len(routing_array)
            utilization_entropy = -np.sum(expert_probs * np.log(expert_probs + 1e-8))
            
            metrics['expert_utilization'] = float(utilization_entropy)
            metrics['num_active_experts'] = float(num_experts)
            metrics['max_expert_usage'] = float(expert_probs.max())
            metrics['min_expert_usage'] = float(expert_probs.min())
        
        # Add routing accuracy if ground truth routing is available
        # (This would require special evaluation data with optimal routing labels)
        
        return metrics


# Example usage and migration helper
def migrate_from_legacy_trainer(
    legacy_config_path: str,
    output_dir: str,
) -> RouterTrainer:
    """Helper function to migrate from legacy router trainer.
    
    Args:
        legacy_config_path: Path to legacy configuration
        output_dir: Output directory for new trainer
        
    Returns:
        New unified router trainer
    """
    from ...bem_core.config.config_loader import load_experiment_config
    
    # Load legacy config and convert to new format
    config = load_experiment_config(legacy_config_path)
    
    # Create new trainer with unified infrastructure
    trainer = RouterTrainer(
        config=config,
        output_dir=output_dir,
        experiment_name=config.name,
    )
    
    return trainer