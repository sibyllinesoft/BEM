"""Unified safety trainer using bem_core infrastructure.

Demonstrates migration of the constitutional safety system to use unified
core infrastructure while preserving safety-specific functionality.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from ...bem_core.training.base_trainer import BaseTrainer, TrainingConfig
from ...bem_core.evaluation.base_evaluator import BaseEvaluator, EvaluationConfig
from ...bem_core.config.base_config import BaseConfig, ExperimentConfig
from .safety_controller import SafetyController
from .constitutional_scorer import ConstitutionalScorer
from .safety_basis import SafetyBasis
from .violation_detector import ViolationDetector
from .lagrangian_optimizer import LagrangianOptimizer


@dataclass
class SafetyConfig(BaseConfig):
    """Configuration for safety-specific parameters."""
    
    # Constitutional principles
    constitutional_principles: List[str] = field(default_factory=lambda: [
        "Be helpful and harmless",
        "Avoid harmful or dangerous content",
        "Respect human autonomy and dignity",
        "Be truthful and accurate",
    ])
    
    # Safety basis configuration
    safety_basis_dim: int = 128
    orthogonality_strength: float = 1.0
    safety_regularization_weight: float = 0.1
    
    # Constitutional scorer settings
    value_model_name: str = "microsoft/DialoGPT-medium"
    scorer_temperature: float = 1.0
    principle_weight: float = 1.0
    
    # Violation detection
    violation_threshold: float = 0.5
    detection_confidence_threshold: float = 0.8
    max_violation_rate: float = 0.05  # 5% maximum violation rate
    
    # Lagrangian optimization
    lagrangian_lr: float = 1e-3
    lagrangian_momentum: float = 0.9
    constraint_tolerance: float = 0.01
    
    # Safety curriculum
    curriculum_enabled: bool = True
    curriculum_stages: List[str] = field(default_factory=lambda: [
        "basic_safety", "nuanced_cases", "edge_cases"
    ])
    
    # Performance preservation
    max_performance_degradation: float = 0.02  # 2% max degradation
    performance_recovery_steps: int = 100
    

class SafetyTrainer(BaseTrainer):
    """Unified trainer for constitutional safety system.
    
    Inherits from BaseTrainer to provide standardized training infrastructure
    while implementing safety-specific training with constitutional principles,
    violation detection, and Lagrangian optimization.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        safety_config: Optional[SafetyConfig] = None,
        output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Initialize safety trainer.
        
        Args:
            config: Complete experiment configuration
            safety_config: Safety-specific configuration
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
        self.safety_config = safety_config or SafetyConfig()
        
        # Safety components (initialized in _setup_model)
        self.base_model = None
        self.safety_controller = None
        self.constitutional_scorer = None
        self.safety_basis = None
        self.violation_detector = None
        self.lagrangian_optimizer = None
        
        # Safety training state
        self.current_curriculum_stage = 0
        self.violation_history = []
        self.performance_history = []
        self.lagrangian_multipliers = {}
        
        self.logger.info(f"Initialized SafetyTrainer with {len(self.safety_config.constitutional_principles)} principles")
    
    def _setup_model(self) -> nn.Module:
        """Set up the constitutional safety system."""
        # Initialize base model
        base_model_name = self.experiment_config.model.base_model
        
        from transformers import AutoModel
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=getattr(torch, self.experiment_config.model.torch_dtype),
        )
        
        # Initialize safety basis (orthogonal safety subspace)
        self.safety_basis = SafetyBasis(
            model_dim=self.experiment_config.model.hidden_size,
            safety_dim=self.safety_config.safety_basis_dim,
            orthogonality_strength=self.safety_config.orthogonality_strength,
        )
        
        # Initialize constitutional scorer
        self.constitutional_scorer = ConstitutionalScorer(
            value_model_name=self.safety_config.value_model_name,
            principles=self.safety_config.constitutional_principles,
            temperature=self.safety_config.scorer_temperature,
        )
        
        # Initialize violation detector
        self.violation_detector = ViolationDetector(
            model_dim=self.experiment_config.model.hidden_size,
            threshold=self.safety_config.violation_threshold,
            confidence_threshold=self.safety_config.detection_confidence_threshold,
        )
        
        # Initialize safety controller (combines all components)
        self.safety_controller = SafetyController(
            base_model=self.base_model,
            safety_basis=self.safety_basis,
            constitutional_scorer=self.constitutional_scorer,
            violation_detector=self.violation_detector,
        )
        
        # Initialize Lagrangian optimizer for constraint handling
        self.lagrangian_optimizer = LagrangianOptimizer(
            lr=self.safety_config.lagrangian_lr,
            momentum=self.safety_config.lagrangian_momentum,
            tolerance=self.safety_config.constraint_tolerance,
        )
        
        self.logger.info("Initialized constitutional safety system")
        self.logger.info(f"Safety basis dimension: {self.safety_config.safety_basis_dim}")
        self.logger.info(f"Constitutional principles: {len(self.safety_config.constitutional_principles)}")
        
        return self.safety_controller
    
    def _compute_loss(self, batch: Dict[str, Any], model_outputs: Any) -> Dict[str, torch.Tensor]:
        """Compute safety-aware loss function with constitutional constraints."""
        # Extract outputs from safety controller
        base_outputs = model_outputs.get('base_outputs')
        safety_scores = model_outputs.get('safety_scores')
        violation_scores = model_outputs.get('violation_scores')
        constitutional_scores = model_outputs.get('constitutional_scores')
        
        # Compute primary utility loss (task performance)
        utility_loss = self._compute_utility_loss(batch, base_outputs)
        
        # Compute safety violation loss
        violation_loss = self._compute_violation_loss(violation_scores, batch)
        
        # Compute constitutional loss (alignment with principles)
        constitutional_loss = self._compute_constitutional_loss(constitutional_scores, batch)
        
        # Compute orthogonality regularization (safety basis)
        orthogonal_loss = self._compute_orthogonality_loss()
        
        # Lagrangian method for constraint optimization
        constraint_violations = self._compute_constraint_violations(
            violation_scores, constitutional_scores
        )
        
        # Update Lagrangian multipliers
        self.lagrangian_multipliers = self.lagrangian_optimizer.update_multipliers(
            constraint_violations, self.lagrangian_multipliers
        )
        
        # Compute Lagrangian loss
        lagrangian_penalty = sum(
            multiplier * violation 
            for multiplier, violation in zip(
                self.lagrangian_multipliers.values(),
                constraint_violations.values()
            )
        )
        
        # Combine losses with safety-first weighting
        safety_weight = self.safety_config.safety_regularization_weight
        constitutional_weight = self.safety_config.principle_weight
        
        total_loss = (
            utility_loss +
            safety_weight * violation_loss +
            constitutional_weight * constitutional_loss +
            self.safety_config.orthogonality_strength * orthogonal_loss +
            lagrangian_penalty
        )
        
        return {
            'loss': total_loss,
            'utility_loss': utility_loss,
            'violation_loss': violation_loss,
            'constitutional_loss': constitutional_loss,
            'orthogonal_loss': orthogonal_loss,
            'lagrangian_penalty': lagrangian_penalty,
        }
    
    def _compute_utility_loss(self, batch: Dict[str, Any], outputs: Any) -> torch.Tensor:
        """Compute primary utility loss (task performance)."""
        if hasattr(outputs, 'loss'):
            return outputs.loss
        
        # Fallback: compute language modeling loss
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        labels = batch.get('labels')
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    def _compute_violation_loss(
        self, 
        violation_scores: torch.Tensor, 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute safety violation loss."""
        if violation_scores is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Penalize high violation scores
        violation_penalty = torch.mean(torch.relu(violation_scores - self.safety_config.violation_threshold))
        
        # Add violation rate constraint
        violation_rate = torch.mean((violation_scores > self.safety_config.violation_threshold).float())
        rate_penalty = torch.relu(violation_rate - self.safety_config.max_violation_rate)
        
        return violation_penalty + rate_penalty
    
    def _compute_constitutional_loss(
        self, 
        constitutional_scores: torch.Tensor, 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute constitutional alignment loss."""
        if constitutional_scores is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Encourage alignment with constitutional principles
        # Higher scores = better alignment, so minimize negative scores
        alignment_loss = torch.mean(torch.relu(-constitutional_scores))
        
        return alignment_loss
    
    def _compute_orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization for safety basis."""
        if self.safety_basis is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return self.safety_basis.compute_orthogonality_loss()
    
    def _compute_constraint_violations(
        self, 
        violation_scores: torch.Tensor,
        constitutional_scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute constraint violations for Lagrangian optimization."""
        violations = {}
        
        if violation_scores is not None:
            # Violation rate constraint
            violation_rate = torch.mean((violation_scores > self.safety_config.violation_threshold).float())
            violations['violation_rate'] = torch.relu(violation_rate - self.safety_config.max_violation_rate)
        
        if constitutional_scores is not None:
            # Constitutional alignment constraint (should be positive)
            min_alignment = torch.min(constitutional_scores)
            violations['constitutional_alignment'] = torch.relu(-min_alignment)
        
        return violations
    
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run safety-specific evaluation."""
        # Create safety evaluator
        eval_config = EvaluationConfig(
            batch_size=self.config.batch_size,
            metrics=[
                'violation_rate', 'constitutional_alignment_score', 
                'safety_coverage', 'performance_preservation'
            ],
            measure_latency=True,
            measure_memory=True,
        )
        
        evaluator = SafetyEvaluator(
            model=self.model,
            config=eval_config,
            safety_config=self.safety_config,
            device=self.device,
            logger=self.logger,
        )
        
        result = evaluator.evaluate(dataloader)
        
        # Track performance history for degradation monitoring
        current_performance = result.metrics.get('task_performance', 0.0)
        self.performance_history.append(current_performance)
        
        # Check for performance degradation
        if len(self.performance_history) > 1:
            performance_drop = self.performance_history[0] - current_performance
            if performance_drop > self.safety_config.max_performance_degradation:
                self.logger.warning(
                    f"Performance degradation detected: {performance_drop:.3f} > "
                    f"{self.safety_config.max_performance_degradation:.3f}"
                )
        
        return result.metrics
    
    def advance_curriculum(self) -> bool:
        """Advance safety curriculum to next stage.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if not self.safety_config.curriculum_enabled:
            return False
        
        if self.current_curriculum_stage < len(self.safety_config.curriculum_stages) - 1:
            self.current_curriculum_stage += 1
            current_stage = self.safety_config.curriculum_stages[self.current_curriculum_stage]
            self.logger.info(f"Advanced to curriculum stage {self.current_curriculum_stage}: {current_stage}")
            return True
        
        return False
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get current safety training metrics."""
        return {
            'current_curriculum_stage': self.current_curriculum_stage,
            'curriculum_stage_name': self.safety_config.curriculum_stages[self.current_curriculum_stage],
            'lagrangian_multipliers': dict(self.lagrangian_multipliers),
            'violation_history_length': len(self.violation_history),
            'performance_history_length': len(self.performance_history),
            'avg_recent_performance': (
                sum(self.performance_history[-10:]) / min(10, len(self.performance_history))
                if self.performance_history else 0.0
            ),
        }


class SafetyEvaluator(BaseEvaluator):
    """Evaluator for constitutional safety system.
    
    Inherits from BaseEvaluator to provide standardized evaluation infrastructure
    while implementing safety-specific metrics and inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig,
        safety_config: SafetyConfig,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None,
    ):
        """Initialize safety evaluator."""
        super().__init__(model, config, device, logger)
        self.safety_config = safety_config
    
    def _run_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run safety controller inference on a batch."""
        # Forward pass through safety controller
        outputs = self.model(**batch)
        
        # Extract safety-specific outputs
        predictions = []
        violation_scores = []
        constitutional_scores = []
        
        if hasattr(outputs, 'logits'):
            # Primary task predictions
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predictions = predicted_ids.cpu().tolist()
        
        if outputs.get('violation_scores') is not None:
            violation_scores = outputs['violation_scores'].cpu().tolist()
        
        if outputs.get('constitutional_scores') is not None:
            constitutional_scores = outputs['constitutional_scores'].cpu().tolist()
        
        return {
            'predictions': predictions,
            'violation_scores': violation_scores,
            'constitutional_scores': constitutional_scores,
            'model_outputs': outputs,
        }
    
    def _compute_component_metrics(
        self, 
        predictions: List[Any], 
        targets: List[Any]
    ) -> Dict[str, float]:
        """Compute safety-specific metrics."""
        metrics = {}
        
        # Extract safety scores from predictions
        all_violation_scores = []
        all_constitutional_scores = []
        
        for pred in predictions:
            if isinstance(pred, dict):
                if 'violation_scores' in pred:
                    all_violation_scores.extend(pred['violation_scores'])
                if 'constitutional_scores' in pred:
                    all_constitutional_scores.extend(pred['constitutional_scores'])
        
        if all_violation_scores:
            import numpy as np
            violation_array = np.array(all_violation_scores)
            
            # Violation rate (percentage above threshold)
            violation_rate = np.mean(violation_array > self.safety_config.violation_threshold)
            metrics['violation_rate'] = float(violation_rate)
            
            # Safety coverage (percentage below threshold)
            metrics['safety_coverage'] = float(1.0 - violation_rate)
            
            # Average violation score
            metrics['avg_violation_score'] = float(np.mean(violation_array))
            
            # Maximum violation score
            metrics['max_violation_score'] = float(np.max(violation_array))
        
        if all_constitutional_scores:
            import numpy as np
            constitutional_array = np.array(all_constitutional_scores)
            
            # Constitutional alignment score
            metrics['constitutional_alignment_score'] = float(np.mean(constitutional_array))
            metrics['min_constitutional_score'] = float(np.min(constitutional_array))
            metrics['constitutional_consistency'] = float(np.std(constitutional_array))
        
        # Performance preservation (if we have baseline performance)
        # This would be computed by comparing to a baseline model
        
        return metrics


# Migration helper function
def migrate_from_legacy_safety_trainer(
    legacy_config_path: str,
    output_dir: str,
    safety_principles: Optional[List[str]] = None,
) -> SafetyTrainer:
    """Helper function to migrate from legacy safety trainer.
    
    Args:
        legacy_config_path: Path to legacy safety configuration
        output_dir: Output directory for new trainer
        safety_principles: Optional custom safety principles
        
    Returns:
        New unified safety trainer
    """
    from ...bem_core.config.config_loader import load_experiment_config
    
    # Load legacy config and convert to new format
    config = load_experiment_config(legacy_config_path)
    
    # Create safety config with custom principles if provided
    safety_config = SafetyConfig()
    if safety_principles:
        safety_config.constitutional_principles = safety_principles
    
    # Create new trainer with unified infrastructure
    trainer = SafetyTrainer(
        config=config,
        safety_config=safety_config,
        output_dir=output_dir,
        experiment_name=config.name,
    )
    
    return trainer