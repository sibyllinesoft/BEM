"""
Online Learning Framework for BEM - Phase 5 Implementation.

Implements safe online adaptation with trust monitors, periodic consolidation,
and rollback mechanisms as specified in TODO.md Phase 5.

Key Features:
- Trust Monitors: Track entropy, KL divergence, spectral radius
- Budget Enforcement: Freeze adaptation when limits exceeded
- Controller Learning: Tiny learning rates for safe online updates
- Consolidation Engine: Periodic static LoRA fitting
- Rollback Mechanism: Revert to last safe state on violations
"""

from typing import Dict, List, Optional, Tuple, Union, NamedTuple, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import deque, defaultdict
import time
import copy

from .telemetry import TelemetryCollector
from .trust_region import TrustRegionProjector, SpectralClamp


class TrustStatus(Enum):
    """Trust status for online learning."""
    TRUSTED = "trusted"
    WARNING = "warning"  
    FROZEN = "frozen"
    ROLLBACK = "rollback"


class OnlineLearningMetrics(NamedTuple):
    """Metrics for online learning monitoring."""
    entropy: float
    kl_divergence: float
    spectral_radius: float
    gradient_norm: float
    parameter_drift: float
    learning_rate: float
    trust_status: TrustStatus
    steps_since_consolidation: int


@dataclass
class TrustBudget:
    """Budget limits for trust monitoring."""
    max_entropy: float = 3.0
    max_kl_divergence: float = 0.1
    max_spectral_radius: float = 2.0
    max_gradient_norm: float = 1.0
    max_parameter_drift: float = 0.5
    
    # Warning thresholds (percentage of max)
    warning_threshold: float = 0.8
    
    # Rolling window for statistics
    window_size: int = 100
    
    # Violation tolerance
    max_violations: int = 5
    violation_window: int = 50


@dataclass 
class OnlineLearningConfig:
    """Configuration for online learning system."""
    
    # Learning rates
    base_learning_rate: float = 1e-5
    min_learning_rate: float = 1e-7
    max_learning_rate: float = 1e-3
    lr_decay_factor: float = 0.9
    lr_recovery_factor: float = 1.1
    
    # Trust monitoring
    trust_budget: TrustBudget = field(default_factory=TrustBudget)
    monitor_frequency: int = 10  # Check every N steps
    
    # Consolidation
    consolidation_frequency: int = 1000  # Steps between consolidations
    consolidation_threshold: float = 0.3  # Trigger early if drift > threshold
    
    # Safety mechanisms
    enable_rollback: bool = True
    rollback_steps: int = 5  # Steps to rollback
    enable_gradient_clipping: bool = True
    gradient_clip_value: float = 1.0
    
    # Memory management
    max_checkpoints: int = 10
    checkpoint_frequency: int = 100
    
    # Adaptive learning
    enable_adaptive_lr: bool = True
    lr_adaptation_window: int = 50
    performance_threshold: float = 0.95


class TrustMonitor:
    """Monitors trust metrics and enforces budgets."""
    
    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.budget = config.trust_budget
        
        # Rolling statistics
        self.metrics_history = deque(maxlen=self.budget.window_size)
        self.violation_history = deque(maxlen=self.budget.violation_window)
        
        # Current state
        self.current_status = TrustStatus.TRUSTED
        self.violation_count = 0
        self.last_safe_metrics = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def update_metrics(self, metrics: OnlineLearningMetrics) -> TrustStatus:
        """Update trust metrics and determine status."""
        self.metrics_history.append(metrics)
        
        # Check individual budget violations
        violations = []
        
        if metrics.entropy > self.budget.max_entropy:
            violations.append(f"entropy: {metrics.entropy:.3f} > {self.budget.max_entropy}")
            
        if metrics.kl_divergence > self.budget.max_kl_divergence:
            violations.append(f"kl_div: {metrics.kl_divergence:.3f} > {self.budget.max_kl_divergence}")
            
        if metrics.spectral_radius > self.budget.max_spectral_radius:
            violations.append(f"spectral: {metrics.spectral_radius:.3f} > {self.budget.max_spectral_radius}")
            
        if metrics.gradient_norm > self.budget.max_gradient_norm:
            violations.append(f"grad_norm: {metrics.gradient_norm:.3f} > {self.budget.max_gradient_norm}")
            
        if metrics.parameter_drift > self.budget.max_parameter_drift:
            violations.append(f"drift: {metrics.parameter_drift:.3f} > {self.budget.max_parameter_drift}")
        
        # Update violation tracking
        self.violation_history.append(len(violations))
        recent_violations = sum(self.violation_history)
        
        # Determine trust status
        if violations:
            self.logger.warning(f"Trust budget violations: {', '.join(violations)}")
            
            if recent_violations > self.budget.max_violations:
                self.current_status = TrustStatus.ROLLBACK
                self.logger.error(f"Too many violations ({recent_violations}), triggering rollback")
            else:
                self.current_status = TrustStatus.FROZEN
                self.logger.warning("Freezing online learning due to violations")
        else:
            # Check warning thresholds
            warning_triggered = any([
                metrics.entropy > self.budget.max_entropy * self.budget.warning_threshold,
                metrics.kl_divergence > self.budget.max_kl_divergence * self.budget.warning_threshold,
                metrics.spectral_radius > self.budget.max_spectral_radius * self.budget.warning_threshold,
                metrics.parameter_drift > self.budget.max_parameter_drift * self.budget.warning_threshold
            ])
            
            if warning_triggered:
                self.current_status = TrustStatus.WARNING
            else:
                self.current_status = TrustStatus.TRUSTED
                self.last_safe_metrics = metrics
        
        return self.current_status
    
    def get_trust_score(self) -> float:
        """Get overall trust score [0, 1]."""
        if not self.metrics_history:
            return 1.0
            
        latest = self.metrics_history[-1]
        
        # Compute normalized scores for each metric
        entropy_score = max(0, 1 - latest.entropy / self.budget.max_entropy)
        kl_score = max(0, 1 - latest.kl_divergence / self.budget.max_kl_divergence)
        spectral_score = max(0, 1 - latest.spectral_radius / self.budget.max_spectral_radius)
        drift_score = max(0, 1 - latest.parameter_drift / self.budget.max_parameter_drift)
        
        # Weighted average
        trust_score = (entropy_score + kl_score + spectral_score + drift_score) / 4.0
        return trust_score
    
    def reset_violations(self):
        """Reset violation history after successful consolidation."""
        self.violation_history.clear()
        self.violation_count = 0
        self.current_status = TrustStatus.TRUSTED


class ConsolidationEngine:
    """Handles periodic consolidation of fast weights to static LoRA."""
    
    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Consolidation tracking
        self.steps_since_consolidation = 0
        self.fast_weight_deltas = []
        self.consolidation_history = []
    
    def should_consolidate(
        self,
        current_metrics: OnlineLearningMetrics,
        trust_status: TrustStatus
    ) -> bool:
        """Determine if consolidation should be triggered."""
        # Time-based consolidation
        time_trigger = self.steps_since_consolidation >= self.config.consolidation_frequency
        
        # Drift-based consolidation
        drift_trigger = (
            current_metrics.parameter_drift > self.config.consolidation_threshold
            and trust_status == TrustStatus.TRUSTED
        )
        
        # Safety consolidation (before rollback)
        safety_trigger = trust_status == TrustStatus.ROLLBACK
        
        return time_trigger or drift_trigger or safety_trigger
    
    def consolidate_weights(
        self,
        online_model: nn.Module,
        base_model: nn.Module,
        target_rank: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        Consolidate fast weights into a static LoRA residual.
        
        Returns dictionary of LoRA parameters to be added to base model.
        """
        self.logger.info("Starting weight consolidation...")
        
        # Compute weight differences
        weight_diffs = {}
        total_diff_norm = 0.0
        
        for (name, online_param), (_, base_param) in zip(
            online_model.named_parameters(), base_model.named_parameters()
        ):
            if online_param.requires_grad:
                diff = online_param.data - base_param.data
                weight_diffs[name] = diff
                total_diff_norm += torch.norm(diff).item() ** 2
        
        total_diff_norm = np.sqrt(total_diff_norm)
        self.logger.info(f"Total parameter drift: {total_diff_norm:.6f}")
        
        # Fit LoRA to weight differences
        consolidated_lora = self._fit_lora_to_diffs(weight_diffs, target_rank)
        
        # Reset fast weights to base
        with torch.no_grad():
            for (name, online_param), (_, base_param) in zip(
                online_model.named_parameters(), base_model.named_parameters()
            ):
                if online_param.requires_grad:
                    online_param.copy_(base_param)
        
        # Update tracking
        self.steps_since_consolidation = 0
        self.consolidation_history.append({
            'timestamp': time.time(),
            'total_drift': total_diff_norm,
            'lora_rank': target_rank
        })
        
        self.logger.info("Weight consolidation complete")
        return consolidated_lora
    
    def _fit_lora_to_diffs(
        self, 
        weight_diffs: Dict[str, torch.Tensor],
        target_rank: int
    ) -> Dict[str, torch.Tensor]:
        """Fit LoRA parameters to approximate weight differences."""
        lora_params = {}
        
        for name, diff in weight_diffs.items():
            if len(diff.shape) != 2:  # Skip non-matrix parameters
                continue
                
            # SVD decomposition of weight difference
            U, S, Vt = torch.svd(diff)
            
            # Take top-k components
            k = min(target_rank, len(S))
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
            
            # Create LoRA A and B matrices
            lora_A = Vt_k.T  # [in_features, rank]
            lora_B = U_k * S_k.unsqueeze(0)  # [out_features, rank]
            
            lora_params[f"{name}.lora_A"] = lora_A
            lora_params[f"{name}.lora_B"] = lora_B
        
        return lora_params
    
    def step(self):
        """Increment step counter."""
        self.steps_since_consolidation += 1


class OnlineLearningController:
    """Main controller for online learning with safety mechanisms."""
    
    def __init__(
        self,
        model: nn.Module,
        config: OnlineLearningConfig,
        telemetry_collector: Optional[TelemetryCollector] = None
    ):
        self.model = model
        self.config = config
        self.telemetry = telemetry_collector
        
        # Components
        self.trust_monitor = TrustMonitor(config)
        self.consolidation_engine = ConsolidationEngine(config)
        
        # State management
        self.checkpoints = deque(maxlen=config.max_checkpoints)
        self.base_model_state = copy.deepcopy(model.state_dict())
        self.current_lr = config.base_learning_rate
        self.step_count = 0
        
        # Spectral monitoring
        self.spectral_clamp = SpectralClamp(max_singular_value=config.trust_budget.max_spectral_radius)
        
        # Optimizer setup
        self.optimizer = None
        self.scheduler = None
        
        # Performance tracking
        self.performance_history = deque(maxlen=config.lr_adaptation_window)
        
        self.logger = logging.getLogger(__name__)
    
    def setup_optimizer(self, optimizer_class=torch.optim.AdamW, **optimizer_kwargs):
        """Setup optimizer for online learning."""
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.current_lr,
            **optimizer_kwargs
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=self.config.lr_decay_factor,
            patience=10,
            verbose=True
        )
    
    def compute_metrics(
        self,
        loss: torch.Tensor,
        base_outputs: Optional[torch.Tensor] = None,
        current_outputs: Optional[torch.Tensor] = None
    ) -> OnlineLearningMetrics:
        """Compute trust monitoring metrics."""
        with torch.no_grad():
            # Entropy computation (from loss or routing if available)
            if hasattr(self.model, 'get_routing_entropy'):
                entropy = self.model.get_routing_entropy()
            else:
                # Fallback: use loss as proxy for entropy
                entropy = loss.item()
            
            # KL divergence between base and current model outputs
            kl_div = 0.0
            if base_outputs is not None and current_outputs is not None:
                kl_div = F.kl_div(
                    F.log_softmax(current_outputs, dim=-1),
                    F.softmax(base_outputs, dim=-1),
                    reduction='mean'
                ).item()
            
            # Spectral radius (largest singular value across all layers)
            spectral_radius = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and len(param.shape) == 2:
                    try:
                        _, S, _ = torch.svd(param.data)
                        spectral_radius = max(spectral_radius, S[0].item())
                    except:
                        continue  # Skip if SVD fails
            
            # Gradient norm
            total_grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = np.sqrt(total_grad_norm)
            
            # Parameter drift from base model
            parameter_drift = 0.0
            for (name, param), (base_name, base_param) in zip(
                self.model.named_parameters(),
                [(n, p) for n, p in self.base_model_state.items()]
            ):
                if param.requires_grad:
                    drift = torch.norm(param.data - base_param).item()
                    parameter_drift += drift ** 2
            parameter_drift = np.sqrt(parameter_drift)
            
            return OnlineLearningMetrics(
                entropy=entropy,
                kl_divergence=kl_div,
                spectral_radius=spectral_radius,
                gradient_norm=total_grad_norm,
                parameter_drift=parameter_drift,
                learning_rate=self.current_lr,
                trust_status=self.trust_monitor.current_status,
                steps_since_consolidation=self.consolidation_engine.steps_since_consolidation
            )
    
    def step(
        self,
        loss: torch.Tensor,
        base_outputs: Optional[torch.Tensor] = None,
        current_outputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Perform one step of online learning with safety checks."""
        self.step_count += 1
        
        # Compute current metrics
        metrics = self.compute_metrics(loss, base_outputs, current_outputs)
        
        # Update trust monitor
        trust_status = self.trust_monitor.update_metrics(metrics)
        
        # Create checkpoint if needed
        if self.step_count % self.config.checkpoint_frequency == 0:
            self._create_checkpoint()
        
        # Handle different trust states
        step_taken = False
        rollback_triggered = False
        
        if trust_status == TrustStatus.ROLLBACK:
            # Rollback to last safe state
            self._rollback()
            rollback_triggered = True
            self.logger.warning("Rollback triggered due to trust violations")
            
        elif trust_status == TrustStatus.FROZEN:
            # Skip optimization step
            self.logger.info("Skipping optimization step due to trust concerns")
            
        elif trust_status in [TrustStatus.TRUSTED, TrustStatus.WARNING]:
            # Proceed with optimization
            if self.optimizer is not None:
                # Apply gradient clipping
                if self.config.enable_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_value
                    )
                
                # Apply spectral clamping to weights
                self._apply_spectral_clamping()
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                step_taken = True
                
                # Adaptive learning rate
                if self.config.enable_adaptive_lr:
                    self._adapt_learning_rate(metrics)
        
        # Check for consolidation
        consolidation_triggered = False
        if self.consolidation_engine.should_consolidate(metrics, trust_status):
            base_model = self._create_base_model()
            consolidated_lora = self.consolidation_engine.consolidate_weights(
                self.model, base_model, target_rank=8
            )
            consolidation_triggered = True
            
            # Reset trust monitor after successful consolidation
            if trust_status != TrustStatus.ROLLBACK:
                self.trust_monitor.reset_violations()
        
        # Update consolidation step counter
        self.consolidation_engine.step()
        
        # Log telemetry
        if self.telemetry:
            self.telemetry.log_online_learning_metrics({
                'metrics': metrics,
                'trust_status': trust_status.value,
                'step_taken': step_taken,
                'rollback_triggered': rollback_triggered,
                'consolidation_triggered': consolidation_triggered,
                'trust_score': self.trust_monitor.get_trust_score()
            })
        
        return {
            'metrics': metrics,
            'trust_status': trust_status,
            'step_taken': step_taken,
            'rollback_triggered': rollback_triggered,
            'consolidation_triggered': consolidation_triggered,
            'trust_score': self.trust_monitor.get_trust_score(),
            'current_lr': self.current_lr
        }
    
    def _create_checkpoint(self):
        """Create model checkpoint for rollback capability."""
        checkpoint = {
            'step': self.step_count,
            'model_state': copy.deepcopy(self.model.state_dict()),
            'optimizer_state': copy.deepcopy(self.optimizer.state_dict()) if self.optimizer else None,
            'metrics': self.trust_monitor.last_safe_metrics,
            'timestamp': time.time()
        }
        self.checkpoints.append(checkpoint)
        
        self.logger.debug(f"Created checkpoint at step {self.step_count}")
    
    def _rollback(self):
        """Rollback to last safe checkpoint."""
        if not self.checkpoints:
            # Fallback to base model
            self.model.load_state_dict(self.base_model_state)
            self.logger.warning("No checkpoints available, rolled back to base model")
            return
        
        # Find most recent safe checkpoint
        safe_checkpoint = None
        for checkpoint in reversed(self.checkpoints):
            if (checkpoint['metrics'] is not None and 
                checkpoint['metrics'].trust_status == TrustStatus.TRUSTED):
                safe_checkpoint = checkpoint
                break
        
        if safe_checkpoint is None:
            # Use most recent checkpoint if no safe ones found
            safe_checkpoint = self.checkpoints[-1]
            self.logger.warning("No safe checkpoints found, using most recent")
        
        # Restore state
        self.model.load_state_dict(safe_checkpoint['model_state'])
        if self.optimizer and safe_checkpoint['optimizer_state']:
            self.optimizer.load_state_dict(safe_checkpoint['optimizer_state'])
        
        self.logger.info(f"Rolled back to checkpoint from step {safe_checkpoint['step']}")
    
    def _create_base_model(self) -> nn.Module:
        """Create a base model copy for consolidation."""
        base_model = copy.deepcopy(self.model)
        base_model.load_state_dict(self.base_model_state)
        return base_model
    
    def _apply_spectral_clamping(self):
        """Apply spectral clamping to model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) == 2:
                param.data = self.spectral_clamp(param.data)
    
    def _adapt_learning_rate(self, metrics: OnlineLearningMetrics):
        """Adapt learning rate based on performance and trust."""
        trust_score = self.trust_monitor.get_trust_score()
        
        # Increase LR if performance is good and trust is high
        if trust_score > 0.9 and metrics.trust_status == TrustStatus.TRUSTED:
            self.current_lr = min(
                self.current_lr * self.config.lr_recovery_factor,
                self.config.max_learning_rate
            )
        # Decrease LR if trust is low
        elif trust_score < 0.5 or metrics.trust_status == TrustStatus.WARNING:
            self.current_lr = max(
                self.current_lr * self.config.lr_decay_factor,
                self.config.min_learning_rate
            )
        
        # Update optimizer learning rate
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        trust_score = self.trust_monitor.get_trust_score()
        
        return {
            'step': self.step_count,
            'trust_status': self.trust_monitor.current_status.value,
            'trust_score': trust_score,
            'current_lr': self.current_lr,
            'num_checkpoints': len(self.checkpoints),
            'steps_since_consolidation': self.consolidation_engine.steps_since_consolidation,
            'violation_count': sum(self.trust_monitor.violation_history),
            'consolidation_history': len(self.consolidation_engine.consolidation_history)
        }


def create_online_learning_controller(
    model: nn.Module,
    config: Optional[OnlineLearningConfig] = None,
    telemetry_collector: Optional[TelemetryCollector] = None
) -> OnlineLearningController:
    """Create online learning controller with default configuration."""
    if config is None:
        config = OnlineLearningConfig()
    
    return OnlineLearningController(
        model=model,
        config=config,
        telemetry_collector=telemetry_collector
    )


def create_default_online_learning_config(
    base_learning_rate: float = 1e-5,
    consolidation_frequency: int = 1000,
    enable_rollback: bool = True
) -> OnlineLearningConfig:
    """Create default online learning configuration."""
    return OnlineLearningConfig(
        base_learning_rate=base_learning_rate,
        consolidation_frequency=consolidation_frequency,
        enable_rollback=enable_rollback,
        trust_budget=TrustBudget(),
        enable_adaptive_lr=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test model
    test_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Linear(32, 10)
    )
    
    # Create online learning controller
    config = create_default_online_learning_config()
    controller = create_online_learning_controller(test_model, config)
    
    # Setup optimizer
    controller.setup_optimizer(torch.optim.AdamW, weight_decay=0.01)
    
    # Simulate training steps
    for step in range(100):
        # Create dummy data
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        
        # Forward pass
        outputs = test_model(x)
        loss = F.cross_entropy(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Online learning step with safety checks
        result = controller.step(loss)
        
        if step % 20 == 0:
            status = controller.get_status_summary()
            print(f"Step {step}:")
            print(f"  Trust status: {status['trust_status']}")
            print(f"  Trust score: {status['trust_score']:.3f}")
            print(f"  Learning rate: {status['current_lr']:.2e}")
            print(f"  Step taken: {result['step_taken']}")
            
            if result['consolidation_triggered']:
                print("  >>> Consolidation triggered")
            if result['rollback_triggered']:
                print("  >>> Rollback triggered")
    
    print(f"\nFinal status: {controller.get_status_summary()}")