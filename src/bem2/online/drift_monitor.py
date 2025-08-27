"""
Drift Monitor for BEM 2.0 Online Learning.

Implements continuous monitoring of KL divergence and parameter norms with 
automatic rollback as specified in TODO.md requirements.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from enum import Enum
from collections import deque
import time
import math
import copy


class DriftLevel(Enum):
    """Levels of drift severity."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DriftThresholds:
    """Thresholds for drift monitoring."""
    
    # KL divergence thresholds
    kl_warning: float = 0.05
    kl_critical: float = 0.1
    kl_emergency: float = 0.2
    
    # Parameter norm thresholds  
    param_norm_warning: float = 0.5
    param_norm_critical: float = 1.0
    param_norm_emergency: float = 2.0
    
    # Spectral radius thresholds
    spectral_warning: float = 1.5
    spectral_critical: float = 2.0
    spectral_emergency: float = 3.0
    
    # Gradient norm thresholds
    grad_norm_warning: float = 5.0
    grad_norm_critical: float = 10.0
    grad_norm_emergency: float = 20.0
    
    # Performance drop thresholds
    performance_drop_warning: float = 0.02  # 2% drop
    performance_drop_critical: float = 0.05  # 5% drop
    performance_drop_emergency: float = 0.1  # 10% drop
    
    # Violation tolerances
    warning_violations_limit: int = 5
    critical_violations_limit: int = 3
    emergency_violations_limit: int = 1
    
    # Time window for violation counting
    violation_window_size: int = 20


@dataclass
class DriftMetrics:
    """Current drift metrics."""
    
    # Core metrics
    kl_divergence: float = 0.0
    parameter_norm: float = 0.0
    spectral_radius: float = 0.0
    gradient_norm: float = 0.0
    
    # Performance metrics
    current_performance: float = 0.0
    baseline_performance: float = 0.0
    performance_drop: float = 0.0
    
    # Derived metrics
    drift_level: DriftLevel = DriftLevel.NORMAL
    drift_score: float = 0.0  # Combined drift score [0, 1]
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    violations_count: Dict[DriftLevel, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize violations count
        if not self.violations_count:
            self.violations_count = {level: 0 for level in DriftLevel}


class DriftMonitor:
    """
    Continuous drift monitor for online learning safety.
    
    Monitors:
    - KL divergence between base and current model
    - Parameter norm drift from original weights
    - Spectral radius of weight matrices
    - Gradient norms during training
    - Performance degradation
    
    Triggers automatic rollback when drift exceeds safety limits.
    """
    
    def __init__(
        self,
        thresholds: DriftThresholds,
        base_model: Optional[nn.Module] = None
    ):
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__)
        
        # Base model for comparison
        self.base_model_state: Optional[Dict[str, torch.Tensor]] = None
        if base_model is not None:
            self.set_base_model(base_model)
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=1000)
        self.violation_history: deque = deque(maxlen=thresholds.violation_window_size)
        
        # Current state
        self.current_metrics = DriftMetrics()
        self.last_safe_metrics: Optional[DriftMetrics] = None
        
        # Rollback trigger state
        self.rollback_triggered = False
        self.rollback_reason = ""
        
        # Performance baseline
        self.baseline_performance: Optional[float] = None
        
        # Statistics
        self.total_checks = 0
        self.total_violations = 0
        self.rollback_count = 0
        
        self.logger.info("DriftMonitor initialized")
    
    def set_base_model(self, base_model: nn.Module):
        """Set the base model for drift comparison."""
        self.base_model_state = {
            name: param.data.clone().detach()
            for name, param in base_model.named_parameters()
        }
        self.logger.info("Base model state captured for drift monitoring")
    
    def set_baseline_performance(self, performance: float):
        """Set baseline performance for regression monitoring."""
        self.baseline_performance = performance
        self.logger.info(f"Baseline performance set to: {performance:.4f}")
    
    def check_drift(
        self,
        model: nn.Module,
        current_performance: Optional[float] = None,
        reference_outputs: Optional[torch.Tensor] = None,
        current_outputs: Optional[torch.Tensor] = None,
        step: int = 0
    ) -> DriftMetrics:
        """
        Perform drift check and return current metrics.
        
        Args:
            model: Current model to check
            current_performance: Current model performance
            reference_outputs: Reference model outputs for KL divergence
            current_outputs: Current model outputs for KL divergence
            step: Current training step
            
        Returns:
            Current drift metrics
        """
        self.total_checks += 1
        
        # Compute core drift metrics
        metrics = DriftMetrics(step=step)
        
        # 1. Parameter norm drift
        if self.base_model_state is not None:
            metrics.parameter_norm = self._compute_parameter_drift(model)
        
        # 2. KL divergence
        if reference_outputs is not None and current_outputs is not None:
            metrics.kl_divergence = self._compute_kl_divergence(
                reference_outputs, current_outputs
            )
        
        # 3. Spectral radius
        metrics.spectral_radius = self._compute_spectral_radius(model)
        
        # 4. Gradient norm
        metrics.gradient_norm = self._compute_gradient_norm(model)
        
        # 5. Performance metrics
        if current_performance is not None:
            metrics.current_performance = current_performance
            if self.baseline_performance is not None:
                metrics.baseline_performance = self.baseline_performance
                metrics.performance_drop = self.baseline_performance - current_performance
        
        # 6. Determine drift level and score
        metrics.drift_level, metrics.drift_score = self._assess_drift_level(metrics)
        
        # 7. Update violation counts
        metrics.violations_count = self._update_violations(metrics)
        
        # 8. Check for rollback trigger
        self._check_rollback_trigger(metrics)
        
        # 9. Update history
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        
        # 10. Update last safe metrics if appropriate
        if metrics.drift_level == DriftLevel.NORMAL:
            self.last_safe_metrics = copy.deepcopy(metrics)
        
        # Log warnings and critical states
        if metrics.drift_level != DriftLevel.NORMAL:
            self.logger.warning(
                f"Drift {metrics.drift_level.value}: "
                f"KL={metrics.kl_divergence:.4f}, "
                f"param_norm={metrics.parameter_norm:.4f}, "
                f"spectral={metrics.spectral_radius:.4f}, "
                f"perf_drop={metrics.performance_drop:.4f}"
            )
        
        return metrics
    
    def _compute_parameter_drift(self, model: nn.Module) -> float:
        """Compute L2 norm of parameter drift from base model."""
        if self.base_model_state is None:
            return 0.0
        
        total_drift = 0.0
        total_params = 0
        
        for name, param in model.named_parameters():
            if name in self.base_model_state:
                base_param = self.base_model_state[name]
                if param.device != base_param.device:
                    base_param = base_param.to(param.device)
                
                drift = torch.norm(param.data - base_param).item()
                total_drift += drift ** 2
                total_params += 1
        
        return math.sqrt(total_drift) if total_params > 0 else 0.0
    
    def _compute_kl_divergence(
        self,
        reference_outputs: torch.Tensor,
        current_outputs: torch.Tensor
    ) -> float:
        """Compute KL divergence between reference and current outputs."""
        try:
            # Ensure outputs are on same device
            if reference_outputs.device != current_outputs.device:
                reference_outputs = reference_outputs.to(current_outputs.device)
            
            # Compute softmax probabilities
            ref_probs = F.softmax(reference_outputs, dim=-1)
            curr_log_probs = F.log_softmax(current_outputs, dim=-1)
            
            # KL divergence: KL(P||Q) = sum(P * log(P/Q))
            kl_div = F.kl_div(curr_log_probs, ref_probs, reduction='mean')
            
            return kl_div.item()
            
        except Exception as e:
            self.logger.warning(f"Error computing KL divergence: {e}")
            return 0.0
    
    def _compute_spectral_radius(self, model: nn.Module) -> float:
        """Compute maximum spectral radius across all weight matrices."""
        max_spectral_radius = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) == 2:  # Matrix parameters only
                try:
                    # Compute largest singular value
                    u, s, v = torch.svd(param.data)
                    spectral_radius = s[0].item()
                    max_spectral_radius = max(max_spectral_radius, spectral_radius)
                except Exception as e:
                    # Skip problematic matrices
                    continue
        
        return max_spectral_radius
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute L2 norm of gradients."""
        total_grad_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
        
        return math.sqrt(total_grad_norm)
    
    def _assess_drift_level(self, metrics: DriftMetrics) -> Tuple[DriftLevel, float]:
        """Assess overall drift level and compute drift score."""
        violations = []
        
        # Check each metric against thresholds
        # KL divergence
        if metrics.kl_divergence >= self.thresholds.kl_emergency:
            violations.append(DriftLevel.EMERGENCY)
        elif metrics.kl_divergence >= self.thresholds.kl_critical:
            violations.append(DriftLevel.CRITICAL)
        elif metrics.kl_divergence >= self.thresholds.kl_warning:
            violations.append(DriftLevel.WARNING)
        
        # Parameter norm
        if metrics.parameter_norm >= self.thresholds.param_norm_emergency:
            violations.append(DriftLevel.EMERGENCY)
        elif metrics.parameter_norm >= self.thresholds.param_norm_critical:
            violations.append(DriftLevel.CRITICAL)
        elif metrics.parameter_norm >= self.thresholds.param_norm_warning:
            violations.append(DriftLevel.WARNING)
        
        # Spectral radius
        if metrics.spectral_radius >= self.thresholds.spectral_emergency:
            violations.append(DriftLevel.EMERGENCY)
        elif metrics.spectral_radius >= self.thresholds.spectral_critical:
            violations.append(DriftLevel.CRITICAL)
        elif metrics.spectral_radius >= self.thresholds.spectral_warning:
            violations.append(DriftLevel.WARNING)
        
        # Performance drop
        if metrics.performance_drop >= self.thresholds.performance_drop_emergency:
            violations.append(DriftLevel.EMERGENCY)
        elif metrics.performance_drop >= self.thresholds.performance_drop_critical:
            violations.append(DriftLevel.CRITICAL)
        elif metrics.performance_drop >= self.thresholds.performance_drop_warning:
            violations.append(DriftLevel.WARNING)
        
        # Determine overall level (worst violation)
        if DriftLevel.EMERGENCY in violations:
            drift_level = DriftLevel.EMERGENCY
        elif DriftLevel.CRITICAL in violations:
            drift_level = DriftLevel.CRITICAL
        elif DriftLevel.WARNING in violations:
            drift_level = DriftLevel.WARNING
        else:
            drift_level = DriftLevel.NORMAL
        
        # Compute drift score (0 = normal, 1 = maximum drift)
        kl_score = min(1.0, metrics.kl_divergence / self.thresholds.kl_emergency)
        norm_score = min(1.0, metrics.parameter_norm / self.thresholds.param_norm_emergency)
        spectral_score = min(1.0, metrics.spectral_radius / self.thresholds.spectral_emergency)
        perf_score = min(1.0, metrics.performance_drop / self.thresholds.performance_drop_emergency)
        
        drift_score = max(kl_score, norm_score, spectral_score, perf_score)
        
        return drift_level, drift_score
    
    def _update_violations(self, metrics: DriftMetrics) -> Dict[DriftLevel, int]:
        """Update violation counts in sliding window."""
        # Add current violation to history
        self.violation_history.append(metrics.drift_level)
        
        # Count violations by level in current window
        violation_counts = {level: 0 for level in DriftLevel}
        for level in self.violation_history:
            violation_counts[level] += 1
        
        return violation_counts
    
    def _check_rollback_trigger(self, metrics: DriftMetrics):
        """Check if rollback should be triggered."""
        # Check violation limits
        emergency_violations = metrics.violations_count[DriftLevel.EMERGENCY]
        critical_violations = metrics.violations_count[DriftLevel.CRITICAL]
        warning_violations = metrics.violations_count[DriftLevel.WARNING]
        
        rollback_reasons = []
        
        # Emergency violations
        if emergency_violations >= self.thresholds.emergency_violations_limit:
            rollback_reasons.append(f"Emergency violations: {emergency_violations}")
        
        # Critical violations
        if critical_violations >= self.thresholds.critical_violations_limit:
            rollback_reasons.append(f"Critical violations: {critical_violations}")
        
        # Warning violations
        if warning_violations >= self.thresholds.warning_violations_limit:
            rollback_reasons.append(f"Warning violations: {warning_violations}")
        
        # Immediate emergency triggers
        if metrics.drift_level == DriftLevel.EMERGENCY:
            rollback_reasons.append(f"Immediate emergency: {metrics.drift_level.value}")
        
        if rollback_reasons:
            self.rollback_triggered = True
            self.rollback_reason = "; ".join(rollback_reasons)
            self.rollback_count += 1
            self.total_violations += 1
            
            self.logger.error(f"ROLLBACK TRIGGERED: {self.rollback_reason}")
    
    def should_rollback(self) -> Tuple[bool, str]:
        """Check if rollback should be triggered."""
        return self.rollback_triggered, self.rollback_reason
    
    def reset_rollback(self):
        """Reset rollback trigger after successful rollback."""
        self.rollback_triggered = False
        self.rollback_reason = ""
        self.violation_history.clear()
        self.logger.info("Rollback trigger reset")
    
    def get_current_metrics(self) -> DriftMetrics:
        """Get current drift metrics."""
        return self.current_metrics
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift monitoring status."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 checks
        
        # Compute statistics over recent history
        kl_values = [m.kl_divergence for m in recent_metrics]
        norm_values = [m.parameter_norm for m in recent_metrics]
        spectral_values = [m.spectral_radius for m in recent_metrics]
        drift_scores = [m.drift_score for m in recent_metrics]
        
        summary = {
            'current_drift_level': self.current_metrics.drift_level.value,
            'current_drift_score': self.current_metrics.drift_score,
            'rollback_triggered': self.rollback_triggered,
            'rollback_reason': self.rollback_reason,
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'rollback_count': self.rollback_count,
            'recent_metrics': {
                'kl_divergence': {
                    'mean': np.mean(kl_values),
                    'max': np.max(kl_values),
                    'current': self.current_metrics.kl_divergence
                },
                'parameter_norm': {
                    'mean': np.mean(norm_values),
                    'max': np.max(norm_values),
                    'current': self.current_metrics.parameter_norm
                },
                'spectral_radius': {
                    'mean': np.mean(spectral_values),
                    'max': np.max(spectral_values),
                    'current': self.current_metrics.spectral_radius
                },
                'drift_score': {
                    'mean': np.mean(drift_scores),
                    'max': np.max(drift_scores),
                    'current': self.current_metrics.drift_score
                }
            },
            'thresholds': {
                'kl_warning': self.thresholds.kl_warning,
                'kl_critical': self.thresholds.kl_critical,
                'kl_emergency': self.thresholds.kl_emergency,
                'param_norm_warning': self.thresholds.param_norm_warning,
                'param_norm_critical': self.thresholds.param_norm_critical,
                'param_norm_emergency': self.thresholds.param_norm_emergency
            }
        }
        
        return summary
    
    def save_metrics(self, filepath: str):
        """Save metrics history to file."""
        metrics_data = [
            {
                'timestamp': m.timestamp,
                'step': m.step,
                'kl_divergence': m.kl_divergence,
                'parameter_norm': m.parameter_norm,
                'spectral_radius': m.spectral_radius,
                'gradient_norm': m.gradient_norm,
                'performance_drop': m.performance_drop,
                'drift_level': m.drift_level.value,
                'drift_score': m.drift_score
            }
            for m in self.metrics_history
        ]
        
        save_data = {
            'metrics_history': metrics_data,
            'thresholds': {
                'kl_warning': self.thresholds.kl_warning,
                'kl_critical': self.thresholds.kl_critical,
                'kl_emergency': self.thresholds.kl_emergency,
                'param_norm_warning': self.thresholds.param_norm_warning,
                'param_norm_critical': self.thresholds.param_norm_critical,
                'param_norm_emergency': self.thresholds.param_norm_emergency,
            },
            'statistics': {
                'total_checks': self.total_checks,
                'total_violations': self.total_violations,
                'rollback_count': self.rollback_count
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"Drift metrics saved to {filepath}")


# Utility functions
def create_drift_monitor(
    kl_warning: float = 0.05,
    kl_critical: float = 0.1,
    param_norm_warning: float = 0.5,
    param_norm_critical: float = 1.0,
    base_model: Optional[nn.Module] = None
) -> DriftMonitor:
    """Create drift monitor with specified thresholds."""
    thresholds = DriftThresholds(
        kl_warning=kl_warning,
        kl_critical=kl_critical,
        param_norm_warning=param_norm_warning,
        param_norm_critical=param_norm_critical
    )
    
    return DriftMonitor(thresholds, base_model)


# Example usage
if __name__ == "__main__":
    # Create models
    base_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    current_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create drift monitor
    monitor = create_drift_monitor(base_model=base_model)
    monitor.set_baseline_performance(0.85)
    
    # Simulate drift over time
    for step in range(100):
        # Add some noise to simulate drift
        with torch.no_grad():
            for param in current_model.parameters():
                param.data += torch.randn_like(param.data) * 0.01
        
        # Create dummy outputs for KL computation
        dummy_input = torch.randn(32, 100)
        base_outputs = base_model(dummy_input)
        current_outputs = current_model(dummy_input)
        
        # Simulate performance degradation
        current_perf = 0.85 - (step * 0.001)
        
        # Check drift
        metrics = monitor.check_drift(
            current_model,
            current_performance=current_perf,
            reference_outputs=base_outputs,
            current_outputs=current_outputs,
            step=step
        )
        
        # Check for rollback
        should_rollback, reason = monitor.should_rollback()
        if should_rollback:
            print(f"Step {step}: ROLLBACK TRIGGERED - {reason}")
            break
        
        if step % 20 == 0:
            print(f"Step {step}: drift_level={metrics.drift_level.value}, "
                  f"score={metrics.drift_score:.3f}")
    
    # Print summary
    summary = monitor.get_drift_summary()
    print(f"\nDrift Summary:")
    print(f"  Total checks: {summary['total_checks']}")
    print(f"  Total violations: {summary['total_violations']}")
    print(f"  Rollback count: {summary['rollback_count']}")
    print(f"  Current drift level: {summary['current_drift_level']}")
    print(f"  Current drift score: {summary['current_drift_score']:.3f}")