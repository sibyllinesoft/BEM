"""
Interface definitions for BEM 2.0 Online Learning System.

Defines common data structures and enums used across the online learning components.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import torch
import time


class SafetyStatus(Enum):
    """Safety status for online updates."""
    SAFE = "safe"
    WARNING = "warning" 
    DANGER = "danger"
    CRITICAL = "critical"


class UpdateDecision(Enum):
    """Decision for whether to apply an update."""
    APPLY = "apply"
    DEFER = "defer"
    REJECT = "reject"
    ROLLBACK = "rollback"


class LearningPhase(Enum):
    """Phase of online learning."""
    WARMUP = "warmup"
    STREAMING = "streaming"
    CONSOLIDATION = "consolidation"
    EVALUATION = "evaluation"
    ROLLBACK = "rollback"


class CanaryStatus(Enum):
    """Status of canary testing."""
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class LearningState:
    """Current state of the online learning system."""
    phase: LearningPhase
    step: int
    updates_applied: int
    rollbacks_triggered: int
    last_update_time: float
    last_checkpoint_time: float
    
    # Current metrics
    kl_divergence: float = 0.0
    parameter_norm: float = 0.0
    performance_score: float = 0.0
    trust_score: float = 1.0
    
    # Flags
    canaries_passed: bool = True
    drift_detected: bool = False
    consolidation_needed: bool = False


class OnlineUpdateResult(NamedTuple):
    """Result of an online update attempt."""
    decision: UpdateDecision
    safety_status: SafetyStatus
    canary_status: CanaryStatus
    
    # Metrics
    kl_divergence: float
    parameter_norm: float
    ewc_loss: float
    replay_loss: float
    
    # Metadata
    update_applied: bool
    rollback_triggered: bool
    checkpoint_created: bool
    drift_detected: bool
    
    # Performance
    performance_delta: float
    time_elapsed: float
    memory_usage: float
    
    # Details
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class UpdateMetrics:
    """Metrics tracked during online updates."""
    
    # Core safety metrics
    kl_divergence: float = 0.0
    parameter_norm: float = 0.0
    spectral_radius: float = 0.0
    gradient_norm: float = 0.0
    
    # Performance metrics  
    loss: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 0.0
    
    # EWC metrics
    ewc_loss: float = 0.0
    fisher_norm: float = 0.0
    
    # Replay metrics
    replay_loss: float = 0.0
    replay_accuracy: float = 0.0
    
    # System metrics
    memory_usage: float = 0.0
    compute_time: float = 0.0
    
    # Tracking
    timestamp: float = field(default_factory=time.time)
    step: int = 0


@dataclass
class SafetyLimits:
    """Safety limits for online learning."""
    
    # KL divergence limits
    max_kl_divergence: float = 0.1
    warning_kl_threshold: float = 0.05
    
    # Parameter norm limits  
    max_parameter_norm: float = 1.0
    warning_norm_threshold: float = 0.5
    
    # Spectral limits
    max_spectral_radius: float = 2.0
    
    # Gradient limits
    max_gradient_norm: float = 1.0
    
    # Performance limits
    min_performance_threshold: float = 0.8
    max_performance_drop: float = 0.1
    
    # Time limits
    max_update_time: float = 30.0  # seconds
    max_memory_usage: float = 8.0  # GB
    
    # Violation tolerances
    max_violations: int = 3
    violation_window: int = 10


@dataclass
class LearningRates:
    """Learning rate configuration for online learning."""
    
    # Base learning rates
    controller_lr: float = 1e-5
    routing_lr: float = 1e-4
    
    # EWC regularization strength
    ewc_lambda: float = 0.1
    
    # Proximal regularization strength  
    prox_lambda: float = 0.05
    
    # Adaptive learning rate parameters
    lr_decay_factor: float = 0.9
    lr_recovery_factor: float = 1.05
    min_lr: float = 1e-7
    max_lr: float = 1e-3
    
    # Scheduling
    warmup_steps: int = 100
    decay_steps: int = 1000