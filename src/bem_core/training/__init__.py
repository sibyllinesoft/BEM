"""Unified training infrastructure for BEM experiments.

Provides base classes and utilities for standardized training across
all BEM components including router, safety, multimodal, and performance variants.
"""

from .base_trainer import BaseTrainer, TrainingConfig
from .config_loader import ConfigLoader, load_experiment_config
from .experiment_runner import ExperimentRunner
from .training_utils import (
    setup_model,
    setup_optimizer,
    setup_scheduler,
    compute_gradient_norm,
    apply_gradient_clipping,
)

__all__ = [
    "BaseTrainer",
    "TrainingConfig",
    "ConfigLoader",
    "load_experiment_config",
    "ExperimentRunner",
    "setup_model",
    "setup_optimizer", 
    "setup_scheduler",
    "compute_gradient_norm",
    "apply_gradient_clipping",
]