"""Unified configuration management for BEM experiments.

Provides template-based configuration system with inheritance,
validation, and standardized experiment configuration patterns.
"""

from .base_config import BaseConfig, ExperimentConfig
from .config_loader import ConfigLoader, load_experiment_config, load_training_config
from .validators import ConfigValidator, validate_config

__all__ = [
    "BaseConfig",
    "ExperimentConfig",
    "ConfigLoader",
    "load_experiment_config",
    "load_training_config",
    "ConfigValidator",
    "validate_config",
]