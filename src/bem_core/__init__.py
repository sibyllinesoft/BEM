"""BEM Core - Unified infrastructure for Block-wise Expert Modules research.

MIT License

Copyright (c) 2024 Nathan Rice and BEM Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""BEM Core - Unified infrastructure for Block-wise Expert Modules research.

This module provides the foundational components that unify training, evaluation,
and configuration management across all BEM variants and experiments.

Key Components:
- training: Unified training infrastructure with base trainers
- evaluation: Consolidated evaluation framework with standard metrics  
- config: Template-based configuration system with inheritance
- utils: Shared utilities for logging, checkpointing, data handling
- interfaces: Component protocols for type safety and extensibility
"""

__version__ = "1.0.0"
__author__ = "BEM Research Team"

# Core modules
from . import training
from . import evaluation
from . import config
from . import utils
from . import interfaces

# Key exports for convenience
from .training.base_trainer import BaseTrainer
from .evaluation.base_evaluator import BaseEvaluator
from .config.base_config import BaseConfig, ExperimentConfig

__all__ = [
    # Modules
    "training",
    "evaluation", 
    "config",
    "utils",
    "interfaces",
    # Key classes
    "BaseTrainer",
    "BaseEvaluator", 
    "BaseConfig",
    "ExperimentConfig",
]