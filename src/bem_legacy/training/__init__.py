"""
BEM v1.1 Training Package

Training infrastructure with governance, mixed precision, and cache-aware optimization.
"""

from .bem_v11_trainer import BEMv11Trainer, BEMv11TrainingConfig
from .governance_trainer import GovernanceAwareTrainer
from .cache_metrics import CacheMetricsCollector

# Aliases for backward compatibility
TrainingConfig = BEMv11TrainingConfig
BEMTrainer = BEMv11Trainer
LoRATrainer = BEMv11Trainer

__all__ = ['BEMv11Trainer', 'BEMv11TrainingConfig', 'GovernanceAwareTrainer', 'CacheMetricsCollector',
           'TrainingConfig', 'BEMTrainer', 'LoRATrainer']