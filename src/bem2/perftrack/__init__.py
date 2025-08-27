"""
BEM 2.0 Performance Track Implementation.

This module implements sophisticated performance optimization variants (PT1-PT4) 
that must shift the Pareto frontier or yield CI-backed Slice-B gains while 
maintaining strict budget parity (±5% params/FLOPs vs v1.3-stack anchor).

Performance Track Variants:
- PT1: Head-Group Gating @ W_O
- PT2: Dynamic Rank Mask (Fixed FLOPs) 
- PT3: Kronecker @ W_down (One Site)
- PT4: Residual FiLM Micro-γ,β

All variants include comprehensive evaluation frameworks for:
- Pareto frontier analysis
- Statistical validation
- Budget parity enforcement
- Latency profiling
- VRAM usage tracking
"""

from .pt1_head_gating import (
    HeadGroupGatingConfig,
    HeadGroupGatingModule,
    AttentionGateController
)
from .pt2_dynamic_mask import (
    DynamicRankMaskConfig,
    DynamicRankMaskModule,
    SparseMaskController
)
from .pt3_kronecker import (
    KroneckerConfig,
    KroneckerModule,
    FusedKroneckerOp
)
from .pt4_residual_film import (
    ResidualFiLMConfig,
    ResidualFiLMModule,
    MicroModulationController
)
from .evaluation import (
    ParetoAnalyzer,
    BudgetValidator,
    PerformanceProfiler
)
from .training import (
    PTTrainingConfig,
    PTTrainer,
    run_pt_variant_training
)

__all__ = [
    # PT1 Head-Group Gating
    'HeadGroupGatingConfig',
    'HeadGroupGatingModule', 
    'AttentionGateController',
    
    # PT2 Dynamic Rank Mask
    'DynamicRankMaskConfig',
    'DynamicRankMaskModule',
    'SparseMaskController',
    
    # PT3 Kronecker Factorization
    'KroneckerConfig',
    'KroneckerModule',
    'FusedKroneckerOp',
    
    # PT4 Residual FiLM
    'ResidualFiLMConfig',
    'ResidualFiLMModule',
    'MicroModulationController',
    
    # Evaluation & Validation
    'ParetoAnalyzer',
    'BudgetValidator', 
    'PerformanceProfiler',
    
    # Training Infrastructure
    'PTTrainingConfig',
    'PTTrainer',
    'run_pt_variant_training'
]