"""
BEM v1.1 Modules - Individual architectural components

This package contains the core components for BEM-v1.1-stable:
- E1: Generated Parallel Low-Rank adapters
- E3: Chunk-sticky routing with hysteresis 
- E4: Attention-logit bias injection
- Governance: Spectral and Frobenius constraints
"""

from .parallel_lora import GeneratedParallelLoRA
from .chunk_sticky_routing import ChunkStickyRouter
from .attention_bias import AttentionLogitBias
from .governance import SpectralGovernance, FrobeniusConstraint, BEMGovernance

__all__ = [
    'GeneratedParallelLoRA',
    'ChunkStickyRouter', 
    'AttentionLogitBias',
    'SpectralGovernance',
    'FrobeniusConstraint',
    'BEMGovernance'
]