"""
High-performance CUDA kernels for BEM operations.

This module provides optimized CUDA kernels for the critical path operations
in the BEM system, specifically the fused generated update kernel:

H = X @ V; H = H ⊙ c; ΔY = H @ U^T

The fused kernel eliminates intermediate memory writes and achieves <15% 
latency overhead vs the frozen base model.
"""

from typing import Optional
import torch

try:
    from .cuda_ops import fused_generated_update
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    fused_generated_update = None

__all__ = ['fused_generated_update', 'CUDA_AVAILABLE']

def check_cuda_availability() -> bool:
    """Check if CUDA kernels are available."""
    return CUDA_AVAILABLE and torch.cuda.is_available()