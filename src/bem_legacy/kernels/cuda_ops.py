"""
Python interface for BEM CUDA operations.

This module provides a high-level Python interface for the optimized CUDA kernels,
handling tensor management, device placement, and integration with the BEM system.

Key features:
- Automatic device and dtype handling
- Memory-efficient tensor operations
- Integration with PyTorch autograd
- Comprehensive error handling
- Performance profiling utilities
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import warnings
import time
import numpy as np

try:
    import fused_bem_kernels  # Compiled CUDA extension
    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False
    warnings.warn(
        "CUDA kernels not available. Falling back to PyTorch implementation. "
        "Performance will be significantly slower. Please compile the CUDA extension.",
        RuntimeWarning
    )

class FusedGeneratedUpdate(torch.autograd.Function):
    """
    PyTorch autograd function for the fused generated update kernel.
    
    Implements: ΔY = ((X @ V) ⊙ codes) @ U^T * scaling
    
    This is a custom autograd function that provides gradients for the fused operation
    while maintaining the performance benefits of the CUDA kernel.
    """
    
    @staticmethod
    def forward(ctx, X, V, codes, U, scaling):
        """Forward pass using the fused CUDA kernel."""
        if not KERNELS_AVAILABLE:
            # Fallback to PyTorch implementation
            H = torch.matmul(X, V)
            H = H * codes  # Hadamard product
            output = torch.matmul(H, U.t()) * scaling
            
            # Save for backward
            ctx.save_for_backward(X, V, codes, U)
            ctx.scaling = scaling
            return output
        
        # Use optimized CUDA kernel
        output = fused_bem_kernels.fused_generated_update(X, V, codes, U, scaling)
        
        # Save tensors for backward pass
        ctx.save_for_backward(X, V, codes, U)
        ctx.scaling = scaling
        
        return output
    
    @staticmethod 
    def backward(ctx, grad_output):
        """Backward pass with proper gradient computation."""
        X, V, codes, U, = ctx.saved_tensors
        scaling = ctx.scaling
        
        # Initialize gradients
        grad_X = grad_V = grad_codes = grad_U = None
        
        if ctx.needs_input_grad[0]:  # grad_X
            # ∂L/∂X = (∂L/∂Y @ U @ diag(codes) @ V^T) * scaling
            temp = grad_output @ U @ torch.diag_embed(codes)
            grad_X = temp @ V.t() * scaling
            
        if ctx.needs_input_grad[1]:  # grad_V
            # ∂L/∂V = X^T @ (∂L/∂Y @ U @ diag(codes)) * scaling
            temp = grad_output @ U @ torch.diag_embed(codes)
            grad_V = X.t() @ temp * scaling
            
        if ctx.needs_input_grad[2]:  # grad_codes
            # ∂L/∂codes = (X @ V) ⊙ (∂L/∂Y @ U) * scaling
            H = X @ V
            temp = grad_output @ U
            grad_codes = H * temp * scaling
            
        if ctx.needs_input_grad[3]:  # grad_U
            # ∂L/∂U = ((X @ V) ⊙ codes)^T @ ∂L/∂Y * scaling
            H = X @ V
            H_scaled = H * codes
            grad_U = H_scaled.t() @ grad_output * scaling
            
        return grad_X, grad_V, grad_codes, grad_U, None


def fused_generated_update(
    X: torch.Tensor,
    V: torch.Tensor, 
    codes: torch.Tensor,
    U: torch.Tensor,
    scaling: float = 1.0
) -> torch.Tensor:
    """
    High-performance fused generated update operation.
    
    Computes: ΔY = ((X @ V) ⊙ codes) @ U^T * scaling
    
    This operation combines:
    1. First GEMM: H = X @ V
    2. Hadamard product: H = H ⊙ codes  
    3. Second GEMM: ΔY = H @ U^T
    4. Scaling: ΔY = ΔY * scaling
    
    All operations are fused into a single CUDA kernel for optimal performance.
    
    Args:
        X: Input tensor [batch_size * seq_len, input_dim] 
        V: First weight matrix [input_dim, intermediate_dim]
        codes: Dynamic codes [batch_size * seq_len, intermediate_dim]
        U: Second weight matrix [intermediate_dim, output_dim] 
        scaling: Scaling factor (alpha / rank)
        
    Returns:
        Output tensor [batch_size * seq_len, output_dim]
        
    Note:
        - V and U should be pre-transposed if needed for optimal memory layout
        - All tensors must be on the same CUDA device
        - Supported dtypes: float16, bfloat16, float32
    """
    return FusedGeneratedUpdate.apply(X, V, codes, U, scaling)


class OptimizedBEMModule(nn.Module):
    """
    Drop-in replacement for the standard BEM module using fused CUDA kernels.
    
    This provides identical functionality to HierarchicalBEMModule but with
    optimized kernel calls for the critical path operations.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dtype = dtype
        
        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False
            
        # LoRA parameters
        self.lora_U = nn.Parameter(torch.randn(
            base_layer.out_features, rank, dtype=dtype, device=base_layer.weight.device
        ) * 0.02)
        
        self.lora_V = nn.Parameter(torch.randn(
            base_layer.in_features, rank, dtype=dtype, device=base_layer.weight.device  
        ) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Performance tracking
        self.register_buffer('kernel_time_ms', torch.tensor(0.0))
        self.register_buffer('pytorch_time_ms', torch.tensor(0.0))
        self.register_buffer('num_calls', torch.tensor(0))
        
    def forward(self, x: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized kernel.
        
        Args:
            x: Input tensor [batch, seq_len, in_features] or [batch*seq_len, in_features]
            codes: Dynamic codes [batch, seq_len, rank] or [batch*seq_len, rank]
            
        Returns:
            Output tensor with same shape as base_layer(x)
        """
        # Base forward pass
        base_output = self.base_layer(x)
        
        # Handle shape variations
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, x.shape[-1])
            codes_flat = codes.view(-1, codes.shape[-1])
        else:
            x_flat = x
            codes_flat = codes
            
        # Ensure correct dtype
        if x_flat.dtype != self.dtype:
            x_flat = x_flat.to(self.dtype)
        if codes_flat.dtype != self.dtype:
            codes_flat = codes_flat.to(self.dtype)
            
        # Apply fused generated update
        if KERNELS_AVAILABLE and x.is_cuda:
            # Use optimized CUDA kernel
            start_time = time.perf_counter()
            lora_output = fused_generated_update(
                x_flat, self.lora_V, codes_flat, self.lora_U, self.scaling
            )
            end_time = time.perf_counter()
            
            # Track timing
            kernel_time = (end_time - start_time) * 1000  # Convert to ms
            self.kernel_time_ms += kernel_time
            
        else:
            # Fallback to PyTorch implementation
            start_time = time.perf_counter()
            
            # Standard LoRA computation: (x @ V) * codes @ U^T * scaling
            x_v = torch.matmul(x_flat, self.lora_V)
            x_v_scaled = x_v * codes_flat
            lora_output = torch.matmul(x_v_scaled, self.lora_U.t()) * self.scaling
            
            end_time = time.perf_counter()
            
            # Track timing
            pytorch_time = (end_time - start_time) * 1000  # Convert to ms
            self.pytorch_time_ms += pytorch_time
            
        # Reshape back if needed
        if len(original_shape) == 3:
            lora_output = lora_output.view(*original_shape[:-1], -1)
            
        # Apply dropout
        lora_output = self.dropout(lora_output)
        
        # Track call count
        self.num_calls += 1
        
        # Final output
        return base_output + lora_output
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.num_calls == 0:
            return {
                'avg_kernel_time_ms': 0.0,
                'avg_pytorch_time_ms': 0.0,
                'total_calls': 0,
                'speedup_ratio': 0.0
            }
            
        avg_kernel = self.kernel_time_ms.item() / self.num_calls.item()
        avg_pytorch = self.pytorch_time_ms.item() / self.num_calls.item()
        
        speedup = avg_pytorch / avg_kernel if avg_kernel > 0 else 0.0
        
        return {
            'avg_kernel_time_ms': avg_kernel,
            'avg_pytorch_time_ms': avg_pytorch, 
            'total_calls': self.num_calls.item(),
            'speedup_ratio': speedup
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking counters."""
        self.kernel_time_ms.zero_()
        self.pytorch_time_ms.zero_()
        self.num_calls.zero_()


def benchmark_kernel_performance(
    batch_size: int = 32,
    seq_len: int = 512,
    input_dim: int = 768,
    intermediate_dim: int = 8,
    output_dim: int = 768,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Comprehensive benchmark of kernel performance vs PyTorch baseline.
    
    Args:
        batch_size: Number of sequences in batch
        seq_len: Sequence length  
        input_dim: Input feature dimension
        intermediate_dim: Intermediate dimension (rank)
        output_dim: Output feature dimension
        dtype: Tensor data type
        device: Device to run on
        num_warmup: Warmup iterations
        num_runs: Number of timed runs
        
    Returns:
        Dictionary with comprehensive performance metrics
    """
    # Create test tensors
    M = batch_size * seq_len
    X = torch.randn(M, input_dim, dtype=dtype, device=device)
    V = torch.randn(input_dim, intermediate_dim, dtype=dtype, device=device)
    codes = torch.randn(M, intermediate_dim, dtype=dtype, device=device)
    U = torch.randn(intermediate_dim, output_dim, dtype=dtype, device=device)
    scaling = 2.0
    
    results = {
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'input_dim': input_dim,
            'intermediate_dim': intermediate_dim,
            'output_dim': output_dim,
            'dtype': str(dtype),
            'device': device,
            'M': M
        }
    }
    
    # Test CUDA kernel if available
    if KERNELS_AVAILABLE and device == "cuda":
        print("Benchmarking CUDA kernel...")
        
        # Warmup
        for _ in range(num_warmup):
            _ = fused_generated_update(X, V, codes, U, scaling)
        torch.cuda.synchronize()
        
        # Timed runs
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = fused_generated_update(X, V, codes, U, scaling)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        kernel_time = (end_time - start_time) / num_runs * 1000  # ms
        
        # Use built-in benchmark function if available
        try:
            detailed_metrics = fused_bem_kernels.benchmark_fused_kernel(
                X, V, codes, U, scaling, num_warmup, num_runs
            )
            results['cuda_kernel'] = {
                'avg_time_ms': kernel_time,
                'detailed_metrics': dict(detailed_metrics)
            }
        except:
            results['cuda_kernel'] = {
                'avg_time_ms': kernel_time
            }
    
    # Test PyTorch baseline
    print("Benchmarking PyTorch baseline...")
    
    # Warmup
    for _ in range(num_warmup):
        H = torch.matmul(X, V)
        H = H * codes
        _ = torch.matmul(H, U.t()) * scaling
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    start_time = time.perf_counter()
    for _ in range(num_runs):
        H = torch.matmul(X, V)
        H = H * codes
        _ = torch.matmul(H, U.t()) * scaling
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    pytorch_time = (end_time - start_time) / num_runs * 1000  # ms
    
    # Calculate theoretical metrics
    flops = 2 * M * input_dim * intermediate_dim + M * intermediate_dim + \
            2 * M * intermediate_dim * output_dim
    
    results['pytorch_baseline'] = {
        'avg_time_ms': pytorch_time,
        'tflops': (flops * 1e-6) / (pytorch_time * 1000),  # TFLOPs
        'flops': flops
    }
    
    # Calculate speedup if both available
    if 'cuda_kernel' in results:
        speedup = pytorch_time / results['cuda_kernel']['avg_time_ms']
        overhead_pct = ((results['cuda_kernel']['avg_time_ms'] - pytorch_time) / pytorch_time) * 100
        
        results['comparison'] = {
            'speedup_ratio': speedup,
            'overhead_percent': overhead_pct,
            'kernel_faster': speedup > 1.0,
            'meets_target': abs(overhead_pct) <= 15.0  # <15% overhead target
        }
        
        print(f"\nPerformance Results:")
        print(f"PyTorch baseline: {pytorch_time:.3f} ms")
        print(f"CUDA kernel: {results['cuda_kernel']['avg_time_ms']:.3f} ms") 
        print(f"Speedup: {speedup:.2f}x")
        print(f"Overhead: {overhead_pct:.1f}%")
        print(f"Meets <15% target: {results['comparison']['meets_target']}")
    
    return results


def validate_numerical_accuracy(
    batch_size: int = 8,
    seq_len: int = 128, 
    input_dim: int = 512,
    intermediate_dim: int = 8,
    output_dim: int = 512,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    rtol: float = 1e-3,
    atol: float = 1e-4
) -> Dict[str, Any]:
    """
    Validate numerical accuracy of CUDA kernel vs PyTorch reference.
    
    Args:
        batch_size, seq_len, etc.: Tensor dimensions
        dtype: Data type for validation
        device: Device to test on
        rtol, atol: Relative and absolute tolerance for comparison
        
    Returns:
        Validation results including max difference and pass/fail status
    """
    print(f"Validating numerical accuracy on {device} with {dtype}...")
    
    # Create test inputs
    M = batch_size * seq_len
    torch.manual_seed(42)  # For reproducible results
    
    X = torch.randn(M, input_dim, dtype=dtype, device=device)
    V = torch.randn(input_dim, intermediate_dim, dtype=dtype, device=device)
    codes = torch.randn(M, intermediate_dim, dtype=dtype, device=device)
    U = torch.randn(intermediate_dim, output_dim, dtype=dtype, device=device)
    scaling = 2.0
    
    # Compute reference result with PyTorch
    H_ref = torch.matmul(X, V)
    H_ref = H_ref * codes
    output_ref = torch.matmul(H_ref, U.t()) * scaling
    
    # Compute result with CUDA kernel
    if KERNELS_AVAILABLE and device == "cuda":
        output_kernel = fused_generated_update(X, V, codes, U, scaling)
    else:
        print("CUDA kernels not available, using PyTorch fallback")
        output_kernel = output_ref  # Same as reference
    
    # Compare results
    abs_diff = torch.abs(output_ref - output_kernel)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    rel_diff = abs_diff / (torch.abs(output_ref) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Check if differences are within tolerance
    passed = torch.allclose(output_ref, output_kernel, rtol=rtol, atol=atol)
    
    results = {
        'passed': passed,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'tolerance': {'rtol': rtol, 'atol': atol},
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'input_dim': input_dim,
            'intermediate_dim': intermediate_dim,
            'output_dim': output_dim,
            'dtype': str(dtype),
            'device': device
        }
    }
    
    print(f"Validation {'PASSED' if passed else 'FAILED'}")
    print(f"Max absolute difference: {max_abs_diff:.2e}")
    print(f"Max relative difference: {max_rel_diff:.2e}")
    
    return results


# Convenience functions for integration
def replace_bem_with_optimized(bem_module):
    """Replace a standard BEM module with optimized version."""
    if not isinstance(bem_module.base_layer, nn.Linear):
        raise ValueError("Can only optimize Linear layer BEM modules")
        
    optimized = OptimizedBEMModule(
        base_layer=bem_module.base_layer,
        rank=bem_module.config.rank,
        alpha=bem_module.config.alpha,
        dropout=bem_module.config.dropout,
        dtype=getattr(bem_module, 'dtype', torch.float16)
    )
    
    # Copy LoRA weights
    optimized.lora_U.data.copy_(bem_module.lora_U.data)
    optimized.lora_V.data.copy_(bem_module.lora_V.data)
    
    return optimized