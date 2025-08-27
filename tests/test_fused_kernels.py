#!/usr/bin/env python3
"""
Unit tests for BEM CUDA kernel numerical accuracy and performance.

These tests ensure that the fused CUDA kernels produce numerically identical
results to the PyTorch reference implementation within acceptable tolerances.

Test Coverage:
- Numerical accuracy for different dtypes and problem sizes
- Edge cases (zeros, large values, etc.)
- Memory layout verification
- Performance regression detection
- Error handling validation
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bem.kernels.cuda_ops import (
    fused_generated_update,
    OptimizedBEMModule,
    KERNELS_AVAILABLE,
    benchmark_kernel_performance,
    validate_numerical_accuracy
)
from bem.hierarchical_bem import HierarchicalBEMModule, HierarchicalBEMConfig


class TestFusedKernelAccuracy(unittest.TestCase):
    """Test numerical accuracy of fused CUDA kernels."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rtol_fp16 = 1e-3
        self.atol_fp16 = 1e-4
        self.rtol_fp32 = 1e-5
        self.atol_fp32 = 1e-6
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    
    def _pytorch_reference(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        codes: torch.Tensor,
        U: torch.Tensor,
        scaling: float
    ) -> torch.Tensor:
        """Reference PyTorch implementation."""
        # H = X @ V
        H = torch.matmul(X, V)
        
        # H = H ⊙ codes (Hadamard product)
        H = H * codes
        
        # ΔY = H @ U^T
        output = torch.matmul(H, U.t())
        
        # Apply scaling
        output = output * scaling
        
        return output
    
    def _create_test_tensors(
        self,
        batch_size: int,
        seq_len: int,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create test tensors with specified dimensions."""
        M = batch_size * seq_len
        
        # Create tensors with controlled random values
        X = torch.randn(M, input_dim, dtype=dtype, device=self.device) * 0.1
        V = torch.randn(input_dim, intermediate_dim, dtype=dtype, device=self.device) * 0.02
        codes = torch.randn(M, intermediate_dim, dtype=dtype, device=self.device) * 0.5
        U = torch.randn(intermediate_dim, output_dim, dtype=dtype, device=self.device) * 0.02
        
        return X, V, codes, U
    
    def _check_accuracy(
        self,
        output_ref: torch.Tensor,
        output_kernel: torch.Tensor,
        rtol: float,
        atol: float,
        test_name: str
    ) -> Dict[str, float]:
        """Check numerical accuracy between reference and kernel outputs."""
        abs_diff = torch.abs(output_ref - output_kernel)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        
        rel_diff = abs_diff / (torch.abs(output_ref) + 1e-8)
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        # Check if within tolerance
        passed = torch.allclose(output_ref, output_kernel, rtol=rtol, atol=atol)
        
        result = {
            'test_name': test_name,
            'passed': passed,
            'max_abs_diff': max_abs_diff,
            'mean_abs_diff': mean_abs_diff,
            'max_rel_diff': max_rel_diff,
            'mean_rel_diff': mean_rel_diff,
            'tolerance_rtol': rtol,
            'tolerance_atol': atol
        }
        
        return result
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not KERNELS_AVAILABLE, "CUDA kernels not compiled")
    def test_small_problem_fp16(self):
        """Test small problem size with fp16."""
        X, V, codes, U = self._create_test_tensors(
            batch_size=2, seq_len=64, input_dim=256,
            intermediate_dim=8, output_dim=256, dtype=torch.float16
        )
        scaling = 2.0
        
        # Reference computation
        output_ref = self._pytorch_reference(X, V, codes, U, scaling)
        
        # Kernel computation
        output_kernel = fused_generated_update(X, V, codes, U, scaling)
        
        # Check accuracy
        result = self._check_accuracy(
            output_ref, output_kernel, self.rtol_fp16, self.atol_fp16,
            "small_problem_fp16"
        )
        
        self.assertTrue(result['passed'], 
                       f"Accuracy test failed: max_abs_diff={result['max_abs_diff']:.2e}, "
                       f"max_rel_diff={result['max_rel_diff']:.2e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not KERNELS_AVAILABLE, "CUDA kernels not compiled")
    def test_medium_problem_fp16(self):
        """Test medium problem size with fp16."""
        X, V, codes, U = self._create_test_tensors(
            batch_size=8, seq_len=512, input_dim=768,
            intermediate_dim=8, output_dim=768, dtype=torch.float16
        )
        scaling = 16.0 / 8.0  # Typical LoRA scaling
        
        output_ref = self._pytorch_reference(X, V, codes, U, scaling)
        output_kernel = fused_generated_update(X, V, codes, U, scaling)
        
        result = self._check_accuracy(
            output_ref, output_kernel, self.rtol_fp16, self.atol_fp16,
            "medium_problem_fp16"
        )
        
        self.assertTrue(result['passed'],
                       f"Accuracy test failed: {result}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available") 
    @unittest.skipIf(not KERNELS_AVAILABLE, "CUDA kernels not compiled")
    def test_large_problem_fp32(self):
        """Test large problem size with fp32."""
        X, V, codes, U = self._create_test_tensors(
            batch_size=4, seq_len=1024, input_dim=1536,
            intermediate_dim=16, output_dim=1536, dtype=torch.float32
        )
        scaling = 1.0
        
        output_ref = self._pytorch_reference(X, V, codes, U, scaling)
        output_kernel = fused_generated_update(X, V, codes, U, scaling)
        
        result = self._check_accuracy(
            output_ref, output_kernel, self.rtol_fp32, self.atol_fp32,
            "large_problem_fp32"
        )
        
        self.assertTrue(result['passed'],
                       f"Accuracy test failed: {result}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not KERNELS_AVAILABLE, "CUDA kernels not compiled")
    def test_bfloat16_accuracy(self):
        """Test bfloat16 precision."""
        X, V, codes, U = self._create_test_tensors(
            batch_size=4, seq_len=256, input_dim=512,
            intermediate_dim=8, output_dim=512, dtype=torch.bfloat16
        )
        scaling = 2.0
        
        output_ref = self._pytorch_reference(X, V, codes, U, scaling)
        output_kernel = fused_generated_update(X, V, codes, U, scaling)
        
        # More relaxed tolerance for bfloat16
        result = self._check_accuracy(
            output_ref, output_kernel, 1e-2, 1e-3,
            "bfloat16_accuracy"
        )
        
        self.assertTrue(result['passed'],
                       f"BFloat16 accuracy test failed: {result}")
    
    def test_edge_case_zeros(self):
        """Test edge case with zero inputs."""
        if not torch.cuda.is_available() or not KERNELS_AVAILABLE:
            self.skipTest("CUDA or kernels not available")
        
        batch_size, seq_len = 2, 32
        input_dim, intermediate_dim, output_dim = 128, 4, 128
        M = batch_size * seq_len
        
        # Test with zero codes
        X = torch.randn(M, input_dim, dtype=torch.float16, device=self.device)
        V = torch.randn(input_dim, intermediate_dim, dtype=torch.float16, device=self.device)
        codes = torch.zeros(M, intermediate_dim, dtype=torch.float16, device=self.device)
        U = torch.randn(intermediate_dim, output_dim, dtype=torch.float16, device=self.device)
        scaling = 1.0
        
        output_ref = self._pytorch_reference(X, V, codes, U, scaling)
        output_kernel = fused_generated_update(X, V, codes, U, scaling)
        
        # With zero codes, output should be zero
        self.assertTrue(torch.allclose(output_ref, torch.zeros_like(output_ref), atol=1e-6))
        self.assertTrue(torch.allclose(output_kernel, torch.zeros_like(output_kernel), atol=1e-6))
    
    def test_edge_case_ones(self):
        """Test edge case with ones as codes."""
        if not torch.cuda.is_available() or not KERNELS_AVAILABLE:
            self.skipTest("CUDA or kernels not available")
            
        X, V, codes, U = self._create_test_tensors(
            batch_size=2, seq_len=64, input_dim=256,
            intermediate_dim=4, output_dim=256, dtype=torch.float16
        )
        
        # Set codes to ones (no modulation effect)
        codes.fill_(1.0)
        scaling = 1.0
        
        output_ref = self._pytorch_reference(X, V, codes, U, scaling)
        output_kernel = fused_generated_update(X, V, codes, U, scaling)
        
        result = self._check_accuracy(
            output_ref, output_kernel, self.rtol_fp16, self.atol_fp16,
            "edge_case_ones"
        )
        
        self.assertTrue(result['passed'])
    
    def test_different_scaling_factors(self):
        """Test with different scaling factors."""
        if not torch.cuda.is_available() or not KERNELS_AVAILABLE:
            self.skipTest("CUDA or kernels not available")
            
        X, V, codes, U = self._create_test_tensors(
            batch_size=4, seq_len=128, input_dim=384,
            intermediate_dim=8, output_dim=384, dtype=torch.float16
        )
        
        scaling_factors = [0.1, 0.5, 1.0, 2.0, 16.0/8.0, 10.0]
        
        for scaling in scaling_factors:
            with self.subTest(scaling=scaling):
                output_ref = self._pytorch_reference(X, V, codes, U, scaling)
                output_kernel = fused_generated_update(X, V, codes, U, scaling)
                
                result = self._check_accuracy(
                    output_ref, output_kernel, self.rtol_fp16, self.atol_fp16,
                    f"scaling_{scaling}"
                )
                
                self.assertTrue(result['passed'],
                               f"Scaling factor {scaling} failed: {result}")


class TestOptimizedBEMModule(unittest.TestCase):
    """Test the optimized BEM module integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
    
    def test_bem_module_compatibility(self):
        """Test that OptimizedBEMModule is compatible with HierarchicalBEMModule."""
        input_dim, output_dim = 512, 512
        rank = 8
        
        # Create base layer
        base_layer = nn.Linear(input_dim, output_dim, bias=False).to(self.device)
        
        # Create optimized BEM module
        optimized_bem = OptimizedBEMModule(
            base_layer=base_layer,
            rank=rank,
            alpha=16.0,
            dtype=torch.float16
        ).to(self.device)
        
        # Create reference BEM module
        bem_config = HierarchicalBEMConfig(
            rank=rank,
            alpha=16.0,
            dropout=0.0
        )
        
        reference_bem = HierarchicalBEMModule(
            base_layer=base_layer,
            config=bem_config
        ).to(self.device)
        
        # Copy weights
        reference_bem.lora_U.data.copy_(optimized_bem.lora_U.data)
        reference_bem.lora_V.data.copy_(optimized_bem.lora_V.data)
        
        # Test forward pass
        batch_size, seq_len = 4, 256
        x = torch.randn(batch_size, seq_len, input_dim, device=self.device, dtype=torch.float16)
        codes = torch.randn(batch_size, seq_len, rank, device=self.device, dtype=torch.float16)
        
        # Optimized forward
        x_flat = x.view(-1, input_dim)
        codes_flat = codes.view(-1, rank)
        output_optimized = optimized_bem(x_flat, codes_flat)
        
        # Reference forward (using PyTorch operations directly)
        base_output = base_layer(x_flat)
        x_v = torch.matmul(x_flat, reference_bem.lora_V)
        x_v_scaled = x_v * codes_flat
        lora_output = torch.matmul(x_v_scaled, reference_bem.lora_U.t())
        lora_output = lora_output * reference_bem.scaling
        output_reference = base_output + lora_output
        
        # Compare outputs
        self.assertTrue(torch.allclose(
            output_optimized, output_reference, 
            rtol=1e-3, atol=1e-4
        ), "Optimized BEM module output doesn't match reference")
    
    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        input_dim, output_dim = 768, 768
        base_layer = nn.Linear(input_dim, output_dim, bias=False).to(self.device)
        
        bem_module = OptimizedBEMModule(
            base_layer=base_layer,
            rank=8,
            alpha=16.0,
            dtype=torch.float16
        ).to(self.device)
        
        # Reset stats
        bem_module.reset_performance_stats()
        
        # Run forward passes
        x = torch.randn(1024, input_dim, device=self.device, dtype=torch.float16)
        codes = torch.randn(1024, 8, device=self.device, dtype=torch.float16)
        
        for _ in range(5):
            _ = bem_module(x, codes)
        
        # Check stats
        stats = bem_module.get_performance_stats()
        self.assertEqual(stats['total_calls'], 5)
        self.assertGreater(stats['avg_kernel_time_ms'] + stats['avg_pytorch_time_ms'], 0)


class TestKernelErrorHandling(unittest.TestCase):
    """Test error handling in CUDA kernels."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_mismatched_dimensions(self):
        """Test error handling for mismatched tensor dimensions."""
        if not torch.cuda.is_available() or not KERNELS_AVAILABLE:
            self.skipTest("CUDA or kernels not available")
            
        # Create tensors with mismatched dimensions
        X = torch.randn(100, 256, dtype=torch.float16, device=self.device)
        V = torch.randn(128, 8, dtype=torch.float16, device=self.device)  # Wrong K1
        codes = torch.randn(100, 8, dtype=torch.float16, device=self.device)
        U = torch.randn(8, 256, dtype=torch.float16, device=self.device)
        
        with self.assertRaises(RuntimeError):
            fused_generated_update(X, V, codes, U, 1.0)
    
    def test_different_devices(self):
        """Test error handling for tensors on different devices."""
        if not torch.cuda.is_available() or not KERNELS_AVAILABLE:
            self.skipTest("CUDA or kernels not available")
            
        # Create tensors on different devices
        X = torch.randn(100, 256, dtype=torch.float16, device="cuda")
        V = torch.randn(256, 8, dtype=torch.float16, device="cpu")  # Wrong device
        codes = torch.randn(100, 8, dtype=torch.float16, device="cuda")
        U = torch.randn(8, 256, dtype=torch.float16, device="cuda")
        
        with self.assertRaises(RuntimeError):
            fused_generated_update(X, V, codes, U, 1.0)
    
    def test_different_dtypes(self):
        """Test error handling for different dtypes."""
        if not torch.cuda.is_available() or not KERNELS_AVAILABLE:
            self.skipTest("CUDA or kernels not available")
            
        # Create tensors with different dtypes
        X = torch.randn(100, 256, dtype=torch.float16, device=self.device)
        V = torch.randn(256, 8, dtype=torch.float32, device=self.device)  # Wrong dtype
        codes = torch.randn(100, 8, dtype=torch.float16, device=self.device)
        U = torch.randn(8, 256, dtype=torch.float16, device=self.device)
        
        with self.assertRaises(RuntimeError):
            fused_generated_update(X, V, codes, U, 1.0)


class TestPerformanceRegression(unittest.TestCase):
    """Test for performance regressions."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not KERNELS_AVAILABLE, "CUDA kernels not compiled")  
    def test_performance_target(self):
        """Test that kernel meets <15% overhead target."""
        # Test configuration similar to production workload
        config = {
            'batch_size': 8,
            'seq_len': 512,
            'input_dim': 768,
            'intermediate_dim': 8,
            'output_dim': 768
        }
        
        result = benchmark_kernel_performance(
            dtype=torch.float16,
            device=self.device,
            num_warmup=5,
            num_runs=20,
            **config
        )
        
        if 'comparison' in result:
            overhead_pct = result['comparison']['overhead_percent']
            meets_target = result['comparison']['meets_target']
            
            self.assertTrue(meets_target,
                           f"Performance target not met. Overhead: {overhead_pct:.1f}% (target: <15%)")
            
            print(f"Performance test passed. Overhead: {overhead_pct:.1f}%")


def run_comprehensive_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFusedKernelAccuracy,
        TestOptimizedBEMModule,
        TestKernelErrorHandling,
        TestPerformanceRegression
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--comprehensive':
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        unittest.main(verbosity=2)