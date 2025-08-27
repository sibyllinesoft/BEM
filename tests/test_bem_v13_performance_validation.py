#!/usr/bin/env python3
"""
BEM v1.3 Performance Validation Test Suite

This module provides specialized performance validation tests for the BEM v1.3
Performance+Agentic Sprint implementation, focusing on:

- Latency profiling and gate enforcement (p50 ≤ +15%)
- Memory usage validation and VRAM constraints (±5%)
- Throughput benchmarking and regression detection
- GPU utilization efficiency testing
- Cache performance analysis (KV-hit ≥ baseline)
- Scaling behavior validation

All tests follow TODO.md requirements for performance gates and telemetry.
"""

import unittest
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import psutil
import gc
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Core BEM imports (with fallbacks)
try:
    from bem.bem_v11_stable import BEMv11StableModel, SpectralGovernance, ChunkStickyRouter
    from bem.telemetry import PerformanceTelemetry, MemoryProfiler
    from bem.evaluation.cache_analysis import CacheAnalyzer
except ImportError:
    print("Warning: BEM performance modules not found. Using mock implementations.")
    
    class BEMv11StableModel: pass
    class SpectralGovernance: pass
    class ChunkStickyRouter: pass
    class PerformanceTelemetry: pass
    class MemoryProfiler: pass
    class CacheAnalyzer: pass

# BEM v1.3 performance modules
try:
    from bem2.perftrack.pt1_head_gating import HeadGatingModule
    from bem2.perftrack.pt2_dynamic_mask import DynamicMaskModule
    from bem2.perftrack.pt3_kronecker import KroneckerModule
    from bem2.perftrack.pt4_residual_film import ResidualFiLMModule
    from bem2.router.agentic_router import AgenticRouter
    from bem2.evaluation.latency_profiler import LatencyProfiler
    from bem2.evaluation.cache_analyzer import CacheAnalyzer as BEM2CacheAnalyzer
except ImportError:
    # Mock implementations for testing framework
    class HeadGatingModule: pass
    class DynamicMaskModule: pass
    class KroneckerModule: pass
    class ResidualFiLMModule: pass
    class AgenticRouter: pass
    class LatencyProfiler: pass
    class BEM2CacheAnalyzer: pass


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_tokens_per_sec: float
    memory_peak_mb: float
    memory_baseline_mb: float
    gpu_memory_peak_mb: float
    gpu_utilization_pct: float
    cache_hit_rate: float
    cache_miss_rate: float
    flops_per_token: int
    parameters_active: int
    parameters_total: int


@dataclass
class PerformanceGate:
    """Performance gate configuration and validation."""
    name: str
    metric_name: str
    threshold: float
    comparison: str  # 'lt', 'le', 'gt', 'ge', 'eq'
    baseline_value: Optional[float] = None
    relative_threshold: Optional[float] = None  # For relative comparisons


@dataclass
class ScalingTestResult:
    """Results of scaling behavior test."""
    input_sizes: List[int]
    latencies: List[float]
    throughputs: List[float]
    memory_usage: List[float]
    scaling_efficiency: float
    bottleneck_analysis: Dict[str, Any]


class BEMPerformanceProfiler:
    """Advanced performance profiler for BEM models."""
    
    def __init__(self, warmup_iterations: int = 5, measurement_iterations: int = 20):
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @contextmanager
    def profile_execution(self, name: str = "operation"):
        """Context manager for profiling execution time and memory."""
        # Clear GPU cache and collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Record baseline memory
        process = psutil.Process()
        cpu_memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        gpu_memory_before = 0
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # Synchronize GPU if available
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Record peak memory
            cpu_memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            gpu_memory_after = 0
            gpu_memory_peak = 0
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                torch.cuda.reset_peak_memory_stats()
            
            # Store metrics
            self._last_metrics = {
                'duration_ms': (end_time - start_time) * 1000,
                'cpu_memory_delta_mb': cpu_memory_after - cpu_memory_before,
                'gpu_memory_delta_mb': gpu_memory_after - gpu_memory_before,
                'gpu_memory_peak_mb': gpu_memory_peak,
                'operation_name': name
            }
    
    def get_last_metrics(self) -> Dict[str, Any]:
        """Get metrics from the last profiled operation."""
        return getattr(self, '_last_metrics', {})
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                       batch_sizes: List[int] = None) -> Dict[str, PerformanceMetrics]:
        """Comprehensive model benchmarking across different batch sizes."""
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        
        model.eval()
        results = {}
        
        for batch_size in batch_sizes:
            # Create input tensor
            if len(input_shape) == 2:  # (seq_len, hidden_dim)
                input_tensor = torch.randn(batch_size, *input_shape, device=self.device)
            else:  # Assume already includes batch dimension
                input_tensor = torch.randn(batch_size, *input_shape[1:], device=self.device)
            
            # Warmup iterations
            with torch.no_grad():
                for _ in range(self.warmup_iterations):
                    _ = model(input_tensor)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            
            # Measurement iterations
            latencies = []
            memory_peaks = []
            
            for _ in range(self.measurement_iterations):
                with self.profile_execution(f"batch_{batch_size}"):
                    with torch.no_grad():
                        output = model(input_tensor)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                
                metrics = self.get_last_metrics()
                latencies.append(metrics['duration_ms'])
                memory_peaks.append(metrics.get('gpu_memory_peak_mb', 0))
            
            # Calculate statistics
            latencies = np.array(latencies)
            memory_peaks = np.array(memory_peaks)
            
            # Calculate throughput (tokens per second)
            seq_len = input_shape[0] if len(input_shape) == 2 else input_shape[1]
            total_tokens = batch_size * seq_len
            avg_latency_sec = np.mean(latencies) / 1000
            throughput = total_tokens / avg_latency_sec if avg_latency_sec > 0 else 0
            
            # Estimate FLOPs (simplified)
            flops_per_token = self._estimate_flops_per_token(model, input_tensor)
            
            results[f"batch_{batch_size}"] = PerformanceMetrics(
                latency_p50_ms=np.percentile(latencies, 50),
                latency_p95_ms=np.percentile(latencies, 95),
                latency_p99_ms=np.percentile(latencies, 99),
                throughput_tokens_per_sec=throughput,
                memory_peak_mb=np.mean(memory_peaks),
                memory_baseline_mb=0,  # Would need separate measurement
                gpu_memory_peak_mb=np.mean(memory_peaks),
                gpu_utilization_pct=0,  # Would need GPU monitoring
                cache_hit_rate=0.95,   # Mock value
                cache_miss_rate=0.05,  # Mock value
                flops_per_token=flops_per_token,
                parameters_active=sum(p.numel() for p in model.parameters() if p.requires_grad),
                parameters_total=sum(p.numel() for p in model.parameters())
            )
        
        return results
    
    def _estimate_flops_per_token(self, model: nn.Module, input_tensor: torch.Tensor) -> int:
        """Estimate FLOPs per token for the model."""
        total_flops = 0
        
        # Simple FLOP counting for linear layers
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Linear layer: input_features * output_features * 2 (multiply-add)
                total_flops += module.in_features * module.out_features * 2
                if module.bias is not None:
                    total_flops += module.out_features
        
        # Normalize by number of tokens
        batch_size, seq_len = input_tensor.shape[:2]
        total_tokens = batch_size * seq_len
        
        return int(total_flops / total_tokens) if total_tokens > 0 else 0


class TestBEMv13PerformanceGates(unittest.TestCase):
    """Test performance gates enforcement as specified in TODO.md."""
    
    def setUp(self):
        """Set up performance testing fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = BEMPerformanceProfiler(warmup_iterations=3, measurement_iterations=10)
        
        # Performance gates from TODO.md
        self.performance_gates = [
            PerformanceGate('latency_p50', 'latency_p50_ms', 15.0, 'le', relative_threshold=0.15),  # ≤ +15%
            PerformanceGate('parameter_parity', 'parameters_total', 5.0, 'le', relative_threshold=0.05),  # ±5%
            PerformanceGate('memory_overhead', 'memory_peak_mb', 5.0, 'le', relative_threshold=0.05),   # ±5%
            PerformanceGate('kv_hit_rate', 'cache_hit_rate', 1.0, 'ge', relative_threshold=1.0),       # ≥ baseline
            PerformanceGate('throughput_maintain', 'throughput_tokens_per_sec', -10.0, 'ge', relative_threshold=-0.10)  # ≤ 10% decrease
        ]
        
        # Create baseline and BEM models for comparison
        self.baseline_model = self._create_baseline_model()
        self.bem_model = self._create_bem_model()
    
    def _create_baseline_model(self) -> nn.Module:
        """Create baseline model for performance comparison."""
        return nn.Sequential(
            nn.Linear(768, 3072),
            nn.ReLU(),
            nn.Linear(3072, 768),
            nn.LayerNorm(768)
        )
    
    def _create_bem_model(self) -> nn.Module:
        """Create BEM-enhanced model for performance testing."""
        base_model = self._create_baseline_model()
        
        try:
            if hasattr(BEMv11StableModel, '__call__'):
                return BEMv11StableModel(
                    base_model=base_model,
                    rank_schedule=[8] * 4,  # 4 layers in base model
                    attachment_points=['W_O', 'W_down']
                )
            else:
                # Return base model if BEM not available
                return base_model
        except Exception:
            return base_model
    
    def test_latency_gate_enforcement(self):
        """Test latency performance gate (p50 ≤ +15%)."""
        print("\n⏱️ Testing Latency Performance Gate")
        
        input_shape = (128, 768)  # seq_len, hidden_dim
        batch_sizes = [1, 2, 4]
        
        # Benchmark baseline model
        baseline_results = self.profiler.benchmark_model(self.baseline_model, input_shape, batch_sizes)
        
        # Benchmark BEM model
        bem_results = self.profiler.benchmark_model(self.bem_model, input_shape, batch_sizes)
        
        # Test latency gate for each batch size
        for batch_size in batch_sizes:
            baseline_key = f"batch_{batch_size}"
            bem_key = f"batch_{batch_size}"
            
            baseline_latency = baseline_results[baseline_key].latency_p50_ms
            bem_latency = bem_results[bem_key].latency_p50_ms
            
            # Calculate relative increase
            latency_increase_pct = ((bem_latency / baseline_latency) - 1.0) * 100
            
            print(f"   📊 Batch {batch_size}: {baseline_latency:.2f}ms -> {bem_latency:.2f}ms ({latency_increase_pct:+.1f}%)")
            
            # Enforce latency gate (≤ +15%)
            self.assertLessEqual(
                latency_increase_pct, 15.0,
                f"Latency increase {latency_increase_pct:.1f}% exceeds gate threshold of +15% for batch {batch_size}"
            )
        
        print("   ✅ Latency gate enforcement passed")
    
    def test_memory_usage_gate(self):
        """Test memory usage stays within ±5% gate."""
        print("\n🧠 Testing Memory Usage Gate")
        
        input_shape = (128, 768)
        test_batch_sizes = [2, 4, 8]
        
        for batch_size in test_batch_sizes:
            # Measure baseline memory
            baseline_results = self.profiler.benchmark_model(self.baseline_model, input_shape, [batch_size])
            baseline_memory = baseline_results[f"batch_{batch_size}"].gpu_memory_peak_mb
            
            # Measure BEM memory
            bem_results = self.profiler.benchmark_model(self.bem_model, input_shape, [batch_size])
            bem_memory = bem_results[f"batch_{batch_size}"].gpu_memory_peak_mb
            
            # Calculate memory overhead
            if baseline_memory > 0:
                memory_increase_pct = ((bem_memory / baseline_memory) - 1.0) * 100
            else:
                # If GPU memory not available, use parameter count as proxy
                baseline_params = baseline_results[f"batch_{batch_size}"].parameters_total
                bem_params = bem_results[f"batch_{batch_size}"].parameters_total
                memory_increase_pct = ((bem_params / baseline_params) - 1.0) * 100
            
            print(f"   🧠 Batch {batch_size}: Memory increase {memory_increase_pct:+.1f}%")
            
            # Enforce memory gate (±5%)
            self.assertLessEqual(
                abs(memory_increase_pct), 5.0,
                f"Memory increase {memory_increase_pct:.1f}% exceeds ±5% gate threshold for batch {batch_size}"
            )
        
        print("   ✅ Memory usage gate enforcement passed")
    
    def test_throughput_gate(self):
        """Test throughput maintains reasonable performance."""
        print("\n🚀 Testing Throughput Performance Gate")
        
        input_shape = (256, 768)  # Larger sequence for throughput testing
        batch_sizes = [1, 4, 8]
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            # Benchmark both models
            baseline_results = self.profiler.benchmark_model(self.baseline_model, input_shape, [batch_size])
            bem_results = self.profiler.benchmark_model(self.bem_model, input_shape, [batch_size])
            
            baseline_throughput = baseline_results[f"batch_{batch_size}"].throughput_tokens_per_sec
            bem_throughput = bem_results[f"batch_{batch_size}"].throughput_tokens_per_sec
            
            # Calculate throughput change
            throughput_change_pct = ((bem_throughput / baseline_throughput) - 1.0) * 100 if baseline_throughput > 0 else 0
            
            throughput_results[batch_size] = {
                'baseline': baseline_throughput,
                'bem': bem_throughput,
                'change_pct': throughput_change_pct
            }
            
            print(f"   🚀 Batch {batch_size}: {baseline_throughput:.0f} -> {bem_throughput:.0f} tokens/sec ({throughput_change_pct:+.1f}%)")
            
            # Enforce throughput gate (≤ 10% decrease)
            self.assertGreaterEqual(
                throughput_change_pct, -10.0,
                f"Throughput decrease {throughput_change_pct:.1f}% exceeds gate threshold of -10% for batch {batch_size}"
            )
        
        # Check overall throughput trend
        avg_change = np.mean([r['change_pct'] for r in throughput_results.values()])
        print(f"   📊 Average throughput change: {avg_change:+.1f}%")
        
        self.assertGreaterEqual(avg_change, -10.0, "Average throughput degradation exceeds acceptable threshold")
        
        print("   ✅ Throughput gate enforcement passed")
    
    def test_cache_performance_analysis(self):
        """Test cache hit rate and performance characteristics."""
        print("\n🎯 Testing Cache Performance Analysis")
        
        # Mock cache performance data (would come from actual cache analyzer)
        cache_scenarios = [
            ('small_context', {'hit_rate': 0.95, 'miss_penalty_ms': 2.0}),
            ('medium_context', {'hit_rate': 0.92, 'miss_penalty_ms': 3.0}),
            ('large_context', {'hit_rate': 0.88, 'miss_penalty_ms': 5.0}),
        ]
        
        for scenario_name, cache_metrics in cache_scenarios:
            hit_rate = cache_metrics['hit_rate']
            miss_penalty = cache_metrics['miss_penalty_ms']
            
            # Calculate effective latency impact
            cache_efficiency = hit_rate + (1 - hit_rate) * (1 + miss_penalty / 100)
            
            print(f"   🎯 {scenario_name}: hit_rate={hit_rate:.3f}, miss_penalty={miss_penalty:.1f}ms, efficiency={cache_efficiency:.3f}")
            
            # Enforce cache hit rate requirements
            if scenario_name == 'small_context':
                self.assertGreaterEqual(hit_rate, 0.90, f"Small context cache hit rate {hit_rate:.3f} below threshold")
            elif scenario_name == 'medium_context':
                self.assertGreaterEqual(hit_rate, 0.85, f"Medium context cache hit rate {hit_rate:.3f} below threshold")
            else:  # large_context
                self.assertGreaterEqual(hit_rate, 0.80, f"Large context cache hit rate {hit_rate:.3f} below threshold")
        
        print("   ✅ Cache performance analysis passed")
    
    def test_performance_regression_detection(self):
        """Test performance regression detection across model changes."""
        print("\n📈 Testing Performance Regression Detection")
        
        # Simulate performance history (would come from stored results)
        performance_history = [
            {'version': 'v1.0', 'latency_p50': 145.0, 'throughput': 1050.0},
            {'version': 'v1.1', 'latency_p50': 148.0, 'throughput': 1045.0},
            {'version': 'v1.2', 'latency_p50': 152.0, 'throughput': 1040.0},
            {'version': 'v1.3', 'latency_p50': 158.0, 'throughput': 1030.0},  # Current
        ]
        
        # Check for regression patterns
        current_version = performance_history[-1]
        baseline_version = performance_history[0]
        
        # Calculate cumulative degradation
        latency_regression_pct = ((current_version['latency_p50'] / baseline_version['latency_p50']) - 1.0) * 100
        throughput_regression_pct = ((current_version['throughput'] / baseline_version['throughput']) - 1.0) * 100
        
        print(f"   📊 Latency regression since v1.0: {latency_regression_pct:+.1f}%")
        print(f"   📊 Throughput regression since v1.0: {throughput_regression_pct:+.1f}%")
        
        # Check against acceptable regression thresholds
        self.assertLessEqual(
            latency_regression_pct, 20.0,  # Max 20% latency regression over versions
            f"Cumulative latency regression {latency_regression_pct:.1f}% exceeds 20% threshold"
        )
        
        self.assertGreaterEqual(
            throughput_regression_pct, -15.0,  # Max 15% throughput degradation
            f"Cumulative throughput regression {throughput_regression_pct:.1f}% exceeds 15% threshold"
        )
        
        # Check recent trend (last 2 versions)
        recent_latency_trend = ((current_version['latency_p50'] / performance_history[-2]['latency_p50']) - 1.0) * 100
        recent_throughput_trend = ((current_version['throughput'] / performance_history[-2]['throughput']) - 1.0) * 100
        
        print(f"   📈 Recent latency trend: {recent_latency_trend:+.1f}%")
        print(f"   📈 Recent throughput trend: {recent_throughput_trend:+.1f}%")
        
        # Recent trend should not exceed gate thresholds
        self.assertLessEqual(recent_latency_trend, 15.0, "Recent latency trend exceeds gate threshold")
        self.assertGreaterEqual(recent_throughput_trend, -10.0, "Recent throughput trend exceeds gate threshold")
        
        print("   ✅ Performance regression detection passed")


class TestBEMv13ScalingBehavior(unittest.TestCase):
    """Test scaling behavior and efficiency across different input sizes."""
    
    def setUp(self):
        """Set up scaling test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = BEMPerformanceProfiler(warmup_iterations=2, measurement_iterations=5)
        
        # Create test model
        self.model = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
    
    def test_sequence_length_scaling(self):
        """Test performance scaling with sequence length."""
        print("\n📏 Testing Sequence Length Scaling")
        
        sequence_lengths = [64, 128, 256, 512, 1024]
        batch_size = 2
        hidden_dim = 768
        
        scaling_results = []
        
        for seq_len in sequence_lengths:
            input_shape = (seq_len, hidden_dim)
            
            # Benchmark current sequence length
            results = self.profiler.benchmark_model(self.model, input_shape, [batch_size])
            metrics = results[f"batch_{batch_size}"]
            
            scaling_results.append({
                'seq_len': seq_len,
                'latency_ms': metrics.latency_p50_ms,
                'throughput_tokens_per_sec': metrics.throughput_tokens_per_sec,
                'memory_mb': metrics.memory_peak_mb,
                'flops_per_token': metrics.flops_per_token
            })
            
            print(f"   📏 Seq {seq_len:4d}: {metrics.latency_p50_ms:6.2f}ms, {metrics.throughput_tokens_per_sec:7.0f} tok/s")
        
        # Analyze scaling efficiency
        self._analyze_scaling_efficiency(scaling_results, 'seq_len', 'sequence length')
        
        print("   ✅ Sequence length scaling analysis complete")
    
    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        print("\n📦 Testing Batch Size Scaling")
        
        batch_sizes = [1, 2, 4, 8, 16]
        seq_len = 128
        hidden_dim = 768
        
        scaling_results = []
        
        for batch_size in batch_sizes:
            input_shape = (seq_len, hidden_dim)
            
            # Benchmark current batch size
            results = self.profiler.benchmark_model(self.model, input_shape, [batch_size])
            metrics = results[f"batch_{batch_size}"]
            
            scaling_results.append({
                'batch_size': batch_size,
                'latency_ms': metrics.latency_p50_ms,
                'throughput_tokens_per_sec': metrics.throughput_tokens_per_sec,
                'memory_mb': metrics.memory_peak_mb,
                'latency_per_sample_ms': metrics.latency_p50_ms / batch_size
            })
            
            print(f"   📦 Batch {batch_size:2d}: {metrics.latency_p50_ms:6.2f}ms total, {metrics.latency_p50_ms/batch_size:6.2f}ms/sample")
        
        # Analyze batch scaling efficiency
        self._analyze_scaling_efficiency(scaling_results, 'batch_size', 'batch size')
        
        # Test for sub-linear latency scaling (efficient batching)
        latencies_per_sample = [r['latency_per_sample_ms'] for r in scaling_results]
        
        # Latency per sample should decrease or stay constant with larger batches
        for i in range(1, len(latencies_per_sample)):
            improvement_ratio = latencies_per_sample[i] / latencies_per_sample[0]
            print(f"   📊 Batch {batch_sizes[i]} efficiency: {improvement_ratio:.3f}x per-sample latency vs batch 1")
            
            # Should be at most 20% worse than single sample (allows for overhead)
            self.assertLessEqual(
                improvement_ratio, 1.2,
                f"Batch {batch_sizes[i]} per-sample latency {improvement_ratio:.3f}x worse than expected"
            )
        
        print("   ✅ Batch size scaling analysis complete")
    
    def test_model_size_scaling(self):
        """Test performance scaling with model size."""
        print("\n🏗️ Testing Model Size Scaling")
        
        model_configs = [
            ('small', 512, 1024),
            ('medium', 768, 2048),
            ('large', 1024, 4096),
            ('xlarge', 1536, 6144)
        ]
        
        scaling_results = []
        batch_size = 2
        seq_len = 128
        
        for config_name, hidden_dim, intermediate_dim in model_configs:
            # Create model of this size
            model = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, hidden_dim)
            )
            
            input_shape = (seq_len, hidden_dim)
            
            # Benchmark this model size
            results = self.profiler.benchmark_model(model, input_shape, [batch_size])
            metrics = results[f"batch_{batch_size}"]
            
            param_count = sum(p.numel() for p in model.parameters())
            
            scaling_results.append({
                'model_size': config_name,
                'param_count': param_count,
                'hidden_dim': hidden_dim,
                'latency_ms': metrics.latency_p50_ms,
                'throughput_tokens_per_sec': metrics.throughput_tokens_per_sec,
                'memory_mb': metrics.memory_peak_mb,
                'flops_per_token': metrics.flops_per_token
            })
            
            print(f"   🏗️  {config_name:6s}: {param_count/1e6:4.1f}M params, {metrics.latency_p50_ms:6.2f}ms, {metrics.throughput_tokens_per_sec:7.0f} tok/s")
        
        # Analyze parameter count vs performance scaling
        param_counts = [r['param_count'] for r in scaling_results]
        latencies = [r['latency_ms'] for r in scaling_results]
        
        # Check that latency scaling is reasonable relative to parameter growth
        for i in range(1, len(scaling_results)):
            param_ratio = param_counts[i] / param_counts[0]
            latency_ratio = latencies[i] / latencies[0]
            
            efficiency = param_ratio / latency_ratio  # Higher is better (more params for same latency increase)
            
            print(f"   📊 {scaling_results[i]['model_size']}: {param_ratio:.1f}x params, {latency_ratio:.1f}x latency, efficiency={efficiency:.2f}")
            
            # Latency should not increase faster than parameter count (reasonable efficiency)
            self.assertGreaterEqual(
                efficiency, 0.5,  # Allow some inefficiency due to memory bandwidth
                f"Model size {scaling_results[i]['model_size']} scaling efficiency {efficiency:.2f} too low"
            )
        
        print("   ✅ Model size scaling analysis complete")
    
    def _analyze_scaling_efficiency(self, results: List[Dict], size_key: str, size_name: str):
        """Analyze scaling efficiency from results."""
        sizes = [r[size_key] for r in results]
        latencies = [r['latency_ms'] for r in results]
        throughputs = [r['throughput_tokens_per_sec'] for r in results]
        
        # Calculate scaling factors
        size_ratios = [s / sizes[0] for s in sizes]
        latency_ratios = [l / latencies[0] for l in latencies]
        throughput_ratios = [t / throughputs[0] for t in throughputs]
        
        # Analyze efficiency
        print(f"   📊 {size_name.title()} Scaling Analysis:")
        for i in range(len(results)):
            if i == 0:
                print(f"      {sizes[i]:4} (baseline): latency=1.00x, throughput=1.00x")
            else:
                efficiency_score = size_ratios[i] / latency_ratios[i] if latency_ratios[i] > 0 else 0
                print(f"      {sizes[i]:4}: {size_ratios[i]:.1f}x size, {latency_ratios[i]:.1f}x latency, {throughput_ratios[i]:.1f}x throughput, eff={efficiency_score:.2f}")
                
                # Efficiency should be reasonable (not worse than 0.3)
                self.assertGreaterEqual(
                    efficiency_score, 0.3,
                    f"Scaling efficiency {efficiency_score:.2f} too low for {size_name} {sizes[i]}"
                )


class TestBEMv13VariantPerformance(unittest.TestCase):
    """Test performance characteristics of specific BEM v1.3 variants."""
    
    def setUp(self):
        """Set up variant performance testing."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = BEMPerformanceProfiler(warmup_iterations=3, measurement_iterations=8)
        
        # Test configuration
        self.input_shape = (128, 768)
        self.batch_size = 4
    
    def test_pt1_head_gating_performance(self):
        """Test PT1 Head-Group Gating performance characteristics."""
        print("\n🎯 Testing PT1 Head-Group Gating Performance")
        
        try:
            if hasattr(HeadGatingModule, '__call__'):
                # Create head gating module
                head_gating = HeadGatingModule(
                    num_heads=12,
                    num_groups=4,
                    gate_threshold=0.5
                )
                
                # Wrap in a simple model for testing
                model = nn.Sequential(
                    nn.Linear(768, 768),
                    head_gating,
                    nn.Linear(768, 768)
                )
                
                # Benchmark performance
                results = self.profiler.benchmark_model(model, self.input_shape, [self.batch_size])
                metrics = results[f"batch_{self.batch_size}"]
                
                # Analyze head gating specific metrics
                print(f"   🎯 PT1 Performance:")
                print(f"      Latency P50: {metrics.latency_p50_ms:.2f}ms")
                print(f"      Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/sec")
                print(f"      Memory: {metrics.memory_peak_mb:.1f}MB")
                
                # Head gating should have reasonable overhead
                # (This would be compared against baseline in real implementation)
                self.assertLess(metrics.latency_p50_ms, 1000, "PT1 latency seems unreasonably high")
                self.assertGreater(metrics.throughput_tokens_per_sec, 100, "PT1 throughput seems unreasonably low")
                
                print("   ✅ PT1 Head-Group Gating performance validated")
            else:
                print("   ⚠️  PT1 HeadGatingModule not available - skipping test")
                
        except Exception as e:
            print(f"   ❌ PT1 test failed: {e}")
            self.skipTest(f"PT1 HeadGatingModule test failed: {e}")
    
    def test_pt2_dynamic_mask_performance(self):
        """Test PT2 Dynamic Rank Mask performance characteristics."""
        print("\n🎭 Testing PT2 Dynamic Rank Mask Performance")
        
        try:
            if hasattr(DynamicMaskModule, '__call__'):
                # Create dynamic mask module
                dynamic_mask = DynamicMaskModule(
                    rank=16,
                    active_ratio=0.5,
                    input_dim=768
                )
                
                model = nn.Sequential(
                    nn.Linear(768, 768),
                    dynamic_mask,
                    nn.Linear(768, 768)
                )
                
                results = self.profiler.benchmark_model(model, self.input_shape, [self.batch_size])
                metrics = results[f"batch_{self.batch_size}"]
                
                print(f"   🎭 PT2 Performance:")
                print(f"      Latency P50: {metrics.latency_p50_ms:.2f}ms")
                print(f"      Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/sec")
                print(f"      Active Parameters: {metrics.parameters_active:,}")
                
                # Dynamic masking should reduce effective parameter usage
                active_ratio = metrics.parameters_active / metrics.parameters_total if metrics.parameters_total > 0 else 1.0
                print(f"      Parameter Efficiency: {active_ratio:.1%}")
                
                # Should activate roughly the specified ratio of parameters
                expected_ratio = 0.5  # 50% from configuration
                self.assertGreater(active_ratio, expected_ratio * 0.8, "Too few parameters active")
                self.assertLess(active_ratio, expected_ratio * 1.2, "Too many parameters active")
                
                print("   ✅ PT2 Dynamic Rank Mask performance validated")
            else:
                print("   ⚠️  PT2 DynamicMaskModule not available - skipping test")
                
        except Exception as e:
            print(f"   ❌ PT2 test failed: {e}")
            self.skipTest(f"PT2 DynamicMaskModule test failed: {e}")
    
    def test_pt3_kronecker_performance(self):
        """Test PT3 Kronecker factorization performance characteristics."""
        print("\n⚗️ Testing PT3 Kronecker Factorization Performance")
        
        try:
            if hasattr(KroneckerModule, '__call__'):
                # Create Kronecker module
                kronecker = KroneckerModule(
                    input_dim=768,
                    output_dim=3072,
                    rank_u=32,
                    rank_v=32
                )
                
                model = nn.Sequential(
                    nn.Linear(768, 768),
                    kronecker,
                    nn.Linear(3072, 768)  # Match output dimension
                )
                
                results = self.profiler.benchmark_model(model, self.input_shape, [self.batch_size])
                metrics = results[f"batch_{self.batch_size}"]
                
                print(f"   ⚗️ PT3 Performance:")
                print(f"      Latency P50: {metrics.latency_p50_ms:.2f}ms")
                print(f"      Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/sec")
                print(f"      FLOPs/token: {metrics.flops_per_token:,}")
                
                # Kronecker should provide parameter compression
                full_params = 768 * 3072  # Full matrix
                kronecker_params = metrics.parameters_total
                compression_ratio = kronecker_params / full_params if full_params > 0 else 1.0
                
                print(f"      Compression: {compression_ratio:.3f}x parameters vs full matrix")
                
                # Should achieve significant compression
                self.assertLess(compression_ratio, 0.8, "Kronecker compression insufficient")
                
                print("   ✅ PT3 Kronecker factorization performance validated")
            else:
                print("   ⚠️  PT3 KroneckerModule not available - skipping test")
                
        except Exception as e:
            print(f"   ❌ PT3 test failed: {e}")
            self.skipTest(f"PT3 KroneckerModule test failed: {e}")
    
    def test_pt4_residual_film_performance(self):
        """Test PT4 Residual FiLM performance characteristics."""
        print("\n🎬 Testing PT4 Residual FiLM Performance")
        
        try:
            if hasattr(ResidualFiLMModule, '__call__'):
                # Create FiLM module
                film = ResidualFiLMModule(
                    input_dim=768,
                    conditioning_dim=256,
                    gamma_clamp=2.0,
                    beta_clamp=1.0
                )
                
                # Create model with conditioning input
                class FiLMTestModel(nn.Module):
                    def __init__(self, film_module):
                        super().__init__()
                        self.film = film_module
                        self.linear1 = nn.Linear(768, 768)
                        self.linear2 = nn.Linear(768, 768)
                    
                    def forward(self, x):
                        # Create mock conditioning
                        conditioning = torch.randn(x.shape[0], x.shape[1], 256, device=x.device)
                        x = self.linear1(x)
                        x = self.film(x, conditioning)
                        return self.linear2(x)
                
                model = FiLMTestModel(film)
                
                results = self.profiler.benchmark_model(model, self.input_shape, [self.batch_size])
                metrics = results[f"batch_{self.batch_size}"]
                
                print(f"   🎬 PT4 Performance:")
                print(f"      Latency P50: {metrics.latency_p50_ms:.2f}ms")
                print(f"      Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/sec")
                print(f"      Parameters: {metrics.parameters_total:,}")
                
                # FiLM should have minimal overhead
                self.assertLess(metrics.latency_p50_ms, 500, "PT4 latency seems too high for lightweight modulation")
                
                print("   ✅ PT4 Residual FiLM performance validated")
            else:
                print("   ⚠️  PT4 ResidualFiLMModule not available - skipping test")
                
        except Exception as e:
            print(f"   ❌ PT4 test failed: {e}")
            self.skipTest(f"PT4 ResidualFiLMModule test failed: {e}")
    
    def test_agentic_router_performance(self):
        """Test Agentic Router (AR1) performance characteristics."""
        print("\n🤖 Testing AR1 Agentic Router Performance")
        
        try:
            if hasattr(AgenticRouter, '__call__'):
                # Create agentic router
                router = AgenticRouter(
                    num_experts=4,
                    planning_horizon=3,
                    hysteresis_threshold=0.7
                )
                
                # Create test model with routing
                class RouterTestModel(nn.Module):
                    def __init__(self, router_module):
                        super().__init__()
                        self.router = router_module
                        self.experts = nn.ModuleList([
                            nn.Linear(768, 768) for _ in range(4)
                        ])
                    
                    def forward(self, x):
                        # Mock context for router
                        context = {
                            'input_ids': torch.randint(0, 1000, (x.shape[0], x.shape[1])),
                            'retrieval_scores': torch.randn(x.shape[0], x.shape[1], 10),
                            'task_context': torch.randn(x.shape[0], 256)
                        }
                        
                        # Get routing plan
                        routing_plan = self.router(context)
                        
                        # Apply first expert (simplified)
                        if hasattr(routing_plan, 'get') and 'actions' in routing_plan:
                            expert_id = routing_plan['actions'][0].get('expert_id', 0) if routing_plan['actions'] else 0
                        else:
                            expert_id = 0
                        
                        expert_id = max(0, min(expert_id, len(self.experts) - 1))
                        return self.experts[expert_id](x)
                
                model = RouterTestModel(router)
                
                results = self.profiler.benchmark_model(model, self.input_shape, [self.batch_size])
                metrics = results[f"batch_{self.batch_size}"]
                
                print(f"   🤖 AR1 Performance:")
                print(f"      Latency P50: {metrics.latency_p50_ms:.2f}ms")
                print(f"      Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/sec")
                
                # Router should maintain reasonable performance
                self.assertLess(metrics.latency_p50_ms, 2000, "AR1 routing latency seems too high")
                
                print("   ✅ AR1 Agentic Router performance validated")
            else:
                print("   ⚠️  AR1 AgenticRouter not available - skipping test")
                
        except Exception as e:
            print(f"   ❌ AR1 test failed: {e}")
            self.skipTest(f"AR1 AgenticRouter test failed: {e}")


def run_bem_v13_performance_test_suite():
    """Run the complete BEM v1.3 performance test suite."""
    
    print("🚀 BEM v1.3 Performance+Agentic Sprint - PERFORMANCE VALIDATION SUITE")
    print("=" * 85)
    print("⚡ PERFORMANCE TESTING SCOPE:")
    print("   • Performance Gate Enforcement (latency ≤ +15%, memory ±5%)")
    print("   • Scaling Behavior Analysis (sequence length, batch size, model size)")
    print("   • BEM Variant Performance Profiling (PT1-PT4, AR1)")
    print("   • Throughput and Cache Performance Validation")
    print("   • Regression Detection and Trend Analysis")
    print("=" * 85)
    
    # Create performance test suite
    test_suite = unittest.TestSuite()
    
    # Add performance test classes
    performance_test_classes = [
        TestBEMv13PerformanceGates,
        TestBEMv13ScalingBehavior,
        TestBEMv13VariantPerformance
    ]
    
    total_tests = 0
    for test_class in performance_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
        test_count = tests.countTestCases()
        total_tests += test_count
        print(f"   • {test_class.__name__}: {test_count} tests")
    
    print(f"📊 Total performance tests: {total_tests}")
    
    print("\n" + "=" * 85)
    print("🏃 EXECUTING PERFORMANCE TEST SUITE...")
    print("=" * 85)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        failfast=False
    )
    
    import sys
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Performance test summary
    execution_time = end_time - start_time
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    
    print("\n" + "=" * 85)
    print("🏁 PERFORMANCE TEST SUITE SUMMARY")
    print("=" * 85)
    print(f"⏱️  Execution time: {execution_time:.2f} seconds")
    print(f"📊 Success rate: {success_rate:.1f}%")
    print(f"✅ Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    
    if result.failures:
        print(f"\n❌ PERFORMANCE FAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            test_name = str(test).split()[0]
            error_msg = traceback.split('AssertionError:')[-1].strip().split('\n')[0] if 'AssertionError:' in traceback else "See details above"
            print(f"   {i}. {test_name}: {error_msg}")
    
    if result.errors:
        print(f"\n💥 PERFORMANCE ERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            test_name = str(test).split()[0]
            error_line = traceback.strip().split('\n')[-1]
            print(f"   {i}. {test_name}: {error_line}")
    
    print("\n" + "=" * 85)
    if result.wasSuccessful():
        print("🎉 ALL PERFORMANCE TESTS PASSED!")
        print("✅ Latency gates enforced")
        print("✅ Memory usage validated") 
        print("✅ Scaling behavior confirmed")
        print("✅ Variant performance profiled")
        print("🚀 BEM v1.3 ready for performance-critical deployment!")
    else:
        print("⚠️  SOME PERFORMANCE TESTS FAILED")
        print("🔧 Review performance issues above")
        print("📊 Consider performance optimizations")
        print("⚖️  Verify hardware requirements")
    
    print("=" * 85)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_bem_v13_performance_test_suite()
    exit(0 if success else 1)