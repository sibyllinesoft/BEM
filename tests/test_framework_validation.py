#!/usr/bin/env python3
"""
Validation script for BEM v1.3 Testing Framework
Tests core functionality without requiring full PyTorch installation
"""

import sys
import traceback
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import unittest
from unittest.mock import Mock, MagicMock

@dataclass
class ParityResult:
    """Result of parameter/FLOP parity validation."""
    param_ratio: float
    flop_ratio: float
    params_within_tolerance: bool
    flops_within_tolerance: bool
    tolerance: float

@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run."""
    accuracy: float
    loss: float
    perplexity: float
    latency_p50: float
    latency_p95: float
    memory_peak: float
    throughput: float
    param_count: int
    flop_count: int
    kv_cache_hits: int
    kv_cache_total: int
    routing_entropy: float
    expert_utilization: Dict[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

def bca_bootstrap_ci(data: np.ndarray, statistic_func, alpha: float = 0.05, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """
    Compute bias-corrected and accelerated (BCa) bootstrap confidence interval.
    
    Args:
        data: Input data array
        statistic_func: Function to compute statistic (e.g., np.mean)
        alpha: Significance level (default 0.05 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(data)
    
    # Original statistic
    theta_hat = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Bias-correction
    z_0 = stats.norm.ppf(np.mean(bootstrap_stats < theta_hat))
    
    # Acceleration (jackknife)
    jackknife_stats = []
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats.append(statistic_func(jackknife_sample))
    
    jackknife_stats = np.array(jackknife_stats)
    theta_dot = np.mean(jackknife_stats)
    
    a_hat = np.sum((theta_dot - jackknife_stats)**3) / (6 * (np.sum((theta_dot - jackknife_stats)**2))**1.5)
    
    # BCa bounds
    z_alpha_2 = stats.norm.ppf(alpha/2)
    z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)
    
    alpha_1 = stats.norm.cdf(z_0 + (z_0 + z_alpha_2)/(1 - a_hat*(z_0 + z_alpha_2)))
    alpha_2 = stats.norm.cdf(z_0 + (z_0 + z_1_alpha_2)/(1 - a_hat*(z_0 + z_1_alpha_2)))
    
    # Handle edge cases
    alpha_1 = max(0, min(1, alpha_1))
    alpha_2 = max(0, min(1, alpha_2))
    
    lower_bound = np.percentile(bootstrap_stats, 100 * alpha_1)
    upper_bound = np.percentile(bootstrap_stats, 100 * alpha_2)
    
    return lower_bound, upper_bound

def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg procedure.
    
    Args:
        p_values: List of p-values to correct
        alpha: FDR level (default 0.05)
        
    Returns:
        Tuple of (reject_flags, corrected_p_values)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and get indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Benjamini-Hochberg procedure
    corrected_p_values = np.zeros_like(p_values)
    reject_flags = np.zeros(n, dtype=bool)
    
    # Calculate adjusted p-values
    for i in range(n-1, -1, -1):
        rank = i + 1
        corrected_p_values[sorted_indices[i]] = min(1.0, sorted_p_values[i] * n / rank)
        if i < n - 1:
            corrected_p_values[sorted_indices[i]] = min(
                corrected_p_values[sorted_indices[i]], 
                corrected_p_values[sorted_indices[i+1]]
            )
    
    # Determine rejections
    reject_flags = corrected_p_values <= alpha
    
    return reject_flags.tolist(), corrected_p_values.tolist()

class TestFrameworkValidation(unittest.TestCase):
    """Validate core testing framework functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
    
    def test_bca_bootstrap_confidence_interval(self):
        """Test BCa bootstrap confidence interval calculation."""
        print("\n🔬 Testing BCa Bootstrap Confidence Intervals")
        
        # Generate test data with known distribution
        true_mean = 10.0
        data = np.random.normal(true_mean, 2.0, size=100)
        
        # Compute BCa CI for mean
        lower, upper = bca_bootstrap_ci(data, np.mean, alpha=0.05, n_bootstrap=1000)
        
        print(f"   📊 Data mean: {np.mean(data):.3f}")
        print(f"   📊 BCa 95% CI: [{lower:.3f}, {upper:.3f}]")
        print(f"   📊 True mean in CI: {lower <= true_mean <= upper}")
        
        # Validate CI properties
        self.assertLess(lower, upper, "Lower bound should be less than upper bound")
        self.assertLess(abs(lower - upper), 3.0, "CI width should be reasonable for this data")
        
        # For large sample from normal distribution, CI should contain true mean most of the time
        # Note: This is probabilistic, so we don't assert it in every test run
        
    def test_fdr_correction(self):
        """Test False Discovery Rate correction."""
        print("\n🔬 Testing FDR Correction (Benjamini-Hochberg)")
        
        # Generate test p-values with some true positives
        p_values = [0.001, 0.004, 0.03, 0.08, 0.12, 0.25, 0.45, 0.67, 0.89]
        
        reject_flags, corrected_p_values = fdr_correction(p_values, alpha=0.05)
        
        print(f"   📊 Original p-values: {p_values}")
        print(f"   📊 Corrected p-values: {[f'{p:.3f}' for p in corrected_p_values]}")
        print(f"   📊 Rejected hypotheses: {sum(reject_flags)}/{len(p_values)}")
        
        # Validate correction properties
        self.assertEqual(len(reject_flags), len(p_values))
        self.assertEqual(len(corrected_p_values), len(p_values))
        
        # Corrected p-values should be >= original p-values
        for orig, corr in zip(p_values, corrected_p_values):
            self.assertGreaterEqual(corr, orig, "Corrected p-value should be >= original")
        
        # Very small p-values should typically be rejected
        self.assertTrue(reject_flags[0], "Very small p-value (0.001) should be rejected")
        
    def test_parity_validation(self):
        """Test parameter and FLOP parity validation logic."""
        print("\n🔬 Testing Parity Validation Logic")
        
        baseline_params = 1000000
        baseline_flops = 5000000
        tolerance = 0.05  # 5%
        
        # Test cases: within tolerance, outside tolerance
        test_cases = [
            ("within_tolerance", 1020000, 5100000, True, True),
            ("params_outside", 1060000, 5100000, False, True),
            ("flops_outside", 1020000, 5300000, True, False),
            ("both_outside", 1060000, 5300000, False, False),
        ]
        
        for case_name, bem_params, bem_flops, expect_param_ok, expect_flop_ok in test_cases:
            with self.subTest(case=case_name):
                # Simulate parity validation logic
                param_ratio = bem_params / baseline_params
                flop_ratio = bem_flops / baseline_flops
                
                params_within_tolerance = abs(param_ratio - 1.0) <= tolerance
                flops_within_tolerance = abs(flop_ratio - 1.0) <= tolerance
                
                result = ParityResult(
                    param_ratio=param_ratio,
                    flop_ratio=flop_ratio,
                    params_within_tolerance=params_within_tolerance,
                    flops_within_tolerance=flops_within_tolerance,
                    tolerance=tolerance
                )
                
                print(f"   📊 {case_name}: params={param_ratio:.3f}x, flops={flop_ratio:.3f}x")
                print(f"      Params OK: {params_within_tolerance}, FLOPs OK: {flops_within_tolerance}")
                
                self.assertEqual(params_within_tolerance, expect_param_ok)
                self.assertEqual(flops_within_tolerance, expect_flop_ok)
    
    def test_experiment_metrics_dataclass(self):
        """Test ExperimentMetrics dataclass functionality."""
        print("\n🔬 Testing ExperimentMetrics Data Structure")
        
        # Create test metrics
        metrics = ExperimentMetrics(
            accuracy=0.85,
            loss=0.32,
            perplexity=1.38,
            latency_p50=45.2,
            latency_p95=67.8,
            memory_peak=2048,
            throughput=125.5,
            param_count=1500000,
            flop_count=7500000,
            kv_cache_hits=8500,
            kv_cache_total=10000,
            routing_entropy=1.85,
            expert_utilization={0: 0.45, 1: 0.55},
            metadata={"model_variant": "PT1", "batch_size": 32}
        )
        
        print(f"   📊 Accuracy: {metrics.accuracy}")
        print(f"   📊 KV Cache Hit Rate: {metrics.kv_cache_hits/metrics.kv_cache_total:.3f}")
        print(f"   📊 Expert Utilization: {metrics.expert_utilization}")
        
        # Validate data structure
        self.assertIsInstance(metrics.accuracy, float)
        self.assertIsInstance(metrics.expert_utilization, dict)
        self.assertIsInstance(metrics.metadata, dict)
        self.assertEqual(metrics.metadata["model_variant"], "PT1")
        
        # Test cache hit rate calculation
        cache_hit_rate = metrics.kv_cache_hits / metrics.kv_cache_total
        self.assertGreaterEqual(cache_hit_rate, 0.0)
        self.assertLessEqual(cache_hit_rate, 1.0)
    
    def test_statistical_significance_testing(self):
        """Test statistical significance testing framework."""
        print("\n🔬 Testing Statistical Significance Framework")
        
        # Generate two groups: baseline and BEM-enhanced
        np.random.seed(42)
        baseline_scores = np.random.normal(0.80, 0.05, size=50)  # 80% accuracy
        bem_scores = np.random.normal(0.83, 0.05, size=50)       # 83% accuracy (improvement)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(baseline_scores, bem_scores)
        
        print(f"   📊 Baseline mean: {np.mean(baseline_scores):.4f}")
        print(f"   📊 BEM mean: {np.mean(bem_scores):.4f}")
        print(f"   📊 t-statistic: {t_stat:.4f}")
        print(f"   📊 p-value: {p_value:.6f}")
        print(f"   📊 Significant at α=0.05: {p_value < 0.05}")
        
        # Validate test results
        self.assertIsInstance(t_stat, (float, np.floating))
        self.assertIsInstance(p_value, (float, np.floating))
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        
        # For this specific test data, we expect statistical significance
        # (but this could vary due to randomness, so we don't assert it)

def run_validation_tests():
    """Run the framework validation tests."""
    print("🚀 Starting BEM v1.3 Testing Framework Validation")
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFrameworkValidation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All framework validation tests passed!")
        print(f"📊 Tests run: {result.testsRun}")
        return True
    else:
        print("❌ Some framework validation tests failed!")
        print(f"📊 Tests run: {result.testsRun}")
        print(f"📊 Failures: {len(result.failures)}")
        print(f"📊 Errors: {len(result.errors)}")
        return False

if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)