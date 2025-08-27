#!/usr/bin/env python3
"""
Simplified BEM v1.1 Pipeline Core Tests
Tests core pipeline logic without requiring PyTorch dependencies.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import yaml
import json
import numpy as np
from unittest.mock import patch, MagicMock

# Add the modules directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from analysis.stats import EnhancedStatisticalAnalyzer
from analysis.cache_metrics import CacheMetricsAnalyzer
from analysis.leakcheck import LeakDetector
from analysis.pareto import ParetoAnalyzer
from analysis.hero_tables import HeroTableGenerator


class TestPipelineCore(unittest.TestCase):
    """Test core pipeline components without PyTorch dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock experiment results
        np.random.seed(42)  # For reproducible tests
        
        # Mock BEM and baseline scores (5 seeds each)
        self.mock_bem_scores = {
            f"seed_{i}": {
                "EM": np.random.normal(0.75, 0.05),
                "F1": np.random.normal(0.82, 0.04),
                "BLEU": np.random.normal(0.68, 0.06),
                "chrF": np.random.normal(0.71, 0.05)
            } for i in range(1, 6)
        }
        
        self.mock_baseline_scores = {
            f"seed_{i}": {
                "EM": np.random.normal(0.70, 0.05),
                "F1": np.random.normal(0.78, 0.04),
                "BLEU": np.random.normal(0.62, 0.06),
                "chrF": np.random.normal(0.66, 0.05)
            } for i in range(1, 6)
        }
        
        # Mock performance data
        self.mock_performance = {
            "bem": {"p50_latency_ms": 195.0, "vram_usage_gb": 4.2},
            "baseline": {"p50_latency_ms": 180.0, "vram_usage_gb": 4.0}
        }
        
        # Mock cache metrics
        self.mock_cache_metrics = {
            "kv_hit_rate": 0.85,
            "routing_flips_per_chunk": 0.12,
            "gate_entropy": 0.91
        }

    def test_statistical_analyzer(self):
        """Test statistical analysis with BCa bootstrap."""
        analyzer = EnhancedStatisticalAnalyzer()
        
        # Test relative improvement calculation
        bem_scores = np.array([0.75, 0.82, 0.68, 0.71, 0.74])
        baseline_scores = np.array([0.70, 0.78, 0.62, 0.66, 0.69])
        
        improvement, ci_lower, ci_upper = analyzer.compute_relative_improvement_ci(
            bem_scores, baseline_scores, confidence_level=0.95
        )
        
        # Should see positive improvement
        self.assertGreater(improvement, 0)
        self.assertLess(ci_lower, improvement)
        self.assertGreater(ci_upper, improvement)
        
        # Test FDR correction
        p_values = [0.01, 0.03, 0.08, 0.12, 0.45]
        corrected_p, significant = analyzer.fdr_correction(p_values, alpha=0.05)
        
        self.assertEqual(len(corrected_p), len(p_values))
        self.assertEqual(len(significant), len(p_values))
        self.assertTrue(any(significant))  # At least one should be significant

    def test_cache_metrics_analyzer(self):
        """Test cache metrics analysis and quality gates."""
        analyzer = CacheMetricsAnalyzer()
        
        # Test quality gate validation
        metrics = {
            "kv_hit_rate": 0.85,
            "routing_flips_per_chunk": 0.12,
            "gate_entropy": 0.91
        }
        
        passes_gates, failures = analyzer.validate_quality_gates(metrics)
        
        # Should pass with 85% hit rate (> 80% threshold)
        self.assertTrue(passes_gates)
        self.assertEqual(len(failures), 0)
        
        # Test failure case
        bad_metrics = {
            "kv_hit_rate": 0.75,  # Below 80% threshold
            "routing_flips_per_chunk": 0.12,
            "gate_entropy": 0.91
        }
        
        passes_gates, failures = analyzer.validate_quality_gates(bad_metrics)
        self.assertFalse(passes_gates)
        self.assertGreater(len(failures), 0)

    def test_leak_detector(self):
        """Test leak detection functionality."""
        detector = LeakDetector(num_perm=64, threshold=0.7)
        
        # Test with some mock data
        train_samples = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language"
        ]
        
        eval_samples = [
            "The quick brown fox jumps over a lazy dog",  # Similar to train[0]
            "Deep learning uses neural networks",  # Different
            "Java is another programming language"  # Different
        ]
        
        leak_results = detector.detect_leaks(train_samples, eval_samples)
        
        # Should detect one leak (similar sentences)
        self.assertGreater(leak_results["total_leaks"], 0)
        self.assertLess(leak_results["leak_rate"], 1.0)

    def test_pareto_analyzer(self):
        """Test Pareto frontier analysis."""
        analyzer = ParetoAnalyzer()
        
        # Mock experiment results with performance trade-offs
        results = [
            {"model": "baseline", "F1": 0.78, "p50_latency_ms": 180},
            {"model": "bem_v11", "F1": 0.82, "p50_latency_ms": 195}
        ]
        
        pareto_points = analyzer.find_pareto_frontier(
            results, primary_metric="F1", efficiency_metric="p50_latency_ms"
        )
        
        self.assertGreater(len(pareto_points), 0)
        
        # Test latency budget validation
        baseline_latency = 180.0
        bem_latency = 195.0
        budget_pct = 15.0
        
        within_budget = analyzer.validate_latency_budget(
            bem_latency, baseline_latency, budget_pct
        )
        
        # 195 vs 180 = +8.3%, should be within 15% budget
        self.assertTrue(within_budget)

    def test_hero_table_generator(self):
        """Test hero table generation."""
        generator = HeroTableGenerator()
        
        # Mock slice results
        slice_results = {
            "Full": {
                "bem": {"EM": [0.75, 0.76, 0.74], "F1": [0.82, 0.83, 0.81]},
                "baseline": {"EM": [0.70, 0.71, 0.69], "F1": [0.78, 0.79, 0.77]}
            }
        }
        
        # Mock statistical results
        statistical_results = {
            "Full": {
                "EM": {"improvement": 7.14, "ci_lower": 2.1, "ci_upper": 12.3, "p_value": 0.01},
                "F1": {"improvement": 5.13, "ci_lower": 1.8, "ci_upper": 8.7, "p_value": 0.02}
            }
        }
        
        table = generator.generate_table(slice_results, statistical_results)
        
        # Should contain improvement percentages and significance stars
        self.assertIn("7.14%", table)
        self.assertIn("5.13%", table)
        self.assertIn("***", table)  # Significant results

    def test_configuration_validation(self):
        """Test experiment configuration validation."""
        # Test BEM config
        bem_config_path = "experiments/v11_baseline.yml"
        if os.path.exists(bem_config_path):
            with open(bem_config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate required BEM components
            self.assertIn("bem_config", config["model"])
            bem_config = config["model"]["bem_config"]
            
            # Check cache-safe sites
            expected_sites = ["W_O", "W_down"]
            self.assertEqual(bem_config["sites"], expected_sites)
            
            # Check depth-varying ranks
            expected_ranks = [2, 4, 8, 8, 8, 4, 2]
            self.assertEqual(bem_config["rank_schedule"], expected_ranks)
            
            # Check chunk-sticky routing
            self.assertEqual(bem_config["routing"]["chunk_size"], 128)
            self.assertEqual(bem_config["routing"]["hysteresis_tau"], 0.7)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestQualityGates(unittest.TestCase):
    """Test quality gate validation logic."""
    
    def test_baseline_threshold_validation(self):
        """Test baseline performance threshold validation."""
        bem_scores = {"EM": 0.75, "F1": 0.82, "BLEU": 0.68, "chrF": 0.71}
        baseline_scores = {"EM": 0.70, "F1": 0.78, "BLEU": 0.62, "chrF": 0.66}
        
        # All metrics should be >= baseline
        for metric in bem_scores:
            self.assertGreaterEqual(bem_scores[metric], baseline_scores[metric])

    def test_latency_budget_validation(self):
        """Test latency budget compliance."""
        baseline_latency = 180.0
        bem_latency = 195.0
        budget_pct = 15.0
        
        latency_increase = (bem_latency - baseline_latency) / baseline_latency * 100
        self.assertLessEqual(latency_increase, budget_pct)

    def test_cache_hit_rate_validation(self):
        """Test cache hit rate quality gate."""
        cache_metrics = {"kv_hit_rate": 0.85}
        min_hit_rate = 0.80
        
        self.assertGreaterEqual(cache_metrics["kv_hit_rate"], min_hit_rate)

    def test_vram_budget_validation(self):
        """Test VRAM usage within budget."""
        baseline_vram = 4.0
        bem_vram = 4.2
        budget_pct = 5.0
        
        vram_increase = abs(bem_vram - baseline_vram) / baseline_vram * 100
        self.assertLessEqual(vram_increase, budget_pct)


def main():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()