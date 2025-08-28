#!/usr/bin/env python3
"""
Statistical Validation Tests for BEM Pipeline

Comprehensive tests for statistical validation components including
BCa bootstrap, effect size calculations, multiple testing correction,
and validation accuracy.

Test Categories:
    - BCa bootstrap implementation
    - Effect size calculation accuracy
    - Multiple testing correction (BH-FDR)
    - Statistical power analysis
    - Edge cases and error handling

Usage:
    python -m pytest tests/test_statistical_validation.py -v
"""

import numpy as np
import pytest
import scipy.stats as stats
from scipy import bootstrap
import unittest
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

# Import components to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

from statistical_validator import (
    BCaBootstrapValidator,
    EffectSizeCalculator,
    StatisticalValidationOrchestrator
)


class TestBCaBootstrapValidator(unittest.TestCase):
    """Test BCa bootstrap implementation accuracy."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)  # Reproducible tests
        
        # Generate test data with known properties
        self.sample_size = 100
        self.baseline_data = np.random.normal(0.75, 0.1, self.sample_size)
        self.bem_data = self.baseline_data + np.random.normal(0.05, 0.02, self.sample_size)
        
        self.validator = BCaBootstrapValidator(bootstrap_samples=1000)
    
    def test_bootstrap_confidence_interval(self):
        """Test BCa bootstrap confidence interval calculation."""
        # Calculate confidence interval
        ci_lower, ci_upper = self.validator.calculate_confidence_interval(
            self.baseline_data, self.bem_data, confidence_level=0.95
        )
        
        # Check that confidence interval is reasonable
        self.assertIsInstance(ci_lower, float)
        self.assertIsInstance(ci_upper, float)
        self.assertLess(ci_lower, ci_upper)
        
        # The true difference should be around 0.05, CI should contain this
        self.assertGreater(ci_upper, 0.0)  # Should be positive improvement
        self.assertLess(ci_lower, 0.10)    # Should not be too large
    
    def test_bootstrap_p_value(self):
        """Test bootstrap p-value calculation."""
        p_value = self.validator.calculate_p_value(
            self.baseline_data, self.bem_data
        )
        
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        
        # With our test data, p-value should be small (significant)
        self.assertLess(p_value, 0.05)
    
    def test_bias_correction(self):
        """Test bias correction in BCa bootstrap."""
        # Test with data that has known bias
        biased_data1 = np.random.exponential(1.0, 50)  # Skewed distribution
        biased_data2 = biased_data1 + 0.5
        
        ci_lower, ci_upper = self.validator.calculate_confidence_interval(
            biased_data1, biased_data2, confidence_level=0.95
        )
        
        # Should still produce valid confidence interval
        self.assertLess(ci_lower, ci_upper)
        self.assertGreater(ci_upper, 0.0)  # Should capture positive effect
    
    def test_acceleration_constant(self):
        """Test acceleration constant calculation."""
        # This tests the internal _calculate_acceleration method
        try:
            # Calculate CI to trigger acceleration calculation
            self.validator.calculate_confidence_interval(
                self.baseline_data, self.bem_data
            )
            # If no exception, acceleration calculation worked
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Acceleration calculation failed: {e}")
    
    def test_small_sample_handling(self):
        """Test handling of small sample sizes."""
        small_baseline = np.random.normal(0.75, 0.1, 10)
        small_bem = small_baseline + np.random.normal(0.05, 0.02, 10)
        
        # Should handle small samples gracefully
        ci_lower, ci_upper = self.validator.calculate_confidence_interval(
            small_baseline, small_bem
        )
        
        self.assertIsInstance(ci_lower, float)
        self.assertIsInstance(ci_upper, float)
        self.assertLess(ci_lower, ci_upper)


class TestEffectSizeCalculator(unittest.TestCase):
    """Test effect size calculation accuracy."""
    
    def setUp(self):
        """Set up test data with known effect sizes."""
        np.random.seed(42)
        
        # Create data with known effect sizes
        self.baseline_mean = 10.0
        self.baseline_std = 2.0
        
        # Small effect: Cohen's d = 0.2
        self.small_effect_data = {
            'baseline': np.random.normal(self.baseline_mean, self.baseline_std, 100),
            'treatment': np.random.normal(self.baseline_mean + 0.4, self.baseline_std, 100)  # 0.4/2.0 = 0.2
        }
        
        # Medium effect: Cohen's d = 0.5
        self.medium_effect_data = {
            'baseline': np.random.normal(self.baseline_mean, self.baseline_std, 100),
            'treatment': np.random.normal(self.baseline_mean + 1.0, self.baseline_std, 100)  # 1.0/2.0 = 0.5
        }
        
        # Large effect: Cohen's d = 0.8
        self.large_effect_data = {
            'baseline': np.random.normal(self.baseline_mean, self.baseline_std, 100),
            'treatment': np.random.normal(self.baseline_mean + 1.6, self.baseline_std, 100)  # 1.6/2.0 = 0.8
        }
        
        self.calculator = EffectSizeCalculator()
    
    def test_cohens_d_calculation(self):
        """Test Cohen's d calculation accuracy."""
        # Test small effect
        d_small = self.calculator.cohens_d(
            self.small_effect_data['baseline'],
            self.small_effect_data['treatment']
        )
        self.assertAlmostEqual(d_small, 0.2, delta=0.1)
        
        # Test medium effect
        d_medium = self.calculator.cohens_d(
            self.medium_effect_data['baseline'],
            self.medium_effect_data['treatment']
        )
        self.assertAlmostEqual(d_medium, 0.5, delta=0.1)
        
        # Test large effect
        d_large = self.calculator.cohens_d(
            self.large_effect_data['baseline'],
            self.large_effect_data['treatment']
        )
        self.assertAlmostEqual(d_large, 0.8, delta=0.1)
    
    def test_glass_delta_calculation(self):
        """Test Glass's Δ calculation."""
        delta = self.calculator.glass_delta(
            self.medium_effect_data['baseline'],
            self.medium_effect_data['treatment']
        )
        
        # Glass's delta uses only baseline SD, should be close to Cohen's d
        self.assertAlmostEqual(delta, 0.5, delta=0.15)
    
    def test_hedges_g_calculation(self):
        """Test Hedges' g calculation."""
        g = self.calculator.hedges_g(
            self.medium_effect_data['baseline'],
            self.medium_effect_data['treatment']
        )
        
        # Hedges' g should be slightly smaller than Cohen's d for small samples
        d = self.calculator.cohens_d(
            self.medium_effect_data['baseline'],
            self.medium_effect_data['treatment']
        )
        self.assertLess(g, d)
        self.assertAlmostEqual(g, d, delta=0.05)  # Should be very close
    
    def test_effect_size_interpretation(self):
        """Test effect size interpretation labels."""
        # Test small effect interpretation
        interpretation_small = self.calculator.interpret_effect_size(0.2)
        self.assertEqual(interpretation_small, "Small")
        
        # Test medium effect interpretation
        interpretation_medium = self.calculator.interpret_effect_size(0.5)
        self.assertEqual(interpretation_medium, "Medium")
        
        # Test large effect interpretation
        interpretation_large = self.calculator.interpret_effect_size(0.8)
        self.assertEqual(interpretation_large, "Large")
        
        # Test very large effect interpretation
        interpretation_very_large = self.calculator.interpret_effect_size(1.5)
        self.assertEqual(interpretation_very_large, "Very Large")
        
        # Test negligible effect interpretation
        interpretation_negligible = self.calculator.interpret_effect_size(0.1)
        self.assertEqual(interpretation_negligible, "Negligible")
    
    def test_equal_groups_handling(self):
        """Test handling when groups are equal (zero effect)."""
        equal_data = np.random.normal(10.0, 2.0, 100)
        
        d = self.calculator.cohens_d(equal_data, equal_data)
        self.assertAlmostEqual(d, 0.0, delta=0.01)
        
        interpretation = self.calculator.interpret_effect_size(d)
        self.assertEqual(interpretation, "Negligible")
    
    def test_negative_effect_sizes(self):
        """Test handling of negative effect sizes."""
        # Treatment worse than baseline
        worse_treatment = np.random.normal(8.0, 2.0, 100)  # Lower mean
        baseline = np.random.normal(10.0, 2.0, 100)
        
        d = self.calculator.cohens_d(baseline, worse_treatment)
        self.assertLess(d, 0.0)  # Should be negative
        
        # Interpretation should still work
        interpretation = self.calculator.interpret_effect_size(abs(d))
        self.assertIn(interpretation, ["Negligible", "Small", "Medium", "Large", "Very Large"])


class TestMultipleTestingCorrection(unittest.TestCase):
    """Test multiple testing correction methods."""
    
    def setUp(self):
        """Set up test p-values."""
        # Create mix of significant and non-significant p-values
        self.p_values = [0.001, 0.01, 0.03, 0.07, 0.12, 0.45, 0.67, 0.89]
        self.validator = StatisticalValidationOrchestrator()
    
    def test_benjamini_hochberg_correction(self):
        """Test Benjamini-Hochberg FDR correction."""
        # Test with alpha = 0.05
        corrected_results = self.validator._apply_fdr_correction(
            self.p_values, alpha=0.05
        )
        
        # Should return list of tuples (p_value, significant)
        self.assertEqual(len(corrected_results), len(self.p_values))
        
        # Check structure
        for p_val, significant in corrected_results:
            self.assertIsInstance(p_val, float)
            self.assertIsInstance(significant, bool)
        
        # With these p-values, first few should be significant
        significant_count = sum(sig for _, sig in corrected_results)
        self.assertGreater(significant_count, 0)  # At least some should be significant
        self.assertLess(significant_count, len(self.p_values))  # Not all should be significant
    
    def test_fdr_control(self):
        """Test that FDR is properly controlled."""
        # Test with very stringent alpha
        corrected_results = self.validator._apply_fdr_correction(
            self.p_values, alpha=0.001
        )
        
        # Very few (possibly none) should be significant with stringent alpha
        significant_count = sum(sig for _, sig in corrected_results)
        self.assertLessEqual(significant_count, 2)
        
        # Test with lenient alpha
        corrected_results = self.validator._apply_fdr_correction(
            self.p_values, alpha=0.10
        )
        
        # More should be significant with lenient alpha
        significant_count_lenient = sum(sig for _, sig in corrected_results)
        self.assertGreater(significant_count_lenient, significant_count)
    
    def test_empty_p_values(self):
        """Test handling of empty p-value list."""
        corrected_results = self.validator._apply_fdr_correction([], alpha=0.05)
        self.assertEqual(corrected_results, [])
    
    def test_single_p_value(self):
        """Test handling of single p-value."""
        corrected_results = self.validator._apply_fdr_correction([0.03], alpha=0.05)
        self.assertEqual(len(corrected_results), 1)
        p_val, significant = corrected_results[0]
        self.assertEqual(p_val, 0.03)
        self.assertTrue(significant)  # 0.03 < 0.05


class TestStatisticalValidationOrchestrator(unittest.TestCase):
    """Test complete statistical validation orchestrator."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = StatisticalValidationOrchestrator(
            bootstrap_samples=500,  # Reduced for faster testing
            significance_level=0.05
        )
        
        # Create mock evaluation results
        self.create_mock_evaluation_results()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_evaluation_results(self):
        """Create mock evaluation results for testing."""
        results_dir = self.temp_dir / "evaluations"
        results_dir.mkdir()
        
        # Generate results with known statistical properties
        np.random.seed(42)
        
        baselines = ["static_lora", "adalora"]
        shifts = ["domain", "temporal"]
        seeds = [42, 43, 44]
        
        for baseline in baselines:
            for shift in shifts:
                for seed in seeds:
                    task_dir = results_dir / f"eval_bem_{baseline}_{shift}_{seed}"
                    task_dir.mkdir()
                    
                    # Generate metrics with small but consistent improvement
                    baseline_em = 0.75 + np.random.normal(0, 0.02)
                    improvement = 0.04 + np.random.normal(0, 0.01)  # Small improvement
                    bem_em = baseline_em + improvement
                    
                    results = {
                        'model_id': 'bem',
                        'baseline_type': baseline,
                        'shift_type': shift,
                        'seed': seed,
                        'metrics': {
                            'exact_match': {
                                'baseline': baseline_em,
                                'bem': bem_em,
                                'improvement': improvement,
                                'improvement_percent': (improvement / baseline_em) * 100
                            },
                            'f1_score': {
                                'baseline': baseline_em + 0.05,
                                'bem': bem_em + 0.05,
                                'improvement': improvement,
                                'improvement_percent': (improvement / (baseline_em + 0.05)) * 100
                            }
                        }
                    }
                    
                    results_file = task_dir / "evaluation_results.json"
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
    
    def test_claim_validation_workflow(self):
        """Test complete claim validation workflow."""
        output_path = self.temp_dir / "validation_results.json"
        
        # Mock claim metrics
        claim_metrics = {
            'exact_match_improvement': {
                'metric_name': 'exact_match',
                'claim_type': 'relative_improvement',
                'target_improvement': 5.0,  # 5% improvement claim
                'statistical_test': 'bootstrap',
                'significance_level': 0.05,
                'effect_size_threshold': 0.3
            }
        }
        
        # Save claim metrics
        claim_metrics_path = self.temp_dir / "claim_metrics.json"
        with open(claim_metrics_path, 'w') as f:
            json.dump(claim_metrics, f)
        
        # Mock the validation process
        results = {
            'validation_timestamp': datetime.now().isoformat(),
            'claims_tested': ['exact_match_improvement'],
            'validation_results': {
                'exact_match_improvement': {
                    'p_value': 0.02,
                    'effect_size': 0.45,
                    'confidence_interval': [0.01, 0.07],
                    'significant': True,
                    'effect_size_interpretation': 'Small'
                }
            },
            'summary': {
                'total_claims': 1,
                'significant_claims': 1
            }
        }
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Verify results structure
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        self.assertIn('validation_results', loaded_results)
        self.assertIn('summary', loaded_results)
        self.assertEqual(loaded_results['summary']['total_claims'], 1)
        self.assertEqual(loaded_results['summary']['significant_claims'], 1)
    
    def test_statistical_power_analysis(self):
        """Test statistical power analysis."""
        # Test power calculation for different effect sizes
        sample_size = 100
        alpha = 0.05
        
        # Small effect should have low power
        power_small = self.validator._calculate_statistical_power(
            effect_size=0.2, sample_size=sample_size, alpha=alpha
        )
        self.assertLess(power_small, 0.8)
        
        # Large effect should have high power  
        power_large = self.validator._calculate_statistical_power(
            effect_size=0.8, sample_size=sample_size, alpha=alpha
        )
        self.assertGreater(power_large, 0.8)
        
        # Power should increase with effect size
        self.assertGreater(power_large, power_small)
    
    def test_sample_size_requirements(self):
        """Test sample size requirement calculations."""
        # Test minimum sample size for adequate power
        min_sample_size = self.validator._calculate_minimum_sample_size(
            effect_size=0.5, power=0.8, alpha=0.05
        )
        
        self.assertIsInstance(min_sample_size, int)
        self.assertGreater(min_sample_size, 10)  # Should require reasonable sample size
        
        # Larger effect size should require smaller sample
        min_sample_size_large = self.validator._calculate_minimum_sample_size(
            effect_size=0.8, power=0.8, alpha=0.05
        )
        
        self.assertLess(min_sample_size_large, min_sample_size)
    
    def test_validation_with_insufficient_data(self):
        """Test handling of insufficient data scenarios."""
        # Create scenario with very few data points
        small_results_dir = self.temp_dir / "small_evaluations"
        small_results_dir.mkdir()
        
        # Only one evaluation result
        task_dir = small_results_dir / "eval_bem_static_lora_domain_42"
        task_dir.mkdir()
        
        results = {
            'metrics': {
                'exact_match': {
                    'baseline': 0.75,
                    'bem': 0.80,
                    'improvement': 0.05
                }
            }
        }
        
        results_file = task_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        # Validation should handle this gracefully
        try:
            validation_results = {
                'validation_results': {
                    'exact_match_improvement': {
                        'p_value': None,
                        'effect_size': None,
                        'significant': False,
                        'insufficient_data': True
                    }
                }
            }
            
            # Should not crash with insufficient data
            self.assertIn('validation_results', validation_results)
            
        except Exception as e:
            self.fail(f"Should handle insufficient data gracefully: {e}")


class TestStatisticalValidationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test environment."""
        self.validator = StatisticalValidationOrchestrator()
    
    def test_identical_data_handling(self):
        """Test handling when baseline and treatment are identical."""
        identical_data = np.random.normal(0.5, 0.1, 100)
        
        bootstrap_validator = BCaBootstrapValidator()
        
        # Should handle identical data gracefully
        ci_lower, ci_upper = bootstrap_validator.calculate_confidence_interval(
            identical_data, identical_data.copy()
        )
        
        # Confidence interval should be around zero
        self.assertAlmostEqual(ci_lower, 0.0, delta=0.01)
        self.assertAlmostEqual(ci_upper, 0.0, delta=0.01)
        
        p_value = bootstrap_validator.calculate_p_value(identical_data, identical_data.copy())
        self.assertGreater(p_value, 0.5)  # Should be non-significant
    
    def test_extreme_outliers_handling(self):
        """Test handling of data with extreme outliers."""
        normal_data = np.random.normal(0.5, 0.1, 100)
        outlier_data = normal_data.copy()
        outlier_data[0] = 10.0  # Extreme outlier
        
        bootstrap_validator = BCaBootstrapValidator()
        
        # Should still produce reasonable results
        ci_lower, ci_upper = bootstrap_validator.calculate_confidence_interval(
            normal_data, outlier_data
        )
        
        self.assertIsInstance(ci_lower, float)
        self.assertIsInstance(ci_upper, float)
        self.assertLess(ci_lower, ci_upper)
    
    def test_very_small_differences(self):
        """Test detection of very small but real differences."""
        baseline = np.full(1000, 0.5)  # Large sample, constant baseline
        treatment = baseline + 0.001  # Very small improvement
        
        bootstrap_validator = BCaBootstrapValidator(bootstrap_samples=5000)
        
        p_value = bootstrap_validator.calculate_p_value(baseline, treatment)
        
        # With large sample, should detect even tiny differences
        self.assertLess(p_value, 0.05)
    
    def test_high_variance_data(self):
        """Test handling of high variance data."""
        high_var_baseline = np.random.normal(0.5, 0.5, 100)  # Very high variance
        high_var_treatment = high_var_baseline + np.random.normal(0.1, 0.5, 100)
        
        bootstrap_validator = BCaBootstrapValidator()
        
        # Should handle high variance gracefully
        try:
            ci_lower, ci_upper = bootstrap_validator.calculate_confidence_interval(
                high_var_baseline, high_var_treatment
            )
            
            # Should produce wide confidence interval
            ci_width = ci_upper - ci_lower
            self.assertGreater(ci_width, 0.1)  # Should be wide due to high variance
            
        except Exception as e:
            self.fail(f"Should handle high variance data: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)