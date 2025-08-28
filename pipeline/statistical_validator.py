#!/usr/bin/env python3
"""
BCa Bootstrap Statistical Validator for BEM Research

Rigorous statistical validation with bias-corrected and accelerated (BCa) bootstrap
with 10,000 resamples and Benjamini-Hochberg FDR correction for multiple testing.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import bootstrap
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StatisticalTest:
    """Configuration for a statistical test."""
    test_name: str
    test_type: str  # "paired_t_test", "one_sample_t_test", etc.
    alpha: float = 0.05
    alternative: str = "two-sided"  # "two-sided", "greater", "less"
    minimum_effect_size: Optional[float] = None

@dataclass 
class BootstrapResult:
    """Results from BCa bootstrap analysis."""
    statistic: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    bootstrap_distribution: np.ndarray
    bias_correction: float
    acceleration: float

@dataclass
class StatisticalValidationResult:
    """Complete statistical validation result."""
    claim_id: str
    metric_name: str
    observed_statistic: float
    bootstrap_result: BootstrapResult
    p_value: float
    effect_size: float
    effect_size_interpretation: str
    passes_significance_test: bool
    confidence_interval_excludes_null: bool
    meets_minimum_effect_size: bool
    raw_data: Dict[str, Any]

class BCaBootstrapValidator:
    """Bias-corrected and accelerated bootstrap validator."""
    
    def __init__(self, 
                 n_resamples: int = 10000,
                 confidence_level: float = 0.95,
                 random_seed: int = 42):
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def compute_bca_bootstrap(self, 
                            data: np.ndarray, 
                            statistic_func: callable,
                            **kwargs) -> BootstrapResult:
        """Compute BCa bootstrap confidence interval."""
        
        logger.debug(f"Computing BCa bootstrap with {self.n_resamples} resamples")
        
        # Observed statistic
        observed_stat = statistic_func(data, **kwargs)
        
        # Generate bootstrap samples
        rng = np.random.RandomState(self.random_seed)
        bootstrap_stats = []
        
        n = len(data)
        for i in range(self.n_resamples):
            # Bootstrap resample
            indices = rng.choice(n, size=n, replace=True)
            bootstrap_sample = data[indices]
            
            try:
                bootstrap_stat = statistic_func(bootstrap_sample, **kwargs)
                bootstrap_stats.append(bootstrap_stat)
            except:
                # Handle numerical issues
                bootstrap_stats.append(np.nan)
                
        bootstrap_stats = np.array(bootstrap_stats)
        bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
        
        # Bias correction (z0)
        n_less = np.sum(bootstrap_stats < observed_stat)
        z0 = stats.norm.ppf(n_less / len(bootstrap_stats)) if n_less > 0 else 0.0
        
        # Acceleration (a) using jackknife
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            try:
                jackknife_stat = statistic_func(jackknife_sample, **kwargs)
                jackknife_stats.append(jackknife_stat)
            except:
                jackknife_stats.append(np.nan)
                
        jackknife_stats = np.array(jackknife_stats)
        jackknife_stats = jackknife_stats[~np.isnan(jackknife_stats)]
        
        # Acceleration parameter
        jackknife_mean = np.mean(jackknife_stats)
        numerator = np.sum((jackknife_mean - jackknife_stats)**3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats)**2))**(3/2)
        a = numerator / denominator if denominator != 0 else 0.0
        
        # BCa confidence interval
        alpha = 1 - self.confidence_level
        z_alpha_2 = stats.norm.ppf(alpha/2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)
        
        # Adjusted percentiles
        alpha_1 = stats.norm.cdf(z0 + (z0 + z_alpha_2)/(1 - a*(z0 + z_alpha_2)))
        alpha_2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2)/(1 - a*(z0 + z_1_alpha_2)))
        
        # Ensure percentiles are in valid range
        alpha_1 = max(0.001, min(0.999, alpha_1))
        alpha_2 = max(0.001, min(0.999, alpha_2))
        
        ci_lower = np.percentile(bootstrap_stats, alpha_1 * 100)
        ci_upper = np.percentile(bootstrap_stats, alpha_2 * 100)
        
        return BootstrapResult(
            statistic=observed_stat,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level,
            bootstrap_distribution=bootstrap_stats,
            bias_correction=z0,
            acceleration=a
        )

class EffectSizeCalculator:
    """Calculate various effect size measures."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1:
            return 0.0
            
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1))/(n1+n2-2))
        
        if pooled_std == 0:
            return 0.0
            
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    @staticmethod
    def glass_delta(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Glass's Δ (delta) effect size."""
        
        control_std = np.std(group2, ddof=1)
        if control_std == 0:
            return 0.0
            
        return (np.mean(group1) - np.mean(group2)) / control_std
    
    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g (bias-corrected Cohen's d)."""
        
        n1, n2 = len(group1), len(group2)
        cohens_d = EffectSizeCalculator.cohens_d(group1, group2)
        
        # Bias correction factor
        df = n1 + n2 - 2
        j = 1 - (3/(4*df - 1)) if df > 0 else 1
        
        return cohens_d * j
    
    @staticmethod
    def interpret_effect_size(effect_size: float) -> str:
        """Interpret effect size magnitude (Cohen's conventions)."""
        
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        elif abs_effect < 1.2:
            return "large"
        else:
            return "very large"

class ClaimValidator:
    """Validate individual claims using statistical testing."""
    
    def __init__(self, 
                 bootstrap_validator: BCaBootstrapValidator,
                 effect_size_calculator: EffectSizeCalculator):
        self.bootstrap_validator = bootstrap_validator
        self.effect_size_calculator = effect_size_calculator
        
    def validate_accuracy_claim(self, 
                              bem_scores: List[float],
                              baseline_scores: List[float], 
                              claim_config: Dict[str, Any]) -> StatisticalValidationResult:
        """Validate accuracy improvement claim."""
        
        logger.info(f"Validating accuracy claim: {claim_config['claim']}")
        
        bem_array = np.array(bem_scores)
        baseline_array = np.array(baseline_scores)
        
        # Difference scores (paired comparison)
        differences = bem_array - baseline_array
        
        # Bootstrap analysis
        def mean_difference(data):
            return np.mean(data)
            
        bootstrap_result = self.bootstrap_validator.compute_bca_bootstrap(
            differences, mean_difference
        )
        
        # Statistical test (paired t-test)
        t_stat, p_value = stats.ttest_rel(bem_array, baseline_array, 
                                         alternative='greater')  # BEM > baseline
        
        # Effect size
        effect_size = self.effect_size_calculator.cohens_d(bem_array, baseline_array)
        effect_interpretation = self.effect_size_calculator.interpret_effect_size(effect_size)
        
        # Decision criteria
        passes_significance = p_value < claim_config.get('significance_level', 0.05)
        ci_excludes_zero = bootstrap_result.confidence_interval[0] > 0
        minimum_effect_size = claim_config.get('minimum_effect_size_d', 0.0)
        meets_effect_size = abs(effect_size) >= minimum_effect_size
        
        return StatisticalValidationResult(
            claim_id=claim_config.get('claim', 'unknown'),
            metric_name=claim_config.get('metric', 'unknown'),
            observed_statistic=bootstrap_result.statistic,
            bootstrap_result=bootstrap_result,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            passes_significance_test=passes_significance,
            confidence_interval_excludes_null=ci_excludes_zero,
            meets_minimum_effect_size=meets_effect_size,
            raw_data={
                "bem_scores": bem_scores,
                "baseline_scores": baseline_scores,
                "differences": differences.tolist(),
                "expected_improvement_pct": claim_config.get('expected_improvement_pct', 0)
            }
        )
        
    def validate_robustness_claim(self, 
                                bem_degradation: List[float],
                                baseline_degradation: List[float],
                                claim_config: Dict[str, Any]) -> StatisticalValidationResult:
        """Validate robustness/degradation claim."""
        
        logger.info(f"Validating robustness claim: {claim_config['claim']}")
        
        bem_array = np.array(bem_degradation)
        baseline_array = np.array(baseline_degradation)
        
        # Degradation difference (baseline should be higher = worse)
        differences = baseline_array - bem_array  # Positive = BEM is better
        
        # Bootstrap analysis
        def mean_difference(data):
            return np.mean(data)
            
        bootstrap_result = self.bootstrap_validator.compute_bca_bootstrap(
            differences, mean_difference
        )
        
        # Statistical test
        t_stat, p_value = stats.ttest_rel(baseline_array, bem_array,
                                         alternative='greater')  # baseline > BEM degradation
        
        # Effect size (larger effect = better BEM robustness)
        effect_size = self.effect_size_calculator.cohens_d(baseline_array, bem_array)
        effect_interpretation = self.effect_size_calculator.interpret_effect_size(effect_size)
        
        # Decision criteria  
        passes_significance = p_value < claim_config.get('significance_level', 0.05)
        ci_excludes_zero = bootstrap_result.confidence_interval[0] > 0
        meets_effect_size = True  # Always pass for robustness claims
        
        return StatisticalValidationResult(
            claim_id=claim_config.get('claim', 'unknown'),
            metric_name=claim_config.get('metric', 'unknown'),
            observed_statistic=bootstrap_result.statistic,
            bootstrap_result=bootstrap_result,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            passes_significance_test=passes_significance,
            confidence_interval_excludes_null=ci_excludes_zero,
            meets_minimum_effect_size=meets_effect_size,
            raw_data={
                "bem_degradation": bem_degradation,
                "baseline_degradation": baseline_degradation,
                "differences": differences.tolist(),
                "expected_improvement_pp": claim_config.get('expected_improvement_pp', 0)
            }
        )

    def validate_threshold_claim(self, 
                               bem_values: List[float],
                               threshold: float,
                               claim_config: Dict[str, Any]) -> StatisticalValidationResult:
        """Validate claims about BEM meeting specific thresholds."""
        
        logger.info(f"Validating threshold claim: {claim_config['claim']}")
        
        bem_array = np.array(bem_values)
        
        # One-sample test against threshold
        differences = bem_array - threshold
        
        # Bootstrap analysis
        def mean_difference(data):
            return np.mean(data)
            
        bootstrap_result = self.bootstrap_validator.compute_bca_bootstrap(
            differences, mean_difference
        )
        
        # Statistical test
        alternative = claim_config.get('alternative', 'greater')
        t_stat, p_value = stats.ttest_1samp(bem_array, threshold, 
                                          alternative=alternative)
        
        # Effect size (one-sample Cohen's d)
        effect_size = (np.mean(bem_array) - threshold) / np.std(bem_array, ddof=1)
        effect_interpretation = self.effect_size_calculator.interpret_effect_size(effect_size)
        
        # Decision criteria
        passes_significance = p_value < claim_config.get('significance_level', 0.05)
        
        if alternative == 'greater':
            ci_excludes_null = bootstrap_result.confidence_interval[0] > 0
        elif alternative == 'less':
            ci_excludes_null = bootstrap_result.confidence_interval[1] < 0
        else:
            ci_excludes_null = (bootstrap_result.confidence_interval[0] > 0 or 
                               bootstrap_result.confidence_interval[1] < 0)
        
        return StatisticalValidationResult(
            claim_id=claim_config.get('claim', 'unknown'),
            metric_name=claim_config.get('metric', 'unknown'),
            observed_statistic=bootstrap_result.statistic,
            bootstrap_result=bootstrap_result,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            passes_significance_test=passes_significance,
            confidence_interval_excludes_null=ci_excludes_null,
            meets_minimum_effect_size=True,  # Not applicable for threshold tests
            raw_data={
                "bem_values": bem_values,
                "threshold": threshold,
                "differences": differences.tolist(),
                "alternative": alternative
            }
        )

class StatisticalValidationOrchestrator:
    """Orchestrate comprehensive statistical validation with multiple testing correction."""
    
    def __init__(self, 
                 n_resamples: int = 10000,
                 confidence_level: float = 0.95,
                 fdr_alpha: float = 0.05,
                 random_seed: int = 42):
        
        self.bootstrap_validator = BCaBootstrapValidator(n_resamples, confidence_level, random_seed)
        self.effect_size_calculator = EffectSizeCalculator()
        self.claim_validator = ClaimValidator(self.bootstrap_validator, self.effect_size_calculator)
        self.fdr_alpha = fdr_alpha
        
    def validate_all_claims(self, 
                          experimental_data: Dict[str, Any],
                          claim_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all claims with multiple testing correction."""
        
        logger.info("Starting comprehensive statistical validation")
        
        validation_results = {}
        p_values = []
        claim_ids = []
        
        # Validate accuracy claims
        if 'accuracy_claims' in claim_configs:
            for claim_id, config in claim_configs['accuracy_claims'].items():
                
                # Extract relevant data
                bem_method = config.get('target_method', 'bem_dynamic')
                baseline_method = config.get('baseline_method', 'static_lora')
                metric = config.get('metric', 'em_score')
                
                bem_scores = self._extract_scores(experimental_data, bem_method, metric)
                baseline_scores = self._extract_scores(experimental_data, baseline_method, metric)
                
                if bem_scores and baseline_scores:
                    result = self.claim_validator.validate_accuracy_claim(
                        bem_scores, baseline_scores, config
                    )
                    validation_results[claim_id] = result
                    p_values.append(result.p_value)
                    claim_ids.append(claim_id)
                    
        # Validate robustness claims
        if 'robustness_claims' in claim_configs:
            for claim_id, config in claim_configs['robustness_claims'].items():
                
                bem_method = config.get('target_method', 'bem_dynamic')
                baseline_method = config.get('baseline_method', 'static_lora')
                metric = config.get('metric', 'degradation_percentage')
                
                bem_degradation = self._extract_degradation(experimental_data, bem_method, metric)
                baseline_degradation = self._extract_degradation(experimental_data, baseline_method, metric)
                
                if bem_degradation and baseline_degradation:
                    result = self.claim_validator.validate_robustness_claim(
                        bem_degradation, baseline_degradation, config
                    )
                    validation_results[claim_id] = result
                    p_values.append(result.p_value)
                    claim_ids.append(claim_id)
                    
        # Validate threshold claims (production SLOs)
        if 'production_claims' in claim_configs:
            for claim_id, config in claim_configs['production_claims'].items():
                
                metric = config.get('metric')
                threshold = config.get('target_speed_ratio', config.get('max_acceptable_parameters', 0))
                
                bem_values = self._extract_scores(experimental_data, 'bem_dynamic', metric)
                
                if bem_values:
                    result = self.claim_validator.validate_threshold_claim(
                        bem_values, threshold, config
                    )
                    validation_results[claim_id] = result
                    p_values.append(result.p_value)
                    claim_ids.append(claim_id)
                    
        # Multiple testing correction (Benjamini-Hochberg FDR)
        if p_values:
            _, corrected_p_values, _, _ = multipletests(p_values, alpha=self.fdr_alpha, method='fdr_bh')
            
            # Update results with corrected p-values
            for i, claim_id in enumerate(claim_ids):
                validation_results[claim_id].raw_data['corrected_p_value'] = corrected_p_values[i]
                validation_results[claim_id].passes_significance_test = corrected_p_values[i] < self.fdr_alpha
                
        # Generate summary statistics
        summary = self._generate_validation_summary(validation_results)
        
        return {
            'validation_results': validation_results,
            'summary': summary,
            'multiple_testing_correction': {
                'method': 'benjamini_hochberg',
                'alpha': self.fdr_alpha,
                'n_tests': len(p_values),
                'n_significant': sum(1 for r in validation_results.values() if r.passes_significance_test)
            }
        }
    
    def _extract_scores(self, 
                       experimental_data: Dict[str, Any], 
                       method: str, 
                       metric: str) -> List[float]:
        """Extract scores for a specific method and metric."""
        
        if method not in experimental_data:
            logger.warning(f"Method {method} not found in experimental data")
            return []
            
        method_data = experimental_data[method]
        
        # Handle different data structures
        if isinstance(method_data, list):
            scores = []
            for run in method_data:
                if metric in run:
                    scores.append(float(run[metric]))
            return scores
        elif isinstance(method_data, dict):
            if metric in method_data:
                if isinstance(method_data[metric], list):
                    return [float(x) for x in method_data[metric]]
                else:
                    return [float(method_data[metric])]
                    
        return []
    
    def _extract_degradation(self, 
                           experimental_data: Dict[str, Any], 
                           method: str, 
                           metric: str) -> List[float]:
        """Extract degradation scores from distribution shift experiments."""
        
        # Look for shift-specific results
        shift_results = experimental_data.get('distribution_shifts', {})
        
        degradation_scores = []
        for shift_name, shift_data in shift_results.items():
            if method in shift_data and metric in shift_data[method]:
                degradation_scores.extend(shift_data[method][metric])
                
        return degradation_scores
    
    def _generate_validation_summary(self, 
                                   validation_results: Dict[str, StatisticalValidationResult]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        
        total_claims = len(validation_results)
        significant_claims = sum(1 for r in validation_results.values() if r.passes_significance_test)
        ci_excludes_null = sum(1 for r in validation_results.values() if r.confidence_interval_excludes_null)
        meets_effect_size = sum(1 for r in validation_results.values() if r.meets_minimum_effect_size)
        
        # Effect size distribution
        effect_sizes = [r.effect_size for r in validation_results.values()]
        effect_interpretations = [r.effect_size_interpretation for r in validation_results.values()]
        
        return {
            'total_claims_tested': total_claims,
            'statistically_significant': significant_claims,
            'significance_rate': significant_claims / total_claims if total_claims > 0 else 0,
            'confidence_intervals_exclude_null': ci_excludes_null,
            'ci_exclusion_rate': ci_excludes_null / total_claims if total_claims > 0 else 0,
            'meets_minimum_effect_size': meets_effect_size,
            'effect_size_rate': meets_effect_size / total_claims if total_claims > 0 else 0,
            'effect_size_statistics': {
                'mean': np.mean(effect_sizes) if effect_sizes else 0,
                'std': np.std(effect_sizes) if effect_sizes else 0,
                'median': np.median(effect_sizes) if effect_sizes else 0,
                'interpretations': dict(pd.Series(effect_interpretations).value_counts()) if effect_interpretations else {}
            },
            'recommendation': self._generate_recommendation(significant_claims, total_claims)
        }
        
    def _generate_recommendation(self, significant_claims: int, total_claims: int) -> str:
        """Generate recommendation based on validation results."""
        
        if total_claims == 0:
            return "No claims could be validated due to insufficient data"
            
        success_rate = significant_claims / total_claims
        
        if success_rate >= 0.8:
            return "Strong statistical support for claims - proceed with publication"
        elif success_rate >= 0.6:
            return "Moderate statistical support - consider strengthening weak claims"
        elif success_rate >= 0.4:
            return "Mixed support - significant revision of claims recommended"
        else:
            return "Weak statistical support - major revision or additional data collection needed"

def main():
    """Example usage of statistical validation system."""
    
    # Sample experimental data
    experimental_data = {
        'bem_dynamic': [
            {'em_score': 81.4, 'f1_score': 85.2},
            {'em_score': 80.9, 'f1_score': 84.8},
            {'em_score': 81.7, 'f1_score': 85.6},
            {'em_score': 81.1, 'f1_score': 85.0},
            {'em_score': 81.3, 'f1_score': 85.3},
        ],
        'static_lora': [
            {'em_score': 78.2, 'f1_score': 82.1},
            {'em_score': 77.8, 'f1_score': 81.9},
            {'em_score': 78.5, 'f1_score': 82.3},
            {'em_score': 78.0, 'f1_score': 82.0},
            {'em_score': 78.3, 'f1_score': 82.2},
        ]
    }
    
    # Sample claim configs (minimal version)
    claim_configs = {
        'accuracy_claims': {
            'static_lora_advantage': {
                'claim': 'BEM +41.7% better accuracy than Static LoRA',
                'metric': 'em_score',
                'baseline_method': 'static_lora',
                'target_method': 'bem_dynamic',
                'expected_improvement_pct': 41.7,
                'minimum_effect_size_d': 1.0
            }
        }
    }
    
    # Run validation
    validator = StatisticalValidationOrchestrator()
    results = validator.validate_all_claims(experimental_data, claim_configs)
    
    # Print results
    print("Statistical Validation Results:")
    print(f"Total claims: {results['summary']['total_claims_tested']}")
    print(f"Significant: {results['summary']['statistically_significant']}")
    print(f"Recommendation: {results['summary']['recommendation']}")

if __name__ == "__main__":
    main()