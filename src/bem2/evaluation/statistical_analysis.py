"""
Statistical Analysis Framework for BEM v1.3
Implements BCa bootstrap confidence intervals and FDR correction for rigorous evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.stats import bootstrap
import warnings
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Results of statistical analysis."""
    metric_name: str
    baseline_mean: float
    treatment_mean: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_corrected: Optional[float]
    significant: bool
    confidence_level: float
    n_bootstrap_samples: int
    method: str


@dataclass
class MultipleTestingResult:
    """Results of multiple hypothesis testing with FDR correction."""
    results: List[StatisticalResult]
    fdr_alpha: float
    num_hypotheses: int
    num_rejected: int
    rejection_threshold: float


class BCaBootstrap:
    """
    Bias-Corrected and Accelerated (BCa) Bootstrap for confidence intervals.
    
    This implementation follows Efron & Tibshirani (1993) and DiCiccio & Efron (1996)
    for computing accurate confidence intervals that correct for bias and skewness.
    """
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def compute_ci(self, 
                   data: np.ndarray, 
                   statistic_fn: Callable[[np.ndarray], float],
                   return_bootstrap_dist: bool = False) -> Tuple[float, float, Optional[np.ndarray]]:
        """
        Compute BCa confidence interval for a given statistic.
        
        Args:
            data: Original sample data
            statistic_fn: Function that computes the statistic from data
            return_bootstrap_dist: Whether to return the bootstrap distribution
            
        Returns:
            Tuple of (ci_lower, ci_upper, bootstrap_dist)
        """
        n = len(data)
        
        # Original statistic
        theta_hat = statistic_fn(data)
        
        # Bootstrap replicates
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = rng.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_fn(bootstrap_sample))
            
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Bias correction (z0)
        z0 = self._compute_bias_correction(bootstrap_stats, theta_hat)
        
        # Acceleration (a_hat)
        a_hat = self._compute_acceleration(data, statistic_fn, theta_hat)
        
        # Adjusted percentiles
        z_alpha_2 = stats.norm.ppf(self.alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - self.alpha / 2)
        
        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1 - a_hat * (z0 + z_alpha_2)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - a_hat * (z0 + z_1_alpha_2)))
        
        # Ensure percentiles are valid
        alpha1 = np.clip(alpha1, 0.001, 0.999)
        alpha2 = np.clip(alpha2, 0.001, 0.999)
        
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha1)
        ci_upper = np.percentile(bootstrap_stats, 100 * alpha2)
        
        bootstrap_dist = bootstrap_stats if return_bootstrap_dist else None
        
        return ci_lower, ci_upper, bootstrap_dist
    
    def _compute_bias_correction(self, bootstrap_stats: np.ndarray, theta_hat: float) -> float:
        """Compute bias correction z0."""
        prop_less = np.mean(bootstrap_stats < theta_hat)
        # Avoid edge cases
        prop_less = np.clip(prop_less, 0.001, 0.999)
        z0 = stats.norm.ppf(prop_less)
        return z0
    
    def _compute_acceleration(self, data: np.ndarray, statistic_fn: Callable, theta_hat: float) -> float:
        """Compute acceleration constant using jackknife."""
        n = len(data)
        jackknife_stats = []
        
        # Jackknife: leave-one-out statistics
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stats.append(statistic_fn(jackknife_sample))
            
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        # Acceleration constant
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        
        if denominator == 0:
            return 0.0
        
        a_hat = numerator / denominator
        return a_hat


class FDRCorrection:
    """
    False Discovery Rate (FDR) correction using Benjamini-Hochberg procedure.
    
    Controls the expected proportion of false discoveries among rejected hypotheses.
    """
    
    @staticmethod
    def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Args:
            p_values: Array of p-values to correct
            alpha: FDR level (default 0.05)
            
        Returns:
            Tuple of (corrected_p_values, rejected_hypotheses, rejection_threshold)
        """
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Benjamini-Hochberg critical values
        critical_values = np.arange(1, m + 1) * alpha / m
        
        # Find largest k such that P(k) <= (k/m) * alpha
        rejections = sorted_p_values <= critical_values
        
        if np.any(rejections):
            # Find the largest index where we reject
            max_reject_idx = np.max(np.where(rejections)[0])
            rejection_threshold = critical_values[max_reject_idx]
        else:
            max_reject_idx = -1
            rejection_threshold = 0.0
        
        # Create boolean array for rejections in original order
        rejected = np.zeros(m, dtype=bool)
        if max_reject_idx >= 0:
            rejected_indices = sorted_indices[:max_reject_idx + 1]
            rejected[rejected_indices] = True
        
        # Corrected p-values (step-up method)
        corrected_p = np.zeros(m)
        for i in range(m):
            original_idx = sorted_indices[i]
            corrected_p[original_idx] = min(1.0, sorted_p_values[i] * m / (i + 1))
        
        return corrected_p, rejected, rejection_threshold


class PairedComparison:
    """
    Paired comparison analysis for A/B testing scenarios.
    
    Handles paired data where each observation has a baseline and treatment measurement.
    """
    
    def __init__(self, bootstrap_samples: int = 10000, confidence_level: float = 0.95):
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.bca = BCaBootstrap(bootstrap_samples, confidence_level)
    
    def compare_metrics(self, 
                       baseline_data: np.ndarray, 
                       treatment_data: np.ndarray,
                       metric_name: str) -> StatisticalResult:
        """
        Compare two sets of measurements using BCa bootstrap.
        
        Args:
            baseline_data: Baseline measurements
            treatment_data: Treatment measurements  
            metric_name: Name of the metric being compared
            
        Returns:
            StatisticalResult with comparison results
        """
        if len(baseline_data) != len(treatment_data):
            raise ValueError("Baseline and treatment data must have same length")
        
        # Compute paired differences
        differences = treatment_data - baseline_data
        
        # Basic statistics
        baseline_mean = np.mean(baseline_data)
        treatment_mean = np.mean(treatment_data)
        effect_size = treatment_mean - baseline_mean
        
        # BCa confidence interval for the mean difference
        def mean_statistic(data):
            return np.mean(data)
        
        ci_lower, ci_upper, _ = self.bca.compute_ci(differences, mean_statistic)
        
        # Two-sided t-test for p-value
        t_stat, p_value = stats.ttest_rel(treatment_data, baseline_data)
        
        # Determine significance based on CI
        significant = not (ci_lower <= 0 <= ci_upper)
        
        return StatisticalResult(
            metric_name=metric_name,
            baseline_mean=baseline_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            p_value_corrected=None,  # Will be filled by multiple testing correction
            significant=significant,
            confidence_level=self.confidence_level,
            n_bootstrap_samples=self.bootstrap_samples,
            method="BCa_bootstrap_paired"
        )


class StatisticalAnalyzer:
    """
    Main statistical analysis framework for BEM v1.3 evaluation.
    
    Coordinates paired comparisons, multiple testing correction, and result reporting.
    """
    
    def __init__(self, 
                 bootstrap_samples: int = 10000,
                 confidence_level: float = 0.95,
                 fdr_alpha: float = 0.05):
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.fdr_alpha = fdr_alpha
        
        self.paired_comparison = PairedComparison(bootstrap_samples, confidence_level)
        self.fdr_correction = FDRCorrection()
        
    def analyze_experiment(self, 
                          baseline_results: Dict[str, np.ndarray],
                          treatment_results: Dict[str, np.ndarray],
                          experiment_name: str) -> MultipleTestingResult:
        """
        Perform complete statistical analysis of an experiment.
        
        Args:
            baseline_results: Dict mapping metric names to baseline measurements
            treatment_results: Dict mapping metric names to treatment measurements
            experiment_name: Name of the experiment for logging
            
        Returns:
            MultipleTestingResult with all comparisons and FDR correction
        """
        logger.info(f"Analyzing experiment: {experiment_name}")
        
        # Validate inputs
        if set(baseline_results.keys()) != set(treatment_results.keys()):
            raise ValueError("Baseline and treatment results must have same metrics")
        
        # Perform paired comparisons for each metric
        comparison_results = []
        for metric_name in baseline_results.keys():
            baseline_data = baseline_results[metric_name]
            treatment_data = treatment_results[metric_name]
            
            if len(baseline_data) == 0 or len(treatment_data) == 0:
                logger.warning(f"Empty data for metric {metric_name}, skipping")
                continue
                
            result = self.paired_comparison.compare_metrics(
                baseline_data, treatment_data, metric_name
            )
            comparison_results.append(result)
            
        if not comparison_results:
            raise ValueError("No valid comparisons could be performed")
        
        # Apply FDR correction
        p_values = np.array([r.p_value for r in comparison_results])
        corrected_p, rejected, rejection_threshold = self.fdr_correction.benjamini_hochberg(
            p_values, self.fdr_alpha
        )
        
        # Update results with corrected p-values and significance
        for i, result in enumerate(comparison_results):
            result.p_value_corrected = corrected_p[i]
            result.significant = rejected[i]
        
        logger.info(f"Analysis complete: {len(comparison_results)} metrics, "
                   f"{np.sum(rejected)} significant after FDR correction")
        
        return MultipleTestingResult(
            results=comparison_results,
            fdr_alpha=self.fdr_alpha,
            num_hypotheses=len(comparison_results),
            num_rejected=int(np.sum(rejected)),
            rejection_threshold=rejection_threshold
        )
    
    def generate_report(self, 
                       analysis_result: MultipleTestingResult,
                       output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive statistical report.
        
        Args:
            analysis_result: Results from analyze_experiment
            output_path: Optional path to save the report
            
        Returns:
            Formatted report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BEM v1.3 Statistical Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Number of metrics analyzed: {analysis_result.num_hypotheses}")
        report_lines.append(f"FDR level (α): {analysis_result.fdr_alpha:.3f}")
        report_lines.append(f"Significant results: {analysis_result.num_rejected}")
        report_lines.append(f"FDR rejection threshold: {analysis_result.rejection_threshold:.6f}")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 40)
        
        # Sort by effect size magnitude for reporting
        sorted_results = sorted(analysis_result.results, 
                              key=lambda x: abs(x.effect_size), reverse=True)
        
        for result in sorted_results:
            status = "SIGNIFICANT" if result.significant else "NOT SIGNIFICANT"
            direction = "↑" if result.effect_size > 0 else "↓"
            
            report_lines.append(f"\nMetric: {result.metric_name}")
            report_lines.append(f"  Status: {status}")
            report_lines.append(f"  Effect: {direction} {result.effect_size:.6f}")
            report_lines.append(f"  Baseline: {result.baseline_mean:.6f}")
            report_lines.append(f"  Treatment: {result.treatment_mean:.6f}")
            report_lines.append(f"  95% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
            report_lines.append(f"  p-value: {result.p_value:.6f}")
            report_lines.append(f"  p-value (FDR corrected): {result.p_value_corrected:.6f}")
        
        # Statistical methodology
        report_lines.append("\n\nMETHODOLOGY")
        report_lines.append("-" * 40)
        report_lines.append("• Confidence intervals: BCa (Bias-Corrected and Accelerated) Bootstrap")
        report_lines.append("• Multiple testing correction: Benjamini-Hochberg FDR procedure")
        report_lines.append(f"• Bootstrap samples: {self.bootstrap_samples:,}")
        report_lines.append(f"• Confidence level: {self.confidence_level:.1%}")
        report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def validate_statistical_assumptions(data: np.ndarray, metric_name: str) -> Dict[str, Any]:
    """
    Validate statistical assumptions for the analysis.
    
    Args:
        data: Data to validate
        metric_name: Name of the metric for logging
        
    Returns:
        Dict with validation results and warnings
    """
    results = {
        'metric': metric_name,
        'n_samples': len(data),
        'warnings': [],
        'recommendations': []
    }
    
    # Check sample size
    if len(data) < 30:
        results['warnings'].append(f"Small sample size (n={len(data)})")
        results['recommendations'].append("Consider collecting more data for robust inference")
    
    # Check for outliers (using IQR method)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    outlier_threshold = 1.5 * IQR
    outliers = np.sum((data < Q1 - outlier_threshold) | (data > Q3 + outlier_threshold))
    
    if outliers > 0.05 * len(data):  # More than 5% outliers
        results['warnings'].append(f"High proportion of outliers ({outliers}/{len(data)})")
        results['recommendations'].append("Consider robust statistical methods or outlier treatment")
    
    # Check for extreme skewness
    if len(data) > 3:
        skewness = stats.skew(data)
        if abs(skewness) > 2:
            results['warnings'].append(f"High skewness ({skewness:.2f})")
            results['recommendations'].append("BCa bootstrap handles skewness well")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Example: Compare two models on multiple metrics
    np.random.seed(42)
    
    # Simulate baseline and treatment data
    n_samples = 100
    baseline_em = np.random.beta(0.6, 0.4, n_samples)  # EM scores
    treatment_em = baseline_em + np.random.normal(0.02, 0.05, n_samples)  # 2% improvement
    
    baseline_f1 = np.random.beta(0.65, 0.35, n_samples)  # F1 scores  
    treatment_f1 = baseline_f1 + np.random.normal(0.015, 0.04, n_samples)  # 1.5% improvement
    
    baseline_bleu = np.random.beta(0.4, 0.6, n_samples)  # BLEU scores
    treatment_bleu = baseline_bleu + np.random.normal(-0.005, 0.03, n_samples)  # Small decline
    
    baseline_results = {
        'exact_match': baseline_em,
        'f1_score': baseline_f1,
        'bleu': baseline_bleu
    }
    
    treatment_results = {
        'exact_match': treatment_em,
        'f1_score': treatment_f1, 
        'bleu': treatment_bleu
    }
    
    # Perform analysis
    analyzer = StatisticalAnalyzer()
    results = analyzer.analyze_experiment(
        baseline_results, treatment_results, "V1_DynamicRank_Test"
    )
    
    # Generate report
    report = analyzer.generate_report(results)
    print(report)