"""
Performance Track Evaluation Framework.

Comprehensive evaluation suite for PT1-PT4 variants including:
- Pareto frontier analysis
- Budget parity validation  
- Statistical significance testing
- Latency profiling
- VRAM usage tracking
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import numpy as np
import time
import gc
import logging
from pathlib import Path
import json

# Statistical testing
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon

logger = logging.getLogger(__name__)


@dataclass
class BudgetConstraints:
    """Budget constraints for performance track variants."""
    
    # Anchor baseline (v1.3-stack)
    baseline_params: int = 124964096    # From v13_anchor.yml
    baseline_flops: int = 1000000       # Placeholder, will be computed
    baseline_memory_mb: float = 512.0   # Estimated
    
    # Tolerance (±5%)
    tolerance: float = 0.05
    
    # Computed bounds
    @property
    def param_bounds(self) -> Tuple[int, int]:
        delta = int(self.baseline_params * self.tolerance)
        return (self.baseline_params - delta, self.baseline_params + delta)
    
    @property
    def flop_bounds(self) -> Tuple[int, int]:
        delta = int(self.baseline_flops * self.tolerance) 
        return (self.baseline_flops - delta, self.baseline_flops + delta)
    
    @property
    def memory_bounds(self) -> Tuple[float, float]:
        delta = self.baseline_memory_mb * self.tolerance
        return (self.baseline_memory_mb - delta, self.baseline_memory_mb + delta)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single variant."""
    
    # Core metrics
    exact_match: float = 0.0
    f1_score: float = 0.0
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    chrf_score: float = 0.0
    
    # Resource usage
    parameters: int = 0
    flops: int = 0
    memory_mb: float = 0.0
    
    # Latency metrics
    inference_latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # Training metrics
    training_loss: float = float('inf')
    validation_loss: float = float('inf')
    convergence_steps: int = -1
    
    # Variant-specific metrics
    variant_specific: Dict[str, float] = None
    
    def __post_init__(self):
        if self.variant_specific is None:
            self.variant_specific = {}
    
    def primary_metric(self) -> float:
        """Get primary performance metric (F1 score)."""
        return self.f1_score
    
    def secondary_metrics(self) -> Dict[str, float]:
        """Get secondary metrics for Pareto analysis."""
        return {
            'exact_match': self.exact_match,
            'bleu': self.bleu_score,
            'rouge': self.rouge_score,
            'chrf': self.chrf_score
        }


class BudgetValidator:
    """Validates budget parity constraints for PT variants."""
    
    def __init__(self, constraints: BudgetConstraints):
        self.constraints = constraints
        
    def validate_parameters(self, param_count: int) -> Dict[str, Union[bool, float, str]]:
        """Validate parameter count against budget."""
        min_params, max_params = self.constraints.param_bounds
        
        is_valid = min_params <= param_count <= max_params
        deviation = (param_count - self.constraints.baseline_params) / self.constraints.baseline_params
        
        return {
            'valid': is_valid,
            'param_count': param_count,
            'baseline': self.constraints.baseline_params,
            'deviation_pct': deviation * 100,
            'bounds': (min_params, max_params),
            'status': 'PASS' if is_valid else 'FAIL'
        }
    
    def validate_flops(self, flop_count: int) -> Dict[str, Union[bool, float, str]]:
        """Validate FLOP count against budget."""
        min_flops, max_flops = self.constraints.flop_bounds
        
        is_valid = min_flops <= flop_count <= max_flops
        deviation = (flop_count - self.constraints.baseline_flops) / self.constraints.baseline_flops
        
        return {
            'valid': is_valid,
            'flop_count': flop_count,
            'baseline': self.constraints.baseline_flops,
            'deviation_pct': deviation * 100,
            'bounds': (min_flops, max_flops),
            'status': 'PASS' if is_valid else 'FAIL'
        }
    
    def validate_memory(self, memory_mb: float) -> Dict[str, Union[bool, float, str]]:
        """Validate memory usage against budget."""
        min_memory, max_memory = self.constraints.memory_bounds
        
        is_valid = min_memory <= memory_mb <= max_memory
        deviation = (memory_mb - self.constraints.baseline_memory_mb) / self.constraints.baseline_memory_mb
        
        return {
            'valid': is_valid,
            'memory_mb': memory_mb,
            'baseline': self.constraints.baseline_memory_mb,
            'deviation_pct': deviation * 100,
            'bounds': (min_memory, max_memory),
            'status': 'PASS' if is_valid else 'FAIL'
        }
    
    def validate_all(self, metrics: PerformanceMetrics) -> Dict[str, Dict]:
        """Validate all budget constraints."""
        return {
            'parameters': self.validate_parameters(metrics.parameters),
            'flops': self.validate_flops(metrics.flops),
            'memory': self.validate_memory(metrics.memory_mb)
        }
    
    def overall_budget_status(self, metrics: PerformanceMetrics) -> str:
        """Get overall budget validation status."""
        validations = self.validate_all(metrics)
        
        all_valid = all(v['valid'] for v in validations.values())
        return 'PASS' if all_valid else 'FAIL'


class ParetoAnalyzer:
    """Analyzes Pareto frontier for performance vs resource trade-offs."""
    
    def __init__(self):
        self.results: List[Tuple[str, PerformanceMetrics]] = []
        
    def add_result(self, variant_name: str, metrics: PerformanceMetrics):
        """Add variant results for Pareto analysis."""
        self.results.append((variant_name, metrics))
        
    def compute_pareto_frontier(
        self, 
        x_metric: str = 'parameters',
        y_metric: str = 'f1_score',
        maximize_y: bool = True
    ) -> List[Tuple[str, PerformanceMetrics]]:
        """
        Compute Pareto frontier.
        
        Args:
            x_metric: Resource metric name (to minimize)
            y_metric: Performance metric name
            maximize_y: Whether to maximize y_metric
            
        Returns:
            List of Pareto optimal (variant_name, metrics) pairs
        """
        if not self.results:
            return []
        
        # Extract x, y coordinates
        points = []
        for name, metrics in self.results:
            if x_metric == 'parameters':
                x = metrics.parameters
            elif x_metric == 'flops':
                x = metrics.flops
            elif x_metric == 'memory_mb':
                x = metrics.memory_mb
            else:
                x = getattr(metrics, x_metric, 0)
                
            if y_metric in ['exact_match', 'f1_score', 'bleu_score', 'rouge_score', 'chrf_score']:
                y = getattr(metrics, y_metric, 0)
            else:
                y = metrics.variant_specific.get(y_metric, 0)
            
            points.append((name, metrics, x, y))
        
        # Sort by x coordinate (resource usage)
        points.sort(key=lambda p: p[2])
        
        # Find Pareto frontier
        pareto_points = []
        best_y = float('-inf') if maximize_y else float('inf')
        
        for name, metrics, x, y in points:
            if maximize_y and y > best_y:
                pareto_points.append((name, metrics))
                best_y = y
            elif not maximize_y and y < best_y:
                pareto_points.append((name, metrics))
                best_y = y
        
        return pareto_points
    
    def pareto_dominance_analysis(self) -> Dict[str, Any]:
        """Comprehensive Pareto dominance analysis."""
        if len(self.results) < 2:
            return {'error': 'Need at least 2 variants for Pareto analysis'}
        
        # Multiple objective Pareto analysis
        frontiers = {}
        
        # Performance vs Parameters
        frontiers['performance_vs_params'] = self.compute_pareto_frontier(
            'parameters', 'f1_score', maximize_y=True
        )
        
        # Performance vs FLOPs
        frontiers['performance_vs_flops'] = self.compute_pareto_frontier(
            'flops', 'f1_score', maximize_y=True
        )
        
        # Performance vs Memory
        frontiers['performance_vs_memory'] = self.compute_pareto_frontier(
            'memory_mb', 'f1_score', maximize_y=True
        )
        
        # BLEU vs Parameters
        frontiers['bleu_vs_params'] = self.compute_pareto_frontier(
            'parameters', 'bleu_score', maximize_y=True
        )
        
        # Find variants that appear in multiple frontiers
        frontier_membership = {}
        for frontier_name, frontier_points in frontiers.items():
            for variant_name, _ in frontier_points:
                if variant_name not in frontier_membership:
                    frontier_membership[variant_name] = []
                frontier_membership[variant_name].append(frontier_name)
        
        # Compute dominance relationships
        dominance = self._compute_dominance_matrix()
        
        return {
            'frontiers': frontiers,
            'frontier_membership': frontier_membership,
            'dominance_matrix': dominance,
            'robust_variants': [
                name for name, memberships in frontier_membership.items()
                if len(memberships) >= 2
            ]
        }
    
    def _compute_dominance_matrix(self) -> Dict[str, Dict[str, str]]:
        """Compute Pareto dominance matrix between all variants."""
        dominance = {}
        
        for i, (name1, metrics1) in enumerate(self.results):
            dominance[name1] = {}
            
            for j, (name2, metrics2) in enumerate(self.results):
                if i == j:
                    dominance[name1][name2] = 'equal'
                    continue
                
                # Check dominance: name1 dominates name2 if:
                # - name1 is better or equal on all objectives
                # - name1 is strictly better on at least one objective
                
                objectives = [
                    ('f1_score', True),      # maximize
                    ('parameters', False),    # minimize
                    ('flops', False),        # minimize
                    ('memory_mb', False),    # minimize
                    ('inference_latency_ms', False)  # minimize
                ]
                
                better_count = 0
                worse_count = 0
                
                for obj_name, maximize in objectives:
                    val1 = getattr(metrics1, obj_name, 0)
                    val2 = getattr(metrics2, obj_name, 0)
                    
                    if maximize:
                        if val1 > val2:
                            better_count += 1
                        elif val1 < val2:
                            worse_count += 1
                    else:
                        if val1 < val2:
                            better_count += 1
                        elif val1 > val2:
                            worse_count += 1
                
                if worse_count == 0 and better_count > 0:
                    dominance[name1][name2] = 'dominates'
                elif better_count == 0 and worse_count > 0:
                    dominance[name1][name2] = 'dominated_by'
                else:
                    dominance[name1][name2] = 'incomparable'
        
        return dominance


class StatisticalValidator:
    """Statistical significance testing for performance improvements."""
    
    @staticmethod
    def validate_improvement(
        baseline_scores: List[float],
        variant_scores: List[float],
        alpha: float = 0.05,
        test_type: str = 'ttest'
    ) -> Dict[str, Any]:
        """
        Test statistical significance of improvement.
        
        Args:
            baseline_scores: List of baseline performance scores
            variant_scores: List of variant performance scores  
            alpha: Significance level
            test_type: 'ttest' or 'wilcoxon'
            
        Returns:
            Dictionary with test results
        """
        if len(baseline_scores) < 3 or len(variant_scores) < 3:
            return {
                'error': 'Need at least 3 samples for statistical testing',
                'valid': False
            }
        
        # Compute basic statistics
        baseline_mean = np.mean(baseline_scores)
        variant_mean = np.mean(variant_scores)
        improvement = variant_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean) * 100 if baseline_mean != 0 else 0
        
        # Perform statistical test
        if test_type == 'ttest':
            statistic, p_value = ttest_ind(variant_scores, baseline_scores, alternative='greater')
            test_name = "Two-sample t-test (one-sided)"
        elif test_type == 'wilcoxon':
            # Paired test if same length, otherwise Mann-Whitney U
            if len(baseline_scores) == len(variant_scores):
                statistic, p_value = wilcoxon(variant_scores, baseline_scores, alternative='greater')
                test_name = "Wilcoxon signed-rank test"
            else:
                from scipy.stats import mannwhitneyu
                statistic, p_value = mannwhitneyu(variant_scores, baseline_scores, alternative='greater')
                test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1) +
             (len(variant_scores) - 1) * np.var(variant_scores, ddof=1)) /
            (len(baseline_scores) + len(variant_scores) - 2)
        )
        cohens_d = improvement / pooled_std if pooled_std != 0 else 0
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # 95% Confidence interval for improvement
        se_diff = np.sqrt(np.var(baseline_scores, ddof=1)/len(baseline_scores) + 
                         np.var(variant_scores, ddof=1)/len(variant_scores))
        t_critical = stats.t.ppf(0.975, len(baseline_scores) + len(variant_scores) - 2)
        ci_lower = improvement - t_critical * se_diff
        ci_upper = improvement + t_critical * se_diff
        
        return {
            'valid': True,
            'test_name': test_name,
            'baseline_mean': baseline_mean,
            'variant_mean': variant_mean,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'p_value': p_value,
            'statistic': statistic,
            'significant': p_value < alpha,
            'alpha': alpha,
            'cohens_d': cohens_d,
            'effect_size': effect_interpretation,
            'confidence_interval_95': (ci_lower, ci_upper),
            'sample_sizes': (len(baseline_scores), len(variant_scores))
        }
    
    @staticmethod
    def slice_b_analysis(
        baseline_results: Dict[str, List[float]],
        variant_results: Dict[str, List[float]],
        slices: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        CI-backed Slice-B analysis for comprehensive evaluation.
        
        Args:
            baseline_results: Dict mapping slice names to score lists
            variant_results: Dict mapping slice names to score lists
            slices: List of slice names to analyze (if None, use all)
            
        Returns:
            Dictionary with per-slice statistical analysis
        """
        if slices is None:
            slices = list(set(baseline_results.keys()) & set(variant_results.keys()))
        
        slice_analyses = {}
        
        for slice_name in slices:
            if slice_name not in baseline_results or slice_name not in variant_results:
                slice_analyses[slice_name] = {'error': f'Missing data for slice {slice_name}'}
                continue
            
            baseline_scores = baseline_results[slice_name]
            variant_scores = variant_results[slice_name]
            
            # Statistical test
            analysis = StatisticalValidator.validate_improvement(
                baseline_scores, variant_scores,
                test_type='wilcoxon'  # More robust for slice analysis
            )
            
            slice_analyses[slice_name] = analysis
        
        # Aggregate analysis
        significant_slices = [
            name for name, results in slice_analyses.items()
            if results.get('significant', False)
        ]
        
        avg_improvement = np.mean([
            results.get('improvement_pct', 0) 
            for results in slice_analyses.values()
            if 'improvement_pct' in results
        ])
        
        slice_analyses['_aggregate'] = {
            'total_slices': len(slices),
            'significant_slices': len(significant_slices),
            'significance_rate': len(significant_slices) / len(slices) if slices else 0,
            'avg_improvement_pct': avg_improvement,
            'robust_improvement': len(significant_slices) >= len(slices) * 0.6  # 60% threshold
        }
        
        return slice_analyses


class PerformanceProfiler:
    """Profiles latency, throughput, and resource usage."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def profile_inference_latency(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Profile inference latency with comprehensive statistics."""
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_data)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
        
        # Timing runs
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                _ = model(input_data)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Compute statistics
        latencies = np.array(latencies)
        
        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p90_ms': np.percentile(latencies, 90),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'num_runs': num_runs,
            'throughput_samples_per_sec': 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        }
    
    def profile_memory_usage(
        self,
        model: nn.Module,
        input_data: torch.Tensor
    ) -> Dict[str, float]:
        """Profile memory usage during inference."""
        if self.device != 'cuda':
            return {'error': 'Memory profiling only supported on CUDA'}
        
        model.eval()
        
        # Clear cache and measure baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        baseline_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_data)
            torch.cuda.synchronize()
        
        # Measure peak and current usage
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        return {
            'baseline_mb': baseline_memory / (1024**2),
            'current_mb': current_memory / (1024**2),
            'peak_mb': peak_memory / (1024**2),
            'allocated_mb': (current_memory - baseline_memory) / (1024**2),
            'peak_delta_mb': (peak_memory - baseline_memory) / (1024**2)
        }
    
    def comprehensive_profile(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        variant_name: str
    ) -> PerformanceMetrics:
        """Create comprehensive performance profile."""
        
        # Latency profiling
        latency_stats = self.profile_inference_latency(model, input_data)
        
        # Memory profiling
        memory_stats = self.profile_memory_usage(model, input_data)
        
        # Parameter count
        param_count = sum(p.numel() for p in model.parameters())
        
        # Create metrics object
        metrics = PerformanceMetrics(
            parameters=param_count,
            memory_mb=memory_stats.get('peak_mb', 0),
            inference_latency_ms=latency_stats['mean_ms'],
            throughput_samples_per_sec=latency_stats['throughput_samples_per_sec'],
            variant_specific={
                'latency_p95_ms': latency_stats['p95_ms'],
                'latency_std_ms': latency_stats['std_ms'],
                'memory_peak_mb': memory_stats.get('peak_mb', 0),
                'memory_allocated_mb': memory_stats.get('allocated_mb', 0)
            }
        )
        
        return metrics


class ComprehensiveEvaluator:
    """Main evaluation coordinator for all PT variants."""
    
    def __init__(self, budget_constraints: BudgetConstraints):
        self.budget_constraints = budget_constraints
        self.budget_validator = BudgetValidator(budget_constraints)
        self.pareto_analyzer = ParetoAnalyzer()
        self.profiler = PerformanceProfiler()
        
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def evaluate_variant(
        self,
        variant_name: str,
        model: nn.Module,
        test_data: torch.Tensor,
        performance_scores: Dict[str, float] = None,
        baseline_scores: Dict[str, List[float]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single PT variant.
        
        Args:
            variant_name: Name of the variant (PT1, PT2, PT3, PT4)
            model: The model to evaluate
            test_data: Test data for profiling
            performance_scores: Dict of performance metric scores
            baseline_scores: Dict of baseline scores for statistical testing
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"Evaluating {variant_name}")
        
        # Performance profiling
        performance_metrics = self.profiler.comprehensive_profile(
            model, test_data, variant_name
        )
        
        # Update with provided performance scores
        if performance_scores:
            for key, value in performance_scores.items():
                if hasattr(performance_metrics, key):
                    setattr(performance_metrics, key, value)
                else:
                    performance_metrics.variant_specific[key] = value
        
        # Budget validation
        budget_results = self.budget_validator.validate_all(performance_metrics)
        budget_status = self.budget_validator.overall_budget_status(performance_metrics)
        
        # Statistical validation (if baseline provided)
        statistical_results = {}
        if baseline_scores:
            for metric_name, variant_scores in performance_scores.items():
                if metric_name in baseline_scores and isinstance(variant_scores, list):
                    statistical_results[metric_name] = StatisticalValidator.validate_improvement(
                        baseline_scores[metric_name],
                        variant_scores
                    )
        
        # Add to Pareto analysis
        self.pareto_analyzer.add_result(variant_name, performance_metrics)
        
        # Compile results
        evaluation_results = {
            'variant_name': variant_name,
            'performance_metrics': asdict(performance_metrics),
            'budget_validation': {
                'results': budget_results,
                'overall_status': budget_status,
                'constraints': asdict(self.budget_constraints)
            },
            'statistical_validation': statistical_results,
            'timestamp': time.time()
        }
        
        self.results[variant_name] = evaluation_results
        return evaluation_results
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final evaluation report."""
        
        if not self.results:
            return {'error': 'No evaluation results available'}
        
        # Pareto analysis
        pareto_analysis = self.pareto_analyzer.pareto_dominance_analysis()
        
        # Summary statistics
        summary_stats = {
            'variants_evaluated': len(self.results),
            'budget_passing_variants': sum(
                1 for r in self.results.values()
                if r['budget_validation']['overall_status'] == 'PASS'
            ),
            'pareto_optimal_variants': len(
                set().union(*[
                    [name for name, _ in frontier]
                    for frontier in pareto_analysis['frontiers'].values()
                ])
            )
        }
        
        # Ranking by primary metric
        variant_rankings = sorted(
            [(name, r['performance_metrics']['f1_score']) 
             for name, r in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Promotion recommendations
        promotion_candidates = []
        for variant_name, results in self.results.items():
            budget_pass = results['budget_validation']['overall_status'] == 'PASS'
            
            # Check if variant appears in any Pareto frontier
            in_pareto = variant_name in pareto_analysis.get('frontier_membership', {})
            
            # Check for statistical significance
            has_significant_improvement = any(
                stat_result.get('significant', False)
                for stat_result in results.get('statistical_validation', {}).values()
            )
            
            if budget_pass and (in_pareto or has_significant_improvement):
                promotion_candidates.append({
                    'variant': variant_name,
                    'budget_valid': budget_pass,
                    'pareto_optimal': in_pareto,
                    'statistically_significant': has_significant_improvement,
                    'f1_score': results['performance_metrics']['f1_score']
                })
        
        # Sort promotion candidates by performance
        promotion_candidates.sort(key=lambda x: x['f1_score'], reverse=True)
        
        final_report = {
            'evaluation_summary': summary_stats,
            'variant_rankings': variant_rankings,
            'pareto_analysis': pareto_analysis,
            'promotion_candidates': promotion_candidates,
            'budget_constraints': asdict(self.budget_constraints),
            'detailed_results': self.results,
            'evaluation_timestamp': time.time(),
            'report_version': '1.0'
        }
        
        return final_report
    
    def save_report(self, report_path: Path):
        """Save evaluation report to file."""
        final_report = self.generate_final_report()
        
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def print_summary(self):
        """Print evaluation summary to console."""
        final_report = self.generate_final_report()
        
        print("\n" + "="*80)
        print("BEM 2.0 PERFORMANCE TRACK EVALUATION SUMMARY")
        print("="*80)
        
        summary = final_report['evaluation_summary']
        print(f"Variants Evaluated: {summary['variants_evaluated']}")
        print(f"Budget-Compliant Variants: {summary['budget_passing_variants']}")
        print(f"Pareto-Optimal Variants: {summary['pareto_optimal_variants']}")
        
        print("\nVARIANT RANKINGS (by F1 score):")
        print("-" * 40)
        for i, (variant, f1_score) in enumerate(final_report['variant_rankings'], 1):
            budget_status = self.results[variant]['budget_validation']['overall_status']
            print(f"{i}. {variant}: {f1_score:.4f} [{budget_status}]")
        
        print("\nPROMOTION CANDIDATES:")
        print("-" * 40)
        candidates = final_report['promotion_candidates']
        if candidates:
            for candidate in candidates:
                print(f"✅ {candidate['variant']}: F1={candidate['f1_score']:.4f}")
                print(f"   Budget Valid: {candidate['budget_valid']}")
                print(f"   Pareto Optimal: {candidate['pareto_optimal']}")
                print(f"   Statistically Significant: {candidate['statistically_significant']}")
                print()
        else:
            print("❌ No variants meet promotion criteria")
        
        print("="*80)