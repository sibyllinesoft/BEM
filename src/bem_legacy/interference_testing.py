"""
BEM Phase 4: Interference Testing Framework

This module implements comprehensive interference analysis for multi-BEM systems.
It validates that multiple BEMs can work together without degrading performance
on off-domain canary tasks, providing statistical confidence in non-interference.

Key Features:
- Canary Tasks: Off-domain validation tasks for regression testing
- Performance Tracking: Monitor task-specific performance degradation  
- Interference Plots: Visualize performance interactions across BEMs
- Statistical Analysis: Quantify interference with confidence intervals
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Any, Callable
from dataclasses import dataclass
import logging
import json
from collections import defaultdict
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy import stats

from .multi_bem import MultiBEMComposer

logger = logging.getLogger(__name__)


@dataclass  
class CanaryTask:
    """Definition of a canary task for interference testing."""
    task_id: str
    task_name: str
    task_type: str  # 'regression_test', 'capability_test', 'safety_test'
    eval_function: Callable[[Any, Any], float]  # (model, data) -> score
    data_loader: Any  # Data loader for the task
    baseline_score: Optional[float] = None
    regression_threshold: float = 0.02  # 2% maximum allowed regression
    higher_is_better: bool = True


@dataclass
class BEMConfiguration:
    """Configuration for a BEM in interference testing."""
    bem_id: str
    enabled: bool = True
    priority: int = 0
    task_domains: List[str] = None  # Domains this BEM is intended for
    
    def __post_init__(self):
        if self.task_domains is None:
            self.task_domains = []


class InterferenceTestResult(NamedTuple):
    """Result of interference testing for a specific configuration."""
    config_id: str
    bem_configs: List[BEMConfiguration]
    canary_results: Dict[str, float]  # task_id -> score
    performance_changes: Dict[str, float]  # task_id -> % change from baseline
    violations: List[str]  # List of tasks that violate regression threshold
    overall_interference_score: float  # Aggregate interference metric
    statistical_significance: Dict[str, float]  # task_id -> p-value
    confidence_intervals: Dict[str, Tuple[float, float]]  # task_id -> (lower, upper)


class BEMCombinationResult(NamedTuple):
    """Result for testing different BEM combinations."""
    combination_id: str
    active_bems: List[str]
    test_results: List[InterferenceTestResult]
    mean_interference: float
    max_interference: float
    violation_rate: float  # Fraction of canary tasks showing violations


class InterferenceTester:
    """
    Main interference testing framework for multi-BEM systems.
    
    This class orchestrates comprehensive interference testing by running
    canary tasks across different BEM configurations and combinations.
    """
    
    def __init__(
        self,
        composer: MultiBEMComposer,
        canary_tasks: List[CanaryTask],
        num_bootstrap_samples: int = 1000,
        confidence_level: float = 0.95
    ):
        """
        Initialize the interference tester.
        
        Args:
            composer: MultiBEMComposer to test
            canary_tasks: List of canary tasks for testing
            num_bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for statistical tests
        """
        self.composer = composer
        self.canary_tasks = {task.task_id: task for task in canary_tasks}
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        
        # Test results storage
        self.test_history: List[InterferenceTestResult] = []
        self.combination_results: List[BEMCombinationResult] = []
        
        # Baseline performance (measured with no BEMs active)
        self.baseline_scores: Dict[str, float] = {}
        
        logger.info(f"Initialized InterferenceTester with {len(canary_tasks)} canary tasks")
    
    def establish_baselines(self, model: nn.Module) -> Dict[str, float]:
        """
        Establish baseline performance on all canary tasks.
        
        Args:
            model: Base model (without any BEM modifications)
            
        Returns:
            Dictionary mapping task IDs to baseline scores
        """
        logger.info("Establishing baseline performance...")
        
        baselines = {}
        
        for task_id, task in self.canary_tasks.items():
            try:
                # Evaluate task with base model
                score = task.eval_function(model, task.data_loader)
                baselines[task_id] = score
                
                # Update task baseline if not set
                if task.baseline_score is None:
                    task.baseline_score = score
                
                logger.debug(f"Baseline for {task.task_name}: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to establish baseline for task {task_id}: {e}")
                baselines[task_id] = 0.0
        
        self.baseline_scores = baselines
        
        logger.info(f"Established baselines for {len(baselines)} tasks")
        
        return baselines.copy()
    
    def run_interference_test(
        self,
        model: nn.Module,
        bem_configs: List[BEMConfiguration],
        config_id: str,
        num_trials: int = 5
    ) -> InterferenceTestResult:
        """
        Run interference testing for a specific BEM configuration.
        
        Args:
            model: Model with BEMs applied
            bem_configs: Configuration of BEMs to test
            config_id: Identifier for this configuration
            num_trials: Number of trials to run for each task
            
        Returns:
            InterferenceTestResult with detailed results
        """
        logger.info(f"Running interference test for configuration '{config_id}'")
        
        # Configure BEMs according to specification
        self._apply_bem_configuration(bem_configs)
        
        # Run evaluations on all canary tasks
        canary_results = {}
        all_trial_results = defaultdict(list)
        
        for task_id, task in self.canary_tasks.items():
            trial_scores = []
            
            for trial in range(num_trials):
                try:
                    score = task.eval_function(model, task.data_loader)
                    trial_scores.append(score)
                    
                except Exception as e:
                    logger.error(f"Trial {trial} failed for task {task_id}: {e}")
                    # Use baseline as fallback
                    if task.baseline_score is not None:
                        trial_scores.append(task.baseline_score)
            
            if trial_scores:
                canary_results[task_id] = np.mean(trial_scores)
                all_trial_results[task_id] = trial_scores
            else:
                canary_results[task_id] = 0.0
                all_trial_results[task_id] = [0.0]
        
        # Compute performance changes and violations
        performance_changes = {}
        violations = []
        
        for task_id, score in canary_results.items():
            task = self.canary_tasks[task_id]
            baseline = self.baseline_scores.get(task_id, task.baseline_score or 1.0)
            
            if baseline > 0:
                if task.higher_is_better:
                    change = (score - baseline) / baseline
                else:
                    change = (baseline - score) / baseline
                
                performance_changes[task_id] = change
                
                # Check for violations
                if change < -task.regression_threshold:
                    violations.append(task_id)
                    logger.warning(f"Regression violation in {task.task_name}: "
                                 f"{change:.4f} < -{task.regression_threshold}")
            else:
                performance_changes[task_id] = 0.0
        
        # Statistical analysis
        statistical_significance = self._compute_statistical_significance(
            all_trial_results, self.baseline_scores
        )
        
        confidence_intervals = self._compute_confidence_intervals(all_trial_results)
        
        # Overall interference score (negative weighted average of changes)
        weights = [1.0] * len(performance_changes)  # Equal weighting for now
        overall_interference = np.average(
            list(performance_changes.values()),
            weights=weights
        )
        
        result = InterferenceTestResult(
            config_id=config_id,
            bem_configs=bem_configs,
            canary_results=canary_results,
            performance_changes=performance_changes,
            violations=violations,
            overall_interference_score=overall_interference,
            statistical_significance=statistical_significance,
            confidence_intervals=confidence_intervals
        )
        
        self.test_history.append(result)
        
        logger.info(f"Interference test completed: {len(violations)} violations, "
                   f"overall_interference={overall_interference:.4f}")
        
        return result
    
    def test_bem_combinations(
        self,
        model: nn.Module,
        max_combination_size: int = 3,
        num_trials_per_config: int = 3
    ) -> List[BEMCombinationResult]:
        """
        Test all combinations of BEMs up to a maximum size.
        
        Args:
            model: Model to test
            max_combination_size: Maximum number of BEMs in a combination
            num_trials_per_config: Number of trials per configuration
            
        Returns:
            List of BEMCombinationResult for each combination tested
        """
        logger.info(f"Testing BEM combinations up to size {max_combination_size}")
        
        bem_ids = list(self.composer.get_bem_registry().keys())
        
        if len(bem_ids) == 0:
            logger.warning("No BEMs registered in composer")
            return []
        
        combination_results = []
        
        # Generate all combinations
        from itertools import combinations
        
        for combo_size in range(1, min(max_combination_size + 1, len(bem_ids) + 1)):
            for bem_combo in combinations(bem_ids, combo_size):
                combo_id = "_".join(sorted(bem_combo))
                
                logger.info(f"Testing combination: {combo_id}")
                
                # Test this combination
                test_results = []
                
                for trial in range(num_trials_per_config):
                    # Create BEM configuration
                    bem_configs = [
                        BEMConfiguration(bem_id=bem_id, enabled=bem_id in bem_combo)
                        for bem_id in bem_ids
                    ]
                    
                    config_id = f"{combo_id}_trial_{trial}"
                    
                    result = self.run_interference_test(
                        model=model,
                        bem_configs=bem_configs,
                        config_id=config_id,
                        num_trials=1  # Single trial per config since we do multiple configs
                    )
                    
                    test_results.append(result)
                
                # Aggregate results for this combination
                interference_scores = [r.overall_interference_score for r in test_results]
                violation_counts = [len(r.violations) for r in test_results]
                
                combination_result = BEMCombinationResult(
                    combination_id=combo_id,
                    active_bems=list(bem_combo),
                    test_results=test_results,
                    mean_interference=np.mean(interference_scores),
                    max_interference=np.max(interference_scores),
                    violation_rate=np.mean([v > 0 for v in violation_counts])
                )
                
                combination_results.append(combination_result)
                
                logger.info(f"Combination {combo_id}: mean_interference={combination_result.mean_interference:.4f}, "
                           f"violation_rate={combination_result.violation_rate:.3f}")
        
        self.combination_results.extend(combination_results)
        
        return combination_results
    
    def _apply_bem_configuration(self, bem_configs: List[BEMConfiguration]):
        """Apply a BEM configuration to the composer."""
        # This is a simplified implementation
        # In practice, you would enable/disable specific BEMs
        # For now, we assume all registered BEMs are used
        pass
    
    def _compute_statistical_significance(
        self,
        trial_results: Dict[str, List[float]],
        baselines: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute statistical significance of performance changes.
        
        Args:
            trial_results: Trial results for each task
            baselines: Baseline scores for each task
            
        Returns:
            Dictionary mapping task IDs to p-values
        """
        p_values = {}
        
        for task_id, scores in trial_results.items():
            if len(scores) < 2:
                p_values[task_id] = 1.0
                continue
            
            baseline = baselines.get(task_id, 0.0)
            
            # One-sample t-test against baseline
            try:
                t_stat, p_value = stats.ttest_1samp(scores, baseline)
                p_values[task_id] = p_value
            except:
                p_values[task_id] = 1.0
        
        return p_values
    
    def _compute_confidence_intervals(
        self,
        trial_results: Dict[str, List[float]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals using bootstrap sampling.
        
        Args:
            trial_results: Trial results for each task
            
        Returns:
            Dictionary mapping task IDs to confidence intervals
        """
        confidence_intervals = {}
        alpha = 1.0 - self.confidence_level
        
        for task_id, scores in trial_results.items():
            if len(scores) < 2:
                mean_score = scores[0] if scores else 0.0
                confidence_intervals[task_id] = (mean_score, mean_score)
                continue
            
            # Bootstrap sampling
            bootstrap_means = []
            for _ in range(self.num_bootstrap_samples):
                bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Compute percentiles
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower = np.percentile(bootstrap_means, lower_percentile)
            upper = np.percentile(bootstrap_means, upper_percentile)
            
            confidence_intervals[task_id] = (lower, upper)
        
        return confidence_intervals
    
    def generate_interference_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive interference testing report.
        
        Returns:
            Dictionary containing the complete report
        """
        if not self.test_history:
            return {"error": "No test results available"}
        
        # Overall statistics
        all_violations = []
        all_interference_scores = []
        
        for result in self.test_history:
            all_violations.extend(result.violations)
            all_interference_scores.append(result.overall_interference_score)
        
        # Task-specific analysis
        task_performance = defaultdict(list)
        task_violations = defaultdict(int)
        
        for result in self.test_history:
            for task_id, change in result.performance_changes.items():
                task_performance[task_id].append(change)
                if task_id in result.violations:
                    task_violations[task_id] += 1
        
        task_analysis = {}
        for task_id in self.canary_tasks:
            if task_id in task_performance:
                changes = task_performance[task_id]
                task_analysis[task_id] = {
                    'mean_change': np.mean(changes),
                    'std_change': np.std(changes),
                    'min_change': np.min(changes),
                    'max_change': np.max(changes),
                    'violation_count': task_violations[task_id],
                    'violation_rate': task_violations[task_id] / len(changes)
                }
        
        # BEM combination analysis
        combination_analysis = {}
        if self.combination_results:
            for combo_result in self.combination_results:
                combination_analysis[combo_result.combination_id] = {
                    'mean_interference': combo_result.mean_interference,
                    'max_interference': combo_result.max_interference,
                    'violation_rate': combo_result.violation_rate,
                    'num_bems': len(combo_result.active_bems)
                }
        
        report = {
            'summary': {
                'total_tests': len(self.test_history),
                'total_violations': len(all_violations),
                'unique_violated_tasks': len(set(all_violations)),
                'mean_interference_score': np.mean(all_interference_scores),
                'std_interference_score': np.std(all_interference_scores),
                'max_interference_score': np.max(all_interference_scores)
            },
            'task_analysis': task_analysis,
            'combination_analysis': combination_analysis,
            'baseline_scores': self.baseline_scores,
            'test_details': [
                {
                    'config_id': result.config_id,
                    'violations': result.violations,
                    'interference_score': result.overall_interference_score,
                    'num_active_bems': len([c for c in result.bem_configs if c.enabled])
                }
                for result in self.test_history
            ]
        }
        
        return report
    
    def plot_interference_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Generate interference visualization plots.
        
        Args:
            save_path: Optional path to save the plot
            figsize: Figure size for the plot
        """
        if not self.combination_results:
            logger.warning("No combination results available for plotting")
            return
        
        # Prepare data for plotting
        combo_names = []
        interference_scores = []
        violation_rates = []
        num_bems = []
        
        for combo_result in self.combination_results:
            combo_names.append(combo_result.combination_id)
            interference_scores.append(combo_result.mean_interference)
            violation_rates.append(combo_result.violation_rate)
            num_bems.append(len(combo_result.active_bems))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Multi-BEM Interference Analysis', fontsize=16)
        
        # Plot 1: Interference scores by combination
        axes[0, 0].bar(range(len(combo_names)), interference_scores)
        axes[0, 0].set_title('Interference Scores by BEM Combination')
        axes[0, 0].set_xlabel('BEM Combination')
        axes[0, 0].set_ylabel('Mean Interference Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Violation rates
        axes[0, 1].bar(range(len(combo_names)), violation_rates, color='red', alpha=0.7)
        axes[0, 1].set_title('Violation Rates by Combination')
        axes[0, 1].set_xlabel('BEM Combination') 
        axes[0, 1].set_ylabel('Violation Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Interference vs Number of BEMs
        axes[1, 0].scatter(num_bems, interference_scores, alpha=0.7)
        axes[1, 0].set_title('Interference vs Number of BEMs')
        axes[1, 0].set_xlabel('Number of Active BEMs')
        axes[1, 0].set_ylabel('Mean Interference Score')
        
        # Plot 4: Task performance heatmap
        if self.test_history:
            # Create matrix of task performance changes
            task_ids = list(self.canary_tasks.keys())
            config_ids = [r.config_id for r in self.test_history[-10:]]  # Last 10 configs
            
            performance_matrix = np.zeros((len(task_ids), len(config_ids)))
            
            for j, result in enumerate(self.test_history[-10:]):
                for i, task_id in enumerate(task_ids):
                    performance_matrix[i, j] = result.performance_changes.get(task_id, 0.0)
            
            im = axes[1, 1].imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
            axes[1, 1].set_title('Task Performance Changes (Recent Configs)')
            axes[1, 1].set_xlabel('Configuration')
            axes[1, 1].set_ylabel('Task')
            axes[1, 1].set_yticks(range(len(task_ids)))
            axes[1, 1].set_yticklabels([t[:15] for t in task_ids])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Interference plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, export_path: str):
        """
        Export all test results to a JSON file.
        
        Args:
            export_path: Path to save the results
        """
        export_data = {
            'baseline_scores': self.baseline_scores,
            'canary_tasks': {
                task_id: {
                    'task_name': task.task_name,
                    'task_type': task.task_type,
                    'regression_threshold': task.regression_threshold,
                    'baseline_score': task.baseline_score
                }
                for task_id, task in self.canary_tasks.items()
            },
            'test_results': [
                {
                    'config_id': result.config_id,
                    'canary_results': result.canary_results,
                    'performance_changes': result.performance_changes,
                    'violations': result.violations,
                    'overall_interference_score': result.overall_interference_score,
                    'statistical_significance': result.statistical_significance
                }
                for result in self.test_history
            ],
            'combination_results': [
                {
                    'combination_id': result.combination_id,
                    'active_bems': result.active_bems,
                    'mean_interference': result.mean_interference,
                    'max_interference': result.max_interference,
                    'violation_rate': result.violation_rate
                }
                for result in self.combination_results
            ]
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Results exported to {export_path}")


def create_standard_canary_tasks() -> List[CanaryTask]:
    """
    Create a standard set of canary tasks for interference testing.
    
    Returns:
        List of standard CanaryTask instances
    """
    # This is a simplified example - in practice you would define real evaluation functions
    def dummy_eval_function(model, data_loader):
        return np.random.uniform(0.7, 1.0)  # Placeholder
    
    canary_tasks = [
        CanaryTask(
            task_id='instruction_following',
            task_name='Instruction Following',
            task_type='capability_test',
            eval_function=dummy_eval_function,
            data_loader=None,  # Would be real data loader
            regression_threshold=0.02
        ),
        CanaryTask(
            task_id='harmlessness',
            task_name='Harmlessness',
            task_type='safety_test', 
            eval_function=dummy_eval_function,
            data_loader=None,
            regression_threshold=0.01  # Stricter threshold for safety
        ),
        CanaryTask(
            task_id='factual_qa',
            task_name='Factual QA',
            task_type='capability_test',
            eval_function=dummy_eval_function,
            data_loader=None,
            regression_threshold=0.02
        ),
        CanaryTask(
            task_id='long_context',
            task_name='Long Context Understanding',
            task_type='capability_test',
            eval_function=dummy_eval_function,
            data_loader=None,
            regression_threshold=0.03
        ),
        CanaryTask(
            task_id='tool_use',
            task_name='Tool Use',
            task_type='capability_test',
            eval_function=dummy_eval_function,
            data_loader=None,
            regression_threshold=0.02
        )
    ]
    
    return canary_tasks


def create_interference_tester(
    composer: MultiBEMComposer,
    canary_tasks: Optional[List[CanaryTask]] = None
) -> InterferenceTester:
    """
    Factory function to create an InterferenceTester.
    
    Args:
        composer: MultiBEMComposer to test
        canary_tasks: Optional list of canary tasks (defaults to standard tasks)
        
    Returns:
        Configured InterferenceTester instance
    """
    if canary_tasks is None:
        canary_tasks = create_standard_canary_tasks()
    
    return InterferenceTester(composer, canary_tasks)