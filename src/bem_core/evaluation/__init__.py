"""Consolidated evaluation framework for BEM experiments.

Provides unified evaluation interfaces, metrics calculation,
and validation protocols across all BEM components.
"""

from .base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult
from .metrics import (
    MetricCalculator,
    compute_accuracy,
    compute_f1_score,
    compute_bleu_score,
    compute_rouge_scores,
    compute_perplexity,
)
from .validation import (
    ValidationProtocol,
    validate_model_outputs,
    check_convergence,
    statistical_significance_test,
)
from .analysis_utils import (
    AnalysisRunner,
    generate_evaluation_report,
    plot_training_curves,
    compare_experiments,
)

__all__ = [
    # Base classes
    "BaseEvaluator",
    "EvaluationConfig", 
    "EvaluationResult",
    # Metrics
    "MetricCalculator",
    "compute_accuracy",
    "compute_f1_score",
    "compute_bleu_score",
    "compute_rouge_scores",
    "compute_perplexity",
    # Validation
    "ValidationProtocol",
    "validate_model_outputs",
    "check_convergence",
    "statistical_significance_test",
    # Analysis
    "AnalysisRunner",
    "generate_evaluation_report",
    "plot_training_curves",
    "compare_experiments",
]