"""Base evaluation framework for BEM components.

Provides the abstract base evaluator and evaluation utilities that all
BEM components inherit from for standardized evaluation protocols.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import MetricCalculator
from ..config.base_config import BaseConfig
from ..utils.logging_utils import setup_logger


@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for evaluation parameters."""
    
    # Evaluation setup
    batch_size: int = 32
    max_eval_samples: Optional[int] = None
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "f1_score", "exact_match", "perplexity"
    ])
    
    # Generation parameters (for generative tasks)
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True
    
    # Performance evaluation
    measure_latency: bool = True
    measure_memory: bool = True
    num_warmup_steps: int = 10
    
    # Output configuration
    save_predictions: bool = False
    save_detailed_results: bool = False
    output_file: Optional[str] = None
    
    # Statistical validation
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    bootstrap_samples: int = 1000


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    latency_stats: Dict[str, float] = field(default_factory=dict)
    memory_stats: Dict[str, float] = field(default_factory=dict)
    
    # Statistical information
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Sample information
    num_samples: int = 0
    evaluation_time: float = 0.0
    
    # Optional detailed results
    predictions: Optional[List[Any]] = None
    targets: Optional[List[Any]] = None
    detailed_metrics: Optional[Dict[str, List[float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "metrics": self.metrics,
            "latency_stats": self.latency_stats,
            "memory_stats": self.memory_stats,
            "confidence_intervals": self.confidence_intervals,
            "statistical_tests": self.statistical_tests,
            "num_samples": self.num_samples,
            "evaluation_time": self.evaluation_time,
        }
        
        if self.predictions is not None:
            result["predictions"] = self.predictions
        if self.targets is not None:
            result["targets"] = self.targets
        if self.detailed_metrics is not None:
            result["detailed_metrics"] = self.detailed_metrics
        
        return result
    
    def summary_string(self) -> str:
        """Generate summary string of key metrics."""
        lines = []
        lines.append(f"Evaluation Summary ({self.num_samples} samples, {self.evaluation_time:.2f}s)")
        lines.append("=" * 60)
        
        # Core metrics
        for metric, value in sorted(self.metrics.items()):
            ci_str = ""
            if metric in self.confidence_intervals:
                ci_low, ci_high = self.confidence_intervals[metric]
                ci_str = f" [95% CI: {ci_low:.3f}-{ci_high:.3f}]"
            lines.append(f"{metric}: {value:.4f}{ci_str}")
        
        # Performance metrics
        if self.latency_stats:
            lines.append("\nLatency Statistics:")
            for stat, value in self.latency_stats.items():
                lines.append(f"  {stat}: {value:.4f}ms")
        
        if self.memory_stats:
            lines.append("\nMemory Statistics:")
            for stat, value in self.memory_stats.items():
                lines.append(f"  {stat}: {value:.2f}MB")
        
        return "\n".join(lines)


class BaseEvaluator(ABC):
    """Abstract base evaluator for all BEM components.
    
    Provides standardized evaluation infrastructure including:
    - Metric calculation and aggregation
    - Performance measurement (latency, memory)
    - Statistical validation and confidence intervals
    - Results saving and reporting
    
    Subclasses must implement:
    - _run_inference(): Run model inference on a batch
    - _compute_component_metrics(): Compute component-specific metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Union[EvaluationConfig, Dict[str, Any]],
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the base evaluator.
        
        Args:
            model: Model to evaluate
            config: Evaluation configuration
            device: Device for evaluation
            logger: Optional logger instance
        """
        self.model = model
        self.config = self._load_config(config)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger = logger or setup_logger(
            name=f"bem_evaluator_{self.__class__.__name__}",
            level=logging.INFO
        )
        
        # Initialize metric calculator
        self.metric_calculator = MetricCalculator()
        
        # Move model to device
        self.model.to(self.device)
        
        self.logger.info(f"Initialized {self.__class__.__name__} on {self.device}")
    
    def _load_config(self, config: Union[EvaluationConfig, Dict[str, Any]]) -> EvaluationConfig:
        """Load and validate evaluation configuration."""
        if isinstance(config, EvaluationConfig):
            return config
        elif isinstance(config, dict):
            return EvaluationConfig(**config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    @abstractmethod
    def _run_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run model inference on a batch.
        
        Args:
            batch: Input batch data
            
        Returns:
            Model outputs and predictions
        """
        pass
    
    @abstractmethod
    def _compute_component_metrics(
        self, 
        predictions: List[Any], 
        targets: List[Any]
    ) -> Dict[str, float]:
        """Compute component-specific metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed metrics
        """
        pass
    
    def evaluate(self, dataloader: DataLoader) -> EvaluationResult:
        """Run complete evaluation on the given dataloader.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Complete evaluation results
        """
        self.logger.info(f"Starting evaluation on {len(dataloader)} batches")
        start_time = time.time()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize tracking
        all_predictions = []
        all_targets = []
        latencies = []
        memory_usage = []
        
        # Warmup (if measuring performance)
        if self.config.measure_latency and self.config.num_warmup_steps > 0:
            self._warmup(dataloader)
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Limit evaluation samples
                if (self.config.max_eval_samples and 
                    batch_idx * dataloader.batch_size >= self.config.max_eval_samples):
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Measure memory before inference
                if self.config.measure_memory and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    memory_before = torch.cuda.memory_allocated() / 1024**2
                
                # Run inference with latency measurement
                if self.config.measure_latency:
                    batch_start = time.time()
                    outputs = self._run_inference(batch)
                    batch_latency = (time.time() - batch_start) * 1000  # Convert to ms
                    latencies.append(batch_latency)
                else:
                    outputs = self._run_inference(batch)
                
                # Measure memory after inference
                if self.config.measure_memory and torch.cuda.is_available():
                    memory_peak = torch.cuda.max_memory_allocated() / 1024**2
                    memory_usage.append(memory_peak - memory_before)
                
                # Collect predictions and targets
                predictions = outputs.get("predictions", [])
                targets = batch.get("labels", batch.get("targets", []))
                
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets)
                
                # Log progress
                if batch_idx % 100 == 0:
                    self.logger.info(f"Processed {batch_idx}/{len(dataloader)} batches")
        
        # Compute metrics
        self.logger.info("Computing metrics...")
        metrics = self._compute_all_metrics(all_predictions, all_targets)
        
        # Performance statistics
        latency_stats = self._compute_latency_stats(latencies) if latencies else {}
        memory_stats = self._compute_memory_stats(memory_usage) if memory_usage else {}
        
        # Statistical validation
        confidence_intervals = self._compute_confidence_intervals(all_predictions, all_targets)
        
        # Create result
        result = EvaluationResult(
            metrics=metrics,
            latency_stats=latency_stats,
            memory_stats=memory_stats,
            confidence_intervals=confidence_intervals,
            num_samples=len(all_predictions),
            evaluation_time=time.time() - start_time,
            predictions=all_predictions if self.config.save_predictions else None,
            targets=all_targets if self.config.save_predictions else None,
        )
        
        # Save results if requested
        if self.config.output_file:
            self._save_results(result)
        
        self.logger.info(f"Evaluation completed in {result.evaluation_time:.2f}s")
        self.logger.info(f"Results: {result.summary_string()}")
        
        return result
    
    def _warmup(self, dataloader: DataLoader) -> None:
        """Run warmup iterations for performance measurement."""
        self.logger.info(f"Running {self.config.num_warmup_steps} warmup steps...")
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.config.num_warmup_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Run inference
                self._run_inference(batch)
    
    def _compute_all_metrics(
        self, 
        predictions: List[Any], 
        targets: List[Any]
    ) -> Dict[str, float]:
        """Compute all requested metrics."""
        metrics = {}
        
        # Compute standard metrics
        for metric_name in self.config.metrics:
            if hasattr(self.metric_calculator, f"compute_{metric_name}"):
                metric_func = getattr(self.metric_calculator, f"compute_{metric_name}")
                try:
                    metrics[metric_name] = metric_func(predictions, targets)
                except Exception as e:
                    self.logger.warning(f"Failed to compute {metric_name}: {e}")
        
        # Compute component-specific metrics
        component_metrics = self._compute_component_metrics(predictions, targets)
        metrics.update(component_metrics)
        
        return metrics
    
    def _compute_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Compute latency statistics."""
        import numpy as np
        
        latencies_array = np.array(latencies)
        
        return {
            "mean": float(np.mean(latencies_array)),
            "std": float(np.std(latencies_array)),
            "min": float(np.min(latencies_array)),
            "max": float(np.max(latencies_array)),
            "p50": float(np.percentile(latencies_array, 50)),
            "p95": float(np.percentile(latencies_array, 95)),
            "p99": float(np.percentile(latencies_array, 99)),
        }
    
    def _compute_memory_stats(self, memory_usage: List[float]) -> Dict[str, float]:
        """Compute memory usage statistics."""
        import numpy as np
        
        memory_array = np.array(memory_usage)
        
        return {
            "mean": float(np.mean(memory_array)),
            "std": float(np.std(memory_array)),
            "min": float(np.min(memory_array)),
            "max": float(np.max(memory_array)),
            "peak": float(np.max(memory_array)),
        }
    
    def _compute_confidence_intervals(
        self, 
        predictions: List[Any], 
        targets: List[Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for metrics using bootstrap."""
        import numpy as np
        from sklearn.utils import resample
        
        confidence_intervals = {}
        
        for metric_name in self.config.metrics:
            if not hasattr(self.metric_calculator, f"compute_{metric_name}"):
                continue
            
            metric_func = getattr(self.metric_calculator, f"compute_{metric_name}")
            bootstrap_scores = []
            
            try:
                # Bootstrap resampling
                for _ in range(self.config.bootstrap_samples):
                    # Resample with replacement
                    indices = resample(range(len(predictions)), n_samples=len(predictions))
                    boot_pred = [predictions[i] for i in indices]
                    boot_targets = [targets[i] for i in indices]
                    
                    # Compute metric on bootstrap sample
                    score = metric_func(boot_pred, boot_targets)
                    bootstrap_scores.append(score)
                
                # Compute confidence interval
                alpha = 1 - self.config.confidence_level
                lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
                upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
                
                confidence_intervals[metric_name] = (float(lower), float(upper))
                
            except Exception as e:
                self.logger.warning(f"Failed to compute CI for {metric_name}: {e}")
        
        return confidence_intervals
    
    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to file."""
        import json
        
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def compare_with_baseline(
        self, 
        current_result: EvaluationResult,
        baseline_result: EvaluationResult
    ) -> Dict[str, Dict[str, float]]:
        """Compare current results with baseline.
        
        Args:
            current_result: Current evaluation results
            baseline_result: Baseline evaluation results
            
        Returns:
            Comparison statistics and significance tests
        """
        comparison = {}
        
        for metric_name in current_result.metrics:
            if metric_name not in baseline_result.metrics:
                continue
            
            current_value = current_result.metrics[metric_name]
            baseline_value = baseline_result.metrics[metric_name]
            
            # Compute relative and absolute differences
            abs_diff = current_value - baseline_value
            rel_diff = abs_diff / baseline_value if baseline_value != 0 else float('inf')
            
            comparison[metric_name] = {
                "current": current_value,
                "baseline": baseline_value,
                "absolute_diff": abs_diff,
                "relative_diff": rel_diff,
                "percent_change": rel_diff * 100,
            }
        
        return comparison