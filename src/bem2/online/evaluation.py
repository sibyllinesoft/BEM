"""
Evaluation component for BEM 2.0 Online Learning System.

This module provides comprehensive evaluation capabilities for the 24-hour soak test
and tracking the +≥1% aggregate improvement goal as specified in TODO.md.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from enum import Enum
import logging
import numpy as np
import torch
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

from .interfaces import OnlineUpdateResult, SafetyStatus, LearningPhase


class EvaluationPhase(Enum):
    """Evaluation phases for online learning"""
    BASELINE = "baseline"
    WARMUP = "warmup" 
    ACTIVE_LEARNING = "active_learning"
    SOAK_TEST = "soak_test"
    VALIDATION = "validation"


class MetricType(Enum):
    """Types of metrics tracked during evaluation"""
    PERFORMANCE = "performance"
    SAFETY = "safety"
    STABILITY = "stability"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"


@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for online learning evaluation"""
    
    # Performance Metrics (for +≥1% improvement goal)
    task_success_rate: float = 0.0
    average_response_quality: float = 0.0
    user_satisfaction_score: float = 0.0
    tool_success_rate: float = 0.0
    
    # Safety Metrics (for 24h soak with no canary regressions)
    canary_pass_rate: float = 1.0
    safety_violations: int = 0
    rollback_count: int = 0
    drift_warnings: int = 0
    
    # Stability Metrics
    kl_divergence_trend: float = 0.0
    parameter_norm_stability: float = 1.0
    gradient_norm_stability: float = 1.0
    
    # Learning Efficiency
    updates_applied: int = 0
    updates_rejected: int = 0
    learning_rate_adjustments: int = 0
    
    # Quality Metrics
    coherence_score: float = 1.0
    consistency_score: float = 1.0
    knowledge_retention: float = 1.0
    
    # Timing Metrics
    average_update_time: float = 0.0
    total_evaluation_time: float = 0.0
    
    def compute_aggregate_improvement(self, baseline_metrics: 'EvaluationMetrics') -> float:
        """
        Compute aggregate improvement over baseline for +≥1% goal.
        
        Args:
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            Percentage improvement (positive = improvement)
        """
        # Weight different metrics for aggregate score
        weights = {
            'task_success_rate': 0.3,
            'average_response_quality': 0.25,
            'user_satisfaction_score': 0.2,
            'tool_success_rate': 0.15,
            'coherence_score': 0.1
        }
        
        current_weighted = (
            self.task_success_rate * weights['task_success_rate'] +
            self.average_response_quality * weights['average_response_quality'] +
            self.user_satisfaction_score * weights['user_satisfaction_score'] +
            self.tool_success_rate * weights['tool_success_rate'] +
            self.coherence_score * weights['coherence_score']
        )
        
        baseline_weighted = (
            baseline_metrics.task_success_rate * weights['task_success_rate'] +
            baseline_metrics.average_response_quality * weights['average_response_quality'] +
            baseline_metrics.user_satisfaction_score * weights['user_satisfaction_score'] +
            baseline_metrics.tool_success_rate * weights['tool_success_rate'] +
            baseline_metrics.coherence_score * weights['coherence_score']
        )
        
        if baseline_weighted == 0:
            return 0.0
            
        improvement = ((current_weighted - baseline_weighted) / baseline_weighted) * 100
        return improvement
    
    def has_canary_regressions(self, baseline_metrics: 'EvaluationMetrics') -> bool:
        """Check if there are any canary regressions vs baseline"""
        return (
            self.canary_pass_rate < baseline_metrics.canary_pass_rate * 0.95 or  # 5% degradation threshold
            self.safety_violations > baseline_metrics.safety_violations or
            self.coherence_score < baseline_metrics.coherence_score * 0.95
        )


class PerformanceWindow:
    """Sliding window for performance tracking"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_average(self, time_window_hours: Optional[float] = None) -> float:
        if not self.values:
            return 0.0
        
        if time_window_hours is None:
            return np.mean(self.values)
        
        # Filter to time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_values = [
            val for val, ts in zip(self.values, self.timestamps)
            if ts >= cutoff_time
        ]
        
        return np.mean(recent_values) if recent_values else 0.0
    
    def get_trend(self) -> float:
        """Get trend slope (positive = improving, negative = degrading)"""
        if len(self.values) < 2:
            return 0.0
        
        values_array = np.array(self.values)
        x = np.arange(len(values_array))
        slope, _ = np.polyfit(x, values_array, 1)
        return slope


@dataclass
class PerformanceTracker:
    """Tracks performance metrics over time with sliding windows"""
    
    # Performance windows for different metrics
    task_success_window: PerformanceWindow = field(default_factory=lambda: PerformanceWindow(1000))
    quality_window: PerformanceWindow = field(default_factory=lambda: PerformanceWindow(1000))
    satisfaction_window: PerformanceWindow = field(default_factory=lambda: PerformanceWindow(1000))
    tool_success_window: PerformanceWindow = field(default_factory=lambda: PerformanceWindow(1000))
    
    # Safety windows
    canary_pass_window: PerformanceWindow = field(default_factory=lambda: PerformanceWindow(1000))
    safety_window: PerformanceWindow = field(default_factory=lambda: PerformanceWindow(1000))
    
    # Stability windows
    kl_divergence_window: PerformanceWindow = field(default_factory=lambda: PerformanceWindow(1000))
    param_norm_window: PerformanceWindow = field(default_factory=lambda: PerformanceWindow(1000))
    
    def update_from_result(self, result: OnlineUpdateResult, timestamp: Optional[datetime] = None):
        """Update all tracking windows from an online update result"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update performance metrics if available
        if hasattr(result, 'task_success_rate') and result.task_success_rate is not None:
            self.task_success_window.add_value(result.task_success_rate, timestamp)
        
        if hasattr(result, 'quality_score') and result.quality_score is not None:
            self.quality_window.add_value(result.quality_score, timestamp)
        
        # Update safety metrics
        canary_success = 1.0 if result.safety_status == SafetyStatus.SAFE else 0.0
        self.canary_pass_window.add_value(canary_success, timestamp)
        
        safety_success = 1.0 if result.safety_status != SafetyStatus.CRITICAL else 0.0
        self.safety_window.add_value(safety_success, timestamp)
        
        # Update stability metrics
        if hasattr(result, 'kl_divergence') and result.kl_divergence is not None:
            self.kl_divergence_window.add_value(result.kl_divergence, timestamp)
    
    def get_current_metrics(self, time_window_hours: float = 1.0) -> EvaluationMetrics:
        """Get current evaluation metrics over specified time window"""
        return EvaluationMetrics(
            task_success_rate=self.task_success_window.get_average(time_window_hours),
            average_response_quality=self.quality_window.get_average(time_window_hours),
            user_satisfaction_score=self.satisfaction_window.get_average(time_window_hours),
            tool_success_rate=self.tool_success_window.get_average(time_window_hours),
            canary_pass_rate=self.canary_pass_window.get_average(time_window_hours),
            kl_divergence_trend=self.kl_divergence_window.get_trend()
        )
    
    def get_trends(self) -> Dict[str, float]:
        """Get performance trends (positive = improving)"""
        return {
            'task_success_trend': self.task_success_window.get_trend(),
            'quality_trend': self.quality_window.get_trend(),
            'satisfaction_trend': self.satisfaction_window.get_trend(),
            'tool_success_trend': self.tool_success_window.get_trend(),
            'canary_pass_trend': self.canary_pass_window.get_trend(),
            'safety_trend': self.safety_window.get_trend(),
            'kl_stability_trend': -self.kl_divergence_window.get_trend()  # Negative KL trend is good
        }


class SoakTestResult(NamedTuple):
    """Result of 24-hour soak test"""
    duration_hours: float
    baseline_metrics: EvaluationMetrics
    final_metrics: EvaluationMetrics
    aggregate_improvement: float
    has_regressions: bool
    success: bool
    summary: str


@dataclass
class OnlineEvaluator:
    """
    Main evaluator for BEM 2.0 online learning system.
    
    Handles the 24-hour soak test and tracks the +≥1% aggregate improvement goal.
    """
    
    performance_tracker: PerformanceTracker = field(default_factory=PerformanceTracker)
    baseline_metrics: Optional[EvaluationMetrics] = None
    evaluation_start_time: Optional[datetime] = None
    current_phase: EvaluationPhase = EvaluationPhase.BASELINE
    
    # Soak test configuration
    target_soak_hours: float = 24.0
    target_improvement_percent: float = 1.0
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Logging
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    
    def start_evaluation(self, phase: EvaluationPhase = EvaluationPhase.BASELINE):
        """Start evaluation in specified phase"""
        with self._lock:
            self.evaluation_start_time = datetime.now()
            self.current_phase = phase
            self.logger.info(f"Starting evaluation phase: {phase.value}")
    
    def set_baseline_metrics(self, metrics: EvaluationMetrics):
        """Set baseline metrics for comparison"""
        with self._lock:
            self.baseline_metrics = metrics
            self.logger.info(f"Baseline metrics established: "
                           f"Task success: {metrics.task_success_rate:.3f}, "
                           f"Quality: {metrics.average_response_quality:.3f}")
    
    def update_metrics(self, result: OnlineUpdateResult):
        """Update metrics from an online learning result"""
        with self._lock:
            self.performance_tracker.update_from_result(result)
    
    def get_current_metrics(self, time_window_hours: float = 1.0) -> EvaluationMetrics:
        """Get current evaluation metrics"""
        with self._lock:
            return self.performance_tracker.get_current_metrics(time_window_hours)
    
    def get_soak_test_progress(self) -> Dict[str, Any]:
        """Get current progress of soak test"""
        if self.evaluation_start_time is None:
            return {"status": "not_started", "progress": 0.0}
        
        elapsed_time = datetime.now() - self.evaluation_start_time
        elapsed_hours = elapsed_time.total_seconds() / 3600
        progress = min(elapsed_hours / self.target_soak_hours, 1.0)
        
        current_metrics = self.get_current_metrics(time_window_hours=1.0)
        
        # Check improvement vs baseline
        improvement = 0.0
        has_regressions = False
        
        if self.baseline_metrics:
            improvement = current_metrics.compute_aggregate_improvement(self.baseline_metrics)
            has_regressions = current_metrics.has_canary_regressions(self.baseline_metrics)
        
        status = "running"
        if progress >= 1.0:
            if improvement >= self.target_improvement_percent and not has_regressions:
                status = "success"
            else:
                status = "failed"
        elif has_regressions:
            status = "regression_detected"
        
        return {
            "status": status,
            "progress": progress,
            "elapsed_hours": elapsed_hours,
            "remaining_hours": max(0, self.target_soak_hours - elapsed_hours),
            "current_improvement_percent": improvement,
            "target_improvement_percent": self.target_improvement_percent,
            "has_regressions": has_regressions,
            "current_metrics": current_metrics,
            "performance_trends": self.performance_tracker.get_trends()
        }
    
    def run_soak_test(self) -> SoakTestResult:
        """
        Run the 24-hour soak test.
        
        This is typically called in a separate thread/process.
        The actual online learning continues while this monitors progress.
        """
        if self.baseline_metrics is None:
            raise ValueError("Baseline metrics must be set before running soak test")
        
        self.start_evaluation(EvaluationPhase.SOAK_TEST)
        start_time = datetime.now()
        
        self.logger.info(f"Starting {self.target_soak_hours}h soak test...")
        self.logger.info(f"Target: +≥{self.target_improvement_percent}% improvement, no canary regressions")
        
        # Monitor progress
        while True:
            progress = self.get_soak_test_progress()
            
            if progress["status"] in ["success", "failed"]:
                break
            elif progress["status"] == "regression_detected":
                self.logger.warning("Canary regressions detected during soak test!")
                # Continue monitoring - system should auto-rollback
            
            # Log progress periodically
            if progress["elapsed_hours"] % 4 < 0.1:  # Every 4 hours approximately
                self.logger.info(f"Soak test progress: {progress['progress']:.1%}, "
                               f"Improvement: {progress['current_improvement_percent']:+.2f}%")
            
            time.sleep(300)  # Check every 5 minutes
        
        # Calculate final results
        end_time = datetime.now()
        duration = end_time - start_time
        duration_hours = duration.total_seconds() / 3600
        
        final_metrics = self.get_current_metrics(time_window_hours=2.0)  # Last 2 hours average
        aggregate_improvement = final_metrics.compute_aggregate_improvement(self.baseline_metrics)
        has_regressions = final_metrics.has_canary_regressions(self.baseline_metrics)
        
        success = (
            duration_hours >= self.target_soak_hours and
            aggregate_improvement >= self.target_improvement_percent and
            not has_regressions
        )
        
        # Create summary
        if success:
            summary = (f"✅ Soak test PASSED: {duration_hours:.1f}h duration, "
                      f"{aggregate_improvement:+.2f}% improvement, no regressions")
        else:
            reasons = []
            if duration_hours < self.target_soak_hours:
                reasons.append(f"insufficient duration ({duration_hours:.1f}h < {self.target_soak_hours}h)")
            if aggregate_improvement < self.target_improvement_percent:
                reasons.append(f"insufficient improvement ({aggregate_improvement:+.2f}% < +{self.target_improvement_percent}%)")
            if has_regressions:
                reasons.append("canary regressions detected")
            
            summary = f"❌ Soak test FAILED: {', '.join(reasons)}"
        
        self.logger.info(summary)
        
        return SoakTestResult(
            duration_hours=duration_hours,
            baseline_metrics=self.baseline_metrics,
            final_metrics=final_metrics,
            aggregate_improvement=aggregate_improvement,
            has_regressions=has_regressions,
            success=success,
            summary=summary
        )
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if self.evaluation_start_time is None:
            return {"error": "Evaluation not started"}
        
        current_metrics = self.get_current_metrics()
        soak_progress = self.get_soak_test_progress()
        trends = self.performance_tracker.get_trends()
        
        elapsed_time = datetime.now() - self.evaluation_start_time
        
        report = {
            "evaluation_summary": {
                "phase": self.current_phase.value,
                "elapsed_time_hours": elapsed_time.total_seconds() / 3600,
                "start_time": self.evaluation_start_time.isoformat()
            },
            "current_metrics": {
                "performance": {
                    "task_success_rate": current_metrics.task_success_rate,
                    "average_response_quality": current_metrics.average_response_quality,
                    "user_satisfaction_score": current_metrics.user_satisfaction_score,
                    "tool_success_rate": current_metrics.tool_success_rate
                },
                "safety": {
                    "canary_pass_rate": current_metrics.canary_pass_rate,
                    "safety_violations": current_metrics.safety_violations,
                    "rollback_count": current_metrics.rollback_count,
                    "drift_warnings": current_metrics.drift_warnings
                },
                "stability": {
                    "kl_divergence_trend": current_metrics.kl_divergence_trend,
                    "parameter_norm_stability": current_metrics.parameter_norm_stability
                }
            },
            "performance_trends": trends,
            "soak_test_progress": soak_progress
        }
        
        # Add baseline comparison if available
        if self.baseline_metrics:
            improvement = current_metrics.compute_aggregate_improvement(self.baseline_metrics)
            report["baseline_comparison"] = {
                "aggregate_improvement_percent": improvement,
                "target_improvement_percent": self.target_improvement_percent,
                "improvement_goal_met": improvement >= self.target_improvement_percent,
                "has_regressions": current_metrics.has_canary_regressions(self.baseline_metrics)
            }
        
        return report
    
    def export_metrics_history(self) -> Dict[str, List[Tuple[datetime, float]]]:
        """Export complete metrics history for analysis"""
        history = {}
        
        # Export all performance windows
        windows = {
            "task_success": self.performance_tracker.task_success_window,
            "quality": self.performance_tracker.quality_window,
            "satisfaction": self.performance_tracker.satisfaction_window,
            "tool_success": self.performance_tracker.tool_success_window,
            "canary_pass": self.performance_tracker.canary_pass_window,
            "safety": self.performance_tracker.safety_window,
            "kl_divergence": self.performance_tracker.kl_divergence_window,
            "param_norm": self.performance_tracker.param_norm_window
        }
        
        for name, window in windows.items():
            history[name] = list(zip(window.timestamps, window.values))
        
        return history


# Utility functions for evaluation

def run_24hour_soak_test(evaluator: OnlineEvaluator, 
                        baseline_metrics: EvaluationMetrics) -> SoakTestResult:
    """
    Convenience function to run the complete 24-hour soak test.
    
    Args:
        evaluator: The online evaluator instance
        baseline_metrics: Baseline metrics for comparison
        
    Returns:
        SoakTestResult with complete test results
    """
    evaluator.set_baseline_metrics(baseline_metrics)
    return evaluator.run_soak_test()


def compute_learning_efficiency(results: List[OnlineUpdateResult]) -> Dict[str, float]:
    """Compute learning efficiency metrics from update results"""
    if not results:
        return {}
    
    total_updates = len(results)
    successful_updates = sum(1 for r in results if r.update_applied)
    rollbacks = sum(1 for r in results if hasattr(r, 'rollback_triggered') and r.rollback_triggered)
    
    return {
        "update_success_rate": successful_updates / total_updates if total_updates > 0 else 0.0,
        "rollback_rate": rollbacks / total_updates if total_updates > 0 else 0.0,
        "learning_efficiency": successful_updates / max(total_updates, 1)
    }


def validate_soak_test_requirements(result: SoakTestResult) -> Tuple[bool, List[str]]:
    """
    Validate that soak test meets all TODO.md requirements.
    
    Returns:
        (success: bool, violations: List[str])
    """
    violations = []
    
    # Check 24-hour duration
    if result.duration_hours < 24.0:
        violations.append(f"Duration {result.duration_hours:.1f}h < required 24h")
    
    # Check +≥1% improvement
    if result.aggregate_improvement < 1.0:
        violations.append(f"Improvement {result.aggregate_improvement:+.2f}% < required +1.0%")
    
    # Check no canary regressions
    if result.has_regressions:
        violations.append("Canary regressions detected")
    
    return len(violations) == 0, violations