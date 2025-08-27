"""
Canary Gate for BEM 2.0 Online Learning Safety.

Implements canary testing to validate updates before applying them.
Must pass safety tests before any online update activation as specified in TODO.md.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from enum import Enum
import time
import copy
from pathlib import Path
import json


class CanaryTestType(Enum):
    """Types of canary tests."""
    PERFORMANCE = "performance"
    SAFETY = "safety"
    COHERENCE = "coherence"
    REGRESSION = "regression"
    DRIFT = "drift"
    MEMORY = "memory"


@dataclass
class CanaryTest:
    """Configuration for a single canary test."""
    
    # Test identification
    name: str
    test_type: CanaryTestType
    description: str
    
    # Test function (returns score between 0-1, higher is better)
    test_function: Callable[[nn.Module, Any], float]
    
    # Thresholds
    min_score: float = 0.8  # Minimum passing score
    baseline_score: Optional[float] = None  # Score from baseline model
    max_regression: float = 0.05  # Maximum allowed regression from baseline
    
    # Test configuration
    timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Data requirements
    requires_data: bool = True
    data_loader: Optional[Any] = None
    
    # Metadata
    priority: int = 1  # Higher priority tests run first
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class CanaryResult:
    """Result of a canary test execution."""
    
    # Test information
    test_name: str
    test_type: CanaryTestType
    
    # Results
    passed: bool
    score: float
    baseline_score: Optional[float] = None
    regression: Optional[float] = None
    
    # Execution information
    execution_time: float = 0.0
    retries_used: int = 0
    error_message: Optional[str] = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class CanaryGate:
    """
    Canary gate for validating online updates before activation.
    
    Implements comprehensive safety testing including:
    - Performance regression tests
    - Safety behavior validation
    - Response coherence checks
    - Memory usage validation
    - Drift detection
    
    Updates are only applied if ALL canary tests pass.
    """
    
    def __init__(self, tests: List[CanaryTest]):
        self.tests = {test.name: test for test in tests}
        self.logger = logging.getLogger(__name__)
        
        # Execution history
        self.test_history: List[List[CanaryResult]] = []
        self.total_executions = 0
        self.total_passed = 0
        
        # Performance tracking
        self.baseline_scores: Dict[str, float] = {}
        self.recent_scores: Dict[str, List[float]] = {}
        
        self.logger.info(f"CanaryGate initialized with {len(self.tests)} tests")
    
    def add_test(self, test: CanaryTest):
        """Add a new canary test."""
        self.tests[test.name] = test
        self.logger.info(f"Added canary test: {test.name}")
    
    def remove_test(self, test_name: str):
        """Remove a canary test."""
        if test_name in self.tests:
            del self.tests[test_name]
            self.logger.info(f"Removed canary test: {test_name}")
    
    def enable_test(self, test_name: str, enabled: bool = True):
        """Enable or disable a specific test."""
        if test_name in self.tests:
            self.tests[test_name].enabled = enabled
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Test {test_name} {status}")
    
    def set_baseline(self, model: nn.Module, data_loader: Any = None):
        """Establish baseline scores for regression detection."""
        self.logger.info("Establishing canary baseline scores...")
        
        for test_name, test in self.tests.items():
            if not test.enabled:
                continue
            
            try:
                # Use test's data loader or provided one
                test_data = data_loader if test.data_loader is None else test.data_loader
                
                # Run test on baseline model
                baseline_score = self._execute_single_test(test, model, test_data)
                
                # Store baseline
                self.baseline_scores[test_name] = baseline_score
                test.baseline_score = baseline_score
                
                self.logger.info(f"Baseline for {test_name}: {baseline_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to establish baseline for {test_name}: {e}")
        
        self.logger.info("Baseline establishment complete")
    
    def run_canaries(
        self,
        model: nn.Module,
        data_loader: Any = None,
        early_stop: bool = True
    ) -> Tuple[bool, List[CanaryResult]]:
        """
        Run all canary tests on the model.
        
        Args:
            model: Model to test
            data_loader: Data for testing (if not specified in individual tests)
            early_stop: Stop on first failure
            
        Returns:
            (all_passed, results) tuple
        """
        self.logger.info("Running canary tests...")
        start_time = time.time()
        
        results = []
        all_passed = True
        
        # Sort tests by priority (higher priority first)
        sorted_tests = sorted(
            [(name, test) for name, test in self.tests.items() if test.enabled],
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        for test_name, test in sorted_tests:
            try:
                # Use test's data loader or provided one
                test_data = data_loader if test.data_loader is None else test.data_loader
                
                # Execute test with retries
                result = self._execute_test_with_retries(test, model, test_data)
                results.append(result)
                
                # Check if test passed
                if not result.passed:
                    all_passed = False
                    self.logger.warning(f"Canary test FAILED: {test_name} (score: {result.score:.4f})")
                    
                    if early_stop:
                        self.logger.info("Early stopping due to canary failure")
                        break
                else:
                    self.logger.debug(f"Canary test PASSED: {test_name} (score: {result.score:.4f})")
                
                # Track recent scores
                if test_name not in self.recent_scores:
                    self.recent_scores[test_name] = []
                self.recent_scores[test_name].append(result.score)
                if len(self.recent_scores[test_name]) > 100:  # Keep last 100 scores
                    self.recent_scores[test_name].pop(0)
                
            except Exception as e:
                # Create error result
                error_result = CanaryResult(
                    test_name=test_name,
                    test_type=test.test_type,
                    passed=False,
                    score=0.0,
                    error_message=str(e),
                    execution_time=0.0
                )
                results.append(error_result)
                all_passed = False
                
                self.logger.error(f"Canary test ERROR: {test_name} - {e}")
                
                if early_stop:
                    break
        
        # Update statistics
        self.test_history.append(results)
        self.total_executions += 1
        if all_passed:
            self.total_passed += 1
        
        execution_time = time.time() - start_time
        status = "PASSED" if all_passed else "FAILED"
        
        self.logger.info(f"Canary tests {status} in {execution_time:.2f}s "
                        f"({len(results)}/{len(sorted_tests)} tests executed)")
        
        return all_passed, results
    
    def _execute_test_with_retries(
        self,
        test: CanaryTest,
        model: nn.Module,
        data_loader: Any
    ) -> CanaryResult:
        """Execute a single test with retry logic."""
        retries_used = 0
        last_error = None
        
        while retries_used <= test.max_retries:
            try:
                start_time = time.time()
                
                # Execute the test
                score = self._execute_single_test(test, model, data_loader)
                execution_time = time.time() - start_time
                
                # Check against thresholds
                passed = score >= test.min_score
                regression = None
                
                # Check for regression if baseline exists
                if test.baseline_score is not None:
                    regression = test.baseline_score - score
                    if regression > test.max_regression:
                        passed = False
                
                return CanaryResult(
                    test_name=test.name,
                    test_type=test.test_type,
                    passed=passed,
                    score=score,
                    baseline_score=test.baseline_score,
                    regression=regression,
                    execution_time=execution_time,
                    retries_used=retries_used
                )
                
            except Exception as e:
                last_error = e
                retries_used += 1
                
                if retries_used <= test.max_retries:
                    self.logger.warning(f"Test {test.name} failed, retry {retries_used}/{test.max_retries}: {e}")
                    time.sleep(1.0)  # Brief pause before retry
        
        # All retries exhausted
        return CanaryResult(
            test_name=test.name,
            test_type=test.test_type,
            passed=False,
            score=0.0,
            baseline_score=test.baseline_score,
            execution_time=0.0,
            retries_used=retries_used,
            error_message=f"Failed after {test.max_retries} retries: {last_error}"
        )
    
    def _execute_single_test(
        self,
        test: CanaryTest,
        model: nn.Module,
        data_loader: Any
    ) -> float:
        """Execute a single test and return score."""
        # Timeout handling
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        try:
            # Execute test function
            if test.requires_data and data_loader is None:
                raise ValueError(f"Test {test.name} requires data but none provided")
            
            score = test.test_function(model, data_loader)
            
            # Validate score
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                raise ValueError(f"Test {test.name} returned invalid score: {score}")
            
            execution_time = time.time() - start_time
            
            # Check timeout
            if execution_time > test.timeout_seconds:
                raise TimeoutError(f"Test {test.name} exceeded timeout of {test.timeout_seconds}s")
            
            return float(score)
            
        except Exception as e:
            self.logger.error(f"Error executing test {test.name}: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get canary gate statistics."""
        enabled_tests = [name for name, test in self.tests.items() if test.enabled]
        disabled_tests = [name for name, test in self.tests.items() if not test.enabled]
        
        # Recent performance statistics
        recent_stats = {}
        for test_name, scores in self.recent_scores.items():
            if scores:
                recent_stats[test_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'count': len(scores)
                }
        
        # Pass rate by test type
        type_stats = {}
        if self.test_history:
            for test_type in CanaryTestType:
                test_results = []
                for execution_results in self.test_history:
                    test_results.extend([r for r in execution_results if r.test_type == test_type])
                
                if test_results:
                    passed_count = sum(1 for r in test_results if r.passed)
                    type_stats[test_type.value] = {
                        'total': len(test_results),
                        'passed': passed_count,
                        'pass_rate': passed_count / len(test_results)
                    }
        
        return {
            'total_tests': len(self.tests),
            'enabled_tests': len(enabled_tests),
            'disabled_tests': len(disabled_tests),
            'test_names': enabled_tests,
            'total_executions': self.total_executions,
            'total_passed': self.total_passed,
            'overall_pass_rate': self.total_passed / max(1, self.total_executions),
            'baseline_scores': self.baseline_scores.copy(),
            'recent_performance': recent_stats,
            'pass_rate_by_type': type_stats
        }
    
    def save_config(self, filepath: str):
        """Save canary configuration (without test functions)."""
        config_data = {
            'tests': [
                {
                    'name': test.name,
                    'test_type': test.test_type.value,
                    'description': test.description,
                    'min_score': test.min_score,
                    'baseline_score': test.baseline_score,
                    'max_regression': test.max_regression,
                    'timeout_seconds': test.timeout_seconds,
                    'max_retries': test.max_retries,
                    'requires_data': test.requires_data,
                    'priority': test.priority,
                    'enabled': test.enabled,
                    'tags': test.tags
                }
                for test in self.tests.values()
            ],
            'baseline_scores': self.baseline_scores,
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Canary configuration saved to {filepath}")


# Predefined canary tests
def create_performance_test(
    name: str = "performance_regression",
    min_score: float = 0.85,
    max_regression: float = 0.05
) -> CanaryTest:
    """Create performance regression test."""
    
    def test_function(model: nn.Module, data_loader) -> float:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch['inputs'], batch['targets']
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Limit evaluation size for speed
                if total >= 1000:
                    break
        
        return correct / total if total > 0 else 0.0
    
    return CanaryTest(
        name=name,
        test_type=CanaryTestType.PERFORMANCE,
        description="Check for performance regression",
        test_function=test_function,
        min_score=min_score,
        max_regression=max_regression,
        timeout_seconds=60.0,
        priority=5
    )


def create_coherence_test(
    name: str = "response_coherence",
    min_score: float = 0.8
) -> CanaryTest:
    """Create response coherence test."""
    
    def test_function(model: nn.Module, data_loader) -> float:
        model.eval()
        coherence_scores = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= 10:  # Limit samples for speed
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch['inputs']
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                outputs = model(inputs)
                
                # Simple coherence check: entropy should not be too high
                probs = torch.softmax(outputs, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                
                # Lower entropy means more coherent/confident predictions
                coherence_score = torch.exp(-entropy.mean()).item()
                coherence_scores.append(coherence_score)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    return CanaryTest(
        name=name,
        test_type=CanaryTestType.COHERENCE,
        description="Check response coherence and confidence",
        test_function=test_function,
        min_score=min_score,
        timeout_seconds=30.0,
        priority=3
    )


def create_memory_test(
    name: str = "memory_usage",
    max_memory_gb: float = 16.0
) -> CanaryTest:
    """Create memory usage test."""
    
    def test_function(model: nn.Module, data_loader) -> float:
        if not torch.cuda.is_available():
            return 1.0  # Pass if no GPU
        
        # Measure memory usage
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # Run a forward pass
        model.eval()
        with torch.no_grad():
            batch = next(iter(data_loader))
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch['inputs']
            
            inputs = inputs.cuda()
            outputs = model(inputs)
        
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        memory_used = memory_after - memory_before
        
        # Score based on memory usage (1.0 if under limit, decreasing linearly)
        score = max(0.0, 1.0 - (memory_used / max_memory_gb))
        return score
    
    return CanaryTest(
        name=name,
        test_type=CanaryTestType.MEMORY,
        description=f"Check memory usage under {max_memory_gb}GB",
        test_function=test_function,
        min_score=0.5,  # Allow up to 50% of limit
        timeout_seconds=15.0,
        priority=2
    )


def create_drift_test(
    name: str = "parameter_drift",
    max_drift: float = 1.0
) -> CanaryTest:
    """Create parameter drift test."""
    
    # Store reference parameters on first call
    _reference_params = {}
    
    def test_function(model: nn.Module, data_loader) -> float:
        nonlocal _reference_params
        
        # Initialize reference on first call
        if not _reference_params:
            _reference_params = {
                name: param.data.clone().detach()
                for name, param in model.named_parameters()
            }
            return 1.0  # Pass on first call
        
        # Compute drift from reference
        total_drift = 0.0
        total_params = 0
        
        for name, param in model.named_parameters():
            if name in _reference_params:
                drift = torch.norm(param.data - _reference_params[name]).item()
                total_drift += drift
                total_params += 1
        
        avg_drift = total_drift / max(1, total_params)
        
        # Score based on drift (1.0 if no drift, decreasing with drift)
        score = max(0.0, 1.0 - (avg_drift / max_drift))
        return score
    
    return CanaryTest(
        name=name,
        test_type=CanaryTestType.DRIFT,
        description=f"Check parameter drift under {max_drift}",
        test_function=test_function,
        min_score=0.7,
        timeout_seconds=10.0,
        priority=4,
        requires_data=False
    )


# Utility function
def create_default_canary_gate() -> CanaryGate:
    """Create canary gate with default tests."""
    tests = [
        create_performance_test(),
        create_coherence_test(),
        create_memory_test(),
        create_drift_test()
    ]
    
    return CanaryGate(tests)


# Example usage
if __name__ == "__main__":
    # Create model and dummy data
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create dummy dataset
    import torch.utils.data as data
    X = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataset = data.TensorDataset(X, y)
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Create canary gate
    gate = create_default_canary_gate()
    
    # Set baseline
    gate.set_baseline(model, data_loader)
    
    # Run canaries
    passed, results = gate.run_canaries(model, data_loader)
    
    print(f"Canaries passed: {passed}")
    for result in results:
        print(f"  {result.test_name}: {'PASS' if result.passed else 'FAIL'} "
              f"(score: {result.score:.3f})")
    
    # Get statistics
    stats = gate.get_statistics()
    print(f"\nStatistics: {stats['overall_pass_rate']:.1%} pass rate")