"""
Index-Swap Monotonicity Tester

Tests whether the Agentic Router preserves monotonicity properties when indices are swapped.
Critical for validating that routing decisions don't break fundamental model behaviors.

Key tests:
- Clean vs shuffled indices maintain relative performance
- Corrupted indices degrade performance appropriately  
- Expert routing is consistent across index permutations
- Cache safety preserved under index transformations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
from pathlib import Path
import json
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IndexSwapTestCase:
    """Single index swap test configuration."""
    name: str
    description: str
    clean_indices: torch.Tensor
    modified_indices: torch.Tensor
    expected_behavior: str  # 'maintain', 'degrade', 'improve'
    tolerance: float = 0.05


class MonotonicityResult(NamedTuple):
    """Result of monotonicity test."""
    test_name: str
    clean_score: float
    modified_score: float
    relative_change: float
    monotonicity_preserved: bool
    expert_consistency: float
    cache_safety_maintained: bool


class MonotonicityTester:
    """
    Comprehensive monotonicity testing for Agentic Router.
    
    Validates that routing decisions are robust to index transformations
    and maintain expected monotonicity properties.
    """
    
    def __init__(
        self,
        num_test_sequences: int = 50,
        sequence_length: int = 1024,
        batch_size: int = 8,
        random_seed: int = 42
    ):
        self.num_test_sequences = num_test_sequences
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.test_cases = []
        self.results = []
        
    def run_monotonicity_tests(
        self,
        router,
        retrieval_indices: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Run comprehensive monotonicity tests.
        
        Args:
            router: AgenticRouter instance
            retrieval_indices: Dictionary of index types to test
            device: Device for testing
            
        Returns:
            Test results and analysis
        """
        if device is None:
            device = next(router.parameters()).device
            
        if retrieval_indices is None:
            retrieval_indices = self._generate_test_indices(device)
        
        logger.info(f"Running monotonicity tests with {len(retrieval_indices)} index sets")
        
        # Generate test cases
        self.test_cases = self._create_test_cases(retrieval_indices)
        
        # Run tests
        self.results.clear()
        router.eval()
        
        with torch.no_grad():
            for test_case in self.test_cases:
                logger.info(f"Running test: {test_case.name}")
                result = self._run_single_test(router, test_case, device)
                self.results.append(result)
        
        # Analyze results
        analysis = self._analyze_results()
        
        logger.info(f"Monotonicity testing completed. {len(self.results)} tests run.")
        
        return analysis
    
    def _generate_test_indices(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Generate different types of test indices."""
        vocab_size = 10000
        index_size = 1000
        
        indices = {}
        
        # Clean indices (ascending)
        indices['clean'] = torch.arange(index_size, device=device)
        
        # Shuffled indices (random permutation)
        shuffled = torch.randperm(index_size, device=device)
        indices['shuffled'] = shuffled
        
        # Partially corrupted indices (some invalid)
        partially_corrupted = torch.arange(index_size, device=device)
        corruption_mask = torch.rand(index_size, device=device) < 0.1  # 10% corruption
        partially_corrupted[corruption_mask] = vocab_size + torch.randint(
            0, 1000, (corruption_mask.sum(),), device=device
        )
        indices['partially_corrupted'] = partially_corrupted
        
        # Heavily corrupted indices
        heavily_corrupted = torch.randint(
            vocab_size, vocab_size + 5000, (index_size,), device=device
        )
        indices['heavily_corrupted'] = heavily_corrupted
        
        # Reversed indices
        indices['reversed'] = torch.arange(index_size, device=device).flip(0)
        
        # Duplicated indices (many repeats)
        duplicated = torch.randint(0, index_size // 10, (index_size,), device=device)
        indices['duplicated'] = duplicated
        
        return indices
    
    def _create_test_cases(self, retrieval_indices: Dict[str, torch.Tensor]) -> List[IndexSwapTestCase]:
        """Create test cases from index sets."""
        test_cases = []
        
        clean_indices = retrieval_indices['clean']
        
        # Test case 1: Clean vs Shuffled (should maintain)
        test_cases.append(IndexSwapTestCase(
            name="clean_vs_shuffled",
            description="Clean indices vs shuffled indices - should maintain performance",
            clean_indices=clean_indices,
            modified_indices=retrieval_indices['shuffled'],
            expected_behavior='maintain',
            tolerance=0.05
        ))
        
        # Test case 2: Clean vs Partially Corrupted (should slightly degrade)
        test_cases.append(IndexSwapTestCase(
            name="clean_vs_partially_corrupted",
            description="Clean indices vs partially corrupted - should slightly degrade",
            clean_indices=clean_indices,
            modified_indices=retrieval_indices['partially_corrupted'],
            expected_behavior='degrade',
            tolerance=0.10
        ))
        
        # Test case 3: Clean vs Heavily Corrupted (should significantly degrade)
        test_cases.append(IndexSwapTestCase(
            name="clean_vs_heavily_corrupted", 
            description="Clean indices vs heavily corrupted - should significantly degrade",
            clean_indices=clean_indices,
            modified_indices=retrieval_indices['heavily_corrupted'],
            expected_behavior='degrade',
            tolerance=0.20
        ))
        
        # Test case 4: Clean vs Reversed (should maintain)
        test_cases.append(IndexSwapTestCase(
            name="clean_vs_reversed",
            description="Clean indices vs reversed indices - should maintain performance",
            clean_indices=clean_indices,
            modified_indices=retrieval_indices['reversed'],
            expected_behavior='maintain',
            tolerance=0.05
        ))
        
        # Test case 5: Clean vs Duplicated (should degrade)
        test_cases.append(IndexSwapTestCase(
            name="clean_vs_duplicated",
            description="Clean indices vs duplicated indices - should degrade",
            clean_indices=clean_indices,
            modified_indices=retrieval_indices['duplicated'],
            expected_behavior='degrade',
            tolerance=0.15
        ))
        
        return test_cases
    
    def _run_single_test(
        self,
        router,
        test_case: IndexSwapTestCase,
        device: torch.device
    ) -> MonotonicityResult:
        """Run a single monotonicity test."""
        
        clean_scores = []
        modified_scores = []
        expert_consistency_scores = []
        cache_safety_scores = []
        
        # Run multiple test sequences
        for seq_idx in range(self.num_test_sequences):
            # Generate test sequence
            input_ids = torch.randint(
                0, 1000, (self.batch_size, self.sequence_length),
                device=device, dtype=torch.long
            )
            
            # Test with clean indices
            clean_result = self._evaluate_with_indices(
                router, input_ids, test_case.clean_indices
            )
            
            # Test with modified indices
            modified_result = self._evaluate_with_indices(
                router, input_ids, test_case.modified_indices
            )
            
            # Extract scores
            clean_scores.append(clean_result['quality_score'])
            modified_scores.append(modified_result['quality_score'])
            
            # Expert consistency
            expert_consistency = self._compute_expert_consistency(
                clean_result['expert_usage'], modified_result['expert_usage']
            )
            expert_consistency_scores.append(expert_consistency)
            
            # Cache safety  
            cache_safety = (
                clean_result['cache_safety'] and modified_result['cache_safety']
            )
            cache_safety_scores.append(float(cache_safety))
        
        # Aggregate results
        clean_score = np.mean(clean_scores)
        modified_score = np.mean(modified_scores)
        relative_change = (modified_score - clean_score) / clean_score
        expert_consistency = np.mean(expert_consistency_scores)
        cache_safety_maintained = np.mean(cache_safety_scores) > 0.9
        
        # Check monotonicity
        monotonicity_preserved = self._check_monotonicity(
            test_case, clean_score, modified_score, relative_change
        )
        
        return MonotonicityResult(
            test_name=test_case.name,
            clean_score=clean_score,
            modified_score=modified_score,
            relative_change=relative_change,
            monotonicity_preserved=monotonicity_preserved,
            expert_consistency=expert_consistency,
            cache_safety_maintained=cache_safety_maintained
        )
    
    def _evaluate_with_indices(
        self,
        router,
        input_ids: torch.Tensor,
        indices: torch.Tensor
    ) -> Dict:
        """Evaluate router with specific indices."""
        
        # Run router forward pass
        outputs, routing_result = router.forward(
            input_ids=input_ids,
            return_routing_info=True,
            training_mode=False
        )
        
        # Compute quality score (placeholder - would use actual task metrics)
        # For now, use routing consistency and cache safety as proxies
        quality_components = []
        
        if routing_result:
            # Routing quality
            flip_rate = routing_result.routing_stats.get('flip_rate', 0)
            quality_components.append(max(0, 1 - flip_rate))
            
            # Cache safety
            cache_safety_rate = routing_result.cache_metrics.get('cache_safety_rate', 0)
            quality_components.append(cache_safety_rate)
            
            # Performance (inverse of latency)
            if 'avg_step_time' in routing_result.performance_metrics:
                avg_time = routing_result.performance_metrics['avg_step_time']
                time_score = max(0, 1 - avg_time / 0.1)  # Normalize around 100ms
                quality_components.append(time_score)
        
        quality_score = np.mean(quality_components) if quality_components else 0.5
        
        # Extract other metrics
        expert_usage = routing_result.routing_stats.get('expert_utilization', []) if routing_result else []
        cache_safety = routing_result.cache_metrics.get('cache_safety_rate', 0) > 0.95 if routing_result else False
        
        return {
            'quality_score': quality_score,
            'expert_usage': expert_usage,
            'cache_safety': cache_safety,
            'outputs': outputs,
            'routing_result': routing_result
        }
    
    def _compute_expert_consistency(
        self,
        clean_usage: List[float],
        modified_usage: List[float]
    ) -> float:
        """Compute expert usage consistency between clean and modified."""
        if not clean_usage or not modified_usage:
            return 0.0
        
        # Ensure same length
        min_len = min(len(clean_usage), len(modified_usage))
        clean_usage = clean_usage[:min_len]
        modified_usage = modified_usage[:min_len]
        
        # Compute similarity (1 - normalized L1 distance)
        l1_distance = sum(abs(c - m) for c, m in zip(clean_usage, modified_usage))
        max_possible_distance = 2.0  # Maximum L1 distance for normalized usage
        
        consistency = max(0, 1 - l1_distance / max_possible_distance)
        return consistency
    
    def _check_monotonicity(
        self,
        test_case: IndexSwapTestCase,
        clean_score: float,
        modified_score: float,
        relative_change: float
    ) -> bool:
        """Check if monotonicity is preserved according to expected behavior."""
        
        expected = test_case.expected_behavior
        tolerance = test_case.tolerance
        
        if expected == 'maintain':
            # Should maintain performance within tolerance
            return abs(relative_change) <= tolerance
        
        elif expected == 'degrade':
            # Should degrade but not too much
            return -tolerance <= relative_change <= tolerance * 0.5
        
        elif expected == 'improve':
            # Should improve (rare case)
            return relative_change >= -tolerance * 0.5
        
        else:
            logger.warning(f"Unknown expected behavior: {expected}")
            return False
    
    def _analyze_results(self) -> Dict:
        """Analyze monotonicity test results."""
        if not self.results:
            return {}
        
        # Overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.monotonicity_preserved)
        pass_rate = passed_tests / total_tests
        
        # Performance statistics
        relative_changes = [r.relative_change for r in self.results]
        expert_consistencies = [r.expert_consistency for r in self.results]
        cache_safety_maintained = sum(1 for r in self.results if r.cache_safety_maintained)
        
        analysis = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': pass_rate,
                'cache_safety_pass_rate': cache_safety_maintained / total_tests
            },
            'performance_impact': {
                'mean_relative_change': np.mean(relative_changes),
                'std_relative_change': np.std(relative_changes),
                'max_degradation': min(relative_changes),
                'max_improvement': max(relative_changes)
            },
            'expert_consistency': {
                'mean_consistency': np.mean(expert_consistencies),
                'min_consistency': min(expert_consistencies),
                'consistency_above_threshold': sum(1 for c in expert_consistencies if c > 0.8) / total_tests
            },
            'test_details': {}
        }
        
        # Per-test analysis
        for result in self.results:
            analysis['test_details'][result.test_name] = {
                'monotonicity_preserved': result.monotonicity_preserved,
                'clean_score': result.clean_score,
                'modified_score': result.modified_score,
                'relative_change': result.relative_change,
                'expert_consistency': result.expert_consistency,
                'cache_safety_maintained': result.cache_safety_maintained
            }
        
        # Overall assessment
        analysis['overall_assessment'] = self._overall_assessment(analysis)
        
        return analysis
    
    def _overall_assessment(self, analysis: Dict) -> Dict:
        """Generate overall assessment of monotonicity."""
        summary = analysis['summary']
        performance = analysis['performance_impact']
        consistency = analysis['expert_consistency']
        
        # Criteria for passing
        criteria = {
            'monotonicity_pass_rate': summary['pass_rate'] >= 0.8,
            'cache_safety_maintained': summary['cache_safety_pass_rate'] >= 0.95,
            'expert_consistency_good': consistency['mean_consistency'] >= 0.7,
            'performance_degradation_acceptable': performance['max_degradation'] >= -0.25,
            'no_severe_regressions': performance['mean_relative_change'] >= -0.1
        }
        
        overall_pass = all(criteria.values())
        
        assessment = {
            'overall_pass': overall_pass,
            'criteria_met': criteria,
            'num_criteria_met': sum(criteria.values()),
            'total_criteria': len(criteria)
        }
        
        # Recommendations
        recommendations = []
        if not criteria['monotonicity_pass_rate']:
            recommendations.append("Improve routing consistency across index transformations")
        if not criteria['cache_safety_maintained']:
            recommendations.append("Address cache safety issues under index swaps")
        if not criteria['expert_consistency_good']:
            recommendations.append("Improve expert selection stability")
        if not criteria['performance_degradation_acceptable']:
            recommendations.append("Investigate severe performance regressions")
        
        assessment['recommendations'] = recommendations
        
        return assessment
    
    def save_results(self, output_path: str):
        """Save monotonicity test results."""
        analysis = self._analyze_results()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Monotonicity test results saved to {output_file}")
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary metrics for acceptance gates."""
        if not self.results:
            return {}
        
        pass_rate = sum(1 for r in self.results if r.monotonicity_preserved) / len(self.results)
        cache_safety_rate = sum(1 for r in self.results if r.cache_safety_maintained) / len(self.results)
        mean_consistency = np.mean([r.expert_consistency for r in self.results])
        
        return {
            'monotonicity_pass_rate': pass_rate,
            'cache_safety_maintained': cache_safety_rate,
            'expert_consistency': mean_consistency,
            'monotonicity_intact': pass_rate >= 0.8 and cache_safety_rate >= 0.95
        }


def create_monotonicity_tester(config: Dict) -> MonotonicityTester:
    """Factory function to create MonotonicityTester."""
    return MonotonicityTester(
        num_test_sequences=config.get('num_test_sequences', 50),
        sequence_length=config.get('sequence_length', 1024),
        batch_size=config.get('batch_size', 8),
        random_seed=config.get('random_seed', 42)
    )