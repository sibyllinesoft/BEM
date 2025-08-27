"""
Slice Analysis for BEM v1.1 Evaluation

Implements Slice A (retrieval-strong) and Slice B (full) analysis
according to TODO.md specifications.

Slice A: Examples meeting coverage/consistency thresholds
Slice B: Full dataset
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class SliceAnalyzer:
    """
    Analyzer for slice-based evaluation of BEM v1.1 models.
    
    Implements the slice definitions from TODO.md:
    - Slice A: Retrieval-strong examples (coverage/consistency thresholds)
    - Slice B: Full dataset
    """
    
    def __init__(
        self,
        coverage_threshold: float = 0.7,
        consistency_threshold: float = 0.8
    ):
        self.coverage_threshold = coverage_threshold
        self.consistency_threshold = consistency_threshold
    
    def analyze_retrieval_strong_slice(
        self,
        predictions: List[str],
        targets: List[str],
        batch_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze Slice A (retrieval-strong) performance.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            batch_data: Optional batch data with retrieval features
            
        Returns:
            Slice A analysis results
        """
        if batch_data is None:
            # Fallback: treat all examples as retrieval-strong
            slice_a_indices = list(range(len(predictions)))
        else:
            # Filter based on retrieval strength
            slice_a_indices = self._identify_retrieval_strong_examples(batch_data)
        
        # Extract slice A data
        slice_a_predictions = [predictions[i] for i in slice_a_indices]
        slice_a_targets = [targets[i] for i in slice_a_indices]
        
        # Compute metrics for slice A
        slice_a_metrics = self._compute_slice_metrics(slice_a_predictions, slice_a_targets)
        
        # Analyze retrieval characteristics
        retrieval_analysis = self._analyze_retrieval_characteristics(
            batch_data, slice_a_indices
        ) if batch_data else {}
        
        return {
            'slice_type': 'A',
            'description': 'Retrieval-strong examples',
            'num_examples': len(slice_a_indices),
            'selection_criteria': {
                'coverage_threshold': self.coverage_threshold,
                'consistency_threshold': self.consistency_threshold
            },
            'metrics': slice_a_metrics,
            'retrieval_analysis': retrieval_analysis,
            'example_indices': slice_a_indices
        }
    
    def analyze_full_slice(
        self,
        predictions: List[str],
        targets: List[str],
        batch_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze Slice B (full dataset) performance.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            batch_data: Optional batch data
            
        Returns:
            Slice B analysis results
        """
        # Compute metrics for full dataset
        full_metrics = self._compute_slice_metrics(predictions, targets)
        
        # Analyze overall characteristics
        overall_analysis = self._analyze_overall_characteristics(
            batch_data
        ) if batch_data else {}
        
        return {
            'slice_type': 'B',
            'description': 'Full dataset',
            'num_examples': len(predictions),
            'metrics': full_metrics,
            'overall_analysis': overall_analysis
        }
    
    def compare_slices(
        self,
        slice_a_results: Dict[str, Any],
        slice_b_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare performance between Slice A and Slice B.
        
        Args:
            slice_a_results: Slice A analysis results
            slice_b_results: Slice B analysis results
            
        Returns:
            Comparative analysis
        """
        comparison = {
            'slice_a_count': slice_a_results['num_examples'],
            'slice_b_count': slice_b_results['num_examples'],
            'slice_a_fraction': slice_a_results['num_examples'] / slice_b_results['num_examples'],
            'metric_comparisons': {}
        }
        
        # Compare key metrics
        key_metrics = ['exact_match', 'f1_score', 'bleu', 'chrf']
        
        for metric in key_metrics:
            if (metric in slice_a_results['metrics'] and 
                metric in slice_b_results['metrics']):
                
                a_value = slice_a_results['metrics'][metric]
                b_value = slice_b_results['metrics'][metric]
                
                comparison['metric_comparisons'][metric] = {
                    'slice_a': a_value,
                    'slice_b': b_value,
                    'difference': a_value - b_value,
                    'relative_improvement': (a_value - b_value) / b_value if b_value > 0 else 0.0
                }
        
        return comparison
    
    def _identify_retrieval_strong_examples(
        self,
        batch_data: List[Dict]
    ) -> List[int]:
        """Identify examples that meet retrieval strength criteria."""
        retrieval_strong_indices = []
        
        for i, example in enumerate(batch_data):
            if 'retrieval_features' in example:
                features = example['retrieval_features']
                coverage = features.get('coverage', 0.0)
                consistency = features.get('consistency', 0.0)
                
                if (coverage >= self.coverage_threshold and 
                    consistency >= self.consistency_threshold):
                    retrieval_strong_indices.append(i)
            else:
                # If no retrieval features, include in slice A by default
                retrieval_strong_indices.append(i)
        
        return retrieval_strong_indices
    
    def _compute_slice_metrics(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Compute evaluation metrics for a slice."""
        if not predictions or not targets:
            return {}
        
        metrics = {}
        
        # Exact Match
        em_scores = [1.0 if pred.strip() == target.strip() else 0.0 
                    for pred, target in zip(predictions, targets)]
        metrics['exact_match'] = np.mean(em_scores)
        
        # F1 Score (token-level)
        f1_scores = []
        for pred, target in zip(predictions, targets):
            f1_scores.append(self._compute_f1(pred, target))
        metrics['f1_score'] = np.mean(f1_scores)
        
        # BLEU (simplified)
        bleu_scores = []
        for pred, target in zip(predictions, targets):
            bleu_scores.append(self._compute_bleu_simple(pred, target))
        metrics['bleu'] = np.mean(bleu_scores)
        
        # chrF (character-level F1)
        chrf_scores = []
        for pred, target in zip(predictions, targets):
            chrf_scores.append(self._compute_chrf(pred, target))
        metrics['chrf'] = np.mean(chrf_scores)
        
        return metrics
    
    def _compute_f1(self, prediction: str, target: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = set(prediction.split())
        target_tokens = set(target.split())
        
        if not pred_tokens and not target_tokens:
            return 1.0
        if not pred_tokens or not target_tokens:
            return 0.0
        
        intersection = pred_tokens & target_tokens
        
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(target_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _compute_bleu_simple(self, prediction: str, target: str) -> float:
        """Compute simplified BLEU score."""
        pred_tokens = prediction.split()
        target_tokens = target.split()
        
        if not pred_tokens or not target_tokens:
            return 0.0
        
        # Simple 1-gram precision
        pred_set = set(pred_tokens)
        target_set = set(target_tokens)
        
        intersection = pred_set & target_set
        return len(intersection) / len(pred_set) if pred_set else 0.0
    
    def _compute_chrf(self, prediction: str, target: str) -> float:
        """Compute character-level F1 score."""
        pred_chars = set(prediction.replace(' ', ''))
        target_chars = set(target.replace(' ', ''))
        
        if not pred_chars and not target_chars:
            return 1.0
        if not pred_chars or not target_chars:
            return 0.0
        
        intersection = pred_chars & target_chars
        
        precision = len(intersection) / len(pred_chars)
        recall = len(intersection) / len(target_chars)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _analyze_retrieval_characteristics(
        self,
        batch_data: List[Dict],
        slice_indices: List[int]
    ) -> Dict[str, Any]:
        """Analyze retrieval characteristics of slice A examples."""
        if not batch_data:
            return {}
        
        slice_data = [batch_data[i] for i in slice_indices]
        
        coverages = []
        consistencies = []
        retrieval_lengths = []
        
        for example in slice_data:
            if 'retrieval_features' in example:
                features = example['retrieval_features']
                coverages.append(features.get('coverage', 0.0))
                consistencies.append(features.get('consistency', 0.0))
                
                if 'retrieved_passages' in features:
                    total_length = sum(len(passage) for passage in features['retrieved_passages'])
                    retrieval_lengths.append(total_length)
        
        analysis = {}
        
        if coverages:
            analysis['coverage_stats'] = {
                'mean': np.mean(coverages),
                'std': np.std(coverages),
                'min': np.min(coverages),
                'max': np.max(coverages)
            }
        
        if consistencies:
            analysis['consistency_stats'] = {
                'mean': np.mean(consistencies),
                'std': np.std(consistencies),
                'min': np.min(consistencies),
                'max': np.max(consistencies)
            }
        
        if retrieval_lengths:
            analysis['retrieval_length_stats'] = {
                'mean': np.mean(retrieval_lengths),
                'std': np.std(retrieval_lengths),
                'min': np.min(retrieval_lengths),
                'max': np.max(retrieval_lengths)
            }
        
        return analysis
    
    def _analyze_overall_characteristics(
        self,
        batch_data: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze overall dataset characteristics."""
        if not batch_data:
            return {}
        
        analysis = {
            'total_examples': len(batch_data),
            'examples_with_retrieval': 0,
            'average_input_length': 0,
            'average_target_length': 0
        }
        
        input_lengths = []
        target_lengths = []
        
        for example in batch_data:
            if 'retrieval_features' in example:
                analysis['examples_with_retrieval'] += 1
            
            if 'input' in example:
                input_lengths.append(len(example['input']))
            if 'target' in example:
                target_lengths.append(len(example['target']))
        
        if input_lengths:
            analysis['average_input_length'] = np.mean(input_lengths)
            analysis['input_length_std'] = np.std(input_lengths)
        
        if target_lengths:
            analysis['average_target_length'] = np.mean(target_lengths)
            analysis['target_length_std'] = np.std(target_lengths)
        
        analysis['retrieval_coverage'] = analysis['examples_with_retrieval'] / len(batch_data)
        
        return analysis


def create_slice_analyzer(
    coverage_threshold: float = 0.7,
    consistency_threshold: float = 0.8
) -> SliceAnalyzer:
    """
    Factory function for slice analyzer.
    
    Args:
        coverage_threshold: Minimum coverage for Slice A
        consistency_threshold: Minimum consistency for Slice A
        
    Returns:
        SliceAnalyzer instance
    """
    return SliceAnalyzer(
        coverage_threshold=coverage_threshold,
        consistency_threshold=consistency_threshold
    )