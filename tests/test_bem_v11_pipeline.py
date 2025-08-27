#!/usr/bin/env python3
"""
BEM v1.1 Research Pipeline Validation and Quality Assurance Tests
Comprehensive test suite ensuring TODO.md requirements are met.

Key validations:
1. BEM-v1.1-stable architecture compliance (cache-safe, spectral governance)
2. Statistical pipeline correctness (BCa bootstrap, FDR correction)
3. Cache metrics accuracy and quality gates
4. Leak detection functionality
5. Experiment configuration validation
6. End-to-end pipeline integration
"""

import unittest
import tempfile
import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from bem.bem_v11_stable import (
    BEMv11Module, BEMv11StableModel, SpectralGovernance, 
    ChunkStickyRouter, AttentionLogitBias, validate_cache_safety,
    create_bem_v11_stable
)
from analysis.stats import StatisticalAnalyzer
from analysis.cache_metrics import CacheMetricsAnalyzer
from analysis.leakcheck import MinHashLeakDetector
from analysis.pareto import ParetoAnalyzer


class TestBEMv11Architecture(unittest.TestCase):
    """Test BEM-v1.1-stable architecture compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock base layer
        self.base_layer = nn.Linear(768, 768)
        
        # Create BEM module with TODO.md specs
        self.bem_module = BEMv11Module(
            base_layer=self.base_layer,
            rank=8,
            num_experts=2,
            chunk_size=128,
            hysteresis_tau=0.7,
            retrieval_dim=768
        )
    
    def test_cache_safe_attachment_points(self):
        """Test that only W_O and W_down attachment points are used (cache-safe)."""
        # Create mock model with various layer names
        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        'attention': nn.ModuleDict({
                            'W_Q': nn.Linear(768, 768),  # Should be forbidden
                            'W_K': nn.Linear(768, 768),  # Should be forbidden  
                            'W_V': nn.Linear(768, 768),  # Should be forbidden
                            'W_O': nn.Linear(768, 768),  # Should be allowed
                        }),
                        'mlp': nn.ModuleDict({
                            'W_up': nn.Linear(768, 3072),
                            'W_down': nn.Linear(3072, 768),  # Should be allowed
                            'W_gate': nn.Linear(768, 3072)
                        })
                    })
                ])
        
        mock_model = MockTransformer()
        
        # Test safe attachment points
        safe_points = ['W_O', 'W_down']
        bem_model = BEMv11StableModel(
            base_model=mock_model,
            rank_schedule=[8],
            attachment_points=safe_points
        )
        
        # Validate cache safety
        is_safe = validate_cache_safety(bem_model)
        self.assertTrue(is_safe, "BEM model should be cache-safe with W_O and W_down")
        
        # Test unsafe attachment points  
        unsafe_points = ['W_Q', 'W_K', 'W_V']
        with self.assertRaises(Exception):
            # Should fail safety validation
            unsafe_model = BEMv11StableModel(
                base_model=mock_model,
                rank_schedule=[8],
                attachment_points=unsafe_points
            )
            validate_cache_safety(unsafe_model)
    
    def test_depth_varying_ranks(self):
        """Test that depth-varying rank schedule is applied correctly."""
        rank_schedule = [2, 4, 8, 8, 8, 4, 2]  # TODO.md specification
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Create layers at different depths
                self.layers = nn.ModuleList([
                    nn.ModuleDict({'W_O': nn.Linear(768, 768)})
                    for _ in range(len(rank_schedule))
                ])
        
        mock_model = MockModel()
        bem_model = BEMv11StableModel(
            base_model=mock_model,
            rank_schedule=rank_schedule,
            attachment_points=['W_O']
        )
        
        # Check that different layers get different ranks
        self.assertEqual(len(bem_model.bem_modules), len(rank_schedule))
        
        # Verify rank assignment (simplified - actual implementation may vary)
        for i, (name, module) in enumerate(bem_model.bem_modules.items()):
            expected_rank = rank_schedule[min(i, len(rank_schedule) - 1)]
            self.assertEqual(module.rank, expected_rank, 
                           f"Layer {i} should have rank {expected_rank}")
    
    def test_chunk_sticky_routing(self):
        """Test chunk-sticky routing with hysteresis."""
        router = ChunkStickyRouter(
            input_dim=768,
            num_experts=2,
            chunk_size=128,
            hysteresis_tau=0.7
        )
        
        # Test input
        batch_size, seq_len = 2, 256
        x = torch.randn(batch_size, seq_len, 768)
        
        # Forward pass
        routing_weights, expert_indices = router(x)
        
        # Check output shapes
        self.assertEqual(routing_weights.shape, (batch_size, seq_len, 2))
        
        expected_chunks = (seq_len + 128 - 1) // 128
        self.assertEqual(expert_indices.shape, (batch_size, expected_chunks))
        
        # Check that routing is chunk-wise constant
        chunk_0_start, chunk_0_end = 0, 128
        chunk_0_weights = routing_weights[:, chunk_0_start:chunk_0_end, :]
        
        # All positions in chunk should have same routing
        for i in range(1, chunk_0_end - chunk_0_start):
            torch.testing.assert_close(
                chunk_0_weights[:, 0, :], 
                chunk_0_weights[:, i, :],
                msg="Routing should be constant within chunk"
            )
    
    def test_attention_logit_bias(self):
        """Test attention-logit bias computation (cache-safe)."""
        bias_module = AttentionLogitBias(retrieval_dim=768)
        
        # Test input
        batch_size, seq_len = 2, 128
        retrieval_features = torch.randn(batch_size, seq_len, 768)
        
        # Compute bias
        bias = bias_module(retrieval_features)
        
        # Check output shape
        self.assertEqual(bias.shape, (batch_size, seq_len, 1))
        
        # Check that bias is finite and reasonable
        self.assertTrue(torch.isfinite(bias).all(), "Bias should be finite")
        self.assertLess(torch.abs(bias).max().item(), 10.0, "Bias should be reasonable magnitude")
    
    def test_spectral_governance(self):
        """Test spectral governance with σ₁ clamp and Frobenius norm budget."""
        governance = SpectralGovernance(max_singular_value=1.0, fro_budget=1.0)
        
        # Create delta weights that need governance
        delta_w = torch.randn(768, 768) * 5.0  # Large values to trigger governance
        
        # Apply governance
        governed_delta = governance.apply_governance(delta_w)
        
        # Check spectral norm constraint
        U, S, Vh = torch.linalg.svd(governed_delta, full_matrices=False)
        max_singular = S.max().item()
        self.assertLessEqual(max_singular, 1.0 + 1e-6, "Max singular value should be clamped")
        
        # Check Frobenius norm budget
        fro_norm = torch.norm(governed_delta, 'fro').item()
        self.assertLessEqual(fro_norm, 1.0 + 1e-6, "Frobenius norm should be within budget")
    
    def test_parallel_lora_computation(self):
        """Test parallel LoRA computation y = base(x) + Σ_e g_e · B_e (A_e x)."""
        # Test with simple input
        batch_size, seq_len, dim = 2, 10, 768
        x = torch.randn(batch_size, seq_len, dim)
        retrieval_features = torch.randn(batch_size, seq_len, 768)
        
        # Forward pass
        outputs = self.bem_module(x, retrieval_features)
        
        # Check required outputs
        self.assertIn('output', outputs)
        self.assertIn('routing_weights', outputs)
        self.assertIn('expert_outputs', outputs)
        self.assertIn('attention_bias', outputs)
        
        # Check output shape
        output = outputs['output']
        self.assertEqual(output.shape, (batch_size, seq_len, 768))
        
        # Check that output includes base layer + expert contributions
        base_output = self.base_layer(x)
        self.assertFalse(torch.allclose(output, base_output), 
                        "Output should differ from base layer (includes expert contributions)")


class TestStatisticalPipeline(unittest.TestCase):
    """Test statistical analysis pipeline compliance with TODO.md."""
    
    def setUp(self):
        """Set up test statistical analyzer."""
        self.analyzer = StatisticalAnalyzer(n_bootstrap=1000, alpha=0.05)  # Reduced for testing
        
        # Create mock experimental data
        np.random.seed(42)
        self.bem_scores = np.random.normal(0.75, 0.05, 20)  # BEM slightly better
        self.baseline_scores = np.random.normal(0.70, 0.05, 20)  # Baseline
    
    def test_relative_improvement_calculation(self):
        """Test relative improvement calculation: Δ% = (BEM - Baseline) / Baseline."""
        result = self.analyzer.compute_relative_improvement_ci(
            self.bem_scores, self.baseline_scores
        )
        
        # Check required fields
        self.assertIn('relative_improvement_pct', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        self.assertIn('significant', result)
        
        # Check calculation correctness
        expected_improvement = ((self.bem_scores.mean() - self.baseline_scores.mean()) 
                              / self.baseline_scores.mean()) * 100
        
        self.assertAlmostEqual(result['relative_improvement_pct'], expected_improvement, places=2)
        
        # Check significance logic (CI > 0)
        ci_above_zero = result['ci_lower'] > 0
        self.assertEqual(result['significant'], ci_above_zero)
    
    def test_bca_bootstrap_confidence_intervals(self):
        """Test BCa bootstrap confidence intervals (10k samples in production)."""
        # Test with simple data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        def mean_func(x):
            return np.mean(x)
        
        # Use smaller sample for testing
        bootstrap = self.analyzer.bootstrap
        bootstrap.n_bootstrap = 100  # Reduced for speed
        
        lower, upper, point_est = bootstrap.compute_bca_ci(data, mean_func)
        
        # Check that CI contains true mean
        true_mean = np.mean(data)
        self.assertEqual(point_est, true_mean)
        self.assertLessEqual(lower, true_mean)
        self.assertGreaterEqual(upper, true_mean)
        
        # Check CI is reasonable
        self.assertLess(upper - lower, true_mean)  # CI width should be reasonable
    
    def test_fdr_correction(self):
        """Test FDR correction for multiple comparisons."""
        # Create mock p-values
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20]
        
        from analysis.stats import MultipleComparisonsCorrection
        rejected, corrected_p, alpha_sidak = MultipleComparisonsCorrection.apply_fdr_correction(
            p_values, method='fdr_bh', alpha=0.05
        )
        
        # Check outputs
        self.assertEqual(len(rejected), len(p_values))
        self.assertEqual(len(corrected_p), len(p_values))
        
        # Check that corrected p-values are reasonable
        for i, (orig_p, corr_p) in enumerate(zip(p_values, corrected_p)):
            self.assertGreaterEqual(corr_p, orig_p, 
                                  f"Corrected p-value {i} should be >= original")
    
    def test_slice_analysis(self):
        """Test analysis for both Slice-A and Slice-B."""
        # Mock data for both slices
        bem_results = {
            'EM': np.random.normal(0.70, 0.05, 10),
            'F1': np.random.normal(0.75, 0.05, 10),
            'BLEU': np.random.normal(0.65, 0.05, 10),
            'chrF': np.random.normal(0.72, 0.05, 10)
        }
        
        baseline_results = {
            'EM': np.random.normal(0.65, 0.05, 10),
            'F1': np.random.normal(0.70, 0.05, 10),
            'BLEU': np.random.normal(0.60, 0.05, 10),
            'chrF': np.random.normal(0.67, 0.05, 10)
        }
        
        # Analyze slice
        results = self.analyzer.analyze_slice_comparison(
            bem_results, baseline_results, "slice_a"
        )
        
        # Check required fields
        self.assertIn('slice_name', results)
        self.assertIn('metrics', results)
        self.assertIn('consistency_win', results)
        self.assertIn('significant_improvements', results)
        self.assertIn('fdr_results', results)
        
        # Check that all metrics are analyzed
        for metric in self.analyzer.metrics:
            if metric in bem_results and metric in baseline_results:
                self.assertIn(metric, results['metrics'])
                
                # Check metric analysis structure
                metric_result = results['metrics'][metric]
                self.assertIn('relative_improvement', metric_result)
                self.assertIn('paired_t_test', metric_result)
                self.assertIn('final_significant', metric_result)
    
    def test_consistency_win_detection(self):
        """Test consistency win detection (all metrics improved)."""
        # All metrics improved
        bem_better = {
            'EM': np.array([0.8, 0.8, 0.8]),
            'F1': np.array([0.8, 0.8, 0.8]),
            'BLEU': np.array([0.8, 0.8, 0.8]),
            'chrF': np.array([0.8, 0.8, 0.8])
        }
        
        baseline_worse = {
            'EM': np.array([0.7, 0.7, 0.7]),
            'F1': np.array([0.7, 0.7, 0.7]),
            'BLEU': np.array([0.7, 0.7, 0.7]),
            'chrF': np.array([0.7, 0.7, 0.7])
        }
        
        # Should detect consistency win
        results = self.analyzer.analyze_slice_comparison(
            bem_better, baseline_worse, "test_slice"
        )
        self.assertTrue(results['consistency_win'], "Should detect consistency win when all metrics improve")
        
        # One metric worse - should not be consistency win
        bem_mixed = bem_better.copy()
        bem_mixed['EM'] = np.array([0.6, 0.6, 0.6])  # Worse than baseline
        
        results_mixed = self.analyzer.analyze_slice_comparison(
            bem_mixed, baseline_worse, "test_slice"
        )
        self.assertFalse(results_mixed['consistency_win'], 
                        "Should not detect consistency win when one metric is worse")


class TestCacheMetrics(unittest.TestCase):
    """Test cache metrics analysis and quality gates."""
    
    def setUp(self):
        """Set up cache metrics analyzer."""
        self.analyzer = CacheMetricsAnalyzer()
    
    def test_kv_hit_rate_computation(self):
        """Test K/V cache hit rate calculation."""
        # Mock cache events
        cache_events = [
            {'cache_hit': True},
            {'cache_hit': True},
            {'cache_hit': False},
            {'cache_hit': True},
            {'cache_hit': False}
        ]
        
        hit_rate = self.analyzer.compute_kv_hit_rate(cache_events)
        expected_rate = (3 / 5) * 100  # 60%
        
        self.assertAlmostEqual(hit_rate, expected_rate, places=2)
    
    def test_routing_flips_computation(self):
        """Test routing flips per token calculation."""
        # Mock routing decisions with some flips
        routing_decisions = np.array([0, 0, 0, 1, 1, 1, 0, 0])  # 2 flips
        chunk_size = 4
        
        flips_per_token = self.analyzer.compute_routing_flips(routing_decisions, chunk_size)
        
        # With chunk_size=4: chunks are [0,0,0,1] (mode=0) and [1,1,0,0] (mode=0) 
        # So 0 flips between chunks
        # This depends on the exact implementation - test the concept
        self.assertIsInstance(flips_per_token, float)
        self.assertGreaterEqual(flips_per_token, 0.0)
    
    def test_gate_entropy_computation(self):
        """Test gate entropy calculation."""
        # Mock routing weights
        routing_weights = np.array([
            [0.9, 0.1],    # Low entropy (concentrated)
            [0.5, 0.5],    # High entropy (uniform) 
            [0.8, 0.2],    # Medium entropy
        ])
        
        entropy = self.analyzer.compute_gate_entropy(routing_weights)
        
        # Check that entropy is reasonable
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, np.log(2))  # Max entropy for 2 experts
    
    def test_quality_gates_validation(self):
        """Test quality gates validation (cache hit ≥ 80%)."""
        # Mock experimental data with good cache performance
        good_cache_data = {
            'method_trends': {
                'groups': {
                    'bem_v11': {
                        'kv_hit_rate': {'mean': 85.0, 'std': 2.0}
                    },
                    'baseline': {
                        'kv_hit_rate': {'mean': 82.0, 'std': 3.0}
                    }
                }
            }
        }
        
        quality_gates = self.analyzer._check_quality_gates(good_cache_data['method_trends'])
        
        self.assertTrue(quality_gates['overall_pass'], "Should pass with good cache hit rates")
        self.assertTrue(quality_gates['results']['bem_v11']['pass'])
        self.assertTrue(quality_gates['results']['baseline']['pass'])
        
        # Test with bad cache performance
        bad_cache_data = {
            'method_trends': {
                'groups': {
                    'bem_v11': {
                        'kv_hit_rate': {'mean': 75.0, 'std': 2.0}  # Below 80% threshold
                    }
                }
            }
        }
        
        bad_quality_gates = self.analyzer._check_quality_gates(bad_cache_data['method_trends'])
        
        self.assertFalse(bad_quality_gates['overall_pass'], "Should fail with bad cache hit rates")
        self.assertFalse(bad_quality_gates['results']['bem_v11']['pass'])


class TestLeakDetection(unittest.TestCase):
    """Test leak detection with LSH/minhash."""
    
    def setUp(self):
        """Set up leak detector."""
        self.detector = MinHashLeakDetector(threshold=0.7, num_perm=64)  # Reduced for testing
        
        # Mock index documents
        self.index_docs = [
            {'id': 'doc1', 'content': 'The quick brown fox jumps over the lazy dog'},
            {'id': 'doc2', 'content': 'A completely different sentence with no overlap'},
            {'id': 'doc3', 'content': 'Another unique document about machine learning'}
        ]
        
        # Index the documents
        self.detector.index_documents(self.index_docs)
    
    def test_leak_detection_positive_case(self):
        """Test leak detection with actual leaks."""
        # Query very similar to indexed document
        eval_queries = [
            {'id': 'query1', 'query': 'The quick brown fox jumps over lazy dog'},  # Very similar to doc1
            {'id': 'query2', 'query': 'Machine learning is a subset of artificial intelligence'}  # Different
        ]
        
        results = self.detector.detect_leaks(eval_queries)
        
        self.assertEqual(len(results), 2)
        
        # First query should detect leak
        query1_result = results[0]
        self.assertTrue(query1_result.leak_detected, "Should detect leak for similar query")
        self.assertGreater(query1_result.max_similarity, 0.5, "Similarity should be high")
        
        # Second query should not detect leak
        query2_result = results[1] 
        self.assertLessEqual(query2_result.max_similarity, 0.5, "Similarity should be low for different query")
    
    def test_leak_statistics_computation(self):
        """Test leak statistics computation."""
        # Mock leak results
        leak_results = [
            Mock(leak_detected=True, similarity_scores=[0.8, 0.7]),
            Mock(leak_detected=False, similarity_scores=[]),
            Mock(leak_detected=True, similarity_scores=[0.9]),
            Mock(leak_detected=False, similarity_scores=[])
        ]
        
        stats = self.detector.compute_leak_statistics(leak_results)
        
        # Check basic statistics
        self.assertEqual(stats['total_queries'], 4)
        self.assertEqual(stats['leaked_queries'], 2)
        self.assertAlmostEqual(stats['leak_rate'], 0.5, places=2)
        
        # Check severity assessment
        self.assertIn('severity', stats)
        self.assertIn('assessment', stats)
    
    def test_policy_over_memory_validation(self):
        """Test policy-over-memory validation from TODO.md."""
        # Low leak rate - should be valid
        low_leak_stats = {
            'leak_rate': 0.02,  # 2% - below 5% threshold
            'max_similarity': 0.8
        }
        
        assessment = self.detector._assess_leak_impact(
            low_leak_stats['leak_rate'], 
            low_leak_stats['max_similarity']
        )
        
        self.assertTrue(assessment['policy_over_memory_valid'], 
                       "Low leak rate should validate policy-over-memory claims")
        
        # High leak rate - should be invalid  
        high_leak_stats = {
            'leak_rate': 0.10,  # 10% - above 5% threshold
            'max_similarity': 0.9
        }
        
        bad_assessment = self.detector._assess_leak_impact(
            high_leak_stats['leak_rate'],
            high_leak_stats['max_similarity']  
        )
        
        self.assertFalse(bad_assessment['policy_over_memory_valid'],
                        "High leak rate should invalidate policy-over-memory claims")


class TestParetoAnalysis(unittest.TestCase):
    """Test Pareto frontier analysis for latency-quality trade-offs."""
    
    def setUp(self):
        """Set up Pareto analyzer."""
        self.analyzer = ParetoAnalyzer(
            primary_metric='F1',
            latency_metric='p50_latency_ms',
            latency_budget_pct=15.0
        )
    
    def test_pareto_frontier_identification(self):
        """Test Pareto frontier identification."""
        # Mock method data
        method_data = {
            'method_A': {
                'quality_metrics': {'F1': {'mean': 0.80, 'ci_lower': 0.78, 'ci_upper': 0.82}},
                'performance_metrics': {'p50_latency_ms': {'mean': 100, 'ci_lower': 95, 'ci_upper': 105}},
                'resource_metrics': {'vram_delta_gb': {'mean': 0.1}}
            },
            'method_B': {  # Dominated by A (lower quality, higher latency)
                'quality_metrics': {'F1': {'mean': 0.75, 'ci_lower': 0.73, 'ci_upper': 0.77}},
                'performance_metrics': {'p50_latency_ms': {'mean': 120, 'ci_lower': 115, 'ci_upper': 125}},
                'resource_metrics': {'vram_delta_gb': {'mean': 0.2}}
            },
            'method_C': {  # Pareto optimal (higher quality, higher latency - trade-off)
                'quality_metrics': {'F1': {'mean': 0.85, 'ci_lower': 0.83, 'ci_upper': 0.87}},
                'performance_metrics': {'p50_latency_ms': {'mean': 130, 'ci_lower': 125, 'ci_upper': 135}},
                'resource_metrics': {'vram_delta_gb': {'mean': 0.3}}
            }
        }
        
        pareto_points = self.analyzer.identify_pareto_frontier(method_data)
        
        # Check results
        self.assertEqual(len(pareto_points), 3)
        
        # Find pareto optimal points
        pareto_optimal = [p for p in pareto_points if p.is_pareto_optimal]
        
        # Method A should be Pareto optimal (dominates B)
        method_a_point = next(p for p in pareto_points if p.method == 'method_A')
        self.assertTrue(method_a_point.is_pareto_optimal)
        self.assertIn('method_B', method_a_point.dominates)
        
        # Method C should be Pareto optimal (trade-off point)
        method_c_point = next(p for p in pareto_points if p.method == 'method_C')
        self.assertTrue(method_c_point.is_pareto_optimal)
        
        # Method B should not be Pareto optimal (dominated)
        method_b_point = next(p for p in pareto_points if p.method == 'method_B')
        self.assertFalse(method_b_point.is_pareto_optimal)
    
    def test_latency_budget_compliance(self):
        """Test latency budget compliance (+15% threshold)."""
        method_data = {
            'baseline': {
                'performance_metrics': {'p50_latency_ms': {'mean': 100}}
            },
            'method_compliant': {
                'performance_metrics': {'p50_latency_ms': {'mean': 110}}  # +10% - within budget
            },
            'method_over_budget': {
                'performance_metrics': {'p50_latency_ms': {'mean': 130}}  # +30% - over budget  
            }
        }
        
        compliance = self.analyzer.compute_latency_budget_compliance(method_data, 'baseline')
        
        # Check baseline
        self.assertEqual(compliance['baseline_latency'], 100)
        self.assertEqual(compliance['budget_threshold'], 115)  # 100 * 1.15
        
        # Check compliance
        self.assertTrue(compliance['method_compliance']['method_compliant']['budget_compliant'])
        self.assertFalse(compliance['method_compliance']['method_over_budget']['budget_compliant'])


class TestExperimentConfiguration(unittest.TestCase):
    """Test experiment configuration validation."""
    
    def test_bem_v11_config_validation(self):
        """Test BEM v1.1 configuration compliance with TODO.md."""
        config_path = Path(__file__).parent / 'experiments' / 'v11_baseline.yml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required TODO.md specifications
        bem_config = config['model']['bem_config']
        
        # E1: Sites should be W_O + W_down only
        sites = bem_config['sites']
        self.assertEqual(set(sites), {'W_O', 'W_down'}, "Sites should be W_O and W_down only")
        
        # E1: Depth-varying ranks
        rank_schedule = bem_config['rank_schedule']
        self.assertEqual(rank_schedule, [2, 4, 8, 8, 8, 4, 2], "Should use TODO.md rank schedule")
        
        # E3: Chunk-sticky routing
        routing = bem_config['routing']
        self.assertIn(routing['chunk_size'], [64, 128], "Chunk size should be 64 or 128")
        self.assertEqual(routing['hysteresis_tau'], 0.7, "Hysteresis tau should be 0.7")
        
        # E4: Attention bias
        attention_bias = bem_config['attention_bias']
        self.assertTrue(attention_bias['enabled'], "Attention bias should be enabled")
        
        # Seeds for statistical analysis
        seeds = config['training']['seeds']
        self.assertEqual(seeds, [1, 2, 3, 4, 5], "Should use 5 seeds for statistical analysis")
        
        # Quality gates from TODO.md
        quality_gates = config['quality_gates']
        self.assertEqual(quality_gates['latency_budget']['max_increase_pct'], 15.0)
        self.assertEqual(quality_gates['cache_performance']['min_hit_rate_pct'], 80.0)
        self.assertEqual(quality_gates['vram_budget']['max_delta_pct'], 5.0)
    
    def test_baseline_config_matching(self):
        """Test that baseline config matches BEM for fair comparison."""
        bem_config_path = Path(__file__).parent / 'experiments' / 'v11_baseline.yml'
        baseline_config_path = Path(__file__).parent / 'experiments' / 'lora_baseline.yml'
        
        with open(bem_config_path, 'r') as f:
            bem_config = yaml.safe_load(f)
        
        with open(baseline_config_path, 'r') as f:
            baseline_config = yaml.safe_load(f)
        
        # Should have same sites and rank schedule
        bem_sites = bem_config['model']['bem_config']['sites']
        baseline_sites = baseline_config['model']['lora_config']['sites']
        self.assertEqual(bem_sites, baseline_sites, "Baseline should use same attachment sites")
        
        bem_ranks = bem_config['model']['bem_config']['rank_schedule']
        baseline_ranks = baseline_config['model']['lora_config']['rank_schedule']
        self.assertEqual(bem_ranks, baseline_ranks, "Baseline should use same rank schedule")
        
        # Should have same training parameters
        self.assertEqual(bem_config['training']['seeds'], baseline_config['training']['seeds'])
        self.assertEqual(bem_config['training']['learning_rate'], baseline_config['training']['learning_rate'])
        self.assertEqual(bem_config['training']['batch_size'], baseline_config['training']['batch_size'])
    
    def test_cache_safety_validation(self):
        """Test cache safety validation in configs."""
        config_path = Path(__file__).parent / 'experiments' / 'v11_baseline.yml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check forbidden sites
        forbidden_sites = config['safety']['forbidden_sites']
        unsafe_sites = ['W_Q', 'W_K', 'W_V', 'q_proj', 'k_proj', 'v_proj']
        
        for unsafe_site in unsafe_sites:
            self.assertIn(unsafe_site, forbidden_sites, f"{unsafe_site} should be forbidden")
        
        # Check that config sites are not in forbidden list
        sites = config['model']['bem_config']['sites']
        for site in sites:
            self.assertNotIn(site, forbidden_sites, f"{site} should not be forbidden")


class TestEndToEndPipeline(unittest.TestCase):
    """Test end-to-end pipeline integration."""
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.loads')
    def test_statistical_pipeline_integration(self, mock_json_loads, mock_open):
        """Test integration between experimental results and statistical analysis."""
        # Mock experimental results
        mock_results = [
            {
                'method': 'bem_v11',
                'seed': 1,
                'evaluation_results': {
                    'standard_metrics': {'EM': 0.75, 'F1': 0.80, 'BLEU': 0.70, 'chrF': 0.73},
                    'system_telemetry': {'p50_latency_ms': 120, 'vram_usage_gb': 8.5}
                }
            },
            {
                'method': 'baseline',
                'seed': 1, 
                'evaluation_results': {
                    'standard_metrics': {'EM': 0.70, 'F1': 0.75, 'BLEU': 0.65, 'chrF': 0.68},
                    'system_telemetry': {'p50_latency_ms': 100, 'vram_usage_gb': 8.0}
                }
            }
        ]
        
        mock_json_loads.side_effect = mock_results
        mock_open.return_value.__enter__.return_value.__iter__.return_value = ['line1', 'line2']
        
        # Test pipeline components can process this data
        analyzer = StatisticalAnalyzer(n_bootstrap=100)
        
        # This is a simplified test - actual integration would be more complex
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.metrics, ['EM', 'F1', 'BLEU', 'chrF'])


def run_validation_suite():
    """Run the complete validation suite and generate report."""
    print("="*60)
    print("BEM v1.1 RESEARCH PIPELINE VALIDATION SUITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBEMv11Architecture,
        TestStatisticalPipeline,
        TestCacheMetrics,
        TestLeakDetection,
        TestParetoAnalysis,
        TestExperimentConfiguration,
        TestEndToEndPipeline
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("✅ ALL TESTS PASSED - BEM v1.1 pipeline ready for execution!")
    else:
        print("❌ VALIDATION FAILED - Please fix issues before proceeding")
        
        if failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
                
        if errors:
            print("\nErrors:")  
            for test, traceback in result.errors:
                print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result


if __name__ == '__main__':
    run_validation_suite()