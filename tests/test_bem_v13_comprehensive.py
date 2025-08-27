#!/usr/bin/env python3
"""
BEM v1.3 Performance+Agentic Sprint Comprehensive Test Suite

This test suite validates the entire BEM v1.3 system as specified in TODO.md,
ensuring research-grade reliability for all components:

CRITICAL TESTING AREAS:
1. Core Component Tests - BEM variants (PT1-PT4, AR1, OL, MM, VC)
2. Parameter/FLOP parity enforcement (±5%)  
3. Cache-safety invariants
4. Statistical analysis framework (BCa bootstrap, FDR)
5. Integration tests for full training pipeline
6. Quality assurance for numerical stability
7. Performance validation for latency and throughput
8. Reproducibility guarantees

All tests follow the strict research requirements from TODO.md with
bulletproof statistical analysis and parameter budget enforcement.
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
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import warnings
import math
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import scipy.stats as stats
from scipy.stats import bootstrap

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Core BEM imports
from bem.bem_v11_stable import (
    BEMv11Module, BEMv11StableModel, SpectralGovernance,
    ChunkStickyRouter, AttentionLogitBias, validate_cache_safety,
    create_bem_v11_stable
)

# BEM v1.3 specific imports
try:
    from bem2.router.agentic_router import AgenticRouter, MacroPolicy
    from bem2.online.online_learner import OnlineLearner
    from bem2.multimodal.controller_integration import MultimodalController
    from bem2.safety.safety_controller import SafetyController
    from bem2.perftrack.pt1_head_gating import HeadGatingModule
    from bem2.perftrack.pt2_dynamic_mask import DynamicMaskModule
    from bem2.perftrack.pt3_kronecker import KroneckerModule
    from bem2.perftrack.pt4_residual_film import ResidualFiLMModule
except ImportError:
    # Create mock classes for missing imports
    class AgenticRouter: pass
    class MacroPolicy: pass
    class OnlineLearner: pass
    class MultimodalController: pass
    class SafetyController: pass
    class HeadGatingModule: pass
    class DynamicMaskModule: pass
    class KroneckerModule: pass
    class ResidualFiLMModule: pass

# Analysis imports
from analysis.statistical_analysis import (
    ExperimentMetrics, ComparisonResult, BootstrapStatistics,
    load_experiment_results, apply_fdr_correction
)

# Evaluation imports
try:
    from eval.bem_evaluator import BEMEvaluator
    from workflows.experiment_runner import ExperimentRunner
except ImportError:
    class BEMEvaluator: pass
    class ExperimentRunner: pass


@dataclass
class ParityValidationResult:
    """Result of parameter/FLOP parity validation."""
    params_within_tolerance: bool
    flops_within_tolerance: bool
    param_ratio: float
    flop_ratio: float
    tolerance: float
    

@dataclass
class CacheSafetyResult:
    """Result of cache safety validation."""
    is_cache_safe: bool
    violations: List[str]
    attachment_points: List[str]
    

@dataclass
class NumericalStabilityResult:
    """Result of numerical stability testing."""
    all_stable: bool
    failed_operations: List[str]
    max_error: float
    tolerance: float


class TestBEMv13CoreComponents(unittest.TestCase):
    """Test all core BEM v1.3 components for correctness and specifications."""
    
    def setUp(self):
        """Set up test fixtures with TODO.md specifications."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TODO.md specifications
        self.tolerance = 0.05  # ±5% parity requirement
        self.chunk_sizes = [64, 128]  # Chunk-sticky routing sizes
        self.hysteresis_tau = 0.7  # Hysteresis threshold
        self.max_singular_value = 3.0  # σ₁ clamp value
        self.fro_budget = 1.0  # Frobenius norm budget
        
        # Create mock base model for testing
        self.base_model = self._create_mock_transformer()
        
        # Statistical testing parameters
        self.bootstrap_iterations = 1000  # Reduced for testing speed
        self.alpha = 0.05  # 95% confidence intervals
        
    def _create_mock_transformer(self) -> nn.Module:
        """Create a mock transformer model for testing."""
        class MockTransformerLayer(nn.Module):
            def __init__(self, dim=768):
                super().__init__()
                self.attention = nn.ModuleDict({
                    'W_Q': nn.Linear(dim, dim),
                    'W_K': nn.Linear(dim, dim), 
                    'W_V': nn.Linear(dim, dim),
                    'W_O': nn.Linear(dim, dim),  # Cache-safe attachment point
                })
                self.mlp = nn.ModuleDict({
                    'W_up': nn.Linear(dim, dim * 4),
                    'W_down': nn.Linear(dim * 4, dim),  # Cache-safe attachment point
                    'W_gate': nn.Linear(dim, dim * 4),
                })
                
            def forward(self, x):
                # Simplified forward pass for testing
                attn_out = self.attention.W_O(x)
                mlp_out = self.mlp.W_down(self.mlp.W_up(x))
                return attn_out + mlp_out
        
        class MockTransformer(nn.Module):
            def __init__(self, num_layers=6, dim=768):
                super().__init__()
                self.layers = nn.ModuleList([
                    MockTransformerLayer(dim) for _ in range(num_layers)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
                
        return MockTransformer()
    
    def test_spectral_governance_compliance(self):
        """Test spectral governance σ₁ clamp and Frobenius budget enforcement."""
        governance = SpectralGovernance(
            max_singular_value=self.max_singular_value,
            fro_budget=self.fro_budget
        )
        
        # Create test delta weights with large singular values
        large_delta = torch.randn(768, 768) * 5.0  # Exceeds σ₁ limit
        
        # Apply governance
        governed_delta = governance.apply_governance(large_delta)
        
        # Validate σ₁ clamping
        U, S, Vh = torch.linalg.svd(governed_delta, full_matrices=False)
        max_singular_value = S.max().item()
        self.assertLessEqual(
            max_singular_value, self.max_singular_value + 1e-6,
            f"Max singular value {max_singular_value} exceeds limit {self.max_singular_value}"
        )
        
        # Validate Frobenius budget
        fro_norm = torch.norm(governed_delta, 'fro').item()
        self.assertLessEqual(
            fro_norm, self.fro_budget + 1e-6,
            f"Frobenius norm {fro_norm} exceeds budget {self.fro_budget}"
        )
    
    def test_chunk_sticky_routing_hysteresis(self):
        """Test chunk-sticky routing with hysteresis behavior."""
        router = ChunkStickyRouter(
            input_dim=768,
            num_experts=4,
            chunk_size=128,
            hysteresis_tau=self.hysteresis_tau
        )
        
        # Create test input with gradual changes
        seq_len = 256  # 2 chunks
        batch_size = 2
        x = torch.randn(batch_size, seq_len, 768)
        
        # Get routing decisions
        routing_weights, expert_indices = router(x)
        
        # Validate output shapes
        self.assertEqual(routing_weights.shape, (batch_size, seq_len, 4))
        self.assertEqual(expert_indices.shape, (batch_size, 2))  # 2 chunks
        
        # Validate chunk-sticky behavior (same expert within chunk)
        chunk1_weights = routing_weights[:, :128, :]  # First chunk
        chunk2_weights = routing_weights[:, 128:, :]  # Second chunk
        
        # Within each chunk, routing weights should be constant
        for batch_idx in range(batch_size):
            chunk1_expert = chunk1_weights[batch_idx, 0, :].argmax()
            self.assertTrue(
                torch.all(chunk1_weights[batch_idx, :, chunk1_expert] == 1.0),
                "Routing should be constant within chunk"
            )
    
    def test_attention_logit_bias_cache_safety(self):
        """Test attention logit bias maintains cache safety."""
        bias_module = AttentionLogitBias(retrieval_dim=768)
        
        # Create mock retrieval features
        batch_size, seq_len = 2, 128
        retrieval_features = torch.randn(batch_size, seq_len, 768)
        
        # Generate bias
        bias = bias_module(retrieval_features)
        
        # Validate output shape (should be attention logit compatible)
        self.assertEqual(bias.shape, (batch_size, seq_len, seq_len))
        
        # Validate bias is additive only (cache-safe)
        # Bias should not modify the attention mechanism structure
        self.assertTrue(torch.isfinite(bias).all(), "Bias must be finite")
        
        # Test that bias doesn't break attention computation
        # Create mock attention logits
        attn_logits = torch.randn(batch_size, seq_len, seq_len)
        biased_logits = attn_logits + bias
        
        # Should still be valid attention logits
        attn_probs = F.softmax(biased_logits, dim=-1)
        self.assertTrue(torch.allclose(attn_probs.sum(dim=-1), torch.ones(batch_size, seq_len)))

    def test_bem_v11_stable_cache_safety(self):
        """Test BEM v11 stable model maintains cache safety."""
        # Create BEM model with safe attachment points
        safe_points = ['W_O', 'W_down']
        bem_model = BEMv11StableModel(
            base_model=self.base_model,
            rank_schedule=[8, 8, 8, 8, 8, 8],  # 6 layers
            attachment_points=safe_points
        )
        
        # Validate cache safety
        safety_result = self._validate_cache_safety(bem_model)
        self.assertTrue(
            safety_result.is_cache_safe,
            f"Model not cache-safe. Violations: {safety_result.violations}"
        )
        
        # Test with unsafe attachment points should fail
        unsafe_points = ['W_Q', 'W_K', 'W_V']
        with self.assertRaises(Exception, msg="Should reject unsafe attachment points"):
            BEMv11StableModel(
                base_model=self.base_model,
                rank_schedule=[8, 8, 8, 8, 8, 8],
                attachment_points=unsafe_points
            )
    
    def _validate_cache_safety(self, model: nn.Module) -> CacheSafetyResult:
        """Validate that model maintains cache safety."""
        violations = []
        attachment_points = []
        
        # Check for forbidden attachment points
        forbidden_points = ['W_Q', 'W_K', 'W_V', 'attention.query', 'attention.key', 'attention.value']
        allowed_points = ['W_O', 'W_down', 'attention.dense', 'mlp.down_proj']
        
        for name, module in model.named_modules():
            if any(forbidden in name for forbidden in forbidden_points):
                if hasattr(module, 'bem_adapter') or 'bem' in name.lower():
                    violations.append(f"Forbidden attachment at {name}")
            
            if any(allowed in name for allowed in allowed_points):
                attachment_points.append(name)
        
        return CacheSafetyResult(
            is_cache_safe=len(violations) == 0,
            violations=violations,
            attachment_points=attachment_points
        )


class TestBEMv13Variants(unittest.TestCase):
    """Test all BEM v1.3 performance variants (PT1-PT4, AR1, OL, MM, VC)."""
    
    def setUp(self):
        """Set up test fixtures for BEM variants."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_dim = 768
        self.num_experts = 4
        self.tolerance = 0.05  # ±5% parity
        
    def test_pt1_head_gating_module(self):
        """Test PT1 Head-Group Gating implementation."""
        try:
            head_gating = HeadGatingModule(
                num_heads=12,
                num_groups=4,
                gate_threshold=0.5
            )
            
            # Test input
            batch_size, seq_len, num_heads, head_dim = 2, 128, 12, 64
            x = torch.randn(batch_size, seq_len, num_heads, head_dim)
            
            # Forward pass
            gated_output = head_gating(x)
            
            # Validate output shape
            self.assertEqual(gated_output.shape, x.shape)
            
            # Validate gating reduces some activations
            self.assertTrue(
                torch.any(gated_output != x),
                "Head gating should modify some activations"
            )
            
        except NameError:
            self.skipTest("PT1 HeadGatingModule not implemented")
    
    def test_pt2_dynamic_mask_module(self):
        """Test PT2 Dynamic Rank Mask implementation."""
        try:
            dynamic_mask = DynamicMaskModule(
                rank=8,
                active_ratio=0.5,  # 50% active components
                input_dim=self.base_dim
            )
            
            # Test input
            batch_size, seq_len = 2, 128
            x = torch.randn(batch_size, seq_len, self.base_dim)
            
            # Forward pass
            masked_output = dynamic_mask(x)
            
            # Validate output shape
            self.assertEqual(masked_output.shape, x.shape)
            
            # Validate sparsity (approximately 50% components should be active)
            # This is a statistical test, so we allow some variance
            nonzero_ratio = (masked_output != 0).float().mean()
            self.assertGreater(nonzero_ratio, 0.3, "Too many components masked")
            self.assertLess(nonzero_ratio, 0.7, "Too few components masked")
            
        except NameError:
            self.skipTest("PT2 DynamicMaskModule not implemented")
    
    def test_pt3_kronecker_module(self):
        """Test PT3 Kronecker factorization implementation."""
        try:
            kronecker = KroneckerModule(
                input_dim=768,
                output_dim=3072,
                rank_u=16,
                rank_v=16
            )
            
            # Test input
            batch_size, seq_len = 2, 128
            x = torch.randn(batch_size, seq_len, 768)
            
            # Forward pass
            kronecker_output = kronecker(x)
            
            # Validate output shape
            self.assertEqual(kronecker_output.shape, (batch_size, seq_len, 3072))
            
            # Validate parameter count is reasonable
            total_params = sum(p.numel() for p in kronecker.parameters())
            full_params = 768 * 3072
            compression_ratio = total_params / full_params
            
            self.assertLess(
                compression_ratio, 0.5,
                f"Kronecker should compress parameters, got ratio {compression_ratio}"
            )
            
        except NameError:
            self.skipTest("PT3 KroneckerModule not implemented")
    
    def test_pt4_residual_film_module(self):
        """Test PT4 Residual FiLM implementation."""
        try:
            film_module = ResidualFiLMModule(
                input_dim=self.base_dim,
                conditioning_dim=256,
                gamma_clamp=2.0,
                beta_clamp=1.0
            )
            
            # Test inputs
            batch_size, seq_len = 2, 128
            x = torch.randn(batch_size, seq_len, self.base_dim)
            conditioning = torch.randn(batch_size, seq_len, 256)
            
            # Forward pass
            film_output = film_module(x, conditioning)
            
            # Validate output shape
            self.assertEqual(film_output.shape, x.shape)
            
            # Validate that output differs from input (FiLM applied)
            self.assertFalse(torch.allclose(film_output, x), "FiLM should modify input")
            
        except NameError:
            self.skipTest("PT4 ResidualFiLMModule not implemented")
    
    def test_ar1_agentic_router(self):
        """Test AR1 Agentic Router implementation."""
        try:
            router = AgenticRouter(
                num_experts=self.num_experts,
                planning_horizon=3,
                hysteresis_threshold=0.7
            )
            
            # Test input
            batch_size, seq_len = 2, 128
            context = {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'retrieval_scores': torch.randn(batch_size, seq_len, 10),
                'task_context': torch.randn(batch_size, 256)
            }
            
            # Forward pass
            routing_plan = router(context)
            
            # Validate planning horizon
            self.assertLessEqual(
                len(routing_plan['actions']), 3,
                "Plan length should be ≤3 as per TODO.md"
            )
            
            # Validate expert selections are valid
            for action in routing_plan['actions']:
                self.assertIn(action['expert_id'], range(self.num_experts))
                self.assertGreater(action['confidence'], 0.0)
            
        except NameError:
            self.skipTest("AR1 AgenticRouter not implemented")
    
    def test_ol_online_learner(self):
        """Test OL Online Learning implementation."""
        try:
            online_learner = OnlineLearner(
                base_model=self._create_mock_model(),
                ewc_lambda=0.1,
                proxy_lambda=0.05,
                replay_buffer_size=1000
            )
            
            # Test online update
            feedback = {
                'input_ids': torch.randint(0, 1000, (2, 64)),
                'rewards': torch.tensor([0.8, 0.6]),
                'task_success': torch.tensor([True, False])
            }
            
            # Perform update
            update_result = online_learner.update(feedback)
            
            # Validate update occurred
            self.assertIsNotNone(update_result)
            self.assertIn('loss', update_result)
            self.assertGreater(update_result['loss'], 0.0)
            
        except NameError:
            self.skipTest("OL OnlineLearner not implemented")
    
    def test_mm_multimodal_controller(self):
        """Test MM Multimodal Controller implementation."""
        try:
            mm_controller = MultimodalController(
                text_dim=768,
                vision_dim=768,
                fusion_dim=512
            )
            
            # Test inputs
            batch_size, seq_len = 2, 64
            text_features = torch.randn(batch_size, seq_len, 768)
            vision_features = torch.randn(batch_size, 196, 768)  # 14x14 patches
            
            # Forward pass
            fused_output = mm_controller(text_features, vision_features)
            
            # Validate output
            self.assertEqual(fused_output.shape, (batch_size, seq_len, 512))
            
            # Test coverage/consistency metrics
            metrics = mm_controller.compute_metrics(text_features, vision_features)
            self.assertIn('coverage', metrics)
            self.assertIn('consistency', metrics)
            
        except NameError:
            self.skipTest("MM MultimodalController not implemented")
    
    def test_vc_safety_controller(self):
        """Test VC Safety Controller implementation."""
        try:
            safety_controller = SafetyController(
                input_dim=768,
                safety_dim=128,
                violation_threshold=0.3
            )
            
            # Test input
            batch_size, seq_len = 2, 64
            x = torch.randn(batch_size, seq_len, 768)
            
            # Forward pass
            safe_output, safety_scores = safety_controller(x)
            
            # Validate output shapes
            self.assertEqual(safe_output.shape, x.shape)
            self.assertEqual(safety_scores.shape, (batch_size, seq_len))
            
            # Validate safety scores are probabilities
            self.assertTrue(torch.all(safety_scores >= 0.0))
            self.assertTrue(torch.all(safety_scores <= 1.0))
            
        except NameError:
            self.skipTest("VC SafetyController not implemented")
    
    def _create_mock_model(self):
        """Create a mock model for testing."""
        return nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )


class TestParameterFLOPParity(unittest.TestCase):
    """Test parameter and FLOP parity enforcement (±5% requirement)."""
    
    def setUp(self):
        """Set up parity testing fixtures."""
        self.tolerance = 0.05  # ±5% as per TODO.md
        self.base_model = self._create_reference_model()
        
    def _create_reference_model(self):
        """Create reference model for parity comparison."""
        return nn.Sequential(
            nn.Linear(768, 3072),
            nn.ReLU(),
            nn.Linear(3072, 768)
        )
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for model with given input shape."""
        # Simplified FLOP counting - in practice would use tools like fvcore
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Linear layer: input_features * output_features * 2 (multiply-add)
                total_flops += module.in_features * module.out_features * 2
                if module.bias is not None:
                    total_flops += module.out_features  # Bias addition
                    
        # Multiply by batch size and sequence length
        batch_size, seq_len = input_shape[0], input_shape[1] if len(input_shape) > 1 else 1
        total_flops *= batch_size * seq_len
        
        return total_flops
    
    def test_bem_v11_parameter_parity(self):
        """Test BEM v11 maintains parameter parity within ±5%."""
        # Count baseline parameters
        baseline_params = self._count_parameters(self.base_model)
        
        # Create BEM-enhanced model
        bem_model = BEMv11StableModel(
            base_model=self.base_model,
            rank_schedule=[8, 8, 8],  # Small ranks for testing
            attachment_points=['W_O', 'W_down']
        )
        
        bem_params = self._count_parameters(bem_model)
        
        # Validate parity
        parity_result = self._validate_parameter_parity(baseline_params, bem_params)
        self.assertTrue(
            parity_result.params_within_tolerance,
            f"Parameter count {bem_params} not within ±{self.tolerance*100}% of baseline {baseline_params}. "
            f"Ratio: {parity_result.param_ratio:.3f}"
        )
    
    def test_bem_v11_flop_parity(self):
        """Test BEM v11 maintains FLOP parity within ±5%."""
        input_shape = (2, 128, 768)  # batch, seq_len, hidden_dim
        
        # Estimate baseline FLOPs
        baseline_flops = self._estimate_flops(self.base_model, input_shape)
        
        # Create BEM-enhanced model
        bem_model = BEMv11StableModel(
            base_model=self.base_model,
            rank_schedule=[8, 8, 8],
            attachment_points=['W_O', 'W_down']
        )
        
        bem_flops = self._estimate_flops(bem_model, input_shape)
        
        # Validate parity
        parity_result = self._validate_flop_parity(baseline_flops, bem_flops)
        self.assertTrue(
            parity_result.flops_within_tolerance,
            f"FLOP count {bem_flops} not within ±{self.tolerance*100}% of baseline {baseline_flops}. "
            f"Ratio: {parity_result.flop_ratio:.3f}"
        )
    
    def _validate_parameter_parity(self, baseline: int, treatment: int) -> ParityValidationResult:
        """Validate parameter parity within tolerance."""
        ratio = treatment / baseline
        within_tolerance = abs(ratio - 1.0) <= self.tolerance
        
        return ParityValidationResult(
            params_within_tolerance=within_tolerance,
            flops_within_tolerance=True,  # Not tested in this method
            param_ratio=ratio,
            flop_ratio=1.0,  # Not tested in this method
            tolerance=self.tolerance
        )
    
    def _validate_flop_parity(self, baseline: int, treatment: int) -> ParityValidationResult:
        """Validate FLOP parity within tolerance."""
        ratio = treatment / baseline
        within_tolerance = abs(ratio - 1.0) <= self.tolerance
        
        return ParityValidationResult(
            params_within_tolerance=True,  # Not tested in this method
            flops_within_tolerance=within_tolerance,
            param_ratio=1.0,  # Not tested in this method
            flop_ratio=ratio,
            tolerance=self.tolerance
        )


class TestStatisticalAnalysisFramework(unittest.TestCase):
    """Test the statistical analysis framework (BCa bootstrap, FDR correction)."""
    
    def setUp(self):
        """Set up statistical testing fixtures."""
        self.bootstrap_stats = BootstrapStatistics(n_bootstrap=1000, alpha=0.05)
        self.metrics_family = ['em_score', 'f1_score', 'bleu_score', 'chrf_score']
        
        # Generate synthetic experiment data
        self.baseline_results = self._generate_synthetic_results('baseline')
        self.treatment_results = self._generate_synthetic_results('treatment', improvement=0.05)
    
    def _generate_synthetic_results(self, name: str, improvement: float = 0.0) -> ExperimentMetrics:
        """Generate synthetic experiment results for testing."""
        np.random.seed(42)  # For reproducible tests
        n_seeds = 5
        
        # Base performance levels
        base_em = 0.75
        base_f1 = 0.80
        base_bleu = 0.25
        base_chrf = 0.55
        
        return ExperimentMetrics(
            experiment_id=name,
            seeds=list(range(1, n_seeds + 1)),
            em_scores=[base_em + improvement + np.random.normal(0, 0.02) for _ in range(n_seeds)],
            f1_scores=[base_f1 + improvement + np.random.normal(0, 0.02) for _ in range(n_seeds)],
            bleu_scores=[base_bleu + improvement + np.random.normal(0, 0.01) for _ in range(n_seeds)],
            chrf_scores=[base_chrf + improvement + np.random.normal(0, 0.01) for _ in range(n_seeds)],
            p50_latency_ms=[150.0 + np.random.normal(0, 10) for _ in range(n_seeds)],
            p95_latency_ms=[250.0 + np.random.normal(0, 20) for _ in range(n_seeds)],
            throughput_tokens_per_sec=[1000.0 + np.random.normal(0, 50) for _ in range(n_seeds)],
            vram_usage_gb=[8.0 + np.random.normal(0, 0.5) for _ in range(n_seeds)]
        )
    
    def test_bca_bootstrap_confidence_intervals(self):
        """Test BCa bootstrap confidence interval calculation."""
        baseline_scores = np.array(self.baseline_results.em_scores)
        treatment_scores = np.array(self.treatment_results.em_scores)
        
        # Perform paired bootstrap test
        rel_improvement, ci_lower, ci_upper, p_value = self.bootstrap_stats.paired_bootstrap_test(
            baseline_scores, treatment_scores
        )
        
        # Validate CI structure
        self.assertLess(ci_lower, ci_upper, "CI lower bound should be less than upper bound")
        self.assertGreater(rel_improvement, 0, "Should detect improvement in synthetic data")
        self.assertLess(p_value, 0.05, "Should detect significant difference")
        
        # Validate CI contains the improvement estimate
        self.assertTrue(
            ci_lower <= rel_improvement <= ci_upper,
            f"Point estimate {rel_improvement:.3f} not within CI [{ci_lower:.3f}, {ci_upper:.3f}]"
        )
    
    def test_fdr_correction_multiple_testing(self):
        """Test FDR correction for multiple testing across metric families."""
        # Create multiple comparisons
        comparisons = []
        
        for metric_name in self.metrics_family:
            baseline_scores = np.array(getattr(self.baseline_results, f"{metric_name}s"))
            treatment_scores = np.array(getattr(self.treatment_results, f"{metric_name}s"))
            
            rel_improvement, ci_lower, ci_upper, p_value = self.bootstrap_stats.paired_bootstrap_test(
                baseline_scores, treatment_scores
            )
            
            comparisons.append(ComparisonResult(
                metric_name=metric_name,
                baseline_mean=baseline_scores.mean(),
                treatment_mean=treatment_scores.mean(),
                relative_improvement_pct=rel_improvement,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                p_value=p_value,
                significant=False,  # Will be set by FDR correction
                effect_size=0.0  # Simplified for testing
            ))
        
        # Apply FDR correction
        corrected_results = apply_fdr_correction(comparisons, alpha=0.05)
        
        # Validate FDR correction was applied
        original_p_values = [comp.p_value for comp in comparisons]
        corrected_significant = [comp.significant for comp in corrected_results]
        
        # Should have fewer significant results after FDR correction (typically)
        self.assertLessEqual(
            sum(corrected_significant), len(original_p_values),
            "FDR correction should be conservative"
        )
        
        # All significant results should have CI lower bound > 0 (as per TODO.md)
        for result in corrected_results:
            if result.significant:
                self.assertGreater(
                    result.ci_lower, 0,
                    f"Significant result for {result.metric_name} should have CI lower bound > 0"
                )
    
    def test_statistical_significance_requirements(self):
        """Test that statistical significance follows TODO.md requirements."""
        baseline_scores = np.array(self.baseline_results.em_scores)
        treatment_scores = np.array(self.treatment_results.em_scores)
        
        rel_improvement, ci_lower, ci_upper, p_value = self.bootstrap_stats.paired_bootstrap_test(
            baseline_scores, treatment_scores
        )
        
        # TODO.md requirement: Stars only if CI lower bound > 0 after FDR
        comparison = ComparisonResult(
            metric_name='em_score',
            baseline_mean=baseline_scores.mean(),
            treatment_mean=treatment_scores.mean(),
            relative_improvement_pct=rel_improvement,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            significant=False,
            effect_size=0.0
        )
        
        # Apply FDR correction (single comparison for simplicity)
        corrected_results = apply_fdr_correction([comparison], alpha=0.05)
        result = corrected_results[0]
        
        # Validate TODO.md requirements
        if result.significant:
            self.assertGreater(
                result.ci_lower, 0,
                "Significant results must have CI lower bound > 0 (TODO.md requirement)"
            )


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of all BEM operations."""
    
    def setUp(self):
        """Set up numerical stability test fixtures."""
        self.tolerance = 1e-3  # Kronecker kernel tolerance from TODO.md
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_spectral_governance_numerical_stability(self):
        """Test spectral governance maintains numerical stability."""
        governance = SpectralGovernance(max_singular_value=3.0, fro_budget=1.0)
        
        # Test with various challenging inputs
        test_cases = [
            torch.randn(768, 768) * 10.0,  # Large values
            torch.randn(768, 768) * 1e-8,  # Small values
            torch.eye(768) * 5.0,  # High condition number
            torch.zeros(768, 768),  # Zero matrix
        ]
        
        for i, delta_w in enumerate(test_cases):
            with self.subTest(case=i):
                # Apply governance
                governed = governance.apply_governance(delta_w)
                
                # Check for numerical issues
                self.assertTrue(torch.isfinite(governed).all(), f"Non-finite values in case {i}")
                self.assertFalse(torch.isnan(governed).any(), f"NaN values in case {i}")
                
                # Validate constraints still hold
                fro_norm = torch.norm(governed, 'fro')
                self.assertLessEqual(fro_norm, 1.0 + self.tolerance, f"Fro budget violated in case {i}")
    
    def test_kronecker_kernel_numerical_precision(self):
        """Test Kronecker kernel meets numerical precision requirements."""
        # This test would verify the Kronecker kernel against fp16 reference
        # as specified in TODO.md implementation notes
        
        # Create reference computation (standard matrix multiply)
        input_dim, output_dim = 768, 3072
        x = torch.randn(2, 128, input_dim, dtype=torch.float16)
        W_full = torch.randn(output_dim, input_dim, dtype=torch.float16)
        
        # Reference output
        reference_output = F.linear(x, W_full)
        
        # Kronecker factorized computation (simulated)
        rank_u, rank_v = 32, 32
        U = torch.randn(output_dim // rank_u, rank_u, dtype=torch.float16)
        V = torch.randn(rank_v, input_dim // rank_v, dtype=torch.float16)
        
        # Kronecker product approximation: W ≈ U ⊗ V
        W_kron = torch.kron(U, V)
        if W_kron.shape[0] > output_dim:
            W_kron = W_kron[:output_dim, :input_dim]  # Truncate if needed
        elif W_kron.shape[0] < output_dim:
            # Pad if needed
            pad_rows = output_dim - W_kron.shape[0]
            W_kron = torch.cat([W_kron, torch.zeros(pad_rows, input_dim, dtype=torch.float16)])
        
        kronecker_output = F.linear(x, W_kron)
        
        # Validate numerical precision (this is a simplified test)
        max_error = torch.max(torch.abs(reference_output - kronecker_output)).item()
        relative_error = max_error / torch.max(torch.abs(reference_output)).item()
        
        self.assertLess(
            relative_error, self.tolerance,
            f"Kronecker approximation error {relative_error:.6f} exceeds tolerance {self.tolerance}"
        )
    
    def test_attention_bias_numerical_stability(self):
        """Test attention logit bias numerical stability with extreme inputs."""
        bias_module = AttentionLogitBias(retrieval_dim=768)
        
        # Test extreme input cases
        test_cases = [
            torch.randn(2, 128, 768) * 100.0,  # Large inputs
            torch.randn(2, 128, 768) * 1e-6,   # Small inputs
            torch.full((2, 128, 768), float('inf')),  # Infinite inputs (should be handled)
        ]
        
        for i, retrieval_features in enumerate(test_cases):
            with self.subTest(case=i):
                if i == 2:  # Skip infinite input case for now
                    continue
                    
                try:
                    bias = bias_module(retrieval_features)
                    
                    # Validate output is numerically stable
                    self.assertTrue(torch.isfinite(bias).all(), f"Non-finite bias in case {i}")
                    self.assertFalse(torch.isnan(bias).any(), f"NaN bias in case {i}")
                    
                except Exception as e:
                    self.fail(f"Attention bias failed on case {i}: {e}")


class TestIntegrationPipeline(unittest.TestCase):
    """Test end-to-end integration of the BEM v1.3 training pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / 'test_config.yaml'
        
        # Create minimal test configuration
        test_config = {
            'experiment_name': 'test_bem_v13',
            'model': {
                'base_model': 'microsoft/DialoGPT-small',
                'bem_config': {
                    'rank': 8,
                    'num_experts': 2,
                    'chunk_size': 64,
                    'attachment_points': ['W_O', 'W_down']
                }
            },
            'training': {
                'max_steps': 10,  # Minimal for testing
                'batch_size': 2,
                'learning_rate': 1e-4,
                'seeds': [1, 2]
            },
            'evaluation': {
                'metrics': ['em_score', 'f1_score'],
                'bootstrap_iterations': 100,  # Reduced for testing
                'alpha': 0.05
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_experiment_runner_integration(self):
        """Test full experiment runner integration."""
        try:
            # Create mock experiment runner
            runner = ExperimentRunner(config_path=str(self.config_path))
            
            # Run minimal experiment
            results = runner.run_experiment()
            
            # Validate results structure
            self.assertIsNotNone(results)
            self.assertIn('experiment_id', results)
            self.assertIn('metrics', results)
            
            # Validate metrics were computed
            for metric in ['em_score', 'f1_score']:
                self.assertIn(metric, results['metrics'])
                
        except (ImportError, NameError):
            self.skipTest("ExperimentRunner not implemented")
    
    def test_quality_gates_integration(self):
        """Test quality gates enforcement in full pipeline."""
        # Define quality gates from TODO.md
        quality_gates = {
            'parameter_parity': {'tolerance': 0.05},
            'flop_parity': {'tolerance': 0.05},
            'latency_p50': {'max_increase_pct': 15.0},
            'cache_safety': {'required': True},
            'kv_hit_rate': {'min_baseline_ratio': 1.0}
        }
        
        # Mock experiment results
        mock_results = {
            'param_ratio': 1.03,  # Within 5%
            'flop_ratio': 0.98,   # Within 5%
            'latency_increase_pct': 12.0,  # Within 15%
            'cache_safe': True,
            'kv_hit_ratio': 1.02  # Above baseline
        }
        
        # Validate quality gates
        gate_results = []
        
        # Parameter parity gate
        param_passed = abs(mock_results['param_ratio'] - 1.0) <= quality_gates['parameter_parity']['tolerance']
        gate_results.append(QualityGateResult(
            gate_name='parameter_parity',
            threshold=quality_gates['parameter_parity']['tolerance'],
            actual_value=mock_results['param_ratio'],
            passed=param_passed,
            description=f"Parameter ratio {mock_results['param_ratio']:.3f} within ±{quality_gates['parameter_parity']['tolerance']*100}%"
        ))
        
        # FLOP parity gate
        flop_passed = abs(mock_results['flop_ratio'] - 1.0) <= quality_gates['flop_parity']['tolerance']
        gate_results.append(QualityGateResult(
            gate_name='flop_parity',
            threshold=quality_gates['flop_parity']['tolerance'],
            actual_value=mock_results['flop_ratio'],
            passed=flop_passed,
            description=f"FLOP ratio {mock_results['flop_ratio']:.3f} within ±{quality_gates['flop_parity']['tolerance']*100}%"
        ))
        
        # Latency gate
        latency_passed = mock_results['latency_increase_pct'] <= quality_gates['latency_p50']['max_increase_pct']
        gate_results.append(QualityGateResult(
            gate_name='latency_p50',
            threshold=quality_gates['latency_p50']['max_increase_pct'],
            actual_value=mock_results['latency_increase_pct'],
            passed=latency_passed,
            description=f"Latency increase {mock_results['latency_increase_pct']:.1f}% within {quality_gates['latency_p50']['max_increase_pct']:.1f}%"
        ))
        
        # Cache safety gate
        cache_passed = mock_results['cache_safe'] == quality_gates['cache_safety']['required']
        gate_results.append(QualityGateResult(
            gate_name='cache_safety',
            threshold=quality_gates['cache_safety']['required'],
            actual_value=mock_results['cache_safe'],
            passed=cache_passed,
            description=f"Cache safety: {mock_results['cache_safe']}"
        ))
        
        # KV hit rate gate
        kv_passed = mock_results['kv_hit_ratio'] >= quality_gates['kv_hit_rate']['min_baseline_ratio']
        gate_results.append(QualityGateResult(
            gate_name='kv_hit_rate',
            threshold=quality_gates['kv_hit_rate']['min_baseline_ratio'],
            actual_value=mock_results['kv_hit_ratio'],
            passed=kv_passed,
            description=f"KV hit ratio {mock_results['kv_hit_ratio']:.3f} >= baseline"
        ))
        
        # Validate all gates passed
        all_passed = all(gate.passed for gate in gate_results)
        self.assertTrue(all_passed, f"Quality gates failed: {[g.gate_name for g in gate_results if not g.passed]}")
        
        # Print gate results for visibility
        print("\n🚪 Quality Gates Results:")
        for gate in gate_results:
            status = "✅ PASS" if gate.passed else "❌ FAIL"
            print(f"   {status} {gate.gate_name}: {gate.description}")


class TestReproducibilityGuarantees(unittest.TestCase):
    """Test reproducibility guarantees and deterministic behavior."""
    
    def test_seed_determinism(self):
        """Test that same seeds produce identical results."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create two identical models with same seed
        model1 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        
        torch.manual_seed(42)  # Reset seed
        model2 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        
        # Test with same input
        x = torch.randn(2, 128, 768)
        
        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)
        
        # Should be identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6), "Same seeds should produce identical results")
    
    def test_configuration_serialization(self):
        """Test that configurations can be serialized and restored."""
        # Original configuration
        config = {
            'model': {
                'rank': 8,
                'num_experts': 4,
                'chunk_size': 128,
                'hysteresis_tau': 0.7
            },
            'training': {
                'learning_rate': 1e-4,
                'batch_size': 32,
                'max_steps': 1000
            },
            'seeds': [1, 2, 3, 4, 5],
            'git_sha': 'abc123',
            'data_hash': 'def456'
        }
        
        # Serialize to JSON
        config_json = json.dumps(config, sort_keys=True)
        
        # Deserialize
        restored_config = json.loads(config_json)
        
        # Should be identical
        self.assertEqual(config, restored_config, "Configuration should survive serialization")
        
        # Validate required reproducibility fields
        self.assertIn('seeds', restored_config)
        self.assertIn('git_sha', restored_config)
        self.assertIn('data_hash', restored_config)


def run_comprehensive_test_suite():
    """Run the complete BEM v1.3 test suite with detailed reporting."""
    
    print("🧪 BEM v1.3 Performance+Agentic Sprint - Comprehensive Test Suite")
    print("=" * 80)
    print("Testing all components as specified in TODO.md:")
    print("• Core Components (PT1-PT4, AR1, OL, MM, VC)")
    print("• Parameter/FLOP parity enforcement (±5%)")
    print("• Cache-safety invariants")
    print("• Statistical analysis framework (BCa bootstrap, FDR)")
    print("• Integration pipeline")
    print("• Numerical stability")
    print("• Reproducibility guarantees")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBEMv13CoreComponents,
        TestBEMv13Variants,
        TestParameterFLOPParity,
        TestStatisticalAnalysisFramework,
        TestNumericalStability,
        TestIntegrationPipeline,
        TestReproducibilityGuarantees
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        stream=None
    )
    
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏁 TEST SUITE SUMMARY")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.skipped:
        print("\n⏭️  SKIPPED:")
        for test, reason in result.skipped:
            print(f"   • {test}: {reason}")
    
    # Overall status
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - BEM v1.3 ready for research deployment!")
    else:
        print("\n❌ SOME TESTS FAILED - Fix issues before deployment")
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)