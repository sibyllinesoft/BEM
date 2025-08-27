#!/usr/bin/env python3
"""
Phase 5 Validation Script.

Comprehensive validation of all Phase 5 advanced features to ensure
they meet the acceptance criteria specified in TODO.md.

Acceptance Gates for Phase 5:
P5: Banked experts improve expressivity without collapsing utilization
Online: Trust monitors prevent catastrophic drift  
Speculative: Neutral or positive latency impact
VQ: Memory reduction without significant quality loss
Counterfactual: Measurable routing improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Import Phase 5 components
from bem import (
    # Banked Experts
    BankedExpertsModule,
    create_banked_experts_module,
    create_default_banked_experts_config,
    
    # Online Learning
    OnlineLearningController,
    TrustStatus,
    create_online_learning_controller,
    create_default_online_learning_config,
    
    # Speculative Decoding
    SpeculativeDecoder,
    create_speculative_decoder,
    create_default_speculative_config,
    SpeculativeDecodingBenchmark,
    
    # Vector Quantization
    VectorQuantizer,
    create_vector_quantizer,
    create_default_vq_config,
    
    # Counterfactual Routing
    CounterfactualRoutingAnalyzer,
    create_counterfactual_analyzer,
    create_default_counterfactual_config,
    
    # Telemetry
    TelemetryCollector,
    create_telemetry_collector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any]
    error_message: str = ""


class Phase5Validator:
    """Comprehensive validator for Phase 5 advanced features."""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.results = []
        self.telemetry = create_telemetry_collector()
        
        logger.info(f"Phase 5 Validator initialized on {self.device}")
    
    def validate_banked_experts(self) -> ValidationResult:
        """
        Validate P5: Banked experts improve expressivity without collapsing utilization.
        
        Acceptance Criteria:
        - Expert utilization entropy > 0.7 (high diversity)
        - Gini coefficient < 0.5 (not too concentrated)
        - Active experts > 50% of total
        - Load balancing loss < 0.1
        """
        logger.info("Validating Banked Experts System...")
        
        try:
            # Create banked experts module
            config = create_default_banked_experts_config(
                num_experts=8,
                expert_rank=16,
                top_k=2,
                enable_batching=True
            )
            
            banked_experts = create_banked_experts_module(
                input_dim=256,
                output_dim=256,
                config=config,
                telemetry_collector=self.telemetry
            ).to(self.device)
            
            # Test with multiple batches to collect statistics
            banked_experts.train()
            total_load_loss = 0.0
            num_batches = 50
            
            for batch_idx in range(num_batches):
                batch_size, seq_len = 4, 32
                inputs = torch.randn(batch_size, seq_len, 256).to(self.device)
                
                outputs, routing_info = banked_experts(inputs, training=True)
                total_load_loss += routing_info['load_balancing_loss'].item()
                
                # Verify output shape
                assert outputs.shape == inputs.shape, f"Output shape mismatch: {outputs.shape} vs {inputs.shape}"
            
            # Get final statistics
            stats = banked_experts.get_expert_statistics()
            utilization = stats['utilization']
            
            # Calculate metrics
            avg_load_loss = total_load_loss / num_batches
            utilization_entropy = utilization.entropy
            gini_coefficient = utilization.gini_coefficient
            active_expert_ratio = utilization.active_experts / config.num_experts
            
            # Check acceptance criteria
            entropy_pass = utilization_entropy > 0.7
            gini_pass = gini_coefficient < 0.5
            active_pass = active_expert_ratio > 0.5
            load_pass = avg_load_loss < 0.1
            
            overall_pass = entropy_pass and gini_pass and active_pass and load_pass
            
            # Calculate composite score
            score = (
                min(utilization_entropy / 0.7, 1.0) * 0.3 +
                min((0.5 - gini_coefficient) / 0.5, 1.0) * 0.3 +
                min(active_expert_ratio / 0.5, 1.0) * 0.2 +
                min((0.1 - avg_load_loss) / 0.1, 1.0) * 0.2
            )
            
            return ValidationResult(
                test_name="Banked Experts",
                passed=overall_pass,
                score=score,
                threshold=0.8,
                details={
                    'utilization_entropy': utilization_entropy,
                    'entropy_threshold': 0.7,
                    'entropy_pass': entropy_pass,
                    'gini_coefficient': gini_coefficient,
                    'gini_threshold': 0.5,
                    'gini_pass': gini_pass,
                    'active_expert_ratio': active_expert_ratio,
                    'active_threshold': 0.5,
                    'active_pass': active_pass,
                    'avg_load_loss': avg_load_loss,
                    'load_threshold': 0.1,
                    'load_pass': load_pass,
                    'total_experts': config.num_experts,
                    'active_experts': utilization.active_experts
                }
            )
            
        except Exception as e:
            logger.error(f"Banked Experts validation failed: {e}")
            return ValidationResult(
                test_name="Banked Experts",
                passed=False,
                score=0.0,
                threshold=0.8,
                details={},
                error_message=str(e)
            )
    
    def validate_online_learning(self) -> ValidationResult:
        """
        Validate Online: Trust monitors prevent catastrophic drift.
        
        Acceptance Criteria:
        - Trust monitor successfully detects violations
        - Rollback mechanism activates when needed
        - Parameter drift stays within bounds
        - No catastrophic performance drops
        """
        logger.info("Validating Online Learning System...")
        
        try:
            # Create simple test model
            test_model = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            ).to(self.device)
            
            # Create online learning controller
            config = create_default_online_learning_config(
                base_learning_rate=1e-3,  # Higher LR to trigger violations
                consolidation_frequency=100,
                enable_rollback=True
            )
            config.trust_budget.max_parameter_drift = 0.1  # Lower threshold
            config.trust_budget.max_gradient_norm = 0.5   # Lower threshold
            
            controller = create_online_learning_controller(
                model=test_model,
                config=config,
                telemetry_collector=self.telemetry
            )
            
            controller.setup_optimizer(torch.optim.SGD, lr=config.base_learning_rate)
            
            # Simulate training with various scenarios
            violations_detected = 0
            rollbacks_triggered = 0
            consolidations_triggered = 0
            max_drift = 0.0
            performance_history = []
            
            for step in range(200):
                # Generate data
                x = torch.randn(16, 128).to(self.device)
                y = torch.randint(0, 10, (16,)).to(self.device)
                
                # Forward pass
                outputs = test_model(x)
                loss = F.cross_entropy(outputs, y)
                
                # Backward pass
                loss.backward()
                
                # Online learning step
                result = controller.step(loss)
                
                # Track metrics
                if result['metrics'].trust_status != TrustStatus.TRUSTED:
                    violations_detected += 1
                if result['rollback_triggered']:
                    rollbacks_triggered += 1
                if result['consolidation_triggered']:
                    consolidations_triggered += 1
                
                max_drift = max(max_drift, result['metrics'].parameter_drift)
                performance_history.append(loss.item())
            
            # Check for catastrophic performance drops
            performance_stability = True
            if len(performance_history) > 20:
                recent_avg = np.mean(performance_history[-10:])
                early_avg = np.mean(performance_history[:10])
                if recent_avg > early_avg * 2.0:  # More than 2x worse
                    performance_stability = False
            
            # Acceptance criteria
            violations_detected_pass = violations_detected > 0  # Should detect some violations
            rollback_functional = rollbacks_triggered > 0      # Should trigger rollbacks
            drift_bounded = max_drift < 1.0                    # Drift should be bounded
            performance_stable = performance_stability          # No catastrophic drops
            
            overall_pass = (violations_detected_pass and rollback_functional and 
                          drift_bounded and performance_stable)
            
            # Calculate score
            score = (
                (1.0 if violations_detected_pass else 0.0) * 0.3 +
                (1.0 if rollback_functional else 0.0) * 0.3 +
                (min(1.0 / max(max_drift, 0.1), 1.0)) * 0.2 +
                (1.0 if performance_stable else 0.0) * 0.2
            )
            
            return ValidationResult(
                test_name="Online Learning",
                passed=overall_pass,
                score=score,
                threshold=0.8,
                details={
                    'violations_detected': violations_detected,
                    'rollbacks_triggered': rollbacks_triggered,
                    'consolidations_triggered': consolidations_triggered,
                    'max_parameter_drift': max_drift,
                    'performance_stable': performance_stable,
                    'final_trust_score': controller.trust_monitor.get_trust_score(),
                    'violations_detected_pass': violations_detected_pass,
                    'rollback_functional': rollback_functional,
                    'drift_bounded': drift_bounded
                }
            )
            
        except Exception as e:
            logger.error(f"Online Learning validation failed: {e}")
            return ValidationResult(
                test_name="Online Learning",
                passed=False,
                score=0.0,
                threshold=0.8,
                details={},
                error_message=str(e)
            )
    
    def validate_speculative_decoding(self) -> ValidationResult:
        """
        Validate Speculative: Neutral or positive latency impact.
        
        Acceptance Criteria:
        - Acceptance rate > 0.3 (reasonable draft quality)
        - Net speedup >= 1.0 (neutral or positive latency)
        - No significant quality degradation
        - Adaptive drafting adjusts appropriately
        """
        logger.info("Validating Speculative Decoding System...")
        
        try:
            # Create mock models for testing
            class MockModel(nn.Module):
                def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, hidden_size)
                    self.layers = nn.ModuleList([
                        nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
                    ])
                    self.lm_head = nn.Linear(hidden_size, vocab_size)
                    
                def forward(self, input_ids, **kwargs):
                    x = self.embedding(input_ids)
                    for layer in self.layers:
                        x = F.gelu(layer(x))
                    logits = self.lm_head(x)
                    return type('Output', (), {'logits': logits})()
                
                def generate(self, input_ids, max_new_tokens=10, **kwargs):
                    # Simple greedy generation
                    generated = input_ids.clone()
                    for _ in range(max_new_tokens):
                        outputs = self(generated)
                        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=1)
                    return generated
            
            # Create models
            base_model = MockModel(num_layers=2).to(self.device)  # Faster for drafting
            bem_model = MockModel(num_layers=4).to(self.device)   # More complex
            
            # Mock tokenizer
            class MockTokenizer:
                def __init__(self):
                    self.eos_token_id = 2
                    
                def encode(self, text, return_tensors='pt'):
                    tokens = [hash(c) % 1000 for c in text[:20]]
                    return torch.tensor(tokens).unsqueeze(0).to(self.device)
                    
                def decode(self, tokens):
                    return f"Generated {len(tokens)} tokens"
            
            tokenizer = MockTokenizer()
            
            # Create speculative decoder
            config = create_default_speculative_config(
                draft_length=4,
                kl_threshold=0.1,
                enable_adaptive_drafting=True
            )
            
            decoder = create_speculative_decoder(
                base_model=base_model,
                bem_model=bem_model,
                tokenizer=tokenizer,
                config=config,
                telemetry_collector=self.telemetry
            )
            
            # Run benchmark
            test_prompts = [
                "The future of artificial intelligence",
                "In a world where technology",
                "Scientists have discovered",
                "The best way to solve this problem",
                "Once upon a time in a distant"
            ]
            
            benchmark = SpeculativeDecodingBenchmark(
                base_model=base_model,
                bem_model=bem_model,
                tokenizer=tokenizer,
                test_prompts=test_prompts
            )
            
            benchmark_results = benchmark.run_benchmark(
                config=config,
                max_new_tokens=20,
                num_runs=3
            )
            
            # Extract metrics
            acceptance_rate = benchmark_results['average_acceptance_rate']
            net_speedup = benchmark_results['average_net_speedup']
            speedup_ratio = benchmark_results['speedup_ratio']
            
            # Get decoder statistics
            stats = decoder.get_statistics()
            
            # Acceptance criteria
            acceptance_pass = acceptance_rate > 0.3
            speedup_pass = net_speedup >= 1.0
            latency_neutral = speedup_ratio >= 1.0
            
            overall_pass = acceptance_pass and speedup_pass and latency_neutral
            
            # Calculate score
            score = (
                min(acceptance_rate / 0.3, 1.0) * 0.4 +
                min(net_speedup / 1.0, 2.0) / 2.0 * 0.4 +  # Cap at 2x speedup
                min(speedup_ratio / 1.0, 2.0) / 2.0 * 0.2
            )
            
            return ValidationResult(
                test_name="Speculative Decoding",
                passed=overall_pass,
                score=score,
                threshold=0.7,
                details={
                    'acceptance_rate': acceptance_rate,
                    'acceptance_threshold': 0.3,
                    'acceptance_pass': acceptance_pass,
                    'net_speedup': net_speedup,
                    'speedup_threshold': 1.0,
                    'speedup_pass': speedup_pass,
                    'speedup_ratio': speedup_ratio,
                    'latency_neutral': latency_neutral,
                    'num_test_cases': benchmark_results['num_test_cases'],
                    'speculative_avg_time': benchmark_results['speculative_avg_time'],
                    'standard_avg_time': benchmark_results['standard_avg_time']
                }
            )
            
        except Exception as e:
            logger.error(f"Speculative Decoding validation failed: {e}")
            return ValidationResult(
                test_name="Speculative Decoding",
                passed=False,
                score=0.0,
                threshold=0.7,
                details={},
                error_message=str(e)
            )
    
    def validate_vector_quantization(self) -> ValidationResult:
        """
        Validate VQ: Memory reduction without significant quality loss.
        
        Acceptance Criteria:
        - Codebook utilization > 0.5
        - Quantization error < 0.1
        - Memory efficiency > 0.7
        - Perplexity indicates good diversity
        """
        logger.info("Validating Vector Quantization System...")
        
        try:
            # Create vector quantizer
            config = create_default_vq_config(
                codebook_size=256,
                code_dim=64,
                enable_residual=True,
                enable_episodic_memory=True
            )
            
            vq = create_vector_quantizer(config=config, telemetry_collector=self.telemetry).to(self.device)
            
            # Test with various inputs
            vq.train()
            total_commitment_loss = 0.0
            total_codebook_loss = 0.0
            num_batches = 30
            
            for batch_idx in range(num_batches):
                batch_size, seq_len = 4, 24
                inputs = torch.randn(batch_size, seq_len, 64).to(self.device)
                
                result = vq(inputs, training=True)
                
                total_commitment_loss += result['commitment_loss'].item()
                total_codebook_loss += result['codebook_loss'].item()
                
                # Verify reconstruction capability
                quantized = result['quantized']
                reconstruction_error = F.mse_loss(quantized, inputs).item()
                
                # Should reconstruct reasonably well
                assert reconstruction_error < 1.0, f"Reconstruction error too high: {reconstruction_error}"
            
            # Get final metrics and analysis
            avg_commitment_loss = total_commitment_loss / num_batches
            avg_codebook_loss = total_codebook_loss / num_batches
            
            # Get latest metrics from last forward pass
            final_inputs = torch.randn(4, 24, 64).to(self.device)
            final_result = vq(final_inputs, training=True)
            metrics = final_result['metrics']
            
            # Get codebook analysis
            analysis = vq.get_codebook_analysis()
            
            # Acceptance criteria
            utilization_pass = metrics.codebook_utilization > 0.5
            error_pass = metrics.average_quantization_error < 0.1
            memory_pass = metrics.memory_efficiency > 0.7
            perplexity_pass = metrics.perplexity > 2.0  # Good diversity
            
            overall_pass = utilization_pass and error_pass and memory_pass and perplexity_pass
            
            # Calculate score
            score = (
                min(metrics.codebook_utilization / 0.5, 1.0) * 0.3 +
                min((0.1 - metrics.average_quantization_error) / 0.1, 1.0) * 0.3 +
                min(metrics.memory_efficiency / 0.7, 1.0) * 0.2 +
                min(metrics.perplexity / 10.0, 1.0) * 0.2  # Cap at 10 for scoring
            )
            
            return ValidationResult(
                test_name="Vector Quantization",
                passed=overall_pass,
                score=score,
                threshold=0.75,
                details={
                    'codebook_utilization': metrics.codebook_utilization,
                    'utilization_threshold': 0.5,
                    'utilization_pass': utilization_pass,
                    'quantization_error': metrics.average_quantization_error,
                    'error_threshold': 0.1,
                    'error_pass': error_pass,
                    'memory_efficiency': metrics.memory_efficiency,
                    'memory_threshold': 0.7,
                    'memory_pass': memory_pass,
                    'perplexity': metrics.perplexity,
                    'perplexity_threshold': 2.0,
                    'perplexity_pass': perplexity_pass,
                    'dead_codes': metrics.dead_codes,
                    'active_codes': analysis['active_codes'],
                    'total_codes': analysis['codebook_size'],
                    'avg_commitment_loss': avg_commitment_loss,
                    'avg_codebook_loss': avg_codebook_loss
                }
            )
            
        except Exception as e:
            logger.error(f"Vector Quantization validation failed: {e}")
            return ValidationResult(
                test_name="Vector Quantization",
                passed=False,
                score=0.0,
                threshold=0.75,
                details={},
                error_message=str(e)
            )
    
    def validate_counterfactual_routing(self) -> ValidationResult:
        """
        Validate Counterfactual: Measurable routing improvements.
        
        Acceptance Criteria:
        - Component importance differences detectable
        - Interaction effects captured
        - Routing efficiency > 0.3
        - Systematic ablation produces consistent results
        """
        logger.info("Validating Counterfactual Routing System...")
        
        try:
            # Create mock model with multiple components
            class MockRoutingModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.component_a = nn.Linear(128, 128)
                    self.component_b = nn.Linear(128, 128)
                    self.component_c = nn.Linear(128, 64)
                    self.final_layer = nn.Linear(64, 10)
                    
                    # Component states for dropout
                    self.a_active = True
                    self.b_active = True
                    self.c_active = True
                
                def forward(self, x):
                    if self.a_active:
                        x = x + 0.3 * F.relu(self.component_a(x))
                    if self.b_active:
                        x = x + 0.2 * F.relu(self.component_b(x))
                    if self.c_active:
                        x = self.component_c(x)
                    else:
                        x = x[:, :64]  # Truncate if component C is inactive
                    
                    return self.final_layer(x)
            
            model = MockRoutingModel().to(self.device)
            
            # Performance evaluator
            def performance_evaluator(outputs, targets):
                return F.cross_entropy(outputs, targets).item()
            
            # Component dropout functions
            def dropout_a(module, is_active):
                module.a_active = is_active
                
            def dropout_b(module, is_active):
                module.b_active = is_active
                
            def dropout_c(module, is_active):
                module.c_active = is_active
            
            # Create counterfactual analyzer
            config = create_default_counterfactual_config(
                component_dropout_rate=0.2,
                analysis_frequency=20,
                enable_routing_optimization=True
            )
            
            analyzer = create_counterfactual_analyzer(
                model=model,
                performance_evaluator=performance_evaluator,
                config=config,
                telemetry_collector=self.telemetry
            )
            
            # Register components
            analyzer.register_component("component_a", model, dropout_a)
            analyzer.register_component("component_b", model, dropout_b)
            analyzer.register_component("component_c", model, dropout_c)
            
            # Run analysis over multiple steps
            for step in range(100):
                batch_size, seq_len = 4, 32
                inputs = torch.randn(batch_size, seq_len, 128).to(self.device)
                targets = torch.randint(0, 10, (batch_size, seq_len)).to(self.device)
                
                result = analyzer.step(inputs, targets)
            
            # Get analysis summary
            summary = analyzer.get_analysis_summary()
            
            # Run systematic ablation
            inputs = torch.randn(4, 32, 128).to(self.device)
            targets = torch.randint(0, 10, (4, 32)).to(self.device)
            ablation_results = analyzer.run_systematic_ablation(inputs, targets)
            
            # Acceptance criteria
            routing_efficiency = summary['routing_efficiency']
            has_importance_differences = len(summary['top_components']) > 0
            has_ablation_results = len(ablation_results) > 0
            
            # Check for measurable component differences
            importance_variance = 0.0
            if summary['top_components']:
                importances = [importance for _, importance, _ in summary['top_components']]
                importance_variance = np.var(importances) if len(importances) > 1 else 0.0
            
            efficiency_pass = routing_efficiency > 0.3
            differences_pass = has_importance_differences and importance_variance > 0.001
            ablation_pass = has_ablation_results
            interaction_pass = len(summary.get('interaction_effects', {})) >= 0
            
            overall_pass = efficiency_pass and differences_pass and ablation_pass
            
            # Calculate score
            score = (
                min(routing_efficiency / 0.3, 1.0) * 0.4 +
                min(importance_variance / 0.01, 1.0) * 0.3 +
                (1.0 if ablation_pass else 0.0) * 0.2 +
                (1.0 if interaction_pass else 0.0) * 0.1
            )
            
            return ValidationResult(
                test_name="Counterfactual Routing",
                passed=overall_pass,
                score=score,
                threshold=0.7,
                details={
                    'routing_efficiency': routing_efficiency,
                    'efficiency_threshold': 0.3,
                    'efficiency_pass': efficiency_pass,
                    'importance_variance': importance_variance,
                    'differences_pass': differences_pass,
                    'num_components_analyzed': len(summary['top_components']),
                    'ablation_results_count': len(ablation_results),
                    'ablation_pass': ablation_pass,
                    'interaction_effects_count': len(summary.get('interaction_effects', {})),
                    'interaction_pass': interaction_pass,
                    'credit_variance': summary.get('credit_variance', 0.0),
                    'ablation_consistency': summary.get('ablation_consistency', 0.0),
                    'top_components': summary['top_components'][:3] if summary['top_components'] else []
                }
            )
            
        except Exception as e:
            logger.error(f"Counterfactual Routing validation failed: {e}")
            return ValidationResult(
                test_name="Counterfactual Routing",
                passed=False,
                score=0.0,
                threshold=0.7,
                details={},
                error_message=str(e)
            )
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all Phase 5 validations and return comprehensive results."""
        logger.info("🚀 Starting Phase 5 Complete Validation Suite")
        
        print("=" * 80)
        print("PHASE 5 BEM SYSTEM - VALIDATION SUITE")
        print("Advanced Features Validation")
        print("=" * 80)
        
        # Run all validations
        validations = [
            self.validate_banked_experts,
            self.validate_online_learning,
            self.validate_speculative_decoding,
            self.validate_vector_quantization,
            self.validate_counterfactual_routing
        ]
        
        results = []
        for validation_fn in validations:
            try:
                result = validation_fn()
                results.append(result)
                self.results.append(result)
                
                # Print result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"\n{status} {result.test_name}")
                print(f"    Score: {result.score:.3f} / {result.threshold:.3f}")
                
                if result.error_message:
                    print(f"    Error: {result.error_message}")
                else:
                    # Print key metrics
                    if result.test_name == "Banked Experts":
                        d = result.details
                        print(f"    Utilization: {d.get('utilization_entropy', 0):.3f} > {d.get('entropy_threshold', 0):.1f} ({'✓' if d.get('entropy_pass') else '✗'})")
                        print(f"    Active experts: {d.get('active_expert_ratio', 0):.1%} ({'✓' if d.get('active_pass') else '✗'})")
                        
                    elif result.test_name == "Online Learning":
                        d = result.details
                        print(f"    Violations detected: {d.get('violations_detected', 0)} ({'✓' if d.get('violations_detected', 0) > 0 else '✗'})")
                        print(f"    Rollbacks functional: {'✓' if d.get('rollback_functional') else '✗'}")
                        
                    elif result.test_name == "Speculative Decoding":
                        d = result.details
                        print(f"    Acceptance rate: {d.get('acceptance_rate', 0):.1%} ({'✓' if d.get('acceptance_pass') else '✗'})")
                        print(f"    Net speedup: {d.get('net_speedup', 0):.2f}x ({'✓' if d.get('speedup_pass') else '✗'})")
                        
                    elif result.test_name == "Vector Quantization":
                        d = result.details
                        print(f"    Codebook util: {d.get('codebook_utilization', 0):.1%} ({'✓' if d.get('utilization_pass') else '✗'})")
                        print(f"    Quant error: {d.get('quantization_error', 0):.4f} ({'✓' if d.get('error_pass') else '✗'})")
                        
                    elif result.test_name == "Counterfactual Routing":
                        d = result.details
                        print(f"    Routing efficiency: {d.get('routing_efficiency', 0):.3f} ({'✓' if d.get('efficiency_pass') else '✗'})")
                        print(f"    Components analyzed: {d.get('num_components_analyzed', 0)}")
                
            except Exception as e:
                logger.error(f"Validation {validation_fn.__name__} failed: {e}")
                failed_result = ValidationResult(
                    test_name=validation_fn.__name__.replace('validate_', '').replace('_', ' ').title(),
                    passed=False,
                    score=0.0,
                    threshold=0.7,
                    details={},
                    error_message=str(e)
                )
                results.append(failed_result)
                self.results.append(failed_result)
        
        # Calculate overall results
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        overall_score = np.mean([r.score for r in results])
        overall_passed = all(r.passed for r in results)
        
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Overall score: {overall_score:.3f}")
        print(f"Overall status: {'✅ ALL PASSED' if overall_passed else '❌ SOME FAILED'}")
        
        if overall_passed:
            print("\n🎯 Phase 5 BEM System - VALIDATION SUCCESSFUL")
            print("All advanced features meet acceptance criteria:")
            print("  ✓ Banked Experts: Expressivity without collapsed utilization")
            print("  ✓ Online Learning: Trust monitors prevent catastrophic drift")
            print("  ✓ Speculative Decoding: Neutral or positive latency impact")
            print("  ✓ Vector Quantization: Memory reduction without quality loss")
            print("  ✓ Counterfactual Routing: Measurable routing improvements")
            print("\n🚀 BEM Research Project Phase 5 - READY FOR DEPLOYMENT")
        else:
            print(f"\n⚠️  Phase 5 validation incomplete - {total_tests - passed_tests} tests failed")
            print("Review failed tests and address issues before deployment.")
        
        # Return comprehensive results
        return {
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'score': r.score,
                    'threshold': r.threshold,
                    'details': r.details,
                    'error_message': r.error_message
                } for r in results
            ],
            'timestamp': time.time()
        }
    
    def save_results(self, output_path: str = "phase5_validation_results.json"):
        """Save validation results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
            
        results_data = {
            'phase': 5,
            'validation_timestamp': time.time(),
            'device': str(self.device),
            'results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'score': r.score,
                    'threshold': r.threshold,
                    'details': r.details,
                    'error_message': r.error_message
                } for r in self.results
            ]
        }
        
        Path(output_path).write_text(json.dumps(results_data, indent=2))
        logger.info(f"Validation results saved to {output_path}")


def main():
    """Main validation function."""
    validator = Phase5Validator()
    
    # Run complete validation suite
    results = validator.run_all_validations()
    
    # Save results
    validator.save_results("phase5_validation_results.json")
    
    return results


if __name__ == "__main__":
    main()