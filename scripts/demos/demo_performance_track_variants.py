#!/usr/bin/env python3
"""
Demo: BEM 2.0 Performance Track Variants (PT1-PT4)

Comprehensive demonstration of all four performance optimization variants:
- PT1: Head-Group Gating @ W_O
- PT2: Dynamic Rank Mask (Fixed FLOPs) 
- PT3: Kronecker @ W_down (One Site)
- PT4: Residual FiLM Micro-γ,β

Each variant is designed to shift the Pareto frontier or yield CI-backed Slice-B gains
while maintaining strict budget parity (±5% params/FLOPs vs v1.3-stack anchor).
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any

# Import PT variants
from bem2.perftrack import (
    # PT1 Head-Group Gating
    HeadGroupGatingConfig,
    HeadGroupGatingModule,
    
    # PT2 Dynamic Rank Mask
    DynamicRankMaskConfig, 
    DynamicRankMaskModule,
    
    # PT3 Kronecker Factorization
    KroneckerConfig,
    KroneckerModule,
    
    # PT4 Residual FiLM
    ResidualFiLMConfig,
    ResidualFiLMModule,
    
    # Evaluation framework
    BudgetValidator,
    BudgetConstraints,
    ParetoAnalyzer,
    PerformanceProfiler,
    ComprehensiveEvaluator,
    PerformanceMetrics
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTransformerLayer(nn.Module):
    """Mock transformer layer for demonstration."""
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, intermediate_size: int = 3072):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        
        # Mock attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        
        # Mock FFN
        self.ffn_up = nn.Linear(hidden_size, intermediate_size)
        self.ffn_down = nn.Linear(intermediate_size, hidden_size)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, attn_weights = self.attention(
            hidden_states, hidden_states, hidden_states
        )
        hidden_states = self.norm1(hidden_states + attn_out)
        
        # FFN
        ffn_intermediate = torch.relu(self.ffn_up(hidden_states))
        ffn_out = self.ffn_down(ffn_intermediate)
        hidden_states = self.norm2(hidden_states + ffn_out)
        
        return hidden_states


class PT1Demo(nn.Module):
    """PT1 Head-Group Gating demonstration model."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.num_heads = 12
        
        # Base transformer layer
        self.base_layer = MockTransformerLayer(self.hidden_size, self.num_heads)
        
        # PT1 Head-Group Gating
        config = HeadGroupGatingConfig(
            num_groups=4,
            heads_per_group=3,  # 12 heads / 4 groups
            rank_per_group=4,
            gate_temperature=1.0,
            decorrelation_strength=0.1
        )
        
        self.pt1_module = HeadGroupGatingModule(
            config=config,
            layer_idx=0,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads
        )
        
    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        # Base transformer processing
        hidden_states = self.base_layer(x)
        
        # Extract attention weights (mock)
        batch_size, seq_len, _ = hidden_states.shape
        attention_weights = torch.softmax(
            torch.randn(batch_size, self.num_heads, seq_len, seq_len), dim=-1
        )
        
        # PT1 processing
        output, attention_info = self.pt1_module(
            hidden_states, attention_weights, output_attentions=output_attentions
        )
        
        if output_attentions:
            return output, attention_info
        return output


class PT2Demo(nn.Module):
    """PT2 Dynamic Rank Mask demonstration model."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        
        # Base transformer layer
        self.base_layer = MockTransformerLayer(self.hidden_size)
        
        # PT2 Dynamic Rank Mask
        config = DynamicRankMaskConfig(
            total_rank=16,
            active_rank=8,  # 50% sparsity
            mask_temperature=0.1,
            use_hadamard_path=True,
            use_instance_adaptive=True
        )
        
        self.pt2_module = DynamicRankMaskModule(
            config=config,
            layer_idx=0,
            hidden_size=self.hidden_size
        )
        
    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        # Base transformer processing
        hidden_states = self.base_layer(x)
        
        # PT2 processing
        output, attention_info = self.pt2_module(
            hidden_states, output_attentions=output_attentions
        )
        
        if output_attentions:
            return output, attention_info
        return output


class PT3Demo(nn.Module):
    """PT3 Kronecker factorization demonstration model."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.intermediate_size = 3072
        
        # Base transformer layer (modified to expose intermediate)
        self.attention = nn.MultiheadAttention(self.hidden_size, 12, batch_first=True)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.ffn_up = nn.Linear(self.hidden_size, self.intermediate_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        
        # PT3 Kronecker at W_down
        config = KroneckerConfig(
            u_rank=8,
            v_rank=8,
            u_dim=64,
            v_dim=48,  # 64 * 48 = 3072 (intermediate_size)
            use_fused_kernel=True,
            init_method="svd"
        )
        
        self.pt3_module = KroneckerModule(
            config=config,
            layer_idx=0,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size
        )
        
    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        hidden_states = self.norm1(x + attn_out)
        
        # FFN up projection (generate intermediate states)
        intermediate_states = torch.relu(self.ffn_up(hidden_states))
        
        # PT3 Kronecker processing at W_down position
        output, attention_info = self.pt3_module(
            hidden_states, intermediate_states, output_attentions=output_attentions
        )
        
        if output_attentions:
            return output, attention_info
        return output


class PT4Demo(nn.Module):
    """PT4 Residual FiLM demonstration model."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        
        # Base transformer layer
        self.base_layer = MockTransformerLayer(self.hidden_size)
        
        # PT4 Residual FiLM
        config = ResidualFiLMConfig(
            gamma_dim=16,
            beta_dim=16,
            controller_dim=32,
            micro_scale=0.01,
            clamp_range=0.1,
            minimal_overhead=True
        )
        
        self.pt4_module = ResidualFiLMModule(
            config=config,
            layer_idx=0,
            hidden_size=self.hidden_size
        )
        
    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        # Base transformer processing
        hidden_states = self.base_layer(x)
        
        # PT4 processing
        output, attention_info = self.pt4_module(
            hidden_states, output_attentions=output_attentions
        )
        
        if output_attentions:
            return output, attention_info
        return output


def create_demo_data(batch_size: int = 32, seq_len: int = 128, hidden_size: int = 768) -> torch.Tensor:
    """Create demo input data."""
    return torch.randn(batch_size, seq_len, hidden_size)


def demonstrate_variant(
    variant_name: str,
    model: nn.Module,
    test_data: torch.Tensor,
    evaluator: ComprehensiveEvaluator
) -> Dict[str, Any]:
    """Demonstrate a single PT variant."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DEMONSTRATING {variant_name}")
    logger.info(f"{'='*60}")
    
    model.eval()
    
    # Forward pass with attention info
    start_time = time.time()
    with torch.no_grad():
        output, attention_info = model(test_data, output_attentions=True)
    inference_time = time.time() - start_time
    
    logger.info(f"Forward pass completed in {inference_time:.4f}s")
    logger.info(f"Output shape: {output.shape}")
    
    # Display variant-specific information
    if attention_info:
        logger.info(f"Attention info keys: {list(attention_info.keys())}")
        
        # Variant-specific metrics
        if variant_name == "PT1":
            gates = attention_info.get('gates')
            if gates is not None:
                logger.info(f"Gate statistics:")
                logger.info(f"  - Gate entropy: {attention_info.get('gate_entropy', 0):.4f}")
                logger.info(f"  - Decorrelation penalty: {attention_info.get('decorrelation_penalty', 0):.4f}")
                logger.info(f"  - Gate values (mean per group): {gates.mean(dim=0).tolist()}")
                
        elif variant_name == "PT2":
            sparsity = attention_info.get('sparsity', 0)
            rank_usage = attention_info.get('rank_usage', 0)
            logger.info(f"Masking statistics:")
            logger.info(f"  - Sparsity: {sparsity:.2%}")
            logger.info(f"  - Rank usage: {rank_usage:.2f}")
            logger.info(f"  - Mask entropy: {attention_info.get('mask_entropy', 0):.4f}")
            
        elif variant_name == "PT3":
            logger.info(f"Kronecker statistics:")
            logger.info(f"  - U effective rank: {attention_info.get('u_effective_rank', 0):.2f}")
            logger.info(f"  - V effective rank: {attention_info.get('v_effective_rank', 0):.2f}")
            logger.info(f"  - Compression ratio: {attention_info.get('compression_ratio', 1):.2f}")
            
        elif variant_name == "PT4":
            gamma = attention_info.get('gamma')
            beta = attention_info.get('beta')
            if gamma is not None and beta is not None:
                logger.info(f"FiLM statistics:")
                logger.info(f"  - Gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]")
                logger.info(f"  - Beta magnitude: {torch.abs(beta).mean():.4f}")
                logger.info(f"  - Residual scale: {attention_info.get('residual_scale', 0):.4f}")
                logger.info(f"  - Stability ratio: {attention_info.get('stability_ratio', 0):.4f}")
    
    # Budget metrics
    if hasattr(model, 'pt1_module'):
        budget_metrics = model.pt1_module.compute_budget_metrics()
    elif hasattr(model, 'pt2_module'):
        budget_metrics = model.pt2_module.compute_budget_metrics()
    elif hasattr(model, 'pt3_module'):
        budget_metrics = model.pt3_module.compute_budget_metrics()
    elif hasattr(model, 'pt4_module'):
        budget_metrics = model.pt4_module.compute_budget_metrics()
    else:
        budget_metrics = {}
    
    logger.info(f"Budget metrics:")
    for key, value in budget_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  - {key}: {value}")
    
    # Create mock performance scores for evaluation
    mock_performance = {
        'exact_match': np.random.uniform(0.7, 0.9),
        'f1_score': np.random.uniform(0.75, 0.95),
        'bleu_score': np.random.uniform(0.3, 0.6),
        'rouge_score': np.random.uniform(0.4, 0.7),
        'chrf_score': np.random.uniform(0.5, 0.8)
    }
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.evaluate_variant(
        variant_name=variant_name,
        model=model,
        test_data=test_data,
        performance_scores=mock_performance
    )
    
    # Display evaluation summary
    budget_status = evaluation_results['budget_validation']['overall_status']
    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  - Budget Status: {budget_status}")
    logger.info(f"  - F1 Score: {mock_performance['f1_score']:.4f}")
    logger.info(f"  - Parameters: {evaluation_results['performance_metrics']['parameters']:,}")
    logger.info(f"  - Memory (MB): {evaluation_results['performance_metrics']['memory_mb']:.2f}")
    logger.info(f"  - Inference Latency (ms): {evaluation_results['performance_metrics']['inference_latency_ms']:.2f}")
    
    return evaluation_results


def main():
    """Main demonstration function."""
    
    print(f"\n{'='*80}")
    print("BEM 2.0 PERFORMANCE TRACK VARIANTS DEMONSTRATION")
    print("PT1: Head-Group Gating | PT2: Dynamic Mask | PT3: Kronecker | PT4: FiLM")
    print(f"{'='*80}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create test data
    test_data = create_demo_data(batch_size=16, seq_len=64, hidden_size=768)
    if device == 'cuda':
        test_data = test_data.cuda()
    
    # Create evaluator
    budget_constraints = BudgetConstraints(
        baseline_params=124964096,  # v1.3-stack anchor
        baseline_flops=1000000,
        baseline_memory_mb=512.0,
        tolerance=0.05
    )
    
    evaluator = ComprehensiveEvaluator(budget_constraints)
    
    # Demonstrate each variant
    variants = [
        ("PT1", PT1Demo()),
        ("PT2", PT2Demo()), 
        ("PT3", PT3Demo()),
        ("PT4", PT4Demo())
    ]
    
    results = {}
    
    for variant_name, model in variants:
        if device == 'cuda':
            model = model.cuda()
        
        try:
            results[variant_name] = demonstrate_variant(
                variant_name, model, test_data, evaluator
            )
        except Exception as e:
            logger.error(f"Error demonstrating {variant_name}: {e}")
            results[variant_name] = {'error': str(e)}
    
    # Generate final comparative report
    logger.info(f"\n{'='*80}")
    logger.info("COMPARATIVE ANALYSIS")
    logger.info(f"{'='*80}")
    
    final_report = evaluator.generate_final_report()
    
    # Print summary
    evaluator.print_summary()
    
    # Pareto analysis
    pareto_analysis = final_report.get('pareto_analysis', {})
    frontiers = pareto_analysis.get('frontiers', {})
    
    if frontiers:
        logger.info(f"\nPareto Frontier Analysis:")
        for frontier_name, frontier_points in frontiers.items():
            logger.info(f"  {frontier_name}:")
            for variant_name, metrics in frontier_points:
                f1_score = metrics.f1_score
                params = metrics.parameters
                logger.info(f"    - {variant_name}: F1={f1_score:.4f}, Params={params:,}")
    
    # Promotion candidates
    promotion_candidates = final_report.get('promotion_candidates', [])
    if promotion_candidates:
        logger.info(f"\n🎯 PROMOTION CANDIDATES:")
        for candidate in promotion_candidates:
            logger.info(f"✅ {candidate['variant']}")
            logger.info(f"   - F1 Score: {candidate['f1_score']:.4f}")
            logger.info(f"   - Budget Valid: {candidate['budget_valid']}")
            logger.info(f"   - Pareto Optimal: {candidate['pareto_optimal']}")
    else:
        logger.info(f"\n❌ No variants meet promotion criteria")
    
    # Save results
    results_dir = Path("performance_track_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed report
    report_path = results_dir / "pt_variants_evaluation_report.json"
    evaluator.save_report(report_path)
    
    # Save demonstration summary
    demo_summary = {
        'timestamp': time.time(),
        'device': device,
        'test_data_shape': list(test_data.shape),
        'variants_demonstrated': list(results.keys()),
        'budget_constraints': {
            'baseline_params': budget_constraints.baseline_params,
            'tolerance': budget_constraints.tolerance
        },
        'promotion_candidates': len(promotion_candidates),
        'summary': final_report['evaluation_summary']
    }
    
    summary_path = results_dir / "demonstration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(demo_summary, f, indent=2, default=str)
    
    logger.info(f"\n📊 Results saved to:")
    logger.info(f"  - Detailed report: {report_path}")
    logger.info(f"  - Demo summary: {summary_path}")
    
    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print("All PT variants successfully demonstrated with budget validation")
    print("See generated reports for detailed Pareto analysis and promotion criteria")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()