# BEM 2.0 Performance Track (PT1-PT4) Implementation

This module implements sophisticated performance optimization variants designed to shift the Pareto frontier or yield CI-backed Slice-B gains while maintaining strict budget parity (±5% params/FLOPs vs v1.3-stack anchor).

## Performance Track Variants

### PT1 - Head-Group Gating @ W_O
**Target**: +0.5–1.5% improvement  
**Approach**: Rank split across G groups with per-group gates derived from attention statistics and retrieval quality. Gates are decorrelated to prevent redundancy.

**Key Features:**
- 4 head groups with per-group rank allocation
- Attention statistics extraction (entropy, variance)
- Retrieval quality scoring for gate computation
- Decorrelation penalty to prevent redundancy
- Spectral governance for stability

**Configuration:**
```python
from bem2.perftrack import HeadGroupGatingConfig, HeadGroupGatingModule

config = HeadGroupGatingConfig(
    num_groups=4,
    heads_per_group=3,
    rank_per_group=4,
    decorrelation_strength=0.1
)
```

### PT2 - Dynamic Rank Mask (Fixed FLOPs)
**Target**: +≥1% primary metric improvement  
**Approach**: k-hot mask over rank components per block with ~50% sparsity, using masked Hadamard path for efficient computation and instance-adaptive rank selection.

**Key Features:**
- k-hot masking with 50% sparsity (8/16 ranks active)
- Instance-adaptive mask controller
- Fast Hadamard Transform for efficient computation
- Straight-through estimator for differentiable masking
- Fixed FLOPs constraint with dynamic sparsity

**Configuration:**
```python
from bem2.perftrack import DynamicRankMaskConfig, DynamicRankMaskModule

config = DynamicRankMaskConfig(
    total_rank=16,
    active_rank=8,
    use_hadamard_path=True,
    target_sparsity=0.5
)
```

### PT3 - Kronecker @ W_down (One Site)
**Target**: +0.5–1.5% chrF/BLEU improvement  
**Approach**: (U⊗V) factorization with fused kernel at W_down attachment point only. Maintains spectral clamps for stability with single attachment point for controlled testing.

**Key Features:**
- Kronecker product factorization (U⊗V)
- Fused CUDA kernels for efficiency
- Memory-efficient computation paths
- SVD initialization for stability
- Orthogonality and diversity regularization
- Single W_down attachment site

**Configuration:**
```python
from bem2.perftrack import KroneckerConfig, KroneckerModule

config = KroneckerConfig(
    u_rank=8, v_rank=8,
    u_dim=64, v_dim=48,
    use_fused_kernel=True,
    init_method="svd"
)
```

### PT4 - Residual FiLM Micro-γ,β
**Target**: Validity improvements with no regressions  
**Approach**: Tiny per-block modulation with strict clamps using controller-driven feature-wise transformations with minimal parameter overhead.

**Key Features:**
- Micro-scale modulation (0.01x scaling)
- Strict parameter clamping (±0.1 range)
- Feature-wise Linear Modulation (FiLM)
- Emergency stabilization mechanisms  
- Minimal parameter overhead
- Channel-wise or spatial-wise modulation options

**Configuration:**
```python
from bem2.perftrack import ResidualFiLMConfig, ResidualFiLMModule

config = ResidualFiLMConfig(
    gamma_dim=16, beta_dim=16,
    micro_scale=0.01,
    clamp_range=0.1,
    minimal_overhead=True
)
```

## Budget Parity Validation

All variants maintain strict budget constraints:

```python
from bem2.perftrack import BudgetValidator, BudgetConstraints

constraints = BudgetConstraints(
    baseline_params=124964096,  # v1.3-stack anchor
    baseline_flops=1000000,
    baseline_memory_mb=512.0,
    tolerance=0.05  # ±5%
)

validator = BudgetValidator(constraints)
validation_result = validator.validate_all(performance_metrics)
```

## Evaluation Framework

### Pareto Frontier Analysis
```python
from bem2.perftrack import ParetoAnalyzer

analyzer = ParetoAnalyzer()
analyzer.add_result("PT1", pt1_metrics)
analyzer.add_result("PT2", pt2_metrics)

pareto_analysis = analyzer.pareto_dominance_analysis()
frontiers = pareto_analysis['frontiers']
```

### Statistical Validation
```python
from bem2.perftrack import StatisticalValidator

# CI-backed Slice-B analysis
slice_results = StatisticalValidator.slice_b_analysis(
    baseline_results={'slice_a': [0.85, 0.87, 0.86]},
    variant_results={'slice_a': [0.87, 0.89, 0.88]}
)
```

### Performance Profiling
```python
from bem2.perftrack import PerformanceProfiler

profiler = PerformanceProfiler(device='cuda')
metrics = profiler.comprehensive_profile(model, test_data, "PT1")

print(f"Latency P95: {metrics.variant_specific['latency_p95_ms']:.2f}ms")
print(f"Memory Peak: {metrics.variant_specific['memory_peak_mb']:.2f}MB")
```

## Training Infrastructure

### Specialized Training Configuration
```python
from bem2.perftrack import PTTrainingConfig, PTTrainer

config = PTTrainingConfig(
    learning_rate=5e-4,
    variant_lr_multipliers={
        'PT1': 1.0,   # Standard LR
        'PT2': 0.5,   # Lower for mask sensitivity  
        'PT3': 1.5,   # Higher for Kronecker factors
        'PT4': 0.1    # Very low for stability
    },
    budget_constraints=budget_constraints,
    enforce_budget_during_training=True
)

trainer = PTTrainer(config)
```

### Variant-Specific Optimizers
Each variant uses specialized parameter grouping:
- **PT1**: Separate rates for gate controller vs projections
- **PT2**: Very low LR for mask parameters 
- **PT3**: Higher LR for Kronecker factors vs projections
- **PT4**: Ultra-conservative rates for stability

## Usage Examples

### Basic Usage
```python
import torch
from bem2.perftrack import HeadGroupGatingModule, HeadGroupGatingConfig

# Create PT1 module
config = HeadGroupGatingConfig(num_groups=4)
pt1_module = HeadGroupGatingModule(
    config=config,
    layer_idx=0,
    hidden_size=768,
    num_attention_heads=12
)

# Forward pass
hidden_states = torch.randn(32, 128, 768)
attention_weights = torch.randn(32, 12, 128, 128)
output, metrics = pt1_module(hidden_states, attention_weights, output_attentions=True)
```

### Comprehensive Evaluation
```python
from bem2.perftrack import ComprehensiveEvaluator, BudgetConstraints

# Setup evaluator
constraints = BudgetConstraints()
evaluator = ComprehensiveEvaluator(constraints)

# Evaluate all variants
for variant_name, model in [("PT1", pt1_model), ("PT2", pt2_model)]:
    results = evaluator.evaluate_variant(
        variant_name=variant_name,
        model=model,
        test_data=test_data,
        performance_scores=scores
    )

# Generate final report
final_report = evaluator.generate_final_report()
promotion_candidates = final_report['promotion_candidates']
```

## Promotion Criteria

Variants are promoted if they meet **ANY** of these criteria:

1. **Pareto Frontier Shift**: Appear in any Pareto-optimal frontier
2. **CI-backed Slice-B Gains**: Statistically significant improvements on evaluation slices
3. **Budget Compliance**: Pass ±5% parameter/FLOP constraints

### Success Metrics
- **Latency P95**: <500ms for inference
- **Throughput**: >1000 requests/minute
- **Memory Efficiency**: <20% VRAM overhead
- **Quality**: 90%+ test coverage, robust error handling

## Architecture Patterns

### Trust Region Constraints
All variants implement spectral governance:
```python
from bem.modules.governance import SpectralGovernance

spectral_gov = SpectralGovernance(
    max_singular_value=1.0,
    layer_specific=True
)

delta_clamped, metrics = spectral_gov.apply_spectral_clamp(
    delta_w, layer_name="pt1_group_0"
)
```

### Invariant Preservation
- **Attachment Points**: W_O + W_down only (cache-safe)
- **Norm/σ₁ Caps**: Applied after composition
- **Residual Compatibility**: All variants maintain residual structure

## Experimental Configurations

Training configurations for each variant:

- **PT1**: [`experiments/PT1_head_gating.yml`](../../experiments/PT1_head_gating.yml)
- **PT2**: [`experiments/PT2_dynamic_mask.yml`](../../experiments/PT2_dynamic_mask.yml)  
- **PT3**: [`experiments/PT3_kronecker.yml`](../../experiments/PT3_kronecker.yml)
- **PT4**: [`experiments/PT4_residual_film.yml`](../../experiments/PT4_residual_film.yml)

## Running the Demo

```bash
# Run comprehensive demonstration
python demo_performance_track_variants.py

# Expected output:
# - Forward pass timings for each variant
# - Budget validation results
# - Pareto frontier analysis
# - Promotion candidate recommendations
```

## Advanced Features

### CUDA Acceleration
PT3 includes fused CUDA kernels for Kronecker operations:
```python
from bem2.perftrack.pt3_kronecker import FusedKroneckerOp

fused_op = FusedKroneckerOp(config)
output = fused_op(input_tensor, U_factor, V_factor)
```

### Emergency Stabilization  
PT4 includes automatic stability monitoring:
```python
# Automatic emergency reset if instability detected
if torch.isnan(output).any():
    pt4_module.emergency_stabilize()
```

### Memory Optimization
All variants include memory-efficient computation paths:
- PT2: Straight-through estimator avoids gradient accumulation
- PT3: Memory-efficient Kronecker avoids explicit product formation
- PT4: Minimal controller networks reduce activation memory

## Performance Benchmarks

Expected performance characteristics:

| Variant | Latency Overhead | Memory Overhead | Parameter Overhead |
|---------|------------------|------------------|--------------------|
| PT1     | <5%              | <10%             | <3%                |
| PT2     | <2%*             | <5%              | <4%                |  
| PT3     | <8%              | <15%             | <5%                |
| PT4     | <1%              | <2%              | <1%                |

*PT2 can be faster due to sparsity

## Citation

If you use this implementation, please cite:

```bibtex
@article{bem2_performance_track_2024,
    title={BEM 2.0 Performance Track: Sophisticated Optimization Variants for Pareto Frontier Enhancement},
    year={2024},
    note={Implementation of PT1-PT4 variants with comprehensive evaluation framework}
}
```