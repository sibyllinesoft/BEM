# BEM v1.3 Developer Guide - Contributing and Extending

## 🏗️ Overview

This developer guide provides comprehensive information for researchers and engineers who want to contribute to, extend, or modify the BEM v1.3 Performance+Agentic Sprint system. The system has 250,000+ lines of code across 6,296+ test files with a comprehensive validation framework.

## 📋 Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Code Architecture](#code-architecture)  
3. [Testing Framework](#testing-framework)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Extending the System](#extending-the-system)
6. [Performance Optimization](#performance-optimization)
7. [Debugging and Profiling](#debugging-and-profiling)

## 🚀 Development Environment Setup

### Prerequisites

```bash
# System requirements
- Python 3.8+ (3.10+ recommended)
- CUDA 11.8+ for GPU acceleration
- 16GB+ RAM (32GB recommended for development)
- 50GB+ disk space for full development environment
```

### Development Installation

```bash
# 1. Clone repository and setup environment
git clone [repository]
cd modules
python -m venv venv
source venv/bin/activate

# 2. Install development dependencies  
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Install pre-commit hooks
pre-commit install

# 4. Run development validation
python validate_structure.py --dev-environment
```

### Development Dependencies

```txt
# Core dependencies (requirements.txt)
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=1.5.0
scikit-learn>=1.2.0
transformers>=4.30.0
accelerate>=0.20.0
peft>=0.4.0

# Development dependencies (requirements-dev.txt)
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0
jupyter>=1.0.0
wandb>=0.15.0
tensorboard>=2.13.0
```

### IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["--tb=short"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": true,
        "**/.pytest_cache": true
    }
}
```

## 🏗️ Code Architecture

### Module Organization

```
bem2/                          # Core BEM v1.3 Implementation
├── __init__.py               # Main exports and version info
├── perftrack/                # Performance Variants (PT1-PT4)
│   ├── __init__.py           # Performance tracking exports
│   ├── pt1_head_gating.py    # Head-group attention gating
│   ├── pt2_dynamic_mask.py   # Dynamic rank masking
│   ├── pt3_kronecker.py      # Kronecker factorization  
│   ├── pt4_residual_film.py  # Residual FiLM modulation
│   ├── training.py           # Performance variant training
│   └── evaluation.py         # Performance evaluation
├── router/                   # Agentic Routing System
│   ├── __init__.py           # Router system exports
│   ├── agentic_router.py     # Main router implementation
│   ├── macro_policy.py       # TRPO macro-policy
│   ├── composition_engine.py # Expert composition
│   ├── training.py           # Router training pipeline
│   └── synthesize_traces.py  # Training data generation
├── online/                   # Online Learning Components
│   ├── __init__.py           # Online learning exports
│   ├── online_learner.py     # EWC/Prox online updates
│   ├── drift_monitor.py      # Drift detection
│   ├── canary_gate.py        # Performance safety gates
│   └── evaluation.py         # Online learning evaluation
├── multimodal/               # Vision-Text Integration
│   ├── __init__.py           # Multimodal exports
│   ├── controller_integration.py # Complete multimodal controller
│   ├── vision_encoder.py     # Vision processing pipeline
│   └── coverage_analysis.py  # Evidence coverage analysis
├── safety/                   # Constitutional Safety
│   ├── __init__.py           # Safety system exports
│   ├── safety_basis.py       # Orthogonal safety basis
│   ├── safety_controller.py  # Safety management
│   └── violation_detector.py # Safety violation detection
└── evaluation/               # Statistical Framework
    ├── __init__.py           # Evaluation framework exports
    ├── statistical_analysis.py # BCa bootstrap + FDR
    ├── evaluation_framework.py # Complete evaluation system
    └── acceptance_validator.py # Gate validation system
```

### Design Principles

**1. Modular Architecture**:
```python
# Each component follows consistent interface patterns
class BEMComponent(abc.ABC):
    """Base class for all BEM components"""
    
    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass implementation"""
        pass
        
    @abc.abstractmethod  
    def get_delta_weights(self) -> Dict[str, torch.Tensor]:
        """Get weight modifications"""
        pass
        
    @abc.abstractmethod
    def validate_budget_parity(self, baseline: 'BEMComponent') -> bool:
        """Ensure ±5% parameter/FLOP parity"""
        pass
```

**2. Configuration-Driven Design**:
```python
# All components use YAML configuration files
@dataclass
class BEMConfig:
    experiment_name: str
    variant_type: str  # "performance", "router", "online", etc.
    model: ModelConfig
    bem_config: VariantConfig
    training: TrainingConfig  
    validation: ValidationConfig
    constraints: ConstraintConfig
    acceptance_gates: AcceptanceConfig
```

**3. Statistical Validation First**:
```python
# Every component includes statistical validation
class ExperimentResult:
    def __init__(self, baseline_metrics, variant_metrics):
        self.baseline_metrics = baseline_metrics
        self.variant_metrics = variant_metrics
        self.statistical_validation = self._compute_statistics()
        self.acceptance_decision = self._validate_gates()
        
    def _compute_statistics(self):
        paired_scores = self.variant_metrics - self.baseline_metrics
        return {
            'bca_bootstrap': bca_bootstrap(paired_scores, n_bootstrap=10000),
            'fdr_correction': benjamini_hochberg_correction([self._compute_p_value()])
        }
```

## 🧪 Testing Framework

### Test Organization

The system includes 6,296+ comprehensive test files organized by component:

```
tests/
├── test_perftrack/           # Performance variant tests
│   ├── test_pt1_head_gating.py
│   ├── test_pt2_dynamic_mask.py  
│   ├── test_pt3_kronecker.py
│   └── test_pt4_residual_film.py
├── test_router/              # Agentic router tests
│   ├── test_agentic_router.py
│   ├── test_macro_policy.py
│   └── test_composition_engine.py
├── test_online/              # Online learning tests
├── test_multimodal/          # Multimodal integration tests
├── test_safety/              # Safety system tests
├── test_evaluation/          # Statistical framework tests
├── test_integration/         # End-to-end integration tests
└── test_performance/         # Performance benchmarking tests
```

### Test Categories

**1. Unit Tests** - Component-level functionality:
```python
class TestDynamicRankMask(unittest.TestCase):
    """Unit tests for PT2 Dynamic Rank Mask component"""
    
    def setUp(self):
        self.config = DynamicRankConfig(
            rank_dim=64,
            active_ratio=0.5,
            hidden_dim=768
        )
        self.module = DynamicRankMask(self.config)
        
    def test_forward_pass_shape_invariance(self):
        """Test that output shape matches input shape"""
        batch_size, seq_len, hidden_dim = 8, 128, 768
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        features = torch.randn(batch_size, seq_len, hidden_dim)
        
        output = self.module(input_tensor, features)
        self.assertEqual(output.shape, input_tensor.shape)
        
    def test_k_hot_sparsity_constraint(self):
        """Test that exactly k components are active"""
        batch_size, seq_len = 4, 64
        features = torch.randn(batch_size, seq_len, 768)
        
        mask = self.module.get_rank_mask(features)
        active_count = mask.sum(dim=-1)
        expected_active = int(self.config.rank_dim * self.config.active_ratio)
        
        self.assertTrue(torch.all(active_count == expected_active))
        
    def test_budget_parity_compliance(self):
        """Test that parameter count is within ±5% of baseline"""
        baseline_params = 354823680
        variant_params = self.module.count_parameters()
        ratio = variant_params / baseline_params
        
        self.assertTrue(0.95 <= ratio <= 1.05, f"Param ratio {ratio} outside ±5%")
```

**2. Integration Tests** - Multi-component interactions:
```python
class TestMultiBEMComposition(unittest.TestCase):
    """Integration tests for multi-BEM composition"""
    
    def setUp(self):
        self.bem1 = DynamicRankMask(config1)
        self.bem2 = ResidualFiLMModule(config2)
        self.composition_engine = CompositionEngine([self.bem1, self.bem2])
        
    def test_orthogonal_subspace_allocation(self):
        """Test that BEMs use orthogonal subspaces"""
        delta_w1 = self.bem1.get_delta_weights()
        delta_w2 = self.bem2.get_delta_weights()
        
        # Compute overlap in weight space
        overlap = self._compute_subspace_overlap(delta_w1, delta_w2)
        self.assertLess(overlap, 0.1, "BEMs must use orthogonal subspaces")
        
    def test_trust_region_projection(self):
        """Test that composed weights respect trust region"""
        combined_delta = self.composition_engine.compose_weights()
        frob_norm = torch.norm(combined_delta, p='fro')
        
        self.assertLessEqual(frob_norm, self.composition_engine.trust_region_bound)
```

**3. Statistical Tests** - Validation framework testing:
```python
class TestStatisticalFramework(unittest.TestCase):
    """Tests for BCa bootstrap and FDR correction"""
    
    def test_bca_bootstrap_coverage(self):
        """Test that BCa bootstrap achieves nominal coverage"""
        # Generate known distribution
        true_mean = 0.1
        n_samples = 1000
        n_experiments = 1000
        coverage_count = 0
        
        for _ in range(n_experiments):
            sample = np.random.normal(true_mean, 0.05, n_samples)
            ci = bca_bootstrap(sample, alpha=0.05)
            
            if ci.lower <= true_mean <= ci.upper:
                coverage_count += 1
                
        coverage_rate = coverage_count / n_experiments
        self.assertGreater(coverage_rate, 0.93)  # Should be ~95%
        
    def test_fdr_control(self):
        """Test that FDR correction controls false discovery rate"""
        n_tests = 1000
        n_true_null = 800  # 80% are null hypotheses
        alpha = 0.05
        
        # Generate p-values: 80% null, 20% alternative
        null_p_values = np.random.uniform(0, 1, n_true_null)
        alt_p_values = np.random.beta(1, 10, n_tests - n_true_null)  # Skewed toward 0
        p_values = np.concatenate([null_p_values, alt_p_values])
        
        fdr_result = benjamini_hochberg_correction(p_values, alpha=alpha)
        
        # Check FDR control
        false_discoveries = np.sum(fdr_result.rejected[:n_true_null])
        total_discoveries = np.sum(fdr_result.rejected)
        
        if total_discoveries > 0:
            fdr_observed = false_discoveries / total_discoveries
            self.assertLessEqual(fdr_observed, alpha * 1.1)  # Allow 10% margin
```

### Running Tests

**Complete Test Suite**:
```bash
# Run all tests with coverage
pytest tests/ --cov=bem2 --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/test_perftrack/ -v
pytest tests/test_router/ -v
pytest tests/test_integration/ -v

# Run tests with performance profiling
pytest tests/test_performance/ --benchmark-only
```

**Continuous Integration**:
```yaml
# .github/workflows/tests.yml
name: BEM v1.3 Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
        
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        
    - name: Run tests
      run: |
        pytest tests/ --cov=bem2 --cov-fail-under=85
        
    - name: Run integration tests
      run: |
        python validate_structure.py --comprehensive
        python test_bem_v13_integration.py
```

## 📝 Contributing Guidelines

### Code Style and Standards

**1. Code Formatting**:
```bash
# Use Black for code formatting
black bem2/ tests/ --line-length 88

# Use isort for import sorting  
isort bem2/ tests/ --profile black

# Use flake8 for linting
flake8 bem2/ tests/ --max-line-length 88
```

**2. Type Hints**:
```python
# All public APIs must include type hints
def run_experiment(
    config: BEMConfig, 
    baseline_model: torch.nn.Module,
    device: torch.device = torch.device("cpu")
) -> ExperimentResult:
    """
    Run BEM experiment with statistical validation.
    
    Args:
        config: Experiment configuration
        baseline_model: Baseline model for comparison
        device: Device for computation
        
    Returns:
        Complete experiment results with statistical validation
        
    Raises:
        BudgetParityError: If variant exceeds ±5% parameter/FLOP budget
        StatisticalValidationError: If bootstrap/FDR computation fails
    """
    pass
```

**3. Documentation Standards**:
```python
class AgenticRouter(nn.Module):
    """
    Agentic router with TRPO-style macro-policy.
    
    This router implements macro-actions at chunk boundaries with trust region
    optimization and hysteresis prevention. Key features:
    
    - TRPO-style trust region optimization with KL divergence constraints
    - Hysteresis mechanism to prevent excessive action switching  
    - Trust region projection to maintain ||ΔW||_F ≤ τ constraints
    - Expert composition with orthogonal subspace allocation
    
    Args:
        state_dim: Dimension of state representation
        action_dim: Dimension of action space
        experts_list: List of expert modules for composition
        trust_region_bound: Maximum Frobenius norm for weight updates
        
    Example:
        >>> router = AgenticRouter(
        ...     state_dim=768,
        ...     action_dim=4,  # (expert_id, span, rank_budget, bias_scale)
        ...     experts_list=[('code', code_expert), ('formal', formal_expert)],
        ...     trust_region_bound=1.0
        ... )
        >>> output = router(tokens, retrieval_features)
    """
```

### Contribution Workflow

**1. Feature Development**:
```bash
# 1. Create feature branch
git checkout -b feature/my-new-variant

# 2. Implement feature with tests
# - Add implementation in bem2/
# - Add comprehensive tests in tests/
# - Update configuration schemas
# - Add documentation

# 3. Run validation
python validate_structure.py --check-implementation
pytest tests/ --cov=bem2

# 4. Create pull request
git push origin feature/my-new-variant
# Open PR with description of changes and test results
```

**2. Pull Request Requirements**:
- [ ] **Code Quality**: All linting and formatting checks pass
- [ ] **Test Coverage**: New code has ≥90% test coverage
- [ ] **Statistical Validation**: New variants include statistical tests
- [ ] **Budget Compliance**: Parameter/FLOP parity validation included
- [ ] **Documentation**: API documentation and usage examples  
- [ ] **Integration Tests**: End-to-end tests demonstrate functionality
- [ ] **Performance Benchmarks**: Latency/memory impact documented

**3. Review Process**:
```
PR Submission → Automated Testing → Code Review → Integration Testing → Merge
     ↓               ↓                  ↓              ↓               ↓
   GitHub CI     Test Coverage      Peer Review    Full Validation   main branch
   Validation    >90% required      by Maintainer  Suite Passes     Integration
```

## 🔧 Extending the System

### Adding New Performance Variants

**Step 1: Create variant implementation**:
```python
# bem2/perftrack/pt5_my_variant.py
class MyNewVariant(BEMComponent):
    """
    My new performance variant for BEM v1.3.
    
    Implements [description of novel technique] with the following features:
    - [Feature 1]
    - [Feature 2]  
    - [Feature 3]
    """
    
    def __init__(self, config: MyVariantConfig):
        super().__init__()
        self.config = config
        self.validate_config()
        self._build_layers()
        
    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass with budget parity guarantee"""
        # Ensure base computation
        base_output = self.base_layer(x)
        
        # Apply variant-specific transformation
        variant_output = self._apply_transformation(x, features)
        
        # Combine with scaling
        return base_output + variant_output * self.scaling
        
    def get_delta_weights(self) -> Dict[str, torch.Tensor]:
        """Get weight modifications for trust region projection"""
        return {
            'layer1': self.variant_layer1.weight,
            'layer2': self.variant_layer2.weight
        }
        
    def validate_budget_parity(self, baseline: BEMComponent) -> bool:
        """Ensure ±5% parameter and FLOP parity"""
        param_ratio = self.count_parameters() / baseline.count_parameters()
        flop_ratio = self.count_flops() / baseline.count_flops()
        
        return (0.95 <= param_ratio <= 1.05 and 
                0.95 <= flop_ratio <= 1.05)
```

**Step 2: Create configuration schema**:
```yaml
# experiments/v5_my_variant.yml
experiment_name: "v5_my_variant"
variant_type: "performance"

model:
  base_model: "microsoft/DialoGPT-medium"
  max_length: 1024
  
bem_config:
  variant: "my_new_variant"
  # Variant-specific parameters
  my_param1: 128
  my_param2: 0.1
  
training:
  learning_rate: 5e-4
  batch_size: 16
  max_epochs: 10
  
validation:
  bootstrap_samples: 10000
  confidence_level: 0.95
  fdr_alpha: 0.05
  
constraints:
  budget_tolerance: 0.05
  latency_budget: 1.15
  vram_budget: 1.05
  
acceptance_gates:
  min_improvement: 0.005
  statistical_significance: true
  ci_lower_bound_positive: true
```

**Step 3: Add comprehensive tests**:
```python
# tests/test_perftrack/test_pt5_my_variant.py
class TestMyNewVariant(unittest.TestCase):
    
    def test_budget_parity_compliance(self):
        """Test ±5% parameter/FLOP parity"""
        baseline = BaselineModel()
        variant = MyNewVariant(config)
        
        self.assertTrue(variant.validate_budget_parity(baseline))
        
    def test_statistical_significance(self):
        """Test that variant achieves expected improvement"""
        results = run_experiment_comparison(baseline, variant)
        
        self.assertGreater(results.bca_confidence_interval.lower, 0)
        self.assertLess(results.fdr_corrected_p_value, 0.05)
        
    def test_performance_gates(self):
        """Test latency and memory constraints"""
        perf_results = benchmark_variant(variant)
        
        self.assertLessEqual(perf_results.latency_ratio, 1.15)
        self.assertLessEqual(perf_results.vram_ratio, 1.05)
```

### Adding New Advanced Components

**Creating a New Advanced Component**:
```python
# bem2/advanced/my_component.py
class MyAdvancedComponent(AdvancedBEMComponent):
    """
    Advanced component with specific capabilities.
    
    This component implements [advanced technique] with the following properties:
    - Statistical validation framework integration
    - Production safety monitoring
    - Hot-swappable composition support
    """
    
    def __init__(self, config: AdvancedConfig):
        super().__init__()
        self.safety_monitor = SafetyMonitor(config.safety_config)
        self.performance_tracker = PerformanceTracker()
        self.composition_engine = CompositionEngine()
        
    def forward_with_monitoring(self, inputs, **kwargs):
        """Forward pass with safety monitoring"""
        # Pre-forward safety check
        self.safety_monitor.pre_forward_check(inputs)
        
        # Execute forward pass
        with self.performance_tracker:
            outputs = self.forward(inputs, **kwargs)
            
        # Post-forward validation
        self.safety_monitor.post_forward_check(outputs)
        
        return outputs
        
    def compose_with_others(self, other_components: List['AdvancedBEMComponent']):
        """Composition with interference prevention"""
        return self.composition_engine.compose_safely(
            [self] + other_components,
            orthogonality_constraint=True,
            trust_region_bound=1.0
        )
```

### Extending Statistical Validation

**Adding Custom Statistical Tests**:
```python
# bem2/evaluation/custom_statistics.py
def my_custom_statistical_test(
    baseline_scores: np.ndarray,
    variant_scores: np.ndarray,
    alpha: float = 0.05
) -> CustomTestResult:
    """
    Custom statistical test for specific use case.
    
    Implements [specific statistical method] for comparing baseline and variant
    performance with the following features:
    - Robust to outliers
    - Handles non-normal distributions
    - Controls for multiple comparisons
    
    Returns:
        CustomTestResult with p-value, effect size, and confidence interval
    """
    paired_differences = variant_scores - baseline_scores
    
    # Implement custom statistical method
    test_statistic = compute_my_test_statistic(paired_differences)
    p_value = compute_p_value(test_statistic)
    effect_size = compute_effect_size(paired_differences)
    
    # Bootstrap confidence interval
    bootstrap_samples = bootstrap_resample(paired_differences, n_samples=10000)
    ci_lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    
    return CustomTestResult(
        test_statistic=test_statistic,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=(ci_lower, ci_upper),
        method_description="My Custom Statistical Test"
    )
```

## ⚡ Performance Optimization

### CUDA Kernel Development

**Creating Custom Fused Kernels**:
```cuda
// bem2/kernels/my_fused_kernel.cu
__global__ void my_fused_operation_kernel(
    const float* input_a,
    const float* input_b, 
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    // Thread and block indexing
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y; 
    int hidden_idx = threadIdx.x;
    
    // Bounds checking
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_dim) {
        return;
    }
    
    // Compute global index
    int global_idx = batch_idx * seq_len * hidden_dim + 
                     seq_idx * hidden_dim + 
                     hidden_idx;
    
    // Fused operation implementation
    float a_val = input_a[global_idx];
    float b_val = input_b[global_idx];
    
    // Custom fused computation
    float result = my_complex_operation(a_val, b_val);
    
    output[global_idx] = result;
}

// Host interface
torch::Tensor my_fused_operation(
    torch::Tensor input_a,
    torch::Tensor input_b
) {
    // Input validation
    TORCH_CHECK(input_a.device() == input_b.device(), "Inputs must be on same device");
    TORCH_CHECK(input_a.dtype() == torch::kFloat32, "Only float32 supported");
    
    // Output tensor allocation
    auto output = torch::zeros_like(input_a);
    
    // Kernel launch configuration
    dim3 block_size(256);
    dim3 grid_size(
        (input_a.size(0) + block_size.x - 1) / block_size.x,
        (input_a.size(1) + block_size.y - 1) / block_size.y
    );
    
    // Launch kernel
    my_fused_operation_kernel<<<grid_size, block_size>>>(
        input_a.data_ptr<float>(),
        input_b.data_ptr<float>(),
        output.data_ptr<float>(),
        input_a.size(0),  // batch_size
        input_a.size(1),  // seq_len  
        input_a.size(2)   // hidden_dim
    );
    
    // Synchronization and error checking
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
```

### Memory Optimization

**Efficient Memory Management**:
```python
# bem2/utils/memory_optimizer.py
class MemoryOptimizer:
    """Memory optimization utilities for BEM components"""
    
    def __init__(self):
        self.memory_pool = {}
        self.allocation_tracker = {}
        
    def optimized_forward(self, module, inputs, **kwargs):
        """Forward pass with memory optimization"""
        # Pre-allocate output tensors
        output_shape = self._infer_output_shape(module, inputs)
        output_tensor = self._get_or_allocate_tensor(output_shape, inputs.device)
        
        # Use gradient checkpointing for memory efficiency
        if self.training:
            return checkpoint(module.forward, inputs, **kwargs)
        else:
            return module(inputs, **kwargs)
            
    def _get_or_allocate_tensor(self, shape, device):
        """Get tensor from pool or allocate new one"""
        key = (shape, device)
        
        if key in self.memory_pool and len(self.memory_pool[key]) > 0:
            return self.memory_pool[key].pop()
        else:
            return torch.zeros(shape, device=device)
            
    def return_tensor(self, tensor):
        """Return tensor to memory pool"""
        key = (tuple(tensor.shape), tensor.device)
        
        if key not in self.memory_pool:
            self.memory_pool[key] = []
            
        # Clear tensor and return to pool
        tensor.zero_()
        self.memory_pool[key].append(tensor)
```

## 🐛 Debugging and Profiling

### Debugging Tools

**Comprehensive Debugging Framework**:
```python
# bem2/debug/debug_tools.py
class BEMDebugger:
    """Comprehensive debugging tools for BEM components"""
    
    def __init__(self, log_level='INFO'):
        self.logger = logging.getLogger('BEM_Debug')
        self.logger.setLevel(getattr(logging, log_level))
        self.trace_data = {}
        
    def trace_forward_pass(self, module, inputs, module_name):
        """Trace forward pass with detailed logging"""
        self.logger.info(f"Starting forward pass: {module_name}")
        
        # Input analysis
        input_stats = self._analyze_tensor(inputs, f"{module_name}_input")
        self.trace_data[f"{module_name}_input"] = input_stats
        
        # Forward pass with timing
        start_time = time.time()
        outputs = module(inputs)
        elapsed_time = time.time() - start_time
        
        # Output analysis  
        output_stats = self._analyze_tensor(outputs, f"{module_name}_output")
        self.trace_data[f"{module_name}_output"] = output_stats
        
        # Log performance metrics
        self.logger.info(f"Completed {module_name}: {elapsed_time:.4f}s")
        
        return outputs
        
    def _analyze_tensor(self, tensor, name):
        """Analyze tensor statistics"""
        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item()
        }
        
    def validate_numerical_stability(self, module):
        """Check for numerical stability issues"""
        issues = []
        
        for name, param in module.named_parameters():
            # Check for exploding gradients
            if param.grad is not None:
                grad_norm = torch.norm(param.grad)
                if grad_norm > 10.0:
                    issues.append(f"Large gradient in {name}: {grad_norm:.4f}")
                    
            # Check parameter magnitudes
            param_norm = torch.norm(param.data)
            if param_norm > 100.0:
                issues.append(f"Large parameters in {name}: {param_norm:.4f}")
                
        return issues
```

### Performance Profiling

**Detailed Performance Profiling**:
```python
# bem2/profiling/performance_profiler.py
class BEMProfiler:
    """Performance profiling for BEM components"""
    
    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
        self.gpu_utilization = {}
        
    def profile_experiment(self, experiment_func, *args, **kwargs):
        """Profile complete experiment execution"""
        # GPU memory baseline
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated()
            
        # CPU profiling
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=True) as prof:
            
            result = experiment_func(*args, **kwargs)
            
        # Collect profiling data
        self._process_profiling_results(prof)
        
        # GPU memory analysis
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage = peak_memory - baseline_memory
            self.memory_data['peak_gpu_memory'] = peak_memory
            self.memory_data['additional_memory'] = memory_usage
            
        return result
        
    def _process_profiling_results(self, prof):
        """Process and store profiling results"""
        # Key function analysis
        key_functions = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20
        )
        
        # Memory timeline
        memory_timeline = prof.profiler.kineto_results.events()
        
        # Store results
        self.timing_data['key_functions'] = key_functions
        self.memory_data['timeline'] = memory_timeline
        
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'timing_analysis': self.timing_data,
            'memory_analysis': self.memory_data,
            'gpu_utilization': self.gpu_utilization,
            'optimization_recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _generate_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Memory optimization suggestions
        if self.memory_data.get('peak_gpu_memory', 0) > 0.8 * torch.cuda.max_memory_allocated():
            recommendations.append("Consider using gradient checkpointing to reduce memory usage")
            
        # Compute optimization suggestions
        cpu_intensive_ops = [op for op in self.timing_data.get('key_functions', [])
                           if 'cpu' in op and op['cpu_time'] > 1000]
        if cpu_intensive_ops:
            recommendations.append("Consider moving CPU-intensive operations to GPU")
            
        return recommendations
```

This comprehensive developer guide provides all necessary information for contributing to and extending the BEM v1.3 system. The framework supports research-grade development with rigorous testing, statistical validation, and performance optimization capabilities.