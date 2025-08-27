# Contributing to BEM

Thank you for your interest in contributing to Block-wise Expert Modules (BEM)! This document provides guidelines and information for contributors.

## 🌟 Ways to Contribute

- **🔬 Research**: Novel architectures, safety mechanisms, optimization techniques
- **⚡ Performance**: Kernel optimizations, memory efficiency, inference improvements
- **🧪 Experiments**: New benchmarks, evaluation metrics, statistical methods
- **📖 Documentation**: Guides, tutorials, API documentation, examples
- **🐛 Bug Reports**: Issues, reproducible test cases, edge case identification
- **💡 Feature Requests**: Enhancement proposals with clear use cases

## 🚀 Getting Started

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/BEM.git
   cd BEM
   git remote add upstream https://github.com/nathanrice/BEM.git
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Additional dev tools
   ```

3. **Download Required Models**
   ```bash
   python scripts/setup/download_models.py --required-only
   ```

4. **Verify Setup**
   ```bash
   python scripts/validation/validate_structure.py
   python scripts/validation/validate_pipeline.py
   pytest tests/ -v
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards (see below)
   - Add comprehensive tests
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run tests
   pytest tests/ -v --cov=src

   # Validate code quality
   python scripts/validation/validate_code_quality.py

   # Run statistical validation for performance claims
   python scripts/validation/validate_statistical_claims.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Code Standards

### Python Style Guidelines

- **Python Version**: 3.9+ required, 3.11+ recommended
- **Type Hints**: Comprehensive type annotations for all functions
- **Docstrings**: Google-style docstrings for all public functions
- **Line Length**: 88 characters maximum (Black formatter)
- **Import Organization**: isort compatible grouping

### Code Quality Requirements

```python
# Example of expected code quality
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class BEMConfig:
    """Configuration for BEM models.
    
    Args:
        num_experts: Number of expert modules
        routing_dim: Dimension of routing vectors  
        safety_threshold: Safety violation threshold
        learning_rate: Training learning rate
    """
    num_experts: int = 8
    routing_dim: int = 256
    safety_threshold: float = 0.95
    learning_rate: float = 1e-4


def compute_routing_scores(
    inputs: torch.Tensor,
    expert_embeddings: torch.Tensor,
    config: BEMConfig
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute routing scores for expert selection.
    
    Args:
        inputs: Input tensor of shape (batch_size, input_dim)
        expert_embeddings: Expert embedding matrix (num_experts, routing_dim)
        config: BEM configuration
        
    Returns:
        Tuple of routing scores and metrics dictionary
        
    Raises:
        ValueError: If input dimensions don't match configuration
    """
    if inputs.size(-1) != config.routing_dim:
        raise ValueError(f"Input dim {inputs.size(-1)} != routing dim {config.routing_dim}")
    
    # Implementation here...
    routing_scores = torch.matmul(inputs, expert_embeddings.t())
    
    metrics = {
        "max_score": float(routing_scores.max()),
        "entropy": float(torch.distributions.Categorical(logits=routing_scores).entropy().mean())
    }
    
    return routing_scores, metrics
```

### Testing Requirements

- **Coverage**: >90% test coverage for all new code
- **Test Types**: Unit tests, integration tests, statistical validation tests
- **Performance**: Benchmarks for performance-critical code
- **Documentation**: Test docstrings explaining test purpose and methodology

```python
import pytest
import torch
from unittest.mock import Mock, patch

from bem.routing import compute_routing_scores, BEMConfig


class TestRoutingScores:
    """Test suite for routing score computation."""
    
    @pytest.fixture
    def config(self) -> BEMConfig:
        """Standard test configuration."""
        return BEMConfig(num_experts=4, routing_dim=128)
    
    @pytest.fixture  
    def sample_inputs(self, config: BEMConfig) -> torch.Tensor:
        """Sample input tensor for testing."""
        return torch.randn(2, config.routing_dim)
    
    def test_routing_scores_shape(self, sample_inputs, config):
        """Test that routing scores have correct shape."""
        expert_embeddings = torch.randn(config.num_experts, config.routing_dim)
        
        scores, metrics = compute_routing_scores(sample_inputs, expert_embeddings, config)
        
        assert scores.shape == (2, config.num_experts)
        assert isinstance(metrics, dict)
        assert "entropy" in metrics
    
    def test_routing_scores_statistical_properties(self, sample_inputs, config):
        """Test statistical properties of routing scores."""
        expert_embeddings = torch.randn(config.num_experts, config.routing_dim)
        
        scores, metrics = compute_routing_scores(sample_inputs, expert_embeddings, config)
        
        # Statistical validation
        assert metrics["entropy"] > 0  # Should have some entropy
        assert torch.isfinite(scores).all()  # No NaN/Inf values
        
    def test_dimension_mismatch_error(self, config):
        """Test error handling for dimension mismatch."""
        wrong_inputs = torch.randn(2, config.routing_dim + 10)
        expert_embeddings = torch.randn(config.num_experts, config.routing_dim)
        
        with pytest.raises(ValueError, match="Input dim .* != routing dim"):
            compute_routing_scores(wrong_inputs, expert_embeddings, config)
```

## 🔬 Research Contributions

### Statistical Validation

All performance claims must include statistical validation:

- **Significance Testing**: p-values with multiple testing correction
- **Effect Size**: Cohen's d or equivalent measures
- **Confidence Intervals**: Bootstrap confidence intervals where appropriate
- **Reproducibility**: Complete experimental configuration and random seeds

### Experiment Configuration

```yaml
# experiments/your_experiment.yaml
experiment:
  name: "novel_routing_method"
  description: "Test novel routing architecture"
  
model:
  type: "bem_v2"
  config:
    num_experts: 8
    routing_method: "your_method"
    
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  
evaluation:
  metrics: ["accuracy", "routing_entropy", "safety_score"]
  statistical_tests: ["bootstrap_ci", "permutation_test"]
  
reproducibility:
  seed: 42
  deterministic: true
```

### Performance Benchmarks

Include benchmarks for performance-sensitive code:

```python
import time
import pytest
from contextlib import contextmanager


@contextmanager
def timer():
    """Simple timing context manager."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")


def test_routing_performance_benchmark(large_batch_inputs, config):
    """Benchmark routing computation performance."""
    expert_embeddings = torch.randn(config.num_experts, config.routing_dim)
    
    # Warm up
    for _ in range(10):
        compute_routing_scores(large_batch_inputs[:10], expert_embeddings, config)
    
    # Benchmark
    with timer():
        for _ in range(100):
            compute_routing_scores(large_batch_inputs, expert_embeddings, config)
    
    # Performance assertions
    with timer() as t:
        scores, metrics = compute_routing_scores(large_batch_inputs, expert_embeddings, config)
    
    # Should complete within reasonable time (adjust based on hardware)
    assert t < 0.1  # Less than 100ms for this batch size
```

## 📖 Documentation Standards

### Code Documentation

- **Public APIs**: Complete docstrings with examples
- **Complex Logic**: Inline comments explaining algorithmic choices
- **Mathematical Operations**: LaTeX math notation where helpful
- **Performance Notes**: Big-O complexity and optimization notes

### User Documentation

- **Tutorials**: Step-by-step guides with expected outputs
- **Examples**: Complete, runnable code examples
- **Integration Guides**: How to integrate BEM with other systems
- **Troubleshooting**: Common issues and solutions

## 🚨 Issue Reporting

### Bug Reports

Please include:
- **Description**: Clear, concise description of the bug
- **Reproduction Steps**: Minimal code to reproduce the issue
- **Environment**: Python version, GPU type, dependency versions
- **Expected vs Actual**: What you expected vs what happened
- **Logs**: Relevant error messages and stack traces

### Feature Requests

Please include:
- **Use Case**: Clear explanation of the need
- **Proposed Solution**: How you envision the feature working
- **Alternatives**: Other approaches you've considered
- **Implementation**: Willingness to contribute implementation

## 🎯 Pull Request Guidelines

### Before Submitting

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Code quality: `python scripts/validation/validate_code_quality.py`
- [ ] Documentation updated
- [ ] CHANGELOG.md entry added (for significant changes)
- [ ] Statistical validation (for performance claims)

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Research contribution

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance benchmarks included (if applicable)
- [ ] Statistical validation included (if applicable)

## Performance Impact
Describe any performance implications

## Breaking Changes
List any breaking changes and migration notes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added and pass
- [ ] No new warnings introduced
```

## 🤝 Community Guidelines

- **Be Respectful**: Treat all contributors with respect and kindness
- **Be Constructive**: Provide actionable feedback and suggestions
- **Be Patient**: Research and complex systems take time to understand
- **Share Knowledge**: Help others learn and grow
- **Stay Focused**: Keep discussions relevant to the project goals

## 📞 Getting Help

- **GitHub Discussions**: General questions and brainstorming
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and tutorials
- **Code Comments**: Detailed explanations in the codebase

## 🙏 Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **Academic Papers**: Research contributors as co-authors (when appropriate)

---

Thank you for contributing to BEM! Your efforts help advance the field of adaptive AI architectures.