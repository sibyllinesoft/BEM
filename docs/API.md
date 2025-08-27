# BEM API Reference

This document provides comprehensive API documentation for the BEM (Basis Extension Modules) system, covering all unified components and their interfaces.

## Table of Contents

- [Core API](#core-api)
- [Configuration System](#configuration-system)
- [Training API](#training-api)
- [Evaluation API](#evaluation-api)
- [Routing System](#routing-system)
- [Safety System](#safety-system)
- [Multimodal Components](#multimodal-components)
- [Online Learning](#online-learning)
- [Performance Tracking](#performance-tracking)
- [Utilities](#utilities)
- [Error Handling](#error-handling)
- [Security Considerations](#security-considerations)

## Core API

### BEMModel

The main entry point for using BEM functionality.

```python
from bem_core import BEMModel, BEMConfig

class BEMModel:
    """Main BEM model interface for dynamic neural adaptation."""
    
    def __init__(self, config: BEMConfig):
        """Initialize BEM model.
        
        Args:
            config: BEM configuration object
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If model initialization fails
        """
    
    def generate(
        self,
        inputs: Union[str, torch.Tensor, Dict[str, Any]],
        context_hints: Optional[List[str]] = None,
        adaptation_strength: float = 1.0,
        max_length: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate outputs with dynamic adaptation.
        
        Args:
            inputs: Input text, tensor, or structured input
            context_hints: Optional context cues for routing
            adaptation_strength: Adaptation intensity (0.0-1.0)
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing:
                - generated_text: Generated output text
                - routing_decisions: Expert routing information
                - adaptation_info: Adaptation metadata
                - safety_scores: Safety evaluation results
                
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If generation fails
        """
    
    def adapt(
        self,
        task_data: Dict[str, Any],
        adaptation_mode: str = "online",
        learning_rate: float = 1e-4,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """Adapt model to new task or domain.
        
        Args:
            task_data: Training data for adaptation
            adaptation_mode: "online", "offline", or "mixed"
            learning_rate: Learning rate for adaptation
            max_steps: Maximum adaptation steps
            
        Returns:
            Adaptation results and metrics
        """
    
    def evaluate(
        self,
        eval_data: Dict[str, Any],
        metrics: List[str],
        return_detailed: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            eval_data: Evaluation dataset
            metrics: List of metrics to compute
            return_detailed: Whether to return detailed results
            
        Returns:
            Evaluation results and statistics
        """
```

### BEMConfig

Configuration system for BEM models.

```python
from bem_core import BEMConfig
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class BEMConfig:
    """Configuration for BEM models."""
    
    # Model Architecture
    base_model: str = "microsoft/DialoGPT-small"
    expert_count: int = 8
    routing_dim: int = 256
    adaptation_rank: int = 16
    
    # Routing Configuration
    routing_strategy: str = "learned"  # "learned", "random", "round_robin"
    routing_temperature: float = 1.0
    routing_top_k: int = 2
    
    # Training Configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    warmup_steps: int = 100
    
    # Safety Configuration
    safety_enabled: bool = True
    safety_threshold: float = 0.95
    constitutional_ai: bool = True
    
    # Performance Configuration
    use_fp16: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False
    
    # Advanced Options
    advanced_routing: bool = False
    multimodal_support: bool = False
    online_learning: bool = False
    
    def validate(self) -> None:
        """Validate configuration parameters."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BEMConfig':
        """Create configuration from dictionary."""
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BEMConfig':
        """Load configuration from YAML file."""
```

## Training API

### Unified Trainer

```python
from bem2.training import UnifiedTrainer

class UnifiedTrainer:
    """Unified training interface for all BEM variants."""
    
    def __init__(
        self,
        model: BEMModel,
        config: BEMConfig,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None
    ):
        """Initialize trainer.
        
        Args:
            model: BEM model to train
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
        """
    
    def train(
        self,
        num_epochs: int,
        save_checkpoints: bool = True,
        checkpoint_frequency: int = 10,
        early_stopping_patience: int = 5
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_epochs: Number of training epochs
            save_checkpoints: Whether to save checkpoints
            checkpoint_frequency: Checkpoint saving frequency
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training results and metrics
        """
    
    def validate(self) -> Dict[str, float]:
        """Run validation evaluation."""
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Save training checkpoint."""
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
```

## Routing System

### Agentic Router

```python
from bem2.router import AgenticRouter, RoutingDecision

class AgenticRouter:
    """Intelligent routing system with agentic capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agentic router.
        
        Args:
            config: Router configuration
        """
    
    def route(
        self,
        inputs: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Route inputs to appropriate experts.
        
        Args:
            inputs: Input tensor
            context: Optional context information
            constraints: Optional routing constraints
            
        Returns:
            Routing decision with expert selections and weights
        """
    
    def learn_routing(
        self,
        feedback: Dict[str, Any],
        learning_rate: float = 1e-4
    ) -> None:
        """Learn from routing feedback."""
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""

@dataclass
class RoutingDecision:
    """Routing decision information."""
    expert_ids: List[int]
    expert_weights: torch.Tensor
    routing_scores: torch.Tensor
    confidence: float
    metadata: Dict[str, Any]
```

## Safety System

### Constitutional Safety

```python
from bem2.safety import ConstitutionalScorer, SafetyController

class ConstitutionalScorer:
    """Constitutional AI safety scoring system."""
    
    def __init__(self, constitution_path: str):
        """Initialize with constitutional principles.
        
        Args:
            constitution_path: Path to constitution file
        """
    
    def score_output(
        self,
        output_text: str,
        context: Optional[str] = None
    ) -> Dict[str, float]:
        """Score output for safety violations.
        
        Args:
            output_text: Generated text to evaluate
            context: Optional context for evaluation
            
        Returns:
            Dictionary of safety scores and violation types
        """
    
    def check_violation(
        self,
        output_text: str,
        threshold: float = 0.95
    ) -> bool:
        """Check if output violates safety constraints."""

class SafetyController:
    """Safety control and intervention system."""
    
    def __init__(
        self,
        scorer: ConstitutionalScorer,
        intervention_threshold: float = 0.95
    ):
        """Initialize safety controller."""
    
    def monitor_generation(
        self,
        model: BEMModel,
        inputs: torch.Tensor,
        max_violations: int = 3
    ) -> Dict[str, Any]:
        """Monitor generation for safety violations."""
    
    def intervene(
        self,
        violation_type: str,
        severity: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Intervene when safety violations are detected."""
```

## Multimodal Components

### Vision-Text Integration

```python
from bem2.multimodal import MultimodalBEM, VisionEncoder

class MultimodalBEM:
    """Multimodal BEM with vision-text capabilities."""
    
    def __init__(self, config: BEMConfig):
        """Initialize multimodal BEM."""
    
    def process_multimodal(
        self,
        text_inputs: str,
        image_inputs: Optional[torch.Tensor] = None,
        audio_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Process multimodal inputs.
        
        Args:
            text_inputs: Text input
            image_inputs: Optional image tensor
            audio_inputs: Optional audio tensor
            
        Returns:
            Multimodal processing results
        """
    
    def cross_modal_routing(
        self,
        modalities: List[str],
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Route based on multiple modalities."""

class VisionEncoder:
    """Vision encoder for multimodal processing."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize vision encoder."""
    
    def encode_image(
        self,
        image: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Encode image to feature representation."""
```

## Online Learning

### Online Learning Framework

```python
from bem2.online import OnlineLearner, StreamingDataset

class OnlineLearner:
    """Online learning system for continuous adaptation."""
    
    def __init__(
        self,
        model: BEMModel,
        config: Dict[str, Any],
        memory_size: int = 10000
    ):
        """Initialize online learner.
        
        Args:
            model: BEM model for online learning
            config: Online learning configuration
            memory_size: Size of experience replay buffer
        """
    
    def process_stream(
        self,
        data_stream: Iterator[Dict[str, Any]],
        adaptation_frequency: int = 100,
        evaluation_frequency: int = 1000
    ) -> Dict[str, Any]:
        """Process streaming data with online adaptation.
        
        Args:
            data_stream: Streaming data iterator
            adaptation_frequency: How often to adapt
            evaluation_frequency: How often to evaluate
            
        Returns:
            Online learning results and metrics
        """
    
    def adapt_online(
        self,
        batch: Dict[str, Any],
        learning_rate: float = 1e-4
    ) -> Dict[str, float]:
        """Perform online adaptation on a batch."""
    
    def detect_drift(
        self,
        recent_performance: List[float],
        historical_baseline: float
    ) -> bool:
        """Detect concept drift in the data stream."""
```

## Performance Tracking

### Performance Variants

```python
from bem2.perftrack import PerformanceTracker, OptimizationVariant

class PerformanceTracker:
    """Performance tracking and optimization system."""
    
    def __init__(self, base_model: BEMModel):
        """Initialize performance tracker."""
    
    def benchmark_variant(
        self,
        variant: OptimizationVariant,
        benchmark_data: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, float]:
        """Benchmark a performance optimization variant."""
    
    def compare_variants(
        self,
        variants: List[OptimizationVariant],
        benchmark_suite: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple optimization variants."""

class OptimizationVariant:
    """Optimization variant implementation."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize optimization variant."""
    
    def apply_optimization(self, model: BEMModel) -> BEMModel:
        """Apply optimization to model."""
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the optimization."""
```

## Evaluation API

### Comprehensive Evaluation Framework

```python
from bem2.evaluation import EvaluationFramework, StatisticalAnalysis

class EvaluationFramework:
    """Comprehensive evaluation system with statistical validation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluation framework."""
    
    def evaluate_comprehensive(
        self,
        model: BEMModel,
        datasets: Dict[str, Any],
        metrics: List[str],
        num_bootstrap_samples: int = 10000
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation with statistical validation.
        
        Args:
            model: Model to evaluate
            datasets: Evaluation datasets
            metrics: Metrics to compute
            num_bootstrap_samples: Bootstrap samples for confidence intervals
            
        Returns:
            Evaluation results with statistical analysis
        """
    
    def ablation_study(
        self,
        base_model: BEMModel,
        components_to_ablate: List[str],
        eval_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform ablation study."""
    
    def compare_methods(
        self,
        models: Dict[str, BEMModel],
        eval_data: Dict[str, Any],
        statistical_tests: List[str] = ["bootstrap", "permutation"]
    ) -> Dict[str, Any]:
        """Compare multiple methods with statistical testing."""

class StatisticalAnalysis:
    """Statistical analysis utilities."""
    
    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        confidence_level: float = 0.95,
        num_bootstrap_samples: int = 10000
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
    
    @staticmethod
    def effect_size_cohens_d(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """Compute Cohen's d effect size."""
    
    @staticmethod
    def multiple_testing_correction(
        p_values: List[float],
        method: str = "benjamini_hochberg"
    ) -> List[float]:
        """Apply multiple testing correction."""
```

## Utilities

### Checkpoint Management

```python
from bem_core.utils import CheckpointManager, ModelCheckpoint

class CheckpointManager:
    """Checkpoint management utilities."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager."""
    
    def save_checkpoint(
        self,
        model: BEMModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model checkpoint."""
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[BEMModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> ModelCheckpoint:
        """Load model checkpoint."""
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""

@dataclass
class ModelCheckpoint:
    """Model checkpoint information."""
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]]
    epoch: int
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str
```

### Logging Utilities

```python
from bem_core.utils import BEMLogger, MetricsLogger

class BEMLogger:
    """Logging utilities for BEM."""
    
    def __init__(self, name: str, level: str = "INFO"):
        """Initialize logger."""
    
    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log training step information."""
    
    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        dataset_name: str
    ) -> None:
        """Log evaluation results."""
    
    def log_routing_decision(
        self,
        routing_decision: RoutingDecision,
        input_info: Dict[str, Any]
    ) -> None:
        """Log routing decisions for analysis."""
```

## Error Handling

### Exception Hierarchy

```python
class BEMException(Exception):
    """Base exception for BEM-related errors."""
    pass

class ConfigurationError(BEMException):
    """Raised when configuration is invalid."""
    pass

class ModelInitializationError(BEMException):
    """Raised when model initialization fails."""
    pass

class RoutingError(BEMException):
    """Raised when routing fails."""
    pass

class SafetyViolationError(BEMException):
    """Raised when safety violations are detected."""
    pass

class TrainingError(BEMException):
    """Raised when training encounters errors."""
    pass

class EvaluationError(BEMException):
    """Raised when evaluation fails."""
    pass
```

### Error Context

```python
from bem_core.utils import ErrorContext

class ErrorContext:
    """Context manager for error handling and debugging."""
    
    def __init__(self, operation: str, debug_info: Dict[str, Any]):
        """Initialize error context."""
    
    def __enter__(self):
        """Enter error context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit error context with cleanup."""
        pass
    
    def add_debug_info(self, key: str, value: Any) -> None:
        """Add debug information."""
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report."""
```

## Security Considerations

### Input Validation

```python
from bem2.security import InputValidator, SecurityManager

class InputValidator:
    """Input validation and sanitization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize input validator."""
    
    def validate_text_input(
        self,
        text: str,
        max_length: int = 4096,
        allowed_characters: Optional[str] = None
    ) -> bool:
        """Validate text input for security."""
    
    def sanitize_input(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sanitize inputs for security."""
    
    def detect_malicious_patterns(
        self,
        text: str
    ) -> List[str]:
        """Detect potentially malicious patterns."""

class SecurityManager:
    """Overall security management."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize security manager."""
    
    def audit_log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security-relevant events."""
    
    def check_rate_limits(
        self,
        user_id: str,
        operation: str
    ) -> bool:
        """Check if operation is within rate limits."""
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security status report."""
```

## Usage Examples

### Basic Usage

```python
import torch
from bem_core import BEMModel, BEMConfig

# Configure BEM
config = BEMConfig(
    base_model="microsoft/DialoGPT-small",
    expert_count=8,
    routing_strategy="learned",
    safety_enabled=True
)

# Initialize model
model = BEMModel(config)

# Generate with dynamic adaptation
result = model.generate(
    inputs="Explain quantum computing to a beginner",
    context_hints=["educational", "technical"],
    adaptation_strength=0.8
)

print(f"Generated: {result['generated_text']}")
print(f"Routing: {result['routing_decisions']}")
print(f"Safety Score: {result['safety_scores']['overall']}")
```

### Training Example

```python
from bem2.training import UnifiedTrainer
from torch.utils.data import DataLoader

# Prepare data
train_dataset = YourDataset(train_data)
val_dataset = YourDataset(val_data)

# Initialize trainer
trainer = UnifiedTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

# Train model
results = trainer.train(
    num_epochs=50,
    save_checkpoints=True,
    early_stopping_patience=5
)

print(f"Final validation accuracy: {results['val_accuracy']:.3f}")
```

### Evaluation Example

```python
from bem2.evaluation import EvaluationFramework

# Initialize evaluation framework
evaluator = EvaluationFramework(config={
    "statistical_validation": True,
    "num_bootstrap_samples": 10000
})

# Evaluate model
eval_results = evaluator.evaluate_comprehensive(
    model=model,
    datasets={"test": test_data},
    metrics=["accuracy", "f1", "safety_score"]
)

print(f"Test Accuracy: {eval_results['accuracy']['mean']:.3f} "
      f"±{eval_results['accuracy']['std']:.3f}")
```

## Version Compatibility

This API documentation is for BEM version 2.0.0 and later. For older versions:

- **Version 1.x**: See legacy documentation in `docs/legacy/`
- **Breaking Changes**: See `CHANGELOG.md` for migration guidance

## Support and Updates

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: Complete guides in `docs/`
- **Examples**: Working demos in `scripts/demos/`
- **API Changes**: Subscribe to repository releases for updates

---

**Last Updated**: December 2024  
**API Version**: 2.0.0  
**Compatibility**: Python 3.9+, PyTorch 2.0+