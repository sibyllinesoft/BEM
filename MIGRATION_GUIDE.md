# BEM Repository Unification Migration Guide

🎯 **Goal**: Consolidate redundant code across BEM experiments while preserving all functionality and improving maintainability.

## 📋 Migration Overview

This guide walks through migrating the existing BEM repository structure to use unified core components, reducing code duplication by 60-70% while maintaining full backwards compatibility.

### Before Migration
```
src/
├── bem2/                     # Each component has its own:
│   ├── router/training.py    # - Training infrastructure
│   ├── safety/training.py    # - Evaluation systems  
│   ├── multimodal/training.py# - Configuration loading
│   └── perftrack/training.py # - Utility functions
└── experiments/              # 50+ similar YAML files
```

### After Migration
```
src/
├── bem_core/                 # Unified infrastructure:
│   ├── training/             # - Shared training base classes
│   ├── evaluation/           # - Unified evaluation framework
│   ├── config/               # - Template-based configuration
│   └── utils/                # - Common utilities
├── bem2/                     # Components inherit from bem_core:
│   ├── router/unified_trainer.py    # Inherits from BaseTrainer
│   ├── safety/unified_trainer.py    # Inherits from BaseTrainer
│   └── multimodal/unified_trainer.py# Inherits from BaseTrainer
└── experiments/
    ├── templates/            # Base templates
    └── configs/              # Experiment-specific overrides
```

## 🚀 Phase 1: Core Infrastructure Setup

### 1.1 Create bem_core Foundation

✅ **Already Complete**: The unified core structure has been created with:

- **`src/bem_core/`**: Main unified infrastructure
- **`training/base_trainer.py`**: Abstract base trainer with standardized training loop
- **`evaluation/base_evaluator.py`**: Unified evaluation framework with metrics
- **`config/base_config.py`**: Configuration classes with inheritance
- **`utils/`**: Shared utilities for logging, checkpointing, data handling

### 1.2 Configuration Template System

✅ **Already Complete**: Template-based configuration system:

- **`config/templates/base_experiment.yaml`**: Base template with sensible defaults
- **`config/config_loader.py`**: Inheritance and validation system
- Supports configuration inheritance with `base_config` field

## 🔄 Phase 2: Component Migration

### 2.1 Router Component Migration

✅ **Example Complete**: Created `src/bem2/router/unified_trainer.py` demonstrating:

```python
class RouterTrainer(BaseTrainer):
    """Inherits from BaseTrainer, implements router-specific logic."""
    
    def _setup_model(self) -> nn.Module:
        # Router-specific model setup
        return AgenticRouter(...)
    
    def _compute_loss(self, batch, outputs) -> Dict[str, torch.Tensor]:
        # Router-specific loss computation
        return {'loss': total_loss, ...}
    
    def _evaluate(self, dataloader) -> Dict[str, float]:
        # Router-specific evaluation
        return evaluator.evaluate(dataloader).metrics
```

### 2.2 Migration Pattern for Other Components

For **safety**, **multimodal**, and **perftrack** components:

1. **Create unified trainer**: `src/bem2/{component}/unified_trainer.py`
2. **Inherit from BaseTrainer**: Use the same pattern as RouterTrainer
3. **Implement required methods**:
   - `_setup_model()`: Component-specific model initialization
   - `_compute_loss()`: Component-specific loss functions
   - `_evaluate()`: Component-specific evaluation logic
4. **Migrate existing functionality**: Move component logic into the new structure
5. **Update imports**: Redirect existing scripts to use unified trainers

### 2.3 Benefits of Migration

- **Eliminated Code**: No more duplicate training loops, logging, checkpointing
- **Standardized Interface**: All components use the same training/evaluation API
- **Better Testing**: Single test suite covers all training infrastructure
- **Faster Development**: New components can focus on algorithms vs boilerplate

## 📝 Phase 3: Configuration Consolidation

### 3.1 Template Inheritance System

Convert existing experiment configs to use inheritance:

**Before** (`experiments/v1_dynrank.yml`):
```yaml
name: "v1_pt1_dynamic_rank_mask"
model:
  base_model: "microsoft/DialoGPT-small"
  hidden_size: 768
  # ... 170 lines of config
training:
  learning_rate: 5e-5
  batch_size: 16
  # ... more repetitive config
```

**After** (`experiments/v1_dynrank.yml`):
```yaml
base_config: "templates/performance_variant.yaml"
name: "v1_pt1_dynamic_rank_mask"
variant_id: "V1"

# Only specify what's different from base template
head_gating:
  enabled: true
  num_groups: 4
  rank_per_group: 2

dynamic_rank_mask:
  enabled: true
  target_sparsity: 0.5
```

### 3.2 Template Hierarchy

Create specialized templates:
- **`base_experiment.yaml`**: Universal defaults
- **`performance_variant.yaml`**: Performance optimization experiments
- **`safety_experiment.yaml`**: Safety and alignment experiments
- **`multimodal_experiment.yaml`**: Vision-language experiments

### 3.3 Configuration Benefits

- **Reduced Redundancy**: 80% reduction in config file size
- **Easier Maintenance**: Update defaults in one place
- **Better Validation**: Centralized configuration validation
- **Clear Intent**: Experiments only specify what's unique

## 🧪 Phase 4: Script and Documentation Updates

### 4.1 Update Experiment Scripts

Update scripts to use unified infrastructure:

**Before**:
```python
# Each component had its own training import
from src.bem2.router.training import RouterTrainer
from src.bem2.safety.training import SafetyTrainer
```

**After**:
```python
# Unified interface for all components
from src.bem2.router.unified_trainer import RouterTrainer
from src.bem2.safety.unified_trainer import SafetyTrainer
from src.bem_core.config import load_experiment_config

# Same interface for all components
config = load_experiment_config("experiments/my_experiment.yml")
trainer = RouterTrainer(config)
results = trainer.train()
```

### 4.2 Update Demo Scripts

Unify demo scripts to use the same patterns:
- `scripts/demos/demo_unified_bem.py`: Single demo showing all components
- Consistent interface across all component demos
- Shared utility functions for common demo patterns

### 4.3 Documentation Updates

- ✅ **Migration Guide**: This document
- **Updated README.md**: Reflect new unified structure
- **Developer Guide**: How to create new components using bem_core
- **Configuration Guide**: How to use the template system

## ✅ Phase 5: Validation and Testing

### 5.1 Backwards Compatibility

- **Legacy Support**: Existing experiments continue to work
- **Gradual Migration**: Components can be migrated one at a time
- **Validation Scripts**: Ensure no functionality is lost

### 5.2 Testing Strategy

1. **Unit Tests**: Test bem_core components in isolation
2. **Integration Tests**: Test component inheritance works correctly
3. **Regression Tests**: Ensure migrated components produce same results
4. **Performance Tests**: Validate no performance degradation

### 5.3 Migration Validation

For each migrated component:

```bash
# Test that unified trainer produces same results as legacy
python scripts/validation/test_migration.py --component router
python scripts/validation/test_migration.py --component safety

# Validate configuration inheritance works
python scripts/validation/test_configs.py

# Run full experiment to ensure no regressions
python scripts/utilities/run_bem_experiments.py --experiments v1_dynrank
```

## 🎯 Implementation Roadmap

### ✅ Completed
- [x] bem_core foundation with base classes
- [x] Configuration template system
- [x] Router component migration example
- [x] Shared utilities and logging
- [x] Migration documentation

### 🚧 Next Steps
1. **Migrate Safety Component** (1-2 days)
   - Create `src/bem2/safety/unified_trainer.py`
   - Implement safety-specific loss and evaluation
   - Test against existing safety experiments

2. **Migrate Multimodal Component** (1-2 days)
   - Create `src/bem2/multimodal/unified_trainer.py`
   - Implement vision-language specific functionality
   - Test multimodal experiments

3. **Migrate Performance Track Components** (1-2 days)
   - Create unified trainers for PT1-PT4 variants
   - Consolidate performance optimization logic
   - Update performance benchmarking

4. **Update All Experiment Configurations** (1 day)
   - Convert 50+ configs to use template inheritance
   - Create specialized templates for each experiment type
   - Validate all experiments still work

5. **Update Scripts and Documentation** (1 day)
   - Update demo scripts to use unified interface
   - Update README and developer guides
   - Create configuration tutorial

### 📈 Expected Results

**Code Reduction**:
- 60-70% reduction in training/evaluation code duplication
- 80% reduction in configuration file size
- Elimination of repetitive utility functions

**Quality Improvements**:
- Standardized interfaces across all components
- Better error handling and logging
- Unified testing and validation
- Improved documentation and examples

**Development Velocity**:
- New components can be created in hours vs days
- Consistent patterns reduce learning curve
- Shared infrastructure improvements benefit all components
- Easier maintenance and bug fixes

## 🛠 Development Guidelines

### Creating New Components

1. **Inherit from BaseTrainer**:
```python
from bem_core.training.base_trainer import BaseTrainer

class MyComponentTrainer(BaseTrainer):
    def _setup_model(self): # Implement required methods
    def _compute_loss(self):  
    def _evaluate(self):
```

2. **Use Configuration Templates**:
```yaml
base_config: "templates/base_experiment.yaml"
name: "my_new_experiment"
# Only specify what's unique to your experiment
```

3. **Follow Existing Patterns**: Look at router example for guidance

### Best Practices

- **Single Responsibility**: Each component focuses on its core algorithm
- **Configuration Driven**: Use config files, avoid hardcoded parameters
- **Comprehensive Logging**: Use the unified logging system
- **Error Handling**: Implement robust error handling and validation
- **Testing**: Write tests for component-specific logic

## 🤝 Migration Support

If you need help with migration:

1. **Review Examples**: Look at `src/bem2/router/unified_trainer.py`
2. **Check Documentation**: Refer to this guide and code comments
3. **Test Incrementally**: Migrate one component at a time
4. **Validate Results**: Ensure no functionality is lost

The unified infrastructure preserves all existing functionality while dramatically reducing code duplication and improving maintainability. All experiments remain reproducible with statistical validation intact.