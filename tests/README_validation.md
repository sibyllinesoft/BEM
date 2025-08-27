# BEM Migration Validation Test Suite

This directory contains comprehensive validation tests for the unified BEM infrastructure migration. The tests ensure that the migration from legacy components to the unified system preserves all functionality while introducing no regressions.

## Overview

The validation suite consists of several test categories:

1. **Unit Tests** - Test individual components and infrastructure
2. **Integration Tests** - Test component interactions and workflows  
3. **Migration Tests** - Compare unified vs legacy implementations
4. **Configuration Tests** - Validate template system and config loading
5. **Performance Tests** - Ensure no performance regressions

## Test Files

### Core Test Files

- `test_unified_infrastructure.py` - Tests for BaseTrainer, BaseEvaluator, and core utilities
- `test_component_migration.py` - Tests for RouterTrainer, SafetyTrainer, MultimodalTrainer migration
- `test_configuration_system.py` - Tests for config templates, inheritance, and loading
- `test_migration_suite.py` - Main test runner that orchestrates all validation

### Validation Scripts

- `scripts/validation/test_migration.py` - Comprehensive migration validation comparing unified vs legacy
- `scripts/validation/validate_all_configs.py` - Configuration system validation and health checks

## Quick Start

### Run All Tests

The fastest way to validate the entire migration:

```bash
# Run complete validation suite
python tests/test_migration_suite.py

# Run with verbose output
python tests/test_migration_suite.py --verbose

# Run only fast tests (skip comprehensive validation)
python tests/test_migration_suite.py --fast
```

### Run Specific Test Categories

```bash
# Run only unit tests
python tests/test_migration_suite.py --unit-tests-only

# Run only integration tests
python tests/test_migration_suite.py --integration-tests-only

# Run with custom output directory
python tests/test_migration_suite.py --output-dir my_results
```

### Run Individual Test Files

```bash
# Test unified infrastructure components
pytest tests/test_unified_infrastructure.py -v

# Test component migration
pytest tests/test_component_migration.py -v

# Test configuration system
pytest tests/test_configuration_system.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Run Validation Scripts Directly

```bash
# Migration validation (compare unified vs legacy)
python scripts/validation/test_migration.py --component all --verbose

# Configuration validation  
python scripts/validation/validate_all_configs.py --verbose

# Component-specific validation
python scripts/validation/test_migration.py --component router
python scripts/validation/test_migration.py --component safety
python scripts/validation/test_migration.py --component multimodal
```

## Test Structure

### Unit Tests (`test_unified_infrastructure.py`)

Tests the core unified infrastructure:

- **BaseTrainer** - Abstract trainer functionality, training loops, checkpointing
- **BaseEvaluator** - Evaluation framework, metrics computation, result serialization  
- **Configuration System** - Config loading, template inheritance, validation
- **Shared Utilities** - Logging, checkpointing, error handling

### Component Migration Tests (`test_component_migration.py`)

Tests that specialized trainers correctly inherit from base infrastructure:

- **RouterTrainer** - Expert routing, load balancing, composition strategies
- **SafetyTrainer** - Constitutional AI, violation detection, safety scoring
- **MultimodalTrainer** - Vision-text fusion, cross-modal attention, alignment
- **Migration Utilities** - Legacy config conversion, backward compatibility

### Configuration Tests (`test_configuration_system.py`)

Tests the configuration template system:

- **Template Loading** - YAML parsing, inheritance resolution, field validation
- **Template Inheritance** - Multi-level inheritance, circular reference detection  
- **Config Validation** - Type checking, required field validation, error handling
- **Experiment Conversion** - Legacy format conversion, backward compatibility

### Migration Validation (`test_migration.py`)

Comprehensive validation comparing unified vs legacy implementations:

- **Result Equivalence** - Statistical comparison of training outcomes
- **Performance Benchmarks** - Memory usage, training speed, convergence
- **Feature Preservation** - Component-specific functionality validation
- **Backward Compatibility** - Legacy checkpoint loading, config conversion

### Configuration Validation (`validate_all_configs.py`)

Health check for the entire configuration system:

- **Template Discovery** - Find and validate all templates
- **Inheritance Validation** - Check inheritance graphs for cycles/orphans
- **Config Loading** - Test loading all experiment configs
- **Error Reporting** - Comprehensive error and warning collection

## Expected Results

### Passing Tests Indicate:

✅ **Infrastructure Migration Successful**
- All unified trainers inherit correctly from BaseTrainer
- Configuration system loads all templates and experiments
- No functionality lost during migration
- Performance maintained or improved

✅ **System Ready for Production**
- All component-specific features preserved
- Backward compatibility maintained  
- Error handling robust
- Configuration system healthy

### Failing Tests Indicate:

❌ **Migration Issues Need Resolution**
- Unified trainers missing functionality
- Configuration loading errors
- Performance regressions
- Backward compatibility broken

## Test Output

### Test Results Structure

```
migration_test_results/
├── migration_validation_summary.md     # Overall summary report
├── pytest_report.html                  # Detailed unit test results  
├── coverage_html/                      # Code coverage reports
├── migration_validation/               # Migration validation results
│   ├── migration_validation_results.json
│   └── migration_validation_report.md
└── config_validation/                  # Configuration validation results
    ├── config_validation_results.json
    └── config_validation_report.md
```

### Reading Results

1. **Start with** `migration_validation_summary.md` for overall status
2. **Check** `pytest_report.html` for detailed unit test results
3. **Review** `migration_validation_report.md` for migration-specific issues
4. **Examine** `config_validation_report.md` for configuration problems
5. **Check logs** in `migration_test_suite.log` for debugging details

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure Python path includes src/
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or run from project root
cd /path/to/BEM && python tests/test_migration_suite.py
```

**Missing Dependencies**
```bash
# Install test dependencies
pip install pytest pytest-html pytest-cov pyyaml

# Install project dependencies  
pip install -r requirements.txt
```

**Permission Errors**
```bash
# Make scripts executable
chmod +x tests/test_migration_suite.py
chmod +x scripts/validation/test_migration.py  
chmod +x scripts/validation/validate_all_configs.py
```

**Legacy Components Not Available**
- Some migration tests skip comparison when legacy components aren't available
- This is expected and tests will still validate unified component functionality
- Look for "SKIPPED" status in test results

### Debug Mode

For detailed debugging information:

```bash
# Maximum verbosity
python tests/test_migration_suite.py --verbose

# Stop on first failure
python tests/test_migration_suite.py --fail-fast

# Run individual components for isolation
python scripts/validation/test_migration.py --component router --verbose
```

## Continuous Integration

### GitHub Actions Integration

Add to `.github/workflows/validation.yml`:

```yaml
name: Migration Validation
on: [push, pull_request]

jobs:
  validate-migration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run migration validation
        run: python tests/test_migration_suite.py --output-dir ci_results
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: validation-results
          path: ci_results/
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: migration-validation
        name: BEM Migration Validation
        entry: python tests/test_migration_suite.py --fast
        language: system
        pass_filenames: false
```

## Development Workflow

### Before Committing Changes

```bash
# Quick validation (recommended for development)
python tests/test_migration_suite.py --fast

# Full validation (recommended before PR)
python tests/test_migration_suite.py
```

### After Major Changes

```bash
# Component-specific validation
python scripts/validation/test_migration.py --component <component>

# Configuration changes
python scripts/validation/validate_all_configs.py

# Performance impact assessment  
python tests/test_migration_suite.py --verbose | grep "performance"
```

### Adding New Components

1. Add unit tests to `test_component_migration.py`
2. Add migration validation to `test_migration.py`
3. Add configuration validation if new templates added
4. Run full test suite to ensure integration

## Contributing

### Adding New Tests

1. **Unit Tests**: Add to appropriate test file (infrastructure, component, config)
2. **Integration Tests**: Add to migration validation scripts
3. **Documentation**: Update this README with new test descriptions
4. **CI Integration**: Ensure new tests run in automated pipeline

### Test Standards

- All tests must be deterministic and reproducible
- Use pytest fixtures for setup/teardown
- Mock external dependencies appropriately
- Include both success and failure test cases
- Document test purpose and expected outcomes

### Review Checklist

- [ ] Tests pass locally with `python tests/test_migration_suite.py`
- [ ] New tests added for new functionality
- [ ] Test coverage maintained or improved
- [ ] Documentation updated
- [ ] CI pipeline passes