# Experiment Configuration Conversion Script

This script systematically converts BEM experiment configurations to use template inheritance, reducing redundancy and improving maintainability.

## Features

### 🔍 **Intelligent Classification**
- **Performance Experiments**: Files matching `PT*`, `v*`, containing performance-related keys
- **Safety Experiments**: Files matching `S*`, `safety`, `alignment`, `mission_c` patterns
- **Multimodal Experiments**: Files matching `MM*`, `VC*`, `vision`, containing vision/multimodal keys
- **Router Experiments**: Files matching `AR*`, `router`, containing routing configuration
- **Baseline Experiments**: Default template for standard configurations

### 🛠 **Smart Conversion**
- Analyzes existing configurations and template content
- Removes redundant fields present in base templates
- Preserves experiment-specific parameters
- Adds appropriate `base_config` inheritance
- Maintains all functionality while reducing file size

### 🔒 **Safety & Validation**
- Creates timestamped backups before conversion
- Validates YAML syntax after conversion
- Checks for required fields and template existence
- Comprehensive error handling and logging

### 📊 **Detailed Reporting**
- Before/after line count comparisons
- Configuration reduction percentages
- Template assignment statistics
- Field removal and preservation tracking
- Validation warnings and errors

## Usage

### Dry Run (Analysis Only)
```bash
# Analyze experiments without making changes
python scripts/convert_experiments_to_templates.py --dry-run

# Specify custom directories
python scripts/convert_experiments_to_templates.py --dry-run \
    --experiments-dir custom_experiments \
    --templates-dir custom_templates
```

### Full Conversion
```bash
# Convert all experiments with automatic backup
python scripts/convert_experiments_to_templates.py

# With custom directories
python scripts/convert_experiments_to_templates.py \
    --experiments-dir experiments \
    --templates-dir experiments/templates
```

## Output Files

### Generated Files
- **Backup Directory**: `experiments_backup_YYYYMMDD_HHMMSS/` - Original files
- **Conversion Report**: `conversion_report_YYYYMMDD_HHMMSS.txt` - Detailed results
- **Conversion Log**: `conversion_log.txt` - Processing log with timestamps

### Converted Configurations
Each experiment file gets converted to:
```yaml
base_config: appropriate_template.yml
# Only experiment-specific overrides remain
experiment_specific_param: value
nested:
  override: different_value
```

## Classification Logic

The script uses both filename patterns and content analysis:

| Type | Filename Patterns | Content Keys | Template |
|------|------------------|--------------|----------|
| Performance | `PT*`, `v*`, `performance`, `optim` | `performance_mode`, `batch_size`, `learning_rate` | `performance_base.yml` |
| Safety | `S*`, `safety`, `alignment`, `mission_c` | `safety_checks`, `alignment_enforcer` | `safety_base.yml` |
| Multimodal | `MM*`, `VC*`, `vision`, `multimodal` | `vision_model`, `image_processor` | `multimodal_base.yml` |
| Router | `AR*`, `router`, `adaptive` | `router_config`, `model_selection` | `router_base.yml` |
| Baseline | `baseline`, `default`, `standard` | (catch-all) | `base.yml` |

## Example Output

```
SUMMARY STATISTICS:
  Total experiments analyzed: 23
  Successful conversions: 22
  Failed conversions: 1
  Overall line reduction: 67.3% (1,245 -> 407 lines)

TEMPLATE USAGE:
  performance_base.yml: 8 experiments
  safety_base.yml: 5 experiments
  multimodal_base.yml: 4 experiments
  router_base.yml: 3 experiments
  base.yml: 2 experiments

INDIVIDUAL CONVERSION RESULTS:
File: PT001_optimization.yml
  Template: performance_base.yml
  Lines: 89 -> 23 (74.2% reduction)
  Removed fields (12): model.architecture, training.optimizer, training.batch_size...
  Preserved fields (5): experiment.learning_rate_schedule, experiment.specific_params...
```

## Safety Features

### Backup System
- Automatic timestamped backups before any changes
- Original files preserved in separate directory
- Easy rollback if needed

### Validation Checks
- YAML syntax validation after conversion
- Required field verification (`base_config` presence)
- Template existence verification
- Comprehensive error reporting

### Logging
- Detailed processing log with timestamps
- Error and warning tracking
- Progress monitoring for large conversions

## Recovery

If something goes wrong:
```bash
# Restore from backup
cp experiments_backup_YYYYMMDD_HHMMSS/* experiments/

# Check validation errors in report
grep "VALIDATION ERRORS" conversion_report_*.txt
```

## Requirements

- Python 3.7+
- PyYAML (`pip install PyYAML`)
- Existing template files in `experiments/templates/`

## Integration

This script is designed to work with the BEM template inheritance system. Ensure your templates are properly configured before running the conversion.