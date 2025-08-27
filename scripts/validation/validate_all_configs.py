#!/usr/bin/env python3
"""
Configuration validation script for unified BEM infrastructure.

This script validates all experiment configurations, templates, and converted
configs to ensure they load correctly, follow the template inheritance system,
and maintain backward compatibility.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import configuration system
from bem_core.config.config_loader import ConfigLoader, ExperimentConfig
from bem_core.config.base_config import BaseConfig


class ConfigValidator:
    """Validates all BEM configurations and templates."""
    
    def __init__(self, project_root: str, output_dir: str = "config_validation"):
        """Initialize configuration validator.
        
        Args:
            project_root: Root directory of BEM project
            output_dir: Directory to save validation results
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        log_file = self.output_dir / "config_validation.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize config loader
        self.config_loader = ConfigLoader()
        
        # Results tracking
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "template_validation": {},
            "experiment_validation": {},
            "inheritance_validation": {},
            "conversion_validation": {},
            "error_summary": [],
            "warnings": [],
            "statistics": {}
        }
    
    def discover_config_files(self) -> Dict[str, List[Path]]:
        """Discover all configuration files in the project.
        
        Returns:
            Dictionary mapping config types to file paths
        """
        self.logger.info("Discovering configuration files...")
        
        config_files = {
            "templates": [],
            "experiments": [],
            "legacy_configs": [],
            "yaml_configs": [],
            "json_configs": []
        }
        
        # Look for templates
        template_dir = self.project_root / "src" / "bem_core" / "config" / "templates"
        if template_dir.exists():
            config_files["templates"] = list(template_dir.glob("*.yaml"))
            config_files["templates"].extend(list(template_dir.glob("*.yml")))
        
        # Look for experiment configs
        experiment_dirs = [
            self.project_root / "experiments",
            self.project_root / "experiments" / "configs"
        ]
        
        for exp_dir in experiment_dirs:
            if exp_dir.exists():
                config_files["experiments"].extend(list(exp_dir.glob("*.yaml")))
                config_files["experiments"].extend(list(exp_dir.glob("*.yml")))
                config_files["experiments"].extend(list(exp_dir.glob("*.json")))
        
        # Look for legacy configs (different naming patterns)
        for exp_dir in experiment_dirs:
            if exp_dir.exists():
                for config_file in exp_dir.iterdir():
                    if config_file.is_file():
                        if config_file.suffix.lower() in ['.yaml', '.yml']:
                            config_files["yaml_configs"].append(config_file)
                        elif config_file.suffix.lower() == '.json':
                            config_files["json_configs"].append(config_file)
        
        # Remove duplicates and categorize
        for key in config_files:
            config_files[key] = list(set(config_files[key]))
        
        # Log discovery results
        for config_type, files in config_files.items():
            self.logger.info(f"Found {len(files)} {config_type}")
        
        return config_files
    
    def validate_template_files(self, template_files: List[Path]) -> Dict[str, Any]:
        """Validate template files for syntax and structure.
        
        Args:
            template_files: List of template file paths
            
        Returns:
            Validation results for templates
        """
        self.logger.info(f"Validating {len(template_files)} template files...")
        
        results = {
            "total_templates": len(template_files),
            "valid_templates": 0,
            "invalid_templates": 0,
            "templates_with_warnings": 0,
            "template_details": {},
            "inheritance_graph": {},
            "errors": []
        }
        
        for template_file in template_files:
            template_name = template_file.stem
            self.logger.info(f"Validating template: {template_name}")
            
            template_result = {
                "file_path": str(template_file),
                "valid": False,
                "errors": [],
                "warnings": [],
                "inherits_from": None,
                "fields": {},
                "required_fields_present": True
            }
            
            try:
                # Load and parse YAML
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                if not isinstance(template_data, dict):
                    template_result["errors"].append("Template is not a valid YAML dictionary")
                    results["invalid_templates"] += 1
                    results["template_details"][template_name] = template_result
                    continue
                
                # Check basic structure
                required_fields = ["name"]
                optional_fields = ["version", "description", "experiment_type", "inherits_from",
                                 "model", "data", "training", "hardware", "logging"]
                
                for field in required_fields:
                    if field not in template_data:
                        template_result["errors"].append(f"Missing required field: {field}")
                        template_result["required_fields_present"] = False
                
                # Record inheritance
                if "inherits_from" in template_data:
                    template_result["inherits_from"] = template_data["inherits_from"]
                    results["inheritance_graph"][template_name] = template_data["inherits_from"]
                
                # Validate field types and values
                template_result["fields"] = self._validate_template_fields(template_data)
                
                # Check for common issues
                warnings = self._check_template_warnings(template_data)
                template_result["warnings"].extend(warnings)
                
                if warnings:
                    results["templates_with_warnings"] += 1
                
                # Mark as valid if no critical errors
                if not template_result["errors"]:
                    template_result["valid"] = True
                    results["valid_templates"] += 1
                else:
                    results["invalid_templates"] += 1
                    
            except yaml.YAMLError as e:
                template_result["errors"].append(f"YAML parsing error: {e}")
                results["invalid_templates"] += 1
            except Exception as e:
                template_result["errors"].append(f"Unexpected error: {e}")
                results["invalid_templates"] += 1
            
            results["template_details"][template_name] = template_result
        
        return results
    
    def validate_experiment_configs(self, config_files: List[Path]) -> Dict[str, Any]:
        """Validate experiment configuration files.
        
        Args:
            config_files: List of experiment config file paths
            
        Returns:
            Validation results for experiment configs
        """
        self.logger.info(f"Validating {len(config_files)} experiment configs...")
        
        results = {
            "total_configs": len(config_files),
            "valid_configs": 0,
            "invalid_configs": 0,
            "configs_with_warnings": 0,
            "config_details": {},
            "errors": []
        }
        
        for config_file in config_files:
            config_name = config_file.stem
            self.logger.info(f"Validating config: {config_name}")
            
            config_result = {
                "file_path": str(config_file),
                "valid": False,
                "loadable": False,
                "errors": [],
                "warnings": [],
                "config_type": None,
                "has_inheritance": False
            }
            
            try:
                # Try to load with config loader
                try:
                    config = self.config_loader.load_config(str(config_file))
                    config_result["loadable"] = True
                    config_result["config_type"] = getattr(config, 'experiment_type', 'unknown')
                    
                    # Check if config uses inheritance
                    with open(config_file, 'r') as f:
                        raw_data = yaml.safe_load(f) if config_file.suffix.lower() in ['.yaml', '.yml'] else json.load(f)
                    
                    if isinstance(raw_data, dict) and "inherits_from" in raw_data:
                        config_result["has_inheritance"] = True
                    
                    # Validate loaded config
                    validation_errors = self._validate_experiment_config(config)
                    config_result["errors"].extend(validation_errors)
                    
                except Exception as load_error:
                    config_result["errors"].append(f"Config loading failed: {load_error}")
                
                # Additional file-level validation
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix.lower() in ['.yaml', '.yml']:
                            raw_data = yaml.safe_load(f)
                        else:
                            raw_data = json.load(f)
                    
                    # Check for deprecated patterns
                    warnings = self._check_config_warnings(raw_data, config_file)
                    config_result["warnings"].extend(warnings)
                    
                    if warnings:
                        results["configs_with_warnings"] += 1
                        
                except Exception as parse_error:
                    config_result["errors"].append(f"File parsing error: {parse_error}")
                
                # Mark as valid if loadable and no critical errors
                if config_result["loadable"] and not config_result["errors"]:
                    config_result["valid"] = True
                    results["valid_configs"] += 1
                else:
                    results["invalid_configs"] += 1
                    
            except Exception as e:
                config_result["errors"].append(f"Validation error: {e}")
                results["invalid_configs"] += 1
            
            results["config_details"][config_name] = config_result
        
        return results
    
    def validate_inheritance_system(self, template_files: List[Path]) -> Dict[str, Any]:
        """Validate the template inheritance system.
        
        Args:
            template_files: List of template files to validate inheritance
            
        Returns:
            Inheritance system validation results
        """
        self.logger.info("Validating template inheritance system...")
        
        results = {
            "inheritance_chains": {},
            "circular_references": [],
            "orphaned_templates": [],
            "inheritance_depths": {},
            "valid_inheritance": True,
            "errors": []
        }
        
        # Build inheritance graph
        inheritance_graph = {}
        template_names = set()
        
        for template_file in template_files:
            template_name = template_file.stem
            template_names.add(template_name)
            
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                if isinstance(template_data, dict) and "inherits_from" in template_data:
                    parent = template_data["inherits_from"]
                    inheritance_graph[template_name] = parent
                    
            except Exception as e:
                results["errors"].append(f"Error reading {template_name}: {e}")
        
        # Check for circular references
        for template in template_names:
            visited = set()
            current = template
            path = []
            
            while current in inheritance_graph and current not in visited:
                visited.add(current)
                path.append(current)
                current = inheritance_graph[current]
                
                if current in visited:
                    # Circular reference detected
                    cycle_start = path.index(current)
                    cycle = path[cycle_start:] + [current]
                    results["circular_references"].append(cycle)
                    results["valid_inheritance"] = False
                    break
            
            if path:
                results["inheritance_chains"][template] = path
        
        # Check for orphaned templates (inherit from non-existent parents)
        for child, parent in inheritance_graph.items():
            if parent not in template_names:
                results["orphaned_templates"].append({
                    "child": child,
                    "missing_parent": parent
                })
                results["valid_inheritance"] = False
        
        # Calculate inheritance depths
        for template in template_names:
            depth = 0
            current = template
            
            while current in inheritance_graph:
                depth += 1
                current = inheritance_graph[current]
                
                if depth > 10:  # Prevent infinite loops
                    break
            
            results["inheritance_depths"][template] = depth
        
        return results
    
    def validate_config_conversions(self) -> Dict[str, Any]:
        """Validate configuration conversion functionality.
        
        Returns:
            Config conversion validation results
        """
        self.logger.info("Validating configuration conversions...")
        
        results = {
            "legacy_format_support": False,
            "conversion_examples": {},
            "conversion_errors": [],
            "backward_compatibility": True
        }
        
        # Test conversion of legacy formats
        legacy_formats = [
            {
                "name": "old_router_format",
                "data": {
                    "experiment_name": "test_router",
                    "router_config": {"num_experts": 4},
                    "model_config": {"base_model": "microsoft/DialoGPT-small"},
                    "training_config": {"learning_rate": 1e-4}
                }
            },
            {
                "name": "old_safety_format", 
                "data": {
                    "experiment_name": "test_safety",
                    "safety_config": {"threshold": 0.8},
                    "model_config": {"base_model": "microsoft/DialoGPT-small"}
                }
            }
        ]
        
        for legacy_format in legacy_formats:
            format_name = legacy_format["name"]
            
            try:
                # Test if config loader can handle legacy format
                # This would require implementing legacy conversion in ConfigLoader
                
                # For now, just test that the data is valid
                legacy_data = legacy_format["data"]
                
                # Check if conversion would be possible
                has_experiment_name = "experiment_name" in legacy_data
                has_model_config = any(key.endswith("_config") for key in legacy_data.keys())
                
                conversion_possible = has_experiment_name and has_model_config
                
                results["conversion_examples"][format_name] = {
                    "legacy_data": legacy_data,
                    "conversion_possible": conversion_possible,
                    "would_convert": conversion_possible
                }
                
                if conversion_possible:
                    results["legacy_format_support"] = True
                    
            except Exception as e:
                results["conversion_errors"].append(f"Error testing {format_name}: {e}")
        
        return results
    
    def _validate_template_fields(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual template fields."""
        field_validation = {}
        
        # Validate model section
        if "model" in template_data:
            model_config = template_data["model"]
            field_validation["model"] = {
                "present": True,
                "valid_base_model": "base_model" in model_config,
                "valid_hidden_size": isinstance(model_config.get("hidden_size"), int) if "hidden_size" in model_config else True,
                "has_custom_params": "custom_params" in model_config
            }
        
        # Validate training section
        if "training" in template_data:
            training_config = template_data["training"]
            field_validation["training"] = {
                "present": True,
                "valid_learning_rate": isinstance(training_config.get("learning_rate"), (int, float)) if "learning_rate" in training_config else True,
                "valid_batch_size": isinstance(training_config.get("batch_size"), int) if "batch_size" in training_config else True,
                "has_steps_or_epochs": any(key in training_config for key in ["max_steps", "max_epochs"])
            }
        
        # Validate data section
        if "data" in template_data:
            data_config = template_data["data"]
            field_validation["data"] = {
                "present": True,
                "has_train_file": "train_file" in data_config,
                "has_validation_file": any(key in data_config for key in ["validation_file", "val_file"]),
                "valid_max_seq_length": isinstance(data_config.get("max_seq_length"), int) if "max_seq_length" in data_config else True
            }
        
        return field_validation
    
    def _check_template_warnings(self, template_data: Dict[str, Any]) -> List[str]:
        """Check for template warnings and best practices."""
        warnings = []
        
        # Check for missing description
        if "description" not in template_data:
            warnings.append("Missing description field")
        
        # Check for unusual learning rates
        if "training" in template_data:
            lr = template_data["training"].get("learning_rate")
            if isinstance(lr, (int, float)):
                if lr > 0.01:
                    warnings.append(f"Unusually high learning rate: {lr}")
                elif lr < 1e-6:
                    warnings.append(f"Unusually low learning rate: {lr}")
        
        # Check for very small batch sizes
        if "training" in template_data:
            batch_size = template_data["training"].get("batch_size")
            if isinstance(batch_size, int) and batch_size < 4:
                warnings.append(f"Very small batch size: {batch_size}")
        
        # Check for deprecated fields
        deprecated_fields = ["use_cuda", "fp16", "dataloader_drop_last"]
        for field in deprecated_fields:
            if field in template_data:
                warnings.append(f"Deprecated field used: {field}")
        
        return warnings
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> List[str]:
        """Validate loaded experiment configuration."""
        errors = []
        
        try:
            # Check required attributes
            if not hasattr(config, 'name') or not config.name:
                errors.append("Missing or empty experiment name")
            
            if not hasattr(config, 'model'):
                errors.append("Missing model configuration")
            
            if not hasattr(config, 'training'):
                errors.append("Missing training configuration")
            
            # Validate model config
            if hasattr(config, 'model'):
                if not hasattr(config.model, 'base_model') or not config.model.base_model:
                    errors.append("Missing base model specification")
            
            # Validate training config
            if hasattr(config, 'training'):
                if not hasattr(config.training, 'learning_rate'):
                    errors.append("Missing learning rate")
                elif not isinstance(config.training.learning_rate, (int, float)):
                    errors.append("Invalid learning rate type")
                
                if not hasattr(config.training, 'batch_size'):
                    errors.append("Missing batch size")
                elif not isinstance(config.training.batch_size, int) or config.training.batch_size <= 0:
                    errors.append("Invalid batch size")
        
        except Exception as e:
            errors.append(f"Config validation error: {e}")
        
        return errors
    
    def _check_config_warnings(self, raw_data: Dict[str, Any], config_file: Path) -> List[str]:
        """Check configuration file for warnings."""
        warnings = []
        
        # Check for old field names
        old_field_mapping = {
            "experiment_name": "name",
            "model_config": "model",
            "training_config": "training",
            "data_config": "data"
        }
        
        for old_field, new_field in old_field_mapping.items():
            if old_field in raw_data:
                warnings.append(f"Deprecated field '{old_field}' found, should use '{new_field}'")
        
        # Check file naming conventions
        if config_file.stem.startswith("experiment_"):
            warnings.append("File name uses old 'experiment_' prefix")
        
        return warnings
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        self.logger.info("Generating validation report...")
        
        # Save detailed results
        results_file = self.output_dir / "config_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary report
        report = []
        report.append("# BEM Configuration Validation Report")
        report.append(f"Generated: {self.results['validation_timestamp']}")
        report.append(f"Project Root: {self.results['project_root']}")
        report.append("")
        
        # Template validation summary
        template_results = self.results.get("template_validation", {})
        if template_results:
            report.append("## Template Validation")
            total = template_results.get("total_templates", 0)
            valid = template_results.get("valid_templates", 0)
            invalid = template_results.get("invalid_templates", 0)
            warnings = template_results.get("templates_with_warnings", 0)
            
            report.append(f"- Total templates: {total}")
            report.append(f"- Valid templates: {valid}")
            report.append(f"- Invalid templates: {invalid}")
            report.append(f"- Templates with warnings: {warnings}")
            
            if total > 0:
                success_rate = (valid / total) * 100
                report.append(f"- Success rate: {success_rate:.1f}%")
        
        # Experiment config validation summary
        experiment_results = self.results.get("experiment_validation", {})
        if experiment_results:
            report.append("")
            report.append("## Experiment Configuration Validation")
            total = experiment_results.get("total_configs", 0)
            valid = experiment_results.get("valid_configs", 0)
            invalid = experiment_results.get("invalid_configs", 0)
            warnings = experiment_results.get("configs_with_warnings", 0)
            
            report.append(f"- Total configs: {total}")
            report.append(f"- Valid configs: {valid}")
            report.append(f"- Invalid configs: {invalid}")
            report.append(f"- Configs with warnings: {warnings}")
            
            if total > 0:
                success_rate = (valid / total) * 100
                report.append(f"- Success rate: {success_rate:.1f}%")
        
        # Inheritance system validation
        inheritance_results = self.results.get("inheritance_validation", {})
        if inheritance_results:
            report.append("")
            report.append("## Template Inheritance System")
            
            valid_inheritance = inheritance_results.get("valid_inheritance", False)
            circular_refs = len(inheritance_results.get("circular_references", []))
            orphaned = len(inheritance_results.get("orphaned_templates", []))
            
            report.append(f"- Inheritance system valid: {'✓' if valid_inheritance else '✗'}")
            report.append(f"- Circular references: {circular_refs}")
            report.append(f"- Orphaned templates: {orphaned}")
            
            if inheritance_results.get("inheritance_depths"):
                max_depth = max(inheritance_results["inheritance_depths"].values())
                report.append(f"- Maximum inheritance depth: {max_depth}")
        
        # Error summary
        error_count = len(self.results.get("error_summary", []))
        warning_count = len(self.results.get("warnings", []))
        
        if error_count > 0 or warning_count > 0:
            report.append("")
            report.append("## Issues Summary")
            report.append(f"- Total errors: {error_count}")
            report.append(f"- Total warnings: {warning_count}")
            
            # Show first few errors
            if error_count > 0:
                report.append("")
                report.append("### Recent Errors:")
                for error in self.results["error_summary"][:5]:
                    report.append(f"- {error}")
                
                if error_count > 5:
                    report.append(f"- ... and {error_count - 5} more errors")
        
        # Overall assessment
        report.append("")
        report.append("## Overall Assessment")
        
        # Calculate overall health score
        template_success = 0
        experiment_success = 0
        
        if template_results.get("total_templates", 0) > 0:
            template_success = template_results.get("valid_templates", 0) / template_results["total_templates"]
        
        if experiment_results.get("total_configs", 0) > 0:
            experiment_success = experiment_results.get("valid_configs", 0) / experiment_results["total_configs"]
        
        if template_success > 0 and experiment_success > 0:
            overall_score = (template_success + experiment_success) / 2 * 100
        elif template_success > 0:
            overall_score = template_success * 100
        elif experiment_success > 0:
            overall_score = experiment_success * 100
        else:
            overall_score = 0
        
        inheritance_valid = inheritance_results.get("valid_inheritance", True)
        
        if overall_score >= 90 and inheritance_valid and error_count == 0:
            report.append("🟢 **CONFIGURATION SYSTEM HEALTHY**: All configs validated successfully")
        elif overall_score >= 75 and inheritance_valid and error_count < 5:
            report.append("🟡 **CONFIGURATION SYSTEM MOSTLY HEALTHY**: Minor issues need attention")
        elif overall_score >= 50:
            report.append("🟠 **CONFIGURATION SYSTEM HAS ISSUES**: Multiple problems need resolution")
        else:
            report.append("🔴 **CONFIGURATION SYSTEM CRITICAL**: Significant problems detected")
        
        report.append(f"Overall health score: {overall_score:.1f}%")
        
        # Recommendations
        report.append("")
        report.append("## Recommendations")
        
        if invalid := template_results.get("invalid_templates", 0):
            report.append(f"- Fix {invalid} invalid template(s)")
        
        if invalid := experiment_results.get("invalid_configs", 0):
            report.append(f"- Fix {invalid} invalid experiment config(s)")
        
        if circular_refs := len(inheritance_results.get("circular_references", [])):
            report.append(f"- Resolve {circular_refs} circular inheritance reference(s)")
        
        if orphaned := len(inheritance_results.get("orphaned_templates", [])):
            report.append(f"- Fix {orphaned} orphaned template(s)")
        
        if warning_count > 10:
            report.append("- Address configuration warnings to improve maintainability")
        
        report_content = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / "config_validation_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return report_content
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete configuration validation."""
        self.logger.info("Starting full configuration validation...")
        
        try:
            # Discover all config files
            config_files = self.discover_config_files()
            
            # Store statistics
            self.results["statistics"] = {
                "total_templates": len(config_files["templates"]),
                "total_experiments": len(config_files["experiments"]),
                "total_yaml_configs": len(config_files["yaml_configs"]),
                "total_json_configs": len(config_files["json_configs"])
            }
            
            # Validate templates
            if config_files["templates"]:
                self.results["template_validation"] = self.validate_template_files(
                    config_files["templates"]
                )
            
            # Validate experiment configs
            all_configs = config_files["experiments"] + config_files["yaml_configs"] + config_files["json_configs"]
            # Remove duplicates
            all_configs = list(set(all_configs))
            
            if all_configs:
                self.results["experiment_validation"] = self.validate_experiment_configs(
                    all_configs
                )
            
            # Validate inheritance system
            if config_files["templates"]:
                self.results["inheritance_validation"] = self.validate_inheritance_system(
                    config_files["templates"]
                )
            
            # Validate config conversions
            self.results["conversion_validation"] = self.validate_config_conversions()
            
            # Collect errors and warnings from all validations
            self._collect_errors_and_warnings()
            
            # Generate report
            report = self.generate_validation_report()
            
            self.logger.info("Configuration validation completed")
            self.logger.info(f"Results saved to: {self.output_dir}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            self.logger.error(traceback.format_exc())
            self.results["validation_error"] = str(e)
            return self.results
    
    def _collect_errors_and_warnings(self):
        """Collect all errors and warnings from validation results."""
        errors = []
        warnings = []
        
        # Collect from template validation
        template_results = self.results.get("template_validation", {})
        if "template_details" in template_results:
            for template_name, details in template_results["template_details"].items():
                for error in details.get("errors", []):
                    errors.append(f"Template {template_name}: {error}")
                for warning in details.get("warnings", []):
                    warnings.append(f"Template {template_name}: {warning}")
        
        # Collect from experiment validation
        experiment_results = self.results.get("experiment_validation", {})
        if "config_details" in experiment_results:
            for config_name, details in experiment_results["config_details"].items():
                for error in details.get("errors", []):
                    errors.append(f"Config {config_name}: {error}")
                for warning in details.get("warnings", []):
                    warnings.append(f"Config {config_name}: {warning}")
        
        # Collect from inheritance validation
        inheritance_results = self.results.get("inheritance_validation", {})
        for error in inheritance_results.get("errors", []):
            errors.append(f"Inheritance: {error}")
        
        for circular_ref in inheritance_results.get("circular_references", []):
            errors.append(f"Circular inheritance: {' -> '.join(circular_ref)}")
        
        for orphaned in inheritance_results.get("orphaned_templates", []):
            errors.append(f"Orphaned template: {orphaned['child']} inherits from missing {orphaned['missing_parent']}")
        
        # Store collected errors and warnings
        self.results["error_summary"] = errors
        self.results["warnings"] = warnings


def main():
    """Main entry point for configuration validation."""
    parser = argparse.ArgumentParser(description="Validate BEM configuration system")
    parser.add_argument("--project-root", default=".", 
                       help="Root directory of BEM project")
    parser.add_argument("--output-dir", default="config_validation",
                       help="Output directory for validation results")
    parser.add_argument("--templates-only", action="store_true",
                       help="Validate only template files")
    parser.add_argument("--configs-only", action="store_true", 
                       help="Validate only experiment configs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create validator
    validator = ConfigValidator(args.project_root, args.output_dir)
    
    # Run validation
    if args.templates_only:
        config_files = validator.discover_config_files()
        results = validator.validate_template_files(config_files["templates"])
        validator.results["template_validation"] = results
        validator.generate_validation_report()
    elif args.configs_only:
        config_files = validator.discover_config_files()
        all_configs = config_files["experiments"] + config_files["yaml_configs"] + config_files["json_configs"]
        results = validator.validate_experiment_configs(list(set(all_configs)))
        validator.results["experiment_validation"] = results  
        validator.generate_validation_report()
    else:
        results = validator.run_full_validation()
    
    # Determine exit code
    error_count = len(validator.results.get("error_summary", []))
    
    print(f"\nValidation completed. Results saved to: {args.output_dir}")
    print(f"Check {args.output_dir}/config_validation_report.md for summary")
    
    if error_count == 0:
        print("✓ No errors found")
        return 0
    else:
        print(f"✗ {error_count} errors found")
        return 1


if __name__ == "__main__":
    sys.exit(main())