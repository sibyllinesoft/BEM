#!/usr/bin/env python3
"""
Systematic conversion script for BEM experiment configurations to use template inheritance.

This script analyzes existing experiment configurations, classifies them by type,
and converts them to use appropriate base templates while preserving all functionality.
"""

import os
import re
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
import logging
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConversionResult:
    """Results of converting a single experiment configuration."""
    filename: str
    original_lines: int
    converted_lines: int
    template_assigned: str
    reduction_percentage: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    removed_fields: List[str] = field(default_factory=list)
    preserved_fields: List[str] = field(default_factory=list)

class ExperimentConverter:
    """Main class for converting experiment configurations to template inheritance."""
    
    def __init__(self, experiments_dir: str = "experiments", templates_dir: str = "src/bem_core/config/templates"):
        self.experiments_dir = Path(experiments_dir)
        self.templates_dir = Path(templates_dir)
        self.backup_dir = Path("experiments_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Load all templates for comparison
        self.templates = self._load_templates()
        
        # Classification patterns
        self.classification_patterns = {
            "performance": {
                "filename_patterns": [r"^PT\d+", r"^v\d+", r"^f\d+", r"^A\d+", r"performance", r"optim", r"speed"],
                "content_keys": ["performance_mode", "batch_size", "learning_rate", "optimizer", "budget"],
                "template": "performance_variant.yaml"
            },
            "safety": {
                "filename_patterns": [r"^S\d+", r"safety", r"alignment", r"mission_c"],
                "content_keys": ["safety_checks", "alignment_enforcer", "mission_critical", "constitutional"],
                "template": "safety_experiment.yaml"
            },
            "multimodal": {
                "filename_patterns": [r"^MM\d+", r"^VC\d+", r"vision", r"multimodal", r"image"],
                "content_keys": ["vision", "multimodal", "vqa", "visual", "image", "coverage_analysis", "consistency"],
                "template": "multimodal_experiment.yaml"
            },
            "router": {
                "filename_patterns": [r"^AR\d+", r"router", r"adaptive"],
                "content_keys": ["router_config", "model_selection", "adaptive_routing", "routing"],
                "template": "performance_variant.yaml"  # Router experiments are performance-focused
            },
            "baseline": {
                "filename_patterns": [r"baseline", r"anchor", r"fleet"],
                "content_keys": [],  # Catch-all
                "template": "base_experiment.yaml"
            }
        }
        
        self.conversion_results = []

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all template configurations for comparison."""
        templates = {}
        
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory {self.templates_dir} not found")
            return templates
            
        for template_file in self.templates_dir.glob("*.yml"):
            try:
                with open(template_file, 'r') as f:
                    templates[template_file.name] = yaml.safe_load(f)
                logger.info(f"Loaded template: {template_file.name}")
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
                
        return templates

    def classify_experiment(self, filepath: Path, config: Dict[str, Any]) -> str:
        """Classify experiment type based on filename and content."""
        filename = filepath.stem.lower()
        
        # Check each classification pattern
        for exp_type, pattern_info in self.classification_patterns.items():
            # Check filename patterns
            for pattern in pattern_info["filename_patterns"]:
                if re.search(pattern, filename):
                    logger.debug(f"Classified {filepath.name} as {exp_type} by filename pattern: {pattern}")
                    return pattern_info["template"]
            
            # Check content keys
            if pattern_info["content_keys"]:
                for key in pattern_info["content_keys"]:
                    if self._has_nested_key(config, key):
                        logger.debug(f"Classified {filepath.name} as {exp_type} by content key: {key}")
                        return pattern_info["template"]
        
        # Default to baseline template
        logger.debug(f"Classified {filepath.name} as baseline (default)")
        return "base_experiment.yaml"

    def _has_nested_key(self, config: Dict[str, Any], key: str) -> bool:
        """Check if a key exists anywhere in the nested configuration."""
        def _search_dict(d, target_key):
            if isinstance(d, dict):
                if target_key in d:
                    return True
                for v in d.values():
                    if _search_dict(v, target_key):
                        return True
            elif isinstance(d, list):
                for item in d:
                    if _search_dict(item, target_key):
                        return True
            return False
        
        return _search_dict(config, key)

    def analyze_experiments(self) -> List[Tuple[Path, str]]:
        """Analyze all experiment files and return classification results."""
        experiment_files = []
        
        for yml_file in self.experiments_dir.glob("*.yml"):
            # Skip template files and already converted files
            if yml_file.parent.name == "templates" or "base_config" in yml_file.read_text():
                continue
                
            try:
                with open(yml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if config is None:
                    logger.warning(f"Empty configuration file: {yml_file}")
                    continue
                
                template = self.classify_experiment(yml_file, config)
                experiment_files.append((yml_file, template))
                logger.info(f"Analyzed {yml_file.name} -> {template}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {yml_file}: {e}")
        
        return experiment_files

    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries for comparison purposes."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge_dicts(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result

    def _find_redundant_fields(self, exp_config: Dict[str, Any], template_config: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Find fields that are redundant with the template and return cleaned config."""
        redundant_fields = []
        cleaned_config = {}
        
        def _compare_recursive(exp_dict: Dict[str, Any], template_dict: Dict[str, Any], path: str = ""):
            nonlocal redundant_fields, cleaned_config
            
            for key, exp_value in exp_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key in template_dict:
                    template_value = template_dict[key]
                    
                    if isinstance(exp_value, dict) and isinstance(template_value, dict):
                        # Recursively compare nested dictionaries
                        nested_cleaned = {}
                        _compare_recursive(exp_value, template_value, current_path)
                        
                        # Only include the key if there are non-redundant nested values
                        nested_result = self._get_nested_path(cleaned_config, current_path.split('.'))
                        if nested_result:
                            self._set_nested_path(cleaned_config, current_path.split('.'), nested_result)
                    elif exp_value == template_value:
                        # Identical values - mark as redundant
                        redundant_fields.append(current_path)
                    else:
                        # Different values - keep in cleaned config
                        self._set_nested_path(cleaned_config, current_path.split('.'), exp_value)
                else:
                    # Key not in template - keep in cleaned config
                    self._set_nested_path(cleaned_config, current_path.split('.'), exp_value)
        
        _compare_recursive(exp_config, template_config)
        return redundant_fields, cleaned_config

    def _get_nested_path(self, d: Dict[str, Any], path: List[str]) -> Any:
        """Get value from nested dictionary path."""
        result = d
        for key in path:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return None
        return result

    def _set_nested_path(self, d: Dict[str, Any], path: List[str], value: Any):
        """Set value in nested dictionary path."""
        current = d
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def convert_experiment(self, filepath: Path, template_name: str) -> ConversionResult:
        """Convert a single experiment file to use template inheritance."""
        logger.info(f"Converting {filepath.name} to use template {template_name}")
        
        # Load original configuration
        with open(filepath, 'r') as f:
            original_content = f.read()
            original_lines = len(original_content.splitlines())
        
        try:
            original_config = yaml.safe_load(original_content)
            if original_config is None:
                original_config = {}
        except Exception as e:
            return ConversionResult(
                filename=filepath.name,
                original_lines=original_lines,
                converted_lines=0,
                template_assigned=template_name,
                reduction_percentage=0,
                errors=[f"Failed to parse YAML: {e}"]
            )

        # Get template configuration for comparison
        template_config = self.templates.get(template_name, {})
        
        # Find redundant fields and create cleaned configuration
        redundant_fields, cleaned_config = self._find_redundant_fields(original_config, template_config)
        
        # Create new configuration with template inheritance
        new_config = {
            "base_config": template_name,
            **cleaned_config
        }
        
        # Write converted configuration
        try:
            with open(filepath, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False, indent=2)
            
            # Calculate new line count
            with open(filepath, 'r') as f:
                converted_content = f.read()
                converted_lines = len(converted_content.splitlines())
        
        except Exception as e:
            return ConversionResult(
                filename=filepath.name,
                original_lines=original_lines,
                converted_lines=0,
                template_assigned=template_name,
                reduction_percentage=0,
                errors=[f"Failed to write converted file: {e}"]
            )

        # Calculate reduction percentage
        reduction_percentage = ((original_lines - converted_lines) / original_lines * 100) if original_lines > 0 else 0
        
        # Gather preserved fields
        preserved_fields = list(self._get_all_keys(cleaned_config))
        
        return ConversionResult(
            filename=filepath.name,
            original_lines=original_lines,
            converted_lines=converted_lines,
            template_assigned=template_name,
            reduction_percentage=reduction_percentage,
            removed_fields=redundant_fields,
            preserved_fields=preserved_fields
        )

    def _get_all_keys(self, d: Dict[str, Any], prefix: str = "") -> Set[str]:
        """Get all keys from nested dictionary."""
        keys = set()
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)
            
            if isinstance(value, dict):
                keys.update(self._get_all_keys(value, full_key))
                
        return keys

    def create_backup(self):
        """Create backup of all experiment files."""
        logger.info(f"Creating backup at {self.backup_dir}")
        self.backup_dir.mkdir(exist_ok=True)
        
        for yml_file in self.experiments_dir.glob("*.yml"):
            if yml_file.parent.name != "templates":
                backup_file = self.backup_dir / yml_file.name
                shutil.copy2(yml_file, backup_file)
                
        logger.info(f"Backup created with {len(list(self.backup_dir.glob('*.yml')))} files")

    def validate_conversions(self) -> List[str]:
        """Validate all converted files for YAML syntax and required fields."""
        validation_errors = []
        
        for yml_file in self.experiments_dir.glob("*.yml"):
            if yml_file.parent.name == "templates":
                continue
                
            try:
                with open(yml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check for base_config field
                if "base_config" not in config:
                    validation_errors.append(f"{yml_file.name}: Missing base_config field")
                
                # Check if base_config template exists
                elif config["base_config"] not in self.templates:
                    validation_errors.append(f"{yml_file.name}: Template {config['base_config']} not found")
                    
            except Exception as e:
                validation_errors.append(f"{yml_file.name}: YAML parsing error: {e}")
        
        return validation_errors

    def generate_report(self) -> str:
        """Generate comprehensive conversion report."""
        total_experiments = len(self.conversion_results)
        successful_conversions = sum(1 for r in self.conversion_results if not r.errors)
        total_original_lines = sum(r.original_lines for r in self.conversion_results)
        total_converted_lines = sum(r.converted_lines for r in self.conversion_results)
        
        # Calculate overall reduction
        overall_reduction = ((total_original_lines - total_converted_lines) / total_original_lines * 100) if total_original_lines > 0 else 0
        
        report = []
        report.append("=" * 80)
        report.append("BEM EXPERIMENT CONFIGURATION CONVERSION REPORT")
        report.append("=" * 80)
        report.append(f"Conversion Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append(f"  Total experiments analyzed: {total_experiments}")
        report.append(f"  Successful conversions: {successful_conversions}")
        report.append(f"  Failed conversions: {total_experiments - successful_conversions}")
        report.append(f"  Overall line reduction: {overall_reduction:.1f}% ({total_original_lines} -> {total_converted_lines} lines)")
        report.append("")
        
        # Template usage statistics
        template_usage = {}
        for result in self.conversion_results:
            template_usage[result.template_assigned] = template_usage.get(result.template_assigned, 0) + 1
        
        report.append("TEMPLATE USAGE:")
        for template, count in sorted(template_usage.items()):
            report.append(f"  {template}: {count} experiments")
        report.append("")
        
        # Individual conversion results
        report.append("INDIVIDUAL CONVERSION RESULTS:")
        report.append("-" * 80)
        
        for result in sorted(self.conversion_results, key=lambda x: x.reduction_percentage, reverse=True):
            report.append(f"File: {result.filename}")
            report.append(f"  Template: {result.template_assigned}")
            report.append(f"  Lines: {result.original_lines} -> {result.converted_lines} ({result.reduction_percentage:.1f}% reduction)")
            
            if result.removed_fields:
                report.append(f"  Removed fields ({len(result.removed_fields)}): {', '.join(result.removed_fields[:5])}")
                if len(result.removed_fields) > 5:
                    report.append(f"    ... and {len(result.removed_fields) - 5} more")
            
            if result.preserved_fields:
                report.append(f"  Preserved fields ({len(result.preserved_fields)}): {', '.join(result.preserved_fields[:5])}")
                if len(result.preserved_fields) > 5:
                    report.append(f"    ... and {len(result.preserved_fields) - 5} more")
            
            if result.warnings:
                report.append(f"  Warnings: {'; '.join(result.warnings)}")
            
            if result.errors:
                report.append(f"  Errors: {'; '.join(result.errors)}")
            
            report.append("")
        
        return "\n".join(report)

    def run_conversion(self) -> str:
        """Run the complete conversion process."""
        logger.info("Starting BEM experiment configuration conversion")
        
        # Create backup
        self.create_backup()
        
        # Analyze experiments
        logger.info("Analyzing experiment configurations...")
        experiment_classifications = self.analyze_experiments()
        
        if not experiment_classifications:
            logger.warning("No experiments found to convert")
            return "No experiments found to convert"
        
        # Convert each experiment
        logger.info(f"Converting {len(experiment_classifications)} experiments...")
        for filepath, template_name in experiment_classifications:
            result = self.convert_experiment(filepath, template_name)
            self.conversion_results.append(result)
            
            if result.errors:
                logger.error(f"Failed to convert {result.filename}: {'; '.join(result.errors)}")
            else:
                logger.info(f"Converted {result.filename}: {result.reduction_percentage:.1f}% reduction")
        
        # Validate conversions
        logger.info("Validating converted configurations...")
        validation_errors = self.validate_conversions()
        
        if validation_errors:
            logger.warning("Validation errors found:")
            for error in validation_errors:
                logger.warning(f"  {error}")
        
        # Generate and save report
        report = self.generate_report()
        
        # Add validation errors to report if any
        if validation_errors:
            report += "\nVALIDATION ERRORS:\n"
            report += "-" * 80 + "\n"
            for error in validation_errors:
                report += f"{error}\n"
        
        report_file = f"conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Conversion complete. Report saved to {report_file}")
        logger.info(f"Backup saved to {self.backup_dir}")
        
        return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert BEM experiments to template inheritance")
    parser.add_argument("--experiments-dir", default="experiments", help="Experiments directory")
    parser.add_argument("--templates-dir", default="experiments/templates", help="Templates directory")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't convert")
    
    args = parser.parse_args()
    
    converter = ExperimentConverter(args.experiments_dir, args.templates_dir)
    
    if args.dry_run:
        logger.info("Running in dry-run mode (analysis only)")
        experiment_classifications = converter.analyze_experiments()
        
        print(f"\nAnalyzed {len(experiment_classifications)} experiments:")
        for filepath, template in experiment_classifications:
            print(f"  {filepath.name} -> {template}")
        
        # Show template usage
        template_usage = {}
        for _, template in experiment_classifications:
            template_usage[template] = template_usage.get(template, 0) + 1
        
        print(f"\nTemplate usage:")
        for template, count in sorted(template_usage.items()):
            print(f"  {template}: {count} experiments")
    
    else:
        report = converter.run_conversion()
        print("\n" + report)


if __name__ == "__main__":
    main()