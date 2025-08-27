#!/usr/bin/env python3
"""
Comprehensive migration validation test suite.

This script runs all validation tests for the unified BEM infrastructure migration,
providing a single entry point for validating the entire system.
"""

import os
import sys
import pytest
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('migration_test_suite.log')
        ]
    )
    
    return logging.getLogger(__name__)


def run_pytest_tests(test_files: list, output_dir: str, logger):
    """Run pytest on specified test files."""
    logger.info("Running pytest validation tests...")
    
    pytest_args = [
        "--verbose",
        "--tb=short",
        f"--html={output_dir}/pytest_report.html",
        "--self-contained-html",
        f"--junit-xml={output_dir}/junit_results.xml",
        "--cov=src",
        f"--cov-report=html:{output_dir}/coverage_html",
        f"--cov-report=xml:{output_dir}/coverage.xml"
    ] + test_files
    
    try:
        result = pytest.main(pytest_args)
        
        if result == 0:
            logger.info("✓ All pytest tests passed")
            return True
        else:
            logger.error("✗ Some pytest tests failed")
            return False
            
    except Exception as e:
        logger.error(f"Error running pytest: {e}")
        return False


def run_migration_validation(output_dir: str, logger):
    """Run the migration validation script."""
    logger.info("Running migration validation...")
    
    script_path = Path(__file__).parent.parent / "scripts" / "validation" / "test_migration.py"
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path),
            "--output-dir", f"{output_dir}/migration_validation",
            "--component", "all"
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            logger.info("✓ Migration validation passed")
            logger.info(result.stdout)
            return True
        else:
            logger.error("✗ Migration validation failed")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Migration validation timed out")
        return False
    except Exception as e:
        logger.error(f"Error running migration validation: {e}")
        return False


def run_config_validation(output_dir: str, logger):
    """Run the configuration validation script."""
    logger.info("Running configuration validation...")
    
    script_path = Path(__file__).parent.parent / "scripts" / "validation" / "validate_all_configs.py"
    project_root = Path(__file__).parent.parent
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path),
            "--project-root", str(project_root),
            "--output-dir", f"{output_dir}/config_validation"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            logger.info("✓ Configuration validation passed")
            logger.info(result.stdout)
            return True
        else:
            logger.error("✗ Configuration validation failed")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Configuration validation timed out")
        return False
    except Exception as e:
        logger.error(f"Error running configuration validation: {e}")
        return False


def run_import_tests(logger):
    """Run basic import tests to ensure all modules load correctly."""
    logger.info("Running import validation tests...")
    
    imports_to_test = [
        # Core infrastructure
        ("src.bem_core.training.base_trainer", ["BaseTrainer", "TrainingConfig"]),
        ("src.bem_core.evaluation.base_evaluator", ["BaseEvaluator", "EvaluationResult"]),
        ("src.bem_core.config.config_loader", ["ConfigLoader", "ExperimentConfig"]),
        
        # Unified trainers
        ("src.bem2.router.unified_trainer", ["RouterTrainer", "RouterEvaluator"]),
        ("src.bem2.safety.unified_trainer", ["SafetyTrainer", "SafetyEvaluator"]),
        ("src.bem2.multimodal.unified_trainer", ["MultimodalTrainer", "MultimodalEvaluator"]),
        
        # Utilities
        ("src.bem_core.utils.logging_utils", ["setup_logger"]),
        ("src.bem_core.utils.checkpoint_utils", ["save_checkpoint", "load_checkpoint"])
    ]
    
    failed_imports = []
    
    for module_name, expected_classes in imports_to_test:
        try:
            module = __import__(module_name, fromlist=expected_classes)
            
            for class_name in expected_classes:
                if not hasattr(module, class_name):
                    failed_imports.append(f"{module_name}.{class_name}")
                    logger.error(f"✗ Missing class: {module_name}.{class_name}")
                else:
                    logger.debug(f"✓ Found: {module_name}.{class_name}")
                    
        except ImportError as e:
            failed_imports.append(f"{module_name}: {e}")
            logger.error(f"✗ Import failed: {module_name} - {e}")
        except Exception as e:
            failed_imports.append(f"{module_name}: {e}")
            logger.error(f"✗ Unexpected error importing {module_name}: {e}")
    
    if not failed_imports:
        logger.info("✓ All import tests passed")
        return True
    else:
        logger.error(f"✗ {len(failed_imports)} import tests failed")
        return False


def generate_summary_report(results: dict, output_dir: str, logger):
    """Generate a summary report of all validation results."""
    logger.info("Generating summary report...")
    
    report = []
    report.append("# BEM Migration Validation Summary")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    
    # Test results summary
    report.append("## Test Results Summary")
    report.append("")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        report.append(f"- {test_name}: {status}")
    
    report.append("")
    report.append(f"**Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)**")
    report.append("")
    
    # Overall assessment
    if passed_tests == total_tests:
        report.append("## 🟢 MIGRATION VALIDATION SUCCESSFUL")
        report.append("")
        report.append("All validation tests passed. The unified BEM infrastructure migration is complete and ready for production use.")
        
    elif passed_tests >= total_tests * 0.8:
        report.append("## 🟡 MIGRATION VALIDATION MOSTLY SUCCESSFUL")
        report.append("")
        report.append("Most validation tests passed, but some issues need attention before production deployment.")
        
    else:
        report.append("## 🔴 MIGRATION VALIDATION FAILED")
        report.append("")
        report.append("Significant issues detected. Migration needs substantial work before production deployment.")
    
    report.append("")
    report.append("## Next Steps")
    report.append("")
    
    failed_tests = [name for name, passed in results.items() if not passed]
    if failed_tests:
        report.append("### Issues to Address:")
        for test_name in failed_tests:
            report.append(f"- Fix issues identified in {test_name}")
        report.append("")
        
    report.append("### Detailed Reports:")
    report.append("- Check pytest_report.html for detailed test results")
    report.append("- Review migration_validation/ for migration-specific results")
    report.append("- Check config_validation/ for configuration system validation")
    report.append("- See migration_test_suite.log for detailed logs")
    
    # Write report
    report_content = "\n".join(report)
    report_file = Path(output_dir) / "migration_validation_summary.md"
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Summary report written to: {report_file}")
    return report_content


def main():
    """Main entry point for migration validation test suite."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive BEM migration validation tests"
    )
    parser.add_argument("--output-dir", default="migration_test_results",
                       help="Output directory for test results")
    parser.add_argument("--fast", action="store_true",
                       help="Run only fast tests (skip comprehensive validation)")
    parser.add_argument("--unit-tests-only", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--integration-tests-only", action="store_true",
                       help="Run only integration tests") 
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first test failure")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting BEM migration validation test suite")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Results tracking
    results = {}
    
    # 1. Basic import tests (always run first)
    logger.info("=" * 60)
    logger.info("PHASE 1: Import Validation")
    logger.info("=" * 60)
    
    results["import_tests"] = run_import_tests(logger)
    
    if args.fail_fast and not results["import_tests"]:
        logger.error("Import tests failed, stopping early")
        return 1
    
    # 2. Unit tests
    if not args.integration_tests_only:
        logger.info("=" * 60)
        logger.info("PHASE 2: Unit Tests")
        logger.info("=" * 60)
        
        test_files = [
            "tests/test_unified_infrastructure.py",
            "tests/test_component_migration.py", 
            "tests/test_configuration_system.py"
        ]
        
        # Filter to existing files
        existing_test_files = []
        for test_file in test_files:
            if os.path.exists(test_file):
                existing_test_files.append(test_file)
            else:
                logger.warning(f"Test file not found: {test_file}")
        
        if existing_test_files:
            results["unit_tests"] = run_pytest_tests(
                existing_test_files, str(output_dir), logger
            )
        else:
            logger.error("No test files found for unit tests")
            results["unit_tests"] = False
        
        if args.fail_fast and not results["unit_tests"]:
            logger.error("Unit tests failed, stopping early")
            return 1
    
    # 3. Integration tests  
    if not args.unit_tests_only and not args.fast:
        logger.info("=" * 60)
        logger.info("PHASE 3: Migration Validation")
        logger.info("=" * 60)
        
        results["migration_validation"] = run_migration_validation(
            str(output_dir), logger
        )
        
        if args.fail_fast and not results["migration_validation"]:
            logger.error("Migration validation failed, stopping early")
            return 1
        
        logger.info("=" * 60)
        logger.info("PHASE 4: Configuration Validation")  
        logger.info("=" * 60)
        
        results["config_validation"] = run_config_validation(
            str(output_dir), logger
        )
        
        if args.fail_fast and not results["config_validation"]:
            logger.error("Configuration validation failed, stopping early")
            return 1
    
    # Generate summary report
    logger.info("=" * 60)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 60)
    
    summary_report = generate_summary_report(results, str(output_dir), logger)
    print("\n" + "=" * 80)
    print("MIGRATION VALIDATION SUMMARY")
    print("=" * 80)
    print(summary_report)
    
    # Determine exit code
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("🎉 All validation tests passed!")
        return 0
    else:
        failed_count = sum(1 for result in results.values() if not result)
        logger.error(f"❌ {failed_count} validation test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())