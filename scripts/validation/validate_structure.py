"""
Lightweight validation script for BEM v1.3 Fast-5 file structure.
Checks that all required files exist and have basic syntax correctness.
"""

import sys
import ast
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def validate_python_syntax(file_path: Path) -> bool:
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False

def validate_yaml_syntax(file_path: Path) -> bool:
    """Validate that a YAML file has correct syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True
    except yaml.YAMLError as e:
        logger.error(f"YAML error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False

def validate_directory_structure() -> Dict[str, bool]:
    """Validate that all required files exist and have correct syntax."""
    results = {}
    
    # Python files to check
    python_files = {
        "bem/controller/stateful.py": "F5.1 Stateful Router",
        "bem/generator/lowrank_diag.py": "F5.2 Low-Rank + Diagonal", 
        "bem/init/svd_warmstart.py": "F5.3 SVD Warm-Start",
        "bem/quant/fp8_qat.py": "F5.4 FP8 QAT",
        "retrieval/hard_negs.py": "F5.5 Hard Negatives",
        "analysis/spectra.py": "Spectral Analysis",
        "bem/bem_v13_fast5.py": "Integration Module",
        "train_fast5.py": "Training Script",
        "analysis/check_parity.py": "Budget Parity Analysis"
    }
    
    # YAML files to check  
    yaml_files = {
        "experiments/f51_stateful.yml": "F5.1 Config",
        "experiments/f52_lowrank_diag.yml": "F5.2 Config",
        "experiments/f53_svd_warm.yml": "F5.3 Config", 
        "experiments/f54_fp8.yml": "F5.4 Config",
        "experiments/f55_hardnegs.yml": "F5.5 Config"
    }
    
    logger.info("📁 Checking Python Files...")
    for file_path, description in python_files.items():
        path = Path(file_path)
        if not path.exists():
            results[description] = False
            logger.error(f"❌ Missing {description}: {file_path}")
        elif not validate_python_syntax(path):
            results[description] = False  
            logger.error(f"❌ Syntax error in {description}: {file_path}")
        else:
            results[description] = True
            logger.info(f"✅ {description}: {file_path}")
    
    logger.info("\n⚙️ Checking YAML Configuration Files...")
    for file_path, description in yaml_files.items():
        path = Path(file_path)
        if not path.exists():
            results[description] = False
            logger.error(f"❌ Missing {description}: {file_path}")
        elif not validate_yaml_syntax(path):
            results[description] = False
            logger.error(f"❌ YAML error in {description}: {file_path}")
        else:
            results[description] = True
            logger.info(f"✅ {description}: {file_path}")
    
    return results

def check_file_content(file_path: Path, expected_patterns: List[str]) -> bool:
    """Check if file contains expected patterns (basic content validation)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_patterns = []
        for pattern in expected_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            logger.warning(f"⚠️ {file_path} missing expected patterns: {missing_patterns}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False

def validate_content_patterns() -> Dict[str, bool]:
    """Validate that files contain expected key components."""
    results = {}
    
    content_checks = {
        "bem/controller/stateful.py": [
            "class StatefulRouter",
            "class S4Lite", 
            "class StatefulBEMRouter",
            "StatefulRouterConfig"
        ],
        "bem/generator/lowrank_diag.py": [
            "class LowRankDiagExpert",
            "class DiagonalPredictor",
            "def lowrank_plus_diag",
            "LowRankDiagConfig"
        ],
        "bem/init/svd_warmstart.py": [
            "class SVDDecomposer",
            "class LoRACheckpointLoader", 
            "class SVDWarmStartTrainer",
            "SVDWarmStartConfig"
        ],
        "bem/quant/fp8_qat.py": [
            "class FP8Quantizer",
            "class FP8Observer",
            "class FP8LoRAExpert",
            "FP8Config"
        ],
        "retrieval/hard_negs.py": [
            "class HardNegativeMiner",
            "class ContradictionDetector",
            "class HardNegativeTrainingLoss",
            "HardNegativeConfig"
        ],
        "bem/bem_v13_fast5.py": [
            "class BEMv13Config",
            "class BEMv13Factory", 
            "def load_bem_v13_from_config",
            "def validate_bem_v13_budget"
        ]
    }
    
    logger.info("\n🔍 Checking Content Patterns...")
    for file_path, patterns in content_checks.items():
        path = Path(file_path)
        if path.exists():
            results[f"content_{path.stem}"] = check_file_content(path, patterns)
        else:
            results[f"content_{path.stem}"] = False
    
    return results

def count_lines_of_code() -> Dict[str, int]:
    """Count lines of code in each Fast-5 component."""
    python_files = [
        "bem/controller/stateful.py",
        "bem/generator/lowrank_diag.py", 
        "bem/init/svd_warmstart.py",
        "bem/quant/fp8_qat.py",
        "retrieval/hard_negs.py",
        "bem/bem_v13_fast5.py",
        "train_fast5.py"
    ]
    
    loc_counts = {}
    total_loc = 0
    
    logger.info("\n📊 Lines of Code Analysis...")
    for file_path in python_files:
        path = Path(file_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                # Count non-empty, non-comment lines
                code_lines = [line for line in lines 
                             if line.strip() and not line.strip().startswith('#')]
                loc = len(code_lines)
                loc_counts[path.stem] = loc
                total_loc += loc
                logger.info(f"  {path.stem}: {loc} lines")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                loc_counts[path.stem] = 0
        else:
            loc_counts[path.stem] = 0
    
    logger.info(f"  Total Fast-5 Implementation: {total_loc} lines of code")
    return loc_counts

def run_structure_validation():
    """Run structure and syntax validation."""
    logger.info("🔍 Starting BEM v1.3 Fast-5 Structure Validation")
    logger.info("=" * 60)
    
    all_results = {}
    
    # 1. Directory Structure & Syntax Check
    all_results['structure'] = validate_directory_structure()
    
    # 2. Content Pattern Check  
    all_results['content'] = validate_content_patterns()
    
    # 3. Lines of Code Analysis
    loc_counts = count_lines_of_code()
    
    # Summary Report
    logger.info("\n" + "=" * 60)
    logger.info("📊 STRUCTURE VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    total_checks = 0
    passed_checks = 0
    
    for category, results in all_results.items():
        category_passed = sum(results.values())
        category_total = len(results)
        
        total_checks += category_total
        passed_checks += category_passed
        
        status = "✅" if category_passed == category_total else "⚠️"
        logger.info(f"{status} {category}: {category_passed}/{category_total} passed")
        
        # Show failed items
        failed_items = [k for k, v in results.items() if not v]
        if failed_items:
            logger.error(f"   Failed: {', '.join(failed_items)}")
    
    logger.info("-" * 60)
    success_rate = (passed_checks / total_checks) * 100
    overall_status = ("✅ PASS" if passed_checks == total_checks 
                     else "⚠️ PARTIAL" if passed_checks > 0 else "❌ FAIL")
    logger.info(f"{overall_status} Overall: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
    
    # Fast-5 Implementation Summary
    logger.info("\n🎯 FAST-5 IMPLEMENTATION SUMMARY:")
    f5_components = [
        ("F5.1", "Stateful Router", "bem/controller/stateful.py"),
        ("F5.2", "Low-Rank + Diagonal", "bem/generator/lowrank_diag.py"),
        ("F5.3", "SVD Warm-Start", "bem/init/svd_warmstart.py"), 
        ("F5.4", "FP8 Generator", "bem/quant/fp8_qat.py"),
        ("F5.5", "Hard Negatives", "retrieval/hard_negs.py")
    ]
    
    implemented_count = 0
    for f5_id, f5_name, f5_path in f5_components:
        if Path(f5_path).exists():
            status = "✅"
            implemented_count += 1
        else:
            status = "❌"
        logger.info(f"  {status} {f5_id}: {f5_name}")
    
    logger.info(f"\n📈 Implementation Status: {implemented_count}/5 Fast-5 variants completed")
    
    if passed_checks == total_checks:
        logger.info("\n🎉 All file structure and syntax checks passed!")
        logger.info("Fast-5 implementation ready for runtime testing.")
        logger.info("\nNext Steps (as per TODO.md):")
        logger.info("1. Run budget parity validation")  
        logger.info("2. Execute Fast-5 training workflow")
        logger.info("3. Statistical analysis with BCa bootstrap")
        logger.info("4. Promote CI-backed winners")
    else:
        logger.warning("\n⚠️ Some validation checks failed. Review errors above.")
    
    return passed_checks == total_checks

if __name__ == "__main__":
    success = run_structure_validation()
    sys.exit(0 if success else 1)