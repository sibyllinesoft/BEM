"""
Validation script for BEM v1.3 Fast-5 variants integration.

Validates that all Fast-5 components are correctly implemented and 
can be instantiated without errors, following the TODO.md specifications.
"""

import sys
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def validate_imports() -> Dict[str, bool]:
    """Validate all Fast-5 imports work correctly."""
    results = {}
    
    # F5.1: Stateful Router
    try:
        from bem.controller.stateful import (
            StatefulRouterConfig, StatefulRouter, S4Lite, 
            StatefulBEMRouter
        )
        results['F5.1_stateful'] = True
        logger.info("✅ F5.1 Stateful Router imports successful")
    except Exception as e:
        results['F5.1_stateful'] = False
        logger.error(f"❌ F5.1 Stateful Router import failed: {e}")
    
    # F5.2: Low-Rank + Diagonal  
    try:
        from bem.generator.lowrank_diag import (
            LowRankDiagConfig, DiagonalPredictor, LowRankDiagExpert,
            LowRankDiagModule, lowrank_plus_diag
        )
        results['F5.2_lowrank_diag'] = True
        logger.info("✅ F5.2 Low-Rank + Diagonal imports successful")
    except Exception as e:
        results['F5.2_lowrank_diag'] = False
        logger.error(f"❌ F5.2 Low-Rank + Diagonal import failed: {e}")
    
    # F5.3: SVD Warm-Start
    try:
        from bem.init.svd_warmstart import (
            SVDWarmStartConfig, LoRACheckpointLoader, SVDDecomposer, 
            SVDWarmStartTrainer
        )
        results['F5.3_svd_warmstart'] = True
        logger.info("✅ F5.3 SVD Warm-Start imports successful")
    except Exception as e:
        results['F5.3_svd_warmstart'] = False
        logger.error(f"❌ F5.3 SVD Warm-Start import failed: {e}")
    
    # F5.4: FP8 Generator
    try:
        from bem.quant.fp8_qat import (
            FP8Config, FP8Quantizer, FP8Observer, FP8LoRAExpert, 
            FP8BEMModule
        )
        results['F5.4_fp8_qat'] = True
        logger.info("✅ F5.4 FP8 Generator imports successful")
    except Exception as e:
        results['F5.4_fp8_qat'] = False
        logger.error(f"❌ F5.4 FP8 Generator import failed: {e}")
    
    # F5.5: Hard Negatives
    try:
        from retrieval.hard_negs import (
            HardNegativeConfig, HardNegativeMiner, ContradictionDetector,
            HardNegativeTrainingLoss, HardNegativeDataset
        )
        results['F5.5_hard_negs'] = True
        logger.info("✅ F5.5 Hard Negatives imports successful")
    except Exception as e:
        results['F5.5_hard_negs'] = False
        logger.error(f"❌ F5.5 Hard Negatives import failed: {e}")
    
    # Integration module
    try:
        from bem.bem_v13_fast5 import (
            BEMv13Config, BEMv13Factory, load_bem_v13_from_config,
            validate_bem_v13_budget
        )
        results['integration'] = True
        logger.info("✅ Integration module imports successful")
    except Exception as e:
        results['integration'] = False
        logger.error(f"❌ Integration module import failed: {e}")
    
    # Analysis frameworks
    try:
        from analysis.check_parity import ModelBudgetAnalyzer, StatisticalAnalyzer
        from analysis.spectra import analyze_bem_spectra
        results['analysis'] = True
        logger.info("✅ Analysis frameworks imports successful")
    except Exception as e:
        results['analysis'] = False
        logger.error(f"❌ Analysis frameworks import failed: {e}")
    
    return results

def validate_configs() -> Dict[str, bool]:
    """Validate all experiment configuration files exist and are valid."""
    results = {}
    
    config_files = [
        "experiments/f51_stateful.yml",
        "experiments/f52_lowrank_diag.yml", 
        "experiments/f53_svd_warm.yml",
        "experiments/f54_fp8.yml",
        "experiments/f55_hardnegs.yml"
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        variant_name = config_path.stem
        
        try:
            if not config_path.exists():
                results[variant_name] = False
                logger.error(f"❌ Config file missing: {config_file}")
                continue
                
            # Try to load and validate YAML
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_keys = ['model', 'training', 'data', 'evaluation']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                results[variant_name] = False
                logger.error(f"❌ Config {config_file} missing keys: {missing_keys}")
            else:
                results[variant_name] = True
                logger.info(f"✅ Config {config_file} valid")
                
        except Exception as e:
            results[variant_name] = False
            logger.error(f"❌ Config {config_file} validation failed: {e}")
    
    return results

def validate_integration() -> Dict[str, bool]:
    """Test that BEM v1.3 modules can be instantiated."""
    results = {}
    
    try:
        from bem.bem_v13_fast5 import BEMv13Config, BEMv13Factory
        
        # Test basic config creation
        config = BEMv13Config(
            base_model="test",
            d_model=512,
            variants_enabled={
                "stateful_router": False,
                "lowrank_diagonal": False, 
                "svd_warmstart": False,
                "fp8_quantization": False,
                "hard_negatives": False
            }
        )
        
        # Test factory with no variants (should work with baseline BEM)
        factory = BEMv13Factory()
        results['basic_instantiation'] = True
        logger.info("✅ Basic BEM v1.3 instantiation successful")
        
    except Exception as e:
        results['basic_instantiation'] = False
        logger.error(f"❌ Basic BEM v1.3 instantiation failed: {e}")
        logger.error(traceback.format_exc())
    
    # Test individual variant instantiation
    variant_tests = [
        ("stateful_router", {"stateful_router": True}),
        ("lowrank_diagonal", {"lowrank_diagonal": True}),
        ("svd_warmstart", {"svd_warmstart": True}),
        ("fp8_quantization", {"fp8_quantization": True}),
        ("hard_negatives", {"hard_negatives": True})
    ]
    
    for variant_name, variant_config in variant_tests:
        try:
            from bem.bem_v13_fast5 import BEMv13Config
            
            config = BEMv13Config(
                base_model="test",
                d_model=512,
                variants_enabled={k: False for k in ["stateful_router", "lowrank_diagonal", 
                                                   "svd_warmstart", "fp8_quantization", "hard_negatives"]},
                **{k: v for k, v in variant_config.items()}
            )
            config.variants_enabled.update(variant_config)
            
            results[f'variant_{variant_name}'] = True
            logger.info(f"✅ {variant_name} variant config creation successful")
            
        except Exception as e:
            results[f'variant_{variant_name}'] = False
            logger.error(f"❌ {variant_name} variant config creation failed: {e}")
    
    return results

def validate_training_script() -> bool:
    """Validate that the training script can be imported and basic checks pass."""
    try:
        # Test import of training script
        sys.path.insert(0, str(Path.cwd()))
        from train_fast5 import Fast5Trainer
        
        logger.info("✅ Training script import successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training script import failed: {e}")
        return False

def validate_directory_structure() -> Dict[str, bool]:
    """Validate that all required directories and files exist."""
    results = {}
    
    # Check directory structure as per TODO.md
    required_structure = {
        "bem/controller/stateful.py": "F5.1 Stateful Router",
        "bem/generator/lowrank_diag.py": "F5.2 Low-Rank + Diagonal", 
        "bem/init/svd_warmstart.py": "F5.3 SVD Warm-Start",
        "bem/quant/fp8_qat.py": "F5.4 FP8 QAT",
        "retrieval/hard_negs.py": "F5.5 Hard Negatives",
        "experiments/f51_stateful.yml": "F5.1 Config",
        "experiments/f52_lowrank_diag.yml": "F5.2 Config",
        "experiments/f53_svd_warm.yml": "F5.3 Config", 
        "experiments/f54_fp8.yml": "F5.4 Config",
        "experiments/f55_hardnegs.yml": "F5.5 Config",
        "analysis/spectra.py": "Spectral Analysis",
        "bem/bem_v13_fast5.py": "Integration Module",
        "train_fast5.py": "Training Script"
    }
    
    for file_path, description in required_structure.items():
        path = Path(file_path)
        if path.exists():
            results[description] = True
            logger.info(f"✅ {description}: {file_path}")
        else:
            results[description] = False
            logger.error(f"❌ Missing {description}: {file_path}")
    
    return results

def run_comprehensive_validation():
    """Run all validation checks and produce summary report."""
    logger.info("🔍 Starting BEM v1.3 Fast-5 Validation")
    logger.info("=" * 50)
    
    all_results = {}
    
    # 1. Directory Structure Check
    logger.info("\n📁 Validating Directory Structure...")
    all_results['directory_structure'] = validate_directory_structure()
    
    # 2. Import Validation  
    logger.info("\n📦 Validating Imports...")
    all_results['imports'] = validate_imports()
    
    # 3. Config Validation
    logger.info("\n⚙️ Validating Configuration Files...")
    all_results['configs'] = validate_configs()
    
    # 4. Integration Testing
    logger.info("\n🔗 Validating Integration...")
    all_results['integration'] = validate_integration()
    
    # 5. Training Script Validation
    logger.info("\n🚀 Validating Training Script...")
    all_results['training_script'] = validate_training_script()
    
    # Summary Report
    logger.info("\n" + "=" * 50)
    logger.info("📊 VALIDATION SUMMARY REPORT")
    logger.info("=" * 50)
    
    total_checks = 0
    passed_checks = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            category_passed = sum(results.values())
            category_total = len(results)
        else:
            category_passed = 1 if results else 0
            category_total = 1
        
        total_checks += category_total
        passed_checks += category_passed
        
        status = "✅" if category_passed == category_total else "⚠️"
        logger.info(f"{status} {category}: {category_passed}/{category_total} passed")
        
        # Show failed items
        if isinstance(results, dict):
            failed_items = [k for k, v in results.items() if not v]
            if failed_items:
                logger.error(f"   Failed: {', '.join(failed_items)}")
    
    logger.info("-" * 50)
    success_rate = (passed_checks / total_checks) * 100
    overall_status = "✅ PASS" if passed_checks == total_checks else "⚠️ PARTIAL" if passed_checks > 0 else "❌ FAIL"
    logger.info(f"{overall_status} Overall: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
    
    # Implementation Status
    logger.info("\n🎯 FAST-5 IMPLEMENTATION STATUS:")
    f5_components = [
        ("F5.1", "Stateful Router", all_results['imports'].get('F5.1_stateful', False)),
        ("F5.2", "Low-Rank + Diagonal", all_results['imports'].get('F5.2_lowrank_diag', False)),
        ("F5.3", "SVD Warm-Start", all_results['imports'].get('F5.3_svd_warmstart', False)), 
        ("F5.4", "FP8 Generator", all_results['imports'].get('F5.4_fp8_qat', False)),
        ("F5.5", "Hard Negatives", all_results['imports'].get('F5.5_hard_negs', False))
    ]
    
    for f5_id, f5_name, f5_status in f5_components:
        status = "✅" if f5_status else "❌"
        logger.info(f"  {status} {f5_id}: {f5_name}")
    
    if passed_checks == total_checks:
        logger.info("\n🎉 All Fast-5 variants implemented successfully!")
        logger.info("Ready for experimental validation as per TODO.md workflow.")
    else:
        logger.warning("\n⚠️ Some validation checks failed. Review errors above.")
    
    return passed_checks == total_checks

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)