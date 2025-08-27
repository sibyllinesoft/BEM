#!/usr/bin/env python3

"""
v1.3-Stack Reproducibility Script
=================================

Single-command validation script for reproducing v1.3-Stack results.
Designed for single-GPU setups with comprehensive validation.
"""

import json
import numpy as np
import subprocess
import sys
from pathlib import Path
import time
import argparse

def check_environment():
    """Validate environment setup"""
    print("🔍 Validating Environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8+ required")
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   ✓ GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        
        if vram_gb < 8:
            print("   ⚠️  Warning: <8GB VRAM may cause OOM errors")
    except ImportError:
        raise RuntimeError("PyTorch not installed")
    
    # Check required packages
    required_packages = [
        'transformers', 'datasets', 'scipy', 'numpy', 
        'pandas', 'tqdm'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            raise RuntimeError(f"Missing package: {package}")
    
    print("✅ Environment validation passed\n")

def download_models():
    """Download required models"""
    print("📥 Downloading Models...")
    
    # Check if models already exist
    model_path = Path("models/dialogpt-small")
    if model_path.exists():
        print("   ✓ Models already downloaded")
        return
    
    # Create models directory
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Download using transformers
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "microsoft/DialoGPT-small"
        print(f"   Downloading {model_name}...")
        
        # Download and cache
        model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=model_path / "model_cache"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_path / "tokenizer_cache"
        )
        
        # Save locally
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print("   ✓ Models downloaded successfully")
    except Exception as e:
        raise RuntimeError(f"Model download failed: {e}")

def prepare_data():
    """Prepare validation dataset"""
    print("📊 Preparing Data...")
    
    data_path = Path("data")
    if all((data_path / f).exists() for f in ["train.jsonl", "val.jsonl", "test.jsonl"]):
        print("   ✓ Data already prepared")
        return
    
    # Run data preparation script
    result = subprocess.run([
        sys.executable, "scripts/prepare_validation_data.py"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Data preparation failed: {result.stderr}")
    
    print("   ✓ Data preparation complete")

def run_baseline_experiment():
    """Run S0 baseline experiment"""
    print("🔬 Running Baseline (S0)...")
    
    baseline_results = Path("logs/S0_baseline/eval.json")
    if baseline_results.exists():
        print("   ✓ Baseline results already exist")
        return
    
    # Run baseline experiment
    result = subprocess.run([
        sys.executable, "train.py",
        "--experiment", "experiments/S0_baseline.yml",
        "--output_dir", "logs/S0_baseline",
        "--seeds", "1,2,3,4,5"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Baseline training failed: {result.stderr}")
    
    print("   ✓ Baseline experiment complete")

def run_v13_stack_experiment():
    """Run S1 v1.3-stack experiment"""
    print("🚀 Running v1.3-Stack (S1)...")
    
    v13_results = Path("logs/S1_stack/eval.json")
    if v13_results.exists():
        print("   ✓ v1.3-Stack results already exist")
        return
    
    # Run v1.3-stack experiment
    result = subprocess.run([
        sys.executable, "train.py", 
        "--experiment", "experiments/S1_stack.yml",
        "--output_dir", "logs/S1_stack",
        "--seeds", "1,2,3,4,5"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"v1.3-Stack training failed: {result.stderr}")
    
    print("   ✓ v1.3-Stack experiment complete")

def run_ablation_studies():
    """Run ablation experiments"""
    print("🧪 Running Ablation Studies...")
    
    ablations = [
        ("A1_no_diagonal", "experiments/A1_no_diagonal.yml"),
        ("A2_no_hard_negatives", "experiments/A2_no_hard_negatives.yml"),
        ("A3_fp16_instead_fp8", "experiments/A3_fp16_instead_fp8.yml")
    ]
    
    for ablation_name, config_path in ablations:
        results_path = Path(f"logs/{ablation_name}/eval.json")
        
        if results_path.exists():
            print(f"   ✓ {ablation_name} already complete")
            continue
        
        print(f"   Running {ablation_name}...")
        result = subprocess.run([
            sys.executable, "train.py",
            "--experiment", config_path,
            "--output_dir", f"logs/{ablation_name}",
            "--seeds", "1,2,3,4,5"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"   ⚠️  {ablation_name} failed: {result.stderr}")
        else:
            print(f"   ✓ {ablation_name} complete")

def run_robustness_tests():
    """Run robustness experiments"""
    print("🛡️ Running Robustness Tests...")
    
    robustness_path = Path("logs/R1_robustness_second_model/eval.json")
    if robustness_path.exists():
        print("   ✓ Robustness tests already complete")
        return
    
    result = subprocess.run([
        sys.executable, "train.py",
        "--experiment", "experiments/R1_robustness_second_model.yml", 
        "--output_dir", "logs/R1_robustness_second_model",
        "--seeds", "1,2,3,4,5"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ⚠️  Robustness test failed: {result.stderr}")
    else:
        print("   ✓ Robustness tests complete")

def run_statistical_analysis():
    """Run final statistical analysis"""
    print("📊 Running Statistical Analysis...")
    
    result = subprocess.run([
        sys.executable, "v13_final_analysis.py"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Statistical analysis failed: {result.stderr}")
    
    # Verify results file exists
    results_file = Path("analysis/v13_final_statistical_results.json")
    if not results_file.exists():
        raise RuntimeError("Statistical results file not generated")
    
    print("   ✓ Statistical analysis complete")

def generate_paper_artifacts():
    """Generate paper tables and figures"""
    print("📝 Generating Paper Artifacts...")
    
    # Generate hero table
    result = subprocess.run([
        sys.executable, "generate_hero_table.py"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ⚠️  Hero table generation failed: {result.stderr}")
    else:
        print("   ✓ Hero table generated")
    
    # Verify key files exist
    key_files = [
        "paper/tables/hero_table_v13.tex",
        "paper/tables/hero_table.csv",
        "paper/sections/abstract_v13.tex",
        "paper/sections/introduction_v13.tex",
        "paper/sections/results_v13.tex",
        "paper/sections/ablation_v13.tex",
        "paper/supplement_v13.tex"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ⚠️  Missing: {file_path}")

def validate_results():
    """Validate final results against expected values"""
    print("✅ Validating Results...")
    
    # Load statistical results
    results_file = Path("analysis/v13_final_statistical_results.json")
    if not results_file.exists():
        raise RuntimeError("Statistical results not found")
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Check promotion decision
    promotion = results['promotion_decisions']['v13_stack']
    if not promotion['promoted_to_main_paper']:
        print("   ⚠️  v1.3-Stack was not promoted")
        return False
    
    # Check quality gates
    quality_gates = results['quality_gates']
    gates_passed = sum(gate['passed'] for gate in quality_gates.values())
    total_gates = len(quality_gates)
    
    print(f"   ✓ Quality Gates: {gates_passed}/{total_gates} passed")
    
    # Check key metrics
    main_comparison = results['main_comparison']
    em_improvement = main_comparison['EM']['relative_improvement_pct']
    f1_improvement = main_comparison['F1']['relative_improvement_pct']
    bleu_improvement = main_comparison['BLEU']['relative_improvement_pct']
    
    print(f"   ✓ EM improvement: +{em_improvement:.1f}%")
    print(f"   ✓ F1 improvement: +{f1_improvement:.1f}%")
    print(f"   ✓ BLEU improvement: +{bleu_improvement:.1f}%")
    
    # Validate expected ranges
    if em_improvement < 3.0 or f1_improvement < 4.0 or bleu_improvement < 7.0:
        print("   ⚠️  Improvements below expected range")
        return False
    
    print("   ✅ All validation checks passed!")
    return True

def print_summary():
    """Print final summary with key results"""
    print("\n" + "="*60)
    print("🎯 v1.3-Stack Reproduction Complete")
    print("="*60)
    
    # Load and display key results
    try:
        with open("analysis/v13_final_statistical_results.json") as f:
            results = json.load(f)
        
        main_comparison = results['main_comparison']
        promotion = results['promotion_decisions']['v13_stack']
        
        print(f"\n📊 Key Results:")
        print(f"   EM: +{main_comparison['EM']['relative_improvement_pct']:.1f}% [CI: {main_comparison['EM']['ci_lower_pct']:.1f}%, {main_comparison['EM']['ci_upper_pct']:.1f}%]")
        print(f"   F1: +{main_comparison['F1']['relative_improvement_pct']:.1f}% [CI: {main_comparison['F1']['ci_lower_pct']:.1f}%, {main_comparison['F1']['ci_upper_pct']:.1f}%]")
        print(f"   BLEU: +{main_comparison['BLEU']['relative_improvement_pct']:.1f}% [CI: {main_comparison['BLEU']['ci_lower_pct']:.1f}%, {main_comparison['BLEU']['ci_upper_pct']:.1f}%]")
        print(f"   chrF: +{main_comparison['chrF']['relative_improvement_pct']:.1f}% [CI: {main_comparison['chrF']['ci_lower_pct']:.1f}%, {main_comparison['chrF']['ci_upper_pct']:.1f}%]")
        
        print(f"\n🎯 Campaign Status: {'✅ SUCCESS' if promotion['promoted_to_main_paper'] else '❌ FAILED'}")
        print(f"   Quality Gates: {promotion['quality_gates_passed']}")
        print(f"   Aggregate Improvement: {promotion['aggregate_improvement_pct']}%")
        print(f"   Recommendation: {promotion['recommendation']}")
        
        print(f"\n📁 Generated Artifacts:")
        print(f"   Statistical Results: analysis/v13_final_statistical_results.json")
        print(f"   Hero Table (LaTeX): paper/tables/hero_table_v13.tex") 
        print(f"   Hero Table (CSV): paper/tables/hero_table.csv")
        print(f"   Paper Sections: paper/sections/*_v13.tex")
        print(f"   Supplement: paper/supplement_v13.tex")
        
    except Exception as e:
        print(f"   ⚠️  Could not load results: {e}")
    
    print(f"\n✅ Reproduction completed successfully!")
    print(f"   Total runtime: {time.time() - start_time:.1f} seconds")

def main():
    """Main reproduction workflow"""
    global start_time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Reproduce v1.3-Stack results")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip training if results already exist")
    parser.add_argument("--fast", action="store_true",
                       help="Skip optional experiments (ablations, robustness)")
    args = parser.parse_args()
    
    print("🎯 v1.3-Stack Reproducibility Script")
    print("="*50)
    
    try:
        # Core validation steps
        check_environment()
        download_models()
        prepare_data()
        
        # Training experiments
        if not args.skip_training:
            run_baseline_experiment()
            run_v13_stack_experiment()
            
            if not args.fast:
                run_ablation_studies()
                run_robustness_tests()
        
        # Analysis and artifacts  
        run_statistical_analysis()
        generate_paper_artifacts()
        
        # Final validation
        if validate_results():
            print_summary()
        else:
            print("❌ Validation failed - results may be inconsistent")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏸️  Reproduction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Reproduction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()