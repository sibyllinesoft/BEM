#!/usr/bin/env python3
"""
BEM v1.3 Performance+Agentic Sprint - Main Execution Script
Runs the complete suite of experiments with statistical validation
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess
from datetime import datetime

# Setup paths
ROOT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
WORKFLOWS_DIR = ROOT_DIR / "workflows"

# Add to Python path
sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_dependencies():
    """Check if required dependencies are available."""
    dependencies = {
        'python': ['python', '--version'],
        'torch': ['python', '-c', 'import torch; print(f"PyTorch {torch.__version__}")'],
        'numpy': ['python', '-c', 'import numpy; print(f"NumPy {numpy.__version__}")'],
        'scipy': ['python', '-c', 'import scipy; print(f"SciPy {scipy.__version__}")'],
        'pandas': ['python', '-c', 'import pandas; print(f"Pandas {pandas.__version__}")'],
        'yaml': ['python', '-c', 'import yaml; print("PyYAML available")'],
    }
    
    logger.info("Checking dependencies...")
    
    for name, cmd in dependencies.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✓ {name}: {result.stdout.strip()}")
            else:
                logger.error(f"✗ {name}: Failed to import")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error(f"✗ {name}: Not available")
            return False
    
    return True


def list_available_experiments():
    """List all available experiment configurations."""
    if not EXPERIMENTS_DIR.exists():
        logger.error(f"Experiments directory not found: {EXPERIMENTS_DIR}")
        return []
    
    experiments = []
    for config_file in EXPERIMENTS_DIR.glob("*.yml"):
        experiments.append(config_file.stem)
    
    return sorted(experiments)


def run_experiment_suite(experiments=None, output_dir=None, skip_baseline=False, 
                        dry_run=False, log_level="INFO"):
    """Run the complete BEM v1.3 experiment suite."""
    
    # Setup output directory
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT_DIR / "results" / f"bem_v13_suite_{timestamp}"
    else:
        output_dir = Path(output_dir)
    
    logger.info(f"BEM v1.3 Performance+Agentic Sprint")
    logger.info(f"Output directory: {output_dir}")
    
    # Check dependencies
    if not dry_run and not check_dependencies():
        logger.error("Dependency check failed. Please install required packages.")
        return False
    
    # List available experiments
    available_experiments = list_available_experiments()
    if not available_experiments:
        logger.error("No experiment configurations found.")
        return False
    
    logger.info(f"Available experiments: {', '.join(available_experiments)}")
    
    # Filter experiments if specified
    if experiments:
        invalid_experiments = set(experiments) - set(available_experiments)
        if invalid_experiments:
            logger.error(f"Invalid experiments specified: {invalid_experiments}")
            return False
        selected_experiments = experiments
    else:
        selected_experiments = available_experiments
    
    logger.info(f"Running experiments: {', '.join(selected_experiments)}")
    
    if dry_run:
        logger.info("DRY RUN - Not executing actual training")
        return True
    
    # Prepare experiment runner command
    runner_script = WORKFLOWS_DIR / "experiment_runner.py"
    cmd = [
        sys.executable, str(runner_script),
        "--config-dir", str(EXPERIMENTS_DIR),
        "--output-root", str(output_dir),
        "--log-level", log_level
    ]
    
    if selected_experiments != available_experiments:
        cmd.extend(["--experiments"] + selected_experiments)
    
    if skip_baseline:
        cmd.append("--skip-baseline")
    
    logger.info(f"Executing command: {' '.join(map(str, cmd))}")
    
    try:
        # Run the experiment suite
        result = subprocess.run(cmd, cwd=ROOT_DIR)
        
        if result.returncode == 0:
            logger.info("🎉 Experiment suite completed successfully!")
            logger.info(f"Results available in: {output_dir}")
            return True
        else:
            logger.error(f"❌ Experiment suite failed with exit code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        logger.info("⚠️  Experiment suite interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to execute experiment suite: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BEM v1.3 Performance+Agentic Sprint - Complete Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_bem_experiments.py
  
  # Run specific experiments
  python run_bem_experiments.py --experiments v1_dynrank v2_gateshaping ar1_pg
  
  # Dry run to check configuration
  python run_bem_experiments.py --dry-run
  
  # Skip baseline comparison (faster)
  python run_bem_experiments.py --skip-baseline
  
  # Custom output directory
  python run_bem_experiments.py --output-dir ./my_results

Available Experiments:
  Performance Variants (PT1-PT4):
    - v1_dynrank:     PT1 Head-Group Gating + Dynamic Rank Mask  
    - v2_gateshaping: PT1 Head-Group Gating + Gate-Shaping v2
    - v3_kron:        PT3 Kronecker @ W_down with fused kernels
    - v4_film:        PT4 Residual FiLM with micro-γ,β modulation
  
  Agentic Router:
    - ar0_bc:         AR0 Behavioral Cloning baseline
    - ar1_pg:         AR1 Policy Gradient with TRPO trust region
  
  Online Learning:
    - ol_shadow:      OL Online Shadow Mode with EWC/Prox regularization
  
  Multimodal Integration:
    - mm_mini:        MM Multimodal Mini with vision features
  
  Safety Alignment:
    - vc_curve:       VC Safety Basis Curve with constitutional constraints

Statistical Analysis:
  All experiments include rigorous statistical validation with:
  • BCa (Bias-Corrected and Accelerated) Bootstrap confidence intervals
  • Benjamini-Hochberg FDR correction for multiple hypothesis testing
  • Performance gate validation with ±5% parameter/FLOP parity constraints
  • Comprehensive evaluation reports with effect sizes and significance testing
        """
    )
    
    parser.add_argument(
        "--experiments", nargs="+", 
        help="Specific experiments to run (default: all)"
    )
    
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory for results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip baseline comparison (faster execution)"
    )
    
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check configuration without running experiments"
    )
    
    parser.add_argument(
        "--list-experiments", action="store_true",
        help="List available experiments and exit"
    )
    
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Handle list experiments
    if args.list_experiments:
        experiments = list_available_experiments()
        if experiments:
            print("Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print("No experiments found.")
        return
    
    # Run experiment suite
    success = run_experiment_suite(
        experiments=args.experiments,
        output_dir=args.output_dir,
        skip_baseline=args.skip_baseline,
        dry_run=args.dry_run,
        log_level=args.log_level
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()