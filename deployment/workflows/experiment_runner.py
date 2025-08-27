"""
Experiment Runner for BEM v1.3 - Automated Training and Evaluation Pipeline
Orchestrates the complete experimental workflow from configuration to statistical analysis
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json
from datetime import datetime
from dataclasses import dataclass

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from bem2.evaluation.evaluation_framework import EvaluationFramework
from bem2.evaluation.statistical_analysis import StatisticalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    config_path: Path
    variant_id: str
    base_config: Optional[str] = None
    gpu_memory_gb: int = 16
    timeout_hours: int = 12
    priority: int = 1  # Lower numbers = higher priority


class ExperimentRunner:
    """
    Main experiment runner for BEM v1.3 system.
    
    Handles training execution, log monitoring, and evaluation coordination.
    """
    
    def __init__(self, 
                 output_root: Path,
                 python_cmd: str = "python",
                 max_concurrent: int = 1):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.python_cmd = python_cmd
        self.max_concurrent = max_concurrent
        self.running_processes = {}
        
        # Initialize evaluation framework
        self.evaluator = EvaluationFramework(
            output_dir=self.output_root / "evaluation_results",
            bootstrap_samples=10000,
            confidence_level=0.95,
            fdr_alpha=0.05
        )
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_root / "experiment_runner.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def discover_experiments(self, config_dir: Path) -> List[ExperimentConfig]:
        """
        Discover all experiment configurations in a directory.
        
        Args:
            config_dir: Directory containing experiment configs
            
        Returns:
            List of experiment configurations sorted by priority
        """
        experiments = []
        
        for config_file in config_dir.glob("*.yml"):
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                experiment = ExperimentConfig(
                    name=config.get('name', config_file.stem),
                    config_path=config_file,
                    variant_id=config.get('variant_id', 'unknown'),
                    base_config=config.get('base_config'),
                    gpu_memory_gb=config.get('hardware', {}).get('min_gpu_memory_gb', 16),
                    timeout_hours=config.get('training', {}).get('max_hours', 12),
                    priority=config.get('priority', 1)
                )
                
                experiments.append(experiment)
                logger.info(f"Discovered experiment: {experiment.name} ({experiment.variant_id})")
                
            except Exception as e:
                logger.error(f"Failed to parse config {config_file}: {e}")
        
        # Sort by priority (lower number = higher priority)
        experiments.sort(key=lambda x: x.priority)
        return experiments
    
    def check_prerequisites(self, experiment: ExperimentConfig) -> bool:
        """
        Check if prerequisites are met for running an experiment.
        
        Args:
            experiment: Experiment configuration
            
        Returns:
            True if prerequisites are met
        """
        checks = {
            'config_exists': experiment.config_path.exists(),
            'python_available': self._check_command_available(self.python_cmd),
            'gpu_memory': self._check_gpu_memory(experiment.gpu_memory_gb),
            'disk_space': self._check_disk_space(self.output_root, min_gb=10),
            'base_config': True  # Will check if base_config specified
        }
        
        # Check base config if specified
        if experiment.base_config:
            base_path = experiment.config_path.parent / experiment.base_config
            checks['base_config'] = base_path.exists()
            if not checks['base_config']:
                logger.error(f"Base config not found: {base_path}")
        
        failed_checks = [name for name, passed in checks.items() if not passed]
        
        if failed_checks:
            logger.error(f"Prerequisites failed for {experiment.name}: {failed_checks}")
            return False
        
        logger.info(f"Prerequisites passed for {experiment.name}")
        return True
    
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in PATH."""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_gpu_memory(self, required_gb: int) -> bool:
        """Check if sufficient GPU memory is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                available_mb = max(int(line.strip()) for line in result.stdout.strip().split('\n'))
                available_gb = available_mb / 1024
                return available_gb >= required_gb
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
        
        # Default to True if we can't check (assume user knows what they're doing)
        return True
    
    def _check_disk_space(self, path: Path, min_gb: int) -> bool:
        """Check if sufficient disk space is available."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (1024**3)
            return free_gb >= min_gb
        except Exception:
            return True  # Default to True if we can't check
    
    def run_training(self, experiment: ExperimentConfig) -> Path:
        """
        Execute training for a single experiment.
        
        Args:
            experiment: Experiment to run
            
        Returns:
            Path to experiment output directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_output_dir = self.output_root / f"{experiment.name}_{timestamp}"
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare training command
        cmd = [
            self.python_cmd, "-m", "bem2.training.train",
            "--config", str(experiment.config_path),
            "--output-dir", str(exp_output_dir),
            "--experiment-name", experiment.name
        ]
        
        logger.info(f"Starting training: {experiment.name}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Output directory: {exp_output_dir}")
        
        # Run training with logging
        log_file = exp_output_dir / "training.log"
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output to log file and monitor progress
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    
                    # Log important messages to main logger
                    if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed']):
                        logger.error(f"{experiment.name}: {line.strip()}")
                    elif any(keyword in line.lower() for keyword in ['epoch', 'step', 'eval']):
                        logger.debug(f"{experiment.name}: {line.strip()}")
                
                process.wait()
                
                if process.returncode == 0:
                    logger.info(f"Training completed successfully: {experiment.name}")
                else:
                    logger.error(f"Training failed: {experiment.name} (exit code: {process.returncode})")
                    
        except Exception as e:
            logger.error(f"Training execution failed for {experiment.name}: {e}")
            
        return exp_output_dir
    
    def run_baseline_comparison(self, 
                              experiment_output: Path,
                              experiment: ExperimentConfig) -> Optional[Path]:
        """
        Run baseline model for comparison if needed.
        
        Args:
            experiment_output: Directory with experiment results
            experiment: Original experiment configuration
            
        Returns:
            Path to baseline output directory or None
        """
        # Check if baseline already exists or if base_config provides baseline
        baseline_dir = self.output_root / "baselines" / experiment.variant_id
        
        if baseline_dir.exists():
            logger.info(f"Using existing baseline: {baseline_dir}")
            return baseline_dir
        
        if not experiment.base_config:
            logger.warning(f"No baseline available for {experiment.name}")
            return None
        
        # Run baseline training
        logger.info(f"Running baseline for {experiment.name}")
        
        baseline_config_path = experiment.config_path.parent / experiment.base_config
        if not baseline_config_path.exists():
            logger.error(f"Baseline config not found: {baseline_config_path}")
            return None
        
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.python_cmd, "-m", "bem2.training.train",
            "--config", str(baseline_config_path),
            "--output-dir", str(baseline_dir),
            "--experiment-name", f"{experiment.name}_baseline"
        ]
        
        try:
            with open(baseline_dir / "baseline_training.log", 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=experiment.timeout_hours * 3600
                )
                
                if process.returncode == 0:
                    logger.info(f"Baseline training completed: {experiment.name}")
                    return baseline_dir
                else:
                    logger.error(f"Baseline training failed: {experiment.name}")
                    return None
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Baseline training timeout: {experiment.name}")
            return None
        except Exception as e:
            logger.error(f"Baseline training error: {e}")
            return None
    
    def run_evaluation(self, 
                      experiment: ExperimentConfig,
                      experiment_output: Path,
                      baseline_output: Optional[Path] = None):
        """
        Run complete evaluation including statistical analysis.
        
        Args:
            experiment: Experiment configuration
            experiment_output: Directory with experiment results
            baseline_output: Directory with baseline results (optional)
        """
        logger.info(f"Starting evaluation: {experiment.name}")
        
        try:
            if baseline_output and baseline_output.exists():
                # Full evaluation with statistical comparison
                result = self.evaluator.evaluate_experiment(
                    experiment_name=experiment.name,
                    treatment_log_dir=experiment_output,
                    baseline_log_dir=baseline_output,
                    config_path=experiment.config_path
                )
                
                # Log key results
                if result.statistical_analysis:
                    sig_results = [r for r in result.statistical_analysis.results if r.significant]
                    logger.info(f"Evaluation complete: {len(sig_results)} significant improvements")
                    
                    for result_stat in sig_results:
                        direction = "↑" if result_stat.effect_size > 0 else "↓"
                        logger.info(f"  {result_stat.metric_name}: {direction} {result_stat.effect_size:.4f}")
                else:
                    logger.warning(f"No statistical analysis performed for {experiment.name}")
                
                logger.info(f"Performance gates: {'PASS' if result.gates_passed else 'FAIL'}")
                
            else:
                # Evaluation without baseline comparison
                logger.warning(f"No baseline available for {experiment.name}, skipping statistical comparison")
                
                # Still validate gates against absolute thresholds
                from bem2.evaluation.evaluation_framework import MetricCollector, GateValidator
                
                collector = MetricCollector()
                metrics = collector.collect_from_logs(experiment_output)
                
                gate_validator = GateValidator.from_config(experiment.config_path)
                gates_passed, gate_results = gate_validator.validate_gates(metrics)
                
                logger.info(f"Performance gates: {'PASS' if gates_passed else 'FAIL'}")
                for gate_name, passed in gate_results.items():
                    status = "PASS" if passed else "FAIL"
                    logger.info(f"  {gate_name}: {status}")
        
        except Exception as e:
            logger.error(f"Evaluation failed for {experiment.name}: {e}")
            raise
    
    def run_experiment_suite(self, 
                           config_dir: Path,
                           experiments: Optional[List[str]] = None,
                           skip_baseline: bool = False) -> List[Dict[str, Any]]:
        """
        Run a complete suite of experiments.
        
        Args:
            config_dir: Directory containing experiment configurations
            experiments: Optional list of specific experiments to run
            skip_baseline: Skip baseline comparison if True
            
        Returns:
            List of experiment results
        """
        logger.info("Starting experiment suite execution")
        logger.info(f"Config directory: {config_dir}")
        
        # Discover experiments
        all_experiments = self.discover_experiments(config_dir)
        
        # Filter experiments if specific ones requested
        if experiments:
            all_experiments = [exp for exp in all_experiments if exp.name in experiments]
            logger.info(f"Filtered to {len(all_experiments)} requested experiments")
        
        if not all_experiments:
            logger.error("No experiments found to run")
            return []
        
        results = []
        
        for i, experiment in enumerate(all_experiments, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"EXPERIMENT {i}/{len(all_experiments)}: {experiment.name} ({experiment.variant_id})")
            logger.info(f"{'='*80}")
            
            # Check prerequisites
            if not self.check_prerequisites(experiment):
                logger.error(f"Skipping {experiment.name} due to failed prerequisites")
                continue
            
            try:
                # Run training
                start_time = time.time()
                experiment_output = self.run_training(experiment)
                training_time = time.time() - start_time
                
                # Run baseline if needed
                baseline_output = None
                if not skip_baseline:
                    baseline_output = self.run_baseline_comparison(experiment_output, experiment)
                
                # Run evaluation
                evaluation_start = time.time()
                self.run_evaluation(experiment, experiment_output, baseline_output)
                evaluation_time = time.time() - evaluation_start
                
                # Record results
                result = {
                    'experiment_name': experiment.name,
                    'variant_id': experiment.variant_id,
                    'status': 'completed',
                    'training_time_minutes': training_time / 60,
                    'evaluation_time_minutes': evaluation_time / 60,
                    'output_directory': str(experiment_output),
                    'baseline_directory': str(baseline_output) if baseline_output else None
                }
                
                results.append(result)
                logger.info(f"COMPLETED: {experiment.name} ({training_time/60:.1f}m training, {evaluation_time/60:.1f}m eval)")
                
            except Exception as e:
                logger.error(f"FAILED: {experiment.name} - {e}")
                results.append({
                    'experiment_name': experiment.name,
                    'variant_id': experiment.variant_id, 
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save suite results
        suite_result_file = self.output_root / f"experiment_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(suite_result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nExperiment suite completed. Results saved to: {suite_result_file}")
        
        # Summary
        completed = len([r for r in results if r['status'] == 'completed'])
        failed = len([r for r in results if r['status'] == 'failed'])
        logger.info(f"Summary: {completed} completed, {failed} failed")
        
        return results


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="BEM v1.3 Experiment Runner")
    parser.add_argument("--config-dir", type=Path, required=True,
                       help="Directory containing experiment configurations")
    parser.add_argument("--output-root", type=Path, required=True,
                       help="Root directory for experiment outputs")
    parser.add_argument("--experiments", nargs="+",
                       help="Specific experiments to run (optional)")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline comparison")
    parser.add_argument("--python-cmd", default="python",
                       help="Python command to use for training")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create experiment runner
    runner = ExperimentRunner(
        output_root=args.output_root,
        python_cmd=args.python_cmd
    )
    
    try:
        # Run experiment suite
        results = runner.run_experiment_suite(
            config_dir=args.config_dir,
            experiments=args.experiments,
            skip_baseline=args.skip_baseline
        )
        
        # Exit with non-zero code if any experiments failed
        failed_count = len([r for r in results if r['status'] == 'failed'])
        sys.exit(1 if failed_count > 0 else 0)
        
    except KeyboardInterrupt:
        logger.info("Experiment suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Experiment suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()