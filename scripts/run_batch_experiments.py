#!/usr/bin/env python3
"""
BEM Paper Factory - Batch Experiment Runner
Executes complete experimental pipeline with multiple seeds for statistical rigor.

Features:
- Runs all baseline and BEM configurations
- Executes multiple seeds per configuration (≥5 for statistical validity)
- Runs special evaluations (index-swap, canary tests)
- Comprehensive logging and progress tracking
- Automatic retry on failures
- Results aggregation for statistical analysis
"""

import argparse
import concurrent.futures
import json
import logging
import subprocess
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import threading
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentJob:
    """Individual experiment job specification."""
    config_file: str
    experiment_id: str
    method_type: str
    approach: str
    seed: int
    output_dir: str
    priority: int = 1  # Higher = more important
    max_retries: int = 2
    timeout_minutes: int = 60
    
@dataclass 
class JobResult:
    """Result of running an experiment job."""
    job: ExperimentJob
    success: bool
    duration_seconds: float
    output_files: List[str]
    error_message: Optional[str] = None
    retry_count: int = 0

class BatchExperimentRunner:
    """
    Orchestrates batch execution of all experiments with multiple seeds.
    """
    
    def __init__(self, 
                 experiments_dir: str = "experiments",
                 output_base_dir: str = "logs",
                 num_seeds: int = 5,
                 max_parallel: int = 4):
        
        self.experiments_dir = Path(experiments_dir)
        self.output_base_dir = Path(output_base_dir)
        self.num_seeds = num_seeds
        self.max_parallel = max_parallel
        
        # Create output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Job tracking
        self.all_jobs: List[ExperimentJob] = []
        self.completed_jobs: List[JobResult] = []
        self.failed_jobs: List[JobResult] = []
        
        # Thread safety
        self.results_lock = threading.Lock()
        
        # Progress tracking
        self.start_time = None
        self.total_jobs = 0
    
    def discover_experiment_configs(self) -> List[str]:
        """Discover all experiment configuration files."""
        if not self.experiments_dir.exists():
            raise FileNotFoundError(f"Experiments directory not found: {self.experiments_dir}")
        
        config_files = list(self.experiments_dir.glob("*.yaml")) + list(self.experiments_dir.glob("*.yml"))
        
        if not config_files:
            raise FileNotFoundError(f"No experiment configs found in {self.experiments_dir}")
        
        logger.info(f"Discovered {len(config_files)} experiment configurations")
        return [str(f) for f in config_files]
    
    def create_experiment_jobs(self, config_files: List[str]) -> List[ExperimentJob]:
        """Create experiment jobs for all configs and seeds."""
        jobs = []
        
        for config_file in config_files:
            # Load config to get metadata
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                experiment_id = config.get('experiment_id', Path(config_file).stem)
                method_type = config.get('method_type', 'unknown')
                approach = config.get('approach', 'unknown')
                
                # Set priority based on method type
                priority = 1
                if method_type == 'bem':
                    priority = 3  # BEM experiments are higher priority
                elif approach in ['mole', 'hyperlora']:
                    priority = 2  # Advanced baselines
                
                # Create jobs for multiple seeds
                for seed in range(1, self.num_seeds + 1):
                    output_dir = self.output_base_dir / f"{experiment_id}_seed{seed}"
                    
                    job = ExperimentJob(
                        config_file=config_file,
                        experiment_id=experiment_id,
                        method_type=method_type,
                        approach=approach,
                        seed=seed,
                        output_dir=str(output_dir),
                        priority=priority,
                        timeout_minutes=90 if method_type == 'bem' else 60  # BEM takes longer
                    )
                    jobs.append(job)
                    
            except Exception as e:
                logger.error(f"Error processing config {config_file}: {e}")
        
        # Sort by priority (higher priority first)
        jobs.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Created {len(jobs)} experiment jobs ({self.num_seeds} seeds per config)")
        return jobs
    
    def run_single_experiment(self, job: ExperimentJob) -> JobResult:
        """Run a single experiment job."""
        logger.info(f"Starting job: {job.experiment_id} (seed {job.seed})")
        
        start_time = time.time()
        
        try:
            # Prepare command
            cmd = [
                'python', 'scripts/train_experiment.py',
                '--config', job.config_file,
                '--seed', str(job.seed),
                '--output-dir', job.output_dir
            ]
            
            # Run experiment with timeout
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,  # Run from project root
                capture_output=True,
                text=True,
                timeout=job.timeout_minutes * 60
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Success - collect output files
                output_files = self._collect_output_files(Path(job.output_dir))
                
                job_result = JobResult(
                    job=job,
                    success=True,
                    duration_seconds=duration,
                    output_files=output_files
                )
                
                logger.info(f"✅ Completed: {job.experiment_id} (seed {job.seed}) "
                           f"in {duration:.1f}s")
                return job_result
            else:
                # Failure
                error_msg = f"Exit code {result.returncode}: {result.stderr[:500]}"
                
                job_result = JobResult(
                    job=job,
                    success=False,
                    duration_seconds=duration,
                    output_files=[],
                    error_message=error_msg
                )
                
                logger.error(f"❌ Failed: {job.experiment_id} (seed {job.seed}): {error_msg}")
                return job_result
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Timeout after {job.timeout_minutes} minutes"
            
            job_result = JobResult(
                job=job,
                success=False,
                duration_seconds=duration,
                output_files=[],
                error_message=error_msg
            )
            
            logger.error(f"⏰ Timeout: {job.experiment_id} (seed {job.seed})")
            return job_result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Exception: {str(e)}"
            
            job_result = JobResult(
                job=job,
                success=False,
                duration_seconds=duration,
                output_files=[],
                error_message=error_msg
            )
            
            logger.error(f"💥 Error: {job.experiment_id} (seed {job.seed}): {e}")
            return job_result
    
    def _collect_output_files(self, output_dir: Path) -> List[str]:
        """Collect all output files from experiment."""
        if not output_dir.exists():
            return []
        
        output_files = []
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                output_files.append(str(file_path))
        
        return output_files
    
    def retry_failed_job(self, failed_result: JobResult) -> Optional[JobResult]:
        """Retry a failed job if retries remaining."""
        job = failed_result.job
        
        if failed_result.retry_count >= job.max_retries:
            logger.warning(f"Max retries exceeded for {job.experiment_id} (seed {job.seed})")
            return failed_result
        
        logger.info(f"Retrying {job.experiment_id} (seed {job.seed}) - "
                   f"attempt {failed_result.retry_count + 1}")
        
        # Run again
        retry_result = self.run_single_experiment(job)
        retry_result.retry_count = failed_result.retry_count + 1
        
        return retry_result
    
    def run_batch_experiments(self, jobs: List[ExperimentJob]) -> Tuple[List[JobResult], List[JobResult]]:
        """Run all experiments in parallel with proper resource management."""
        self.start_time = time.time()
        self.total_jobs = len(jobs)
        
        logger.info(f"🚀 Starting batch execution of {self.total_jobs} jobs "
                   f"with max parallelism: {self.max_parallel}")
        
        completed = []
        failed = []
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(self.run_single_experiment, job): job 
                            for job in jobs}
            
            # Process completed jobs
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    result = future.result()
                    
                    with self.results_lock:
                        if result.success:
                            completed.append(result)
                        else:
                            # Try to retry if possible
                            retry_result = self.retry_failed_job(result)
                            
                            if retry_result.success:
                                completed.append(retry_result)
                            else:
                                failed.append(retry_result)
                        
                        # Progress update
                        total_processed = len(completed) + len(failed)
                        elapsed = time.time() - self.start_time
                        eta = elapsed * (self.total_jobs / total_processed - 1) if total_processed > 0 else 0
                        
                        logger.info(f"Progress: {total_processed}/{self.total_jobs} "
                                   f"({100*total_processed/self.total_jobs:.1f}%) "
                                   f"- ETA: {eta/60:.1f}m")
                
                except Exception as e:
                    logger.error(f"Error processing job result: {e}")
                    failed.append(JobResult(
                        job=job,
                        success=False,
                        duration_seconds=0,
                        output_files=[],
                        error_message=f"Processing error: {str(e)}"
                    ))
        
        self.completed_jobs = completed
        self.failed_jobs = failed
        
        total_time = time.time() - self.start_time
        logger.info(f"🎯 Batch execution completed in {total_time/60:.1f} minutes")
        logger.info(f"✅ Success: {len(completed)}, ❌ Failed: {len(failed)}")
        
        return completed, failed
    
    def run_special_evaluations(self, completed_jobs: List[JobResult]) -> Dict[str, Any]:
        """Run special evaluations on successful BEM experiments."""
        logger.info("Starting special evaluations...")
        
        special_eval_results = {}
        
        # Find BEM experiments to evaluate
        bem_jobs = [job for job in completed_jobs 
                   if job.job.method_type == 'bem' and 'p3' in job.job.approach.lower()]
        
        if not bem_jobs:
            logger.warning("No BEM P3+ experiments found for special evaluation")
            return special_eval_results
        
        # Run special evaluations on a representative BEM model
        representative_job = bem_jobs[0]  # Use first successful BEM job
        model_path = Path(representative_job.job.output_dir) / "model"  # Would be actual model path
        
        # Create special evaluation config
        special_config = {
            'test_dataset': 'validation_set',
            'canary_tasks': [
                {'name': 'arithmetic', 'baseline_performance': 0.95, 'interference_threshold': 0.02},
                {'name': 'copy_task', 'baseline_performance': 0.99, 'interference_threshold': 0.01},
                {'name': 'simple_qa', 'baseline_performance': 0.90, 'interference_threshold': 0.02}
            ],
            'latency_configurations': ['static_lora', 'bem_p3_retrieval', 'bem_p4_composition'],
            'test_prompts': ['test prompt ' + str(i) for i in range(20)]
        }
        
        # Save config
        special_config_file = self.output_base_dir / "special_eval_config.yaml"
        with open(special_config_file, 'w') as f:
            yaml.dump(special_config, f)
        
        # Run special evaluations
        try:
            special_output_dir = self.output_base_dir / "special_evaluations"
            
            cmd = [
                'python', 'scripts/run_special_evaluations.py',
                '--model-path', str(model_path),
                '--config', str(special_config_file),
                '--output-dir', str(special_output_dir),
                '--tests', 'all'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes
            )
            
            if result.returncode == 0:
                logger.info("✅ Special evaluations completed successfully")
                
                # Load results
                try:
                    summary_file = special_output_dir / "special_evaluations_summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            special_eval_results = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load special evaluation results: {e}")
            else:
                logger.error(f"Special evaluations failed: {result.stderr[:500]}")
                
        except subprocess.TimeoutExpired:
            logger.error("Special evaluations timed out")
        except Exception as e:
            logger.error(f"Error running special evaluations: {e}")
        
        return special_eval_results
    
    def generate_batch_report(self, 
                            completed_jobs: List[JobResult],
                            failed_jobs: List[JobResult],
                            special_eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive batch execution report."""
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Group results by experiment type
        results_by_experiment = {}
        for job in completed_jobs:
            exp_id = job.job.experiment_id
            if exp_id not in results_by_experiment:
                results_by_experiment[exp_id] = {
                    'experiment_id': exp_id,
                    'method_type': job.job.method_type,
                    'approach': job.job.approach,
                    'successful_seeds': [],
                    'total_seeds': 0,
                    'avg_duration': 0,
                    'output_directories': []
                }
            
            results_by_experiment[exp_id]['successful_seeds'].append(job.job.seed)
            results_by_experiment[exp_id]['output_directories'].append(job.job.output_dir)
        
        # Count total seeds and calculate stats
        for exp_id in results_by_experiment:
            exp_data = results_by_experiment[exp_id]
            exp_data['total_seeds'] = len(exp_data['successful_seeds'])
            
            # Calculate average duration
            exp_jobs = [job for job in completed_jobs if job.job.experiment_id == exp_id]
            exp_data['avg_duration'] = sum(job.duration_seconds for job in exp_jobs) / len(exp_jobs)
        
        # Generate report
        report = {
            'batch_execution_summary': {
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                                          time.localtime(self.start_time)) if self.start_time else None,
                'total_duration_minutes': total_time / 60,
                'total_jobs': self.total_jobs,
                'successful_jobs': len(completed_jobs),
                'failed_jobs': len(failed_jobs),
                'success_rate': len(completed_jobs) / self.total_jobs if self.total_jobs > 0 else 0,
                'target_seeds_per_experiment': self.num_seeds,
                'max_parallel_jobs': self.max_parallel
            },
            'experiment_results': results_by_experiment,
            'failed_experiments': [
                {
                    'experiment_id': job.job.experiment_id,
                    'seed': job.job.seed,
                    'error_message': job.error_message,
                    'retry_count': job.retry_count
                }
                for job in failed_jobs
            ],
            'special_evaluations': special_eval_results,
            'statistical_readiness': {
                'experiments_with_sufficient_seeds': sum(
                    1 for exp in results_by_experiment.values() 
                    if exp['total_seeds'] >= 5
                ),
                'total_experiments': len(results_by_experiment),
                'ready_for_statistical_analysis': all(
                    exp['total_seeds'] >= 5 
                    for exp in results_by_experiment.values()
                )
            }
        }
        
        return report
    
    def save_batch_report(self, report: Dict[str, Any]) -> None:
        """Save comprehensive batch report."""
        report_file = self.output_base_dir / "batch_execution_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Batch report saved to: {report_file}")
        
        # Print summary
        summary = report['batch_execution_summary']
        logger.info(f"📊 BATCH EXECUTION SUMMARY:")
        logger.info(f"  Total time: {summary['total_duration_minutes']:.1f} minutes")
        logger.info(f"  Success rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"  Jobs: {summary['successful_jobs']}/{summary['total_jobs']}")
        
        readiness = report['statistical_readiness']
        if readiness['ready_for_statistical_analysis']:
            logger.info(f"✅ Ready for statistical analysis!")
        else:
            logger.warning(f"⚠️  {readiness['experiments_with_sufficient_seeds']}/{readiness['total_experiments']} "
                         f"experiments have sufficient seeds")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete experimental pipeline."""
        logger.info("🚀 Starting complete experimental pipeline...")
        
        # Discover experiments
        config_files = self.discover_experiment_configs()
        
        # Create jobs
        self.all_jobs = self.create_experiment_jobs(config_files)
        
        # Run batch experiments
        completed, failed = self.run_batch_experiments(self.all_jobs)
        
        # Run special evaluations
        special_results = self.run_special_evaluations(completed)
        
        # Generate and save report
        report = self.generate_batch_report(completed, failed, special_results)
        self.save_batch_report(report)
        
        logger.info("🎉 Complete experimental pipeline finished!")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='BEM Paper Factory - Batch Experiment Runner')
    parser.add_argument('--experiments-dir', default='experiments', 
                       help='Directory containing experiment configs')
    parser.add_argument('--output-dir', default='logs', 
                       help='Base output directory')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of seeds per experiment (minimum 5 for statistics)')
    parser.add_argument('--max-parallel', type=int, default=4,
                       help='Maximum parallel experiments')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running experiments')
    
    args = parser.parse_args()
    
    if args.seeds < 5:
        logger.warning("Recommend at least 5 seeds for statistical validity")
    
    # Initialize batch runner
    runner = BatchExperimentRunner(
        experiments_dir=args.experiments_dir,
        output_base_dir=args.output_dir,
        num_seeds=args.seeds,
        max_parallel=args.max_parallel
    )
    
    if args.dry_run:
        logger.info("🔍 DRY RUN - Validating configuration...")
        config_files = runner.discover_experiment_configs()
        jobs = runner.create_experiment_jobs(config_files)
        
        logger.info(f"✅ Configuration valid!")
        logger.info(f"  Experiment configs: {len(config_files)}")
        logger.info(f"  Total jobs: {len(jobs)} ({args.seeds} seeds each)")
        logger.info(f"  Estimated duration: {len(jobs) * 45 / args.max_parallel / 60:.1f} hours")
        return
    
    # Run complete pipeline
    try:
        report = runner.run_complete_pipeline()
        
        if report['statistical_readiness']['ready_for_statistical_analysis']:
            logger.info("🎯 All experiments completed successfully - ready for paper generation!")
        else:
            logger.warning("⚠️  Some experiments failed - review results before proceeding")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)

if __name__ == '__main__':
    main()