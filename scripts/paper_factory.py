#!/usr/bin/env python3
"""
BEM Paper Factory - Complete Automated Paper Generation System
Master orchestration script for NeurIPS 2025 submission with statistical rigor.
"""

import argparse
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('paper_factory.log')
    ]
)
logger = logging.getLogger(__name__)


class PaperFactory:
    """Complete automated paper generation system."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.scripts_dir = self.root_dir / "scripts"
        self.start_time = time.time()
        
        # Pipeline stages
        self.stages = [
            ("experiments", "Run experimental validation", self._run_experiments),
            ("statistics", "Perform statistical analysis", self._run_statistics), 
            ("assembly", "Assemble paper components", self._assemble_paper),
            ("validation", "Validate reproducibility", self._validate_reproducibility),
            ("packaging", "Create submission package", self._create_package)
        ]
    
    def _log_stage_start(self, stage_name: str, description: str):
        """Log the start of a pipeline stage."""
        logger.info("=" * 80)
        logger.info(f"STAGE: {stage_name.upper()} - {description}")
        logger.info("=" * 80)
    
    def _run_command(self, cmd: List[str], description: str, 
                    timeout: Optional[int] = None, cwd: Optional[Path] = None) -> bool:
        """Run a command with proper logging and error handling."""
        logger.info(f"Running: {description}")
        logger.debug(f"Command: {' '.join(map(str, cmd))}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or self.root_dir
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} - SUCCESS")
                if result.stdout.strip():
                    logger.debug(f"Output: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ {description} - FAILED")
                logger.error(f"Return code: {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} - TIMEOUT after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"❌ {description} - EXCEPTION: {e}")
            return False
    
    def _run_experiments(self, **kwargs) -> bool:
        """Stage 1: Run all experiments with multiple seeds."""
        self._log_stage_start("experiments", "Running experimental validation")
        
        if kwargs.get('skip_experiments', False):
            logger.info("Skipping experiments as requested")
            return True
        
        # Check if experimental results already exist
        logs_dir = self.root_dir / "logs"
        if logs_dir.exists():
            experiment_files = list(logs_dir.glob("*_seed_*"))
            if len(experiment_files) >= 25:  # 5 methods × 5 seeds minimum
                logger.info(f"Found {len(experiment_files)} existing experiment files")
                if not kwargs.get('force_rerun', False):
                    logger.info("Skipping experiments - use --force-rerun to override")
                    return True
        
        # Run batch experiments
        cmd = [
            'python', str(self.scripts_dir / 'run_batch_experiments.py'),
            '--config-dir', str(self.root_dir / 'experiments'),
            '--output-dir', str(logs_dir),
            '--num-seeds', str(kwargs.get('num_seeds', 5)),
            '--parallel-jobs', str(kwargs.get('parallel_jobs', 4))
        ]
        
        # This is a long-running process - set appropriate timeout
        timeout = kwargs.get('experiment_timeout', 14400)  # 4 hours default
        
        return self._run_command(cmd, "Batch experiments", timeout=timeout)
    
    def _run_statistics(self, **kwargs) -> bool:
        """Stage 2: Perform complete statistical analysis."""
        self._log_stage_start("statistics", "Performing statistical analysis")
        
        cmd = [
            'python', str(self.scripts_dir / 'run_statistical_pipeline.py'),
            '--experiments-dir', str(self.root_dir / 'experiments'),
            '--logs-dir', str(self.root_dir / 'logs'),
            '--output-dir', str(self.root_dir / 'analysis' / 'results'),
            '--claims-file', str(self.root_dir / 'paper' / 'claims.yaml'),
            '--bootstrap-samples', str(kwargs.get('bootstrap_samples', 10000)),
            '--confidence-level', str(kwargs.get('confidence_level', 0.95))
        ]
        
        return self._run_command(cmd, "Statistical analysis pipeline", timeout=1800)  # 30 min
    
    def _assemble_paper(self, **kwargs) -> bool:
        """Stage 3: Assemble complete paper."""
        self._log_stage_start("assembly", "Assembling paper components")
        
        cmd = [
            'python', str(self.scripts_dir / 'assemble_paper.py'),
            '--root-dir', str(self.root_dir),
            '--skip-experiments'  # We already ran experiments
        ]
        
        return self._run_command(cmd, "Paper assembly", timeout=900)  # 15 min
    
    def _validate_reproducibility(self, **kwargs) -> bool:
        """Stage 4: Validate reproducibility and quality."""
        self._log_stage_start("validation", "Validating reproducibility")
        
        cmd = [
            'python', str(self.scripts_dir / 'validate_reproducibility.py'),
            '--root-dir', str(self.root_dir),
            '--validation-only'
        ]
        
        return self._run_command(cmd, "Reproducibility validation", timeout=300)  # 5 min
    
    def _create_package(self, **kwargs) -> bool:
        """Stage 5: Create final submission package."""
        self._log_stage_start("packaging", "Creating submission package")
        
        cmd = [
            'python', str(self.scripts_dir / 'validate_reproducibility.py'),
            '--root-dir', str(self.root_dir),
            '--create-package'
        ]
        
        return self._run_command(cmd, "Package creation", timeout=600)  # 10 min
    
    def run_complete_pipeline(self, **kwargs) -> Dict[str, bool]:
        """Run the complete paper factory pipeline."""
        logger.info("🏭 Starting BEM Paper Factory Pipeline")
        logger.info(f"Root directory: {self.root_dir}")
        logger.info(f"Started at: {datetime.now().isoformat()}")
        
        results = {}
        
        # Run each stage
        for stage_id, description, stage_func in self.stages:
            stage_start = time.time()
            
            try:
                success = stage_func(**kwargs)
                results[stage_id] = success
                
                elapsed = time.time() - stage_start
                status = "✅ SUCCESS" if success else "❌ FAILED"
                logger.info(f"{status} - {stage_id} completed in {elapsed:.1f}s")
                
                if not success and not kwargs.get('continue_on_failure', False):
                    logger.error(f"Pipeline failed at stage: {stage_id}")
                    break
                    
            except Exception as e:
                logger.error(f"Exception in stage {stage_id}: {e}")
                results[stage_id] = False
                
                if not kwargs.get('continue_on_failure', False):
                    break
        
        # Final report
        total_elapsed = time.time() - self.start_time
        self._generate_final_report(results, total_elapsed)
        
        return results
    
    def _generate_final_report(self, results: Dict[str, bool], total_elapsed: float):
        """Generate final pipeline report."""
        logger.info("=" * 80)
        logger.info("🏁 PAPER FACTORY PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        # Stage results
        for stage_id, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"  {stage_id.upper()}: {status}")
        
        logger.info(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        
        # Check for key outputs
        key_outputs = [
            (self.root_dir / "paper" / "main.pdf", "Paper PDF"),
            (self.root_dir / "analysis" / "results" / "claims_validation.json", "Statistical validation"),
            (self.root_dir / "validation_report.json", "Reproducibility report")
        ]
        
        logger.info("\n📋 Key outputs:")
        for output_path, description in key_outputs:
            exists = "✅" if output_path.exists() else "❌"
            logger.info(f"  {exists} {description}: {output_path}")
        
        # Overall success assessment
        all_critical_success = all(results.get(stage) for stage in ['statistics', 'assembly', 'validation'])
        
        if all_critical_success:
            logger.info("\n🎉 PIPELINE SUCCESSFUL - Paper ready for review!")
        else:
            logger.warning("\n⚠️  PIPELINE HAD FAILURES - Review required before submission")
        
        # Next steps
        logger.info("\n📝 Next steps:")
        if all_critical_success:
            logger.info("  1. Review generated paper in paper/main.pdf")
            logger.info("  2. Check validation report for any warnings")
            logger.info("  3. Submit to NeurIPS 2025!")
        else:
            logger.info("  1. Check logs for specific failure details")
            logger.info("  2. Address any failed stages")
            logger.info("  3. Re-run pipeline with --continue-on-failure if needed")
    
    def run_specific_stage(self, stage_name: str, **kwargs) -> bool:
        """Run a specific stage of the pipeline."""
        for stage_id, description, stage_func in self.stages:
            if stage_id == stage_name:
                logger.info(f"Running specific stage: {stage_name}")
                return stage_func(**kwargs)
        
        logger.error(f"Unknown stage: {stage_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='BEM Paper Factory - Complete Automated Paper Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python paper_factory.py
  
  # Skip experiments, use existing logs
  python paper_factory.py --skip-experiments
  
  # Run specific stage only
  python paper_factory.py --stage statistics
  
  # Force re-run experiments
  python paper_factory.py --force-rerun --experiment-timeout 7200
        """
    )
    
    # Pipeline control
    parser.add_argument('--root-dir', type=Path, default='.',
                       help='Root directory of the project')
    parser.add_argument('--stage', type=str,
                       choices=['experiments', 'statistics', 'assembly', 'validation', 'packaging'],
                       help='Run specific stage only')
    parser.add_argument('--continue-on-failure', action='store_true',
                       help='Continue pipeline even if stages fail')
    
    # Experiment control
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip experimental runs and use existing results')
    parser.add_argument('--force-rerun', action='store_true', 
                       help='Force re-run experiments even if results exist')
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of random seeds per experiment')
    parser.add_argument('--parallel-jobs', type=int, default=4,
                       help='Number of parallel experiment jobs')
    parser.add_argument('--experiment-timeout', type=int, default=14400,
                       help='Timeout for experiments in seconds (default: 4 hours)')
    
    # Statistical analysis control
    parser.add_argument('--bootstrap-samples', type=int, default=10000,
                       help='Number of bootstrap samples')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                       help='Confidence level for intervals')
    
    # Logging control
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Create factory and run
    factory = PaperFactory(args.root_dir)
    
    if args.stage:
        # Run specific stage
        success = factory.run_specific_stage(args.stage, **vars(args))
        return 0 if success else 1
    else:
        # Run complete pipeline
        results = factory.run_complete_pipeline(**vars(args))
        
        # Return success if all critical stages passed
        critical_stages = ['statistics', 'assembly', 'validation']
        all_critical_passed = all(results.get(stage, False) for stage in critical_stages)
        return 0 if all_critical_passed else 1


if __name__ == '__main__':
    exit(main())