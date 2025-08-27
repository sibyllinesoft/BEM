#!/usr/bin/env python3
"""
v1.3-Stack Ablation Campaign Runner

This script orchestrates the execution of Phase 3 ablation studies for the v1.3-stack
validation campaign. It runs all three critical ablations (A1, A2, A3) with proper
experimental controls and statistical rigor.

Ablation Studies:
- A1: S1 without diagonal head (tests F5.2 contribution)
- A2: S1 without hard negatives (tests F5.5 contribution) 
- A3: S1 with FP16 instead of FP8 (tests F5.4 efficiency contribution)

Statistical Protocol:
- 5 seeds per ablation for statistical power
- Paired comparison against S1 full stack
- BCa bootstrap with 10k iterations
- FDR correction across ablations
- Quality gate validation for each variant

Author: Claude Code (Experiment Orchestrator)
Created: 2024-08-23
Campaign: v1.3-Stack Phase 3 Ablation Validation
"""

import os
import sys
import json
import logging
import subprocess
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.ablation_analysis import AblationAnalyzer, print_ablation_report, save_ablation_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ablation_campaign.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AblationCampaignRunner:
    """Orchestrates v1.3-stack ablation campaign execution."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.experiments_dir = project_root / 'experiments'
        self.logs_dir = project_root / 'logs'
        self.scripts_dir = project_root / 'scripts'
        
        # Ensure directories exist
        self.logs_dir.mkdir(exist_ok=True)
        
        # Ablation configuration
        self.ablation_configs = {
            'A1_no_diagonal': {
                'config_file': 'A1_no_diagonal.yml',
                'description': 'S1 without F5.2 diagonal head',
                'ingredient': 'F5.2 Low-Rank + Diagonal',
                'expected_impact': 'Quality degradation (-1 to -2%)'
            },
            'A2_no_hard_negatives': {
                'config_file': 'A2_no_hard_negatives.yml', 
                'description': 'S1 without F5.5 hard negatives',
                'ingredient': 'F5.5 Hard Negatives Training',
                'expected_impact': 'Quality degradation (-1 to -2%)'
            },
            'A3_fp16_instead_fp8': {
                'config_file': 'A3_fp16_instead_fp8.yml',
                'description': 'S1 with FP16 instead of F5.4 FP8',
                'ingredient': 'F5.4 FP8 Quantization-Aware Training', 
                'expected_impact': 'Efficiency degradation (latency/memory)'
            }
        }
        
        # Reference full stack
        self.full_stack_config = 'S1_stack.yml'
        
    def validate_prerequisites(self) -> bool:
        """Validate that all prerequisites are in place."""
        logger.info("🔍 Validating prerequisites...")
        
        # Check S1 full stack results exist
        s1_results = self.logs_dir / 'S1_stack' / 'eval.json'
        if not s1_results.exists():
            logger.error(f"❌ S1 full stack results not found: {s1_results}")
            logger.info("   Run S1 stack experiment first: python train_fast5.py --config experiments/S1_stack.yml")
            return False
        logger.info("✅ S1 full stack results found")
        
        # Check ablation configs exist
        for ablation_id, config_info in self.ablation_configs.items():
            config_path = self.experiments_dir / config_info['config_file']
            if not config_path.exists():
                logger.error(f"❌ Ablation config not found: {config_path}")
                return False
        logger.info("✅ All ablation configs found")
        
        # Check training script exists
        train_script = self.project_root / 'train_fast5.py'
        if not train_script.exists():
            train_script = self.project_root / 'train.py'
            if not train_script.exists():
                logger.error("❌ Training script not found (train_fast5.py or train.py)")
                return False
        logger.info("✅ Training script found")
        
        # Check data files exist
        data_files = ['data/train.jsonl', 'data/val.jsonl', 'data/hard_negs.jsonl']
        for data_file in data_files:
            if not (self.project_root / data_file).exists():
                logger.error(f"❌ Data file not found: {data_file}")
                return False
        logger.info("✅ All data files found")
        
        return True
        
    def run_single_ablation(
        self, 
        ablation_id: str, 
        config_info: Dict[str, Any],
        resume: bool = False
    ) -> bool:
        """Run single ablation experiment with all seeds."""
        
        config_path = self.experiments_dir / config_info['config_file']
        logs_path = self.logs_dir / ablation_id
        
        logger.info(f"🧪 Starting {ablation_id}:")
        logger.info(f"   Description: {config_info['description']}")
        logger.info(f"   Ingredient: {config_info['ingredient']}")
        logger.info(f"   Expected: {config_info['expected_impact']}")
        logger.info(f"   Config: {config_path}")
        logger.info(f"   Logs: {logs_path}")
        
        # Create logs directory
        logs_path.mkdir(exist_ok=True)
        
        # Check if already completed
        if resume and (logs_path / 'eval.json').exists():
            logger.info(f"✅ {ablation_id} already completed, skipping")
            return True
            
        if self.dry_run:
            logger.info(f"🔥 DRY RUN: Would run {ablation_id}")
            return True
            
        # Run training experiment
        train_script = self.project_root / 'train_fast5.py'
        if not train_script.exists():
            train_script = self.project_root / 'train.py'
            
        cmd = [
            'python3', str(train_script),
            '--config', str(config_path),
            '--output_dir', str(logs_path),
            '--no_wandb_log',  # Prevent W&B spam during campaign
            '--quiet'  # Reduce verbose output
        ]
        
        logger.info(f"   Executing: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per ablation
            )
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {ablation_id} completed successfully ({duration:.1f}s)")
                return True
            else:
                logger.error(f"❌ {ablation_id} failed (returncode: {result.returncode})")
                logger.error(f"   STDOUT: {result.stdout[-1000:]}")  # Last 1000 chars
                logger.error(f"   STDERR: {result.stderr[-1000:]}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {ablation_id} timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"❌ {ablation_id} failed with exception: {e}")
            return False
            
    def run_all_ablations(self, resume: bool = False, parallel: bool = False) -> Dict[str, bool]:
        """Run all ablation experiments."""
        
        logger.info("🚀 Starting v1.3-Stack Ablation Campaign")
        logger.info(f"   Total ablations: {len(self.ablation_configs)}")
        logger.info(f"   Parallel execution: {parallel}")
        logger.info(f"   Resume mode: {resume}")
        
        results = {}
        
        if parallel:
            # TODO: Implement parallel execution if needed
            logger.warning("⚠️ Parallel execution not yet implemented, running sequentially")
            
        # Sequential execution
        for ablation_id, config_info in self.ablation_configs.items():
            success = self.run_single_ablation(ablation_id, config_info, resume=resume)
            results[ablation_id] = success
            
            if not success:
                logger.error(f"❌ Ablation campaign failed at {ablation_id}")
                break
                
            # Brief pause between experiments
            if not self.dry_run and len(self.ablation_configs) > 1:
                logger.info("   Cooling down for 30 seconds...")
                time.sleep(30)
                
        return results
        
    def analyze_campaign_results(self) -> Optional[Any]:
        """Analyze all ablation results using statistical framework."""
        
        logger.info("📊 Analyzing ablation campaign results...")
        
        # Check that all results exist
        s1_results_path = self.logs_dir / 'S1_stack' / 'eval.json'
        if not s1_results_path.exists():
            logger.error("❌ S1 full stack results not found for comparison")
            return None
            
        ablation_results_paths = {}
        missing_results = []
        
        for ablation_id in self.ablation_configs.keys():
            results_path = self.logs_dir / ablation_id / 'eval.json'
            if results_path.exists():
                ablation_results_paths[ablation_id] = str(results_path)
            else:
                missing_results.append(ablation_id)
                
        if missing_results:
            logger.error(f"❌ Missing results for: {missing_results}")
            return None
            
        # Set up analysis
        analyzer = AblationAnalyzer(n_bootstrap=10000, alpha=0.05, min_effect_size=0.01)
        
        campaign_config = {
            'campaign_id': 'v1.3-stack-phase3-ablations',
            'full_stack_config': 'S1_stack',
            'ablations': {
                ablation_id: {
                    'ablation_type': f"F5.X_{ablation_id.split('_', 1)[1]}_removal",
                    'ingredient_removed': config_info['ingredient'],
                    'hypothesis': config_info['expected_impact']
                }
                for ablation_id, config_info in self.ablation_configs.items()
            }
        }
        
        try:
            # Run statistical analysis
            campaign_result = analyzer.analyze_ablation_campaign(
                full_stack_results_path=str(s1_results_path),
                ablation_results_paths=ablation_results_paths,
                campaign_config=campaign_config
            )
            
            # Print results
            print_ablation_report(campaign_result)
            
            # Save detailed results
            results_file = self.project_root / 'analysis' / 'ablation_campaign_results.json'
            results_file.parent.mkdir(exist_ok=True)
            save_ablation_report(campaign_result, str(results_file))
            
            logger.info(f"✅ Ablation analysis complete: {results_file}")
            return campaign_result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return None
            
    def generate_campaign_summary(self, execution_results: Dict[str, bool], analysis_result: Any = None):
        """Generate comprehensive campaign summary."""
        
        summary = {
            'campaign_metadata': {
                'campaign_id': 'v1.3-stack-phase3-ablations',
                'timestamp': datetime.now().isoformat(),
                'total_ablations': len(self.ablation_configs),
                'dry_run': self.dry_run
            },
            'execution_results': execution_results,
            'execution_summary': {
                'successful_ablations': sum(1 for success in execution_results.values() if success),
                'failed_ablations': sum(1 for success in execution_results.values() if not success),
                'execution_success': all(execution_results.values())
            }
        }
        
        if analysis_result:
            summary['analysis_available'] = True
            summary['promotion_recommendation'] = analysis_result.promotion_recommendation
        else:
            summary['analysis_available'] = False
            
        # Save summary
        summary_file = self.project_root / 'analysis' / 'ablation_campaign_summary.json'
        summary_file.parent.mkdir(exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"📋 Campaign summary saved: {summary_file}")
        return summary

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='v1.3-Stack Ablation Campaign Runner')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no actual execution)')
    parser.add_argument('--resume', action='store_true', help='Resume mode (skip completed experiments)')
    parser.add_argument('--parallel', action='store_true', help='Run ablations in parallel (experimental)')
    parser.add_argument('--analyze-only', action='store_true', help='Skip execution, only run analysis')
    parser.add_argument('--project-root', type=str, default='.', help='Project root directory')
    
    args = parser.parse_args()
    
    # Set up campaign runner
    project_root = Path(args.project_root).resolve()
    runner = AblationCampaignRunner(project_root, dry_run=args.dry_run)
    
    logger.info("🔬 v1.3-STACK ABLATION CAMPAIGN RUNNER")
    logger.info("="*50)
    logger.info(f"Project Root: {project_root}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Analyze Only: {args.analyze_only}")
    
    # Validate prerequisites
    if not args.analyze_only and not runner.validate_prerequisites():
        logger.error("❌ Prerequisites validation failed")
        sys.exit(1)
        
    # Execute ablations
    execution_results = {}
    if not args.analyze_only:
        execution_results = runner.run_all_ablations(resume=args.resume, parallel=args.parallel)
        
        # Check execution success
        if not all(execution_results.values()):
            logger.error("❌ Ablation campaign execution failed")
            failed_ablations = [aid for aid, success in execution_results.items() if not success]
            logger.error(f"   Failed ablations: {failed_ablations}")
        else:
            logger.info("✅ All ablations executed successfully")
    else:
        logger.info("⏭️ Skipping execution (analyze-only mode)")
        
    # Analyze results
    analysis_result = None
    if not args.dry_run:
        analysis_result = runner.analyze_campaign_results()
        
    # Generate campaign summary
    campaign_summary = runner.generate_campaign_summary(execution_results, analysis_result)
    
    # Final status
    if args.dry_run:
        logger.info("🔥 DRY RUN COMPLETE - No actual experiments executed")
    elif args.analyze_only:
        if analysis_result:
            logger.info("📊 ANALYSIS COMPLETE - Statistical validation finished")
        else:
            logger.error("❌ ANALYSIS FAILED - Check logs for details")
    elif all(execution_results.values()) and analysis_result:
        logger.info("🎯 CAMPAIGN SUCCESS - All ablations completed and analyzed")
        
        if analysis_result.all_ingredients_causal:
            logger.info("✅ VALIDATION SUCCESSFUL - All ingredients show causal contribution")
        else:
            logger.warning("⚠️ VALIDATION PARTIAL - Some ingredients lack clear causal evidence")
            
    else:
        logger.error("❌ CAMPAIGN INCOMPLETE - Check logs and rerun with --resume")
        sys.exit(1)

if __name__ == "__main__":
    main()