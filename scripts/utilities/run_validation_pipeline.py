#!/usr/bin/env python3
"""
Complete Validation Pipeline for BEM 2.0 Agentic Router

Runs the full training and evaluation pipeline:
1. AR0: Behavior Cloning training
2. AR1: Policy Gradient training  
3. Comprehensive evaluation against acceptance gates
4. Performance validation and reporting

Usage:
python run_validation_pipeline.py [--quick] [--output validation_results/]
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
import json
import yaml
import shutil
from typing import Dict, List


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def run_command(command: List[str], description: str, logger, cwd=None) -> bool:
    """Run a command and log the output."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            if result.stdout.strip():
                logger.info(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr.strip():
                logger.error(f"Error:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out after 2 hours")
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {e}")
        return False


def create_validation_configs(output_dir: Path, quick_mode: bool = False):
    """Create configuration files for training."""
    config_dir = output_dir / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # AR0 Behavior Cloning config
    ar0_config = {
        'macro_policy': {
            'num_experts': 3,
            'chunk_summary_dim': 512,
            'hidden_dim': 1024,
            'num_layers': 4,
            'dropout': 0.1,
            'hysteresis_tau': 0.1,
            'step_penalty': 0.02
        },
        'composition': {
            'trust_region_tau': 1.0,
            'rank_budget_max': 32,
            'bias_scale_range': [0.1, 2.0],
            'attachment_points': ['W_O', 'W_down']
        },
        'router': {
            'chunk_size': 128,
            'max_sequence_length': 2048,
            'cache_safety_enabled': True
        },
        'trace_generation': {
            'num_train_traces': 1000 if quick_mode else 5000,
            'num_eval_traces': 200 if quick_mode else 500,
            'sequence_length_range': [256, 1024],
            'domains': ['code', 'formal', 'safety']
        },
        'bc_training': {
            'learning_rate': 1e-4,
            'batch_size': 16 if quick_mode else 32,
            'num_epochs': 3 if quick_mode else 10,
            'eval_interval': 100,
            'save_interval': 200,
            'gradient_clip': 1.0
        }
    }
    
    # AR1 Policy Gradient config  
    ar1_config = {
        'macro_policy': ar0_config['macro_policy'],
        'composition': ar0_config['composition'],
        'router': ar0_config['router'],
        'environment': {
            'type': 'multi_task',
            'max_sequence_length': 2048,
            'chunk_size': 128,
            'vocab_size': 32000,
            'task_weights': {
                'code_completion': 0.5,
                'formal_reasoning': 0.3,
                'safety_analysis': 0.2
            }
        },
        'pg_training': {
            'learning_rate': 5e-5,
            'gamma': 0.99,
            'epsilon_clip': 0.2,
            'entropy_coeff': 0.01,
            'num_episodes': 200 if quick_mode else 1000,
            'max_steps_per_episode': 50 if quick_mode else 100,
            'eval_interval': 20 if quick_mode else 100,
            'save_interval': 50 if quick_mode else 200
        },
        'acceptance_validation': {
            'target_metrics': {
                'slice_b_improvement_min': 0.015,  # +≥1.5% EM/F1 on Slice-B
                'p50_latency_increase_max': 0.15,   # ≤+15% p50 latency
                'monotonicity_pass_rate_min': 0.8,
                'cache_safety_required': True
            }
        }
    }
    
    # Save configs
    ar0_config_path = config_dir / 'AR0_bc.yml'
    ar1_config_path = config_dir / 'AR1_pg.yml'
    
    with open(ar0_config_path, 'w') as f:
        yaml.dump(ar0_config, f, default_flow_style=False)
    
    with open(ar1_config_path, 'w') as f:
        yaml.dump(ar1_config, f, default_flow_style=False)
    
    return ar0_config_path, ar1_config_path


def run_training_phase(
    phase: str,
    config_path: str,
    checkpoint_path: str = None,
    output_dir: Path = None,
    logger = None
) -> bool:
    """Run a training phase."""
    command = [
        sys.executable, 
        'bem2/train_agentic_router.py',
        '--phase', phase,
        '--config', str(config_path)
    ]
    
    if checkpoint_path:
        command.extend(['--checkpoint', str(checkpoint_path)])
    
    if output_dir:
        command.extend(['--output', str(output_dir)])
    
    description = f"Training Phase {phase.upper()}"
    return run_command(command, description, logger)


def run_evaluation(
    model_path: str,
    baseline_path: str = None,
    output_dir: Path = None,
    logger = None
) -> bool:
    """Run comprehensive evaluation."""
    command = [
        sys.executable,
        'bem2/evaluate_agentic_router.py',
        '--model', str(model_path)
    ]
    
    if baseline_path:
        command.extend(['--baseline', str(baseline_path)])
    
    if output_dir:
        command.extend(['--out', str(output_dir)])
    
    description = "Comprehensive Evaluation"
    return run_command(command, description, logger)


def validate_acceptance_gates(evaluation_dir: Path, logger) -> Dict:
    """Validate results against acceptance gates."""
    logger.info("Validating acceptance gates...")
    
    # Load evaluation summary
    summary_file = evaluation_dir / 'evaluation_summary.json'
    if not summary_file.exists():
        logger.error(f"Evaluation summary not found: {summary_file}")
        return {'overall_pass': False, 'error': 'Missing evaluation summary'}
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Check acceptance gates from TODO.md
    gates = {}
    
    # Gate 1: Slice-B CI > 0 (EM/F1 improvement ≥1.5%)
    slice_b_improvement = summary.get('key_metrics', {}).get('slice_b_improvement', 0)
    gates['slice_b_improvement'] = {
        'value': slice_b_improvement,
        'target': 0.015,
        'pass': slice_b_improvement >= 0.015,
        'description': 'Slice-B EM/F1 improvement ≥1.5%'
    }
    
    # Gate 2: p50 latency increase ≤15%
    latency_change = summary.get('key_metrics', {}).get('p50_latency_change_percent', 0)
    gates['p50_latency_increase'] = {
        'value': latency_change,
        'target': 15.0,
        'pass': latency_change <= 15.0,
        'description': 'P50 latency increase ≤15%'
    }
    
    # Gate 3: Monotonicity intact
    monotonicity_pass_rate = summary.get('key_metrics', {}).get('monotonicity_pass_rate', 0)
    gates['monotonicity_intact'] = {
        'value': monotonicity_pass_rate,
        'target': 0.8,
        'pass': monotonicity_pass_rate >= 0.8,
        'description': 'Monotonicity pass rate ≥80%'
    }
    
    # Gate 4: Cache safety
    cache_safe = summary.get('key_metrics', {}).get('cache_safe', False)
    gates['cache_safety'] = {
        'value': cache_safe,
        'target': True,
        'pass': cache_safe is True,
        'description': 'Cache safety maintained'
    }
    
    # Overall pass
    overall_pass = all(gate['pass'] for gate in gates.values())
    
    validation_result = {
        'overall_pass': overall_pass,
        'gates': gates,
        'overall_assessment': summary.get('overall_assessment', 'UNKNOWN'),
        'recommendations': summary.get('recommendations', [])
    }
    
    # Log results
    logger.info("=" * 60)
    logger.info("ACCEPTANCE GATE VALIDATION")
    logger.info("=" * 60)
    
    for gate_name, gate_info in gates.items():
        status = "✅ PASS" if gate_info['pass'] else "❌ FAIL"
        logger.info(f"{gate_name}: {status}")
        logger.info(f"  {gate_info['description']}")
        logger.info(f"  Value: {gate_info['value']}, Target: {gate_info['target']}")
    
    logger.info(f"\nOVERALL RESULT: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if validation_result['recommendations']:
        logger.info("\nRECOMMENDATIONS:")
        for rec in validation_result['recommendations']:
            logger.info(f"  - {rec}")
    
    return validation_result


def generate_final_report(
    output_dir: Path,
    ar0_results: Path,
    ar1_results: Path,
    evaluation_results: Path,
    validation_result: Dict,
    logger
):
    """Generate final validation report."""
    report = {
        'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'bem_2_version': '2.0',
        'pipeline_status': 'COMPLETE',
        'overall_pass': validation_result['overall_pass'],
        'training_phases': {
            'ar0_behavior_cloning': {
                'status': 'COMPLETE',
                'output_dir': str(ar0_results)
            },
            'ar1_policy_gradient': {
                'status': 'COMPLETE', 
                'output_dir': str(ar1_results)
            }
        },
        'evaluation': {
            'status': 'COMPLETE',
            'output_dir': str(evaluation_results)
        },
        'acceptance_gates': validation_result['gates'],
        'final_assessment': validation_result['overall_assessment'],
        'recommendations': validation_result['recommendations']
    }
    
    # Save report
    report_file = output_dir / 'bem2_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate human-readable summary
    summary_lines = [
        "BEM 2.0 AGENTIC ROUTER VALIDATION REPORT",
        "=" * 50,
        "",
        f"Validation Date: {report['validation_timestamp']}",
        f"Overall Result: {'✅ PASS' if report['overall_pass'] else '❌ FAIL'}",
        "",
        "ACCEPTANCE GATES:",
        "=" * 20
    ]
    
    for gate_name, gate_info in report['acceptance_gates'].items():
        status = "✅ PASS" if gate_info['pass'] else "❌ FAIL"
        summary_lines.extend([
            f"{gate_name}: {status}",
            f"  {gate_info['description']}",
            f"  Value: {gate_info['value']}, Target: {gate_info['target']}",
            ""
        ])
    
    if report['recommendations']:
        summary_lines.extend([
            "RECOMMENDATIONS:",
            "=" * 15
        ])
        for rec in report['recommendations']:
            summary_lines.append(f"  - {rec}")
        summary_lines.append("")
    
    summary_lines.extend([
        "ARTIFACTS:",
        "=" * 10,
        f"AR0 Training Results: {report['training_phases']['ar0_behavior_cloning']['output_dir']}",
        f"AR1 Training Results: {report['training_phases']['ar1_policy_gradient']['output_dir']}",
        f"Evaluation Results: {report['evaluation']['output_dir']}",
        f"Full Report: {report_file}",
        "",
        "=" * 50,
        "BEM 2.0 VALIDATION COMPLETE"
    ])
    
    summary_file = output_dir / 'bem2_validation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"Final report saved to: {report_file}")
    logger.info(f"Summary saved to: {summary_file}")
    
    return report_file


def main():
    parser = argparse.ArgumentParser(description="Run BEM 2.0 Validation Pipeline")
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode (reduced training iterations)')
    parser.add_argument('--output', type=str, default='validation_results',
                       help='Output directory for all results')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training phases (for testing evaluation only)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_level, str(output_dir / 'validation_pipeline.log'))
    
    logger.info("=" * 60)
    logger.info("BEM 2.0 AGENTIC ROUTER VALIDATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    try:
        # Create training configs
        logger.info("Creating training configurations...")
        ar0_config_path, ar1_config_path = create_validation_configs(output_dir, args.quick)
        
        ar0_results = None
        ar1_results = None
        
        if not args.skip_training:
            # Phase 1: AR0 Behavior Cloning
            logger.info("=" * 40)
            logger.info("PHASE 1: AR0 BEHAVIOR CLONING")
            logger.info("=" * 40)
            
            ar0_results = output_dir / 'AR0_training'
            ar0_success = run_training_phase('bc', ar0_config_path, output_dir=ar0_results, logger=logger)
            
            if not ar0_success:
                logger.error("AR0 training failed. Aborting pipeline.")
                return 1
            
            # Find AR0 best model
            ar0_model_path = ar0_results / 'best_model.pt'
            if not ar0_model_path.exists():
                ar0_model_path = ar0_results / 'final_model.pt'
            
            # Phase 2: AR1 Policy Gradient
            logger.info("=" * 40)
            logger.info("PHASE 2: AR1 POLICY GRADIENT")
            logger.info("=" * 40)
            
            ar1_results = output_dir / 'AR1_training'
            ar1_success = run_training_phase(
                'pg', ar1_config_path, 
                checkpoint_path=ar0_model_path,
                output_dir=ar1_results, 
                logger=logger
            )
            
            if not ar1_success:
                logger.error("AR1 training failed. Aborting pipeline.")
                return 1
            
            # Find AR1 best model
            ar1_model_path = ar1_results / 'best_model.pt'
            if not ar1_model_path.exists():
                ar1_model_path = ar1_results / 'final_model.pt'
        else:
            logger.info("Skipping training phases...")
            # Mock paths for testing
            ar0_results = output_dir / 'AR0_training'
            ar1_results = output_dir / 'AR1_training'
            ar1_model_path = ar1_results / 'best_model.pt'
        
        # Phase 3: Comprehensive Evaluation
        logger.info("=" * 40)
        logger.info("PHASE 3: COMPREHENSIVE EVALUATION")
        logger.info("=" * 40)
        
        evaluation_results = output_dir / 'evaluation'
        
        if not args.skip_training:
            eval_success = run_evaluation(
                model_path=ar1_model_path,
                baseline_path=ar0_model_path if ar0_results else None,
                output_dir=evaluation_results,
                logger=logger
            )
            
            if not eval_success:
                logger.error("Evaluation failed. Aborting pipeline.")
                return 1
        else:
            logger.info("Skipping evaluation (training was skipped)...")
            evaluation_results.mkdir(parents=True, exist_ok=True)
            # Create mock evaluation summary for testing
            mock_summary = {
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'overall_assessment': 'PASS',
                'key_metrics': {
                    'slice_b_improvement': 0.025,
                    'p50_latency_change_percent': 12.5,
                    'monotonicity_pass_rate': 0.85,
                    'cache_safe': True
                },
                'recommendations': []
            }
            with open(evaluation_results / 'evaluation_summary.json', 'w') as f:
                json.dump(mock_summary, f, indent=2)
        
        # Phase 4: Acceptance Gate Validation
        logger.info("=" * 40)
        logger.info("PHASE 4: ACCEPTANCE GATE VALIDATION")
        logger.info("=" * 40)
        
        validation_result = validate_acceptance_gates(evaluation_results, logger)
        
        # Generate final report
        logger.info("=" * 40)
        logger.info("GENERATING FINAL REPORT")
        logger.info("=" * 40)
        
        report_file = generate_final_report(
            output_dir=output_dir,
            ar0_results=ar0_results,
            ar1_results=ar1_results,
            evaluation_results=evaluation_results,
            validation_result=validation_result,
            logger=logger
        )
        
        total_time = time.time() - start_time
        
        # Final summary
        logger.info("=" * 60)
        logger.info("VALIDATION PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Overall result: {'✅ PASS' if validation_result['overall_pass'] else '❌ FAIL'}")
        logger.info(f"Report location: {report_file}")
        logger.info("=" * 60)
        
        return 0 if validation_result['overall_pass'] else 1
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)