#!/usr/bin/env python3
"""
Comprehensive Agentic Router Evaluation Script

Runs complete evaluation suite including:
- Latency profiling with p50/p95 tracking
- Index-swap monotonicity testing  
- Cache safety validation
- Acceptance gate validation
- Performance benchmarking

Usage:
python bem2/evaluate_agentic_router.py --model logs/AR1/best_model.pt --baseline logs/AR0/best_model.pt --out evaluation_results/
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
import sys
import json
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from bem2.router.macro_policy import create_macro_policy
from bem2.router.composition_engine import create_composition_engine, create_default_experts
from bem2.router.agentic_router import create_agentic_router
from bem2.evaluation import LatencyProfiler, MonotonicityTester, CacheAnalyzer, AcceptanceValidator


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


def load_router_from_checkpoint(checkpoint_path: str, config: dict, device: torch.device):
    """Load router from checkpoint."""
    logger = logging.getLogger(__name__)
    
    # Create router components
    logger.info("Creating router components...")
    
    policy_config = config.get('macro_policy', {})
    macro_policy = create_macro_policy(policy_config, num_experts=3)
    
    expert_deltas = create_default_experts()
    composition_config = config.get('composition', {})
    composition_engine = create_composition_engine(composition_config, expert_deltas)
    
    router_config = config.get('router', {})
    agentic_router = create_agentic_router(
        config=router_config,
        macro_policy=macro_policy,
        composition_engine=composition_engine
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'router_state_dict' in checkpoint:
        agentic_router.load_state_dict(checkpoint['router_state_dict'])
    elif 'policy_state_dict' in checkpoint:
        agentic_router.macro_policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        logger.warning("No recognized state dict found in checkpoint")
    
    agentic_router.to(device)
    return agentic_router


def run_comprehensive_evaluation(
    router,
    baseline_router,
    config: dict,
    device: torch.device,
    output_dir: Path
) -> dict:
    """Run comprehensive evaluation suite."""
    logger = logging.getLogger(__name__)
    
    evaluation_results = {}
    
    # 1. Latency Profiling
    logger.info("=" * 60)
    logger.info("LATENCY PROFILING")
    logger.info("=" * 60)
    
    latency_config = config.get('latency_profiling', {})
    profiler = LatencyProfiler(
        warmup_iterations=latency_config.get('warmup_iterations', 10),
        measurement_iterations=latency_config.get('measurement_iterations', 100)
    )
    
    latency_results = profiler.profile_router(router, device=device)
    profiler.save_results(str(output_dir / 'latency_results.json'))
    evaluation_results['latency'] = latency_results
    
    logger.info(f"Latency profiling completed:")
    logger.info(f"  Mean latency: {latency_results['overall']['mean_latency']:.4f}s")
    logger.info(f"  P50 latency: {latency_results['overall']['p50_latency']:.4f}s") 
    logger.info(f"  P95 latency: {latency_results['overall']['p95_latency']:.4f}s")
    
    # 2. Monotonicity Testing
    logger.info("=" * 60)
    logger.info("MONOTONICITY TESTING")
    logger.info("=" * 60)
    
    monotonicity_config = config.get('monotonicity_testing', {})
    tester = MonotonicityTester(
        num_test_sequences=monotonicity_config.get('num_test_sequences', 50),
        sequence_length=monotonicity_config.get('sequence_length', 1024)
    )
    
    monotonicity_results = tester.run_monotonicity_tests(router, device=device)
    tester.save_results(str(output_dir / 'monotonicity_results.json'))
    evaluation_results['monotonicity'] = monotonicity_results
    
    logger.info(f"Monotonicity testing completed:")
    logger.info(f"  Pass rate: {monotonicity_results['summary']['pass_rate']:.2%}")
    logger.info(f"  Cache safety: {monotonicity_results['summary']['cache_safety_pass_rate']:.2%}")
    
    # 3. Cache Analysis
    logger.info("=" * 60)
    logger.info("CACHE ANALYSIS")
    logger.info("=" * 60)
    
    cache_config = config.get('cache_analysis', {})
    analyzer = CacheAnalyzer(
        chunk_size=cache_config.get('chunk_size', 128)
    )
    
    cache_results = analyzer.analyze_cache_safety(router, device=device)
    analyzer.save_results(str(output_dir / 'cache_results.json'))
    evaluation_results['cache'] = cache_results
    
    logger.info(f"Cache analysis completed:")
    if cache_results.get('summary'):
        logger.info(f"  Cache safe: {cache_results['summary']['cache_safe']}")
        logger.info(f"  Violation rate: {cache_results['summary']['violation_rate']:.4f}")
    
    # 4. Performance Comparison (if baseline available)
    if baseline_router:
        logger.info("=" * 60)
        logger.info("BASELINE COMPARISON")
        logger.info("=" * 60)
        
        baseline_latency = profiler.profile_router(baseline_router, device=device)
        evaluation_results['baseline_latency'] = baseline_latency
        
        # Compare key metrics
        router_p50 = latency_results['overall']['p50_latency']
        baseline_p50 = baseline_latency['overall']['p50_latency']
        latency_change = ((router_p50 - baseline_p50) / baseline_p50) * 100
        
        logger.info(f"Performance comparison:")
        logger.info(f"  P50 latency change: {latency_change:+.1f}%")
        logger.info(f"  Router P50: {router_p50:.4f}s")
        logger.info(f"  Baseline P50: {baseline_p50:.4f}s")
        
        evaluation_results['comparison'] = {
            'p50_latency_change_percent': latency_change,
            'router_p50': router_p50,
            'baseline_p50': baseline_p50
        }
    
    # 5. Acceptance Validation
    logger.info("=" * 60)
    logger.info("ACCEPTANCE VALIDATION")
    logger.info("=" * 60)
    
    # Prepare baseline metrics
    baseline_metrics = {}
    if baseline_router:
        baseline_metrics = {
            'p50_latency': baseline_latency['overall']['p50_latency'],
            'parameters': sum(p.numel() for p in baseline_router.parameters()),
            'average_score': 0.7  # Placeholder baseline score
        }
    
    acceptance_config = config.get('acceptance_validation', {})
    validator = AcceptanceValidator(
        baseline_metrics=baseline_metrics,
        target_metrics=acceptance_config.get('target_metrics', {})
    )
    
    acceptance_results = validator.validate_acceptance(
        router=router,
        baseline_router=baseline_router,
        device=device
    )
    validator.save_results(str(output_dir / 'acceptance_results.json'))
    evaluation_results['acceptance'] = acceptance_results
    
    logger.info(f"Acceptance validation completed:")
    logger.info(f"  Overall status: {acceptance_results.overall_status.value}")
    logger.info(f"  Pass rate: {acceptance_results.summary['pass_rate']:.2%}")
    logger.info(f"  Failed gates: {acceptance_results.summary['failed']}")
    
    return evaluation_results


def generate_summary_report(
    evaluation_results: dict,
    output_dir: Path
):
    """Generate comprehensive summary report."""
    logger = logging.getLogger(__name__)
    
    summary = {
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall_assessment': 'PASS',  # Will be updated based on results
        'key_metrics': {},
        'acceptance_gates': {},
        'recommendations': []
    }
    
    # Extract key metrics
    latency = evaluation_results.get('latency', {})
    if latency.get('overall'):
        summary['key_metrics'].update({
            'p50_latency_ms': latency['overall']['p50_latency'] * 1000,
            'p95_latency_ms': latency['overall']['p95_latency'] * 1000,
            'mean_latency_ms': latency['overall']['mean_latency'] * 1000
        })
    
    monotonicity = evaluation_results.get('monotonicity', {})
    if monotonicity.get('summary'):
        summary['key_metrics']['monotonicity_pass_rate'] = monotonicity['summary']['pass_rate']
    
    cache = evaluation_results.get('cache', {})
    if cache.get('summary'):
        summary['key_metrics']['cache_safe'] = cache['summary']['cache_safe']
        summary['key_metrics']['cache_violation_rate'] = cache['summary']['violation_rate']
    
    # Performance comparison
    comparison = evaluation_results.get('comparison', {})
    if comparison:
        summary['key_metrics']['p50_latency_change_percent'] = comparison['p50_latency_change_percent']
    
    # Acceptance gates
    acceptance = evaluation_results.get('acceptance')
    if acceptance:
        summary['acceptance_gates'] = {
            'overall_status': acceptance.overall_status.value,
            'pass_rate': acceptance.summary['pass_rate'],
            'failed_gates': [gate.name for gate in acceptance.gates if gate.status.value == 'FAIL'],
            'passed_gates': [gate.name for gate in acceptance.gates if gate.status.value == 'PASS']
        }
        
        # Update overall assessment
        if acceptance.overall_status.value != 'PASS':
            summary['overall_assessment'] = acceptance.overall_status.value
        
        # Add recommendations
        summary['recommendations'].extend(acceptance.recommendations)
    
    # Additional recommendations based on metrics
    if summary['key_metrics'].get('p50_latency_change_percent', 0) > 15:
        summary['recommendations'].append("Latency increase exceeds 15% threshold - optimize routing performance")
    
    if summary['key_metrics'].get('monotonicity_pass_rate', 1.0) < 0.8:
        summary['recommendations'].append("Monotonicity pass rate below 80% - improve routing stability")
    
    if not summary['key_metrics'].get('cache_safe', True):
        summary['recommendations'].append("Cache safety violations detected - review routing implementation")
    
    # Save summary report
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate human-readable report
    report_lines = [
        "AGENTIC ROUTER EVALUATION REPORT",
        "=" * 50,
        "",
        f"Evaluation Date: {summary['evaluation_timestamp']}",
        f"Overall Assessment: {summary['overall_assessment']}",
        "",
        "KEY METRICS:",
        f"  P50 Latency: {summary['key_metrics'].get('p50_latency_ms', 0):.1f}ms",
        f"  P95 Latency: {summary['key_metrics'].get('p95_latency_ms', 0):.1f}ms",
        f"  Latency Change: {summary['key_metrics'].get('p50_latency_change_percent', 0):+.1f}%",
        f"  Monotonicity Pass Rate: {summary['key_metrics'].get('monotonicity_pass_rate', 0):.1%}",
        f"  Cache Safe: {summary['key_metrics'].get('cache_safe', 'Unknown')}",
        "",
        "ACCEPTANCE GATES:",
        f"  Overall Status: {summary['acceptance_gates'].get('overall_status', 'Unknown')}",
        f"  Pass Rate: {summary['acceptance_gates'].get('pass_rate', 0):.1%}",
        f"  Failed Gates: {', '.join(summary['acceptance_gates'].get('failed_gates', []))}",
        "",
        "RECOMMENDATIONS:"
    ]
    
    for rec in summary['recommendations']:
        report_lines.append(f"  - {rec}")
    
    report_lines.extend([
        "",
        "=" * 50,
        "For detailed results, see individual result files in this directory."
    ])
    
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Summary report saved to {output_dir / 'evaluation_summary.json'}")
    logger.info(f"Human-readable report saved to {output_dir / 'evaluation_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agentic Router")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained router checkpoint')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline router checkpoint')
    parser.add_argument('--config', type=str, default='experiments/AR1_pg.yml',
                       help='Path to configuration file')
    parser.add_argument('--out', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_level, str(output_dir / 'evaluation.log'))
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = {}
        logger.warning(f"Config file not found: {args.config}, using defaults")
    
    # Load main router
    logger.info(f"Loading router from {args.model}")
    router = load_router_from_checkpoint(args.model, config, device)
    
    # Load baseline router if provided
    baseline_router = None
    if args.baseline and Path(args.baseline).exists():
        logger.info(f"Loading baseline router from {args.baseline}")
        baseline_router = load_router_from_checkpoint(args.baseline, config, device)
    
    logger.info(f"Router loaded with {sum(p.numel() for p in router.parameters())} parameters")
    
    # Run comprehensive evaluation
    logger.info("Starting comprehensive evaluation...")
    start_time = time.time()
    
    evaluation_results = run_comprehensive_evaluation(
        router=router,
        baseline_router=baseline_router,
        config=config,
        device=device,
        output_dir=output_dir
    )
    
    evaluation_time = time.time() - start_time
    logger.info(f"Evaluation completed in {evaluation_time:.1f} seconds")
    
    # Save complete results
    with open(output_dir / 'complete_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Generate summary report
    generate_summary_report(evaluation_results, output_dir)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    
    summary_file = output_dir / 'evaluation_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        logger.info(f"Overall Assessment: {summary['overall_assessment']}")
        
        if summary['key_metrics']:
            logger.info("Key Metrics:")
            for key, value in summary['key_metrics'].items():
                logger.info(f"  {key}: {value}")
        
        if summary['recommendations']:
            logger.info("Recommendations:")
            for rec in summary['recommendations']:
                logger.info(f"  - {rec}")
    
    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()