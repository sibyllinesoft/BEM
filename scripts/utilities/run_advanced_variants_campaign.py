#!/usr/bin/env python3
"""
Advanced BEM Variants Campaign Executor

Runs V2 (Dual-Path), V7 (FiLM-lite), and V11 (Learned Cache) variants
against B1 baseline with comprehensive quality gates and statistical analysis.

This integrates with the existing batch experiment infrastructure while
adding advanced variant-specific validation and analysis.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.run_batch_experiments import BatchExperimentRunner, ExperimentJob
from bem.advanced_variants import AdvancedVariantsRunner
from analysis.statistical_analysis import StatisticalAnalyzer
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedVariantsCampaignExecutor:
    """
    Orchestrates the complete advanced BEM variants campaign with
    statistical rigor and quality gate validation.
    """
    
    def __init__(
        self,
        experiments_dir: str = "experiments",
        output_dir: str = "logs/advanced_variants",
        num_seeds: int = 5,
        max_parallel: int = 2  # Conservative for resource management
    ):
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.num_seeds = num_seeds
        self.max_parallel = max_parallel
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.batch_runner = BatchExperimentRunner(
            experiments_dir=str(self.experiments_dir),
            output_base_dir=str(self.output_dir),
            num_seeds=num_seeds,
            max_parallel=max_parallel
        )
        
        self.variants_runner = AdvancedVariantsRunner(
            output_dir=str(self.output_dir),
            num_seeds=num_seeds
        )
        
        # Define experiment configurations
        self.variant_configs = [
            "V2_dual_path.yaml",
            "V7_film_lite.yaml", 
            "V11_learned_cache_policy.yaml"
        ]
        
        self.baseline_config = "B1_bem_v11_stable.yaml"
        
    def verify_configurations(self) -> List[str]:
        """Verify that all required experiment configurations exist."""
        all_configs = [self.baseline_config] + self.variant_configs
        existing_configs = []
        missing_configs = []
        
        for config in all_configs:
            config_path = self.experiments_dir / config
            if config_path.exists():
                existing_configs.append(str(config_path))
                logger.info(f"✓ Found config: {config}")
            else:
                missing_configs.append(config)
                logger.error(f"✗ Missing config: {config}")
                
        if missing_configs:
            logger.error(f"Missing {len(missing_configs)} required configurations!")
            logger.error("Please ensure all variant configs are generated.")
            return []
            
        logger.info(f"All {len(existing_configs)} configurations verified")
        return existing_configs
        
    def create_experiment_jobs(self, config_files: List[str]) -> List[ExperimentJob]:
        """Create experiment jobs for all configurations and seeds."""
        jobs = []
        
        for config_file in config_files:
            config_path = Path(config_file)
            experiment_id = config_path.stem
            
            # Load config to determine variant type
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            architecture = config_data.get('model', {}).get('architecture', 'unknown')
            
            # Determine method type and approach for job classification
            if 'dual_path' in architecture:
                method_type = 'V2_dual_path_lora'
                approach = 'orthogonal_dual_branch'
            elif 'film_lite' in architecture:
                method_type = 'V7_film_lite_bem' 
                approach = 'film_modulation'
            elif 'learned_cache' in architecture:
                method_type = 'V11_learned_cache_bem'
                approach = 'adaptive_kv_cache'
            elif 'bem_v11_stable' in architecture:
                method_type = 'B1_bem_v11_stable'
                approach = 'baseline'
            else:
                method_type = 'unknown'
                approach = 'unknown'
                
            # Create jobs for all seeds
            for seed in range(1, self.num_seeds + 1):
                job = ExperimentJob(
                    config_file=str(config_file),
                    experiment_id=f"{experiment_id}_seed{seed}",
                    method_type=method_type,
                    approach=approach,
                    seed=seed,
                    output_dir=str(self.output_dir / f"{experiment_id}_seed{seed}"),
                    priority=1,  # All variants have same priority
                    max_retries=2,
                    timeout_minutes=90  # Longer timeout for advanced variants
                )
                jobs.append(job)
                
        logger.info(f"Created {len(jobs)} experiment jobs")
        return jobs
        
    def run_experiments(self, jobs: List[ExperimentJob]) -> Dict[str, Any]:
        """Run all experiment jobs with progress tracking."""
        logger.info("Starting experiment execution...")
        logger.info(f"Total jobs: {len(jobs)}")
        logger.info(f"Configurations: {len(set(job.config_file for job in jobs))}")
        logger.info(f"Seeds per config: {self.num_seeds}")
        
        # Run experiments
        completed_jobs, failed_jobs = self.batch_runner.run_batch_experiments(jobs)
        
        # Analyze results
        results_summary = {
            'total_jobs': len(jobs),
            'completed_jobs': len(completed_jobs),
            'failed_jobs': len(failed_jobs),
            'success_rate': len(completed_jobs) / len(jobs) * 100,
            'completed_by_config': {},
            'failed_by_config': {}
        }
        
        # Group by configuration
        for job_result in completed_jobs:
            config_name = Path(job_result.job.config_file).stem
            if config_name not in results_summary['completed_by_config']:
                results_summary['completed_by_config'][config_name] = []
            results_summary['completed_by_config'][config_name].append(job_result.job.seed)
            
        for job_result in failed_jobs:
            config_name = Path(job_result.job.config_file).stem
            if config_name not in results_summary['failed_by_config']:
                results_summary['failed_by_config'][config_name] = []
            results_summary['failed_by_config'][config_name].append(job_result.job.seed)
            
        logger.info(f"Execution complete: {results_summary['success_rate']:.1f}% success rate")
        
        return results_summary
        
    def collect_results(self) -> Dict[str, Dict[str, Any]]:
        """Collect and aggregate results from all completed experiments."""
        logger.info("Collecting experimental results...")
        
        results_by_config = {}
        
        # Scan output directory for results
        for config_dir in self.output_dir.iterdir():
            if not config_dir.is_dir():
                continue
                
            # Extract config name and seed
            parts = config_dir.name.split('_seed')
            if len(parts) != 2:
                continue
                
            config_name = parts[0]
            seed = int(parts[1])
            
            # Load results if available
            results_file = config_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        seed_results = json.load(f)
                        
                    if config_name not in results_by_config:
                        results_by_config[config_name] = {}
                        
                    results_by_config[config_name][f'seed_{seed}'] = seed_results
                    
                except Exception as e:
                    logger.warning(f"Failed to load results from {results_file}: {e}")
                    
        # Aggregate results across seeds
        aggregated_results = {}
        for config_name, seed_results in results_by_config.items():
            aggregated_results[config_name] = self._aggregate_seed_results(seed_results)
            
        logger.info(f"Collected results for {len(aggregated_results)} configurations")
        return aggregated_results
        
    def _aggregate_seed_results(self, seed_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate results across multiple seeds with statistics."""
        if not seed_results:
            return {}
            
        # Extract metrics from all seeds
        metrics = {}
        for seed_name, results in seed_results.items():
            eval_metrics = results.get('eval_metrics', {})
            for metric_name, value in eval_metrics.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)
                
        # Compute statistics
        aggregated = {}
        for metric_name, values in metrics.items():
            if values:
                aggregated[f"{metric_name}_mean"] = np.mean(values)
                aggregated[f"{metric_name}_std"] = np.std(values)
                aggregated[f"{metric_name}_ci_lower"] = np.percentile(values, 2.5)
                aggregated[f"{metric_name}_ci_upper"] = np.percentile(values, 97.5)
                aggregated[f"{metric_name}_values"] = values
                
        aggregated['num_seeds'] = len(seed_results)
        return aggregated
        
    def run_quality_gate_validation(
        self, 
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, bool]]:
        """Run quality gate validation for all variants."""
        logger.info("Running quality gate validation...")
        
        # Get baseline results
        baseline_results = results.get('B1_bem_v11_stable', {})
        if not baseline_results:
            logger.error("No baseline results found for quality gate validation!")
            return {}
            
        # Validate each variant
        validations = {}
        for config_name, config_results in results.items():
            if config_name == 'B1_bem_v11_stable':
                continue  # Skip baseline
                
            # Map config name to variant name
            if 'V2_dual_path' in config_name:
                variant_name = 'V2_dual_path'
            elif 'V7_film_lite' in config_name:
                variant_name = 'V7_film_lite'  
            elif 'V11_learned_cache' in config_name:
                variant_name = 'V11_learned_cache'
            else:
                logger.warning(f"Unknown variant: {config_name}")
                continue
                
            validations[variant_name] = self.variants_runner.validate_quality_gates(
                variant_name, config_results, baseline_results
            )
            
        return validations
        
    def run_statistical_analysis(
        self, 
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run statistical analysis with FDR correction."""
        logger.info("Running statistical analysis...")
        
        # Initialize statistical analyzer
        analyzer = StatisticalAnalyzer()
        
        # Get baseline and variant results
        baseline_name = 'B1_bem_v11_stable'
        baseline_results = results.get(baseline_name, {})
        
        if not baseline_results:
            logger.error("No baseline results for statistical analysis!")
            return {}
            
        # Compare each variant to baseline
        comparisons = {}
        primary_metrics = ['BLEU', 'chrF', 'EM', 'F1']
        
        for config_name, config_results in results.items():
            if config_name == baseline_name:
                continue
                
            config_comparisons = {}
            for metric in primary_metrics:
                baseline_values = baseline_results.get(f'{metric}_values', [])
                variant_values = config_results.get(f'{metric}_values', [])
                
                if baseline_values and variant_values:
                    # Paired t-test
                    stat_result = analyzer.paired_t_test(
                        baseline_values, variant_values, 
                        alternative='two-sided'
                    )
                    
                    # Bootstrap confidence intervals
                    ci_result = analyzer.bootstrap_confidence_interval(
                        baseline_values, variant_values,
                        metric_func=lambda x, y: np.mean(y) - np.mean(x),  # Difference in means
                        n_bootstrap=10000,
                        confidence_level=0.95
                    )
                    
                    config_comparisons[metric] = {
                        'baseline_mean': np.mean(baseline_values),
                        'variant_mean': np.mean(variant_values),
                        'difference': np.mean(variant_values) - np.mean(baseline_values),
                        'p_value': stat_result.get('p_value', 1.0),
                        'confidence_interval': ci_result,
                        't_statistic': stat_result.get('t_statistic', 0.0)
                    }
                    
            comparisons[config_name] = config_comparisons
            
        # Apply FDR correction
        all_p_values = []
        comparison_keys = []
        
        for config_name, config_comparisons in comparisons.items():
            for metric, comparison in config_comparisons.items():
                all_p_values.append(comparison['p_value'])
                comparison_keys.append((config_name, metric))
                
        if all_p_values:
            corrected_p_values = analyzer.fdr_correction(all_p_values)
            
            # Update comparisons with corrected p-values
            for i, (config_name, metric) in enumerate(comparison_keys):
                comparisons[config_name][metric]['p_value_corrected'] = corrected_p_values[i]
                comparisons[config_name][metric]['significant_corrected'] = corrected_p_values[i] < 0.05
                
        return {
            'comparisons': comparisons,
            'statistical_power': {
                'num_seeds': self.num_seeds,
                'significance_level': 0.05,
                'fdr_correction': 'benjamini_hochberg',
                'confidence_level': 0.95
            }
        }
        
    def generate_campaign_report(
        self,
        execution_summary: Dict[str, Any],
        results: Dict[str, Dict[str, Any]], 
        quality_gates: Dict[str, Dict[str, bool]],
        statistical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive campaign report."""
        logger.info("Generating campaign report...")
        
        # Summary statistics
        total_variants = len([k for k in results.keys() if k != 'B1_bem_v11_stable'])
        variants_with_results = sum(1 for k, v in results.items() if k != 'B1_bem_v11_stable' and v)
        
        # Quality gate summary
        gate_summary = {}
        for variant_name, gates in quality_gates.items():
            passed_gates = sum(gates.values())
            total_gates = len(gates)
            gate_summary[variant_name] = {
                'passed': passed_gates,
                'total': total_gates,
                'pass_rate': passed_gates / total_gates if total_gates > 0 else 0.0,
                'all_passed': passed_gates == total_gates
            }
            
        # Statistical significance summary
        significance_summary = {}
        for config_name, comparisons in statistical_analysis.get('comparisons', {}).items():
            significant_improvements = 0
            total_comparisons = len(comparisons)
            
            for metric, comparison in comparisons.items():
                if comparison.get('significant_corrected', False) and comparison.get('difference', 0) > 0:
                    significant_improvements += 1
                    
            significance_summary[config_name] = {
                'significant_improvements': significant_improvements,
                'total_comparisons': total_comparisons,
                'improvement_rate': significant_improvements / total_comparisons if total_comparisons > 0 else 0.0
            }
            
        # Overall campaign assessment
        successful_variants = sum(
            1 for variant_name in gate_summary.keys()
            if gate_summary[variant_name]['all_passed']
        )
        
        report = {
            'campaign_overview': {
                'total_variants_tested': total_variants,
                'variants_with_complete_results': variants_with_results,
                'variants_passing_all_gates': successful_variants,
                'overall_success_rate': successful_variants / total_variants if total_variants > 0 else 0.0,
                'execution_summary': execution_summary
            },
            'quality_gate_results': gate_summary,
            'statistical_analysis_results': significance_summary,
            'detailed_results': results,
            'detailed_quality_gates': quality_gates,
            'detailed_statistical_analysis': statistical_analysis,
            'recommendations': self._generate_recommendations(
                gate_summary, significance_summary, statistical_analysis
            )
        }
        
        return report
        
    def _generate_recommendations(
        self,
        gate_summary: Dict[str, Any],
        significance_summary: Dict[str, Any],
        statistical_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # Check for successful variants
        successful_variants = [
            variant for variant, summary in gate_summary.items()
            if summary['all_passed'] and significance_summary.get(variant, {}).get('improvement_rate', 0) > 0
        ]
        
        if successful_variants:
            recommendations.append(
                f"✅ SUCCESSFUL VARIANTS: {', '.join(successful_variants)} passed all quality gates "
                f"and showed significant improvements. Recommend for publication."
            )
        else:
            recommendations.append(
                "⚠️  NO FULLY SUCCESSFUL VARIANTS: All variants failed at least one quality gate "
                "or showed no significant improvements."
            )
            
        # Specific variant recommendations
        for variant, gates in gate_summary.items():
            if not gates['all_passed']:
                failed_gates = gates['total'] - gates['passed']
                recommendations.append(
                    f"❌ {variant}: Failed {failed_gates} quality gates. "
                    f"Review architectural assumptions and hyperparameters."
                )
                
        # Statistical power recommendations
        if self.num_seeds < 5:
            recommendations.append(
                f"⚠️  STATISTICAL POWER: Only {self.num_seeds} seeds used. "
                f"Recommend ≥5 seeds for adequate statistical power."
            )
            
        return recommendations
        
    def save_results(self, report: Dict[str, Any]) -> str:
        """Save complete campaign results."""
        output_file = self.output_dir / "advanced_variants_campaign_report.json"
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Campaign report saved: {output_file}")
        return str(output_file)
        
    def run_campaign(self) -> Dict[str, Any]:
        """Run complete advanced variants campaign."""
        logger.info("🚀 Starting Advanced BEM Variants Campaign")
        
        # 1. Verify configurations
        config_files = self.verify_configurations()
        if not config_files:
            raise RuntimeError("Required configurations not found!")
            
        # 2. Create experiment jobs
        jobs = self.create_experiment_jobs(config_files)
        
        # 3. Run experiments
        execution_summary = self.run_experiments(jobs)
        
        # 4. Collect results
        results = self.collect_results()
        
        # 5. Quality gate validation
        quality_gates = self.run_quality_gate_validation(results)
        
        # 6. Statistical analysis
        statistical_analysis = self.run_statistical_analysis(results)
        
        # 7. Generate report
        report = self.generate_campaign_report(
            execution_summary, results, quality_gates, statistical_analysis
        )
        
        # 8. Save results
        output_file = self.save_results(report)
        
        logger.info("✅ Advanced BEM Variants Campaign Complete!")
        logger.info(f"📊 Report available at: {output_file}")
        
        return report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Advanced BEM Variants Campaign")
    parser.add_argument('--experiments-dir', default='experiments', 
                       help='Directory containing experiment configs')
    parser.add_argument('--output-dir', default='logs/advanced_variants',
                       help='Output directory for results')  
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of seeds per experiment')
    parser.add_argument('--max-parallel', type=int, default=2,
                       help='Maximum parallel experiments')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configs without running experiments')
    
    args = parser.parse_args()
    
    # Create executor
    executor = AdvancedVariantsCampaignExecutor(
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
        max_parallel=args.max_parallel
    )
    
    if args.dry_run:
        # Just verify configurations
        logger.info("🔍 Dry run: Verifying configurations...")
        config_files = executor.verify_configurations()
        
        if config_files:
            jobs = executor.create_experiment_jobs(config_files)
            logger.info(f"✅ Would create {len(jobs)} experiment jobs")
            logger.info("Configuration verification successful!")
        else:
            logger.error("❌ Configuration verification failed!")
            return 1
    else:
        # Run full campaign
        try:
            report = executor.run_campaign()
            
            # Print summary
            overview = report['campaign_overview']
            print("\n" + "="*60)
            print("ADVANCED BEM VARIANTS CAMPAIGN SUMMARY")
            print("="*60)
            print(f"Variants tested: {overview['total_variants_tested']}")
            print(f"Success rate: {overview['overall_success_rate']:.1%}")
            print(f"Variants passing all gates: {overview['variants_passing_all_gates']}")
            
            # Print recommendations
            print("\nRECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  {rec}")
                
            return 0
            
        except Exception as e:
            logger.error(f"Campaign failed: {e}")
            return 1
            
    return 0


if __name__ == "__main__":
    sys.exit(main())