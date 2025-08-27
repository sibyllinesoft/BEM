#!/usr/bin/env python3
"""
BEM Paper Factory - Statistical Analysis Pipeline
Comprehensive pipeline for rigorous statistical analysis of experimental results.

Pipeline stages:
1. Harvest and aggregate experimental logs
2. Compute bootstrap confidence intervals (10k resamples)
3. Apply Holm-Bonferroni multiple comparison correction
4. Validate all claims against pre-registered tests
5. Generate publication-ready statistical results
6. Create final claim validation report

This script orchestrates the complete statistical analysis workflow.
"""

import argparse
import json
import logging
import subprocess
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalPipeline:
    """
    Orchestrates the complete statistical analysis pipeline for BEM paper.
    """
    
    def __init__(self, 
                 logs_dir: str,
                 claims_file: str,
                 output_dir: str = "analysis",
                 bootstrap_samples: int = 10000,
                 confidence_level: float = 0.95):
        
        self.logs_dir = Path(logs_dir)
        self.claims_file = Path(claims_file)
        self.output_dir = Path(output_dir)
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline stages
        self.pipeline_results = {}
        self.start_time = None
    
    def stage_1_harvest_logs(self) -> Dict[str, Any]:
        """Stage 1: Harvest and aggregate all experimental logs."""
        logger.info("📊 Stage 1: Harvesting experimental logs...")
        
        stage_start = time.time()
        
        # Use the logs directory itself as the root for harvesting
        log_roots = [str(self.logs_dir)] if self.logs_dir.exists() else []
        
        if not log_roots:
            raise FileNotFoundError(f"No experimental logs found in {self.logs_dir}")
        
        # Run log harvesting script
        harvest_output = self.output_dir / "harvested_runs.jsonl"
        summary_output = self.output_dir / "harvest_summary.json"
        
        try:
            cmd = [
                'python', 'scripts/harvest_logs.py',
                '--roots'] + log_roots + [
                '--out', str(harvest_output),
                '--summary', str(summary_output)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Log harvesting failed: {result.stderr}")
            
            # Load harvest summary
            with open(summary_output, 'r') as f:
                harvest_summary = json.load(f)
            
            stage_duration = time.time() - stage_start
            
            stage_result = {
                'success': True,
                'duration_seconds': stage_duration,
                'harvested_data_file': str(harvest_output),
                'summary': harvest_summary,
                'log_roots_processed': log_roots
            }
            
            logger.info(f"✅ Stage 1 completed in {stage_duration:.1f}s")
            logger.info(f"   Harvested: {harvest_summary['harvest_summary']['total_experiments']} experiments")
            
            return stage_result
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Log harvesting timed out")
        except Exception as e:
            raise RuntimeError(f"Log harvesting error: {str(e)}")
    
    def stage_2_statistical_analysis(self, harvested_data_file: str) -> Dict[str, Any]:
        """Stage 2: Compute bootstrap CIs and statistical tests."""
        logger.info("🔬 Stage 2: Computing statistical analysis...")
        
        stage_start = time.time()
        
        # Run statistical analysis script
        stats_output = self.output_dir / "statistical_results.json"
        
        try:
            cmd = [
                'python', 'analysis/stats.py',
                '--in', harvested_data_file,
                '--claims', str(self.claims_file),
                '--out', str(stats_output),
                '--bootstrap', str(self.bootstrap_samples),
                '--correction', 'holm-bonferroni'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes for bootstrap analysis
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Statistical analysis failed: {result.stderr}")
            
            # Load statistical results
            with open(stats_output, 'r') as f:
                stats_results = json.load(f)
            
            stage_duration = time.time() - stage_start
            
            stage_result = {
                'success': True,
                'duration_seconds': stage_duration,
                'stats_results_file': str(stats_output),
                'summary': stats_results.get('summary', {}),
                'bootstrap_samples': self.bootstrap_samples,
                'confidence_level': self.confidence_level
            }
            
            logger.info(f"✅ Stage 2 completed in {stage_duration:.1f}s")
            logger.info(f"   Claims validated: {stats_results['summary']['passed_claims']}/{stats_results['summary']['total_claims']}")
            
            return stage_result
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Statistical analysis timed out")
        except Exception as e:
            raise RuntimeError(f"Statistical analysis error: {str(e)}")
    
    def stage_3_method_comparison(self, harvested_data_file: str) -> Dict[str, Any]:
        """Stage 3: Compare methods and compute effect sizes."""
        logger.info("⚖️ Stage 3: Computing method comparisons...")
        
        stage_start = time.time()
        
        # Load harvested data for processing
        experiments_data = []
        with open(harvested_data_file, 'r') as f:
            for line in f:
                experiments_data.append(json.loads(line.strip()))
        
        # Group by method type
        methods_data = self._group_experiments_by_method(experiments_data)
        
        # Compute pairwise comparisons
        comparisons = self._compute_method_comparisons(methods_data)
        
        # Save comparison results
        comparison_output = self.output_dir / "method_comparisons.json"
        with open(comparison_output, 'w') as f:
            json.dump(comparisons, f, indent=2, default=str)
        
        stage_duration = time.time() - stage_start
        
        stage_result = {
            'success': True,
            'duration_seconds': stage_duration,
            'comparisons_file': str(comparison_output),
            'methods_analyzed': list(methods_data.keys()),
            'total_comparisons': len(comparisons.get('pairwise_comparisons', {}))
        }
        
        logger.info(f"✅ Stage 3 completed in {stage_duration:.1f}s")
        logger.info(f"   Methods compared: {len(methods_data)}")
        
        return stage_result
    
    def _group_experiments_by_method(self, experiments_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group experiments by method type and approach."""
        methods = {}
        
        for exp in experiments_data:
            method_key = f"{exp.get('method_type', 'unknown')}_{exp.get('approach', 'unknown')}"
            
            if method_key not in methods:
                methods[method_key] = []
            
            methods[method_key].append(exp)
        
        return methods
    
    def _compute_method_comparisons(self, methods_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Compute statistical comparisons between methods."""
        comparisons = {
            'pairwise_comparisons': {},
            'effect_sizes': {},
            'summary_statistics': {}
        }
        
        # Get method names
        method_names = list(methods_data.keys())
        
        # Define metrics to compare
        metrics = ['exact_match', 'f1_score', 'bleu', 'chrF']
        
        # Compute summary statistics for each method
        for method_name, experiments in methods_data.items():
            method_stats = {}
            
            for metric in metrics:
                values = []
                for exp in experiments:
                    if 'evaluation_results' in exp and 'standard_metrics' in exp['evaluation_results']:
                        metric_value = exp['evaluation_results']['standard_metrics'].get(metric)
                        if metric_value is not None:
                            values.append(metric_value)
                
                if values:
                    method_stats[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n_samples': len(values),
                        'values': values  # For detailed analysis
                    }
            
            comparisons['summary_statistics'][method_name] = method_stats
        
        # Compute pairwise comparisons (simplified implementation)
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], i+1):
                comparison_key = f"{method1}_vs_{method2}"
                
                method1_stats = comparisons['summary_statistics'].get(method1, {})
                method2_stats = comparisons['summary_statistics'].get(method2, {})
                
                comparison_result = {}
                
                for metric in metrics:
                    if metric in method1_stats and metric in method2_stats:
                        # Compute effect size (Cohen's d approximation)
                        mean1 = method1_stats[metric]['mean']
                        mean2 = method2_stats[metric]['mean']
                        std1 = method1_stats[metric]['std']
                        std2 = method2_stats[metric]['std']
                        
                        pooled_std = np.sqrt((std1**2 + std2**2) / 2) if (std1**2 + std2**2) > 0 else 1
                        effect_size = (mean1 - mean2) / pooled_std
                        
                        comparison_result[metric] = {
                            'method1_mean': mean1,
                            'method2_mean': mean2,
                            'difference': mean1 - mean2,
                            'effect_size': float(effect_size),
                            'effect_magnitude': self._classify_effect_size(effect_size)
                        }
                
                comparisons['pairwise_comparisons'][comparison_key] = comparison_result
        
        return comparisons
    
    def _classify_effect_size(self, effect_size: float) -> str:
        """Classify effect size magnitude (Cohen's d)."""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def stage_4_claim_validation(self, stats_results_file: str) -> Dict[str, Any]:
        """Stage 4: Final claim validation and report generation."""
        logger.info("🎯 Stage 4: Final claim validation...")
        
        stage_start = time.time()
        
        # Load statistical results
        with open(stats_results_file, 'r') as f:
            stats_results = json.load(f)
        
        # Load claims ledger
        with open(self.claims_file, 'r') as f:
            claims_ledger = yaml.safe_load(f)
        
        # Generate comprehensive validation report
        validation_report = self._generate_comprehensive_validation_report(
            stats_results, claims_ledger
        )
        
        # Save validation report
        validation_output = self.output_dir / "final_claim_validation.json"
        with open(validation_output, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Generate human-readable summary
        self._generate_validation_summary(validation_report)
        
        stage_duration = time.time() - stage_start
        
        stage_result = {
            'success': True,
            'duration_seconds': stage_duration,
            'validation_file': str(validation_output),
            'ready_for_publication': validation_report.get('overall_assessment', {}).get('ready_for_publication', False),
            'total_claims': validation_report.get('summary', {}).get('total_claims', 0),
            'validated_claims': validation_report.get('summary', {}).get('validated_claims', 0)
        }
        
        logger.info(f"✅ Stage 4 completed in {stage_duration:.1f}s")
        
        return stage_result
    
    def _generate_comprehensive_validation_report(self, 
                                                stats_results: Dict[str, Any], 
                                                claims_ledger: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive claim validation report."""
        
        claim_results = stats_results.get('claim_results', {})
        
        validation_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_info': {
                'bootstrap_samples': self.bootstrap_samples,
                'confidence_level': self.confidence_level,
                'multiple_correction': 'holm-bonferroni'
            },
            'summary': {
                'total_claims': len(claims_ledger.get('claims', {})),
                'validated_claims': 0,
                'failed_claims': 0,
                'missing_claims': 0
            },
            'detailed_results': {},
            'failed_claims_analysis': [],
            'statistical_quality_checks': {},
            'overall_assessment': {}
        }
        
        # Process each claim
        for claim_id, claim_spec in claims_ledger.get('claims', {}).items():
            if claim_id in claim_results:
                claim_result = claim_results[claim_id]
                passes = claim_result.get('pass_status', False)
                
                detailed_result = {
                    'claim_id': claim_id,
                    'assertion': claim_spec.get('assertion', ''),
                    'pass_status': passes,
                    'confidence_interval': claim_result.get('confidence_interval'),
                    'effect_size': claim_result.get('effect_size'),
                    'p_value': claim_result.get('p_value'),
                    'corrected_p_value': claim_result.get('corrected_p_value'),
                    'test_type': claim_spec.get('statistical_test', ''),
                    'pass_rule': claim_spec.get('pass_rule', ''),
                    'figure_ref': claim_spec.get('figure_ref', ''),
                    'datasets': claim_spec.get('datasets', []),
                    'statistical_quality': self._assess_claim_quality(claim_result)
                }
                
                validation_report['detailed_results'][claim_id] = detailed_result
                
                if passes:
                    validation_report['summary']['validated_claims'] += 1
                else:
                    validation_report['summary']['failed_claims'] += 1
                    validation_report['failed_claims_analysis'].append({
                        'claim_id': claim_id,
                        'assertion': claim_spec.get('assertion', ''),
                        'failure_reason': self._analyze_failure_reason(claim_result, claim_spec),
                        'recommendation': self._generate_failure_recommendation(claim_result, claim_spec)
                    })
            else:
                validation_report['summary']['missing_claims'] += 1
                validation_report['failed_claims_analysis'].append({
                    'claim_id': claim_id,
                    'assertion': claim_spec.get('assertion', ''),
                    'failure_reason': 'No experimental data found for this claim',
                    'recommendation': 'Ensure experiments matching the claim specification are included'
                })
        
        # Overall assessment
        validation_report['overall_assessment'] = self._generate_overall_assessment(validation_report)
        
        return validation_report
    
    def _assess_claim_quality(self, claim_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the statistical quality of a claim result."""
        quality = {
            'confidence_interval_available': claim_result.get('confidence_interval') is not None,
            'effect_size_available': claim_result.get('effect_size') is not None,
            'p_value_available': claim_result.get('p_value') is not None,
            'multiple_correction_applied': claim_result.get('corrected_p_value') is not None
        }
        
        # Assess effect size magnitude if available
        if claim_result.get('effect_size') is not None:
            effect_size = abs(claim_result['effect_size'])
            quality['effect_size_magnitude'] = self._classify_effect_size(effect_size)
        
        # Assess confidence interval width if available
        if claim_result.get('confidence_interval'):
            ci = claim_result['confidence_interval']
            if len(ci) == 2:
                ci_width = ci[1] - ci[0]
                quality['confidence_interval_width'] = ci_width
                quality['confidence_interval_precision'] = "narrow" if ci_width < 0.1 else "wide"
        
        return quality
    
    def _analyze_failure_reason(self, claim_result: Dict[str, Any], claim_spec: Dict[str, Any]) -> str:
        """Analyze why a claim failed validation."""
        if 'error' in claim_result.get('test_result', {}):
            return f"Test execution error: {claim_result['test_result']['error']}"
        
        pass_rule = claim_spec.get('pass_rule', '')
        ci = claim_result.get('confidence_interval')
        
        if ci and 'ci_lower_bound' in pass_rule:
            return f"Confidence interval lower bound ({ci[0]:.3f}) does not meet threshold in pass rule: {pass_rule}"
        
        return "Statistical test criteria not met"
    
    def _generate_failure_recommendation(self, claim_result: Dict[str, Any], claim_spec: Dict[str, Any]) -> str:
        """Generate recommendation for addressing claim failure."""
        if 'error' in claim_result.get('test_result', {}):
            return "Review experimental data and ensure proper test implementation"
        
        ci = claim_result.get('confidence_interval')
        if ci and len(ci) == 2 and ci[0] < 0:
            return "Consider revising claim or improving experimental methodology - current results show negative or insignificant effect"
        
        return "Review experimental setup and consider increasing sample size or improving methodology"
    
    def _generate_overall_assessment(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall readiness assessment."""
        summary = validation_report['summary']
        
        total_claims = summary['total_claims']
        validated_claims = summary['validated_claims']
        failed_claims = summary['failed_claims']
        
        pass_rate = validated_claims / total_claims if total_claims > 0 else 0
        
        # Determine publication readiness
        ready_for_publication = (
            failed_claims == 0 and  # No failed claims
            summary['missing_claims'] == 0 and  # No missing claims
            pass_rate >= 0.95  # At least 95% pass rate
        )
        
        assessment = {
            'ready_for_publication': ready_for_publication,
            'pass_rate': pass_rate,
            'quality_grade': 'A' if pass_rate >= 0.95 else 'B' if pass_rate >= 0.8 else 'C',
            'critical_issues': failed_claims > 0,
            'recommendations': []
        }
        
        # Generate recommendations
        if not ready_for_publication:
            if failed_claims > 0:
                assessment['recommendations'].append("Address all failed claims before submission")
            if summary['missing_claims'] > 0:
                assessment['recommendations'].append("Ensure all claims have corresponding experimental data")
            if pass_rate < 0.95:
                assessment['recommendations'].append("Improve experimental methodology to increase claim validation rate")
        else:
            assessment['recommendations'].append("All claims validated - paper ready for submission!")
        
        return assessment
    
    def _generate_validation_summary(self, validation_report: Dict[str, Any]) -> None:
        """Generate human-readable validation summary."""
        summary_file = self.output_dir / "validation_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("BEM Paper Factory - Final Claim Validation Summary\n")
            f.write("=" * 55 + "\n\n")
            
            # Overall assessment
            assessment = validation_report['overall_assessment']
            if assessment['ready_for_publication']:
                f.write("🎉 READY FOR PUBLICATION\n\n")
            else:
                f.write("❌ NOT READY FOR PUBLICATION\n\n")
            
            # Summary statistics
            summary = validation_report['summary']
            f.write(f"Total Claims: {summary['total_claims']}\n")
            f.write(f"Validated Claims: {summary['validated_claims']}\n")
            f.write(f"Failed Claims: {summary['failed_claims']}\n")
            f.write(f"Missing Claims: {summary['missing_claims']}\n")
            f.write(f"Pass Rate: {assessment['pass_rate']*100:.1f}%\n\n")
            
            # Failed claims details
            if validation_report['failed_claims_analysis']:
                f.write("FAILED CLAIMS ANALYSIS:\n")
                f.write("-" * 25 + "\n")
                
                for failed_claim in validation_report['failed_claims_analysis'][:5]:  # Show first 5
                    f.write(f"Claim: {failed_claim['claim_id']}\n")
                    f.write(f"Assertion: {failed_claim['assertion'][:80]}...\n")
                    f.write(f"Reason: {failed_claim['failure_reason']}\n")
                    f.write(f"Recommendation: {failed_claim['recommendation']}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            for rec in assessment['recommendations']:
                f.write(f"• {rec}\n")
        
        logger.info(f"Validation summary saved to: {summary_file}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete statistical analysis pipeline."""
        logger.info("🚀 Starting complete statistical analysis pipeline...")
        
        self.start_time = time.time()
        pipeline_results = {
            'pipeline_start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stages': {},
            'overall_success': True,
            'final_assessment': {}
        }
        
        try:
            # Stage 1: Harvest logs
            stage1_result = self.stage_1_harvest_logs()
            pipeline_results['stages']['stage_1_harvest'] = stage1_result
            
            # Stage 2: Statistical analysis
            stage2_result = self.stage_2_statistical_analysis(
                stage1_result['harvested_data_file']
            )
            pipeline_results['stages']['stage_2_statistics'] = stage2_result
            
            # Stage 3: Method comparison
            stage3_result = self.stage_3_method_comparison(
                stage1_result['harvested_data_file']
            )
            pipeline_results['stages']['stage_3_comparison'] = stage3_result
            
            # Stage 4: Final validation
            stage4_result = self.stage_4_claim_validation(
                stage2_result['stats_results_file']
            )
            pipeline_results['stages']['stage_4_validation'] = stage4_result
            
            # Overall assessment
            pipeline_results['final_assessment'] = {
                'ready_for_publication': stage4_result['ready_for_publication'],
                'total_claims_validated': stage4_result['validated_claims'],
                'total_claims': stage4_result['total_claims'],
                'pipeline_duration_minutes': (time.time() - self.start_time) / 60
            }
            
            # Save pipeline results
            pipeline_output = self.output_dir / "statistical_pipeline_results.json"
            with open(pipeline_output, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            logger.info("🎯 Statistical pipeline completed successfully!")
            
            if pipeline_results['final_assessment']['ready_for_publication']:
                logger.info("✅ All statistical analysis complete - ready for paper generation!")
            else:
                logger.warning("⚠️  Statistical issues detected - review before proceeding")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            pipeline_results['overall_success'] = False
            pipeline_results['error'] = str(e)
            return pipeline_results

def main():
    parser = argparse.ArgumentParser(description='BEM Paper Factory - Statistical Analysis Pipeline')
    parser.add_argument('--logs-dir', required=True, help='Directory containing experimental logs')
    parser.add_argument('--claims', default='paper/claims.yaml', help='Claims ledger file')
    parser.add_argument('--output-dir', default='analysis', help='Output directory for analysis results')
    parser.add_argument('--bootstrap', type=int, default=10000, help='Number of bootstrap resamples')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = StatisticalPipeline(
        logs_dir=args.logs_dir,
        claims_file=args.claims,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap,
        confidence_level=args.confidence
    )
    
    # Run complete pipeline
    try:
        results = pipeline.run_complete_pipeline()
        
        if results['overall_success']:
            if results['final_assessment']['ready_for_publication']:
                logger.info("🎉 Statistical analysis complete - ready for paper generation!")
            else:
                logger.warning("⚠️  Review statistical results before proceeding")
        else:
            logger.error("❌ Statistical pipeline failed")
            exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        exit(1)

if __name__ == '__main__':
    main()