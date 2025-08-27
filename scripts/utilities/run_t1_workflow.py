#!/usr/bin/env python3
"""
BEM 2.0 Complete T1 Tracking Workflow Orchestrator
Executes the comprehensive tracking and evaluation workflows with rigorous statistical analysis.

This is the master orchestrator that runs the complete T1 workflow:
T1.1 - Data Collection (harvest runs from all pillars)
T1.2 - Statistical Analysis (BCa bootstrap with FDR correction) 
T1.3 - Specialized Analysis (router audit, soak, hallucinations, violations)

Key Features:
- Orchestrates complete end-to-end T1 workflow
- Rigorous statistical methodology with BCa bootstrap and FDR correction
- Comprehensive specialized analysis across all safety and performance dimensions
- Generates consolidated reports with actionable insights
- Supports both full analysis and targeted subset analysis
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class T1WorkflowOrchestrator:
    """
    Master orchestrator for the complete BEM 2.0 T1 tracking workflow.
    """
    
    def __init__(self, 
                 results_dir: str = "results",
                 analysis_dir: str = "analysis",
                 output_prefix: str = "bem2"):
        """
        Initialize T1 workflow orchestrator.
        
        Args:
            results_dir: Base directory containing pillar results
            analysis_dir: Directory for analysis outputs
            output_prefix: Prefix for output files
        """
        self.results_dir = Path(results_dir)
        self.analysis_dir = Path(analysis_dir)
        self.output_prefix = output_prefix
        
        # Ensure analysis directory exists
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output files
        self.consolidated_runs_file = self.analysis_dir / f"runs_{output_prefix}.jsonl"
        self.harvest_report_file = self.analysis_dir / f"harvest_report_{output_prefix}.json"
        self.statistical_report_file = self.analysis_dir / f"statistical_report_{output_prefix}.json"
        self.pareto_report_file = self.analysis_dir / f"pareto_analysis_{output_prefix}.json"
        self.pareto_plot_file = self.analysis_dir / f"pareto_plot_{output_prefix}.png"
        self.router_audit_file = self.analysis_dir / f"router_audit_{output_prefix}.json"
        self.soak_analysis_file = self.analysis_dir / f"soak_analysis_{output_prefix}.json"
        self.hallucination_analysis_file = self.analysis_dir / f"hallucination_analysis_{output_prefix}.json"
        self.violation_analysis_file = self.analysis_dir / f"violation_analysis_{output_prefix}.json"
        self.master_report_file = self.analysis_dir / f"T1_master_report_{output_prefix}.json"
        
        logger.info(f"Initialized T1 workflow orchestrator")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Analysis directory: {self.analysis_dir}")
        logger.info(f"Consolidated runs output: {self.consolidated_runs_file}")
    
    def execute_t1_1_data_collection(self, pillars: Optional[List[str]] = None) -> bool:
        """
        Execute T1.1: Data Collection and Consolidation
        
        Args:
            pillars: Optional list of specific pillars to collect
            
        Returns:
            Success status
        """
        logger.info("="*60)
        logger.info("EXECUTING T1.1: DATA COLLECTION")
        logger.info("="*60)
        
        try:
            # Build command
            cmd = [
                sys.executable, 
                "analysis/collect.py",
                "--results-dir", str(self.results_dir),
                "--output", str(self.consolidated_runs_file)
            ]
            
            if pillars:
                cmd.extend(["--pillars"] + pillars)
            
            # Execute data collection
            logger.info(f"Running data collection: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Data collection failed: {result.stderr}")
                return False
            
            logger.info("Data collection output:")
            logger.info(result.stdout)
            
            # Verify output file exists and has content
            if not self.consolidated_runs_file.exists():
                logger.error(f"Consolidated runs file not created: {self.consolidated_runs_file}")
                return False
            
            # Count lines to verify content
            with open(self.consolidated_runs_file, 'r') as f:
                line_count = sum(1 for _ in f)
            
            logger.info(f"✅ T1.1 Data Collection completed successfully")
            logger.info(f"📊 Consolidated {line_count} experimental runs")
            logger.info(f"📁 Output: {self.consolidated_runs_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ T1.1 Data Collection failed: {e}")
            return False
    
    def execute_t1_2_statistical_analysis(self) -> bool:
        """
        Execute T1.2: Statistical Analysis with BCa Bootstrap and FDR Correction
        
        Returns:
            Success status
        """
        logger.info("="*60)
        logger.info("EXECUTING T1.2: STATISTICAL ANALYSIS")
        logger.info("="*60)
        
        try:
            # Build command for statistical analysis
            cmd = [
                sys.executable,
                "analysis/stats.py",
                "--in", str(self.consolidated_runs_file),
                "--out", str(self.statistical_report_file),
                "--bootstrap", "10000",  # BCa bootstrap with 10k samples
                "--correction", "fdr_bh"  # Benjamini-Hochberg FDR correction
            ]
            
            # Execute statistical analysis
            logger.info(f"Running statistical analysis: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Statistical analysis failed: {result.stderr}")
                return False
            
            logger.info("Statistical analysis output:")
            logger.info(result.stdout)
            
            # Verify output
            if not self.statistical_report_file.exists():
                logger.error(f"Statistical report not created: {self.statistical_report_file}")
                return False
            
            logger.info(f"✅ T1.2 Statistical Analysis completed successfully")
            logger.info(f"📊 BCa bootstrap (10k samples) with FDR correction applied")
            logger.info(f"📁 Output: {self.statistical_report_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ T1.2 Statistical Analysis failed: {e}")
            return False
    
    def execute_pareto_analysis(self, primary_metric: str = "F1") -> bool:
        """
        Execute Pareto frontier analysis for performance vs resource trade-offs.
        
        Args:
            primary_metric: Primary quality metric for Pareto analysis
            
        Returns:
            Success status
        """
        logger.info("="*60)
        logger.info("EXECUTING PARETO FRONTIER ANALYSIS")
        logger.info("="*60)
        
        try:
            cmd = [
                sys.executable,
                "analysis/pareto.py",
                "--in", str(self.consolidated_runs_file),
                "--out", str(self.pareto_report_file),
                "--plot", str(self.pareto_plot_file),
                "--primary-metric", primary_metric,
                "--latency-metric", "p50_latency_ms",
                "--budget-pct", "15.0"
            ]
            
            # Execute Pareto analysis
            logger.info(f"Running Pareto analysis: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Pareto analysis failed: {result.stderr}")
                return False
            
            logger.info("Pareto analysis output:")
            logger.info(result.stdout)
            
            # Verify outputs
            if not self.pareto_report_file.exists():
                logger.error(f"Pareto report not created: {self.pareto_report_file}")
                return False
            
            logger.info(f"✅ Pareto Analysis completed successfully")
            logger.info(f"📊 Performance vs resource trade-offs analyzed")
            logger.info(f"📁 Report: {self.pareto_report_file}")
            if self.pareto_plot_file.exists():
                logger.info(f"📈 Plot: {self.pareto_plot_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pareto Analysis failed: {e}")
            return False
    
    def execute_t1_3_specialized_analyses(self) -> Dict[str, bool]:
        """
        Execute T1.3: All Specialized Analysis Components
        
        Returns:
            Dict mapping analysis type to success status
        """
        logger.info("="*60)
        logger.info("EXECUTING T1.3: SPECIALIZED ANALYSES")
        logger.info("="*60)
        
        results = {}
        
        # Router Action Audit
        logger.info("\n🔍 Running Router Action Audit Analysis...")
        try:
            cmd = [
                sys.executable,
                "analysis/router_audit.py",
                "--input", str(self.consolidated_runs_file),
                "--output", str(self.router_audit_file),
                "--experts", "8",
                "--confidence-threshold", "0.8"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Router audit completed")
                results['router_audit'] = True
            else:
                logger.warning(f"⚠️ Router audit failed: {result.stderr}")
                results['router_audit'] = False
                
        except Exception as e:
            logger.warning(f"⚠️ Router audit error: {e}")
            results['router_audit'] = False
        
        # Online Learning Soak Analysis
        logger.info("\n📈 Running Online Learning Soak Analysis...")
        try:
            cmd = [
                sys.executable,
                "analysis/soak.py",
                "--input", str(self.consolidated_runs_file),
                "--output", str(self.soak_analysis_file),
                "--metric", "accuracy",
                "--stability-window", "100"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Soak analysis completed")
                results['soak_analysis'] = True
            else:
                logger.warning(f"⚠️ Soak analysis failed: {result.stderr}")
                results['soak_analysis'] = False
                
        except Exception as e:
            logger.warning(f"⚠️ Soak analysis error: {e}")
            results['soak_analysis'] = False
        
        # Hallucination Analysis
        logger.info("\n🔍 Running Hallucination Analysis...")
        try:
            cmd = [
                sys.executable,
                "analysis/hallucinations.py",
                "--input", str(self.consolidated_runs_file),
                "--output", str(self.hallucination_analysis_file),
                "--threshold", "0.3"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Hallucination analysis completed")
                results['hallucination_analysis'] = True
            else:
                logger.warning(f"⚠️ Hallucination analysis failed: {result.stderr}")
                results['hallucination_analysis'] = False
                
        except Exception as e:
            logger.warning(f"⚠️ Hallucination analysis error: {e}")
            results['hallucination_analysis'] = False
        
        # Constitutional AI Violation Analysis
        logger.info("\n⚖️ Running Constitutional Violation Analysis...")
        try:
            cmd = [
                sys.executable,
                "analysis/violations.py",
                "--input", str(self.consolidated_runs_file),
                "--output", str(self.violation_analysis_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Violation analysis completed")
                results['violation_analysis'] = True
            else:
                logger.warning(f"⚠️ Violation analysis failed: {result.stderr}")
                results['violation_analysis'] = False
                
        except Exception as e:
            logger.warning(f"⚠️ Violation analysis error: {e}")
            results['violation_analysis'] = False
        
        # Summary
        successful_analyses = sum(results.values())
        total_analyses = len(results)
        
        logger.info(f"\n📊 T1.3 Specialized Analyses Summary:")
        logger.info(f"✅ Successful: {successful_analyses}/{total_analyses}")
        
        for analysis, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {analysis}")
        
        return results
    
    def generate_master_report(self, specialized_results: Dict[str, bool]) -> Dict[str, Any]:
        """
        Generate comprehensive master T1 tracking report.
        
        Args:
            specialized_results: Results from specialized analyses
            
        Returns:
            Master report data
        """
        logger.info("="*60)
        logger.info("GENERATING MASTER T1 REPORT")
        logger.info("="*60)
        
        master_report = {
            'T1_workflow_metadata': {
                'execution_timestamp': time.time(),
                'workflow_version': 'BEM_2.0_T1',
                'orchestrator_version': '1.0.0',
                'consolidated_runs_file': str(self.consolidated_runs_file),
                'analysis_directory': str(self.analysis_dir)
            },
            'execution_summary': {
                'T1_1_data_collection': self.consolidated_runs_file.exists(),
                'T1_2_statistical_analysis': self.statistical_report_file.exists(),
                'pareto_analysis': self.pareto_report_file.exists(),
                'T1_3_specialized_analyses': specialized_results
            },
            'output_files': {},
            'component_summaries': {},
            'overall_assessment': {},
            'recommendations': []
        }
        
        # Load and summarize component reports
        try:
            # Harvest report summary
            if self.harvest_report_file.exists():
                with open(self.harvest_report_file, 'r') as f:
                    harvest_data = json.load(f)
                master_report['component_summaries']['data_collection'] = {
                    'total_runs': harvest_data.get('harvest_metadata', {}).get('total_runs', 0),
                    'validation_rate': harvest_data.get('validation_summary', {}).get('validation_rate', 0),
                    'pillars_covered': len(harvest_data.get('pillar_breakdown', {}))
                }
                master_report['output_files']['harvest_report'] = str(self.harvest_report_file)
            
            # Statistical analysis summary
            if self.statistical_report_file.exists():
                with open(self.statistical_report_file, 'r') as f:
                    stats_data = json.load(f)
                master_report['component_summaries']['statistical_analysis'] = {
                    'total_claims': stats_data.get('summary', {}).get('total_claims', 0),
                    'passed_claims': stats_data.get('summary', {}).get('passed_claims', 0),
                    'pass_rate': stats_data.get('summary', {}).get('pass_rate', 0),
                    'methodology': 'BCa bootstrap (10k samples) + FDR correction'
                }
                master_report['output_files']['statistical_report'] = str(self.statistical_report_file)
            
            # Pareto analysis summary
            if self.pareto_report_file.exists():
                with open(self.pareto_report_file, 'r') as f:
                    pareto_data = json.load(f)
                master_report['component_summaries']['pareto_analysis'] = {
                    'methods_analyzed': pareto_data.get('analysis_metadata', {}).get('total_methods', 0),
                    'pareto_optimal_methods': pareto_data.get('analysis_metadata', {}).get('pareto_optimal_methods', 0),
                    'best_overall': pareto_data.get('recommendations', {}).get('best_overall', {}).get('method', 'unknown')
                }
                master_report['output_files']['pareto_report'] = str(self.pareto_report_file)
                if self.pareto_plot_file.exists():
                    master_report['output_files']['pareto_plot'] = str(self.pareto_plot_file)
            
            # Specialized analyses summaries
            if specialized_results.get('router_audit', False) and self.router_audit_file.exists():
                with open(self.router_audit_file, 'r') as f:
                    router_data = json.load(f)
                master_report['component_summaries']['router_audit'] = {
                    'total_decisions': router_data.get('audit_metadata', {}).get('total_decisions', 0),
                    'specialization_detected': router_data.get('task_specialization', {}).get('specialization_detected', False),
                    'patterns_detected': len(router_data.get('detected_patterns', []))
                }
                master_report['output_files']['router_audit'] = str(self.router_audit_file)
            
            if specialized_results.get('soak_analysis', False) and self.soak_analysis_file.exists():
                with open(self.soak_analysis_file, 'r') as f:
                    soak_data = json.load(f)
                master_report['component_summaries']['soak_analysis'] = {
                    'runs_analyzed': soak_data.get('soak_analysis_metadata', {}).get('total_runs_analyzed', 0),
                    'stable_runs_rate': soak_data.get('aggregate_analysis', {}).get('stability_summary', {}).get('stable_runs_rate', 0),
                    'convergence_rate': soak_data.get('aggregate_analysis', {}).get('convergence_summary', {}).get('convergence_rate', 0)
                }
                master_report['output_files']['soak_analysis'] = str(self.soak_analysis_file)
            
            if specialized_results.get('hallucination_analysis', False) and self.hallucination_analysis_file.exists():
                with open(self.hallucination_analysis_file, 'r') as f:
                    hall_data = json.load(f)
                master_report['component_summaries']['hallucination_analysis'] = {
                    'instances_analyzed': hall_data.get('hallucination_analysis_metadata', {}).get('total_instances_analyzed', 0),
                    'safety_level': hall_data.get('safety_assessment', {}).get('safety_level', 'unknown'),
                    'safety_score': hall_data.get('safety_assessment', {}).get('safety_score', 0)
                }
                master_report['output_files']['hallucination_analysis'] = str(self.hallucination_analysis_file)
            
            if specialized_results.get('violation_analysis', False) and self.violation_analysis_file.exists():
                with open(self.violation_analysis_file, 'r') as f:
                    viol_data = json.load(f)
                master_report['component_summaries']['violation_analysis'] = {
                    'instances_analyzed': viol_data.get('violation_analysis_metadata', {}).get('total_instances_analyzed', 0),
                    'compliance_score': viol_data.get('constitutional_compliance', {}).get('overall_compliance_score', 1.0),
                    'critical_violations': viol_data.get('constitutional_compliance', {}).get('critical_violations', 0)
                }
                master_report['output_files']['violation_analysis'] = str(self.violation_analysis_file)
        
        except Exception as e:
            logger.warning(f"Error loading component summaries: {e}")
        
        # Generate overall assessment
        master_report['overall_assessment'] = self._generate_overall_assessment(master_report)
        
        # Generate recommendations
        master_report['recommendations'] = self._generate_master_recommendations(master_report)
        
        # Save master report
        with open(self.master_report_file, 'w') as f:
            json.dump(master_report, f, indent=2, default=str)
        
        logger.info(f"📋 Master T1 report generated: {self.master_report_file}")
        
        return master_report
    
    def _generate_overall_assessment(self, master_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall T1 workflow assessment."""
        
        execution_summary = master_report.get('execution_summary', {})
        component_summaries = master_report.get('component_summaries', {})
        
        # Count successful components
        successful_components = 0
        total_components = 0
        
        # Core components
        if execution_summary.get('T1_1_data_collection', False):
            successful_components += 1
        total_components += 1
        
        if execution_summary.get('T1_2_statistical_analysis', False):
            successful_components += 1
        total_components += 1
        
        # Specialized analyses
        specialized = execution_summary.get('T1_3_specialized_analyses', {})
        for analysis, success in specialized.items():
            if success:
                successful_components += 1
            total_components += 1
        
        # Success rate
        success_rate = successful_components / total_components if total_components > 0 else 0
        
        # Data quality assessment
        data_collection = component_summaries.get('data_collection', {})
        total_runs = data_collection.get('total_runs', 0)
        validation_rate = data_collection.get('validation_rate', 0)
        
        # Statistical rigor assessment
        stats = component_summaries.get('statistical_analysis', {})
        statistical_rigor = 'high' if stats.get('methodology') == 'BCa bootstrap (10k samples) + FDR correction' else 'unknown'
        
        # Safety assessment
        safety_scores = []
        hallucination = component_summaries.get('hallucination_analysis', {})
        if 'safety_score' in hallucination:
            safety_scores.append(hallucination['safety_score'])
        
        violation = component_summaries.get('violation_analysis', {})
        if 'compliance_score' in violation:
            safety_scores.append(violation['compliance_score'])
        
        overall_safety_score = np.mean(safety_scores) if safety_scores else None
        
        return {
            'workflow_success_rate': success_rate,
            'successful_components': successful_components,
            'total_components': total_components,
            'data_quality': {
                'total_runs_collected': total_runs,
                'validation_rate': validation_rate,
                'data_quality_score': validation_rate
            },
            'statistical_rigor': statistical_rigor,
            'safety_assessment': {
                'overall_safety_score': overall_safety_score,
                'safety_components_analyzed': len(safety_scores)
            },
            'completeness_score': success_rate,
            'overall_grade': self._compute_overall_grade(success_rate, validation_rate, overall_safety_score)
        }
    
    def _compute_overall_grade(self, 
                              success_rate: float, 
                              validation_rate: float, 
                              safety_score: Optional[float]) -> str:
        """Compute overall workflow grade."""
        
        # Weight different factors
        weighted_score = success_rate * 0.4  # Workflow completion
        
        if validation_rate is not None:
            weighted_score += validation_rate * 0.3  # Data quality
        
        if safety_score is not None:
            weighted_score += safety_score * 0.3  # Safety assessment
        else:
            weighted_score += 0.15  # Partial credit for missing safety
        
        # Convert to letter grade
        if weighted_score >= 0.9:
            return 'A'
        elif weighted_score >= 0.8:
            return 'B'
        elif weighted_score >= 0.7:
            return 'C'
        elif weighted_score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _generate_master_recommendations(self, master_report: Dict[str, Any]) -> List[str]:
        """Generate master-level recommendations."""
        
        recommendations = []
        
        execution_summary = master_report.get('execution_summary', {})
        component_summaries = master_report.get('component_summaries', {})
        overall_assessment = master_report.get('overall_assessment', {})
        
        # Workflow completeness
        success_rate = overall_assessment.get('workflow_success_rate', 0)
        if success_rate < 0.8:
            recommendations.append(
                f"WORKFLOW COMPLETION: Only {success_rate:.1%} of T1 components completed successfully. "
                f"Investigate and resolve component failures for comprehensive analysis."
            )
        
        # Data quality
        data_quality = overall_assessment.get('data_quality', {})
        validation_rate = data_quality.get('validation_rate', 0)
        
        if validation_rate < 0.8:
            recommendations.append(
                f"DATA QUALITY: Validation rate is {validation_rate:.1%}. "
                f"Improve data collection and validation processes for more reliable analysis."
            )
        
        # Statistical analysis
        stats = component_summaries.get('statistical_analysis', {})
        if stats:
            pass_rate = stats.get('pass_rate', 0)
            if pass_rate < 0.5:
                recommendations.append(
                    f"STATISTICAL SIGNIFICANCE: Only {pass_rate:.1%} of statistical claims passed. "
                    f"Review experimental design and increase sample sizes for stronger evidence."
                )
        
        # Safety assessment
        safety = overall_assessment.get('safety_assessment', {})
        safety_score = safety.get('overall_safety_score')
        
        if safety_score is not None and safety_score < 0.8:
            recommendations.append(
                f"SAFETY CONCERNS: Overall safety score is {safety_score:.2f}. "
                f"Address hallucination and constitutional violation issues before deployment."
            )
        
        # Specialized analysis recommendations
        if not execution_summary.get('T1_3_specialized_analyses', {}).get('router_audit', False):
            recommendations.append(
                "ROUTER INTERPRETABILITY: Router audit analysis failed. "
                "Ensure routing decision data is properly logged for interpretability analysis."
            )
        
        if not execution_summary.get('T1_3_specialized_analyses', {}).get('soak_analysis', False):
            recommendations.append(
                "STABILITY ANALYSIS: Soak analysis failed. "
                "Verify online learning logs are available for stability assessment."
            )
        
        # Performance optimization
        pareto = component_summaries.get('pareto_analysis', {})
        if pareto and pareto.get('pareto_optimal_methods', 0) == 0:
            recommendations.append(
                "PERFORMANCE OPTIMIZATION: No Pareto-optimal methods identified. "
                "Review performance vs resource trade-offs and optimize model configurations."
            )
        
        # Overall grade-based recommendations
        overall_grade = overall_assessment.get('overall_grade', 'F')
        if overall_grade in ['D', 'F']:
            recommendations.append(
                f"COMPREHENSIVE REVIEW NEEDED: Overall T1 workflow grade is {overall_grade}. "
                f"Conduct systematic review of experimental setup, data collection, and analysis methods."
            )
        
        return recommendations[:10]  # Top 10 recommendations
    
    def print_final_summary(self, master_report: Dict[str, Any]):
        """Print final T1 workflow summary."""
        
        print("\n" + "="*80)
        print("BEM 2.0 T1 TRACKING WORKFLOW - FINAL SUMMARY")
        print("="*80)
        
        # Execution summary
        execution = master_report.get('execution_summary', {})
        print(f"\n📋 Workflow Execution:")
        print(f"  ✅ T1.1 Data Collection: {'Success' if execution.get('T1_1_data_collection', False) else 'Failed'}")
        print(f"  ✅ T1.2 Statistical Analysis: {'Success' if execution.get('T1_2_statistical_analysis', False) else 'Failed'}")
        print(f"  ✅ Pareto Analysis: {'Success' if execution.get('pareto_analysis', False) else 'Failed'}")
        
        specialized = execution.get('T1_3_specialized_analyses', {})
        print(f"  📊 T1.3 Specialized Analyses:")
        for analysis, success in specialized.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"    {status}: {analysis.replace('_', ' ').title()}")
        
        # Data summary
        data_collection = master_report.get('component_summaries', {}).get('data_collection', {})
        if data_collection:
            print(f"\n📊 Data Collection Summary:")
            print(f"  Total Runs: {data_collection.get('total_runs', 0):,}")
            print(f"  Validation Rate: {data_collection.get('validation_rate', 0):.1%}")
            print(f"  Pillars Covered: {data_collection.get('pillars_covered', 0)}")
        
        # Statistical summary
        stats = master_report.get('component_summaries', {}).get('statistical_analysis', {})
        if stats:
            print(f"\n📈 Statistical Analysis Summary:")
            print(f"  Claims Tested: {stats.get('total_claims', 0)}")
            print(f"  Claims Passed: {stats.get('passed_claims', 0)}")
            print(f"  Pass Rate: {stats.get('pass_rate', 0):.1%}")
            print(f"  Methodology: {stats.get('methodology', 'Unknown')}")
        
        # Overall assessment
        assessment = master_report.get('overall_assessment', {})
        print(f"\n🎯 Overall Assessment:")
        print(f"  Workflow Success Rate: {assessment.get('workflow_success_rate', 0):.1%}")
        print(f"  Completeness Score: {assessment.get('completeness_score', 0):.1%}")
        print(f"  Overall Grade: {assessment.get('overall_grade', 'N/A')}")
        
        # Safety assessment
        safety = assessment.get('safety_assessment', {})
        if safety.get('overall_safety_score') is not None:
            print(f"  Safety Score: {safety['overall_safety_score']:.2f}/1.00")
        
        # Key recommendations
        recommendations = master_report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Key Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Output files
        output_files = master_report.get('output_files', {})
        print(f"\n📁 Generated Outputs ({len(output_files)} files):")
        print(f"  📋 Master Report: {master_report.get('T1_workflow_metadata', {}).get('analysis_directory', 'unknown')}/T1_master_report_bem2.json")
        
        for file_type, file_path in output_files.items():
            print(f"  📄 {file_type.replace('_', ' ').title()}: {file_path}")
        
        # Final status
        overall_grade = assessment.get('overall_grade', 'F')
        success_rate = assessment.get('workflow_success_rate', 0)
        
        print(f"\n🎉 BEM 2.0 T1 Tracking Workflow Complete!")
        print(f"   Grade: {overall_grade} | Success Rate: {success_rate:.1%}")
        
        if overall_grade in ['A', 'B'] and success_rate > 0.8:
            print(f"   ✅ Excellent workflow execution with comprehensive analysis")
        elif overall_grade in ['B', 'C'] and success_rate > 0.6:
            print(f"   ⚠️  Good execution with some areas for improvement")
        else:
            print(f"   ❌ Significant issues detected - review recommendations")
    
    def run_complete_t1_workflow(self, 
                                pillars: Optional[List[str]] = None,
                                primary_metric: str = "F1",
                                skip_specialized: bool = False) -> bool:
        """
        Execute the complete T1 tracking workflow.
        
        Args:
            pillars: Optional list of specific pillars to analyze
            primary_metric: Primary metric for Pareto analysis
            skip_specialized: Skip specialized analyses (faster execution)
            
        Returns:
            Overall success status
        """
        workflow_start_time = time.time()
        
        print("\n" + "="*80)
        print("🚀 STARTING BEM 2.0 T1 TRACKING WORKFLOW")
        print("="*80)
        print(f"📅 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Primary metric: {primary_metric}")
        print(f"🎯 Pillars: {pillars if pillars else 'All available'}")
        print(f"⚡ Skip specialized: {skip_specialized}")
        
        # Track component success
        component_success = {
            'T1_1': False,
            'T1_2': False,
            'pareto': False,
            'T1_3': {}
        }
        
        # T1.1: Data Collection
        component_success['T1_1'] = self.execute_t1_1_data_collection(pillars)
        if not component_success['T1_1']:
            logger.error("❌ T1.1 failed - cannot proceed with analysis")
            return False
        
        # T1.2: Statistical Analysis
        component_success['T1_2'] = self.execute_t1_2_statistical_analysis()
        
        # Pareto Analysis
        component_success['pareto'] = self.execute_pareto_analysis(primary_metric)
        
        # T1.3: Specialized Analyses
        if not skip_specialized:
            component_success['T1_3'] = self.execute_t1_3_specialized_analyses()
        else:
            logger.info("⏭️  Skipping specialized analyses (skip_specialized=True)")
            component_success['T1_3'] = {}
        
        # Generate master report
        master_report = self.generate_master_report(component_success['T1_3'])
        
        # Calculate total time
        workflow_end_time = time.time()
        total_time = workflow_end_time - workflow_start_time
        
        # Print final summary
        self.print_final_summary(master_report)
        
        print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
        print(f"📋 Master report: {self.master_report_file}")
        
        # Determine overall success
        overall_assessment = master_report.get('overall_assessment', {})
        success_rate = overall_assessment.get('workflow_success_rate', 0)
        overall_success = success_rate >= 0.6  # At least 60% components successful
        
        if overall_success:
            logger.info("🎉 BEM 2.0 T1 Tracking Workflow completed successfully!")
        else:
            logger.warning("⚠️ BEM 2.0 T1 Tracking Workflow completed with issues")
        
        return overall_success


def main():
    """Main entry point for T1 workflow orchestrator."""
    
    parser = argparse.ArgumentParser(
        description='BEM 2.0 Complete T1 Tracking Workflow Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete T1 workflow on all pillars
  python run_t1_workflow.py
  
  # Run on specific pillars only
  python run_t1_workflow.py --pillars AR0 MM0 VC0
  
  # Skip specialized analyses for faster execution
  python run_t1_workflow.py --skip-specialized
  
  # Use different primary metric for Pareto analysis
  python run_t1_workflow.py --primary-metric BLEU
  
  # Run only data collection
  python run_t1_workflow.py --t1-1-only
        """
    )
    
    parser.add_argument('--results-dir', default='results',
                       help='Base directory containing pillar results')
    parser.add_argument('--analysis-dir', default='analysis',
                       help='Directory for analysis outputs')
    parser.add_argument('--output-prefix', default='bem2',
                       help='Prefix for output files')
    parser.add_argument('--pillars', nargs='+',
                       choices=['AR0', 'AR1', 'OL0', 'MM0', 'VC0', 'PT1', 'PT2', 'PT3', 'PT4'],
                       help='Specific pillars to analyze')
    parser.add_argument('--primary-metric', default='F1',
                       choices=['F1', 'EM', 'BLEU', 'chrF', 'accuracy'],
                       help='Primary metric for Pareto analysis')
    parser.add_argument('--skip-specialized', action='store_true',
                       help='Skip specialized analyses for faster execution')
    parser.add_argument('--t1-1-only', action='store_true',
                       help='Run only T1.1 data collection')
    parser.add_argument('--t1-2-only', action='store_true',
                       help='Run only T1.2 statistical analysis (requires existing consolidated runs)')
    parser.add_argument('--t1-3-only', action='store_true',
                       help='Run only T1.3 specialized analyses (requires existing consolidated runs)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = T1WorkflowOrchestrator(
        results_dir=args.results_dir,
        analysis_dir=args.analysis_dir,
        output_prefix=args.output_prefix
    )
    
    try:
        if args.t1_1_only:
            # Run only data collection
            success = orchestrator.execute_t1_1_data_collection(args.pillars)
            print(f"\n🎯 T1.1 Data Collection: {'✅ Success' if success else '❌ Failed'}")
            
        elif args.t1_2_only:
            # Run only statistical analysis
            success = orchestrator.execute_t1_2_statistical_analysis()
            print(f"\n🎯 T1.2 Statistical Analysis: {'✅ Success' if success else '❌ Failed'}")
            
        elif args.t1_3_only:
            # Run only specialized analyses
            results = orchestrator.execute_t1_3_specialized_analyses()
            successful = sum(results.values())
            total = len(results)
            print(f"\n🎯 T1.3 Specialized Analyses: {successful}/{total} successful")
            
        else:
            # Run complete workflow
            success = orchestrator.run_complete_t1_workflow(
                pillars=args.pillars,
                primary_metric=args.primary_metric,
                skip_specialized=args.skip_specialized
            )
            
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Workflow failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()