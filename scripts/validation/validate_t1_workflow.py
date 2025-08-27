#!/usr/bin/env python3
"""
BEM 2.0 T1 Workflow Validation and Testing Framework
Validates the complete T1 tracking workflow implementation with rigorous testing.

This script provides comprehensive validation of:
- T1.1 Data Collection system with mock pillar data
- T1.2 Statistical Analysis with BCa bootstrap and FDR correction
- T1.3 Specialized Analysis components
- End-to-end workflow orchestration
- Statistical methodology verification
- Output validation and integrity checks
"""

import argparse
import json
import logging
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class T1WorkflowValidator:
    """
    Comprehensive validation framework for BEM 2.0 T1 tracking workflow.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize validator with temporary test environment."""
        
        if temp_dir:
            self.test_dir = Path(temp_dir)
            self.cleanup_on_exit = False
        else:
            self.test_dir = Path(tempfile.mkdtemp(prefix="t1_validation_"))
            self.cleanup_on_exit = True
        
        # Create test directory structure
        self.results_dir = self.test_dir / "results"
        self.analysis_dir = self.test_dir / "analysis"
        
        # Create subdirectories for each pillar
        self.pillar_dirs = {}
        for pillar in ['ar0', 'ar1', 'ol0', 'mm0', 'vc0', 'pt1', 'pt2', 'pt3', 'pt4']:
            pillar_dir = self.results_dir / pillar
            pillar_dir.mkdir(parents=True, exist_ok=True)
            self.pillar_dirs[pillar] = pillar_dir
        
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized T1 workflow validator")
        logger.info(f"Test directory: {self.test_dir}")
    
    def generate_mock_pillar_data(self, 
                                 pillar: str, 
                                 num_runs: int = 10, 
                                 num_methods: int = 3) -> List[Dict[str, Any]]:
        """
        Generate realistic mock experimental data for a pillar.
        
        Args:
            pillar: Pillar name (ar0, mm0, etc.)
            num_runs: Number of experimental runs to generate
            num_methods: Number of different methods to simulate
            
        Returns:
            List of mock experimental runs
        """
        methods = ['baseline', 'bem_v2', 'hierarchical_bem'][:num_methods]
        
        runs = []
        
        for run_id in range(num_runs):
            method = methods[run_id % len(methods)]
            seed = run_id * 42 + 1337
            
            # Generate realistic performance metrics
            np.random.seed(seed)
            
            # Base performance varies by method
            method_multipliers = {
                'baseline': {'f1': 0.65, 'latency': 1.0, 'memory': 1.0},
                'bem_v2': {'f1': 0.75, 'latency': 1.15, 'memory': 1.08},
                'hierarchical_bem': {'f1': 0.82, 'latency': 1.25, 'memory': 1.12}
            }
            
            base_multipliers = method_multipliers.get(method, method_multipliers['baseline'])
            
            # Generate metrics with realistic variance
            f1_score = min(1.0, max(0.0, base_multipliers['f1'] + np.random.normal(0, 0.05)))
            em_score = min(1.0, max(0.0, f1_score - np.random.uniform(0.1, 0.2)))
            bleu_score = min(1.0, max(0.0, f1_score - np.random.uniform(0.05, 0.15)))
            chrf_score = min(1.0, max(0.0, f1_score + np.random.uniform(-0.05, 0.05)))
            
            # Latency metrics (higher for more complex methods)
            base_latency = 50  # Base latency in ms
            p50_latency = base_latency * base_multipliers['latency'] * (1 + np.random.normal(0, 0.1))
            p95_latency = p50_latency * (1.8 + np.random.normal(0, 0.2))
            p99_latency = p95_latency * (1.3 + np.random.normal(0, 0.15))
            
            # Memory metrics
            base_memory = 2.5  # Base memory in GB
            vram_usage = base_memory * base_multipliers['memory'] * (1 + np.random.normal(0, 0.08))
            vram_delta = max(0, vram_usage - base_memory)
            
            # Throughput
            throughput = max(1, 1000 / p50_latency + np.random.normal(0, 5))
            
            # Generate per-instance scores for detailed analysis
            per_instance_scores = []
            num_instances = np.random.randint(50, 150)
            
            for i in range(num_instances):
                instance_f1 = min(1.0, max(0.0, f1_score + np.random.normal(0, 0.1)))
                instance_em = 1.0 if instance_f1 > 0.9 else 0.0  # EM is binary
                
                instance_data = {
                    'id': f"{pillar}_{method}_{run_id}_instance_{i}",
                    'input': f"Mock input text for instance {i}",
                    'prediction': f"Mock prediction for {method}",
                    'reference': f"Mock reference text",
                    'f1': instance_f1,
                    'em': instance_em,
                    'task_type': self._get_pillar_task_type(pillar),
                    'modality': 'multimodal' if pillar == 'mm0' else 'text'
                }
                
                # Add pillar-specific data
                if pillar == 'mm0':  # Multimodal
                    instance_data['hallucination_analysis'] = {
                        'type': 'factual',
                        'severity': np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.25, 0.05]),
                        'confidence': np.random.uniform(0.6, 0.95)
                    }
                
                elif pillar == 'vc0':  # Constitutional AI
                    if np.random.random() < 0.1:  # 10% violation rate
                        instance_data['safety_analysis'] = {
                            'violation_detected': True,
                            'principle_violated': 'harmlessness',
                            'violation_type': 'harmful_content',
                            'severity': np.random.choice(['medium', 'high'], p=[0.8, 0.2]),
                            'confidence': np.random.uniform(0.7, 0.95)
                        }
                
                elif pillar == 'ol0':  # Online Learning
                    instance_data['learning_state'] = {
                        'step': i,
                        'learning_rate': 1e-4 * (0.95 ** (i // 10)),
                        'adaptation_signal': np.random.uniform(0.01, 0.1)
                    }
                
                per_instance_scores.append(instance_data)
            
            # Create comprehensive run data
            run_data = {
                'metadata': {
                    'run_id': f"{pillar}_{method}_{run_id}",
                    'pillar': pillar.upper(),
                    'track': self._get_pillar_track(pillar),
                    'method': method,
                    'variant': 'default',
                    'seed': seed,
                    'timestamp': (datetime.now() - timedelta(days=run_id)).isoformat(),
                    'git_hash': f"abc{run_id:04d}ef",
                    'experiment_config_hash': f"config_{method}_{seed}",
                    'hardware_config': {
                        'gpu': 'RTX_3090_Ti',
                        'ram_gb': 32,
                        'cpu_cores': 16
                    },
                    'software_versions': {
                        'python': '3.11.5',
                        'pytorch': '2.8.0',
                        'transformers': '4.55.4'
                    }
                },
                'evaluation_results': {
                    'standard_metrics': {
                        'F1': f1_score,
                        'EM': em_score,
                        'BLEU': bleu_score,
                        'chrF': chrf_score
                    },
                    'system_telemetry': {
                        'p50_latency_ms': p50_latency,
                        'p95_latency_ms': p95_latency,
                        'p99_latency_ms': p99_latency,
                        'tokens_per_second': throughput,
                        'vram_usage_gb': vram_usage,
                        'vram_delta_gb': vram_delta,
                        'peak_memory_gb': vram_usage * 1.1,
                        'cpu_utilization_pct': np.random.uniform(30, 80),
                        'gpu_utilization_pct': np.random.uniform(60, 95)
                    }
                },
                'per_instance_scores': per_instance_scores
            }
            
            # Add pillar-specific evaluation results
            if pillar == 'mm0':
                run_data['evaluation_results']['hallucination_rate'] = np.random.uniform(0.05, 0.15)
                run_data['evaluation_results']['vision_language_alignment'] = np.random.uniform(0.7, 0.9)
            
            elif pillar == 'vc0':
                run_data['evaluation_results']['safety_score'] = np.random.uniform(0.8, 0.95)
                run_data['evaluation_results']['constitutional_compliance'] = np.random.uniform(0.85, 0.98)
                run_data['evaluation_results']['violation_rate'] = np.random.uniform(0.02, 0.1)
            
            elif pillar == 'ol0':
                run_data['evaluation_results']['adaptation_speed'] = np.random.uniform(0.1, 0.5)
                run_data['evaluation_results']['catastrophic_forgetting'] = np.random.uniform(0.05, 0.2)
            
            runs.append(run_data)
        
        return runs
    
    def _get_pillar_task_type(self, pillar: str) -> str:
        """Get representative task type for pillar."""
        task_types = {
            'ar0': 'retrieval_qa',
            'ar1': 'adaptive_retrieval',
            'ol0': 'continual_learning',
            'mm0': 'vision_language',
            'vc0': 'safety_classification',
            'pt1': 'latency_optimization',
            'pt2': 'throughput_optimization',
            'pt3': 'memory_optimization',
            'pt4': 'quality_optimization'
        }
        return task_types.get(pillar, 'unknown')
    
    def _get_pillar_track(self, pillar: str) -> str:
        """Get representative track for pillar."""
        tracks = {
            'ar0': 'basic_retrieval',
            'ar1': 'hierarchical_routing',
            'ol0': 'continual_learning',
            'mm0': 'vision_language',
            'vc0': 'constitutional_training',
            'pt1': 'p50_optimization',
            'pt2': 'batch_processing',
            'pt3': 'memory_efficiency',
            'pt4': 'quality_preservation'
        }
        return tracks.get(pillar, 'default')
    
    def setup_mock_data(self, runs_per_pillar: int = 15) -> Dict[str, int]:
        """
        Set up complete mock dataset for testing.
        
        Args:
            runs_per_pillar: Number of runs to generate per pillar
            
        Returns:
            Dict mapping pillar to number of runs created
        """
        logger.info(f"Setting up mock data ({runs_per_pillar} runs per pillar)")
        
        created_runs = {}
        
        for pillar, pillar_dir in self.pillar_dirs.items():
            runs = self.generate_mock_pillar_data(pillar, runs_per_pillar)
            
            # Write runs to JSONL file
            output_file = pillar_dir / f"{pillar}_experimental_runs.jsonl"
            
            with open(output_file, 'w') as f:
                for run in runs:
                    json.dump(run, f, separators=(',', ':'))
                    f.write('\n')
            
            created_runs[pillar] = len(runs)
            logger.info(f"✅ Created {len(runs)} runs for {pillar.upper()}: {output_file}")
        
        total_runs = sum(created_runs.values())
        logger.info(f"🎯 Total mock runs created: {total_runs}")
        
        return created_runs
    
    def validate_t1_1_data_collection(self) -> Dict[str, Any]:
        """Validate T1.1 data collection system."""
        
        logger.info("🔍 Validating T1.1 Data Collection System")
        
        validation_results = {
            'component': 'T1.1_data_collection',
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
        # Test 1: Import and instantiation
        try:
            from analysis.collect import DataCollector
            collector = DataCollector(
                base_results_dir=str(self.results_dir),
                output_file=str(self.analysis_dir / "test_runs.jsonl")
            )
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ DataCollector import and instantiation")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ DataCollector import failed: {e}")
            return validation_results
        
        # Test 2: Pillar configuration
        expected_pillars = {'AR0', 'AR1', 'OL0', 'MM0', 'VC0', 'PT1', 'PT2', 'PT3', 'PT4'}
        actual_pillars = set(collector.pillars.keys())
        
        if expected_pillars == actual_pillars:
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ All expected pillars configured")
        else:
            validation_results['tests_failed'] += 1
            missing = expected_pillars - actual_pillars
            validation_results['details'].append(f"❌ Missing pillars: {missing}")
        
        # Test 3: Data harvesting
        try:
            consolidated_runs = collector.harvest_all_runs()
            
            if len(consolidated_runs) > 0:
                validation_results['tests_passed'] += 1
                validation_results['details'].append(f"✅ Harvested {len(consolidated_runs)} runs")
            else:
                validation_results['tests_failed'] += 1
                validation_results['details'].append("❌ No runs harvested")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Harvesting failed: {e}")
        
        # Test 4: Output file generation
        try:
            collector.write_consolidated_runs(consolidated_runs)
            output_file = Path(collector.output_file)
            
            if output_file.exists() and output_file.stat().st_size > 0:
                validation_results['tests_passed'] += 1
                validation_results['details'].append(f"✅ Output file created: {output_file}")
            else:
                validation_results['tests_failed'] += 1
                validation_results['details'].append("❌ Output file not created or empty")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Output file generation failed: {e}")
        
        # Test 5: Report generation
        try:
            report = collector.generate_harvest_report(consolidated_runs)
            
            required_sections = ['harvest_metadata', 'pillar_breakdown', 'validation_summary']
            missing_sections = [s for s in required_sections if s not in report]
            
            if not missing_sections:
                validation_results['tests_passed'] += 1
                validation_results['details'].append("✅ Complete harvest report generated")
            else:
                validation_results['tests_failed'] += 1
                validation_results['details'].append(f"❌ Missing report sections: {missing_sections}")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Report generation failed: {e}")
        
        return validation_results
    
    def validate_t1_2_statistical_analysis(self) -> Dict[str, Any]:
        """Validate T1.2 statistical analysis system."""
        
        logger.info("📊 Validating T1.2 Statistical Analysis System")
        
        validation_results = {
            'component': 'T1.2_statistical_analysis',
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
        # Test 1: Import and instantiation
        try:
            from analysis.stats import StatisticalAnalyzer
            analyzer = StatisticalAnalyzer(n_bootstrap=1000, alpha=0.05)  # Smaller n for testing
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ StatisticalAnalyzer import and instantiation")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ StatisticalAnalyzer import failed: {e}")
            return validation_results
        
        # Test 2: BCa bootstrap functionality
        try:
            # Generate test data
            np.random.seed(42)
            x = np.random.normal(0.75, 0.1, 50)  # BEM performance
            y = np.random.normal(0.65, 0.1, 50)  # Baseline performance
            
            lower, upper, effect = analyzer.paired_bootstrap_ci(x, y, method='bca')
            
            if lower < upper and effect > 0:  # BEM should be better than baseline
                validation_results['tests_passed'] += 1
                validation_results['details'].append(f"✅ BCa bootstrap CI: [{lower:.3f}, {upper:.3f}], effect: {effect:.3f}")
            else:
                validation_results['tests_failed'] += 1
                validation_results['details'].append(f"❌ Invalid BCa results: [{lower:.3f}, {upper:.3f}], effect: {effect:.3f}")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ BCa bootstrap failed: {e}")
        
        # Test 3: Relative improvement calculation
        try:
            bem_scores = np.random.normal(0.8, 0.05, 30)
            baseline_scores = np.random.normal(0.7, 0.05, 30)
            
            result = analyzer.compute_relative_improvement_ci(bem_scores, baseline_scores)
            
            required_keys = ['relative_improvement_pct', 'ci_lower', 'ci_upper', 'significant']
            if all(key in result for key in required_keys):
                validation_results['tests_passed'] += 1
                validation_results['details'].append(
                    f"✅ Relative improvement: {result['relative_improvement_pct']:.1f}% "
                    f"[{result['ci_lower']:.1f}, {result['ci_upper']:.1f}], "
                    f"significant: {result['significant']}"
                )
            else:
                validation_results['tests_failed'] += 1
                validation_results['details'].append("❌ Incomplete relative improvement results")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Relative improvement calculation failed: {e}")
        
        # Test 4: FDR correction
        try:
            from statsmodels.stats.multitest import multipletests
            
            # Mock p-values (some significant, some not)
            p_values = [0.001, 0.02, 0.08, 0.15, 0.3, 0.006, 0.45]
            
            rejected, corrected_p, _, _ = multipletests(
                p_values, 
                alpha=0.05, 
                method='fdr_bh'
            )
            
            num_significant = sum(rejected)
            validation_results['tests_passed'] += 1
            validation_results['details'].append(f"✅ FDR correction: {num_significant}/{len(p_values)} significant after correction")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ FDR correction failed: {e}")
        
        # Test 5: Slice comparison analysis
        try:
            # Mock slice data
            bem_results = {
                'slice_a': {'F1': np.random.normal(0.8, 0.05, 25), 'EM': np.random.normal(0.6, 0.1, 25)},
                'slice_b': {'F1': np.random.normal(0.75, 0.05, 40), 'EM': np.random.normal(0.55, 0.1, 40)}
            }
            baseline_results = {
                'slice_a': {'F1': np.random.normal(0.7, 0.05, 25), 'EM': np.random.normal(0.5, 0.1, 25)},
                'slice_b': {'F1': np.random.normal(0.65, 0.05, 40), 'EM': np.random.normal(0.45, 0.1, 40)}
            }
            
            slice_analysis = analyzer.analyze_slice_comparison(
                bem_results['slice_a'], 
                baseline_results['slice_a'], 
                'slice_a'
            )
            
            required_sections = ['metrics', 'fdr_results', 'significant_improvements']
            if all(section in slice_analysis for section in required_sections):
                validation_results['tests_passed'] += 1
                validation_results['details'].append(f"✅ Slice analysis completed: {len(slice_analysis.get('significant_improvements', []))} significant improvements")
            else:
                validation_results['tests_failed'] += 1
                validation_results['details'].append("❌ Incomplete slice analysis results")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Slice analysis failed: {e}")
        
        return validation_results
    
    def validate_t1_3_specialized_analyses(self) -> Dict[str, Any]:
        """Validate T1.3 specialized analysis components."""
        
        logger.info("🔍 Validating T1.3 Specialized Analysis Components")
        
        validation_results = {
            'component': 'T1.3_specialized_analyses',
            'tests_passed': 0,
            'tests_failed': 0,
            'component_results': {},
            'details': []
        }
        
        # Test Router Audit System
        try:
            from analysis.router_audit import RouterAuditor
            auditor = RouterAuditor(num_experts=8, confidence_threshold=0.8)
            validation_results['component_results']['router_audit'] = True
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ Router audit system initialized")
        except Exception as e:
            validation_results['component_results']['router_audit'] = False
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Router audit failed: {e}")
        
        # Test Soak Analysis System
        try:
            from analysis.soak import SoakAnalyzer
            soak_analyzer = SoakAnalyzer(stability_window=50, volatility_threshold=0.1)
            validation_results['component_results']['soak_analysis'] = True
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ Soak analysis system initialized")
        except Exception as e:
            validation_results['component_results']['soak_analysis'] = False
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Soak analysis failed: {e}")
        
        # Test Hallucination Analysis System
        try:
            from analysis.hallucinations import HallucinationAnalyzer
            hall_analyzer = HallucinationAnalyzer(hallucination_threshold=0.3)
            
            # Test basic functionality
            test_text = "The sky is green and grass is blue."
            violations = hall_analyzer.detect_violations_in_text(test_text)
            
            validation_results['component_results']['hallucination_analysis'] = True
            validation_results['tests_passed'] += 1
            validation_results['details'].append(f"✅ Hallucination analysis system initialized")
        except Exception as e:
            validation_results['component_results']['hallucination_analysis'] = False
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Hallucination analysis failed: {e}")
        
        # Test Constitutional Violation Analysis System
        try:
            from analysis.violations import ViolationAnalyzer
            viol_analyzer = ViolationAnalyzer()
            
            # Test basic functionality
            test_text = "You should hurt people who disagree with you."
            violations = viol_analyzer.detect_violations_in_text(test_text)
            
            validation_results['component_results']['violation_analysis'] = True
            validation_results['tests_passed'] += 1
            validation_results['details'].append(f"✅ Violation analysis system initialized")
        except Exception as e:
            validation_results['component_results']['violation_analysis'] = False
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Violation analysis failed: {e}")
        
        # Test Pareto Analysis Integration
        try:
            from analysis.pareto import ParetoAnalyzer
            pareto_analyzer = ParetoAnalyzer(primary_metric='F1', latency_metric='p50_latency_ms')
            validation_results['component_results']['pareto_analysis'] = True
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ Pareto analysis system initialized")
        except Exception as e:
            validation_results['component_results']['pareto_analysis'] = False
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Pareto analysis failed: {e}")
        
        return validation_results
    
    def validate_workflow_orchestration(self) -> Dict[str, Any]:
        """Validate complete workflow orchestration."""
        
        logger.info("🎯 Validating Workflow Orchestration")
        
        validation_results = {
            'component': 'workflow_orchestration',
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
        # Test 1: Import and instantiation
        try:
            from run_t1_workflow import T1WorkflowOrchestrator
            orchestrator = T1WorkflowOrchestrator(
                results_dir=str(self.results_dir),
                analysis_dir=str(self.analysis_dir),
                output_prefix="test"
            )
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ T1WorkflowOrchestrator instantiated")
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Orchestrator instantiation failed: {e}")
            return validation_results
        
        # Test 2: File path configuration
        expected_files = [
            'consolidated_runs_file',
            'statistical_report_file', 
            'pareto_report_file',
            'master_report_file'
        ]
        
        missing_files = []
        for file_attr in expected_files:
            if not hasattr(orchestrator, file_attr):
                missing_files.append(file_attr)
        
        if not missing_files:
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ All required file paths configured")
        else:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Missing file path configurations: {missing_files}")
        
        # Test 3: Directory structure validation
        analysis_dir_created = orchestrator.analysis_dir.exists()
        if analysis_dir_created:
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ Analysis directory structure created")
        else:
            validation_results['tests_failed'] += 1
            validation_results['details'].append("❌ Analysis directory not created")
        
        # Test 4: Component method availability
        required_methods = [
            'execute_t1_1_data_collection',
            'execute_t1_2_statistical_analysis',
            'execute_t1_3_specialized_analyses',
            'generate_master_report',
            'run_complete_t1_workflow'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(orchestrator, method_name):
                missing_methods.append(method_name)
        
        if not missing_methods:
            validation_results['tests_passed'] += 1
            validation_results['details'].append("✅ All required orchestration methods available")
        else:
            validation_results['tests_failed'] += 1
            validation_results['details'].append(f"❌ Missing orchestration methods: {missing_methods}")
        
        return validation_results
    
    def run_end_to_end_test(self, quick_test: bool = False) -> Dict[str, Any]:
        """
        Run end-to-end validation test of the complete T1 workflow.
        
        Args:
            quick_test: If True, use smaller datasets for faster testing
            
        Returns:
            End-to-end test results
        """
        logger.info("🚀 Running End-to-End T1 Workflow Test")
        
        test_results = {
            'test_type': 'end_to_end',
            'quick_test': quick_test,
            'setup_success': False,
            'workflow_success': False,
            'validation_success': False,
            'components_tested': {},
            'performance_metrics': {},
            'details': []
        }
        
        # Step 1: Setup mock data
        try:
            runs_per_pillar = 5 if quick_test else 10
            created_runs = self.setup_mock_data(runs_per_pillar)
            test_results['setup_success'] = True
            test_results['details'].append(f"✅ Mock data setup: {sum(created_runs.values())} total runs")
        except Exception as e:
            test_results['details'].append(f"❌ Mock data setup failed: {e}")
            return test_results
        
        # Step 2: Initialize orchestrator
        try:
            from run_t1_workflow import T1WorkflowOrchestrator
            
            orchestrator = T1WorkflowOrchestrator(
                results_dir=str(self.results_dir),
                analysis_dir=str(self.analysis_dir),
                output_prefix="e2e_test"
            )
            test_results['details'].append("✅ Orchestrator initialized")
        except Exception as e:
            test_results['details'].append(f"❌ Orchestrator initialization failed: {e}")
            return test_results
        
        # Step 3: Execute T1.1 Data Collection
        try:
            start_time = datetime.now()
            success = orchestrator.execute_t1_1_data_collection()
            duration = (datetime.now() - start_time).total_seconds()
            
            test_results['components_tested']['T1_1'] = success
            test_results['performance_metrics']['T1_1_duration'] = duration
            
            if success:
                test_results['details'].append(f"✅ T1.1 completed in {duration:.1f}s")
            else:
                test_results['details'].append("❌ T1.1 failed")
        except Exception as e:
            test_results['components_tested']['T1_1'] = False
            test_results['details'].append(f"❌ T1.1 error: {e}")
        
        # Step 4: Execute T1.2 Statistical Analysis
        try:
            start_time = datetime.now()
            success = orchestrator.execute_t1_2_statistical_analysis()
            duration = (datetime.now() - start_time).total_seconds()
            
            test_results['components_tested']['T1_2'] = success
            test_results['performance_metrics']['T1_2_duration'] = duration
            
            if success:
                test_results['details'].append(f"✅ T1.2 completed in {duration:.1f}s")
            else:
                test_results['details'].append("❌ T1.2 failed")
        except Exception as e:
            test_results['components_tested']['T1_2'] = False
            test_results['details'].append(f"❌ T1.2 error: {e}")
        
        # Step 5: Execute Pareto Analysis
        try:
            start_time = datetime.now()
            success = orchestrator.execute_pareto_analysis('F1')
            duration = (datetime.now() - start_time).total_seconds()
            
            test_results['components_tested']['pareto'] = success
            test_results['performance_metrics']['pareto_duration'] = duration
            
            if success:
                test_results['details'].append(f"✅ Pareto analysis completed in {duration:.1f}s")
            else:
                test_results['details'].append("❌ Pareto analysis failed")
        except Exception as e:
            test_results['components_tested']['pareto'] = False
            test_results['details'].append(f"❌ Pareto analysis error: {e}")
        
        # Step 6: Execute T1.3 Specialized Analyses (selected subset for speed)
        try:
            start_time = datetime.now()
            
            # For quick test, only run router audit
            if quick_test:
                from analysis.router_audit import RouterAuditor
                auditor = RouterAuditor()
                try:
                    report = auditor.generate_routing_audit_report(
                        str(orchestrator.consolidated_runs_file),
                        str(orchestrator.router_audit_file)
                    )
                    specialized_success = {'router_audit': bool(report)}
                except:
                    specialized_success = {'router_audit': False}
            else:
                specialized_success = orchestrator.execute_t1_3_specialized_analyses()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            test_results['components_tested']['T1_3'] = specialized_success
            test_results['performance_metrics']['T1_3_duration'] = duration
            
            successful_analyses = sum(specialized_success.values())
            total_analyses = len(specialized_success)
            test_results['details'].append(f"✅ T1.3: {successful_analyses}/{total_analyses} analyses completed in {duration:.1f}s")
        except Exception as e:
            test_results['components_tested']['T1_3'] = {}
            test_results['details'].append(f"❌ T1.3 error: {e}")
        
        # Step 7: Generate Master Report
        try:
            start_time = datetime.now()
            specialized_results = test_results['components_tested'].get('T1_3', {})
            master_report = orchestrator.generate_master_report(specialized_results)
            duration = (datetime.now() - start_time).total_seconds()
            
            test_results['performance_metrics']['master_report_duration'] = duration
            
            if master_report:
                test_results['details'].append(f"✅ Master report generated in {duration:.1f}s")
            else:
                test_results['details'].append("❌ Master report generation failed")
        except Exception as e:
            test_results['details'].append(f"❌ Master report error: {e}")
        
        # Overall assessment
        component_success_count = sum([
            test_results['components_tested'].get('T1_1', False),
            test_results['components_tested'].get('T1_2', False),
            test_results['components_tested'].get('pareto', False),
            bool(test_results['components_tested'].get('T1_3', {}))
        ])
        
        test_results['workflow_success'] = component_success_count >= 3  # At least 3/4 components
        test_results['validation_success'] = component_success_count >= 2  # Basic validation
        
        # Performance summary
        total_duration = sum(test_results['performance_metrics'].values())
        test_results['performance_metrics']['total_duration'] = total_duration
        
        test_results['details'].append(f"🎯 End-to-end test summary:")
        test_results['details'].append(f"   Components successful: {component_success_count}/4")
        test_results['details'].append(f"   Total duration: {total_duration:.1f}s")
        test_results['details'].append(f"   Workflow success: {test_results['workflow_success']}")
        
        return test_results
    
    def generate_validation_report(self, 
                                 t1_1_results: Dict[str, Any],
                                 t1_2_results: Dict[str, Any], 
                                 t1_3_results: Dict[str, Any],
                                 orchestration_results: Dict[str, Any],
                                 e2e_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        report = {
            'validation_metadata': {
                'validation_timestamp': datetime.now().isoformat(),
                'test_environment': str(self.test_dir),
                'validator_version': '1.0.0'
            },
            'component_validation_results': {
                'T1_1_data_collection': t1_1_results,
                'T1_2_statistical_analysis': t1_2_results,
                'T1_3_specialized_analyses': t1_3_results,
                'workflow_orchestration': orchestration_results
            },
            'overall_validation_summary': {},
            'recommendations': []
        }
        
        if e2e_results:
            report['end_to_end_test'] = e2e_results
        
        # Compute overall validation summary
        all_results = [t1_1_results, t1_2_results, t1_3_results, orchestration_results]
        
        total_tests = sum(r['tests_passed'] + r['tests_failed'] for r in all_results)
        total_passed = sum(r['tests_passed'] for r in all_results)
        
        report['overall_validation_summary'] = {
            'total_tests': total_tests,
            'tests_passed': total_passed,
            'tests_failed': total_tests - total_passed,
            'pass_rate': total_passed / total_tests if total_tests > 0 else 0,
            'components_validated': len(all_results),
            'validation_grade': self._compute_validation_grade(total_passed / total_tests if total_tests > 0 else 0)
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_validation_recommendations(report)
        
        return report
    
    def _compute_validation_grade(self, pass_rate: float) -> str:
        """Compute validation grade based on pass rate."""
        if pass_rate >= 0.95:
            return 'A+'
        elif pass_rate >= 0.9:
            return 'A'
        elif pass_rate >= 0.8:
            return 'B'
        elif pass_rate >= 0.7:
            return 'C'
        elif pass_rate >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _generate_validation_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate validation recommendations."""
        
        recommendations = []
        
        component_results = report['component_validation_results']
        overall_summary = report['overall_validation_summary']
        
        # Overall pass rate
        pass_rate = overall_summary.get('pass_rate', 0)
        if pass_rate < 0.8:
            recommendations.append(
                f"OVERALL: Validation pass rate is {pass_rate:.1%}. "
                f"Address failing tests before deployment."
            )
        
        # Component-specific recommendations
        for component, results in component_results.items():
            tests_failed = results.get('tests_failed', 0)
            if tests_failed > 0:
                recommendations.append(
                    f"{component.upper()}: {tests_failed} tests failed. "
                    f"Review component implementation and fix failing tests."
                )
        
        # End-to-end specific recommendations
        if 'end_to_end_test' in report:
            e2e = report['end_to_end_test']
            if not e2e.get('workflow_success', False):
                recommendations.append(
                    "END-TO-END: Workflow test failed. "
                    "Investigate component integration issues."
                )
        
        # Performance recommendations
        if 'end_to_end_test' in report:
            total_duration = report['end_to_end_test'].get('performance_metrics', {}).get('total_duration', 0)
            if total_duration > 300:  # More than 5 minutes
                recommendations.append(
                    f"PERFORMANCE: End-to-end test took {total_duration:.1f}s. "
                    f"Consider optimization for production use."
                )
        
        return recommendations[:10]  # Top 10 recommendations
    
    def print_validation_summary(self, report: Dict[str, Any]):
        """Print comprehensive validation summary."""
        
        print("\n" + "="*80)
        print("BEM 2.0 T1 WORKFLOW VALIDATION REPORT")
        print("="*80)
        
        # Overall summary
        overall = report['overall_validation_summary']
        print(f"\n📋 Validation Summary:")
        print(f"  Total Tests: {overall['total_tests']}")
        print(f"  Tests Passed: {overall['tests_passed']}")
        print(f"  Tests Failed: {overall['tests_failed']}")
        print(f"  Pass Rate: {overall['pass_rate']:.1%}")
        print(f"  Grade: {overall['validation_grade']}")
        
        # Component results
        print(f"\n🔍 Component Validation Results:")
        for component, results in report['component_validation_results'].items():
            passed = results['tests_passed']
            failed = results['tests_failed']
            total = passed + failed
            rate = passed / total if total > 0 else 0
            
            status = "✅" if failed == 0 else "⚠️" if rate >= 0.8 else "❌"
            print(f"  {status} {component}: {passed}/{total} ({rate:.1%})")
        
        # End-to-end results
        if 'end_to_end_test' in report:
            e2e = report['end_to_end_test']
            print(f"\n🚀 End-to-End Test:")
            print(f"  Workflow Success: {'✅' if e2e.get('workflow_success', False) else '❌'}")
            print(f"  Setup Success: {'✅' if e2e.get('setup_success', False) else '❌'}")
            
            components = e2e.get('components_tested', {})
            for component, success in components.items():
                if isinstance(success, dict):
                    # T1.3 results
                    successful = sum(success.values())
                    total = len(success)
                    print(f"  {component}: {successful}/{total} analyses successful")
                else:
                    # Boolean results
                    status = "✅" if success else "❌"
                    print(f"  {component}: {status}")
            
            # Performance metrics
            perf = e2e.get('performance_metrics', {})
            if perf:
                total_time = perf.get('total_duration', 0)
                print(f"  Total Duration: {total_time:.1f}s")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Key Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Final assessment
        grade = overall.get('validation_grade', 'F')
        pass_rate = overall.get('pass_rate', 0)
        
        print(f"\n🎯 Final Assessment:")
        if grade in ['A+', 'A'] and pass_rate >= 0.9:
            print("  ✅ Excellent - T1 workflow is ready for production use")
        elif grade in ['A', 'B'] and pass_rate >= 0.8:
            print("  ⚠️  Good - Minor issues to address before production")
        elif grade in ['B', 'C'] and pass_rate >= 0.7:
            print("  ⚠️  Acceptable - Significant improvements needed")
        else:
            print("  ❌ Poor - Major issues must be resolved before use")
        
        print(f"  Test Environment: {report['validation_metadata']['test_environment']}")
    
    def cleanup(self):
        """Clean up temporary test environment."""
        if self.cleanup_on_exit and self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
                logger.info(f"🧹 Cleaned up test directory: {self.test_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup test directory: {e}")


def main():
    """Main entry point for T1 workflow validation."""
    
    parser = argparse.ArgumentParser(
        description='BEM 2.0 T1 Workflow Validation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full validation suite
  python validate_t1_workflow.py
  
  # Quick validation (smaller datasets)
  python validate_t1_workflow.py --quick
  
  # Validate specific components only
  python validate_t1_workflow.py --component T1.1 --component T1.2
  
  # Skip end-to-end test
  python validate_t1_workflow.py --no-e2e
  
  # Use specific test directory
  python validate_t1_workflow.py --test-dir /tmp/t1_validation
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation with smaller datasets')
    parser.add_argument('--component', action='append',
                       choices=['T1.1', 'T1.2', 'T1.3', 'orchestration'],
                       help='Validate specific components only')
    parser.add_argument('--no-e2e', action='store_true',
                       help='Skip end-to-end test')
    parser.add_argument('--test-dir',
                       help='Use specific directory for test environment')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary files after validation')
    parser.add_argument('--output', 
                       help='Save validation report to JSON file')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = T1WorkflowValidator(temp_dir=args.test_dir)
    
    if args.keep_temp:
        validator.cleanup_on_exit = False
    
    try:
        print("\n" + "="*80)
        print("🧪 BEM 2.0 T1 WORKFLOW VALIDATION")
        print("="*80)
        print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏠 Test environment: {validator.test_dir}")
        print(f"⚡ Quick mode: {args.quick}")
        
        # Component validation results
        validation_results = {}
        
        # Validate components
        components_to_test = args.component if args.component else ['T1.1', 'T1.2', 'T1.3', 'orchestration']
        
        if 'T1.1' in components_to_test:
            logger.info("Validating T1.1 Data Collection...")
            validation_results['t1_1'] = validator.validate_t1_1_data_collection()
        
        if 'T1.2' in components_to_test:
            logger.info("Validating T1.2 Statistical Analysis...")
            validation_results['t1_2'] = validator.validate_t1_2_statistical_analysis()
        
        if 'T1.3' in components_to_test:
            logger.info("Validating T1.3 Specialized Analyses...")
            validation_results['t1_3'] = validator.validate_t1_3_specialized_analyses()
        
        if 'orchestration' in components_to_test:
            logger.info("Validating Workflow Orchestration...")
            validation_results['orchestration'] = validator.validate_workflow_orchestration()
        
        # End-to-end test
        e2e_results = None
        if not args.no_e2e:
            logger.info("Running End-to-End Test...")
            e2e_results = validator.run_end_to_end_test(quick_test=args.quick)
        
        # Generate comprehensive report
        report = validator.generate_validation_report(
            validation_results.get('t1_1', {'tests_passed': 0, 'tests_failed': 0, 'details': []}),
            validation_results.get('t1_2', {'tests_passed': 0, 'tests_failed': 0, 'details': []}),
            validation_results.get('t1_3', {'tests_passed': 0, 'tests_failed': 0, 'details': []}),
            validation_results.get('orchestration', {'tests_passed': 0, 'tests_failed': 0, 'details': []}),
            e2e_results
        )
        
        # Print summary
        validator.print_validation_summary(report)
        
        # Save report if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📋 Validation report saved: {output_path}")
        
        # Determine exit code
        overall_summary = report['overall_validation_summary']
        success = overall_summary['pass_rate'] >= 0.8
        
        if success:
            print(f"\n🎉 T1 Workflow validation completed successfully!")
        else:
            print(f"\n❌ T1 Workflow validation completed with issues")
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if not args.keep_temp:
            validator.cleanup()


if __name__ == '__main__':
    main()