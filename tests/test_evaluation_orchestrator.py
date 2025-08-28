"""
Tests for evaluation_orchestrator.py - Master evaluation orchestrator.

This module tests the evaluation orchestrator that coordinates all evaluation
components and manages the complete BEM validation pipeline.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
import asyncio
from typing import Dict, List, Any, Tuple

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

from evaluation_orchestrator import (
    EvaluationCoordinator,
    ResourceManager,
    TaskScheduler,
    ResultsCollector,
    QualityAssurance,
    EvaluationOrchestrator,
    PerformanceMonitor,
    DependencyTracker
)


class TestEvaluationCoordinator(unittest.TestCase):
    """Test evaluation coordination functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.coordinator = EvaluationCoordinator()
        
        # Mock evaluation components
        self.mock_components = {
            "baseline_evaluator": Mock(),
            "statistical_validator": Mock(),
            "shift_generator": Mock(),
            "routing_auditor": Mock(),
            "spectral_monitor": Mock(),
            "retrieval_ablator": Mock()
        }
    
    def test_coordinate_evaluation_pipeline(self):
        """Test evaluation pipeline coordination."""
        # Mock component results
        for component in self.mock_components.values():
            component.run.return_value = {"status": "success", "metrics": {"accuracy": 0.85}}
        
        pipeline_config = {
            "models": ["bem_model_v1"],
            "baselines": ["static_lora", "adalora"],
            "evaluation_seeds": [42, 123, 456],
            "shifts": ["domain", "temporal", "adversarial"]
        }
        
        coordination_results = self.coordinator.coordinate_evaluation_pipeline(
            components=self.mock_components,
            config=pipeline_config
        )
        
        self.assertIsInstance(coordination_results, dict)
        self.assertIn("component_results", coordination_results)
        self.assertIn("coordination_metrics", coordination_results)
        self.assertIn("pipeline_status", coordination_results)
        self.assertIn("execution_timeline", coordination_results)
    
    def test_manage_evaluation_dependencies(self):
        """Test evaluation dependency management."""
        dependencies = {
            "statistical_validator": ["baseline_evaluator"],
            "routing_auditor": ["baseline_evaluator"],
            "spectral_monitor": ["baseline_evaluator"],
            "retrieval_ablator": ["shift_generator", "baseline_evaluator"]
        }
        
        execution_order = self.coordinator.manage_evaluation_dependencies(dependencies)
        
        self.assertIsInstance(execution_order, list)
        # Baseline evaluator should come first
        self.assertEqual(execution_order[0], "baseline_evaluator")
        # Retrieval ablator should come after shift_generator
        retrieval_idx = execution_order.index("retrieval_ablator")
        shift_idx = execution_order.index("shift_generator")
        self.assertGreater(retrieval_idx, shift_idx)
    
    def test_coordinate_parallel_execution(self):
        """Test parallel execution coordination."""
        parallel_groups = [
            ["baseline_evaluator"],  # Level 0
            ["shift_generator", "statistical_validator"],  # Level 1
            ["routing_auditor", "spectral_monitor"],  # Level 2
            ["retrieval_ablator"]  # Level 3
        ]
        
        with patch('asyncio.gather') as mock_gather:
            mock_gather.return_value = asyncio.Future()
            mock_gather.return_value.set_result([{"status": "success"}])
            
            parallel_results = asyncio.run(
                self.coordinator.coordinate_parallel_execution(
                    self.mock_components, parallel_groups
                )
            )
            
            self.assertIsInstance(parallel_results, dict)
            self.assertIn("parallel_results", parallel_results)
            self.assertIn("execution_stats", parallel_results)


class TestResourceManager(unittest.TestCase):
    """Test resource management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.resource_manager = ResourceManager()
        
        # Mock system resources
        self.resource_limits = {
            "max_memory_gb": 32,
            "max_gpu_memory_gb": 16,
            "max_cpu_cores": 8,
            "max_concurrent_tasks": 4
        }
    
    def test_monitor_system_resources(self):
        """Test system resource monitoring."""
        with patch('psutil.virtual_memory') as mock_memory:
            with patch('psutil.cpu_count') as mock_cpu:
                mock_memory.return_value.total = 32 * 1024**3  # 32GB
                mock_memory.return_value.available = 16 * 1024**3  # 16GB available
                mock_cpu.return_value = 8
                
                resource_status = self.resource_manager.monitor_system_resources()
                
                self.assertIsInstance(resource_status, dict)
                self.assertIn("memory", resource_status)
                self.assertIn("cpu", resource_status)
                self.assertIn("available_memory_gb", resource_status["memory"])
    
    def test_allocate_resources(self):
        """Test resource allocation."""
        task_requirements = {
            "task_1": {"memory_gb": 8, "gpu_memory_gb": 4, "cpu_cores": 2},
            "task_2": {"memory_gb": 6, "gpu_memory_gb": 3, "cpu_cores": 2},
            "task_3": {"memory_gb": 10, "gpu_memory_gb": 8, "cpu_cores": 3},
        }
        
        allocation_plan = self.resource_manager.allocate_resources(
            task_requirements, self.resource_limits
        )
        
        self.assertIsInstance(allocation_plan, dict)
        self.assertIn("allocation_schedule", allocation_plan)
        self.assertIn("resource_utilization", allocation_plan)
        self.assertIn("conflicts", allocation_plan)
    
    def test_optimize_resource_usage(self):
        """Test resource usage optimization."""
        current_allocation = {
            "task_1": {"memory_gb": 8, "gpu_memory_gb": 4, "priority": "high"},
            "task_2": {"memory_gb": 12, "gpu_memory_gb": 6, "priority": "medium"},
            "task_3": {"memory_gb": 6, "gpu_memory_gb": 3, "priority": "low"},
        }
        
        optimized_allocation = self.resource_manager.optimize_resource_usage(
            current_allocation, self.resource_limits
        )
        
        self.assertIsInstance(optimized_allocation, dict)
        self.assertIn("optimized_schedule", optimized_allocation)
        self.assertIn("efficiency_gain", optimized_allocation)
    
    def test_detect_resource_bottlenecks(self):
        """Test resource bottleneck detection."""
        resource_usage_history = [
            {"timestamp": 1, "memory_usage": 0.9, "gpu_usage": 0.8, "cpu_usage": 0.6},
            {"timestamp": 2, "memory_usage": 0.95, "gpu_usage": 0.9, "cpu_usage": 0.7},
            {"timestamp": 3, "memory_usage": 0.99, "gpu_usage": 0.95, "cpu_usage": 0.8},
        ]
        
        bottlenecks = self.resource_manager.detect_resource_bottlenecks(
            resource_usage_history
        )
        
        self.assertIsInstance(bottlenecks, dict)
        self.assertIn("bottleneck_types", bottlenecks)
        self.assertIn("severity", bottlenecks)
        # Memory should be identified as a bottleneck
        self.assertIn("memory", bottlenecks["bottleneck_types"])


class TestTaskScheduler(unittest.TestCase):
    """Test task scheduling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = TaskScheduler()
        
        # Mock evaluation tasks
        self.evaluation_tasks = [
            {
                "id": "baseline_eval",
                "priority": 1,
                "estimated_duration": 300,  # 5 minutes
                "resource_requirements": {"memory_gb": 4, "gpu_memory_gb": 2},
                "dependencies": []
            },
            {
                "id": "statistical_validation",
                "priority": 2,
                "estimated_duration": 600,  # 10 minutes
                "resource_requirements": {"memory_gb": 8, "gpu_memory_gb": 0},
                "dependencies": ["baseline_eval"]
            },
            {
                "id": "shift_generation",
                "priority": 2,
                "estimated_duration": 900,  # 15 minutes
                "resource_requirements": {"memory_gb": 6, "gpu_memory_gb": 4},
                "dependencies": ["baseline_eval"]
            }
        ]
    
    def test_schedule_tasks(self):
        """Test task scheduling."""
        resource_constraints = {
            "max_memory_gb": 16,
            "max_gpu_memory_gb": 8,
            "max_concurrent_tasks": 2
        }
        
        schedule = self.scheduler.schedule_tasks(
            self.evaluation_tasks, resource_constraints
        )
        
        self.assertIsInstance(schedule, dict)
        self.assertIn("execution_plan", schedule)
        self.assertIn("estimated_completion_time", schedule)
        self.assertIn("resource_utilization", schedule)
        
        # Verify dependencies are respected
        execution_plan = schedule["execution_plan"]
        baseline_time = next(task["start_time"] for task in execution_plan 
                           if task["id"] == "baseline_eval")
        dependent_tasks = [task for task in execution_plan 
                          if "baseline_eval" in task.get("dependencies", [])]
        
        for dependent_task in dependent_tasks:
            self.assertGreater(dependent_task["start_time"], baseline_time)
    
    def test_optimize_task_ordering(self):
        """Test task ordering optimization."""
        optimized_order = self.scheduler.optimize_task_ordering(
            self.evaluation_tasks,
            optimization_strategy="minimize_total_time"
        )
        
        self.assertIsInstance(optimized_order, list)
        self.assertEqual(len(optimized_order), len(self.evaluation_tasks))
        
        # High priority tasks should come early
        baseline_idx = next(i for i, task in enumerate(optimized_order) 
                           if task["id"] == "baseline_eval")
        self.assertEqual(baseline_idx, 0)  # Should be first
    
    def test_handle_task_failures(self):
        """Test task failure handling."""
        failed_task = self.evaluation_tasks[1]  # statistical_validation
        remaining_tasks = self.evaluation_tasks[2:]  # shift_generation
        
        failure_response = self.scheduler.handle_task_failures(
            failed_task, remaining_tasks
        )
        
        self.assertIsInstance(failure_response, dict)
        self.assertIn("recovery_plan", failure_response)
        self.assertIn("affected_tasks", failure_response)
        self.assertIn("alternative_approaches", failure_response)


class TestResultsCollector(unittest.TestCase):
    """Test results collection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = ResultsCollector()
        
        # Mock evaluation results
        self.mock_results = {
            "baseline_evaluator": {
                "bem_model": {"accuracy": 0.87, "f1": 0.84, "em": 0.79},
                "static_lora": {"accuracy": 0.81, "f1": 0.78, "em": 0.74},
                "adalora": {"accuracy": 0.83, "f1": 0.80, "em": 0.76}
            },
            "statistical_validator": {
                "significance_tests": {
                    "bem_vs_static": {"p_value": 0.001, "significant": True},
                    "bem_vs_adalora": {"p_value": 0.023, "significant": True}
                },
                "effect_sizes": {
                    "bem_vs_static": {"cohens_d": 0.8, "magnitude": "large"},
                    "bem_vs_adalora": {"cohens_d": 0.6, "magnitude": "medium"}
                }
            },
            "shift_generator": {
                "domain_shifts": {"validation_score": 0.92, "diversity_score": 0.88},
                "temporal_shifts": {"validation_score": 0.89, "diversity_score": 0.85},
                "adversarial_shifts": {"validation_score": 0.86, "diversity_score": 0.82}
            }
        }
    
    def test_collect_evaluation_results(self):
        """Test evaluation results collection."""
        collected_results = self.collector.collect_evaluation_results(self.mock_results)
        
        self.assertIsInstance(collected_results, dict)
        self.assertIn("aggregated_metrics", collected_results)
        self.assertIn("component_summaries", collected_results)
        self.assertIn("cross_component_analysis", collected_results)
    
    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        aggregated = self.collector.aggregate_metrics(
            self.mock_results["baseline_evaluator"]
        )
        
        self.assertIsInstance(aggregated, dict)
        self.assertIn("mean_accuracy", aggregated)
        self.assertIn("std_accuracy", aggregated)
        self.assertIn("best_model", aggregated)
        self.assertIn("performance_ranking", aggregated)
    
    def test_validate_result_consistency(self):
        """Test result consistency validation."""
        consistency_check = self.collector.validate_result_consistency(self.mock_results)
        
        self.assertIsInstance(consistency_check, dict)
        self.assertIn("consistency_score", consistency_check)
        self.assertIn("inconsistencies", consistency_check)
        self.assertIn("validation_status", consistency_check)
    
    def test_generate_comprehensive_summary(self):
        """Test comprehensive summary generation."""
        summary = self.collector.generate_comprehensive_summary(self.mock_results)
        
        self.assertIsInstance(summary, dict)
        self.assertIn("executive_summary", summary)
        self.assertIn("key_findings", summary)
        self.assertIn("performance_overview", summary)
        self.assertIn("statistical_significance", summary)
        self.assertIn("recommendations", summary)


class TestQualityAssurance(unittest.TestCase):
    """Test quality assurance functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.qa = QualityAssurance()
        
        # Mock evaluation data for QA
        self.evaluation_data = {
            "results": {
                "model_a": {"accuracy": 0.85, "f1": 0.82},
                "model_b": {"accuracy": 0.78, "f1": 0.75}
            },
            "metadata": {
                "evaluation_date": "2024-01-15",
                "dataset_size": 1000,
                "random_seed": 42
            }
        }
    
    def test_validate_evaluation_quality(self):
        """Test evaluation quality validation."""
        quality_report = self.qa.validate_evaluation_quality(self.evaluation_data)
        
        self.assertIsInstance(quality_report, dict)
        self.assertIn("quality_score", quality_report)
        self.assertIn("quality_checks", quality_report)
        self.assertIn("issues_found", quality_report)
        self.assertIn("recommendations", quality_report)
    
    def test_check_statistical_validity(self):
        """Test statistical validity checking."""
        statistical_data = {
            "sample_sizes": {"model_a": 1000, "model_b": 1000},
            "p_values": {"comparison_1": 0.001, "comparison_2": 0.045},
            "effect_sizes": {"comparison_1": 0.8, "comparison_2": 0.3},
            "confidence_intervals": {
                "model_a": {"lower": 0.82, "upper": 0.88},
                "model_b": {"lower": 0.74, "upper": 0.82}
            }
        }
        
        validity_check = self.qa.check_statistical_validity(statistical_data)
        
        self.assertIsInstance(validity_check, dict)
        self.assertIn("validity_status", validity_check)
        self.assertIn("statistical_power", validity_check)
        self.assertIn("multiple_testing_correction", validity_check)
    
    def test_detect_evaluation_anomalies(self):
        """Test evaluation anomaly detection."""
        # Create data with potential anomalies
        anomalous_results = {
            "model_scores": [0.85, 0.83, 0.87, 0.15, 0.86],  # One very low score
            "training_curves": {
                "model_a": [0.5, 0.6, 0.7, 0.8, 0.85],  # Normal progression
                "model_b": [0.5, 0.6, 0.3, 0.8, 0.85],  # Anomalous dip
            }
        }
        
        anomaly_report = self.qa.detect_evaluation_anomalies(anomalous_results)
        
        self.assertIsInstance(anomaly_report, dict)
        self.assertIn("anomalies_detected", anomaly_report)
        self.assertIn("anomaly_types", anomaly_report)
        self.assertIn("severity_levels", anomaly_report)
        
        # Should detect the anomalous score
        self.assertTrue(anomaly_report["anomalies_detected"])
    
    def test_verify_reproducibility(self):
        """Test reproducibility verification."""
        reproducibility_data = {
            "run_1": {"accuracy": 0.85, "f1": 0.82, "seed": 42},
            "run_2": {"accuracy": 0.851, "f1": 0.821, "seed": 42},  # Very similar
            "run_3": {"accuracy": 0.849, "f1": 0.819, "seed": 42},
        }
        
        reproducibility_check = self.qa.verify_reproducibility(reproducibility_data)
        
        self.assertIsInstance(reproducibility_check, dict)
        self.assertIn("reproducibility_score", reproducibility_check)
        self.assertIn("variance_analysis", reproducibility_check)
        self.assertIn("consistency_status", reproducibility_check)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def test_monitor_evaluation_performance(self):
        """Test evaluation performance monitoring."""
        performance_data = {
            "task_durations": {
                "baseline_eval": 300,  # 5 minutes
                "statistical_validation": 600,  # 10 minutes
                "shift_generation": 900  # 15 minutes
            },
            "resource_usage": {
                "peak_memory_gb": 12,
                "peak_gpu_memory_gb": 6,
                "average_cpu_usage": 0.7
            },
            "throughput_metrics": {
                "samples_per_second": 25,
                "total_samples_processed": 5000
            }
        }
        
        performance_report = self.monitor.monitor_evaluation_performance(performance_data)
        
        self.assertIsInstance(performance_report, dict)
        self.assertIn("performance_summary", performance_report)
        self.assertIn("efficiency_metrics", performance_report)
        self.assertIn("bottleneck_analysis", performance_report)
        self.assertIn("optimization_suggestions", performance_report)
    
    def test_track_resource_utilization(self):
        """Test resource utilization tracking."""
        utilization_history = [
            {"timestamp": 1, "memory": 0.6, "gpu": 0.4, "cpu": 0.5},
            {"timestamp": 2, "memory": 0.8, "gpu": 0.7, "cpu": 0.6},
            {"timestamp": 3, "memory": 0.9, "gpu": 0.9, "cpu": 0.8},
        ]
        
        utilization_analysis = self.monitor.track_resource_utilization(utilization_history)
        
        self.assertIsInstance(utilization_analysis, dict)
        self.assertIn("utilization_trends", utilization_analysis)
        self.assertIn("peak_usage", utilization_analysis)
        self.assertIn("efficiency_score", utilization_analysis)
    
    def test_analyze_performance_trends(self):
        """Test performance trend analysis."""
        historical_performance = [
            {"date": "2024-01-01", "avg_accuracy": 0.82, "processing_time": 1800},
            {"date": "2024-01-08", "avg_accuracy": 0.84, "processing_time": 1650},
            {"date": "2024-01-15", "avg_accuracy": 0.87, "processing_time": 1500},
        ]
        
        trend_analysis = self.monitor.analyze_performance_trends(historical_performance)
        
        self.assertIsInstance(trend_analysis, dict)
        self.assertIn("accuracy_trend", trend_analysis)
        self.assertIn("efficiency_trend", trend_analysis)
        self.assertIn("projected_performance", trend_analysis)


class TestDependencyTracker(unittest.TestCase):
    """Test dependency tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = DependencyTracker()
        
        # Mock dependency graph
        self.dependency_graph = {
            "baseline_evaluator": [],
            "statistical_validator": ["baseline_evaluator"],
            "shift_generator": ["baseline_evaluator"],
            "routing_auditor": ["baseline_evaluator", "statistical_validator"],
            "spectral_monitor": ["baseline_evaluator"],
            "retrieval_ablator": ["shift_generator", "baseline_evaluator"],
            "evaluation_orchestrator": [
                "routing_auditor", "spectral_monitor", "retrieval_ablator", "statistical_validator"
            ]
        }
    
    def test_track_component_dependencies(self):
        """Test component dependency tracking."""
        dependency_analysis = self.tracker.track_component_dependencies(
            self.dependency_graph
        )
        
        self.assertIsInstance(dependency_analysis, dict)
        self.assertIn("dependency_levels", dependency_analysis)
        self.assertIn("critical_path", dependency_analysis)
        self.assertIn("parallel_opportunities", dependency_analysis)
    
    def test_detect_circular_dependencies(self):
        """Test circular dependency detection."""
        # Add a circular dependency
        circular_graph = self.dependency_graph.copy()
        circular_graph["baseline_evaluator"] = ["evaluation_orchestrator"]  # Creates cycle
        
        circular_analysis = self.tracker.detect_circular_dependencies(circular_graph)
        
        self.assertIsInstance(circular_analysis, dict)
        self.assertIn("has_cycles", circular_analysis)
        self.assertIn("cycles_found", circular_analysis)
        
        # Should detect the cycle
        self.assertTrue(circular_analysis["has_cycles"])
    
    def test_optimize_execution_order(self):
        """Test execution order optimization."""
        optimized_order = self.tracker.optimize_execution_order(
            self.dependency_graph
        )
        
        self.assertIsInstance(optimized_order, list)
        
        # baseline_evaluator should come first (no dependencies)
        self.assertEqual(optimized_order[0], "baseline_evaluator")
        
        # evaluation_orchestrator should come last (most dependencies)
        self.assertEqual(optimized_order[-1], "evaluation_orchestrator")
        
        # Dependencies should be satisfied
        for i, component in enumerate(optimized_order):
            dependencies = self.dependency_graph[component]
            for dep in dependencies:
                dep_index = optimized_order.index(dep)
                self.assertLess(dep_index, i, f"{dep} should come before {component}")


class TestEvaluationOrchestrator(unittest.TestCase):
    """Test the main evaluation orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = EvaluationOrchestrator()
        
        # Mock configuration
        self.evaluation_config = {
            "models": ["bem_model_v1"],
            "baselines": ["static_lora", "adalora", "moelora"],
            "datasets": ["squad", "nq", "trivia"],
            "shifts": ["domain", "temporal", "adversarial"],
            "statistical_config": {
                "bootstrap_samples": 1000,  # Reduced for testing
                "significance_level": 0.05,
                "fdr_correction": True
            },
            "resource_limits": {
                "max_memory_gb": 32,
                "max_gpu_memory_gb": 16,
                "max_concurrent_tasks": 4
            }
        }
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator.coordinator)
        self.assertIsNotNone(self.orchestrator.resource_manager)
        self.assertIsNotNone(self.orchestrator.task_scheduler)
        self.assertIsNotNone(self.orchestrator.results_collector)
        self.assertIsNotNone(self.orchestrator.quality_assurance)
        self.assertIsNotNone(self.orchestrator.performance_monitor)
        self.assertIsNotNone(self.orchestrator.dependency_tracker)
    
    @patch('evaluation_orchestrator.BaselineEvaluator')
    @patch('evaluation_orchestrator.StatisticalValidator')
    @patch('evaluation_orchestrator.ShiftGenerator')
    def test_run_comprehensive_evaluation(self, mock_shift, mock_stats, mock_baseline):
        """Test running comprehensive evaluation."""
        # Mock component responses
        mock_baseline.return_value.evaluate.return_value = {
            "bem_model_v1": {"accuracy": 0.87, "f1": 0.84},
            "static_lora": {"accuracy": 0.81, "f1": 0.78}
        }
        
        mock_stats.return_value.validate.return_value = {
            "statistical_significance": True,
            "p_values": {"bem_vs_static": 0.001},
            "effect_sizes": {"bem_vs_static": 0.8}
        }
        
        mock_shift.return_value.generate_all_shifts.return_value = {
            "domain": [{"text": "shifted text", "shift_type": "domain"}]
        }
        
        # Run evaluation
        results = self.orchestrator.run_comprehensive_evaluation(self.evaluation_config)
        
        self.assertIsInstance(results, dict)
        self.assertIn("evaluation_results", results)
        self.assertIn("statistical_analysis", results)
        self.assertIn("quality_report", results)
        self.assertIn("execution_summary", results)
    
    def test_orchestration_with_failures(self):
        """Test orchestration with component failures."""
        # Mock a component failure
        mock_components = {
            "baseline_evaluator": Mock(),
            "statistical_validator": Mock(),
            "shift_generator": Mock()
        }
        
        # Make one component fail
        mock_components["shift_generator"].run.side_effect = Exception("Component failed")
        mock_components["baseline_evaluator"].run.return_value = {"status": "success"}
        mock_components["statistical_validator"].run.return_value = {"status": "success"}
        
        failure_handling = self.orchestrator.handle_component_failures(
            mock_components,
            failed_component="shift_generator"
        )
        
        self.assertIsInstance(failure_handling, dict)
        self.assertIn("recovery_plan", failure_handling)
        self.assertIn("alternative_execution", failure_handling)
        self.assertIn("impact_assessment", failure_handling)
    
    def test_save_and_load_orchestration_state(self):
        """Test saving and loading orchestration state."""
        orchestration_state = {
            "config": self.evaluation_config,
            "execution_status": "in_progress",
            "completed_components": ["baseline_evaluator", "statistical_validator"],
            "pending_components": ["shift_generator", "routing_auditor"],
            "results": {"baseline_evaluator": {"status": "success"}}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "orchestration_state.json"
            
            # Save state
            self.orchestrator.save_orchestration_state(orchestration_state, state_path)
            self.assertTrue(state_path.exists())
            
            # Load state
            loaded_state = self.orchestrator.load_orchestration_state(state_path)
            self.assertEqual(loaded_state["execution_status"], "in_progress")
            self.assertEqual(len(loaded_state["completed_components"]), 2)


class TestEvaluationOrchestratorIntegration(unittest.TestCase):
    """Integration tests for evaluation orchestrator."""
    
    def test_end_to_end_orchestration_workflow(self):
        """Test complete end-to-end orchestration workflow."""
        # Create minimal configuration for integration test
        minimal_config = {
            "models": ["test_model"],
            "baselines": ["test_baseline"],
            "datasets": ["test_dataset"],
            "shifts": ["domain"],
            "statistical_config": {
                "bootstrap_samples": 100,  # Minimal for testing
                "significance_level": 0.05
            },
            "resource_limits": {
                "max_memory_gb": 8,
                "max_concurrent_tasks": 2
            }
        }
        
        orchestrator = EvaluationOrchestrator()
        
        # Mock all components to return success
        with patch.multiple(
            'evaluation_orchestrator',
            BaselineEvaluator=Mock(),
            StatisticalValidator=Mock(),
            ShiftGenerator=Mock(),
            RoutingAuditor=Mock(),
            SpectralMonitor=Mock(),
            RetrievalAblator=Mock()
        ) as mocks:
            
            # Configure mock returns
            for mock_obj in mocks.values():
                mock_instance = mock_obj.return_value
                mock_instance.run.return_value = {
                    "status": "success",
                    "metrics": {"accuracy": 0.85, "f1": 0.82}
                }
            
            # Run orchestration
            results = orchestrator.run_comprehensive_evaluation(minimal_config)
            
            # Verify results structure
            self.assertIsInstance(results, dict)
            required_sections = [
                "evaluation_results", "quality_report", "execution_summary"
            ]
            
            for section in required_sections:
                self.assertIn(section, results)
            
            # Verify execution summary
            execution_summary = results["execution_summary"]
            self.assertIn("total_duration", execution_summary)
            self.assertIn("components_executed", execution_summary)
            self.assertIn("success_rate", execution_summary)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)