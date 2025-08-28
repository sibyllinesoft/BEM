"""
Tests for routing_auditor.py - Hierarchical routing analysis and validation.

This module tests the routing audit system that analyzes MoE routing decisions
and validates hierarchical expert selection patterns.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Any, Tuple

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

from routing_auditor import (
    RoutingPatternAnalyzer,
    HierarchicalValidator,
    ExpertLoadBalancer,
    RoutingAuditReport,
    RoutingAuditor,
    ExpertUtilizationTracker,
    RoutingDiversityMetrics
)


class TestRoutingPatternAnalyzer(unittest.TestCase):
    """Test routing pattern analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RoutingPatternAnalyzer()
        
        # Mock routing data - shape: (batch_size, sequence_length, num_experts)
        self.sample_routing_weights = torch.randn(32, 128, 8)
        self.sample_routing_weights = torch.softmax(self.sample_routing_weights, dim=-1)
        
        # Mock expert assignments
        self.sample_expert_assignments = torch.randint(0, 8, (32, 128))
        
        # Mock input features
        self.sample_inputs = torch.randn(32, 128, 768)
    
    def test_analyze_routing_patterns(self):
        """Test routing pattern analysis."""
        patterns = self.analyzer.analyze_routing_patterns(
            self.sample_routing_weights,
            self.sample_expert_assignments,
            self.sample_inputs
        )
        
        self.assertIsInstance(patterns, dict)
        self.assertIn("expert_utilization", patterns)
        self.assertIn("routing_diversity", patterns)
        self.assertIn("specialization_scores", patterns)
        self.assertIn("load_balance_score", patterns)
    
    def test_compute_expert_specialization(self):
        """Test expert specialization computation."""
        specialization_scores = self.analyzer.compute_expert_specialization(
            self.sample_routing_weights,
            self.sample_inputs
        )
        
        self.assertIsInstance(specialization_scores, torch.Tensor)
        self.assertEqual(specialization_scores.shape[0], 8)  # num_experts
        self.assertTrue(torch.all(specialization_scores >= 0))
    
    def test_compute_routing_diversity(self):
        """Test routing diversity computation."""
        diversity_metrics = self.analyzer.compute_routing_diversity(
            self.sample_routing_weights
        )
        
        self.assertIsInstance(diversity_metrics, dict)
        self.assertIn("entropy", diversity_metrics)
        self.assertIn("gini_coefficient", diversity_metrics)
        self.assertIn("effective_experts", diversity_metrics)
        
        # Check that entropy is reasonable
        entropy = diversity_metrics["entropy"]
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
    
    def test_detect_routing_anomalies(self):
        """Test routing anomaly detection."""
        # Create anomalous routing (all weights go to one expert)
        anomalous_weights = torch.zeros_like(self.sample_routing_weights)
        anomalous_weights[:, :, 0] = 1.0  # All weight to expert 0
        
        anomalies = self.analyzer.detect_routing_anomalies(anomalous_weights)
        
        self.assertIsInstance(anomalies, dict)
        self.assertIn("overloaded_experts", anomalies)
        self.assertIn("underutilized_experts", anomalies)
        self.assertIn("routing_collapse_score", anomalies)
        
        # Should detect expert 0 as overloaded
        overloaded = anomalies["overloaded_experts"]
        self.assertIn(0, overloaded)


class TestHierarchicalValidator(unittest.TestCase):
    """Test hierarchical routing validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = HierarchicalValidator()
        
        # Mock hierarchical routing data
        # Layer 1: coarse-grained routing
        self.layer1_weights = torch.randn(32, 128, 4)
        self.layer1_weights = torch.softmax(self.layer1_weights, dim=-1)
        
        # Layer 2: fine-grained routing within each coarse expert
        self.layer2_weights = torch.randn(32, 128, 4, 2)  # 2 sub-experts per coarse expert
        self.layer2_weights = torch.softmax(self.layer2_weights, dim=-1)
    
    def test_validate_hierarchical_consistency(self):
        """Test hierarchical routing consistency validation."""
        consistency_score = self.validator.validate_hierarchical_consistency(
            self.layer1_weights,
            self.layer2_weights
        )
        
        self.assertIsInstance(consistency_score, float)
        self.assertGreaterEqual(consistency_score, 0.0)
        self.assertLessEqual(consistency_score, 1.0)
    
    def test_analyze_expert_hierarchy(self):
        """Test expert hierarchy analysis."""
        hierarchy_analysis = self.validator.analyze_expert_hierarchy(
            [self.layer1_weights, self.layer2_weights]
        )
        
        self.assertIsInstance(hierarchy_analysis, dict)
        self.assertIn("layer_specialization", hierarchy_analysis)
        self.assertIn("hierarchy_depth_utilization", hierarchy_analysis)
        self.assertIn("cross_layer_dependencies", hierarchy_analysis)
    
    def test_compute_routing_tree_metrics(self):
        """Test routing tree metrics computation."""
        tree_metrics = self.validator.compute_routing_tree_metrics(
            [self.layer1_weights, self.layer2_weights]
        )
        
        self.assertIsInstance(tree_metrics, dict)
        self.assertIn("tree_depth", tree_metrics)
        self.assertIn("branching_factor", tree_metrics)
        self.assertIn("leaf_utilization", tree_metrics)
        self.assertIn("path_diversity", tree_metrics)


class TestExpertLoadBalancer(unittest.TestCase):
    """Test expert load balancing analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.balancer = ExpertLoadBalancer()
        self.num_experts = 8
        self.batch_size = 32
        self.sequence_length = 128
        
        # Mock routing weights
        self.routing_weights = torch.randn(self.batch_size, self.sequence_length, self.num_experts)
        self.routing_weights = torch.softmax(self.routing_weights, dim=-1)
    
    def test_compute_load_balance_metrics(self):
        """Test load balance metrics computation."""
        load_metrics = self.balancer.compute_load_balance_metrics(self.routing_weights)
        
        self.assertIsInstance(load_metrics, dict)
        self.assertIn("coefficient_of_variation", load_metrics)
        self.assertIn("load_imbalance_ratio", load_metrics)
        self.assertIn("expert_utilization_variance", load_metrics)
        self.assertIn("effective_expert_ratio", load_metrics)
    
    def test_detect_load_imbalance(self):
        """Test load imbalance detection."""
        # Create imbalanced routing
        imbalanced_weights = torch.zeros_like(self.routing_weights)
        imbalanced_weights[:, :, 0] = 0.8  # Most weight to expert 0
        imbalanced_weights[:, :, 1:] = 0.2 / (self.num_experts - 1)  # Rest distributed
        
        imbalance_analysis = self.balancer.detect_load_imbalance(
            imbalanced_weights,
            threshold=0.1
        )
        
        self.assertIsInstance(imbalance_analysis, dict)
        self.assertIn("is_imbalanced", imbalance_analysis)
        self.assertIn("imbalance_severity", imbalance_analysis)
        self.assertIn("overloaded_experts", imbalance_analysis)
        self.assertIn("underloaded_experts", imbalance_analysis)
        
        # Should detect imbalance
        self.assertTrue(imbalance_analysis["is_imbalanced"])
    
    def test_suggest_load_balancing_strategies(self):
        """Test load balancing strategy suggestions."""
        load_metrics = self.balancer.compute_load_balance_metrics(self.routing_weights)
        
        strategies = self.balancer.suggest_load_balancing_strategies(
            load_metrics,
            self.routing_weights
        )
        
        self.assertIsInstance(strategies, list)
        for strategy in strategies:
            self.assertIn("strategy_type", strategy)
            self.assertIn("description", strategy)
            self.assertIn("expected_improvement", strategy)


class TestExpertUtilizationTracker(unittest.TestCase):
    """Test expert utilization tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ExpertUtilizationTracker()
        self.num_experts = 8
    
    def test_track_utilization(self):
        """Test utilization tracking."""
        # Simulate multiple batches
        for i in range(5):
            routing_weights = torch.randn(32, 128, self.num_experts)
            routing_weights = torch.softmax(routing_weights, dim=-1)
            
            self.tracker.update(routing_weights, batch_id=i)
        
        utilization_stats = self.tracker.get_utilization_stats()
        
        self.assertIsInstance(utilization_stats, dict)
        self.assertIn("mean_utilization", utilization_stats)
        self.assertIn("utilization_variance", utilization_stats)
        self.assertIn("expert_rankings", utilization_stats)
        self.assertIn("temporal_trends", utilization_stats)
    
    def test_utilization_history(self):
        """Test utilization history tracking."""
        # Track multiple updates
        for i in range(10):
            routing_weights = torch.randn(16, 64, self.num_experts)
            routing_weights = torch.softmax(routing_weights, dim=-1)
            self.tracker.update(routing_weights, batch_id=i)
        
        history = self.tracker.get_utilization_history()
        
        self.assertIsInstance(history, dict)
        self.assertEqual(len(history["batch_ids"]), 10)
        self.assertEqual(len(history["utilization_per_batch"]), 10)
    
    def test_utilization_anomaly_detection(self):
        """Test detection of utilization anomalies."""
        # Create normal utilization pattern
        for i in range(5):
            normal_weights = torch.ones(32, 128, self.num_experts) / self.num_experts
            self.tracker.update(normal_weights, batch_id=i)
        
        # Create anomalous utilization
        anomalous_weights = torch.zeros(32, 128, self.num_experts)
        anomalous_weights[:, :, 0] = 1.0
        self.tracker.update(anomalous_weights, batch_id=5)
        
        anomalies = self.tracker.detect_anomalies()
        
        self.assertIsInstance(anomalies, dict)
        self.assertIn("anomalous_batches", anomalies)
        self.assertIn("anomaly_scores", anomalies)


class TestRoutingDiversityMetrics(unittest.TestCase):
    """Test routing diversity metrics computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = RoutingDiversityMetrics()
        self.routing_weights = torch.randn(32, 128, 8)
        self.routing_weights = torch.softmax(self.routing_weights, dim=-1)
    
    def test_compute_entropy_metrics(self):
        """Test entropy-based diversity metrics."""
        entropy_metrics = self.metrics.compute_entropy_metrics(self.routing_weights)
        
        self.assertIsInstance(entropy_metrics, dict)
        self.assertIn("shannon_entropy", entropy_metrics)
        self.assertIn("normalized_entropy", entropy_metrics)
        self.assertIn("entropy_variance", entropy_metrics)
        
        # Check entropy bounds
        shannon_entropy = entropy_metrics["shannon_entropy"]
        self.assertGreaterEqual(shannon_entropy, 0.0)
        self.assertLessEqual(shannon_entropy, np.log(8))  # log(num_experts)
    
    def test_compute_gini_coefficient(self):
        """Test Gini coefficient computation."""
        gini_coeff = self.metrics.compute_gini_coefficient(self.routing_weights)
        
        self.assertIsInstance(gini_coeff, float)
        self.assertGreaterEqual(gini_coeff, 0.0)
        self.assertLessEqual(gini_coeff, 1.0)
    
    def test_compute_effective_experts(self):
        """Test effective number of experts computation."""
        effective_experts = self.metrics.compute_effective_experts(self.routing_weights)
        
        self.assertIsInstance(effective_experts, float)
        self.assertGreaterEqual(effective_experts, 1.0)
        self.assertLessEqual(effective_experts, 8.0)  # num_experts
    
    def test_compute_diversity_over_time(self):
        """Test diversity metrics over time."""
        # Simulate temporal routing data
        temporal_routing = []
        for t in range(10):
            routing_t = torch.randn(32, 64, 8)
            routing_t = torch.softmax(routing_t, dim=-1)
            temporal_routing.append(routing_t)
        
        temporal_diversity = self.metrics.compute_diversity_over_time(temporal_routing)
        
        self.assertIsInstance(temporal_diversity, dict)
        self.assertIn("diversity_trend", temporal_diversity)
        self.assertIn("stability_score", temporal_diversity)
        self.assertIn("temporal_variance", temporal_diversity)


class TestRoutingAuditReport(unittest.TestCase):
    """Test routing audit report generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.report = RoutingAuditReport()
        
        # Mock audit data
        self.audit_data = {
            "routing_patterns": {
                "expert_utilization": [0.15, 0.12, 0.18, 0.10, 0.13, 0.11, 0.12, 0.09],
                "routing_diversity": {"entropy": 2.85, "gini_coefficient": 0.12},
                "load_balance_score": 0.87
            },
            "hierarchical_validation": {
                "consistency_score": 0.92,
                "hierarchy_depth_utilization": [0.8, 0.6]
            },
            "anomalies": {
                "routing_collapse_score": 0.05,
                "overloaded_experts": [],
                "underutilized_experts": [7]
            }
        }
    
    def test_generate_summary_report(self):
        """Test summary report generation."""
        summary = self.report.generate_summary_report(self.audit_data)
        
        self.assertIsInstance(summary, dict)
        self.assertIn("overall_health_score", summary)
        self.assertIn("key_findings", summary)
        self.assertIn("recommendations", summary)
        self.assertIn("metrics_summary", summary)
        
        # Check health score is reasonable
        health_score = summary["overall_health_score"]
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)
    
    def test_generate_detailed_report(self):
        """Test detailed report generation."""
        detailed_report = self.report.generate_detailed_report(self.audit_data)
        
        self.assertIsInstance(detailed_report, str)
        self.assertIn("Routing Audit Report", detailed_report)
        self.assertIn("Expert Utilization", detailed_report)
        self.assertIn("Load Balancing", detailed_report)
        self.assertIn("Recommendations", detailed_report)
    
    def test_save_audit_report(self):
        """Test saving audit report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "audit_report.json"
            
            self.report.save_audit_report(self.audit_data, report_path)
            
            self.assertTrue(report_path.exists())
            
            # Verify saved content
            with open(report_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIn("audit_data", saved_data)
            self.assertIn("summary", saved_data)
            self.assertIn("timestamp", saved_data)


class TestRoutingAuditor(unittest.TestCase):
    """Test the main routing auditor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.auditor = RoutingAuditor()
        
        # Mock model with routing layers
        self.mock_model = Mock()
        self.mock_model.routing_layers = [Mock(), Mock()]
        
        # Mock routing data for multiple layers
        self.routing_data = {
            "layer_0": {
                "routing_weights": torch.randn(32, 128, 8),
                "expert_assignments": torch.randint(0, 8, (32, 128)),
                "inputs": torch.randn(32, 128, 768)
            },
            "layer_1": {
                "routing_weights": torch.randn(32, 128, 8),
                "expert_assignments": torch.randint(0, 8, (32, 128)),
                "inputs": torch.randn(32, 128, 768)
            }
        }
        
        # Apply softmax to routing weights
        for layer_data in self.routing_data.values():
            layer_data["routing_weights"] = torch.softmax(
                layer_data["routing_weights"], dim=-1
            )
    
    def test_auditor_initialization(self):
        """Test auditor initialization."""
        self.assertIsNotNone(self.auditor.pattern_analyzer)
        self.assertIsNotNone(self.auditor.hierarchical_validator)
        self.assertIsNotNone(self.auditor.load_balancer)
        self.assertIsNotNone(self.auditor.utilization_tracker)
        self.assertIsNotNone(self.auditor.diversity_metrics)
        self.assertIsNotNone(self.auditor.report_generator)
    
    def test_comprehensive_routing_audit(self):
        """Test comprehensive routing audit."""
        audit_results = self.auditor.comprehensive_audit(
            self.routing_data,
            model_config={"num_experts": 8, "num_layers": 2}
        )
        
        self.assertIsInstance(audit_results, dict)
        self.assertIn("routing_patterns", audit_results)
        self.assertIn("hierarchical_analysis", audit_results)
        self.assertIn("load_balancing", audit_results)
        self.assertIn("diversity_metrics", audit_results)
        self.assertIn("anomaly_detection", audit_results)
        self.assertIn("overall_assessment", audit_results)
    
    def test_layer_wise_audit(self):
        """Test layer-wise routing audit."""
        layer_results = {}
        
        for layer_name, layer_data in self.routing_data.items():
            layer_audit = self.auditor.audit_layer(
                layer_data["routing_weights"],
                layer_data["expert_assignments"],
                layer_data["inputs"],
                layer_name=layer_name
            )
            layer_results[layer_name] = layer_audit
        
        # Verify each layer audit
        for layer_name, layer_audit in layer_results.items():
            self.assertIsInstance(layer_audit, dict)
            self.assertIn("patterns", layer_audit)
            self.assertIn("load_balance", layer_audit)
            self.assertIn("diversity", layer_audit)
            self.assertIn("anomalies", layer_audit)
    
    def test_temporal_routing_analysis(self):
        """Test temporal routing analysis across batches."""
        # Simulate temporal routing data
        temporal_data = []
        for t in range(5):
            batch_data = {
                "routing_weights": torch.randn(16, 64, 8),
                "expert_assignments": torch.randint(0, 8, (16, 64)),
                "timestamp": t
            }
            batch_data["routing_weights"] = torch.softmax(
                batch_data["routing_weights"], dim=-1
            )
            temporal_data.append(batch_data)
        
        temporal_analysis = self.auditor.temporal_analysis(temporal_data)
        
        self.assertIsInstance(temporal_analysis, dict)
        self.assertIn("utilization_trends", temporal_analysis)
        self.assertIn("diversity_evolution", temporal_analysis)
        self.assertIn("stability_metrics", temporal_analysis)
        self.assertIn("anomaly_timeline", temporal_analysis)
    
    def test_routing_health_assessment(self):
        """Test overall routing health assessment."""
        health_assessment = self.auditor.assess_routing_health(self.routing_data)
        
        self.assertIsInstance(health_assessment, dict)
        self.assertIn("overall_health_score", health_assessment)
        self.assertIn("health_breakdown", health_assessment)
        self.assertIn("critical_issues", health_assessment)
        self.assertIn("recommendations", health_assessment)
        
        # Health score should be valid
        health_score = health_assessment["overall_health_score"]
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)


class TestRoutingAuditIntegration(unittest.TestCase):
    """Integration tests for routing audit components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.auditor = RoutingAuditor()
        
        # Create realistic routing scenario
        self.num_experts = 12
        self.batch_size = 64
        self.sequence_length = 256
        self.num_layers = 3
        
        # Generate multi-layer routing data with realistic patterns
        self.multi_layer_data = self.generate_realistic_routing_data()
    
    def generate_realistic_routing_data(self):
        """Generate realistic routing data for testing."""
        routing_data = {}
        
        for layer in range(self.num_layers):
            # Create routing weights with some specialization patterns
            routing_weights = torch.randn(self.batch_size, self.sequence_length, self.num_experts)
            
            # Add specialization bias (some experts more likely to be chosen)
            if layer == 0:  # First layer - broad specialization
                routing_weights[:, :, :4] += 0.5  # Bias toward first 4 experts
            elif layer == 1:  # Middle layer - balanced
                pass  # Keep random
            else:  # Last layer - specific specialization
                routing_weights[:, :, -4:] += 0.8  # Bias toward last 4 experts
            
            routing_weights = torch.softmax(routing_weights, dim=-1)
            expert_assignments = torch.multinomial(
                routing_weights.view(-1, self.num_experts), 1
            ).view(self.batch_size, self.sequence_length)
            
            routing_data[f"layer_{layer}"] = {
                "routing_weights": routing_weights,
                "expert_assignments": expert_assignments,
                "inputs": torch.randn(self.batch_size, self.sequence_length, 768)
            }
        
        return routing_data
    
    def test_end_to_end_audit_pipeline(self):
        """Test complete end-to-end audit pipeline."""
        # Run comprehensive audit
        audit_results = self.auditor.comprehensive_audit(
            self.multi_layer_data,
            model_config={
                "num_experts": self.num_experts,
                "num_layers": self.num_layers,
                "model_type": "MoE-Transformer"
            }
        )
        
        # Verify comprehensive results structure
        required_sections = [
            "routing_patterns", "hierarchical_analysis", "load_balancing",
            "diversity_metrics", "anomaly_detection", "overall_assessment"
        ]
        
        for section in required_sections:
            self.assertIn(section, audit_results)
        
        # Verify assessment quality
        assessment = audit_results["overall_assessment"]
        self.assertIn("health_score", assessment)
        self.assertIn("key_insights", assessment)
        self.assertIn("recommendations", assessment)
    
    def test_audit_report_generation(self):
        """Test complete audit report generation."""
        # Run audit
        audit_results = self.auditor.comprehensive_audit(self.multi_layer_data)
        
        # Generate report
        report = self.auditor.report_generator.generate_detailed_report(audit_results)
        
        # Verify report content
        self.assertIsInstance(report, str)
        self.assertIn("Executive Summary", report)
        self.assertIn("Routing Pattern Analysis", report)
        self.assertIn("Expert Utilization", report)
        self.assertIn("Load Balancing Assessment", report)
        self.assertIn("Recommendations", report)
    
    def test_routing_optimization_suggestions(self):
        """Test routing optimization suggestions."""
        audit_results = self.auditor.comprehensive_audit(self.multi_layer_data)
        
        # Extract optimization suggestions
        load_balancing = audit_results["load_balancing"]
        anomalies = audit_results["anomaly_detection"]
        
        # Verify optimization suggestions exist
        if "load_imbalance" in load_balancing:
            self.assertIn("balancing_strategies", load_balancing)
        
        if anomalies.get("has_anomalies", False):
            self.assertIn("remediation_suggestions", anomalies)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)