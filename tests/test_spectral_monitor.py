"""
Tests for spectral_monitor.py - Spectral governance and stability monitoring.

This module tests the spectral monitoring system that tracks gradient conflicts,
spectral radius bounds, and numerical stability in MoE training.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
import warnings
from typing import Dict, List, Any, Tuple

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

from spectral_monitor import (
    SpectralAnalyzer,
    GradientConflictDetector,
    StabilityMonitor,
    SpectralGovernor,
    NumericalStabilityTracker,
    SpectralMetrics,
    SpectralMonitoringSystem
)


class TestSpectralAnalyzer(unittest.TestCase):
    """Test spectral analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SpectralAnalyzer()
        
        # Create mock model parameters
        self.mock_parameters = [
            torch.randn(64, 32, requires_grad=True),  # Weight matrix 1
            torch.randn(32, 16, requires_grad=True),  # Weight matrix 2
            torch.randn(16, requires_grad=True),       # Bias vector
        ]
        
        # Create mock gradients
        for param in self.mock_parameters:
            param.grad = torch.randn_like(param) * 0.1
    
    def test_compute_spectral_radius(self):
        """Test spectral radius computation."""
        weight_matrix = torch.randn(50, 50)
        spectral_radius = self.analyzer.compute_spectral_radius(weight_matrix)
        
        self.assertIsInstance(spectral_radius, float)
        self.assertGreaterEqual(spectral_radius, 0.0)
        
        # For a random matrix, spectral radius should be reasonable
        self.assertLess(spectral_radius, 10.0)
    
    def test_compute_frobenius_norm(self):
        """Test Frobenius norm computation."""
        matrix = torch.randn(20, 30)
        frobenius_norm = self.analyzer.compute_frobenius_norm(matrix)
        
        self.assertIsInstance(frobenius_norm, float)
        self.assertGreaterEqual(frobenius_norm, 0.0)
        
        # Verify against PyTorch implementation
        expected_norm = torch.norm(matrix, p='fro').item()
        self.assertAlmostEqual(frobenius_norm, expected_norm, places=5)
    
    def test_analyze_parameter_spectrum(self):
        """Test parameter spectrum analysis."""
        weight_matrix = torch.randn(64, 64)
        spectrum_analysis = self.analyzer.analyze_parameter_spectrum(weight_matrix)
        
        self.assertIsInstance(spectrum_analysis, dict)
        self.assertIn("eigenvalues", spectrum_analysis)
        self.assertIn("spectral_radius", spectrum_analysis)
        self.assertIn("condition_number", spectrum_analysis)
        self.assertIn("rank", spectrum_analysis)
        
        # Check eigenvalues are complex numbers or real
        eigenvalues = spectrum_analysis["eigenvalues"]
        self.assertIsInstance(eigenvalues, (torch.Tensor, np.ndarray))
    
    def test_compute_gradient_spectrum(self):
        """Test gradient spectrum computation."""
        gradients = [param.grad for param in self.mock_parameters if param.grad is not None]
        
        spectrum_metrics = self.analyzer.compute_gradient_spectrum(gradients)
        
        self.assertIsInstance(spectrum_metrics, dict)
        self.assertIn("max_eigenvalue", spectrum_metrics)
        self.assertIn("min_eigenvalue", spectrum_metrics)
        self.assertIn("spectral_gap", spectrum_metrics)
        self.assertIn("gradient_norms", spectrum_metrics)
    
    def test_detect_spectral_anomalies(self):
        """Test spectral anomaly detection."""
        # Create a matrix with known spectral properties
        # Construct a matrix with large spectral radius
        eigenvalues = torch.tensor([5.0, -4.0, 2.0, 1.0])
        Q = torch.randn(4, 4)
        Q, _ = torch.qr(Q)  # Orthogonal matrix
        anomalous_matrix = Q @ torch.diag(eigenvalues) @ Q.T
        
        anomalies = self.analyzer.detect_spectral_anomalies(
            anomalous_matrix,
            spectral_radius_threshold=3.0,
            condition_number_threshold=100.0
        )
        
        self.assertIsInstance(anomalies, dict)
        self.assertIn("spectral_radius_violation", anomalies)
        self.assertIn("ill_conditioning", anomalies)
        self.assertIn("rank_deficiency", anomalies)
        
        # Should detect spectral radius violation
        self.assertTrue(anomalies["spectral_radius_violation"])


class TestGradientConflictDetector(unittest.TestCase):
    """Test gradient conflict detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = GradientConflictDetector()
        
        # Create mock gradients from different experts/tasks
        self.expert_gradients = {
            "expert_0": [torch.randn(32, 16) * 0.1, torch.randn(16) * 0.05],
            "expert_1": [torch.randn(32, 16) * 0.1, torch.randn(16) * 0.05],
            "expert_2": [torch.randn(32, 16) * 0.1, torch.randn(16) * 0.05],
        }
    
    def test_compute_gradient_similarity(self):
        """Test gradient similarity computation."""
        grad1 = [torch.randn(10, 5), torch.randn(5)]
        grad2 = [torch.randn(10, 5), torch.randn(5)]
        
        similarity = self.detector.compute_gradient_similarity(grad1, grad2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_detect_gradient_conflicts(self):
        """Test gradient conflict detection between experts."""
        conflicts = self.detector.detect_gradient_conflicts(self.expert_gradients)
        
        self.assertIsInstance(conflicts, dict)
        self.assertIn("conflict_matrix", conflicts)
        self.assertIn("high_conflict_pairs", conflicts)
        self.assertIn("average_conflict_score", conflicts)
        
        # Conflict matrix should be symmetric
        conflict_matrix = conflicts["conflict_matrix"]
        num_experts = len(self.expert_gradients)
        self.assertEqual(conflict_matrix.shape, (num_experts, num_experts))
        
        # Diagonal should be 1 (self-similarity)
        for i in range(num_experts):
            self.assertAlmostEqual(conflict_matrix[i, i], 1.0, places=5)
    
    def test_analyze_gradient_interference(self):
        """Test gradient interference analysis."""
        interference_analysis = self.detector.analyze_gradient_interference(
            self.expert_gradients
        )
        
        self.assertIsInstance(interference_analysis, dict)
        self.assertIn("destructive_interference", interference_analysis)
        self.assertIn("constructive_interference", interference_analysis)
        self.assertIn("interference_strength", interference_analysis)
    
    def test_compute_conflict_severity(self):
        """Test conflict severity computation."""
        # Create highly conflicting gradients
        conflicting_gradients = {
            "expert_0": [torch.ones(10, 5)],
            "expert_1": [-torch.ones(10, 5)],  # Opposite direction
        }
        
        severity = self.detector.compute_conflict_severity(conflicting_gradients)
        
        self.assertIsInstance(severity, float)
        self.assertGreaterEqual(severity, 0.0)
        
        # High conflict should result in high severity
        self.assertGreater(severity, 0.5)


class TestStabilityMonitor(unittest.TestCase):
    """Test stability monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = StabilityMonitor()
        
        # Create mock training history
        self.training_history = []
        for step in range(50):
            metrics = {
                "loss": 2.0 * np.exp(-step * 0.1) + np.random.normal(0, 0.1),
                "gradient_norm": 1.0 + np.random.normal(0, 0.2),
                "spectral_radius": 0.95 + np.random.normal(0, 0.05),
                "step": step
            }
            self.training_history.append(metrics)
    
    def test_analyze_training_stability(self):
        """Test training stability analysis."""
        stability_analysis = self.monitor.analyze_training_stability(
            self.training_history
        )
        
        self.assertIsInstance(stability_analysis, dict)
        self.assertIn("stability_score", stability_analysis)
        self.assertIn("convergence_rate", stability_analysis)
        self.assertIn("oscillation_detection", stability_analysis)
        self.assertIn("instability_periods", stability_analysis)
        
        # Stability score should be reasonable
        stability_score = stability_analysis["stability_score"]
        self.assertIsInstance(stability_score, float)
        self.assertGreaterEqual(stability_score, 0.0)
        self.assertLessEqual(stability_score, 1.0)
    
    def test_detect_numerical_instabilities(self):
        """Test numerical instability detection."""
        # Create history with instability
        unstable_history = self.training_history.copy()
        # Add some unstable points
        unstable_history[25]["gradient_norm"] = 100.0  # Gradient explosion
        unstable_history[30]["loss"] = float('nan')    # NaN loss
        
        instabilities = self.monitor.detect_numerical_instabilities(unstable_history)
        
        self.assertIsInstance(instabilities, dict)
        self.assertIn("gradient_explosion", instabilities)
        self.assertIn("nan_detection", instabilities)
        self.assertIn("inf_detection", instabilities)
        self.assertIn("unstable_steps", instabilities)
        
        # Should detect the instabilities we added
        self.assertTrue(instabilities["gradient_explosion"]["detected"])
        self.assertTrue(instabilities["nan_detection"]["detected"])
    
    def test_compute_stability_metrics(self):
        """Test stability metrics computation."""
        stability_metrics = self.monitor.compute_stability_metrics(
            self.training_history
        )
        
        self.assertIsInstance(stability_metrics, dict)
        self.assertIn("loss_variance", stability_metrics)
        self.assertIn("gradient_variance", stability_metrics)
        self.assertIn("spectral_variance", stability_metrics)
        self.assertIn("trend_analysis", stability_metrics)
    
    def test_predict_instability(self):
        """Test instability prediction."""
        # Create recent history
        recent_history = self.training_history[-10:]
        
        instability_prediction = self.monitor.predict_instability(
            recent_history,
            lookhead_steps=5
        )
        
        self.assertIsInstance(instability_prediction, dict)
        self.assertIn("instability_probability", instability_prediction)
        self.assertIn("confidence", instability_prediction)
        self.assertIn("risk_factors", instability_prediction)
        
        # Probability should be valid
        prob = instability_prediction["instability_probability"]
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


class TestSpectralGovernor(unittest.TestCase):
    """Test spectral governance functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.governor = SpectralGovernor()
        
        # Create mock model
        self.mock_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Linear(16, 8)
        )
        
        # Add some gradients
        dummy_input = torch.randn(10, 64)
        dummy_target = torch.randn(10, 8)
        loss = nn.MSELoss()(self.mock_model(dummy_input), dummy_target)
        loss.backward()
    
    def test_apply_spectral_normalization(self):
        """Test spectral normalization application."""
        # Get original spectral radius
        original_weights = self.mock_model[0].weight.data.clone()
        
        self.governor.apply_spectral_normalization(self.mock_model)
        
        # Check that weights have been normalized
        normalized_weights = self.mock_model[0].weight.data
        self.assertFalse(torch.equal(original_weights, normalized_weights))
        
        # Spectral radius should be controlled
        spectral_radius = self.governor._compute_spectral_radius(normalized_weights)
        self.assertLessEqual(spectral_radius, 1.1)  # Allow small numerical tolerance
    
    def test_enforce_spectral_bounds(self):
        """Test spectral bound enforcement."""
        # Create matrix with large spectral radius
        large_matrix = torch.randn(32, 32) * 10
        
        bounded_matrix = self.governor.enforce_spectral_bounds(
            large_matrix,
            max_spectral_radius=1.0
        )
        
        # Check spectral radius is bounded
        bounded_radius = self.governor._compute_spectral_radius(bounded_matrix)
        self.assertLessEqual(bounded_radius, 1.1)  # Small tolerance for numerical errors
    
    def test_gradient_clipping_with_spectral_info(self):
        """Test gradient clipping using spectral information."""
        # Get model gradients
        gradients = [param.grad for param in self.mock_model.parameters() 
                    if param.grad is not None]
        
        original_norms = [torch.norm(grad).item() for grad in gradients]
        
        self.governor.gradient_clipping_with_spectral_info(
            self.mock_model,
            max_norm=1.0
        )
        
        # Check gradients have been clipped
        clipped_norms = [torch.norm(param.grad).item() 
                        for param in self.mock_model.parameters() 
                        if param.grad is not None]
        
        # At least some gradients should be clipped if they were large
        if any(norm > 1.0 for norm in original_norms):
            self.assertTrue(all(norm <= 1.1 for norm in clipped_norms))
    
    def test_adaptive_spectral_control(self):
        """Test adaptive spectral control."""
        control_metrics = {
            "spectral_radius": 1.5,
            "gradient_norm": 2.0,
            "loss_trend": "increasing"
        }
        
        control_actions = self.governor.adaptive_spectral_control(
            self.mock_model,
            control_metrics
        )
        
        self.assertIsInstance(control_actions, dict)
        self.assertIn("spectral_normalization_applied", control_actions)
        self.assertIn("gradient_scaling_factor", control_actions)
        self.assertIn("stability_interventions", control_actions)


class TestNumericalStabilityTracker(unittest.TestCase):
    """Test numerical stability tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = NumericalStabilityTracker()
    
    def test_track_numerical_health(self):
        """Test numerical health tracking."""
        # Create model with potential numerical issues
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 5)
        )
        
        # Add some gradients
        dummy_input = torch.randn(5, 10)
        dummy_target = torch.randn(5, 5)
        loss = nn.MSELoss()(model(dummy_input), dummy_target)
        loss.backward()
        
        health_metrics = self.tracker.track_numerical_health(model)
        
        self.assertIsInstance(health_metrics, dict)
        self.assertIn("gradient_health", health_metrics)
        self.assertIn("parameter_health", health_metrics)
        self.assertIn("overall_stability_score", health_metrics)
        self.assertIn("risk_indicators", health_metrics)
    
    def test_detect_precision_issues(self):
        """Test precision issue detection."""
        # Create tensors with precision issues
        normal_tensor = torch.randn(100)
        tiny_tensor = torch.full((100,), 1e-8)
        huge_tensor = torch.full((100,), 1e8)
        
        precision_analysis = self.tracker.detect_precision_issues([
            normal_tensor, tiny_tensor, huge_tensor
        ])
        
        self.assertIsInstance(precision_analysis, dict)
        self.assertIn("underflow_risk", precision_analysis)
        self.assertIn("overflow_risk", precision_analysis)
        self.assertIn("dynamic_range", precision_analysis)
        
        # Should detect underflow and overflow risks
        self.assertTrue(precision_analysis["underflow_risk"])
        self.assertTrue(precision_analysis["overflow_risk"])
    
    def test_monitor_activation_statistics(self):
        """Test activation statistics monitoring."""
        # Create sample activations
        activations = {
            "layer_0": torch.randn(32, 64),
            "layer_1": torch.randn(32, 32),
            "layer_2": torch.randn(32, 16)
        }
        
        activation_stats = self.tracker.monitor_activation_statistics(activations)
        
        self.assertIsInstance(activation_stats, dict)
        for layer_name in activations.keys():
            self.assertIn(layer_name, activation_stats)
            layer_stats = activation_stats[layer_name]
            self.assertIn("mean", layer_stats)
            self.assertIn("std", layer_stats)
            self.assertIn("min", layer_stats)
            self.assertIn("max", layer_stats)


class TestSpectralMetrics(unittest.TestCase):
    """Test spectral metrics computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SpectralMetrics()
    
    def test_compute_comprehensive_metrics(self):
        """Test comprehensive spectral metrics computation."""
        # Create test matrices
        weight_matrices = [
            torch.randn(32, 16),
            torch.randn(16, 8),
            torch.randn(8, 4)
        ]
        
        gradients = [torch.randn_like(w) * 0.1 for w in weight_matrices]
        
        comprehensive_metrics = self.metrics.compute_comprehensive_metrics(
            weight_matrices, gradients
        )
        
        self.assertIsInstance(comprehensive_metrics, dict)
        self.assertIn("spectral_radii", comprehensive_metrics)
        self.assertIn("condition_numbers", comprehensive_metrics)
        self.assertIn("frobenius_norms", comprehensive_metrics)
        self.assertIn("gradient_alignment", comprehensive_metrics)
        self.assertIn("stability_indicators", comprehensive_metrics)
    
    def test_temporal_spectral_analysis(self):
        """Test temporal spectral analysis."""
        # Create temporal spectral data
        temporal_data = []
        for t in range(20):
            metrics = {
                "spectral_radius": 0.9 + 0.1 * np.sin(t * 0.1) + np.random.normal(0, 0.05),
                "condition_number": 10 + np.random.normal(0, 2),
                "gradient_norm": 1.0 + np.random.normal(0, 0.2),
                "timestamp": t
            }
            temporal_data.append(metrics)
        
        temporal_analysis = self.metrics.temporal_spectral_analysis(temporal_data)
        
        self.assertIsInstance(temporal_analysis, dict)
        self.assertIn("spectral_trends", temporal_analysis)
        self.assertIn("stability_periods", temporal_analysis)
        self.assertIn("anomaly_detection", temporal_analysis)
        self.assertIn("periodicity_analysis", temporal_analysis)
    
    def test_cross_layer_spectral_correlation(self):
        """Test cross-layer spectral correlation analysis."""
        # Create multi-layer spectral data
        layer_metrics = {
            "layer_0": {"spectral_radius": 0.8, "condition_number": 5.0},
            "layer_1": {"spectral_radius": 0.9, "condition_number": 8.0},
            "layer_2": {"spectral_radius": 0.7, "condition_number": 3.0}
        }
        
        correlation_analysis = self.metrics.cross_layer_spectral_correlation(layer_metrics)
        
        self.assertIsInstance(correlation_analysis, dict)
        self.assertIn("correlation_matrix", correlation_analysis)
        self.assertIn("dominant_patterns", correlation_analysis)
        self.assertIn("layer_interactions", correlation_analysis)


class TestSpectralMonitoringSystem(unittest.TestCase):
    """Test the complete spectral monitoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitoring_system = SpectralMonitoringSystem()
        
        # Create mock model for testing
        self.mock_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Linear(16, 8)
        )
        
        # Create mock training step
        dummy_input = torch.randn(16, 64)
        dummy_target = torch.randn(16, 8)
        self.mock_loss = nn.MSELoss()(self.mock_model(dummy_input), dummy_target)
        self.mock_loss.backward()
    
    def test_monitoring_system_initialization(self):
        """Test monitoring system initialization."""
        self.assertIsNotNone(self.monitoring_system.spectral_analyzer)
        self.assertIsNotNone(self.monitoring_system.gradient_conflict_detector)
        self.assertIsNotNone(self.monitoring_system.stability_monitor)
        self.assertIsNotNone(self.monitoring_system.spectral_governor)
        self.assertIsNotNone(self.monitoring_system.numerical_tracker)
        self.assertIsNotNone(self.monitoring_system.metrics_calculator)
    
    def test_comprehensive_spectral_monitoring(self):
        """Test comprehensive spectral monitoring."""
        monitoring_results = self.monitoring_system.comprehensive_monitor(
            model=self.mock_model,
            step=1,
            loss=self.mock_loss.item()
        )
        
        self.assertIsInstance(monitoring_results, dict)
        self.assertIn("spectral_analysis", monitoring_results)
        self.assertIn("gradient_conflicts", monitoring_results)
        self.assertIn("stability_assessment", monitoring_results)
        self.assertIn("numerical_health", monitoring_results)
        self.assertIn("governance_actions", monitoring_results)
        self.assertIn("overall_spectral_health", monitoring_results)
    
    def test_monitoring_over_training_steps(self):
        """Test monitoring over multiple training steps."""
        # Simulate multiple training steps
        for step in range(10):
            # Create new gradients for each step
            dummy_input = torch.randn(16, 64)
            dummy_target = torch.randn(16, 8)
            loss = nn.MSELoss()(self.mock_model(dummy_input), dummy_target)
            loss.backward()
            
            # Monitor this step
            results = self.monitoring_system.comprehensive_monitor(
                model=self.mock_model,
                step=step,
                loss=loss.item()
            )
            
            self.assertIsInstance(results, dict)
            self.assertIn("spectral_analysis", results)
        
        # Get temporal analysis
        temporal_analysis = self.monitoring_system.get_temporal_analysis()
        
        self.assertIsInstance(temporal_analysis, dict)
        self.assertIn("spectral_trends", temporal_analysis)
        self.assertIn("stability_evolution", temporal_analysis)
    
    def test_save_and_load_monitoring_data(self):
        """Test saving and loading monitoring data."""
        # Run some monitoring steps
        for step in range(5):
            dummy_input = torch.randn(16, 64)
            dummy_target = torch.randn(16, 8)
            loss = nn.MSELoss()(self.mock_model(dummy_input), dummy_target)
            loss.backward()
            
            self.monitoring_system.comprehensive_monitor(
                model=self.mock_model,
                step=step,
                loss=loss.item()
            )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "spectral_monitoring.json"
            
            # Save monitoring data
            self.monitoring_system.save_monitoring_data(save_path)
            self.assertTrue(save_path.exists())
            
            # Load monitoring data
            loaded_data = self.monitoring_system.load_monitoring_data(save_path)
            self.assertIsInstance(loaded_data, dict)
            self.assertIn("monitoring_history", loaded_data)
            self.assertIn("summary_statistics", loaded_data)
    
    def test_generate_monitoring_report(self):
        """Test monitoring report generation."""
        # Run monitoring for several steps
        for step in range(8):
            dummy_input = torch.randn(16, 64)
            dummy_target = torch.randn(16, 8)
            loss = nn.MSELoss()(self.mock_model(dummy_input), dummy_target)
            loss.backward()
            
            self.monitoring_system.comprehensive_monitor(
                model=self.mock_model,
                step=step,
                loss=loss.item()
            )
        
        # Generate report
        report = self.monitoring_system.generate_monitoring_report()
        
        self.assertIsInstance(report, str)
        self.assertIn("Spectral Monitoring Report", report)
        self.assertIn("Spectral Analysis Summary", report)
        self.assertIn("Stability Assessment", report)
        self.assertIn("Recommendations", report)


class TestSpectralMonitoringIntegration(unittest.TestCase):
    """Integration tests for spectral monitoring components."""
    
    def test_end_to_end_monitoring_workflow(self):
        """Test complete end-to-end monitoring workflow."""
        # Create a realistic model
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        monitoring_system = SpectralMonitoringSystem(
            config={
                "spectral_radius_threshold": 1.2,
                "gradient_conflict_threshold": 0.8,
                "stability_window": 10
            }
        )
        
        # Simulate training with monitoring
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        monitoring_results = []
        
        for epoch in range(5):
            for step in range(20):
                # Forward pass
                inputs = torch.randn(32, 128)
                targets = torch.randint(0, 10, (32,))
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Monitor before optimizer step
                monitor_result = monitoring_system.comprehensive_monitor(
                    model=model,
                    step=epoch * 20 + step,
                    loss=loss.item()
                )
                monitoring_results.append(monitor_result)
                
                optimizer.step()
        
        # Verify monitoring captured important metrics
        self.assertEqual(len(monitoring_results), 100)  # 5 epochs * 20 steps
        
        # Check that each monitoring result has expected structure
        for result in monitoring_results:
            self.assertIn("spectral_analysis", result)
            self.assertIn("stability_assessment", result)
            self.assertIn("overall_spectral_health", result)
        
        # Generate final report
        final_report = monitoring_system.generate_monitoring_report()
        self.assertIsInstance(final_report, str)
        self.assertIn("training steps monitored", final_report.lower())


if __name__ == '__main__':
    # Suppress some numerical warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)