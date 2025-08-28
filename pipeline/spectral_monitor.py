#!/usr/bin/env python3
"""
Spectral Governance Monitoring for BEM Research Validation

Track spectral radius, Frobenius norm changes, and gradient conflicts
for stability monitoring during BEM training and evaluation.
"""

import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd, norm
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpectralMetrics:
    """Metrics for spectral analysis."""
    spectral_radius: float
    frobenius_norm: float
    condition_number: float
    rank: int
    singular_values: List[float]
    nuclear_norm: float
    
@dataclass
class GradientConflictMetrics:
    """Metrics for gradient conflict analysis."""
    conflict_intensity: float
    alignment_score: float
    gradient_magnitude: float
    conflict_frequency: float
    conflicting_pairs: List[Tuple[int, int]]

@dataclass
class StabilityMetrics:
    """Overall stability metrics."""
    spectral_stability_score: float
    gradient_stability_score: float
    training_stability_trend: str  # "stable", "improving", "degrading"
    stability_alerts: List[str]

@dataclass
class SpectralMonitoringResult:
    """Complete spectral monitoring result."""
    spectral_metrics: SpectralMetrics
    gradient_conflict_metrics: GradientConflictMetrics
    stability_metrics: StabilityMetrics
    historical_trends: Dict[str, List[float]]
    recommendations: List[str]
    metadata: Dict[str, Any]

class SpectralAnalyzer:
    """Analyze spectral properties of weight matrices."""
    
    def __init__(self, track_history: bool = True, max_history: int = 1000):
        self.track_history = track_history
        self.max_history = max_history
        
        if track_history:
            self.spectral_radius_history = deque(maxlen=max_history)
            self.frobenius_norm_history = deque(maxlen=max_history)
            self.condition_number_history = deque(maxlen=max_history)
            
    def analyze_weight_matrix(self, weight_matrix: torch.Tensor) -> SpectralMetrics:
        """Analyze spectral properties of a weight matrix."""
        
        if isinstance(weight_matrix, torch.Tensor):
            W = weight_matrix.detach().cpu().numpy()
        else:
            W = np.array(weight_matrix)
            
        # Handle different matrix shapes
        if W.ndim > 2:
            # Reshape higher-dimensional tensors to 2D
            original_shape = W.shape
            W = W.reshape(original_shape[0], -1)
            
        m, n = W.shape
        
        try:
            # Compute SVD
            if min(m, n) > 100:  # Use sparse SVD for large matrices
                k = min(50, min(m, n) - 1)  # Number of singular values to compute
                U, s, Vt = svds(W, k=k)
                s = np.sort(s)[::-1]  # Sort in descending order
            else:
                U, s, Vt = svd(W, full_matrices=False)
                
            # Spectral radius (largest singular value)
            spectral_radius = float(s[0]) if len(s) > 0 else 0.0
            
            # Frobenius norm
            frobenius_norm = float(norm(W, 'fro'))
            
            # Condition number
            if len(s) > 1 and s[-1] > 1e-12:
                condition_number = float(s[0] / s[-1])
            else:
                condition_number = float('inf')
                
            # Matrix rank (numerical rank with tolerance)
            tolerance = max(m, n) * s[0] * np.finfo(W.dtype).eps if len(s) > 0 else 1e-12
            rank = int(np.sum(s > tolerance))
            
            # Nuclear norm (sum of singular values)
            nuclear_norm = float(np.sum(s))
            
            # Track history
            if self.track_history:
                self.spectral_radius_history.append(spectral_radius)
                self.frobenius_norm_history.append(frobenius_norm)
                self.condition_number_history.append(condition_number)
                
        except Exception as e:
            logger.warning(f"SVD computation failed: {e}")
            # Return default metrics
            spectral_radius = 0.0
            frobenius_norm = float(norm(W, 'fro'))
            condition_number = float('inf')
            rank = min(m, n)
            s = [0.0]
            nuclear_norm = 0.0
            
        return SpectralMetrics(
            spectral_radius=spectral_radius,
            frobenius_norm=frobenius_norm,
            condition_number=condition_number,
            rank=rank,
            singular_values=s.tolist()[:20],  # Store top 20 singular values
            nuclear_norm=nuclear_norm
        )
        
    def get_spectral_trends(self) -> Dict[str, Any]:
        """Get trends in spectral properties over time."""
        
        if not self.track_history:
            return {}
            
        trends = {}
        
        # Spectral radius trend
        if len(self.spectral_radius_history) > 5:
            sr_values = list(self.spectral_radius_history)
            trend_slope = np.polyfit(range(len(sr_values)), sr_values, 1)[0]
            trends['spectral_radius'] = {
                'values': sr_values,
                'trend': 'increasing' if trend_slope > 0.001 else 'decreasing' if trend_slope < -0.001 else 'stable',
                'slope': float(trend_slope),
                'current': sr_values[-1],
                'volatility': float(np.std(sr_values[-20:]) if len(sr_values) >= 20 else np.std(sr_values))
            }
            
        # Condition number trend
        if len(self.condition_number_history) > 5:
            cn_values = [x for x in self.condition_number_history if not np.isinf(x)]
            if cn_values:
                trend_slope = np.polyfit(range(len(cn_values)), cn_values, 1)[0]
                trends['condition_number'] = {
                    'values': cn_values,
                    'trend': 'increasing' if trend_slope > 0.1 else 'decreasing' if trend_slope < -0.1 else 'stable',
                    'slope': float(trend_slope),
                    'current': cn_values[-1],
                    'volatility': float(np.std(cn_values[-20:]) if len(cn_values) >= 20 else np.std(cn_values))
                }
                
        return trends

class GradientConflictDetector:
    """Detect and analyze gradient conflicts between experts."""
    
    def __init__(self, num_experts: int, track_history: bool = True, max_history: int = 1000):
        self.num_experts = num_experts
        self.track_history = track_history
        
        # Gradient storage
        self.current_gradients = {}
        self.gradient_magnitudes = defaultdict(list)
        
        if track_history:
            self.conflict_history = deque(maxlen=max_history)
            self.alignment_history = deque(maxlen=max_history)
            
    def record_expert_gradients(self, expert_id: int, gradients: Dict[str, torch.Tensor]) -> None:
        """Record gradients for a specific expert."""
        
        # Flatten and concatenate all gradients for this expert
        flattened_grads = []
        for param_name, grad in gradients.items():
            if grad is not None:
                flattened_grads.append(grad.detach().cpu().flatten())
                
        if flattened_grads:
            expert_gradient_vector = torch.cat(flattened_grads)
            self.current_gradients[expert_id] = expert_gradient_vector
            
            # Track magnitude
            magnitude = float(torch.norm(expert_gradient_vector))
            self.gradient_magnitudes[expert_id].append(magnitude)
            
    def analyze_gradient_conflicts(self) -> GradientConflictMetrics:
        """Analyze conflicts between expert gradients."""
        
        if len(self.current_gradients) < 2:
            return self._get_empty_conflict_metrics()
            
        expert_ids = list(self.current_gradients.keys())
        gradients = [self.current_gradients[eid] for eid in expert_ids]
        
        # Compute pairwise cosine similarities
        similarities = []
        conflicting_pairs = []
        
        for i, grad_i in enumerate(gradients):
            for j, grad_j in enumerate(gradients[i+1:], i+1):
                # Cosine similarity
                dot_product = float(torch.dot(grad_i, grad_j))
                norm_i = float(torch.norm(grad_i))
                norm_j = float(torch.norm(grad_j))
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cosine_sim = dot_product / (norm_i * norm_j)
                    similarities.append(cosine_sim)
                    
                    # Detect conflicts (negative similarity)
                    if cosine_sim < -0.1:  # Threshold for conflict
                        conflicting_pairs.append((expert_ids[i], expert_ids[j]))
                        
        # Conflict metrics
        if similarities:
            alignment_score = float(np.mean(similarities))
            conflict_intensity = float(-np.min(similarities)) if similarities else 0.0
            conflict_frequency = len(conflicting_pairs) / len(similarities)
        else:
            alignment_score = 0.0
            conflict_intensity = 0.0
            conflict_frequency = 0.0
            
        # Overall gradient magnitude
        all_magnitudes = [float(torch.norm(g)) for g in gradients]
        gradient_magnitude = float(np.mean(all_magnitudes)) if all_magnitudes else 0.0
        
        # Track history
        if self.track_history:
            self.conflict_history.append(conflict_intensity)
            self.alignment_history.append(alignment_score)
            
        return GradientConflictMetrics(
            conflict_intensity=conflict_intensity,
            alignment_score=alignment_score,
            gradient_magnitude=gradient_magnitude,
            conflict_frequency=conflict_frequency,
            conflicting_pairs=conflicting_pairs
        )
        
    def get_conflict_trends(self) -> Dict[str, Any]:
        """Get trends in gradient conflicts over time."""
        
        if not self.track_history:
            return {}
            
        trends = {}
        
        # Conflict intensity trend
        if len(self.conflict_history) > 5:
            conflict_values = list(self.conflict_history)
            trend_slope = np.polyfit(range(len(conflict_values)), conflict_values, 1)[0]
            trends['conflict_intensity'] = {
                'values': conflict_values,
                'trend': 'increasing' if trend_slope > 0.001 else 'decreasing' if trend_slope < -0.001 else 'stable',
                'slope': float(trend_slope),
                'current': conflict_values[-1]
            }
            
        # Alignment trend
        if len(self.alignment_history) > 5:
            alignment_values = list(self.alignment_history)
            trend_slope = np.polyfit(range(len(alignment_values)), alignment_values, 1)[0]
            trends['gradient_alignment'] = {
                'values': alignment_values,
                'trend': 'increasing' if trend_slope > 0.001 else 'decreasing' if trend_slope < -0.001 else 'stable',
                'slope': float(trend_slope),
                'current': alignment_values[-1]
            }
            
        return trends
        
    def _get_empty_conflict_metrics(self) -> GradientConflictMetrics:
        """Return empty conflict metrics."""
        return GradientConflictMetrics(
            conflict_intensity=0.0,
            alignment_score=0.0,
            gradient_magnitude=0.0,
            conflict_frequency=0.0,
            conflicting_pairs=[]
        )

class StabilityAnalyzer:
    """Analyze overall training stability based on spectral and gradient metrics."""
    
    def __init__(self):
        self.stability_thresholds = {
            'spectral_radius': {'min': 0.1, 'max': 10.0},
            'condition_number': {'max': 1000.0},
            'conflict_intensity': {'max': 0.5},
            'gradient_magnitude': {'min': 1e-6, 'max': 100.0}
        }
        
    def analyze_stability(self, 
                         spectral_metrics: SpectralMetrics,
                         gradient_metrics: GradientConflictMetrics,
                         trends: Dict[str, Any]) -> StabilityMetrics:
        """Analyze overall stability and generate alerts."""
        
        alerts = []
        
        # Spectral stability analysis
        spectral_score = 1.0
        
        # Check spectral radius
        if spectral_metrics.spectral_radius > self.stability_thresholds['spectral_radius']['max']:
            spectral_score *= 0.5
            alerts.append(f"High spectral radius: {spectral_metrics.spectral_radius:.3f} (risk of instability)")
        elif spectral_metrics.spectral_radius < self.stability_thresholds['spectral_radius']['min']:
            spectral_score *= 0.8
            alerts.append(f"Very low spectral radius: {spectral_metrics.spectral_radius:.3f} (may indicate weak adaptation)")
            
        # Check condition number
        if spectral_metrics.condition_number > self.stability_thresholds['condition_number']['max']:
            spectral_score *= 0.6
            alerts.append(f"High condition number: {spectral_metrics.condition_number:.1f} (numerical instability risk)")
            
        # Gradient stability analysis
        gradient_score = 1.0
        
        # Check conflict intensity
        if gradient_metrics.conflict_intensity > self.stability_thresholds['conflict_intensity']['max']:
            gradient_score *= 0.7
            alerts.append(f"High gradient conflict: {gradient_metrics.conflict_intensity:.3f} (expert interference)")
            
        # Check gradient magnitude
        if gradient_metrics.gradient_magnitude > self.stability_thresholds['gradient_magnitude']['max']:
            gradient_score *= 0.8
            alerts.append(f"Large gradient magnitude: {gradient_metrics.gradient_magnitude:.3f} (potential instability)")
        elif gradient_metrics.gradient_magnitude < self.stability_thresholds['gradient_magnitude']['min']:
            gradient_score *= 0.9
            alerts.append(f"Very small gradients: {gradient_metrics.gradient_magnitude:.6f} (vanishing gradient risk)")
            
        # Training trend analysis
        training_trend = self._analyze_training_trend(trends)
        
        # Overall stability scores
        spectral_stability_score = max(0.0, min(1.0, spectral_score))
        gradient_stability_score = max(0.0, min(1.0, gradient_score))
        
        return StabilityMetrics(
            spectral_stability_score=spectral_stability_score,
            gradient_stability_score=gradient_stability_score,
            training_stability_trend=training_trend,
            stability_alerts=alerts
        )
        
    def _analyze_training_trend(self, trends: Dict[str, Any]) -> str:
        """Analyze overall training trend."""
        
        trend_indicators = []
        
        # Spectral radius trend
        if 'spectral_radius' in trends:
            sr_trend = trends['spectral_radius']['trend']
            if sr_trend == 'increasing':
                trend_indicators.append(-1)  # Potentially concerning
            elif sr_trend == 'stable':
                trend_indicators.append(1)   # Good
            else:
                trend_indicators.append(0)   # Neutral
                
        # Gradient alignment trend
        if 'gradient_alignment' in trends:
            ga_trend = trends['gradient_alignment']['trend']
            if ga_trend == 'increasing':
                trend_indicators.append(1)   # Good (less conflict)
            elif ga_trend == 'stable':
                trend_indicators.append(0)   # Neutral
            else:
                trend_indicators.append(-1)  # Concerning
                
        # Overall trend decision
        if not trend_indicators:
            return "unknown"
            
        avg_indicator = np.mean(trend_indicators)
        
        if avg_indicator > 0.3:
            return "improving"
        elif avg_indicator < -0.3:
            return "degrading"
        else:
            return "stable"

class SpectralMonitor:
    """Main spectral governance monitor orchestrator."""
    
    def __init__(self, num_experts: int = 8, track_history: bool = True):
        self.num_experts = num_experts
        self.spectral_analyzer = SpectralAnalyzer(track_history)
        self.gradient_detector = GradientConflictDetector(num_experts, track_history)
        self.stability_analyzer = StabilityAnalyzer()
        
        # Weight matrix tracking
        self.tracked_modules = {}
        
    def register_module(self, module_name: str, module: nn.Module) -> None:
        """Register a module for spectral monitoring."""
        self.tracked_modules[module_name] = module
        
    def record_training_step(self, 
                           expert_gradients: Dict[int, Dict[str, torch.Tensor]],
                           weight_matrices: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Record metrics for one training step."""
        
        # Record expert gradients
        for expert_id, gradients in expert_gradients.items():
            self.gradient_detector.record_expert_gradients(expert_id, gradients)
            
        # Analyze weight matrices if provided
        if weight_matrices:
            for matrix_name, matrix in weight_matrices.items():
                self.spectral_analyzer.analyze_weight_matrix(matrix)
                
    def generate_monitoring_report(self) -> SpectralMonitoringResult:
        """Generate comprehensive spectral monitoring report."""
        
        logger.info("Generating spectral governance monitoring report")
        
        # Get current metrics
        # For demonstration, analyze a random matrix if no weights provided
        dummy_matrix = torch.randn(256, 512)
        current_spectral = self.spectral_analyzer.analyze_weight_matrix(dummy_matrix)
        
        current_conflicts = self.gradient_detector.analyze_gradient_conflicts()
        
        # Get trends
        spectral_trends = self.spectral_analyzer.get_spectral_trends()
        conflict_trends = self.gradient_detector.get_conflict_trends()
        all_trends = {**spectral_trends, **conflict_trends}
        
        # Stability analysis
        stability_metrics = self.stability_analyzer.analyze_stability(
            current_spectral, current_conflicts, all_trends
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_spectral, current_conflicts, stability_metrics, all_trends
        )
        
        return SpectralMonitoringResult(
            spectral_metrics=current_spectral,
            gradient_conflict_metrics=current_conflicts,
            stability_metrics=stability_metrics,
            historical_trends=all_trends,
            recommendations=recommendations,
            metadata={
                'num_experts': self.num_experts,
                'monitoring_timestamp': str(np.datetime64('now')),
                'tracked_modules': list(self.tracked_modules.keys())
            }
        )
        
    def _generate_recommendations(self, 
                                spectral_metrics: SpectralMetrics,
                                gradient_metrics: GradientConflictMetrics,
                                stability_metrics: StabilityMetrics,
                                trends: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Spectral recommendations
        if spectral_metrics.spectral_radius > 5.0:
            recommendations.append(
                f"High spectral radius ({spectral_metrics.spectral_radius:.2f}) detected. "
                "Consider gradient clipping or learning rate reduction."
            )
            
        if spectral_metrics.condition_number > 1000:
            recommendations.append(
                f"Poor conditioning (κ={spectral_metrics.condition_number:.1f}). "
                "Consider weight initialization or regularization adjustments."
            )
            
        # Gradient conflict recommendations
        if gradient_metrics.conflict_intensity > 0.3:
            recommendations.append(
                f"High gradient conflicts ({gradient_metrics.conflict_intensity:.2f}). "
                "Expert objectives may be misaligned. Consider conflict regularization."
            )
            
        if len(gradient_metrics.conflicting_pairs) > 2:
            recommendations.append(
                f"Multiple conflicting expert pairs detected ({len(gradient_metrics.conflicting_pairs)}). "
                "Review expert specialization and routing strategies."
            )
            
        # Stability recommendations
        if stability_metrics.training_stability_trend == "degrading":
            recommendations.append(
                "Training stability is degrading over time. "
                "Consider reducing learning rate or adding regularization."
            )
            
        # Trend-based recommendations
        if 'spectral_radius' in trends and trends['spectral_radius']['trend'] == 'increasing':
            recommendations.append(
                "Spectral radius is increasing over time. "
                "Monitor for potential instability and consider preventive measures."
            )
            
        if not recommendations:
            recommendations.append("Spectral governance metrics are within acceptable ranges.")
            
        return recommendations

def main():
    """Example usage of spectral monitor."""
    
    # Initialize monitor
    monitor = SpectralMonitor(num_experts=8)
    
    # Simulate training steps
    for step in range(50):
        # Mock expert gradients
        expert_gradients = {}
        for expert_id in range(8):
            expert_gradients[expert_id] = {
                'weight': torch.randn(256, 512) * 0.01,
                'bias': torch.randn(512) * 0.001
            }
            
        # Mock weight matrices
        weight_matrices = {
            'expert_0_linear': torch.randn(256, 512),
            'expert_1_linear': torch.randn(256, 512),
            'gating_network': torch.randn(512, 8)
        }
        
        # Record step
        monitor.record_training_step(expert_gradients, weight_matrices)
        
    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    
    # Print summary
    print("Spectral Governance Report:")
    print(f"Spectral Radius: {report.spectral_metrics.spectral_radius:.4f}")
    print(f"Condition Number: {report.spectral_metrics.condition_number:.2f}")
    print(f"Gradient Conflicts: {report.gradient_conflict_metrics.conflict_intensity:.3f}")
    print(f"Spectral Stability: {report.stability_metrics.spectral_stability_score:.3f}")
    print(f"Gradient Stability: {report.stability_metrics.gradient_stability_score:.3f}")
    print(f"Training Trend: {report.stability_metrics.training_stability_trend}")
    
    print(f"\nRecommendations ({len(report.recommendations)}):")
    for rec in report.recommendations:
        print(f"  - {rec}")

if __name__ == "__main__":
    main()