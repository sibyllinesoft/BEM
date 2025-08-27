"""
Counterfactual Routing for BEM - Phase 5 Implementation.

Implements credit assignment through component dropouts and routing optimization
as specified in TODO.md Phase 5.

Key Features:
- Component Dropout: Randomly disable routing components
- Credit Assignment: Measure performance impact of each component
- Routing Optimization: Improve routing based on counterfactuals
- Ablation Analysis: Systematic component importance measurement
"""

from typing import Dict, List, Optional, Tuple, Union, NamedTuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict, deque
import itertools
import random
from abc import ABC, abstractmethod

from .telemetry import TelemetryCollector


class ComponentType(Enum):
    """Types of components that can be dropped for counterfactual analysis."""
    PREFIX_ROUTER = "prefix_router"
    CHUNK_ROUTER = "chunk_router" 
    TOKEN_ROUTER = "token_router"
    UNCERTAINTY_HEAD = "uncertainty_head"
    EXPERT_BANK = "expert_bank"
    RETRIEVAL_FEATURES = "retrieval_features"
    TRUST_REGION = "trust_region"
    VQ_CODEBOOK = "vq_codebook"


class CounterfactualMetrics(NamedTuple):
    """Metrics for counterfactual routing analysis."""
    component_importance_scores: Dict[str, float]
    dropout_impacts: Dict[str, float]
    routing_efficiency: float
    credit_assignment_variance: float
    ablation_consistency: float
    interaction_effects: Dict[Tuple[str, str], float]


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual routing system."""
    
    # Dropout parameters
    component_dropout_rate: float = 0.1
    min_active_components: int = 1
    dropout_schedule: str = "fixed"  # "fixed", "adaptive", "scheduled"
    
    # Analysis parameters
    analysis_frequency: int = 100  # Steps between analyses
    analysis_window: int = 50  # Number of samples per analysis
    importance_smoothing: float = 0.9  # EMA smoothing for importance scores
    
    # Credit assignment
    performance_metric: str = "loss"  # "loss", "accuracy", "perplexity", "custom"
    baseline_estimation: str = "moving_average"  # "moving_average", "control_group"
    baseline_window: int = 200
    
    # Interaction analysis
    enable_interaction_analysis: bool = True
    max_interaction_order: int = 2  # Pairwise interactions
    interaction_significance_threshold: float = 0.05
    
    # Optimization
    enable_routing_optimization: bool = True
    optimization_frequency: int = 500
    learning_rate_adjustment: float = 0.1
    gradient_scaling: bool = True
    
    # Ablation studies
    systematic_ablation: bool = True
    ablation_sample_size: int = 20
    statistical_significance: float = 0.05


class ComponentDropoutMask:
    """Manages dropout masks for different components."""
    
    def __init__(self, config: CounterfactualConfig):
        self.config = config
        self.active_dropouts = set()
        self.dropout_history = deque(maxlen=1000)
        
    def generate_dropout_mask(
        self,
        available_components: List[str],
        current_step: int
    ) -> Dict[str, bool]:
        """
        Generate dropout mask for components.
        
        Returns:
            Dictionary mapping component names to whether they should be active (True) or dropped (False)
        """
        mask = {component: True for component in available_components}
        
        if self.config.dropout_schedule == "fixed":
            # Fixed probability dropout
            num_to_drop = max(0, int(len(available_components) * self.config.component_dropout_rate))
            num_to_drop = min(num_to_drop, len(available_components) - self.config.min_active_components)
            
            if num_to_drop > 0:
                components_to_drop = random.sample(available_components, num_to_drop)
                for component in components_to_drop:
                    mask[component] = False
                    
        elif self.config.dropout_schedule == "adaptive":
            # Adaptive dropout based on component importance
            # (This would require importance scores from previous analysis)
            pass  # Implementation would depend on stored importance scores
            
        elif self.config.dropout_schedule == "scheduled":
            # Scheduled dropout that increases over time
            schedule_factor = min(1.0, current_step / 10000)  # Ramp up over 10k steps
            effective_rate = self.config.component_dropout_rate * schedule_factor
            
            num_to_drop = max(0, int(len(available_components) * effective_rate))
            num_to_drop = min(num_to_drop, len(available_components) - self.config.min_active_components)
            
            if num_to_drop > 0:
                components_to_drop = random.sample(available_components, num_to_drop)
                for component in components_to_drop:
                    mask[component] = False
        
        # Record dropout decision
        dropped_components = [c for c, active in mask.items() if not active]
        self.dropout_history.append({
            'step': current_step,
            'dropped': dropped_components,
            'active': [c for c, active in mask.items() if active]
        })
        
        return mask


class PerformanceBaseline:
    """Tracks performance baseline for credit assignment."""
    
    def __init__(self, config: CounterfactualConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.baseline_window)
        self.baseline_value = 0.0
        self.baseline_variance = 1.0
        
    def update(self, performance_value: float):
        """Update baseline with new performance measurement."""
        self.performance_history.append(performance_value)
        
        if self.config.baseline_estimation == "moving_average":
            self.baseline_value = np.mean(self.performance_history)
            self.baseline_variance = np.var(self.performance_history) if len(self.performance_history) > 1 else 1.0
        
    def get_performance_impact(self, current_performance: float) -> float:
        """Calculate performance impact relative to baseline."""
        if self.config.performance_metric == "loss":
            # For loss, lower is better, so negative impact means improvement
            impact = current_performance - self.baseline_value
        else:
            # For metrics like accuracy, higher is better
            impact = self.baseline_value - current_performance
            
        # Normalize by baseline variance
        normalized_impact = impact / (np.sqrt(self.baseline_variance) + 1e-8)
        return normalized_impact


class ComponentImportanceTracker:
    """Tracks importance scores for different components."""
    
    def __init__(self, config: CounterfactualConfig):
        self.config = config
        self.importance_scores = defaultdict(float)
        self.impact_history = defaultdict(list)
        self.confidence_intervals = defaultdict(tuple)
        
    def update_importance(
        self,
        component: str,
        performance_impact: float,
        was_dropped: bool
    ):
        """Update importance score for a component."""
        # Record the impact
        self.impact_history[component].append({
            'impact': performance_impact,
            'dropped': was_dropped,
            'timestamp': len(self.impact_history[component])
        })
        
        # Keep only recent history
        max_history = 200
        if len(self.impact_history[component]) > max_history:
            self.impact_history[component] = self.impact_history[component][-max_history:]
        
        # Calculate importance as the average impact when component is dropped
        dropped_impacts = [
            entry['impact'] for entry in self.impact_history[component]
            if entry['dropped']
        ]
        
        if dropped_impacts:
            new_importance = np.mean(dropped_impacts)
            
            # Apply EMA smoothing
            self.importance_scores[component] = (
                self.config.importance_smoothing * self.importance_scores[component] +
                (1 - self.config.importance_smoothing) * new_importance
            )
            
            # Update confidence interval
            if len(dropped_impacts) > 3:
                std_error = np.std(dropped_impacts) / np.sqrt(len(dropped_impacts))
                margin = 1.96 * std_error  # 95% confidence interval
                mean_impact = np.mean(dropped_impacts)
                self.confidence_intervals[component] = (
                    mean_impact - margin,
                    mean_impact + margin
                )
    
    def get_ranked_components(self) -> List[Tuple[str, float, Tuple[float, float]]]:
        """Get components ranked by importance with confidence intervals."""
        ranked = []
        for component, importance in self.importance_scores.items():
            ci = self.confidence_intervals.get(component, (importance, importance))
            ranked.append((component, importance, ci))
        
        # Sort by importance (higher impact when dropped = more important)
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class InteractionAnalyzer:
    """Analyzes interaction effects between components."""
    
    def __init__(self, config: CounterfactualConfig):
        self.config = config
        self.interaction_data = defaultdict(list)
        self.interaction_scores = defaultdict(float)
        
    def record_interaction(
        self,
        dropped_components: List[str],
        performance_impact: float
    ):
        """Record performance impact for a specific component combination."""
        if len(dropped_components) <= self.config.max_interaction_order:
            # Create interaction key (sorted for consistency)
            interaction_key = tuple(sorted(dropped_components))
            self.interaction_data[interaction_key].append(performance_impact)
    
    def analyze_interactions(self, component_importance: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        """Analyze pairwise interaction effects."""
        interactions = {}
        
        # Only analyze pairwise interactions for now
        for interaction_key, impacts in self.interaction_data.items():
            if len(interaction_key) == 2 and len(impacts) >= 5:  # Need sufficient data
                comp_a, comp_b = interaction_key
                
                # Expected impact if components were independent
                expected_impact = component_importance.get(comp_a, 0) + component_importance.get(comp_b, 0)
                
                # Actual impact when both are dropped
                actual_impact = np.mean(impacts)
                
                # Interaction effect is the difference
                interaction_effect = actual_impact - expected_impact
                interactions[(comp_a, comp_b)] = interaction_effect
        
        return interactions


class CounterfactualRoutingAnalyzer:
    """Main analyzer for counterfactual routing experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        config: CounterfactualConfig,
        performance_evaluator: Callable[[torch.Tensor, torch.Tensor], float],
        telemetry_collector: Optional[TelemetryCollector] = None
    ):
        self.model = model
        self.config = config
        self.performance_evaluator = performance_evaluator
        self.telemetry = telemetry_collector
        
        # Components
        self.dropout_mask = ComponentDropoutMask(config)
        self.performance_baseline = PerformanceBaseline(config)
        self.importance_tracker = ComponentImportanceTracker(config)
        self.interaction_analyzer = InteractionAnalyzer(config)
        
        # State tracking
        self.step_count = 0
        self.analysis_results = deque(maxlen=100)
        
        # Component registry
        self.registered_components = set()
        self.component_hooks = {}
        
        self.logger = logging.getLogger(__name__)
    
    def register_component(
        self, 
        component_name: str, 
        component_module: nn.Module,
        dropout_fn: Callable[[nn.Module, bool], None]
    ):
        """
        Register a component for counterfactual analysis.
        
        Args:
            component_name: Name of the component
            component_module: The PyTorch module
            dropout_fn: Function to enable/disable the component
        """
        self.registered_components.add(component_name)
        self.component_hooks[component_name] = {
            'module': component_module,
            'dropout_fn': dropout_fn,
            'is_active': True
        }
        
        self.logger.info(f"Registered component: {component_name}")
    
    def step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        perform_analysis: bool = None
    ) -> Dict[str, Any]:
        """
        Perform one step of counterfactual analysis.
        
        Args:
            inputs: Model inputs
            targets: Target outputs for performance evaluation
            perform_analysis: Whether to perform analysis this step (auto if None)
            
        Returns:
            Dictionary with analysis results
        """
        self.step_count += 1
        
        # Decide whether to perform analysis
        if perform_analysis is None:
            perform_analysis = (self.step_count % self.config.analysis_frequency == 0)
        
        if perform_analysis and len(self.registered_components) > 0:
            return self._perform_counterfactual_analysis(inputs, targets)
        else:
            # Normal forward pass without analysis
            outputs = self.model(inputs)
            performance = self.performance_evaluator(outputs, targets)
            self.performance_baseline.update(performance)
            
            return {
                'performance': performance,
                'analysis_performed': False,
                'step': self.step_count
            }
    
    def _perform_counterfactual_analysis(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """Perform comprehensive counterfactual analysis."""
        analysis_start_time = time.time() if 'time' in globals() else 0
        
        # Collect baseline performance
        baseline_outputs = self.model(inputs)
        baseline_performance = self.performance_evaluator(baseline_outputs, targets)
        self.performance_baseline.update(baseline_performance)
        
        # Perform dropout experiments
        component_impacts = {}
        interaction_data = []
        
        for _ in range(self.config.analysis_window):
            # Generate dropout mask
            dropout_mask = self.dropout_mask.generate_dropout_mask(
                list(self.registered_components),
                self.step_count
            )
            
            # Apply dropout mask
            dropped_components = []
            for component_name, is_active in dropout_mask.items():
                if component_name in self.component_hooks:
                    self.component_hooks[component_name]['dropout_fn'](
                        self.component_hooks[component_name]['module'],
                        is_active
                    )
                    self.component_hooks[component_name]['is_active'] = is_active
                    if not is_active:
                        dropped_components.append(component_name)
            
            # Forward pass with dropout
            try:
                outputs = self.model(inputs)
                performance = self.performance_evaluator(outputs, targets)
                
                # Calculate impact
                impact = self.performance_baseline.get_performance_impact(performance)
                
                # Update component importance
                for component_name in self.registered_components:
                    was_dropped = component_name in dropped_components
                    self.importance_tracker.update_importance(
                        component_name, impact, was_dropped
                    )
                
                # Record for interaction analysis
                if dropped_components:
                    self.interaction_analyzer.record_interaction(dropped_components, impact)
                    interaction_data.append((dropped_components.copy(), impact))
                
            except Exception as e:
                self.logger.warning(f"Error during counterfactual analysis: {e}")
                continue
            
            finally:
                # Restore all components
                for component_name in self.component_hooks:
                    self.component_hooks[component_name]['dropout_fn'](
                        self.component_hooks[component_name]['module'],
                        True  # Restore to active state
                    )
                    self.component_hooks[component_name]['is_active'] = True
        
        # Analyze results
        ranked_components = self.importance_tracker.get_ranked_components()
        component_importance_dict = dict(self.importance_tracker.importance_scores)
        
        # Interaction analysis
        interaction_effects = {}
        if self.config.enable_interaction_analysis:
            interaction_effects = self.interaction_analyzer.analyze_interactions(
                component_importance_dict
            )
        
        # Compute routing efficiency
        routing_efficiency = self._compute_routing_efficiency(ranked_components)
        
        # Credit assignment variance
        credit_variance = self._compute_credit_variance()
        
        # Ablation consistency
        ablation_consistency = self._compute_ablation_consistency()
        
        # Create metrics
        metrics = CounterfactualMetrics(
            component_importance_scores=component_importance_dict,
            dropout_impacts={comp: impacts[-1]['impact'] if impacts else 0.0 
                           for comp, impacts in self.importance_tracker.impact_history.items()},
            routing_efficiency=routing_efficiency,
            credit_assignment_variance=credit_variance,
            ablation_consistency=ablation_consistency,
            interaction_effects=interaction_effects
        )
        
        # Store results
        analysis_result = {
            'step': self.step_count,
            'baseline_performance': baseline_performance,
            'ranked_components': ranked_components,
            'interaction_effects': interaction_effects,
            'metrics': metrics,
            'analysis_time': time.time() - analysis_start_time if analysis_start_time else 0
        }
        
        self.analysis_results.append(analysis_result)
        
        # Log telemetry
        if self.telemetry:
            self.telemetry.log_counterfactual_metrics(metrics)
        
        # Routing optimization
        if (self.config.enable_routing_optimization and 
            self.step_count % self.config.optimization_frequency == 0):
            self._optimize_routing(ranked_components)
        
        return analysis_result
    
    def _compute_routing_efficiency(self, ranked_components: List[Tuple[str, float, Tuple[float, float]]]) -> float:
        """Compute overall routing efficiency metric."""
        if not ranked_components:
            return 0.0
        
        # Efficiency is based on the concentration of importance in top components
        importances = [importance for _, importance, _ in ranked_components]
        total_importance = sum(abs(imp) for imp in importances)
        
        if total_importance == 0:
            return 0.0
        
        # Calculate Gini coefficient for importance distribution
        sorted_importances = sorted([abs(imp) for imp in importances])
        n = len(sorted_importances)
        cumulative = np.cumsum(sorted_importances)
        gini = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_importances) / cumulative[-1]) / n
        
        # Higher Gini means more concentrated importance (more efficient routing)
        return gini
    
    def _compute_credit_variance(self) -> float:
        """Compute variance in credit assignment."""
        all_variances = []
        
        for component, history in self.importance_tracker.impact_history.items():
            if len(history) > 1:
                impacts = [entry['impact'] for entry in history if entry['dropped']]
                if len(impacts) > 1:
                    all_variances.append(np.var(impacts))
        
        return np.mean(all_variances) if all_variances else 0.0
    
    def _compute_ablation_consistency(self) -> float:
        """Compute consistency of ablation results."""
        # This would compare results across multiple ablation runs
        # For now, return a placeholder based on confidence intervals
        
        consistency_scores = []
        for component, importance, (ci_low, ci_high) in self.importance_tracker.get_ranked_components():
            if ci_high != ci_low:
                # Relative confidence interval width (smaller = more consistent)
                relative_width = (ci_high - ci_low) / (abs(importance) + 1e-8)
                consistency = max(0, 1 - relative_width)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _optimize_routing(self, ranked_components: List[Tuple[str, float, Tuple[float, float]]]):
        """Optimize routing based on component importance analysis."""
        if not ranked_components:
            return
        
        self.logger.info("Performing routing optimization based on component importance")
        
        # Adjust learning rates based on component importance
        for component_name, importance, (ci_low, ci_high) in ranked_components:
            if component_name in self.component_hooks:
                component_module = self.component_hooks[component_name]['module']
                
                # Scale gradients based on importance
                if hasattr(component_module, 'parameters'):
                    importance_scale = abs(importance)
                    
                    for param in component_module.parameters():
                        if param.grad is not None and self.config.gradient_scaling:
                            # Scale gradients for more important components
                            param.grad *= (1.0 + importance_scale * self.config.learning_rate_adjustment)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        if not self.analysis_results:
            return {'status': 'no_analysis_performed'}
        
        latest = self.analysis_results[-1]
        ranked_components = self.importance_tracker.get_ranked_components()
        
        summary = {
            'step': self.step_count,
            'num_components': len(self.registered_components),
            'top_components': ranked_components[:5],  # Top 5 most important
            'bottom_components': ranked_components[-5:] if len(ranked_components) > 5 else [],
            'routing_efficiency': latest['metrics'].routing_efficiency,
            'credit_variance': latest['metrics'].credit_assignment_variance,
            'ablation_consistency': latest['metrics'].ablation_consistency,
            'num_analyses_performed': len(self.analysis_results),
            'interaction_effects': latest['interaction_effects']
        }
        
        return summary
    
    def run_systematic_ablation(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        components_to_test: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Run systematic ablation study on specified components."""
        if components_to_test is None:
            components_to_test = list(self.registered_components)
        
        self.logger.info(f"Running systematic ablation on {len(components_to_test)} components")
        
        # Get baseline performance
        baseline_outputs = self.model(inputs)
        baseline_performance = self.performance_evaluator(baseline_outputs, targets)
        
        results = {}
        
        # Test each component individually
        for component_name in components_to_test:
            component_impacts = []
            
            for _ in range(self.config.ablation_sample_size):
                # Drop this component
                if component_name in self.component_hooks:
                    self.component_hooks[component_name]['dropout_fn'](
                        self.component_hooks[component_name]['module'],
                        False  # Deactivate
                    )
                
                try:
                    # Test performance without this component
                    outputs = self.model(inputs)
                    performance = self.performance_evaluator(outputs, targets)
                    impact = performance - baseline_performance
                    component_impacts.append(impact)
                    
                except Exception as e:
                    self.logger.warning(f"Error testing component {component_name}: {e}")
                    
                finally:
                    # Restore component
                    if component_name in self.component_hooks:
                        self.component_hooks[component_name]['dropout_fn'](
                            self.component_hooks[component_name]['module'],
                            True  # Reactivate
                        )
            
            if component_impacts:
                results[component_name] = {
                    'mean_impact': np.mean(component_impacts),
                    'std_impact': np.std(component_impacts),
                    'samples': len(component_impacts)
                }
        
        return results


def create_counterfactual_analyzer(
    model: nn.Module,
    performance_evaluator: Callable[[torch.Tensor, torch.Tensor], float],
    config: Optional[CounterfactualConfig] = None,
    telemetry_collector: Optional[TelemetryCollector] = None
) -> CounterfactualRoutingAnalyzer:
    """Create counterfactual routing analyzer with default configuration."""
    if config is None:
        config = CounterfactualConfig()
    
    return CounterfactualRoutingAnalyzer(
        model=model,
        config=config,
        performance_evaluator=performance_evaluator,
        telemetry_collector=telemetry_collector
    )


def create_default_counterfactual_config(
    component_dropout_rate: float = 0.1,
    analysis_frequency: int = 100,
    enable_routing_optimization: bool = True
) -> CounterfactualConfig:
    """Create default counterfactual configuration."""
    return CounterfactualConfig(
        component_dropout_rate=component_dropout_rate,
        analysis_frequency=analysis_frequency,
        enable_routing_optimization=enable_routing_optimization,
        enable_interaction_analysis=True,
        systematic_ablation=True,
        performance_metric="loss"
    )


# Example usage and testing
if __name__ == "__main__":
    # Mock model for testing
    class MockBEMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix_router = nn.Linear(128, 64)
            self.chunk_router = nn.Linear(128, 64) 
            self.token_router = nn.Linear(128, 64)
            self.uncertainty_head = nn.Linear(128, 1)
            self.final_layer = nn.Linear(64, 10)
            
            # Component states
            self.prefix_active = True
            self.chunk_active = True
            self.token_active = True
            self.uncertainty_active = True
        
        def forward(self, x):
            # Simulate routing through different components
            out = x
            
            if self.prefix_active:
                out = out + 0.1 * self.prefix_router(x)
            if self.chunk_active:
                out = out + 0.1 * self.chunk_router(x)
            if self.token_active:
                out = out + 0.1 * self.token_router(x)
                
            if self.uncertainty_active:
                uncertainty = torch.sigmoid(self.uncertainty_head(x))
                out = out * uncertainty
            
            return self.final_layer(out)
    
    # Component dropout functions
    def prefix_dropout(module, is_active):
        module.prefix_active = is_active
        
    def chunk_dropout(module, is_active):
        module.chunk_active = is_active
        
    def token_dropout(module, is_active):
        module.token_active = is_active
        
    def uncertainty_dropout(module, is_active):
        module.uncertainty_active = is_active
    
    # Performance evaluator
    def performance_evaluator(outputs, targets):
        return F.cross_entropy(outputs, targets).item()
    
    # Create model and analyzer
    model = MockBEMModel()
    config = create_default_counterfactual_config(
        component_dropout_rate=0.2,
        analysis_frequency=10
    )
    
    analyzer = create_counterfactual_analyzer(
        model=model,
        performance_evaluator=performance_evaluator,
        config=config
    )
    
    # Register components
    analyzer.register_component("prefix_router", model, prefix_dropout)
    analyzer.register_component("chunk_router", model, chunk_dropout)
    analyzer.register_component("token_router", model, token_dropout)
    analyzer.register_component("uncertainty_head", model, uncertainty_dropout)
    
    print(f"Registered components: {analyzer.registered_components}")
    
    # Run counterfactual analysis
    batch_size, seq_len, feature_dim = 8, 32, 128
    num_classes = 10
    
    for step in range(50):
        inputs = torch.randn(batch_size, seq_len, feature_dim)
        targets = torch.randint(0, num_classes, (batch_size, seq_len))
        
        result = analyzer.step(inputs, targets)
        
        if result['analysis_performed'] and step % 20 == 0:
            summary = analyzer.get_analysis_summary()
            print(f"\nStep {step} - Analysis Summary:")
            print(f"  Routing efficiency: {summary['routing_efficiency']:.3f}")
            print(f"  Credit variance: {summary['credit_variance']:.3f}")
            print(f"  Top components:")
            for name, importance, (ci_low, ci_high) in summary['top_components']:
                print(f"    {name}: {importance:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    
    # Run systematic ablation
    print(f"\nRunning systematic ablation study...")
    inputs = torch.randn(batch_size, seq_len, feature_dim)
    targets = torch.randint(0, num_classes, (batch_size, seq_len))
    
    ablation_results = analyzer.run_systematic_ablation(inputs, targets)
    
    print("Ablation Results:")
    for component, result in ablation_results.items():
        print(f"  {component}: {result['mean_impact']:.4f} ± {result['std_impact']:.4f}")
    
    print("Counterfactual routing analysis test completed!")