#!/usr/bin/env python3
"""
Hierarchical Routing Auditor for BEM Research Validation

Comprehensive routing analysis for expert loads, entropy, utilization skew,
and KV-cache validation. Ensures proper expert utilization and system stability.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RoutingMetrics:
    """Metrics for routing analysis."""
    expert_loads: List[float]
    load_balance_coefficient: float
    entropy: float
    utilization_skew: float
    gini_coefficient: float
    effective_experts: int
    routing_concentration: float

@dataclass
class KVCacheMetrics:
    """Metrics for KV-cache validation."""
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    invalidation_events: int
    zero_invalidation_validated: bool
    cache_efficiency: float

@dataclass
class RoutingAuditResult:
    """Complete routing audit result."""
    routing_metrics: RoutingMetrics
    kv_cache_metrics: KVCacheMetrics
    expert_utilization_stats: Dict[str, float]
    routing_stability_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]

class ExpertLoadBalancer:
    """Monitor and analyze expert load balancing."""
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.expert_counters = np.zeros(num_experts)
        self.routing_history = []
        
    def record_routing_decision(self, expert_weights: torch.Tensor) -> None:
        """Record a routing decision for analysis."""
        
        # Convert to numpy and normalize
        if isinstance(expert_weights, torch.Tensor):
            weights = expert_weights.detach().cpu().numpy()
        else:
            weights = np.array(expert_weights)
            
        # Record per-expert load
        self.expert_counters += weights
        
        # Store routing decision
        self.routing_history.append(weights.copy())
        
    def compute_load_balance_metrics(self) -> Dict[str, float]:
        """Compute load balancing metrics."""
        
        if len(self.routing_history) == 0:
            return self._get_empty_metrics()
            
        routing_matrix = np.array(self.routing_history)  # [decisions, experts]
        
        # Expert loads (total weight per expert)
        expert_loads = np.sum(routing_matrix, axis=0)
        total_load = np.sum(expert_loads)
        
        if total_load == 0:
            return self._get_empty_metrics()
            
        normalized_loads = expert_loads / total_load
        
        # Load Balance Coefficient (variance of normalized loads)
        load_balance_coeff = np.var(normalized_loads)
        
        # Entropy (higher = more balanced)
        entropy = stats.entropy(normalized_loads + 1e-10)  # Add small epsilon
        
        # Gini coefficient (inequality measure)
        gini_coeff = self._compute_gini_coefficient(expert_loads)
        
        # Utilization skew (ratio of max to min load)
        min_load = np.min(expert_loads)
        max_load = np.max(expert_loads)
        utilization_skew = max_load / (min_load + 1e-10)
        
        # Effective number of experts (based on entropy)
        effective_experts = np.exp(entropy)
        
        # Routing concentration (how focused routing decisions are)
        mean_concentration = np.mean([np.max(weights) for weights in self.routing_history])
        
        return {
            'expert_loads': expert_loads.tolist(),
            'normalized_loads': normalized_loads.tolist(),
            'load_balance_coefficient': float(load_balance_coeff),
            'entropy': float(entropy),
            'utilization_skew': float(utilization_skew),
            'gini_coefficient': float(gini_coeff),
            'effective_experts': float(effective_experts),
            'routing_concentration': float(mean_concentration),
            'total_decisions': len(self.routing_history)
        }
        
    def _compute_gini_coefficient(self, loads: np.ndarray) -> float:
        """Compute Gini coefficient for load distribution."""
        
        if len(loads) == 0 or np.sum(loads) == 0:
            return 0.0
            
        # Sort loads
        sorted_loads = np.sort(loads)
        n = len(sorted_loads)
        
        # Compute Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_loads)) / (n * np.sum(sorted_loads)) - (n + 1) / n
        
        return max(0.0, gini)  # Ensure non-negative
        
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics when no data is available."""
        return {
            'expert_loads': [0.0] * self.num_experts,
            'normalized_loads': [0.0] * self.num_experts,
            'load_balance_coefficient': 0.0,
            'entropy': 0.0,
            'utilization_skew': 1.0,
            'gini_coefficient': 0.0,
            'effective_experts': 0.0,
            'routing_concentration': 0.0,
            'total_decisions': 0
        }

class KVCacheValidator:
    """Validate KV-cache behavior and efficiency."""
    
    def __init__(self):
        self.cache_operations = []
        self.invalidation_events = []
        self.cache_stats = defaultdict(int)
        
    def record_cache_access(self, 
                           cache_key: str,
                           hit: bool,
                           expert_id: Optional[int] = None) -> None:
        """Record a cache access event."""
        
        self.cache_operations.append({
            'key': cache_key,
            'hit': hit,
            'expert_id': expert_id,
            'timestamp': len(self.cache_operations)
        })
        
        if hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
            
    def record_cache_invalidation(self, 
                                cache_key: str,
                                reason: str = "expert_switch") -> None:
        """Record a cache invalidation event."""
        
        self.invalidation_events.append({
            'key': cache_key,
            'reason': reason,
            'timestamp': len(self.cache_operations)
        })
        
        self.cache_stats['invalidations'] += 1
        
    def validate_kv_cache_safety(self) -> Dict[str, Any]:
        """Validate KV-cache safety properties."""
        
        total_operations = len(self.cache_operations)
        total_hits = self.cache_stats['hits']
        total_misses = self.cache_stats['misses']
        total_invalidations = len(self.invalidation_events)
        
        if total_operations == 0:
            return self._get_empty_cache_metrics()
            
        # Hit rate
        hit_rate = total_hits / total_operations
        
        # Zero invalidation validation (ideal for BEM)
        zero_invalidation_validated = total_invalidations == 0
        
        # Cache efficiency (hit rate adjusted for invalidations)
        invalidation_penalty = total_invalidations / max(total_operations, 1)
        cache_efficiency = hit_rate * (1 - invalidation_penalty)
        
        # Analyze invalidation patterns
        invalidation_analysis = self._analyze_invalidation_patterns()
        
        return {
            'cache_hits': total_hits,
            'cache_misses': total_misses,
            'total_operations': total_operations,
            'hit_rate': hit_rate,
            'invalidation_events': total_invalidations,
            'zero_invalidation_validated': zero_invalidation_validated,
            'cache_efficiency': cache_efficiency,
            'invalidation_analysis': invalidation_analysis,
            'safety_score': self._compute_safety_score(zero_invalidation_validated, hit_rate)
        }
        
    def _analyze_invalidation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in cache invalidations."""
        
        if not self.invalidation_events:
            return {'pattern': 'none', 'frequency': 0.0, 'reasons': {}}
            
        # Invalidation frequency
        total_ops = len(self.cache_operations)
        frequency = len(self.invalidation_events) / max(total_ops, 1)
        
        # Reason analysis
        reasons = defaultdict(int)
        for event in self.invalidation_events:
            reasons[event['reason']] += 1
            
        # Temporal clustering
        timestamps = [event['timestamp'] for event in self.invalidation_events]
        temporal_variance = np.var(timestamps) if len(timestamps) > 1 else 0.0
        
        return {
            'frequency': frequency,
            'reasons': dict(reasons),
            'temporal_variance': temporal_variance,
            'pattern': 'clustered' if temporal_variance > 100 else 'uniform'
        }
        
    def _compute_safety_score(self, zero_invalidation: bool, hit_rate: float) -> float:
        """Compute overall cache safety score."""
        
        base_score = hit_rate
        
        # Bonus for zero invalidation (BEM's key advantage)
        if zero_invalidation:
            base_score += 0.2
            
        # Penalty for high invalidation rate
        invalidation_rate = len(self.invalidation_events) / max(len(self.cache_operations), 1)
        penalty = min(0.3, invalidation_rate * 0.5)
        
        return max(0.0, min(1.0, base_score - penalty))
        
    def _get_empty_cache_metrics(self) -> Dict[str, Any]:
        """Return empty cache metrics."""
        return {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_operations': 0,
            'hit_rate': 0.0,
            'invalidation_events': 0,
            'zero_invalidation_validated': True,
            'cache_efficiency': 0.0,
            'invalidation_analysis': {'pattern': 'none', 'frequency': 0.0, 'reasons': {}},
            'safety_score': 0.0
        }

class RoutingEntropyAnalyzer:
    """Analyze routing entropy and expert specialization patterns."""
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.input_routing_pairs = []
        
    def record_input_routing(self, 
                           input_representation: torch.Tensor,
                           routing_weights: torch.Tensor) -> None:
        """Record input and corresponding routing decision."""
        
        # Store input features and routing
        if isinstance(input_representation, torch.Tensor):
            input_repr = input_representation.detach().cpu().numpy()
        else:
            input_repr = np.array(input_representation)
            
        if isinstance(routing_weights, torch.Tensor):
            routing = routing_weights.detach().cpu().numpy()
        else:
            routing = np.array(routing_weights)
            
        self.input_routing_pairs.append((input_repr.flatten(), routing))
        
    def analyze_routing_entropy(self) -> Dict[str, float]:
        """Analyze routing entropy and patterns."""
        
        if not self.input_routing_pairs:
            return self._get_empty_entropy_metrics()
            
        # Extract routing decisions
        routing_decisions = np.array([pair[1] for pair in self.input_routing_pairs])
        
        # Global entropy (across all decisions)
        mean_routing = np.mean(routing_decisions, axis=0)
        global_entropy = stats.entropy(mean_routing + 1e-10)
        
        # Per-decision entropy
        decision_entropies = [stats.entropy(routing + 1e-10) for routing in routing_decisions]
        mean_decision_entropy = np.mean(decision_entropies)
        entropy_variance = np.var(decision_entropies)
        
        # Expert specialization (how consistently each expert is used)
        expert_consistency = []
        for expert_idx in range(self.num_experts):
            expert_weights = routing_decisions[:, expert_idx]
            consistency = 1 - np.var(expert_weights)  # Higher variance = lower consistency
            expert_consistency.append(max(0.0, consistency))
            
        # Routing diversity (Jensen-Shannon divergence between decisions)
        routing_diversity = self._compute_routing_diversity(routing_decisions)
        
        return {
            'global_entropy': float(global_entropy),
            'mean_decision_entropy': float(mean_decision_entropy),
            'entropy_variance': float(entropy_variance),
            'expert_consistency': expert_consistency,
            'mean_consistency': float(np.mean(expert_consistency)),
            'routing_diversity': float(routing_diversity),
            'specialization_score': float(1 - mean_decision_entropy / np.log(self.num_experts))
        }
        
    def _compute_routing_diversity(self, routing_decisions: np.ndarray) -> float:
        """Compute average pairwise Jensen-Shannon divergence."""
        
        n_decisions = len(routing_decisions)
        if n_decisions < 2:
            return 0.0
            
        js_divergences = []
        
        # Sample pairs to avoid O(n^2) computation for large datasets
        max_pairs = min(1000, n_decisions * (n_decisions - 1) // 2)
        
        for i in range(min(n_decisions, 32)):  # Sample subset for efficiency
            for j in range(i + 1, min(n_decisions, 32)):
                p = routing_decisions[i] + 1e-10
                q = routing_decisions[j] + 1e-10
                
                # Normalize to probability distributions
                p = p / np.sum(p)
                q = q / np.sum(q)
                
                js_div = jensenshannon(p, q)
                js_divergences.append(js_div)
                
        return np.mean(js_divergences) if js_divergences else 0.0
        
    def _get_empty_entropy_metrics(self) -> Dict[str, float]:
        """Return empty entropy metrics."""
        return {
            'global_entropy': 0.0,
            'mean_decision_entropy': 0.0,
            'entropy_variance': 0.0,
            'expert_consistency': [0.0] * self.num_experts,
            'mean_consistency': 0.0,
            'routing_diversity': 0.0,
            'specialization_score': 0.0
        }

class RoutingAuditor:
    """Main routing auditor orchestrator."""
    
    def __init__(self, num_experts: int = 8):
        self.num_experts = num_experts
        self.load_balancer = ExpertLoadBalancer(num_experts)
        self.cache_validator = KVCacheValidator()
        self.entropy_analyzer = RoutingEntropyAnalyzer(num_experts)
        
    def record_routing_event(self, 
                           input_representation: Optional[torch.Tensor],
                           routing_weights: torch.Tensor,
                           cache_key: Optional[str] = None,
                           cache_hit: bool = True,
                           expert_id: Optional[int] = None) -> None:
        """Record a complete routing event for analysis."""
        
        # Record load balancing
        self.load_balancer.record_routing_decision(routing_weights)
        
        # Record cache behavior
        if cache_key is not None:
            self.cache_validator.record_cache_access(cache_key, cache_hit, expert_id)
            
        # Record routing entropy
        if input_representation is not None:
            self.entropy_analyzer.record_input_routing(input_representation, routing_weights)
            
    def record_cache_invalidation(self, cache_key: str, reason: str = "expert_switch") -> None:
        """Record cache invalidation event."""
        self.cache_validator.record_cache_invalidation(cache_key, reason)
        
    def generate_audit_report(self) -> RoutingAuditResult:
        """Generate comprehensive routing audit report."""
        
        logger.info("Generating comprehensive routing audit report")
        
        # Compute all metrics
        load_metrics = self.load_balancer.compute_load_balance_metrics()
        cache_metrics = self.cache_validator.validate_kv_cache_safety()
        entropy_metrics = self.entropy_analyzer.analyze_routing_entropy()
        
        # Create structured metrics objects
        routing_metrics = RoutingMetrics(
            expert_loads=load_metrics['expert_loads'],
            load_balance_coefficient=load_metrics['load_balance_coefficient'],
            entropy=load_metrics['entropy'],
            utilization_skew=load_metrics['utilization_skew'],
            gini_coefficient=load_metrics['gini_coefficient'],
            effective_experts=int(load_metrics['effective_experts']),
            routing_concentration=load_metrics['routing_concentration']
        )
        
        kv_cache_metrics = KVCacheMetrics(
            cache_hits=cache_metrics['cache_hits'],
            cache_misses=cache_metrics['cache_misses'],
            cache_hit_rate=cache_metrics['hit_rate'],
            invalidation_events=cache_metrics['invalidation_events'],
            zero_invalidation_validated=cache_metrics['zero_invalidation_validated'],
            cache_efficiency=cache_metrics['cache_efficiency']
        )
        
        # Expert utilization statistics
        expert_stats = self._compute_expert_utilization_stats(load_metrics, entropy_metrics)
        
        # Overall routing stability score
        stability_score = self._compute_routing_stability_score(
            routing_metrics, kv_cache_metrics, entropy_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            routing_metrics, kv_cache_metrics, entropy_metrics, stability_score
        )
        
        return RoutingAuditResult(
            routing_metrics=routing_metrics,
            kv_cache_metrics=kv_cache_metrics,
            expert_utilization_stats=expert_stats,
            routing_stability_score=stability_score,
            recommendations=recommendations,
            metadata={
                'num_experts': self.num_experts,
                'total_routing_decisions': load_metrics['total_decisions'],
                'analysis_timestamp': str(np.datetime64('now')),
                'raw_metrics': {
                    'load_balance': load_metrics,
                    'cache_validation': cache_metrics,
                    'entropy_analysis': entropy_metrics
                }
            }
        )
        
    def _compute_expert_utilization_stats(self, 
                                        load_metrics: Dict[str, Any],
                                        entropy_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Compute expert utilization statistics."""
        
        expert_loads = load_metrics['expert_loads']
        expert_consistency = entropy_metrics['expert_consistency']
        
        if not expert_loads:
            return {}
            
        total_load = sum(expert_loads)
        
        stats = {}
        for i, (load, consistency) in enumerate(zip(expert_loads, expert_consistency)):
            utilization_pct = (load / max(total_load, 1)) * 100
            stats[f'expert_{i}'] = {
                'utilization_percent': utilization_pct,
                'consistency_score': consistency,
                'specialization_index': utilization_pct * consistency
            }
            
        return stats
        
    def _compute_routing_stability_score(self, 
                                       routing_metrics: RoutingMetrics,
                                       kv_cache_metrics: KVCacheMetrics,
                                       entropy_metrics: Dict[str, Any]) -> float:
        """Compute overall routing stability score (0-1)."""
        
        # Load balance score (lower coefficient is better)
        load_balance_score = 1 / (1 + routing_metrics.load_balance_coefficient)
        
        # Cache efficiency score
        cache_score = kv_cache_metrics.cache_efficiency
        
        # Expert specialization score  
        specialization_score = entropy_metrics.get('specialization_score', 0.0)
        
        # Entropy stability score (low variance is better)
        entropy_variance = entropy_metrics.get('entropy_variance', 1.0)
        entropy_stability_score = 1 / (1 + entropy_variance)
        
        # Weighted combination
        stability_score = (
            0.3 * load_balance_score +
            0.3 * cache_score + 
            0.2 * specialization_score +
            0.2 * entropy_stability_score
        )
        
        return max(0.0, min(1.0, stability_score))
        
    def _generate_recommendations(self, 
                                routing_metrics: RoutingMetrics,
                                kv_cache_metrics: KVCacheMetrics,
                                entropy_metrics: Dict[str, Any],
                                stability_score: float) -> List[str]:
        """Generate actionable recommendations based on audit results."""
        
        recommendations = []
        
        # Load balancing recommendations
        if routing_metrics.load_balance_coefficient > 0.1:
            recommendations.append(
                f"High load imbalance detected (coefficient: {routing_metrics.load_balance_coefficient:.3f}). "
                "Consider adjusting gating network or expert capacity."
            )
            
        if routing_metrics.utilization_skew > 5.0:
            recommendations.append(
                f"Severe utilization skew detected ({routing_metrics.utilization_skew:.1f}x). "
                "Some experts are underutilized. Consider load balancing regularization."
            )
            
        # Cache recommendations
        if not kv_cache_metrics.zero_invalidation_validated:
            recommendations.append(
                f"Cache invalidations detected ({kv_cache_metrics.invalidation_events} events). "
                "This may indicate expert switching issues affecting BEM's KV-cache safety."
            )
            
        if kv_cache_metrics.cache_hit_rate < 0.8:
            recommendations.append(
                f"Low cache hit rate ({kv_cache_metrics.cache_hit_rate:.2%}). "
                "Consider increasing cache size or improving cache key design."
            )
            
        # Specialization recommendations  
        specialization_score = entropy_metrics.get('specialization_score', 0.0)
        if specialization_score < 0.3:
            recommendations.append(
                f"Low expert specialization ({specialization_score:.2f}). "
                "Experts may not be learning distinct behaviors. Consider specialization regularization."
            )
            
        # Overall stability
        if stability_score < 0.6:
            recommendations.append(
                f"Overall routing stability is low ({stability_score:.2f}). "
                "Consider reviewing routing architecture and training procedures."
            )
        elif stability_score > 0.8:
            recommendations.append(
                f"Excellent routing stability ({stability_score:.2f}). "
                "Current configuration is performing well."
            )
            
        if not recommendations:
            recommendations.append("Routing system is performing within acceptable parameters.")
            
        return recommendations

def main():
    """Example usage of routing auditor."""
    
    # Initialize auditor
    auditor = RoutingAuditor(num_experts=8)
    
    # Simulate routing events
    for i in range(100):
        # Mock input representation
        input_repr = torch.randn(512)
        
        # Mock routing weights (with some expert preferences)
        routing_weights = torch.softmax(torch.randn(8) + torch.tensor([1, 0, -1, 0, 2, -1, 0, 1]), dim=0)
        
        # Mock cache behavior
        cache_key = f"input_{i}"
        cache_hit = np.random.random() > 0.2  # 80% hit rate
        
        # Record event
        auditor.record_routing_event(
            input_representation=input_repr,
            routing_weights=routing_weights,
            cache_key=cache_key,
            cache_hit=cache_hit
        )
        
        # Occasional cache invalidation
        if np.random.random() < 0.05:
            auditor.record_cache_invalidation(cache_key, "expert_switch")
            
    # Generate audit report
    audit_result = auditor.generate_audit_report()
    
    # Print summary
    print("Routing Audit Summary:")
    print(f"Load Balance Coefficient: {audit_result.routing_metrics.load_balance_coefficient:.4f}")
    print(f"Effective Experts: {audit_result.routing_metrics.effective_experts}")
    print(f"Cache Hit Rate: {audit_result.kv_cache_metrics.cache_hit_rate:.2%}")
    print(f"Zero Invalidation Validated: {audit_result.kv_cache_metrics.zero_invalidation_validated}")
    print(f"Routing Stability Score: {audit_result.routing_stability_score:.3f}")
    print(f"Recommendations: {len(audit_result.recommendations)}")
    
    for rec in audit_result.recommendations:
        print(f"  - {rec}")

if __name__ == "__main__":
    main()