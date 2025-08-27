"""
Cache Analysis for BEM v1.1 

Advanced cache analysis and optimization suggestions for BEM-v1.1-stable
focused on KV cache efficiency and routing behavior analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math


class CacheAnalyzer:
    """
    Advanced cache analysis for BEM v1.1 models.
    
    Provides detailed analysis of:
    - KV cache hit rates and efficiency
    - Routing stability and flip patterns
    - Memory usage optimization
    - Performance bottleneck identification
    """
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_cache_efficiency(
        self,
        routing_weights: torch.Tensor,
        chunk_size: int = 128,
        sequence_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze KV cache efficiency from routing patterns.
        
        Args:
            routing_weights: [batch_size, seq_len, num_experts]
            chunk_size: Chunk size for routing decisions
            sequence_length: Optional sequence length override
            
        Returns:
            Comprehensive cache efficiency analysis
        """
        batch_size, seq_len, num_experts = routing_weights.shape
        if sequence_length:
            seq_len = min(seq_len, sequence_length)
        
        analysis = {
            'chunk_size': chunk_size,
            'sequence_length': seq_len,
            'batch_size': batch_size,
            'num_experts': num_experts
        }
        
        # Chunk-level analysis
        num_chunks = math.ceil(seq_len / chunk_size)
        chunk_consistency_scores = []
        chunk_dominant_experts = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)
            
            chunk_routing = routing_weights[:, start_idx:end_idx, :]  # [batch, chunk_len, experts]
            
            # Consistency within chunk (how stable is routing within chunk)
            chunk_consistency = self._compute_chunk_consistency(chunk_routing)
            chunk_consistency_scores.append(chunk_consistency)
            
            # Dominant expert per chunk
            chunk_mean_routing = chunk_routing.mean(dim=1)  # [batch, experts]
            dominant_expert = torch.argmax(chunk_mean_routing, dim=1)  # [batch]
            chunk_dominant_experts.append(dominant_expert)
        
        # Cache hit rate estimation
        cache_hit_rate = self._estimate_cache_hit_rate(
            routing_weights, chunk_size, chunk_consistency_scores
        )
        
        # Routing stability across chunks
        stability_metrics = self._analyze_routing_stability(chunk_dominant_experts)
        
        # Expert utilization analysis
        utilization_analysis = self._analyze_expert_utilization(routing_weights)
        
        analysis.update({
            'cache_hit_rate': cache_hit_rate,
            'chunk_consistency': {
                'mean': np.mean(chunk_consistency_scores),
                'std': np.std(chunk_consistency_scores),
                'min': np.min(chunk_consistency_scores),
                'max': np.max(chunk_consistency_scores)
            },
            'stability_metrics': stability_metrics,
            'utilization_analysis': utilization_analysis,
            'num_chunks': num_chunks
        })
        
        return analysis
    
    def _compute_chunk_consistency(self, chunk_routing: torch.Tensor) -> float:
        """
        Compute consistency of routing within a chunk.
        
        Args:
            chunk_routing: [batch, chunk_len, experts]
            
        Returns:
            Consistency score (0 = completely inconsistent, 1 = perfectly consistent)
        """
        batch_size, chunk_len, num_experts = chunk_routing.shape
        
        if chunk_len <= 1:
            return 1.0  # Single token chunks are perfectly consistent
        
        # Compute variance of routing weights within chunk
        chunk_var = torch.var(chunk_routing, dim=1)  # [batch, experts]
        total_variance = torch.sum(chunk_var, dim=1)  # [batch]
        
        # Consistency is inverse of variance (lower variance = higher consistency)
        mean_variance = torch.mean(total_variance).item()
        
        # Normalize to [0, 1] range (assuming max variance is when routing is uniform random)
        max_possible_variance = num_experts * 0.25  # Variance of uniform distribution over [0,1]
        consistency = max(0.0, 1.0 - (mean_variance / max_possible_variance))
        
        return consistency
    
    def _estimate_cache_hit_rate(
        self,
        routing_weights: torch.Tensor,
        chunk_size: int,
        chunk_consistency_scores: List[float]
    ) -> Dict[str, float]:
        """Estimate KV cache hit rate based on routing stability."""
        
        batch_size, seq_len, num_experts = routing_weights.shape
        num_chunks = math.ceil(seq_len / chunk_size)
        
        # Base cache hit rate from chunk consistency
        mean_consistency = np.mean(chunk_consistency_scores)
        
        # Perfect consistency -> high cache hit rate
        # Low consistency -> low cache hit rate
        base_hit_rate = mean_consistency
        
        # Adjust for chunk boundary effects
        boundary_penalty = 0.05  # 5% penalty for chunk boundaries
        adjusted_hit_rate = base_hit_rate * (1.0 - boundary_penalty)
        
        # Compute theoretical maximum hit rate
        # (depends on chunk size and sequence length)
        total_cache_opportunities = seq_len - num_chunks  # Positions that could reuse cache
        if seq_len > 0:
            theoretical_max = total_cache_opportunities / seq_len
        else:
            theoretical_max = 0.0
        
        # Final hit rate estimate
        estimated_hit_rate = min(adjusted_hit_rate, theoretical_max)
        
        return {
            'estimated_hit_rate': estimated_hit_rate,
            'base_consistency': mean_consistency,
            'theoretical_max': theoretical_max,
            'cache_opportunities': total_cache_opportunities
        }
    
    def _analyze_routing_stability(
        self,
        chunk_dominant_experts: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze stability of routing decisions across chunks."""
        
        if len(chunk_dominant_experts) <= 1:
            return {'flip_rate': 0.0, 'stability_score': 1.0}
        
        # Count flips between adjacent chunks
        total_flips = 0
        total_decisions = 0
        
        for i in range(1, len(chunk_dominant_experts)):
            prev_experts = chunk_dominant_experts[i-1]
            curr_experts = chunk_dominant_experts[i]
            
            # Count flips for each sequence in batch
            flips = (prev_experts != curr_experts).sum().item()
            total_flips += flips
            total_decisions += prev_experts.shape[0]  # batch size
        
        flip_rate = total_flips / total_decisions if total_decisions > 0 else 0.0
        stability_score = 1.0 - flip_rate
        
        # Additional stability metrics
        expert_transitions = defaultdict(int)
        for i in range(1, len(chunk_dominant_experts)):
            prev_experts = chunk_dominant_experts[i-1]
            curr_experts = chunk_dominant_experts[i]
            
            for prev, curr in zip(prev_experts.cpu().numpy(), curr_experts.cpu().numpy()):
                expert_transitions[(prev, curr)] += 1
        
        return {
            'flip_rate': flip_rate,
            'stability_score': stability_score,
            'total_flips': total_flips,
            'total_decisions': total_decisions,
            'expert_transitions': dict(expert_transitions)
        }
    
    def _analyze_expert_utilization(
        self,
        routing_weights: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze how evenly experts are utilized."""
        
        # Average routing weights across batch and sequence
        mean_routing = routing_weights.mean(dim=(0, 1))  # [num_experts]
        
        # Expert utilization statistics
        utilization_stats = {
            'mean_utilization': mean_routing.cpu().numpy().tolist(),
            'utilization_std': torch.std(mean_routing).item(),
            'min_utilization': torch.min(mean_routing).item(),
            'max_utilization': torch.max(mean_routing).item()
        }
        
        # Compute entropy of utilization (higher = more balanced)
        probs = mean_routing + 1e-8  # Add small epsilon
        probs = probs / probs.sum()
        entropy = -(probs * torch.log(probs)).sum().item()
        max_entropy = math.log(len(mean_routing))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Load balancing score (1.0 = perfect balance, 0.0 = completely imbalanced)
        load_balance_score = normalized_entropy
        
        utilization_stats.update({
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'load_balance_score': load_balance_score
        })
        
        return utilization_stats
    
    def suggest_optimizations(
        self,
        cache_analysis: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, str]]:
        """
        Suggest optimizations based on cache analysis.
        
        Args:
            cache_analysis: Results from analyze_cache_efficiency
            performance_metrics: Optional performance metrics
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Cache hit rate suggestions
        hit_rate = cache_analysis.get('cache_hit_rate', {}).get('estimated_hit_rate', 0.0)
        
        if hit_rate < 0.5:
            suggestions.append({
                'type': 'cache_efficiency',
                'priority': 'high',
                'suggestion': 'Low cache hit rate detected. Consider increasing hysteresis threshold or chunk size.',
                'details': f'Current hit rate: {hit_rate:.2%}. Target: >80%'
            })
        
        # Routing stability suggestions
        stability = cache_analysis.get('stability_metrics', {})
        flip_rate = stability.get('flip_rate', 0.0)
        
        if flip_rate > 0.2:
            suggestions.append({
                'type': 'routing_stability',
                'priority': 'medium',
                'suggestion': 'High routing flip rate. Increase hysteresis threshold or improve routing consistency.',
                'details': f'Current flip rate: {flip_rate:.2%}. Target: <10%'
            })
        
        # Expert utilization suggestions
        utilization = cache_analysis.get('utilization_analysis', {})
        load_balance = utilization.get('load_balance_score', 1.0)
        
        if load_balance < 0.6:
            suggestions.append({
                'type': 'load_balancing',
                'priority': 'medium',
                'suggestion': 'Uneven expert utilization. Consider load balancing penalties or expert capacity adjustments.',
                'details': f'Load balance score: {load_balance:.2f}. Target: >0.8'
            })
        
        # Chunk size suggestions
        chunk_consistency = cache_analysis.get('chunk_consistency', {})
        mean_consistency = chunk_consistency.get('mean', 1.0)
        
        if mean_consistency < 0.7:
            suggestions.append({
                'type': 'chunk_sizing',
                'priority': 'low',
                'suggestion': 'Low within-chunk consistency. Consider smaller chunk sizes or improved context modeling.',
                'details': f'Chunk consistency: {mean_consistency:.2f}. Target: >0.8'
            })
        
        # Performance-based suggestions
        if performance_metrics:
            latency = performance_metrics.get('latency_per_token_ms', 0.0)
            throughput = performance_metrics.get('tokens_per_second', 0.0)
            
            if latency > 50:  # >50ms per token is slow
                suggestions.append({
                    'type': 'performance',
                    'priority': 'high',
                    'suggestion': 'High latency detected. Optimize cache usage, reduce routing computation, or use smaller models.',
                    'details': f'Current latency: {latency:.1f}ms/token. Target: <20ms/token'
                })
            
            if throughput < 100:  # <100 tokens/sec is slow
                suggestions.append({
                    'type': 'performance', 
                    'priority': 'medium',
                    'suggestion': 'Low throughput. Consider batch size optimization, mixed precision, or hardware acceleration.',
                    'details': f'Current throughput: {throughput:.1f} tokens/sec. Target: >500 tokens/sec'
                })
        
        return suggestions
    
    def generate_cache_report(
        self,
        cache_analysis: Dict[str, Any],
        suggestions: List[Dict[str, str]]
    ) -> str:
        """Generate human-readable cache analysis report."""
        
        report_lines = [
            "# BEM v1.1 Cache Analysis Report",
            "",
            f"**Analysis Date**: {cache_analysis.get('timestamp', 'N/A')}",
            f"**Sequence Length**: {cache_analysis.get('sequence_length', 'N/A')}",
            f"**Chunk Size**: {cache_analysis.get('chunk_size', 'N/A')}",
            f"**Number of Chunks**: {cache_analysis.get('num_chunks', 'N/A')}",
            "",
            "## Cache Efficiency Metrics",
            ""
        ]
        
        # Cache hit rate
        hit_rate_info = cache_analysis.get('cache_hit_rate', {})
        estimated_hit_rate = hit_rate_info.get('estimated_hit_rate', 0.0)
        theoretical_max = hit_rate_info.get('theoretical_max', 0.0)
        
        report_lines.extend([
            f"- **Estimated Cache Hit Rate**: {estimated_hit_rate:.2%}",
            f"- **Theoretical Maximum**: {theoretical_max:.2%}",
            f"- **Cache Efficiency**: {(estimated_hit_rate/theoretical_max*100) if theoretical_max > 0 else 0:.1f}%",
            ""
        ])
        
        # Routing stability
        stability_info = cache_analysis.get('stability_metrics', {})
        flip_rate = stability_info.get('flip_rate', 0.0)
        stability_score = stability_info.get('stability_score', 0.0)
        
        report_lines.extend([
            "## Routing Stability",
            "",
            f"- **Flip Rate**: {flip_rate:.2%}",
            f"- **Stability Score**: {stability_score:.3f}",
            f"- **Total Flips**: {stability_info.get('total_flips', 0)}",
            ""
        ])
        
        # Expert utilization
        utilization_info = cache_analysis.get('utilization_analysis', {})
        load_balance = utilization_info.get('load_balance_score', 0.0)
        
        report_lines.extend([
            "## Expert Utilization",
            "",
            f"- **Load Balance Score**: {load_balance:.3f}",
            f"- **Utilization Entropy**: {utilization_info.get('normalized_entropy', 0.0):.3f}",
            f"- **Min Utilization**: {utilization_info.get('min_utilization', 0.0):.3f}",
            f"- **Max Utilization**: {utilization_info.get('max_utilization', 0.0):.3f}",
            ""
        ])
        
        # Optimization suggestions
        if suggestions:
            report_lines.extend([
                "## Optimization Suggestions",
                ""
            ])
            
            # Group by priority
            high_priority = [s for s in suggestions if s['priority'] == 'high']
            medium_priority = [s for s in suggestions if s['priority'] == 'medium']
            low_priority = [s for s in suggestions if s['priority'] == 'low']
            
            for priority, priority_suggestions in [('High', high_priority), 
                                                 ('Medium', medium_priority), 
                                                 ('Low', low_priority)]:
                if priority_suggestions:
                    report_lines.extend([f"### {priority} Priority", ""])
                    for i, suggestion in enumerate(priority_suggestions, 1):
                        report_lines.append(f"{i}. **{suggestion['suggestion']}**")
                        report_lines.append(f"   {suggestion['details']}")
                        report_lines.append("")
        
        return '\n'.join(report_lines)