"""
Cache Metrics Collection for BEM v1.1

Comprehensive cache metrics collection and analysis for BEM v1.1-stable
according to TODO.md specifications:

- KV hit% (cache efficiency)
- Tokens/s (throughput)  
- Latency (p50/p95)
- VRAM usage
- Flip rates and routing stability
- Gate entropy and utilization
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import math


class CacheMetricsCollector:
    """
    Comprehensive cache metrics collection for BEM v1.1.
    
    Tracks all metrics specified in TODO.md for cache efficiency,
    performance, and routing behavior analysis.
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        percentiles: List[float] = [50, 95, 99]
    ):
        self.history_size = history_size
        self.percentiles = percentiles
        
        # Metric histories
        self.latencies = deque(maxlen=history_size)
        self.throughputs = deque(maxlen=history_size)
        self.flip_rates = deque(maxlen=history_size)
        self.gate_entropies = deque(maxlen=history_size)
        self.kv_hit_rates = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        
        # Per-layer tracking
        self.layer_metrics = defaultdict(lambda: {
            'flip_rates': deque(maxlen=history_size),
            'entropies': deque(maxlen=history_size),
            'utilizations': deque(maxlen=history_size)
        })
        
        # Cache state tracking
        self.cache_states = {}
        self.routing_history = defaultdict(list)
        
    def start_timing(self) -> float:
        """Start timing for latency measurement."""
        return time.time()
    
    def end_timing(self, start_time: float, num_tokens: int) -> Tuple[float, float]:
        """
        End timing and compute latency/throughput.
        
        Args:
            start_time: Start time from start_timing()
            num_tokens: Number of tokens processed
            
        Returns:
            (latency_ms, tokens_per_second)
        """
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        tokens_per_second = num_tokens / (end_time - start_time) if end_time > start_time else 0.0
        
        # Store metrics
        self.latencies.append(latency_ms)
        self.throughputs.append(tokens_per_second)
        
        return latency_ms, tokens_per_second
    
    def collect_from_bem_output(self, bem_aux_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Collect cache metrics from BEM auxiliary output.
        
        Args:
            bem_aux_info: BEM auxiliary information from forward pass
            
        Returns:
            Dictionary of collected metrics
        """
        metrics = {}
        
        for layer_name, layer_info in bem_aux_info.items():
            if 'routing_info' in layer_info:
                routing_info = layer_info['routing_info']
                
                # Gate entropy
                if 'gate_entropy' in routing_info:
                    entropy = routing_info['gate_entropy']
                    if isinstance(entropy, torch.Tensor):
                        entropy = entropy.item()
                    self.gate_entropies.append(entropy)
                    self.layer_metrics[layer_name]['entropies'].append(entropy)
                
                # Flip rate
                if 'flip_rate' in routing_info:
                    flip_rate = routing_info['flip_rate']
                    if isinstance(flip_rate, torch.Tensor):
                        flip_rate = flip_rate.item()
                    self.flip_rates.append(flip_rate)
                    self.layer_metrics[layer_name]['flip_rates'].append(flip_rate)
                
                # Expert utilization
                if 'expert_utilization' in routing_info:
                    utilization = routing_info['expert_utilization']
                    if isinstance(utilization, torch.Tensor):
                        utilization = utilization.cpu().numpy()
                    
                    # Compute utilization statistics
                    util_std = np.std(utilization) if len(utilization) > 1 else 0.0
                    util_entropy = self._compute_utilization_entropy(utilization)
                    
                    self.layer_metrics[layer_name]['utilizations'].append(util_std)
                    metrics[f'{layer_name}_util_std'] = util_std
                    metrics[f'{layer_name}_util_entropy'] = util_entropy
        
        # Aggregate metrics across layers
        if self.gate_entropies:
            metrics['gate_entropy'] = np.mean(list(self.gate_entropies)[-10:])  # Recent average
        
        if self.flip_rates:
            metrics['flip_rate'] = np.mean(list(self.flip_rates)[-10:])
        
        # Expert utilization summary
        recent_utils = []
        for layer_metrics in self.layer_metrics.values():
            if layer_metrics['utilizations']:
                recent_utils.extend(list(layer_metrics['utilizations'])[-5:])
        
        if recent_utils:
            metrics['expert_utilization_std'] = np.mean(recent_utils)
        
        return metrics
    
    def _compute_utilization_entropy(self, utilization: np.ndarray) -> float:
        """Compute entropy of expert utilization distribution."""
        if len(utilization) == 0:
            return 0.0
        
        # Add small epsilon to avoid log(0)
        probs = utilization + 1e-8
        probs = probs / np.sum(probs)
        
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def estimate_kv_cache_hit_rate(
        self,
        routing_weights: torch.Tensor,
        chunk_size: int = 128
    ) -> float:
        """
        Estimate KV cache hit rate based on routing stability.
        
        Cache hits occur when routing doesn't change within chunks,
        allowing reuse of cached K/V representations.
        
        Args:
            routing_weights: [batch_size, seq_len, num_experts]
            chunk_size: Chunk size for cache alignment
            
        Returns:
            Estimated cache hit rate [0, 1]
        """
        batch_size, seq_len, num_experts = routing_weights.shape
        num_chunks = math.ceil(seq_len / chunk_size)
        
        total_cache_opportunities = 0
        cache_hits = 0
        
        for b in range(batch_size):
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, seq_len)
                
                if end_idx - start_idx < 2:  # Need at least 2 tokens for cache reuse
                    continue
                
                chunk_routing = routing_weights[b, start_idx:end_idx]  # [chunk_len, num_experts]
                
                # Check routing consistency within chunk
                first_routing = chunk_routing[0]
                consistent_positions = 0
                
                for pos in range(1, chunk_routing.shape[0]):
                    if torch.allclose(chunk_routing[pos], first_routing, atol=1e-6):
                        consistent_positions += 1
                
                total_cache_opportunities += chunk_routing.shape[0] - 1
                cache_hits += consistent_positions
        
        hit_rate = cache_hits / max(1, total_cache_opportunities)
        self.kv_hit_rates.append(hit_rate)
        
        return hit_rate
    
    def collect_memory_metrics(self) -> Dict[str, float]:
        """Collect VRAM usage metrics."""
        metrics = {}
        
        if torch.cuda.is_available():
            # Current memory usage
            current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            self.memory_usage.append(current_memory_mb)
            
            metrics.update({
                'vram_current_mb': current_memory_mb,
                'vram_peak_mb': peak_memory_mb,
                'vram_utilization': current_memory_mb / (24 * 1024)  # Assume 24GB budget
            })
        
        return metrics
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cache and performance report.
        
        Returns all metrics specified in TODO.md with statistics.
        """
        report = {
            'timestamp': time.time(),
            'total_samples': len(self.latencies)
        }
        
        # Latency statistics (p50/p95/p99)
        if self.latencies:
            latencies_array = np.array(list(self.latencies))
            for p in self.percentiles:
                report[f'latency_p{p}_ms'] = np.percentile(latencies_array, p)
            report['latency_mean_ms'] = np.mean(latencies_array)
            report['latency_std_ms'] = np.std(latencies_array)
        
        # Throughput statistics  
        if self.throughputs:
            throughputs_array = np.array(list(self.throughputs))
            report['tokens_per_second_mean'] = np.mean(throughputs_array)
            report['tokens_per_second_p50'] = np.percentile(throughputs_array, 50)
            report['tokens_per_second_p95'] = np.percentile(throughputs_array, 95)
        
        # Cache efficiency
        if self.kv_hit_rates:
            hit_rates_array = np.array(list(self.kv_hit_rates))
            report['kv_cache_hit_rate_mean'] = np.mean(hit_rates_array)
            report['kv_cache_hit_rate_std'] = np.std(hit_rates_array)
        
        # Routing stability  
        if self.flip_rates:
            flip_rates_array = np.array(list(self.flip_rates))
            report['routing_flip_rate_mean'] = np.mean(flip_rates_array)
            report['routing_stability'] = 1.0 - np.mean(flip_rates_array)  # Higher is more stable
        
        # Gate behavior
        if self.gate_entropies:
            entropies_array = np.array(list(self.gate_entropies))
            report['gate_entropy_mean'] = np.mean(entropies_array)
            report['gate_entropy_std'] = np.std(entropies_array)
        
        # Memory usage
        if self.memory_usage:
            memory_array = np.array(list(self.memory_usage))
            report['vram_usage_mean_mb'] = np.mean(memory_array)
            report['vram_usage_peak_mb'] = np.max(memory_array)
        
        # Per-layer breakdown
        layer_reports = {}
        for layer_name, layer_data in self.layer_metrics.items():
            layer_report = {}
            
            if layer_data['flip_rates']:
                layer_report['flip_rate'] = np.mean(list(layer_data['flip_rates'])[-10:])
                
            if layer_data['entropies']:
                layer_report['entropy'] = np.mean(list(layer_data['entropies'])[-10:])
                
            if layer_data['utilizations']:
                layer_report['utilization_std'] = np.mean(list(layer_data['utilizations'])[-10:])
            
            if layer_report:
                layer_reports[layer_name] = layer_report
        
        if layer_reports:
            report['per_layer_metrics'] = layer_reports
        
        # Cache safety assessment
        report['cache_safety_score'] = self._compute_cache_safety_score()
        
        return report
    
    def _compute_cache_safety_score(self) -> float:
        """
        Compute overall cache safety score based on routing stability.
        
        Score of 1.0 = perfect cache safety, 0.0 = no cache efficiency.
        """
        if not self.flip_rates or not self.kv_hit_rates:
            return 0.5  # Unknown
        
        # Recent stability metrics
        recent_flip_rate = np.mean(list(self.flip_rates)[-20:]) if self.flip_rates else 0.5
        recent_hit_rate = np.mean(list(self.kv_hit_rates)[-20:]) if self.kv_hit_rates else 0.5
        
        # Combine flip rate and hit rate into safety score
        stability_score = 1.0 - recent_flip_rate  # Lower flip rate = higher score
        cache_score = recent_hit_rate  # Higher hit rate = higher score
        
        # Weighted combination
        safety_score = 0.6 * stability_score + 0.4 * cache_score
        
        return max(0.0, min(1.0, safety_score))
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.latencies.clear()
        self.throughputs.clear()
        self.flip_rates.clear()
        self.gate_entropies.clear()
        self.kv_hit_rates.clear()
        self.memory_usage.clear()
        self.layer_metrics.clear()
        self.cache_states.clear()
        self.routing_history.clear()
    
    def export_metrics(self, filepath: str):
        """Export comprehensive metrics to JSON file."""
        import json
        
        report = self.get_comprehensive_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Cache metrics exported to {filepath}")


def benchmark_bem_model(
    model: torch.nn.Module,
    dataloader,
    num_batches: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark BEM model performance and cache metrics.
    
    Args:
        model: BEM v1.1 model to benchmark
        dataloader: DataLoader with test data
        num_batches: Number of batches to benchmark
        device: Device for benchmarking
        
    Returns:
        Comprehensive benchmark results
    """
    collector = CacheMetricsCollector()
    model.eval()
    
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            batch_size, seq_len = batch['input_ids'].shape
            total_tokens += batch_size * seq_len
            
            # Time forward pass
            start_time = collector.start_timing()
            
            outputs = model(**batch, return_aux_info=True)
            
            latency_ms, tokens_per_sec = collector.end_timing(start_time, batch_size * seq_len)
            
            # Collect BEM metrics
            if hasattr(outputs, 'bem_aux_info'):
                metrics = collector.collect_from_bem_output(outputs.bem_aux_info)
                
                # Estimate cache hit rate if routing weights available
                for layer_info in outputs.bem_aux_info.values():
                    if 'routing_info' in layer_info and 'routing_weights' in layer_info['routing_info']:
                        routing_weights = layer_info['routing_info']['routing_weights']
                        collector.estimate_kv_cache_hit_rate(routing_weights)
                        break
            
            # Collect memory metrics
            collector.collect_memory_metrics()
    
    # Generate final report
    benchmark_results = collector.get_comprehensive_report()
    benchmark_results.update({
        'total_batches': num_batches,
        'total_tokens': total_tokens,
        'avg_batch_size': total_tokens / num_batches if num_batches > 0 else 0
    })
    
    return benchmark_results