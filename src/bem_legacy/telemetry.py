"""
Comprehensive telemetry and monitoring system for hierarchical BEM.
Tracks performance, routing behavior, and system health metrics.

Features:
- Real-time performance monitoring
- Routing pattern analysis
- Memory and compute profiling  
- Gradient flow monitoring
- Cache efficiency tracking
- Exportable metrics for analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import time
import psutil
import gc
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from contextlib import contextmanager
import json
from pathlib import Path

from .controller import RoutingLevel, RoutingState
from .hierarchical_bem import FullHierarchicalBEM


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    forward_time: float = 0.0
    controller_time: float = 0.0
    bem_time: float = 0.0
    memory_used: float = 0.0
    memory_peak: float = 0.0
    gpu_utilization: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    latency_per_token: float = 0.0


@dataclass
class RoutingMetrics:
    """Container for routing behavior metrics."""
    entropy_mean: float = 0.0
    entropy_std: float = 0.0
    code_norm_mean: float = 0.0
    code_norm_std: float = 0.0
    uncertainty_mean: float = 0.0
    uncertainty_std: float = 0.0
    utilization_active_fraction: float = 0.0
    routing_distribution: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [prefix, chunk, token]
    stability_score: float = 0.0


@dataclass
class SystemMetrics:
    """Container for system-level metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gradient_norm_controller: float = 0.0
    gradient_norm_bem: float = 0.0
    parameter_norm_controller: float = 0.0
    parameter_norm_bem: float = 0.0


class TelemetryCollector:
    """
    Main telemetry collection system for hierarchical BEM.
    
    Collects, aggregates, and exports comprehensive metrics for:
    - Performance profiling
    - Routing behavior analysis
    - System health monitoring
    - Training diagnostics
    """
    
    def __init__(
        self,
        model: FullHierarchicalBEM,
        collection_interval: int = 100,  # Collect metrics every N steps
        history_length: int = 1000,      # Keep N recent measurements
        export_path: Optional[str] = None
    ):
        self.model = model
        self.collection_interval = collection_interval
        self.history_length = history_length
        self.export_path = Path(export_path) if export_path else Path("telemetry_logs")
        self.export_path.mkdir(exist_ok=True)
        
        # Metric histories (using deque for efficient sliding window)
        self.performance_history = deque(maxlen=history_length)
        self.routing_history = deque(maxlen=history_length)
        self.system_history = deque(maxlen=history_length)
        
        # Current step counter
        self.step = 0
        
        # Timing contexts
        self._timing_stack = []
        self._current_timings = {}
        
        # GPU monitoring (set up before memory tracking)
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.gpu_device = next(model.parameters()).device
        
        # Memory tracking
        self._memory_baseline = self._get_memory_usage()
        
        # Gradient tracking hooks
        self._gradient_hooks = []
        self._setup_gradient_hooks()
        
        # Cache for expensive computations
        self._routing_stability_cache = deque(maxlen=50)  # Last 50 code samples for stability
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory = {}
        
        # CPU memory
        process = psutil.Process()
        memory['cpu_mb'] = process.memory_info().rss / 1024 / 1024
        memory['cpu_percent'] = process.memory_percent()
        
        # GPU memory
        if self.cuda_available:
            memory['gpu_mb'] = torch.cuda.memory_allocated(self.gpu_device) / 1024 / 1024
            memory['gpu_peak_mb'] = torch.cuda.max_memory_allocated(self.gpu_device) / 1024 / 1024
            memory['gpu_reserved_mb'] = torch.cuda.memory_reserved(self.gpu_device) / 1024 / 1024
        else:
            memory['gpu_mb'] = 0.0
            memory['gpu_peak_mb'] = 0.0
            memory['gpu_reserved_mb'] = 0.0
        
        return memory
    
    def _setup_gradient_hooks(self):
        """Setup hooks to monitor gradient flow."""
        def create_hook(name: str):
            def hook(grad):
                if grad is not None:
                    self._current_timings[f'grad_norm_{name}'] = grad.norm().item()
                return grad
            return hook
        
        # Hook controller gradients
        for name, param in self.model.controller.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(create_hook(f'controller_{name}'))
                self._gradient_hooks.append(hook)
        
        # Hook BEM module gradients
        for module_name, bem_module in self.model.bem_modules.items():
            for param_name, param in bem_module.named_parameters():
                if param.requires_grad and 'lora' in param_name:
                    hook = param.register_hook(create_hook(f'{module_name}_{param_name}'))
                    self._gradient_hooks.append(hook)
    
    @contextmanager
    def timing_context(self, name: str):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        self._timing_stack.append(name)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self._current_timings[name] = end_time - start_time
            self._timing_stack.pop()
    
    def collect_performance_metrics(
        self,
        batch_size: int = 1,
        sequence_length: int = 512
    ) -> PerformanceMetrics:
        """Collect performance metrics."""
        memory = self._get_memory_usage()
        
        # Calculate throughput
        total_time = sum(self._current_timings.values())
        total_tokens = batch_size * sequence_length
        
        metrics = PerformanceMetrics(
            forward_time=self._current_timings.get('forward_pass', 0.0),
            controller_time=self._current_timings.get('controller', 0.0),
            bem_time=self._current_timings.get('bem_computation', 0.0),
            memory_used=memory['gpu_mb'],
            memory_peak=memory['gpu_peak_mb'],
            gpu_utilization=self._get_gpu_utilization(),
            throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0.0,
            latency_per_token=total_time / total_tokens if total_tokens > 0 else 0.0
        )
        
        return metrics
    
    def collect_routing_metrics(
        self,
        routing_info: Dict[str, Any]
    ) -> RoutingMetrics:
        """Collect routing behavior metrics."""
        metrics = RoutingMetrics()
        
        # Extract routing state information
        if 'routing_state' in routing_info:
            routing_state = routing_info['routing_state']
            
            # Entropy metrics
            if hasattr(routing_state, 'entropy') and routing_state.entropy is not None:
                entropy = routing_state.entropy
                if isinstance(entropy, torch.Tensor):
                    metrics.entropy_mean = entropy.mean().item()
                    metrics.entropy_std = entropy.std().item() if entropy.numel() > 1 else 0.0
            
            # Uncertainty metrics
            if hasattr(routing_state, 'uncertainty') and routing_state.uncertainty is not None:
                uncertainty = routing_state.uncertainty
                if isinstance(uncertainty, torch.Tensor):
                    metrics.uncertainty_mean = uncertainty.mean().item()
                    metrics.uncertainty_std = uncertainty.std().item() if uncertainty.numel() > 1 else 0.0
            
            # Utilization metrics
            if hasattr(routing_state, 'utilization') and routing_state.utilization is not None:
                util = routing_state.utilization
                metrics.utilization_active_fraction = util.get('active_fraction', 0.0)
        
        # Code norm metrics (collect from all layers)
        code_norms = []
        for layer_name, info in routing_info.get('layers', {}).items():
            if 'code_norm' in info:
                code_norms.append(info['code_norm'])
        
        if code_norms:
            code_norms = np.array(code_norms)
            metrics.code_norm_mean = code_norms.mean()
            metrics.code_norm_std = code_norms.std()
        
        # Routing distribution (from model statistics)
        routing_stats = self.model.get_routing_statistics()
        if 'global_stats' in routing_stats and 'routing_distribution' in routing_stats['global_stats']:
            metrics.routing_distribution = routing_stats['global_stats']['routing_distribution']
        
        # Stability score (computed from recent routing history)
        stability_score = self._compute_routing_stability()
        metrics.stability_score = stability_score
        
        return metrics
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        # System resources
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        
        # GPU memory
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        if self.cuda_available:
            gpu_memory_used = torch.cuda.memory_allocated(self.gpu_device) / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(self.gpu_device).total_memory / 1024**3
        
        # Gradient norms
        controller_grad_norm = self._compute_parameter_norm(
            self.model.controller.parameters(), grad=True
        )
        bem_grad_norm = self._compute_parameter_norm(
            self.model.get_bem_parameters(), grad=True
        )
        
        # Parameter norms
        controller_param_norm = self._compute_parameter_norm(
            self.model.controller.parameters(), grad=False
        )
        bem_param_norm = self._compute_parameter_norm(
            self.model.get_bem_parameters(), grad=False
        )
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_info.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gradient_norm_controller=controller_grad_norm,
            gradient_norm_bem=bem_grad_norm,
            parameter_norm_controller=controller_param_norm,
            parameter_norm_bem=bem_param_norm
        )
        
        return metrics
    
    def _compute_parameter_norm(
        self,
        parameters,
        grad: bool = False
    ) -> float:
        """Compute L2 norm of parameters or gradients."""
        total_norm = 0.0
        param_count = 0
        
        for param in parameters:
            if param is not None:
                if grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                elif not grad:
                    param_norm = param.data.norm(2)
                else:
                    continue
                
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if not self.cuda_available:
            return 0.0
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_device.index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0  # Return 0 if pynvml not available
    
    def _compute_routing_stability(self) -> float:
        """Compute routing stability score from recent history."""
        if len(self._routing_stability_cache) < 2:
            return 1.0  # Perfect stability if no history
        
        # Convert to tensor for computation
        codes = list(self._routing_stability_cache)
        if not codes:
            return 1.0
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(codes) - 1):
            code1 = codes[i].flatten() if isinstance(codes[i], torch.Tensor) else torch.tensor(codes[i]).flatten()
            code2 = codes[i + 1].flatten() if isinstance(codes[i + 1], torch.Tensor) else torch.tensor(codes[i + 1]).flatten()
            
            # Cosine similarity
            sim = F.cosine_similarity(code1.unsqueeze(0), code2.unsqueeze(0))
            similarities.append(sim.item())
        
        # Return average similarity as stability score
        return np.mean(similarities) if similarities else 1.0
    
    def step_update(
        self,
        routing_info: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        sequence_length: int = 512
    ):
        """Update telemetry for current step."""
        self.step += 1
        
        # Collect metrics at specified intervals
        if self.step % self.collection_interval == 0:
            # Performance metrics
            perf_metrics = self.collect_performance_metrics(batch_size, sequence_length)
            self.performance_history.append((self.step, perf_metrics))
            
            # Routing metrics
            if routing_info:
                routing_metrics = self.collect_routing_metrics(routing_info)
                self.routing_history.append((self.step, routing_metrics))
                
                # Update stability cache
                if 'codes' in routing_info:
                    self._routing_stability_cache.append(routing_info['codes'])
            
            # System metrics
            system_metrics = self.collect_system_metrics()
            self.system_history.append((self.step, system_metrics))
        
        # Clear current timings
        self._current_timings.clear()
        
        # Reset peak memory
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get latest metrics."""
        metrics = {}
        
        if self.performance_history:
            _, latest_perf = self.performance_history[-1]
            metrics['performance'] = latest_perf.__dict__
        
        if self.routing_history:
            _, latest_routing = self.routing_history[-1]
            metrics['routing'] = latest_routing.__dict__
        
        if self.system_history:
            _, latest_system = self.system_history[-1]
            metrics['system'] = latest_system.__dict__
        
        return metrics
    
    def get_performance_summary(self, last_n_steps: int = 100) -> Dict[str, Any]:
        """Get performance summary over recent steps."""
        if not self.performance_history:
            return {}
        
        # Get recent history
        recent_history = list(self.performance_history)[-last_n_steps:]
        
        # Extract metrics
        forward_times = [perf.forward_time for _, perf in recent_history]
        throughputs = [perf.throughput_tokens_per_sec for _, perf in recent_history]
        latencies = [perf.latency_per_token for _, perf in recent_history]
        memory_usage = [perf.memory_used for _, perf in recent_history]
        
        summary = {
            'forward_time': {
                'mean': np.mean(forward_times),
                'std': np.std(forward_times),
                'p50': np.percentile(forward_times, 50),
                'p95': np.percentile(forward_times, 95),
                'p99': np.percentile(forward_times, 99)
            },
            'throughput': {
                'mean': np.mean(throughputs),
                'std': np.std(throughputs),
                'min': np.min(throughputs),
                'max': np.max(throughputs)
            },
            'latency_per_token': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95)
            },
            'memory_usage': {
                'mean': np.mean(memory_usage),
                'max': np.max(memory_usage),
                'trend': 'increasing' if memory_usage[-1] > memory_usage[0] else 'stable'
            }
        }
        
        return summary
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Export metrics to JSON file."""
        if filename is None:
            filename = f"telemetry_step_{self.step}.json"
        
        filepath = self.export_path / filename
        
        # Prepare export data
        export_data = {
            'metadata': {
                'step': self.step,
                'collection_interval': self.collection_interval,
                'history_length': len(self.performance_history),
                'export_timestamp': time.time()
            },
            'performance_history': [
                {
                    'step': step,
                    'metrics': metrics.__dict__
                }
                for step, metrics in self.performance_history
            ],
            'routing_history': [
                {
                    'step': step,
                    'metrics': metrics.__dict__
                }
                for step, metrics in self.routing_history
            ],
            'system_history': [
                {
                    'step': step,
                    'metrics': metrics.__dict__
                }
                for step, metrics in self.system_history
            ],
            'summary': {
                'performance': self.get_performance_summary(),
                'current_metrics': self.get_current_metrics()
            }
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Telemetry exported to: {filepath}")
        return str(filepath)
    
    def generate_report(self, last_n_steps: int = 500) -> str:
        """Generate a human-readable telemetry report."""
        summary = self.get_performance_summary(last_n_steps)
        current_metrics = self.get_current_metrics()
        
        report = f"""
Hierarchical BEM Telemetry Report
==================================

Step: {self.step}
Collection Interval: {self.collection_interval}
History Length: {len(self.performance_history)} entries

Performance Summary (Last {last_n_steps} steps):
------------------------------------------------
Forward Time:
  Mean: {summary.get('forward_time', {}).get('mean', 0):.4f}s
  P95: {summary.get('forward_time', {}).get('p95', 0):.4f}s
  P99: {summary.get('forward_time', {}).get('p99', 0):.4f}s

Throughput:
  Mean: {summary.get('throughput', {}).get('mean', 0):.2f} tokens/sec
  Max: {summary.get('throughput', {}).get('max', 0):.2f} tokens/sec

Memory Usage:
  Mean: {summary.get('memory_usage', {}).get('mean', 0):.2f} MB
  Peak: {summary.get('memory_usage', {}).get('max', 0):.2f} MB
  Trend: {summary.get('memory_usage', {}).get('trend', 'unknown')}

Current State:
--------------
"""
        
        if 'routing' in current_metrics:
            routing = current_metrics['routing']
            report += f"""
Routing Behavior:
  Entropy: {routing.get('entropy_mean', 0):.4f} ± {routing.get('entropy_std', 0):.4f}
  Code Norm: {routing.get('code_norm_mean', 0):.4f} ± {routing.get('code_norm_std', 0):.4f}
  Uncertainty: {routing.get('uncertainty_mean', 0):.4f} ± {routing.get('uncertainty_std', 0):.4f}
  Stability: {routing.get('stability_score', 0):.4f}
  Routing Distribution: Prefix={routing.get('routing_distribution', [0,0,0])[0]:.2%}, Chunk={routing.get('routing_distribution', [0,0,0])[1]:.2%}, Token={routing.get('routing_distribution', [0,0,0])[2]:.2%}
"""
        
        if 'system' in current_metrics:
            system = current_metrics['system']
            report += f"""
System Resources:
  CPU: {system.get('cpu_percent', 0):.1f}%
  Memory: {system.get('memory_percent', 0):.1f}%
  GPU Memory: {system.get('gpu_memory_used', 0):.2f}/{system.get('gpu_memory_total', 0):.2f} GB
  Controller Grad Norm: {system.get('gradient_norm_controller', 0):.4f}
  BEM Grad Norm: {system.get('gradient_norm_bem', 0):.4f}
"""
        
        return report
    
    def cleanup(self):
        """Cleanup resources."""
        # Remove gradient hooks
        for hook in self._gradient_hooks:
            hook.remove()
        self._gradient_hooks.clear()


# Factory functions

def create_telemetry_collector(
    model: FullHierarchicalBEM,
    collection_interval: int = 100,
    history_length: int = 1000,
    export_path: Optional[str] = None
) -> TelemetryCollector:
    """Create telemetry collector for hierarchical BEM."""
    return TelemetryCollector(
        model=model,
        collection_interval=collection_interval,
        history_length=history_length,
        export_path=export_path
    )


# Context manager for easy profiling

@contextmanager
def profile_bem_operation(
    collector: TelemetryCollector,
    operation_name: str,
    collect_routing_info: bool = True
):
    """Context manager for profiling BEM operations."""
    with collector.timing_context(operation_name):
        start_step = collector.step
        
        try:
            yield collector
        finally:
            # Update telemetry
            if collect_routing_info:
                # This would need to be passed in from the actual operation
                routing_info = {}  # Placeholder
                collector.step_update(routing_info=routing_info)
            else:
                collector.step_update()


# Analysis utilities

def analyze_routing_patterns(
    telemetry_data: List[Tuple[int, RoutingMetrics]],
    window_size: int = 50
) -> Dict[str, Any]:
    """Analyze routing patterns from telemetry data."""
    if len(telemetry_data) < window_size:
        return {}
    
    # Extract metrics
    steps, metrics = zip(*telemetry_data)
    entropies = [m.entropy_mean for m in metrics]
    stabilities = [m.stability_score for m in metrics]
    code_norms = [m.code_norm_mean for m in metrics]
    
    # Compute trends
    entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0]
    stability_trend = np.polyfit(range(len(stabilities)), stabilities, 1)[0]
    
    analysis = {
        'entropy_analysis': {
            'mean': np.mean(entropies),
            'trend': 'increasing' if entropy_trend > 0 else 'decreasing',
            'stability': np.std(entropies) < 0.1
        },
        'routing_stability': {
            'mean': np.mean(stabilities),
            'trend': 'improving' if stability_trend > 0 else 'degrading',
            'consistent': np.std(stabilities) < 0.05
        },
        'code_magnitude': {
            'mean': np.mean(code_norms),
            'trend': np.polyfit(range(len(code_norms)), code_norms, 1)[0],
            'stability': np.std(code_norms)
        }
    }
    
    return analysis


def detect_performance_regressions(
    performance_history: List[Tuple[int, PerformanceMetrics]],
    baseline_steps: int = 100,
    threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """Detect performance regressions in telemetry data."""
    if len(performance_history) < baseline_steps * 2:
        return []
    
    # Split into baseline and recent periods
    baseline_data = performance_history[:baseline_steps]
    recent_data = performance_history[-baseline_steps:]
    
    # Compute means for comparison
    baseline_throughput = np.mean([p.throughput_tokens_per_sec for _, p in baseline_data])
    recent_throughput = np.mean([p.throughput_tokens_per_sec for _, p in recent_data])
    
    baseline_latency = np.mean([p.latency_per_token for _, p in baseline_data])
    recent_latency = np.mean([p.latency_per_token for _, p in recent_data])
    
    regressions = []
    
    # Check throughput regression
    throughput_change = (recent_throughput - baseline_throughput) / baseline_throughput
    if throughput_change < -threshold:
        regressions.append({
            'metric': 'throughput',
            'change_percent': throughput_change * 100,
            'baseline': baseline_throughput,
            'current': recent_throughput,
            'severity': 'high' if throughput_change < -0.2 else 'medium'
        })
    
    # Check latency regression
    latency_change = (recent_latency - baseline_latency) / baseline_latency
    if latency_change > threshold:
        regressions.append({
            'metric': 'latency',
            'change_percent': latency_change * 100,
            'baseline': baseline_latency,
            'current': recent_latency,
            'severity': 'high' if latency_change > 0.2 else 'medium'
        })
    
    return regressions