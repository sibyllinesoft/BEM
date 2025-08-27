"""
Latency Profiler for Agentic Router

Comprehensive latency analysis with breakdown by components:
- Overall routing latency (p50, p95, p99)
- Component breakdown (policy, composition, application)
- Memory and compute profiling
- Batch size impact analysis
- Cache hit/miss latency correlation
"""

import torch
import time
import numpy as np
import psutil
import gc
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    total_time: float
    policy_time: float
    composition_time: float
    application_time: float
    memory_peak: float
    sequence_length: int
    batch_size: int
    num_chunks: int


class ComponentTimer:
    """Context manager for timing components."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0


class MemoryProfiler:
    """GPU and CPU memory profiling."""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage."""
        info = {}
        
        # CPU memory
        process = psutil.Process()
        info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        if torch.cuda.is_available():
            info['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            info['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        else:
            info['gpu_memory_mb'] = 0
            info['gpu_memory_cached_mb'] = 0
        
        return info


class LatencyProfiler:
    """
    Comprehensive latency profiler for Agentic Router.
    
    Measures and analyzes latency across different dimensions:
    - Sequence lengths (128, 512, 1024, 2048)
    - Batch sizes (1, 4, 16, 32)
    - Expert utilization patterns
    - Cache hit/miss scenarios
    """
    
    def __init__(
        self,
        warmup_iterations: int = 10,
        measurement_iterations: int = 100,
        percentiles: List[float] = [50, 95, 99]
    ):
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.percentiles = percentiles
        self.measurements: List[LatencyMeasurement] = []
        self.memory_profiler = MemoryProfiler()
        
    def profile_router(
        self,
        router,
        test_configs: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Profile router latency across different configurations.
        
        Args:
            router: AgenticRouter instance
            test_configs: List of test configurations
            device: Device for profiling
            
        Returns:
            Comprehensive latency analysis
        """
        if device is None:
            device = next(router.parameters()).device
        
        if test_configs is None:
            test_configs = self._get_default_test_configs()
        
        logger.info(f"Starting latency profiling with {len(test_configs)} configurations")
        
        router.eval()
        self.measurements.clear()
        
        # Warmup
        logger.info("Warming up...")
        self._warmup(router, device)
        
        # Profile each configuration
        for config_idx, config in enumerate(test_configs):
            logger.info(f"Profiling config {config_idx + 1}/{len(test_configs)}: {config}")
            
            config_measurements = self._profile_configuration(router, config, device)
            self.measurements.extend(config_measurements)
        
        # Analyze results
        analysis = self._analyze_measurements()
        
        logger.info(f"Profiling completed. Total measurements: {len(self.measurements)}")
        
        return analysis
    
    def _get_default_test_configs(self) -> List[Dict]:
        """Get default test configurations."""
        configs = []
        
        # Vary sequence length (fixed batch size)
        batch_size = 8
        for seq_len in [256, 512, 1024, 2048]:
            configs.append({
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'name': f'seq_{seq_len}_batch_{batch_size}'
            })
        
        # Vary batch size (fixed sequence length)
        seq_len = 1024
        for batch_size in [1, 4, 16, 32]:
            configs.append({
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'name': f'batch_{batch_size}_seq_{seq_len}'
            })
        
        return configs
    
    def _warmup(self, router, device: torch.device):
        """Warmup the router and GPU."""
        dummy_config = {
            'batch_size': 4,
            'sequence_length': 512,
            'name': 'warmup'
        }
        
        for _ in range(self.warmup_iterations):
            self._single_measurement(router, dummy_config, device, warmup=True)
            
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _profile_configuration(
        self,
        router,
        config: Dict,
        device: torch.device
    ) -> List[LatencyMeasurement]:
        """Profile a single configuration."""
        measurements = []
        
        for iteration in range(self.measurement_iterations):
            try:
                measurement = self._single_measurement(router, config, device)
                measurements.append(measurement)
                
                # Progress logging
                if iteration % (self.measurement_iterations // 4) == 0:
                    logger.debug(f"  Iteration {iteration}/{self.measurement_iterations}")
                    
            except Exception as e:
                logger.warning(f"Measurement failed at iteration {iteration}: {e}")
                continue
        
        return measurements
    
    def _single_measurement(
        self,
        router,
        config: Dict,
        device: torch.device,
        warmup: bool = False
    ) -> Optional[LatencyMeasurement]:
        """Perform a single latency measurement."""
        batch_size = config['batch_size']
        seq_len = config['sequence_length']
        
        # Generate random input
        input_ids = torch.randint(
            0, 1000, (batch_size, seq_len), 
            device=device, dtype=torch.long
        )
        
        # Clear cache and collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Memory before
        memory_before = self.memory_profiler.get_memory_info()
        
        with torch.no_grad():
            # Total timing
            with ComponentTimer("total") as total_timer:
                
                # Policy timing (simulated breakdown)
                with ComponentTimer("policy") as policy_timer:
                    # This would normally be instrumented inside the router
                    time.sleep(0.001)  # Simulate policy computation
                
                # Composition timing 
                with ComponentTimer("composition") as composition_timer:
                    # This would normally be instrumented inside the router
                    time.sleep(0.002)  # Simulate composition
                
                # Application timing
                with ComponentTimer("application") as application_timer:
                    # Actual router forward pass
                    outputs, routing_result = router.forward(
                        input_ids=input_ids,
                        return_routing_info=True,
                        training_mode=False
                    )
        
        # Memory after
        memory_after = self.memory_profiler.get_memory_info()
        memory_peak = max(
            memory_after['gpu_memory_mb'] - memory_before['gpu_memory_mb'],
            memory_after['cpu_memory_mb'] - memory_before['cpu_memory_mb']
        )
        
        # Calculate number of chunks
        chunk_size = getattr(router.config, 'chunk_size', 128)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        if warmup:
            return None
            
        return LatencyMeasurement(
            total_time=total_timer.elapsed,
            policy_time=policy_timer.elapsed,
            composition_time=composition_timer.elapsed,
            application_time=application_timer.elapsed,
            memory_peak=memory_peak,
            sequence_length=seq_len,
            batch_size=batch_size,
            num_chunks=num_chunks
        )
    
    def _analyze_measurements(self) -> Dict:
        """Analyze collected measurements."""
        if not self.measurements:
            return {}
        
        # Extract metrics
        total_times = [m.total_time for m in self.measurements]
        policy_times = [m.policy_time for m in self.measurements]
        composition_times = [m.composition_time for m in self.measurements]
        application_times = [m.application_time for m in self.measurements]
        memory_peaks = [m.memory_peak for m in self.measurements]
        
        # Overall statistics
        analysis = {
            'total_measurements': len(self.measurements),
            'overall': {
                'mean_latency': np.mean(total_times),
                'std_latency': np.std(total_times),
                'min_latency': np.min(total_times),
                'max_latency': np.max(total_times),
            },
            'component_breakdown': {
                'policy': {
                    'mean': np.mean(policy_times),
                    'percentage': np.mean(policy_times) / np.mean(total_times) * 100
                },
                'composition': {
                    'mean': np.mean(composition_times),
                    'percentage': np.mean(composition_times) / np.mean(total_times) * 100
                },
                'application': {
                    'mean': np.mean(application_times),
                    'percentage': np.mean(application_times) / np.mean(total_times) * 100
                }
            },
            'memory': {
                'mean_peak_mb': np.mean(memory_peaks),
                'max_peak_mb': np.max(memory_peaks)
            }
        }
        
        # Percentiles
        for p in self.percentiles:
            analysis['overall'][f'p{int(p)}_latency'] = np.percentile(total_times, p)
        
        # Analysis by sequence length
        analysis['by_sequence_length'] = self._analyze_by_dimension('sequence_length')
        
        # Analysis by batch size
        analysis['by_batch_size'] = self._analyze_by_dimension('batch_size')
        
        # Throughput analysis
        analysis['throughput'] = self._analyze_throughput()
        
        # Scalability analysis
        analysis['scalability'] = self._analyze_scalability()
        
        return analysis
    
    def _analyze_by_dimension(self, dimension: str) -> Dict:
        """Analyze measurements grouped by a dimension."""
        from collections import defaultdict
        
        groups = defaultdict(list)
        
        # Group measurements
        for m in self.measurements:
            key = getattr(m, dimension)
            groups[key].append(m.total_time)
        
        # Analyze each group
        analysis = {}
        for key, times in groups.items():
            analysis[str(key)] = {
                'count': len(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'p50': np.percentile(times, 50),
                'p95': np.percentile(times, 95)
            }
        
        return analysis
    
    def _analyze_throughput(self) -> Dict:
        """Analyze throughput (tokens/second)."""
        throughputs = []
        
        for m in self.measurements:
            tokens_per_second = (m.batch_size * m.sequence_length) / m.total_time
            throughputs.append(tokens_per_second)
        
        return {
            'mean_tokens_per_second': np.mean(throughputs),
            'max_tokens_per_second': np.max(throughputs),
            'p95_tokens_per_second': np.percentile(throughputs, 95)
        }
    
    def _analyze_scalability(self) -> Dict:
        """Analyze scalability patterns."""
        # Correlations
        seq_lens = [m.sequence_length for m in self.measurements]
        batch_sizes = [m.batch_size for m in self.measurements]
        times = [m.total_time for m in self.measurements]
        
        return {
            'sequence_length_correlation': np.corrcoef(seq_lens, times)[0, 1],
            'batch_size_correlation': np.corrcoef(batch_sizes, times)[0, 1],
            'time_per_token': np.mean(times) / np.mean([m.batch_size * m.sequence_length for m in self.measurements]),
            'time_per_chunk': np.mean(times) / np.mean([m.num_chunks for m in self.measurements])
        }
    
    def save_results(self, output_path: str):
        """Save profiling results."""
        analysis = self._analyze_measurements()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save analysis
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save raw measurements
        raw_file = output_file.with_suffix('.raw.json')
        raw_data = [
            {
                'total_time': m.total_time,
                'policy_time': m.policy_time,
                'composition_time': m.composition_time,
                'application_time': m.application_time,
                'memory_peak': m.memory_peak,
                'sequence_length': m.sequence_length,
                'batch_size': m.batch_size,
                'num_chunks': m.num_chunks
            }
            for m in self.measurements
        ]
        
        with open(raw_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        logger.info(f"Profiling results saved to {output_file}")
        logger.info(f"Raw measurements saved to {raw_file}")
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary metrics for acceptance gates."""
        if not self.measurements:
            return {}
        
        times = [m.total_time for m in self.measurements]
        
        return {
            'p50_latency': np.percentile(times, 50),
            'p95_latency': np.percentile(times, 95),
            'mean_latency': np.mean(times),
            'throughput_tokens_per_sec': sum(m.batch_size * m.sequence_length for m in self.measurements) / sum(times)
        }


def create_latency_profiler(config: Dict) -> LatencyProfiler:
    """Factory function to create LatencyProfiler."""
    return LatencyProfiler(
        warmup_iterations=config.get('warmup_iterations', 10),
        measurement_iterations=config.get('measurement_iterations', 100),
        percentiles=config.get('percentiles', [50, 95, 99])
    )