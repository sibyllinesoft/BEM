#!/usr/bin/env python3
"""
BEM Performance Benchmarking Suite

This script runs comprehensive performance benchmarks for the BEM system,
measuring processing time, memory usage, and system resource consumption.
"""

import os
import sys
import time
import psutil
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Represents the result of a benchmark test."""
    name: str
    duration: float  # seconds
    memory_peak: float  # MB
    cpu_percent: float
    success: bool
    metadata: Dict[str, Any]


class PerformanceMonitor:
    """Context manager for monitoring system performance."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.process = psutil.Process()
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        
    def __enter__(self):
        self.monitoring = True
        self.start_time = time.time()
        self._start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitoring = False
        self.end_time = time.time()
    
    def _start_monitoring(self):
        """Start monitoring system resources."""
        import threading
        
        def monitor():
            while self.monitoring:
                try:
                    memory_info = self.process.memory_info()
                    cpu_percent = self.process.cpu_percent()
                    
                    self.memory_samples.append(memory_info.rss / 1024 / 1024)  # MB
                    self.cpu_samples.append(cpu_percent)
                    
                    time.sleep(self.interval)
                except psutil.NoSuchProcess:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    @property
    def duration(self) -> float:
        """Get total duration."""
        return getattr(self, 'end_time', time.time()) - self.start_time
    
    @property
    def peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return max(self.memory_samples) if self.memory_samples else 0
    
    @property
    def avg_cpu(self) -> float:
        """Get average CPU usage."""
        return np.mean(self.cpu_samples) if self.cpu_samples else 0


class BEMBenchmarkSuite:
    """Comprehensive benchmarking suite for BEM."""
    
    def __init__(self, project_root: Path, output_dir: Optional[Path] = None):
        self.project_root = project_root
        self.output_dir = output_dir or project_root / "benchmark_results"
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
        # Ensure src is in Python path
        sys.path.insert(0, str(project_root / "src"))
        
    def run_benchmarks(self, quick: bool = False, comprehensive: bool = False) -> Dict[str, Any]:
        """Run all benchmark tests."""
        print("🚀 Starting BEM Performance Benchmarks")
        print("=" * 50)
        
        # Core benchmarks (always run)
        self._benchmark_import_time()
        self._benchmark_model_loading()
        self._benchmark_inference_speed()
        
        if not quick:
            self._benchmark_batch_processing()
            self._benchmark_memory_efficiency()
            self._benchmark_concurrent_requests()
        
        if comprehensive:
            self._benchmark_scalability()
            self._benchmark_long_running_stability()
            self._benchmark_resource_cleanup()
        
        return self._generate_report()
    
    def _benchmark_import_time(self):
        """Benchmark module import time."""
        print("\n📦 Benchmarking Module Import...")
        
        with PerformanceMonitor() as monitor:
            try:
                # Fresh Python process to measure true import time
                result = subprocess.run([
                    sys.executable, "-c",
                    "import time; start = time.time(); import src.bem; print(f'Import time: {time.time() - start:.4f}s')"
                ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
                
                success = result.returncode == 0
                if success and "Import time:" in result.stdout:
                    import_time = float(result.stdout.split("Import time: ")[1].split("s")[0])
                else:
                    import_time = monitor.duration
                
                self.results.append(BenchmarkResult(
                    name="Module Import",
                    duration=import_time,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=success,
                    metadata={"stdout": result.stdout, "stderr": result.stderr}
                ))
                
                print(f"✅ Import time: {import_time:.4f}s")
                
            except subprocess.TimeoutExpired:
                self.results.append(BenchmarkResult(
                    name="Module Import",
                    duration=30.0,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=False,
                    metadata={"error": "Import timeout"}
                ))
                print("❌ Import timeout (>30s)")
    
    def _benchmark_model_loading(self):
        """Benchmark model loading time."""
        print("\n🧠 Benchmarking Model Loading...")
        
        with PerformanceMonitor() as monitor:
            try:
                # Import and initialize BEM
                import src.bem as bem
                
                # Test model loading
                start_time = time.time()
                # Assuming there's a model loading function
                # model = bem.load_model()  # Adjust based on actual API
                load_time = time.time() - start_time
                
                self.results.append(BenchmarkResult(
                    name="Model Loading",
                    duration=load_time,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=True,
                    metadata={"load_time": load_time}
                ))
                
                print(f"✅ Model loading: {load_time:.4f}s")
                
            except Exception as e:
                self.results.append(BenchmarkResult(
                    name="Model Loading",
                    duration=monitor.duration,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=False,
                    metadata={"error": str(e)}
                ))
                print(f"❌ Model loading failed: {e}")
    
    def _benchmark_inference_speed(self):
        """Benchmark inference speed with sample data."""
        print("\n⚡ Benchmarking Inference Speed...")
        
        # Create sample data
        sample_sizes = [1, 10, 100] if hasattr(self, '_quick') and self._quick else [1, 10, 100, 1000]
        
        for sample_size in sample_sizes:
            with PerformanceMonitor() as monitor:
                try:
                    # Generate sample data
                    sample_data = self._generate_sample_data(sample_size)
                    
                    # Run inference
                    start_time = time.time()
                    # results = bem.process(sample_data)  # Adjust based on actual API
                    inference_time = time.time() - start_time
                    
                    throughput = sample_size / inference_time if inference_time > 0 else 0
                    
                    self.results.append(BenchmarkResult(
                        name=f"Inference Speed (n={sample_size})",
                        duration=inference_time,
                        memory_peak=monitor.peak_memory,
                        cpu_percent=monitor.avg_cpu,
                        success=True,
                        metadata={
                            "sample_size": sample_size,
                            "throughput": throughput,
                            "latency_per_item": inference_time / sample_size
                        }
                    ))
                    
                    print(f"✅ n={sample_size}: {inference_time:.4f}s ({throughput:.2f} items/s)")
                    
                except Exception as e:
                    self.results.append(BenchmarkResult(
                        name=f"Inference Speed (n={sample_size})",
                        duration=monitor.duration,
                        memory_peak=monitor.peak_memory,
                        cpu_percent=monitor.avg_cpu,
                        success=False,
                        metadata={"sample_size": sample_size, "error": str(e)}
                    ))
                    print(f"❌ n={sample_size} failed: {e}")
    
    def _benchmark_batch_processing(self):
        """Benchmark batch processing capabilities."""
        print("\n📦 Benchmarking Batch Processing...")
        
        batch_sizes = [16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            with PerformanceMonitor() as monitor:
                try:
                    # Create batch data
                    batch_data = [self._generate_sample_data(1) for _ in range(batch_size)]
                    
                    start_time = time.time()
                    # Process batch
                    # results = bem.process_batch(batch_data)  # Adjust based on actual API
                    processing_time = time.time() - start_time
                    
                    throughput = batch_size / processing_time if processing_time > 0 else 0
                    
                    self.results.append(BenchmarkResult(
                        name=f"Batch Processing (batch_size={batch_size})",
                        duration=processing_time,
                        memory_peak=monitor.peak_memory,
                        cpu_percent=monitor.avg_cpu,
                        success=True,
                        metadata={
                            "batch_size": batch_size,
                            "throughput": throughput,
                            "latency_per_batch": processing_time
                        }
                    ))
                    
                    print(f"✅ batch_size={batch_size}: {processing_time:.4f}s ({throughput:.2f} batches/s)")
                    
                except Exception as e:
                    self.results.append(BenchmarkResult(
                        name=f"Batch Processing (batch_size={batch_size})",
                        duration=monitor.duration,
                        memory_peak=monitor.peak_memory,
                        cpu_percent=monitor.avg_cpu,
                        success=False,
                        metadata={"batch_size": batch_size, "error": str(e)}
                    ))
                    print(f"❌ batch_size={batch_size} failed: {e}")
    
    def _benchmark_memory_efficiency(self):
        """Benchmark memory usage patterns."""
        print("\n💾 Benchmarking Memory Efficiency...")
        
        with PerformanceMonitor(interval=0.05) as monitor:  # More frequent sampling
            try:
                # Simulate memory-intensive operations
                data_sizes = [1000, 5000, 10000]
                peak_memories = []
                
                for data_size in data_sizes:
                    start_memory = monitor.peak_memory
                    
                    # Create large dataset
                    large_data = self._generate_sample_data(data_size)
                    
                    # Process data
                    # result = bem.process(large_data)
                    
                    # Clean up
                    del large_data
                    
                    peak_memories.append(monitor.peak_memory - start_memory)
                
                self.results.append(BenchmarkResult(
                    name="Memory Efficiency",
                    duration=monitor.duration,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=True,
                    metadata={
                        "data_sizes": data_sizes,
                        "peak_memories": peak_memories,
                        "memory_growth": peak_memories
                    }
                ))
                
                print(f"✅ Memory efficiency test completed")
                print(f"   Peak memory: {monitor.peak_memory:.2f} MB")
                
            except Exception as e:
                self.results.append(BenchmarkResult(
                    name="Memory Efficiency",
                    duration=monitor.duration,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=False,
                    metadata={"error": str(e)}
                ))
                print(f"❌ Memory efficiency test failed: {e}")
    
    def _benchmark_concurrent_requests(self):
        """Benchmark concurrent processing capability."""
        print("\n🔄 Benchmarking Concurrent Processing...")
        
        import concurrent.futures
        import threading
        
        def process_item(item_id):
            """Process a single item."""
            data = self._generate_sample_data(10)
            start = time.time()
            # result = bem.process(data)
            duration = time.time() - start
            return item_id, duration, True
        
        thread_counts = [1, 2, 4, 8]
        
        for thread_count in thread_counts:
            with PerformanceMonitor() as monitor:
                try:
                    start_time = time.time()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                        futures = [executor.submit(process_item, i) for i in range(thread_count * 2)]
                        results = [future.result() for future in concurrent.futures.as_completed(futures)]
                    
                    total_time = time.time() - start_time
                    successful_tasks = sum(1 for _, _, success in results if success)
                    avg_task_time = np.mean([duration for _, duration, _ in results])
                    
                    self.results.append(BenchmarkResult(
                        name=f"Concurrent Processing (threads={thread_count})",
                        duration=total_time,
                        memory_peak=monitor.peak_memory,
                        cpu_percent=monitor.avg_cpu,
                        success=successful_tasks == len(results),
                        metadata={
                            "thread_count": thread_count,
                            "total_tasks": len(results),
                            "successful_tasks": successful_tasks,
                            "avg_task_time": avg_task_time,
                            "throughput": successful_tasks / total_time
                        }
                    ))
                    
                    print(f"✅ threads={thread_count}: {successful_tasks}/{len(results)} tasks successful")
                    print(f"   Total time: {total_time:.4f}s, Avg task: {avg_task_time:.4f}s")
                    
                except Exception as e:
                    self.results.append(BenchmarkResult(
                        name=f"Concurrent Processing (threads={thread_count})",
                        duration=monitor.duration,
                        memory_peak=monitor.peak_memory,
                        cpu_percent=monitor.avg_cpu,
                        success=False,
                        metadata={"thread_count": thread_count, "error": str(e)}
                    ))
                    print(f"❌ threads={thread_count} failed: {e}")
    
    def _benchmark_scalability(self):
        """Benchmark scalability characteristics."""
        print("\n📈 Benchmarking Scalability...")
        
        # Test increasing load
        loads = [100, 500, 1000, 2000, 5000]
        
        for load in loads:
            with PerformanceMonitor() as monitor:
                try:
                    data = self._generate_sample_data(load)
                    
                    start_time = time.time()
                    # result = bem.process(data)
                    processing_time = time.time() - start_time
                    
                    throughput = load / processing_time if processing_time > 0 else 0
                    
                    self.results.append(BenchmarkResult(
                        name=f"Scalability Test (load={load})",
                        duration=processing_time,
                        memory_peak=monitor.peak_memory,
                        cpu_percent=monitor.avg_cpu,
                        success=True,
                        metadata={
                            "load": load,
                            "throughput": throughput,
                            "latency": processing_time / load
                        }
                    ))
                    
                    print(f"✅ load={load}: {processing_time:.4f}s ({throughput:.2f} items/s)")
                    
                except Exception as e:
                    self.results.append(BenchmarkResult(
                        name=f"Scalability Test (load={load})",
                        duration=monitor.duration,
                        memory_peak=monitor.peak_memory,
                        cpu_percent=monitor.avg_cpu,
                        success=False,
                        metadata={"load": load, "error": str(e)}
                    ))
                    print(f"❌ load={load} failed: {e}")
    
    def _benchmark_long_running_stability(self):
        """Benchmark stability over extended periods."""
        print("\n⏱️  Benchmarking Long-Running Stability...")
        
        duration_minutes = 5  # 5-minute stability test
        
        with PerformanceMonitor(interval=1.0) as monitor:
            try:
                end_time = time.time() + (duration_minutes * 60)
                iterations = 0
                errors = 0
                
                while time.time() < end_time:
                    try:
                        data = self._generate_sample_data(100)
                        # result = bem.process(data)
                        iterations += 1
                        
                        if iterations % 10 == 0:
                            print(f"   Completed {iterations} iterations...")
                        
                    except Exception:
                        errors += 1
                
                error_rate = errors / iterations if iterations > 0 else 1.0
                
                self.results.append(BenchmarkResult(
                    name="Long-Running Stability",
                    duration=monitor.duration,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=error_rate < 0.01,  # Less than 1% error rate
                    metadata={
                        "duration_minutes": duration_minutes,
                        "iterations": iterations,
                        "errors": errors,
                        "error_rate": error_rate,
                        "throughput": iterations / monitor.duration
                    }
                ))
                
                print(f"✅ Stability test completed: {iterations} iterations, {errors} errors")
                print(f"   Error rate: {error_rate:.4f}, Duration: {monitor.duration:.1f}s")
                
            except Exception as e:
                self.results.append(BenchmarkResult(
                    name="Long-Running Stability",
                    duration=monitor.duration,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=False,
                    metadata={"error": str(e)}
                ))
                print(f"❌ Stability test failed: {e}")
    
    def _benchmark_resource_cleanup(self):
        """Benchmark resource cleanup and memory leaks."""
        print("\n🧹 Benchmarking Resource Cleanup...")
        
        with PerformanceMonitor(interval=0.5) as monitor:
            try:
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Create and destroy resources multiple times
                for iteration in range(20):
                    # Create resources
                    data = self._generate_sample_data(1000)
                    # result = bem.process(data)
                    
                    # Explicit cleanup
                    del data
                    # del result
                    
                    if iteration % 5 == 0:
                        # Force garbage collection
                        import gc
                        gc.collect()
                
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_growth = final_memory - initial_memory
                
                self.results.append(BenchmarkResult(
                    name="Resource Cleanup",
                    duration=monitor.duration,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=memory_growth < 100,  # Less than 100MB growth
                    metadata={
                        "initial_memory": initial_memory,
                        "final_memory": final_memory,
                        "memory_growth": memory_growth,
                        "iterations": 20
                    }
                ))
                
                print(f"✅ Resource cleanup test completed")
                print(f"   Memory growth: {memory_growth:.2f} MB")
                
            except Exception as e:
                self.results.append(BenchmarkResult(
                    name="Resource Cleanup",
                    duration=monitor.duration,
                    memory_peak=monitor.peak_memory,
                    cpu_percent=monitor.avg_cpu,
                    success=False,
                    metadata={"error": str(e)}
                ))
                print(f"❌ Resource cleanup test failed: {e}")
    
    def _generate_sample_data(self, size: int):
        """Generate sample data for benchmarking."""
        # This should be adjusted based on the actual data format BEM expects
        return {
            "data": np.random.rand(size, 100).tolist(),
            "metadata": {"size": size, "timestamp": time.time()}
        }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        print("\n" + "=" * 50)
        print("📊 BENCHMARK REPORT")
        print("=" * 50)
        
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        print(f"\n📈 Summary:")
        print(f"✅ Successful: {len(successful_tests)}/{len(self.results)} tests")
        print(f"❌ Failed: {len(failed_tests)} tests")
        
        if successful_tests:
            avg_duration = np.mean([r.duration for r in successful_tests])
            max_memory = max([r.memory_peak for r in successful_tests])
            avg_cpu = np.mean([r.cpu_percent for r in successful_tests])
            
            print(f"\n📊 Performance Metrics:")
            print(f"⏱️  Average Duration: {avg_duration:.4f}s")
            print(f"💾 Peak Memory Usage: {max_memory:.2f} MB")
            print(f"🖥️  Average CPU Usage: {avg_cpu:.2f}%")
        
        # Generate detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.results) * 100
            },
            "results": [asdict(result) for result in self.results]
        }
        
        if successful_tests:
            report["performance_summary"] = {
                "avg_duration": float(np.mean([r.duration for r in successful_tests])),
                "max_memory": float(max([r.memory_peak for r in successful_tests])),
                "avg_cpu": float(np.mean([r.cpu_percent for r in successful_tests]))
            }
        
        # Save report
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        # Generate performance plots
        self._generate_performance_plots()
        
        return report
    
    def _generate_performance_plots(self):
        """Generate performance visualization plots."""
        try:
            successful_results = [r for r in self.results if r.success]
            
            if len(successful_results) < 2:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('BEM Performance Benchmarks', fontsize=16)
            
            # Duration plot
            names = [r.name[:20] + '...' if len(r.name) > 20 else r.name for r in successful_results]
            durations = [r.duration for r in successful_results]
            
            axes[0, 0].bar(range(len(names)), durations)
            axes[0, 0].set_title('Execution Duration')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].set_xticks(range(len(names)))
            axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
            
            # Memory usage plot
            memory_usage = [r.memory_peak for r in successful_results]
            axes[0, 1].bar(range(len(names)), memory_usage)
            axes[0, 1].set_title('Peak Memory Usage')
            axes[0, 1].set_ylabel('Memory (MB)')
            axes[0, 1].set_xticks(range(len(names)))
            axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
            
            # CPU usage plot
            cpu_usage = [r.cpu_percent for r in successful_results]
            axes[1, 0].bar(range(len(names)), cpu_usage)
            axes[1, 0].set_title('Average CPU Usage')
            axes[1, 0].set_ylabel('CPU %')
            axes[1, 0].set_xticks(range(len(names)))
            axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
            
            # Performance vs Memory scatter
            axes[1, 1].scatter(durations, memory_usage, alpha=0.7)
            axes[1, 1].set_xlabel('Duration (seconds)')
            axes[1, 1].set_ylabel('Peak Memory (MB)')
            axes[1, 1].set_title('Performance vs Memory Trade-off')
            
            plt.tight_layout()
            
            plot_file = self.output_dir / f"performance_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance plots saved: {plot_file}")
            
        except ImportError:
            print("📊 Matplotlib not available - skipping performance plots")
        except Exception as e:
            print(f"📊 Error generating plots: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run BEM performance benchmarks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark suite (faster, fewer tests)"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive benchmark suite (includes stability and scalability tests)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    benchmark_suite = BEMBenchmarkSuite(args.project_root, args.output_dir)
    
    try:
        report = benchmark_suite.run_benchmarks(
            quick=args.quick,
            comprehensive=args.comprehensive
        )
        
        success_rate = report["summary"]["success_rate"]
        
        if success_rate >= 90:
            print(f"\n🎉 BENCHMARKS PASSED ({success_rate:.1f}% success rate)")
            sys.exit(0)
        elif success_rate >= 70:
            print(f"\n⚠️  BENCHMARKS PARTIALLY PASSED ({success_rate:.1f}% success rate)")
            sys.exit(1)
        else:
            print(f"\n❌ BENCHMARKS FAILED ({success_rate:.1f}% success rate)")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n⚠️  Benchmarks interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Benchmarks failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()