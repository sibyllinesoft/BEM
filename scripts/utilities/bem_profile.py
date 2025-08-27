#!/usr/bin/env python3
"""
Comprehensive performance profiling suite for BEM CUDA kernels.

This script provides end-to-end performance analysis of the BEM system,
focusing on achieving the critical <15% latency overhead requirement.

Features:
- CUDA kernel vs PyTorch baseline comparison
- Memory bandwidth utilization analysis
- Numerical accuracy validation
- End-to-end BEM system profiling
- Hardware utilization metrics
- Detailed latency breakdown

Target: RTX 3090 Ti with <15% latency overhead
"""

import argparse
import json
import time
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bem.kernels.cuda_ops import (
    benchmark_kernel_performance, 
    validate_numerical_accuracy,
    KERNELS_AVAILABLE
)
from bem.hierarchical_bem import HierarchicalBEMConfig, create_hierarchical_bem
from bem.controller import HierarchicalController, RoutingLevel


def get_system_info() -> Dict[str, Any]:
    """Get system and hardware information."""
    info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'kernels_available': KERNELS_AVAILABLE,
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
        })
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f'device_{i}'] = {
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count,
            }
    
    return info


def profile_kernel_performance(
    config: Dict[str, Any],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Dict[str, Any]:
    """Profile kernel performance across different problem sizes."""
    print("\n" + "="*60)
    print("CUDA KERNEL PERFORMANCE PROFILING")
    print("="*60)
    
    results = {'config': config, 'benchmarks': []}
    
    # Test configurations (realistic BEM scenarios)
    test_configs = [
        # Small models / debugging
        {'batch_size': 1, 'seq_len': 128, 'input_dim': 768, 'intermediate_dim': 8, 'output_dim': 768},
        {'batch_size': 1, 'seq_len': 512, 'input_dim': 768, 'intermediate_dim': 8, 'output_dim': 768},
        
        # Medium models
        {'batch_size': 4, 'seq_len': 512, 'input_dim': 768, 'intermediate_dim': 8, 'output_dim': 768},
        {'batch_size': 8, 'seq_len': 512, 'input_dim': 1024, 'intermediate_dim': 8, 'output_dim': 1024},
        
        # Production scenarios
        {'batch_size': 16, 'seq_len': 512, 'input_dim': 768, 'intermediate_dim': 8, 'output_dim': 768},
        {'batch_size': 32, 'seq_len': 256, 'input_dim': 768, 'intermediate_dim': 8, 'output_dim': 768},
        
        # Large models
        {'batch_size': 8, 'seq_len': 1024, 'input_dim': 1536, 'intermediate_dim': 16, 'output_dim': 1536},
        {'batch_size': 4, 'seq_len': 2048, 'input_dim': 2048, 'intermediate_dim': 16, 'output_dim': 2048},
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\nTest {i+1}/{len(test_configs)}: {test_config}")
        
        try:
            benchmark_result = benchmark_kernel_performance(
                dtype=dtype,
                device=device,
                num_warmup=config.get('num_warmup', 10),
                num_runs=config.get('num_runs', 100),
                **test_config
            )
            
            benchmark_result['test_id'] = i
            results['benchmarks'].append(benchmark_result)
            
            # Print key metrics
            if 'comparison' in benchmark_result:
                comp = benchmark_result['comparison']
                target_met = "✓" if comp['meets_target'] else "✗"
                print(f"  Overhead: {comp['overhead_percent']:.1f}% {target_met}")
                print(f"  Speedup: {comp['speedup_ratio']:.2f}x")
            
        except Exception as e:
            print(f"  Error: {e}")
            results['benchmarks'].append({
                'test_id': i,
                'config': test_config,
                'error': str(e)
            })
    
    return results


def profile_memory_bandwidth(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Dict[str, Any]:
    """Analyze memory bandwidth utilization."""
    print("\n" + "="*60)
    print("MEMORY BANDWIDTH ANALYSIS")
    print("="*60)
    
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    # RTX 3090 Ti theoretical bandwidth
    theoretical_bandwidth_gb_s = 936.2
    
    # Test with large problem size to maximize bandwidth utilization
    config = {
        'batch_size': 32,
        'seq_len': 1024,
        'input_dim': 2048,
        'intermediate_dim': 16,
        'output_dim': 2048
    }
    
    print(f"Testing configuration: {config}")
    
    result = benchmark_kernel_performance(
        dtype=dtype,
        device=device,
        num_warmup=20,
        num_runs=200,
        **config
    )
    
    # Calculate bandwidth utilization
    if 'cuda_kernel' in result and 'detailed_metrics' in result['cuda_kernel']:
        achieved_bandwidth = result['cuda_kernel']['detailed_metrics'].get('bandwidth_gb_s', 0)
        utilization_pct = (achieved_bandwidth / theoretical_bandwidth_gb_s) * 100
        
        bandwidth_analysis = {
            'theoretical_bandwidth_gb_s': theoretical_bandwidth_gb_s,
            'achieved_bandwidth_gb_s': achieved_bandwidth,
            'utilization_percent': utilization_pct,
            'config': config,
            'meets_target': utilization_pct >= 80.0,  # Target >80% utilization
        }
        
        print(f"Theoretical bandwidth: {theoretical_bandwidth_gb_s:.1f} GB/s")
        print(f"Achieved bandwidth: {achieved_bandwidth:.1f} GB/s")
        print(f"Utilization: {utilization_pct:.1f}%")
        print(f"Target met (>80%): {'✓' if bandwidth_analysis['meets_target'] else '✗'}")
        
        return bandwidth_analysis
    
    return {'error': 'Could not measure bandwidth'}


def validate_accuracy_comprehensive(
    device: str = "cuda"
) -> Dict[str, Any]:
    """Comprehensive numerical accuracy validation."""
    print("\n" + "="*60)
    print("NUMERICAL ACCURACY VALIDATION")
    print("="*60)
    
    results = {'validations': []}
    
    # Test different dtypes and sizes
    test_cases = [
        {'dtype': torch.float32, 'size': 'small', 'rtol': 1e-5, 'atol': 1e-6},
        {'dtype': torch.float16, 'size': 'medium', 'rtol': 1e-3, 'atol': 1e-4},
        {'dtype': torch.bfloat16, 'size': 'medium', 'rtol': 1e-2, 'atol': 1e-3},
    ]
    
    size_configs = {
        'small': {'batch_size': 4, 'seq_len': 128, 'input_dim': 256, 'intermediate_dim': 8, 'output_dim': 256},
        'medium': {'batch_size': 8, 'seq_len': 512, 'input_dim': 768, 'intermediate_dim': 8, 'output_dim': 768},
        'large': {'batch_size': 16, 'seq_len': 1024, 'input_dim': 1536, 'intermediate_dim': 16, 'output_dim': 1536}
    }
    
    for i, test_case in enumerate(test_cases):
        print(f"\nValidation {i+1}: {test_case['dtype']} - {test_case['size']}")
        
        config = size_configs[test_case['size']]
        
        try:
            validation_result = validate_numerical_accuracy(
                dtype=test_case['dtype'],
                device=device,
                rtol=test_case['rtol'],
                atol=test_case['atol'],
                **config
            )
            
            validation_result['test_case'] = test_case
            results['validations'].append(validation_result)
            
            status = "PASS" if validation_result['passed'] else "FAIL"
            print(f"  Result: {status}")
            print(f"  Max abs diff: {validation_result['max_abs_diff']:.2e}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results['validations'].append({
                'test_case': test_case,
                'error': str(e)
            })
    
    return results


def profile_end_to_end_bem(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda"
) -> Dict[str, Any]:
    """Profile end-to-end BEM system performance."""
    print("\n" + "="*60)
    print("END-TO-END BEM SYSTEM PROFILING")
    print("="*60)
    
    try:
        # Load base model
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Configure BEM
        bem_config = HierarchicalBEMConfig(
            rank=8,
            alpha=16.0,
            chunk_size=32,
            enable_uncertainty=True,
            enable_token_routing=True
        )
        
        # Create hierarchical BEM
        bem_model = create_hierarchical_bem(
            base_model=base_model,
            config=bem_config,
        )
        
        # Test sequences
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly, we must consider",
            "The benefits of renewable energy include",
        ]
        
        results = {
            'model_name': model_name,
            'bem_config': bem_config.__dict__,
            'benchmarks': []
        }
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nBenchmark {i+1}: '{prompt[:30]}...'")
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Baseline timing (frozen model)
            base_model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(5):
                    _ = base_model(**inputs)
                torch.cuda.synchronize()
                
                # Timed runs
                start_time = time.perf_counter()
                for _ in range(20):
                    _ = base_model(**inputs)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                baseline_time = (end_time - start_time) / 20 * 1000  # ms
            
            # BEM timing
            bem_model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(5):
                    _ = bem_model(**inputs)
                torch.cuda.synchronize()
                
                # Timed runs
                start_time = time.perf_counter()
                for _ in range(20):
                    _ = bem_model(**inputs)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                bem_time = (end_time - start_time) / 20 * 1000  # ms
            
            # Calculate overhead
            overhead_pct = ((bem_time - baseline_time) / baseline_time) * 100
            meets_target = overhead_pct <= 15.0
            
            benchmark_result = {
                'prompt': prompt,
                'seq_len': inputs['input_ids'].shape[1],
                'baseline_time_ms': baseline_time,
                'bem_time_ms': bem_time,
                'overhead_percent': overhead_pct,
                'meets_target': meets_target
            }
            
            results['benchmarks'].append(benchmark_result)
            
            print(f"  Baseline: {baseline_time:.2f} ms")
            print(f"  BEM: {bem_time:.2f} ms")
            print(f"  Overhead: {overhead_pct:.1f}% {'✓' if meets_target else '✗'}")
        
        return results
        
    except Exception as e:
        return {'error': str(e)}


def generate_report(
    results: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate comprehensive performance report."""
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE REPORT")
    print("="*60)
    
    report_file = output_dir / "profile_results.json"
    
    # Save raw results
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Raw results saved to: {report_file}")
    
    # Generate summary
    summary_file = output_dir / "profile_summary.md"
    
    with open(summary_file, 'w') as f:
        f.write("# BEM CUDA Kernels Performance Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System information
        if 'system_info' in results:
            f.write("## System Information\n\n")
            sys_info = results['system_info']
            f.write(f"- CUDA Available: {sys_info.get('cuda_available', 'N/A')}\n")
            f.write(f"- Kernels Available: {sys_info.get('kernels_available', 'N/A')}\n")
            f.write(f"- PyTorch Version: {sys_info.get('torch_version', 'N/A')}\n")
            
            if 'device_0' in sys_info:
                device = sys_info['device_0']
                f.write(f"- GPU: {device.get('name', 'N/A')}\n")
                f.write(f"- Memory: {device.get('total_memory_gb', 0):.1f} GB\n")
            f.write("\n")
        
        # Kernel performance summary
        if 'kernel_performance' in results and 'benchmarks' in results['kernel_performance']:
            f.write("## Kernel Performance Summary\n\n")
            benchmarks = results['kernel_performance']['benchmarks']
            
            passed_tests = 0
            total_tests = 0
            
            for benchmark in benchmarks:
                if 'comparison' in benchmark:
                    total_tests += 1
                    if benchmark['comparison']['meets_target']:
                        passed_tests += 1
            
            f.write(f"- Tests Passed: {passed_tests}/{total_tests}\n")
            f.write(f"- Target: <15% latency overhead\n")
            
            if total_tests > 0:
                overheads = []
                speedups = []
                for benchmark in benchmarks:
                    if 'comparison' in benchmark:
                        overheads.append(benchmark['comparison']['overhead_percent'])
                        speedups.append(benchmark['comparison']['speedup_ratio'])
                
                if overheads:
                    f.write(f"- Average Overhead: {np.mean(overheads):.1f}%\n")
                    f.write(f"- Average Speedup: {np.mean(speedups):.2f}x\n")
            f.write("\n")
        
        # Memory bandwidth summary
        if 'memory_bandwidth' in results and 'utilization_percent' in results['memory_bandwidth']:
            f.write("## Memory Bandwidth Analysis\n\n")
            bw = results['memory_bandwidth']
            f.write(f"- Theoretical: {bw.get('theoretical_bandwidth_gb_s', 0):.1f} GB/s\n")
            f.write(f"- Achieved: {bw.get('achieved_bandwidth_gb_s', 0):.1f} GB/s\n")
            f.write(f"- Utilization: {bw.get('utilization_percent', 0):.1f}%\n")
            f.write(f"- Target Met (>80%): {'✓' if bw.get('meets_target', False) else '✗'}\n\n")
        
        # Accuracy validation summary
        if 'accuracy_validation' in results and 'validations' in results['accuracy_validation']:
            f.write("## Numerical Accuracy Summary\n\n")
            validations = results['accuracy_validation']['validations']
            
            passed_validations = sum(1 for v in validations if v.get('passed', False))
            total_validations = len(validations)
            
            f.write(f"- Validations Passed: {passed_validations}/{total_validations}\n")
            
            for validation in validations:
                if 'test_case' in validation:
                    dtype = validation['test_case']['dtype']
                    status = "PASS" if validation.get('passed', False) else "FAIL"
                    max_diff = validation.get('max_abs_diff', 0)
                    f.write(f"- {dtype}: {status} (max_diff: {max_diff:.2e})\n")
            f.write("\n")
        
        # End-to-end summary
        if 'end_to_end' in results and 'benchmarks' in results['end_to_end']:
            f.write("## End-to-End BEM Performance\n\n")
            benchmarks = results['end_to_end']['benchmarks']
            
            passed_e2e = sum(1 for b in benchmarks if b.get('meets_target', False))
            total_e2e = len(benchmarks)
            
            f.write(f"- Tests Passed: {passed_e2e}/{total_e2e}\n")
            
            if benchmarks:
                overheads = [b.get('overhead_percent', 0) for b in benchmarks]
                f.write(f"- Average Overhead: {np.mean(overheads):.1f}%\n")
            f.write("\n")
        
        # Final assessment
        f.write("## Overall Assessment\n\n")
        
        # Count total passed tests
        total_passed = 0
        total_tests = 0
        
        if 'kernel_performance' in results:
            benchmarks = results['kernel_performance'].get('benchmarks', [])
            for b in benchmarks:
                if 'comparison' in b:
                    total_tests += 1
                    if b['comparison']['meets_target']:
                        total_passed += 1
        
        if 'memory_bandwidth' in results:
            total_tests += 1
            if results['memory_bandwidth'].get('meets_target', False):
                total_passed += 1
        
        if 'accuracy_validation' in results:
            validations = results['accuracy_validation'].get('validations', [])
            for v in validations:
                total_tests += 1
                if v.get('passed', False):
                    total_passed += 1
        
        if 'end_to_end' in results:
            benchmarks = results['end_to_end'].get('benchmarks', [])
            for b in benchmarks:
                total_tests += 1
                if b.get('meets_target', False):
                    total_passed += 1
        
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            f.write(f"**Overall Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})**\n\n")
            
            if success_rate >= 80:
                f.write("✅ **PRODUCTION READY**: Kernels meet performance targets\n")
            elif success_rate >= 60:
                f.write("⚠️ **NEEDS OPTIMIZATION**: Some performance targets missed\n")
            else:
                f.write("❌ **NOT READY**: Significant performance issues detected\n")
    
    print(f"Summary report saved to: {summary_file}")


def main():
    """Main profiling function."""
    parser = argparse.ArgumentParser(description="BEM CUDA Kernels Performance Profiler")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--dtype", default="float16", help="Data type (float16/bfloat16/float32)")
    parser.add_argument("--output-dir", default="logs", help="Output directory for results")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of timed runs")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end profiling")
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                       help="Model for end-to-end testing")
    
    args = parser.parse_args()
    
    # Convert dtype string
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    dtype = dtype_map.get(args.dtype, torch.float16)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("BEM CUDA KERNELS PERFORMANCE PROFILER")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Data Type: {args.dtype}")
    print(f"Output Directory: {output_dir}")
    print(f"Target: <15% latency overhead on RTX 3090 Ti")
    
    # Check prerequisites
    if not torch.cuda.is_available() and args.device == "cuda":
        print("\nError: CUDA not available but CUDA device requested")
        sys.exit(1)
    
    if not KERNELS_AVAILABLE and args.device == "cuda":
        print("\nWarning: CUDA kernels not compiled. Only PyTorch baseline will be tested.")
    
    # Run profiling suite
    results = {}
    
    # System information
    results['system_info'] = get_system_info()
    
    # Profile configuration
    profile_config = {
        'device': args.device,
        'dtype': args.dtype,
        'num_warmup': args.num_warmup,
        'num_runs': args.num_runs,
        'target_overhead_pct': 15.0
    }
    results['profile_config'] = profile_config
    
    try:
        # 1. Kernel performance profiling
        results['kernel_performance'] = profile_kernel_performance(
            profile_config, args.device, dtype
        )
        
        # 2. Memory bandwidth analysis
        if args.device == "cuda":
            results['memory_bandwidth'] = profile_memory_bandwidth(args.device, dtype)
        
        # 3. Numerical accuracy validation
        results['accuracy_validation'] = validate_accuracy_comprehensive(args.device)
        
        # 4. End-to-end BEM profiling
        if not args.skip_e2e:
            results['end_to_end'] = profile_end_to_end_bem(args.model_name, args.device)
        
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
        results['interrupted'] = True
    except Exception as e:
        print(f"\nError during profiling: {e}")
        results['error'] = str(e)
    
    # Generate comprehensive report
    generate_report(results, output_dir)
    
    print(f"\nProfiling complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()