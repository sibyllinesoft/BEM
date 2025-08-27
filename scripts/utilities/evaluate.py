"""
BEM v1.1 Evaluation Script

Main evaluation script for BEM-v1.1-stable according to TODO.md specifications.

Usage:
    python evaluate.py --ckpt logs/B1/best.pt --suite eval/suites/main.yml --slice both --out logs/B1/eval.json
    python evaluate.py --ckpt logs/B1/best.pt --mode index_swap --indices indices/clean.faiss,indices/shuffled.faiss,indices/corrupt.faiss --out logs/B1/indexswap.json

Features:
- EM, F1, BLEU, chrF metrics
- Slice A (retrieval-strong) and Slice B (full) analysis
- Cache metrics (KV hit%, tokens/s, latency p50/p95)
- Index-swap monotonicity testing
- Statistical significance analysis
- Comprehensive reporting
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from datetime import datetime

# Add bem package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bem.models import create_bem_v11_model
from bem.evaluation import BEMv11Evaluator
from bem.training.cache_metrics import benchmark_bem_model

# HuggingFace imports
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


def load_evaluation_suite(suite_path: str) -> Dict[str, Any]:
    """Load evaluation suite configuration."""
    with open(suite_path, 'r') as f:
        suite_config = yaml.safe_load(f)
    
    required_fields = ['datasets', 'metrics']
    for field in required_fields:
        if field not in suite_config:
            raise ValueError(f"Missing required field in evaluation suite: {field}")
    
    return suite_config


def create_evaluation_dataset(dataset_config: Dict[str, Any]) -> Dataset:
    """Create evaluation dataset from configuration."""
    if 'huggingface_dataset' in dataset_config:
        dataset_name = dataset_config['huggingface_dataset']
        split = dataset_config.get('split', 'test')
        
        if 'subset' in dataset_config:
            dataset = load_dataset(dataset_name, dataset_config['subset'], split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    elif 'jsonl_path' in dataset_config:
        jsonl_path = dataset_config['jsonl_path']
        with open(jsonl_path, 'r') as f:
            data = [json.loads(line) for line in f]
        dataset = Dataset.from_list(data)
    else:
        raise ValueError("Must specify either 'huggingface_dataset' or 'jsonl_path' in dataset config")
    
    return dataset


def filter_slice_data(dataset: Dataset, slice_type: str, slice_config: Optional[Dict] = None) -> Dataset:
    """Filter dataset based on slice criteria."""
    if slice_type == 'A' and slice_config:
        # Slice A: retrieval-strong examples
        # Filter based on coverage/consistency thresholds
        coverage_threshold = slice_config.get('coverage_threshold', 0.7)
        consistency_threshold = slice_config.get('consistency_threshold', 0.8)
        
        def filter_fn(example):
            # Check if example has retrieval features
            if 'retrieval_features' in example:
                coverage = example['retrieval_features'].get('coverage', 0.0)
                consistency = example['retrieval_features'].get('consistency', 0.0)
                return coverage >= coverage_threshold and consistency >= consistency_threshold
            return False
        
        filtered_dataset = dataset.filter(filter_fn)
        print(f"🔍 Slice A filtered: {len(filtered_dataset)}/{len(dataset)} examples")
        return filtered_dataset
    
    elif slice_type == 'B':
        # Slice B: full dataset
        return dataset
    
    else:
        raise ValueError(f"Unknown slice type: {slice_type}")


def create_eval_data_collator(tokenizer, max_length: int = 4096):
    """Create data collator for evaluation."""
    
    def collate_fn(batch):
        inputs = []
        targets = []
        retrieval_contexts = []
        
        for item in batch:
            if 'input' in item and 'target' in item:
                # Standard input-target format
                inputs.append(item['input'])
                targets.append(item['target'])
            elif 'question' in item and 'answer' in item:
                # QA format
                inputs.append(item['question'])
                targets.append(item['answer'])
            elif 'text' in item:
                # Text completion format
                text = item['text']
                # Split at some point for input/target
                split_point = len(text) // 2
                inputs.append(text[:split_point])
                targets.append(text[split_point:])
            else:
                raise ValueError("Unknown data format for evaluation")
            
            # Retrieval context if available
            retrieval_contexts.append(item.get('retrieval_context', {}))
        
        # Tokenize inputs
        input_tokenized = tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Tokenize targets (for computing metrics)
        target_tokenized = tokenizer(
            targets,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_tokenized['input_ids'],
            'attention_mask': input_tokenized['attention_mask'],
            'labels': target_tokenized['input_ids'],
            'target_texts': targets,  # Keep original text for metrics
            'retrieval_context': retrieval_contexts if any(retrieval_contexts) else None
        }
    
    return collate_fn


def run_standard_evaluation(
    model_path: str,
    suite_config: Dict[str, Any],
    slice_type: str,
    output_path: str,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Run standard evaluation on specified slice."""
    
    print(f"🔬 Running standard evaluation (slice: {slice_type})")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create BEM evaluator
    evaluator = BEMv11Evaluator.from_pretrained(model_path, device=device)
    
    # Process each dataset in the suite
    all_results = {}
    
    for dataset_name, dataset_config in suite_config['datasets'].items():
        print(f"📊 Evaluating on {dataset_name}...")
        
        # Create dataset
        dataset = create_evaluation_dataset(dataset_config)
        
        # Apply slice filtering
        if slice_type in ['A', 'both']:
            slice_config = suite_config.get('slices', {}).get('A', {})
            dataset_slice_a = filter_slice_data(dataset, 'A', slice_config)
        
        if slice_type in ['B', 'both']:
            dataset_slice_b = filter_slice_data(dataset, 'B', None)
        
        # Create dataloaders
        data_collator = create_eval_data_collator(tokenizer)
        
        dataset_results = {}
        
        if slice_type in ['A', 'both'] and len(dataset_slice_a) > 0:
            print(f"  Slice A: {len(dataset_slice_a)} examples")
            dataloader_a = DataLoader(
                dataset_slice_a,
                batch_size=4,  # Small batch size for evaluation
                shuffle=False,
                collate_fn=data_collator
            )
            
            results_a = evaluator.evaluate(
                eval_dataloader=dataloader_a,
                slice_type='A',
                compute_cache_metrics=True
            )
            dataset_results['slice_A'] = results_a
        
        if slice_type in ['B', 'both']:
            print(f"  Slice B: {len(dataset_slice_b)} examples")
            dataloader_b = DataLoader(
                dataset_slice_b,
                batch_size=4,
                shuffle=False, 
                collate_fn=data_collator
            )
            
            results_b = evaluator.evaluate(
                eval_dataloader=dataloader_b,
                slice_type='B',
                compute_cache_metrics=True
            )
            dataset_results['slice_B'] = results_b
        
        all_results[dataset_name] = dataset_results
    
    # Aggregate results
    evaluation_summary = {
        'model_path': model_path,
        'evaluation_suite': suite_config,
        'slice_type': slice_type,
        'results_by_dataset': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add aggregated metrics across datasets
    evaluation_summary['aggregated_metrics'] = aggregate_results(all_results, slice_type)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    
    print(f"💾 Evaluation results saved to {output_path}")
    
    return evaluation_summary


def run_index_swap_evaluation(
    model_path: str,
    index_paths: List[str],
    eval_dataset_path: str,
    output_path: str,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Run index-swap monotonicity evaluation."""
    
    print("🔄 Running index-swap monotonicity evaluation")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load evaluation dataset
    if eval_dataset_path.endswith('.jsonl'):
        with open(eval_dataset_path, 'r') as f:
            eval_data = [json.loads(line) for line in f]
        eval_dataset = Dataset.from_list(eval_data)
    else:
        eval_dataset = load_dataset(eval_dataset_path, split='test')
    
    # Create dataloader
    data_collator = create_eval_data_collator(tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator
    )
    
    # Index labels (from TODO.md)
    index_labels = ['clean', 'shuffled', 'corrupt'][:len(index_paths)]
    
    results_by_index = {}
    
    # Evaluate with each index
    for index_path, label in zip(index_paths, index_labels):
        print(f"📊 Evaluating with {label} index...")
        
        # Create evaluator with this specific index
        evaluator = BEMv11Evaluator.from_pretrained(
            model_path, 
            retrieval_index_path=index_path,
            device=device
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            eval_dataloader=eval_dataloader,
            slice_type='B',  # Use full dataset for index swap
            compute_cache_metrics=True
        )
        
        results_by_index[label] = results
    
    # Check monotonicity
    monotonicity_results = check_index_swap_monotonicity(results_by_index, index_labels)
    
    # Compile final results
    index_swap_summary = {
        'model_path': model_path,
        'index_paths': {label: path for label, path in zip(index_labels, index_paths)},
        'results_by_index': results_by_index,
        'monotonicity_analysis': monotonicity_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(index_swap_summary, f, indent=2, default=str)
    
    print(f"💾 Index-swap results saved to {output_path}")
    
    return index_swap_summary


def check_index_swap_monotonicity(
    results_by_index: Dict[str, Dict],
    index_labels: List[str]
) -> Dict[str, Any]:
    """Check monotonicity: clean > shuffled > corrupt."""
    
    print("🔍 Checking index-swap monotonicity...")
    
    # Expected order from TODO.md
    expected_order = ['clean', 'shuffled', 'corrupt']
    
    monotonicity_check = {
        'expected_order': expected_order,
        'passed': True,
        'violations': [],
        'metric_comparisons': {}
    }
    
    # Key metrics to check
    key_metrics = ['exact_match', 'f1_score', 'bleu', 'chrf']
    
    for metric in key_metrics:
        if all(metric in results_by_index.get(label, {}) for label in index_labels):
            values = []
            for label in index_labels:
                if label in expected_order:
                    metric_data = results_by_index[label][metric]
                    value = metric_data['mean'] if isinstance(metric_data, dict) else metric_data
                    values.append(value)
            
            if len(values) >= 2:
                # Check if monotonic (non-increasing)
                is_monotonic = all(values[i] >= values[i+1] for i in range(len(values)-1))
                
                monotonicity_check['metric_comparisons'][metric] = {
                    'values': {label: val for label, val in zip(index_labels, values)},
                    'is_monotonic': is_monotonic,
                    'violations': []
                }
                
                # Record violations
                for i in range(len(values)-1):
                    if values[i] < values[i+1]:
                        violation = f"{index_labels[i]} ({values[i]:.4f}) < {index_labels[i+1]} ({values[i+1]:.4f})"
                        monotonicity_check['violations'].append(f"{metric}: {violation}")
                        monotonicity_check['metric_comparisons'][metric]['violations'].append(violation)
                        monotonicity_check['passed'] = False
    
    if monotonicity_check['passed']:
        print("✅ Index-swap monotonicity test PASSED")
    else:
        print("❌ Index-swap monotonicity test FAILED")
        for violation in monotonicity_check['violations']:
            print(f"  - {violation}")
    
    return monotonicity_check


def aggregate_results(all_results: Dict[str, Dict], slice_type: str) -> Dict[str, Any]:
    """Aggregate results across datasets."""
    
    aggregated = {}
    
    # Collect all metrics across datasets and slices
    all_metrics = defaultdict(list)
    
    for dataset_name, dataset_results in all_results.items():
        for slice_name, slice_results in dataset_results.items():
            if slice_type == 'both' or slice_name.endswith(slice_type):
                for metric_name, metric_data in slice_results.items():
                    if isinstance(metric_data, dict) and 'mean' in metric_data:
                        all_metrics[metric_name].append(metric_data['mean'])
                    elif isinstance(metric_data, (int, float)):
                        all_metrics[metric_name].append(metric_data)
    
    # Compute aggregated statistics
    for metric_name, values in all_metrics.items():
        if values:
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    return aggregated


def generate_evaluation_report(results: Dict[str, Any], report_path: str):
    """Generate human-readable evaluation report."""
    
    report_lines = [
        "# BEM v1.1 Evaluation Report",
        "",
        f"**Model**: {results.get('model_path', 'Unknown')}",
        f"**Evaluation Date**: {results.get('timestamp', 'Unknown')}",
        f"**Slice Type**: {results.get('slice_type', 'Unknown')}",
        ""
    ]
    
    # Core metrics summary
    if 'aggregated_metrics' in results:
        report_lines.extend([
            "## Aggregated Metrics",
            ""
        ])
        
        metrics = results['aggregated_metrics']
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                mean_val = metric_data['mean']
                std_val = metric_data.get('std', 0)
                report_lines.append(f"- **{metric_name.upper()}**: {mean_val:.4f} ± {std_val:.4f}")
    
    # Index swap results if available
    if 'monotonicity_analysis' in results:
        mono_analysis = results['monotonicity_analysis']
        status = "✅ PASSED" if mono_analysis['passed'] else "❌ FAILED"
        
        report_lines.extend([
            "",
            "## Index-Swap Monotonicity Test",
            "",
            f"**Status**: {status}",
            ""
        ])
        
        if 'metric_comparisons' in mono_analysis:
            report_lines.append("**Metric Comparisons**:")
            report_lines.append("")
            
            for metric, comparison in mono_analysis['metric_comparisons'].items():
                values = comparison.get('values', {})
                is_mono = comparison.get('is_monotonic', False)
                mono_status = "✅" if is_mono else "❌"
                
                report_lines.append(f"- {metric.upper()} {mono_status}:")
                for idx_label, value in values.items():
                    report_lines.append(f"  - {idx_label}: {value:.4f}")
                report_lines.append("")
    
    # Dataset-specific results
    if 'results_by_dataset' in results:
        report_lines.extend([
            "",
            "## Results by Dataset",
            ""
        ])
        
        for dataset_name, dataset_results in results['results_by_dataset'].items():
            report_lines.append(f"### {dataset_name}")
            report_lines.append("")
            
            for slice_name, slice_results in dataset_results.items():
                report_lines.append(f"**{slice_name}**:")
                
                # Core metrics
                for metric in ['exact_match', 'f1_score', 'bleu', 'chrf']:
                    if metric in slice_results:
                        metric_data = slice_results[metric]
                        if isinstance(metric_data, dict) and 'mean' in metric_data:
                            value = metric_data['mean']
                            report_lines.append(f"- {metric}: {value:.4f}")
                
                report_lines.append("")
    
    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Evaluation report saved to {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='BEM v1.1 Evaluation Script')
    
    # Required arguments
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--out', type=str, required=True,
                       help='Output path for evaluation results')
    
    # Evaluation modes
    parser.add_argument('--mode', type=str, choices=['standard', 'index_swap'], default='standard',
                       help='Evaluation mode')
    
    # Standard evaluation arguments
    parser.add_argument('--suite', type=str,
                       help='Path to evaluation suite YAML (for standard mode)')
    parser.add_argument('--slice', type=str, choices=['A', 'B', 'both'], default='both',
                       help='Which slice(s) to evaluate (for standard mode)')
    
    # Index swap arguments
    parser.add_argument('--indices', type=str,
                       help='Comma-separated list of index paths (for index_swap mode)')
    parser.add_argument('--dataset', type=str,
                       help='Evaluation dataset path (for index_swap mode)')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for evaluation')
    parser.add_argument('--report', action='store_true',
                       help='Generate human-readable report')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'standard':
        if not args.suite:
            parser.error("--suite is required for standard evaluation mode")
    elif args.mode == 'index_swap':
        if not args.indices or not args.dataset:
            parser.error("--indices and --dataset are required for index_swap mode")
    
    print(f"🔬 BEM v1.1 Evaluation")
    print(f"   Mode: {args.mode}")
    print(f"   Checkpoint: {args.ckpt}")
    print(f"   Output: {args.out}")
    print()
    
    try:
        if args.mode == 'standard':
            # Load evaluation suite
            suite_config = load_evaluation_suite(args.suite)
            
            # Run standard evaluation
            results = run_standard_evaluation(
                model_path=args.ckpt,
                suite_config=suite_config,
                slice_type=args.slice,
                output_path=args.out,
                device=args.device
            )
            
        elif args.mode == 'index_swap':
            # Parse index paths
            index_paths = [p.strip() for p in args.indices.split(',')]
            
            # Run index swap evaluation
            results = run_index_swap_evaluation(
                model_path=args.ckpt,
                index_paths=index_paths,
                eval_dataset_path=args.dataset,
                output_path=args.out,
                device=args.device
            )
        
        # Generate report if requested
        if args.report:
            report_path = args.out.replace('.json', '_report.md')
            generate_evaluation_report(results, report_path)
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    from collections import defaultdict
    sys.exit(main())