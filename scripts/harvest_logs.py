#!/usr/bin/env python3
"""
BEM Paper Factory - Log Harvesting Script
Centralizes metrics and artifacts from distributed experiment logs.

Aggregates results from:
- Training logs (loss curves, metrics)
- Evaluation results (per-instance predictions)
- System telemetry (latency, memory, cache stats)
- Routing decisions (gate activations, utilization)
- Statistical metadata (seeds, SHAs, environment)
"""

import argparse
import json
import logging
import os
import glob
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogHarvester:
    """
    Centralizes experimental logs from multiple runs into structured format.
    """
    
    def __init__(self, root_dirs: List[str]):
        self.root_dirs = [Path(d) for d in root_dirs]
        self.harvested_data = []
        
    def discover_experiment_logs(self) -> List[Path]:
        """Discover all experiment log directories."""
        log_dirs = []
        
        for root in self.root_dirs:
            if not root.exists():
                logger.warning(f"Root directory not found: {root}")
                continue
                
            # Find all subdirectories with results.json or trainer_state.json
            for log_dir in root.rglob("*"):
                if log_dir.is_dir():
                    # Check for standard training artifacts
                    has_results = (log_dir / "results.json").exists()
                    has_trainer_state = (log_dir / "trainer_state.json").exists()
                    has_eval_results = (log_dir / "eval_results.json").exists()
                    
                    if has_results or has_trainer_state or has_eval_results:
                        log_dirs.append(log_dir)
                        
        logger.info(f"Discovered {len(log_dirs)} experiment log directories")
        return log_dirs
    
    def extract_experiment_metadata(self, log_dir: Path) -> Dict[str, Any]:
        """Extract experiment configuration and metadata."""
        metadata = {
            'experiment_path': str(log_dir),
            'experiment_name': log_dir.name,
        }
        
        # Try to find config file
        config_files = [
            log_dir / "config.yaml",
            log_dir / "config.json", 
            log_dir / "training_args.json",
            log_dir / "experiment_config.yaml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    if config_file.suffix == '.yaml':
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                    else:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                    metadata['config'] = config
                    break
                except Exception as e:
                    logger.warning(f"Error reading config {config_file}: {e}")
        
        # Extract key experimental parameters
        if 'config' in metadata:
            config = metadata['config']
            metadata.update({
                'experiment_id': config.get('experiment_id', 'unknown'),
                'method_type': config.get('method_type', 'unknown'),
                'approach': config.get('approach', 'unknown'),
                'seed': config.get('seed', None),
                'model_name': config.get('model', {}).get('base_model', 'unknown'),
                'rank': self._extract_rank(config),
                'attach_points': self._extract_attach_points(config),
            })
        
        # Try to extract git SHA and environment info
        self._extract_reproducibility_info(log_dir, metadata)
        
        return metadata
    
    def _extract_rank(self, config: Dict[str, Any]) -> Optional[int]:
        """Extract LoRA rank from config."""
        # Static LoRA
        if 'lora' in config and 'rank' in config['lora']:
            return config['lora']['rank']
        
        # BEM
        if 'bem' in config and 'generator' in config['bem']:
            return config['bem']['generator'].get('rank', None)
        
        return None
    
    def _extract_attach_points(self, config: Dict[str, Any]) -> List[str]:
        """Extract target modules/attach points from config."""
        # Static LoRA
        if 'lora' in config and 'target_modules' in config['lora']:
            return config['lora']['target_modules']
        
        # BEM  
        if 'bem' in config and 'generator' in config['bem']:
            return config['bem']['generator'].get('target_modules', [])
        
        return []
    
    def _extract_reproducibility_info(self, log_dir: Path, metadata: Dict[str, Any]) -> None:
        """Extract git SHA, versions, and environment info."""
        # Git SHA
        git_file = log_dir / "git_sha.txt"
        if git_file.exists():
            try:
                with open(git_file, 'r') as f:
                    metadata['git_sha'] = f.read().strip()
            except Exception as e:
                logger.warning(f"Error reading git SHA from {git_file}: {e}")
        
        # Environment manifest
        env_files = [
            log_dir / "environment.json",
            log_dir / "env_info.yaml", 
            log_dir / "manifest.json"
        ]
        
        for env_file in env_files:
            if env_file.exists():
                try:
                    if env_file.suffix == '.yaml':
                        with open(env_file, 'r') as f:
                            env_info = yaml.safe_load(f)
                    else:
                        with open(env_file, 'r') as f:
                            env_info = json.load(f)
                    metadata['environment'] = env_info
                    break
                except Exception as e:
                    logger.warning(f"Error reading environment from {env_file}: {e}")
    
    def extract_training_metrics(self, log_dir: Path) -> Dict[str, Any]:
        """Extract training metrics and loss curves."""
        metrics = {}
        
        # Trainer state (Transformers)
        trainer_state_file = log_dir / "trainer_state.json"
        if trainer_state_file.exists():
            try:
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
                
                # Extract final metrics
                if 'log_history' in trainer_state and trainer_state['log_history']:
                    final_log = trainer_state['log_history'][-1]
                    metrics.update({
                        'final_train_loss': final_log.get('train_loss', None),
                        'final_eval_loss': final_log.get('eval_loss', None),
                        'total_steps': trainer_state.get('global_step', None),
                        'epochs': trainer_state.get('epoch', None)
                    })
                    
                    # Extract loss curves
                    train_losses = [log.get('train_loss') for log in trainer_state['log_history'] if 'train_loss' in log]
                    eval_losses = [log.get('eval_loss') for log in trainer_state['log_history'] if 'eval_loss' in log]
                    
                    metrics.update({
                        'train_loss_curve': [x for x in train_losses if x is not None],
                        'eval_loss_curve': [x for x in eval_losses if x is not None]
                    })
                    
            except Exception as e:
                logger.warning(f"Error reading trainer state from {trainer_state_file}: {e}")
        
        return metrics
    
    def extract_evaluation_results(self, log_dir: Path) -> Dict[str, Any]:
        """Extract evaluation results and per-instance predictions."""
        results = {}
        
        # Standard evaluation results
        eval_files = [
            log_dir / "eval_results.json",
            log_dir / "results.json",
            log_dir / "test_results.json"
        ]
        
        for eval_file in eval_files:
            if eval_file.exists():
                try:
                    with open(eval_file, 'r') as f:
                        eval_results = json.load(f)
                    results.update(eval_results)
                except Exception as e:
                    logger.warning(f"Error reading eval results from {eval_file}: {e}")
        
        # Per-instance predictions
        predictions_file = log_dir / "predictions.jsonl"
        if predictions_file.exists():
            try:
                predictions = []
                with open(predictions_file, 'r') as f:
                    for line in f:
                        predictions.append(json.loads(line.strip()))
                results['predictions'] = predictions
                results['num_predictions'] = len(predictions)
            except Exception as e:
                logger.warning(f"Error reading predictions from {predictions_file}: {e}")
        
        # BEM-specific metrics
        bem_metrics_file = log_dir / "bem_metrics.json"
        if bem_metrics_file.exists():
            try:
                with open(bem_metrics_file, 'r') as f:
                    bem_metrics = json.load(f)
                results['bem_metrics'] = bem_metrics
            except Exception as e:
                logger.warning(f"Error reading BEM metrics from {bem_metrics_file}: {e}")
        
        return results
    
    def extract_system_telemetry(self, log_dir: Path) -> Dict[str, Any]:
        """Extract system performance metrics."""
        telemetry = {}
        
        # Latency profiling
        latency_file = log_dir / "latency_profile.json"
        if latency_file.exists():
            try:
                with open(latency_file, 'r') as f:
                    latency_data = json.load(f)
                telemetry['latency'] = latency_data
            except Exception as e:
                logger.warning(f"Error reading latency profile from {latency_file}: {e}")
        
        # Memory usage
        memory_file = log_dir / "memory_profile.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                telemetry['memory'] = memory_data
            except Exception as e:
                logger.warning(f"Error reading memory profile from {memory_file}: {e}")
        
        # Throughput metrics
        throughput_file = log_dir / "throughput.json"
        if throughput_file.exists():
            try:
                with open(throughput_file, 'r') as f:
                    throughput_data = json.load(f)
                telemetry['throughput'] = throughput_data
            except Exception as e:
                logger.warning(f"Error reading throughput from {throughput_file}: {e}")
        
        # Cache statistics
        cache_file = log_dir / "cache_stats.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                telemetry['cache'] = cache_data
            except Exception as e:
                logger.warning(f"Error reading cache stats from {cache_file}: {e}")
        
        return telemetry
    
    def extract_routing_data(self, log_dir: Path) -> Dict[str, Any]:
        """Extract BEM routing decisions and gate utilization."""
        routing_data = {}
        
        # Gate activations and utilization
        gate_file = log_dir / "gate_activations.jsonl"
        if gate_file.exists():
            try:
                activations = []
                with open(gate_file, 'r') as f:
                    for line in f:
                        activations.append(json.loads(line.strip()))
                routing_data['gate_activations'] = activations
                
                # Compute utilization statistics
                if activations:
                    all_gates = [act.get('gates', []) for act in activations]
                    if all_gates and len(all_gates[0]) > 0:
                        gate_matrix = np.array(all_gates)
                        routing_data['gate_utilization'] = {
                            'mean_activation': float(np.mean(gate_matrix, axis=0).tolist()),
                            'gate_entropy': float(self._compute_gate_entropy(gate_matrix)),
                            'unused_gates': int(np.sum(np.mean(gate_matrix, axis=0) < 0.01))
                        }
                        
            except Exception as e:
                logger.warning(f"Error reading gate activations from {gate_file}: {e}")
        
        # Routing accuracy (if available)
        routing_acc_file = log_dir / "routing_accuracy.json"
        if routing_acc_file.exists():
            try:
                with open(routing_acc_file, 'r') as f:
                    routing_acc = json.load(f)
                routing_data['routing_accuracy'] = routing_acc
            except Exception as e:
                logger.warning(f"Error reading routing accuracy from {routing_acc_file}: {e}")
        
        return routing_data
    
    def _compute_gate_entropy(self, gate_matrix: np.ndarray) -> float:
        """Compute average gate entropy across all routing decisions."""
        # gate_matrix: [num_examples, num_gates]
        entropies = []
        for i in range(gate_matrix.shape[0]):
            gates = gate_matrix[i]
            # Normalize to probabilities
            if np.sum(gates) > 0:
                probs = gates / np.sum(gates)
                # Compute entropy
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                entropies.append(entropy)
        
        return float(np.mean(entropies)) if entropies else 0.0
    
    def harvest_experiment(self, log_dir: Path) -> Dict[str, Any]:
        """Harvest all data from a single experiment directory."""
        logger.info(f"Harvesting experiment: {log_dir}")
        
        experiment_data = {
            'metadata': self.extract_experiment_metadata(log_dir),
            'training_metrics': self.extract_training_metrics(log_dir),
            'evaluation_results': self.extract_evaluation_results(log_dir),
            'system_telemetry': self.extract_system_telemetry(log_dir),
            'routing_data': self.extract_routing_data(log_dir),
        }
        
        # Flatten some commonly used fields to top level
        experiment_data.update({
            'experiment_id': experiment_data['metadata'].get('experiment_id', 'unknown'),
            'method_type': experiment_data['metadata'].get('method_type', 'unknown'),
            'approach': experiment_data['metadata'].get('approach', 'unknown'),
            'seed': experiment_data['metadata'].get('seed', None),
            'rank': experiment_data['metadata'].get('rank', None),
            'git_sha': experiment_data['metadata'].get('git_sha', None),
        })
        
        return experiment_data
    
    def harvest_all_experiments(self) -> None:
        """Harvest data from all discovered experiments."""
        log_dirs = self.discover_experiment_logs()
        
        for log_dir in log_dirs:
            try:
                experiment_data = self.harvest_experiment(log_dir)
                self.harvested_data.append(experiment_data)
            except Exception as e:
                logger.error(f"Error harvesting experiment {log_dir}: {e}")
    
    def validate_harvested_data(self) -> Dict[str, Any]:
        """Validate harvested data for completeness and consistency."""
        if not self.harvested_data:
            return {'valid': False, 'error': 'No data harvested'}
        
        validation_report = {
            'valid': True,
            'total_experiments': len(self.harvested_data),
            'issues': []
        }
        
        # Check for required fields
        required_fields = ['experiment_id', 'method_type', 'seed']
        for i, exp in enumerate(self.harvested_data):
            for field in required_fields:
                if field not in exp or exp[field] is None:
                    validation_report['issues'].append(
                        f"Experiment {i}: Missing required field '{field}'"
                    )
        
        # Check for rank consistency within method types
        method_ranks = defaultdict(list)
        for exp in self.harvested_data:
            if exp.get('rank') is not None:
                method_ranks[exp['method_type']].append(exp['rank'])
        
        for method, ranks in method_ranks.items():
            unique_ranks = set(ranks)
            if len(unique_ranks) > 1:
                validation_report['issues'].append(
                    f"Method '{method}' has inconsistent ranks: {unique_ranks}"
                )
        
        # Check for seed diversity
        seeds = [exp.get('seed') for exp in self.harvested_data if exp.get('seed') is not None]
        if len(set(seeds)) < 3:
            validation_report['issues'].append(
                f"Insufficient seed diversity: {len(set(seeds))} unique seeds found"
            )
        
        validation_report['valid'] = len(validation_report['issues']) == 0
        
        if validation_report['issues']:
            logger.warning(f"Validation issues found: {len(validation_report['issues'])}")
            for issue in validation_report['issues']:
                logger.warning(f"  - {issue}")
        
        return validation_report
    
    def export_to_jsonl(self, output_path: str) -> None:
        """Export harvested data to JSONL format for statistical analysis."""
        with open(output_path, 'w') as f:
            for experiment in self.harvested_data:
                json_line = json.dumps(experiment, default=str)
                f.write(json_line + '\n')
        
        logger.info(f"Exported {len(self.harvested_data)} experiments to {output_path}")
    
    def generate_summary_report(self, output_path: str) -> None:
        """Generate human-readable summary of harvested data."""
        summary = {
            'harvest_summary': {
                'total_experiments': len(self.harvested_data),
                'method_types': {},
                'seeds_found': set(),
                'experiment_ids': set(),
            },
            'data_completeness': {},
            'quality_checks': {}
        }
        
        # Method type breakdown
        for exp in self.harvested_data:
            method = exp.get('method_type', 'unknown')
            if method not in summary['harvest_summary']['method_types']:
                summary['harvest_summary']['method_types'][method] = 0
            summary['harvest_summary']['method_types'][method] += 1
            
            if exp.get('seed') is not None:
                summary['harvest_summary']['seeds_found'].add(exp['seed'])
                
            if exp.get('experiment_id') is not None:
                summary['harvest_summary']['experiment_ids'].add(exp['experiment_id'])
        
        # Convert sets to lists for JSON serialization
        summary['harvest_summary']['seeds_found'] = sorted(list(summary['harvest_summary']['seeds_found']))
        summary['harvest_summary']['experiment_ids'] = sorted(list(summary['harvest_summary']['experiment_ids']))
        
        # Data completeness analysis
        fields_to_check = [
            'evaluation_results', 'system_telemetry', 'routing_data', 
            'training_metrics', 'git_sha', 'rank'
        ]
        
        for field in fields_to_check:
            count = sum(1 for exp in self.harvested_data 
                       if exp.get(field) is not None and exp[field] != {})
            summary['data_completeness'][field] = {
                'present': count,
                'missing': len(self.harvested_data) - count,
                'completion_rate': count / len(self.harvested_data) if self.harvested_data else 0
            }
        
        # Validation report
        validation = self.validate_harvested_data()
        summary['quality_checks'] = validation
        
        # Write summary
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='BEM Paper Factory - Log Harvester')
    parser.add_argument('--roots', nargs='+', required=True, 
                       help='Root directories to search for experiment logs')
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--summary', help='Optional summary report path')
    
    args = parser.parse_args()
    
    # Initialize harvester
    harvester = LogHarvester(args.roots)
    
    # Harvest all experiments
    logger.info("Starting log harvest...")
    harvester.harvest_all_experiments()
    
    # Validate data
    validation = harvester.validate_harvested_data()
    if not validation['valid']:
        logger.error("Data validation failed!")
        if 'issues' in validation:
            logger.error(f"Issues: {validation['issues']}")
        if 'error' in validation:
            logger.error(f"Error: {validation['error']}")
        return
    
    # Export results
    harvester.export_to_jsonl(args.out)
    
    # Generate summary report if requested
    if args.summary:
        harvester.generate_summary_report(args.summary)
    
    logger.info("Log harvest complete!")

if __name__ == '__main__':
    main()