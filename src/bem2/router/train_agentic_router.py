#!/usr/bin/env python3
"""
Agentic Router Training Script

Implements the complete AR0→AR1 training pipeline:
- AR0: Behavior Cloning from synthetic traces
- AR1: Policy Gradient from task rewards

Usage:
python bem2/router/train_agentic_router.py --config experiments/AR0_bc.yml --phase bc
python bem2/router/train_agentic_router.py --config experiments/AR1_pg.yml --phase pg --load_bc logs/AR0/best_model.pt
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
import sys
import json
import time
import numpy as np

# Add parent directory to path for imports  
sys.path.append(str(Path(__file__).parent.parent.parent))

from bem2.router.macro_policy import create_macro_policy
from bem2.router.composition_engine import create_composition_engine, create_default_experts
from bem2.router.agentic_router import create_agentic_router
from bem2.router.trace_generator import TraceGenerator
from bem2.router.training import create_bc_trainer, create_pg_trainer, BCDataset
from bem2.router import TraceGenerator


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('agentic_router_training.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_router_components(config: dict, device: torch.device):
    """Create router components from configuration."""
    logger = logging.getLogger(__name__)
    
    # Create macro-policy
    logger.info("Creating macro-policy...")
    policy_config = config.get('macro_policy', {})
    macro_policy = create_macro_policy(policy_config, num_experts=3)
    
    # Create expert deltas
    logger.info("Creating expert bank...")
    expert_deltas = create_default_experts()
    
    # Create composition engine
    logger.info("Creating composition engine...")
    composition_config = config.get('composition', {})
    composition_engine = create_composition_engine(composition_config, expert_deltas)
    
    # Create agentic router
    logger.info("Creating agentic router...")
    router_config = config.get('router', {})
    agentic_router = create_agentic_router(
        config=router_config,
        macro_policy=macro_policy,
        composition_engine=composition_engine
    )
    
    return agentic_router, macro_policy, composition_engine


def train_bc_phase(
    config: dict,
    macro_policy,
    device: torch.device,
    output_dir: str
) -> dict:
    """Train using Behavior Cloning (AR0 phase)."""
    logger = logging.getLogger(__name__)
    logger.info("Starting BC training phase...")
    
    # Load or generate synthetic traces
    trace_config = config.get('trace_generation', {})
    traces_path = trace_config.get('traces_path', 'data/router_traces.jsonl')
    
    if Path(traces_path).exists():
        logger.info(f"Loading traces from {traces_path}")
        traces = TraceGenerator.load_traces(traces_path)
    else:
        logger.info("Generating synthetic traces...")
        generator = TraceGenerator(
            chunk_size=trace_config.get('chunk_size', 128),
            max_chunks_per_trace=trace_config.get('max_chunks_per_trace', 20),
            seed=trace_config.get('seed', 42)
        )
        
        num_traces = trace_config.get('num_traces', 10000)
        domain_dist = trace_config.get('domain_distribution', {
            'Code': 0.4, 'Formal': 0.3, 'Safety': 0.2, 'Mixed': 0.1
        })
        
        traces = generator.generate_dataset(
            num_traces=num_traces,
            domain_distribution=domain_dist,
            output_path=traces_path
        )
    
    logger.info(f"Training with {len(traces)} traces")
    
    # Split traces into train/eval
    split_ratio = config.get('train_eval_split', 0.8)
    split_idx = int(len(traces) * split_ratio)
    train_traces = traces[:split_idx]
    eval_traces = traces[split_idx:]
    
    logger.info(f"Train traces: {len(train_traces)}, Eval traces: {len(eval_traces)}")
    
    # Create trainer
    training_config = config.get('training', {})
    trainer = create_bc_trainer(macro_policy, training_config, device)
    
    # Train
    start_time = time.time()
    training_metrics = trainer.train(
        train_traces=train_traces,
        eval_traces=eval_traces,
        output_dir=output_dir
    )
    training_time = time.time() - start_time
    
    logger.info(f"BC training completed in {training_time:.2f} seconds")
    logger.info(f"Best eval loss: {training_metrics['best_eval_loss']:.4f}")
    
    return training_metrics


def train_pg_phase(
    config: dict,
    agentic_router,
    device: torch.device,
    output_dir: str,
    bc_checkpoint: str = None
) -> dict:
    """Train using Policy Gradient (AR1 phase).""" 
    logger = logging.getLogger(__name__)
    logger.info("Starting PG training phase...")
    
    # Load BC checkpoint if provided
    if bc_checkpoint and Path(bc_checkpoint).exists():
        logger.info(f"Loading BC checkpoint from {bc_checkpoint}")
        checkpoint = torch.load(bc_checkpoint, map_location=device)
        agentic_router.macro_policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Create trainer
    training_config = config.get('training', {})
    trainer = create_pg_trainer(agentic_router, training_config, device)
    
    # Create dummy environment (would be actual task environment)
    class DummyEnvironment:
        def reset(self):
            return torch.randint(0, 1000, (1, 1024))
        
        def step(self, action):
            reward = np.random.normal(0, 0.1)  # Dummy reward
            done = np.random.random() < 0.1
            return reward, done
    
    environment = DummyEnvironment()
    
    # Train
    start_time = time.time()
    num_episodes = config.get('num_episodes', 1000)
    training_metrics = trainer.train(
        environment=environment,
        num_episodes=num_episodes,
        output_dir=output_dir
    )
    training_time = time.time() - start_time
    
    logger.info(f"PG training completed in {training_time:.2f} seconds")
    avg_reward = np.mean(training_metrics['episode_rewards'][-100:])  # Last 100 episodes
    logger.info(f"Average reward (last 100 episodes): {avg_reward:.3f}")
    
    return training_metrics


def evaluate_router(
    agentic_router,
    config: dict,
    device: torch.device,
    output_dir: str
) -> dict:
    """Evaluate trained router."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating router...")
    
    # Generate test sequences
    batch_size = config.get('eval_batch_size', 16)
    seq_len = config.get('eval_seq_len', 1024)
    num_batches = config.get('eval_batches', 10)
    
    eval_metrics = {
        'routing_stats': [],
        'cache_metrics': [],
        'performance_metrics': [],
        'latencies': []
    }
    
    agentic_router.eval()
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Generate random input
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            
            # Time the forward pass
            start_time = time.time()
            outputs, routing_result = agentic_router.forward(
                input_ids=input_ids,
                return_routing_info=True,
                training_mode=False
            )
            latency = time.time() - start_time
            
            if routing_result:
                eval_metrics['routing_stats'].append(routing_result.routing_stats)
                eval_metrics['cache_metrics'].append(routing_result.cache_metrics)
                eval_metrics['performance_metrics'].append(routing_result.performance_metrics)
                eval_metrics['latencies'].append(latency)
    
    # Aggregate metrics
    aggregated_metrics = {}
    
    if eval_metrics['latencies']:
        latencies = eval_metrics['latencies']
        aggregated_metrics.update({
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'max_latency': np.max(latencies)
        })
    
    if eval_metrics['routing_stats']:
        # Average routing statistics
        flip_rates = [stats['flip_rate'] for stats in eval_metrics['routing_stats']]
        aggregated_metrics.update({
            'avg_flip_rate': np.mean(flip_rates),
            'std_flip_rate': np.std(flip_rates)
        })
        
        # Expert utilization
        expert_utils = [stats['expert_utilization'] for stats in eval_metrics['routing_stats']]
        if expert_utils:
            avg_expert_util = np.mean(expert_utils, axis=0).tolist()
            aggregated_metrics['expert_utilization'] = avg_expert_util
    
    if eval_metrics['cache_metrics']:
        cache_safety_rates = [metrics['cache_safety_rate'] for metrics in eval_metrics['cache_metrics']]
        aggregated_metrics.update({
            'avg_cache_safety_rate': np.mean(cache_safety_rates),
            'std_cache_safety_rate': np.std(cache_safety_rates)
        })
    
    # Save evaluation results
    eval_output_path = Path(output_dir) / 'eval_results.json'
    with open(eval_output_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    logger.info(f"Evaluation results saved to {eval_output_path}")
    logger.info(f"Average latency: {aggregated_metrics.get('avg_latency', 0):.3f}s")
    logger.info(f"P95 latency: {aggregated_metrics.get('p95_latency', 0):.3f}s")
    logger.info(f"Cache safety rate: {aggregated_metrics.get('avg_cache_safety_rate', 0):.3f}")
    
    return aggregated_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Agentic Router")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--phase', type=str, choices=['bc', 'pg', 'both'], default='both',
                       help='Training phase to run')
    parser.add_argument('--load_bc', type=str, default=None,
                       help='Path to BC checkpoint for PG training')
    parser.add_argument('--output_dir', type=str, default='logs/agentic_router',
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create router components
    agentic_router, macro_policy, composition_engine = create_router_components(config, device)
    agentic_router.to(device)
    
    logger.info(f"Router created with {sum(p.numel() for p in agentic_router.parameters())} parameters")
    
    # Training
    all_metrics = {}
    
    if args.phase in ['bc', 'both']:
        logger.info("=" * 50)
        logger.info("BEHAVIOR CLONING PHASE (AR0)")
        logger.info("=" * 50)
        
        bc_metrics = train_bc_phase(config, macro_policy, device, str(output_dir))
        all_metrics['bc_phase'] = bc_metrics
    
    if args.phase in ['pg', 'both']:
        logger.info("=" * 50)
        logger.info("POLICY GRADIENT PHASE (AR1)")
        logger.info("=" * 50)
        
        pg_metrics = train_pg_phase(
            config, agentic_router, device, str(output_dir), 
            bc_checkpoint=args.load_bc
        )
        all_metrics['pg_phase'] = pg_metrics
    
    # Evaluation
    logger.info("=" * 50)
    logger.info("EVALUATION")
    logger.info("=" * 50)
    
    eval_metrics = evaluate_router(agentic_router, config, device, str(output_dir))
    all_metrics['evaluation'] = eval_metrics
    
    # Save all metrics
    with open(output_dir / 'training_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    logger.info("Training completed!")
    logger.info(f"Results saved to: {output_dir}")
    
    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    
    if 'evaluation' in all_metrics:
        eval_results = all_metrics['evaluation']
        logger.info(f"Final P50 latency: {eval_results.get('p50_latency', 0):.3f}s")
        logger.info(f"Final P95 latency: {eval_results.get('p95_latency', 0):.3f}s")
        logger.info(f"Cache safety rate: {eval_results.get('avg_cache_safety_rate', 0):.3f}")
        logger.info(f"Expert utilization: {eval_results.get('expert_utilization', [])}")
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()