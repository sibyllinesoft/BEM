#!/usr/bin/env python3
"""
Training Pipeline for Agentic Router (AR0→AR1)

Implements the complete training pipeline:
1. AR0: Behavior Cloning from synthetic traces
2. AR1: Policy Gradient fine-tuning with task rewards

Usage:
python bem2/train_agentic_router.py --phase bc --config experiments/AR0_bc.yml
python bem2/train_agentic_router.py --phase pg --config experiments/AR1_pg.yml --checkpoint logs/AR0/best_model.pt
"""

import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from typing import Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from bem2.router.macro_policy import create_macro_policy
from bem2.router.trace_generator import create_trace_generator
from bem2.router.composition_engine import create_composition_engine, create_default_experts
from bem2.router.agentic_router import create_agentic_router
from bem2.router.training import BCTrainer, PGTrainer


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def create_router_from_config(config: Dict, device: torch.device):
    """Create router components from configuration."""
    logger = logging.getLogger(__name__)
    
    # Create router components
    logger.info("Creating router components...")
    
    policy_config = config.get('macro_policy', {})
    macro_policy = create_macro_policy(policy_config, num_experts=3)
    
    expert_deltas = create_default_experts()
    composition_config = config.get('composition', {})
    composition_engine = create_composition_engine(composition_config, expert_deltas)
    
    router_config = config.get('router', {})
    agentic_router = create_agentic_router(
        config=router_config,
        macro_policy=macro_policy,
        composition_engine=composition_engine
    )
    
    agentic_router.to(device)
    return agentic_router


def train_behavior_cloning(
    router, 
    config: Dict, 
    device: torch.device,
    output_dir: Path
) -> Dict:
    """Train router using behavior cloning."""
    logger = logging.getLogger(__name__)
    
    # Generate synthetic traces
    logger.info("Generating synthetic traces...")
    trace_config = config.get('trace_generation', {})
    trace_generator = create_trace_generator(trace_config)
    
    num_train_traces = trace_config.get('num_train_traces', 5000)
    num_eval_traces = trace_config.get('num_eval_traces', 500)
    
    train_traces = trace_generator.generate_dataset(
        num_traces=num_train_traces,
        domains=['code', 'formal', 'safety']
    )
    
    eval_traces = trace_generator.generate_dataset(
        num_traces=num_eval_traces,
        domains=['code', 'formal', 'safety']
    )
    
    logger.info(f"Generated {len(train_traces)} training traces, {len(eval_traces)} eval traces")
    
    # Create trainer
    bc_config = config.get('bc_training', {})
    trainer = BCTrainer(
        router=router,
        device=device,
        learning_rate=bc_config.get('learning_rate', 1e-4),
        batch_size=bc_config.get('batch_size', 32),
        gradient_clip=bc_config.get('gradient_clip', 1.0)
    )
    
    # Train
    logger.info("Starting behavior cloning training...")
    training_results = trainer.train(
        train_traces=train_traces,
        eval_traces=eval_traces,
        num_epochs=bc_config.get('num_epochs', 10),
        eval_interval=bc_config.get('eval_interval', 500),
        save_interval=bc_config.get('save_interval', 1000),
        output_dir=str(output_dir)
    )
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'router_state_dict': router.state_dict(),
        'policy_state_dict': router.macro_policy.state_dict(),
        'config': config,
        'training_results': training_results
    }, final_model_path)
    
    logger.info(f"BC training completed. Final model saved to {final_model_path}")
    
    return training_results


def train_policy_gradient(
    router,
    config: Dict,
    device: torch.device,
    output_dir: Path,
    checkpoint_path: Optional[str] = None
) -> Dict:
    """Train router using policy gradient."""
    logger = logging.getLogger(__name__)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'router_state_dict' in checkpoint:
            router.load_state_dict(checkpoint['router_state_dict'])
        elif 'policy_state_dict' in checkpoint:
            router.macro_policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            logger.warning("No recognized state dict found in checkpoint")
    
    # Create trainer
    pg_config = config.get('pg_training', {})
    trainer = PGTrainer(
        router=router,
        device=device,
        learning_rate=pg_config.get('learning_rate', 5e-5),
        gamma=pg_config.get('gamma', 0.99),
        epsilon_clip=pg_config.get('epsilon_clip', 0.2),
        entropy_coeff=pg_config.get('entropy_coeff', 0.01)
    )
    
    # Create task environment (placeholder)
    from bem2.router.task_environment import create_task_environment
    env_config = config.get('environment', {})
    environment = create_task_environment(env_config)
    
    # Train
    logger.info("Starting policy gradient training...")
    training_results = trainer.train(
        environment=environment,
        num_episodes=pg_config.get('num_episodes', 1000),
        max_steps_per_episode=pg_config.get('max_steps_per_episode', 100),
        eval_interval=pg_config.get('eval_interval', 100),
        save_interval=pg_config.get('save_interval', 200),
        output_dir=str(output_dir)
    )
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'router_state_dict': router.state_dict(),
        'policy_state_dict': router.macro_policy.state_dict(),
        'config': config,
        'training_results': training_results
    }, final_model_path)
    
    logger.info(f"PG training completed. Final model saved to {final_model_path}")
    
    return training_results


def main():
    parser = argparse.ArgumentParser(description="Train Agentic Router")
    parser.add_argument('--phase', type=str, choices=['bc', 'pg'], required=True,
                       help='Training phase: bc (behavior cloning) or pg (policy gradient)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to continue from (for PG phase)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for logs and models')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('logs') / f"AR{'0' if args.phase == 'bc' else '1'}" / time.strftime('%Y%m%d_%H%M%S')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_level, str(output_dir / 'training.log'))
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {args.config}")
    
    # Create router
    router = create_router_from_config(config, device)
    logger.info(f"Router created with {sum(p.numel() for p in router.parameters())} parameters")
    
    # Run training
    start_time = time.time()
    
    if args.phase == 'bc':
        logger.info("Starting Behavior Cloning (AR0) training...")
        training_results = train_behavior_cloning(router, config, device, output_dir)
    else:  # pg
        logger.info("Starting Policy Gradient (AR1) training...")
        training_results = train_policy_gradient(router, config, device, output_dir, args.checkpoint)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f} seconds")
    
    # Save training summary
    summary = {
        'phase': args.phase,
        'config_file': args.config,
        'checkpoint_file': args.checkpoint,
        'device': str(device),
        'training_time_seconds': training_time,
        'output_directory': str(output_dir),
        'router_parameters': sum(p.numel() for p in router.parameters()),
        'training_results': training_results
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Training summary saved to {output_dir / 'training_summary.json'}")
    
    # Find and report best model
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        logger.info(f"Best model available at: {best_model_path}")
    else:
        final_model_path = output_dir / 'final_model.pt'
        if final_model_path.exists():
            logger.info(f"Final model available at: {final_model_path}")
    
    logger.info("=" * 60)
    logger.info(f"TRAINING PHASE {args.phase.upper()} COMPLETE")
    logger.info("=" * 60)
    
    if args.phase == 'bc':
        logger.info("Next step: Run Policy Gradient phase with:")
        logger.info(f"python bem2/train_agentic_router.py --phase pg --config experiments/AR1_pg.yml --checkpoint {output_dir}/best_model.pt")
    else:
        logger.info("Next step: Run comprehensive evaluation with:")
        logger.info(f"python bem2/evaluate_agentic_router.py --model {output_dir}/best_model.pt --out evaluation_results/")


if __name__ == "__main__":
    main()