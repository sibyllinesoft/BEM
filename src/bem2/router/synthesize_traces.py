#!/usr/bin/env python3
"""
Synthetic Trace Generation Script

Generates synthetic routing traces for behavior cloning training.
Used in AR0 phase to create training data for the macro-policy.

Usage:
python bem2/router/synthesize_traces.py --experts Code,Formal,Safety --out data/router_traces.jsonl
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from bem2.router.trace_generator import TraceGenerator, create_trace_generator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic routing traces")
    parser.add_argument('--experts', type=str, default='Code,Formal,Safety',
                       help='Comma-separated list of expert names')
    parser.add_argument('--num_traces', type=int, default=10000,
                       help='Number of traces to generate')
    parser.add_argument('--out', type=str, required=True,
                       help='Output path for traces')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--chunk_size', type=int, default=128,
                       help='Chunk size for routing')
    parser.add_argument('--max_chunks', type=int, default=20,
                       help='Maximum chunks per trace')
    parser.add_argument('--domain_dist', type=str, default='Code:0.4,Formal:0.3,Safety:0.2,Mixed:0.1',
                       help='Domain distribution (name:prob pairs)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting synthetic trace generation")
    logger.info(f"Experts: {args.experts}")
    logger.info(f"Output: {args.out}")
    logger.info(f"Number of traces: {args.num_traces}")
    
    # Parse domain distribution
    domain_distribution = {}
    for pair in args.domain_dist.split(','):
        name, prob = pair.split(':')
        domain_distribution[name.strip()] = float(prob)
    
    logger.info(f"Domain distribution: {domain_distribution}")
    
    # Create trace generator
    config = {
        'chunk_size': args.chunk_size,
        'max_chunks_per_trace': args.max_chunks,
        'seed': args.seed
    }
    
    generator = create_trace_generator(config)
    
    # Generate traces
    logger.info("Generating traces...")
    traces = generator.generate_dataset(
        num_traces=args.num_traces,
        domain_distribution=domain_distribution,
        output_path=args.out
    )
    
    # Summary statistics
    total_steps = sum(len(trace.actions) for trace in traces)
    expert_usage = [0, 0, 0]  # Code, Formal, Safety
    
    for trace in traces:
        for i in trace.metadata['expert_usage']:
            if i < len(expert_usage):
                expert_usage[i] += trace.metadata['expert_usage'][i]
    
    total_expert_steps = sum(expert_usage)
    expert_percentages = [usage / total_expert_steps * 100 if total_expert_steps > 0 else 0 
                         for usage in expert_usage]
    
    logger.info(f"Generated {len(traces)} traces with {total_steps} total routing steps")
    logger.info(f"Expert usage: Code={expert_percentages[0]:.1f}%, "
               f"Formal={expert_percentages[1]:.1f}%, "
               f"Safety={expert_percentages[2]:.1f}%")
    
    avg_reward = sum(trace.metadata['avg_reward'] for trace in traces) / len(traces)
    logger.info(f"Average trace reward: {avg_reward:.3f}")
    
    logger.info(f"Traces saved to: {args.out}")

if __name__ == "__main__":
    main()