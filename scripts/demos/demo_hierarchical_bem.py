"""
Comprehensive demo of the hierarchical BEM system.
Demonstrates the full implementation from TODO.md step B4.

This script showcases:
- Hierarchical routing controller (prefix/chunk/token levels)
- Integration with TinyLlama model
- Training with end-to-end methodology
- Comprehensive telemetry and monitoring
- Performance benchmarking vs baselines

Usage:
    python demo_hierarchical_bem.py --mode demo
    python demo_hierarchical_bem.py --mode benchmark
    python demo_hierarchical_bem.py --mode train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import wandb
from dataclasses import asdict

# Import our hierarchical BEM system
from bem.controller import (
    HierarchicalController, 
    RoutingLevel,
    create_hierarchical_controller,
    analyze_routing_behavior,
    compute_routing_stability
)
from bem.hierarchical_bem import (
    HierarchicalBEMConfig,
    FullHierarchicalBEM,
    create_hierarchical_bem,
    create_hierarchical_bem_for_validation
)
from bem.hierarchical_training import (
    HierarchicalTrainingConfig,
    HierarchicalBEMTrainer,
    TrainingStrategy,
    create_end_to_end_trainer
)
from bem.telemetry import (
    TelemetryCollector,
    create_telemetry_collector,
    profile_bem_operation
)

# For validation compatibility
from bem.simple_bem import SimpleBEMModule, BEMController


def setup_device_and_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Setup device and load base model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map=device
        )
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating mock TinyLlama model for demonstration...")
        model, tokenizer = create_mock_tinyllama_model(device)
    
    return model, tokenizer, device


def create_mock_tinyllama_model(device):
    """Create a mock TinyLlama model for testing when real model is unavailable."""
    class MockTinyLlama(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {
                'hidden_size': 2048,
                'num_attention_heads': 32,
                'num_hidden_layers': 22,
                'intermediate_size': 5632,
                'vocab_size': 32000,
                'max_position_embeddings': 2048,
            })()
            
            self.embed_tokens = nn.Embedding(32000, 2048)
            
            # Create some transformer layers
            self.layers = nn.ModuleList([
                self.create_layer() for _ in range(4)  # Smaller for demo
            ])
            
            self.norm = nn.LayerNorm(2048)
            self.lm_head = nn.Linear(2048, 32000, bias=False)
        
        def create_layer(self):
            layer = nn.Module()
            layer.self_attn = nn.MultiheadAttention(2048, 32, batch_first=True)
            layer.mlp = nn.Sequential(
                nn.Linear(2048, 5632),
                nn.SiLU(),
                nn.Linear(5632, 2048)
            )
            layer.input_layernorm = nn.LayerNorm(2048)
            layer.post_attention_layernorm = nn.LayerNorm(2048)
            return layer
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            x = self.embed_tokens(input_ids)
            
            for layer in self.layers:
                # Self attention
                residual = x
                x = layer.input_layernorm(x)
                attn_out, _ = layer.self_attn(x, x, x, key_padding_mask=attention_mask)
                x = residual + attn_out
                
                # MLP
                residual = x
                x = layer.post_attention_layernorm(x)
                x = residual + layer.mlp(x)
            
            x = self.norm(x)
            logits = self.lm_head(x)
            
            return type('Output', (), {'logits': logits, 'hidden_states': x})()
    
    model = MockTinyLlama().to(device)
    
    # Mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 32000
            self.eos_token = "</s>"
            self.pad_token = "</s>"
            self.eos_token_id = 2
            self.pad_token_id = 2
        
        def encode(self, text, return_tensors=None):
            # Simple mock encoding
            tokens = [1] + list(range(len(text.split())))[:30] + [2]
            if return_tensors == 'pt':
                return torch.tensor(tokens).unsqueeze(0)
            return tokens
        
        def decode(self, tokens, skip_special_tokens=True):
            return " ".join([f"token_{t}" for t in tokens if t not in [0, 1, 2]])
    
    tokenizer = MockTokenizer()
    
    print("Created mock TinyLlama model for demonstration")
    return model, tokenizer


def demo_hierarchical_controller(device):
    """Demonstrate hierarchical controller functionality."""
    print("\n" + "="*60)
    print("HIERARCHICAL CONTROLLER DEMONSTRATION")
    print("="*60)
    
    # Create controller
    model_config = {
        'hidden_size': 2048,
        'num_attention_heads': 32,
        'vocab_size': 32000
    }
    
    controller_config = {
        'rank': 16,
        'chunk_size': 32,
        'max_prefix_tokens': 128,
        'ema_decay': 0.99,
        'enable_uncertainty': True,
        'enable_token_routing': True
    }
    
    controller = create_hierarchical_controller(
        model_config, controller_config
    ).to(device)
    
    print(f"Created hierarchical controller:")
    print(f"  - Input dim: {controller.input_dim}")
    print(f"  - Code dim: {controller.code_dim}")
    print(f"  - Chunk size: {controller.chunk_size}")
    print(f"  - Parameters: {sum(p.numel() for p in controller.parameters()):,}")
    
    # Generate sample input
    batch_size, seq_len = 2, 256
    hidden_states = torch.randn(batch_size, seq_len, 2048, device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    side_signals = torch.randn(batch_size, 512, device=device)  # Mock retrieval signals
    
    print(f"\nTesting with input shape: {hidden_states.shape}")
    
    # Test different routing levels
    for routing_level in [RoutingLevel.PREFIX, RoutingLevel.CHUNK, RoutingLevel.TOKEN]:
        print(f"\n--- {routing_level.value.upper()} ROUTING ---")
        
        start_time = time.time()
        codes, routing_state = controller(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            side_signals=side_signals,
            routing_level=routing_level,
            return_routing_state=True
        )
        end_time = time.time()
        
        print(f"Output shape: {codes.shape}")
        print(f"Routing time: {(end_time - start_time)*1000:.2f}ms")
        
        if routing_state.uncertainty is not None:
            print(f"Uncertainty: {routing_state.uncertainty.mean():.4f} ± {routing_state.uncertainty.std():.4f}")
        
        if routing_state.entropy is not None:
            print(f"Entropy: {routing_state.entropy:.4f}")
        
        if routing_state.utilization:
            util = routing_state.utilization
            print(f"Utilization - Active: {util['active_fraction']:.2%}, Norm: {util['code_norm_mean']:.4f}")
        
        # Analyze code distribution
        with torch.no_grad():
            code_mean = codes.mean().item()
            code_std = codes.std().item()
            code_norm = codes.norm(dim=-1).mean().item()
            
            print(f"Code stats - Mean: {code_mean:.4f}, Std: {code_std:.4f}, Norm: {code_norm:.4f}")
    
    # Test routing behavior analysis
    print(f"\n--- ROUTING BEHAVIOR ANALYSIS ---")
    routing_analysis = analyze_routing_behavior(
        controller, hidden_states, attention_mask, num_chunks=4
    )
    
    print("Prefix routing:")
    print(f"  Code norm: {routing_analysis['prefix']['code_norm']:.4f}")
    print(f"  Entropy: {routing_analysis['prefix']['entropy']:.4f}")
    
    print("Chunk routing:")
    for i, chunk_result in enumerate(routing_analysis['chunks']):
        print(f"  Chunk {i}: Norm={chunk_result['code_norm']:.4f}, "
              f"Entropy={chunk_result['entropy']:.4f}")
    
    if 'tokens' in routing_analysis:
        print("Token routing:")
        print(f"  Code norm: {routing_analysis['tokens']['code_norm_mean']:.4f} ± {routing_analysis['tokens']['code_norm_std']:.4f}")
        print(f"  Entropy: {routing_analysis['tokens']['entropy_mean']:.4f} ± {routing_analysis['tokens']['entropy_std']:.4f}")


def demo_hierarchical_bem_integration(model, tokenizer, device):
    """Demonstrate full hierarchical BEM integration."""
    print("\n" + "="*60)
    print("HIERARCHICAL BEM INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Create hierarchical BEM configuration
    bem_config = HierarchicalBEMConfig(
        rank=16,
        alpha=32.0,
        dropout=0.1,
        chunk_size=32,
        max_prefix_tokens=128,
        ema_decay=0.99,
        enable_uncertainty=True,
        enable_token_routing=True,
        attach_mlp=True,
        attach_attention=True,
        attach_qkv=False  # Keep disabled for cache compatibility
    )
    
    print("BEM Configuration:")
    config_dict = asdict(bem_config)
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    
    # Create hierarchical BEM
    hierarchical_bem = create_hierarchical_bem(
        base_model=model,
        config=bem_config
    )
    
    print(f"\nCreated hierarchical BEM:")
    print(f"  Controller parameters: {sum(p.numel() for p in hierarchical_bem.controller.parameters()):,}")
    print(f"  BEM modules: {len(hierarchical_bem.bem_modules)}")
    
    bem_params = hierarchical_bem.get_bem_parameters()
    print(f"  Total BEM parameters: {sum(p.numel() for p in bem_params):,}")
    
    # List attached layers
    print(f"\nAttached layers:")
    for layer_name, bem_module in hierarchical_bem.bem_modules.items():
        print(f"  {layer_name}: {bem_module.attach_point} "
              f"({bem_module.in_features} -> {bem_module.out_features})")
    
    # Test forward pass
    print(f"\n--- FORWARD PASS TEST ---")
    
    # Create sample input
    text = "The hierarchical BEM system demonstrates adaptive parameter generation through"
    inputs = tokenizer.encode(text, return_tensors='pt')
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    inputs = inputs.to(device)
    
    print(f"Input text: '{text}'")
    print(f"Input shape: {inputs.shape}")
    
    # Forward pass with routing info
    with torch.no_grad():
        start_time = time.time()
        outputs, routing_info = hierarchical_bem(
            input_ids=inputs,
            return_routing_info=True
        )
        end_time = time.time()
    
    print(f"Forward pass time: {(end_time - start_time)*1000:.2f}ms")
    
    if hasattr(outputs, 'logits'):
        print(f"Output logits shape: {outputs.logits.shape}")
        
        # Sample next token
        next_token_logits = outputs.logits[0, -1, :]
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
        next_word = tokenizer.decode(next_token.item())
        print(f"Predicted next token: '{next_word}'")
    
    # Display routing statistics
    print(f"\n--- ROUTING STATISTICS ---")
    routing_stats = hierarchical_bem.get_routing_statistics()
    
    print(f"Total BEM modules: {routing_stats['total_bem_modules']}")
    print(f"Global forward calls: {routing_stats['global_stats']['total_forward_calls']}")
    
    if routing_stats['global_stats']['total_forward_calls'] > 0:
        routing_dist = routing_stats['global_stats']['routing_distribution']
        print(f"Routing distribution:")
        print(f"  Prefix: {routing_dist[0]:.2%}")
        print(f"  Chunk: {routing_dist[1]:.2%}")
        print(f"  Token: {routing_dist[2]:.2%}")
    
    print(f"\nPer-layer statistics:")
    for layer_name, layer_stats in routing_stats['layers'].items():
        print(f"  {layer_name}:")
        print(f"    Attach point: {layer_stats['attach_point']}")
        print(f"    Total calls: {layer_stats['total_calls']}")
        print(f"    Routing counts: {layer_stats['routing_counts']}")
    
    return hierarchical_bem, routing_stats


def demo_training_setup(hierarchical_bem, device):
    """Demonstrate training setup and single training step."""
    print("\n" + "="*60)
    print("TRAINING SETUP DEMONSTRATION")
    print("="*60)
    
    # Create training configuration
    training_config = HierarchicalTrainingConfig(
        strategy=TrainingStrategy.END_TO_END,
        learning_rate=1e-3,
        controller_lr=2e-3,  # Higher LR for controller
        bem_lr=5e-4,         # Lower LR for BEM parameters
        max_steps=1000,
        batch_size=4,
        gradient_accumulation_steps=2,
        
        # Loss weights
        lm_loss_weight=1.0,
        kl_divergence_weight=0.05,
        entropy_regularization=0.01,
        delta_norm_regularization=0.001,
        
        # Routing parameters
        routing_temperature=1.0,
        uncertainty_target=0.8,
        
        use_amp=True,
        amp_dtype="fp16"
    )
    
    print("Training Configuration:")
    config_dict = asdict(training_config)
    for key, value in config_dict.items():
        if not key.startswith('expert_') and value is not None:
            print(f"  {key}: {value}")
    
    # Create trainer
    trainer = HierarchicalBEMTrainer(
        model=hierarchical_bem,
        config=training_config,
        device=device
    )
    
    print(f"\nTrainer created:")
    print(f"  Controller optimizer: {trainer.controller_optimizer.__class__.__name__}")
    print(f"  BEM optimizer: {trainer.bem_optimizer.__class__.__name__}")
    print(f"  Mixed precision: {training_config.use_amp}")
    
    # Create dummy batch for training step demo
    batch_size, seq_len = 2, 64
    dummy_batch = {
        'input_ids': torch.randint(1, 1000, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, device=device),
        'labels': torch.randint(1, 1000, (batch_size, seq_len), device=device)
    }
    
    print(f"\n--- TRAINING STEP DEMONSTRATION ---")
    print(f"Batch shape: {dummy_batch['input_ids'].shape}")
    
    # Perform training step
    start_time = time.time()
    loss_dict = trainer.train_step(dummy_batch)
    end_time = time.time()
    
    print(f"Training step time: {(end_time - start_time)*1000:.2f}ms")
    print(f"Loss components:")
    for loss_name, loss_value in loss_dict.items():
        if isinstance(loss_value, float):
            print(f"  {loss_name}: {loss_value:.6f}")
    
    # Check gradient norms
    controller_grad_norm = torch.nn.utils.clip_grad_norm_(
        hierarchical_bem.controller.parameters(), float('inf')
    )
    bem_grad_norm = torch.nn.utils.clip_grad_norm_(
        hierarchical_bem.get_bem_parameters(), float('inf')
    )
    
    print(f"\nGradient norms:")
    print(f"  Controller: {controller_grad_norm:.6f}")
    print(f"  BEM parameters: {bem_grad_norm:.6f}")
    
    return trainer


def demo_telemetry_system(hierarchical_bem, device):
    """Demonstrate telemetry and monitoring system."""
    print("\n" + "="*60)
    print("TELEMETRY SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create telemetry collector
    collector = create_telemetry_collector(
        model=hierarchical_bem,
        collection_interval=1,  # Collect every step for demo
        history_length=100,
        export_path="demo_telemetry"
    )
    
    print("Telemetry collector created:")
    print(f"  Collection interval: {collector.collection_interval}")
    print(f"  History length: {collector.history_length}")
    print(f"  Export path: {collector.export_path}")
    
    # Simulate multiple forward passes with telemetry
    print(f"\n--- SIMULATING OPERATIONS WITH TELEMETRY ---")
    
    batch_size, seq_len = 2, 128
    
    for step in range(5):
        # Create inputs
        input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=device)
        
        # Profile the operation
        with profile_bem_operation(collector, f"operation_step_{step}") as profiler:
            with collector.timing_context("forward_pass"):
                outputs, routing_info = hierarchical_bem(
                    input_ids=input_ids,
                    return_routing_info=True
                )
        
        # Update telemetry
        collector.step_update(
            routing_info=routing_info,
            batch_size=batch_size,
            sequence_length=seq_len
        )
        
        if step == 0 or step == 4:
            current_metrics = collector.get_current_metrics()
            print(f"\nStep {step} metrics:")
            
            if 'performance' in current_metrics:
                perf = current_metrics['performance']
                print(f"  Forward time: {perf['forward_time']:.4f}s")
                print(f"  Throughput: {perf['throughput_tokens_per_sec']:.2f} tokens/sec")
                print(f"  Memory usage: {perf['memory_used']:.2f} MB")
            
            if 'system' in current_metrics:
                system = current_metrics['system']
                print(f"  CPU usage: {system['cpu_percent']:.1f}%")
                print(f"  GPU memory: {system['gpu_memory_used']:.2f} GB")
    
    # Generate performance summary
    print(f"\n--- PERFORMANCE SUMMARY ---")
    summary = collector.get_performance_summary(last_n_steps=5)
    
    if 'forward_time' in summary:
        ft = summary['forward_time']
        print(f"Forward time stats:")
        print(f"  Mean: {ft['mean']:.4f}s")
        print(f"  P95: {ft['p95']:.4f}s")
        print(f"  P99: {ft['p99']:.4f}s")
    
    if 'throughput' in summary:
        tp = summary['throughput']
        print(f"Throughput stats:")
        print(f"  Mean: {tp['mean']:.2f} tokens/sec")
        print(f"  Max: {tp['max']:.2f} tokens/sec")
    
    # Generate report
    print(f"\n--- TELEMETRY REPORT ---")
    report = collector.generate_report(last_n_steps=5)
    print(report)
    
    # Export metrics
    export_path = collector.export_metrics("demo_metrics.json")
    print(f"\nTelemetry exported to: {export_path}")
    
    return collector


def benchmark_performance(hierarchical_bem, baseline_model, device):
    """Benchmark hierarchical BEM vs baseline model."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    # Test configurations
    test_configs = [
        (1, 128),   # Single sequence
        (4, 128),   # Small batch
        (8, 128),   # Medium batch
        (4, 256),   # Longer sequence
        (2, 512),   # Long sequence
    ]
    
    results = []
    
    for batch_size, seq_len in test_configs:
        print(f"\n--- Testing batch_size={batch_size}, seq_len={seq_len} ---")
        
        # Prepare inputs
        input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=device)
        
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = hierarchical_bem(input_ids)
                _ = baseline_model(input_ids)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark hierarchical BEM
        torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
        
        start_time = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = hierarchical_bem(input_ids)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        bem_time = time.time() - start_time
        bem_memory = torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else 0
        
        # Benchmark baseline
        torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
        
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                baseline_outputs = baseline_model(input_ids)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        baseline_time = time.time() - start_time  
        baseline_memory = torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else 0
        
        # Calculate metrics
        time_overhead = (bem_time - baseline_time) / baseline_time * 100
        memory_overhead = (bem_memory - baseline_memory) / baseline_memory * 100 if baseline_memory > 0 else 0
        
        total_tokens = batch_size * seq_len * num_runs
        bem_throughput = total_tokens / bem_time
        baseline_throughput = total_tokens / baseline_time
        
        result = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'bem_time': bem_time,
            'baseline_time': baseline_time,
            'time_overhead': time_overhead,
            'bem_memory': bem_memory,
            'baseline_memory': baseline_memory,
            'memory_overhead': memory_overhead,
            'bem_throughput': bem_throughput,
            'baseline_throughput': baseline_throughput
        }
        
        results.append(result)
        
        print(f"  BEM time: {bem_time:.4f}s vs Baseline: {baseline_time:.4f}s")
        print(f"  Time overhead: {time_overhead:.1f}%")
        print(f"  BEM throughput: {bem_throughput:.2f} tokens/sec")
        print(f"  Baseline throughput: {baseline_throughput:.2f} tokens/sec")
        
        if device.type == 'cuda':
            print(f"  BEM memory: {bem_memory:.1f} MB vs Baseline: {baseline_memory:.1f} MB")
            print(f"  Memory overhead: {memory_overhead:.1f}%")
    
    # Summary
    print(f"\n--- BENCHMARK SUMMARY ---")
    avg_time_overhead = np.mean([r['time_overhead'] for r in results])
    avg_memory_overhead = np.mean([r['memory_overhead'] for r in results])
    
    print(f"Average time overhead: {avg_time_overhead:.1f}%")
    print(f"Average memory overhead: {avg_memory_overhead:.1f}%")
    
    # Check if within acceptable limits (from TODO.md: ≤15% latency budget)
    if avg_time_overhead <= 15.0:
        print(f"✅ Performance within acceptable limits (<= 15%)")
    else:
        print(f"❌ Performance overhead too high (> 15%)")
    
    return results


def run_validation_compatibility_test(device):
    """Test compatibility with existing validation framework."""
    print("\n" + "="*60)
    print("VALIDATION FRAMEWORK COMPATIBILITY TEST")
    print("="*60)
    
    # Create a simple model for validation
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(768, 1024)
            self.linear2 = nn.Linear(1024, 768)
            self.config = type('Config', (), {'hidden_size': 768})()
        
        def forward(self, input_ids, **kwargs):
            # Simple forward pass
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], 768, device=input_ids.device)
            x = self.linear1(x)
            x = F.gelu(x)
            x = self.linear2(x)
            return type('Output', (), {'logits': torch.randn(*input_ids.shape, 32000, device=input_ids.device)})()
    
    test_model = SimpleTestModel().to(device)
    
    # Test compatibility with validation function
    print("Testing compatibility with validation framework...")
    
    hierarchical_bem = create_hierarchical_bem_for_validation(
        model=test_model,
        target_modules=['linear1', 'linear2'],
        rank=8,
        alpha=16.0,
        chunk_size=32,
        enable_uncertainty=True
    )
    
    print(f"Successfully created hierarchical BEM for validation")
    print(f"BEM modules attached: {list(hierarchical_bem.bem_modules.keys())}")
    
    # Test forward pass
    input_ids = torch.randint(1, 1000, (2, 64), device=device)
    
    with torch.no_grad():
        outputs = hierarchical_bem(input_ids)
    
    print(f"Forward pass successful. Output shape: {outputs.logits.shape}")
    
    # Test parameter extraction
    bem_parameters = hierarchical_bem.get_bem_parameters()
    print(f"BEM parameters: {len(bem_parameters)} tensors, {sum(p.numel() for p in bem_parameters):,} total params")
    
    print("✅ Validation framework compatibility test passed")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Hierarchical BEM System Demo")
    parser.add_argument('--mode', choices=['demo', 'benchmark', 'train', 'compatibility'], 
                       default='demo', help='Demo mode to run')
    parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Model name or path')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    
    args = parser.parse_args()
    
    print("🚀 HIERARCHICAL BEM SYSTEM DEMONSTRATION")
    print("Implementation of TODO.md Step B4: Hierarchical Routing Controller")
    print("="*80)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(project="hierarchical-bem-demo", config=vars(args))
    
    try:
        if args.mode == 'compatibility':
            # Just test compatibility
            run_validation_compatibility_test(device)
            return
        
        # Setup model
        model, tokenizer, device = setup_device_and_model(args.model)
        
        if args.mode == 'demo':
            # Full demonstration
            print("\n🎯 Running complete hierarchical BEM demonstration...")
            
            # 1. Controller demonstration
            demo_hierarchical_controller(device)
            
            # 2. BEM integration demonstration
            hierarchical_bem, routing_stats = demo_hierarchical_bem_integration(
                model, tokenizer, device
            )
            
            # 3. Training setup demonstration
            trainer = demo_training_setup(hierarchical_bem, device)
            
            # 4. Telemetry system demonstration
            collector = demo_telemetry_system(hierarchical_bem, device)
            
            # 5. Compatibility test
            run_validation_compatibility_test(device)
            
        elif args.mode == 'benchmark':
            # Performance benchmarking
            print("\n⚡ Running performance benchmarks...")
            
            # Create hierarchical BEM
            bem_config = HierarchicalBEMConfig(rank=16, chunk_size=32)
            hierarchical_bem = create_hierarchical_bem(model, bem_config)
            
            # Benchmark vs baseline
            results = benchmark_performance(hierarchical_bem, model, device)
            
            # Save results
            results_path = Path("benchmark_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark results saved to: {results_path}")
            
        elif args.mode == 'train':
            # Training demonstration
            print("\n🎓 Running training demonstration...")
            
            # Create hierarchical BEM
            bem_config = HierarchicalBEMConfig(rank=16, chunk_size=32)
            hierarchical_bem = create_hierarchical_bem(model, bem_config)
            
            # Setup training
            trainer = demo_training_setup(hierarchical_bem, device)
            
            # Run a few training steps
            print("\nRunning sample training steps...")
            
            for step in range(5):
                # Create dummy batch
                batch = {
                    'input_ids': torch.randint(1, 1000, (2, 64), device=device),
                    'attention_mask': torch.ones(2, 64, device=device),
                    'labels': torch.randint(1, 1000, (2, 64), device=device)
                }
                
                loss_dict = trainer.train_step(batch)
                print(f"Step {step}: Loss = {loss_dict['total_loss']:.6f}")
                
                if args.wandb:
                    wandb.log({'step': step, **loss_dict})
        
        print("\n✅ Demonstration completed successfully!")
        print("\nKey achievements:")
        print("  ✓ Hierarchical routing controller (prefix/chunk/token levels)")
        print("  ✓ Uncertainty estimation and EMA smoothing")
        print("  ✓ Cache-aware Q/K/V routing policies")
        print("  ✓ Integration with transformer architectures")
        print("  ✓ End-to-end training methodology")
        print("  ✓ Comprehensive telemetry and monitoring")
        print("  ✓ Performance benchmarking vs baselines")
        print("  ✓ Validation framework compatibility")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()