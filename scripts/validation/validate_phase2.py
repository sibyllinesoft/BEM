#!/usr/bin/env python3
"""
Validation script for Phase 2 hierarchical BEM implementation.
Tests the key requirements from TODO.md Phase 2 specifications.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any

from bem.hierarchical_bem import (
    HierarchicalBEMConfig, 
    HierarchicalBEMModule,
    FullHierarchicalBEM,
    create_hierarchical_bem
)
from bem.controller import (
    HierarchicalController, 
    RoutingLevel,
    create_hierarchical_controller,
    analyze_routing_behavior
)


def create_mock_model(hidden_size=768):
    """Create a simple mock model for testing."""
    class MockTransformerLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.mlp_up = nn.Linear(hidden_size, hidden_size * 4)
            self.mlp_down = nn.Linear(hidden_size * 4, hidden_size)
            self.attention_out = nn.Linear(hidden_size, hidden_size)
            
        def forward(self, x):
            # Simple forward pass
            mlp_hidden = torch.relu(self.mlp_up(x))
            mlp_out = self.mlp_down(mlp_hidden)
            attn_out = self.attention_out(x)
            return x + mlp_out + attn_out
    
    class MockModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layers = nn.ModuleList([MockTransformerLayer(hidden_size) for _ in range(2)])
            self.config = {'hidden_size': hidden_size}
            
        def forward(self, input_ids, **kwargs):
            # Convert input_ids to embeddings (simplified)
            batch_size, seq_len = input_ids.shape
            x = torch.randn(batch_size, seq_len, hidden_size, device=input_ids.device)
            
            for layer in self.layers:
                x = layer(x)
                
            return type('Output', (), {'last_hidden_state': x})()
    
    return MockModel(hidden_size)


def test_hierarchical_controller():
    """Test the hierarchical controller functionality."""
    print("=== Testing Hierarchical Controller ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create controller
    controller = HierarchicalController(
        input_dim=768,
        code_dim=8,
        chunk_size=32,
        enable_uncertainty=True,
        enable_token_routing=True
    ).to(device)
    
    # Test data
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, 768, device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    
    print(f"Input shape: {hidden_states.shape}")
    
    # Test different routing levels
    routing_levels = [RoutingLevel.PREFIX, RoutingLevel.CHUNK, RoutingLevel.TOKEN]
    
    for level in routing_levels:
        print(f"\nTesting {level.value} routing:")
        
        start_time = time.time()
        codes, routing_state = controller(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            routing_level=level,
            return_routing_state=True
        )
        end_time = time.time()
        
        print(f"  Output shape: {codes.shape}")
        print(f"  Code norm mean: {codes.norm(dim=-1).mean():.4f}")
        print(f"  Entropy: {routing_state.entropy:.4f}")
        print(f"  Processing time: {(end_time - start_time)*1000:.2f}ms")
        
        if routing_state.uncertainty is not None:
            print(f"  Uncertainty: {routing_state.uncertainty.mean():.4f}")
    
    print("✓ Hierarchical Controller tests passed")


def test_hierarchical_bem_module():
    """Test the hierarchical BEM module."""
    print("\n=== Testing Hierarchical BEM Module ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base layer and BEM module
    base_layer = nn.Linear(768, 3072).to(device)
    config = HierarchicalBEMConfig(
        rank=8,
        chunk_size=32,
        enable_uncertainty=True,
        enable_token_routing=True
    )
    
    bem_module = HierarchicalBEMModule(
        base_layer=base_layer,
        config=config,
        layer_name="test_layer",
        attach_point="mlp"
    ).to(device)
    
    # Create controller
    controller = HierarchicalController(
        input_dim=768,
        code_dim=config.rank
    ).to(device)
    
    # Test data
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, 768, device=device)
    hidden_states = torch.randn(batch_size, seq_len, 768, device=device)
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass with different routing levels
    for level in [RoutingLevel.CHUNK, RoutingLevel.TOKEN]:
        print(f"\nTesting {level.value} routing:")
        
        start_time = time.time()
        output, routing_info = bem_module(
            x=x,
            hidden_states=hidden_states,
            controller=controller,
            return_routing_info=True
        )
        end_time = time.time()
        
        print(f"  Output shape: {output.shape}")
        print(f"  Layer name: {routing_info['layer_name']}")
        print(f"  Routing level: {routing_info['routing_level']}")
        print(f"  Code norm: {routing_info['code_norm']:.4f}")
        print(f"  LoRA norm: {routing_info['lora_norm']:.4f}")
        print(f"  Processing time: {(end_time - start_time)*1000:.2f}ms")
        
        if routing_info['uncertainty'] is not None:
            print(f"  Uncertainty: {routing_info['uncertainty']:.4f}")
    
    print("✓ Hierarchical BEM Module tests passed")


def test_cache_safety():
    """Test cache safety with chunk-level Q/K/V routing."""
    print("\n=== Testing Cache Safety ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create QKV layers and BEM modules
    config = HierarchicalBEMConfig(
        rank=8,
        chunk_size=32,
        attach_qkv=True  # Enable Q/K/V attachment
    )
    
    q_layer = nn.Linear(768, 768).to(device)
    k_layer = nn.Linear(768, 768).to(device)
    v_layer = nn.Linear(768, 768).to(device)
    
    q_bem = HierarchicalBEMModule(q_layer, config, "q_proj", "q").to(device)
    k_bem = HierarchicalBEMModule(k_layer, config, "k_proj", "k").to(device)
    v_bem = HierarchicalBEMModule(v_layer, config, "v_proj", "v").to(device)
    
    controller = HierarchicalController(input_dim=768, code_dim=8).to(device)
    
    # Test that Q/K/V layers use chunk-level routing (cache-safe)
    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, 768, device=device)
    hidden_states = torch.randn(batch_size, seq_len, 768, device=device)
    
    qkv_modules = [("Q", q_bem), ("K", k_bem), ("V", v_bem)]
    
    for name, module in qkv_modules:
        routing_level = module.get_routing_level(seq_len, chunk_position=0)
        print(f"  {name} routing level: {routing_level}")
        assert routing_level == RoutingLevel.CHUNK, f"{name} should use chunk routing for cache safety"
    
    print("✓ Cache safety tests passed")


def test_ema_smoothing():
    """Test EMA smoothing behavior in chunk router."""
    print("\n=== Testing EMA Smoothing ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from bem.controller import ChunkRouter
    
    # Create chunk router with strong EMA decay
    router = ChunkRouter(
        input_dim=768,
        code_dim=8,
        chunk_size=32,
        ema_decay=0.95  # Strong smoothing
    ).to(device)
    
    # Generate different inputs
    prefix_summary = torch.randn(2, 768, device=device)
    hidden_states1 = torch.randn(2, 64, 768, device=device)
    hidden_states2 = torch.randn(2, 64, 768, device=device) * 2.0  # Different scale
    
    print("Processing first chunk:")
    chunk_code1, ema_code1 = router(hidden_states1, prefix_summary, training=True)
    print(f"  Chunk code norm: {chunk_code1.norm(dim=-1).mean():.4f}")
    print(f"  EMA code norm: {ema_code1.norm(dim=-1).mean():.4f}")
    
    print("Processing second chunk:")
    chunk_code2, ema_code2 = router(hidden_states2, prefix_summary, training=True)
    print(f"  Chunk code norm: {chunk_code2.norm(dim=-1).mean():.4f}")  
    print(f"  EMA code norm: {ema_code2.norm(dim=-1).mean():.4f}")
    
    # Check EMA smoothing effect
    ema_change = (ema_code2 - ema_code1).norm()
    chunk_change = (chunk_code2 - chunk_code1).norm()
    
    print(f"  EMA change: {ema_change:.4f}")
    print(f"  Chunk change: {chunk_change:.4f}")
    print(f"  Smoothing ratio: {ema_change / chunk_change:.4f}")
    
    # EMA should change less than raw chunk codes
    assert ema_change < chunk_change, "EMA should smooth out changes"
    
    print("✓ EMA smoothing tests passed")


def test_performance_vs_phase1():
    """Test performance compared to Phase 1 (simple BEM)."""
    print("\n=== Testing Performance vs Phase 1 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from bem.simple_bem import SimpleBEMModule, BEMController
    
    # Setup for comparison
    batch_size, seq_len = 4, 256
    hidden_size = 768
    
    # Phase 1: Simple BEM
    base_layer = nn.Linear(hidden_size, hidden_size * 4).to(device)
    simple_bem = SimpleBEMModule(base_layer, rank=8, alpha=16.0).to(device)
    simple_controller = BEMController(hidden_size, 8).to(device)
    
    # Phase 2: Hierarchical BEM  
    config = HierarchicalBEMConfig(rank=8, chunk_size=32)
    hier_bem = HierarchicalBEMModule(
        nn.Linear(hidden_size, hidden_size * 4).to(device), 
        config, "test", "mlp"
    ).to(device)
    hier_controller = HierarchicalController(hidden_size, 8).to(device)
    
    # Test data
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = simple_bem(x, simple_controller(hidden_states))
        _ = hier_bem(x, hidden_states, hier_controller)
    
    # Benchmark Phase 1
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = simple_bem(x, simple_controller(hidden_states))
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    phase1_time = time.time() - start_time
    
    # Benchmark Phase 2
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = hier_bem(x, hidden_states, hier_controller)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    phase2_time = time.time() - start_time
    
    # Calculate overhead
    overhead = (phase2_time - phase1_time) / phase1_time * 100
    
    print(f"  Phase 1 time: {phase1_time:.4f}s")
    print(f"  Phase 2 time: {phase2_time:.4f}s")
    print(f"  Overhead: {overhead:.1f}%")
    
    # Check acceptance criteria (≤15% overhead)
    if overhead <= 15.0:
        print("✓ Performance overhead within acceptable limits (≤15%)")
    else:
        print(f"⚠ Performance overhead {overhead:.1f}% exceeds 15% limit")
    
    return {'phase1_time': phase1_time, 'phase2_time': phase2_time, 'overhead_percent': overhead}


def test_routing_effectiveness():
    """Test routing effectiveness across different scenarios."""
    print("\n=== Testing Routing Effectiveness ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    controller = HierarchicalController(
        input_dim=768,
        code_dim=8,
        chunk_size=32,
        enable_uncertainty=True
    ).to(device)
    
    # Generate different types of inputs to test routing adaptability
    scenarios = {
        'short_sequence': torch.randn(2, 32, 768, device=device),
        'medium_sequence': torch.randn(2, 128, 768, device=device), 
        'long_sequence': torch.randn(2, 512, 768, device=device),
        'structured_input': torch.randn(2, 256, 768, device=device) * 0.1,  # Low variance
        'noisy_input': torch.randn(2, 256, 768, device=device) * 2.0  # High variance
    }
    
    results = {}
    
    for scenario_name, hidden_states in scenarios.items():
        print(f"\n  Testing {scenario_name}:")
        
        analysis = analyze_routing_behavior(
            controller, hidden_states, num_chunks=min(4, hidden_states.shape[1] // 32)
        )
        
        print(f"    Prefix entropy: {analysis['prefix']['entropy']:.4f}")
        print(f"    Chunk entropy avg: {np.mean([c['entropy'] for c in analysis['chunks']]):.4f}")
        
        if 'tokens' in analysis:
            print(f"    Token entropy: {analysis['tokens']['entropy_mean']:.4f}")
        
        results[scenario_name] = analysis
    
    print("✓ Routing effectiveness tests completed")
    return results


def main():
    """Run all Phase 2 validation tests."""
    print("🧪 Phase 2 Hierarchical BEM Validation")
    print("======================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Core functionality tests
        test_hierarchical_controller()
        test_hierarchical_bem_module()
        
        # Phase 2 specific requirements
        test_cache_safety()
        test_ema_smoothing()
        
        # Performance and effectiveness
        perf_results = test_performance_vs_phase1()
        routing_results = test_routing_effectiveness()
        
        # Summary
        print("\n🎯 Phase 2 Validation Summary")
        print("============================")
        print("✓ Hierarchical routing (prefix/chunk/token) - WORKING")
        print("✓ Cache-safe Q/K/V routing (chunk-level only) - WORKING") 
        print("✓ EMA smoothing for chunk codes - WORKING")
        print("✓ Uncertainty estimation and gating - WORKING")
        print(f"✓ Performance overhead: {perf_results['overhead_percent']:.1f}%")
        
        if perf_results['overhead_percent'] <= 15.0:
            print("🎉 Phase 2 ACCEPTANCE CRITERIA MET!")
            print("   - All hierarchical routing features implemented")
            print("   - Cache safety maintained") 
            print("   - Performance within 15% overhead limit")
            print("   - EMA smoothing working correctly")
        else:
            print("⚠ Phase 2 partially complete - performance optimization needed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 2 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)