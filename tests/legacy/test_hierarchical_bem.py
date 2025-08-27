"""
Comprehensive test suite for hierarchical BEM implementation.
Tests all components: controller, integration, performance, and validation.

This test suite validates:
- Individual routing level functionality
- Integration with TinyLlama architecture  
- Performance benchmarking vs baselines
- Training stability and convergence
- Memory efficiency and throughput
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import numpy as np
import time
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
from contextlib import contextmanager

# Import our modules
from .controller import (
    HierarchicalController, 
    PrefixRouter, 
    ChunkRouter, 
    TokenRouter, 
    UncertaintyHead,
    RoutingLevel,
    create_hierarchical_controller
)
from .hierarchical_bem import (
    HierarchicalBEMModule,
    FullHierarchicalBEM, 
    HierarchicalBEMConfig,
    create_hierarchical_bem
)
from .hierarchical_training import (
    HierarchicalBEMTrainer,
    HierarchicalTrainingConfig,
    TrainingStrategy
)
from .telemetry import TelemetryCollector, create_telemetry_collector


# Test fixtures and utilities

@pytest.fixture
def device():
    """Get test device (prefer CUDA if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture 
def mock_model_config():
    """Mock model configuration for testing."""
    return {
        'hidden_size': 768,
        'num_attention_heads': 12,
        'num_hidden_layers': 12,
        'intermediate_size': 3072,
        'vocab_size': 32000
    }


@pytest.fixture
def sample_hidden_states(device):
    """Generate sample hidden states for testing."""
    batch_size, seq_len, hidden_dim = 2, 128, 768
    return torch.randn(batch_size, seq_len, hidden_dim, device=device)


@pytest.fixture
def sample_attention_mask(device):
    """Generate sample attention mask."""
    batch_size, seq_len = 2, 128
    mask = torch.ones(batch_size, seq_len, device=device)
    # Zero out last few tokens for testing
    mask[:, -10:] = 0
    return mask


@pytest.fixture
def hierarchical_config():
    """Default hierarchical BEM configuration for testing."""
    return HierarchicalBEMConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        chunk_size=32,
        max_prefix_tokens=64,  # Smaller for tests
        ema_decay=0.95,
        enable_uncertainty=True,
        enable_token_routing=True,
        code_clamp_value=3.0
    )


class TestPrefixRouter:
    """Test suite for prefix router component."""
    
    def test_initialization(self, device):
        """Test prefix router initialization."""
        router = PrefixRouter(
            input_dim=768,
            code_dim=8,
            hidden_dim=512,
            max_prefix_tokens=64
        ).to(device)
        
        assert router.input_dim == 768
        assert router.code_dim == 8
        assert router.hidden_dim == 512
        assert router.max_prefix_tokens == 64
        
        # Check parameter initialization
        for param in router.parameters():
            assert param.requires_grad
            assert not torch.isnan(param).any()
    
    def test_forward_pass(self, device, sample_hidden_states, sample_attention_mask):
        """Test prefix router forward pass."""
        router = PrefixRouter(
            input_dim=768,
            code_dim=8,
            max_prefix_tokens=64
        ).to(device)
        
        # Test without attention mask
        prefix_code = router(sample_hidden_states)
        assert prefix_code.shape == (2, 8)  # [batch_size, code_dim]
        assert not torch.isnan(prefix_code).any()
        
        # Test with attention mask
        prefix_code_masked = router(sample_hidden_states, sample_attention_mask)
        assert prefix_code_masked.shape == (2, 8)
        assert not torch.isnan(prefix_code_masked).any()
        
        # Results should be different with mask
        assert not torch.allclose(prefix_code, prefix_code_masked, atol=1e-6)
    
    def test_prefix_length_handling(self, device):
        """Test prefix router with different sequence lengths."""
        router = PrefixRouter(
            input_dim=768,
            code_dim=8,
            max_prefix_tokens=64
        ).to(device)
        
        # Test with shorter sequence (should use entire sequence)
        short_seq = torch.randn(2, 32, 768, device=device)
        prefix_code_short = router(short_seq)
        assert prefix_code_short.shape == (2, 8)
        
        # Test with longer sequence (should truncate to max_prefix_tokens)
        long_seq = torch.randn(2, 200, 768, device=device)
        prefix_code_long = router(long_seq)
        assert prefix_code_long.shape == (2, 8)
        
        # Both should be valid
        assert not torch.isnan(prefix_code_short).any()
        assert not torch.isnan(prefix_code_long).any()


class TestChunkRouter:
    """Test suite for chunk router component."""
    
    def test_initialization(self, device):
        """Test chunk router initialization."""
        router = ChunkRouter(
            input_dim=768,
            code_dim=8,
            chunk_size=32,
            ema_decay=0.95
        ).to(device)
        
        assert router.chunk_size == 32
        assert router.ema_decay == 0.95
        assert router.ema_initialized.item() == False
    
    def test_forward_pass(self, device, sample_hidden_states):
        """Test chunk router forward pass."""
        router = ChunkRouter(
            input_dim=768,
            code_dim=8,
            chunk_size=32
        ).to(device)
        
        prefix_summary = torch.randn(2, 768, device=device)
        
        # Test forward pass
        chunk_code, ema_code = router(
            sample_hidden_states,
            prefix_summary,
            chunk_start=0,
            training=True
        )
        
        assert chunk_code.shape == (2, 8)
        assert ema_code.shape == (2, 8)
        assert not torch.isnan(chunk_code).any()
        assert not torch.isnan(ema_code).any()
        
        # EMA should be initialized after first call
        assert router.ema_initialized.item() == True
    
    def test_ema_behavior(self, device, sample_hidden_states):
        """Test EMA smoothing behavior."""
        router = ChunkRouter(
            input_dim=768,
            code_dim=8,
            chunk_size=32,
            ema_decay=0.9
        ).to(device)
        
        prefix_summary = torch.randn(2, 768, device=device)
        
        # First call - initializes EMA
        chunk_code1, ema_code1 = router(
            sample_hidden_states, prefix_summary, training=True
        )
        
        # Second call - should update EMA
        chunk_code2, ema_code2 = router(
            sample_hidden_states, prefix_summary, training=True  
        )
        
        # EMA codes should be different but similar
        assert not torch.allclose(ema_code1, ema_code2, atol=1e-6)
        cosine_sim = F.cosine_similarity(ema_code1, ema_code2, dim=-1)
        assert cosine_sim.mean() > 0.5  # Should be reasonably similar
    
    def test_side_signals(self, device, sample_hidden_states):
        """Test side signal integration."""
        router = ChunkRouter(
            input_dim=768,
            code_dim=8,
            side_signal_dim=256
        ).to(device)
        
        prefix_summary = torch.randn(2, 768, device=device)
        side_signals = torch.randn(2, 256, device=device)
        
        # Without side signals
        chunk_code1, _ = router(sample_hidden_states, prefix_summary)
        
        # With side signals
        chunk_code2, _ = router(
            sample_hidden_states, prefix_summary, side_signals=side_signals
        )
        
        # Results should be different
        assert not torch.allclose(chunk_code1, chunk_code2, atol=1e-4)


class TestTokenRouter:
    """Test suite for token router component."""
    
    def test_initialization_and_forward(self, device, sample_hidden_states):
        """Test token router initialization and forward pass."""
        router = TokenRouter(
            input_dim=768,
            code_dim=8,
            hidden_dim=512
        ).to(device)
        
        prefix_summary = torch.randn(2, 768, device=device)
        
        token_codes = router(sample_hidden_states, prefix_summary)
        
        assert token_codes.shape == (2, 128, 8)  # [batch, seq_len, code_dim]
        assert not torch.isnan(token_codes).any()
    
    def test_sequence_length_variability(self, device):
        """Test token router with different sequence lengths."""
        router = TokenRouter(input_dim=768, code_dim=8).to(device)
        prefix_summary = torch.randn(3, 768, device=device)
        
        # Test different sequence lengths
        for seq_len in [16, 64, 256]:
            hidden_states = torch.randn(3, seq_len, 768, device=device)
            token_codes = router(hidden_states, prefix_summary)
            assert token_codes.shape == (3, seq_len, 8)


class TestUncertaintyHead:
    """Test suite for uncertainty head component."""
    
    def test_uncertainty_estimation(self, device):
        """Test uncertainty head functionality."""
        head = UncertaintyHead(
            input_dim=768,
            hidden_dim=256,
            temperature_init=1.0
        ).to(device)
        
        # Test with different input shapes
        features_2d = torch.randn(4, 768, device=device)
        uncertainty_2d = head(features_2d)
        assert uncertainty_2d.shape == (4, 1)
        assert (uncertainty_2d >= 0).all() and (uncertainty_2d <= 1).all()
        
        features_3d = torch.randn(4, 32, 768, device=device)  
        uncertainty_3d = head(features_3d)
        assert uncertainty_3d.shape == (4, 32, 1)
        assert (uncertainty_3d >= 0).all() and (uncertainty_3d <= 1).all()
    
    def test_temperature_learning(self, device):
        """Test learnable temperature parameter."""
        head = UncertaintyHead(input_dim=768, temperature_init=2.0).to(device)
        
        initial_temp = head.temperature.item()
        assert abs(initial_temp - 2.0) < 0.1
        
        # Temperature should be learnable
        assert head.log_temperature.requires_grad


class TestHierarchicalController:
    """Test suite for main hierarchical controller."""
    
    def test_initialization(self, device, mock_model_config):
        """Test controller initialization."""
        controller = create_hierarchical_controller(
            mock_model_config,
            {'rank': 8, 'chunk_size': 32}
        ).to(device)
        
        assert controller.input_dim == 768
        assert controller.code_dim == 8
        assert controller.chunk_size == 32
        assert hasattr(controller, 'prefix_router')
        assert hasattr(controller, 'chunk_router')
        assert hasattr(controller, 'token_router')
        assert hasattr(controller, 'uncertainty_head')
    
    def test_routing_levels(self, device, sample_hidden_states, sample_attention_mask):
        """Test different routing levels."""
        controller = HierarchicalController(
            input_dim=768,
            code_dim=8,
            chunk_size=32
        ).to(device)
        
        # Test prefix routing
        prefix_codes = controller(
            sample_hidden_states,
            sample_attention_mask,
            routing_level=RoutingLevel.PREFIX
        )
        assert prefix_codes.shape == (2, 8)
        
        # Test chunk routing
        chunk_codes = controller(
            sample_hidden_states,
            sample_attention_mask, 
            routing_level=RoutingLevel.CHUNK
        )
        assert chunk_codes.shape == (2, 8)
        
        # Test token routing
        token_codes = controller(
            sample_hidden_states,
            sample_attention_mask,
            routing_level=RoutingLevel.TOKEN
        )
        assert token_codes.shape == (2, 128, 8)
    
    def test_routing_state_return(self, device, sample_hidden_states):
        """Test routing state information return."""
        controller = HierarchicalController(
            input_dim=768,
            code_dim=8,
            enable_uncertainty=True
        ).to(device)
        
        codes, routing_state = controller(
            sample_hidden_states,
            routing_level=RoutingLevel.CHUNK,
            return_routing_state=True
        )
        
        assert codes.shape == (2, 8)
        assert routing_state.prefix_code is not None
        assert routing_state.chunk_code is not None
        assert routing_state.ema_chunk_code is not None
        assert routing_state.uncertainty is not None
        assert routing_state.entropy is not None
        assert routing_state.utilization is not None
    
    def test_code_clamping(self, device):
        """Test code norm clamping functionality."""
        controller = HierarchicalController(
            input_dim=768,
            code_dim=8,
            code_clamp_value=2.0
        ).to(device)
        
        # Create input that might produce large codes
        large_input = torch.randn(2, 64, 768, device=device) * 10
        
        codes = controller(large_input, routing_level=RoutingLevel.PREFIX)
        
        # Check that norms are clamped
        norms = codes.norm(dim=-1)
        assert (norms <= 2.1).all()  # Allow small numerical error


class TestHierarchicalBEMModule:
    """Test suite for hierarchical BEM module."""
    
    def test_bem_module_creation(self, device, hierarchical_config):
        """Test BEM module creation and initialization."""
        base_layer = nn.Linear(768, 2048).to(device)
        
        bem_module = HierarchicalBEMModule(
            base_layer=base_layer,
            config=hierarchical_config,
            layer_name='test_layer',
            attach_point='mlp'
        )
        
        assert bem_module.in_features == 768
        assert bem_module.out_features == 2048
        assert bem_module.config.rank == 8
        assert bem_module.layer_name == 'test_layer'
        assert bem_module.attach_point == 'mlp'
        
        # Check LoRA parameters
        assert bem_module.lora_U.shape == (2048, 8)
        assert bem_module.lora_V.shape == (768, 8)
        assert bem_module.lora_U.requires_grad
        assert bem_module.lora_V.requires_grad
        
        # Check base layer is frozen
        assert not bem_module.base_layer.weight.requires_grad
    
    def test_forward_pass(self, device, hierarchical_config):
        """Test BEM module forward pass."""
        base_layer = nn.Linear(768, 2048).to(device)
        bem_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'test', 'mlp'
        ).to(device)
        
        controller = HierarchicalController(
            input_dim=768, code_dim=8, chunk_size=32
        ).to(device)
        
        # Test 2D input
        x_2d = torch.randn(4, 768, device=device)
        hidden_states = torch.randn(4, 64, 768, device=device)
        
        output_2d = bem_module(x_2d, hidden_states, controller)
        assert output_2d.shape == (4, 2048)
        assert not torch.isnan(output_2d).any()
        
        # Test 3D input  
        x_3d = torch.randn(4, 32, 768, device=device)
        output_3d = bem_module(x_3d, hidden_states, controller)
        assert output_3d.shape == (4, 32, 2048)
        assert not torch.isnan(output_3d).any()
    
    def test_routing_level_determination(self, device, hierarchical_config):
        """Test automatic routing level determination."""
        base_layer = nn.Linear(768, 2048).to(device)
        
        # MLP attach point
        mlp_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'mlp', 'mlp'
        ).to(device)
        
        routing_level = mlp_module.get_routing_level(seq_len=128)
        assert routing_level == RoutingLevel.TOKEN  # Should use token for short sequences
        
        routing_level_long = mlp_module.get_routing_level(seq_len=1024)  
        assert routing_level_long == RoutingLevel.CHUNK  # Should use chunk for long sequences
        
        # Attention attach point (Q/K/V)
        attn_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'attn', 'q'
        ).to(device)
        
        routing_level_attn = attn_module.get_routing_level(seq_len=128)
        assert routing_level_attn == RoutingLevel.CHUNK  # Always chunk for Q/K/V
    
    def test_routing_info_return(self, device, hierarchical_config):
        """Test routing information return."""
        base_layer = nn.Linear(768, 2048).to(device)
        bem_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'test', 'mlp'
        ).to(device)
        
        controller = HierarchicalController(
            input_dim=768, code_dim=8
        ).to(device)
        
        x = torch.randn(2, 32, 768, device=device)
        hidden_states = torch.randn(2, 64, 768, device=device)
        
        output, routing_info = bem_module(
            x, hidden_states, controller, return_routing_info=True
        )
        
        assert 'routing_level' in routing_info
        assert 'routing_state' in routing_info
        assert 'code_norm' in routing_info
        assert 'lora_norm' in routing_info
        assert 'layer_name' in routing_info
        assert 'attach_point' in routing_info


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_inference_speed(self, device, hierarchical_config):
        """Benchmark inference speed vs baseline."""
        base_layer = nn.Linear(768, 2048).to(device)
        bem_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'test', 'mlp'
        ).to(device)
        
        controller = HierarchicalController(
            input_dim=768, code_dim=8
        ).to(device)
        
        # Prepare inputs
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, 768, device=device)
        hidden_states = torch.randn(batch_size, seq_len, 768, device=device)
        
        # Warm up
        for _ in range(10):
            _ = bem_module(x, hidden_states, controller)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark BEM
        start_time = time.time()
        num_runs = 100
        
        for _ in range(num_runs):
            output = bem_module(x, hidden_states, controller)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        bem_time = time.time() - start_time
        
        # Benchmark baseline (base layer only)
        start_time = time.time()
        
        for _ in range(num_runs):
            baseline_output = base_layer(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        baseline_time = time.time() - start_time
        
        # Calculate overhead
        overhead = (bem_time - baseline_time) / baseline_time
        
        print(f"BEM time: {bem_time:.4f}s")
        print(f"Baseline time: {baseline_time:.4f}s") 
        print(f"Overhead: {overhead:.2%}")
        
        # Should be reasonable overhead (less than 50% for this test)
        assert overhead < 0.5, f"Overhead too high: {overhead:.2%}"
    
    def test_memory_usage(self, device, hierarchical_config):
        """Test memory usage vs baseline."""
        if device.type != 'cuda':
            pytest.skip("CUDA required for memory testing")
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Baseline memory
        baseline_mem_start = torch.cuda.memory_allocated()
        
        base_layer = nn.Linear(768, 2048).to(device)
        baseline_mem_after = torch.cuda.memory_allocated()
        baseline_params_mem = baseline_mem_after - baseline_mem_start
        
        # BEM memory
        bem_mem_start = torch.cuda.memory_allocated()
        
        bem_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'test', 'mlp'
        ).to(device)
        
        controller = HierarchicalController(
            input_dim=768, code_dim=8
        ).to(device)
        
        bem_mem_after = torch.cuda.memory_allocated()
        bem_total_mem = bem_mem_after - baseline_mem_start
        
        # Calculate additional memory for BEM
        bem_additional_mem = bem_total_mem - baseline_params_mem
        
        print(f"Baseline parameters: {baseline_params_mem / 1024**2:.2f} MB")
        print(f"BEM additional memory: {bem_additional_mem / 1024**2:.2f} MB")
        print(f"Total BEM memory: {bem_total_mem / 1024**2:.2f} MB")
        
        # BEM should add reasonable amount of memory
        mem_ratio = bem_additional_mem / baseline_params_mem
        assert mem_ratio < 0.5, f"BEM adds too much memory: {mem_ratio:.2f}x baseline"
    
    def test_throughput_scaling(self, device, hierarchical_config):
        """Test throughput scaling with batch size."""
        base_layer = nn.Linear(768, 2048).to(device)
        bem_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'test', 'mlp'
        ).to(device)
        
        controller = HierarchicalController(
            input_dim=768, code_dim=8
        ).to(device)
        
        seq_len = 128
        throughputs = []
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, seq_len, 768, device=device)
            hidden_states = torch.randn(batch_size, seq_len, 768, device=device)
            
            # Warm up
            for _ in range(5):
                _ = bem_module(x, hidden_states, controller)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Time multiple runs
            start_time = time.time()
            num_runs = 50
            
            for _ in range(num_runs):
                _ = bem_module(x, hidden_states, controller)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            total_time = time.time() - start_time
            total_tokens = num_runs * batch_size * seq_len
            throughput = total_tokens / total_time
            
            throughputs.append(throughput)
            print(f"Batch {batch_size}: {throughput:.2f} tokens/sec")
        
        # Throughput should generally increase with batch size
        assert throughputs[-1] > throughputs[0], "Throughput should scale with batch size"


class TestIntegrationWithTinyLlama:
    """Integration tests with TinyLlama-like architecture."""
    
    def create_mock_tinyllama_layer(self, device):
        """Create a mock TinyLlama layer for testing."""
        class MockTinyLlamaLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(768, 3072),
                    nn.GELU(),
                    nn.Linear(3072, 768)
                )
                self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
                self.norm1 = nn.LayerNorm(768)
                self.norm2 = nn.LayerNorm(768)
            
            def forward(self, x, attention_mask=None):
                # Self-attention
                attn_out, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
                x = self.norm1(x + attn_out)
                
                # MLP
                mlp_out = self.mlp(x)
                x = self.norm2(x + mlp_out)
                
                return x
        
        return MockTinyLlamaLayer().to(device)
    
    def test_tinyllama_integration(self, device, hierarchical_config):
        """Test integration with TinyLlama-like architecture."""
        # Create mock model
        mock_layer = self.create_mock_tinyllama_layer(device)
        
        # Create hierarchical BEM for MLP
        base_mlp_layer = mock_layer.mlp[0]  # First linear layer
        bem_module = HierarchicalBEMModule(
            base_mlp_layer, hierarchical_config, 'mlp.0', 'mlp'
        ).to(device)
        
        controller = HierarchicalController(
            input_dim=768, code_dim=8
        ).to(device)
        
        # Test forward pass
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, 768, device=device)
        
        # Original forward pass
        original_output = mock_layer(x)
        
        # Replace MLP layer with BEM
        mock_layer.mlp[0] = bem_module
        
        # BEM forward pass (need to provide hidden states and controller)
        def bem_forward(x):
            hidden_states = x  # Use input as hidden states for this test
            return bem_module(x, hidden_states, controller)
        
        mock_layer.mlp[0].forward = lambda x: bem_forward(x)
        
        # Forward pass with BEM
        bem_output = mock_layer(x)
        
        # Outputs should have same shape
        assert bem_output.shape == original_output.shape
        assert not torch.isnan(bem_output).any()
        
        # Outputs should be different (BEM is adapting)
        assert not torch.allclose(bem_output, original_output, atol=1e-4)


class TestTrainingIntegration:
    """Test training integration and stability."""
    
    def test_gradient_flow(self, device, hierarchical_config):
        """Test that gradients flow correctly through the system."""
        base_layer = nn.Linear(768, 2048).to(device)
        bem_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'test', 'mlp'
        ).to(device)
        
        controller = HierarchicalController(
            input_dim=768, code_dim=8
        ).to(device)
        
        # Create dummy target
        x = torch.randn(2, 32, 768, device=device, requires_grad=True)
        hidden_states = torch.randn(2, 64, 768, device=device)
        target = torch.randn(2, 32, 2048, device=device)
        
        # Forward pass
        output = bem_module(x, hidden_states, controller)
        
        # Compute loss
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for BEM parameters
        assert bem_module.lora_U.grad is not None
        assert bem_module.lora_V.grad is not None
        assert not torch.isnan(bem_module.lora_U.grad).any()
        assert not torch.isnan(bem_module.lora_V.grad).any()
        
        # Check controller gradients
        for param in controller.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
        
        # Check that base layer remains frozen
        assert base_layer.weight.grad is None
        assert base_layer.bias.grad is None
    
    def test_training_step(self, device, hierarchical_config, mock_model_config):
        """Test a complete training step."""
        # Create minimal setup
        base_layer = nn.Linear(768, 2048).to(device)
        bem_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'test', 'mlp'
        ).to(device)
        
        # Mock the full hierarchical BEM  
        class MockFullBEM(nn.Module):
            def __init__(self):
                super().__init__()
                self.controller = HierarchicalController(
                    input_dim=768, code_dim=8
                ).to(device)
                self.bem_modules = nn.ModuleDict({'test': bem_module})
                self.config = hierarchical_config
            
            def forward(self, input_ids, **kwargs):
                # Simplified forward for testing
                batch_size, seq_len = input_ids.shape
                hidden_states = torch.randn(batch_size, seq_len, 768, device=device)
                x = torch.randn(batch_size, seq_len, 768, device=device)
                
                output = bem_module(x, hidden_states, self.controller)
                
                # Mock logits output
                logits = torch.randn(batch_size, seq_len, 32000, device=device)
                
                return type('MockOutput', (), {'logits': logits})()
            
            def get_bem_parameters(self):
                return [bem_module.lora_U, bem_module.lora_V]
            
            def get_routing_statistics(self):
                return {
                    'total_bem_modules': 1,
                    'global_stats': {'routing_distribution': [0.33, 0.33, 0.34]}
                }
        
        mock_model = MockFullBEM()
        
        # Create trainer
        training_config = HierarchicalTrainingConfig(
            learning_rate=1e-3,
            max_steps=10,
            batch_size=2,
            log_interval=1
        )
        
        trainer = HierarchicalBEMTrainer(
            model=mock_model,
            config=training_config,
            device=device
        )
        
        # Create mock batch
        batch = {
            'input_ids': torch.randint(0, 32000, (2, 64), device=device),
            'labels': torch.randint(0, 32000, (2, 64), device=device),
            'attention_mask': torch.ones(2, 64, device=device)
        }
        
        # Perform training step
        loss_dict = trainer.train_step(batch)
        
        # Check that loss is computed
        assert 'total_loss' in loss_dict
        assert 'lm_loss' in loss_dict
        assert not np.isnan(loss_dict['total_loss'])
        assert loss_dict['total_loss'] > 0
        
        print(f"Training step completed. Total loss: {loss_dict['total_loss']:.4f}")


class TestTelemetryIntegration:
    """Test telemetry system integration."""
    
    def test_telemetry_collection(self, device, hierarchical_config, mock_model_config):
        """Test telemetry collection during BEM operation."""
        # Create simple BEM setup
        base_layer = nn.Linear(768, 2048).to(device)
        bem_module = HierarchicalBEMModule(
            base_layer, hierarchical_config, 'test', 'mlp'
        ).to(device)
        
        controller = HierarchicalController(
            input_dim=768, code_dim=8
        ).to(device)
        
        # Mock full BEM model
        class MockBEMForTelemetry(nn.Module):
            def __init__(self):
                super().__init__()
                self.controller = controller
                self.bem_modules = nn.ModuleDict({'test': bem_module})
                self.config = hierarchical_config
            
            def get_routing_statistics(self):
                return {
                    'total_bem_modules': 1,
                    'global_stats': {
                        'routing_distribution': [0.4, 0.4, 0.2],
                        'total_forward_calls': 10
                    },
                    'layers': {
                        'test': {
                            'total_calls': 10,
                            'routing_counts': [4, 4, 2]
                        }
                    }
                }
        
        mock_model = MockBEMForTelemetry()
        
        # Create telemetry collector
        collector = create_telemetry_collector(
            model=mock_model,
            collection_interval=1,  # Collect every step
            history_length=100
        )
        
        # Simulate operation with telemetry
        with collector.timing_context('forward_pass'):
            x = torch.randn(2, 32, 768, device=device)
            hidden_states = torch.randn(2, 64, 768, device=device)
            
            with collector.timing_context('bem_computation'):
                output, routing_info = bem_module(
                    x, hidden_states, controller, return_routing_info=True
                )
        
        # Update telemetry
        collector.step_update(routing_info={'routing_state': routing_info})
        
        # Check that metrics were collected
        current_metrics = collector.get_current_metrics()
        
        assert 'performance' in current_metrics
        assert 'system' in current_metrics
        assert current_metrics['performance']['forward_time'] > 0
        
        # Check performance summary
        summary = collector.get_performance_summary(last_n_steps=1)
        assert 'forward_time' in summary
        assert 'throughput' in summary
        
        print(f"Telemetry collection test passed. Forward time: {current_metrics['performance']['forward_time']:.4f}s")


# Utility functions for running tests

def run_performance_benchmarks(device, hierarchical_config):
    """Run comprehensive performance benchmarks."""
    print(f"\nRunning performance benchmarks on {device}...")
    
    benchmark_tests = TestPerformanceBenchmarks()
    
    print("\n1. Inference Speed Test:")
    benchmark_tests.test_inference_speed(device, hierarchical_config)
    
    if device.type == 'cuda':
        print("\n2. Memory Usage Test:")
        benchmark_tests.test_memory_usage(device, hierarchical_config)
    
    print("\n3. Throughput Scaling Test:")
    benchmark_tests.test_throughput_scaling(device, hierarchical_config)


def run_integration_tests(device, hierarchical_config):
    """Run integration tests."""
    print(f"\nRunning integration tests on {device}...")
    
    integration_tests = TestIntegrationWithTinyLlama()
    
    print("1. TinyLlama Integration Test:")
    integration_tests.test_tinyllama_integration(device, hierarchical_config)
    
    training_tests = TestTrainingIntegration()
    
    print("2. Gradient Flow Test:")
    training_tests.test_gradient_flow(device, hierarchical_config)
    
    print("3. Training Step Test:")
    mock_model_config = {'hidden_size': 768, 'vocab_size': 32000}
    training_tests.test_training_step(device, hierarchical_config, mock_model_config)


if __name__ == "__main__":
    """Run all tests when script is executed directly."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = HierarchicalBEMConfig()
    
    print(f"Running hierarchical BEM tests on {device}")
    print("=" * 60)
    
    try:
        # Run core component tests
        print("Testing individual components...")
        
        # Can also run with pytest for more detailed output:
        # pytest -v test_hierarchical_bem.py::TestHierarchicalController
        
        # Run performance benchmarks
        run_performance_benchmarks(device, config)
        
        # Run integration tests
        run_integration_tests(device, config)
        
        print("\n" + "=" * 60)
        print("All tests completed successfully! ✅")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise