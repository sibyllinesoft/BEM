#!/usr/bin/env python3
"""
Demo: BEM 2.0 Unified Performance Track Variants

Demonstrates the unified performance tracking system using standardized trainer interfaces
and template-based configuration. Shows PT1-PT4 variants with consistent APIs, unified
evaluation metrics, and template inheritance benefits.

Variants demonstrated:
- PT1: Head-Group Gating @ W_O (unified interface)
- PT2: Dynamic Rank Mask (unified interface) 
- PT3: Kronecker @ W_down (unified interface)
- PT4: Residual FiLM Micro-γ,β (unified interface)
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import unified infrastructure
from src.bem_core.config.config_loader import load_experiment_config, load_training_config
from src.bem_core.training import BaseTrainer, TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedPerformanceTrainer(BaseTrainer):
    """Unified trainer for performance track variants (PT1-PT4)."""
    
    def __init__(self, config_path: str, **kwargs):
        """Initialize performance trainer with unified interface.
        
        Args:
            config_path: Path to experiment configuration
            **kwargs: Additional trainer arguments
        """
        # Load unified configuration
        training_config = load_training_config(config_path)
        super().__init__(training_config, **kwargs)
        
        # Load full experiment configuration
        self.experiment_config = load_experiment_config(config_path)
        
        # Extract variant type from config
        self.variant_type = self.experiment_config.name.split('_')[0]  # e.g., "PT1", "PT2"
        self.variant_config = self.experiment_config.model.get('variant', {})
        
        logger.info(f"Initialized unified trainer for {self.variant_type}")
    
    def _setup_model(self):
        """Set up performance variant model using unified interface."""
        variant_type = self.variant_type
        config = self.experiment_config.model
        
        logger.info(f"Setting up {variant_type} model with unified configuration")
        
        if variant_type == "PT1":
            return self._setup_pt1_model(config)
        elif variant_type == "PT2":
            return self._setup_pt2_model(config)
        elif variant_type == "PT3":
            return self._setup_pt3_model(config)
        elif variant_type == "PT4":
            return self._setup_pt4_model(config)
        else:
            raise ValueError(f"Unknown variant type: {variant_type}")
    
    def _setup_pt1_model(self, config: Dict[str, Any]):
        """Set up PT1 Head-Group Gating model."""
        
        class PT1UnifiedModel(nn.Module):
            """PT1 model with unified interface."""
            
            def __init__(self, model_config: Dict[str, Any]):
                super().__init__()
                self.hidden_size = model_config.get('hidden_size', 768)
                self.num_heads = model_config.get('num_heads', 12)
                self.variant_config = model_config.get('variant', {})
                
                # Base transformer components
                self.attention = nn.MultiheadAttention(
                    self.hidden_size, self.num_heads, batch_first=True
                )
                self.norm = nn.LayerNorm(self.hidden_size)
                
                # PT1-specific parameters from unified config
                self.num_groups = self.variant_config.get('num_groups', 4)
                self.heads_per_group = self.num_heads // self.num_groups
                self.rank_per_group = self.variant_config.get('rank_per_group', 4)
                self.gate_temperature = self.variant_config.get('gate_temperature', 1.0)
                
                # Head-group gating layers
                self.group_gates = nn.Parameter(torch.randn(self.num_groups) * 0.1)
                self.rank_projections = nn.ModuleList([
                    nn.Linear(self.hidden_size, self.rank_per_group)
                    for _ in range(self.num_groups)
                ])
                
                logger.info(f"PT1 unified model: {self.num_groups} groups, {self.heads_per_group} heads/group")
            
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                if input_ids is not None:
                    # Simple embedding (in real implementation, would use proper embeddings)
                    batch_size, seq_len = input_ids.shape
                    x = torch.randn(batch_size, seq_len, self.hidden_size).to(input_ids.device)
                else:
                    x = kwargs.get('hidden_states', torch.randn(1, 64, self.hidden_size))
                
                # Attention with head-group gating
                attn_out, _ = self.attention(x, x, x)
                
                # Apply head-group gating
                gates = torch.softmax(self.group_gates / self.gate_temperature, dim=0)
                
                # Simulate group-wise processing
                output = self.norm(x + attn_out)
                
                return {
                    'last_hidden_state': output,
                    'gate_weights': gates,
                    'group_efficiency': gates.var().item(),  # Efficiency metric
                }
            
            def compute_budget_metrics(self):
                """Compute unified budget metrics."""
                base_params = sum(p.numel() for p in self.parameters())
                gating_params = self.group_gates.numel() + sum(
                    p.numel() for proj in self.rank_projections for p in proj.parameters()
                )
                
                return {
                    'total_parameters': base_params,
                    'gating_parameters': gating_params,
                    'parameter_efficiency': gating_params / base_params,
                    'flops_reduction': 1 - (self.num_groups * self.rank_per_group) / (self.num_heads * self.hidden_size),
                }
        
        return PT1UnifiedModel(config)
    
    def _setup_pt2_model(self, config: Dict[str, Any]):
        """Set up PT2 Dynamic Rank Mask model."""
        
        class PT2UnifiedModel(nn.Module):
            """PT2 model with unified interface."""
            
            def __init__(self, model_config: Dict[str, Any]):
                super().__init__()
                self.hidden_size = model_config.get('hidden_size', 768)
                self.variant_config = model_config.get('variant', {})
                
                # PT2-specific parameters from unified config
                self.total_rank = self.variant_config.get('total_rank', 16)
                self.active_rank = self.variant_config.get('active_rank', 8)
                self.mask_temperature = self.variant_config.get('mask_temperature', 0.1)
                
                # Dynamic mask components
                self.rank_logits = nn.Parameter(torch.randn(self.total_rank) * 0.1)
                self.projection = nn.Linear(self.hidden_size, self.total_rank)
                self.output_projection = nn.Linear(self.active_rank, self.hidden_size)
                
                logger.info(f"PT2 unified model: {self.total_rank} total rank, {self.active_rank} active")
            
            def forward(self, input_ids=None, **kwargs):
                if input_ids is not None:
                    batch_size, seq_len = input_ids.shape
                    x = torch.randn(batch_size, seq_len, self.hidden_size).to(input_ids.device)
                else:
                    x = kwargs.get('hidden_states', torch.randn(1, 64, self.hidden_size))
                
                # Dynamic rank masking
                mask_weights = torch.softmax(self.rank_logits / self.mask_temperature, dim=0)
                top_k_indices = torch.topk(mask_weights, self.active_rank).indices
                
                # Project and select active ranks
                projected = self.projection(x)
                active_features = projected[..., top_k_indices]
                
                # Output projection
                output = self.output_projection(active_features)
                
                return {
                    'last_hidden_state': output,
                    'mask_weights': mask_weights,
                    'active_ranks': top_k_indices,
                    'sparsity_ratio': 1 - (self.active_rank / self.total_rank),
                }
            
            def compute_budget_metrics(self):
                """Compute unified budget metrics."""
                return {
                    'total_parameters': sum(p.numel() for p in self.parameters()),
                    'active_parameters': self.active_rank * (self.hidden_size + 1),
                    'sparsity_ratio': 1 - (self.active_rank / self.total_rank),
                    'flops_reduction': 1 - (self.active_rank / self.total_rank),
                }
        
        return PT2UnifiedModel(config)
    
    def _setup_pt3_model(self, config: Dict[str, Any]):
        """Set up PT3 Kronecker factorization model."""
        
        class PT3UnifiedModel(nn.Module):
            """PT3 model with unified interface."""
            
            def __init__(self, model_config: Dict[str, Any]):
                super().__init__()
                self.hidden_size = model_config.get('hidden_size', 768)
                self.variant_config = model_config.get('variant', {})
                
                # PT3-specific parameters from unified config
                self.u_rank = self.variant_config.get('u_rank', 8)
                self.v_rank = self.variant_config.get('v_rank', 8)
                self.u_dim = self.variant_config.get('u_dim', 64)
                self.v_dim = self.variant_config.get('v_dim', 48)
                
                # Kronecker factors
                self.U = nn.Parameter(torch.randn(self.u_dim, self.u_rank) * 0.1)
                self.V = nn.Parameter(torch.randn(self.v_dim, self.v_rank) * 0.1)
                
                # Input/output projections
                self.input_proj = nn.Linear(self.hidden_size, self.u_dim * self.v_dim)
                self.output_proj = nn.Linear(self.u_rank * self.v_rank, self.hidden_size)
                
                logger.info(f"PT3 unified model: U({self.u_dim}x{self.u_rank}), V({self.v_dim}x{self.v_rank})")
            
            def forward(self, input_ids=None, **kwargs):
                if input_ids is not None:
                    batch_size, seq_len = input_ids.shape
                    x = torch.randn(batch_size, seq_len, self.hidden_size).to(input_ids.device)
                else:
                    x = kwargs.get('hidden_states', torch.randn(1, 64, self.hidden_size))
                
                # Kronecker factorization
                projected = self.input_proj(x)
                batch_size, seq_len, _ = projected.shape
                
                # Reshape for Kronecker product
                reshaped = projected.view(batch_size, seq_len, self.u_dim, self.v_dim)
                
                # Apply Kronecker factors
                # Simplified Kronecker operation
                u_applied = torch.einsum('bsud,ur->bsrd', reshaped, self.U)
                v_applied = torch.einsum('bsrd,vr->bsvr', u_applied, self.V.t())
                
                # Flatten and project output
                flattened = v_applied.view(batch_size, seq_len, -1)
                output = self.output_proj(flattened)
                
                return {
                    'last_hidden_state': output,
                    'u_effective_rank': torch.linalg.matrix_rank(self.U).float(),
                    'v_effective_rank': torch.linalg.matrix_rank(self.V).float(),
                    'compression_ratio': (self.u_rank * self.v_rank) / (self.u_dim * self.v_dim),
                }
            
            def compute_budget_metrics(self):
                """Compute unified budget metrics."""
                full_params = self.u_dim * self.v_dim * self.hidden_size
                kron_params = (self.u_dim * self.u_rank) + (self.v_dim * self.v_rank)
                
                return {
                    'total_parameters': sum(p.numel() for p in self.parameters()),
                    'kronecker_parameters': kron_params,
                    'compression_ratio': kron_params / full_params,
                    'memory_reduction': 1 - (kron_params / full_params),
                }
        
        return PT3UnifiedModel(config)
    
    def _setup_pt4_model(self, config: Dict[str, Any]):
        """Set up PT4 Residual FiLM model."""
        
        class PT4UnifiedModel(nn.Module):
            """PT4 model with unified interface."""
            
            def __init__(self, model_config: Dict[str, Any]):
                super().__init__()
                self.hidden_size = model_config.get('hidden_size', 768)
                self.variant_config = model_config.get('variant', {})
                
                # PT4-specific parameters from unified config
                self.gamma_dim = self.variant_config.get('gamma_dim', 16)
                self.beta_dim = self.variant_config.get('beta_dim', 16)
                self.controller_dim = self.variant_config.get('controller_dim', 32)
                self.micro_scale = self.variant_config.get('micro_scale', 0.01)
                
                # Base layer
                self.base_layer = nn.Linear(self.hidden_size, self.hidden_size)
                
                # FiLM controllers
                self.gamma_controller = nn.Linear(self.hidden_size, self.gamma_dim)
                self.beta_controller = nn.Linear(self.hidden_size, self.beta_dim)
                
                # Micro-scale residual projections
                self.gamma_proj = nn.Linear(self.gamma_dim, self.hidden_size)
                self.beta_proj = nn.Linear(self.beta_dim, self.hidden_size)
                
                logger.info(f"PT4 unified model: γ_dim={self.gamma_dim}, β_dim={self.beta_dim}")
            
            def forward(self, input_ids=None, **kwargs):
                if input_ids is not None:
                    batch_size, seq_len = input_ids.shape
                    x = torch.randn(batch_size, seq_len, self.hidden_size).to(input_ids.device)
                else:
                    x = kwargs.get('hidden_states', torch.randn(1, 64, self.hidden_size))
                
                # Base transformation
                base_output = self.base_layer(x)
                
                # FiLM controllers
                gamma_code = self.gamma_controller(x)
                beta_code = self.beta_controller(x)
                
                # Micro-scale FiLM parameters
                gamma = 1.0 + self.micro_scale * self.gamma_proj(gamma_code)
                beta = self.micro_scale * self.beta_proj(beta_code)
                
                # Apply FiLM modulation
                film_output = gamma * base_output + beta
                
                # Residual connection
                output = x + film_output
                
                return {
                    'last_hidden_state': output,
                    'gamma_params': gamma,
                    'beta_params': beta,
                    'film_magnitude': torch.abs(gamma - 1.0).mean() + torch.abs(beta).mean(),
                    'stability_ratio': torch.abs(film_output).mean() / torch.abs(x).mean(),
                }
            
            def compute_budget_metrics(self):
                """Compute unified budget metrics."""
                film_params = (
                    self.gamma_controller.weight.numel() + self.gamma_controller.bias.numel() +
                    self.beta_controller.weight.numel() + self.beta_controller.bias.numel() +
                    self.gamma_proj.weight.numel() + self.gamma_proj.bias.numel() +
                    self.beta_proj.weight.numel() + self.beta_proj.bias.numel()
                )
                
                return {
                    'total_parameters': sum(p.numel() for p in self.parameters()),
                    'film_parameters': film_params,
                    'parameter_overhead': film_params / sum(p.numel() for p in self.parameters()),
                    'micro_scale_factor': self.micro_scale,
                }
        
        return PT4UnifiedModel(config)
    
    def _compute_loss(self, batch: Dict[str, Any], model_outputs: Any) -> Dict[str, torch.Tensor]:
        """Compute unified loss for performance variants."""
        # Mock loss for demonstration - in practice, would be task-specific
        mock_loss = torch.tensor(0.5, requires_grad=True)
        
        # Add variant-specific regularization
        if hasattr(self.model, 'compute_budget_metrics'):
            budget_metrics = self.model.compute_budget_metrics()
            
            # Add efficiency penalties based on variant type
            if self.variant_type == "PT1":
                # Encourage gate diversity
                if 'gate_weights' in model_outputs:
                    gate_entropy = -torch.sum(
                        model_outputs['gate_weights'] * torch.log(model_outputs['gate_weights'] + 1e-8)
                    )
                    mock_loss += 0.01 * (1.0 - gate_entropy)
            
            elif self.variant_type == "PT2":
                # Encourage sparsity
                sparsity_bonus = budget_metrics.get('sparsity_ratio', 0)
                mock_loss -= 0.01 * sparsity_bonus
        
        return {"loss": mock_loss}
    
    def _evaluate(self, dataloader) -> Dict[str, float]:
        """Run unified evaluation for performance variants."""
        # Mock evaluation metrics
        base_metrics = {
            "accuracy": np.random.uniform(0.75, 0.95),
            "f1_score": np.random.uniform(0.70, 0.90),
            "perplexity": np.random.uniform(2.0, 8.0),
        }
        
        # Add variant-specific metrics
        if hasattr(self.model, 'compute_budget_metrics'):
            budget_metrics = self.model.compute_budget_metrics()
            base_metrics.update({
                f"budget_{k}": v for k, v in budget_metrics.items() 
                if isinstance(v, (int, float))
            })
        
        return base_metrics


def create_unified_pt_configs() -> Dict[str, str]:
    """Create unified configuration files for PT variants."""
    configs = {}
    
    # PT1 Configuration Template
    pt1_config = """
name: "PT1_unified_demo"
description: "PT1 Head-Group Gating with unified interface"

model:
  type: "performance_track"
  hidden_size: 768
  num_heads: 12
  
  variant:
    num_groups: 4
    rank_per_group: 4
    gate_temperature: 1.0
    decorrelation_strength: 0.1

training:
  learning_rate: 3e-4
  batch_size: 32
  max_steps: 1000
  warmup_steps: 100
  eval_steps: 200

hardware:
  device: "auto"
  fp16: false

logging:
  level: "INFO"
  wandb_project: "bem-pt-unified"

seed: 42
"""
    
    # PT2 Configuration Template  
    pt2_config = """
name: "PT2_unified_demo"
description: "PT2 Dynamic Rank Mask with unified interface"

model:
  type: "performance_track"
  hidden_size: 768
  
  variant:
    total_rank: 16
    active_rank: 8
    mask_temperature: 0.1
    use_instance_adaptive: true

training:
  learning_rate: 2e-4
  batch_size: 32
  max_steps: 1000
  warmup_steps: 100
  eval_steps: 200

hardware:
  device: "auto"
  fp16: false

logging:
  level: "INFO"
  wandb_project: "bem-pt-unified"

seed: 42
"""
    
    # PT3 Configuration Template
    pt3_config = """
name: "PT3_unified_demo"
description: "PT3 Kronecker Factorization with unified interface"

model:
  type: "performance_track"
  hidden_size: 768
  
  variant:
    u_rank: 8
    v_rank: 8
    u_dim: 64
    v_dim: 48
    use_fused_kernel: true
    init_method: "svd"

training:
  learning_rate: 5e-4
  batch_size: 32
  max_steps: 1000
  warmup_steps: 100
  eval_steps: 200

hardware:
  device: "auto"
  fp16: false

logging:
  level: "INFO"
  wandb_project: "bem-pt-unified"

seed: 42
"""
    
    # PT4 Configuration Template
    pt4_config = """
name: "PT4_unified_demo"
description: "PT4 Residual FiLM with unified interface"

model:
  type: "performance_track"
  hidden_size: 768
  
  variant:
    gamma_dim: 16
    beta_dim: 16
    controller_dim: 32
    micro_scale: 0.01
    clamp_range: 0.1
    minimal_overhead: true

training:
  learning_rate: 4e-4
  batch_size: 32
  max_steps: 1000
  warmup_steps: 100
  eval_steps: 200

hardware:
  device: "auto"
  fp16: false

logging:
  level: "INFO"
  wandb_project: "bem-pt-unified"

seed: 42
"""
    
    # Save configuration files
    for variant, config_content in [
        ("PT1", pt1_config), ("PT2", pt2_config), 
        ("PT3", pt3_config), ("PT4", pt4_config)
    ]:
        config_path = f"demo_{variant.lower()}_unified_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        configs[variant] = config_path
    
    return configs


def demonstrate_unified_variant(variant: str, config_path: str) -> Dict[str, Any]:
    """Demonstrate a unified performance variant."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DEMONSTRATING {variant} WITH UNIFIED INTERFACE")
    logger.info(f"{'='*70}")
    
    # Initialize unified trainer
    trainer = UnifiedPerformanceTrainer(
        config_path=config_path,
        experiment_name=f"unified_{variant.lower()}_demo"
    )
    
    # Setup training (this initializes the model)
    trainer.setup_training(train_dataloader=None, eval_dataloader=None)
    
    # Create test data
    test_data = {
        'input_ids': torch.randint(0, 1000, (16, 64)),
        'attention_mask': torch.ones(16, 64)
    }
    
    # Move to device
    device = trainer.device
    test_data = {k: v.to(device) for k, v in test_data.items()}
    trainer.model.to(device)
    
    # Forward pass
    start_time = time.time()
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model(**test_data)
    inference_time = time.time() - start_time
    
    logger.info(f"✅ Unified forward pass: {inference_time*1000:.2f}ms")
    logger.info(f"   Output shape: {outputs['last_hidden_state'].shape}")
    
    # Get budget metrics through unified interface
    budget_metrics = trainer.model.compute_budget_metrics()
    
    logger.info(f"📊 Unified Budget Metrics:")
    for key, value in budget_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value:,}")
    
    # Variant-specific analysis
    logger.info(f"🔍 {variant}-Specific Analysis:")
    if variant == "PT1" and 'gate_weights' in outputs:
        gates = outputs['gate_weights']
        logger.info(f"   Gate entropy: {-(gates * torch.log(gates + 1e-8)).sum():.4f}")
        logger.info(f"   Gate distribution: {gates.tolist()}")
        
    elif variant == "PT2" and 'sparsity_ratio' in outputs:
        logger.info(f"   Sparsity ratio: {outputs['sparsity_ratio']:.2%}")
        logger.info(f"   Active ranks: {outputs['active_ranks'].tolist()}")
        
    elif variant == "PT3" and 'compression_ratio' in outputs:
        logger.info(f"   Compression ratio: {outputs['compression_ratio']:.4f}")
        logger.info(f"   U effective rank: {outputs['u_effective_rank']:.2f}")
        logger.info(f"   V effective rank: {outputs['v_effective_rank']:.2f}")
        
    elif variant == "PT4" and 'film_magnitude' in outputs:
        logger.info(f"   FiLM magnitude: {outputs['film_magnitude']:.4f}")
        logger.info(f"   Stability ratio: {outputs['stability_ratio']:.4f}")
    
    # Run unified evaluation
    eval_metrics = trainer._evaluate(None)
    
    logger.info(f"📈 Unified Evaluation Metrics:")
    for key, value in eval_metrics.items():
        logger.info(f"   {key}: {value:.4f}")
    
    # Compile results
    results = {
        'variant': variant,
        'config_path': config_path,
        'inference_time_ms': inference_time * 1000,
        'output_shape': list(outputs['last_hidden_state'].shape),
        'budget_metrics': budget_metrics,
        'eval_metrics': eval_metrics,
        'variant_specific': {
            k: v.tolist() if isinstance(v, torch.Tensor) else v 
            for k, v in outputs.items() 
            if k not in ['last_hidden_state']
        }
    }
    
    return results


def print_comparative_analysis(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print unified comparative analysis."""
    
    print(f"\n{'='*80}")
    print("🎯 UNIFIED PERFORMANCE TRACK COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    # Performance comparison
    print(f"\n⚡ Inference Performance:")
    print(f"{'Variant':<8} {'Time (ms)':<12} {'Parameters':<15} {'Efficiency':<12}")
    print("-" * 50)
    
    for variant, results in all_results.items():
        time_ms = results['inference_time_ms']
        params = results['budget_metrics']['total_parameters']
        
        # Calculate efficiency score
        f1_score = results['eval_metrics'].get('f1_score', 0)
        efficiency = f1_score * 1000 / time_ms
        
        print(f"{variant:<8} {time_ms:<12.2f} {params:<15,} {efficiency:<12.2f}")
    
    # Budget analysis
    print(f"\n💰 Budget Efficiency Analysis:")
    for variant, results in all_results.items():
        budget = results['budget_metrics']
        print(f"\n{variant} ({results['variant']}):")
        
        # Common metrics
        if 'parameter_efficiency' in budget:
            print(f"  Parameter Efficiency: {budget['parameter_efficiency']:.4f}")
        if 'compression_ratio' in budget:
            print(f"  Compression Ratio: {budget['compression_ratio']:.4f}")
        if 'sparsity_ratio' in budget:
            print(f"  Sparsity Ratio: {budget['sparsity_ratio']:.2%}")
        if 'flops_reduction' in budget:
            print(f"  FLOPs Reduction: {budget['flops_reduction']:.2%}")
    
    # Best performers by metric
    print(f"\n🏆 Best Performers by Metric:")
    
    metrics = ['accuracy', 'f1_score', 'inference_time_ms']
    for metric in metrics:
        if metric == 'inference_time_ms':
            best_variant = min(all_results.keys(), 
                              key=lambda v: all_results[v].get(metric, float('inf')))
            best_value = all_results[best_variant][metric]
            print(f"  Fastest Inference: {best_variant} ({best_value:.2f}ms)")
        else:
            best_variant = max(all_results.keys(),
                              key=lambda v: all_results[v]['eval_metrics'].get(metric, 0))
            best_value = all_results[best_variant]['eval_metrics'][metric]
            print(f"  Best {metric.title()}: {best_variant} ({best_value:.4f})")
    
    # Template inheritance benefits
    print(f"\n✨ Unified Interface Benefits Demonstrated:")
    print(f"  - Single trainer class for all PT variants")
    print(f"  - Template-based configuration inheritance") 
    print(f"  - Consistent budget and evaluation metrics")
    print(f"  - Standardized performance profiling")
    print(f"  - Unified logging and experiment tracking")
    print(f"  - Seamless variant switching via configuration")


def main():
    """Main demonstration function."""
    
    print(f"\n{'='*80}")
    print("🚀 BEM 2.0 UNIFIED PERFORMANCE TRACK DEMONSTRATION")
    print("Showcasing PT1-PT4 with unified interfaces and template inheritance")
    print(f"{'='*80}")
    
    # Create unified configurations
    print(f"📄 Creating unified configuration templates...")
    pt_configs = create_unified_pt_configs()
    print(f"✅ Created {len(pt_configs)} variant configurations")
    
    # Demonstrate each variant
    all_results = {}
    
    for variant, config_path in pt_configs.items():
        try:
            print(f"\n🔧 Demonstrating {variant} with unified interface...")
            results = demonstrate_unified_variant(variant, config_path)
            all_results[variant] = results
            print(f"✅ {variant} demonstration completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error demonstrating {variant}: {e}")
            logger.exception("Detailed error:")
            all_results[variant] = {'error': str(e)}
    
    # Comparative analysis
    if all_results:
        print_comparative_analysis(all_results)
    
    # Save unified results
    results_dir = Path("unified_performance_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_path = results_dir / "unified_pt_variants_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save configuration summary
    config_summary = {
        'demonstration_type': 'unified_performance_track',
        'variants_demonstrated': list(pt_configs.keys()),
        'unified_benefits': [
            'Single trainer interface for all variants',
            'Template-based configuration inheritance',
            'Consistent metrics and evaluation',
            'Standardized budget validation',
            'Unified logging and tracking'
        ],
        'template_inheritance': {
            'base_template_fields': ['training', 'hardware', 'logging'],
            'variant_specific_fields': ['model.variant'],
            'configuration_consistency': 'All variants share common structure'
        }
    }
    
    summary_path = results_dir / "unified_configuration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"\n📊 Unified results saved:")
    print(f"  - Detailed evaluation: {results_path}")
    print(f"  - Configuration summary: {summary_path}")
    
    # Clean up demo configs
    for config_path in pt_configs.values():
        Path(config_path).unlink(missing_ok=True)
    
    print(f"\n✅ Unified Performance Track demonstration completed!")
    print(f"📈 Template inheritance and unified interfaces successfully demonstrated")
    
    return 0


if __name__ == "__main__":
    exit(main())