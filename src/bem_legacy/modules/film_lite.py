"""
V7 - FiLM-lite Extension Implementation

Based on B1 architecture with FiLM-style modulation: y = γ(z) ⊙ y + β(z)
Applied to MLP output only with tiny MLP on retrieval features for γ, β generation.
Same base architecture and routing as B1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

from .parallel_lora import GeneratedParallelLoRA
from .chunk_sticky_routing import ChunkStickyRouter
from .attention_bias import MultiScaleAttentionBias as AttentionBias
from .governance import SpectralGovernance


class FiLMConditioner(nn.Module):
    """
    FiLM-style conditioning module that generates γ(z) and β(z) from retrieval features.
    Applied as: output = γ(z) ⊙ input + β(z)
    """
    
    def __init__(
        self,
        retrieval_dim: int,
        target_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.retrieval_dim = retrieval_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        
        # Tiny MLP for γ (scale) generation
        self.gamma_network = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
            nn.Sigmoid()  # Scale factors [0, 2] centered around 1
        )
        
        # Tiny MLP for β (shift) generation  
        self.beta_network = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
            nn.Tanh()  # Shift factors [-1, 1]
        )
        
        # Initialize to identity transformation initially
        self._init_identity()
        
    def _init_identity(self):
        """Initialize to approximate identity transformation."""
        # Gamma network should output ~1.0 initially
        with torch.no_grad():
            # Last layer of gamma network should output logits for sigmoid(x) ≈ 1
            nn.init.normal_(self.gamma_network[-1].weight, std=0.01)
            nn.init.constant_(self.gamma_network[-1].bias, 0.0)
            
            # Beta network should output ~0.0 initially
            nn.init.normal_(self.beta_network[-1].weight, std=0.01) 
            nn.init.zeros_(self.beta_network[-1].bias)
            
    def forward(self, retrieval_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate FiLM parameters from retrieval features.
        
        Args:
            retrieval_features: [batch_size, retrieval_dim]
            
        Returns:
            gamma: Scale parameters [batch_size, target_dim]
            beta: Shift parameters [batch_size, target_dim]
        """
        # Generate scale parameters γ(z) ∈ [0, 2]
        gamma_logits = self.gamma_network(retrieval_features)
        gamma = 2.0 * gamma_logits  # Scale sigmoid [0,1] to [0,2]
        
        # Generate shift parameters β(z) ∈ [-1, 1]
        beta = self.beta_network(retrieval_features)
        
        return gamma, beta
        
    def apply_conditioning(
        self, 
        input_tensor: torch.Tensor,
        retrieval_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply FiLM conditioning: output = γ(z) ⊙ input + β(z)
        
        Args:
            input_tensor: [batch_size, seq_len, target_dim]
            retrieval_features: [batch_size, retrieval_dim]
            
        Returns:
            Dictionary with conditioned output and FiLM parameters
        """
        batch_size, seq_len, target_dim = input_tensor.shape
        
        # Generate FiLM parameters
        gamma, beta = self.forward(retrieval_features)  # [batch_size, target_dim]
        
        # Expand for sequence dimension
        gamma_expanded = gamma.unsqueeze(1)  # [batch_size, 1, target_dim]
        beta_expanded = beta.unsqueeze(1)    # [batch_size, 1, target_dim]
        
        # Apply FiLM transformation: γ ⊙ input + β
        conditioned_output = gamma_expanded * input_tensor + beta_expanded
        
        return {
            'output': conditioned_output,
            'gamma': gamma,
            'beta': beta,
            'input': input_tensor
        }


class FiLMEnhancedBEMLayer(nn.Module):
    """
    BEM layer enhanced with FiLM conditioning on MLP output.
    Based on B1 architecture with added FiLM modulation.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        retrieval_dim: int,
        rank: int,
        num_experts: int = 2,
        alpha: float = 16.0,
        dropout: float = 0.0,
        film_feature_dim: int = 64,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7,
        enable_attention_bias: bool = True
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.retrieval_dim = retrieval_dim
        self.rank = rank
        self.enable_attention_bias = enable_attention_bias
        
        # Base BEM-v1.1 components (same as B1)
        self.bem_lora = GeneratedParallelLoRA(
            base_layer=base_layer,
            retrieval_dim=retrieval_dim,
            rank=rank,
            num_experts=num_experts,
            alpha=alpha,
            dropout=dropout
        )
        
        # Chunk-sticky routing (same as B1)
        self.router = ChunkStickyRouter(
            chunk_size=chunk_size,
            hysteresis_tau=hysteresis_tau
        )
        
        # Attention bias (same as B1, for applicable layers)
        if enable_attention_bias:
            self.attention_bias = AttentionBias(
                retrieval_dim=retrieval_dim,
                bias_type='logit'
            )
        else:
            self.attention_bias = None
            
        # FiLM conditioning on MLP output
        self.film_conditioner = FiLMConditioner(
            retrieval_dim=retrieval_dim,
            target_dim=base_layer.out_features,
            hidden_dim=film_feature_dim
        )
        
        # Spectral governance (same as B1)
        self.governance = SpectralGovernance(
            max_singular_value=2.0,
            fro_budget=0.1
        )
        
    def forward(
        self,
        x: torch.Tensor,
        retrieval_features: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with FiLM-enhanced BEM.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_features: [batch_size, retrieval_dim]
            position_ids: Optional position IDs for routing
            attention_scores: Optional attention scores for bias (if attention layer)
            
        Returns:
            Dictionary with output, FiLM parameters, BEM outputs, etc.
        """
        # Base BEM forward pass (same as B1)
        bem_result = self.bem_lora(x, retrieval_features)
        bem_output = bem_result['output']  # base(x) + expert_outputs
        
        # Apply attention bias if this is an attention layer
        if self.attention_bias is not None and attention_scores is not None:
            bias_result = self.attention_bias(attention_scores, retrieval_features)
            # Note: attention bias would be applied in the attention mechanism itself
            attention_bias = bias_result['bias']
        else:
            attention_bias = None
            
        # Apply FiLM conditioning to the BEM output
        # This is the key V7 innovation: γ(z) ⊙ bem_output + β(z)
        film_result = self.film_conditioner.apply_conditioning(
            bem_output, retrieval_features
        )
        
        final_output = film_result['output']
        
        return {
            'output': final_output,
            'bem_output': bem_output,
            'base_output': bem_result['base_output'],
            'expert_outputs': bem_result['expert_outputs'],
            'gates': bem_result['gates'],
            'film_gamma': film_result['gamma'],
            'film_beta': film_result['beta'],
            'attention_bias': attention_bias,
            'film_conditioned': True
        }


class FiLMLiteBEM(nn.Module):
    """
    V7 FiLM-lite BEM implementation.
    
    Based on B1 (BEM-v1.1-stable) architecture with FiLM-style modulation
    applied to MLP outputs. Same base architecture and routing as B1.
    """
    
    def __init__(
        self,
        base_layers: Dict[str, nn.Linear],
        retrieval_dim: int,
        rank_schedule: Dict[str, int],
        num_experts: int = 2,
        alpha: float = 16.0,
        dropout: float = 0.0,
        film_feature_dim: int = 64,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7
    ):
        super().__init__()
        
        self.base_layers = base_layers
        self.film_bem_modules = nn.ModuleDict()
        
        for name, layer in base_layers.items():
            rank = rank_schedule.get(name, 8)  # Default rank if not specified
            
            # Determine if this layer should have attention bias
            is_attention_layer = any(pattern in name for pattern in ['out_proj', 'W_O'])
            
            self.film_bem_modules[name] = FiLMEnhancedBEMLayer(
                base_layer=layer,
                retrieval_dim=retrieval_dim,
                rank=rank,
                num_experts=num_experts,
                alpha=alpha,
                dropout=dropout,
                film_feature_dim=film_feature_dim,
                chunk_size=chunk_size,
                hysteresis_tau=hysteresis_tau,
                enable_attention_bias=is_attention_layer
            )
            
    def forward(
        self,
        layer_name: str,
        x: torch.Tensor,
        retrieval_features: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific layer.
        """
        if layer_name not in self.film_bem_modules:
            # Fallback to base layer
            return {'output': self.base_layers[layer_name](x)}
            
        return self.film_bem_modules[layer_name](
            x, retrieval_features, position_ids, attention_scores
        )
        
    def get_film_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get FiLM parameters from all layers (for analysis/visualization).
        """
        film_params = {}
        for layer_name, module in self.film_bem_modules.items():
            if hasattr(module, '_last_film_gamma') and hasattr(module, '_last_film_beta'):
                film_params[layer_name] = {
                    'gamma': module._last_film_gamma,
                    'beta': module._last_film_beta
                }
        return film_params


def create_film_lite_bem_for_model(
    model: nn.Module,
    retrieval_dim: int,
    rank_schedule: Optional[Dict[str, int]] = None,
    attachment_points: Optional[Dict[str, str]] = None,
    **kwargs
) -> FiLMLiteBEM:
    """
    Create FiLM-lite BEM for specified model layers.
    
    Args:
        model: Base transformer model
        retrieval_dim: Dimension of retrieval features
        rank_schedule: Rank per layer name (same as B1: [2,4,8,8,8,4,2])
        attachment_points: Which layers to attach to (default: W_O and W_down)
        
    Returns:
        FiLMLiteBEM instance
    """
    if attachment_points is None:
        attachment_points = {
            'attention': ['out_proj', 'W_O'],
            'mlp': ['down_proj', 'W_down']
        }
        
    if rank_schedule is None:
        # Same rank schedule as B1
        rank_schedule = {}
        default_ranks = [2, 4, 8, 8, 8, 4, 2]
        layer_idx = 0
        
    # Find layers to attach to
    base_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer should have LoRA attached
            should_attach = False
            for category, patterns in attachment_points.items():
                if any(pattern in name for pattern in patterns):
                    should_attach = True
                    break
                    
            if should_attach:
                # Determine rank for this layer
                if name in rank_schedule:
                    rank = rank_schedule[name]
                else:
                    rank = default_ranks[min(layer_idx, len(default_ranks) - 1)]
                    rank_schedule[name] = rank
                    
                base_layers[name] = module
                layer_idx += 1
                
    print(f"Created FiLM-lite BEM for {len(base_layers)} layers")
    print(f"Rank schedule: {rank_schedule}")
    
    return FiLMLiteBEM(
        base_layers=base_layers,
        retrieval_dim=retrieval_dim,
        rank_schedule=rank_schedule,
        **kwargs
    )