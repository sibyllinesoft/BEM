"""
Generated Parallel Low-Rank (E1) Implementation

The key innovation: adapters are generated dynamically from retrieval context
rather than being static learned parameters.

Architecture: y = base(x) + Σ_e g_e · B_e (A_e x)
where A_e, B_e are generated from retrieval features z.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class AdapterGenerator(nn.Module):
    """
    Generates LoRA adapter matrices A_e and B_e from retrieval features.
    This is the core innovation - dynamic generation rather than static parameters.
    """
    
    def __init__(
        self,
        retrieval_dim: int,
        in_features: int,
        out_features: int,
        rank: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.retrieval_dim = retrieval_dim
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Generator networks for A and B matrices
        self.A_generator = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features * rank)
        )
        
        self.B_generator = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rank * out_features)
        )
        
        # Initialize to small values
        self._init_weights()
        
    def _init_weights(self):
        """Initialize generator weights to produce small LoRA matrices."""
        for module in [self.A_generator, self.B_generator]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)
        
        # Final layers produce very small values initially
        nn.init.normal_(self.A_generator[-1].weight, std=0.001)
        nn.init.normal_(self.B_generator[-1].weight, std=0.001)
    
    def forward(self, retrieval_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate LoRA A and B matrices from retrieval features.
        
        Args:
            retrieval_features: [batch_size, retrieval_dim]
            
        Returns:
            A: [batch_size, in_features, rank] 
            B: [batch_size, rank, out_features]
        """
        batch_size = retrieval_features.shape[0]
        
        # Generate A matrix [in_features, rank]
        A_flat = self.A_generator(retrieval_features)  # [batch_size, in_features * rank]
        A = A_flat.view(batch_size, self.in_features, self.rank)
        
        # Generate B matrix [rank, out_features] 
        B_flat = self.B_generator(retrieval_features)  # [batch_size, rank * out_features]
        B = B_flat.view(batch_size, self.rank, self.out_features)
        
        return A, B


class GeneratedExpert(nn.Module):
    """
    Single expert that generates its LoRA matrices dynamically.
    Implements B_e (A_e x) where A_e, B_e are generated from context.
    """
    
    def __init__(
        self,
        retrieval_dim: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Adapter generator
        self.adapter_gen = AdapterGenerator(
            retrieval_dim=retrieval_dim,
            in_features=in_features,
            out_features=out_features,
            rank=rank
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        retrieval_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: B_e (A_e x) with generated matrices.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_features: [batch_size, retrieval_dim]
            
        Returns:
            Expert output [batch_size, seq_len, out_features]
        """
        batch_size, seq_len, in_features = x.shape
        
        # Generate A and B matrices from retrieval context
        A, B = self.adapter_gen(retrieval_features)  # [batch, in, rank], [batch, rank, out]
        
        # Apply A matrix: A_e x
        # x: [batch, seq, in] @ A: [batch, in, rank] -> [batch, seq, rank]
        Ax = torch.bmm(x, A)
        Ax = self.dropout(Ax)
        
        # Apply B matrix: B_e (A_e x)  
        # Ax: [batch, seq, rank] @ B: [batch, rank, out] -> [batch, seq, out]
        BAx = torch.bmm(Ax, B)
        
        return BAx * self.scaling


class GeneratedParallelLoRA(nn.Module):
    """
    Generated Parallel LoRA implementation (E1).
    
    Key innovation: Multiple experts with generated (not learned) LoRA matrices
    based on retrieval context. Architecture: y = base(x) + Σ_e g_e · B_e (A_e x)
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        retrieval_dim: int,
        rank: int,
        num_experts: int = 2,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.retrieval_dim = retrieval_dim
        self.rank = rank
        self.num_experts = num_experts
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Generated experts
        self.experts = nn.ModuleList([
            GeneratedExpert(
                retrieval_dim=retrieval_dim,
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
        # Expert gate network
        self.gate_network = nn.Sequential(
            nn.Linear(retrieval_dim, retrieval_dim // 2),
            nn.ReLU(),
            nn.Linear(retrieval_dim // 2, num_experts)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        retrieval_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing y = base(x) + Σ_e g_e · B_e (A_e x).
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_features: [batch_size, retrieval_dim]
            
        Returns:
            Dictionary with output, expert_outputs, gates, etc.
        """
        batch_size, seq_len, _ = x.shape
        
        # Base layer output
        base_output = self.base_layer(x)
        
        # Compute expert gates g_e
        gate_logits = self.gate_network(retrieval_features)  # [batch_size, num_experts]
        gates = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
        
        # Compute expert outputs B_e (A_e x)
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x, retrieval_features)  # [batch_size, seq_len, out_features]
            expert_outputs.append(expert_out)
            
        # Stack expert outputs
        expert_stack = torch.stack(expert_outputs, dim=0)  # [num_experts, batch, seq, out]
        
        # Apply gating: Σ_e g_e · B_e (A_e x)
        gates_expanded = gates.T.unsqueeze(-1).unsqueeze(-1)  # [num_experts, batch, 1, 1]
        gated_experts = expert_stack * gates_expanded  # Broadcasting
        combined_expert_output = gated_experts.sum(dim=0)  # [batch, seq, out]
        
        # Final output: base + gated experts
        output = base_output + combined_expert_output
        
        return {
            'output': output,
            'base_output': base_output,
            'expert_outputs': expert_outputs,
            'gates': gates,
            'gate_logits': gate_logits,
            'combined_expert_output': combined_expert_output
        }


class MultiLayerGeneratedLoRA(nn.Module):
    """
    Applies Generated Parallel LoRA to multiple layers with depth-varying ranks.
    """
    
    def __init__(
        self,
        base_layers: Dict[str, nn.Linear],
        retrieval_dim: int,
        rank_schedule: Dict[str, int],
        num_experts: int = 2,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.base_layers = base_layers
        self.lora_modules = nn.ModuleDict()
        
        for name, layer in base_layers.items():
            rank = rank_schedule.get(name, 8)  # Default rank if not specified
            
            self.lora_modules[name] = GeneratedParallelLoRA(
                base_layer=layer,
                retrieval_dim=retrieval_dim,
                rank=rank,
                num_experts=num_experts,
                alpha=alpha,
                dropout=dropout
            )
            
    def forward(
        self,
        layer_name: str,
        x: torch.Tensor,
        retrieval_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific layer.
        """
        if layer_name not in self.lora_modules:
            # Fallback to base layer
            return {'output': self.base_layers[layer_name](x)}
            
        return self.lora_modules[layer_name](x, retrieval_features)


def create_generated_lora_for_model(
    model: nn.Module,
    retrieval_dim: int,
    rank_schedule: Optional[Dict[str, int]] = None,
    attachment_points: Optional[Dict[str, str]] = None,
    **kwargs
) -> MultiLayerGeneratedLoRA:
    """
    Create Generated Parallel LoRA for specified model layers.
    
    Args:
        model: Base transformer model
        retrieval_dim: Dimension of retrieval features
        rank_schedule: Rank per layer name
        attachment_points: Which layers to attach to (default: W_O and W_down)
        
    Returns:
        MultiLayerGeneratedLoRA instance
    """
    if attachment_points is None:
        attachment_points = {
            'attention': ['out_proj', 'W_O'],
            'mlp': ['down_proj', 'W_down']
        }
        
    if rank_schedule is None:
        # Default depth-varying schedule from TODO.md
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
                
    print(f"Created Generated LoRA for {len(base_layers)} layers")
    print(f"Rank schedule: {rank_schedule}")
    
    return MultiLayerGeneratedLoRA(
        base_layers=base_layers,
        retrieval_dim=retrieval_dim,
        rank_schedule=rank_schedule,
        **kwargs
    )