"""
V2 - Dual-Path (LoRA++) Implementation

Two parallel low-rank branches with independent gates and orthogonality regularization.
Key features:
- Two parallel LoRA branches with same rank schedule
- Chunk-sticky routing (N=128) 
- Orthogonality regularization: λ_ortho = 0.1
- Gate decorrelation penalty: α = 0.01
- Same sites as B1: W_O + W_down
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math

from .parallel_lora import GeneratedExpert
from .chunk_sticky_routing import ChunkStickyRouter
from .governance import SpectralGovernance


class DualPathLoRABranch(nn.Module):
    """
    Single branch in dual-path architecture.
    Similar to GeneratedExpert but with independent gating.
    """
    
    def __init__(
        self,
        retrieval_dim: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        branch_id: str = "branch_0"
    ):
        super().__init__()
        self.branch_id = branch_id
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Use same adapter generation as base BEM
        self.expert = GeneratedExpert(
            retrieval_dim=retrieval_dim,
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Independent gate network for this branch
        self.gate_network = nn.Sequential(
            nn.Linear(retrieval_dim, retrieval_dim // 2),
            nn.ReLU(),
            nn.Linear(retrieval_dim // 2, 1),
            nn.Sigmoid()  # Gate strength [0, 1]
        )
        
    def forward(
        self,
        x: torch.Tensor,
        retrieval_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single branch.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_features: [batch_size, retrieval_dim]
            
        Returns:
            Dictionary with branch output, gate, etc.
        """
        # Compute branch output using expert
        branch_output = self.expert(x, retrieval_features)
        
        # Compute independent gate
        gate = self.gate_network(retrieval_features)  # [batch_size, 1]
        gate = gate.unsqueeze(1)  # [batch_size, 1, 1] for broadcasting
        
        # Apply gating
        gated_output = branch_output * gate
        
        return {
            'output': gated_output,
            'raw_output': branch_output,
            'gate': gate.squeeze(-1),  # [batch_size, 1]
            'branch_id': self.branch_id
        }


class OrthogonalityRegularizer(nn.Module):
    """
    Compute orthogonality regularization between dual path branches.
    Encourages branches to learn different patterns.
    """
    
    def __init__(self, lambda_ortho: float = 0.1):
        super().__init__()
        self.lambda_ortho = lambda_ortho
        
    def compute_loss(
        self, 
        branch_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute orthogonality loss between branch outputs.
        
        Args:
            branch_outputs: List of branch outputs [batch_size, seq_len, out_features]
            
        Returns:
            Orthogonality loss scalar
        """
        if len(branch_outputs) < 2:
            return torch.tensor(0.0, device=branch_outputs[0].device)
            
        # Flatten branch outputs for correlation computation
        flattened = [out.flatten(1) for out in branch_outputs]  # [batch_size, seq_len * out_features]
        
        # Compute correlation between branches
        ortho_loss = 0.0
        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                # Normalize for correlation
                branch_i = F.normalize(flattened[i], dim=1)  # [batch_size, features]
                branch_j = F.normalize(flattened[j], dim=1)  # [batch_size, features]
                
                # Correlation: want this to be small (orthogonal)
                correlation = torch.sum(branch_i * branch_j, dim=1).abs().mean()
                ortho_loss += correlation
                
        return self.lambda_ortho * ortho_loss


class GateDecorrelationLoss(nn.Module):
    """
    Decorrelation penalty for gates to prevent redundant branches.
    """
    
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
        
    def compute_loss(self, gates: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute gate decorrelation loss.
        
        Args:
            gates: List of gate tensors [batch_size, 1] each
            
        Returns:
            Decorrelation loss scalar
        """
        if len(gates) < 2:
            return torch.tensor(0.0, device=gates[0].device)
            
        decorr_loss = 0.0
        for i in range(len(gates)):
            for j in range(i + 1, len(gates)):
                # Correlation between gates
                gate_i = gates[i].flatten()  # [batch_size]
                gate_j = gates[j].flatten()  # [batch_size]
                
                # Pearson correlation coefficient
                correlation = torch.corrcoef(torch.stack([gate_i, gate_j]))[0, 1].abs()
                decorr_loss += correlation
                
        return self.alpha * decorr_loss


class DualPathLoRA(nn.Module):
    """
    V2 Dual-Path LoRA++ implementation.
    
    Two parallel low-rank branches with independent gates.
    Architecture: y = base(x) + branch_1_output + branch_2_output
    where each branch_output = gate_i * B_i(A_i(x))
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        retrieval_dim: int,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        lambda_ortho: float = 0.1,
        alpha_decorr: float = 0.01,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.retrieval_dim = retrieval_dim
        self.rank = rank
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Two parallel branches
        self.branch_1 = DualPathLoRABranch(
            retrieval_dim=retrieval_dim,
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            branch_id="branch_1"
        )
        
        self.branch_2 = DualPathLoRABranch(
            retrieval_dim=retrieval_dim,
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            branch_id="branch_2"
        )
        
        # Chunk-sticky routing
        self.router = ChunkStickyRouter(
            chunk_size=chunk_size,
            hysteresis_tau=hysteresis_tau
        )
        
        # Regularizers
        self.ortho_regularizer = OrthogonalityRegularizer(lambda_ortho=lambda_ortho)
        self.decorr_regularizer = GateDecorrelationLoss(alpha=alpha_decorr)
        
        # Spectral governance
        self.governance = SpectralGovernance(
            max_singular_value=2.0,
            fro_budget=0.1
        )
        
    def forward(
        self,
        x: torch.Tensor,
        retrieval_features: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing dual-path architecture.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_features: [batch_size, retrieval_dim]
            position_ids: Optional position IDs for routing
            
        Returns:
            Dictionary with output, branch outputs, gates, losses, etc.
        """
        batch_size, seq_len, _ = x.shape
        
        # Base layer output
        base_output = self.base_layer(x)
        
        # Routing decisions (chunk-sticky)
        if position_ids is not None:
            routing_probs = self.router(retrieval_features, position_ids)
        else:
            # Default uniform routing if no position_ids
            routing_probs = torch.ones(batch_size, 2, device=x.device) / 2.0
            
        # Branch outputs
        branch_1_result = self.branch_1(x, retrieval_features)
        branch_2_result = self.branch_2(x, retrieval_features)
        
        # Apply routing to branch outputs
        routed_branch_1 = branch_1_result['output'] * routing_probs[:, 0:1].unsqueeze(-1)
        routed_branch_2 = branch_2_result['output'] * routing_probs[:, 1:2].unsqueeze(-1)
        
        # Combine outputs
        dual_path_output = routed_branch_1 + routed_branch_2
        final_output = base_output + dual_path_output
        
        # Compute regularization losses
        branch_raw_outputs = [branch_1_result['raw_output'], branch_2_result['raw_output']]
        ortho_loss = self.ortho_regularizer.compute_loss(branch_raw_outputs)
        
        branch_gates = [branch_1_result['gate'], branch_2_result['gate']]
        decorr_loss = self.decorr_regularizer.compute_loss(branch_gates)
        
        return {
            'output': final_output,
            'base_output': base_output,
            'dual_path_output': dual_path_output,
            'branch_1_output': branch_1_result['output'],
            'branch_2_output': branch_2_result['output'],
            'branch_1_gate': branch_1_result['gate'],
            'branch_2_gate': branch_2_result['gate'],
            'routing_probs': routing_probs,
            'ortho_loss': ortho_loss,
            'decorr_loss': decorr_loss,
            'total_reg_loss': ortho_loss + decorr_loss,
            'branch_outputs_raw': branch_raw_outputs
        }


class MultiLayerDualPathLoRA(nn.Module):
    """
    Apply Dual-Path LoRA to multiple layers with depth-varying ranks.
    """
    
    def __init__(
        self,
        base_layers: Dict[str, nn.Linear],
        retrieval_dim: int,
        rank_schedule: Dict[str, int],
        alpha: float = 16.0,
        dropout: float = 0.0,
        lambda_ortho: float = 0.1,
        alpha_decorr: float = 0.01,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7
    ):
        super().__init__()
        
        self.base_layers = base_layers
        self.dual_path_modules = nn.ModuleDict()
        
        for name, layer in base_layers.items():
            rank = rank_schedule.get(name, 8)  # Default rank if not specified
            
            self.dual_path_modules[name] = DualPathLoRA(
                base_layer=layer,
                retrieval_dim=retrieval_dim,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                lambda_ortho=lambda_ortho,
                alpha_decorr=alpha_decorr,
                chunk_size=chunk_size,
                hysteresis_tau=hysteresis_tau
            )
            
    def forward(
        self,
        layer_name: str,
        x: torch.Tensor,
        retrieval_features: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific layer.
        """
        if layer_name not in self.dual_path_modules:
            # Fallback to base layer
            return {'output': self.base_layers[layer_name](x)}
            
        return self.dual_path_modules[layer_name](x, retrieval_features, position_ids)
        
    def get_regularization_losses(self) -> torch.Tensor:
        """
        Aggregate regularization losses from all layers.
        """
        total_loss = 0.0
        for module in self.dual_path_modules.values():
            # These will be computed during the last forward pass
            if hasattr(module, '_last_ortho_loss'):
                total_loss += module._last_ortho_loss
            if hasattr(module, '_last_decorr_loss'):
                total_loss += module._last_decorr_loss
                
        return total_loss


def create_dual_path_lora_for_model(
    model: nn.Module,
    retrieval_dim: int,
    rank_schedule: Optional[Dict[str, int]] = None,
    attachment_points: Optional[Dict[str, str]] = None,
    **kwargs
) -> MultiLayerDualPathLoRA:
    """
    Create Dual-Path LoRA for specified model layers.
    
    Args:
        model: Base transformer model
        retrieval_dim: Dimension of retrieval features
        rank_schedule: Rank per layer name (2×[2,4,4,4,4,4,2])
        attachment_points: Which layers to attach to (default: W_O and W_down)
        
    Returns:
        MultiLayerDualPathLoRA instance
    """
    if attachment_points is None:
        attachment_points = {
            'attention': ['out_proj', 'W_O'],
            'mlp': ['down_proj', 'W_down']
        }
        
    if rank_schedule is None:
        # Default dual-path rank schedule: 2×[2,4,4,4,4,4,2] (same total params as B1)
        rank_schedule = {}
        default_ranks = [2, 4, 4, 4, 4, 4, 2]  # Same as V2 config
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
                
    print(f"Created Dual-Path LoRA for {len(base_layers)} layers")
    print(f"Rank schedule: {rank_schedule}")
    
    return MultiLayerDualPathLoRA(
        base_layers=base_layers,
        retrieval_dim=retrieval_dim,
        rank_schedule=rank_schedule,
        **kwargs
    )