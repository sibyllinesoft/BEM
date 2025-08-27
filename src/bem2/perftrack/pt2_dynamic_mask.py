"""
PT2 - Dynamic Rank Mask (Fixed FLOPs) Implementation.

Implements k-hot mask over rank components per block with ~50% sparsity,
using masked Hadamard path for efficient computation and instance-adaptive
rank selection.

Expected Performance: +≥1% primary metric improvement  
Budget Constraint: Fixed FLOPs, ±5% params vs v1.3-stack anchor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import logging

from ..security.parameter_protection import ParameterProtector
from bem.modules.governance import SpectralGovernance

logger = logging.getLogger(__name__)


@dataclass
class DynamicRankMaskConfig:
    """Configuration for PT2 Dynamic Rank Mask."""
    
    # Rank masking parameters
    total_rank: int = 16         # Total available rank
    active_rank: int = 8         # k in k-hot mask (~50% sparsity)
    mask_temperature: float = 0.1    # Gumbel softmax temperature
    
    # Hadamard path parameters
    use_hadamard_path: bool = True   # Use efficient Hadamard computation
    hadamard_dim: int = 16           # Must be power of 2 for fast Hadamard
    
    # Instance adaptation
    use_instance_adaptive: bool = True   # Enable instance-specific masking
    adaptation_dim: int = 32             # Dimension for adaptation network
    
    # Mask learning parameters
    mask_learning_rate: float = 1e-3    # Separate LR for mask parameters
    mask_momentum: float = 0.9           # Momentum for mask optimization
    straight_through: bool = True        # Use straight-through estimator
    
    # Trust region constraints  
    max_singular_value: float = 1.0     # Spectral clamp
    fro_budget: float = 0.8             # Conservative Frobenius budget
    
    # Budget constraints
    budget_tolerance: float = 0.05      # ±5% tolerance
    target_sparsity: float = 0.5        # Target ~50% sparsity


class FastHadamardTransform(nn.Module):
    """Fast Hadamard Transform for efficient masked computation."""
    
    def __init__(self, dim: int):
        super().__init__()
        assert dim > 0 and (dim & (dim - 1)) == 0, f"dim must be power of 2, got {dim}"
        self.dim = dim
        self.register_buffer('hadamard_matrix', self._generate_hadamard_matrix(dim))
    
    def _generate_hadamard_matrix(self, n: int) -> torch.Tensor:
        """Generate Hadamard matrix of size n x n."""
        if n == 1:
            return torch.tensor([[1.0]])
        
        # Recursive construction using Sylvester's construction
        h_half = self._generate_hadamard_matrix(n // 2)
        h_full = torch.zeros(n, n)
        
        h_full[:n//2, :n//2] = h_half
        h_full[:n//2, n//2:] = h_half  
        h_full[n//2:, :n//2] = h_half
        h_full[n//2:, n//2:] = -h_half
        
        return h_full / math.sqrt(n)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fast Hadamard transform."""
        # x: [..., dim]
        original_shape = x.shape
        x_flat = x.view(-1, self.dim)
        
        # Apply Hadamard transform
        transformed = torch.matmul(x_flat, self.hadamard_matrix.T)
        
        return transformed.view(original_shape)


class InstanceAdaptiveController(nn.Module):
    """Controller for instance-adaptive rank selection."""
    
    def __init__(self, config: DynamicRankMaskConfig, hidden_size: int):
        super().__init__()
        self.config = config
        
        # Adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(hidden_size, config.adaptation_dim),
            nn.LayerNorm(config.adaptation_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.adaptation_dim, config.total_rank)
        )
        
        # Learnable mask logits
        self.mask_logits = nn.Parameter(
            torch.randn(config.total_rank) * 0.1
        )
        
        # EMA statistics for stability
        self.register_buffer('mask_ema', torch.ones(config.total_rank) / config.total_rank)
        self.ema_momentum = 0.99
        
    def compute_instance_mask(
        self, 
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute instance-adaptive k-hot mask."""
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Instance-specific logits
        instance_logits = self.adaptation_net(hidden_states.mean(dim=1))  # [B, total_rank]
        
        # Combine with learned mask logits
        combined_logits = instance_logits + self.mask_logits.unsqueeze(0)
        
        if training:
            # Gumbel softmax for differentiable sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(combined_logits) + 1e-8) + 1e-8)
            logits_with_noise = (combined_logits + gumbel_noise) / self.config.mask_temperature
            
            # Convert to k-hot using top-k
            _, top_k_indices = torch.topk(logits_with_noise, self.config.active_rank, dim=-1)
            
            # Create k-hot mask
            mask = torch.zeros_like(combined_logits)
            mask.scatter_(-1, top_k_indices, 1.0)
            
            if self.config.straight_through:
                # Straight-through estimator
                soft_mask = F.softmax(combined_logits / self.config.mask_temperature, dim=-1)
                top_k_soft = torch.zeros_like(soft_mask)
                top_k_soft.scatter_(-1, top_k_indices, 1.0)
                
                # Straight-through: forward uses hard mask, backward uses soft
                mask = top_k_soft + soft_mask - soft_mask.detach()
            
            # Update EMA statistics
            batch_mask_mean = mask.mean(dim=0)
            self.mask_ema.mul_(self.ema_momentum).add_(
                batch_mask_mean, alpha=1 - self.ema_momentum
            )
        else:
            # Deterministic top-k for inference
            _, top_k_indices = torch.topk(combined_logits, self.config.active_rank, dim=-1)
            mask = torch.zeros_like(combined_logits)
            mask.scatter_(-1, top_k_indices, 1.0)
        
        # Compute metrics
        sparsity = 1.0 - mask.mean().item()
        entropy = -torch.sum(self.mask_ema * torch.log(self.mask_ema + 1e-8))
        
        metrics = {
            'mask': mask,
            'sparsity': sparsity,
            'mask_entropy': entropy,
            'instance_logits': instance_logits,
            'selected_indices': top_k_indices if not training or not self.config.straight_through else None
        }
        
        return mask, metrics


class SparseMaskController(nn.Module):
    """Main controller for sparse rank masking."""
    
    def __init__(self, config: DynamicRankMaskConfig, hidden_size: int):
        super().__init__()
        self.config = config
        
        # Instance adaptation controller
        self.instance_controller = InstanceAdaptiveController(config, hidden_size)
        
        # Fast Hadamard transform for efficient computation
        if config.use_hadamard_path:
            # Ensure hadamard_dim is compatible
            hadamard_dim = max(16, 2 ** math.ceil(math.log2(config.total_rank)))
            self.hadamard_transform = FastHadamardTransform(hadamard_dim)
            self.hadamard_dim = hadamard_dim
        else:
            self.hadamard_transform = None
            self.hadamard_dim = config.total_rank
        
        # Learnable scaling factors
        self.rank_scales = nn.Parameter(torch.ones(config.total_rank))
        
    def apply_sparse_mask(
        self,
        rank_activations: torch.Tensor,  # [B, S, total_rank]
        hidden_states: torch.Tensor      # [B, S, D] - for adaptation
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply sparse masking to rank activations."""
        
        # Get instance-adaptive mask
        mask, mask_metrics = self.instance_controller.compute_instance_mask(
            hidden_states, training=self.training
        )
        
        # Apply mask to rank activations
        masked_activations = rank_activations * mask.unsqueeze(1)  # [B, S, total_rank]
        
        # Apply learnable scaling
        scaled_activations = masked_activations * self.rank_scales.unsqueeze(0).unsqueeze(0)
        
        # Efficient computation via Hadamard path
        if self.config.use_hadamard_path and self.hadamard_transform is not None:
            # Pad to Hadamard dimension if needed
            if scaled_activations.shape[-1] < self.hadamard_dim:
                padding = torch.zeros(
                    *scaled_activations.shape[:-1], 
                    self.hadamard_dim - scaled_activations.shape[-1],
                    device=scaled_activations.device,
                    dtype=scaled_activations.dtype
                )
                padded_activations = torch.cat([scaled_activations, padding], dim=-1)
            else:
                padded_activations = scaled_activations[:, :, :self.hadamard_dim]
            
            # Apply Hadamard transform
            transformed_activations = self.hadamard_transform(padded_activations)
            
            # Take only the original dimensions
            final_activations = transformed_activations[:, :, :self.config.total_rank]
        else:
            final_activations = scaled_activations
        
        # Additional metrics
        rank_usage = mask.sum(dim=-1).float()  # [B]
        efficiency = rank_usage / self.config.total_rank
        
        all_metrics = {
            **mask_metrics,
            'rank_usage': rank_usage.mean().item(),
            'efficiency': efficiency.mean().item(),
            'rank_scales': self.rank_scales.detach().clone(),
            'masked_activations': masked_activations
        }
        
        return final_activations, all_metrics


class DynamicRankMaskModule(nn.Module):
    """
    PT2 Dynamic Rank Mask Module.
    
    Implements k-hot masking over rank components with ~50% sparsity,
    using masked Hadamard path for efficient computation.
    """
    
    def __init__(
        self,
        config: DynamicRankMaskConfig,
        layer_idx: int,
        hidden_size: int
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        
        # Core low-rank projections
        self.U = nn.Linear(hidden_size, config.total_rank, bias=False)
        self.V = nn.Linear(config.total_rank, hidden_size, bias=False)
        
        # Sparse mask controller
        self.mask_controller = SparseMaskController(config, hidden_size)
        
        # Spectral governance
        self.spectral_gov = SpectralGovernance(
            max_singular_value=config.max_singular_value,
            layer_specific=True
        )
        
        # Parameter protection
        self.param_protector = ParameterProtector(
            max_norm=config.max_singular_value,
            clip_gradient=True
        )
        
        # Budget tracking
        self.register_buffer('flop_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('param_count', torch.tensor(0, dtype=torch.long))
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize projection parameters with proper scaling."""
        # Orthogonal initialization for U
        nn.init.orthogonal_(self.U.weight)
        
        # Xavier initialization for V (scaled for sparsity)
        nn.init.xavier_uniform_(
            self.V.weight, 
            gain=math.sqrt(2.0 / self.config.target_sparsity)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, S, D]
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with dynamic rank masking."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Forward through U projection
        rank_activations = self.U(hidden_states)  # [B, S, total_rank]
        
        # Apply sparse masking
        masked_activations, mask_metrics = self.mask_controller.apply_sparse_mask(
            rank_activations, hidden_states
        )
        
        # Forward through V projection
        delta_w = self.V(masked_activations)  # [B, S, D]
        
        # Apply spectral governance
        delta_w_clamped, spectral_metrics = self.spectral_gov.apply_spectral_clamp(
            delta_w.view(-1, hidden_dim),
            layer_name=f"pt2_layer_{self.layer_idx}",
            target_sigma1=self.config.max_singular_value
        )
        delta_w_clamped = delta_w_clamped.view(batch_size, seq_len, hidden_dim)
        
        # Output with residual connection
        output = hidden_states + delta_w_clamped
        
        # Update FLOP counter
        current_flops = self._compute_flops(batch_size, seq_len, mask_metrics['sparsity'])
        self.flop_count.add_(current_flops)
        
        if output_attentions:
            attention_info = {
                **mask_metrics,
                **spectral_metrics,
                'delta_w': delta_w_clamped,
                'rank_activations': rank_activations,
                'current_flops': current_flops,
                'cumulative_flops': self.flop_count.item()
            }
            return output, attention_info
        
        return output, None
    
    def _compute_flops(self, batch_size: int, seq_len: int, sparsity: float) -> int:
        """Compute FLOPs for current forward pass."""
        # U projection: B * S * D * total_rank
        u_flops = batch_size * seq_len * self.hidden_size * self.config.total_rank
        
        # V projection: B * S * active_rank * D (considering sparsity)
        active_rank = int(self.config.total_rank * (1 - sparsity))
        v_flops = batch_size * seq_len * active_rank * self.hidden_size
        
        # Mask computation (approximate)
        mask_flops = batch_size * seq_len * self.config.adaptation_dim
        
        # Hadamard transform if used
        hadamard_flops = 0
        if self.config.use_hadamard_path:
            hadamard_flops = batch_size * seq_len * self.mask_controller.hadamard_dim * math.log2(self.mask_controller.hadamard_dim)
        
        return u_flops + v_flops + mask_flops + hadamard_flops
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        u_params = self.U.weight.numel()
        v_params = self.V.weight.numel()
        controller_params = sum(p.numel() for p in self.mask_controller.parameters())
        
        return {
            'U_projection': u_params,
            'V_projection': v_params,
            'mask_controller': controller_params,
            'total': u_params + v_params + controller_params
        }
    
    def get_effective_rank(self) -> float:
        """Get current effective rank based on masking."""
        with torch.no_grad():
            # Get average mask from EMA
            avg_mask = self.mask_controller.instance_controller.mask_ema
            effective_rank = torch.sum(avg_mask).item()
            return effective_rank
    
    def compute_budget_metrics(self, batch_size: int = 32, seq_len: int = 128) -> Dict[str, float]:
        """Compute comprehensive budget metrics."""
        param_counts = self.get_parameter_count()
        effective_rank = self.get_effective_rank()
        
        # Compute expected FLOPs based on average sparsity
        expected_sparsity = 1.0 - (self.config.active_rank / self.config.total_rank)
        expected_flops = self._compute_flops(batch_size, seq_len, expected_sparsity)
        
        # Memory estimate
        memory_mb = (
            param_counts['total'] * 4 +  # Parameters
            batch_size * seq_len * self.hidden_size * 4 +  # Base activations
            batch_size * seq_len * self.config.total_rank * 4  # Rank activations
        ) / (1024 * 1024)
        
        return {
            'parameters': param_counts['total'],
            'effective_rank': effective_rank,
            'target_sparsity': self.config.target_sparsity,
            'expected_flops': expected_flops,
            'memory_mb': memory_mb,
            'total_rank': self.config.total_rank,
            'active_rank': self.config.active_rank
        }
    
    def get_mask_statistics(self) -> Dict[str, float]:
        """Get detailed mask statistics for analysis."""
        with torch.no_grad():
            mask_ema = self.mask_controller.instance_controller.mask_ema
            
            return {
                'mask_entropy': -torch.sum(mask_ema * torch.log(mask_ema + 1e-8)).item(),
                'mask_variance': torch.var(mask_ema).item(),
                'mask_mean': torch.mean(mask_ema).item(),
                'mask_max': torch.max(mask_ema).item(),
                'mask_min': torch.min(mask_ema).item(),
                'effective_diversity': torch.sum(mask_ema > 0.1).item()  # Ranks with >10% usage
            }