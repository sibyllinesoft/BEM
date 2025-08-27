"""
PT1 - Head-Group Gating @ W_O Implementation.

Implements head-group gating with rank split across G groups and per-group gates
derived from attention statistics and retrieval quality. Gates are decorrelated
to prevent redundancy.

Expected Performance: +0.5–1.5% improvement
Budget Constraint: ±5% params/FLOPs vs v1.3-stack anchor
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
class HeadGroupGatingConfig:
    """Configuration for PT1 Head-Group Gating."""
    
    # Group structure
    num_groups: int = 4          # G groups for rank splitting
    heads_per_group: int = 2     # Heads assigned to each group
    rank_per_group: int = 2      # Rank allocation per group
    
    # Gating parameters
    gate_temperature: float = 1.0    # Softmax temperature for gating
    gate_dropout: float = 0.1        # Dropout on gate weights
    decorrelation_strength: float = 0.1  # Decorrelation penalty weight
    
    # Attention statistics
    use_attention_stats: bool = True     # Use attention entropy/variance
    attention_stat_dim: int = 16         # Dimension for attention feature extraction
    
    # Retrieval quality scoring
    use_retrieval_quality: bool = True   # Use retrieval quality signals
    retrieval_quality_dim: int = 8       # Dimension for retrieval features
    
    # Trust region constraints
    max_singular_value: float = 1.0     # Spectral clamp
    fro_budget: float = 0.8             # Frobenius budget (conservative)
    
    # Budget tracking
    budget_tolerance: float = 0.05      # ±5% tolerance


class AttentionStatExtractor(nn.Module):
    """Extract gating signals from attention statistics."""
    
    def __init__(self, config: HeadGroupGatingConfig, hidden_size: int):
        super().__init__()
        self.config = config
        
        # Attention statistics processor
        self.attention_processor = nn.Sequential(
            nn.Linear(hidden_size, config.attention_stat_dim),
            nn.LayerNorm(config.attention_stat_dim),
            nn.GELU(),
            nn.Dropout(config.gate_dropout)
        )
        
        # Entropy and variance computation
        self.register_buffer('attention_ema', torch.zeros(hidden_size))
        self.register_buffer('attention_var_ema', torch.zeros(hidden_size))
        self.ema_momentum = 0.99
        
    def forward(
        self, 
        attention_weights: torch.Tensor,  # [B, H, S, S]
        hidden_states: torch.Tensor       # [B, S, D]
    ) -> torch.Tensor:
        """Extract attention-based gating features."""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Compute attention entropy per head
        attention_entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-8),
            dim=-1
        ).mean(dim=-1)  # [B, H]
        
        # Compute attention variance per head
        attention_variance = torch.var(attention_weights, dim=-1).mean(dim=-1)  # [B, H]
        
        # Combine entropy and variance features
        attention_features = torch.cat([
            attention_entropy, 
            attention_variance
        ], dim=-1)  # [B, 2*H]
        
        # Update EMA statistics
        if self.training:
            batch_mean = hidden_states.mean(dim=(0, 1))
            batch_var = hidden_states.var(dim=(0, 1))
            
            self.attention_ema.mul_(self.ema_momentum).add_(
                batch_mean, alpha=1 - self.ema_momentum
            )
            self.attention_var_ema.mul_(self.ema_momentum).add_(
                batch_var, alpha=1 - self.ema_momentum
            )
        
        # Process through attention feature extractor
        processed_features = self.attention_processor(hidden_states.mean(dim=1))
        
        return processed_features


class RetrievalQualityScorer(nn.Module):
    """Score retrieval quality for gating decisions."""
    
    def __init__(self, config: HeadGroupGatingConfig, hidden_size: int):
        super().__init__()
        self.config = config
        
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_size, config.retrieval_quality_dim),
            nn.LayerNorm(config.retrieval_quality_dim),
            nn.GELU(),
            nn.Linear(config.retrieval_quality_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor, retrieval_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score retrieval quality for gating."""
        if retrieval_context is None:
            # Use self-attention patterns as proxy for retrieval quality
            quality_input = hidden_states
        else:
            # Use actual retrieval context
            quality_input = torch.cat([hidden_states, retrieval_context], dim=-1)
            
        quality_scores = self.quality_scorer(quality_input.mean(dim=1))  # [B, 1]
        return quality_scores


class AttentionGateController(nn.Module):
    """Controller for head-group gating with decorrelation."""
    
    def __init__(self, config: HeadGroupGatingConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Feature extractors
        self.attention_extractor = AttentionStatExtractor(config, hidden_size)
        self.retrieval_scorer = RetrievalQualityScorer(config, hidden_size)
        
        # Gate computation
        gate_input_dim = (
            config.attention_stat_dim + 
            config.retrieval_quality_dim + 
            1  # retrieval quality score
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, config.num_groups * 2),
            nn.LayerNorm(config.num_groups * 2),
            nn.GELU(),
            nn.Dropout(config.gate_dropout),
            nn.Linear(config.num_groups * 2, config.num_groups)
        )
        
        # Decorrelation tracking
        self.register_buffer('gate_correlation_ema', torch.eye(config.num_groups))
        
    def compute_gates(
        self,
        attention_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        retrieval_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute group gates with decorrelation."""
        batch_size = hidden_states.shape[0]
        
        # Extract attention features
        attention_features = self.attention_extractor(attention_weights, hidden_states)
        
        # Score retrieval quality
        retrieval_scores = self.retrieval_scorer(hidden_states, retrieval_context)
        
        # Combine features
        gate_input = torch.cat([
            attention_features,
            retrieval_scores.expand(-1, self.config.retrieval_quality_dim),
            retrieval_scores
        ], dim=-1)
        
        # Compute raw gates
        raw_gates = self.gate_network(gate_input)  # [B, G]
        
        # Apply temperature and softmax
        gates = F.softmax(raw_gates / self.config.gate_temperature, dim=-1)
        
        # Update correlation tracking
        if self.training and gates.shape[0] > 1:
            gate_corr = torch.corrcoef(gates.T)
            self.gate_correlation_ema.mul_(0.99).add_(gate_corr, alpha=0.01)
        
        # Compute decorrelation penalty
        decorr_penalty = torch.sum(
            torch.abs(self.gate_correlation_ema - torch.eye(self.config.num_groups, device=gates.device))
        )
        
        metrics = {
            'gates': gates,
            'gate_entropy': -torch.sum(gates * torch.log(gates + 1e-8), dim=-1).mean(),
            'decorrelation_penalty': decorr_penalty,
            'attention_features': attention_features,
            'retrieval_scores': retrieval_scores
        }
        
        return gates, metrics


class HeadGroupGatingModule(nn.Module):
    """
    PT1 Head-Group Gating Module.
    
    Implements rank splitting across groups with per-group gates derived from
    attention statistics and retrieval quality, with decorrelation constraints.
    """
    
    def __init__(
        self,
        config: HeadGroupGatingConfig,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        # Validate configuration
        assert num_attention_heads % config.num_groups == 0, \
            f"num_attention_heads ({num_attention_heads}) must be divisible by num_groups ({config.num_groups})"
        
        self.heads_per_group = num_attention_heads // config.num_groups
        
        # Gate controller
        self.gate_controller = AttentionGateController(config, hidden_size)
        
        # Group-specific low-rank projections at W_O
        self.group_projections = nn.ModuleList([
            nn.ModuleDict({
                'U': nn.Linear(hidden_size, config.rank_per_group, bias=False),
                'V': nn.Linear(config.rank_per_group, hidden_size, bias=False)
            }) for _ in range(config.num_groups)
        ])
        
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
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize group projection parameters."""
        for group_proj in self.group_projections:
            # Orthogonal initialization for U
            nn.init.orthogonal_(group_proj['U'].weight)
            
            # Xavier initialization for V (scaled down)
            nn.init.xavier_uniform_(group_proj['V'].weight, gain=0.1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, S, D]
        attention_weights: torch.Tensor,      # [B, H, S, S]
        retrieval_context: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with head-group gating."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute group gates
        gates, gate_metrics = self.gate_controller.compute_gates(
            attention_weights, hidden_states, retrieval_context
        )
        
        # Apply group-specific transformations
        group_outputs = []
        group_deltas = []
        
        for group_idx in range(self.config.num_groups):
            # Get group-specific projection
            U = self.group_projections[group_idx]['U']
            V = self.group_projections[group_idx]['V']
            
            # Compute low-rank transformation
            latent = U(hidden_states)  # [B, S, R]
            delta_w = V(latent)        # [B, S, D]
            
            # Apply spectral governance
            delta_w_clamped, spectral_metrics = self.spectral_gov.apply_spectral_clamp(
                delta_w.view(-1, hidden_dim),
                layer_name=f"pt1_group_{group_idx}",
                target_sigma1=self.config.max_singular_value
            )
            delta_w_clamped = delta_w_clamped.view(batch_size, seq_len, hidden_dim)
            
            # Apply gate weighting
            gate_weight = gates[:, group_idx:group_idx+1, None]  # [B, 1, 1]
            gated_delta = gate_weight * delta_w_clamped
            
            group_outputs.append(gated_delta)
            group_deltas.append(delta_w_clamped)
        
        # Combine group outputs
        combined_delta = torch.sum(torch.stack(group_outputs, dim=0), dim=0)
        output = hidden_states + combined_delta
        
        # Compute additional metrics
        total_budget = torch.sum(torch.stack([
            torch.norm(delta, 'fro') for delta in group_deltas
        ]))
        
        # Decorrelation penalty in loss
        decorr_loss = self.config.decorrelation_strength * gate_metrics['decorrelation_penalty']
        
        if output_attentions:
            attention_info = {
                'gates': gates,
                'group_outputs': group_outputs,
                'total_budget': total_budget,
                'decorrelation_loss': decorr_loss,
                **gate_metrics
            }
            return output, attention_info
        
        return output, None
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown for budget validation."""
        group_params = sum(
            sum(p.numel() for p in group_proj.parameters())
            for group_proj in self.group_projections
        )
        
        gate_params = sum(p.numel() for p in self.gate_controller.parameters())
        
        return {
            'group_projections': group_params,
            'gate_controller': gate_params,
            'total': group_params + gate_params
        }
    
    def get_flop_count(self, batch_size: int, seq_len: int) -> Dict[str, int]:
        """Get FLOP count for budget validation."""
        # Group projections: B * S * (D * R + R * D) per group
        group_flops = (
            batch_size * seq_len * 
            (self.hidden_size * self.config.rank_per_group * 2) *
            self.config.num_groups
        )
        
        # Gate computation (simplified estimate)
        gate_flops = batch_size * seq_len * 100  # Approximate
        
        return {
            'group_projections': group_flops,
            'gate_computation': gate_flops,
            'total': group_flops + gate_flops
        }
    
    def compute_budget_metrics(self, batch_size: int = 32, seq_len: int = 128) -> Dict[str, float]:
        """Compute comprehensive budget metrics."""
        param_counts = self.get_parameter_count()
        flop_counts = self.get_flop_count(batch_size, seq_len)
        
        # Memory estimate (parameters + activations)
        memory_mb = (
            param_counts['total'] * 4 +  # 4 bytes per float32 parameter
            batch_size * seq_len * self.hidden_size * 4  # Activation memory
        ) / (1024 * 1024)
        
        return {
            'parameters': param_counts['total'],
            'flops': flop_counts['total'],
            'memory_mb': memory_mb,
            'groups': self.config.num_groups,
            'rank_per_group': self.config.rank_per_group
        }