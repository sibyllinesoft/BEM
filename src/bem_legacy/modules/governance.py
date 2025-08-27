"""
Governance Components for BEM v1.1

Implements the governance requirements from TODO.md:
- Spectral clamp: σ₁ per layer
- Frobenius trust-region: τ projection on sum of deltas  
- Gate decorrelation penalty
- Flip penalty to discourage rapid routing changes

These components ensure training stability and prevent pathological behaviors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class SpectralGovernance(nn.Module):
    """
    Spectral governance with σ₁ clamp per layer.
    
    Enforces maximum singular value constraints on weight updates
    to prevent exploding gradients and maintain stability.
    """
    
    def __init__(
        self,
        max_singular_value: float = 1.0,
        layer_specific: bool = True,
        smoothing_momentum: float = 0.9
    ):
        super().__init__()
        self.max_singular_value = max_singular_value
        self.layer_specific = layer_specific
        self.smoothing_momentum = smoothing_momentum
        
        # Track layer-specific sigma_1 values
        self.register_buffer('layer_sigma1_ema', torch.tensor(0.0))
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))
    
    def apply_spectral_clamp(
        self,
        delta_w: torch.Tensor,
        layer_name: Optional[str] = None,
        target_sigma1: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply spectral clamping to weight delta.
        
        Args:
            delta_w: Weight delta tensor [out_features, in_features]  
            layer_name: Layer name for layer-specific tracking
            target_sigma1: Override for max singular value
            
        Returns:
            Clamped delta weights and statistics
        """
        if target_sigma1 is None:
            target_sigma1 = self.max_singular_value
        
        # Compute SVD
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        
        # Current maximum singular value
        current_sigma1 = S[0].item()
        
        # Update EMA tracking
        if self.training:
            if self.update_count == 0:
                self.layer_sigma1_ema = current_sigma1
            else:
                self.layer_sigma1_ema = (
                    self.smoothing_momentum * self.layer_sigma1_ema + 
                    (1 - self.smoothing_momentum) * current_sigma1
                )
            self.update_count += 1
        
        # Apply clamping if needed
        if current_sigma1 > target_sigma1:
            # Clamp singular values
            S_clamped = torch.clamp(S, max=target_sigma1)
            
            # Reconstruct with clamped values
            delta_w_clamped = U @ torch.diag(S_clamped) @ Vh
            clamped = True
        else:
            delta_w_clamped = delta_w
            clamped = False
        
        stats = {
            'original_sigma1': current_sigma1,
            'clamped_sigma1': S[0].item() if clamped else current_sigma1,
            'was_clamped': clamped,
            'sigma1_ema': self.layer_sigma1_ema.item(),
            'condition_number': (S[0] / S[-1]).item() if len(S) > 1 else 1.0
        }
        
        return delta_w_clamped, stats


class FrobeniusConstraint(nn.Module):
    """
    Frobenius trust-region constraint.
    
    Projects sum of delta weights to satisfy Frobenius norm budget τ.
    Ensures that total weight change magnitude stays within bounds.
    """
    
    def __init__(
        self,
        fro_budget: float = 1.0,
        adaptive_budget: bool = True,
        budget_decay: float = 0.99
    ):
        super().__init__()
        self.base_fro_budget = fro_budget
        self.adaptive_budget = adaptive_budget
        self.budget_decay = budget_decay
        
        # Dynamic budget adjustment
        self.register_buffer('current_budget', torch.tensor(fro_budget))
        self.register_buffer('violation_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_updates', torch.tensor(0, dtype=torch.long))
    
    def apply_frobenius_constraint(
        self,
        delta_weights: List[torch.Tensor],
        layer_names: Optional[List[str]] = None
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        Apply Frobenius norm constraint to sum of deltas.
        
        Args:
            delta_weights: List of weight delta tensors
            layer_names: Optional layer names for tracking
            
        Returns:
            Constrained delta weights and statistics
        """
        if not delta_weights:
            return delta_weights, {}
        
        # Compute Frobenius norm of sum of deltas
        total_fro_norm_sq = sum(torch.norm(dw, 'fro') ** 2 for dw in delta_weights)
        total_fro_norm = torch.sqrt(total_fro_norm_sq)
        
        # Check if constraint is violated
        current_budget = self.current_budget.item()
        constraint_violated = total_fro_norm > current_budget
        
        if constraint_violated:
            # Project to satisfy constraint
            scale_factor = current_budget / total_fro_norm
            constrained_deltas = [dw * scale_factor for dw in delta_weights]
            
            # Update violation tracking
            if self.training:
                self.violation_count += 1
        else:
            constrained_deltas = delta_weights
            scale_factor = 1.0
        
        # Update statistics
        if self.training:
            self.total_updates += 1
            
            # Adaptive budget adjustment
            if self.adaptive_budget:
                violation_rate = self.violation_count.float() / self.total_updates.float()
                
                # If violation rate is too high, increase budget
                if violation_rate > 0.1:  # More than 10% violations
                    self.current_budget *= 1.05
                # If violation rate is very low, decrease budget
                elif violation_rate < 0.02:  # Less than 2% violations
                    self.current_budget *= self.budget_decay
                
                # Clamp budget to reasonable range
                self.current_budget = torch.clamp(
                    self.current_budget,
                    min=self.base_fro_budget * 0.1,
                    max=self.base_fro_budget * 5.0
                )
        
        stats = {
            'total_fro_norm': total_fro_norm.item(),
            'fro_budget': current_budget,
            'constraint_violated': constraint_violated,
            'scale_factor': scale_factor,
            'violation_rate': (self.violation_count.float() / max(1, self.total_updates)).item(),
            'adaptive_budget': self.current_budget.item()
        }
        
        return constrained_deltas, stats


class GateDecorrelationPenalty(nn.Module):
    """
    Gate decorrelation penalty to encourage diverse expert usage.
    
    Penalizes high correlation between different experts' usage patterns
    to promote specialization and prevent mode collapse.
    """
    
    def __init__(
        self,
        penalty_weight: float = 0.01,
        target_correlation: float = 0.0,
        temporal_smoothing: float = 0.9
    ):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.target_correlation = target_correlation
        self.temporal_smoothing = temporal_smoothing
        
        # Track expert usage patterns
        self.register_buffer('expert_usage_ema', None)
    
    def compute_decorrelation_loss(
        self,
        routing_weights: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute decorrelation penalty for routing weights.
        
        Args:
            routing_weights: [batch_size, seq_len, num_experts]
            expert_indices: Optional hard expert indices
            
        Returns:
            Dictionary with penalty loss and statistics
        """
        batch_size, seq_len, num_experts = routing_weights.shape
        
        # Compute expert usage patterns (average over batch and sequence)
        expert_usage = routing_weights.mean(dim=(0, 1))  # [num_experts]
        
        # Update EMA of expert usage
        if self.expert_usage_ema is None:
            self.expert_usage_ema = expert_usage.detach()
        else:
            self.expert_usage_ema = (
                self.temporal_smoothing * self.expert_usage_ema +
                (1 - self.temporal_smoothing) * expert_usage.detach()
            )
        
        # Compute correlation matrix between experts
        # Reshape for correlation computation
        expert_activations = routing_weights.transpose(1, 2)  # [batch, experts, seq]
        expert_activations_flat = expert_activations.reshape(num_experts, -1)  # [experts, batch*seq]
        
        # Center the activations
        expert_mean = expert_activations_flat.mean(dim=1, keepdim=True)
        expert_centered = expert_activations_flat - expert_mean
        
        # Compute correlation matrix
        expert_cov = torch.mm(expert_centered, expert_centered.t()) / (expert_centered.shape[1] - 1)
        expert_std = torch.sqrt(torch.diag(expert_cov))
        expert_corr = expert_cov / torch.outer(expert_std, expert_std)
        
        # Penalty: sum of squared off-diagonal correlations
        mask = 1.0 - torch.eye(num_experts, device=expert_corr.device)
        off_diag_corr = expert_corr * mask
        decorrelation_penalty = (off_diag_corr ** 2).sum() * self.penalty_weight
        
        # Usage balance penalty (encourage equal usage)
        uniform_usage = torch.ones_like(expert_usage) / num_experts
        usage_entropy = -(expert_usage * torch.log(expert_usage + 1e-8)).sum()
        max_entropy = math.log(num_experts)
        usage_balance_penalty = (max_entropy - usage_entropy) * self.penalty_weight
        
        return {
            'decorrelation_loss': decorrelation_penalty,
            'usage_balance_loss': usage_balance_penalty,
            'total_penalty': decorrelation_penalty + usage_balance_penalty,
            'expert_correlations': expert_corr,
            'expert_usage': expert_usage,
            'expert_usage_ema': self.expert_usage_ema,
            'usage_entropy': usage_entropy / max_entropy  # Normalized entropy
        }


class FlipPenalty(nn.Module):
    """
    Flip penalty to discourage rapid routing changes.
    
    Works in conjunction with hysteresis in chunk-sticky routing
    to maintain cache efficiency by penalizing frequent expert switches.
    """
    
    def __init__(
        self,
        penalty_weight: float = 0.1,
        temporal_window: int = 10,
        target_flip_rate: float = 0.1
    ):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.temporal_window = temporal_window
        self.target_flip_rate = target_flip_rate
        
        # Circular buffer for tracking flips
        self.register_buffer('flip_history', torch.zeros(temporal_window))
        self.register_buffer('history_index', torch.tensor(0, dtype=torch.long))
    
    def compute_flip_penalty(
        self,
        current_expert_indices: torch.Tensor,
        previous_expert_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flip penalty based on routing changes.
        
        Args:
            current_expert_indices: Current routing decisions [batch_size, num_chunks]
            previous_expert_indices: Previous routing decisions (same shape)
            
        Returns:
            Dictionary with flip penalty and statistics
        """
        if previous_expert_indices is None:
            # No penalty for first timestep
            flip_penalty = torch.tensor(0.0, device=current_expert_indices.device)
            flip_rate = torch.tensor(0.0, device=current_expert_indices.device)
        else:
            # Compute flip rate
            flips = (current_expert_indices != previous_expert_indices).float()
            current_flip_rate = flips.mean()
            
            # Update flip history
            if self.training:
                idx = self.history_index % self.temporal_window
                self.flip_history[idx] = current_flip_rate.detach()
                self.history_index += 1
            
            # Compute average flip rate over temporal window
            valid_entries = min(self.history_index.item(), self.temporal_window)
            avg_flip_rate = self.flip_history[:valid_entries].mean()
            
            # Penalty for exceeding target flip rate
            excess_flips = torch.clamp(current_flip_rate - self.target_flip_rate, min=0.0)
            flip_penalty = excess_flips ** 2 * self.penalty_weight
            
            flip_rate = current_flip_rate
        
        return {
            'flip_penalty': flip_penalty,
            'current_flip_rate': flip_rate,
            'avg_flip_rate': self.flip_history[:min(self.history_index.item(), self.temporal_window)].mean(),
            'target_flip_rate': torch.tensor(self.target_flip_rate, device=flip_penalty.device)
        }


class BEMGovernance(nn.Module):
    """
    Combined governance module for BEM v1.1.
    
    Integrates all governance components:
    - Spectral clamping per layer
    - Frobenius trust-region constraints  
    - Gate decorrelation penalties
    - Flip penalties for routing stability
    """
    
    def __init__(
        self,
        max_singular_value: float = 1.0,
        fro_budget: float = 1.0,
        decorrelation_weight: float = 0.01,
        flip_penalty_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.spectral_gov = SpectralGovernance(max_singular_value, **kwargs)
        self.frobenius_constraint = FrobeniusConstraint(fro_budget, **kwargs)
        self.decorrelation_penalty = GateDecorrelationPenalty(decorrelation_weight, **kwargs)
        self.flip_penalty = FlipPenalty(flip_penalty_weight, **kwargs)
        
    def apply_governance(
        self,
        delta_weights: List[torch.Tensor],
        routing_weights: torch.Tensor,
        current_expert_indices: torch.Tensor,
        previous_expert_indices: Optional[torch.Tensor] = None,
        layer_names: Optional[List[str]] = None
    ) -> Tuple[List[torch.Tensor], Dict[str, any]]:
        """
        Apply all governance constraints.
        
        Args:
            delta_weights: List of weight delta tensors
            routing_weights: Routing weights [batch, seq, experts]
            current_expert_indices: Current expert choices
            previous_expert_indices: Previous expert choices (for flip penalty)
            layer_names: Optional layer names
            
        Returns:
            Governed delta weights and comprehensive statistics
        """
        all_stats = {}
        
        # Apply spectral governance to each layer
        governed_deltas = []
        for i, delta_w in enumerate(delta_weights):
            layer_name = layer_names[i] if layer_names else f"layer_{i}"
            governed_delta, spectral_stats = self.spectral_gov.apply_spectral_clamp(
                delta_w, layer_name
            )
            governed_deltas.append(governed_delta)
            all_stats[f'spectral_{layer_name}'] = spectral_stats
        
        # Apply Frobenius constraint to sum of deltas
        governed_deltas, fro_stats = self.frobenius_constraint.apply_frobenius_constraint(
            governed_deltas, layer_names
        )
        all_stats['frobenius'] = fro_stats
        
        # Compute penalties (these contribute to loss, don't modify weights)
        decorr_stats = self.decorrelation_penalty.compute_decorrelation_loss(routing_weights)
        all_stats['decorrelation'] = decorr_stats
        
        flip_stats = self.flip_penalty.compute_flip_penalty(
            current_expert_indices, previous_expert_indices
        )
        all_stats['flip'] = flip_stats
        
        # Total penalty for loss
        total_penalty = decorr_stats['total_penalty'] + flip_stats['flip_penalty']
        all_stats['total_governance_penalty'] = total_penalty
        
        return governed_deltas, all_stats


def create_bem_governance(
    config: Optional[Dict] = None
) -> BEMGovernance:
    """
    Factory function for BEM governance.
    
    Args:
        config: Configuration dictionary with governance parameters
        
    Returns:
        BEMGovernance instance
    """
    if config is None:
        config = {}
    
    # Default values from TODO.md
    defaults = {
        'max_singular_value': 1.0,
        'fro_budget': 1.0, 
        'decorrelation_weight': 0.01,
        'flip_penalty_weight': 0.1
    }
    
    # Merge with provided config
    final_config = {**defaults, **config}
    
    return BEMGovernance(**final_config)