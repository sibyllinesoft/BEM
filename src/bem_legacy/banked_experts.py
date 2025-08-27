"""
Banked Experts System for BEM - Phase 5 Implementation.

Implements MoE-style expert banking with top-k mixture-of-LoRAs, batch routing,
and load balancing as specified in TODO.md Phase 5.

Key Features:
- Expert Bank: Collection of specialized LoRA experts
- Top-k Router: Select best k experts for each input
- Batch Optimization: Group inputs by expert selection patterns  
- Load Balancing: Prevent expert collapse with auxiliary losses
- Fused Path: Maintain kernel efficiency from earlier phases
"""

from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import logging
from collections import defaultdict, Counter
import math

from .telemetry import TelemetryCollector


class ExpertUtilizationStats(NamedTuple):
    """Statistics tracking expert utilization."""
    expert_usage_counts: torch.Tensor
    total_selections: int
    entropy: float
    gini_coefficient: float
    active_experts: int


@dataclass
class BankedExpertsConfig:
    """Configuration for banked experts system."""
    
    # Expert bank parameters
    num_experts: int = 8
    expert_rank: int = 8
    top_k: int = 2
    
    # Load balancing
    load_balance_weight: float = 0.1
    diversity_weight: float = 0.05
    
    # Batch optimization
    enable_batching: bool = True
    batch_merge_threshold: float = 0.8
    min_batch_size: int = 4
    
    # Expert initialization
    init_std: float = 0.02
    orthogonal_init: bool = True
    
    # Gating network
    gate_hidden_dim: Optional[int] = None  # If None, uses input_dim // 2
    gate_dropout: float = 0.1
    gate_activation: str = "gelu"
    
    # Capacity factors
    capacity_factor: float = 1.25
    drop_tokens: bool = False
    
    # Advanced features
    enable_expert_dropout: bool = True
    expert_dropout_rate: float = 0.1
    enable_gradient_clipping: bool = True
    expert_grad_clip: float = 1.0


class LoRAExpert(nn.Module):
    """Individual LoRA expert in the bank."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        expert_id: int,
        init_std: float = 0.02,
        orthogonal_init: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.expert_id = expert_id
        
        # LoRA matrices - down and up projection
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, output_dim, bias=False)
        
        # Scaling factor (learnable per expert)
        self.scaling = nn.Parameter(torch.ones(1) * (rank ** -0.5))
        
        # Expert-specific normalization
        self.norm = nn.LayerNorm(rank)
        
        self._initialize_weights(init_std, orthogonal_init)
    
    def _initialize_weights(self, init_std: float, orthogonal_init: bool):
        """Initialize expert weights."""
        if orthogonal_init:
            # Orthogonal initialization for better expert diversity
            nn.init.orthogonal_(self.lora_A.weight, gain=init_std)
            nn.init.orthogonal_(self.lora_B.weight, gain=init_std)
        else:
            # Standard normal initialization
            nn.init.normal_(self.lora_A.weight, std=init_std)
            nn.init.normal_(self.lora_B.weight, std=init_std)
        
        # Initialize B to zero for stable training
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA expert."""
        # x: [batch_size, seq_len, input_dim] or [tokens, input_dim]
        h = self.lora_A(x)  # [*, rank]
        h = self.norm(h)
        h = h * self.scaling
        output = self.lora_B(h)  # [*, output_dim]
        return output


class TopKGatingNetwork(nn.Module):
    """Top-k gating network for expert selection."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim or (input_dim // 2)
        
        # Gating network layers
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_experts)
        )
        
        # Noise for load balancing (training only)
        self.noise_std = 0.1
        
        # Track gating statistics
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
    
    def forward(
        self, 
        x: torch.Tensor, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of gating network.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            training: Whether in training mode
            
        Returns:
            gates: Top-k gate values [batch_size * seq_len, top_k]
            indices: Expert indices [batch_size * seq_len, top_k] 
            load_balancing_loss: Auxiliary loss for load balancing
        """
        # Flatten for processing
        batch_size, seq_len = x.shape[:2]
        x_flat = x.view(-1, self.input_dim)  # [batch_size * seq_len, input_dim]
        
        # Compute gate logits
        gate_logits = self.gate_network(x_flat)  # [tokens, num_experts]
        
        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(
            gate_logits, self.top_k, dim=-1
        )  # [tokens, top_k]
        
        # Convert to probabilities
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Update expert usage statistics
        if training:
            self._update_expert_stats(top_k_indices)
        
        # Compute load balancing loss
        load_balancing_loss = self._compute_load_balancing_loss(gate_logits)
        
        return top_k_gates, top_k_indices, load_balancing_loss
    
    def _update_expert_stats(self, expert_indices: torch.Tensor):
        """Update expert usage statistics."""
        with torch.no_grad():
            # Count expert usage
            expert_mask = F.one_hot(
                expert_indices.view(-1), 
                num_classes=self.num_experts
            ).float().sum(dim=0)
            
            self.expert_counts += expert_mask
            self.total_tokens += expert_indices.numel()
    
    def _compute_load_balancing_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        # Gate probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Mean probability per expert (should be 1/num_experts)
        mean_probs = gate_probs.mean(dim=0)
        
        # Encourage uniform distribution
        uniform_target = 1.0 / self.num_experts
        load_loss = F.mse_loss(mean_probs, torch.full_like(mean_probs, uniform_target))
        
        return load_loss
    
    def get_expert_utilization(self) -> ExpertUtilizationStats:
        """Get expert utilization statistics."""
        with torch.no_grad():
            usage_counts = self.expert_counts.clone()
            total_selections = int(self.total_tokens.item())
            
            if total_selections == 0:
                return ExpertUtilizationStats(
                    expert_usage_counts=usage_counts,
                    total_selections=0,
                    entropy=0.0,
                    gini_coefficient=1.0,
                    active_experts=0
                )
            
            # Normalize to probabilities
            usage_probs = usage_counts / usage_counts.sum()
            
            # Compute entropy (higher = more uniform)
            entropy = -torch.sum(usage_probs * torch.log(usage_probs + 1e-8)).item()
            
            # Compute Gini coefficient (lower = more uniform)
            sorted_probs, _ = torch.sort(usage_probs)
            n = len(sorted_probs)
            index = torch.arange(1, n + 1, dtype=torch.float32, device=usage_probs.device)
            gini = (2 * torch.sum(index * sorted_probs) / (n * torch.sum(sorted_probs)) - (n + 1) / n).item()
            
            # Count active experts (usage > 1% of uniform)
            threshold = 0.01 / self.num_experts
            active_experts = int((usage_probs > threshold).sum().item())
            
            return ExpertUtilizationStats(
                expert_usage_counts=usage_counts,
                total_selections=total_selections,
                entropy=entropy,
                gini_coefficient=gini,
                active_experts=active_experts
            )


class BatchedExpertRouter(nn.Module):
    """Batched routing for efficient expert computation."""
    
    def __init__(
        self,
        config: BankedExpertsConfig,
        merge_threshold: float = 0.8,
        min_batch_size: int = 4
    ):
        super().__init__()
        self.config = config
        self.merge_threshold = merge_threshold
        self.min_batch_size = min_batch_size
        
    def route_tokens(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Route tokens to experts with batching optimization.
        
        Args:
            tokens: Input tokens [num_tokens, input_dim]
            expert_indices: Expert indices [num_tokens, top_k]
            expert_weights: Expert weights [num_tokens, top_k]
            
        Returns:
            Dictionary with routing information for batched execution
        """
        num_tokens, top_k = expert_indices.shape
        
        # Create routing map
        routing_map = {}
        token_expert_assignments = []
        
        # Group tokens by expert selection patterns
        pattern_groups = defaultdict(list)
        
        for token_idx in range(num_tokens):
            # Create pattern signature (sorted expert indices)
            pattern = tuple(sorted(expert_indices[token_idx].tolist()))
            pattern_groups[pattern].append(token_idx)
        
        # Process each pattern group
        for pattern, token_indices in pattern_groups.items():
            if len(token_indices) >= self.min_batch_size:
                # Large enough group - create batch
                routing_map[f"batch_{len(routing_map)}"] = {
                    'token_indices': torch.tensor(token_indices, device=tokens.device),
                    'expert_pattern': pattern,
                    'tokens': tokens[token_indices],
                    'weights': expert_weights[token_indices]
                }
            else:
                # Small group - merge with similar patterns or process individually
                merged = False
                for existing_key, existing_batch in routing_map.items():
                    if self._patterns_similar(pattern, existing_batch['expert_pattern']):
                        # Merge with existing batch
                        existing_indices = existing_batch['token_indices']
                        new_indices = torch.tensor(token_indices, device=tokens.device)
                        
                        routing_map[existing_key]['token_indices'] = torch.cat([
                            existing_indices, new_indices
                        ])
                        routing_map[existing_key]['tokens'] = torch.cat([
                            existing_batch['tokens'], tokens[token_indices]
                        ])
                        routing_map[existing_key]['weights'] = torch.cat([
                            existing_batch['weights'], expert_weights[token_indices]
                        ])
                        merged = True
                        break
                
                if not merged:
                    # Process individually
                    for i, token_idx in enumerate(token_indices):
                        routing_map[f"individual_{token_idx}"] = {
                            'token_indices': torch.tensor([token_idx], device=tokens.device),
                            'expert_pattern': pattern,
                            'tokens': tokens[token_idx:token_idx+1],
                            'weights': expert_weights[token_idx:token_idx+1]
                        }
        
        return routing_map
    
    def _patterns_similar(self, pattern1: Tuple, pattern2: Tuple) -> bool:
        """Check if two expert selection patterns are similar enough to merge."""
        set1, set2 = set(pattern1), set(pattern2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return True
        
        jaccard_similarity = intersection / union
        return jaccard_similarity >= self.merge_threshold


class BankedExpertsModule(nn.Module):
    """Main banked experts module implementing MoE-style routing."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: BankedExpertsConfig,
        telemetry_collector: Optional[TelemetryCollector] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.telemetry = telemetry_collector
        
        # Expert bank
        self.experts = nn.ModuleList([
            LoRAExpert(
                input_dim=input_dim,
                output_dim=output_dim,
                rank=config.expert_rank,
                expert_id=i,
                init_std=config.init_std,
                orthogonal_init=config.orthogonal_init
            ) for i in range(config.num_experts)
        ])
        
        # Gating network
        self.gating_network = TopKGatingNetwork(
            input_dim=input_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            hidden_dim=config.gate_hidden_dim,
            dropout=config.gate_dropout,
            activation=config.gate_activation
        )
        
        # Batched router for efficiency
        if config.enable_batching:
            self.batch_router = BatchedExpertRouter(
                config=config,
                merge_threshold=config.batch_merge_threshold,
                min_batch_size=config.min_batch_size
            )
        
        # Expert dropout for regularization
        if config.enable_expert_dropout:
            self.expert_dropout = nn.Dropout(config.expert_dropout_rate)
        
        # Load balancing loss tracking
        self.register_buffer('cumulative_load_loss', torch.tensor(0.0))
        self.register_buffer('loss_steps', torch.tensor(0))
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through banked experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            training: Whether in training mode
            
        Returns:
            output: Expert-routed output [batch_size, seq_len, output_dim]
            routing_info: Dictionary with routing statistics and losses
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Gating - select top-k experts
        expert_weights, expert_indices, load_loss = self.gating_network(x, training)
        # expert_weights: [batch_size * seq_len, top_k]
        # expert_indices: [batch_size * seq_len, top_k]
        
        # Flatten tokens for processing
        tokens = x.view(-1, input_dim)  # [num_tokens, input_dim]
        
        # Route tokens through selected experts
        if self.config.enable_batching and hasattr(self, 'batch_router'):
            output = self._batched_expert_computation(
                tokens, expert_indices, expert_weights
            )
        else:
            output = self._individual_expert_computation(
                tokens, expert_indices, expert_weights
            )
        
        # Reshape output
        output = output.view(batch_size, seq_len, self.output_dim)
        
        # Collect telemetry
        routing_info = {
            'load_balancing_loss': load_loss,
            'expert_utilization': self.gating_network.get_expert_utilization(),
            'num_active_experts': len(torch.unique(expert_indices)),
            'routing_entropy': self._compute_routing_entropy(expert_weights)
        }
        
        # Update cumulative loss tracking
        if training:
            self.cumulative_load_loss += load_loss.detach()
            self.loss_steps += 1
        
        if self.telemetry:
            self.telemetry.log_routing_metrics(routing_info)
        
        return output, routing_info
    
    def _batched_expert_computation(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute expert outputs with batching optimization."""
        num_tokens = tokens.shape[0]
        output = torch.zeros(
            num_tokens, self.output_dim,
            device=tokens.device, dtype=tokens.dtype
        )
        
        # Route tokens to batches
        routing_map = self.batch_router.route_tokens(
            tokens, expert_indices, expert_weights
        )
        
        # Process each batch
        for batch_key, batch_info in routing_map.items():
            batch_tokens = batch_info['tokens']
            batch_weights = batch_info['weights']
            batch_indices = batch_info['token_indices']
            expert_pattern = batch_info['expert_pattern']
            
            # Compute outputs for experts in this pattern
            batch_output = torch.zeros(
                len(batch_tokens), self.output_dim,
                device=tokens.device, dtype=tokens.dtype
            )
            
            for k_idx, expert_id in enumerate(expert_pattern):
                # Get weights for this expert across all tokens in batch
                expert_weights_for_batch = batch_weights[:, k_idx]
                
                # Apply expert dropout if enabled
                if hasattr(self, 'expert_dropout') and self.training:
                    expert_weights_for_batch = self.expert_dropout(expert_weights_for_batch)
                
                # Compute expert output
                expert_output = self.experts[expert_id](batch_tokens)
                
                # Weight and accumulate
                weighted_output = expert_output * expert_weights_for_batch.unsqueeze(-1)
                batch_output += weighted_output
            
            # Store results
            output[batch_indices] = batch_output
        
        return output
    
    def _individual_expert_computation(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute expert outputs individually (fallback method)."""
        num_tokens = tokens.shape[0]
        output = torch.zeros(
            num_tokens, self.output_dim,
            device=tokens.device, dtype=tokens.dtype
        )
        
        # Process each token individually
        for token_idx in range(num_tokens):
            token_input = tokens[token_idx:token_idx+1]
            
            # Process top-k experts for this token
            for k_idx in range(self.config.top_k):
                expert_id = expert_indices[token_idx, k_idx].item()
                weight = expert_weights[token_idx, k_idx]
                
                # Apply expert dropout if enabled
                if hasattr(self, 'expert_dropout') and self.training:
                    weight = self.expert_dropout(weight)
                
                # Compute expert output
                expert_output = self.experts[expert_id](token_input)
                
                # Weight and accumulate
                output[token_idx] += weight * expert_output.squeeze(0)
        
        return output
    
    def _compute_routing_entropy(self, expert_weights: torch.Tensor) -> float:
        """Compute entropy of routing decisions."""
        with torch.no_grad():
            # Average weights across all tokens and normalize
            avg_weights = expert_weights.mean(dim=0)
            avg_weights = avg_weights / (avg_weights.sum() + 1e-8)
            
            # Compute entropy
            entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-8))
            return entropy.item()
    
    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """Get auxiliary losses for training."""
        losses = {}
        
        # Load balancing loss
        if self.loss_steps > 0:
            avg_load_loss = self.cumulative_load_loss / self.loss_steps
            losses['load_balancing'] = avg_load_loss * self.config.load_balance_weight
        
        # Diversity loss to encourage expert specialization
        if self.config.diversity_weight > 0:
            diversity_loss = self._compute_diversity_loss()
            losses['diversity'] = diversity_loss * self.config.diversity_weight
        
        return losses
    
    def _compute_diversity_loss(self) -> torch.Tensor:
        """Compute diversity loss to encourage expert specialization."""
        # Compare expert parameters to encourage diversity
        diversity_loss = 0.0
        
        for i in range(self.config.num_experts):
            for j in range(i + 1, self.config.num_experts):
                expert_i_params = torch.cat([
                    self.experts[i].lora_A.weight.view(-1),
                    self.experts[i].lora_B.weight.view(-1)
                ])
                expert_j_params = torch.cat([
                    self.experts[j].lora_A.weight.view(-1),
                    self.experts[j].lora_B.weight.view(-1)
                ])
                
                # Encourage orthogonality between experts
                similarity = F.cosine_similarity(
                    expert_i_params.unsqueeze(0),
                    expert_j_params.unsqueeze(0)
                )
                diversity_loss += similarity.abs()
        
        # Normalize by number of pairs
        num_pairs = self.config.num_experts * (self.config.num_experts - 1) / 2
        diversity_loss = diversity_loss / num_pairs
        
        return diversity_loss
    
    def get_expert_statistics(self) -> Dict[str, any]:
        """Get comprehensive expert utilization and performance statistics."""
        stats = {
            'utilization': self.gating_network.get_expert_utilization(),
            'auxiliary_losses': self.get_auxiliary_losses()
        }
        
        # Expert weight statistics
        expert_norms = []
        for expert in self.experts:
            norm_a = torch.norm(expert.lora_A.weight).item()
            norm_b = torch.norm(expert.lora_B.weight).item()
            expert_norms.append({'A': norm_a, 'B': norm_b, 'total': norm_a + norm_b})
        
        stats['expert_norms'] = expert_norms
        stats['scaling_factors'] = [expert.scaling.item() for expert in self.experts]
        
        return stats


def create_banked_experts_module(
    input_dim: int,
    output_dim: int,
    config: Optional[BankedExpertsConfig] = None,
    telemetry_collector: Optional[TelemetryCollector] = None
) -> BankedExpertsModule:
    """Create a banked experts module with default configuration."""
    if config is None:
        config = BankedExpertsConfig()
    
    return BankedExpertsModule(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config,
        telemetry_collector=telemetry_collector
    )


def create_default_banked_experts_config(
    num_experts: int = 8,
    expert_rank: int = 8,
    top_k: int = 2,
    enable_batching: bool = True
) -> BankedExpertsConfig:
    """Create default banked experts configuration."""
    return BankedExpertsConfig(
        num_experts=num_experts,
        expert_rank=expert_rank,
        top_k=top_k,
        enable_batching=enable_batching,
        load_balance_weight=0.1,
        diversity_weight=0.05
    )


# Example usage and testing
if __name__ == "__main__":
    # Test banked experts module
    config = create_default_banked_experts_config(num_experts=4, top_k=2)
    module = create_banked_experts_module(
        input_dim=512,
        output_dim=512,
        config=config
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, 512)
    
    output, routing_info = module(x, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Load balancing loss: {routing_info['load_balancing_loss']:.4f}")
    print(f"Active experts: {routing_info['num_active_experts']}")
    print(f"Routing entropy: {routing_info['routing_entropy']:.4f}")
    
    # Get expert statistics
    stats = module.get_expert_statistics()
    print("\nExpert Utilization:")
    util = stats['utilization']
    print(f"  Entropy: {util.entropy:.4f}")
    print(f"  Gini coefficient: {util.gini_coefficient:.4f}")
    print(f"  Active experts: {util.active_experts}/{config.num_experts}")