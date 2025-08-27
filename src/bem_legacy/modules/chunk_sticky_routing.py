"""
Chunk-Sticky Routing with Hysteresis (E3)

Implements piecewise-constant routing decisions per N=128 tokens with hysteresis
to prevent frequent routing changes that would violate KV cache efficiency.

Key features:
- Decisions made per chunk (64/128 tokens)
- Hysteresis: only flip if Δlogit > τ=0.7
- Gate entropy and utilization logging
- Cache-safe: no token-wise K/V violations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class ChunkStickyRouter(nn.Module):
    """
    Chunk-sticky routing with hysteresis (E3).
    
    Makes routing decisions once per chunk of N tokens with hysteresis to
    prevent rapid switching that would invalidate KV cache.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7,
        routing_hidden_dim: int = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.chunk_size = chunk_size
        self.hysteresis_tau = hysteresis_tau
        
        if routing_hidden_dim is None:
            routing_hidden_dim = max(input_dim // 2, 64)
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, routing_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(routing_hidden_dim, routing_hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(routing_hidden_dim // 2, num_experts)
        )
        
        # Initialize router to produce balanced initial logits
        self._init_router_weights()
        
        # Buffers for tracking routing history
        self.register_buffer('routing_history', torch.zeros(10, dtype=torch.long))
        self.register_buffer('flip_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('chunk_count', torch.tensor(0, dtype=torch.long))
        
    def _init_router_weights(self):
        """Initialize router to produce balanced logits initially."""
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Final layer bias to ensure balanced initial routing
        nn.init.zeros_(self.router[-1].bias)
        nn.init.normal_(self.router[-1].weight, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor,
        prev_expert_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute chunk-sticky routing with hysteresis.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            prev_expert_indices: Previous expert choices [batch_size] for hysteresis
            
        Returns:
            Dictionary containing routing outputs and metrics
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Calculate number of chunks
        num_chunks = math.ceil(seq_len / self.chunk_size)
        
        # Output tensors
        routing_weights = torch.zeros(batch_size, seq_len, self.num_experts, device=device)
        expert_indices = torch.zeros(batch_size, num_chunks, dtype=torch.long, device=device)
        chunk_logits_all = []
        flip_decisions = []
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, seq_len)
            chunk_len = end_idx - start_idx
            
            # Extract chunk
            chunk_x = x[:, start_idx:end_idx, :]  # [batch_size, chunk_len, input_dim]
            
            # Compute routing logits for chunk (average over tokens)
            chunk_logits = self.router(chunk_x).mean(dim=1)  # [batch_size, num_experts]
            chunk_logits_all.append(chunk_logits)
            
            # Get top expert for each sequence in batch
            top_experts = torch.argmax(chunk_logits, dim=-1)  # [batch_size]
            
            # Apply hysteresis if not first chunk
            if chunk_idx > 0 and prev_expert_indices is not None:
                # Get previous expert choice
                prev_experts = expert_indices[:, chunk_idx - 1]
                
                # Calculate logit difference between current top and previous choice
                current_logits = chunk_logits.gather(1, top_experts.unsqueeze(1)).squeeze(1)
                prev_logits = chunk_logits.gather(1, prev_experts.unsqueeze(1)).squeeze(1) 
                logit_diff = current_logits - prev_logits
                
                # Only switch if difference exceeds hysteresis threshold
                switch_mask = logit_diff > self.hysteresis_tau
                final_experts = torch.where(switch_mask, top_experts, prev_experts)
                
                # Track flips for analysis
                flips = switch_mask.sum().item()
                self.flip_count += flips
                flip_decisions.append(switch_mask)
                
            else:
                final_experts = top_experts
                flip_decisions.append(torch.ones_like(top_experts, dtype=torch.bool))
            
            # Store expert indices for this chunk
            expert_indices[:, chunk_idx] = final_experts
            
            # Create one-hot routing weights for this chunk
            chunk_weights = F.one_hot(final_experts, num_classes=self.num_experts).float()
            
            # Apply routing weights to all tokens in chunk
            routing_weights[:, start_idx:end_idx] = chunk_weights.unsqueeze(1).expand(-1, chunk_len, -1)
        
        # Update chunk count
        self.chunk_count += num_chunks
        
        # Compute routing statistics
        stats = self._compute_routing_stats(
            routing_weights, expert_indices, chunk_logits_all, flip_decisions
        )
        
        return {
            'routing_weights': routing_weights,
            'expert_indices': expert_indices,
            'chunk_logits': torch.stack(chunk_logits_all, dim=1),  # [batch, num_chunks, num_experts]
            **stats
        }
    
    def _compute_routing_stats(
        self,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor, 
        chunk_logits_all: List[torch.Tensor],
        flip_decisions: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute routing statistics for monitoring."""
        
        # Gate entropy per chunk
        chunk_entropies = []
        for logits in chunk_logits_all:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            chunk_entropies.append(entropy)
        
        avg_entropy = torch.stack(chunk_entropies).mean() if chunk_entropies else torch.tensor(0.0)
        
        # Expert utilization (what fraction of chunks use each expert)
        batch_size, num_chunks = expert_indices.shape
        utilization = torch.zeros(self.num_experts, device=expert_indices.device)
        
        for expert_idx in range(self.num_experts):
            expert_usage = (expert_indices == expert_idx).float().mean()
            utilization[expert_idx] = expert_usage
            
        # Flip rate (excluding first chunk which always "flips")
        total_decisions = batch_size * max(1, len(flip_decisions) - 1)
        flip_rate = torch.tensor(0.0, device=expert_indices.device)
        
        if len(flip_decisions) > 1:
            total_flips = sum(mask.sum().item() for mask in flip_decisions[1:])
            flip_rate = torch.tensor(total_flips / total_decisions, device=expert_indices.device)
        
        # Routing concentration (how concentrated are the routing decisions?)
        routing_concentration = torch.norm(utilization, p=2).item()  # L2 norm
        
        return {
            'gate_entropy': avg_entropy,
            'expert_utilization': utilization,
            'flip_rate': flip_rate,
            'routing_concentration': torch.tensor(routing_concentration, device=expert_indices.device),
            'num_chunks': torch.tensor(len(chunk_logits_all), dtype=torch.long, device=expert_indices.device)
        }
    
    def get_cache_alignment_report(self) -> Dict[str, any]:
        """
        Generate report on cache alignment properties.
        
        Returns:
            Report on chunk alignment and cache safety
        """
        return {
            'chunk_size': self.chunk_size,
            'hysteresis_tau': self.hysteresis_tau,
            'cache_safe': True,
            'chunk_aligned': True,
            'token_wise_routing': False,
            'total_flips': self.flip_count.item(),
            'total_chunks': self.chunk_count.item(),
            'historical_flip_rate': self.flip_count.item() / max(1, self.chunk_count.item())
        }


class AdaptiveChunkRouter(ChunkStickyRouter):
    """
    Extension with adaptive chunk sizing based on content.
    
    Still maintains cache safety but can adjust chunk boundaries
    based on semantic breaks or attention patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        base_chunk_size: int = 128,
        min_chunk_size: int = 64,
        max_chunk_size: int = 256,
        **kwargs
    ):
        super().__init__(input_dim, num_experts, base_chunk_size, **kwargs)
        self.base_chunk_size = base_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Boundary detector
        self.boundary_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def _detect_chunk_boundaries(self, x: torch.Tensor) -> List[int]:
        """
        Detect semantic chunk boundaries while respecting size constraints.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            List of boundary positions
        """
        seq_len = x.shape[1]
        
        # Score each position as potential boundary
        boundary_scores = self.boundary_detector(x).squeeze(-1)  # [batch_size, seq_len]
        avg_scores = boundary_scores.mean(dim=0)  # [seq_len]
        
        # Find boundaries respecting size constraints
        boundaries = [0]
        pos = self.min_chunk_size
        
        while pos < seq_len - self.min_chunk_size:
            # Search window for next boundary
            search_end = min(pos + self.max_chunk_size - self.min_chunk_size, seq_len)
            
            # Find highest scoring boundary in valid range
            window_scores = avg_scores[pos:search_end]
            if len(window_scores) > 0:
                best_offset = torch.argmax(window_scores).item()
                boundary_pos = pos + best_offset
            else:
                boundary_pos = min(pos + self.base_chunk_size, seq_len)
                
            boundaries.append(boundary_pos)
            pos = boundary_pos + self.min_chunk_size
        
        if boundaries[-1] < seq_len:
            boundaries.append(seq_len)
            
        return boundaries


def create_chunk_sticky_router(
    input_dim: int,
    num_experts: int,
    chunk_size: int = 128,
    adaptive: bool = False,
    **kwargs
) -> ChunkStickyRouter:
    """
    Factory function for chunk-sticky routers.
    
    Args:
        input_dim: Input feature dimension
        num_experts: Number of experts to route between
        chunk_size: Base chunk size
        adaptive: Whether to use adaptive chunk sizing
        **kwargs: Additional router arguments
        
    Returns:
        ChunkStickyRouter instance
    """
    if adaptive:
        return AdaptiveChunkRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            base_chunk_size=chunk_size,
            **kwargs
        )
    else:
        return ChunkStickyRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            chunk_size=chunk_size,
            **kwargs
        )