"""
Macro-Policy for Agentic Router

Implements the neural policy π_macro that outputs macro-actions for BEM routing:
- State: chunk summary + retrieval/vision/value features + previous action
- Action: (expert_id, scope, span, rank_budget, bias_scale)
- Trust region constraints and action hysteresis
- Support for both BC and PG training phases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, NamedTuple
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class MacroAction:
    """
    Macro-action specification for BEM routing.
    
    Attributes:
        expert_id: Which expert to activate (Code, Formal, Safety, etc.)
        scope: How much context to use (local, global)
        span: Number of chunks to apply this action
        rank_budget: Budget for low-rank decomposition 
        bias_scale: Scale factor for attention bias
    """
    expert_id: int
    scope: str  # 'local' or 'global'
    span: int   # 1-4 chunks
    rank_budget: int  # 8, 16, 32, 64
    bias_scale: float  # 0.1 to 2.0
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert action to tensor representation."""
        scope_id = 0 if self.scope == 'local' else 1
        return torch.tensor([
            self.expert_id,
            scope_id, 
            self.span,
            self.rank_budget,
            self.bias_scale
        ], device=device, dtype=torch.float32)
    
    @classmethod
    def from_tensor(cls, action_tensor: torch.Tensor) -> 'MacroAction':
        """Create action from tensor representation."""
        expert_id = int(action_tensor[0].item())
        scope = 'local' if action_tensor[1].item() < 0.5 else 'global'
        span = int(action_tensor[2].item())
        rank_budget = int(action_tensor[3].item())
        bias_scale = float(action_tensor[4].item())
        
        return cls(
            expert_id=expert_id,
            scope=scope,
            span=span,
            rank_budget=rank_budget,
            bias_scale=bias_scale
        )


class MacroPolicyState(NamedTuple):
    """State representation for macro-policy."""
    chunk_summary: torch.Tensor      # [batch, hidden_dim] - chunk content summary
    retrieval_features: torch.Tensor # [batch, retr_dim] - retrieval quality/relevance 
    vision_features: torch.Tensor    # [batch, vision_dim] - visual features (optional)
    value_features: torch.Tensor     # [batch, value_dim] - safety/constitution scores
    prev_action: Optional[torch.Tensor]  # [batch, action_dim] - previous macro-action
    chunk_index: torch.Tensor        # [batch] - position in sequence


class MacroPolicy(nn.Module):
    """
    Neural macro-policy for agentic BEM routing.
    
    Architecture:
    - State encoder: processes all input features 
    - Action embedder: embeds previous action
    - Policy head: outputs action distribution
    - Value head: for policy gradient training
    """
    
    def __init__(
        self,
        num_experts: int = 3,  # Code, Formal, Safety
        chunk_summary_dim: int = 512,
        retrieval_dim: int = 64,
        vision_dim: int = 768,
        value_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        hysteresis_tau: float = 0.5,
        max_span: int = 4
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.hysteresis_tau = hysteresis_tau
        self.max_span = max_span
        
        # Input projections
        self.chunk_proj = nn.Linear(chunk_summary_dim, hidden_dim)
        self.retrieval_proj = nn.Linear(retrieval_dim, hidden_dim // 4)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim // 4)
        self.value_proj = nn.Linear(value_dim, hidden_dim // 4)
        self.action_proj = nn.Linear(5, hidden_dim // 4)  # 5 action components
        self.pos_proj = nn.Linear(1, hidden_dim // 4)
        
        # State encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Policy heads
        self.expert_head = nn.Linear(hidden_dim, num_experts)
        self.scope_head = nn.Linear(hidden_dim, 2)  # local vs global
        self.span_head = nn.Linear(hidden_dim, max_span)  # 1-4 chunks
        self.rank_head = nn.Linear(hidden_dim, 4)  # 8, 16, 32, 64
        self.bias_head = nn.Linear(hidden_dim, 1)   # continuous bias scale
        
        # Value head for PG training
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Action history for hysteresis
        self.register_buffer('action_history', torch.zeros(10, 5))  # last 10 actions
        self.register_buffer('hysteresis_count', torch.tensor(0, dtype=torch.long))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for balanced initial policy."""
        # Initialize policy heads with small weights
        for head in [self.expert_head, self.scope_head, self.span_head, self.rank_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
        
        # Bias head centered around 1.0
        nn.init.zeros_(self.bias_head.weight)
        nn.init.constant_(self.bias_head.bias, 0.0)  # will be sigmoid'd and scaled
        
        # Value head small initialization
        nn.init.xavier_uniform_(self.value_head.weight, gain=0.1)
        nn.init.zeros_(self.value_head.bias)
    
    def encode_state(self, state: MacroPolicyState) -> torch.Tensor:
        """
        Encode policy state into hidden representation.
        
        Args:
            state: MacroPolicyState with all input features
            
        Returns:
            Encoded state tensor [batch, hidden_dim]
        """
        batch_size = state.chunk_summary.shape[0]
        device = state.chunk_summary.device
        
        # Project all input components
        chunk_enc = self.chunk_proj(state.chunk_summary)  # [batch, hidden_dim]
        retr_enc = self.retrieval_proj(state.retrieval_features)  # [batch, hidden_dim//4]
        vision_enc = self.vision_proj(state.vision_features)  # [batch, hidden_dim//4] 
        value_enc = self.value_proj(state.value_features)  # [batch, hidden_dim//4]
        pos_enc = self.pos_proj(state.chunk_index.float().unsqueeze(1))  # [batch, hidden_dim//4]
        
        # Handle previous action
        if state.prev_action is not None:
            action_enc = self.action_proj(state.prev_action)  # [batch, hidden_dim//4]
        else:
            action_enc = torch.zeros(batch_size, self.hidden_dim // 4, device=device)
        
        # Concatenate auxiliary features
        aux_features = torch.cat([retr_enc, vision_enc, value_enc, action_enc, pos_enc], dim=-1)
        
        # Combine chunk summary with auxiliary features
        combined_state = chunk_enc + F.linear(
            aux_features, 
            torch.eye(self.hidden_dim, device=device)[:, :aux_features.shape[-1]]
        )
        
        # Apply transformer encoding
        # Add sequence dimension for transformer (treat as seq_len=1)
        state_seq = combined_state.unsqueeze(1)  # [batch, 1, hidden_dim]
        encoded_seq = self.state_encoder(state_seq)  # [batch, 1, hidden_dim]
        encoded_state = encoded_seq.squeeze(1)  # [batch, hidden_dim]
        
        return encoded_state
    
    def forward(
        self,
        state: MacroPolicyState,
        sample: bool = True,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of macro-policy.
        
        Args:
            state: Current policy state
            sample: Whether to sample actions or use argmax
            temperature: Temperature for action sampling
            
        Returns:
            Dictionary with action distributions and sampled actions
        """
        # Encode state
        encoded_state = self.encode_state(state)  # [batch, hidden_dim]
        
        # Compute action logits
        expert_logits = self.expert_head(encoded_state) / temperature
        scope_logits = self.scope_head(encoded_state) / temperature  
        span_logits = self.span_head(encoded_state) / temperature
        rank_logits = self.rank_head(encoded_state) / temperature
        bias_raw = self.bias_head(encoded_state)  # Raw bias output
        
        # Convert rank logits to actual rank values
        rank_values = torch.tensor([8, 16, 32, 64], device=encoded_state.device, dtype=torch.float32)
        
        # Sample or select actions
        if sample:
            expert_dist = torch.distributions.Categorical(logits=expert_logits)
            expert_actions = expert_dist.sample()
            expert_log_probs = expert_dist.log_prob(expert_actions)
            
            scope_dist = torch.distributions.Categorical(logits=scope_logits)
            scope_actions = scope_dist.sample()
            scope_log_probs = scope_dist.log_prob(scope_actions)
            
            span_dist = torch.distributions.Categorical(logits=span_logits)
            span_actions = span_dist.sample() + 1  # 1-4
            span_log_probs = span_dist.log_prob(span_actions - 1)
            
            rank_dist = torch.distributions.Categorical(logits=rank_logits)
            rank_indices = rank_dist.sample()
            rank_actions = rank_values[rank_indices]
            rank_log_probs = rank_dist.log_prob(rank_indices)
            
        else:
            expert_actions = torch.argmax(expert_logits, dim=-1)
            expert_log_probs = F.log_softmax(expert_logits, dim=-1).gather(1, expert_actions.unsqueeze(1)).squeeze(1)
            
            scope_actions = torch.argmax(scope_logits, dim=-1)
            scope_log_probs = F.log_softmax(scope_logits, dim=-1).gather(1, scope_actions.unsqueeze(1)).squeeze(1)
            
            span_actions = torch.argmax(span_logits, dim=-1) + 1
            span_log_probs = F.log_softmax(span_logits, dim=-1).gather(1, (span_actions - 1).unsqueeze(1)).squeeze(1)
            
            rank_indices = torch.argmax(rank_logits, dim=-1)
            rank_actions = rank_values[rank_indices]
            rank_log_probs = F.log_softmax(rank_logits, dim=-1).gather(1, rank_indices.unsqueeze(1)).squeeze(1)
        
        # Bias scale (continuous, sigmoid to [0.1, 2.0] range)
        bias_actions = torch.sigmoid(bias_raw).squeeze(-1) * 1.9 + 0.1
        
        # Apply hysteresis if we have previous actions
        if state.prev_action is not None:
            expert_actions, scope_actions, span_actions, rank_actions, bias_actions = self._apply_hysteresis(
                expert_actions, scope_actions, span_actions, rank_actions, bias_actions,
                state.prev_action, expert_logits, scope_logits, span_logits, rank_logits, bias_raw
            )
        
        # Combine log probs
        total_log_probs = expert_log_probs + scope_log_probs + span_log_probs + rank_log_probs
        
        # Compute state value
        state_values = self.value_head(encoded_state).squeeze(-1)
        
        # Create action tensor
        actions = torch.stack([
            expert_actions.float(),
            scope_actions.float(),
            span_actions.float(), 
            rank_actions.float(),
            bias_actions
        ], dim=-1)  # [batch, 5]
        
        return {
            'actions': actions,
            'log_probs': total_log_probs,
            'state_values': state_values,
            'expert_logits': expert_logits,
            'scope_logits': scope_logits,
            'span_logits': span_logits,
            'rank_logits': rank_logits,
            'bias_raw': bias_raw,
            'encoded_state': encoded_state
        }
    
    def _apply_hysteresis(
        self,
        expert_actions: torch.Tensor,
        scope_actions: torch.Tensor,
        span_actions: torch.Tensor,
        rank_actions: torch.Tensor,
        bias_actions: torch.Tensor,
        prev_action: torch.Tensor,
        expert_logits: torch.Tensor,
        scope_logits: torch.Tensor,
        span_logits: torch.Tensor,
        rank_logits: torch.Tensor,
        bias_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply action hysteresis to prevent thrashing.
        
        Only flip actions if the improvement exceeds hysteresis threshold τ.
        """
        batch_size = expert_actions.shape[0]
        device = expert_actions.device
        
        # Extract previous actions
        prev_expert = prev_action[:, 0].long()
        prev_scope = prev_action[:, 1].long()
        prev_span = (prev_action[:, 2] - 1).long()  # Convert back to 0-3
        prev_rank_indices = torch.searchsorted(
            torch.tensor([8, 16, 32, 64], device=device), prev_action[:, 3]
        ).clamp(0, 3)
        
        # Compute logit differences
        current_expert_logits = expert_logits.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
        prev_expert_logits = expert_logits.gather(1, prev_expert.unsqueeze(1)).squeeze(1)
        expert_diff = current_expert_logits - prev_expert_logits
        
        current_scope_logits = scope_logits.gather(1, scope_actions.unsqueeze(1)).squeeze(1)
        prev_scope_logits = scope_logits.gather(1, prev_scope.unsqueeze(1)).squeeze(1)
        scope_diff = current_scope_logits - prev_scope_logits
        
        current_span_logits = span_logits.gather(1, (span_actions - 1).unsqueeze(1)).squeeze(1)
        prev_span_logits = span_logits.gather(1, prev_span.unsqueeze(1)).squeeze(1)
        span_diff = current_span_logits - prev_span_logits
        
        current_rank_indices = torch.searchsorted(
            torch.tensor([8, 16, 32, 64], device=device), rank_actions
        ).clamp(0, 3)
        current_rank_logits = rank_logits.gather(1, current_rank_indices.unsqueeze(1)).squeeze(1)
        prev_rank_logits = rank_logits.gather(1, prev_rank_indices.unsqueeze(1)).squeeze(1)
        rank_diff = current_rank_logits - prev_rank_logits
        
        # Combined improvement score (weighted)
        total_improvement = (
            0.4 * expert_diff +      # Expert choice is most important
            0.2 * scope_diff +       # Scope less critical
            0.2 * span_diff +        # Span less critical  
            0.2 * rank_diff          # Rank less critical
        )
        
        # Apply hysteresis: only flip if improvement > τ
        should_flip = total_improvement > self.hysteresis_tau
        
        # For actions that shouldn't flip, revert to previous
        final_expert = torch.where(should_flip, expert_actions, prev_expert)
        final_scope = torch.where(should_flip, scope_actions, prev_scope)
        final_span = torch.where(should_flip, span_actions, (prev_span + 1).float())
        final_rank = torch.where(should_flip, rank_actions, prev_action[:, 3])
        
        # For bias, use a continuous hysteresis (smaller threshold)
        bias_diff = torch.abs(bias_actions - prev_action[:, 4])
        bias_should_change = bias_diff > (self.hysteresis_tau * 0.2)  # Smaller threshold for continuous
        final_bias = torch.where(bias_should_change, bias_actions, prev_action[:, 4])
        
        # Update hysteresis stats
        flip_count = should_flip.sum().item()
        self.hysteresis_count += flip_count
        
        if flip_count > 0:
            logger.debug(f"Hysteresis: {flip_count}/{batch_size} actions flipped, "
                        f"avg improvement: {total_improvement[should_flip].mean():.3f}")
        
        return final_expert, final_scope, final_span, final_rank, final_bias
    
    def get_hysteresis_stats(self) -> Dict[str, float]:
        """Get hysteresis statistics for monitoring."""
        return {
            'total_hysteresis_blocks': self.hysteresis_count.item(),
            'hysteresis_tau': self.hysteresis_tau
        }


def create_macro_policy(
    config: Dict,
    num_experts: int = 3
) -> MacroPolicy:
    """
    Factory function to create MacroPolicy from configuration.
    
    Args:
        config: Configuration dictionary
        num_experts: Number of expert types
        
    Returns:
        Configured MacroPolicy instance
    """
    return MacroPolicy(
        num_experts=num_experts,
        chunk_summary_dim=config.get('chunk_summary_dim', 512),
        retrieval_dim=config.get('retrieval_dim', 64),
        vision_dim=config.get('vision_dim', 768), 
        value_dim=config.get('value_dim', 32),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1),
        hysteresis_tau=config.get('hysteresis_tau', 0.5),
        max_span=config.get('max_span', 4)
    )