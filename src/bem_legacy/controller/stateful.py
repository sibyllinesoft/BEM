"""
F5.1 - Stateful Router with GRU/SSM over chunk summaries.

Adds a tiny recurrent/SSM head over chunk summaries to stabilize routing,
reduce flip thrash, and harness long-range structure.

Mechanism: s_t = SSM(LN(summary_t), s_{t-1}); gates_t = MLP([features_t, s_t])
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class StatefulRouterConfig:
    """Configuration for stateful router."""
    d_feat: int = 512          # Feature dimension
    d_state: int = 64          # State dimension (kept small for efficiency)
    code_dim: int = 8          # Output code dimension
    chunk_size: int = 128      # Chunk size for processing
    dropout: float = 0.1       # Dropout rate
    flip_penalty_beta: float = 0.01  # Penalty for routing flips
    hidden_norm_clamp: float = 5.0   # Clamp for hidden state norms
    use_ssm: bool = False      # Use SSM instead of GRU
    hysteresis_tau: float = 0.7 # Hysteresis threshold


class S4Lite(nn.Module):
    """Lightweight S4 implementation for sequence modeling."""
    
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Diagonal state space parameters
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Input and output projections
        self.in_proj = nn.Linear(d_model, d_state * 2)  # B and C
        self.out_proj = nn.Linear(d_state, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters following S4 conventions."""
        # Initialize A to be stable (negative real parts)
        with torch.no_grad():
            A_init = torch.randn(self.d_state)
            A_init = -torch.exp(A_init)  # Ensure negative
            self.A_log.copy_(torch.log(-A_init))  # Store log for stability
            
        # Initialize projections
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of S4Lite.
        
        Args:
            x: Input sequence [batch, seq_len, d_model]
            state: Previous state [batch, d_state] or None
            
        Returns:
            output: Output sequence [batch, seq_len, d_model]
            final_state: Final state [batch, d_state]
        """
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            state = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
            
        # Get A matrix (ensure it's negative)
        A = -torch.exp(self.A_log)  # [d_state]
        
        # Project input to get B and C
        BC = self.in_proj(x)  # [batch, seq_len, 2*d_state]
        B, C = BC.chunk(2, dim=-1)  # Each [batch, seq_len, d_state]
        
        # Apply state space recursion
        states = []
        current_state = state
        
        for t in range(seq_len):
            # Update state: x_{t+1} = A * x_t + B * u_t
            current_state = A * current_state + B[:, t]  # [batch, d_state]
            states.append(current_state)
            
        states = torch.stack(states, dim=1)  # [batch, seq_len, d_state]
        
        # Output: y_t = C * x_t + D * u_t
        output = self.out_proj(states) + self.D * x  # [batch, seq_len, d_model]
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
        final_state = states[:, -1]  # [batch, d_state]
        
        return output, final_state


class StatefulRouter(nn.Module):
    """
    Stateful router that maintains memory across chunks to reduce flip thrash.
    
    Features:
    - GRU or S4-lite for maintaining state across chunks
    - Hysteresis mechanism to prevent excessive switching
    - Flip penalty during training to encourage stability
    - Hidden state norm clamping for numerical stability
    """
    
    def __init__(self, config: StatefulRouterConfig):
        super().__init__()
        self.config = config
        
        # State transition model (GRU or S4-lite)
        if config.use_ssm:
            self.state_model = S4Lite(
                d_model=config.d_feat,
                d_state=config.d_state,
                dropout=config.dropout
            )
            self.state_dim = config.d_feat  # S4 outputs same dimension
        else:
            self.state_model = nn.GRU(
                input_size=config.d_feat,
                hidden_size=config.d_state,
                batch_first=True,
                dropout=config.dropout if config.dropout > 0 else 0
            )
            self.state_dim = config.d_state
            
        # Layer normalization for input features
        self.feature_norm = nn.LayerNorm(config.d_feat)
        
        # MLP for generating codes from [features, state]
        self.code_mlp = nn.Sequential(
            nn.LayerNorm(config.d_feat + self.state_dim),
            nn.Linear(config.d_feat + self.state_dim, 4 * config.d_feat),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.d_feat, config.code_dim)
        )
        
        # Buffer to track previous routing decisions for metrics
        self.register_buffer('prev_codes', torch.zeros(1, config.code_dim))
        self.register_buffer('flip_count', torch.zeros(1))
        self.register_buffer('total_chunks', torch.zeros(1))
        
    def _compute_chunk_summaries(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute chunk-level summaries from token features.
        
        Args:
            features: Token features [batch, seq_len, d_feat]
            
        Returns:
            chunk_summaries: [batch, num_chunks, d_feat]
        """
        batch_size, seq_len, d_feat = features.shape
        chunk_size = self.config.chunk_size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        # Pad sequence if needed
        pad_len = num_chunks * chunk_size - seq_len
        if pad_len > 0:
            features = F.pad(features, (0, 0, 0, pad_len))
            
        # Reshape to chunks and compute mean
        features_chunked = features.view(batch_size, num_chunks, chunk_size, d_feat)
        chunk_summaries = features_chunked.mean(dim=2)  # [batch, num_chunks, d_feat]
        
        return chunk_summaries
        
    def _expand_chunk_codes_to_tokens(
        self, 
        chunk_codes: torch.Tensor, 
        original_seq_len: int
    ) -> torch.Tensor:
        """
        Expand chunk-level codes to token-level codes.
        
        Args:
            chunk_codes: [batch, num_chunks, code_dim]
            original_seq_len: Original sequence length before chunking
            
        Returns:
            token_codes: [batch, seq_len, code_dim]
        """
        batch_size, num_chunks, code_dim = chunk_codes.shape
        chunk_size = self.config.chunk_size
        
        # Repeat each chunk code for all tokens in the chunk
        token_codes = chunk_codes.unsqueeze(2).repeat(1, 1, chunk_size, 1)
        token_codes = token_codes.view(batch_size, num_chunks * chunk_size, code_dim)
        
        # Truncate to original sequence length
        token_codes = token_codes[:, :original_seq_len]
        
        return token_codes
        
    def _compute_flip_penalty(self, codes: torch.Tensor, prev_codes: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for routing flips to encourage stability.
        
        Args:
            codes: Current codes [batch, num_chunks, code_dim]
            prev_codes: Previous codes [batch, num_chunks, code_dim]
            
        Returns:
            flip_penalty: Scalar penalty term
        """
        if prev_codes.shape != codes.shape:
            return torch.tensor(0.0, device=codes.device, dtype=codes.dtype)
            
        # Compute L2 distance between consecutive chunk codes
        code_diff = torch.norm(codes[:, 1:] - codes[:, :-1], p=2, dim=-1)  # [batch, num_chunks-1]
        
        # Also penalize difference from previous batch's final codes
        if prev_codes.numel() > 0:
            first_diff = torch.norm(codes[:, 0] - prev_codes[:, -1], p=2, dim=-1)  # [batch]
            total_diff = torch.cat([first_diff.unsqueeze(1), code_diff], dim=1)  # [batch, num_chunks]
        else:
            total_diff = code_diff
            
        return self.config.flip_penalty_beta * total_diff.mean()
        
    def forward(
        self, 
        features: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of stateful router.
        
        Args:
            features: Input token features [batch, seq_len, d_feat]
            hidden_state: Previous hidden state (optional)
            return_metrics: Whether to return routing metrics
            
        Returns:
            Dictionary containing:
                - codes: Routing codes [batch, seq_len, code_dim]
                - hidden_state: Updated hidden state
                - chunk_codes: Chunk-level codes [batch, num_chunks, code_dim]
                - flip_penalty: Penalty for routing instability (if training)
                - metrics: Routing metrics (if requested)
        """
        batch_size, seq_len, d_feat = features.shape
        
        # Normalize input features
        features_norm = self.feature_norm(features)
        
        # Compute chunk summaries
        chunk_summaries = self._compute_chunk_summaries(features_norm)  # [batch, num_chunks, d_feat]
        num_chunks = chunk_summaries.shape[1]
        
        # Pass chunk summaries through state model
        if self.config.use_ssm:
            # S4-lite processes the entire sequence
            state_output, final_hidden = self.state_model(chunk_summaries, hidden_state)
        else:
            # GRU processes sequence
            if hidden_state is None:
                hidden_state = torch.zeros(
                    1, batch_size, self.config.d_state,
                    device=features.device, dtype=features.dtype
                )
            state_output, final_hidden = self.state_model(chunk_summaries, hidden_state)
            
        # Clamp hidden state norms for numerical stability
        if self.config.hidden_norm_clamp > 0:
            if self.config.use_ssm:
                final_hidden = torch.clamp(
                    final_hidden, 
                    -self.config.hidden_norm_clamp, 
                    self.config.hidden_norm_clamp
                )
            else:
                final_hidden = torch.clamp(
                    final_hidden, 
                    -self.config.hidden_norm_clamp, 
                    self.config.hidden_norm_clamp
                )
        
        # Combine chunk summaries with state output
        if self.config.use_ssm:
            combined_features = torch.cat([chunk_summaries, state_output], dim=-1)
        else:
            # For GRU, state_output is already the processed features
            # Concatenate original features with GRU outputs
            combined_features = torch.cat([chunk_summaries, state_output], dim=-1)
            
        # Generate codes via MLP
        chunk_codes = self.code_mlp(combined_features)  # [batch, num_chunks, code_dim]
        
        # Expand chunk codes to token level
        token_codes = self._expand_chunk_codes_to_tokens(chunk_codes, seq_len)
        
        # Compute flip penalty for training
        flip_penalty = torch.tensor(0.0, device=features.device, dtype=features.dtype)
        if self.training and self.prev_codes.numel() > 0:
            # Reshape prev_codes to match current batch if needed
            if self.prev_codes.shape[0] == batch_size:
                flip_penalty = self._compute_flip_penalty(chunk_codes, self.prev_codes.view(batch_size, -1, self.config.code_dim))
        
        # Update tracking buffers
        if self.training:
            with torch.no_grad():
                self.prev_codes = chunk_codes.detach().clone()
                
                # Update flip counting metrics
                if hasattr(self, '_last_chunk_codes'):
                    code_changes = torch.norm(
                        chunk_codes.argmax(dim=-1).float() - self._last_chunk_codes.argmax(dim=-1).float(),
                        p=1, dim=1
                    ).sum()
                    self.flip_count += code_changes
                    
                self._last_chunk_codes = chunk_codes.detach().clone()
                self.total_chunks += num_chunks * batch_size
        
        result = {
            'codes': token_codes,
            'hidden_state': final_hidden,
            'chunk_codes': chunk_codes,
            'flip_penalty': flip_penalty
        }
        
        if return_metrics:
            metrics = self._compute_metrics(chunk_codes)
            result['metrics'] = metrics
            
        return result
        
    def _compute_metrics(self, chunk_codes: torch.Tensor) -> Dict[str, float]:
        """Compute routing metrics for analysis."""
        with torch.no_grad():
            # Entropy of routing distribution
            probs = F.softmax(chunk_codes, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            
            # Flip rate
            flip_rate = (self.flip_count / (self.total_chunks + 1e-8)).item()
            
            # Code magnitude statistics
            code_norm = torch.norm(chunk_codes, p=2, dim=-1).mean().item()
            
            return {
                'routing_entropy': entropy,
                'flip_rate': flip_rate,
                'code_norm': code_norm,
                'total_chunks': self.total_chunks.item()
            }
            
    def reset_state(self):
        """Reset internal state buffers."""
        self.prev_codes.zero_()
        self.flip_count.zero_()
        self.total_chunks.zero_()
        if hasattr(self, '_last_chunk_codes'):
            delattr(self, '_last_chunk_codes')


class StatefulBEMRouter(nn.Module):
    """
    Enhanced BEM router that integrates stateful routing with the existing
    chunk-sticky routing mechanism from v1.1.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        config: Optional[StatefulRouterConfig] = None,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7
    ):
        super().__init__()
        
        if config is None:
            config = StatefulRouterConfig(
                d_feat=input_dim,
                code_dim=num_experts,
                chunk_size=chunk_size
            )
        
        self.config = config
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.chunk_size = chunk_size
        self.hysteresis_tau = hysteresis_tau
        
        # Stateful router core
        self.stateful_router = StatefulRouter(config)
        
        # Convert codes to expert probabilities
        self.code_to_expert = nn.Linear(config.code_dim, num_experts)
        
        # Store previous routing decisions for hysteresis
        self.register_buffer('prev_routing', torch.zeros(1, dtype=torch.long))
        
    def forward(
        self, 
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute stateful chunk-sticky routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            hidden_state: Previous hidden state (optional)
            
        Returns:
            routing_weights: Soft routing weights [batch_size, seq_len, num_experts]
            expert_indices: Hard expert indices [batch_size, num_chunks]
            aux_info: Auxiliary information (codes, metrics, etc.)
        """
        batch_size, seq_len, _ = x.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Get stateful codes
        router_output = self.stateful_router(x, hidden_state, return_metrics=True)
        codes = router_output['codes']  # [batch_size, seq_len, code_dim]
        
        # Convert codes to expert logits
        expert_logits = self.code_to_expert(codes)  # [batch_size, seq_len, num_experts]
        
        # Apply softmax to get probabilities
        routing_probs = F.softmax(expert_logits, dim=-1)
        
        # Chunk-wise routing decisions with hysteresis
        expert_indices = torch.zeros(batch_size, num_chunks, dtype=torch.long, device=x.device)
        routing_weights = torch.zeros_like(routing_probs)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, seq_len)
            
            # Average logits over chunk
            chunk_logits = expert_logits[:, start_idx:end_idx].mean(dim=1)  # [batch_size, num_experts]
            
            # Get top expert for this chunk
            chunk_expert = torch.argmax(chunk_logits, dim=-1)  # [batch_size]
            
            # Apply hysteresis - only switch if difference is significant
            if chunk_idx > 0:
                prev_expert = expert_indices[:, chunk_idx - 1]
                logit_diff = chunk_logits.gather(1, chunk_expert.unsqueeze(1)) - \
                           chunk_logits.gather(1, prev_expert.unsqueeze(1))
                
                # Only switch if logit difference exceeds threshold
                switch_mask = logit_diff.squeeze(1) > self.hysteresis_tau
                chunk_expert = torch.where(switch_mask, chunk_expert, prev_expert)
            
            expert_indices[:, chunk_idx] = chunk_expert
            
            # Set routing weights for this chunk (one-hot)
            chunk_weights = F.one_hot(chunk_expert, num_classes=self.num_experts).float()
            routing_weights[:, start_idx:end_idx] = chunk_weights.unsqueeze(1).expand(-1, end_idx - start_idx, -1)
        
        aux_info = {
            'codes': codes,
            'hidden_state': router_output['hidden_state'],
            'chunk_codes': router_output['chunk_codes'],
            'flip_penalty': router_output['flip_penalty'],
            'metrics': router_output.get('metrics', {})
        }
        
        return routing_weights, expert_indices, aux_info


def create_stateful_router_config(**kwargs) -> StatefulRouterConfig:
    """Factory function to create StatefulRouterConfig with validation."""
    return StatefulRouterConfig(**kwargs)