"""
Hierarchical routing controller for BEM (Bolt-on Expert Module).
Implements prefix/chunk/token level routing with uncertainty estimation and EMA smoothing.

This is the core controller system that generates dynamic codes for the generated BEM variant.
Based on TODO.md step B4 specifications and director's notes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
from dataclasses import dataclass
from enum import Enum

class RoutingLevel(Enum):
    """Routing granularity levels for hierarchical control."""
    PREFIX = "prefix"      # Once per sequence (first 128 tokens)
    CHUNK = "chunk"        # Every N tokens (default N=32)
    TOKEN = "token"        # Per-token (MLPs only)

@dataclass
class RoutingState:
    """State container for routing decisions and telemetry."""
    prefix_code: Optional[torch.Tensor] = None      # [batch, code_dim]
    chunk_code: Optional[torch.Tensor] = None       # [batch, code_dim]
    token_code: Optional[torch.Tensor] = None       # [batch, seq_len, code_dim]
    uncertainty: Optional[torch.Tensor] = None      # [batch] or [batch, seq_len]
    entropy: Optional[torch.Tensor] = None          # Routing entropy for telemetry
    utilization: Optional[Dict[str, float]] = None  # Expert utilization stats
    chunk_position: int = 0                         # Current chunk position
    ema_chunk_code: Optional[torch.Tensor] = None   # EMA-smoothed chunk code


class PrefixRouter(nn.Module):
    """
    Prefix router: processes first 128 tokens to generate coarse style/skill prior.
    This provides the global context and domain/style bias for the entire sequence.
    """
    
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_prefix_tokens: int = 128
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
            
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.max_prefix_tokens = max_prefix_tokens
        
        # Attention-based pooling for prefix summary
        self.prefix_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Global query for attention pooling
        self.global_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        
        # Prefix code generator
        self.prefix_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, code_dim)
        )
        
        # Initialize with small weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with careful scaling."""
        for module in self.prefix_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Increased from 0.02
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,  # [batch, seq_len, dim]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate prefix code from initial tokens.
        
        Args:
            hidden_states: Hidden states [batch, seq_len, dim]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            prefix_code: Global prefix code [batch, code_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Take first max_prefix_tokens
        prefix_len = min(seq_len, self.max_prefix_tokens)
        prefix_states = hidden_states[:, :prefix_len, :]  # [batch, prefix_len, dim]
        
        if attention_mask is not None:
            prefix_mask = attention_mask[:, :prefix_len]
        else:
            prefix_mask = None
        
        # Attention-based pooling with global query
        global_query = self.global_query.expand(batch_size, 1, -1)  # [batch, 1, dim]
        
        prefix_summary, _ = self.prefix_attention(
            query=global_query,
            key=prefix_states,
            value=prefix_states,
            key_padding_mask=~prefix_mask.bool() if prefix_mask is not None else None,
            need_weights=False
        )
        
        prefix_summary = prefix_summary.squeeze(1)  # [batch, dim]
        
        # Generate prefix code
        prefix_code = self.prefix_net(prefix_summary)  # [batch, code_dim]
        
        return prefix_code


class ChunkRouter(nn.Module):
    """
    Chunk router: recomputes every N tokens for environment/retrieval reactivity.
    Includes EMA smoothing for stability and side signal integration.
    """
    
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        chunk_size: int = 32,
        ema_decay: float = 0.99,
        side_signal_dim: Optional[int] = None
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
            
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.ema_decay = ema_decay
        
        # Side signal projection if provided
        self.side_signal_dim = side_signal_dim
        if side_signal_dim is not None:
            self.side_projection = nn.Linear(side_signal_dim, input_dim)
        
        # Chunk code generator
        chunk_input_dim = input_dim * 2  # current chunk + prefix summary
        self.chunk_net = nn.Sequential(
            nn.LayerNorm(chunk_input_dim),
            nn.Linear(chunk_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, code_dim)
        )
        
        # EMA buffer for chunk codes (registered as buffer, not parameter)
        self.register_buffer('ema_chunk_code', torch.zeros(1, code_dim))
        self.register_buffer('ema_initialized', torch.tensor(False))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with careful scaling."""
        for module in self.chunk_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Increased from 0.02
                nn.init.zeros_(module.bias)
        
        if hasattr(self, 'side_projection'):
            nn.init.xavier_uniform_(self.side_projection.weight, gain=0.02)
            nn.init.zeros_(self.side_projection.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, dim]
        prefix_summary: torch.Tensor,  # [batch, dim]
        chunk_start: int = 0,
        side_signals: Optional[torch.Tensor] = None,  # [batch, side_dim]
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate chunk code with EMA smoothing.
        
        Args:
            hidden_states: Hidden states [batch, seq_len, dim]
            prefix_summary: Prefix summary [batch, dim]
            chunk_start: Starting position of current chunk
            side_signals: Optional side signals [batch, side_dim]
            training: Whether in training mode (affects EMA update)
            
        Returns:
            chunk_code: Current chunk code [batch, code_dim]
            ema_code: EMA-smoothed chunk code [batch, code_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Extract current chunk
        chunk_end = min(chunk_start + self.chunk_size, seq_len)
        chunk_states = hidden_states[:, chunk_start:chunk_end, :]  # [batch, chunk_len, dim]
        
        # Mean pooling over chunk
        chunk_summary = chunk_states.mean(dim=1)  # [batch, dim]
        
        # Add side signals if provided
        if side_signals is not None and hasattr(self, 'side_projection'):
            side_projected = self.side_projection(side_signals)  # [batch, dim]
            chunk_summary = chunk_summary + side_projected
        
        # Concatenate with prefix summary
        chunk_input = torch.cat([chunk_summary, prefix_summary], dim=-1)  # [batch, 2*dim]
        
        # Generate chunk code
        chunk_code = self.chunk_net(chunk_input)  # [batch, code_dim]
        
        # Update EMA (only during training)
        if training:
            if not self.ema_initialized.item():
                # Initialize EMA with first chunk
                self.ema_chunk_code.copy_(chunk_code.mean(dim=0, keepdim=True))
                self.ema_initialized.copy_(torch.tensor(True))
            else:
                # Update EMA
                current_mean = chunk_code.mean(dim=0, keepdim=True)
                self.ema_chunk_code.mul_(self.ema_decay).add_(current_mean, alpha=1 - self.ema_decay)
        
        # Expand EMA code for batch
        ema_code = self.ema_chunk_code.expand(batch_size, -1)  # [batch, code_dim]
        
        return chunk_code, ema_code


class TokenRouter(nn.Module):
    """
    Token router: fine-grained per-token control (MLPs only for cache compatibility).
    Lightweight design to minimize per-token overhead.
    """
    
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = min(input_dim * 2, 512)  # Smaller for token-level
            
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        
        # Lightweight token code generator
        self.token_net = nn.Sequential(
            nn.LayerNorm(input_dim * 2),  # current token + prefix summary
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, code_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with careful scaling."""
        for module in self.token_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.05)  # Increased from 0.01
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, dim]
        prefix_summary: torch.Tensor   # [batch, dim]
    ) -> torch.Tensor:
        """
        Generate per-token codes.
        
        Args:
            hidden_states: Hidden states [batch, seq_len, dim]
            prefix_summary: Prefix summary [batch, dim]
            
        Returns:
            token_codes: Per-token codes [batch, seq_len, code_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Expand prefix summary for concatenation
        prefix_expanded = prefix_summary.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, dim]
        
        # Concatenate token states with prefix summary
        token_input = torch.cat([hidden_states, prefix_expanded], dim=-1)  # [batch, seq_len, 2*dim]
        
        # Generate per-token codes
        token_codes = self.token_net(token_input)  # [batch, seq_len, code_dim]
        
        return token_codes


class UncertaintyHead(nn.Module):
    """
    Uncertainty estimation head to down-weight deltas when confidence is low.
    Provides adaptive control over the magnitude of BEM interventions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        temperature_init: float = 1.0
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Learnable temperature for uncertainty scaling
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for uncertainty estimation."""
        for module in self.uncertainty_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Increased from 0.02
                nn.init.zeros_(module.bias)
    
    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return torch.exp(self.log_temperature)
    
    def forward(
        self,
        features: torch.Tensor  # [batch, ...] - any shape with feature dim as last
    ) -> torch.Tensor:
        """
        Estimate uncertainty for given features.
        
        Args:
            features: Input features [..., input_dim]
            
        Returns:
            uncertainty: Uncertainty scores [..., 1] in [0, 1]
        """
        # Preserve original shape
        original_shape = features.shape[:-1]
        features_flat = features.view(-1, self.input_dim)
        
        # Compute raw uncertainty
        uncertainty_raw = self.uncertainty_net(features_flat)  # [*, 1]
        
        # Apply temperature scaling
        uncertainty_scaled = uncertainty_raw / self.temperature
        uncertainty_scaled = torch.sigmoid(uncertainty_scaled)
        
        # Reshape back to original
        uncertainty = uncertainty_scaled.view(*original_shape, 1)
        
        return uncertainty


class HierarchicalController(nn.Module):
    """
    Main hierarchical routing controller combining prefix/chunk/token routers.
    This is the core controller from TODO.md step B4.
    
    Features:
    - Three-level hierarchical routing (prefix/chunk/token)
    - Uncertainty estimation with learnable temperature
    - EMA smoothing for chunk codes
    - Side signal integration (retrieval, style tokens, etc.)
    - Comprehensive telemetry and monitoring
    """
    
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        chunk_size: int = 32,
        max_prefix_tokens: int = 128,
        ema_decay: float = 0.99,
        side_signal_dim: Optional[int] = None,
        enable_uncertainty: bool = True,
        enable_token_routing: bool = True,
        code_clamp_value: float = 3.0  # Clamp ||c|| to prevent scale drift
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
            
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.max_prefix_tokens = max_prefix_tokens
        self.enable_uncertainty = enable_uncertainty
        self.enable_token_routing = enable_token_routing
        self.code_clamp_value = code_clamp_value
        
        # Initialize routing components
        self.prefix_router = PrefixRouter(
            input_dim=input_dim,
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_prefix_tokens=max_prefix_tokens
        )
        
        self.chunk_router = ChunkRouter(
            input_dim=input_dim,
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            chunk_size=chunk_size,
            ema_decay=ema_decay,
            side_signal_dim=side_signal_dim
        )
        
        if enable_token_routing:
            self.token_router = TokenRouter(
                input_dim=input_dim,
                code_dim=code_dim,
                hidden_dim=hidden_dim // 2,  # Smaller for token-level
                dropout=dropout
            )
        
        if enable_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 4,  # Small uncertainty head
                dropout=dropout
            )
        
        # Layer normalization for code clamping
        self.code_norm = nn.LayerNorm(code_dim, elementwise_affine=False)
    
    def _clamp_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Clamp code norms to prevent scale drift.
        
        Args:
            codes: Input codes [..., code_dim]
            
        Returns:
            clamped_codes: Norm-clamped codes [..., code_dim]
        """
        # Compute L2 norm over last dimension
        norms = torch.norm(codes, dim=-1, keepdim=True)  # [..., 1]
        
        # Clamp norms to maximum value
        scale_factor = torch.clamp(norms, max=self.code_clamp_value) / (norms + 1e-8)
        
        # Apply scaling
        clamped_codes = codes * scale_factor
        
        return clamped_codes
    
    def _compute_entropy(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of code distribution for telemetry.
        
        Args:
            codes: Input codes [batch, code_dim] or [batch, seq_len, code_dim]
            
        Returns:
            entropy: Code entropy [batch] or [batch, seq_len]
        """
        # Normalize codes to probabilities
        codes_prob = F.softmax(codes, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(codes_prob * torch.log(codes_prob + 1e-8)).sum(dim=-1)
        
        return entropy
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, input_dim]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
        side_signals: Optional[torch.Tensor] = None,  # [batch, side_signal_dim]
        routing_level: Union[RoutingLevel, str] = RoutingLevel.CHUNK,
        chunk_position: int = 0,
        return_routing_state: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, RoutingState]]:
        """
        Main forward pass with hierarchical routing.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, input_dim]
            attention_mask: Attention mask [batch, seq_len]
            side_signals: Optional side signals [batch, side_signal_dim]
            routing_level: Which routing level to use
            chunk_position: Current chunk position (for chunk routing)
            return_routing_state: Whether to return detailed routing state
            
        Returns:
            codes: Generated codes [batch, code_dim] or [batch, seq_len, code_dim]
            routing_state: Detailed routing state (if requested)
        """
        if isinstance(routing_level, str):
            routing_level = RoutingLevel(routing_level)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Always compute prefix summary (needed for all levels)
        prefix_code = self.prefix_router(hidden_states, attention_mask)
        
        # Create prefix summary from hidden states for uncertainty estimation
        # Take mean of first max_prefix_tokens for consistency with prefix router
        prefix_len = min(seq_len, self.max_prefix_tokens)
        prefix_summary = hidden_states[:, :prefix_len, :].mean(dim=1)  # [batch, input_dim]
        
        # Initialize routing state
        routing_state = RoutingState(
            prefix_code=prefix_code,
            chunk_position=chunk_position
        )
        
        # Generate codes based on routing level
        if routing_level == RoutingLevel.PREFIX:
            # Use prefix code only
            codes = prefix_code
            
        elif routing_level == RoutingLevel.CHUNK:
            # Chunk-level routing
            chunk_code, ema_chunk_code = self.chunk_router(
                hidden_states=hidden_states,
                prefix_summary=prefix_summary,
                chunk_start=chunk_position,
                side_signals=side_signals,
                training=self.training
            )
            
            # Use EMA-smoothed chunk code
            codes = ema_chunk_code
            routing_state.chunk_code = chunk_code
            routing_state.ema_chunk_code = ema_chunk_code
            
        elif routing_level == RoutingLevel.TOKEN:
            # Token-level routing (if enabled)
            if not self.enable_token_routing:
                raise ValueError("Token routing is disabled")
            
            token_codes = self.token_router(hidden_states, prefix_summary)
            codes = token_codes
            routing_state.token_code = token_codes
            
        else:
            raise ValueError(f"Unknown routing level: {routing_level}")
        
        # Apply code clamping
        codes = self._clamp_codes(codes)
        
        # Compute uncertainty if enabled
        if self.enable_uncertainty:
            # Use appropriate features for uncertainty estimation
            if routing_level == RoutingLevel.TOKEN:
                uncertainty_features = hidden_states.mean(dim=1)  # [batch, input_dim]
            else:
                uncertainty_features = prefix_summary  # [batch, input_dim]
            
            uncertainty = self.uncertainty_head(uncertainty_features)  # [batch, 1]
            routing_state.uncertainty = uncertainty.squeeze(-1)  # [batch]
            
            # Apply uncertainty scaling to codes
            if routing_level == RoutingLevel.TOKEN:
                # Expand uncertainty for token-level
                uncertainty_expanded = uncertainty.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, 1]
                codes = codes * uncertainty_expanded
            else:
                codes = codes * uncertainty
        
        # Compute telemetry
        entropy = self._compute_entropy(codes)
        routing_state.entropy = entropy.mean() if entropy.dim() > 1 else entropy.mean(dim=0)
        
        # Compute utilization (simplified - track non-zero activations)
        with torch.no_grad():
            active_fraction = (codes.abs() > 0.01).float().mean()
            routing_state.utilization = {
                'active_fraction': active_fraction.item(),
                'code_norm_mean': codes.norm(dim=-1).mean().item(),
                'code_norm_std': codes.norm(dim=-1).std().item()
            }
        
        if return_routing_state:
            return codes, routing_state
        else:
            return codes


# Factory functions for creating controllers

def create_hierarchical_controller(
    model_config: Dict[str, Any],
    controller_config: Optional[Dict[str, Any]] = None
) -> HierarchicalController:
    """
    Factory function to create hierarchical controller from model config.
    
    Args:
        model_config: Model configuration dict
        controller_config: Optional controller-specific config
        
    Returns:
        HierarchicalController instance
    """
    if controller_config is None:
        controller_config = {}
    
    # Extract dimensions from model config
    hidden_size = model_config.get('hidden_size', 768)
    
    # Default controller parameters
    defaults = {
        'input_dim': hidden_size,
        'code_dim': controller_config.get('rank', 8),
        'hidden_dim': hidden_size * 4,
        'dropout': 0.1,
        'chunk_size': 32,
        'max_prefix_tokens': 128,
        'ema_decay': 0.99,
        'enable_uncertainty': True,
        'enable_token_routing': True,
        'code_clamp_value': 3.0
    }
    
    # Override with provided config
    defaults.update(controller_config)
    
    return HierarchicalController(**defaults)


# Analysis utilities for debugging and monitoring

def analyze_routing_behavior(
    controller: HierarchicalController,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_chunks: int = 4
) -> Dict[str, Any]:
    """
    Analyze routing behavior across different levels and chunks.
    
    Args:
        controller: HierarchicalController instance
        hidden_states: Input hidden states [batch, seq_len, dim]
        attention_mask: Optional attention mask
        num_chunks: Number of chunks to analyze
        
    Returns:
        Analysis results dictionary
    """
    controller.eval()
    results = {}
    
    batch_size, seq_len, _ = hidden_states.shape
    chunk_size = controller.chunk_size
    
    with torch.no_grad():
        # Analyze prefix routing
        prefix_codes, prefix_state = controller(
            hidden_states, attention_mask, 
            routing_level=RoutingLevel.PREFIX,
            return_routing_state=True
        )
        
        results['prefix'] = {
            'code_norm': prefix_codes.norm(dim=-1).mean().item(),
            'entropy': prefix_state.entropy.item(),
            'utilization': prefix_state.utilization
        }
        
        # Analyze chunk routing across multiple chunks
        chunk_results = []
        for chunk_idx in range(min(num_chunks, seq_len // chunk_size)):
            chunk_start = chunk_idx * chunk_size
            
            chunk_codes, chunk_state = controller(
                hidden_states, attention_mask,
                routing_level=RoutingLevel.CHUNK,
                chunk_position=chunk_start,
                return_routing_state=True
            )
            
            chunk_results.append({
                'position': chunk_start,
                'code_norm': chunk_codes.norm(dim=-1).mean().item(),
                'entropy': chunk_state.entropy.item(),
                'utilization': chunk_state.utilization,
                'uncertainty': chunk_state.uncertainty.mean().item() if chunk_state.uncertainty is not None else None
            })
        
        results['chunks'] = chunk_results
        
        # Analyze token routing if enabled
        if controller.enable_token_routing:
            token_codes, token_state = controller(
                hidden_states, attention_mask,
                routing_level=RoutingLevel.TOKEN,
                return_routing_state=True
            )
            
            results['tokens'] = {
                'code_norm_mean': token_codes.norm(dim=-1).mean().item(),
                'code_norm_std': token_codes.norm(dim=-1).std().item(),
                'entropy_mean': token_state.entropy.mean().item(),
                'entropy_std': token_state.entropy.std().item(),
                'utilization': token_state.utilization
            }
    
    return results


def compute_routing_stability(
    controller: HierarchicalController,
    hidden_states_list: list,  # List of hidden states tensors
    routing_level: RoutingLevel = RoutingLevel.CHUNK
) -> Dict[str, float]:
    """
    Compute stability metrics for routing decisions across multiple inputs.
    
    Args:
        controller: HierarchicalController instance
        hidden_states_list: List of hidden state tensors
        routing_level: Routing level to analyze
        
    Returns:
        Stability metrics dictionary
    """
    controller.eval()
    
    codes_list = []
    entropies = []
    
    with torch.no_grad():
        for hidden_states in hidden_states_list:
            codes, routing_state = controller(
                hidden_states,
                routing_level=routing_level,
                return_routing_state=True
            )
            
            codes_list.append(codes)
            entropies.append(routing_state.entropy)
    
    # Stack codes for analysis
    stacked_codes = torch.stack(codes_list, dim=0)  # [num_inputs, batch, ...]
    stacked_entropies = torch.stack(entropies, dim=0)  # [num_inputs, ...]
    
    # Compute stability metrics
    code_std = stacked_codes.std(dim=0).mean().item()  # Std across inputs
    entropy_std = stacked_entropies.std(dim=0).mean().item()
    
    # Cosine similarity between consecutive codes
    cosine_sims = []
    for i in range(len(codes_list) - 1):
        codes_flat1 = codes_list[i].flatten(1)  # [batch, features]
        codes_flat2 = codes_list[i+1].flatten(1)
        
        cosine_sim = F.cosine_similarity(codes_flat1, codes_flat2, dim=-1)
        cosine_sims.append(cosine_sim.mean().item())
    
    avg_cosine_similarity = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
    
    return {
        'code_stability': 1.0 - code_std,  # Higher is more stable
        'entropy_stability': 1.0 - entropy_std,
        'cosine_similarity': avg_cosine_similarity,
        'overall_stability': (1.0 - code_std + 1.0 - entropy_std + avg_cosine_similarity) / 3.0
    }