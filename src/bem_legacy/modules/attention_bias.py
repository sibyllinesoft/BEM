"""
Attention-Logit Bias (E4) Implementation

Adds bias(z) to attention scores based on retrieval features without modifying
the K/V representations. This preserves cache safety while allowing retrieval
context to influence attention patterns.

Key properties:
- Cache-safe: no K/V modifications
- Additive bias to attention logits before softmax
- Based on retrieval features
- Can be position-aware or global
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class AttentionLogitBias(nn.Module):
    """
    Attention-logit bias from retrieval features (E4).
    
    Adds bias(z) to attention scores without modifying K/V cache.
    This allows retrieval context to influence attention while maintaining
    cache efficiency.
    """
    
    def __init__(
        self,
        retrieval_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 1,
        position_aware: bool = True,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.retrieval_dim = retrieval_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.position_aware = position_aware
        self.max_seq_len = max_seq_len
        
        # Main bias network
        self.bias_net = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads if num_heads > 1 else 1)
        )
        
        # Position-aware components (optional)
        if position_aware:
            # Position encoding for bias modulation
            self.pos_encoding = nn.Parameter(
                torch.randn(max_seq_len, hidden_dim // 2) * 0.02
            )
            
            # Position modulation network
            self.pos_mod_net = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        
        # Temperature parameter for bias scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to produce small initial biases."""
        for module in [self.bias_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
        
        # Final layer should start very small
        nn.init.normal_(self.bias_net[-1].weight, std=0.001)
        nn.init.zeros_(self.bias_net[-1].bias)
    
    def forward(
        self,
        retrieval_features: torch.Tensor,
        seq_len: int,
        head_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute attention bias from retrieval features.
        
        Args:
            retrieval_features: [batch_size, retrieval_dim] or [batch_size, seq_len, retrieval_dim]
            seq_len: Sequence length for output bias
            head_dim: If provided, return bias for specific head dimension
            
        Returns:
            bias: Attention bias to add to logits
                 Shape depends on inputs:
                 - Global: [batch_size, 1, 1, 1] (broadcast to all positions)
                 - Per-position: [batch_size, num_heads, seq_len, seq_len] or similar
        """
        batch_size = retrieval_features.shape[0]
        device = retrieval_features.device
        
        # Handle different input shapes
        if retrieval_features.dim() == 2:
            # Global retrieval features [batch_size, retrieval_dim]
            global_features = retrieval_features
            per_position = False
        else:
            # Per-position retrieval features [batch_size, seq_len, retrieval_dim]
            global_features = retrieval_features.mean(dim=1)  # [batch_size, retrieval_dim]
            per_position = True
        
        # Compute base bias
        base_bias = self.bias_net(global_features)  # [batch_size, num_heads or 1]
        
        if self.num_heads > 1:
            bias_shape = [batch_size, self.num_heads, 1, 1]
            bias = base_bias.view(batch_size, self.num_heads, 1, 1)
        else:
            bias_shape = [batch_size, 1, 1, 1]
            bias = base_bias.view(batch_size, 1, 1, 1)
        
        # Apply position-aware modulation if enabled
        if self.position_aware and per_position:
            pos_bias = self._compute_position_bias(
                retrieval_features, seq_len, batch_size, device
            )
            bias = bias + pos_bias
        
        # Scale by temperature
        bias = bias * self.temperature
        
        return bias
    
    def _compute_position_bias(
        self,
        retrieval_features: torch.Tensor,
        seq_len: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute position-aware bias modulation.
        
        Args:
            retrieval_features: [batch_size, seq_len, retrieval_dim]
            seq_len: Sequence length
            batch_size: Batch size
            device: Device
            
        Returns:
            Position-modulated bias
        """
        # Get position encodings for this sequence length
        pos_enc = self.pos_encoding[:seq_len]  # [seq_len, hidden_dim // 2]
        
        # Expand for batch
        pos_enc_batch = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq_len, hidden_dim//2]
        
        # Compute position modulation
        pos_mod = self.pos_mod_net(pos_enc_batch)  # [batch_size, seq_len, 1]
        
        # Create attention matrix bias
        # Simple approach: use position modulation as query bias
        if self.num_heads > 1:
            pos_bias = pos_mod.view(batch_size, 1, seq_len, 1)  # Broadcast over heads and keys
            pos_bias = pos_bias.expand(-1, self.num_heads, -1, seq_len)
        else:
            pos_bias = pos_mod.view(batch_size, 1, seq_len, 1)
            pos_bias = pos_bias.expand(-1, 1, -1, seq_len)
        
        return pos_bias * 0.1  # Small modulation factor


class MultiScaleAttentionBias(nn.Module):
    """
    Multi-scale attention bias that can operate at different granularities.
    
    Provides biases at:
    - Global level (same bias for all positions)  
    - Chunk level (same bias within chunks)
    - Token level (different bias per token)
    """
    
    def __init__(
        self,
        retrieval_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 1,
        chunk_size: int = 128,
        scales: list = ['global', 'chunk', 'token']
    ):
        super().__init__()
        self.retrieval_dim = retrieval_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.scales = scales
        
        # Bias networks for different scales
        self.bias_networks = nn.ModuleDict()
        
        if 'global' in scales:
            self.bias_networks['global'] = AttentionLogitBias(
                retrieval_dim, hidden_dim, num_heads, position_aware=False
            )
        
        if 'chunk' in scales:
            self.bias_networks['chunk'] = AttentionLogitBias(
                retrieval_dim, hidden_dim, num_heads, position_aware=False  
            )
        
        if 'token' in scales:
            self.bias_networks['token'] = AttentionLogitBias(
                retrieval_dim, hidden_dim, num_heads, position_aware=True
            )
        
        # Scale mixing weights
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
    
    def forward(
        self,
        retrieval_features: torch.Tensor,
        seq_len: int,
        chunk_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale attention bias.
        
        Args:
            retrieval_features: Global retrieval features [batch_size, retrieval_dim]
            seq_len: Sequence length
            chunk_features: Optional chunk-level features [batch_size, num_chunks, retrieval_dim]
            
        Returns:
            Dictionary with bias components and combined bias
        """
        batch_size = retrieval_features.shape[0]
        device = retrieval_features.device
        
        biases = {}
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        combined_bias = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, device=device
        )
        
        for i, scale in enumerate(self.scales):
            if scale == 'global':
                bias = self.bias_networks[scale](retrieval_features, seq_len)
                biases[scale] = bias
                combined_bias += scale_weights[i] * bias
                
            elif scale == 'chunk' and chunk_features is not None:
                # Apply chunk-level bias
                num_chunks = chunk_features.shape[1]
                chunk_bias = torch.zeros_like(combined_bias)
                
                for chunk_idx in range(num_chunks):
                    start_pos = chunk_idx * self.chunk_size
                    end_pos = min(start_pos + self.chunk_size, seq_len)
                    
                    chunk_feat = chunk_features[:, chunk_idx]  # [batch_size, retrieval_dim]
                    chunk_b = self.bias_networks[scale](chunk_feat, end_pos - start_pos)
                    
                    # Apply to chunk positions
                    chunk_bias[:, :, start_pos:end_pos, start_pos:end_pos] = chunk_b
                
                biases[scale] = chunk_bias
                combined_bias += scale_weights[i] * chunk_bias
                
            elif scale == 'token':
                # Token-level features needed
                token_features = retrieval_features.unsqueeze(1).expand(-1, seq_len, -1)
                bias = self.bias_networks[scale](token_features, seq_len)
                biases[scale] = bias
                combined_bias += scale_weights[i] * bias
        
        return {
            'combined_bias': combined_bias,
            'scale_biases': biases,
            'scale_weights': scale_weights
        }


def create_attention_bias(
    retrieval_dim: int,
    num_heads: int = 1,
    bias_type: str = 'simple',
    **kwargs
) -> nn.Module:
    """
    Factory function for attention bias modules.
    
    Args:
        retrieval_dim: Dimension of retrieval features
        num_heads: Number of attention heads
        bias_type: Type of bias ('simple', 'multiscale')
        **kwargs: Additional arguments
        
    Returns:
        Attention bias module
    """
    if bias_type == 'simple':
        return AttentionLogitBias(
            retrieval_dim=retrieval_dim,
            num_heads=num_heads,
            **kwargs
        )
    elif bias_type == 'multiscale':
        return MultiScaleAttentionBias(
            retrieval_dim=retrieval_dim,
            num_heads=num_heads,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")