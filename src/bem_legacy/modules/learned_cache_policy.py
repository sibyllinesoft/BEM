"""
V11 - Learned Cache Policy Implementation

Extends B1 with learned K/V update policy. Controller emits "update-KV" events.
K/V frozen except during update windows. Windows align with chunk boundaries (N=128).
Policy learned based on retrieval quality and context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math

from .parallel_lora import GeneratedParallelLoRA
from .chunk_sticky_routing import ChunkStickyRouter
from .attention_bias import MultiScaleAttentionBias as AttentionBias
from .governance import SpectralGovernance


class CachePolicyController(nn.Module):
    """
    Learned controller that decides when to update K/V cache.
    Emits "update-KV" events based on retrieval quality and context.
    """
    
    def __init__(
        self,
        retrieval_dim: int,
        hidden_dim: int = 128,
        window_size: int = 64,
        policy_lr: float = 1e-5
    ):
        super().__init__()
        self.retrieval_dim = retrieval_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.policy_lr = policy_lr
        
        # Policy network that decides update probability
        self.policy_network = nn.Sequential(
            nn.Linear(retrieval_dim + 3, hidden_dim),  # +3 for cache stats
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Update probability [0, 1]
        )
        
        # Cache quality estimator
        self.quality_estimator = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Quality score [0, 1]
        )
        
        # Running statistics for cache performance
        self.register_buffer('cache_hit_rate', torch.tensor(0.5))
        self.register_buffer('avg_update_interval', torch.tensor(32.0))
        self.register_buffer('quality_ema', torch.tensor(0.5))
        self.ema_momentum = 0.99
        
        # Initialize policy to be conservative (low update rate initially)
        self._init_conservative_policy()
        
    def _init_conservative_policy(self):
        """Initialize policy to update conservatively."""
        with torch.no_grad():
            # Policy network should output low probabilities initially
            nn.init.normal_(self.policy_network[-2].weight, std=0.01)
            nn.init.constant_(self.policy_network[-2].bias, -2.0)  # sigmoid(-2) ≈ 0.12
            
    def update_cache_stats(
        self, 
        hit_rate: torch.Tensor,
        update_interval: torch.Tensor,
        quality: torch.Tensor
    ):
        """Update running cache statistics."""
        with torch.no_grad():
            self.cache_hit_rate = self.ema_momentum * self.cache_hit_rate + (1 - self.ema_momentum) * hit_rate.mean()
            self.avg_update_interval = self.ema_momentum * self.avg_update_interval + (1 - self.ema_momentum) * update_interval.mean()
            self.quality_ema = self.ema_momentum * self.quality_ema + (1 - self.ema_momentum) * quality.mean()
            
    def forward(
        self,
        retrieval_features: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decide whether to update K/V cache.
        
        Args:
            retrieval_features: [batch_size, retrieval_dim]
            position_ids: [batch_size, seq_len] for window alignment
            
        Returns:
            Dictionary with update decisions and cache metrics
        """
        batch_size = retrieval_features.shape[0]
        device = retrieval_features.device
        
        # Estimate retrieval quality
        quality_scores = self.quality_estimator(retrieval_features)  # [batch_size, 1]
        
        # Cache statistics as features
        cache_stats = torch.stack([
            self.cache_hit_rate.expand(batch_size),
            self.avg_update_interval.expand(batch_size) / 128.0,  # Normalize
            self.quality_ema.expand(batch_size)
        ], dim=1)  # [batch_size, 3]
        
        # Combine retrieval features with cache stats
        policy_input = torch.cat([retrieval_features, cache_stats], dim=1)
        
        # Policy decision
        update_prob = self.policy_network(policy_input)  # [batch_size, 1]
        
        # Sample update decisions (differentiable via Gumbel trick for training)
        if self.training:
            # Gumbel-Softmax for differentiable sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(update_prob) + 1e-8) + 1e-8)
            update_logits = torch.log(update_prob + 1e-8) - torch.log(1 - update_prob + 1e-8)
            update_decisions = torch.sigmoid((update_logits + gumbel_noise) / 0.1)
        else:
            # Hard decisions during inference
            update_decisions = (update_prob > 0.5).float()
            
        # Window alignment: align with chunk boundaries
        window_aligned_decisions = self._align_with_windows(
            update_decisions, position_ids
        )
        
        return {
            'update_decisions': update_decisions.squeeze(-1),  # [batch_size]
            'update_prob': update_prob.squeeze(-1),           # [batch_size]
            'quality_scores': quality_scores.squeeze(-1),     # [batch_size]
            'window_aligned_decisions': window_aligned_decisions,
            'cache_stats': {
                'hit_rate': self.cache_hit_rate,
                'avg_interval': self.avg_update_interval,
                'quality_ema': self.quality_ema
            }
        }
        
    def _align_with_windows(
        self, 
        decisions: torch.Tensor,
        position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Align update decisions with chunk/window boundaries."""
        # Simple window alignment: quantize positions to window boundaries
        window_starts = (position_ids // self.window_size) * self.window_size
        
        # Decision applies to entire window
        batch_size, seq_len = position_ids.shape
        aligned_decisions = decisions.unsqueeze(1).expand(batch_size, seq_len)
        
        return aligned_decisions


class LearnedCacheKVLayer(nn.Module):
    """
    K/V layer with learned cache update policy.
    Updates K/V cache only during controller-specified windows.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        # K/V cache storage
        self.register_buffer(
            'cached_k', 
            torch.zeros(1, max_seq_len, num_heads, self.head_dim)
        )
        self.register_buffer(
            'cached_v',
            torch.zeros(1, max_seq_len, num_heads, self.head_dim) 
        )
        self.register_buffer('cache_valid_length', torch.tensor(0))
        self.register_buffer('last_update_pos', torch.tensor(0))
        
        # Cache performance tracking
        self.register_buffer('cache_hits', torch.tensor(0.0))
        self.register_buffer('cache_misses', torch.tensor(0.0))
        
    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: torch.Tensor,
        update_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with learned cache policy.
        
        Args:
            k: Key tensor [batch_size, seq_len, num_heads, head_dim]
            v: Value tensor [batch_size, seq_len, num_heads, head_dim]
            position_ids: [batch_size, seq_len]
            update_mask: [batch_size, seq_len] - where to update cache
            
        Returns:
            Dictionary with cached K/V and cache metrics
        """
        batch_size, seq_len, num_heads, head_dim = k.shape
        device = k.device
        
        # Initialize cache if needed
        if self.cached_k.shape[0] != batch_size:
            self.cached_k = self.cached_k.expand(batch_size, -1, -1, -1).contiguous()
            self.cached_v = self.cached_v.expand(batch_size, -1, -1, -1).contiguous()
            
        # Determine cache hits and misses
        max_pos = position_ids.max().item()
        cache_hit_mask = position_ids < self.cache_valid_length
        
        # Update cache statistics
        hits = cache_hit_mask.float().sum()
        misses = (~cache_hit_mask).float().sum()
        self.cache_hits = 0.99 * self.cache_hits + 0.01 * hits
        self.cache_misses = 0.99 * self.cache_misses + 0.01 * misses
        
        # Apply update policy: only update where mask is True
        k_out = k.clone()
        v_out = v.clone()
        
        # Use cached values where available and no update requested
        for b in range(batch_size):
            for s in range(seq_len):
                pos = position_ids[b, s].item()
                if pos < self.cache_valid_length and not update_mask[b, s]:
                    # Use cached values
                    k_out[b, s] = self.cached_k[b, pos]
                    v_out[b, s] = self.cached_v[b, pos]
                elif update_mask[b, s]:
                    # Update cache with new values
                    if pos < self.max_seq_len:
                        self.cached_k[b, pos] = k[b, s]
                        self.cached_v[b, pos] = v[b, s]
                        
        # Update cache valid length
        if update_mask.any():
            new_valid_length = max(self.cache_valid_length.item(), max_pos + 1)
            self.cache_valid_length = min(new_valid_length, self.max_seq_len)
            self.last_update_pos = max_pos
            
        # Cache performance metrics
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses + 1e-8)
        update_interval = max_pos - self.last_update_pos.item() + 1
        
        return {
            'k': k_out,
            'v': v_out,
            'cache_metrics': {
                'hit_rate': hit_rate,
                'update_interval': torch.tensor(update_interval, device=device),
                'cache_valid_length': self.cache_valid_length,
                'num_updates': update_mask.sum()
            }
        }


class LearnedCacheBEMLayer(nn.Module):
    """
    BEM layer with learned cache policy.
    Extends B1 architecture with intelligent K/V caching.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        retrieval_dim: int,
        rank: int,
        num_experts: int = 2,
        alpha: float = 16.0,
        dropout: float = 0.0,
        window_size: int = 64,
        policy_lr: float = 1e-5,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.retrieval_dim = retrieval_dim
        self.rank = rank
        
        # Base BEM-v1.1 components (same as B1)
        self.bem_lora = GeneratedParallelLoRA(
            base_layer=base_layer,
            retrieval_dim=retrieval_dim,
            rank=rank,
            num_experts=num_experts,
            alpha=alpha,
            dropout=dropout
        )
        
        # Chunk-sticky routing (same as B1)
        self.router = ChunkStickyRouter(
            chunk_size=chunk_size,
            hysteresis_tau=hysteresis_tau
        )
        
        # Learned cache policy controller
        self.cache_controller = CachePolicyController(
            retrieval_dim=retrieval_dim,
            window_size=window_size,
            policy_lr=policy_lr
        )
        
        # K/V cache layer (if applicable)
        self.kv_cache = None
        if 'attention' in str(type(base_layer)).lower() or hasattr(base_layer, 'num_heads'):
            # Assume attention layer
            d_model = base_layer.out_features
            num_heads = getattr(base_layer, 'num_heads', 8)
            self.kv_cache = LearnedCacheKVLayer(
                d_model=d_model,
                num_heads=num_heads
            )
            
        # Spectral governance (same as B1)
        self.governance = SpectralGovernance(
            max_singular_value=2.0,
            fro_budget=0.1
        )
        
    def forward(
        self,
        x: torch.Tensor,
        retrieval_features: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with learned cache policy.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_features: [batch_size, retrieval_dim]
            position_ids: Position IDs for cache alignment
            k, v: Key/Value tensors if attention layer
            
        Returns:
            Dictionary with output, cache decisions, metrics, etc.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Create default position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
        # Base BEM forward pass (same as B1)
        bem_result = self.bem_lora(x, retrieval_features)
        bem_output = bem_result['output']
        
        # Cache policy decision
        cache_decision = self.cache_controller(retrieval_features, position_ids)
        update_decisions = cache_decision['window_aligned_decisions']  # [batch_size, seq_len]
        
        # Apply cache policy to K/V if this is an attention layer
        cache_metrics = {}
        k_out, v_out = k, v
        if self.kv_cache is not None and k is not None and v is not None:
            cache_result = self.kv_cache(k, v, position_ids, update_decisions)
            k_out, v_out = cache_result['k'], cache_result['v']
            cache_metrics = cache_result['cache_metrics']
            
            # Update controller statistics
            self.cache_controller.update_cache_stats(
                cache_metrics['hit_rate'],
                cache_metrics['update_interval'],
                cache_decision['quality_scores'].mean()
            )
            
        return {
            'output': bem_output,
            'base_output': bem_result['base_output'],
            'expert_outputs': bem_result['expert_outputs'],
            'gates': bem_result['gates'],
            'k': k_out,
            'v': v_out,
            'cache_decisions': cache_decision,
            'cache_metrics': cache_metrics,
            'cache_enabled': self.kv_cache is not None
        }


class LearnedCacheBEM(nn.Module):
    """
    V11 Learned Cache Policy BEM implementation.
    
    Extends B1 with learned K/V update policy. Controller emits "update-KV" events.
    K/V frozen except during update windows aligned with chunk boundaries.
    """
    
    def __init__(
        self,
        base_layers: Dict[str, nn.Linear],
        retrieval_dim: int,
        rank_schedule: Dict[str, int],
        num_experts: int = 2,
        alpha: float = 16.0,
        dropout: float = 0.0,
        window_size: int = 64,
        policy_lr: float = 1e-5,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7
    ):
        super().__init__()
        
        self.base_layers = base_layers
        self.cache_bem_modules = nn.ModuleDict()
        
        for name, layer in base_layers.items():
            rank = rank_schedule.get(name, 8)  # Default rank if not specified
            
            self.cache_bem_modules[name] = LearnedCacheBEMLayer(
                base_layer=layer,
                retrieval_dim=retrieval_dim,
                rank=rank,
                num_experts=num_experts,
                alpha=alpha,
                dropout=dropout,
                window_size=window_size,
                policy_lr=policy_lr,
                chunk_size=chunk_size,
                hysteresis_tau=hysteresis_tau
            )
            
    def forward(
        self,
        layer_name: str,
        x: torch.Tensor,
        retrieval_features: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific layer.
        """
        if layer_name not in self.cache_bem_modules:
            # Fallback to base layer
            result = {'output': self.base_layers[layer_name](x)}
            if k is not None:
                result['k'] = k
            if v is not None:
                result['v'] = v
            return result
            
        return self.cache_bem_modules[layer_name](
            x, retrieval_features, position_ids, k, v
        )
        
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics from all layers.
        """
        stats = {}
        for layer_name, module in self.cache_bem_modules.items():
            if hasattr(module, 'kv_cache') and module.kv_cache is not None:
                cache_stats = {
                    'hit_rate': module.kv_cache.cache_hits / (module.kv_cache.cache_hits + module.kv_cache.cache_misses + 1e-8),
                    'valid_length': module.kv_cache.cache_valid_length.item(),
                    'last_update_pos': module.kv_cache.last_update_pos.item()
                }
                controller_stats = module.cache_controller.cache_stats if hasattr(module, 'cache_controller') else {}
                stats[layer_name] = {**cache_stats, **controller_stats}
        return stats
        
    def reset_caches(self):
        """Reset all cache states."""
        for module in self.cache_bem_modules.values():
            if hasattr(module, 'kv_cache') and module.kv_cache is not None:
                module.kv_cache.cached_k.zero_()
                module.kv_cache.cached_v.zero_()
                module.kv_cache.cache_valid_length.zero_()
                module.kv_cache.last_update_pos.zero_()
                module.kv_cache.cache_hits.zero_()
                module.kv_cache.cache_misses.zero_()


def create_learned_cache_bem_for_model(
    model: nn.Module,
    retrieval_dim: int,
    rank_schedule: Optional[Dict[str, int]] = None,
    attachment_points: Optional[Dict[str, str]] = None,
    **kwargs
) -> LearnedCacheBEM:
    """
    Create Learned Cache BEM for specified model layers.
    
    Args:
        model: Base transformer model
        retrieval_dim: Dimension of retrieval features
        rank_schedule: Rank per layer name (same as B1: [2,4,8,8,8,4,2])
        attachment_points: Which layers to attach to (default: W_O and W_down)
        
    Returns:
        LearnedCacheBEM instance
    """
    if attachment_points is None:
        attachment_points = {
            'attention': ['out_proj', 'W_O'],
            'mlp': ['down_proj', 'W_down'],
            'kv_layers': ['kv_windows']  # Special marker for K/V layers
        }
        
    if rank_schedule is None:
        # Same rank schedule as B1
        rank_schedule = {}
        default_ranks = [2, 4, 8, 8, 8, 4, 2]
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
                
    print(f"Created Learned Cache BEM for {len(base_layers)} layers")
    print(f"Rank schedule: {rank_schedule}")
    
    return LearnedCacheBEM(
        base_layers=base_layers,
        retrieval_dim=retrieval_dim,
        rank_schedule=rank_schedule,
        **kwargs
    )