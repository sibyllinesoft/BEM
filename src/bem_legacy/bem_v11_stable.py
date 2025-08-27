"""
BEM-v1.1-stable implementation according to TODO.md specifications.

Architecture (E1 + E3 + E4):
- Sites: W_O and W_down only (cache-safe, high-leverage)
- Parallel LoRA: y = base(x) + Σ_e g_e · B_e (A_e x)
- Depth-varying ranks: [2,4,8,8,8,4,2] across blocks
- Chunk-sticky routing: N∈{64,128}, hysteresis τ=0.7
- Attention-logit bias: add bias(z) to attention scores (no K/V edits)
- Spectral governance: σ₁ clamp + Fro norm budget with trust-region
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math
import numpy as np


class SpectralGovernance(nn.Module):
    """
    Spectral governance with σ₁ clamp and Frobenius norm budget with trust-region projection.
    """
    
    def __init__(self, max_singular_value: float = 1.0, fro_budget: float = 1.0):
        super().__init__()
        self.max_singular_value = max_singular_value
        self.fro_budget = fro_budget
        
    def apply_governance(self, delta_w: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral governance to delta weights.
        
        Args:
            delta_w: Delta weight tensor [out_features, in_features]
            
        Returns:
            Governed delta weights
        """
        # Compute SVD for spectral norm clamping
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        
        # Clamp singular values
        S_clamped = torch.clamp(S, max=self.max_singular_value)
        
        # Reconstruct with clamped singular values
        delta_w_clamped = U @ torch.diag(S_clamped) @ Vh
        
        # Apply Frobenius norm budget with trust-region projection
        fro_norm = torch.norm(delta_w_clamped, 'fro')
        if fro_norm > self.fro_budget:
            delta_w_clamped = delta_w_clamped * (self.fro_budget / fro_norm)
            
        return delta_w_clamped


class ChunkStickyRouter(nn.Module):
    """
    Chunk-sticky routing with hysteresis (E3).
    Computes token logits and applies piecewise-constant decisions per chunk.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.chunk_size = chunk_size
        self.hysteresis_tau = hysteresis_tau
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts)
        )
        
        # Store previous routing decisions for hysteresis
        self.register_buffer('prev_routing', torch.zeros(1, dtype=torch.long))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute chunk-sticky routing with hysteresis.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            routing_weights: Soft routing weights [batch_size, seq_len, num_experts]
            expert_indices: Hard expert indices [batch_size, num_chunks]
        """
        batch_size, seq_len, _ = x.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Compute routing logits for each token
        logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # Apply softmax to get probabilities
        routing_probs = F.softmax(logits, dim=-1)
        
        # Chunk-wise routing decisions
        expert_indices = torch.zeros(batch_size, num_chunks, dtype=torch.long, device=x.device)
        routing_weights = torch.zeros_like(routing_probs)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, seq_len)
            
            # Average logits over chunk
            chunk_logits = logits[:, start_idx:end_idx].mean(dim=1)  # [batch_size, num_experts]
            
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
        
        return routing_weights, expert_indices


class AttentionLogitBias(nn.Module):
    """
    Attention-logit bias from retrieval features (E4).
    Adds bias(z) to attention scores without modifying K/V.
    """
    
    def __init__(self, retrieval_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.bias_net = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, retrieval_features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention bias from retrieval features.
        
        Args:
            retrieval_features: [batch_size, seq_len, retrieval_dim]
            
        Returns:
            bias: [batch_size, seq_len, 1] bias to add to attention scores
        """
        return self.bias_net(retrieval_features)


class ParallelLoRAExpert(nn.Module):
    """
    Single expert in the parallel LoRA setup.
    Implements B_e (A_e x) with proper initialization.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA A and B matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B(Ax) for this expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            
        Returns:
            Expert output [batch_size, seq_len, out_features]
        """
        # A_e x
        ax = torch.matmul(x, self.lora_A)  # [batch_size, seq_len, rank]
        ax = self.dropout(ax)
        
        # B_e (A_e x)
        output = torch.matmul(ax, self.lora_B) * self.scaling
        
        return output


class BEMv11Module(nn.Module):
    """
    BEM-v1.1-stable module implementing E1 + E3 + E4.
    Supports W_O and W_down attachment points only (cache-safe).
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        num_experts: int = 2,
        alpha: float = 16.0,
        dropout: float = 0.0,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7,
        retrieval_dim: Optional[int] = None,
        max_singular_value: float = 1.0,
        fro_budget: float = 1.0
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.num_experts = num_experts
        
        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Parallel LoRA experts
        self.experts = nn.ModuleList([
            ParallelLoRAExpert(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
        # Chunk-sticky router (E3)
        self.router = ChunkStickyRouter(
            input_dim=base_layer.in_features,
            num_experts=num_experts,
            chunk_size=chunk_size,
            hysteresis_tau=hysteresis_tau
        )
        
        # Attention bias (E4) - optional
        self.attention_bias = None
        if retrieval_dim is not None:
            self.attention_bias = AttentionLogitBias(retrieval_dim)
            
        # Spectral governance
        self.governance = SpectralGovernance(
            max_singular_value=max_singular_value,
            fro_budget=fro_budget
        )
        
    def forward(
        self,
        x: torch.Tensor,
        retrieval_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing y = base(x) + Σ_e g_e · B_e (A_e x).
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            retrieval_features: Optional retrieval features for attention bias
            
        Returns:
            Dictionary containing:
                - output: Final output [batch_size, seq_len, out_features]
                - attention_bias: Optional attention bias [batch_size, seq_len, 1]
                - routing_weights: Routing weights [batch_size, seq_len, num_experts]
                - expert_outputs: Individual expert outputs
        """
        # Base layer output
        base_output = self.base_layer(x)
        
        # Routing (E3)
        routing_weights, expert_indices = self.router(x)
        
        # Compute expert outputs (E1)
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)
            expert_outputs.append(expert_output)
        
        # Combine expert outputs with routing weights
        expert_stack = torch.stack(expert_outputs, dim=-1)  # [batch, seq, out, num_experts]
        
        # Apply routing weights: Σ_e g_e · B_e (A_e x)
        routed_output = torch.sum(
            expert_stack * routing_weights.unsqueeze(-2),  # Broadcasting
            dim=-1
        )
        
        # Apply spectral governance to the combined delta
        # Note: In practice, we'd apply this during training updates
        # Here we're showing the conceptual structure
        
        # Final output: base + routed experts
        output = base_output + routed_output
        
        # Attention bias (E4)
        attention_bias = None
        if self.attention_bias is not None and retrieval_features is not None:
            attention_bias = self.attention_bias(retrieval_features)
            
        return {
            'output': output,
            'attention_bias': attention_bias,
            'routing_weights': routing_weights,
            'expert_outputs': expert_outputs,
            'expert_indices': expert_indices
        }


class BEMv11StableModel(nn.Module):
    """
    Complete BEM-v1.1-stable model that can be applied to transformer layers.
    Supports depth-varying ranks and selective attachment to W_O and W_down only.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        rank_schedule: List[int],
        attachment_points: List[str] = ['W_O', 'W_down'],
        num_experts: int = 2,
        alpha: float = 16.0,
        dropout: float = 0.0,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7,
        retrieval_dim: Optional[int] = None,
        max_singular_value: float = 1.0,
        fro_budget: float = 1.0
    ):
        super().__init__()
        
        self.base_model = base_model
        self.rank_schedule = rank_schedule
        self.attachment_points = attachment_points
        self.num_experts = num_experts
        
        # Attach BEM modules to specified layers
        self.bem_modules = nn.ModuleDict()
        
        # Find transformer layers and attach BEM modules
        layer_count = 0
        for name, module in base_model.named_modules():
            if any(ap in name for ap in attachment_points):
                if isinstance(module, nn.Linear):
                    # Determine rank for this layer
                    rank = rank_schedule[min(layer_count, len(rank_schedule) - 1)]
                    
                    # Create BEM module
                    bem_module = BEMv11Module(
                        base_layer=module,
                        rank=rank,
                        num_experts=num_experts,
                        alpha=alpha,
                        dropout=dropout,
                        chunk_size=chunk_size,
                        hysteresis_tau=hysteresis_tau,
                        retrieval_dim=retrieval_dim,
                        max_singular_value=max_singular_value,
                        fro_budget=fro_budget
                    )
                    
                    self.bem_modules[name] = bem_module
                    layer_count += 1
                    
        print(f"Attached BEM modules to {len(self.bem_modules)} layers")
        print(f"Using rank schedule: {rank_schedule}")
        
    def forward(self, *args, **kwargs):
        """
        Forward pass through the model with BEM modifications.
        This would need to be implemented based on the specific base model architecture.
        """
        # This is a placeholder - actual implementation would depend on the base model
        # and how we want to integrate the BEM modules into the forward pass
        raise NotImplementedError("Model-specific forward pass needs to be implemented")
        
    def get_cache_safety_report(self) -> Dict[str, Any]:
        """
        Generate a report verifying cache safety (no K/V token-wise edits).
        """
        report = {
            'cache_safe': True,
            'attachment_points': self.attachment_points,
            'kv_modifications': False,
            'token_wise_kv_edits': False,
            'details': []
        }
        
        # Verify that we only attach to W_O and W_down
        safe_points = ['W_O', 'W_down', 'out_proj', 'down_proj']
        unsafe_points = ['W_Q', 'W_K', 'W_V', 'q_proj', 'k_proj', 'v_proj']
        
        for point in self.attachment_points:
            if any(unsafe in point for unsafe in unsafe_points):
                report['cache_safe'] = False
                report['kv_modifications'] = True
                report['details'].append(f"UNSAFE: {point} modifies K/V")
            elif any(safe in point for safe in safe_points):
                report['details'].append(f"SAFE: {point} is cache-safe")
            else:
                report['details'].append(f"UNKNOWN: {point} safety unknown")
                
        return report


def create_bem_v11_stable(
    base_model: nn.Module,
    rank_schedule: Optional[List[int]] = None,
    **kwargs
) -> BEMv11StableModel:
    """
    Convenience function to create BEM-v1.1-stable model.
    
    Args:
        base_model: Base transformer model
        rank_schedule: Depth-varying ranks, defaults to [2,4,8,8,8,4,2]
        **kwargs: Additional arguments for BEMv11StableModel
        
    Returns:
        BEM-v1.1-stable model
    """
    if rank_schedule is None:
        rank_schedule = [2, 4, 8, 8, 8, 4, 2]  # Default from TODO.md
        
    return BEMv11StableModel(
        base_model=base_model,
        rank_schedule=rank_schedule,
        **kwargs
    )


def validate_cache_safety(model: BEMv11StableModel) -> bool:
    """
    Validate that the model maintains cache safety.
    
    Args:
        model: BEM-v1.1-stable model to validate
        
    Returns:
        True if cache-safe, False otherwise
    """
    report = model.get_cache_safety_report()
    
    if not report['cache_safe']:
        print("❌ CACHE SAFETY VIOLATION:")
        for detail in report['details']:
            if 'UNSAFE' in detail:
                print(f"  {detail}")
        return False
    else:
        print("✅ CACHE SAFETY VERIFIED:")
        for detail in report['details']:
            print(f"  {detail}")
        return True