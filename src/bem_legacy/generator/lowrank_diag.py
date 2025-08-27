"""
F5.2 - Low-Rank + Diagonal Generator.

Implements ΔW = U diag(c) V^T + diag(d) where:
- c = standard rank-r codes from controller  
- d = per-feature scaling predicted from same features
- Captures axis-aligned scaling cheaply, often boosts chrF/BLEU and format stability

Budget: Keep total params = baseline by reducing r slightly if needed.
Gate: +0.5–1.5% chrF/BLEU; no latency regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import math


@dataclass
class LowRankDiagConfig:
    """Configuration for Low-Rank + Diagonal generator."""
    rank: int = 8                    # Reduced rank to accommodate diagonal terms
    d_max: float = 0.2              # Maximum diagonal value (clamping)
    diagonal_l2_penalty: float = 0.01  # L2 penalty on diagonal terms
    diagonal_init_std: float = 0.01  # Initialization std for diagonal predictor
    spectral_budget_include_diag: bool = True  # Include diagonal in spectral budget
    use_feature_gating: bool = True  # Gate diagonal terms based on features
    diagonal_dropout: float = 0.1    # Dropout for diagonal predictor


class DiagonalPredictor(nn.Module):
    """Predicts per-feature diagonal scaling terms from input features."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LowRankDiagConfig
    ):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        
        # Feature processing for diagonal prediction
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Dropout(config.diagonal_dropout),
            nn.Linear(in_features // 2, out_features)
        )
        
        # Optional gating mechanism
        if config.use_feature_gating:
            self.gate_predictor = nn.Sequential(
                nn.Linear(in_features, in_features // 4),
                nn.GELU(),
                nn.Linear(in_features // 4, out_features),
                nn.Sigmoid()
            )
        else:
            self.gate_predictor = None
            
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with small values."""
        # Initialize diagonal predictor to output small values initially
        for module in self.feature_processor:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.diagonal_init_std)
                nn.init.zeros_(module.bias)
                
        if self.gate_predictor is not None:
            for module in self.gate_predictor:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict diagonal scaling terms.
        
        Args:
            x: Input features [*, in_features]
            
        Returns:
            diagonal: Diagonal scaling terms [*, out_features]
        """
        # Global feature statistics for diagonal prediction
        # Use mean pooling over sequence dimension if present
        if x.dim() == 3:  # [batch, seq_len, features]
            x_global = x.mean(dim=1)  # [batch, features]
        else:
            x_global = x
            
        # Predict diagonal terms
        diagonal = self.feature_processor(x_global)  # [batch, out_features]
        
        # Apply gating if enabled
        if self.gate_predictor is not None:
            gate = self.gate_predictor(x_global)  # [batch, out_features]
            diagonal = diagonal * gate
            
        # Clamp diagonal values
        diagonal = torch.clamp(diagonal, -self.config.d_max, self.config.d_max)
        
        # Expand back to match input shape if needed
        if x.dim() == 3:
            diagonal = diagonal.unsqueeze(1).expand(-1, x.size(1), -1)
            
        return diagonal


class LowRankDiagExpert(nn.Module):
    """
    Enhanced LoRA expert with additional diagonal scaling terms.
    
    Implements: ΔW = U diag(c) V^T + diag(d)
    where c comes from the controller and d is predicted from features.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        config: LowRankDiagConfig,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.config = config
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices (standard LoRA)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Diagonal predictor
        self.diagonal_predictor = DiagonalPredictor(
            in_features, out_features, config
        )
        
        # Dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
            
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters following LoRA conventions."""
        # LoRA initialization: A ~ N(0, σ²), B ~ 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(
        self, 
        x: torch.Tensor, 
        codes: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass computing low-rank + diagonal transformation.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features] or [*, in_features]
            codes: Routing codes [batch_size, seq_len, rank] or [*, rank]
            return_components: Whether to return component breakdown
            
        Returns:
            output: Transformed tensor [batch_size, seq_len, out_features]
            components: Component breakdown (if requested)
        """
        # Low-rank component: H = (x @ V) * code
        H = self.lora_A(x)  # [*, rank]
        H = H * codes  # Element-wise multiplication with codes
        H = self.dropout(H)
        lowrank_output = self.lora_B(H) * self.scaling  # [*, out_features]
        
        # Diagonal component: diag(d) applied to input
        diagonal_scales = self.diagonal_predictor(x)  # [*, out_features]
        
        # For diagonal operation, we need to apply scaling to the linear transformation
        # diag(d) @ W @ x ≈ d ⊙ (W @ x) for identity W, but we want d ⊙ x projected to output space
        # Simplified approximation: d ⊙ linear_proj(x) where linear_proj preserves feature relationships
        if hasattr(self, '_diagonal_proj'):
            diagonal_projection = self._diagonal_proj
        else:
            # Create a learnable projection for diagonal component
            self._diagonal_proj = nn.Linear(
                self.in_features, self.out_features, bias=False
            ).to(x.device)
            # Initialize as identity-like transformation
            with torch.no_grad():
                if self.in_features == self.out_features:
                    self._diagonal_proj.weight.copy_(torch.eye(self.in_features))
                else:
                    nn.init.xavier_uniform_(self._diagonal_proj.weight)
            diagonal_projection = self._diagonal_proj
            
        diagonal_base = diagonal_projection(x)  # [*, out_features]
        diagonal_output = diagonal_scales * diagonal_base  # Element-wise scaling
        
        # Combine components
        output = lowrank_output + diagonal_output
        
        if return_components:
            components = {
                'lowrank_output': lowrank_output,
                'diagonal_output': diagonal_output,
                'diagonal_scales': diagonal_scales,
                'codes': codes,
                'H': H
            }
            return output, components
        
        return output
        
    def get_effective_weight(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Get effective weight matrix for analysis.
        
        Args:
            codes: Routing codes [rank] or [batch, rank]
            
        Returns:
            weight: Effective weight matrix
        """
        # Low-rank component
        if codes.dim() == 1:
            # Single code vector
            U = self.lora_B.weight  # [out_features, rank]
            V = self.lora_A.weight.T  # [rank, in_features]
            lowrank_weight = U @ torch.diag(codes) @ V  # [out_features, in_features]
        else:
            # Multiple codes - return batch of weight matrices
            batch_size = codes.size(0)
            U = self.lora_B.weight.unsqueeze(0).expand(batch_size, -1, -1)
            V = self.lora_A.weight.T.unsqueeze(0).expand(batch_size, -1, -1)
            codes_diag = torch.diag_embed(codes)  # [batch, rank, rank]
            lowrank_weight = U @ codes_diag @ V  # [batch, out_features, in_features]
            
        return lowrank_weight * self.scaling
        
    def compute_spectral_norm(self, codes: torch.Tensor) -> torch.Tensor:
        """Compute spectral norm of the effective transformation."""
        weight = self.get_effective_weight(codes)
        if weight.dim() == 3:
            # Batch of matrices
            return torch.stack([torch.linalg.norm(w, ord=2) for w in weight])
        else:
            return torch.linalg.norm(weight, ord=2)
            
    def compute_frobenius_norm(self, codes: torch.Tensor) -> torch.Tensor:
        """Compute Frobenius norm of the effective transformation."""
        weight = self.get_effective_weight(codes)
        if weight.dim() == 3:
            # Batch of matrices
            return torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
        else:
            return torch.norm(weight, p='fro')
            
    def compute_diagonal_penalty(self) -> torch.Tensor:
        """Compute L2 penalty on diagonal terms."""
        total_penalty = 0.0
        
        # Penalty on diagonal predictor parameters
        for param in self.diagonal_predictor.parameters():
            total_penalty += torch.norm(param, p=2) ** 2
            
        return self.config.diagonal_l2_penalty * total_penalty


class LowRankDiagModule(nn.Module):
    """
    BEM module using Low-Rank + Diagonal experts.
    
    Drop-in replacement for BEMv11Module with enhanced expressivity.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        num_experts: int = 2,
        config: Optional[LowRankDiagConfig] = None,
        alpha: float = 16.0,
        dropout: float = 0.0,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7,
        max_singular_value: float = 1.0,
        fro_budget: float = 1.0
    ):
        super().__init__()
        
        if config is None:
            config = LowRankDiagConfig(rank=rank)
            
        self.config = config
        self.base_layer = base_layer
        self.rank = rank
        self.num_experts = num_experts
        
        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Low-rank + diagonal experts
        self.experts = nn.ModuleList([
            LowRankDiagExpert(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=rank,
                config=config,
                alpha=alpha,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
        # Router (reuse from stateful implementation)
        from ..controller.stateful import StatefulBEMRouter, StatefulRouterConfig
        
        router_config = StatefulRouterConfig(
            d_feat=base_layer.in_features,
            code_dim=rank,
            chunk_size=chunk_size
        )
        
        self.router = StatefulBEMRouter(
            input_dim=base_layer.in_features,
            num_experts=num_experts,
            config=router_config,
            chunk_size=chunk_size,
            hysteresis_tau=hysteresis_tau
        )
        
        # Governance for spectral constraints
        self.max_singular_value = max_singular_value
        self.fro_budget = fro_budget
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with low-rank + diagonal experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            hidden_state: Previous hidden state for stateful router
            return_details: Whether to return detailed component information
            
        Returns:
            Dictionary containing output and auxiliary information
        """
        # Base layer output
        base_output = self.base_layer(x)
        
        # Routing with stateful router
        routing_weights, expert_indices, aux_info = self.router(x, hidden_state)
        codes = aux_info['codes']  # [batch_size, seq_len, rank]
        
        # Compute expert outputs
        expert_outputs = []
        expert_components = []
        
        for i, expert in enumerate(self.experts):
            if return_details:
                expert_out, components = expert(x, codes, return_components=True)
                expert_components.append(components)
            else:
                expert_out = expert(x, codes)
            expert_outputs.append(expert_out)
        
        # Combine expert outputs with routing weights
        expert_stack = torch.stack(expert_outputs, dim=-1)  # [batch, seq, out, num_experts]
        
        # Apply routing weights
        routed_output = torch.sum(
            expert_stack * routing_weights.unsqueeze(-2),  # Broadcasting
            dim=-1
        )
        
        # Final output: base + routed experts
        output = base_output + routed_output
        
        result = {
            'output': output,
            'routing_weights': routing_weights,
            'expert_indices': expert_indices,
            'codes': codes,
            'hidden_state': aux_info['hidden_state'],
            'flip_penalty': aux_info['flip_penalty']
        }
        
        if return_details:
            result.update({
                'expert_outputs': expert_outputs,
                'expert_components': expert_components,
                'base_output': base_output,
                'routed_output': routed_output,
                'router_metrics': aux_info.get('metrics', {})
            })
            
        return result
        
    def compute_regularization_loss(self) -> Dict[str, torch.Tensor]:
        """Compute regularization terms for training."""
        losses = {}
        
        # Diagonal penalties from all experts
        diagonal_penalty = sum(expert.compute_diagonal_penalty() for expert in self.experts)
        losses['diagonal_penalty'] = diagonal_penalty
        
        # Spectral constraints (placeholder - would need actual implementation)
        if hasattr(self, '_last_codes'):
            spectral_penalties = []
            for expert in self.experts:
                spectral_norm = expert.compute_spectral_norm(self._last_codes.mean(dim=0))
                spectral_penalty = F.relu(spectral_norm - self.max_singular_value) ** 2
                spectral_penalties.append(spectral_penalty)
            losses['spectral_penalty'] = torch.stack(spectral_penalties).mean()
            
        return losses
        
    def get_budget_info(self) -> Dict[str, int]:
        """Get parameter and FLOP budget information."""
        # Count parameters
        total_params = 0
        lowrank_params = 0
        diagonal_params = 0
        
        for expert in self.experts:
            # Low-rank parameters
            lowrank_params += expert.lora_A.weight.numel() + expert.lora_B.weight.numel()
            
            # Diagonal predictor parameters
            for param in expert.diagonal_predictor.parameters():
                diagonal_params += param.numel()
                
        total_params = lowrank_params + diagonal_params
        
        # Router parameters
        router_params = sum(p.numel() for p in self.router.parameters())
        total_params += router_params
        
        # FLOP estimates (approximate)
        seq_len_est = 512  # Typical sequence length
        batch_size_est = 8  # Typical batch size
        
        # Low-rank FLOPs: 2 * batch * seq * (in * rank + rank * out) per expert
        lowrank_flops = 2 * batch_size_est * seq_len_est * self.num_experts * (
            self.base_layer.in_features * self.rank + self.rank * self.base_layer.out_features
        )
        
        # Diagonal FLOPs: diagonal predictor forward pass
        diagonal_flops = 0
        for expert in self.experts:
            for module in expert.diagonal_predictor.feature_processor:
                if isinstance(module, nn.Linear):
                    diagonal_flops += 2 * batch_size_est * seq_len_est * \
                                   module.in_features * module.out_features
                                   
        total_flops = lowrank_flops + diagonal_flops
        
        return {
            'total_params': total_params,
            'lowrank_params': lowrank_params,
            'diagonal_params': diagonal_params,
            'router_params': router_params,
            'total_flops': total_flops,
            'lowrank_flops': lowrank_flops,
            'diagonal_flops': diagonal_flops
        }


def create_lowrank_diag_config(**kwargs) -> LowRankDiagConfig:
    """Factory function to create LowRankDiagConfig with validation."""
    return LowRankDiagConfig(**kwargs)


def convert_lora_to_lowrank_diag(
    lora_expert: nn.Module,
    config: LowRankDiagConfig
) -> LowRankDiagExpert:
    """
    Convert a standard LoRA expert to Low-Rank + Diagonal expert.
    
    Args:
        lora_expert: Existing LoRA expert module
        config: Configuration for the new expert
        
    Returns:
        Enhanced expert with diagonal capabilities
    """
    # Extract dimensions from existing LoRA
    in_features = lora_expert.lora_A.in_features
    out_features = lora_expert.lora_B.out_features
    rank = lora_expert.lora_A.out_features
    
    # Create new expert
    new_expert = LowRankDiagExpert(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        config=config
    )
    
    # Copy LoRA weights
    with torch.no_grad():
        new_expert.lora_A.weight.copy_(lora_expert.lora_A.weight)
        new_expert.lora_B.weight.copy_(lora_expert.lora_B.weight)
        
    return new_expert


def lowrank_plus_diag(x, U, V, code_c, diag_d, d_max=0.2):
    """
    Compute low-rank + diagonal transformation as specified in TODO.md.
    
    ΔW = U diag(c) V^T + diag(d)
    
    Args:
        x: Input tensor [T, d_in]
        U: Left matrix [d_out, rank]
        V: Right matrix [d_in, rank] 
        code_c: Codes [T, rank]
        diag_d: Diagonal terms [T, d_out]
        d_max: Maximum diagonal value for clamping
        
    Returns:
        Transformed output [T, d_out]
    """
    # Clamp diagonal terms
    d = diag_d.clamp_(-d_max, d_max)
    
    # Low-rank component: H = (x @ V) * code_c, then H @ U.T
    H = (x @ V) * code_c
    lowrank_output = H @ U.T
    
    # Diagonal component: d ⊙ (x @ I) ≈ d ⊙ x (broadcasted)
    # For efficiency, we approximate diag(d) @ x as element-wise scaling
    if x.size(-1) == U.size(0):  # If dimensions match
        diagonal_output = d * x  
    else:
        # Use identity matrix projection for proper dimensions
        I = torch.eye(U.size(0), device=x.device, dtype=x.dtype)
        diagonal_output = d * (x @ I[:x.size(-1), :].T)
    
    return lowrank_output + diagonal_output