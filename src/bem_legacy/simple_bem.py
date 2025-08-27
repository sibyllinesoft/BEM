"""
Simple BEM implementation for validation experiment.
This is the "dirt simple" version focused on proving the core concept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class BEMController(nn.Module):
    """
    Simple controller that takes task instruction as input and predicts a code vector c.
    The code vector is used to interpolate between static LoRAs.
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
            hidden_dim = input_dim * 4
            
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP controller
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, code_dim),
            nn.Softmax(dim=-1)  # Ensure coefficients sum to 1
        )
        
        # Initialize with small weights
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate interpolation coefficients from input features.
        
        Args:
            features: Input features [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            
        Returns:
            code: Interpolation coefficients [batch_size, code_dim] or [batch_size, seq_len, code_dim]
        """
        # If 3D, pool over sequence dimension
        if features.dim() == 3:
            # Mean pooling over sequence dimension
            features = features.mean(dim=1)
        
        code = self.net(features)
        return code


class SimpleBEMModule(nn.Module):
    """
    Simple BEM module that wraps a base linear layer and applies dynamic LoRA-style updates.
    Uses the "generated" variant: ΔW = U * diag(code) * V^T
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.02
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Get dimensions from base layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # LoRA parameters: ΔW = U * diag(code) * V^T
        self.lora_U = nn.Parameter(torch.randn(self.out_features, rank) * init_scale)
        self.lora_V = nn.Parameter(torch.randn(self.in_features, rank) * init_scale)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic LoRA update.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features] or [batch_size, in_features]
            code: Dynamic code vector [batch_size, rank]
            
        Returns:
            output: [batch_size, seq_len, out_features] or [batch_size, out_features]
        """
        # Base forward pass
        base_output = self.base_layer(x)
        
        # Dynamic LoRA computation: x @ (U * diag(code) * V^T)^T
        # = x @ V @ diag(code) @ U^T
        # = (x @ V) * code @ U^T
        
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, self.in_features)  # [batch_size * seq_len, in_features]
            # Expand code for sequence dimension
            code_expanded = code.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, rank]
            code_flat = code_expanded.contiguous().view(-1, self.rank)  # [batch_size * seq_len, rank]
        else:
            x_flat = x
            code_flat = code
        
        # Compute dynamic LoRA update
        x_v = torch.matmul(x_flat, self.lora_V)  # [*, rank]
        x_v_scaled = x_v * code_flat  # Element-wise multiplication with code
        lora_output = torch.matmul(x_v_scaled, self.lora_U.t())  # [*, out_features]
        
        # Reshape back if needed
        if len(original_shape) == 3:
            lora_output = lora_output.view(original_shape[0], original_shape[1], -1)
        
        # Apply dropout and scaling
        lora_output = self.dropout(lora_output) * self.scaling
        
        return base_output + lora_output


def create_bem_from_linear(
    linear_layer: nn.Linear,
    controller_input_dim: int,
    rank: int = 8,
    alpha: float = 16.0,
    controller_hidden_dim: Optional[int] = None,
    dropout: float = 0.1
) -> Tuple[SimpleBEMModule, BEMController]:
    """
    Convenience function to create a BEM module and controller from a linear layer.
    
    Args:
        linear_layer: Base linear layer to wrap
        controller_input_dim: Input dimension for the controller
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        controller_hidden_dim: Hidden dimension for controller
        dropout: Dropout rate
        
    Returns:
        bem_module: SimpleBEMModule instance
        controller: BEMController instance
    """
    bem_module = SimpleBEMModule(
        base_layer=linear_layer,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    controller = BEMController(
        input_dim=controller_input_dim,
        code_dim=rank,
        hidden_dim=controller_hidden_dim,
        dropout=dropout
    )
    
    return bem_module, controller


# Utility functions for debugging and analysis
def analyze_code_distribution(codes: torch.Tensor) -> Dict[str, float]:
    """Analyze the distribution of generated codes."""
    with torch.no_grad():
        return {
            'mean': codes.mean().item(),
            'std': codes.std().item(),
            'min': codes.min().item(),
            'max': codes.max().item(),
            'entropy': -(codes * torch.log(codes + 1e-8)).sum(dim=-1).mean().item()
        }


def compute_effective_rank(codes: torch.Tensor) -> float:
    """Compute the effective rank of the code distribution."""
    with torch.no_grad():
        # Normalize codes
        codes_norm = F.normalize(codes, p=1, dim=-1)
        # Compute entropy-based effective rank
        entropy = -(codes_norm * torch.log(codes_norm + 1e-8)).sum(dim=-1).mean()
        effective_rank = torch.exp(entropy).item()
        return effective_rank