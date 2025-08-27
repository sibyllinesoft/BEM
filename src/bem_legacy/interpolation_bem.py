"""
Interpolation BEM for the validation experiment.
This module handles interpolation between two static LoRAs based on controller output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy


class StaticLoRA(nn.Module):
    """A standard static LoRA module."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA forward pass: x @ A^T @ B^T"""
        result = self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result
    
    def get_delta_weight(self) -> torch.Tensor:
        """Get the effective weight delta: B @ A"""
        return (self.lora_B @ self.lora_A) * self.scaling


class InterpolationBEM(nn.Module):
    """
    BEM module that interpolates between two static LoRAs.
    This is the core module for the validation experiment.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        lora_json: StaticLoRA,
        lora_summary: StaticLoRA,
        controller_input_dim: int,
        controller_hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.lora_json = lora_json
        self.lora_summary = lora_summary
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Controller that outputs 2D interpolation weights
        if controller_hidden_dim is None:
            controller_hidden_dim = controller_input_dim * 4
            
        self.controller = nn.Sequential(
            nn.LayerNorm(controller_input_dim),
            nn.Linear(controller_input_dim, controller_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(controller_hidden_dim, controller_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(controller_hidden_dim // 2, 2),  # 2D output for interpolation
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        # Initialize controller with small weights
        for module in self.controller:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with interpolated LoRA.
        
        Args:
            x: Input tensor for the linear layer
            features: Features for the controller (e.g., task instruction embeddings)
            
        Returns:
            output: Base output + interpolated LoRA output
        """
        # Base forward pass
        base_output = self.base_layer(x)
        
        # Get interpolation weights from controller
        if features.dim() == 3:
            # Pool sequence dimension
            features = features.mean(dim=1)
        
        # Ensure features match controller precision
        features = features.to(next(self.controller.parameters()).dtype)
        weights = self.controller(features)  # [batch_size, 2]
        
        # Get LoRA outputs
        json_output = self.lora_json(x)
        summary_output = self.lora_summary(x)
        
        # Interpolate: c[0] * json_output + c[1] * summary_output
        interpolated_output = (
            weights[:, 0:1].unsqueeze(-1) * json_output +
            weights[:, 1:2].unsqueeze(-1) * summary_output
        )
        
        return base_output + interpolated_output
    
    def get_interpolation_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Get the current interpolation weights for analysis."""
        if features.dim() == 3:
            features = features.mean(dim=1)
        # Ensure features match controller precision
        features = features.to(next(self.controller.parameters()).dtype)
        return self.controller(features)
    
    def get_effective_delta_weight(self, features: torch.Tensor) -> torch.Tensor:
        """Get the effective weight delta for the current input."""
        weights = self.get_interpolation_weights(features)
        
        json_delta = self.lora_json.get_delta_weight()
        summary_delta = self.lora_summary.get_delta_weight()
        
        # Interpolate weight deltas
        effective_delta = (
            weights[0, 0] * json_delta + 
            weights[0, 1] * summary_delta
        )
        
        return effective_delta


def create_interpolation_bem(
    base_layer: nn.Linear,
    controller_input_dim: int,
    rank: int = 8,
    alpha: float = 16.0,
    controller_hidden_dim: Optional[int] = None,
    dropout: float = 0.1
) -> InterpolationBEM:
    """
    Create an InterpolationBEM with fresh LoRAs.
    
    Args:
        base_layer: Base linear layer
        controller_input_dim: Input dimension for controller
        rank: LoRA rank
        alpha: LoRA alpha
        controller_hidden_dim: Controller hidden dimension
        dropout: Dropout rate
        
    Returns:
        InterpolationBEM instance
    """
    # Create two static LoRAs
    lora_json = StaticLoRA(
        in_features=base_layer.in_features,
        out_features=base_layer.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    lora_summary = StaticLoRA(
        in_features=base_layer.in_features,
        out_features=base_layer.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    return InterpolationBEM(
        base_layer=base_layer,
        lora_json=lora_json,
        lora_summary=lora_summary,
        controller_input_dim=controller_input_dim,
        controller_hidden_dim=controller_hidden_dim,
        dropout=dropout
    )


def load_pretrained_loras_into_bem(
    base_layer: nn.Linear,
    json_lora_state: Dict[str, torch.Tensor],
    summary_lora_state: Dict[str, torch.Tensor],
    controller_input_dim: int,
    rank: int = 8,
    alpha: float = 16.0,
    controller_hidden_dim: Optional[int] = None,
    dropout: float = 0.1
) -> InterpolationBEM:
    """
    Create an InterpolationBEM with pre-trained LoRA weights.
    
    Args:
        base_layer: Base linear layer
        json_lora_state: State dict for JSON LoRA
        summary_lora_state: State dict for summary LoRA
        controller_input_dim: Input dimension for controller
        rank: LoRA rank
        alpha: LoRA alpha
        controller_hidden_dim: Controller hidden dimension
        dropout: Dropout rate
        
    Returns:
        InterpolationBEM instance with loaded LoRAs
    """
    # Create BEM
    bem = create_interpolation_bem(
        base_layer=base_layer,
        controller_input_dim=controller_input_dim,
        rank=rank,
        alpha=alpha,
        controller_hidden_dim=controller_hidden_dim,
        dropout=dropout
    )
    
    # Load pre-trained LoRA weights
    bem.lora_json.load_state_dict(json_lora_state)
    bem.lora_summary.load_state_dict(summary_lora_state)
    
    # Freeze the LoRAs - only train the controller
    for param in bem.lora_json.parameters():
        param.requires_grad = False
    for param in bem.lora_summary.parameters():
        param.requires_grad = False
    
    return bem


# Analysis utilities
def analyze_interpolation_behavior(
    bem: InterpolationBEM,
    json_features: torch.Tensor,
    summary_features: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Analyze how the BEM interpolates between tasks."""
    with torch.no_grad():
        json_weights = bem.get_interpolation_weights(json_features)
        summary_weights = bem.get_interpolation_weights(summary_features)
        
        return {
            'json_weights': json_weights,
            'summary_weights': summary_weights,
            'json_preference': json_weights[:, 0] - json_weights[:, 1],
            'summary_preference': summary_weights[:, 1] - summary_weights[:, 0],
            'weight_difference': torch.abs(json_weights - summary_weights)
        }


def compute_task_specialization_score(
    bem: InterpolationBEM,
    json_features: torch.Tensor,
    summary_features: torch.Tensor
) -> float:
    """
    Compute a score indicating how well the controller specializes for different tasks.
    Higher scores mean better task-specific adaptation.
    """
    with torch.no_grad():
        json_weights = bem.get_interpolation_weights(json_features)
        summary_weights = bem.get_interpolation_weights(summary_features)
        
        # JSON task should prefer JSON LoRA (index 0)
        json_specialization = json_weights[:, 0].mean()
        
        # Summary task should prefer summary LoRA (index 1)  
        summary_specialization = summary_weights[:, 1].mean()
        
        # Overall specialization score
        score = (json_specialization + summary_specialization) / 2.0
        
        return score.item()