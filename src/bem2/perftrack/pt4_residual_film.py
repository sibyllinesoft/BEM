"""
PT4 - Residual FiLM Micro-γ,β Implementation.

Implements tiny per-block modulation with strict clamps using controller-driven
feature-wise transformations with minimal parameter overhead.

Expected Performance: Validity improvements with no regressions
Budget Constraint: ±5% params/FLOPs vs v1.3-stack anchor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import logging

from ..security.parameter_protection import ParameterProtector
from bem.modules.governance import SpectralGovernance

logger = logging.getLogger(__name__)


@dataclass
class ResidualFiLMConfig:
    """Configuration for PT4 Residual FiLM."""
    
    # FiLM parameters
    gamma_dim: int = 16              # Dimension for γ (scaling) parameters
    beta_dim: int = 16               # Dimension for β (shift) parameters
    controller_dim: int = 32         # Hidden dimension for controller network
    
    # Micro-modulation parameters
    micro_scale: float = 0.01        # Tiny scaling for micro-modulation
    clamp_range: float = 0.1         # Strict clamping range [-clamp, +clamp]
    residual_scale: float = 0.1      # Scale for residual connection
    
    # Controller parameters
    controller_layers: int = 2       # Depth of controller network
    controller_activation: str = "gelu"  # Activation function
    controller_dropout: float = 0.05  # Light dropout
    
    # Feature-wise transformation
    use_channel_wise: bool = True    # Apply channel-wise modulation
    use_spatial_wise: bool = False   # Apply spatial-wise modulation (expensive)
    normalize_features: bool = True  # Normalize features before modulation
    
    # Stability constraints
    max_gamma: float = 1.2           # Maximum gamma value
    min_gamma: float = 0.8           # Minimum gamma value
    max_beta: float = 0.1            # Maximum beta magnitude
    
    # Budget constraints
    budget_tolerance: float = 0.05   # ±5% tolerance
    minimal_overhead: bool = True    # Ensure minimal parameter overhead


class MicroModulationController(nn.Module):
    """Controller network for computing γ and β parameters."""
    
    def __init__(self, config: ResidualFiLMConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Controller network layers
        controller_layers = []
        input_dim = hidden_size
        
        for i in range(config.controller_layers):
            if i < config.controller_layers - 1:
                # Hidden layers
                controller_layers.extend([
                    nn.Linear(input_dim, config.controller_dim),
                    nn.LayerNorm(config.controller_dim),
                    self._get_activation(config.controller_activation),
                    nn.Dropout(config.controller_dropout)
                ])
                input_dim = config.controller_dim
            else:
                # Final layer - output γ and β
                output_dim = config.gamma_dim + config.beta_dim
                controller_layers.append(nn.Linear(input_dim, output_dim))
        
        self.controller = nn.Sequential(*controller_layers)
        
        # Feature normalization
        if config.normalize_features:
            self.feature_norm = nn.LayerNorm(hidden_size)
        else:
            self.feature_norm = nn.Identity()
        
        # Initialize with small weights for minimal impact
        self._initialize_minimal()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation, nn.GELU())
    
    def _initialize_minimal(self):
        """Initialize parameters for minimal initial impact."""
        for module in self.controller.modules():
            if isinstance(module, nn.Linear):
                # Very small initialization
                nn.init.uniform_(module.weight, -self.config.micro_scale, self.config.micro_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute γ and β parameters from hidden states.
        
        Args:
            hidden_states: Input tensor [B, S, D]
            
        Returns:
            gamma: Scaling parameters [B, S, gamma_dim] or [B, gamma_dim]
            beta: Shift parameters [B, S, beta_dim] or [B, beta_dim]
            metrics: Dictionary of controller metrics
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Normalize features
        normalized_features = self.feature_norm(hidden_states)
        
        # Global pooling for controller input (reduce to per-sequence)
        if self.config.use_spatial_wise:
            # Keep spatial dimension - more expensive
            controller_input = normalized_features  # [B, S, D]
            controller_output = self.controller(controller_input)  # [B, S, gamma_dim + beta_dim]
        else:
            # Global average pooling - more efficient
            controller_input = torch.mean(normalized_features, dim=1)  # [B, D]
            controller_output = self.controller(controller_input)  # [B, gamma_dim + beta_dim]
            
            # Broadcast to sequence dimension
            controller_output = controller_output.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Split into γ and β
        gamma_raw = controller_output[..., :self.config.gamma_dim]
        beta_raw = controller_output[..., self.config.gamma_dim:]
        
        # Apply strict clamping and scaling
        gamma = torch.clamp(
            gamma_raw * self.config.micro_scale,
            min=-self.config.clamp_range,
            max=self.config.clamp_range
        )
        # Convert to multiplicative form: γ = 1 + small_adjustment
        gamma = 1.0 + gamma
        gamma = torch.clamp(gamma, min=self.config.min_gamma, max=self.config.max_gamma)
        
        beta = torch.clamp(
            beta_raw * self.config.micro_scale,
            min=-self.config.max_beta,
            max=self.config.max_beta
        )
        
        # Compute metrics
        metrics = {
            'gamma_mean': torch.mean(gamma).detach(),
            'gamma_std': torch.std(gamma).detach(),
            'gamma_range': (torch.max(gamma) - torch.min(gamma)).detach(),
            'beta_mean': torch.mean(beta).detach(),
            'beta_std': torch.std(beta).detach(),
            'beta_magnitude': torch.mean(torch.abs(beta)).detach(),
            'controller_activation': torch.mean(torch.abs(controller_output)).detach()
        }
        
        return gamma, beta, metrics


class FeatureWiseModulation(nn.Module):
    """Feature-wise linear modulation (FiLM) layer."""
    
    def __init__(self, config: ResidualFiLMConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Projection layers for γ and β to match feature dimensions
        if config.use_channel_wise:
            # Project to full hidden dimension for channel-wise modulation
            self.gamma_proj = nn.Linear(config.gamma_dim, hidden_size, bias=False)
            self.beta_proj = nn.Linear(config.beta_dim, hidden_size, bias=False)
        else:
            # Use lower-dimensional modulation
            assert config.gamma_dim <= hidden_size and config.beta_dim <= hidden_size
            self.gamma_proj = nn.Identity()
            self.beta_proj = nn.Identity()
        
        # Initialize projection layers with tiny weights
        if hasattr(self.gamma_proj, 'weight'):
            nn.init.uniform_(self.gamma_proj.weight, -config.micro_scale, config.micro_scale)
        if hasattr(self.beta_proj, 'weight'):
            nn.init.uniform_(self.beta_proj.weight, -config.micro_scale, config.micro_scale)
    
    def forward(
        self,
        features: torch.Tensor,      # [B, S, D]
        gamma: torch.Tensor,         # [B, S, gamma_dim] or [B, gamma_dim]
        beta: torch.Tensor           # [B, S, beta_dim] or [B, beta_dim]
    ) -> torch.Tensor:
        """Apply feature-wise modulation."""
        
        # Project γ and β to feature dimensions
        if self.config.use_channel_wise:
            gamma_projected = self.gamma_proj(gamma)  # [B, S, D] or [B, D]
            beta_projected = self.beta_proj(beta)     # [B, S, D] or [B, D]
        else:
            # Use only the first gamma_dim and beta_dim features
            gamma_projected = torch.cat([
                gamma,
                torch.ones_like(features[..., self.config.gamma_dim:])
            ], dim=-1)
            beta_projected = torch.cat([
                beta,
                torch.zeros_like(features[..., self.config.beta_dim:])
            ], dim=-1)
        
        # Ensure broadcasting compatibility
        if gamma_projected.dim() == 2:  # [B, D]
            gamma_projected = gamma_projected.unsqueeze(1)
        if beta_projected.dim() == 2:   # [B, D]
            beta_projected = beta_projected.unsqueeze(1)
        
        # Apply FiLM transformation: γ ⊙ x + β
        modulated_features = gamma_projected * features + beta_projected
        
        return modulated_features


class ResidualFiLMModule(nn.Module):
    """
    PT4 Residual FiLM Module.
    
    Implements tiny per-block modulation with strict clamps and minimal
    parameter overhead for validity improvements.
    """
    
    def __init__(
        self,
        config: ResidualFiLMConfig,
        layer_idx: int,
        hidden_size: int
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        
        # Controller for computing γ and β
        self.controller = MicroModulationController(config, hidden_size)
        
        # Feature-wise modulation layer
        self.modulation = FeatureWiseModulation(config, hidden_size)
        
        # Spectral governance (very conservative for stability)
        self.spectral_gov = SpectralGovernance(
            max_singular_value=0.5,  # Extra conservative
            layer_specific=True
        )
        
        # Parameter protection
        self.param_protector = ParameterProtector(
            max_norm=config.clamp_range,
            clip_gradient=True
        )
        
        # Residual scaling parameter (learnable but initialized small)
        self.residual_scale = nn.Parameter(
            torch.tensor(config.residual_scale)
        )
        
        # Track stability metrics
        self.register_buffer('modulation_magnitude_ema', torch.tensor(0.0))
        self.register_buffer('output_variance_ema', torch.tensor(1.0))
        
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, S, D]
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with residual FiLM modulation."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute γ and β parameters
        gamma, beta, controller_metrics = self.controller(hidden_states)
        
        # Apply feature-wise modulation
        modulated_features = self.modulation(hidden_states, gamma, beta)
        
        # Compute delta (change from original)
        delta = modulated_features - hidden_states
        
        # Apply spectral governance to delta
        delta_clamped, spectral_metrics = self.spectral_gov.apply_spectral_clamp(
            delta.view(-1, hidden_dim),
            layer_name=f"pt4_film_{self.layer_idx}",
            target_sigma1=self.config.clamp_range
        )
        delta_clamped = delta_clamped.view(batch_size, seq_len, hidden_dim)
        
        # Apply residual scaling
        scaled_delta = self.residual_scale * delta_clamped
        
        # Final output with residual connection
        output = hidden_states + scaled_delta
        
        # Update stability tracking
        if self.training:
            modulation_magnitude = torch.mean(torch.abs(scaled_delta))
            output_variance = torch.var(output)
            
            self.modulation_magnitude_ema.mul_(0.99).add_(modulation_magnitude.detach(), alpha=0.01)
            self.output_variance_ema.mul_(0.99).add_(output_variance.detach(), alpha=0.01)
        
        # Stability check - emergency clamp if needed
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.warning(f"NaN/Inf detected in PT4 layer {self.layer_idx}, falling back to identity")
            output = hidden_states
        
        if output_attentions:
            attention_info = {
                **controller_metrics,
                **spectral_metrics,
                'gamma': gamma,
                'beta': beta,
                'delta': scaled_delta,
                'residual_scale': self.residual_scale.detach(),
                'modulation_magnitude': self.modulation_magnitude_ema,
                'output_variance': self.output_variance_ema,
                'stability_ratio': self.modulation_magnitude_ema / (self.output_variance_ema + 1e-8)
            }
            return output, attention_info
        
        return output, None
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        controller_params = sum(p.numel() for p in self.controller.parameters())
        modulation_params = sum(p.numel() for p in self.modulation.parameters())
        residual_params = 1  # residual_scale
        
        return {
            'controller': controller_params,
            'modulation': modulation_params,
            'residual_scale': residual_params,
            'total': controller_params + modulation_params + residual_params
        }
    
    def get_flop_count(self, batch_size: int, seq_len: int) -> Dict[str, int]:
        """Get FLOP count for budget validation."""
        
        # Controller computation
        controller_flops = batch_size * seq_len * (
            self.hidden_size * self.config.controller_dim +  # First layer
            self.config.controller_dim * (self.config.gamma_dim + self.config.beta_dim)  # Output layer
        )
        
        # If not spatial-wise, reduce by sequence length factor
        if not self.config.use_spatial_wise:
            controller_flops = batch_size * (
                self.hidden_size * self.config.controller_dim +
                self.config.controller_dim * (self.config.gamma_dim + self.config.beta_dim)
            )
        
        # Modulation computation
        if self.config.use_channel_wise:
            modulation_flops = batch_size * seq_len * (
                self.config.gamma_dim * self.hidden_size +  # gamma projection
                self.config.beta_dim * self.hidden_size +   # beta projection
                self.hidden_size * 2  # element-wise multiply and add
            )
        else:
            modulation_flops = batch_size * seq_len * self.hidden_size  # element-wise ops only
        
        return {
            'controller': controller_flops,
            'modulation': modulation_flops,
            'total': controller_flops + modulation_flops
        }
    
    def get_minimal_overhead_ratio(self) -> float:
        """Get overhead ratio compared to base transformer layer."""
        # Estimate base transformer layer parameters (simplified)
        base_params = (
            self.hidden_size * self.hidden_size * 4 +  # QKV + output projections
            self.hidden_size * 4 * self.hidden_size * 2  # FFN
        )
        
        # Our parameter count
        our_params = self.get_parameter_count()['total']
        
        # Overhead ratio
        overhead_ratio = our_params / base_params
        
        return overhead_ratio
    
    def compute_budget_metrics(self, batch_size: int = 32, seq_len: int = 128) -> Dict[str, float]:
        """Compute comprehensive budget metrics."""
        param_counts = self.get_parameter_count()
        flop_counts = self.get_flop_count(batch_size, seq_len)
        overhead_ratio = self.get_minimal_overhead_ratio()
        
        # Memory estimate (minimal due to small parameters)
        memory_mb = (
            param_counts['total'] * 4 +  # Parameters (tiny)
            batch_size * seq_len * self.hidden_size * 4 +  # Hidden states
            batch_size * seq_len * (self.config.gamma_dim + self.config.beta_dim) * 4  # γ, β
        ) / (1024 * 1024)
        
        return {
            **param_counts,
            **flop_counts,
            'memory_mb': memory_mb,
            'overhead_ratio': overhead_ratio,
            'gamma_dim': self.config.gamma_dim,
            'beta_dim': self.config.beta_dim,
            'minimal_design': self.config.minimal_overhead,
            'residual_scale_value': self.residual_scale.item()
        }
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability and safety metrics."""
        with torch.no_grad():
            # Current parameter magnitudes
            controller_norm = sum(
                torch.norm(p).item() for p in self.controller.parameters()
            )
            
            modulation_norm = sum(
                torch.norm(p).item() for p in self.modulation.parameters()
            )
            
            return {
                'controller_param_norm': controller_norm,
                'modulation_param_norm': modulation_norm,
                'residual_scale': self.residual_scale.item(),
                'modulation_magnitude_ema': self.modulation_magnitude_ema.item(),
                'output_variance_ema': self.output_variance_ema.item(),
                'stability_indicator': self.modulation_magnitude_ema.item() / max(self.output_variance_ema.item(), 1e-6),
                'parameter_safety': float(controller_norm < 1.0 and modulation_norm < 1.0)
            }
    
    def emergency_stabilize(self):
        """Emergency stabilization - reset parameters to safe values."""
        logger.warning(f"Emergency stabilization triggered for PT4 layer {self.layer_idx}")
        
        # Reset controller to minimal values
        for module in self.controller.controller.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -self.config.micro_scale/10, self.config.micro_scale/10)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Reset modulation projections
        if hasattr(self.modulation.gamma_proj, 'weight'):
            nn.init.uniform_(self.modulation.gamma_proj.weight, -self.config.micro_scale/10, self.config.micro_scale/10)
        if hasattr(self.modulation.beta_proj, 'weight'):
            nn.init.uniform_(self.modulation.beta_proj.weight, -self.config.micro_scale/10, self.config.micro_scale/10)
        
        # Reset residual scale
        self.residual_scale.data.fill_(self.config.residual_scale / 10)
        
        logger.info(f"PT4 layer {self.layer_idx} stabilization complete")