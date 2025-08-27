"""
PT3 - Kronecker @ W_down (One Site) Implementation.

Implements (U⊗V) factorization with fused kernel at W_down attachment point only.
Maintains spectral clamps for stability and provides controlled testing with 
single attachment point.

Expected Performance: +0.5–1.5% chrF/BLEU improvement
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
class KroneckerConfig:
    """Configuration for PT3 Kronecker Factorization."""
    
    # Kronecker factorization parameters
    u_rank: int = 8              # Rank of U factor
    v_rank: int = 8              # Rank of V factor  
    u_dim: int = 64              # First dimension factor
    v_dim: int = 64              # Second dimension factor
    
    # Fused kernel parameters
    use_fused_kernel: bool = True        # Use custom CUDA kernel
    block_size: int = 128                # Block size for tiling
    memory_efficient: bool = True        # Use memory-efficient implementation
    
    # Initialization parameters
    init_method: str = "svd"             # "svd", "random", "xavier"
    init_scale: float = 0.1              # Scaling factor for initialization
    
    # Training parameters
    orthogonal_reg: float = 0.01         # Orthogonality regularization strength
    diversity_reg: float = 0.001         # Diversity regularization for factors
    
    # Trust region constraints
    max_singular_value: float = 1.0      # Spectral clamp
    fro_budget: float = 0.8              # Conservative Frobenius budget
    
    # Attachment configuration
    attachment_site: str = "W_down"      # Fixed to W_down only
    
    # Budget constraints
    budget_tolerance: float = 0.05       # ±5% tolerance


class FusedKroneckerOp(nn.Module):
    """
    Fused CUDA kernel for efficient Kronecker product computation.
    
    Implements optimized (U⊗V)x computation with memory efficiency.
    Falls back to standard PyTorch implementation if CUDA unavailable.
    """
    
    def __init__(self, config: KroneckerConfig):
        super().__init__()
        self.config = config
        self.use_fused = config.use_fused_kernel and torch.cuda.is_available()
        
        if self.use_fused:
            try:
                # Try to import custom CUDA kernels
                from bem.kernels.cuda_ops import kronecker_product_forward, kronecker_product_backward
                self.cuda_forward = kronecker_product_forward
                self.cuda_backward = kronecker_product_backward
                logger.info("Using fused CUDA kernels for Kronecker operations")
            except ImportError:
                logger.warning("CUDA kernels not available, falling back to PyTorch implementation")
                self.use_fused = False
    
    def forward(
        self, 
        x: torch.Tensor,          # Input tensor [B, ..., u_dim * v_dim]
        U: torch.Tensor,          # U factor [u_rank, u_dim]
        V: torch.Tensor           # V factor [v_rank, v_dim]
    ) -> torch.Tensor:
        """
        Compute (U ⊗ V) @ x efficiently.
        
        Args:
            x: Input of shape [..., u_dim * v_dim]
            U: Left factor of shape [u_rank, u_dim] 
            V: Right factor of shape [v_rank, v_dim]
            
        Returns:
            Output of shape [..., u_rank * v_rank]
        """
        if self.use_fused and x.is_cuda and U.is_cuda and V.is_cuda:
            return self._fused_forward(x, U, V)
        else:
            return self._pytorch_forward(x, U, V)
    
    def _fused_forward(self, x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """CUDA-accelerated forward pass."""
        try:
            # Reshape input for batch processing
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, self.config.u_dim * self.config.v_dim)
            
            # Call CUDA kernel
            output_flat = self.cuda_forward(
                x_flat, U, V,
                self.config.u_dim, self.config.v_dim,
                self.config.u_rank, self.config.v_rank,
                self.config.block_size
            )
            
            # Reshape back to original batch dimensions
            output_shape = batch_shape + (self.config.u_rank * self.config.v_rank,)
            return output_flat.view(output_shape)
            
        except Exception as e:
            logger.warning(f"CUDA kernel failed ({e}), falling back to PyTorch")
            return self._pytorch_forward(x, U, V)
    
    def _pytorch_forward(self, x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch implementation."""
        batch_shape = x.shape[:-1]
        
        # Reshape input to matrix form
        x_matrix = x.view(-1, self.config.u_dim, self.config.v_dim)  # [B, u_dim, v_dim]
        
        if self.config.memory_efficient:
            # Memory-efficient computation: avoid explicit Kronecker product
            # (U ⊗ V) @ vec(X) = vec(V^T @ X^T @ U^T)
            
            # X @ V^T: [B, u_dim, v_rank]
            temp = torch.matmul(x_matrix, V.T)
            
            # U @ temp^T: [u_rank, v_rank, B] -> [B, u_rank, v_rank]
            result = torch.matmul(U, temp.transpose(-2, -1)).transpose(-2, -1).transpose(-3, -1)
            
            # Flatten to [B, u_rank * v_rank]
            output = result.contiguous().view(-1, self.config.u_rank * self.config.v_rank)
        else:
            # Standard Kronecker product computation
            kron_uv = torch.kron(U, V)  # [u_rank * v_rank, u_dim * v_dim]
            x_flat = x.view(-1, self.config.u_dim * self.config.v_dim)
            output = torch.matmul(x_flat, kron_uv.T)
        
        # Reshape to original batch dimensions
        output_shape = batch_shape + (self.config.u_rank * self.config.v_rank,)
        return output.view(output_shape)


class KroneckerFactorization(nn.Module):
    """Core Kronecker factorization module with regularization."""
    
    def __init__(self, config: KroneckerConfig):
        super().__init__()
        self.config = config
        
        # Kronecker factors
        self.U = nn.Parameter(torch.randn(config.u_rank, config.u_dim))
        self.V = nn.Parameter(torch.randn(config.v_rank, config.v_dim))
        
        # Fused operation
        self.fused_op = FusedKroneckerOp(config)
        
        # Initialize parameters
        self._initialize_factors()
        
        # Track orthogonality and diversity
        self.register_buffer('u_orthogonality', torch.tensor(0.0))
        self.register_buffer('v_orthogonality', torch.tensor(0.0))
        self.register_buffer('factor_diversity', torch.tensor(0.0))
        
    def _initialize_factors(self):
        """Initialize U and V factors using specified method."""
        if self.config.init_method == "svd":
            # Initialize via SVD of random matrix
            random_matrix = torch.randn(
                self.config.u_dim * self.config.v_dim, 
                self.config.u_rank * self.config.v_rank
            )
            u_full, s, v_full = torch.svd(random_matrix)
            
            # Extract factors
            u_init = u_full[:self.config.u_dim, :self.config.u_rank].T
            v_init = v_full[:self.config.v_dim, :self.config.v_rank].T
            
            # Scale by singular values
            s_sqrt = torch.sqrt(s[:min(self.config.u_rank, self.config.v_rank)] + 1e-6)
            u_init *= s_sqrt.unsqueeze(-1)
            v_init *= s_sqrt.unsqueeze(-1)
            
        elif self.config.init_method == "random":
            u_init = torch.randn(self.config.u_rank, self.config.u_dim)
            v_init = torch.randn(self.config.v_rank, self.config.v_dim)
            
        elif self.config.init_method == "xavier":
            u_init = torch.empty(self.config.u_rank, self.config.u_dim)
            v_init = torch.empty(self.config.v_rank, self.config.v_dim)
            nn.init.xavier_uniform_(u_init)
            nn.init.xavier_uniform_(v_init)
            
        else:
            raise ValueError(f"Unknown init_method: {self.config.init_method}")
        
        # Apply scaling and orthogonalization
        u_init = F.normalize(u_init, dim=-1) * self.config.init_scale
        v_init = F.normalize(v_init, dim=-1) * self.config.init_scale
        
        self.U.data.copy_(u_init)
        self.V.data.copy_(v_init)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with Kronecker factorization.
        
        Args:
            x: Input tensor [..., u_dim * v_dim]
            
        Returns:
            output: Transformed tensor [..., u_rank * v_rank]
            metrics: Dictionary of regularization metrics
        """
        # Apply Kronecker transformation
        output = self.fused_op(x, self.U, self.V)
        
        # Compute regularization metrics
        metrics = self._compute_regularization_metrics()
        
        return output, metrics
    
    def _compute_regularization_metrics(self) -> Dict[str, torch.Tensor]:
        """Compute orthogonality and diversity regularization terms."""
        
        # Orthogonality regularization: penalize deviation from orthogonal
        u_gram = torch.matmul(self.U, self.U.T)
        v_gram = torch.matmul(self.V, self.V.T)
        
        u_identity = torch.eye(self.config.u_rank, device=self.U.device)
        v_identity = torch.eye(self.config.v_rank, device=self.V.device)
        
        u_orthogonal_loss = torch.norm(u_gram - u_identity, 'fro') ** 2
        v_orthogonal_loss = torch.norm(v_gram - v_identity, 'fro') ** 2
        
        orthogonal_reg = self.config.orthogonal_reg * (u_orthogonal_loss + v_orthogonal_loss)
        
        # Diversity regularization: encourage diverse factors
        u_diversity = -torch.sum(torch.var(self.U, dim=1))
        v_diversity = -torch.sum(torch.var(self.V, dim=1))
        diversity_reg = self.config.diversity_reg * (u_diversity + v_diversity)
        
        # Update tracking buffers
        self.u_orthogonality.mul_(0.9).add_(u_orthogonal_loss.detach(), alpha=0.1)
        self.v_orthogonality.mul_(0.9).add_(v_orthogonal_loss.detach(), alpha=0.1)
        self.factor_diversity.mul_(0.9).add_(-(u_diversity + v_diversity).detach(), alpha=0.1)
        
        return {
            'orthogonal_reg': orthogonal_reg,
            'diversity_reg': diversity_reg,
            'u_orthogonality': u_orthogonal_loss.detach(),
            'v_orthogonality': v_orthogonal_loss.detach(),
            'u_condition_number': torch.cond(self.U).detach(),
            'v_condition_number': torch.cond(self.V).detach()
        }
    
    def get_effective_rank(self) -> Tuple[float, float]:
        """Compute effective ranks of U and V factors."""
        with torch.no_grad():
            u_svd = torch.svd(self.U)[1]
            v_svd = torch.svd(self.V)[1]
            
            # Effective rank via normalized entropy
            u_normalized = u_svd / (torch.sum(u_svd) + 1e-8)
            v_normalized = v_svd / (torch.sum(v_svd) + 1e-8)
            
            u_eff_rank = torch.exp(-torch.sum(u_normalized * torch.log(u_normalized + 1e-8)))
            v_eff_rank = torch.exp(-torch.sum(v_normalized * torch.log(v_normalized + 1e-8)))
            
            return u_eff_rank.item(), v_eff_rank.item()


class KroneckerModule(nn.Module):
    """
    PT3 Kronecker Module for W_down attachment.
    
    Implements (U⊗V) factorization with fused kernel at W_down site only,
    maintaining spectral clamps for stability.
    """
    
    def __init__(
        self,
        config: KroneckerConfig,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Validate dimensions
        assert config.u_dim * config.v_dim == intermediate_size, \
            f"u_dim * v_dim ({config.u_dim * config.v_dim}) must equal intermediate_size ({intermediate_size})"
        
        assert config.u_rank * config.v_rank <= hidden_size, \
            f"u_rank * v_rank ({config.u_rank * config.v_rank}) must not exceed hidden_size ({hidden_size})"
        
        # Core Kronecker factorization
        self.kronecker = KroneckerFactorization(config)
        
        # Final projection to hidden_size (for residual compatibility)
        self.output_projection = nn.Linear(
            config.u_rank * config.v_rank, 
            hidden_size, 
            bias=False
        )
        
        # Spectral governance
        self.spectral_gov = SpectralGovernance(
            max_singular_value=config.max_singular_value,
            layer_specific=True
        )
        
        # Parameter protection
        self.param_protector = ParameterProtector(
            max_norm=config.max_singular_value,
            clip_gradient=True
        )
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, S, D] 
        intermediate_states: torch.Tensor,    # [B, S, intermediate_size] from W_down input
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with Kronecker factorization at W_down.
        
        Args:
            hidden_states: Original hidden states for residual
            intermediate_states: Intermediate activations from FFN
            output_attentions: Whether to return attention information
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply Kronecker factorization to intermediate states
        kronecker_output, kronecker_metrics = self.kronecker(intermediate_states)
        
        # Project to hidden dimension
        projected_output = self.output_projection(kronecker_output)
        
        # Apply spectral governance
        delta_w_clamped, spectral_metrics = self.spectral_gov.apply_spectral_clamp(
            projected_output.view(-1, hidden_dim),
            layer_name=f"pt3_kronecker_{self.layer_idx}",
            target_sigma1=self.config.max_singular_value
        )
        delta_w_clamped = delta_w_clamped.view(batch_size, seq_len, hidden_dim)
        
        # Residual connection
        output = hidden_states + delta_w_clamped
        
        if output_attentions:
            # Get effective ranks
            u_eff_rank, v_eff_rank = self.kronecker.get_effective_rank()
            
            attention_info = {
                **kronecker_metrics,
                **spectral_metrics,
                'kronecker_output': kronecker_output,
                'delta_w': delta_w_clamped,
                'u_effective_rank': u_eff_rank,
                'v_effective_rank': v_eff_rank,
                'compression_ratio': (self.config.u_dim * self.config.v_dim) / (self.config.u_rank * self.config.v_rank)
            }
            return output, attention_info
        
        return output, None
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        kronecker_params = self.kronecker.U.numel() + self.kronecker.V.numel()
        projection_params = self.output_projection.weight.numel()
        
        return {
            'kronecker_factors': kronecker_params,
            'output_projection': projection_params,
            'total': kronecker_params + projection_params
        }
    
    def get_flop_count(self, batch_size: int, seq_len: int) -> Dict[str, int]:
        """Get FLOP count for budget validation."""
        
        # Kronecker product computation
        if self.config.memory_efficient:
            # Memory-efficient: B*S*(u_dim*v_rank + u_rank*v_dim)
            kronecker_flops = batch_size * seq_len * (
                self.config.u_dim * self.config.v_rank +
                self.config.u_rank * self.config.v_dim  
            )
        else:
            # Standard: B*S*u_dim*v_dim*(u_rank*v_rank)
            kronecker_flops = batch_size * seq_len * (
                self.config.u_dim * self.config.v_dim * 
                self.config.u_rank * self.config.v_rank
            )
        
        # Output projection
        projection_flops = batch_size * seq_len * (
            self.config.u_rank * self.config.v_rank * self.hidden_size
        )
        
        return {
            'kronecker_computation': kronecker_flops,
            'output_projection': projection_flops,
            'total': kronecker_flops + projection_flops
        }
    
    def get_compression_metrics(self) -> Dict[str, float]:
        """Get compression and efficiency metrics."""
        # Original parameter count for equivalent full matrix
        original_params = self.config.u_dim * self.config.v_dim * self.hidden_size
        
        # Current parameter count
        current_params = self.get_parameter_count()['total']
        
        # Compression ratio
        compression_ratio = original_params / current_params
        
        # Effective ranks
        u_eff_rank, v_eff_rank = self.kronecker.get_effective_rank()
        
        return {
            'compression_ratio': compression_ratio,
            'parameter_reduction': 1.0 - (current_params / original_params),
            'u_effective_rank': u_eff_rank,
            'v_effective_rank': v_eff_rank,
            'rank_utilization_u': u_eff_rank / self.config.u_rank,
            'rank_utilization_v': v_eff_rank / self.config.v_rank,
            'total_rank_utilization': (u_eff_rank * v_eff_rank) / (self.config.u_rank * self.config.v_rank)
        }
    
    def compute_budget_metrics(self, batch_size: int = 32, seq_len: int = 128) -> Dict[str, float]:
        """Compute comprehensive budget metrics."""
        param_counts = self.get_parameter_count()
        flop_counts = self.get_flop_count(batch_size, seq_len)
        compression_metrics = self.get_compression_metrics()
        
        # Memory estimate
        memory_mb = (
            param_counts['total'] * 4 +  # Parameters
            batch_size * seq_len * self.hidden_size * 4 +  # Hidden states
            batch_size * seq_len * self.intermediate_size * 4 +  # Intermediate states
            batch_size * seq_len * self.config.u_rank * self.config.v_rank * 4  # Kronecker output
        ) / (1024 * 1024)
        
        return {
            **param_counts,
            **flop_counts,
            **compression_metrics,
            'memory_mb': memory_mb,
            'u_rank': self.config.u_rank,
            'v_rank': self.config.v_rank,
            'attachment_site': self.config.attachment_site
        }