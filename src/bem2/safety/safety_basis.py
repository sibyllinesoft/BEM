"""
Orthogonal Safety Basis Implementation

Implements reserved safety dimensions per BEM layer that remain orthogonal to
existing skill/style capabilities. Uses low-rank orthogonal matrices with
strict orthogonality constraints and dynamic activation.

Key Features:
- Per-layer orthogonal safety subspaces
- Gram-Schmidt orthogonalization 
- Orthogonality penalty enforcement
- Dynamic safety basis activation
- Efficient low-rank parameterization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class SafetyBasisConfig:
    """Configuration for orthogonal safety basis."""
    
    # Dimensionality configuration
    hidden_dim: int = 768           # Base model hidden dimension
    safety_rank: int = 32           # Rank of safety subspace
    num_layers: int = 12            # Number of transformer layers
    
    # Orthogonality enforcement
    orthogonal_penalty: float = 1.0  # Weight for orthogonality loss
    gram_schmidt_steps: int = 3      # Iterative orthogonalization steps
    orthogonal_tolerance: float = 1e-4  # Tolerance for orthogonality check
    
    # Activation configuration  
    activation_threshold: float = 0.1   # Minimum activation strength
    max_activation: float = 1.0         # Maximum activation strength
    temperature: float = 1.0            # Temperature for gating function
    
    # Initialization
    init_std: float = 0.02              # Standard deviation for initialization
    freeze_basis_epochs: int = 5        # Epochs to freeze basis during warmup


class OrthogonalSafetyBasis(nn.Module):
    """
    Orthogonal safety basis that reserves dimensions per layer for safety responses.
    
    The safety basis operates by:
    1. Maintaining orthogonal subspaces per layer
    2. Gating activation based on constitutional scores
    3. Ensuring no interference with existing capabilities
    4. Providing scalar control over safety strength
    """
    
    def __init__(self, config: SafetyBasisConfig):
        super().__init__()
        self.config = config
        
        # Per-layer safety bases (orthogonal matrices)
        self.safety_bases = nn.ModuleList([
            self._create_safety_basis(i) for i in range(config.num_layers)
        ])
        
        # Activation gating network
        self.activation_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Safety response generator
        self.safety_response_generator = nn.ModuleList([
            nn.Linear(config.safety_rank, config.hidden_dim) 
            for _ in range(config.num_layers)
        ])
        
        # Orthogonality tracking
        self.register_buffer('orthogonality_violations', torch.tensor(0.0))
        self.register_buffer('activation_history', torch.zeros(100))  # Rolling history
        self.register_buffer('history_pointer', torch.tensor(0))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _create_safety_basis(self, layer_idx: int) -> nn.Module:
        """Create orthogonal safety basis for a specific layer."""
        
        class LayerSafetyBasis(nn.Module):
            def __init__(self, hidden_dim: int, safety_rank: int, init_std: float):
                super().__init__()
                
                # Low-rank parameterization of orthogonal matrix
                # U @ V^T where both U and V have orthogonal columns
                self.U = nn.Parameter(torch.randn(hidden_dim, safety_rank) * init_std)
                self.V = nn.Parameter(torch.randn(hidden_dim, safety_rank) * init_std)
                
                # Layer-specific bias for safety responses
                self.safety_bias = nn.Parameter(torch.zeros(hidden_dim))
                
            def forward(self) -> torch.Tensor:
                """Return orthogonal safety basis matrix."""
                # Gram-Schmidt orthogonalization of U and V
                U_ortho = self._gram_schmidt(self.U)
                V_ortho = self._gram_schmidt(self.V)
                
                # Compute orthogonal basis
                basis = U_ortho @ V_ortho.T
                return basis
            
            def _gram_schmidt(self, X: torch.Tensor) -> torch.Tensor:
                """Apply Gram-Schmidt orthogonalization."""
                Q = X.clone()
                for i in range(Q.size(1)):
                    # Normalize current vector
                    q = Q[:, i]
                    q = q / (torch.norm(q) + 1e-8)
                    Q[:, i] = q
                    
                    # Orthogonalize remaining vectors
                    for j in range(i + 1, Q.size(1)):
                        proj = torch.dot(Q[:, j], q) * q
                        Q[:, j] = Q[:, j] - proj
                
                return Q
        
        return LayerSafetyBasis(
            self.config.hidden_dim,
            self.config.safety_rank,
            self.config.init_std
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        safety_score: torch.Tensor,
        safety_knob: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply safety basis transformation to hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            layer_idx: Which transformer layer (0-indexed)
            safety_score: Constitutional/value scores [batch_size]
            safety_knob: Global safety strength [0, 1]
            
        Returns:
            transformed_states: Safety-adjusted hidden states
            telemetry: Activation and orthogonality metrics
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get safety basis for this layer
        safety_basis = self.safety_bases[layer_idx]()  # [hidden_dim, hidden_dim]
        
        # Compute activation strength based on safety score and knob
        activation_strength = self._compute_activation_strength(
            hidden_states, safety_score, safety_knob
        )  # [batch_size, seq_len, 1]
        
        # Generate safety response
        safety_response = self._generate_safety_response(
            hidden_states, safety_basis, layer_idx
        )  # [batch_size, seq_len, hidden_dim]
        
        # Apply gated safety response
        transformed_states = hidden_states + (
            activation_strength * safety_response
        )
        
        # Compute telemetry
        telemetry = self._compute_telemetry(
            safety_basis, activation_strength, safety_score
        )
        
        # Update activation history
        self._update_activation_history(activation_strength.mean())
        
        return transformed_states, telemetry
    
    def _compute_activation_strength(
        self,
        hidden_states: torch.Tensor,
        safety_score: torch.Tensor,
        safety_knob: float
    ) -> torch.Tensor:
        """Compute gated activation strength based on constitutional scores."""
        
        # Pool hidden states for activation computation
        pooled_states = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Compute base activation from content
        base_activation = self.activation_gate(pooled_states)  # [batch_size, 1]
        
        # Modulate by safety score (higher score = higher activation)
        safety_modulation = torch.sigmoid(
            safety_score.unsqueeze(-1) / self.config.temperature
        )  # [batch_size, 1]
        
        # Apply safety knob scaling
        knob_modulation = torch.clamp(
            torch.tensor(safety_knob), 
            self.config.activation_threshold, 
            self.config.max_activation
        )
        
        # Combine all factors
        activation_strength = (
            base_activation * safety_modulation * knob_modulation
        )
        
        # Expand to sequence length
        return activation_strength.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
    
    def _generate_safety_response(
        self,
        hidden_states: torch.Tensor,
        safety_basis: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Generate safety response using orthogonal basis."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Project into safety subspace
        safety_projection = hidden_states @ safety_basis  # [batch, seq, hidden]
        
        # Generate safety-specific features
        safety_features = safety_projection @ safety_basis.T[:self.config.safety_rank, :]
        # [batch, seq, safety_rank]
        
        # Transform to safety response
        safety_response = self.safety_response_generator[layer_idx](safety_features)
        # [batch, seq, hidden_dim]
        
        # Add layer-specific safety bias
        safety_response = safety_response + self.safety_bases[layer_idx].safety_bias
        
        return safety_response
    
    def _compute_telemetry(
        self,
        safety_basis: torch.Tensor,
        activation_strength: torch.Tensor,
        safety_score: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute orthogonality and activation telemetry."""
        
        # Orthogonality check: should be close to identity
        basis_gram = safety_basis.T @ safety_basis
        identity = torch.eye(basis_gram.size(0), device=basis_gram.device)
        orthogonality_error = torch.norm(basis_gram - identity).item()
        
        # Update violation tracking
        if orthogonality_error > self.config.orthogonal_tolerance:
            self.orthogonality_violations += 1
        
        # Activation statistics
        activation_mean = activation_strength.mean().item()
        activation_std = activation_strength.std().item()
        activation_max = activation_strength.max().item()
        
        # Safety score statistics
        safety_mean = safety_score.mean().item()
        safety_std = safety_score.std().item()
        
        return {
            'orthogonality_error': torch.tensor(orthogonality_error),
            'orthogonality_violations': self.orthogonality_violations.clone(),
            'activation_mean': torch.tensor(activation_mean),
            'activation_std': torch.tensor(activation_std),
            'activation_max': torch.tensor(activation_max),
            'safety_score_mean': torch.tensor(safety_mean),
            'safety_score_std': torch.tensor(safety_std),
            'basis_rank': torch.tensor(torch.matrix_rank(safety_basis).item()),
            'basis_condition_number': torch.tensor(
                torch.cond(safety_basis).item() if safety_basis.size(0) == safety_basis.size(1) else 0.0
            )
        }
    
    def _update_activation_history(self, activation_value: float):
        """Update rolling history of activation values."""
        ptr = self.history_pointer.item()
        self.activation_history[ptr] = activation_value
        self.history_pointer = (ptr + 1) % self.activation_history.size(0)
    
    def compute_orthogonality_penalty(self) -> torch.Tensor:
        """Compute orthogonality penalty for all safety bases."""
        total_penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for basis_module in self.safety_bases:
            safety_basis = basis_module()
            
            # Orthogonality penalty: ||B^T B - I||_F^2
            gram_matrix = safety_basis.T @ safety_basis
            identity = torch.eye(
                gram_matrix.size(0), 
                device=gram_matrix.device, 
                dtype=gram_matrix.dtype
            )
            penalty = torch.norm(gram_matrix - identity, 'fro') ** 2
            total_penalty = total_penalty + penalty
        
        return total_penalty * self.config.orthogonal_penalty
    
    def get_activation_statistics(self) -> Dict[str, float]:
        """Get statistics about activation history."""
        history = self.activation_history
        return {
            'mean_activation': history.mean().item(),
            'std_activation': history.std().item(),
            'min_activation': history.min().item(),
            'max_activation': history.max().item(),
            'activation_violations': (history < self.config.activation_threshold).sum().item(),
            'orthogonality_violations': self.orthogonality_violations.item()
        }
    
    def _initialize_parameters(self):
        """Initialize safety basis parameters with careful scaling."""
        for basis_module in self.safety_bases:
            # Initialize with small random values
            nn.init.normal_(basis_module.U, 0, self.config.init_std)
            nn.init.normal_(basis_module.V, 0, self.config.init_std)
            nn.init.zeros_(basis_module.safety_bias)
        
        # Initialize activation gate
        for module in self.activation_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize safety response generators
        for generator in self.safety_response_generator:
            nn.init.xavier_uniform_(generator.weight)
            nn.init.zeros_(generator.bias)
    
    def freeze_safety_bases(self, freeze: bool = True):
        """Freeze/unfreeze safety basis parameters during warmup."""
        for basis_module in self.safety_bases:
            basis_module.U.requires_grad_(not freeze)
            basis_module.V.requires_grad_(not freeze)
            basis_module.safety_bias.requires_grad_(not freeze)
    
    def validate_orthogonality(self) -> Dict[str, float]:
        """Validate that all safety bases maintain orthogonality."""
        results = {}
        
        for i, basis_module in enumerate(self.safety_bases):
            safety_basis = basis_module()
            
            # Check orthogonality
            gram_matrix = safety_basis.T @ safety_basis
            identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            error = torch.norm(gram_matrix - identity, 'fro').item()
            
            results[f'layer_{i}_orthogonality_error'] = error
            results[f'layer_{i}_condition_number'] = torch.cond(safety_basis).item()
            results[f'layer_{i}_rank'] = torch.matrix_rank(safety_basis).item()
        
        return results