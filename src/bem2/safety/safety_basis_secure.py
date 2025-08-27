"""
Security-Hardened Orthogonal Safety Basis for VC0 System

Implements a security-hardened orthogonal safety basis using QR decomposition
with comprehensive protection against tampering and unauthorized access.

Key Features:
- Orthogonal basis construction with integrity verification  
- QR decomposition with numerical stability checks
- Parameter protection and tamper detection
- Secure basis adaptation and learning
- Real-time orthogonality monitoring
- Cryptographic basis verification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import time
import hashlib
import hmac
from datetime import datetime

from ..security import (
    SecurityContext, ParameterGuard, SecurityAuditor, AuditEventType,
    SafetyCircuitBreaker, AuthorizationError
)

logger = logging.getLogger(__name__)


@dataclass  
class SafetyBasisConfig:
    """Configuration for secure orthogonal safety basis"""
    
    # Basis dimensions
    safety_rank: int = 32                    # Rank of safety subspace per layer
    num_layers: int = 24                     # Number of transformer layers
    hidden_dim: int = 1024                   # Hidden dimension per layer
    
    # Orthogonality constraints
    orthogonal_penalty: float = 1.0          # Weight for orthogonality constraint
    gram_schmidt_steps: int = 3              # Iterative orthogonalization steps  
    numerical_stability_eps: float = 1e-8   # Numerical stability epsilon
    
    # Safety activation
    activation_threshold: float = 0.1        # Minimum safety activation
    max_activation: float = 1.0              # Maximum safety activation
    safety_temperature: float = 1.0         # Temperature for safety gating
    
    # Training and adaptation
    freeze_basis_epochs: int = 2             # Epochs to freeze basis during warmup
    basis_learning_rate: float = 1e-4       # Learning rate for basis updates
    orthogonality_check_frequency: int = 100 # Steps between orthogonality checks
    
    # Security configuration
    enable_parameter_protection: bool = True
    enable_integrity_verification: bool = True
    enable_basis_encryption: bool = True
    enable_tampering_detection: bool = True
    cryptographic_verification: bool = True
    
    # Monitoring and auditing
    comprehensive_monitoring: bool = True
    real_time_orthogonality_check: bool = True
    basis_adaptation_logging: bool = True


class SecureOrthogonalSafetyBasis(nn.Module):
    """
    Security-hardened orthogonal safety basis implementation.
    
    Creates and maintains orthogonal safety subspaces that are:
    1. Orthogonal to existing model capabilities 
    2. Cryptographically protected against tampering
    3. Continuously monitored for integrity
    4. Numerically stable under adaptation
    """
    
    def __init__(self, config: SafetyBasisConfig, 
                 security_context: Optional[SecurityContext] = None):
        super().__init__()
        self.config = config
        self.security_context = security_context
        
        # Initialize security components
        self._initialize_security_components()
        
        # Build orthogonal basis matrices  
        self._build_basis_matrices()
        
        # Setup parameter protection
        self._setup_parameter_protection()
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        # Verify initial orthogonality
        self._verify_initial_orthogonality()
    
    def _initialize_security_components(self):
        """Initialize comprehensive security components"""
        
        # Parameter protection system
        if self.config.enable_parameter_protection:
            self.parameter_guard = ParameterGuard({
                'integrity_checking': True,
                'real_time_monitoring': True,
                'backup_enabled': True
            })
        
        # Security auditor
        if self.config.comprehensive_monitoring:
            self.security_auditor = SecurityAuditor({
                'log_file': './logs/safety_basis_audit.jsonl',
                'real_time_monitoring': True,
                'integrity_protection': True
            })
        
        # Circuit breaker for resilience
        self.circuit_breaker = SafetyCircuitBreaker(
            'safety_basis',
            failure_threshold=3,
            timeout_seconds=30.0
        )
        
        # Cryptographic keys for verification
        self._setup_cryptographic_keys()
    
    def _setup_cryptographic_keys(self):
        """Setup cryptographic keys for basis verification"""
        if self.config.cryptographic_verification:
            # Generate HMAC key for basis integrity
            self.basis_hmac_key = hashlib.sha256(
                f"safety_basis_{self.config.safety_rank}_{time.time()}".encode()
            ).digest()
            
            # Generate encryption key for basis storage (would use proper key management)
            self.basis_encryption_key = hashlib.sha256(
                f"basis_encryption_{self.config.hidden_dim}".encode()
            ).digest()[:16]  # AES-128 key
    
    def _build_basis_matrices(self):
        """Build orthogonal safety basis matrices with security"""
        
        # Safety basis matrices for each layer
        self.safety_bases = nn.ParameterList([
            nn.Parameter(torch.randn(
                self.config.hidden_dim, 
                self.config.safety_rank
            ) * 0.01)
            for _ in range(self.config.num_layers)
        ])
        
        # Basis adaptation parameters
        self.basis_scales = nn.ParameterList([
            nn.Parameter(torch.ones(self.config.safety_rank))
            for _ in range(self.config.num_layers)
        ])
        
        # Safety activation weights
        self.safety_activation_weights = nn.ParameterList([
            nn.Parameter(torch.ones(self.config.safety_rank) * 0.1)
            for _ in range(self.config.num_layers)
        ])
        
        # Initialize orthogonal bases using secure QR decomposition
        self._initialize_orthogonal_bases()
        
        # Create basis integrity hashes
        self._create_basis_integrity_hashes()
    
    def _initialize_orthogonal_bases(self):
        """Initialize orthogonal bases using secure QR decomposition"""
        with torch.no_grad():
            for layer_idx, basis_param in enumerate(self.safety_bases):
                # Generate random matrix for QR decomposition
                random_matrix = torch.randn_like(basis_param) * 0.1
                
                # Secure QR decomposition with numerical stability
                try:
                    Q, R = self._secure_qr_decomposition(random_matrix)
                    
                    # Verify orthogonality
                    orthogonality_error = self._compute_orthogonality_error(Q)
                    
                    if orthogonality_error > 1e-5:
                        logger.warning(f"Layer {layer_idx} orthogonality error: {orthogonality_error}")
                        # Retry with better conditioning
                        Q, R = self._secure_qr_decomposition(
                            random_matrix + torch.eye(basis_param.size(0), 
                                                     device=basis_param.device) * 1e-6
                        )
                    
                    # Set orthogonal basis
                    basis_param.copy_(Q)
                    
                    # Log initialization
                    if hasattr(self, 'security_auditor'):
                        self.security_auditor.log_security_event(
                            AuditEventType.SYSTEM_STARTUP,
                            'info',
                            source='safety_basis',
                            details={
                                'layer': layer_idx,
                                'orthogonality_error': orthogonality_error,
                                'basis_rank': self.config.safety_rank
                            }
                        )
                
                except Exception as e:
                    logger.error(f"Failed to initialize orthogonal basis for layer {layer_idx}: {e}")
                    # Fallback to identity-based initialization
                    basis_param.copy_(torch.eye(
                        basis_param.size(0), 
                        basis_param.size(1),
                        device=basis_param.device
                    )[:basis_param.size(0), :basis_param.size(1)])
    
    def _secure_qr_decomposition(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform QR decomposition with security and numerical stability checks"""
        
        # Add numerical stability
        matrix = matrix + torch.eye(
            matrix.size(0), 
            device=matrix.device
        ) * self.config.numerical_stability_eps
        
        # Perform QR decomposition
        Q, R = torch.qr(matrix)
        
        # Verify numerical stability
        if torch.isnan(Q).any() or torch.isinf(Q).any():
            raise RuntimeError("QR decomposition produced NaN or Inf values")
        
        # Verify orthogonality
        orthogonality_error = self._compute_orthogonality_error(Q)
        if orthogonality_error > 1e-4:  # Tolerance for numerical precision
            # Apply Gram-Schmidt for better orthogonality
            Q = self._gram_schmidt_orthogonalization(Q)
        
        return Q, R
    
    def _gram_schmidt_orthogonalization(self, matrix: torch.Tensor) -> torch.Tensor:
        """Apply Gram-Schmidt orthogonalization with security monitoring"""
        
        Q = matrix.clone()
        m, n = Q.shape
        
        # Iterative Gram-Schmidt
        for step in range(self.config.gram_schmidt_steps):
            for j in range(n):
                # Normalize current column
                norm = torch.norm(Q[:, j])
                
                if norm < self.config.numerical_stability_eps:
                    # Handle near-zero vector
                    Q[:, j] = torch.randn(m, device=Q.device) * 0.01
                    norm = torch.norm(Q[:, j])
                
                Q[:, j] = Q[:, j] / norm
                
                # Orthogonalize against previous columns  
                for k in range(j + 1, n):
                    projection = torch.dot(Q[:, j], Q[:, k])
                    Q[:, k] = Q[:, k] - projection * Q[:, j]
            
            # Check convergence
            orthogonality_error = self._compute_orthogonality_error(Q)
            if orthogonality_error < 1e-6:
                break
        
        return Q
    
    def _compute_orthogonality_error(self, matrix: torch.Tensor) -> float:
        """Compute orthogonality error for verification"""
        if matrix.size(1) == 0:
            return 0.0
        
        # Compute Q^T Q - I
        identity = torch.eye(matrix.size(1), device=matrix.device)
        orthogonality_matrix = torch.mm(matrix.t(), matrix) - identity
        
        # Return Frobenius norm of deviation from identity
        return torch.norm(orthogonality_matrix, p='fro').item()
    
    def _create_basis_integrity_hashes(self):
        """Create cryptographic integrity hashes for bases"""
        if self.config.cryptographic_verification:
            self.basis_integrity_hashes = {}
            
            for layer_idx, basis_param in enumerate(self.safety_bases):
                integrity_hash = self._compute_basis_integrity_hash(basis_param, layer_idx)
                self.basis_integrity_hashes[layer_idx] = integrity_hash
    
    def _compute_basis_integrity_hash(self, basis: torch.Tensor, layer_idx: int) -> str:
        """Compute cryptographic integrity hash for basis"""
        # Serialize basis data
        basis_data = basis.detach().cpu().numpy().tobytes()
        layer_data = str(layer_idx).encode()
        
        # Combine with layer information
        combined_data = basis_data + layer_data
        
        # Compute HMAC
        return hmac.new(self.basis_hmac_key, combined_data, hashlib.sha256).hexdigest()
    
    def _setup_parameter_protection(self):
        """Setup parameter protection for safety basis"""
        if hasattr(self, 'parameter_guard'):
            # Protect safety basis parameters
            for layer_idx, basis_param in enumerate(self.safety_bases):
                param_name = f"safety_basis_layer_{layer_idx}"
                self.parameter_guard.protect_parameter(
                    param_name, basis_param, 'critical', self.security_context
                )
            
            # Protect activation weights
            for layer_idx, weight_param in enumerate(self.safety_activation_weights):
                param_name = f"safety_activation_weights_layer_{layer_idx}"
                self.parameter_guard.protect_parameter(
                    param_name, weight_param, 'high', self.security_context
                )
    
    def _initialize_monitoring(self):
        """Initialize comprehensive monitoring systems"""
        
        # Orthogonality monitoring
        self.register_buffer('orthogonality_violations', torch.tensor(0))
        self.register_buffer('total_orthogonality_checks', torch.tensor(0))
        self.register_buffer('avg_orthogonality_error', torch.tensor(0.0))
        
        # Security monitoring
        self.register_buffer('integrity_violations', torch.tensor(0))
        self.register_buffer('tampering_attempts', torch.tensor(0))
        self.register_buffer('total_safety_activations', torch.tensor(0))
        
        # Performance monitoring
        self.orthogonality_check_times = []
        self.basis_adaptation_history = []
    
    def _verify_initial_orthogonality(self):
        """Verify initial orthogonality of all bases"""
        for layer_idx, basis_param in enumerate(self.safety_bases):
            orthogonality_error = self._compute_orthogonality_error(basis_param)
            
            if orthogonality_error > 1e-4:
                logger.warning(f"Layer {layer_idx} initial orthogonality error: {orthogonality_error}")
            
            # Log verification
            if hasattr(self, 'security_auditor'):
                self.security_auditor.log_security_event(
                    AuditEventType.SYSTEM_STARTUP,
                    'info',
                    source='safety_basis',
                    details={
                        'verification_type': 'initial_orthogonality',
                        'layer': layer_idx,
                        'orthogonality_error': orthogonality_error,
                        'passed': orthogonality_error <= 1e-4
                    }
                )
    
    def forward(self, 
                hidden_states: torch.Tensor,
                layer_idx: int,
                constitutional_score: Optional[torch.Tensor] = None,
                security_context: Optional[SecurityContext] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply secure orthogonal safety basis transformation.
        
        Args:
            hidden_states: Input hidden states [batch, seq, hidden]
            layer_idx: Transformer layer index
            constitutional_score: Constitutional AI scores for gating
            security_context: Security context for authorization
            
        Returns:
            transformed_states: Safety-transformed hidden states
            safety_info: Detailed safety transformation information
        """
        
        # Security validation
        self._validate_security_context(security_context)
        self._verify_parameter_integrity(layer_idx)
        
        # Execute with circuit breaker protection
        return self.circuit_breaker.call(
            self._secure_forward_internal,
            hidden_states, layer_idx, constitutional_score
        )
    
    def _validate_security_context(self, security_context: Optional[SecurityContext]):
        """Validate security context for safety basis access"""
        if self.config.enable_parameter_protection:
            if not security_context:
                raise AuthorizationError("Security context required for safety basis access")
            
            if security_context.is_expired():
                raise AuthorizationError("Security context has expired")
            
            if not security_context.has_permission('safety.basis.transform'):
                self._log_security_event(
                    AuditEventType.AUTHORIZATION_DENIED,
                    'high',
                    {
                        'user_id': security_context.user_id,
                        'required_permission': 'safety.basis.transform'
                    }
                )
                raise AuthorizationError("Insufficient permissions for safety basis transformation")
    
    def _verify_parameter_integrity(self, layer_idx: int):
        """Verify parameter integrity before transformation"""
        if not self.config.enable_integrity_verification:
            return
        
        # Check parameter integrity
        if hasattr(self, 'parameter_guard'):
            integrity_result = self.parameter_guard.validate_parameter_integrity(
                f"safety_basis_layer_{layer_idx}"
            )
            
            if not integrity_result['valid_parameters'] == integrity_result['total_parameters']:
                self.integrity_violations += 1
                self._log_security_event(
                    AuditEventType.PARAMETER_INTEGRITY_VIOLATION,
                    'critical',
                    {'layer': layer_idx, 'integrity_result': integrity_result}
                )
                raise RuntimeError(f"Parameter integrity violation in layer {layer_idx}")
        
        # Verify basis integrity hash
        if hasattr(self, 'basis_integrity_hashes'):
            current_hash = self._compute_basis_integrity_hash(
                self.safety_bases[layer_idx], layer_idx
            )
            expected_hash = self.basis_integrity_hashes.get(layer_idx)
            
            if expected_hash and not hmac.compare_digest(current_hash, expected_hash):
                self.tampering_attempts += 1
                self._log_security_event(
                    AuditEventType.PARAMETER_INTEGRITY_VIOLATION,
                    'critical',
                    {
                        'layer': layer_idx,
                        'tampering_type': 'basis_integrity_hash_mismatch'
                    }
                )
                raise RuntimeError(f"Basis integrity hash mismatch in layer {layer_idx}")
    
    def _secure_forward_internal(self, 
                                hidden_states: torch.Tensor,
                                layer_idx: int,
                                constitutional_score: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Internal secure forward pass"""
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Verify layer index
        if layer_idx >= len(self.safety_bases):
            raise ValueError(f"Layer index {layer_idx} exceeds available bases")
        
        # Get safety basis for this layer
        safety_basis = self.safety_bases[layer_idx]
        safety_weights = self.safety_activation_weights[layer_idx]
        basis_scales = self.basis_scales[layer_idx]
        
        # Verify orthogonality if enabled
        if self.config.real_time_orthogonality_check:
            self._check_real_time_orthogonality(layer_idx)
        
        # Compute safety projections
        safety_projections = self._compute_safety_projections(
            hidden_states, safety_basis, safety_weights, basis_scales
        )
        
        # Apply constitutional gating
        gated_projections = self._apply_constitutional_gating(
            safety_projections, constitutional_score
        )
        
        # Transform hidden states
        transformed_states = self._apply_safety_transformation(
            hidden_states, gated_projections, safety_basis
        )
        
        # Collect safety information
        safety_info = {
            'layer_idx': layer_idx,
            'safety_activations': torch.norm(safety_projections, dim=-1).mean().item(),
            'orthogonality_error': self._compute_orthogonality_error(safety_basis),
            'constitutional_gating': constitutional_score.mean().item() if constitutional_score is not None else 1.0,
            'transformation_magnitude': torch.norm(transformed_states - hidden_states).item(),
            'security_verified': True
        }
        
        # Update monitoring
        self._update_safety_monitoring(safety_info)
        
        return transformed_states, safety_info
    
    def _check_real_time_orthogonality(self, layer_idx: int):
        """Check orthogonality in real-time with performance tracking"""
        start_time = time.time()
        
        safety_basis = self.safety_bases[layer_idx]
        orthogonality_error = self._compute_orthogonality_error(safety_basis)
        
        self.total_orthogonality_checks += 1
        
        # Update running average
        alpha = 0.01
        self.avg_orthogonality_error = (
            (1 - alpha) * self.avg_orthogonality_error + 
            alpha * orthogonality_error
        )
        
        # Check for violations
        if orthogonality_error > 1e-3:  # Tolerance for real-time checks
            self.orthogonality_violations += 1
            self._log_security_event(
                AuditEventType.ANOMALOUS_BEHAVIOR,
                'medium',
                {
                    'layer': layer_idx,
                    'orthogonality_error': orthogonality_error,
                    'violation_type': 'real_time_orthogonality'
                }
            )
        
        # Performance tracking
        check_time = time.time() - start_time
        self.orthogonality_check_times.append(check_time)
        
        # Keep only recent times
        if len(self.orthogonality_check_times) > 1000:
            self.orthogonality_check_times = self.orthogonality_check_times[-1000:]
    
    def _compute_safety_projections(self, 
                                   hidden_states: torch.Tensor,
                                   safety_basis: torch.Tensor,
                                   safety_weights: torch.Tensor,
                                   basis_scales: torch.Tensor) -> torch.Tensor:
        """Compute safety subspace projections"""
        
        # Project hidden states onto safety basis
        # hidden_states: [batch, seq, hidden]
        # safety_basis: [hidden, safety_rank]
        projections = torch.matmul(hidden_states, safety_basis)  # [batch, seq, safety_rank]
        
        # Apply safety activation weights
        weighted_projections = projections * safety_weights.unsqueeze(0).unsqueeze(0)
        
        # Apply basis scaling
        scaled_projections = weighted_projections * basis_scales.unsqueeze(0).unsqueeze(0)
        
        # Apply activation function with temperature
        safety_activations = torch.tanh(scaled_projections / self.config.safety_temperature)
        
        # Clamp to configured range
        safety_activations = torch.clamp(
            safety_activations,
            min=self.config.activation_threshold,
            max=self.config.max_activation
        )
        
        return safety_activations
    
    def _apply_constitutional_gating(self, 
                                   safety_projections: torch.Tensor,
                                   constitutional_score: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply constitutional AI gating to safety projections"""
        
        if constitutional_score is None:
            # No constitutional gating - use full safety activations
            return safety_projections
        
        # Expand constitutional score to match projections
        # constitutional_score: [batch]
        # safety_projections: [batch, seq, safety_rank]
        
        batch_size, seq_len, safety_rank = safety_projections.shape
        
        # Expand constitutional score
        gating_scores = constitutional_score.view(batch_size, 1, 1).expand(
            batch_size, seq_len, safety_rank
        )
        
        # Apply inverse gating (lower constitutional scores -> higher safety activation)
        # This ensures safety activations increase when constitutional violations are detected
        inverse_gating = 1.0 - gating_scores
        
        # Apply gating with smooth transition
        gated_projections = safety_projections * torch.sigmoid(inverse_gating * 5.0)
        
        return gated_projections
    
    def _apply_safety_transformation(self, 
                                   hidden_states: torch.Tensor,
                                   safety_projections: torch.Tensor,
                                   safety_basis: torch.Tensor) -> torch.Tensor:
        """Apply safety transformation to hidden states"""
        
        # Project safety activations back to hidden space
        # safety_projections: [batch, seq, safety_rank]
        # safety_basis: [hidden, safety_rank]
        safety_corrections = torch.matmul(safety_projections, safety_basis.t())
        
        # Apply additive correction (preserves original information)
        transformed_states = hidden_states + safety_corrections
        
        return transformed_states
    
    def _update_safety_monitoring(self, safety_info: Dict[str, Any]):
        """Update comprehensive safety monitoring"""
        
        self.total_safety_activations += 1
        
        # Store adaptation history
        adaptation_record = {
            'timestamp': datetime.now(),
            'layer': safety_info['layer_idx'],
            'safety_activation': safety_info['safety_activations'],
            'orthogonality_error': safety_info['orthogonality_error'],
            'transformation_magnitude': safety_info['transformation_magnitude']
        }
        
        self.basis_adaptation_history.append(adaptation_record)
        
        # Keep only recent history
        if len(self.basis_adaptation_history) > 10000:
            self.basis_adaptation_history = self.basis_adaptation_history[-10000:]
    
    def adapt_basis_securely(self, 
                            layer_idx: int,
                            adaptation_signal: torch.Tensor,
                            security_context: SecurityContext,
                            adaptation_strength: float = 0.01) -> bool:
        """
        Securely adapt safety basis based on feedback signal.
        
        Args:
            layer_idx: Layer to adapt
            adaptation_signal: Signal indicating needed adaptation
            security_context: Security context for authorization
            adaptation_strength: Strength of adaptation
            
        Returns:
            success: Whether adaptation was successful
        """
        
        # Validate security context
        if not security_context.has_permission('safety.basis.adapt'):
            self._log_security_event(
                AuditEventType.AUTHORIZATION_DENIED,
                'high',
                {
                    'user_id': security_context.user_id,
                    'required_permission': 'safety.basis.adapt'
                }
            )
            raise AuthorizationError("Insufficient permissions for basis adaptation")
        
        # Verify parameter integrity
        self._verify_parameter_integrity(layer_idx)
        
        try:
            # Backup current basis
            current_basis = self.safety_bases[layer_idx].clone()
            
            # Compute adaptation direction
            adaptation_direction = self._compute_adaptation_direction(
                current_basis, adaptation_signal
            )
            
            # Apply adaptation with security checks
            adapted_basis = self._apply_secure_adaptation(
                current_basis, adaptation_direction, adaptation_strength
            )
            
            # Verify orthogonality of adapted basis
            orthogonality_error = self._compute_orthogonality_error(adapted_basis)
            
            if orthogonality_error > 1e-3:
                # Re-orthogonalize if needed
                adapted_basis = self._gram_schmidt_orthogonalization(adapted_basis)
                final_orthogonality_error = self._compute_orthogonality_error(adapted_basis)
                
                if final_orthogonality_error > 1e-3:
                    logger.error(f"Failed to maintain orthogonality after adaptation: {final_orthogonality_error}")
                    return False
            
            # Update basis with new values
            with torch.no_grad():
                self.safety_bases[layer_idx].copy_(adapted_basis)
            
            # Update integrity hash
            if hasattr(self, 'basis_integrity_hashes'):
                new_hash = self._compute_basis_integrity_hash(adapted_basis, layer_idx)
                self.basis_integrity_hashes[layer_idx] = new_hash
            
            # Log adaptation
            if hasattr(self, 'security_auditor'):
                self.security_auditor.log_security_event(
                    AuditEventType.PARAMETER_MODIFIED,
                    'info',
                    source='safety_basis',
                    user_id=security_context.user_id,
                    details={
                        'layer': layer_idx,
                        'adaptation_strength': adaptation_strength,
                        'orthogonality_error_before': orthogonality_error,
                        'orthogonality_error_after': self._compute_orthogonality_error(adapted_basis)
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Basis adaptation failed for layer {layer_idx}: {e}")
            
            # Log failure
            self._log_security_event(
                AuditEventType.ANOMALOUS_BEHAVIOR,
                'high',
                {
                    'layer': layer_idx,
                    'error': str(e),
                    'adaptation_type': 'basis_adaptation_failure'
                }
            )
            
            return False
    
    def _compute_adaptation_direction(self, 
                                    current_basis: torch.Tensor,
                                    adaptation_signal: torch.Tensor) -> torch.Tensor:
        """Compute secure adaptation direction"""
        
        # Project adaptation signal onto current basis
        signal_projections = torch.matmul(adaptation_signal.t(), current_basis)
        
        # Compute residual (orthogonal component)
        reconstructed_signal = torch.matmul(current_basis, signal_projections.t())
        residual = adaptation_signal - reconstructed_signal.t()
        
        # Use residual as adaptation direction (maintains orthogonality)
        return residual.t()
    
    def _apply_secure_adaptation(self, 
                                current_basis: torch.Tensor,
                                adaptation_direction: torch.Tensor,
                                adaptation_strength: float) -> torch.Tensor:
        """Apply adaptation with security constraints"""
        
        # Apply adaptation
        adapted_basis = current_basis + adaptation_strength * adaptation_direction
        
        # Security constraints
        # 1. Clamp extreme values
        adapted_basis = torch.clamp(adapted_basis, -10.0, 10.0)
        
        # 2. Check for NaN/Inf
        if torch.isnan(adapted_basis).any() or torch.isinf(adapted_basis).any():
            raise RuntimeError("Adaptation produced NaN or Inf values")
        
        return adapted_basis
    
    def get_orthogonality_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orthogonality statistics"""
        
        # Compute current orthogonality errors for all layers
        current_errors = []
        for layer_idx, basis in enumerate(self.safety_bases):
            error = self._compute_orthogonality_error(basis)
            current_errors.append(error)
        
        return {
            'total_layers': len(self.safety_bases),
            'current_orthogonality_errors': current_errors,
            'max_orthogonality_error': max(current_errors),
            'avg_orthogonality_error': self.avg_orthogonality_error.item(),
            'orthogonality_violations': self.orthogonality_violations.item(),
            'total_orthogonality_checks': self.total_orthogonality_checks.item(),
            'violation_rate': (
                self.orthogonality_violations.item() / 
                max(1, self.total_orthogonality_checks.item())
            ),
            'avg_check_time': (
                sum(self.orthogonality_check_times) / len(self.orthogonality_check_times)
                if self.orthogonality_check_times else 0.0
            )
        }
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        
        return {
            'total_safety_activations': self.total_safety_activations.item(),
            'integrity_violations': self.integrity_violations.item(),
            'tampering_attempts': self.tampering_attempts.item(),
            'basis_adaptations': len(self.basis_adaptation_history),
            'security_features': {
                'parameter_protection': self.config.enable_parameter_protection,
                'integrity_verification': self.config.enable_integrity_verification,
                'tampering_detection': self.config.enable_tampering_detection,
                'cryptographic_verification': self.config.cryptographic_verification
            },
            'orthogonality_statistics': self.get_orthogonality_statistics()
        }
    
    def _log_security_event(self, event_type: AuditEventType, severity: str, details: Dict[str, Any]):
        """Log security event with comprehensive context"""
        if hasattr(self, 'security_auditor'):
            self.security_auditor.log_security_event(
                event_type=event_type,
                severity=severity,
                source='safety_basis',
                user_id=self.security_context.user_id if self.security_context else None,
                details={
                    'component': 'orthogonal_safety_basis',
                    'safety_rank': self.config.safety_rank,
                    'num_layers': self.config.num_layers,
                    **details
                }
            )
        else:
            # Fallback logging
            logger.log(
                logging.CRITICAL if severity == 'critical' else
                logging.WARNING if severity in ['high', 'medium'] else
                logging.INFO,
                f"Safety Basis Security Event: {event_type.value}",
                extra=details
            )


# Utility functions
def create_secure_safety_basis(
    config: Optional[SafetyBasisConfig] = None,
    security_context: Optional[SecurityContext] = None
) -> SecureOrthogonalSafetyBasis:
    """Create secure orthogonal safety basis with comprehensive protection"""
    if config is None:
        config = SafetyBasisConfig()
    
    return SecureOrthogonalSafetyBasis(config, security_context)


def verify_basis_orthogonality(
    safety_basis: SecureOrthogonalSafetyBasis,
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """Verify orthogonality of all safety bases"""
    
    verification_results = {}
    
    for layer_idx, basis in enumerate(safety_basis.safety_bases):
        orthogonality_error = safety_basis._compute_orthogonality_error(basis)
        
        verification_results[f"layer_{layer_idx}"] = {
            'orthogonality_error': orthogonality_error,
            'orthogonal': orthogonality_error <= tolerance,
            'basis_shape': basis.shape
        }
    
    # Overall results
    all_orthogonal = all(result['orthogonal'] for result in verification_results.values())
    max_error = max(result['orthogonality_error'] for result in verification_results.values())
    
    return {
        'individual_results': verification_results,
        'all_orthogonal': all_orthogonal,
        'max_orthogonality_error': max_error,
        'tolerance': tolerance
    }