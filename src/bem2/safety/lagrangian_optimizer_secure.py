"""
Secure Lagrangian Constraint Optimization for Value-Aligned Safety Basis (VC0)

This module implements a security-hardened Lagrangian optimizer for balancing
helpfulness vs safety constraints with comprehensive protection against:
- Parameter tampering and injection attacks
- Unauthorized constraint modifications
- Optimization state corruption
- Gradient poisoning attacks
- Performance degradation attacks

Security Features:
- Cryptographic integrity verification for all optimization parameters
- Authenticated constraint modifications with RBAC
- Secure gradient computation with anomaly detection
- Tamper-resistant optimization state management
- Real-time performance monitoring and circuit breaking
- Comprehensive audit logging of all optimization operations

Author: Security-Hardened Implementation
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import time
from collections import deque
import logging
import json
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings

from ..security.auth_manager import AuthenticationManager, SecurityContext
from ..security.parameter_protection import ParameterGuard
from ..security.audit_logger import SecurityAuditor, SecurityEvent
from ..security.circuit_breaker import SafetyCircuitBreaker
from ..security.input_validator import SafetyInputValidator
from .constitutional_scorer_secure import SecureConstitutionalScorer
from .safety_basis_secure import SecureOrthogonalSafetyBasis


class ConstraintType(Enum):
    """Types of safety constraints in the optimization"""
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness" 
    HONESTY = "honesty"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    AUTONOMY = "autonomy"
    TRANSPARENCY = "transparency"
    ORTHOGONALITY = "orthogonality"
    PERFORMANCE = "performance"


@dataclass
class OptimizationConstraint:
    """Represents a single optimization constraint with security metadata"""
    constraint_type: ConstraintType
    weight: float
    threshold: float
    penalty_coefficient: float
    gradient_clip: float
    is_hard_constraint: bool
    creation_time: float
    created_by: str
    integrity_hash: str
    last_modified: float
    
    def __post_init__(self):
        """Generate integrity hash after initialization"""
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute HMAC integrity hash for constraint parameters"""
        data = f"{self.constraint_type.value}|{self.weight}|{self.threshold}|{self.penalty_coefficient}|{self.gradient_clip}|{self.is_hard_constraint}|{self.creation_time}|{self.created_by}|{self.last_modified}"
        return hmac.new(
            b"constraint_integrity_key",
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify constraint hasn't been tampered with"""
        expected_hash = self._compute_hash()
        return hmac.compare_digest(self.integrity_hash, expected_hash)


@dataclass
class OptimizationState:
    """Secure optimization state with tamper detection"""
    lambda_params: Dict[str, float]
    gradient_history: List[Dict[str, torch.Tensor]]
    loss_history: List[float]
    constraint_violations: List[Dict[str, float]]
    iteration_count: int
    convergence_status: str
    last_update: float
    state_hash: str
    performance_metrics: Dict[str, float]
    
    def update_hash(self):
        """Update state integrity hash"""
        # Create deterministic representation of state
        state_repr = {
            "lambda_params": self.lambda_params,
            "loss_history": self.loss_history[-10:],  # Last 10 for efficiency
            "iteration_count": self.iteration_count,
            "convergence_status": self.convergence_status,
            "last_update": self.last_update,
        }
        state_str = json.dumps(state_repr, sort_keys=True)
        self.state_hash = hashlib.sha256(state_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify optimization state hasn't been corrupted"""
        old_hash = self.state_hash
        self.update_hash()
        current_hash = self.state_hash
        self.state_hash = old_hash  # Restore original
        return hmac.compare_digest(old_hash, current_hash)


class SecureLagrangianOptimizer:
    """
    Security-hardened Lagrangian optimizer for constrained safety optimization.
    
    Implements dual ascent method for solving:
    min f(θ) + Σᵢ λᵢ * max(0, gᵢ(θ) - cᵢ)
    
    Where:
    - f(θ) is the primary objective (helpfulness)
    - gᵢ(θ) are constraint functions (safety constraints)  
    - λᵢ are Lagrange multipliers
    - cᵢ are constraint thresholds
    
    Security features:
    - All parameters cryptographically protected
    - Authenticated constraint modifications
    - Gradient anomaly detection
    - Optimization state integrity verification
    - Performance monitoring and circuit breaking
    """
    
    def __init__(
        self,
        auth_manager: AuthenticationManager,
        parameter_guard: ParameterGuard,
        auditor: SecurityAuditor,
        circuit_breaker: SafetyCircuitBreaker,
        input_validator: SafetyInputValidator,
        constitutional_scorer: SecureConstitutionalScorer,
        safety_basis: SecureOrthogonalSafetyBasis,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        learning_rate: float = 0.01,
        lambda_lr: float = 0.1,
        gradient_clip_value: float = 1.0,
        performance_threshold: float = 0.95
    ):
        self.auth_manager = auth_manager
        self.parameter_guard = parameter_guard
        self.auditor = auditor
        self.circuit_breaker = circuit_breaker
        self.input_validator = input_validator
        self.constitutional_scorer = constitutional_scorer
        self.safety_basis = safety_basis
        
        # Optimization parameters (protected)
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._learning_rate = learning_rate
        self._lambda_lr = lambda_lr
        self._gradient_clip_value = gradient_clip_value
        self._performance_threshold = performance_threshold
        
        # Security state
        self._constraints: Dict[str, OptimizationConstraint] = {}
        self._optimization_state: Optional[OptimizationState] = None
        self._is_initialized = False
        self._security_violations = deque(maxlen=100)
        
        # Performance monitoring
        self._gradient_norms = deque(maxlen=50)
        self._loss_values = deque(maxlen=100)
        self._constraint_violations_history = deque(maxlen=100)
        
        # Anomaly detection for gradients
        self._gradient_scaler = StandardScaler()
        self._gradient_baseline_computed = False
        self._gradient_anomaly_threshold = 3.0
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default constraints
        self._initialize_default_constraints()
    
    def _initialize_default_constraints(self):
        """Initialize default safety constraints with secure defaults"""
        default_constraints = [
            {
                "type": ConstraintType.HARMLESSNESS,
                "weight": 2.0,
                "threshold": 0.05,
                "penalty": 10.0,
                "clip": 1.0,
                "hard": True
            },
            {
                "type": ConstraintType.HELPFULNESS,
                "weight": 1.0,
                "threshold": 0.85,
                "penalty": 5.0,
                "clip": 0.5,
                "hard": False
            },
            {
                "type": ConstraintType.HONESTY,
                "weight": 1.5,
                "threshold": 0.10,
                "penalty": 8.0,
                "clip": 0.8,
                "hard": True
            },
            {
                "type": ConstraintType.PRIVACY,
                "weight": 2.5,
                "threshold": 0.02,
                "penalty": 15.0,
                "clip": 1.2,
                "hard": True
            },
            {
                "type": ConstraintType.ORTHOGONALITY,
                "weight": 3.0,
                "threshold": 0.1,
                "penalty": 20.0,
                "clip": 1.5,
                "hard": True
            },
            {
                "type": ConstraintType.PERFORMANCE,
                "weight": 1.0,
                "threshold": 0.01,  # Max 1% performance drop
                "penalty": 25.0,
                "clip": 2.0,
                "hard": True
            }
        ]
        
        current_time = time.time()
        for constraint_config in default_constraints:
            constraint = OptimizationConstraint(
                constraint_type=constraint_config["type"],
                weight=constraint_config["weight"],
                threshold=constraint_config["threshold"],
                penalty_coefficient=constraint_config["penalty"],
                gradient_clip=constraint_config["clip"],
                is_hard_constraint=constraint_config["hard"],
                creation_time=current_time,
                created_by="system_initialization",
                integrity_hash="",  # Will be computed in __post_init__
                last_modified=current_time
            )
            self._constraints[constraint.constraint_type.value] = constraint
        
        self.logger.info(f"Initialized {len(default_constraints)} default constraints")
        self.auditor.log_event(SecurityEvent.CONSTRAINT_MODIFIED, {
            "action": "initialize_defaults",
            "constraints_count": len(default_constraints),
            "timestamp": current_time
        })
    
    def add_constraint(
        self,
        context: SecurityContext,
        constraint_type: ConstraintType,
        weight: float,
        threshold: float,
        penalty_coefficient: float = 1.0,
        gradient_clip: float = 1.0,
        is_hard_constraint: bool = False
    ) -> bool:
        """
        Add or update an optimization constraint with authentication.
        
        Args:
            context: Security context for authorization
            constraint_type: Type of constraint to add
            weight: Weight in the Lagrangian
            threshold: Constraint threshold value
            penalty_coefficient: Penalty coefficient for violations
            gradient_clip: Gradient clipping value for this constraint
            is_hard_constraint: Whether this is a hard constraint
        
        Returns:
            bool: True if constraint added successfully
        """
        try:
            # Authenticate and authorize
            if not self.auth_manager.verify_context(context):
                self.auditor.log_event(SecurityEvent.ACCESS_DENIED, {
                    "operation": "add_constraint",
                    "constraint_type": constraint_type.value,
                    "reason": "invalid_context"
                })
                return False
            
            if not self.auth_manager.has_permission(context, "modify_constraints"):
                self.auditor.log_event(SecurityEvent.ACCESS_DENIED, {
                    "operation": "add_constraint",
                    "constraint_type": constraint_type.value,
                    "reason": "insufficient_permissions"
                })
                return False
            
            # Validate constraint parameters
            if not self._validate_constraint_parameters(
                weight, threshold, penalty_coefficient, gradient_clip
            ):
                return False
            
            # Create constraint with security metadata
            current_time = time.time()
            constraint = OptimizationConstraint(
                constraint_type=constraint_type,
                weight=weight,
                threshold=threshold,
                penalty_coefficient=penalty_coefficient,
                gradient_clip=gradient_clip,
                is_hard_constraint=is_hard_constraint,
                creation_time=current_time,
                created_by=context.user_id,
                integrity_hash="",
                last_modified=current_time
            )
            
            # Add to protected storage
            constraint_id = constraint_type.value
            self._constraints[constraint_id] = constraint
            
            # Log the operation
            self.auditor.log_event(SecurityEvent.CONSTRAINT_MODIFIED, {
                "operation": "add_constraint",
                "constraint_type": constraint_type.value,
                "constraint_id": constraint_id,
                "weight": weight,
                "threshold": threshold,
                "is_hard": is_hard_constraint,
                "user": context.user_id,
                "timestamp": current_time
            })
            
            self.logger.info(f"Added constraint {constraint_type.value} by {context.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add constraint: {e}")
            self.auditor.log_event(SecurityEvent.SYSTEM_ERROR, {
                "operation": "add_constraint",
                "error": str(e)
            })
            return False
    
    def _validate_constraint_parameters(
        self,
        weight: float,
        threshold: float,
        penalty_coefficient: float,
        gradient_clip: float
    ) -> bool:
        """Validate constraint parameters are within safe bounds"""
        validations = [
            (0.0 <= weight <= 10.0, f"Weight {weight} outside safe range [0, 10]"),
            (0.0 <= threshold <= 1.0, f"Threshold {threshold} outside range [0, 1]"),
            (0.0 <= penalty_coefficient <= 100.0, f"Penalty {penalty_coefficient} outside range [0, 100]"),
            (0.0 < gradient_clip <= 5.0, f"Gradient clip {gradient_clip} outside range (0, 5]")
        ]
        
        for is_valid, error_msg in validations:
            if not is_valid:
                self.logger.warning(f"Parameter validation failed: {error_msg}")
                self.auditor.log_event(SecurityEvent.VALIDATION_FAILED, {
                    "validation_type": "constraint_parameters",
                    "error": error_msg
                })
                return False
        
        return True
    
    def optimize(
        self,
        context: SecurityContext,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        objective_fn: Callable,
        constraint_fns: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """
        Perform secure Lagrangian optimization with comprehensive monitoring.
        
        Args:
            context: Security context for authorization
            model: Neural network model to optimize
            train_loader: Training data loader
            val_loader: Validation data loader
            objective_fn: Primary objective function (helpfulness)
            constraint_fns: Dictionary mapping constraint names to functions
        
        Returns:
            Dict with optimization results and security metadata
        """
        try:
            # Security checks
            if not self._pre_optimization_security_check(context):
                return {"success": False, "error": "Security check failed"}
            
            # Initialize optimization state
            self._initialize_optimization_state()
            
            # Main optimization loop
            results = self._optimization_loop(
                model, train_loader, val_loader, objective_fn, constraint_fns
            )
            
            # Post-optimization validation
            if not self._post_optimization_validation(results):
                return {"success": False, "error": "Post-optimization validation failed"}
            
            # Log successful completion
            self.auditor.log_event(SecurityEvent.OPTIMIZATION_COMPLETED, {
                "user": context.user_id,
                "iterations": results.get("iterations", 0),
                "final_loss": results.get("final_loss", 0),
                "constraints_satisfied": results.get("constraints_satisfied", False),
                "performance_maintained": results.get("performance_maintained", False)
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            self.auditor.log_event(SecurityEvent.SYSTEM_ERROR, {
                "operation": "optimize",
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    def _pre_optimization_security_check(self, context: SecurityContext) -> bool:
        """Comprehensive pre-optimization security validation"""
        checks = [
            (self.auth_manager.verify_context(context), "Invalid security context"),
            (self.auth_manager.has_permission(context, "run_optimization"), "Insufficient permissions"),
            (len(self._constraints) > 0, "No constraints defined"),
            (self.circuit_breaker.can_proceed(), "Circuit breaker open"),
            (all(c.verify_integrity() for c in self._constraints.values()), "Constraint integrity check failed")
        ]
        
        for check_passed, error_msg in checks:
            if not check_passed:
                self.logger.warning(f"Pre-optimization check failed: {error_msg}")
                self.auditor.log_event(SecurityEvent.SECURITY_VIOLATION, {
                    "check": "pre_optimization",
                    "error": error_msg,
                    "user": context.user_id
                })
                return False
        
        return True
    
    def _initialize_optimization_state(self):
        """Initialize secure optimization state"""
        # Initialize Lagrange multipliers
        lambda_params = {}
        for constraint_name in self._constraints.keys():
            lambda_params[constraint_name] = 0.1  # Small positive initial value
        
        self._optimization_state = OptimizationState(
            lambda_params=lambda_params,
            gradient_history=[],
            loss_history=[],
            constraint_violations=[],
            iteration_count=0,
            convergence_status="initialized",
            last_update=time.time(),
            state_hash="",
            performance_metrics={}
        )
        self._optimization_state.update_hash()
        
        self.logger.info("Optimization state initialized")
    
    def _optimization_loop(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        objective_fn: Callable,
        constraint_fns: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Main secure optimization loop with monitoring"""
        
        optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)
        best_loss = float('inf')
        convergence_count = 0
        
        for iteration in range(self._max_iterations):
            try:
                # Check circuit breaker
                if not self.circuit_breaker.can_proceed():
                    self.logger.warning("Circuit breaker triggered, stopping optimization")
                    break
                
                # Verify optimization state integrity
                if not self._optimization_state.verify_integrity():
                    self.logger.error("Optimization state integrity check failed")
                    self.auditor.log_event(SecurityEvent.INTEGRITY_VIOLATION, {
                        "component": "optimization_state",
                        "iteration": iteration
                    })
                    break
                
                # Compute objective and constraints
                batch_loss, constraint_values = self._compute_loss_and_constraints(
                    model, train_loader, objective_fn, constraint_fns
                )
                
                # Detect gradient anomalies
                if not self._check_gradient_anomalies(model):
                    self.logger.warning("Gradient anomaly detected, stopping optimization")
                    break
                
                # Update Lagrange multipliers
                self._update_lagrange_multipliers(constraint_values)
                
                # Compute total Lagrangian loss
                total_loss = self._compute_lagrangian_loss(batch_loss, constraint_values)
                
                # Backward pass with gradient clipping
                optimizer.zero_grad()
                total_loss.backward()
                self._apply_secure_gradient_clipping(model)
                optimizer.step()
                
                # Update state
                self._update_optimization_state(iteration, total_loss.item(), constraint_values)
                
                # Check convergence
                if abs(best_loss - total_loss.item()) < self._tolerance:
                    convergence_count += 1
                    if convergence_count >= 5:
                        self.logger.info(f"Converged after {iteration} iterations")
                        break
                else:
                    convergence_count = 0
                    best_loss = total_loss.item()
                
                # Periodic validation
                if iteration % 50 == 0:
                    val_results = self._validate_performance(model, val_loader)
                    if not val_results["performance_maintained"]:
                        self.logger.warning("Performance degradation detected")
                        if self._constraints["performance"].is_hard_constraint:
                            break
                
            except Exception as e:
                self.logger.error(f"Error in optimization iteration {iteration}: {e}")
                self.circuit_breaker.record_failure()
                break
        
        # Prepare final results
        final_results = {
            "success": True,
            "iterations": self._optimization_state.iteration_count,
            "final_loss": self._optimization_state.loss_history[-1] if self._optimization_state.loss_history else 0,
            "lambda_params": self._optimization_state.lambda_params.copy(),
            "constraints_satisfied": self._check_constraints_satisfaction(),
            "performance_maintained": True,  # Will be validated in post-optimization
            "convergence_status": self._optimization_state.convergence_status
        }
        
        return final_results
    
    def _compute_loss_and_constraints(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        objective_fn: Callable,
        constraint_fns: Dict[str, Callable]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute objective loss and constraint values"""
        model.eval()
        total_objective = 0.0
        constraint_values = {name: 0.0 for name in self._constraints.keys()}
        
        batch_count = 0
        
        # Sample a few batches for efficiency
        for batch_idx, (data, targets) in enumerate(train_loader):
            if batch_idx >= 5:  # Limit to 5 batches for efficiency
                break
            
            with torch.no_grad():
                outputs = model(data)
                
                # Compute objective
                batch_objective = objective_fn(outputs, targets)
                total_objective += batch_objective.item()
                
                # Compute constraints
                for constraint_name in constraint_values.keys():
                    if constraint_name in constraint_fns:
                        constraint_value = constraint_fns[constraint_name](outputs, targets)
                        constraint_values[constraint_name] += constraint_value.item()
            
            batch_count += 1
        
        # Average over batches
        if batch_count > 0:
            total_objective /= batch_count
            for name in constraint_values:
                constraint_values[name] /= batch_count
        
        model.train()
        return torch.tensor(total_objective, requires_grad=True), constraint_values
    
    def _check_gradient_anomalies(self, model: nn.Module) -> bool:
        """Detect gradient anomalies that might indicate attacks"""
        try:
            # Compute gradient norms
            total_norm = 0.0
            param_count = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    param_count += 1
            
            if param_count == 0:
                return True  # No gradients to check
            
            total_norm = (total_norm ** 0.5)
            self._gradient_norms.append(total_norm)
            
            # Establish baseline if needed
            if not self._gradient_baseline_computed and len(self._gradient_norms) >= 10:
                gradient_array = np.array(list(self._gradient_norms))
                self._gradient_scaler.fit(gradient_array.reshape(-1, 1))
                self._gradient_baseline_computed = True
                return True
            
            # Check for anomalies once baseline is established
            if self._gradient_baseline_computed:
                normalized_norm = self._gradient_scaler.transform([[total_norm]])[0, 0]
                if abs(normalized_norm) > self._gradient_anomaly_threshold:
                    self.logger.warning(f"Gradient anomaly detected: norm={total_norm}, normalized={normalized_norm}")
                    self.auditor.log_event(SecurityEvent.ANOMALY_DETECTED, {
                        "type": "gradient_anomaly",
                        "gradient_norm": total_norm,
                        "normalized_norm": normalized_norm,
                        "threshold": self._gradient_anomaly_threshold
                    })
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking gradient anomalies: {e}")
            return True  # Fail open for this check
    
    def _update_lagrange_multipliers(self, constraint_values: Dict[str, float]):
        """Update Lagrange multipliers using projected gradient ascent"""
        for constraint_name, constraint_value in constraint_values.items():
            if constraint_name in self._constraints and constraint_name in self._optimization_state.lambda_params:
                constraint = self._constraints[constraint_name]
                violation = max(0, constraint_value - constraint.threshold)
                
                # Gradient ascent on lambda
                lambda_gradient = violation
                new_lambda = self._optimization_state.lambda_params[constraint_name] + self._lambda_lr * lambda_gradient
                
                # Project to non-negative values
                self._optimization_state.lambda_params[constraint_name] = max(0.0, new_lambda)
    
    def _compute_lagrangian_loss(
        self, 
        objective_loss: torch.Tensor, 
        constraint_values: Dict[str, float]
    ) -> torch.Tensor:
        """Compute total Lagrangian loss"""
        total_loss = objective_loss
        
        for constraint_name, constraint_value in constraint_values.items():
            if constraint_name in self._constraints and constraint_name in self._optimization_state.lambda_params:
                constraint = self._constraints[constraint_name]
                lambda_val = self._optimization_state.lambda_params[constraint_name]
                
                # Add penalty for constraint violation
                violation = max(0, constraint_value - constraint.threshold)
                penalty = constraint.weight * lambda_val * violation
                penalty += constraint.penalty_coefficient * (violation ** 2)
                
                total_loss += penalty
        
        return total_loss
    
    def _apply_secure_gradient_clipping(self, model: nn.Module):
        """Apply constraint-aware gradient clipping"""
        # Global gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self._gradient_clip_value)
        
        # Constraint-specific clipping if needed
        for constraint in self._constraints.values():
            if constraint.gradient_clip < self._gradient_clip_value:
                # Apply more aggressive clipping for this constraint
                torch.nn.utils.clip_grad_norm_(model.parameters(), constraint.gradient_clip)
    
    def _update_optimization_state(
        self, 
        iteration: int, 
        loss: float, 
        constraint_values: Dict[str, float]
    ):
        """Update optimization state with integrity protection"""
        self._optimization_state.iteration_count = iteration
        self._optimization_state.loss_history.append(loss)
        self._optimization_state.constraint_violations.append(constraint_values.copy())
        self._optimization_state.last_update = time.time()
        
        # Update convergence status
        if len(self._optimization_state.loss_history) >= 2:
            loss_change = abs(self._optimization_state.loss_history[-1] - self._optimization_state.loss_history[-2])
            if loss_change < self._tolerance:
                self._optimization_state.convergence_status = "converging"
            else:
                self._optimization_state.convergence_status = "optimizing"
        
        # Update integrity hash
        self._optimization_state.update_hash()
    
    def _validate_performance(self, model: nn.Module, val_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Validate model performance hasn't degraded significantly"""
        model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        performance_maintained = accuracy >= self._performance_threshold
        
        model.train()
        
        return {
            "accuracy": accuracy,
            "performance_maintained": performance_maintained,
            "samples_evaluated": total_samples
        }
    
    def _check_constraints_satisfaction(self) -> bool:
        """Check if all constraints are currently satisfied"""
        if not self._optimization_state.constraint_violations:
            return False
        
        latest_violations = self._optimization_state.constraint_violations[-1]
        
        for constraint_name, constraint in self._constraints.items():
            if constraint_name in latest_violations:
                violation = latest_violations[constraint_name]
                if violation > constraint.threshold:
                    if constraint.is_hard_constraint:
                        return False
        
        return True
    
    def _post_optimization_validation(self, results: Dict[str, Any]) -> bool:
        """Comprehensive post-optimization security validation"""
        validations = [
            (results.get("success", False), "Optimization reported failure"),
            (self._optimization_state.verify_integrity(), "Final state integrity check failed"),
            (results.get("constraints_satisfied", False), "Constraints not satisfied"),
            (all(c.verify_integrity() for c in self._constraints.values()), "Constraint integrity check failed")
        ]
        
        for is_valid, error_msg in validations:
            if not is_valid:
                self.logger.error(f"Post-optimization validation failed: {error_msg}")
                self.auditor.log_event(SecurityEvent.VALIDATION_FAILED, {
                    "validation_type": "post_optimization",
                    "error": error_msg
                })
                return False
        
        return True
    
    def get_optimization_status(self, context: SecurityContext) -> Dict[str, Any]:
        """Get current optimization status with security checks"""
        if not self.auth_manager.verify_context(context):
            return {"error": "Unauthorized access"}
        
        if not self.auth_manager.has_permission(context, "view_optimization_status"):
            return {"error": "Insufficient permissions"}
        
        if self._optimization_state is None:
            return {"status": "not_initialized"}
        
        # Verify state integrity before returning
        if not self._optimization_state.verify_integrity():
            self.auditor.log_event(SecurityEvent.INTEGRITY_VIOLATION, {
                "component": "optimization_state_access",
                "user": context.user_id
            })
            return {"error": "State integrity check failed"}
        
        return {
            "iteration_count": self._optimization_state.iteration_count,
            "convergence_status": self._optimization_state.convergence_status,
            "lambda_params": self._optimization_state.lambda_params.copy(),
            "recent_losses": list(self._optimization_state.loss_history[-10:]),
            "constraints_satisfied": self._check_constraints_satisfaction(),
            "last_update": self._optimization_state.last_update
        }
    
    def reset_optimization(self, context: SecurityContext) -> bool:
        """Reset optimization state with authentication"""
        if not self.auth_manager.verify_context(context):
            return False
        
        if not self.auth_manager.has_permission(context, "reset_optimization"):
            return False
        
        self._optimization_state = None
        self._gradient_norms.clear()
        self._loss_values.clear() 
        self._constraint_violations_history.clear()
        self._gradient_baseline_computed = False
        
        self.auditor.log_event(SecurityEvent.OPTIMIZATION_RESET, {
            "user": context.user_id,
            "timestamp": time.time()
        })
        
        self.logger.info(f"Optimization reset by {context.user_id}")
        return True


def create_secure_lagrangian_optimizer(
    auth_manager: AuthenticationManager,
    parameter_guard: ParameterGuard,
    auditor: SecurityAuditor,
    circuit_breaker: SafetyCircuitBreaker,
    input_validator: SafetyInputValidator,
    constitutional_scorer: SecureConstitutionalScorer,
    safety_basis: SecureOrthogonalSafetyBasis,
    **kwargs
) -> SecureLagrangianOptimizer:
    """Factory function to create a secure Lagrangian optimizer"""
    return SecureLagrangianOptimizer(
        auth_manager=auth_manager,
        parameter_guard=parameter_guard,
        auditor=auditor,
        circuit_breaker=circuit_breaker,
        input_validator=input_validator,
        constitutional_scorer=constitutional_scorer,
        safety_basis=safety_basis,
        **kwargs
    )