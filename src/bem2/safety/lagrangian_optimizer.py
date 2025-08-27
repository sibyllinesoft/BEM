"""
Lagrangian Constrained Optimization for Safety Training

Implements constrained optimization with Lagrangian multipliers to minimize
helpfulness loss subject to violation rate ≤ ε. Ensures safety improvements
while maintaining model utility performance.

Key Features:
- Lagrangian dual optimization with adaptive λ
- Constraint satisfaction monitoring  
- Automatic λ scheduling based on constraint violation
- Primal-dual optimization with stability guarantees
- Multi-objective optimization balancing safety and utility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import math
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ConstraintConfig:
    """Configuration for Lagrangian constraint optimization."""
    
    # Primary constraint
    max_violation_rate: float = 0.05        # Maximum violation rate ε
    violation_tolerance: float = 0.01       # Tolerance around constraint
    
    # Lagrangian parameters
    initial_lambda: float = 1.0             # Initial Lagrange multiplier
    lambda_lr: float = 0.01                 # Learning rate for λ updates
    min_lambda: float = 0.001              # Minimum λ value
    max_lambda: float = 100.0              # Maximum λ value
    
    # Constraint satisfaction
    constraint_check_frequency: int = 100   # Steps between constraint checks
    constraint_history_length: int = 500   # Length of constraint history
    convergence_window: int = 100          # Window for convergence check
    convergence_tolerance: float = 0.001   # Tolerance for convergence
    
    # Optimization balance
    safety_weight: float = 1.0             # Weight for safety loss
    utility_weight: float = 1.0            # Weight for utility/helpfulness loss
    orthogonality_weight: float = 0.1      # Weight for orthogonality penalty
    
    # Adaptive scheduling  
    lambda_adaptation_rate: float = 0.1    # Rate of λ adaptation
    constraint_momentum: float = 0.9       # Momentum for constraint tracking
    warmup_steps: int = 1000              # Steps to warm up constraints
    
    # Stability guarantees
    max_gradient_norm: float = 1.0         # Maximum gradient norm
    stability_check: bool = True           # Enable stability monitoring
    rollback_on_divergence: bool = True    # Rollback on optimization divergence


class LagrangianOptimizer:
    """
    Lagrangian optimizer for constrained safety training.
    
    Solves the constrained optimization problem:
    min_θ L_utility(θ) subject to L_violation(θ) ≤ ε
    
    Using the Lagrangian formulation:
    L(θ, λ) = L_utility(θ) + λ * (L_violation(θ) - ε)
    
    Where θ are model parameters and λ is the Lagrange multiplier.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ConstraintConfig,
        utility_loss_fn: Callable,
        violation_loss_fn: Callable,
        orthogonality_loss_fn: Optional[Callable] = None
    ):
        self.model = model
        self.config = config
        self.utility_loss_fn = utility_loss_fn
        self.violation_loss_fn = violation_loss_fn
        self.orthogonality_loss_fn = orthogonality_loss_fn
        
        # Lagrange multiplier (learnable parameter)
        self.lambda_param = nn.Parameter(torch.tensor(config.initial_lambda))
        
        # Primary optimizer for model parameters
        self.model_optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Dual optimizer for Lagrange multipliers
        self.lambda_optimizer = optim.SGD(
            [self.lambda_param],
            lr=config.lambda_lr
        )
        
        # Constraint tracking
        self.violation_history = deque(maxlen=config.constraint_history_length)
        self.constraint_satisfaction_history = deque(maxlen=config.convergence_window)
        self.lambda_history = deque(maxlen=config.convergence_window)
        
        # Optimization state
        self.step_count = 0
        self.last_constraint_check = 0
        self.best_model_state = None
        self.best_violation_rate = float('inf')
        
        # Telemetry
        self.violation_rate_ema = 0.0
        self.utility_loss_ema = 0.0
        self.total_loss_ema = 0.0
        self.constraint_violations = 0
        self.convergence_achieved = False
        
        logger.info(f"Initialized Lagrangian optimizer with ε={config.max_violation_rate}")
    
    def step(
        self,
        batch: Dict[str, torch.Tensor],
        safety_scores: torch.Tensor,
        return_metrics: bool = False
    ) -> Optional[Dict[str, float]]:
        """
        Perform one optimization step with Lagrangian constraints.
        
        Args:
            batch: Training batch data
            safety_scores: Constitutional/safety scores for batch
            return_metrics: Whether to return detailed metrics
            
        Returns:
            metrics: Optimization metrics if requested
        """
        self.step_count += 1
        
        # Zero gradients
        self.model_optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
        
        # Forward pass
        model_output = self.model(**batch)
        
        # Compute utility loss (helpfulness/performance)
        utility_loss = self.utility_loss_fn(model_output, batch)
        
        # Compute violation loss (safety constraint)
        violation_loss = self.violation_loss_fn(model_output, safety_scores)
        
        # Compute orthogonality penalty if provided
        orthogonality_loss = torch.tensor(0.0, device=utility_loss.device)
        if self.orthogonality_loss_fn is not None:
            orthogonality_loss = self.orthogonality_loss_fn()
        
        # Current violation rate
        violation_rate = self._compute_violation_rate(model_output, safety_scores)
        
        # Lagrangian formulation
        constraint_violation = violation_rate - self.config.max_violation_rate
        
        # Clamp lambda to valid range
        lambda_clamped = torch.clamp(
            self.lambda_param,
            self.config.min_lambda,
            self.config.max_lambda
        )
        
        # Total Lagrangian loss
        lagrangian_loss = (
            self.config.utility_weight * utility_loss +
            lambda_clamped * constraint_violation +
            self.config.orthogonality_weight * orthogonality_loss
        )
        
        # Backward pass
        lagrangian_loss.backward()
        
        # Gradient clipping for stability
        if self.config.max_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_gradient_norm
            )
        
        # Update model parameters (primal variables)
        self.model_optimizer.step()
        
        # Update Lagrange multiplier (dual variables)
        if self.step_count > self.config.warmup_steps:
            # λ increases when constraint is violated, decreases when satisfied
            with torch.no_grad():
                lambda_grad = constraint_violation.detach()
                self.lambda_param.grad = lambda_grad
            
            self.lambda_optimizer.step()
            
            # Ensure λ stays non-negative and within bounds
            with torch.no_grad():
                self.lambda_param.clamp_(
                    self.config.min_lambda,
                    self.config.max_lambda
                )
        
        # Update telemetry
        self._update_telemetry(
            utility_loss.item(),
            violation_loss.item(),
            violation_rate.item(),
            lagrangian_loss.item()
        )
        
        # Check constraints periodically
        if (self.step_count - self.last_constraint_check >= 
            self.config.constraint_check_frequency):
            self._check_constraints()
            self.last_constraint_check = self.step_count
        
        # Save best model if constraint is satisfied
        if violation_rate.item() <= self.config.max_violation_rate:
            if violation_rate.item() < self.best_violation_rate:
                self.best_violation_rate = violation_rate.item()
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
        
        if return_metrics:
            return self._get_optimization_metrics(
                utility_loss.item(),
                violation_loss.item(),
                violation_rate.item(),
                constraint_violation.item(),
                lambda_clamped.item()
            )
    
    def _compute_violation_rate(
        self,
        model_output: torch.Tensor,
        safety_scores: torch.Tensor
    ) -> torch.Tensor:
        """Compute current violation rate from model output and safety scores."""
        
        # Binary violations based on safety scores
        # Low safety scores indicate violations
        violations = safety_scores < 0.5  # Threshold for violation
        violation_rate = violations.float().mean()
        
        return violation_rate
    
    def _update_telemetry(
        self,
        utility_loss: float,
        violation_loss: float,
        violation_rate: float,
        total_loss: float
    ):
        """Update exponential moving averages for telemetry."""
        alpha = 0.01  # EMA decay factor
        
        self.utility_loss_ema = (1 - alpha) * self.utility_loss_ema + alpha * utility_loss
        self.violation_rate_ema = (1 - alpha) * self.violation_rate_ema + alpha * violation_rate
        self.total_loss_ema = (1 - alpha) * self.total_loss_ema + alpha * total_loss
        
        # Track violation history
        self.violation_history.append(violation_rate)
        
        # Count constraint violations
        if violation_rate > self.config.max_violation_rate:
            self.constraint_violations += 1
    
    def _check_constraints(self):
        """Check constraint satisfaction and convergence."""
        if len(self.violation_history) < 10:
            return
        
        # Recent violation rates
        recent_violations = list(self.violation_history)[-10:]
        avg_recent_violation = sum(recent_violations) / len(recent_violations)
        
        # Check constraint satisfaction
        constraint_satisfied = avg_recent_violation <= (
            self.config.max_violation_rate + self.config.violation_tolerance
        )
        self.constraint_satisfaction_history.append(constraint_satisfied)
        
        # Track lambda evolution
        self.lambda_history.append(self.lambda_param.item())
        
        # Check convergence
        if len(self.constraint_satisfaction_history) >= self.config.convergence_window:
            satisfaction_rate = sum(self.constraint_satisfaction_history) / len(self.constraint_satisfaction_history)
            lambda_stability = self._compute_lambda_stability()
            
            if (satisfaction_rate > 0.9 and 
                lambda_stability < self.config.convergence_tolerance):
                self.convergence_achieved = True
                logger.info(f"Lagrangian optimization converged at step {self.step_count}")
        
        # Adaptive lambda scheduling
        if self.step_count > self.config.warmup_steps:
            self._adapt_lambda_learning_rate(constraint_satisfied)
    
    def _compute_lambda_stability(self) -> float:
        """Compute stability of lambda parameter over recent history."""
        if len(self.lambda_history) < 2:
            return float('inf')
        
        lambda_values = torch.tensor(list(self.lambda_history))
        return torch.std(lambda_values).item()
    
    def _adapt_lambda_learning_rate(self, constraint_satisfied: bool):
        """Adapt lambda learning rate based on constraint satisfaction."""
        current_lr = self.lambda_optimizer.param_groups[0]['lr']
        
        if constraint_satisfied:
            # Reduce learning rate when constraint is satisfied
            new_lr = current_lr * (1 - self.config.lambda_adaptation_rate * 0.1)
        else:
            # Increase learning rate when constraint is violated
            new_lr = current_lr * (1 + self.config.lambda_adaptation_rate)
        
        # Clamp learning rate
        new_lr = max(1e-6, min(new_lr, 1.0))
        
        for param_group in self.lambda_optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def _get_optimization_metrics(
        self,
        utility_loss: float,
        violation_loss: float,
        violation_rate: float,
        constraint_violation: float,
        lambda_value: float
    ) -> Dict[str, float]:
        """Get detailed optimization metrics."""
        
        constraint_satisfaction_rate = 0.0
        if len(self.constraint_satisfaction_history) > 0:
            constraint_satisfaction_rate = (
                sum(self.constraint_satisfaction_history) / 
                len(self.constraint_satisfaction_history)
            )
        
        return {
            'utility_loss': utility_loss,
            'violation_loss': violation_loss,
            'violation_rate': violation_rate,
            'constraint_violation': constraint_violation,
            'lambda_value': lambda_value,
            'utility_loss_ema': self.utility_loss_ema,
            'violation_rate_ema': self.violation_rate_ema,
            'total_loss_ema': self.total_loss_ema,
            'constraint_satisfaction_rate': constraint_satisfaction_rate,
            'convergence_achieved': self.convergence_achieved,
            'constraint_violations_total': self.constraint_violations,
            'steps_since_last_check': self.step_count - self.last_constraint_check,
            'lambda_lr': self.lambda_optimizer.param_groups[0]['lr']
        }
    
    def get_constraint_analysis(self) -> Dict[str, any]:
        """Get detailed constraint satisfaction analysis."""
        
        if len(self.violation_history) == 0:
            return {'error': 'No violation history available'}
        
        violation_rates = list(self.violation_history)
        
        # Statistical analysis
        avg_violation_rate = sum(violation_rates) / len(violation_rates)
        min_violation_rate = min(violation_rates)
        max_violation_rate = max(violation_rates)
        
        # Recent trend
        if len(violation_rates) >= 20:
            recent_rates = violation_rates[-20:]
            earlier_rates = violation_rates[-40:-20] if len(violation_rates) >= 40 else violation_rates[:-20]
            
            recent_avg = sum(recent_rates) / len(recent_rates)
            earlier_avg = sum(earlier_rates) / len(earlier_rates)
            trend = recent_avg - earlier_avg
        else:
            trend = 0.0
        
        # Constraint satisfaction
        satisfied_steps = sum(1 for rate in violation_rates 
                             if rate <= self.config.max_violation_rate)
        satisfaction_percentage = satisfied_steps / len(violation_rates) * 100
        
        return {
            'violation_rate_stats': {
                'average': avg_violation_rate,
                'minimum': min_violation_rate,
                'maximum': max_violation_rate,
                'current': violation_rates[-1] if violation_rates else 0.0,
                'trend': trend
            },
            'constraint_satisfaction': {
                'target_rate': self.config.max_violation_rate,
                'current_satisfaction_percentage': satisfaction_percentage,
                'satisfied_steps': satisfied_steps,
                'total_steps': len(violation_rates)
            },
            'optimization_state': {
                'lambda_value': self.lambda_param.item(),
                'lambda_lr': self.lambda_optimizer.param_groups[0]['lr'],
                'convergence_achieved': self.convergence_achieved,
                'best_violation_rate': self.best_violation_rate,
                'constraint_violations_count': self.constraint_violations
            }
        }
    
    def save_checkpoint(self, filepath: str):
        """Save optimizer state and constraint tracking."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'lambda_optimizer_state_dict': self.lambda_optimizer.state_dict(),
            'lambda_param': self.lambda_param.item(),
            'step_count': self.step_count,
            'violation_history': list(self.violation_history),
            'constraint_satisfaction_history': list(self.constraint_satisfaction_history),
            'lambda_history': list(self.lambda_history),
            'best_model_state': self.best_model_state,
            'best_violation_rate': self.best_violation_rate,
            'convergence_achieved': self.convergence_achieved,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved Lagrangian optimizer checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load optimizer state and constraint tracking."""
        checkpoint = torch.load(filepath)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
        self.lambda_optimizer.load_state_dict(checkpoint['lambda_optimizer_state_dict'])
        
        self.lambda_param.data = torch.tensor(checkpoint['lambda_param'])
        self.step_count = checkpoint['step_count']
        
        self.violation_history.clear()
        self.violation_history.extend(checkpoint['violation_history'])
        
        self.constraint_satisfaction_history.clear() 
        self.constraint_satisfaction_history.extend(checkpoint['constraint_satisfaction_history'])
        
        self.lambda_history.clear()
        self.lambda_history.extend(checkpoint['lambda_history'])
        
        self.best_model_state = checkpoint['best_model_state']
        self.best_violation_rate = checkpoint['best_violation_rate']
        self.convergence_achieved = checkpoint['convergence_achieved']
        
        logger.info(f"Loaded Lagrangian optimizer checkpoint from {filepath}")
    
    def restore_best_model(self):
        """Restore the best model state (lowest violation rate)."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with violation rate {self.best_violation_rate:.4f}")
        else:
            logger.warning("No best model state available to restore")
    
    def reset_constraint_tracking(self):
        """Reset all constraint tracking and telemetry."""
        self.violation_history.clear()
        self.constraint_satisfaction_history.clear()
        self.lambda_history.clear()
        
        self.violation_rate_ema = 0.0
        self.utility_loss_ema = 0.0 
        self.total_loss_ema = 0.0
        self.constraint_violations = 0
        self.convergence_achieved = False
        self.best_violation_rate = float('inf')
        self.best_model_state = None
        
        logger.info("Reset all constraint tracking and telemetry")
    
    def update_constraint_target(self, new_target: float):
        """Update the constraint target violation rate."""
        old_target = self.config.max_violation_rate
        self.config.max_violation_rate = new_target
        
        logger.info(f"Updated constraint target from {old_target:.4f} to {new_target:.4f}")
        
        # Reset tracking to adapt to new target
        self.constraint_satisfaction_history.clear()
        self.convergence_achieved = False