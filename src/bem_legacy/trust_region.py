"""
BEM Phase 4: Trust Region Projection Module

This module implements global budget constraints for multi-BEM composition.
It ensures that the combined ΔW matrices from multiple BEMs stay within
spectral and Frobenius norm budgets using trust region projection.

Key Features:
- Norm Calculator: Combined Frobenius norm across all BEMs
- Spectral Clamps: Global σ₁ limits on composed ΔW
- Projection Operator: Scale down ΔW_sum when budget exceeded
- Numerical Stability: Use fp32 accumulators for norm calculations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrustRegionBudget:
    """
    Defines the budget constraints for trust region projection.
    """
    spectral_budget: float  # Maximum σ₁ (largest singular value)
    frobenius_budget: float  # Maximum Frobenius norm
    layer_name: str
    
    def __post_init__(self):
        assert self.spectral_budget > 0
        assert self.frobenius_budget > 0


class BudgetViolation(NamedTuple):
    """Information about budget violations."""
    layer_name: str
    violation_type: str  # 'spectral' or 'frobenius'
    actual_value: float
    budget_value: float
    violation_ratio: float


class TrustRegionProjectionResult(NamedTuple):
    """Result of trust region projection."""
    projected_deltas: Dict[str, torch.Tensor]
    violations: List[BudgetViolation]
    scaling_factors: Dict[str, float]
    pre_projection_norms: Dict[str, Dict[str, float]]
    post_projection_norms: Dict[str, Dict[str, float]]


class NormCalculator:
    """
    Calculates various norms of ΔW matrices with numerical stability.
    
    Uses fp32 accumulators for critical calculations to maintain precision.
    """
    
    @staticmethod
    def frobenius_norm(delta_w: torch.Tensor, use_fp32: bool = True) -> torch.Tensor:
        """
        Compute Frobenius norm with optional fp32 precision.
        
        Args:
            delta_w: Matrix to compute norm for
            use_fp32: Whether to use fp32 for accumulation
            
        Returns:
            Frobenius norm as scalar tensor
        """
        if use_fp32 and delta_w.dtype != torch.float32:
            delta_w_fp32 = delta_w.float()
            norm = torch.norm(delta_w_fp32, p='fro')
            return norm.to(delta_w.dtype)
        else:
            return torch.norm(delta_w, p='fro')
    
    @staticmethod
    def spectral_norm(delta_w: torch.Tensor, use_fp32: bool = True) -> torch.Tensor:
        """
        Compute spectral norm (largest singular value) with optional fp32 precision.
        
        Args:
            delta_w: Matrix to compute norm for
            use_fp32: Whether to use fp32 for SVD computation
            
        Returns:
            Spectral norm as scalar tensor
        """
        if use_fp32 and delta_w.dtype != torch.float32:
            delta_w_fp32 = delta_w.float()
            # Use SVD to get largest singular value
            try:
                _, s, _ = torch.linalg.svd(delta_w_fp32, full_matrices=False)
                spectral_norm = s[0] if len(s) > 0 else torch.tensor(0.0)
                return spectral_norm.to(delta_w.dtype)
            except Exception as e:
                logger.warning(f"SVD failed, falling back to operator norm: {e}")
                # Fallback to matrix 2-norm
                norm = torch.linalg.matrix_norm(delta_w_fp32, ord=2)
                return norm.to(delta_w.dtype)
        else:
            try:
                _, s, _ = torch.linalg.svd(delta_w, full_matrices=False)
                return s[0] if len(s) > 0 else torch.tensor(0.0, device=delta_w.device, dtype=delta_w.dtype)
            except Exception as e:
                logger.warning(f"SVD failed, falling back to operator norm: {e}")
                return torch.linalg.matrix_norm(delta_w, ord=2)
    
    @staticmethod
    def combined_frobenius_norm(delta_ws: List[torch.Tensor], use_fp32: bool = True) -> torch.Tensor:
        """
        Compute combined Frobenius norm of multiple matrices.
        
        Args:
            delta_ws: List of matrices
            use_fp32: Whether to use fp32 for accumulation
            
        Returns:
            Combined Frobenius norm
        """
        if not delta_ws:
            return torch.tensor(0.0)
        
        if use_fp32:
            # Accumulate squared norms in fp32
            total_squared_norm = torch.tensor(0.0, dtype=torch.float32, device=delta_ws[0].device)
            for delta_w in delta_ws:
                squared_norm = torch.sum(delta_w.float() ** 2)
                total_squared_norm = total_squared_norm + squared_norm
            
            combined_norm = torch.sqrt(total_squared_norm)
            return combined_norm.to(delta_ws[0].dtype)
        else:
            # Direct computation
            total_squared_norm = sum(torch.sum(dw ** 2) for dw in delta_ws)
            return torch.sqrt(total_squared_norm)
    
    @staticmethod
    def compute_all_norms(delta_w: torch.Tensor, use_fp32: bool = True) -> Dict[str, float]:
        """
        Compute all norm types for a matrix.
        
        Args:
            delta_w: Matrix to analyze
            use_fp32: Whether to use fp32 precision
            
        Returns:
            Dictionary with all norm values
        """
        return {
            'frobenius': NormCalculator.frobenius_norm(delta_w, use_fp32).item(),
            'spectral': NormCalculator.spectral_norm(delta_w, use_fp32).item(),
            'nuclear': torch.linalg.matrix_norm(delta_w.float() if use_fp32 else delta_w, ord='nuc').item(),
            'max': torch.max(torch.abs(delta_w)).item(),
            'mean': torch.mean(torch.abs(delta_w)).item()
        }


class SpectralClamp:
    """
    Applies spectral clamping to limit the largest singular value.
    """
    
    def __init__(self, max_spectral_norm: float, use_fp32: bool = True):
        """
        Initialize spectral clamp.
        
        Args:
            max_spectral_norm: Maximum allowed spectral norm
            use_fp32: Whether to use fp32 for SVD computations
        """
        self.max_spectral_norm = max_spectral_norm
        self.use_fp32 = use_fp32
        
    def apply(self, delta_w: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Apply spectral clamping to a matrix.
        
        Args:
            delta_w: Matrix to clamp
            
        Returns:
            Tuple of (clamped_matrix, scaling_factor)
        """
        current_spectral_norm = NormCalculator.spectral_norm(delta_w, self.use_fp32)
        
        if current_spectral_norm <= self.max_spectral_norm:
            return delta_w, 1.0
        
        # Compute scaling factor
        scaling_factor = self.max_spectral_norm / current_spectral_norm.item()
        
        # Apply scaling
        clamped_delta_w = delta_w * scaling_factor
        
        logger.debug(f"Applied spectral clamp: {current_spectral_norm:.6f} -> {self.max_spectral_norm}, "
                    f"scaling_factor={scaling_factor:.6f}")
        
        return clamped_delta_w, scaling_factor


class TrustRegionProjector:
    """
    Main trust region projection engine for multi-BEM composition.
    
    This class implements the core trust region projection algorithm:
    ΔW_sum ← ΔW_sum * min(1, τ / ||ΔW_sum||_F)
    """
    
    def __init__(self, budgets: Dict[str, TrustRegionBudget], use_fp32: bool = True):
        """
        Initialize the trust region projector.
        
        Args:
            budgets: Dictionary mapping layer names to budget constraints
            use_fp32: Whether to use fp32 for numerical stability
        """
        self.budgets = budgets
        self.use_fp32 = use_fp32
        self.norm_calculator = NormCalculator()
        
        # Initialize spectral clamps
        self.spectral_clamps = {
            layer_name: SpectralClamp(budget.spectral_budget, use_fp32)
            for layer_name, budget in budgets.items()
        }
        
        logger.info(f"Initialized TrustRegionProjector with {len(budgets)} layer budgets")
    
    def project_multi_bem_deltas(
        self,
        bem_deltas: Dict[str, Dict[str, torch.Tensor]]
    ) -> TrustRegionProjectionResult:
        """
        Project multiple BEM deltas to respect global trust region budgets.
        
        Args:
            bem_deltas: Nested dict {bem_id: {layer_name: delta_w}}
            
        Returns:
            TrustRegionProjectionResult with projected deltas and diagnostic info
        """
        # First, compute combined deltas per layer
        combined_deltas = self._compute_combined_deltas(bem_deltas)
        
        # Compute pre-projection norms
        pre_projection_norms = {
            layer_name: self.norm_calculator.compute_all_norms(delta_w, self.use_fp32)
            for layer_name, delta_w in combined_deltas.items()
        }
        
        # Apply projections
        projected_deltas = {}
        violations = []
        scaling_factors = {}
        
        for layer_name, combined_delta in combined_deltas.items():
            if layer_name not in self.budgets:
                # No budget constraint for this layer
                projected_deltas[layer_name] = combined_delta
                scaling_factors[layer_name] = 1.0
                continue
            
            budget = self.budgets[layer_name]
            
            # Apply spectral clamp first
            spectrally_clamped, spectral_scaling = self.spectral_clamps[layer_name].apply(combined_delta)
            
            # Then apply Frobenius norm projection
            frobenius_norm = self.norm_calculator.frobenius_norm(spectrally_clamped, self.use_fp32)
            
            if frobenius_norm <= budget.frobenius_budget:
                # Within budget
                projected_deltas[layer_name] = spectrally_clamped
                scaling_factors[layer_name] = spectral_scaling
            else:
                # Apply trust region projection
                frobenius_scaling = budget.frobenius_budget / frobenius_norm.item()
                final_scaling = spectral_scaling * frobenius_scaling
                
                projected_deltas[layer_name] = combined_delta * final_scaling
                scaling_factors[layer_name] = final_scaling
                
                # Record violation
                violations.append(BudgetViolation(
                    layer_name=layer_name,
                    violation_type='frobenius',
                    actual_value=frobenius_norm.item(),
                    budget_value=budget.frobenius_budget,
                    violation_ratio=frobenius_norm.item() / budget.frobenius_budget
                ))
                
                logger.debug(f"Applied trust region projection to {layer_name}: "
                           f"norm {frobenius_norm:.6f} -> {budget.frobenius_budget}, "
                           f"final_scaling={final_scaling:.6f}")
            
            # Check for spectral violations (already clamped, but record for diagnostics)
            if spectral_scaling < 1.0:
                original_spectral = self.norm_calculator.spectral_norm(combined_delta, self.use_fp32)
                violations.append(BudgetViolation(
                    layer_name=layer_name,
                    violation_type='spectral',
                    actual_value=original_spectral.item(),
                    budget_value=budget.spectral_budget,
                    violation_ratio=original_spectral.item() / budget.spectral_budget
                ))
        
        # Compute post-projection norms
        post_projection_norms = {
            layer_name: self.norm_calculator.compute_all_norms(delta_w, self.use_fp32)
            for layer_name, delta_w in projected_deltas.items()
        }
        
        if violations:
            logger.info(f"Trust region projection applied: {len(violations)} violations detected and corrected")
        else:
            logger.debug("No budget violations detected")
        
        return TrustRegionProjectionResult(
            projected_deltas=projected_deltas,
            violations=violations,
            scaling_factors=scaling_factors,
            pre_projection_norms=pre_projection_norms,
            post_projection_norms=post_projection_norms
        )
    
    def _compute_combined_deltas(
        self,
        bem_deltas: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine deltas from multiple BEMs per layer.
        
        Args:
            bem_deltas: Nested dict {bem_id: {layer_name: delta_w}}
            
        Returns:
            Dictionary {layer_name: combined_delta_w}
        """
        combined_deltas = {}
        
        # Get all layer names
        all_layer_names = set()
        for bem_id, layer_deltas in bem_deltas.items():
            all_layer_names.update(layer_deltas.keys())
        
        # Combine deltas for each layer
        for layer_name in all_layer_names:
            layer_deltas = []
            
            for bem_id, bem_layer_deltas in bem_deltas.items():
                if layer_name in bem_layer_deltas:
                    layer_deltas.append(bem_layer_deltas[layer_name])
            
            if layer_deltas:
                # Sum all deltas for this layer
                if self.use_fp32:
                    # Use fp32 accumulation for numerical stability
                    combined = torch.zeros_like(layer_deltas[0], dtype=torch.float32)
                    for delta in layer_deltas:
                        combined = combined + delta.float()
                    combined_deltas[layer_name] = combined.to(layer_deltas[0].dtype)
                else:
                    combined_deltas[layer_name] = sum(layer_deltas)
        
        return combined_deltas
    
    def update_budget(self, layer_name: str, budget: TrustRegionBudget):
        """
        Update budget for a specific layer.
        
        Args:
            layer_name: Name of the layer
            budget: New budget constraints
        """
        self.budgets[layer_name] = budget
        self.spectral_clamps[layer_name] = SpectralClamp(budget.spectral_budget, self.use_fp32)
        
        logger.info(f"Updated budget for {layer_name}: spectral={budget.spectral_budget}, "
                   f"frobenius={budget.frobenius_budget}")
    
    def get_budget_info(self) -> Dict[str, Dict[str, float]]:
        """Get information about all budget constraints."""
        return {
            layer_name: {
                'spectral_budget': budget.spectral_budget,
                'frobenius_budget': budget.frobenius_budget
            }
            for layer_name, budget in self.budgets.items()
        }


class AdaptiveTrustRegion:
    """
    Adaptive trust region that adjusts budgets based on training dynamics.
    """
    
    def __init__(
        self,
        initial_projector: TrustRegionProjector,
        adaptation_rate: float = 0.1,
        violation_threshold: float = 0.1
    ):
        """
        Initialize adaptive trust region.
        
        Args:
            initial_projector: Initial TrustRegionProjector
            adaptation_rate: Rate of budget adaptation
            violation_threshold: Threshold for triggering adaptations
        """
        self.projector = initial_projector
        self.adaptation_rate = adaptation_rate
        self.violation_threshold = violation_threshold
        self.violation_history: List[List[BudgetViolation]] = []
        
    def project_and_adapt(
        self,
        bem_deltas: Dict[str, Dict[str, torch.Tensor]]
    ) -> TrustRegionProjectionResult:
        """
        Project deltas and adapt budgets based on violation patterns.
        
        Args:
            bem_deltas: BEM deltas to project
            
        Returns:
            Projection result with potentially updated budgets
        """
        # Apply projection
        result = self.projector.project_multi_bem_deltas(bem_deltas)
        
        # Record violations
        self.violation_history.append(result.violations)
        
        # Keep only recent history
        if len(self.violation_history) > 100:
            self.violation_history = self.violation_history[-100:]
        
        # Check if adaptation is needed
        if len(self.violation_history) >= 10:
            self._adapt_budgets()
        
        return result
    
    def _adapt_budgets(self):
        """Adapt budgets based on violation history."""
        recent_violations = self.violation_history[-10:]
        
        # Count violations per layer
        layer_violation_counts = {}
        layer_violation_ratios = {}
        
        for violations in recent_violations:
            for violation in violations:
                layer_name = violation.layer_name
                
                if layer_name not in layer_violation_counts:
                    layer_violation_counts[layer_name] = 0
                    layer_violation_ratios[layer_name] = []
                
                layer_violation_counts[layer_name] += 1
                layer_violation_ratios[layer_name].append(violation.violation_ratio)
        
        # Adapt budgets for layers with frequent violations
        for layer_name, violation_count in layer_violation_counts.items():
            violation_rate = violation_count / len(recent_violations)
            
            if violation_rate > self.violation_threshold:
                # Increase budget for this layer
                current_budget = self.projector.budgets[layer_name]
                mean_violation_ratio = np.mean(layer_violation_ratios[layer_name])
                
                adaptation_factor = 1.0 + self.adaptation_rate * (mean_violation_ratio - 1.0)
                
                new_budget = TrustRegionBudget(
                    spectral_budget=current_budget.spectral_budget * adaptation_factor,
                    frobenius_budget=current_budget.frobenius_budget * adaptation_factor,
                    layer_name=layer_name
                )
                
                self.projector.update_budget(layer_name, new_budget)
                
                logger.info(f"Adapted budget for {layer_name}: violation_rate={violation_rate:.3f}, "
                           f"adaptation_factor={adaptation_factor:.3f}")


def create_trust_region_projector(
    layer_names: List[str],
    spectral_budget: float = 1.0,
    frobenius_budget: float = 5.0,
    use_fp32: bool = True
) -> TrustRegionProjector:
    """
    Factory function to create a TrustRegionProjector.
    
    Args:
        layer_names: List of layer names to apply budgets to
        spectral_budget: Default spectral budget
        frobenius_budget: Default Frobenius budget
        use_fp32: Whether to use fp32 precision
        
    Returns:
        Configured TrustRegionProjector
    """
    budgets = {
        layer_name: TrustRegionBudget(
            spectral_budget=spectral_budget,
            frobenius_budget=frobenius_budget,
            layer_name=layer_name
        )
        for layer_name in layer_names
    }
    
    return TrustRegionProjector(budgets, use_fp32)


def create_adaptive_trust_region(
    projector: TrustRegionProjector,
    adaptation_rate: float = 0.1,
    violation_threshold: float = 0.1
) -> AdaptiveTrustRegion:
    """
    Factory function to create an AdaptiveTrustRegion.
    
    Args:
        projector: Base TrustRegionProjector
        adaptation_rate: Rate of adaptation
        violation_threshold: Threshold for triggering adaptation
        
    Returns:
        Configured AdaptiveTrustRegion
    """
    return AdaptiveTrustRegion(projector, adaptation_rate, violation_threshold)