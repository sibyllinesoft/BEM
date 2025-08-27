"""
BEM Phase 4: Subspace Management Module

This module implements orthogonal subspace allocation for multi-BEM composition.
Each BEM is assigned disjoint column blocks in the U/V matrices to prevent interference.

Key Features:
- Subspace Planner: Allocate disjoint U/V column blocks
- Orthogonality Enforcer: Maintain orthogonal constraints during training
- Capacity Manager: Distribute rank capacity across multiple BEMs
- Validation: Check orthogonality preservation across updates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubspaceAllocation:
    """
    Defines the subspace allocation for a single BEM.
    """
    bem_id: str
    u_start_idx: int
    u_end_idx: int  
    v_start_idx: int
    v_end_idx: int
    allocated_rank: int
    
    def __post_init__(self):
        assert self.u_end_idx > self.u_start_idx
        assert self.v_end_idx > self.v_start_idx
        assert self.allocated_rank == (self.u_end_idx - self.u_start_idx)
        assert self.allocated_rank == (self.v_end_idx - self.v_start_idx)


class SubspaceValidationResult(NamedTuple):
    """Result of orthogonality validation."""
    is_valid: bool
    max_overlap: float
    failed_pairs: List[Tuple[str, str]]
    validation_details: Dict[str, float]


class SubspacePlanner:
    """
    Plans orthogonal subspace allocations for multiple BEMs.
    
    This class ensures that each BEM gets a disjoint column block
    in both U and V matrices, preventing interference between BEMs.
    """
    
    def __init__(self, total_rank: int, min_rank_per_bem: int = 4):
        """
        Initialize the subspace planner.
        
        Args:
            total_rank: Total rank budget for all BEMs combined
            min_rank_per_bem: Minimum rank allocation per BEM
        """
        self.total_rank = total_rank
        self.min_rank_per_bem = min_rank_per_bem
        self.allocations: Dict[str, SubspaceAllocation] = {}
        self.allocated_rank = 0
        
        logger.info(f"Initialized SubspacePlanner with total_rank={total_rank}, min_rank_per_bem={min_rank_per_bem}")
    
    def allocate_subspace(self, bem_id: str, requested_rank: int) -> SubspaceAllocation:
        """
        Allocate a subspace for a new BEM.
        
        Args:
            bem_id: Unique identifier for the BEM
            requested_rank: Requested rank allocation
            
        Returns:
            SubspaceAllocation object defining the allocated subspace
            
        Raises:
            ValueError: If allocation would exceed total rank or minimum requirements not met
        """
        if bem_id in self.allocations:
            raise ValueError(f"BEM {bem_id} already has an allocated subspace")
        
        if requested_rank < self.min_rank_per_bem:
            logger.warning(f"Requested rank {requested_rank} < min_rank_per_bem {self.min_rank_per_bem}, using minimum")
            requested_rank = self.min_rank_per_bem
        
        if self.allocated_rank + requested_rank > self.total_rank:
            available = self.total_rank - self.allocated_rank
            if available >= self.min_rank_per_bem:
                logger.warning(f"Requested rank {requested_rank} > available {available}, allocating {available}")
                requested_rank = available
            else:
                raise ValueError(f"Cannot allocate {requested_rank} rank, only {available} available")
        
        # Allocate consecutive column blocks
        u_start = self.allocated_rank
        u_end = u_start + requested_rank
        v_start = self.allocated_rank  # V allocation follows U allocation
        v_end = v_start + requested_rank
        
        allocation = SubspaceAllocation(
            bem_id=bem_id,
            u_start_idx=u_start,
            u_end_idx=u_end,
            v_start_idx=v_start,
            v_end_idx=v_end,
            allocated_rank=requested_rank
        )
        
        self.allocations[bem_id] = allocation
        self.allocated_rank += requested_rank
        
        logger.info(f"Allocated subspace for BEM {bem_id}: U[{u_start}:{u_end}], V[{v_start}:{v_end}], rank={requested_rank}")
        
        return allocation
    
    def get_allocation(self, bem_id: str) -> Optional[SubspaceAllocation]:
        """Get the allocation for a specific BEM."""
        return self.allocations.get(bem_id)
    
    def get_all_allocations(self) -> Dict[str, SubspaceAllocation]:
        """Get all current allocations."""
        return self.allocations.copy()
    
    def remove_allocation(self, bem_id: str) -> bool:
        """
        Remove a BEM's allocation. Note: This creates fragmentation.
        
        Args:
            bem_id: BEM to deallocate
            
        Returns:
            True if removed, False if not found
        """
        if bem_id not in self.allocations:
            return False
        
        allocation = self.allocations[bem_id]
        self.allocated_rank -= allocation.allocated_rank
        del self.allocations[bem_id]
        
        logger.info(f"Removed allocation for BEM {bem_id}")
        logger.warning("Subspace fragmentation created - consider defragmentation")
        
        return True
    
    def get_capacity_info(self) -> Dict[str, int]:
        """Get information about capacity usage."""
        return {
            'total_rank': self.total_rank,
            'allocated_rank': self.allocated_rank,
            'available_rank': self.total_rank - self.allocated_rank,
            'num_bems': len(self.allocations)
        }


class OrthogonalityEnforcer:
    """
    Enforces orthogonality constraints during training.
    
    This class provides methods to validate and maintain orthogonality
    between BEM subspaces during parameter updates.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize the orthogonality enforcer.
        
        Args:
            tolerance: Maximum allowed overlap between subspaces
        """
        self.tolerance = tolerance
        self.validation_history: List[SubspaceValidationResult] = []
        
    def validate_orthogonality(
        self,
        u_matrices: Dict[str, torch.Tensor],
        v_matrices: Dict[str, torch.Tensor], 
        allocations: Dict[str, SubspaceAllocation]
    ) -> SubspaceValidationResult:
        """
        Validate orthogonality between BEM subspaces.
        
        Args:
            u_matrices: Dictionary mapping BEM IDs to their U matrices
            v_matrices: Dictionary mapping BEM IDs to their V matrices
            allocations: Dictionary mapping BEM IDs to their allocations
            
        Returns:
            SubspaceValidationResult with validation details
        """
        failed_pairs = []
        max_overlap = 0.0
        validation_details = {}
        
        bem_ids = list(allocations.keys())
        
        for i, bem_id_1 in enumerate(bem_ids):
            for bem_id_2 in bem_ids[i+1:]:
                alloc_1 = allocations[bem_id_1]
                alloc_2 = allocations[bem_id_2]
                
                # Check if subspaces are supposed to be disjoint
                u_overlap = not (alloc_1.u_end_idx <= alloc_2.u_start_idx or 
                               alloc_2.u_end_idx <= alloc_1.u_start_idx)
                v_overlap = not (alloc_1.v_end_idx <= alloc_2.v_start_idx or 
                               alloc_2.v_end_idx <= alloc_1.v_start_idx)
                
                if u_overlap or v_overlap:
                    # This should not happen with proper allocation
                    logger.error(f"Allocation overlap detected between {bem_id_1} and {bem_id_2}")
                    failed_pairs.append((bem_id_1, bem_id_2))
                    continue
                
                # Extract subspace matrices
                u1 = u_matrices[bem_id_1][:, alloc_1.u_start_idx:alloc_1.u_end_idx]
                u2 = u_matrices[bem_id_2][:, alloc_2.u_start_idx:alloc_2.u_end_idx]
                v1 = v_matrices[bem_id_1][:, alloc_1.v_start_idx:alloc_1.v_end_idx]
                v2 = v_matrices[bem_id_2][:, alloc_2.v_start_idx:alloc_2.v_end_idx]
                
                # Compute cross-correlations (should be close to zero for orthogonal subspaces)
                u_cross_corr = torch.abs(torch.matmul(u1.t(), u2)).max().item()
                v_cross_corr = torch.abs(torch.matmul(v1.t(), v2)).max().item()
                
                max_cross_corr = max(u_cross_corr, v_cross_corr)
                max_overlap = max(max_overlap, max_cross_corr)
                
                validation_details[f'{bem_id_1}_{bem_id_2}_u_corr'] = u_cross_corr
                validation_details[f'{bem_id_1}_{bem_id_2}_v_corr'] = v_cross_corr
                
                if max_cross_corr > self.tolerance:
                    failed_pairs.append((bem_id_1, bem_id_2))
                    logger.warning(f"Orthogonality violation between {bem_id_1} and {bem_id_2}: "
                                 f"max_cross_corr={max_cross_corr:.6f} > tolerance={self.tolerance}")
        
        is_valid = len(failed_pairs) == 0 and max_overlap <= self.tolerance
        
        result = SubspaceValidationResult(
            is_valid=is_valid,
            max_overlap=max_overlap,
            failed_pairs=failed_pairs,
            validation_details=validation_details
        )
        
        self.validation_history.append(result)
        
        if is_valid:
            logger.debug(f"Orthogonality validation passed: max_overlap={max_overlap:.8f}")
        else:
            logger.error(f"Orthogonality validation failed: {len(failed_pairs)} violations, max_overlap={max_overlap:.8f}")
        
        return result
    
    def enforce_orthogonality(
        self,
        u_matrices: Dict[str, torch.Tensor],
        v_matrices: Dict[str, torch.Tensor],
        allocations: Dict[str, SubspaceAllocation]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Enforce orthogonality by projecting overlapping components.
        
        This method uses Gram-Schmidt orthogonalization to enforce
        orthogonality between BEM subspaces when violations are detected.
        
        Args:
            u_matrices: Dictionary mapping BEM IDs to their U matrices
            v_matrices: Dictionary mapping BEM IDs to their V matrices  
            allocations: Dictionary mapping BEM IDs to their allocations
            
        Returns:
            Tuple of (corrected_u_matrices, corrected_v_matrices)
        """
        corrected_u = {bem_id: matrix.clone() for bem_id, matrix in u_matrices.items()}
        corrected_v = {bem_id: matrix.clone() for bem_id, matrix in v_matrices.items()}
        
        bem_ids = list(allocations.keys())
        
        # Apply Gram-Schmidt process in order of BEM priority (first come, first served)
        for i, bem_id_1 in enumerate(bem_ids):
            alloc_1 = allocations[bem_id_1]
            
            for bem_id_2 in bem_ids[i+1:]:
                alloc_2 = allocations[bem_id_2]
                
                # Extract subspaces
                u1 = corrected_u[bem_id_1][:, alloc_1.u_start_idx:alloc_1.u_end_idx]
                u2 = corrected_u[bem_id_2][:, alloc_2.u_start_idx:alloc_2.u_end_idx]
                v1 = corrected_v[bem_id_1][:, alloc_1.v_start_idx:alloc_1.v_end_idx]
                v2 = corrected_v[bem_id_2][:, alloc_2.v_start_idx:alloc_2.v_end_idx]
                
                # Orthogonalize U subspaces
                u2_corrected = self._gram_schmidt_project_out(u2, u1)
                corrected_u[bem_id_2][:, alloc_2.u_start_idx:alloc_2.u_end_idx] = u2_corrected
                
                # Orthogonalize V subspaces  
                v2_corrected = self._gram_schmidt_project_out(v2, v1)
                corrected_v[bem_id_2][:, alloc_2.v_start_idx:alloc_2.v_end_idx] = v2_corrected
        
        logger.info(f"Applied orthogonality enforcement to {len(bem_ids)} BEMs")
        
        return corrected_u, corrected_v
    
    def _gram_schmidt_project_out(self, target: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """
        Project out components of basis from target using Gram-Schmidt.
        
        Args:
            target: Matrix to project [dim, rank_target]
            basis: Basis to project out [dim, rank_basis]
            
        Returns:
            Projected target matrix
        """
        target_corrected = target.clone()
        
        # QR decomposition of basis for numerical stability
        Q, _ = torch.linalg.qr(basis)
        
        # Project out basis components from target
        projection = torch.matmul(Q, torch.matmul(Q.t(), target))
        target_corrected = target - projection
        
        # Renormalize columns
        norms = torch.norm(target_corrected, dim=0, keepdim=True)
        target_corrected = target_corrected / (norms + 1e-8)
        
        return target_corrected
    
    def get_validation_history(self) -> List[SubspaceValidationResult]:
        """Get the history of validation results."""
        return self.validation_history.copy()


class CapacityManager:
    """
    Manages rank capacity distribution across multiple BEMs.
    
    This class provides utilities for optimizing rank allocation
    based on BEM performance and utilization patterns.
    """
    
    def __init__(self, planner: SubspacePlanner):
        """
        Initialize the capacity manager.
        
        Args:
            planner: SubspacePlanner instance to manage
        """
        self.planner = planner
        self.utilization_history: Dict[str, List[float]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
    def record_utilization(self, bem_id: str, utilization: float):
        """
        Record utilization for a BEM.
        
        Args:
            bem_id: BEM identifier
            utilization: Utilization metric (0.0 to 1.0)
        """
        if bem_id not in self.utilization_history:
            self.utilization_history[bem_id] = []
        
        self.utilization_history[bem_id].append(utilization)
        
        # Keep only recent history
        if len(self.utilization_history[bem_id]) > 100:
            self.utilization_history[bem_id] = self.utilization_history[bem_id][-100:]
    
    def record_performance(self, bem_id: str, performance: float):
        """
        Record performance for a BEM.
        
        Args:
            bem_id: BEM identifier
            performance: Performance metric (higher is better)
        """
        if bem_id not in self.performance_history:
            self.performance_history[bem_id] = []
        
        self.performance_history[bem_id].append(performance)
        
        # Keep only recent history
        if len(self.performance_history[bem_id]) > 100:
            self.performance_history[bem_id] = self.performance_history[bem_id][-100:]
    
    def get_utilization_stats(self, bem_id: str) -> Optional[Dict[str, float]]:
        """Get utilization statistics for a BEM."""
        if bem_id not in self.utilization_history or not self.utilization_history[bem_id]:
            return None
        
        values = self.utilization_history[bem_id]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'recent_mean': np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        }
    
    def get_performance_stats(self, bem_id: str) -> Optional[Dict[str, float]]:
        """Get performance statistics for a BEM."""
        if bem_id not in self.performance_history or not self.performance_history[bem_id]:
            return None
        
        values = self.performance_history[bem_id]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'recent_mean': np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        }
    
    def suggest_reallocation(self) -> Optional[Dict[str, int]]:
        """
        Suggest rank reallocation based on utilization and performance.
        
        Returns:
            Dictionary mapping BEM IDs to suggested new ranks, or None if no changes needed
        """
        allocations = self.planner.get_all_allocations()
        if len(allocations) < 2:
            return None
        
        suggestions = {}
        total_current_rank = sum(alloc.allocated_rank for alloc in allocations.values())
        
        # Calculate efficiency scores for each BEM
        efficiency_scores = {}
        for bem_id in allocations:
            util_stats = self.get_utilization_stats(bem_id)
            perf_stats = self.get_performance_stats(bem_id)
            
            if util_stats is None or perf_stats is None:
                efficiency_scores[bem_id] = 1.0  # Neutral score for new BEMs
                continue
            
            # Efficiency = performance * utilization (higher is better)
            efficiency = perf_stats['recent_mean'] * util_stats['recent_mean']
            efficiency_scores[bem_id] = efficiency
        
        # Redistribute rank based on efficiency
        total_efficiency = sum(efficiency_scores.values())
        if total_efficiency == 0:
            return None
        
        reallocation_needed = False
        for bem_id, allocation in allocations.items():
            current_rank = allocation.allocated_rank
            target_proportion = efficiency_scores[bem_id] / total_efficiency
            suggested_rank = max(self.planner.min_rank_per_bem, 
                               int(total_current_rank * target_proportion))
            
            if abs(suggested_rank - current_rank) > 1:  # Only suggest significant changes
                suggestions[bem_id] = suggested_rank
                reallocation_needed = True
        
        return suggestions if reallocation_needed else None


def create_subspace_planner(total_rank: int, min_rank_per_bem: int = 4) -> SubspacePlanner:
    """
    Factory function to create a SubspacePlanner.
    
    Args:
        total_rank: Total rank budget
        min_rank_per_bem: Minimum rank per BEM
        
    Returns:
        Configured SubspacePlanner instance
    """
    return SubspacePlanner(total_rank, min_rank_per_bem)


def create_orthogonality_enforcer(tolerance: float = 1e-6) -> OrthogonalityEnforcer:
    """
    Factory function to create an OrthogonalityEnforcer.
    
    Args:
        tolerance: Orthogonality tolerance
        
    Returns:
        Configured OrthogonalityEnforcer instance  
    """
    return OrthogonalityEnforcer(tolerance)


def create_capacity_manager(planner: SubspacePlanner) -> CapacityManager:
    """
    Factory function to create a CapacityManager.
    
    Args:
        planner: SubspacePlanner to manage
        
    Returns:
        Configured CapacityManager instance
    """
    return CapacityManager(planner)