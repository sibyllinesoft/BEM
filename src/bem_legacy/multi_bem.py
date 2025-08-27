"""
BEM Phase 4: Multi-BEM Composer Module

This module implements the composition orchestrator for multiple BEMs.
It coordinates multiple BEM instances with subspace assignments, applies
trust region projection, and manages routing coordination.

Key Features:
- BEM Registry: Manage multiple BEM instances with subspace assignments
- Composition Engine: Sum ΔW matrices with trust region projection
- Routing Coordination: Coordinate multiple controllers simultaneously
- State Management: Handle multiple EMA states and uncertainty heads
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple, Any, Union
from dataclasses import dataclass
import logging
from collections import defaultdict

from .subspace import SubspacePlanner, OrthogonalityEnforcer, SubspaceAllocation
from .trust_region import TrustRegionProjector, TrustRegionBudget, TrustRegionProjectionResult
from .simple_bem import SimpleBEMModule, BEMController
from .hierarchical_bem import HierarchicalBEMModule, FullHierarchicalBEM
from .retrieval_bem import RetrievalAwareBEMModule

logger = logging.getLogger(__name__)


@dataclass
class MultiBEMConfig:
    """Configuration for multi-BEM composition."""
    total_rank: int
    min_rank_per_bem: int = 4
    spectral_budget: float = 1.0
    frobenius_budget: float = 5.0
    use_fp32_projection: bool = True
    orthogonality_tolerance: float = 1e-6
    enable_adaptive_trust_region: bool = False
    adaptation_rate: float = 0.1
    violation_threshold: float = 0.1


class BEMRegistryEntry(NamedTuple):
    """Entry in the BEM registry."""
    bem_module: Union[SimpleBEMModule, HierarchicalBEMModule, RetrievalAwareBEMModule]
    allocation: SubspaceAllocation
    controller: Optional[BEMController]
    priority: int  # Higher priority BEMs get preference in orthogonality enforcement


class CompositionState(NamedTuple):
    """State information for multi-BEM composition."""
    bem_codes: Dict[str, torch.Tensor]  # BEM ID -> code tensor
    combined_deltas: Dict[str, torch.Tensor]  # layer name -> combined ΔW
    trust_region_result: TrustRegionProjectionResult
    orthogonality_valid: bool
    composition_timestamp: int


class MultiBEMComposer(nn.Module):
    """
    Main orchestrator for multi-BEM composition.
    
    This class manages multiple BEM instances, enforces subspace reservations,
    applies trust region projection, and coordinates routing across BEMs.
    """
    
    def __init__(self, config: MultiBEMConfig, layer_names: List[str]):
        """
        Initialize the multi-BEM composer.
        
        Args:
            config: Configuration for composition
            layer_names: List of layer names that BEMs can attach to
        """
        super().__init__()
        
        self.config = config
        self.layer_names = layer_names
        
        # Initialize core components
        self.subspace_planner = SubspacePlanner(
            total_rank=config.total_rank,
            min_rank_per_bem=config.min_rank_per_bem
        )
        
        self.orthogonality_enforcer = OrthogonalityEnforcer(
            tolerance=config.orthogonality_tolerance
        )
        
        # Create trust region budgets for each layer
        budgets = {
            layer_name: TrustRegionBudget(
                spectral_budget=config.spectral_budget,
                frobenius_budget=config.frobenius_budget,
                layer_name=layer_name
            )
            for layer_name in layer_names
        }
        
        self.trust_region_projector = TrustRegionProjector(
            budgets=budgets,
            use_fp32=config.use_fp32_projection
        )
        
        # BEM registry
        self.bem_registry: Dict[str, BEMRegistryEntry] = {}
        
        # State management
        self.composition_history: List[CompositionState] = []
        self.composition_counter = 0
        
        # Module registry for parameters
        self.bem_modules = nn.ModuleDict()
        
        logger.info(f"Initialized MultiBEMComposer with total_rank={config.total_rank}, "
                   f"layers={len(layer_names)}")
    
    def register_bem(
        self,
        bem_id: str,
        bem_module: Union[SimpleBEMModule, HierarchicalBEMModule, RetrievalAwareBEMModule],
        controller: Optional[BEMController] = None,
        requested_rank: Optional[int] = None,
        priority: int = 0
    ) -> SubspaceAllocation:
        """
        Register a new BEM with the composer.
        
        Args:
            bem_id: Unique identifier for the BEM
            bem_module: BEM module instance
            controller: Optional controller for the BEM
            requested_rank: Requested rank (defaults to BEM's rank)
            priority: Priority for orthogonality enforcement (higher = more priority)
            
        Returns:
            SubspaceAllocation for the registered BEM
            
        Raises:
            ValueError: If BEM already registered or allocation fails
        """
        if bem_id in self.bem_registry:
            raise ValueError(f"BEM {bem_id} is already registered")
        
        # Determine requested rank
        if requested_rank is None:
            if hasattr(bem_module, 'rank'):
                requested_rank = bem_module.rank
            else:
                requested_rank = self.config.min_rank_per_bem
        
        # Allocate subspace
        allocation = self.subspace_planner.allocate_subspace(bem_id, requested_rank)
        
        # Create registry entry
        entry = BEMRegistryEntry(
            bem_module=bem_module,
            allocation=allocation,
            controller=controller,
            priority=priority
        )
        
        self.bem_registry[bem_id] = entry
        
        # Register with PyTorch's module system
        self.bem_modules[bem_id] = bem_module
        
        logger.info(f"Registered BEM {bem_id} with rank={requested_rank}, priority={priority}")
        
        return allocation
    
    def unregister_bem(self, bem_id: str) -> bool:
        """
        Unregister a BEM from the composer.
        
        Args:
            bem_id: BEM to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if bem_id not in self.bem_registry:
            return False
        
        # Remove from subspace planner
        self.subspace_planner.remove_allocation(bem_id)
        
        # Remove from registry
        del self.bem_registry[bem_id]
        del self.bem_modules[bem_id]
        
        logger.info(f"Unregistered BEM {bem_id}")
        
        return True
    
    def forward(
        self,
        x: torch.Tensor,
        layer_name: str,
        routing_context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Forward pass through composed BEMs.
        
        Args:
            x: Input tensor
            layer_name: Name of the layer being processed
            routing_context: Context information for routing
            
        Returns:
            Output tensor with composed BEM modifications
        """
        if layer_name not in self.layer_names:
            # No BEMs attached to this layer
            return x
        
        # Generate codes for each BEM
        bem_codes = self._generate_bem_codes(x, layer_name, routing_context)
        
        # Compute individual BEM deltas
        bem_deltas = self._compute_bem_deltas(x, layer_name, bem_codes)
        
        # Apply trust region projection
        projection_result = self.trust_region_projector.project_multi_bem_deltas(
            {layer_name: bem_deltas} if bem_deltas else {}
        )
        
        # Apply combined deltas
        if layer_name in projection_result.projected_deltas:
            delta_output = torch.matmul(x, projection_result.projected_deltas[layer_name].t())
            output = x + delta_output
        else:
            output = x
        
        # Update composition state
        self._update_composition_state(bem_codes, projection_result)
        
        return output
    
    def _generate_bem_codes(
        self,
        x: torch.Tensor,
        layer_name: str,
        routing_context: Optional[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate codes for all registered BEMs.
        
        Args:
            x: Input tensor
            layer_name: Current layer name
            routing_context: Routing context
            
        Returns:
            Dictionary mapping BEM IDs to their generated codes
        """
        bem_codes = {}
        
        for bem_id, entry in self.bem_registry.items():
            try:
                if entry.controller is not None:
                    # Use the BEM's controller
                    code = entry.controller.generate_code(x, routing_context)
                elif hasattr(entry.bem_module, 'generate_code'):
                    # BEM has built-in code generation
                    code = entry.bem_module.generate_code(x, routing_context)
                else:
                    # Fallback to uniform code
                    batch_size = x.shape[0]
                    code = torch.ones(
                        batch_size, entry.allocation.allocated_rank,
                        device=x.device, dtype=x.dtype
                    )
                    logger.warning(f"Using fallback uniform code for BEM {bem_id}")
                
                bem_codes[bem_id] = code
                
            except Exception as e:
                logger.error(f"Failed to generate code for BEM {bem_id}: {e}")
                # Use zero code as safe fallback
                batch_size = x.shape[0]
                bem_codes[bem_id] = torch.zeros(
                    batch_size, entry.allocation.allocated_rank,
                    device=x.device, dtype=x.dtype
                )
        
        return bem_codes
    
    def _compute_bem_deltas(
        self,
        x: torch.Tensor,
        layer_name: str,
        bem_codes: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute individual BEM deltas.
        
        Args:
            x: Input tensor
            layer_name: Current layer name
            bem_codes: Generated codes for each BEM
            
        Returns:
            Dictionary mapping BEM IDs to their delta matrices
        """
        bem_deltas = {}
        
        for bem_id, code in bem_codes.items():
            entry = self.bem_registry[bem_id]
            allocation = entry.allocation
            
            try:
                # Extract U and V matrices for this BEM's subspace
                if hasattr(entry.bem_module, 'lora_U') and hasattr(entry.bem_module, 'lora_V'):
                    # Simple BEM module
                    U_full = entry.bem_module.lora_U
                    V_full = entry.bem_module.lora_V
                    
                    # Extract subspace
                    U = U_full[:, allocation.u_start_idx:allocation.u_end_idx]
                    V = V_full[:, allocation.v_start_idx:allocation.v_end_idx]
                    
                    # Compute ΔW = U * diag(code_mean) * V^T
                    code_mean = torch.mean(code, dim=0)  # Average across batch
                    delta_w = torch.matmul(U * code_mean, V.t())
                    
                    bem_deltas[bem_id] = delta_w
                    
                elif hasattr(entry.bem_module, 'compute_delta_w'):
                    # Custom delta computation
                    delta_w = entry.bem_module.compute_delta_w(code, allocation)
                    bem_deltas[bem_id] = delta_w
                    
                else:
                    logger.warning(f"BEM {bem_id} does not support delta computation")
                    
            except Exception as e:
                logger.error(f"Failed to compute delta for BEM {bem_id}: {e}")
        
        return bem_deltas
    
    def _update_composition_state(
        self,
        bem_codes: Dict[str, torch.Tensor],
        trust_region_result: TrustRegionProjectionResult
    ):
        """
        Update the internal composition state.
        
        Args:
            bem_codes: Generated BEM codes
            trust_region_result: Trust region projection result
        """
        # Validate orthogonality
        orthogonality_valid = self._validate_orthogonality()
        
        # Create composition state
        state = CompositionState(
            bem_codes=bem_codes,
            combined_deltas=trust_region_result.projected_deltas,
            trust_region_result=trust_region_result,
            orthogonality_valid=orthogonality_valid,
            composition_timestamp=self.composition_counter
        )
        
        self.composition_history.append(state)
        self.composition_counter += 1
        
        # Keep only recent history
        if len(self.composition_history) > 100:
            self.composition_history = self.composition_history[-100:]
    
    def _validate_orthogonality(self) -> bool:
        """
        Validate orthogonality of current BEM subspaces.
        
        Returns:
            True if orthogonality is valid
        """
        try:
            u_matrices = {}
            v_matrices = {}
            allocations = {}
            
            for bem_id, entry in self.bem_registry.items():
                if hasattr(entry.bem_module, 'lora_U') and hasattr(entry.bem_module, 'lora_V'):
                    u_matrices[bem_id] = entry.bem_module.lora_U
                    v_matrices[bem_id] = entry.bem_module.lora_V
                    allocations[bem_id] = entry.allocation
            
            if len(u_matrices) < 2:
                return True  # Can't have interference with < 2 BEMs
            
            result = self.orthogonality_enforcer.validate_orthogonality(
                u_matrices, v_matrices, allocations
            )
            
            return result.is_valid
            
        except Exception as e:
            logger.error(f"Orthogonality validation failed: {e}")
            return False
    
    def enforce_orthogonality(self) -> bool:
        """
        Enforce orthogonality constraints across all BEMs.
        
        Returns:
            True if enforcement was successful
        """
        try:
            u_matrices = {}
            v_matrices = {}
            allocations = {}
            
            # Collect matrices and sort by priority
            bem_priorities = [(entry.priority, bem_id, entry) 
                             for bem_id, entry in self.bem_registry.items()]
            bem_priorities.sort(reverse=True)  # Higher priority first
            
            ordered_bem_ids = [bem_id for _, bem_id, _ in bem_priorities]
            
            for bem_id, entry in self.bem_registry.items():
                if hasattr(entry.bem_module, 'lora_U') and hasattr(entry.bem_module, 'lora_V'):
                    u_matrices[bem_id] = entry.bem_module.lora_U
                    v_matrices[bem_id] = entry.bem_module.lora_V
                    allocations[bem_id] = entry.allocation
            
            # Apply orthogonality enforcement with priority ordering
            corrected_u, corrected_v = self.orthogonality_enforcer.enforce_orthogonality(
                u_matrices, v_matrices, allocations
            )
            
            # Update the BEM modules with corrected matrices
            for bem_id, corrected_u_matrix in corrected_u.items():
                entry = self.bem_registry[bem_id]
                if hasattr(entry.bem_module, 'lora_U'):
                    entry.bem_module.lora_U.data.copy_(corrected_u_matrix)
                    entry.bem_module.lora_V.data.copy_(corrected_v[bem_id])
            
            logger.info("Orthogonality enforcement completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Orthogonality enforcement failed: {e}")
            return False
    
    def get_composition_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current composition state.
        
        Returns:
            Dictionary with composition statistics
        """
        stats = {
            'num_bems': len(self.bem_registry),
            'total_allocated_rank': self.subspace_planner.allocated_rank,
            'available_rank': self.subspace_planner.total_rank - self.subspace_planner.allocated_rank,
            'composition_counter': self.composition_counter,
            'orthogonality_valid': self._validate_orthogonality(),
        }
        
        # BEM-specific stats
        bem_stats = {}
        for bem_id, entry in self.bem_registry.items():
            bem_stats[bem_id] = {
                'allocated_rank': entry.allocation.allocated_rank,
                'priority': entry.priority,
                'u_range': f"[{entry.allocation.u_start_idx}:{entry.allocation.u_end_idx}]",
                'v_range': f"[{entry.allocation.v_start_idx}:{entry.allocation.v_end_idx}]"
            }
        
        stats['bem_stats'] = bem_stats
        
        # Recent composition history
        if self.composition_history:
            recent_state = self.composition_history[-1]
            stats['recent_composition'] = {
                'trust_region_violations': len(recent_state.trust_region_result.violations),
                'orthogonality_valid': recent_state.orthogonality_valid,
                'num_active_bems': len(recent_state.bem_codes)
            }
        
        return stats
    
    def get_bem_registry(self) -> Dict[str, BEMRegistryEntry]:
        """Get a copy of the BEM registry."""
        return self.bem_registry.copy()
    
    def save_composition_checkpoint(self) -> Dict[str, Any]:
        """
        Save a checkpoint of the current composition state.
        
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = {
            'config': self.config,
            'layer_names': self.layer_names,
            'subspace_allocations': self.subspace_planner.get_all_allocations(),
            'trust_region_budgets': self.trust_region_projector.get_budget_info(),
            'composition_counter': self.composition_counter,
            'bem_priorities': {bem_id: entry.priority for bem_id, entry in self.bem_registry.items()}
        }
        
        return checkpoint
    
    def load_composition_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Load a composition checkpoint.
        
        Args:
            checkpoint: Checkpoint data to load
        """
        # Note: This is a partial implementation
        # In a full implementation, you would need to restore BEM modules as well
        self.composition_counter = checkpoint.get('composition_counter', 0)
        
        # Update priorities if available
        if 'bem_priorities' in checkpoint:
            for bem_id, priority in checkpoint['bem_priorities'].items():
                if bem_id in self.bem_registry:
                    entry = self.bem_registry[bem_id]
                    self.bem_registry[bem_id] = entry._replace(priority=priority)
        
        logger.info("Loaded composition checkpoint")


def create_multi_bem_composer(
    config: MultiBEMConfig,
    layer_names: List[str]
) -> MultiBEMComposer:
    """
    Factory function to create a MultiBEMComposer.
    
    Args:
        config: Configuration for the composer
        layer_names: List of layer names
        
    Returns:
        Configured MultiBEMComposer instance
    """
    return MultiBEMComposer(config, layer_names)


def create_default_multi_bem_config(
    total_rank: int,
    num_layers: int
) -> MultiBEMConfig:
    """
    Create a default configuration for multi-BEM composition.
    
    Args:
        total_rank: Total rank budget
        num_layers: Number of layers (used to set budgets)
        
    Returns:
        Default MultiBEMConfig
    """
    # Scale budgets based on number of layers to prevent total budget explosion
    base_frobenius_budget = max(1.0, 10.0 / num_layers)
    
    return MultiBEMConfig(
        total_rank=total_rank,
        min_rank_per_bem=4,
        spectral_budget=1.0,
        frobenius_budget=base_frobenius_budget,
        use_fp32_projection=True,
        orthogonality_tolerance=1e-6,
        enable_adaptive_trust_region=False,
        adaptation_rate=0.1,
        violation_threshold=0.1
    )