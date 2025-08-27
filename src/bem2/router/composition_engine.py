"""
Composition Engine for Agentic Router

Safely composes BEM deltas with trust-region projection:
- Cache-safe application at W_O/W_down attachment points only
- Trust region projection: ΔW ← ΔW · min(1, τ/||ΔW||_F)
- Action parameter integration (rank_budget, bias_scale)
- Hysteresis-aware composition with minimal KV cache impact
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
from dataclasses import dataclass

from .macro_policy import MacroAction
from ...bem.trust_region import TrustRegionProjector, TrustRegionBudget
from ...bem.modules.chunk_sticky_routing import ChunkStickyRouter

logger = logging.getLogger(__name__)


@dataclass 
class BEMCompositionConfig:
    """Configuration for BEM composition."""
    attachment_points: List[str]  # ['attention.w_o', 'mlp.w_down']
    trust_region_tau: float = 1.0
    spectral_budget: float = 2.0
    frobenius_budget: float = 5.0
    cache_safe_only: bool = True


class CompositionResult(NamedTuple):
    """Result of BEM composition."""
    composed_deltas: Dict[str, torch.Tensor]  # Layer name -> composed delta
    scaling_factors: Dict[str, float]         # Layer name -> scaling factor
    trust_region_violations: List            # List of violations
    cache_safety_report: Dict                # Cache safety metrics
    composition_stats: Dict                  # Composition statistics


class ExpertBank:
    """
    Bank of expert modules for different domains.
    
    Each expert contains pre-trained LoRA-style deltas for specific attachment points.
    """
    
    def __init__(self, config: BEMCompositionConfig):
        self.config = config
        self.experts = {}
        self.expert_metadata = {}
        
    def register_expert(
        self,
        expert_id: int,
        expert_name: str,
        deltas: Dict[str, torch.Tensor],
        metadata: Optional[Dict] = None
    ):
        """
        Register an expert with its deltas.
        
        Args:
            expert_id: Numerical ID for the expert
            expert_name: Human-readable name
            deltas: Dictionary mapping layer names to delta tensors
            metadata: Optional metadata about the expert
        """
        # Validate attachment points
        for layer_name in deltas.keys():
            if layer_name not in self.config.attachment_points:
                logger.warning(f"Expert {expert_name} has delta for non-attachment point: {layer_name}")
        
        self.experts[expert_id] = {
            'name': expert_name,
            'deltas': deltas
        }
        
        self.expert_metadata[expert_id] = metadata or {}
        
        logger.info(f"Registered expert {expert_id} ({expert_name}) with {len(deltas)} deltas")
    
    def get_expert_deltas(
        self, 
        expert_id: int,
        rank_budget: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get deltas for an expert, optionally with rank budgeting.
        
        Args:
            expert_id: ID of expert to retrieve
            rank_budget: Optional rank limit for compression
            
        Returns:
            Dictionary of layer deltas
        """
        if expert_id not in self.experts:
            raise ValueError(f"Expert {expert_id} not registered")
        
        deltas = self.experts[expert_id]['deltas'].copy()
        
        # Apply rank budgeting if specified
        if rank_budget is not None:
            deltas = self._apply_rank_budget(deltas, rank_budget)
        
        return deltas
    
    def _apply_rank_budget(
        self,
        deltas: Dict[str, torch.Tensor],
        rank_budget: int
    ) -> Dict[str, torch.Tensor]:
        """
        Apply rank budget constraint to deltas via SVD truncation.
        
        Args:
            deltas: Original deltas
            rank_budget: Maximum rank to preserve
            
        Returns:
            Rank-constrained deltas
        """
        constrained_deltas = {}
        
        for layer_name, delta in deltas.items():
            if delta.dim() != 2:
                # Skip non-2D tensors
                constrained_deltas[layer_name] = delta
                continue
                
            # Apply SVD truncation
            try:
                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
                
                # Truncate to rank budget
                k = min(rank_budget, len(S))
                U_trunc = U[:, :k]
                S_trunc = S[:k]
                Vh_trunc = Vh[:k, :]
                
                # Reconstruct delta
                delta_constrained = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
                constrained_deltas[layer_name] = delta_constrained
                
                # Log compression
                original_rank = len(S)
                compression_ratio = k / original_rank
                logger.debug(f"Rank budget {layer_name}: {original_rank}→{k} "
                           f"(compression: {compression_ratio:.2f})")
                
            except Exception as e:
                logger.warning(f"SVD failed for {layer_name}, using original delta: {e}")
                constrained_deltas[layer_name] = delta
        
        return constrained_deltas
    
    def list_experts(self) -> Dict[int, Dict]:
        """List all registered experts."""
        return {
            expert_id: {
                'name': info['name'],
                'layers': list(info['deltas'].keys()),
                'metadata': self.expert_metadata.get(expert_id, {})
            }
            for expert_id, info in self.experts.items()
        }


class CompositionEngine:
    """
    Main engine for composing BEM deltas with trust-region constraints.
    
    Implements the core composition algorithm:
    1. Retrieve expert deltas based on macro-action
    2. Apply rank budgeting and bias scaling
    3. Compose deltas across experts  
    4. Apply trust-region projection
    5. Ensure cache safety
    """
    
    def __init__(
        self,
        config: BEMCompositionConfig,
        expert_bank: Optional[ExpertBank] = None
    ):
        self.config = config
        self.expert_bank = expert_bank or ExpertBank(config)
        
        # Initialize trust region projector
        budgets = {}
        for layer_name in config.attachment_points:
            budgets[layer_name] = TrustRegionBudget(
                spectral_budget=config.spectral_budget,
                frobenius_budget=config.frobenius_budget,
                layer_name=layer_name
            )
        
        self.trust_region_projector = TrustRegionProjector(budgets)
        
        # Cache safety tracker
        self.cache_violations = []
        
        logger.info(f"Initialized CompositionEngine with {len(config.attachment_points)} attachment points")
    
    def compose_action(
        self,
        action: MacroAction,
        chunk_index: int = 0,
        sequence_length: int = 1
    ) -> CompositionResult:
        """
        Compose deltas for a single macro-action.
        
        Args:
            action: MacroAction specifying expert and parameters
            chunk_index: Current chunk position
            sequence_length: Total sequence length in chunks
            
        Returns:
            CompositionResult with composed deltas and diagnostics
        """
        # Retrieve expert deltas with rank budgeting
        expert_deltas = self.expert_bank.get_expert_deltas(
            expert_id=action.expert_id,
            rank_budget=action.rank_budget
        )
        
        # Apply bias scaling
        scaled_deltas = self._apply_bias_scaling(expert_deltas, action.bias_scale)
        
        # Apply scope constraints
        scoped_deltas = self._apply_scope_constraints(
            scaled_deltas, 
            action.scope, 
            chunk_index, 
            sequence_length
        )
        
        # Single expert composition (no multi-expert mixing for now)
        bem_deltas = {f"expert_{action.expert_id}": scoped_deltas}
        
        # Apply trust region projection
        projection_result = self.trust_region_projector.project_multi_bem_deltas(bem_deltas)
        
        # Extract composed deltas
        composed_deltas = projection_result.projected_deltas
        
        # Generate cache safety report
        cache_safety_report = self._generate_cache_safety_report(
            action, chunk_index, sequence_length
        )
        
        # Compilation statistics
        composition_stats = {
            'expert_id': action.expert_id,
            'expert_name': self.expert_bank.experts[action.expert_id]['name'],
            'rank_budget': action.rank_budget,
            'bias_scale': action.bias_scale,
            'scope': action.scope,
            'span': action.span,
            'num_layers_affected': len(composed_deltas),
            'total_parameters': sum(delta.numel() for delta in composed_deltas.values()),
        }
        
        return CompositionResult(
            composed_deltas=composed_deltas,
            scaling_factors=projection_result.scaling_factors,
            trust_region_violations=projection_result.violations,
            cache_safety_report=cache_safety_report,
            composition_stats=composition_stats
        )
    
    def compose_sequence(
        self,
        actions: List[MacroAction],
        chunk_boundaries: List[int]
    ) -> Dict[int, CompositionResult]:
        """
        Compose deltas for a sequence of actions across chunks.
        
        Args:
            actions: List of macro-actions
            chunk_boundaries: Chunk boundary positions
            
        Returns:
            Dictionary mapping chunk indices to composition results
        """
        results = {}
        
        # Apply each action to its designated chunk span
        for action_idx, action in enumerate(actions):
            start_chunk = action_idx
            end_chunk = min(action_idx + action.span, len(chunk_boundaries))
            
            # Compose for each chunk in span
            for chunk_idx in range(start_chunk, end_chunk):
                if chunk_idx not in results:
                    result = self.compose_action(
                        action=action,
                        chunk_index=chunk_idx,
                        sequence_length=len(chunk_boundaries)
                    )
                    results[chunk_idx] = result
        
        return results
    
    def _apply_bias_scaling(
        self,
        deltas: Dict[str, torch.Tensor],
        bias_scale: float
    ) -> Dict[str, torch.Tensor]:
        """Apply bias scaling to deltas."""
        scaled_deltas = {}
        
        for layer_name, delta in deltas.items():
            scaled_deltas[layer_name] = delta * bias_scale
        
        return scaled_deltas
    
    def _apply_scope_constraints(
        self,
        deltas: Dict[str, torch.Tensor],
        scope: str,
        chunk_index: int,
        sequence_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        Apply scope constraints (local vs global context).
        
        For local scope, we might reduce the effective rank or apply masking.
        For global scope, we use full deltas.
        """
        if scope == 'local':
            # For local scope, slightly reduce delta magnitude to limit context
            local_factor = 0.8
            return {
                layer_name: delta * local_factor 
                for layer_name, delta in deltas.items()
            }
        else:
            # Global scope uses full deltas
            return deltas
    
    def _generate_cache_safety_report(
        self,
        action: MacroAction,
        chunk_index: int,
        sequence_length: int
    ) -> Dict:
        """Generate cache safety compliance report."""
        
        # Check if action violates cache safety
        violations = []
        
        # Cache safety requires chunk-sticky routing
        if action.span > 4:  # Spans too long might cause cache issues
            violations.append(f"Action span {action.span} exceeds recommended maximum of 4")
        
        # Bias scale too high might cause numerical instability
        if action.bias_scale > 2.0:
            violations.append(f"Bias scale {action.bias_scale} exceeds safe maximum of 2.0")
        
        return {
            'cache_safe': len(violations) == 0,
            'violations': violations,
            'attachment_points_only': self.config.cache_safe_only,
            'chunk_aligned': True,  # Actions are chunk-aligned by design
            'action_span': action.span,
            'chunk_index': chunk_index,
            'sequence_length': sequence_length
        }
    
    def apply_deltas_to_model(
        self,
        model: nn.Module,
        composed_deltas: Dict[str, torch.Tensor],
        attachment_points: Optional[Dict[str, str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply composed deltas to model at attachment points.
        
        Args:
            model: PyTorch model to modify
            composed_deltas: Composed delta tensors
            attachment_points: Mapping from delta keys to model parameter paths
            
        Returns:
            Dictionary of original weights for potential rollback
        """
        if attachment_points is None:
            # Default attachment points
            attachment_points = {
                layer_name: layer_name.replace('.', '/')
                for layer_name in composed_deltas.keys()
            }
        
        original_weights = {}
        
        for delta_key, delta in composed_deltas.items():
            if delta_key in attachment_points:
                param_path = attachment_points[delta_key]
                
                try:
                    # Navigate to parameter
                    param = model
                    for attr in param_path.split('/'):
                        param = getattr(param, attr)
                    
                    # Store original for rollback
                    original_weights[param_path] = param.data.clone()
                    
                    # Apply delta (cache-safe additive modification)
                    param.data += delta
                    
                    logger.debug(f"Applied delta to {param_path}: shape {delta.shape}")
                    
                except AttributeError as e:
                    logger.error(f"Failed to apply delta to {param_path}: {e}")
        
        return original_weights
    
    def rollback_deltas(
        self,
        model: nn.Module,
        original_weights: Dict[str, torch.Tensor]
    ):
        """Rollback applied deltas using stored original weights."""
        for param_path, original_weight in original_weights.items():
            try:
                # Navigate to parameter
                param = model
                for attr in param_path.split('/'):
                    param = getattr(param, attr)
                
                # Restore original weight
                param.data.copy_(original_weight)
                
                logger.debug(f"Rolled back {param_path}")
                
            except AttributeError as e:
                logger.error(f"Failed to rollback {param_path}: {e}")
    
    def get_composition_stats(self) -> Dict:
        """Get overall composition statistics."""
        return {
            'num_experts': len(self.expert_bank.experts),
            'attachment_points': self.config.attachment_points,
            'trust_region_tau': self.config.trust_region_tau,
            'cache_violations': len(self.cache_violations),
            'trust_region_budget': self.trust_region_projector.get_budget_info()
        }


def create_composition_engine(
    config: Dict,
    expert_deltas: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
) -> CompositionEngine:
    """
    Factory function to create CompositionEngine from configuration.
    
    Args:
        config: Configuration dictionary
        expert_deltas: Optional pre-loaded expert deltas
        
    Returns:
        Configured CompositionEngine instance
    """
    # Create composition config
    composition_config = BEMCompositionConfig(
        attachment_points=config.get('attachment_points', ['attention.w_o', 'mlp.w_down']),
        trust_region_tau=config.get('trust_region_tau', 1.0),
        spectral_budget=config.get('spectral_budget', 2.0),
        frobenius_budget=config.get('frobenius_budget', 5.0),
        cache_safe_only=config.get('cache_safe_only', True)
    )
    
    # Create expert bank
    expert_bank = ExpertBank(composition_config)
    
    # Register experts if provided
    if expert_deltas:
        expert_names = ['Code', 'Formal', 'Safety']
        for expert_id, deltas in expert_deltas.items():
            expert_name = expert_names[expert_id] if expert_id < len(expert_names) else f'Expert_{expert_id}'
            expert_bank.register_expert(expert_id, expert_name, deltas)
    
    return CompositionEngine(composition_config, expert_bank)


def create_default_experts() -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Create default expert deltas for testing.
    
    Returns:
        Dictionary mapping expert IDs to their deltas
    """
    # Create synthetic expert deltas
    experts = {}
    
    for expert_id in range(3):  # Code, Formal, Safety
        deltas = {}
        
        # Create deltas for standard attachment points
        for layer_name in ['attention.w_o', 'mlp.w_down']:
            if layer_name == 'attention.w_o':
                # Attention output projection typically smaller
                delta = torch.randn(512, 512) * 0.1
            else:
                # MLP down projection typically larger  
                delta = torch.randn(2048, 512) * 0.1
            
            # Add expert-specific bias
            if expert_id == 0:  # Code expert - smaller, focused deltas
                delta *= 0.8
            elif expert_id == 1:  # Formal expert - larger, more structured
                delta *= 1.2
            else:  # Safety expert - conservative deltas
                delta *= 0.6
            
            deltas[layer_name] = delta
        
        experts[expert_id] = deltas
    
    return experts