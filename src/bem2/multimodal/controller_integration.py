"""
Multimodal Controller Integration for BEM 2.0

Extends the hierarchical controller to accept vision features while maintaining
cache-safety. Vision features are projected to controller dimension and integrated
at the chunk level, aligned with patch windows.

Key Principle: Vision features ONLY feed the controller, never the generator sites.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from ..controller import HierarchicalController, RoutingState, RoutingLevel
from .vision_encoder import VisionFeatures


@dataclass
class MultimodalRoutingState(RoutingState):
    """Extended routing state with multimodal features."""
    vision_features: Optional[VisionFeatures] = None
    visual_attention: Optional[torch.Tensor] = None  # [batch, num_patches]
    cross_modal_alignment: Optional[torch.Tensor] = None  # [batch] - alignment score
    conflict_gate_active: bool = False  # Whether conflict gating is active


class VisionProjector(nn.Module):
    """
    Projects vision features to controller dimension.
    Handles CLS, pool, patch, and region embeddings.
    """
    
    def __init__(
        self,
        vision_dim: int,
        controller_dim: int,
        projection_mode: str = "adaptive",  # "linear", "mlp", "adaptive"
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.controller_dim = controller_dim
        self.projection_mode = projection_mode
        
        if projection_mode == "linear":
            # Simple linear projection
            self.cls_projector = nn.Linear(vision_dim, controller_dim)
            self.pool_projector = nn.Linear(vision_dim, controller_dim)
            self.patch_projector = nn.Linear(vision_dim, controller_dim)
            self.region_projector = nn.Linear(vision_dim, controller_dim)
            
        elif projection_mode == "mlp":
            # MLP projection for better expressivity
            def create_mlp(input_dim, output_dim):
                return nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, output_dim)
                )
            
            self.cls_projector = create_mlp(vision_dim, controller_dim)
            self.pool_projector = create_mlp(vision_dim, controller_dim)
            self.patch_projector = create_mlp(vision_dim, controller_dim)
            self.region_projector = create_mlp(vision_dim, controller_dim)
            
        elif projection_mode == "adaptive":
            # Adaptive projection with feature-dependent gating
            self.base_projector = nn.Linear(vision_dim, controller_dim)
            
            # Gating networks for different feature types
            self.feature_gate = nn.Sequential(
                nn.Linear(vision_dim, controller_dim // 4),
                nn.GELU(),
                nn.Linear(controller_dim // 4, controller_dim),
                nn.Sigmoid()
            )
            
        else:
            raise ValueError(f"Unknown projection mode: {projection_mode}")
        
        # Cross-modal attention for alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=controller_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection weights."""
        def init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.apply(init_module)
    
    def forward(
        self, 
        vision_features: VisionFeatures,
        text_context: Optional[torch.Tensor] = None  # [batch, seq_len, controller_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Project vision features to controller dimension.
        
        Args:
            vision_features: Input vision features
            text_context: Optional text context for cross-modal attention
            
        Returns:
            Dictionary of projected features
        """
        batch_size = vision_features.batch_size
        
        if self.projection_mode == "adaptive":
            # Adaptive projection with gating
            cls_projected = self.base_projector(vision_features.cls_embedding)
            cls_gate = self.feature_gate(vision_features.cls_embedding)
            cls_projected = cls_projected * cls_gate
            
            pool_projected = self.base_projector(vision_features.pool_embedding)
            pool_gate = self.feature_gate(vision_features.pool_embedding)
            pool_projected = pool_projected * pool_gate
            
            # For sequences, apply per-element
            patch_projected = self.base_projector(vision_features.patch_embeddings)
            patch_gates = self.feature_gate(vision_features.patch_embeddings)
            patch_projected = patch_projected * patch_gates
            
            region_projected = self.base_projector(vision_features.region_summaries)
            region_gates = self.feature_gate(vision_features.region_summaries)
            region_projected = region_projected * region_gates
            
        else:
            # Standard projection
            cls_projected = self.cls_projector(vision_features.cls_embedding)
            pool_projected = self.pool_projector(vision_features.pool_embedding)
            patch_projected = self.patch_projector(vision_features.patch_embeddings)
            region_projected = self.region_projector(vision_features.region_summaries)
        
        # Cross-modal attention if text context is provided
        cross_modal_features = None
        if text_context is not None:
            # Use CLS as query, text as key/value
            cls_query = cls_projected.unsqueeze(1)  # [batch, 1, controller_dim]
            cross_modal_features, attention_weights = self.cross_attention(
                query=cls_query,
                key=text_context,
                value=text_context
            )
            cross_modal_features = cross_modal_features.squeeze(1)  # [batch, controller_dim]
        
        return {
            'cls': cls_projected,
            'pool': pool_projected,
            'patches': patch_projected,
            'regions': region_projected,
            'cross_modal': cross_modal_features
        }


class ConflictGate(nn.Module):
    """
    Conflict gate that disables visual conditioning when consistency is low.
    Implements automatic fallback to text-only conditioning.
    """
    
    def __init__(
        self,
        controller_dim: int,
        consistency_threshold: float = 0.5,
        coverage_threshold: float = 0.3,
        gate_smoothing: float = 0.9  # EMA smoothing for gate decisions
    ):
        super().__init__()
        
        self.controller_dim = controller_dim
        self.consistency_threshold = consistency_threshold
        self.coverage_threshold = coverage_threshold
        self.gate_smoothing = gate_smoothing
        
        # Learned conflict detection
        self.conflict_detector = nn.Sequential(
            nn.Linear(controller_dim * 2, controller_dim),  # vision + text features
            nn.LayerNorm(controller_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(controller_dim, 1),
            nn.Sigmoid()
        )
        
        # EMA buffer for gate smoothing
        self.register_buffer('gate_ema', torch.tensor(1.0))
        self.register_buffer('ema_initialized', torch.tensor(False))
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [batch, controller_dim]
        text_features: torch.Tensor,   # [batch, controller_dim]
        consistency_score: Optional[torch.Tensor] = None,  # [batch]
        coverage_score: Optional[torch.Tensor] = None,     # [batch]
        training: bool = True
    ) -> Tuple[torch.Tensor, bool]:
        """
        Determine whether to gate visual conditioning.
        
        Args:
            vision_features: Projected vision features [batch, controller_dim]
            text_features: Text features [batch, controller_dim]
            consistency_score: Consistency score [batch]
            coverage_score: Coverage score [batch] 
            training: Whether in training mode
            
        Returns:
            gate_weight: Gating weight [batch] (0=text-only, 1=multimodal)
            conflict_detected: Whether conflict gating is active
        """
        batch_size = vision_features.shape[0]
        
        # Learned conflict detection
        combined_features = torch.cat([vision_features, text_features], dim=-1)
        learned_gate = self.conflict_detector(combined_features).squeeze(-1)  # [batch]
        
        # Rule-based gating from scores
        rule_gate = torch.ones(batch_size, device=vision_features.device)
        
        if consistency_score is not None:
            consistency_mask = consistency_score > self.consistency_threshold
            rule_gate = rule_gate * consistency_mask.float()
        
        if coverage_score is not None:
            coverage_mask = coverage_score > self.coverage_threshold
            rule_gate = rule_gate * coverage_mask.float()
        
        # Combine learned and rule-based gating
        gate_weight = learned_gate * rule_gate
        
        # Apply EMA smoothing during training
        if training:
            current_gate = gate_weight.mean()
            
            if not self.ema_initialized.item():
                self.gate_ema.copy_(current_gate)
                self.ema_initialized.copy_(torch.tensor(True))
            else:
                self.gate_ema.mul_(self.gate_smoothing).add_(
                    current_gate, alpha=1 - self.gate_smoothing
                )
        
        # Determine if conflict gating is active
        conflict_detected = self.gate_ema.item() < 0.8  # Threshold for conflict detection
        
        return gate_weight, conflict_detected


class ChunkAligner(nn.Module):
    """
    Aligns vision patch embeddings with text chunk windows for cache-safe integration.
    Ensures chunk routing decisions are synchronized with patch-level features.
    """
    
    def __init__(
        self,
        patch_grid_size: Tuple[int, int] = (14, 14),  # ViT patch grid
        chunk_size: int = 32,  # Text chunk size
        controller_dim: int = 512
    ):
        super().__init__()
        
        self.patch_grid_size = patch_grid_size
        self.chunk_size = chunk_size
        self.controller_dim = controller_dim
        
        num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.num_patches = num_patches
        
        # Patch-to-chunk alignment network
        self.alignment_net = nn.Sequential(
            nn.Linear(controller_dim, controller_dim // 2),
            nn.LayerNorm(controller_dim // 2),
            nn.GELU(),
            nn.Linear(controller_dim // 2, 1)
        )
    
    def forward(
        self,
        patch_features: torch.Tensor,  # [batch, num_patches, controller_dim]
        chunk_position: int,
        text_length: int
    ) -> torch.Tensor:
        """
        Align patch features with current text chunk.
        
        Args:
            patch_features: Patch-level vision features
            chunk_position: Current chunk position in text
            text_length: Total text length
            
        Returns:
            chunk_aligned_features: Features aligned with current chunk [batch, controller_dim]
        """
        batch_size, num_patches, _ = patch_features.shape
        
        # Compute chunk progress (0.0 to 1.0)
        chunk_end = min(chunk_position + self.chunk_size, text_length)
        chunk_progress = (chunk_position + (chunk_end - chunk_position) / 2) / text_length
        
        # Map chunk progress to spatial attention over patches
        # Early chunks -> focus on top/left, later chunks -> bottom/right
        h, w = self.patch_grid_size
        patch_positions = torch.zeros(num_patches, 2, device=patch_features.device)
        
        for i in range(num_patches):
            row = i // w
            col = i % w
            patch_positions[i, 0] = row / (h - 1)  # Normalized row
            patch_positions[i, 1] = col / (w - 1)  # Normalized col
        
        # Compute spatial bias based on chunk progress
        # Reading order: top-left to bottom-right
        spatial_bias = patch_positions[:, 0] * 0.7 + patch_positions[:, 1] * 0.3  # Reading order weight
        target_position = chunk_progress
        
        # Distance-based attention weights
        spatial_distances = torch.abs(spatial_bias - target_position)
        spatial_weights = torch.exp(-spatial_distances * 3.0)  # Sharpness parameter
        spatial_weights = F.softmax(spatial_weights, dim=0)
        
        # Apply learned alignment on top of spatial bias
        alignment_scores = self.alignment_net(patch_features).squeeze(-1)  # [batch, num_patches]
        learned_weights = F.softmax(alignment_scores, dim=-1)
        
        # Combine spatial and learned weights
        combined_weights = 0.7 * spatial_weights.unsqueeze(0) + 0.3 * learned_weights
        combined_weights = F.softmax(combined_weights, dim=-1)
        
        # Weighted average of patch features
        chunk_aligned_features = (patch_features * combined_weights.unsqueeze(-1)).sum(dim=1)
        
        return chunk_aligned_features


class MultimodalController(HierarchicalController):
    """
    Extended hierarchical controller with multimodal conditioning.
    
    Maintains cache-safety by only feeding vision features to the controller,
    never to generator sites (W_down/W_O).
    """
    
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        vision_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        chunk_size: int = 32,
        max_prefix_tokens: int = 128,
        ema_decay: float = 0.99,
        enable_uncertainty: bool = True,
        enable_token_routing: bool = True,
        code_clamp_value: float = 3.0,
        # Multimodal-specific parameters
        projection_mode: str = "adaptive",
        consistency_threshold: float = 0.5,
        coverage_threshold: float = 0.3,
        patch_grid_size: Tuple[int, int] = (14, 14),
        enable_conflict_gating: bool = True
    ):
        # Initialize base controller
        super().__init__(
            input_dim=input_dim,
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            chunk_size=chunk_size,
            max_prefix_tokens=max_prefix_tokens,
            ema_decay=ema_decay,
            side_signal_dim=None,  # Will be set by vision projector
            enable_uncertainty=enable_uncertainty,
            enable_token_routing=enable_token_routing,
            code_clamp_value=code_clamp_value
        )
        
        self.vision_dim = vision_dim
        self.patch_grid_size = patch_grid_size
        self.enable_conflict_gating = enable_conflict_gating
        
        # Vision projector
        self.vision_projector = VisionProjector(
            vision_dim=vision_dim,
            controller_dim=input_dim,
            projection_mode=projection_mode,
            dropout=dropout
        )
        
        # Conflict gating
        if enable_conflict_gating:
            self.conflict_gate = ConflictGate(
                controller_dim=input_dim,
                consistency_threshold=consistency_threshold,
                coverage_threshold=coverage_threshold
            )
        
        # Chunk alignment
        self.chunk_aligner = ChunkAligner(
            patch_grid_size=patch_grid_size,
            chunk_size=chunk_size,
            controller_dim=input_dim
        )
        
        # Update chunk router to accept larger side signals
        # (CLS + regions + chunk-aligned patches)
        side_signal_dim = input_dim * 3  # CLS + regions + patches
        self.chunk_router.side_signal_dim = side_signal_dim
        if hasattr(self.chunk_router, 'side_projection'):
            self.chunk_router.side_projection = nn.Linear(side_signal_dim, input_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, input_dim]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
        vision_features: Optional[VisionFeatures] = None,  # Vision features
        routing_level: Union[RoutingLevel, str] = RoutingLevel.CHUNK,
        chunk_position: int = 0,
        return_routing_state: bool = False,
        enable_vision_conditioning: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, MultimodalRoutingState]]:
        """
        Multimodal forward pass with vision conditioning.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, input_dim]
            attention_mask: Attention mask [batch, seq_len]
            vision_features: Optional vision features
            routing_level: Which routing level to use
            chunk_position: Current chunk position
            return_routing_state: Whether to return detailed routing state
            enable_vision_conditioning: Whether to enable vision conditioning
            
        Returns:
            codes: Generated codes [batch, code_dim] or [batch, seq_len, code_dim]
            routing_state: Detailed multimodal routing state (if requested)
        """
        if isinstance(routing_level, str):
            routing_level = RoutingLevel(routing_level)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Process vision features if provided
        side_signals = None
        gate_weight = None
        conflict_detected = False
        
        if vision_features is not None and enable_vision_conditioning:
            # Project vision features to controller dimension
            projected_vision = self.vision_projector(
                vision_features, 
                text_context=hidden_states
            )
            
            # Apply conflict gating if enabled
            if self.enable_conflict_gating:
                # Use prefix summary as text context for gating
                prefix_len = min(seq_len, self.max_prefix_tokens)
                text_summary = hidden_states[:, :prefix_len, :].mean(dim=1)
                
                gate_weight, conflict_detected = self.conflict_gate(
                    vision_features=projected_vision['cls'],
                    text_features=text_summary,
                    consistency_score=vision_features.consistency_score,
                    coverage_score=vision_features.coverage_score,
                    training=self.training
                )
                
                # Apply gating
                for key in projected_vision:
                    if projected_vision[key] is not None:
                        if projected_vision[key].dim() == 2:  # [batch, dim]
                            projected_vision[key] = projected_vision[key] * gate_weight.unsqueeze(-1)
                        elif projected_vision[key].dim() == 3:  # [batch, seq, dim]
                            projected_vision[key] = projected_vision[key] * gate_weight.unsqueeze(-1).unsqueeze(-1)
            
            # Prepare side signals for chunk routing
            if routing_level == RoutingLevel.CHUNK:
                # Align patches with current chunk
                chunk_aligned_patches = self.chunk_aligner(
                    patch_features=projected_vision['patches'],
                    chunk_position=chunk_position,
                    text_length=seq_len
                )
                
                # Combine CLS, regions, and chunk-aligned patches
                side_signals = torch.cat([
                    projected_vision['cls'],  # [batch, input_dim]
                    projected_vision['regions'].mean(dim=1),  # [batch, input_dim] - average regions
                    chunk_aligned_patches  # [batch, input_dim]
                ], dim=-1)  # [batch, input_dim * 3]
        
        # Call parent forward with vision-augmented side signals
        if return_routing_state:
            codes, routing_state = super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                side_signals=side_signals,
                routing_level=routing_level,
                chunk_position=chunk_position,
                return_routing_state=True
            )
            
            # Create extended multimodal routing state
            multimodal_state = MultimodalRoutingState(
                prefix_code=routing_state.prefix_code,
                chunk_code=routing_state.chunk_code,
                token_code=routing_state.token_code,
                uncertainty=routing_state.uncertainty,
                entropy=routing_state.entropy,
                utilization=routing_state.utilization,
                chunk_position=routing_state.chunk_position,
                ema_chunk_code=routing_state.ema_chunk_code,
                vision_features=vision_features,
                visual_attention=vision_features.spatial_attention if vision_features else None,
                cross_modal_alignment=gate_weight,
                conflict_gate_active=conflict_detected
            )
            
            return codes, multimodal_state
        else:
            codes = super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                side_signals=side_signals,
                routing_level=routing_level,
                chunk_position=chunk_position,
                return_routing_state=False
            )
            
            return codes


def create_multimodal_controller(
    model_config: Dict[str, Any],
    vision_config: Optional[Dict[str, Any]] = None,
    controller_config: Optional[Dict[str, Any]] = None
) -> MultimodalController:
    """
    Factory function to create multimodal controller.
    
    Args:
        model_config: Model configuration dict
        vision_config: Vision-specific configuration
        controller_config: Controller-specific configuration
        
    Returns:
        MultimodalController instance
    """
    if vision_config is None:
        vision_config = {}
    if controller_config is None:
        controller_config = {}
    
    # Extract dimensions
    hidden_size = model_config.get('hidden_size', 768)
    vision_dim = vision_config.get('vision_dim', 512)
    
    # Default parameters
    defaults = {
        'input_dim': hidden_size,
        'code_dim': controller_config.get('rank', 8),
        'vision_dim': vision_dim,
        'hidden_dim': hidden_size * 4,
        'dropout': 0.1,
        'chunk_size': 32,
        'max_prefix_tokens': 128,
        'ema_decay': 0.99,
        'enable_uncertainty': True,
        'enable_token_routing': True,
        'code_clamp_value': 3.0,
        'projection_mode': 'adaptive',
        'consistency_threshold': 0.5,
        'coverage_threshold': 0.3,
        'patch_grid_size': (14, 14),
        'enable_conflict_gating': True
    }
    
    # Override with provided configs
    defaults.update(controller_config)
    defaults.update(vision_config)
    
    return MultimodalController(**defaults)