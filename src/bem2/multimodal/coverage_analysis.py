"""
Coverage and Consistency Analysis for BEM 2.0 Multimodal Conditioning

Implements sophisticated coverage analysis and consistency monitoring to reduce
hallucination and improve multimodal alignment. Provides automatic fallback
mechanisms when visual conditioning conflicts with text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
import numpy as np
from enum import Enum

from .vision_encoder import VisionFeatures


class ConsistencyLevel(Enum):
    """Consistency quality levels."""
    HIGH = "high"       # >0.8
    MEDIUM = "medium"   # 0.5-0.8  
    LOW = "low"         # 0.3-0.5
    CONFLICT = "conflict"  # <0.3


class CoverageMetrics(NamedTuple):
    """Coverage analysis metrics."""
    spatial_entropy: torch.Tensor      # [batch] - Spatial attention entropy
    object_coverage: torch.Tensor      # [batch] - Object detection coverage
    region_diversity: torch.Tensor     # [batch] - Diversity of region features
    attention_spread: torch.Tensor     # [batch] - Attention distribution spread
    overall_score: torch.Tensor        # [batch] - Combined coverage score


class ConsistencyMetrics(NamedTuple):
    """Consistency analysis metrics."""
    global_local_consistency: torch.Tensor   # [batch] - CLS vs region consistency
    cross_modal_alignment: torch.Tensor      # [batch] - Vision-text alignment  
    temporal_consistency: torch.Tensor       # [batch] - Consistency over time
    semantic_coherence: torch.Tensor         # [batch] - Semantic feature coherence
    overall_score: torch.Tensor              # [batch] - Combined consistency score


@dataclass
class ConflictAnalysis:
    """Analysis of multimodal conflicts."""
    conflict_detected: bool
    confidence_level: ConsistencyLevel
    coverage_score: float
    consistency_score: float
    recommended_action: str  # "proceed", "gate_vision", "fallback_text"
    failure_modes: List[str]  # List of detected failure modes


class ObjectDetector(nn.Module):
    """
    Lightweight object detection for coverage analysis.
    Uses region features to detect and localize key objects.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_object_classes: int = 80,  # COCO classes
        detection_threshold: float = 0.5
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_object_classes = num_object_classes
        self.detection_threshold = detection_threshold
        
        # Object classification head
        self.object_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_object_classes),
            nn.Sigmoid()  # Multi-class probabilities
        )
        
        # Spatial localization head
        self.spatial_localizer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 4),  # [x, y, w, h] bounding box
            nn.Sigmoid()  # Normalized coordinates
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 8),
            nn.GELU(),
            nn.Linear(feature_dim // 8, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        region_features: torch.Tensor  # [batch, num_regions, feature_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Detect objects in regions.
        
        Args:
            region_features: Region-level features
            
        Returns:
            Detection results dictionary
        """
        batch_size, num_regions, _ = region_features.shape
        
        # Object classification
        object_probs = self.object_classifier(region_features)  # [batch, num_regions, num_classes]
        
        # Spatial localization
        spatial_coords = self.spatial_localizer(region_features)  # [batch, num_regions, 4]
        
        # Confidence scores
        confidence_scores = self.confidence_estimator(region_features).squeeze(-1)  # [batch, num_regions]
        
        # Apply detection threshold
        detection_mask = (object_probs.max(dim=-1)[0] > self.detection_threshold) & \
                        (confidence_scores > self.detection_threshold)
        
        return {
            'object_probs': object_probs,
            'spatial_coords': spatial_coords,
            'confidence_scores': confidence_scores,
            'detection_mask': detection_mask,
            'num_detections': detection_mask.sum(dim=-1).float()  # [batch]
        }


class CoverageAnalyzer(nn.Module):
    """
    Comprehensive coverage analysis for multimodal features.
    Analyzes spatial distribution, object coverage, and attention patterns.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        num_regions: int = 8,
        patch_grid_size: Tuple[int, int] = (14, 14)
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.num_regions = num_regions
        self.patch_grid_size = patch_grid_size
        
        # Object detector
        self.object_detector = ObjectDetector(vision_dim)
        
        # Coverage quality estimator
        self.coverage_estimator = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.LayerNorm(vision_dim // 2),
            nn.GELU(),
            nn.Linear(vision_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _compute_spatial_entropy(
        self,
        attention_weights: Optional[torch.Tensor],
        patch_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial entropy of attention distribution."""
        if attention_weights is not None:
            # Use provided attention weights
            normalized_attention = F.softmax(attention_weights, dim=-1)
        else:
            # Compute attention from patch magnitudes
            patch_norms = patch_embeddings.norm(dim=-1)  # [batch, num_patches]
            normalized_attention = F.softmax(patch_norms, dim=-1)
        
        # Compute entropy
        entropy = -(normalized_attention * torch.log(normalized_attention + 1e-8)).sum(dim=-1)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(normalized_attention.shape[1])
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def _compute_region_diversity(self, region_features: torch.Tensor) -> torch.Tensor:
        """Compute diversity of region features."""
        batch_size, num_regions, feature_dim = region_features.shape
        
        # Compute pairwise cosine similarities between regions
        region_norms = F.normalize(region_features, dim=-1)  # [batch, num_regions, feature_dim]
        
        # Expand for pairwise comparison
        regions_expanded_1 = region_norms.unsqueeze(2)  # [batch, num_regions, 1, feature_dim]
        regions_expanded_2 = region_norms.unsqueeze(1)  # [batch, 1, num_regions, feature_dim]
        
        # Compute cosine similarities
        similarities = (regions_expanded_1 * regions_expanded_2).sum(dim=-1)  # [batch, num_regions, num_regions]
        
        # Remove diagonal (self-similarities)
        eye = torch.eye(num_regions, device=region_features.device)
        similarities = similarities * (1 - eye.unsqueeze(0))
        
        # Diversity is inverse of average similarity
        avg_similarity = similarities.sum(dim=(1, 2)) / (num_regions * (num_regions - 1))
        diversity = 1.0 - avg_similarity
        
        return diversity
    
    def _compute_attention_spread(
        self,
        attention_weights: Optional[torch.Tensor],
        patch_grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Compute spatial spread of attention."""
        if attention_weights is None:
            # Return neutral score if no attention
            return torch.ones(attention_weights.shape[0] if attention_weights is not None else 1, 
                             device=attention_weights.device if attention_weights is not None else 'cpu')
        
        batch_size, num_patches = attention_weights.shape
        h, w = patch_grid_size
        
        # Create spatial coordinates for patches
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, h, device=attention_weights.device),
            torch.linspace(0, 1, w, device=attention_weights.device),
            indexing='ij'
        )
        
        coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1)  # [num_patches, 2]
        
        # Compute attention-weighted centroid
        centroids = torch.einsum('bn,nd->bd', attention_weights, coords)  # [batch, 2]
        
        # Compute spread as weighted variance from centroid
        coord_diffs = coords.unsqueeze(0) - centroids.unsqueeze(1)  # [batch, num_patches, 2]
        weighted_variances = torch.einsum('bn,bnd->bd', attention_weights, coord_diffs ** 2)  # [batch, 2]
        
        # Total spread as sum of variances
        spread = weighted_variances.sum(dim=1)  # [batch]
        
        return spread
    
    def forward(
        self,
        vision_features: VisionFeatures
    ) -> CoverageMetrics:
        """
        Analyze coverage quality of vision features.
        
        Args:
            vision_features: Input vision features
            
        Returns:
            Coverage metrics
        """
        batch_size = vision_features.batch_size
        
        # 1. Spatial entropy analysis
        spatial_entropy = self._compute_spatial_entropy(
            vision_features.spatial_attention,
            vision_features.patch_embeddings
        )
        
        # 2. Object detection coverage
        detection_results = self.object_detector(vision_features.region_summaries)
        object_coverage = detection_results['num_detections'] / self.num_regions  # Normalize by num regions
        
        # 3. Region diversity
        region_diversity = self._compute_region_diversity(vision_features.region_summaries)
        
        # 4. Attention spread
        attention_spread = self._compute_attention_spread(
            vision_features.spatial_attention,
            self.patch_grid_size
        )
        
        # 5. Overall coverage score (learned combination)
        # Combine all features for learned scoring
        combined_features = torch.stack([
            spatial_entropy,
            object_coverage,
            region_diversity,
            attention_spread
        ], dim=-1)  # [batch, 4]
        
        # Use region features as context for coverage estimation
        region_context = vision_features.region_summaries.mean(dim=1)  # [batch, vision_dim]
        overall_score = self.coverage_estimator(region_context).squeeze(-1)  # [batch]
        
        return CoverageMetrics(
            spatial_entropy=spatial_entropy,
            object_coverage=object_coverage,
            region_diversity=region_diversity,
            attention_spread=attention_spread,
            overall_score=overall_score
        )


class ConsistencyAnalyzer(nn.Module):
    """
    Comprehensive consistency analysis between multimodal features.
    Detects conflicts between vision and text modalities.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        text_dim: int = 512,
        num_regions: int = 8
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_regions = num_regions
        
        # Cross-modal alignment network
        self.cross_modal_aligner = nn.MultiheadAttention(
            embed_dim=min(vision_dim, text_dim),
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature projectors for alignment
        self.vision_projector = nn.Linear(vision_dim, min(vision_dim, text_dim))
        self.text_projector = nn.Linear(text_dim, min(vision_dim, text_dim))
        
        # Consistency estimator
        self.consistency_estimator = nn.Sequential(
            nn.Linear(min(vision_dim, text_dim) * 2, min(vision_dim, text_dim)),
            nn.LayerNorm(min(vision_dim, text_dim)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(min(vision_dim, text_dim), 1),
            nn.Sigmoid()
        )
        
        # Temporal consistency tracker
        self.register_buffer('prev_vision_features', torch.zeros(1, vision_dim))
        self.register_buffer('prev_text_features', torch.zeros(1, text_dim))
        self.register_buffer('consistency_history', torch.zeros(10))  # Track last 10 scores
        self.register_buffer('history_index', torch.tensor(0))
    
    def _compute_global_local_consistency(
        self,
        cls_embedding: torch.Tensor,
        region_summaries: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency between global CLS and local region features."""
        # Already computed in vision encoder, but we can refine it here
        cls_expanded = cls_embedding.unsqueeze(1).expand(-1, self.num_regions, -1)
        similarities = F.cosine_similarity(cls_expanded, region_summaries, dim=-1)
        consistency = similarities.mean(dim=-1)
        return consistency
    
    def _compute_cross_modal_alignment(
        self,
        vision_features: torch.Tensor,  # [batch, vision_dim]
        text_features: torch.Tensor     # [batch, text_dim]
    ) -> torch.Tensor:
        """Compute alignment between vision and text features."""
        # Project to common space
        vision_proj = self.vision_projector(vision_features)  # [batch, min_dim]
        text_proj = self.text_projector(text_features)        # [batch, min_dim]
        
        # Cross-modal attention
        vision_query = vision_proj.unsqueeze(1)  # [batch, 1, min_dim]
        text_key_value = text_proj.unsqueeze(1)  # [batch, 1, min_dim]
        
        aligned_vision, attention_weights = self.cross_modal_aligner(
            query=vision_query,
            key=text_key_value,
            value=text_key_value
        )
        
        aligned_vision = aligned_vision.squeeze(1)  # [batch, min_dim]
        
        # Compute alignment score
        alignment = F.cosine_similarity(aligned_vision, text_proj, dim=-1)
        
        return alignment
    
    def _update_temporal_consistency(
        self,
        current_vision: torch.Tensor,  # [batch, vision_dim] 
        current_text: torch.Tensor,    # [batch, text_dim]
        consistency_score: torch.Tensor  # [batch]
    ) -> torch.Tensor:
        """Update and compute temporal consistency."""
        batch_size = current_vision.shape[0]
        
        # For simplicity, use batch mean for temporal tracking
        curr_vision_mean = current_vision.mean(dim=0)  # [vision_dim]
        curr_text_mean = current_text.mean(dim=0)      # [text_dim]
        curr_consistency_mean = consistency_score.mean().item()
        
        # Compute temporal consistency if we have previous features
        if self.prev_vision_features.norm() > 0:
            vision_temporal = F.cosine_similarity(
                curr_vision_mean.unsqueeze(0), 
                self.prev_vision_features.unsqueeze(0)
            ).item()
            
            text_temporal = F.cosine_similarity(
                curr_text_mean.unsqueeze(0),
                self.prev_text_features.unsqueeze(0)
            ).item()
            
            temporal_consistency = (vision_temporal + text_temporal) / 2
        else:
            temporal_consistency = 1.0  # Perfect consistency for first timestep
        
        # Update history
        current_idx = self.history_index.item()
        self.consistency_history[current_idx] = curr_consistency_mean
        self.history_index.copy_(torch.tensor((current_idx + 1) % 10))
        
        # Update previous features
        self.prev_vision_features.copy_(curr_vision_mean)
        self.prev_text_features.copy_(curr_text_mean)
        
        # Return temporal consistency for all batch items
        return torch.tensor(temporal_consistency, device=current_vision.device).expand(batch_size)
    
    def _compute_semantic_coherence(
        self,
        vision_features: VisionFeatures
    ) -> torch.Tensor:
        """Compute semantic coherence within vision features."""
        # Check coherence between CLS, pool, and region features
        cls_pool_similarity = F.cosine_similarity(
            vision_features.cls_embedding, 
            vision_features.pool_embedding, 
            dim=-1
        )
        
        # Average similarity between CLS and regions
        cls_expanded = vision_features.cls_embedding.unsqueeze(1)
        cls_region_similarities = F.cosine_similarity(
            cls_expanded, 
            vision_features.region_summaries, 
            dim=-1
        ).mean(dim=-1)
        
        # Combined coherence score
        coherence = (cls_pool_similarity + cls_region_similarities) / 2
        
        return coherence
    
    def forward(
        self,
        vision_features: VisionFeatures,
        text_features: torch.Tensor  # [batch, text_dim]
    ) -> ConsistencyMetrics:
        """
        Analyze consistency between vision and text features.
        
        Args:
            vision_features: Vision features
            text_features: Text features (e.g., prefix summary)
            
        Returns:
            Consistency metrics
        """
        # 1. Global-local consistency (within vision)
        global_local_consistency = self._compute_global_local_consistency(
            vision_features.cls_embedding,
            vision_features.region_summaries
        )
        
        # 2. Cross-modal alignment (vision-text)
        cross_modal_alignment = self._compute_cross_modal_alignment(
            vision_features.cls_embedding,
            text_features
        )
        
        # 3. Temporal consistency (across time)
        temporal_consistency = self._update_temporal_consistency(
            vision_features.cls_embedding,
            text_features,
            cross_modal_alignment
        )
        
        # 4. Semantic coherence (within vision)
        semantic_coherence = self._compute_semantic_coherence(vision_features)
        
        # 5. Overall consistency score (learned combination)
        combined_features = torch.cat([
            vision_features.cls_embedding,
            text_features
        ], dim=-1)
        
        overall_score = self.consistency_estimator(combined_features).squeeze(-1)
        
        return ConsistencyMetrics(
            global_local_consistency=global_local_consistency,
            cross_modal_alignment=cross_modal_alignment,
            temporal_consistency=temporal_consistency,
            semantic_coherence=semantic_coherence,
            overall_score=overall_score
        )


class ConsistencyGate(nn.Module):
    """
    Main consistency gate that combines coverage and consistency analysis
    to determine when to enable/disable visual conditioning.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        text_dim: int = 512,
        num_regions: int = 8,
        patch_grid_size: Tuple[int, int] = (14, 14),
        # Thresholds for gating decisions
        high_consistency_threshold: float = 0.8,
        medium_consistency_threshold: float = 0.5,
        low_consistency_threshold: float = 0.3,
        coverage_threshold: float = 0.4
    ):
        super().__init__()
        
        self.high_consistency_threshold = high_consistency_threshold
        self.medium_consistency_threshold = medium_consistency_threshold
        self.low_consistency_threshold = low_consistency_threshold
        self.coverage_threshold = coverage_threshold
        
        # Analysis components
        self.coverage_analyzer = CoverageAnalyzer(
            vision_dim=vision_dim,
            num_regions=num_regions,
            patch_grid_size=patch_grid_size
        )
        
        self.consistency_analyzer = ConsistencyAnalyzer(
            vision_dim=vision_dim,
            text_dim=text_dim,
            num_regions=num_regions
        )
        
        # Decision fusion network
        self.decision_network = nn.Sequential(
            nn.Linear(8, 16),  # 4 coverage + 4 consistency metrics
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 3),  # [proceed, gate, fallback] probabilities
            nn.Softmax(dim=-1)
        )
        
        # Statistics tracking
        self.register_buffer('gate_count', torch.tensor(0))
        self.register_buffer('total_count', torch.tensor(0))
        self.register_buffer('avg_coverage', torch.tensor(0.0))
        self.register_buffer('avg_consistency', torch.tensor(0.0))
    
    def _classify_consistency_level(
        self, 
        consistency_score: torch.Tensor
    ) -> Tuple[ConsistencyLevel, torch.Tensor]:
        """Classify consistency level and return gating weights."""
        batch_size = consistency_score.shape[0]
        
        # Create level masks
        high_mask = consistency_score > self.high_consistency_threshold
        medium_mask = (consistency_score > self.medium_consistency_threshold) & ~high_mask
        low_mask = (consistency_score > self.low_consistency_threshold) & ~high_mask & ~medium_mask
        conflict_mask = ~high_mask & ~medium_mask & ~low_mask
        
        # Assign weights based on levels
        weights = torch.zeros_like(consistency_score)
        weights[high_mask] = 1.0      # Full visual conditioning
        weights[medium_mask] = 0.7    # Reduced visual conditioning
        weights[low_mask] = 0.3       # Minimal visual conditioning
        weights[conflict_mask] = 0.0  # Text-only fallback
        
        # Determine overall level (use mode for batch)
        level_counts = torch.tensor([
            high_mask.sum().item(),
            medium_mask.sum().item(),
            low_mask.sum().item(),
            conflict_mask.sum().item()
        ])
        
        dominant_level_idx = level_counts.argmax().item()
        levels = [ConsistencyLevel.HIGH, ConsistencyLevel.MEDIUM, 
                 ConsistencyLevel.LOW, ConsistencyLevel.CONFLICT]
        dominant_level = levels[dominant_level_idx]
        
        return dominant_level, weights
    
    def _detect_failure_modes(
        self,
        coverage_metrics: CoverageMetrics,
        consistency_metrics: ConsistencyMetrics
    ) -> List[str]:
        """Detect specific failure modes."""
        failure_modes = []
        
        # Low coverage failures
        if coverage_metrics.spatial_entropy.mean() < 0.3:
            failure_modes.append("low_spatial_diversity")
        
        if coverage_metrics.object_coverage.mean() < 0.2:
            failure_modes.append("insufficient_object_detection")
        
        if coverage_metrics.region_diversity.mean() < 0.4:
            failure_modes.append("homogeneous_regions")
        
        # Consistency failures
        if consistency_metrics.global_local_consistency.mean() < 0.4:
            failure_modes.append("global_local_mismatch")
        
        if consistency_metrics.cross_modal_alignment.mean() < 0.3:
            failure_modes.append("vision_text_misalignment")
        
        if consistency_metrics.temporal_consistency.mean() < 0.5:
            failure_modes.append("temporal_instability")
        
        if consistency_metrics.semantic_coherence.mean() < 0.4:
            failure_modes.append("semantic_incoherence")
        
        return failure_modes
    
    def forward(
        self,
        vision_features: VisionFeatures,
        text_features: torch.Tensor,  # [batch, text_dim]
        return_analysis: bool = False
    ) -> Tuple[torch.Tensor, ConflictAnalysis]:
        """
        Main gating decision.
        
        Args:
            vision_features: Vision features
            text_features: Text context features
            return_analysis: Whether to return detailed analysis
            
        Returns:
            gate_weights: Gating weights [batch] (0=text-only, 1=full-multimodal)
            conflict_analysis: Detailed conflict analysis
        """
        batch_size = vision_features.batch_size
        
        # 1. Analyze coverage
        coverage_metrics = self.coverage_analyzer(vision_features)
        
        # 2. Analyze consistency
        consistency_metrics = self.consistency_analyzer(vision_features, text_features)
        
        # 3. Combine metrics for decision
        combined_metrics = torch.stack([
            coverage_metrics.spatial_entropy,
            coverage_metrics.object_coverage,
            coverage_metrics.region_diversity,
            coverage_metrics.attention_spread,
            consistency_metrics.global_local_consistency,
            consistency_metrics.cross_modal_alignment,
            consistency_metrics.temporal_consistency,
            consistency_metrics.semantic_coherence
        ], dim=-1)  # [batch, 8]
        
        # 4. Neural decision fusion
        decision_logits = self.decision_network(combined_metrics)  # [batch, 3]
        
        # 5. Rule-based consistency classification
        overall_consistency = consistency_metrics.overall_score
        overall_coverage = coverage_metrics.overall_score
        
        consistency_level, rule_weights = self._classify_consistency_level(overall_consistency)
        
        # 6. Combine neural and rule-based decisions
        # Neural weights: proceed=1.0, gate=0.5, fallback=0.0
        neural_weights = decision_logits[:, 0] * 1.0 + decision_logits[:, 1] * 0.5 + decision_logits[:, 2] * 0.0
        
        # Final weights as average of neural and rule-based
        final_weights = (neural_weights + rule_weights) / 2
        
        # 7. Apply coverage threshold
        coverage_mask = overall_coverage > self.coverage_threshold
        final_weights = final_weights * coverage_mask.float()
        
        # 8. Update statistics
        self.total_count += batch_size
        gated_count = (final_weights < 0.5).sum().item()
        self.gate_count += gated_count
        
        # EMA update for averages
        alpha = 0.01
        self.avg_coverage.mul_(1 - alpha).add_(overall_coverage.mean(), alpha=alpha)
        self.avg_consistency.mul_(1 - alpha).add_(overall_consistency.mean(), alpha=alpha)
        
        # 9. Create conflict analysis
        failure_modes = self._detect_failure_modes(coverage_metrics, consistency_metrics)
        
        # Determine recommended action
        if final_weights.mean() > 0.8:
            recommended_action = "proceed"
        elif final_weights.mean() > 0.3:
            recommended_action = "gate_vision"
        else:
            recommended_action = "fallback_text"
        
        conflict_analysis = ConflictAnalysis(
            conflict_detected=final_weights.mean() < 0.5,
            confidence_level=consistency_level,
            coverage_score=overall_coverage.mean().item(),
            consistency_score=overall_consistency.mean().item(),
            recommended_action=recommended_action,
            failure_modes=failure_modes
        )
        
        if return_analysis:
            conflict_analysis.coverage_metrics = coverage_metrics
            conflict_analysis.consistency_metrics = consistency_metrics
        
        return final_weights, conflict_analysis
    
    def get_statistics(self) -> Dict[str, float]:
        """Get gating statistics."""
        total = max(self.total_count.item(), 1)  # Avoid division by zero
        
        return {
            'gate_rate': self.gate_count.item() / total,
            'total_processed': total,
            'avg_coverage': self.avg_coverage.item(),
            'avg_consistency': self.avg_consistency.item()
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.gate_count.zero_()
        self.total_count.zero_()
        self.avg_coverage.zero_()
        self.avg_consistency.zero_()