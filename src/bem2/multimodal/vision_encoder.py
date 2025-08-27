"""
Vision Encoder Module for BEM 2.0 Multimodal Conditioning

Implements CLIP-based vision encoding with region-level processing,
CLS/pool embeddings extraction, and coverage analysis features.

Cache-Safety: Vision features are extracted separately and fed only to
the controller, never to generator sites.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, using mock vision encoder")


@dataclass
class VisionFeatures:
    """Container for extracted vision features."""
    cls_embedding: torch.Tensor  # [batch, vision_dim] - CLS token
    pool_embedding: torch.Tensor  # [batch, vision_dim] - pooled features
    patch_embeddings: torch.Tensor  # [batch, num_patches, vision_dim] - patch-level
    region_summaries: torch.Tensor  # [batch, num_regions, vision_dim] - region-level
    spatial_attention: Optional[torch.Tensor] = None  # [batch, num_patches] - attention weights
    coverage_score: Optional[torch.Tensor] = None  # [batch] - coverage metric
    consistency_score: Optional[torch.Tensor] = None  # [batch] - consistency metric
    
    @property
    def batch_size(self) -> int:
        return self.cls_embedding.shape[0]
    
    @property  
    def vision_dim(self) -> int:
        return self.cls_embedding.shape[1]
    
    @property
    def num_patches(self) -> int:
        return self.patch_embeddings.shape[1]
        
    @property
    def num_regions(self) -> int:
        return self.region_summaries.shape[1]


class VisionEncoder(nn.Module):
    """
    Vision encoder based on CLIP with region-level processing.
    
    Extracts:
    - CLS embeddings (global image representation)
    - Pool embeddings (average pooled patches)
    - Patch embeddings (aligned with ViT patch windows)
    - Region summaries (spatially grouped patches)
    """
    
    def __init__(
        self,
        model_path: str = "models/vision",
        vision_dim: int = 512,
        num_regions: int = 8,  # For region summaries
        enable_coverage_analysis: bool = True,
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        self.model_path = Path(model_path)
        self.vision_dim = vision_dim
        self.num_regions = num_regions
        self.enable_coverage_analysis = enable_coverage_analysis
        
        # Load vision encoder metadata
        self.metadata = self._load_metadata()
        
        # Initialize CLIP model
        self.clip_model = self._load_clip_model()
        self.processor = self._load_processor()
        
        # Region pooling for spatial summarization
        # Creates region summaries from patch embeddings
        self.region_pooler = nn.AdaptiveAvgPool2d((num_regions, 1))
        
        # Spatial attention for coverage analysis
        if enable_coverage_analysis:
            self.spatial_attention = nn.Sequential(
                nn.Linear(vision_dim, vision_dim // 4),
                nn.ReLU(),
                nn.Linear(vision_dim // 4, 1),
                nn.Sigmoid()
            )
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(vision_dim)
        
        # Freeze CLIP parameters (only train additional layers)
        self._freeze_clip_parameters()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load vision encoder metadata."""
        metadata_path = self.model_path / "encoder_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        else:
            # Default metadata for mock setup
            return {
                "encoder_name": "clip-vit-base-patch32",
                "vision_dim": self.vision_dim,
                "input_resolution": 224,
                "status": "mock"
            }
    
    def _load_clip_model(self) -> nn.Module:
        """Load CLIP vision model."""
        if not HAS_TRANSFORMERS or self.metadata.get("status") == "mock":
            # Mock vision model for development
            return self._create_mock_clip_model()
        
        try:
            model = CLIPModel.from_pretrained(self.model_path)
            return model.vision_model
        except Exception as e:
            print(f"Warning: Failed to load CLIP model, using mock: {e}")
            return self._create_mock_clip_model()
    
    def _create_mock_clip_model(self) -> nn.Module:
        """Create mock CLIP model for development."""
        class MockCLIPVision(nn.Module):
            def __init__(self, vision_dim: int, num_patches: int = 196):
                super().__init__()
                self.vision_dim = vision_dim
                self.num_patches = num_patches
                
                # Mock embeddings
                self.patch_embedding = nn.Conv2d(3, vision_dim, kernel_size=32, stride=32)
                self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, vision_dim))
                self.class_embedding = nn.Parameter(torch.randn(1, 1, vision_dim))
                
                # Mock transformer layers (simplified)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(vision_dim, 8, dim_feedforward=vision_dim*4),
                    num_layers=6
                )
                
            def forward(self, pixel_values):
                batch_size = pixel_values.shape[0]
                
                # Mock patch embedding
                patches = self.patch_embedding(pixel_values)  # [B, vision_dim, H, W]
                patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, vision_dim]
                
                # Add class token
                class_tokens = self.class_embedding.expand(batch_size, -1, -1)
                x = torch.cat([class_tokens, patches], dim=1)
                
                # Add position embeddings
                x = x + self.position_embedding[:, :x.shape[1]]
                
                # Transformer forward
                x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
                
                # Return in CLIP format
                return type('CLIPVisionOutput', (), {
                    'last_hidden_state': x,
                    'pooler_output': x[:, 0]  # CLS token
                })()
        
        return MockCLIPVision(self.vision_dim)
    
    def _load_processor(self):
        """Load CLIP processor."""
        if not HAS_TRANSFORMERS or self.metadata.get("status") == "mock":
            return None
        
        try:
            return CLIPProcessor.from_pretrained(self.model_path)
        except:
            return None
    
    def _freeze_clip_parameters(self):
        """Freeze CLIP parameters to avoid disrupting pretrained features."""
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def _compute_region_summaries(
        self, 
        patch_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute region summaries from patch embeddings.
        
        Args:
            patch_embeddings: Patch embeddings [batch, num_patches, vision_dim]
            
        Returns:
            region_summaries: Region summaries [batch, num_regions, vision_dim]
        """
        batch_size, num_patches, vision_dim = patch_embeddings.shape
        
        # Reshape for spatial pooling (assuming square patch grid)
        patch_size = int(np.sqrt(num_patches))
        if patch_size * patch_size != num_patches:
            # Handle non-square case with padding
            patch_size = int(np.sqrt(num_patches)) + 1
            padded_patches = F.pad(
                patch_embeddings, 
                (0, 0, 0, patch_size * patch_size - num_patches)
            )
        else:
            padded_patches = patch_embeddings
        
        # Reshape to spatial grid
        spatial_patches = padded_patches.view(
            batch_size, patch_size, patch_size, vision_dim
        ).permute(0, 3, 1, 2)  # [batch, vision_dim, H, W]
        
        # Apply region pooling
        region_features = self.region_pooler(spatial_patches)  # [batch, vision_dim, num_regions, 1]
        region_features = region_features.squeeze(-1).transpose(1, 2)  # [batch, num_regions, vision_dim]
        
        return region_features
    
    def _compute_coverage_score(
        self, 
        patch_embeddings: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute coverage score based on patch activation distribution.
        
        Args:
            patch_embeddings: Patch embeddings [batch, num_patches, vision_dim]
            attention_weights: Optional attention weights [batch, num_patches]
            
        Returns:
            coverage_score: Coverage score [batch]
        """
        # Compute patch activation magnitudes
        patch_norms = patch_embeddings.norm(dim=-1)  # [batch, num_patches]
        
        # Use attention weights if available, otherwise uniform
        if attention_weights is not None:
            weights = attention_weights
        else:
            weights = torch.ones_like(patch_norms) / patch_norms.shape[1]
        
        # Compute weighted entropy as coverage measure
        # Higher entropy = more uniform coverage
        normalized_weights = F.softmax(weights, dim=-1)
        entropy = -(normalized_weights * torch.log(normalized_weights + 1e-8)).sum(dim=-1)
        max_entropy = np.log(patch_norms.shape[1])  # Maximum possible entropy
        
        coverage_score = entropy / max_entropy  # Normalize to [0, 1]
        return coverage_score
    
    def _compute_consistency_score(
        self,
        cls_embedding: torch.Tensor,
        region_summaries: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency score between global and regional features.
        
        Args:
            cls_embedding: Global CLS embedding [batch, vision_dim]
            region_summaries: Region summaries [batch, num_regions, vision_dim]
            
        Returns:
            consistency_score: Consistency score [batch]
        """
        # Expand CLS for comparison
        cls_expanded = cls_embedding.unsqueeze(1).expand(-1, self.num_regions, -1)
        
        # Compute cosine similarity between CLS and each region
        similarities = F.cosine_similarity(cls_expanded, region_summaries, dim=-1)
        
        # Consistency is the mean similarity
        consistency_score = similarities.mean(dim=-1)
        
        return consistency_score
    
    def preprocess_images(self, images: List) -> torch.Tensor:
        """
        Preprocess images for CLIP input.
        
        Args:
            images: List of PIL images or image arrays
            
        Returns:
            pixel_values: Preprocessed pixel values [batch, 3, H, W]
        """
        if self.processor is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            return inputs['pixel_values']
        else:
            # Mock preprocessing for development
            batch_size = len(images)
            resolution = self.metadata.get("input_resolution", 224)
            return torch.randn(batch_size, 3, resolution, resolution)
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        return_attention: bool = False
    ) -> VisionFeatures:
        """
        Extract vision features from images.
        
        Args:
            pixel_values: Input pixel values [batch, 3, H, W]
            return_attention: Whether to compute spatial attention
            
        Returns:
            VisionFeatures: Extracted vision features
        """
        batch_size = pixel_values.shape[0]
        
        # Forward through CLIP vision model
        with torch.no_grad():  # CLIP is frozen
            vision_outputs = self.clip_model(pixel_values)
        
        # Extract features
        hidden_states = vision_outputs.last_hidden_state  # [batch, seq_len, vision_dim]
        cls_embedding = hidden_states[:, 0]  # First token is CLS
        patch_embeddings = hidden_states[:, 1:]  # Rest are patches
        
        # Normalize features
        cls_embedding = self.feature_norm(cls_embedding)
        patch_embeddings = self.feature_norm(patch_embeddings)
        
        # Compute pool embedding (alternative to CLS)
        pool_embedding = patch_embeddings.mean(dim=1)
        pool_embedding = self.feature_norm(pool_embedding)
        
        # Compute region summaries
        region_summaries = self._compute_region_summaries(patch_embeddings)
        region_summaries = self.feature_norm(region_summaries)
        
        # Compute spatial attention if enabled
        spatial_attention = None
        if self.enable_coverage_analysis and return_attention:
            attention_logits = self.spatial_attention(patch_embeddings).squeeze(-1)  # [batch, num_patches]
            spatial_attention = F.softmax(attention_logits, dim=-1)
        
        # Compute coverage and consistency scores
        coverage_score = None
        consistency_score = None
        if self.enable_coverage_analysis:
            coverage_score = self._compute_coverage_score(patch_embeddings, spatial_attention)
            consistency_score = self._compute_consistency_score(cls_embedding, region_summaries)
        
        return VisionFeatures(
            cls_embedding=cls_embedding,
            pool_embedding=pool_embedding,
            patch_embeddings=patch_embeddings,
            region_summaries=region_summaries,
            spatial_attention=spatial_attention,
            coverage_score=coverage_score,
            consistency_score=consistency_score
        )
    
    def get_patch_grid_size(self) -> Tuple[int, int]:
        """Get patch grid dimensions."""
        # For ViT-B/32, typically 7x7 = 49 patches for 224x224 input
        # For ViT-B/16, typically 14x14 = 196 patches for 224x224 input
        model_name = self.metadata.get("encoder_name", "")
        if "patch32" in model_name:
            return (7, 7)
        elif "patch16" in model_name:
            return (14, 14)
        else:
            # Default assumption
            return (14, 14)
    
    def extract_features_batch(
        self, 
        images: List, 
        batch_size: int = 8
    ) -> List[VisionFeatures]:
        """
        Extract features for a large batch of images with batching.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            List of VisionFeatures
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            pixel_values = self.preprocess_images(batch_images)
            
            # Move to device if using GPU
            if next(self.parameters()).is_cuda:
                pixel_values = pixel_values.cuda()
            
            features = self.forward(pixel_values, return_attention=True)
            all_features.append(features)
        
        return all_features


def create_vision_encoder(
    model_path: str = "models/vision",
    config: Optional[Dict[str, Any]] = None
) -> VisionEncoder:
    """
    Factory function to create vision encoder.
    
    Args:
        model_path: Path to vision model
        config: Optional configuration dict
        
    Returns:
        VisionEncoder instance
    """
    if config is None:
        config = {}
    
    defaults = {
        "vision_dim": 512,
        "num_regions": 8,
        "enable_coverage_analysis": True
    }
    defaults.update(config)
    
    return VisionEncoder(model_path=model_path, **defaults)