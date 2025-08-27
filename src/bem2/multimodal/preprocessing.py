"""
Vision Preprocessing for BEM 2.0 Multimodal Conditioning

Handles efficient preprocessing and caching of vision features to amortize
preprocessing costs. Implements chunk-aligned feature caching and per-chunk
summaries as specified in the TODO.md requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import pickle
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import logging

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .vision_encoder import VisionEncoder, VisionFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CachedVisionFeatures:
    """Cached vision features with metadata."""
    features: VisionFeatures
    image_hash: str
    timestamp: float
    patch_grid_size: Tuple[int, int]
    chunk_summaries: Optional[Dict[int, torch.Tensor]] = None  # chunk_id -> summary
    preprocessing_time_ms: float = 0.0


class ImageHasher:
    """Efficient image hashing for cache keys."""
    
    @staticmethod
    def hash_image(image) -> str:
        """Compute hash of image content."""
        if isinstance(image, str):
            # Image path
            with open(image, 'rb') as f:
                content = f.read()
        elif hasattr(image, 'tobytes'):
            # PIL Image
            content = image.tobytes()
        elif isinstance(image, torch.Tensor):
            # Tensor
            content = image.cpu().numpy().tobytes()
        elif isinstance(image, np.ndarray):
            # Numpy array
            content = image.tobytes()
        else:
            # Fallback: convert to string
            content = str(image).encode()
        
        return hashlib.md5(content).hexdigest()
    
    @staticmethod
    def hash_batch(images: List) -> List[str]:
        """Hash a batch of images efficiently."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            hashes = list(executor.map(ImageHasher.hash_image, images))
        return hashes


class VisionCache:
    """
    LRU cache for vision features with disk persistence.
    Implements efficient caching to amortize preprocessing costs.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache/vision_features",
        max_memory_items: int = 1000,
        max_disk_items: int = 10000,
        cache_ttl_hours: int = 24
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_items = max_memory_items
        self.max_disk_items = max_disk_items
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        
        # In-memory cache (LRU)
        self.memory_cache: Dict[str, CachedVisionFeatures] = {}
        self.access_order: List[str] = []
        
        # Statistics
        self.cache_stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Load cache index
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load disk cache index."""
        index_path = self.cache_dir / "cache_index.json"
        self.disk_index = {}
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self.disk_index = json.load(f)
                # Clean up expired entries
                self._cleanup_expired()
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.disk_index = {}
    
    def _save_cache_index(self):
        """Save disk cache index."""
        index_path = self.cache_dir / "cache_index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump(self.disk_index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries from disk cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.disk_index.items():
            if current_time - entry['timestamp'] > self.cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_disk_entry(key)
    
    def _remove_disk_entry(self, key: str):
        """Remove entry from disk cache."""
        if key in self.disk_index:
            filepath = self.cache_dir / f"{key}.pkl"
            if filepath.exists():
                filepath.unlink()
            del self.disk_index[key]
    
    def _evict_memory_lru(self):
        """Evict least recently used item from memory."""
        if len(self.memory_cache) >= self.max_memory_items:
            lru_key = self.access_order.pop(0)
            if lru_key in self.memory_cache:
                del self.memory_cache[lru_key]
                self.cache_stats['evictions'] += 1
    
    def _update_access_order(self, key: str):
        """Update access order for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get(self, image_hash: str) -> Optional[CachedVisionFeatures]:
        """Get cached vision features."""
        self.cache_stats['total_requests'] += 1
        
        # Check memory cache first
        if image_hash in self.memory_cache:
            self._update_access_order(image_hash)
            self.cache_stats['memory_hits'] += 1
            return self.memory_cache[image_hash]
        
        # Check disk cache
        if image_hash in self.disk_index:
            entry_info = self.disk_index[image_hash]
            
            # Check if expired
            if time.time() - entry_info['timestamp'] > self.cache_ttl_seconds:
                self._remove_disk_entry(image_hash)
                self.cache_stats['misses'] += 1
                return None
            
            # Load from disk
            filepath = self.cache_dir / f"{image_hash}.pkl"
            try:
                with open(filepath, 'rb') as f:
                    cached_features = pickle.load(f)
                
                # Add to memory cache
                self._evict_memory_lru()
                self.memory_cache[image_hash] = cached_features
                self._update_access_order(image_hash)
                
                self.cache_stats['disk_hits'] += 1
                return cached_features
                
            except Exception as e:
                logger.warning(f"Failed to load cached features for {image_hash}: {e}")
                self._remove_disk_entry(image_hash)
        
        # Cache miss
        self.cache_stats['misses'] += 1
        return None
    
    def put(self, image_hash: str, features: CachedVisionFeatures):
        """Cache vision features."""
        # Add to memory cache
        self._evict_memory_lru()
        self.memory_cache[image_hash] = features
        self._update_access_order(image_hash)
        
        # Save to disk cache
        if len(self.disk_index) >= self.max_disk_items:
            # Remove oldest disk entry
            oldest_key = min(self.disk_index.keys(), 
                           key=lambda k: self.disk_index[k]['timestamp'])
            self._remove_disk_entry(oldest_key)
        
        # Save to disk
        filepath = self.cache_dir / f"{image_hash}.pkl"
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(features, f)
            
            self.disk_index[image_hash] = {
                'timestamp': features.timestamp,
                'filepath': str(filepath)
            }
            self._save_cache_index()
            
        except Exception as e:
            logger.warning(f"Failed to cache features for {image_hash}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache_stats.copy()
        stats.update({
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_size': len(self.disk_index),
            'hit_rate': (stats['memory_hits'] + stats['disk_hits']) / max(stats['total_requests'], 1),
            'memory_hit_rate': stats['memory_hits'] / max(stats['total_requests'], 1),
            'disk_hit_rate': stats['disk_hits'] / max(stats['total_requests'], 1)
        })
        return stats
    
    def clear(self):
        """Clear all cached data."""
        self.memory_cache.clear()
        self.access_order.clear()
        
        # Remove disk files
        for key in list(self.disk_index.keys()):
            self._remove_disk_entry(key)
        
        self._save_cache_index()
        
        # Reset stats
        for key in self.cache_stats:
            self.cache_stats[key] = 0


class ChunkAlignedProcessor:
    """
    Processes vision features aligned with text chunk windows.
    Creates per-chunk summaries to optimize routing decisions.
    """
    
    def __init__(
        self,
        chunk_size: int = 32,
        patch_grid_size: Tuple[int, int] = (14, 14),
        summary_method: str = "attention_weighted"  # "mean", "max", "attention_weighted"
    ):
        self.chunk_size = chunk_size
        self.patch_grid_size = patch_grid_size
        self.summary_method = summary_method
        
        # Initialize attention weights for spatial-temporal alignment
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=512,  # Assume 512-dim features
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def align_patches_to_chunks(
        self,
        vision_features: VisionFeatures,
        text_length: int,
        num_chunks: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Align vision patches to text chunks for cache-safe integration.
        
        Args:
            vision_features: Vision features with patch embeddings
            text_length: Length of text sequence
            num_chunks: Number of text chunks (computed if not provided)
            
        Returns:
            Dictionary mapping chunk_id to chunk-aligned features
        """
        if num_chunks is None:
            num_chunks = (text_length + self.chunk_size - 1) // self.chunk_size
        
        patch_embeddings = vision_features.patch_embeddings  # [batch, num_patches, dim]
        batch_size, num_patches, feature_dim = patch_embeddings.shape
        
        chunk_aligned_features = {}
        
        for chunk_id in range(num_chunks):
            # Compute chunk progress (0.0 to 1.0)
            chunk_start = chunk_id * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, text_length)
            chunk_progress = (chunk_start + chunk_end) / (2 * text_length)
            
            # Map chunk progress to spatial attention over patches
            chunk_features = self._compute_chunk_attention(
                patch_embeddings, chunk_progress
            )
            
            chunk_aligned_features[chunk_id] = chunk_features
        
        return chunk_aligned_features
    
    def _compute_chunk_attention(
        self,
        patch_embeddings: torch.Tensor,  # [batch, num_patches, dim]
        chunk_progress: float
    ) -> torch.Tensor:
        """Compute chunk-specific attention over patches."""
        batch_size, num_patches, feature_dim = patch_embeddings.shape
        h, w = self.patch_grid_size
        
        # Create spatial bias based on reading order and chunk progress
        spatial_positions = self._get_spatial_positions()  # [num_patches, 2]
        
        # Reading order bias: top-left to bottom-right
        reading_order_bias = (spatial_positions[:, 0] * 0.7 + 
                             spatial_positions[:, 1] * 0.3)  # [num_patches]
        
        # Compute attention weights based on chunk progress
        target_position = chunk_progress
        spatial_distances = torch.abs(reading_order_bias - target_position)
        spatial_weights = torch.exp(-spatial_distances * 3.0)  # Sharpness
        spatial_weights = spatial_weights / spatial_weights.sum()
        
        # Apply spatial attention
        spatial_weights = spatial_weights.unsqueeze(0).unsqueeze(-1)  # [1, num_patches, 1]
        chunk_features = (patch_embeddings * spatial_weights).sum(dim=1)  # [batch, dim]
        
        return chunk_features
    
    def _get_spatial_positions(self) -> torch.Tensor:
        """Get normalized spatial positions for patches."""
        h, w = self.patch_grid_size
        num_patches = h * w
        
        positions = torch.zeros(num_patches, 2)
        for i in range(num_patches):
            row = i // w
            col = i % w
            positions[i, 0] = row / (h - 1)  # Normalized row
            positions[i, 1] = col / (w - 1)  # Normalized col
        
        return positions
    
    def create_chunk_summaries(
        self,
        vision_features: VisionFeatures,
        text_length: int
    ) -> Dict[int, torch.Tensor]:
        """
        Create per-chunk vision summaries for efficient caching.
        
        Args:
            vision_features: Vision features
            text_length: Length of text sequence
            
        Returns:
            Dictionary of chunk summaries
        """
        # Get chunk-aligned features
        chunk_features = self.align_patches_to_chunks(vision_features, text_length)
        
        summaries = {}
        for chunk_id, features in chunk_features.items():
            if self.summary_method == "mean":
                summary = features.mean(dim=0, keepdim=True)
            elif self.summary_method == "max":
                summary = features.max(dim=0, keepdim=True)[0]
            elif self.summary_method == "attention_weighted":
                # Use region summaries as context for attention
                if features.dim() == 3:  # [batch, seq, dim]
                    # Apply self-attention for summary
                    summary, _ = self.spatial_attention(features, features, features)
                    summary = summary.mean(dim=1, keepdim=True)  # [batch, 1, dim]
                else:  # [batch, dim]
                    summary = features.unsqueeze(1)  # [batch, 1, dim]
            else:
                # Default to mean
                summary = features.mean(dim=0, keepdim=True)
            
            summaries[chunk_id] = summary
        
        return summaries


class VisionPreprocessor:
    """
    Main vision preprocessing pipeline with caching and chunk alignment.
    Implements the complete MM0 preprocessing stack.
    """
    
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        cache_dir: str = "cache/vision_features",
        enable_caching: bool = True,
        cache_config: Optional[Dict[str, Any]] = None,
        chunk_config: Optional[Dict[str, Any]] = None
    ):
        self.vision_encoder = vision_encoder
        self.enable_caching = enable_caching
        
        # Initialize cache
        if enable_caching:
            cache_config = cache_config or {}
            self.cache = VisionCache(cache_dir=cache_dir, **cache_config)
        
        # Initialize chunk processor
        chunk_config = chunk_config or {}
        self.chunk_processor = ChunkAlignedProcessor(**chunk_config)
        
        # Image hasher
        self.hasher = ImageHasher()
        
        # Statistics
        self.preprocessing_stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'preprocessing_time_ms': 0.0,
            'avg_preprocessing_time_ms': 0.0
        }
    
    def preprocess_image(
        self,
        image,
        text_length: Optional[int] = None,
        force_recompute: bool = False
    ) -> CachedVisionFeatures:
        """
        Preprocess a single image with caching.
        
        Args:
            image: Input image
            text_length: Length of associated text (for chunk alignment)
            force_recompute: Whether to force recomputation
            
        Returns:
            Cached vision features
        """
        start_time = time.time()
        
        # Compute image hash
        image_hash = self.hasher.hash_image(image)
        
        # Check cache if enabled
        if self.enable_caching and not force_recompute:
            cached_features = self.cache.get(image_hash)
            if cached_features is not None:
                self.preprocessing_stats['cache_hits'] += 1
                self.preprocessing_stats['total_processed'] += 1
                return cached_features
        
        # Preprocess image
        if isinstance(image, str) and HAS_PIL:
            # Load image from path
            pil_image = Image.open(image).convert('RGB')
            pixel_values = self.vision_encoder.preprocess_images([pil_image])
        elif hasattr(image, 'size') and HAS_PIL:  # PIL Image
            pixel_values = self.vision_encoder.preprocess_images([image])
        else:
            # Assume preprocessed tensor
            pixel_values = image.unsqueeze(0) if image.dim() == 3 else image
        
        # Extract vision features
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values, return_attention=True)
        
        # Create chunk summaries if text length provided
        chunk_summaries = None
        if text_length is not None:
            chunk_summaries = self.chunk_processor.create_chunk_summaries(
                vision_features, text_length
            )
        
        # Create cached features
        preprocessing_time = (time.time() - start_time) * 1000
        cached_features = CachedVisionFeatures(
            features=vision_features,
            image_hash=image_hash,
            timestamp=time.time(),
            patch_grid_size=self.vision_encoder.get_patch_grid_size(),
            chunk_summaries=chunk_summaries,
            preprocessing_time_ms=preprocessing_time
        )
        
        # Cache if enabled
        if self.enable_caching:
            self.cache.put(image_hash, cached_features)
        
        # Update statistics
        self.preprocessing_stats['total_processed'] += 1
        self.preprocessing_stats['preprocessing_time_ms'] += preprocessing_time
        self.preprocessing_stats['avg_preprocessing_time_ms'] = (
            self.preprocessing_stats['preprocessing_time_ms'] / 
            self.preprocessing_stats['total_processed']
        )
        
        return cached_features
    
    def preprocess_batch(
        self,
        images: List,
        text_lengths: Optional[List[int]] = None,
        batch_size: int = 8,
        force_recompute: bool = False
    ) -> List[CachedVisionFeatures]:
        """
        Preprocess a batch of images efficiently.
        
        Args:
            images: List of images
            text_lengths: List of text lengths for chunk alignment
            batch_size: Processing batch size
            force_recompute: Whether to force recomputation
            
        Returns:
            List of cached vision features
        """
        if text_lengths is None:
            text_lengths = [None] * len(images)
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_text_lengths = text_lengths[i:i + batch_size]
            
            batch_results = []
            for image, text_length in zip(batch_images, batch_text_lengths):
                features = self.preprocess_image(
                    image, text_length, force_recompute
                )
                batch_results.append(features)
            
            results.extend(batch_results)
        
        return results
    
    def get_chunk_features(
        self,
        image_hash: str,
        chunk_id: int
    ) -> Optional[torch.Tensor]:
        """
        Get cached chunk features for efficient routing.
        
        Args:
            image_hash: Hash of source image
            chunk_id: Chunk identifier
            
        Returns:
            Chunk features if cached, None otherwise
        """
        if not self.enable_caching:
            return None
        
        cached_features = self.cache.get(image_hash)
        if cached_features is None or cached_features.chunk_summaries is None:
            return None
        
        return cached_features.chunk_summaries.get(chunk_id)
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        stats = self.preprocessing_stats.copy()
        
        if self.enable_caching:
            cache_stats = self.cache.get_stats()
            stats.update({
                'cache_hit_rate': cache_stats['hit_rate'],
                'memory_cache_size': cache_stats['memory_cache_size'],
                'disk_cache_size': cache_stats['disk_cache_size']
            })
        
        return stats
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.enable_caching:
            self.cache.clear()
        
        # Reset stats
        for key in self.preprocessing_stats:
            if key.endswith('_ms'):
                self.preprocessing_stats[key] = 0.0
            else:
                self.preprocessing_stats[key] = 0


def create_vision_preprocessor(
    vision_encoder: VisionEncoder,
    config: Optional[Dict[str, Any]] = None
) -> VisionPreprocessor:
    """
    Factory function to create vision preprocessor.
    
    Args:
        vision_encoder: Vision encoder instance
        config: Optional configuration
        
    Returns:
        VisionPreprocessor instance
    """
    if config is None:
        config = {}
    
    defaults = {
        'cache_dir': 'cache/vision_features',
        'enable_caching': True,
        'cache_config': {
            'max_memory_items': 1000,
            'max_disk_items': 10000,
            'cache_ttl_hours': 24
        },
        'chunk_config': {
            'chunk_size': 32,
            'patch_grid_size': vision_encoder.get_patch_grid_size(),
            'summary_method': 'attention_weighted'
        }
    }
    
    # Update with provided config
    for key in ['cache_config', 'chunk_config']:
        if key in config and isinstance(config[key], dict):
            defaults[key].update(config[key])
        elif key in config:
            defaults[key] = config[key]
    
    defaults.update({k: v for k, v in config.items() 
                    if k not in ['cache_config', 'chunk_config']})
    
    return VisionPreprocessor(vision_encoder, **defaults)