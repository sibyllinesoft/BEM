#!/usr/bin/env python3
"""
Vision Feature Precomputation Script for BEM 2.0 Multimodal Conditioning

Precomputes and caches vision features for VQA datasets to amortize
preprocessing costs during training and evaluation.

Usage:
    python bem2/multimodal/precompute.py --encoder models/vision --images data/vqa/images --out data/vqa/vis_feats.parquet
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import logging
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .vision_encoder import VisionEncoder, create_vision_encoder
from .preprocessing import VisionPreprocessor, create_vision_preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Simple dataset for batch image processing."""
    
    def __init__(self, image_paths: List[Path], text_lengths: List[int] = None):
        self.image_paths = image_paths
        self.text_lengths = text_lengths or [None] * len(image_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return {
            'image_path': self.image_paths[idx],
            'text_length': self.text_lengths[idx],
            'image_id': self.image_paths[idx].stem
        }


def collect_vqa_images(image_dir: Path) -> List[Path]:
    """Collect all VQA images from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f"**/*{ext}"))
        image_paths.extend(image_dir.glob(f"**/*{ext.upper()}"))
    
    return sorted(image_paths)


def precompute_vision_features(
    vision_encoder: VisionEncoder,
    preprocessor: VisionPreprocessor,
    image_paths: List[Path],
    output_path: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    max_images: int = None
) -> Dict[str, Any]:
    """
    Precompute vision features for all images.
    
    Args:
        vision_encoder: Vision encoder model
        preprocessor: Vision preprocessor with caching
        image_paths: List of image paths to process
        output_path: Output file path
        batch_size: Batch size for processing
        num_workers: Number of worker threads
        max_images: Maximum number of images to process (for testing)
        
    Returns:
        Statistics about precomputation
    """
    if max_images:
        image_paths = image_paths[:max_images]
    
    logger.info(f"Precomputing features for {len(image_paths)} images")
    logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=lambda x: x  # Return list of dicts
    )
    
    # Results storage
    feature_records = []
    total_processing_time = 0
    cache_hits = 0
    cache_misses = 0
    
    start_time = time.time()
    
    # Process in batches
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Processing images"):
            batch_start_time = time.time()
            
            # Extract batch information
            batch_image_paths = [item['image_path'] for item in batch_data]
            batch_image_ids = [item['image_id'] for item in batch_data]
            batch_text_lengths = [item['text_length'] for item in batch_data]
            
            # Preprocess batch
            try:
                cached_features_list = preprocessor.preprocess_batch(
                    batch_image_paths,
                    text_lengths=batch_text_lengths,
                    batch_size=len(batch_image_paths)
                )
                
                # Store results
                for i, cached_features in enumerate(cached_features_list):
                    record = {
                        'image_id': batch_image_ids[i],
                        'image_path': str(batch_image_paths[i]),
                        'image_hash': cached_features.image_hash,
                        'timestamp': cached_features.timestamp,
                        'preprocessing_time_ms': cached_features.preprocessing_time_ms,
                        'patch_grid_size': cached_features.patch_grid_size,
                        'vision_dim': cached_features.features.vision_dim,
                        'num_patches': cached_features.features.num_patches,
                        'num_regions': cached_features.features.num_regions,
                        'has_chunk_summaries': cached_features.chunk_summaries is not None,
                        'coverage_score': cached_features.features.coverage_score.mean().item() if cached_features.features.coverage_score is not None else None,
                        'consistency_score': cached_features.features.consistency_score.mean().item() if cached_features.features.consistency_score is not None else None
                    }
                    
                    feature_records.append(record)
                
                # Update statistics
                batch_time = time.time() - batch_start_time
                total_processing_time += batch_time
                
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                continue
    
    # Get preprocessing statistics
    preprocessing_stats = preprocessor.get_preprocessing_stats()
    cache_hits = preprocessing_stats.get('cache_hits', 0)
    total_processed = preprocessing_stats.get('total_processed', len(image_paths))
    cache_misses = total_processed - cache_hits
    
    # Create dataframe and save
    if feature_records:
        df = pd.DataFrame(feature_records)
        
        # Save to parquet for efficient storage
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        logger.info(f"✅ Saved {len(feature_records)} feature records to {output_path}")
    else:
        logger.error("❌ No features were successfully processed")
    
    # Compute final statistics
    total_time = time.time() - start_time
    
    stats = {
        'total_images': len(image_paths),
        'successfully_processed': len(feature_records),
        'total_time_seconds': total_time,
        'avg_time_per_image_ms': (total_time * 1000) / len(image_paths) if image_paths else 0,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'cache_hit_rate': cache_hits / max(total_processed, 1),
        'batch_size': batch_size,
        'preprocessing_stats': preprocessing_stats
    }
    
    # Save statistics
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Precompute vision features for VQA")
    parser.add_argument("--encoder", required=True, type=Path,
                       help="Path to vision encoder model")
    parser.add_argument("--images", required=True, type=Path,
                       help="Directory containing VQA images")
    parser.add_argument("--out", required=True, type=Path,
                       help="Output parquet file for features")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of worker threads")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process (for testing)")
    parser.add_argument("--cache-dir", type=Path, default="cache/vision_features",
                       help="Directory for feature caching")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear existing cache before processing")
    
    args = parser.parse_args()
    
    print("👁️  BEM 2.0 Vision Feature Precomputation")
    print("=" * 50)
    
    # Validate inputs
    if not args.encoder.exists():
        print(f"❌ Vision encoder not found: {args.encoder}")
        return 1
    
    if not args.images.exists():
        print(f"❌ Image directory not found: {args.images}")
        return 1
    
    # Collect images
    print("📁 Collecting VQA images...")
    image_paths = collect_vqa_images(args.images)
    
    if not image_paths:
        print(f"❌ No images found in {args.images}")
        return 1
    
    print(f"   Found {len(image_paths)} images")
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
        print(f"   Limited to {len(image_paths)} images for testing")
    
    # Create vision encoder
    print("🔧 Loading vision encoder...")
    try:
        vision_encoder = create_vision_encoder(str(args.encoder))
        print(f"   ✅ Loaded encoder: {vision_encoder.metadata.get('encoder_name', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load vision encoder: {e}")
        return 1
    
    # Create preprocessor with caching
    print("⚙️  Setting up preprocessor...")
    preprocessor_config = {
        'cache_dir': str(args.cache_dir),
        'enable_caching': True,
        'cache_config': {
            'max_memory_items': 1000,
            'max_disk_items': 10000,
            'cache_ttl_hours': 24
        }
    }
    
    preprocessor = create_vision_preprocessor(vision_encoder, preprocessor_config)
    
    # Clear cache if requested
    if args.clear_cache:
        print("🗑️  Clearing existing cache...")
        preprocessor.clear_cache()
    
    print(f"   Cache directory: {args.cache_dir}")
    
    # Precompute features
    print("🚀 Starting feature precomputation...")
    
    try:
        stats = precompute_vision_features(
            vision_encoder=vision_encoder,
            preprocessor=preprocessor,
            image_paths=image_paths,
            output_path=args.out,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_images
        )
        
        # Print summary
        print("\n📊 Precomputation Summary:")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Successfully processed: {stats['successfully_processed']}")
        print(f"   Total time: {stats['total_time_seconds']:.1f}s")
        print(f"   Avg time per image: {stats['avg_time_per_image_ms']:.1f}ms")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
        
        print(f"\n💾 Output saved to: {args.out}")
        print(f"   Statistics saved to: {args.out.parent / f'{args.out.stem}_stats.json'}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Precomputation failed: {e}")
        logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    exit(main())