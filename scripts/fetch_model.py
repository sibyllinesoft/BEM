#!/usr/bin/env python3
"""
Model Fetching Script for BEM Real Runs Campaign

Downloads and caches base models with SHA logging for reproducibility.
Supports TinyLlama-1.1B, Qwen2-1.5B-Instruct, and other base models.

Usage:
    python scripts/fetch_model.py --name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out models/tinyllama
    python scripts/fetch_model.py --name Qwen/Qwen2-1.5B-Instruct --out models/qwen2-1_5b
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig
)

# Approved model configurations for TODO.md requirements
APPROVED_MODELS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "size": "1.1B",
        "context_length": 2048,
        "license": "Apache-2.0",
        "description": "TinyLlama 1.1B parameter chat model"
    },
    "Qwen/Qwen2-1.5B-Instruct": {
        "size": "1.5B", 
        "context_length": 32768,
        "license": "Apache-2.0",
        "description": "Qwen2 1.5B parameter instruct model"
    },
    "microsoft/DialoGPT-small": {
        "size": "117M",
        "context_length": 1024, 
        "license": "MIT",
        "description": "Small DialoGPT for quick testing"
    },
}

def compute_model_hash(model_path: Path) -> str:
    """Compute hash of model files for reproducibility."""
    model_files = []
    
    # Collect all model files
    for ext in ["*.bin", "*.safetensors", "*.json", "*.txt"]:
        model_files.extend(model_path.glob(ext))
    
    # Sort for consistent hashing
    model_files.sort(key=lambda x: x.name)
    
    hasher = hashlib.sha256()
    for file_path in model_files:
        if file_path.is_file() and file_path.stat().st_size > 0:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
    
    return hasher.hexdigest()

def download_model(model_name: str, output_dir: Path, 
                  force_download: bool = False) -> Dict[str, Any]:
    """Download and cache model with metadata logging."""
    
    if model_name not in APPROVED_MODELS:
        raise ValueError(f"Model {model_name} not in approved list. "
                        f"Available: {list(APPROVED_MODELS.keys())}")
    
    print(f"📥 Downloading model: {model_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if not force_download and (output_dir / "config.json").exists():
        print("✅ Model already exists. Use --force to re-download.")
        model_hash = compute_model_hash(output_dir)
        
        # Load existing metadata if available
        metadata_file = output_dir / "download_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            metadata["model_hash"] = model_hash  # Update hash
            metadata["status"] = "cached"
            return metadata
        else:
            # Create basic metadata for existing model
            return {
                "model_name": model_name,
                "model_info": APPROVED_MODELS[model_name],
                "output_dir": str(output_dir),
                "model_hash": model_hash,
                "parameter_count": 0,  # Unknown for cached
                "status": "cached"
            }
    
    try:
        # Download tokenizer
        print("📚 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(output_dir / "tokenizer_cache")
        )
        tokenizer.save_pretrained(output_dir)
        
        # Download model
        print("🧠 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Save space
            cache_dir=str(output_dir / "model_cache")
        )
        model.save_pretrained(output_dir)
        
        # Download config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.save_pretrained(output_dir)
        
        # Compute hash for reproducibility
        model_hash = compute_model_hash(output_dir)
        
        # Create metadata
        metadata = {
            "model_name": model_name,
            "model_info": APPROVED_MODELS[model_name],
            "output_dir": str(output_dir),
            "model_hash": model_hash,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "download_timestamp": None,  # Will be set by caller
            "file_sizes": {},
            "parameter_count": getattr(model.config, 'num_parameters', None) or 
                             sum(p.numel() for p in model.parameters()),
            "status": "downloaded"
        }
        
        # Record file sizes
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(output_dir)
                metadata["file_sizes"][str(relative_path)] = file_path.stat().st_size
        
        # Save metadata
        with open(output_dir / "download_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Download complete!")
        print(f"📊 Model hash: {model_hash[:16]}...")
        print(f"📈 Parameters: {metadata['parameter_count']:,}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        # Clean up partial download
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise

def update_base_lock(metadata: Dict[str, Any], lock_file: Path):
    """Update the base model lock file with new model metadata."""
    
    # Load existing lock or create new
    if lock_file.exists():
        with open(lock_file, 'r') as f:
            lock_data = json.load(f)
        
        # Handle legacy single-model format
        if "model_name" in lock_data and "models" not in lock_data:
            # Convert legacy format to new format
            old_model = lock_data.copy()
            lock_data = {
                "models": {
                    old_model["model_name"]: {
                        "hash": "legacy",
                        "output_dir": "unknown",
                        "parameter_count": old_model.get("parameters", "unknown"),
                        "transformers_version": "unknown",
                        "download_timestamp": None
                    }
                },
                "last_updated": None
            }
    else:
        lock_data = {"models": {}, "last_updated": None}
    
    # Ensure models key exists
    if "models" not in lock_data:
        lock_data["models"] = {}
    
    # Update with new model
    model_name = metadata["model_name"]
    lock_data["models"][model_name] = {
        "hash": metadata["model_hash"],
        "output_dir": metadata["output_dir"], 
        "parameter_count": metadata.get("parameter_count", 0),
        "transformers_version": metadata.get("transformers_version", "unknown"),
        "download_timestamp": metadata.get("download_timestamp")
    }
    
    import time
    lock_data["last_updated"] = time.time()
    
    # Ensure directory exists
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save updated lock
    with open(lock_file, 'w') as f:
        json.dump(lock_data, f, indent=2)
    
    print(f"🔒 Updated base lock: {lock_file}")

def main():
    parser = argparse.ArgumentParser(description="Download models for BEM experiments")
    parser.add_argument("--name", required=True, 
                       choices=list(APPROVED_MODELS.keys()),
                       help="Model name to download")
    parser.add_argument("--out", required=True, type=Path,
                       help="Output directory for model")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if cached")
    parser.add_argument("--update-lock", type=Path, 
                       default="manifests/base_lock.json",
                       help="Update base lock file with model metadata")
    
    args = parser.parse_args()
    
    print("🤖 BEM Model Fetcher")
    print("=" * 50)
    print(f"Model: {args.name}")
    print(f"Output: {args.out}")
    print(f"Model info: {APPROVED_MODELS[args.name]}")
    
    try:
        # Download model
        import time
        start_time = time.time()
        metadata = download_model(args.name, args.out, args.force)
        metadata["download_timestamp"] = start_time
        
        # Update lock file
        if args.update_lock:
            update_base_lock(metadata, args.update_lock)
        
        print(f"🎉 Success! Model ready at: {args.out}")
        
    except Exception as e:
        print(f"💥 Failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())