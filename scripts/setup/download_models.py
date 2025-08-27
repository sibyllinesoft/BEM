#!/usr/bin/env python3
"""
Model Download and Management Script for BEM Repository

This script handles downloading and setting up model artifacts without
tracking large files in git. Models are downloaded on-demand and cached
locally.
"""

import os
import json
import hashlib
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model downloads and verification."""
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path(__file__).parent.parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / "model_registry.json"
        
    def get_model_registry(self) -> Dict[str, Any]:
        """Load model registry configuration."""
        default_registry = {
            "base_models": {
                "dialogpt-small": {
                    "source": "microsoft/DialoGPT-small",
                    "type": "huggingface",
                    "description": "Small conversational model for demonstration",
                    "size_mb": 117,
                    "required_files": [
                        "config.json",
                        "generation_config.json", 
                        "model.safetensors",
                        "tokenizer.json",
                        "vocab.json",
                        "merges.txt"
                    ]
                }
            },
            "vision_models": {
                "clip-vision": {
                    "source": "openai/clip-vit-base-patch32",
                    "type": "huggingface",
                    "description": "Vision encoder for multimodal experiments",
                    "size_mb": 605,
                    "required_files": [
                        "config.json",
                        "preprocessor_config.json",
                        "model.safetensors"
                    ]
                }
            },
            "safety_models": {
                "constitutional": {
                    "source": "custom",
                    "type": "generated",
                    "description": "Constitutional AI safety model components",
                    "size_mb": 50,
                    "required_files": [
                        "constitution.json",
                        "safety_classifier.pt",
                        "value_head.pt"
                    ]
                }
            }
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Save default registry
            with open(self.config_file, 'w') as f:
                json.dump(default_registry, f, indent=2)
            return default_registry
    
    def download_huggingface_model(self, model_id: str, target_dir: Path) -> bool:
        """Download model from Hugging Face Hub."""
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Downloading {model_id} to {target_dir}")
            
            snapshot_download(
                repo_id=model_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            
            # Create download metadata
            metadata = {
                "source": model_id,
                "downloaded_at": str(pd.Timestamp.now()),
                "type": "huggingface"
            }
            
            with open(target_dir / "download_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            return False
    
    def setup_model(self, category: str, model_name: str, force: bool = False) -> bool:
        """Set up a specific model."""
        registry = self.get_model_registry()
        
        if category not in registry:
            logger.error(f"Unknown model category: {category}")
            return False
            
        if model_name not in registry[category]:
            logger.error(f"Unknown model: {model_name} in category {category}")
            return False
            
        model_config = registry[category][model_name]
        target_dir = self.models_dir / model_name
        
        # Check if already downloaded
        if target_dir.exists() and not force:
            logger.info(f"Model {model_name} already exists. Use --force to re-download.")
            return True
            
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Download based on type
        if model_config["type"] == "huggingface":
            return self.download_huggingface_model(model_config["source"], target_dir)
        elif model_config["type"] == "generated":
            return self.generate_placeholder_model(model_config, target_dir)
        else:
            logger.error(f"Unknown model type: {model_config['type']}")
            return False
    
    def generate_placeholder_model(self, config: Dict, target_dir: Path) -> bool:
        """Generate placeholder model files for testing."""
        try:
            import torch
            import json
            
            # Create placeholder files based on configuration
            for filename in config["required_files"]:
                file_path = target_dir / filename
                
                if filename.endswith('.json'):
                    # Create minimal JSON config
                    placeholder_config = {
                        "model_type": "placeholder",
                        "generated": True,
                        "description": config["description"]
                    }
                    with open(file_path, 'w') as f:
                        json.dump(placeholder_config, f, indent=2)
                        
                elif filename.endswith('.pt'):
                    # Create small placeholder tensor
                    placeholder_tensor = torch.randn(10, 10)
                    torch.save(placeholder_tensor, file_path)
                    
                else:
                    # Create empty file
                    file_path.touch()
                    
            logger.info(f"Generated placeholder model: {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate placeholder model: {e}")
            return False
    
    def setup_all_models(self, force: bool = False):
        """Set up all models in the registry."""
        registry = self.get_model_registry()
        
        for category, models in registry.items():
            logger.info(f"Setting up {category}...")
            for model_name in models:
                self.setup_model(category, model_name, force=force)
    
    def list_models(self):
        """List all available models."""
        registry = self.get_model_registry()
        
        print("Available Models:")
        print("="*50)
        
        for category, models in registry.items():
            print(f"\n{category.upper()}:")
            for model_name, config in models.items():
                status = "✓" if (self.models_dir / model_name).exists() else "✗"
                size_mb = config.get('size_mb', 'Unknown')
                print(f"  {status} {model_name:<20} ({size_mb} MB) - {config['description']}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BEM Model Download Manager")
    parser.add_argument("--setup-all", action="store_true", help="Setup all models")
    parser.add_argument("--setup", nargs=2, metavar=('CATEGORY', 'MODEL'), help="Setup specific model")
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--force", action="store_true", help="Force re-download existing models")
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.list:
        manager.list_models()
    elif args.setup_all:
        manager.setup_all_models(force=args.force)
    elif args.setup:
        category, model_name = args.setup
        success = manager.setup_model(category, model_name, force=args.force)
        if not success:
            exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()