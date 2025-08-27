#!/usr/bin/env python3
"""
Training Script for BEM 2.0 Multimodal Conditioning (MM0)

Implements the complete training pipeline for multimodal BEM with vision
conditioning, cache safety validation, and VQA evaluation.

Usage:
    python train_multimodal.py --config experiments/MM0.yml --output outputs/MM0
"""

import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from bem2.multimodal import (
    VisionEncoder, 
    MultimodalController,
    ConsistencyGate,
    VisionPreprocessor,
    MultimodalTrainer,
    MultimodalTrainingConfig,
    create_vision_encoder,
    create_multimodal_controller,
    create_vision_preprocessor,
    create_multimodal_trainer
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VQADataset(Dataset):
    """VQA dataset for multimodal training."""
    
    def __init__(
        self,
        data_file: str,
        image_dir: str,
        max_samples: Optional[int] = None
    ):
        self.image_dir = Path(image_dir)
        self.data = self._load_data(data_file, max_samples)
    
    def _load_data(self, data_file: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Load VQA data from JSONL file."""
        data = []
        
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                item = json.loads(line.strip())
                
                # Expected format:
                # {
                #   "image_id": "123",
                #   "image_filename": "image_123.jpg", 
                #   "question": "What color is the car?",
                #   "answers": ["red", "crimson"],
                #   "detected_objects": ["car", "tree"],
                #   "visual_facts": ["there is a red car in the image"]
                # }
                
                # Add full image path
                image_path = self.image_dir / item['image_filename']
                item['image_path'] = image_path
                
                data.append(item)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yml') or config_path.endswith('.yaml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return config


def create_mock_vqa_data(output_dir: Path, num_samples: int = 1000):
    """Create mock VQA data for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock train data
    train_data = []
    for i in range(num_samples):
        item = {
            "image_id": f"train_{i:06d}",
            "image_filename": f"train_{i:06d}.jpg",
            "question": f"What is the main object in image {i}?",
            "answers": [f"object_{i % 10}", f"item_{i % 10}"],
            "detected_objects": [f"object_{i % 10}"],
            "visual_facts": [f"there is an object_{i % 10} in the image"],
            "object_attributes": {f"object_{i % 10}": ["red", "large"]},
            "spatial_facts": [f"object_{i % 10} is in the center"]
        }
        train_data.append(item)
    
    # Save train data
    train_file = output_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Create smaller eval set
    eval_data = train_data[:num_samples // 10]  # 10% for eval
    eval_file = output_dir / "val.jsonl"
    with open(eval_file, 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')
    
    # Create mock images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create empty image files (just for testing)
    for item in train_data:
        image_path = images_dir / item['image_filename']
        image_path.touch()
    
    logger.info(f"Created mock VQA data: {len(train_data)} train, {len(eval_data)} eval")
    return train_file, eval_file, images_dir


def setup_model(config: Dict[str, Any]) -> tuple:
    """Set up multimodal model components."""
    model_config = config['model']
    
    # Create vision encoder
    logger.info("Creating vision encoder...")
    vision_encoder = create_vision_encoder(
        model_path=model_config['vision']['encoder_path'],
        config=model_config['vision']
    )
    
    # Create multimodal controller
    logger.info("Creating multimodal controller...")
    controller = create_multimodal_controller(
        model_config={'hidden_size': model_config['controller']['input_dim']},
        vision_config=model_config['vision'],
        controller_config=model_config['controller']
    )
    
    # Create consistency gate
    logger.info("Creating consistency gate...")
    consistency_gate = ConsistencyGate(
        vision_dim=model_config['vision']['vision_dim'],
        text_dim=model_config['controller']['input_dim'],
        num_regions=model_config['vision']['num_regions'],
        patch_grid_size=tuple(model_config['vision']['patch_grid_size']),
        high_consistency_threshold=0.8,
        medium_consistency_threshold=model_config['multimodal']['consistency_threshold'],
        low_consistency_threshold=0.3,
        coverage_threshold=model_config['multimodal']['coverage_threshold']
    )
    
    # Create vision preprocessor
    logger.info("Creating vision preprocessor...")
    cache_config = config.get('cache', {})
    preprocessor = create_vision_preprocessor(
        vision_encoder,
        config={
            'cache_dir': cache_config.get('vision_cache', {}).get('cache_dir', 'cache/vision_features'),
            'enable_caching': True,
            'cache_config': cache_config.get('vision_cache', {}),
            'chunk_config': cache_config.get('chunk_processing', {})
        }
    )
    
    return vision_encoder, controller, consistency_gate, preprocessor


def setup_datasets(config: Dict[str, Any]) -> tuple:
    """Set up training and evaluation datasets."""
    data_config = config['data']
    
    # Check if data files exist, create mock data if not
    train_file = Path(data_config['train_data'])
    eval_file = Path(data_config['eval_data'])
    image_dir = Path(data_config['image_dir'])
    
    if not train_file.exists() or not eval_file.exists() or not image_dir.exists():
        logger.warning("VQA data files not found, creating mock data for testing...")
        train_file, eval_file, image_dir = create_mock_vqa_data(
            train_file.parent, num_samples=1000
        )
    
    # Create datasets
    train_dataset = VQADataset(str(train_file), str(image_dir))
    eval_dataset = VQADataset(str(eval_file), str(image_dir))
    
    logger.info(f"Loaded datasets: {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="Train BEM 2.0 Multimodal Model")
    parser.add_argument("--config", required=True, type=str,
                       help="Path to experiment configuration file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with smaller dataset")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run without actual training")
    
    args = parser.parse_args()
    
    print("🚀 BEM 2.0 Multimodal Training")
    print("=" * 50)
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output:
        config['output']['output_dir'] = args.output
    
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    config_save_path = output_dir / "config.yml"
    with open(config_save_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    # Set random seed
    if 'seed' in config['output']:
        torch.manual_seed(config['output']['seed'])
        
    # Set device
    device = config['hardware']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Setup model components
    logger.info("Setting up model components...")
    vision_encoder, controller, consistency_gate, preprocessor = setup_model(config)
    
    # Move models to device
    vision_encoder = vision_encoder.to(device)
    controller = controller.to(device)
    consistency_gate = consistency_gate.to(device)
    
    # Setup datasets
    logger.info("Setting up datasets...")
    train_dataset, eval_dataset = setup_datasets(config)
    
    # Debug mode: use smaller datasets
    if args.debug:
        logger.info("Debug mode: using smaller datasets")
        train_data = train_dataset.data[:100]
        eval_data = eval_dataset.data[:20]
        train_dataset.data = train_data
        eval_dataset.data = eval_data
    
    # Convert datasets to format expected by trainer
    def dataset_to_list(dataset):
        return [
            {
                'image': item['image_path'],
                'question': item['question'],
                'answers': item['answers'],
                'detected_objects': item.get('detected_objects', []),
                'visual_facts': item.get('visual_facts', []),
                'object_attributes': item.get('object_attributes', {}),
                'spatial_facts': item.get('spatial_facts', [])
            }
            for item in dataset.data
        ]
    
    train_data_list = dataset_to_list(train_dataset)
    eval_data_list = dataset_to_list(eval_dataset)
    
    # Create training configuration
    training_config_dict = config['training'].copy()
    
    # Map config to MultimodalTrainingConfig fields
    training_config_dict.update({
        'vision_dim': config['model']['vision']['vision_dim'],
        'controller_dim': config['model']['controller']['input_dim'],
        'code_dim': config['model']['controller']['code_dim'],
        'primary_loss_weight': training_config_dict['loss_weights']['primary'],
        'coverage_loss_weight': training_config_dict['loss_weights']['coverage'],
        'consistency_loss_weight': training_config_dict['loss_weights']['consistency'],
        'conflict_loss_weight': training_config_dict['loss_weights']['conflict'],
        'hallucination_loss_weight': training_config_dict['loss_weights']['hallucination'],
        'consistency_threshold': config['model']['multimodal']['consistency_threshold'],
        'coverage_threshold': config['model']['multimodal']['coverage_threshold']
    })
    
    # Remove nested dict
    del training_config_dict['loss_weights']
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_multimodal_trainer(
        model=controller,
        vision_encoder=vision_encoder,
        preprocessor=preprocessor,
        consistency_gate=consistency_gate,
        config=training_config_dict,
        output_dir=str(output_dir)
    )
    
    if args.dry_run:
        logger.info("✅ Dry run completed - all components initialized successfully")
        return 0
    
    # Train model
    logger.info("Starting training...")
    
    try:
        trainer.train(
            train_dataset=train_data_list,
            eval_dataset=eval_data_list,
            resume_from=args.resume
        )
        
        # Get training summary
        summary = trainer.get_training_summary()
        
        # Save training summary
        summary_path = output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✅ Training completed successfully!")
        
        # Print final metrics
        if summary.get('best_metrics'):
            print("\n🎯 Best Results:")
            for metric, value in summary['best_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save current state
        trainer.save_checkpoint(trainer.step)
        logger.info(f"Checkpoint saved at step {trainer.step}")
        
        return 1
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    exit(main())