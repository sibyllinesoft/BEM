#!/usr/bin/env python3
"""
Demo: BEM 2.0 Unified Multimodal System

Demonstrates the unified multimodal BEM system using the standardized trainer interface
and template-based configuration system. Shows vision conditioning, coverage/consistency
analysis, and conflict gating with the new unified API.

Usage:
    python demo_unified_multimodal.py --image path/to/image.jpg --question "What do you see?"
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import unified infrastructure
from src.bem_core.config.config_loader import load_experiment_config, load_training_config
from src.bem_core.training import BaseTrainer, TrainingConfig

# Import multimodal components (these would be part of unified trainers when available)
try:
    from bem2.multimodal import (
        VisionEncoder, MultimodalController, ConsistencyGate, VisionPreprocessor,
        create_vision_encoder, create_multimodal_controller, create_vision_preprocessor
    )
except ImportError:
    # Fallback to mock implementations for demo
    logging.warning("Multimodal components not available, using mock implementations")
    
    class MockVisionEncoder:
        def __init__(self, **kwargs):
            self.config = kwargs
        
        def to(self, device):
            return self
    
    class MockMultimodalController:
        def __init__(self, **kwargs):
            self.config = kwargs
            
        def to(self, device):
            return self
        
        def __call__(self, hidden_states, vision_features, **kwargs):
            batch_size, seq_len, hidden_dim = hidden_states.shape
            codes = torch.randn(batch_size, seq_len, hidden_dim)
            
            # Mock routing state
            routing_state = type('RoutingState', (), {
                'uncertainty': torch.rand(batch_size, seq_len),
                'entropy': torch.rand(1),
                'conflict_gate_active': True
            })()
            
            return codes, routing_state
    
    class MockConsistencyGate:
        def __init__(self, **kwargs):
            self.config = kwargs
        
        def to(self, device):
            return self
        
        def coverage_analyzer(self, vision_features):
            return type('CoverageMetrics', (), {
                'overall_score': torch.rand(1),
                'spatial_entropy': torch.rand(1),
                'object_coverage': torch.rand(1),
                'region_diversity': torch.rand(1)
            })()
        
        def consistency_analyzer(self, vision_features, text_features):
            return type('ConsistencyMetrics', (), {
                'overall_score': torch.rand(1),
                'cross_modal_alignment': torch.rand(1),
                'global_local_consistency': torch.rand(1),
                'temporal_consistency': torch.rand(1)
            })()
        
        def __call__(self, vision_features, text_features, return_analysis=False):
            gate_weights = torch.rand(1)
            if return_analysis:
                conflict_analysis = type('ConflictAnalysis', (), {
                    'conflict_detected': torch.rand(1) > 0.5,
                    'confidence_level': type('ConfidenceLevel', (), {'value': 'MEDIUM'})(),
                    'recommended_action': 'proceed',
                    'failure_modes': []
                })()
                return gate_weights, conflict_analysis
            return gate_weights
    
    class MockVisionPreprocessor:
        def __init__(self, encoder, config):
            self.encoder = encoder
            self.config = config
        
        def preprocess_image(self, image_path, text_length=10):
            # Mock vision features
            vision_features = type('VisionFeatures', (), {
                'coverage_score': torch.rand(1),
                'consistency_score': torch.rand(1),
            })()
            
            cached_features = type('CachedFeatures', (), {
                'features': vision_features
            })()
            
            return cached_features
    
    # Mock factory functions
    def create_vision_encoder(**kwargs):
        return MockVisionEncoder(**kwargs)
    
    def create_multimodal_controller(**kwargs):
        return MockMultimodalController(**kwargs)
    
    def create_vision_preprocessor(encoder, config):
        return MockVisionPreprocessor(encoder, config)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    """Unified trainer for multimodal BEM components."""
    
    def __init__(self, config_path: str, **kwargs):
        """Initialize multimodal trainer.
        
        Args:
            config_path: Path to experiment configuration
            **kwargs: Additional trainer arguments
        """
        # Load training configuration 
        training_config = load_training_config(config_path)
        super().__init__(training_config, **kwargs)
        
        # Load full experiment configuration
        self.experiment_config = load_experiment_config(config_path)
        
        # Initialize multimodal components
        self.vision_encoder = None
        self.controller = None
        self.consistency_gate = None
        self.preprocessor = None
    
    def _setup_model(self):
        """Set up multimodal model components."""
        config = self.experiment_config
        
        # Create vision encoder
        vision_config = config.model.get('vision', {})
        self.vision_encoder = create_vision_encoder(
            model_path=vision_config.get('model_path', "models/vision"),
            config={
                'vision_dim': vision_config.get('vision_dim', 512),
                'num_regions': vision_config.get('num_regions', 8),
                'enable_coverage_analysis': vision_config.get('enable_coverage_analysis', True)
            }
        )
        
        # Create multimodal controller
        controller_config = config.model.get('controller', {})
        self.controller = create_multimodal_controller(
            model_config={'hidden_size': config.model.get('hidden_size', 768)},
            vision_config={
                'vision_dim': vision_config.get('vision_dim', 512),
                'num_regions': vision_config.get('num_regions', 8),
                'patch_grid_size': vision_config.get('patch_grid_size', (14, 14)),
                'projection_mode': vision_config.get('projection_mode', 'adaptive')
            },
            controller_config={
                'code_dim': controller_config.get('code_dim', 8),
                'chunk_size': controller_config.get('chunk_size', 32),
                'enable_uncertainty': controller_config.get('enable_uncertainty', True)
            }
        )
        
        # Create consistency gate
        gate_config = config.model.get('consistency_gate', {})
        self.consistency_gate = MockConsistencyGate(
            vision_dim=vision_config.get('vision_dim', 512),
            text_dim=config.model.get('hidden_size', 768),
            num_regions=vision_config.get('num_regions', 8),
            patch_grid_size=vision_config.get('patch_grid_size', (14, 14)),
            high_consistency_threshold=gate_config.get('high_threshold', 0.8),
            medium_consistency_threshold=gate_config.get('medium_threshold', 0.5),
            low_consistency_threshold=gate_config.get('low_threshold', 0.3),
            coverage_threshold=gate_config.get('coverage_threshold', 0.4)
        )
        
        # Create vision preprocessor
        preprocess_config = config.model.get('preprocessing', {})
        self.preprocessor = create_vision_preprocessor(
            self.vision_encoder,
            config={
                'enable_caching': preprocess_config.get('enable_caching', True),
                'cache_dir': preprocess_config.get('cache_dir', 'cache/demo_vision')
            }
        )
        
        # Return the main controller as the primary model
        return self.controller
    
    def _compute_loss(self, batch: Dict[str, Any], model_outputs: Any) -> Dict[str, torch.Tensor]:
        """Compute multimodal training loss."""
        # This would implement the actual multimodal loss
        # For demo purposes, return a mock loss
        mock_loss = torch.tensor(0.5, requires_grad=True)
        return {"loss": mock_loss}
    
    def _evaluate(self, dataloader) -> Dict[str, float]:
        """Run multimodal evaluation."""
        # This would implement actual evaluation metrics
        # For demo purposes, return mock metrics
        return {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "multimodal_alignment": 0.78,
            "consistency_score": 0.74
        }
    
    def encode_question(self, question: str) -> torch.Tensor:
        """Encode question text to hidden states."""
        # Simple word-based encoding for demo
        words = question.lower().split()[:128]
        
        batch_size = 1
        seq_len = len(words)
        hidden_dim = self.experiment_config.model.get('hidden_size', 768)
        
        # Generate somewhat meaningful embeddings
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(self.device)
        
        # Add structure based on question words
        for i, word in enumerate(words):
            if word in ['what', 'where', 'when', 'who', 'how', 'why']:
                hidden_states[0, i] += torch.tensor([2.0] * hidden_dim).to(self.device)
            elif word in ['color', 'object', 'person', 'car', 'building']:
                hidden_states[0, i] += torch.tensor([1.5] * hidden_dim).to(self.device)
        
        return hidden_states
    
    def process_multimodal_input(
        self, 
        image_path: Path, 
        question: str
    ) -> Dict[str, Any]:
        """Process image and question through unified multimodal system."""
        logger.info(f"Processing with unified API: {image_path}")
        logger.info(f"Question: {question}")
        
        start_time = time.time()
        
        # Step 1: Vision preprocessing
        preprocessing_start = time.time()
        try:
            cached_features = self.preprocessor.preprocess_image(
                image_path, text_length=len(question.split())
            )
            vision_features = cached_features.features
            preprocessing_time = time.time() - preprocessing_start
            
            logger.info(f"✅ Vision preprocessing: {preprocessing_time*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"Vision preprocessing failed: {e}")
            return {'error': f'Vision preprocessing failed: {e}'}
        
        # Step 2: Question encoding
        encoding_start = time.time()
        hidden_states = self.encode_question(question)
        encoding_time = time.time() - encoding_start
        
        logger.info(f"✅ Question encoding: {encoding_time*1000:.1f}ms")
        
        # Step 3: Multimodal processing
        controller_start = time.time()
        with torch.no_grad():
            codes, routing_state = self.controller(
                hidden_states=hidden_states,
                vision_features=vision_features,
                routing_level='chunk',
                return_routing_state=True,
                enable_vision_conditioning=True
            )
        controller_time = time.time() - controller_start
        
        logger.info(f"✅ Unified multimodal processing: {controller_time*1000:.1f}ms")
        
        # Step 4: Consistency analysis
        analysis_start = time.time()
        text_features = hidden_states.mean(dim=1)  # Simple text summary
        
        coverage_metrics = self.consistency_gate.coverage_analyzer(vision_features)
        consistency_metrics = self.consistency_gate.consistency_analyzer(
            vision_features, text_features
        )
        gate_weights, conflict_analysis = self.consistency_gate(
            vision_features, text_features, return_analysis=True
        )
        
        analysis_time = time.time() - analysis_start
        
        logger.info(f"✅ Unified consistency analysis: {analysis_time*1000:.1f}ms")
        
        # Step 5: Response generation (mock)
        generation_start = time.time()
        
        if 'color' in question.lower():
            mock_answer = "The image contains various colors including blue and red objects."
        elif 'what' in question.lower():
            mock_answer = "The image shows several objects including vehicles and buildings."
        elif 'where' in question.lower():
            mock_answer = "The objects are positioned throughout the scene, with some in the foreground."
        else:
            mock_answer = "This appears to be a complex scene with multiple elements."
        
        if conflict_analysis.conflict_detected:
            mock_answer = "I can see the image, but I'm not completely certain about all the details. " + mock_answer
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Compile unified results
        results = {
            'question': question,
            'answer': mock_answer,
            'image_path': str(image_path),
            'unified_timing': {
                'preprocessing_ms': preprocessing_time * 1000,
                'encoding_ms': encoding_time * 1000,
                'multimodal_ms': controller_time * 1000,
                'analysis_ms': analysis_time * 1000,
                'generation_ms': generation_time * 1000,
                'total_ms': total_time * 1000
            },
            'unified_metrics': {
                'coverage_score': getattr(coverage_metrics.overall_score, 'mean', lambda: coverage_metrics.overall_score)().item(),
                'consistency_score': getattr(consistency_metrics.overall_score, 'mean', lambda: consistency_metrics.overall_score)().item(),
                'conflict_detected': conflict_analysis.conflict_detected,
                'confidence_level': conflict_analysis.confidence_level.value,
                'code_norm': codes.norm().item(),
                'uncertainty_score': getattr(routing_state.uncertainty, 'mean', lambda: torch.tensor(0.0))().item(),
            }
        }
        
        return results


def create_mock_multimodal_config() -> str:
    """Create a mock configuration file for demonstration."""
    config_content = """
# Unified Multimodal BEM Configuration (MM0 Converted)
name: "MM0_unified_demo"
description: "Unified multimodal BEM system demonstration"

# Model configuration
model:
  type: "multimodal_bem"
  hidden_size: 768
  
  vision:
    model_path: "models/vision"
    vision_dim: 512
    num_regions: 8
    patch_grid_size: [14, 14]
    projection_mode: "adaptive"
    enable_coverage_analysis: true
  
  controller:
    code_dim: 8
    chunk_size: 32
    enable_uncertainty: true
  
  consistency_gate:
    high_threshold: 0.8
    medium_threshold: 0.5
    low_threshold: 0.3
    coverage_threshold: 0.4
  
  preprocessing:
    enable_caching: true
    cache_dir: "cache/demo_vision"

# Training configuration
training:
  learning_rate: 5e-5
  batch_size: 16
  max_steps: 1000
  warmup_steps: 100
  eval_steps: 100
  logging_steps: 50
  save_steps: 500

# Hardware configuration
hardware:
  device: "auto"
  fp16: false
  gradient_checkpointing: false

# Logging configuration
logging:
  level: "INFO"
  log_to_file: true
  wandb_project: null

# Random seed
seed: 42
deterministic: true
"""
    
    config_path = Path("demo_multimodal_config.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return str(config_path)


def print_unified_results(results: Dict[str, Any]) -> None:
    """Print formatted results using unified interface."""
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print("\n" + "="*70)
    print("🎯 BEM 2.0 UNIFIED Multimodal Analysis Results")
    print("="*70)
    
    print(f"\n❓ Question: {results['question']}")
    print(f"🖼️  Image: {results['image_path']}")
    print(f"💬 Answer: {results['answer']}")
    
    print(f"\n⚡ Unified Performance Metrics:")
    timing = results['unified_timing']
    print(f"  Vision Preprocessing: {timing['preprocessing_ms']:.1f}ms")
    print(f"  Question Encoding: {timing['encoding_ms']:.1f}ms")
    print(f"  Multimodal Processing: {timing['multimodal_ms']:.1f}ms")
    print(f"  Consistency Analysis: {timing['analysis_ms']:.1f}ms")
    print(f"  Answer Generation: {timing['generation_ms']:.1f}ms")
    print(f"  🕐 Total Unified Time: {timing['total_ms']:.1f}ms")
    
    print(f"\n🤖 Unified System Metrics:")
    metrics = results['unified_metrics']
    print(f"  Coverage Score: {metrics['coverage_score']:.3f}")
    print(f"  Consistency Score: {metrics['consistency_score']:.3f}")
    print(f"  Code Norm: {metrics['code_norm']:.3f}")
    print(f"  Uncertainty Score: {metrics['uncertainty_score']:.3f}")
    print(f"  Conflict Detected: {'Yes' if metrics['conflict_detected'] else 'No'}")
    print(f"  Confidence Level: {metrics['confidence_level']}")
    
    print(f"\n✨ Benefits of Unified Interface:")
    print(f"  - Single API for all multimodal components")
    print(f"  - Template-based configuration system") 
    print(f"  - Standardized metrics and logging")
    print(f"  - Unified training/evaluation pipeline")
    print(f"  - Consistent error handling and validation")


def main():
    parser = argparse.ArgumentParser(description="BEM 2.0 Unified Multimodal Demo")
    parser.add_argument("--question", required=True, type=str,
                       help="Question to ask about the image")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to image file (optional - will use mock if not provided)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file (will create demo config if not provided)")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("🚀 BEM 2.0 Unified Multimodal System Demo")
    print("=" * 60)
    
    # Create or use configuration
    if args.config:
        config_path = args.config
    else:
        print("ℹ️  Creating demo configuration...")
        config_path = create_mock_multimodal_config()
        print(f"✅ Demo config created: {config_path}")
    
    # Handle image path
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return 1
    else:
        print("ℹ️  No image provided, using mock image for demo")
        image_path = Path("mock_image.jpg")
    
    try:
        # Initialize unified trainer
        print(f"🔧 Initializing unified multimodal trainer...")
        trainer = MultimodalTrainer(
            config_path=config_path,
            experiment_name="unified_multimodal_demo"
        )
        
        # Setup training (this initializes all components)
        print(f"🛠️  Setting up unified training pipeline...")
        # Note: In a real scenario, we'd have actual dataloaders
        trainer.setup_training(train_dataloader=None, eval_dataloader=None)
        
        print(f"✅ Unified system initialized with template configuration")
        
        # Process multimodal input
        results = trainer.process_multimodal_input(image_path, args.question)
        
        # Display results
        print_unified_results(results)
        
        # Save results if requested
        if args.output and 'error' not in results:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_path}")
        
        print(f"\n✅ Unified multimodal demo completed successfully!")
        print(f"📈 Configuration inheritance and unified API demonstrated")
        
        # Clean up demo config if we created it
        if not args.config:
            Path(config_path).unlink()
        
        return 0
        
    except Exception as e:
        logger.error(f"Unified demo failed: {e}")
        logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    exit(main())