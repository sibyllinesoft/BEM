#!/usr/bin/env python3
"""
Demo Script for BEM 2.0 Multimodal Conditioning

Demonstrates the multimodal BEM system with vision conditioning,
coverage/consistency analysis, and conflict gating.

Usage:
    python demo_multimodal.py --image path/to/image.jpg --question "What do you see?"
"""

import argparse
import logging
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

# Add project root to path  
sys.path.append(str(Path(__file__).parent))

from bem2.multimodal import (
    VisionEncoder,
    MultimodalController,
    ConsistencyGate,
    VisionPreprocessor,
    create_vision_encoder,
    create_multimodal_controller,
    create_vision_preprocessor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalBEMDemo:
    """Demo class for BEM 2.0 multimodal system."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.setup_models()
    
    def setup_models(self):
        """Initialize all model components."""
        logger.info("Setting up BEM 2.0 multimodal system...")
        
        # Create vision encoder
        self.vision_encoder = create_vision_encoder(
            model_path="models/vision",
            config={
                'vision_dim': 512,
                'num_regions': 8,
                'enable_coverage_analysis': True
            }
        ).to(self.device)
        
        # Create multimodal controller
        self.controller = create_multimodal_controller(
            model_config={'hidden_size': 768},
            vision_config={
                'vision_dim': 512,
                'num_regions': 8,
                'patch_grid_size': (14, 14),
                'projection_mode': 'adaptive'
            },
            controller_config={
                'code_dim': 8,
                'chunk_size': 32,
                'enable_uncertainty': True
            }
        ).to(self.device)
        
        # Create consistency gate
        self.consistency_gate = ConsistencyGate(
            vision_dim=512,
            text_dim=768,
            num_regions=8,
            patch_grid_size=(14, 14),
            high_consistency_threshold=0.8,
            medium_consistency_threshold=0.5,
            low_consistency_threshold=0.3,
            coverage_threshold=0.4
        ).to(self.device)
        
        # Create vision preprocessor
        self.preprocessor = create_vision_preprocessor(
            self.vision_encoder,
            config={
                'enable_caching': True,
                'cache_dir': 'cache/demo_vision'
            }
        )
        
        logger.info("✅ All components initialized")
    
    def encode_question(self, question: str) -> torch.Tensor:
        """Mock question encoding (would use actual tokenizer)."""
        # Simple word-based encoding for demo
        words = question.lower().split()[:128]  # Limit length
        
        # Create mock hidden states
        batch_size = 1
        seq_len = len(words)
        hidden_dim = 768
        
        # Generate somewhat meaningful embeddings
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(self.device)
        
        # Add some structure based on question words
        for i, word in enumerate(words):
            if word in ['what', 'where', 'when', 'who', 'how', 'why']:
                hidden_states[0, i] += torch.tensor([2.0] * hidden_dim).to(self.device)
            elif word in ['color', 'object', 'person', 'car', 'building']:
                hidden_states[0, i] += torch.tensor([1.5] * hidden_dim).to(self.device)
        
        return hidden_states
    
    def process_image_and_question(
        self, 
        image_path: Path, 
        question: str
    ) -> dict:
        """Process image and question through multimodal system."""
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Question: {question}")
        
        start_time = time.time()
        
        # Step 1: Preprocess image
        preprocessing_start = time.time()
        try:
            cached_features = self.preprocessor.preprocess_image(
                image_path, text_length=len(question.split())
            )
            vision_features = cached_features.features
            preprocessing_time = time.time() - preprocessing_start
            
            logger.info(f"✅ Vision preprocessing: {preprocessing_time*1000:.1f}ms")
            logger.info(f"   Coverage score: {vision_features.coverage_score.mean().item() if vision_features.coverage_score is not None else 'N/A'}")
            logger.info(f"   Consistency score: {vision_features.consistency_score.mean().item() if vision_features.consistency_score is not None else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Vision preprocessing failed: {e}")
            return {'error': f'Vision preprocessing failed: {e}'}
        
        # Step 2: Encode question
        encoding_start = time.time()
        hidden_states = self.encode_question(question)
        encoding_time = time.time() - encoding_start
        
        logger.info(f"✅ Question encoding: {encoding_time*1000:.1f}ms")
        
        # Step 3: Multimodal controller forward pass
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
        
        logger.info(f"✅ Multimodal controller: {controller_time*1000:.1f}ms")
        logger.info(f"   Code norm: {codes.norm().item():.3f}")
        logger.info(f"   Uncertainty: {routing_state.uncertainty.mean().item() if routing_state.uncertainty is not None else 'N/A'}")
        
        # Step 4: Coverage and consistency analysis
        analysis_start = time.time()
        text_features = hidden_states.mean(dim=1)  # Simple text summary
        
        # Run coverage analysis
        coverage_metrics = self.consistency_gate.coverage_analyzer(vision_features)
        
        # Run consistency analysis  
        consistency_metrics = self.consistency_gate.consistency_analyzer(
            vision_features, text_features
        )
        
        # Run conflict gating
        gate_weights, conflict_analysis = self.consistency_gate(
            vision_features, text_features, return_analysis=True
        )
        
        analysis_time = time.time() - analysis_start
        
        logger.info(f"✅ Coverage/consistency analysis: {analysis_time*1000:.1f}ms")
        
        # Step 5: Generate response (mock)
        generation_start = time.time()
        
        # Mock answer generation based on codes and question
        if 'color' in question.lower():
            mock_answer = "The image contains various colors including blue and red objects."
        elif 'what' in question.lower():
            mock_answer = "The image shows several objects including vehicles and buildings."
        elif 'where' in question.lower():
            mock_answer = "The objects are positioned throughout the scene, with some in the foreground."
        else:
            mock_answer = "This appears to be a complex scene with multiple elements."
        
        # Add conflict gating effect
        if conflict_analysis.conflict_detected:
            mock_answer = "I can see the image, but I'm not completely certain about all the details. " + mock_answer
        
        generation_time = time.time() - generation_start
        
        logger.info(f"✅ Answer generation: {generation_time*1000:.1f}ms")
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            'question': question,
            'answer': mock_answer,
            'image_path': str(image_path),
            'timing': {
                'preprocessing_ms': preprocessing_time * 1000,
                'encoding_ms': encoding_time * 1000,
                'controller_ms': controller_time * 1000,
                'analysis_ms': analysis_time * 1000,
                'generation_ms': generation_time * 1000,
                'total_ms': total_time * 1000
            },
            'vision_analysis': {
                'coverage_score': coverage_metrics.overall_score.mean().item(),
                'spatial_entropy': coverage_metrics.spatial_entropy.mean().item(),
                'object_coverage': coverage_metrics.object_coverage.mean().item(),
                'region_diversity': coverage_metrics.region_diversity.mean().item()
            },
            'consistency_analysis': {
                'overall_score': consistency_metrics.overall_score.mean().item(),
                'cross_modal_alignment': consistency_metrics.cross_modal_alignment.mean().item(),
                'global_local_consistency': consistency_metrics.global_local_consistency.mean().item(),
                'temporal_consistency': consistency_metrics.temporal_consistency.mean().item()
            },
            'conflict_gating': {
                'conflict_detected': conflict_analysis.conflict_detected,
                'confidence_level': conflict_analysis.confidence_level.value,
                'recommended_action': conflict_analysis.recommended_action,
                'failure_modes': conflict_analysis.failure_modes,
                'gate_weight': gate_weights.mean().item()
            },
            'multimodal_features': {
                'code_norm': codes.norm().item(),
                'uncertainty_score': routing_state.uncertainty.mean().item() if routing_state.uncertainty is not None else 0.0,
                'routing_entropy': routing_state.entropy.item() if routing_state.entropy is not None else 0.0,
                'gate_active': routing_state.conflict_gate_active
            }
        }
        
        return results
    
    def print_results(self, results: dict):
        """Print formatted results."""
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n" + "="*60)
        print("🎯 BEM 2.0 Multimodal Analysis Results")
        print("="*60)
        
        print(f"\n❓ Question: {results['question']}")
        print(f"🖼️  Image: {results['image_path']}")
        print(f"💬 Answer: {results['answer']}")
        
        print(f"\n⚡ Performance:")
        timing = results['timing']
        print(f"  Vision Preprocessing: {timing['preprocessing_ms']:.1f}ms")
        print(f"  Question Encoding: {timing['encoding_ms']:.1f}ms")
        print(f"  Multimodal Controller: {timing['controller_ms']:.1f}ms")
        print(f"  Coverage/Consistency: {timing['analysis_ms']:.1f}ms")
        print(f"  Answer Generation: {timing['generation_ms']:.1f}ms")
        print(f"  🕐 Total Time: {timing['total_ms']:.1f}ms")
        
        print(f"\n👁️  Vision Analysis:")
        vision = results['vision_analysis']
        print(f"  Coverage Score: {vision['coverage_score']:.3f}")
        print(f"  Spatial Entropy: {vision['spatial_entropy']:.3f}")
        print(f"  Object Coverage: {vision['object_coverage']:.3f}")
        print(f"  Region Diversity: {vision['region_diversity']:.3f}")
        
        print(f"\n🔗 Consistency Analysis:")
        consistency = results['consistency_analysis']
        print(f"  Overall Score: {consistency['overall_score']:.3f}")
        print(f"  Cross-Modal Alignment: {consistency['cross_modal_alignment']:.3f}")
        print(f"  Global-Local Consistency: {consistency['global_local_consistency']:.3f}")
        print(f"  Temporal Consistency: {consistency['temporal_consistency']:.3f}")
        
        print(f"\n🚪 Conflict Gating:")
        gating = results['conflict_gating']
        print(f"  Conflict Detected: {'Yes' if gating['conflict_detected'] else 'No'}")
        print(f"  Confidence Level: {gating['confidence_level']}")
        print(f"  Recommended Action: {gating['recommended_action']}")
        print(f"  Gate Weight: {gating['gate_weight']:.3f}")
        if gating['failure_modes']:
            print(f"  Failure Modes: {', '.join(gating['failure_modes'])}")
        
        print(f"\n🎭 Multimodal Features:")
        features = results['multimodal_features']
        print(f"  Code Norm: {features['code_norm']:.3f}")
        print(f"  Uncertainty Score: {features['uncertainty_score']:.3f}")
        print(f"  Routing Entropy: {features['routing_entropy']:.3f}")
        print(f"  Vision Gate Active: {'Yes' if features['gate_active'] else 'No'}")


def main():
    parser = argparse.ArgumentParser(description="BEM 2.0 Multimodal Demo")
    parser.add_argument("--question", required=True, type=str,
                       help="Question to ask about the image")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to image file (optional - will use mock if not provided)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("🚀 BEM 2.0 Multimodal Conditioning Demo")
    print("=" * 50)
    
    # Set device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    # Create demo system
    demo = MultimodalBEMDemo(device=device)
    
    # Handle image path
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return 1
    else:
        # Use mock image path for demo
        print("ℹ️  No image provided, using mock image for demo")
        image_path = Path("mock_image.jpg")
    
    try:
        # Process image and question
        results = demo.process_image_and_question(image_path, args.question)
        
        # Print results
        demo.print_results(results)
        
        # Save results if requested
        if args.output and 'error' not in results:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_path}")
        
        print(f"\n✅ Demo completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    exit(main())