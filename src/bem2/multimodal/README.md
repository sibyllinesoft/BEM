# BEM 2.0 Multimodal Conditioning (MM0)

This package implements multimodal conditioning for BEM 2.0, adding vision capabilities while maintaining cache-safety and performance requirements.

## Overview

MM0 extends the BEM architecture with vision conditioning while preserving the critical cache-safety property. Vision features are fed **only** to the controller, never to generator sites (W_down/W_O), ensuring that the generative model remains text-only and maintains its caching benefits.

### Key Features

- **Cache-Safe Design**: Vision features restricted to controller only
- **CLIP-Based Vision**: Uses ViT-B/32 for robust feature extraction
- **Coverage Analysis**: Spatial attention distribution monitoring
- **Consistency Gating**: Automatic fallback for low-confidence predictions
- **Conflict Detection**: Multi-modal alignment validation
- **VQA Evaluation**: Comprehensive visual question answering metrics
- **Hallucination Detection**: Object, attribute, and spatial consistency checking

## Architecture

```
Image → CLIP Encoder → Vision Features → Controller Only
                                      ↓
Text → Tokenizer → Hidden States → Controller + Generator
                                     ↓           ↓
                              Vision-Conditioned   Text-Only
                                 Routing         Generation
```

## Components

### Core Modules

- **`vision_encoder.py`**: CLIP-based vision feature extraction
- **`controller_integration.py`**: Multimodal controller with vision projection
- **`coverage_analysis.py`**: Spatial coverage and consistency analysis
- **`evaluation.py`**: VQA evaluation and hallucination detection
- **`preprocessing.py`**: Vision feature caching and batch processing
- **`training.py`**: Multimodal training with cache-safety validation

### Scripts

- **`train_multimodal.py`**: Main training script
- **`demo_multimodal.py`**: Interactive demonstration
- **`precompute.py`**: Batch vision feature precomputation
- **`vqa_suite.py`**: Comprehensive VQA evaluation

### Configuration

- **`MM0.yml`**: Complete experiment configuration with acceptance gates

## Quick Start

### 1. Training

```bash
# Train with default configuration
python train_multimodal.py --config experiments/MM0.yml --output outputs/MM0

# Resume from checkpoint
python train_multimodal.py --config experiments/MM0.yml --resume outputs/MM0/checkpoint_1000.pt

# Debug with smaller dataset
python train_multimodal.py --config experiments/MM0.yml --debug
```

### 2. Demo

```bash
# Demo with real image
python demo_multimodal.py --image path/to/image.jpg --question "What do you see?"

# Demo with mock data
python demo_multimodal.py --question "What objects are in the image?"
```

### 3. Evaluation

```bash
# Run comprehensive VQA evaluation
python -m eval.vqa_suite --config experiments/MM0.yml --checkpoint outputs/MM0/best_model.pt
```

### 4. Vision Feature Precomputation

```bash
# Precompute features for faster training
python -m bem2.multimodal.precompute --encoder models/vision --images data/vqa/images --out data/vqa/vis_feats.parquet
```

## Usage Examples

### Basic Multimodal Processing

```python
from bem2.multimodal import (
    create_vision_encoder,
    create_multimodal_controller, 
    create_vision_preprocessor
)

# Initialize components
vision_encoder = create_vision_encoder("models/vision")
controller = create_multimodal_controller(
    model_config={'hidden_size': 768},
    vision_config={'vision_dim': 512, 'num_regions': 8}
)
preprocessor = create_vision_preprocessor(vision_encoder)

# Process image and text
vision_features = preprocessor.preprocess_image("image.jpg", text_length=50)
hidden_states = tokenize_and_embed("What's in this image?")

# Get vision-conditioned codes
codes, routing_state = controller(
    hidden_states=hidden_states,
    vision_features=vision_features.features,
    enable_vision_conditioning=True
)
```

### Coverage and Consistency Analysis

```python
from bem2.multimodal import ConsistencyGate

gate = ConsistencyGate(
    vision_dim=512,
    text_dim=768,
    num_regions=8,
    patch_grid_size=(14, 14)
)

# Analyze multimodal consistency
gate_weights, conflict_analysis = gate(
    vision_features, text_features, return_analysis=True
)

print(f"Conflict detected: {conflict_analysis.conflict_detected}")
print(f"Confidence level: {conflict_analysis.confidence_level}")
print(f"Recommended action: {conflict_analysis.recommended_action}")
```

### VQA Evaluation

```python
from bem2.multimodal import VQAEvaluator

evaluator = VQAEvaluator()

# Evaluate batch of predictions
metrics = evaluator.evaluate_batch(
    predictions=["A red car", "Blue sky"],
    ground_truths=[["red car", "vehicle"], ["blue sky", "clear sky"]], 
    questions=["What color is the car?", "What's the weather like?"]
)

print(f"Exact Match: {metrics.exact_match:.3f}")
print(f"F1 Score: {metrics.f1_score:.3f}")
```

## Configuration

The system is configured via YAML files. Key parameters:

```yaml
model:
  controller:
    code_dim: 8
    chunk_size: 32
    enable_uncertainty: true
  
  vision:
    vision_dim: 512
    num_regions: 8 
    patch_grid_size: [14, 14]
    enable_coverage_analysis: true
  
  multimodal:
    consistency_threshold: 0.5
    coverage_threshold: 0.3
    enable_conflict_gating: true

training:
  loss_weights:
    primary: 1.0
    coverage: 0.1
    consistency: 0.1
    conflict: 0.05
    hallucination: 0.1
```

## Cache-Safety Guarantees

The MM0 implementation maintains strict cache-safety:

1. **Generator Isolation**: Vision features never reach W_down or W_O matrices
2. **Controller-Only**: All vision conditioning happens in the routing controller
3. **Gradient Validation**: Automatic checking ensures no gradients flow to generator
4. **Chunk Alignment**: Vision patches align with text chunks for consistent routing

## Performance Targets

From the MM0 specification:

- **VQA Improvement**: +≥2% EM/F1 on validation set
- **Hallucination Reduction**: ≥10% reduction in hallucination rate  
- **Latency Constraint**: ≤+15% p50 latency increase
- **Coverage Quality**: ≥0.4 average coverage score
- **Consistency Quality**: ≥0.5 average consistency score

## File Structure

```
bem2/multimodal/
├── __init__.py                 # Package exports
├── vision_encoder.py          # CLIP-based vision encoding
├── controller_integration.py  # Multimodal controller
├── coverage_analysis.py       # Coverage/consistency analysis
├── evaluation.py              # VQA evaluation framework  
├── preprocessing.py           # Vision preprocessing pipeline
├── training.py               # Multimodal training system
├── precompute.py             # Vision feature precomputation
└── README.md                 # This file

experiments/
└── MM0.yml                   # Experiment configuration

scripts/
├── train_multimodal.py       # Training script
├── demo_multimodal.py        # Interactive demo
└── eval/
    └── vqa_suite.py          # VQA evaluation suite
```

## Research Background

MM0 implements findings from recent multimodal AI research:

- **Cache-Safety**: Novel approach preserving text-only generation benefits
- **Coverage Analysis**: Spatial attention distribution for hallucination detection
- **Conflict Gating**: Automatic quality assessment and fallback mechanisms
- **Chunk Alignment**: Synchronizing vision patches with text processing windows

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size in config or use gradient checkpointing
2. **Vision Model Not Found**: Ensure CLIP model is downloaded to `models/vision/`
3. **Cache Directory Permissions**: Check write permissions for `cache/vision_features/`
4. **Import Errors**: Verify all dependencies are installed and modules are in PYTHONPATH

### Debug Mode

Use `--debug` flag for development:

```bash
python train_multimodal.py --config experiments/MM0.yml --debug
```

This enables:
- Smaller datasets (100 train, 20 eval samples)
- Verbose logging
- Additional validation checks
- Performance profiling

## Contributing

When modifying the multimodal system:

1. **Maintain Cache-Safety**: Never allow vision features to reach generator
2. **Preserve Interfaces**: Keep backward compatibility with existing APIs
3. **Add Tests**: Include unit tests for new functionality
4. **Update Docs**: Document changes and new parameters
5. **Validate Performance**: Ensure latency and quality requirements are met

## License

This implementation is part of the BEM 2.0 research project.