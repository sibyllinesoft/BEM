"""
BEM 2.0 Multimodal Conditioning (MM0) Package

This package implements multimodal conditioning for BEM 2.0, adding vision capabilities
while maintaining cache-safety and performance requirements.

Key Components:
- Vision feature extraction (CLIP-based)
- Controller integration with cache-safe design
- Coverage/consistency monitoring
- Conflict gating for hallucination reduction
- VQA evaluation framework

Cache-Safety Invariant:
- Vision features ONLY fed to controller
- Generator sites remain text-only (W_down/W_O)
- Chunk routing aligned with patch windows
"""

from .vision_encoder import VisionEncoder, VisionFeatures, create_vision_encoder
from .controller_integration import MultimodalController, create_multimodal_controller
from .coverage_analysis import CoverageAnalyzer, ConsistencyGate
from .evaluation import VQAEvaluator, HallucinationMetrics, MultimodalEvaluator
from .preprocessing import VisionPreprocessor, create_vision_preprocessor
from .training import MultimodalTrainer, MultimodalTrainingConfig, create_multimodal_trainer

__all__ = [
    'VisionEncoder',
    'VisionFeatures',
    'create_vision_encoder',
    'MultimodalController', 
    'create_multimodal_controller',
    'CoverageAnalyzer',
    'ConsistencyGate',
    'VQAEvaluator',
    'HallucinationMetrics',
    'MultimodalEvaluator',
    'VisionPreprocessor',
    'create_vision_preprocessor',
    'MultimodalTrainer',
    'MultimodalTrainingConfig',
    'create_multimodal_trainer'
]

__version__ = "2.0.0"