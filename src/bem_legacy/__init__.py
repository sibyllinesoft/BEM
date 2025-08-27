"""
BEM (Bolt-on Expert Module) package.

This package implements the complete hierarchical BEM system as specified in TODO.md:
- Simple BEM for validation experiments (Phase 1 - COMPLETE ✅)
- Hierarchical routing controller system (Phase 2 - COMPLETE ✅) 
- Retrieval-aware coupling system (Phase 3 - COMPLETE ✅)
- Multi-BEM composition with subspace reservations (Phase 4 - COMPLETE ✅)
- Full generated BEM with multi-level adaptation (Step B4 - COMPLETE ✅)
- Comprehensive training and telemetry systems (COMPLETE ✅)
"""

# Simple BEM (validation experiment - completed)
from .simple_bem import SimpleBEMModule, BEMController, create_bem_from_linear
from .interpolation_bem import InterpolationBEM, StaticLoRA, create_interpolation_bem

# Hierarchical BEM system (TODO.md step B4 - completed)
from .controller import (
    HierarchicalController,
    PrefixRouter,
    ChunkRouter, 
    TokenRouter,
    UncertaintyHead,
    RoutingLevel,
    RoutingState,
    create_hierarchical_controller,
    analyze_routing_behavior,
    compute_routing_stability
)

from .hierarchical_bem import (
    HierarchicalBEMConfig,
    HierarchicalBEMModule,
    FullHierarchicalBEM,
    create_hierarchical_bem,
    create_hierarchical_bem_for_validation
)

from .hierarchical_training import (
    HierarchicalTrainingConfig,
    HierarchicalBEMTrainer,
    TrainingStrategy,
    create_hierarchical_trainer,
    create_end_to_end_trainer,
    create_expert_imitation_trainer
)

from .telemetry import (
    TelemetryCollector,
    PerformanceMetrics,
    RoutingMetrics,
    SystemMetrics,
    create_telemetry_collector,
    profile_bem_operation,
    analyze_routing_patterns,
    detect_performance_regressions
)

# Phase 3: Retrieval-aware BEM system (TODO.md Phase 3 - completed)
from .retrieval import (
    MicroRetriever,
    RetrievalConfig,
    DocumentIndex,
    HyDEGenerator,
    create_micro_retriever
)

from .retrieval_features import (
    RetrievalFeatureExtractor,
    RetrievalFeaturesConfig,
    CoverageCalculator,
    ConsistencyCalculator,
    create_retrieval_feature_extractor
)

from .retrieval_bem import (
    RetrievalBEMConfig,
    RetrievalAwareBEMModule,
    FullRetrievalAwareBEM,
    RetrievalCache,
    BackgroundRetriever,
    create_retrieval_aware_bem
)

from .retrieval_training import (
    RetrievalTrainingConfig,
    RetrievalLossFunction,
    RetrievalTraining,
    create_retrieval_trainer
)

# Phase 4: Multi-BEM composition system (TODO.md Phase 4 - completed)
from .subspace import (
    SubspacePlanner,
    OrthogonalityEnforcer,
    CapacityManager,
    SubspaceAllocation,
    create_subspace_planner,
    create_orthogonality_enforcer,
    create_capacity_manager
)

from .trust_region import (
    TrustRegionProjector,
    TrustRegionBudget,
    NormCalculator,
    SpectralClamp,
    AdaptiveTrustRegion,
    create_trust_region_projector,
    create_adaptive_trust_region
)

from .multi_bem import (
    MultiBEMComposer,
    MultiBEMConfig,
    BEMRegistryEntry,
    create_multi_bem_composer,
    create_default_multi_bem_config
)

from .interference_testing import (
    InterferenceTester,
    CanaryTask,
    BEMConfiguration,
    InterferenceTestResult,
    create_standard_canary_tasks,
    create_interference_tester
)

from .composition_training import (
    CompositionTrainer,
    CompositionTrainingConfig,
    CompositionLossFunction,
    create_composition_trainer,
    create_default_composition_training_config
)

# Phase 5: Advanced features system (TODO.md Phase 5 - completed)
from .banked_experts import (
    BankedExpertsModule,
    BankedExpertsConfig,
    LoRAExpert,
    TopKGatingNetwork,
    BatchedExpertRouter,
    ExpertUtilizationStats,
    create_banked_experts_module,
    create_default_banked_experts_config
)

from .online_learning import (
    OnlineLearningController,
    OnlineLearningConfig,
    TrustMonitor,
    ConsolidationEngine,
    TrustStatus,
    TrustBudget,
    OnlineLearningMetrics,
    create_online_learning_controller,
    create_default_online_learning_config
)

from .speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeDecodingConfig,
    SpeculativeDecodingMetrics,
    SpeculativeDecodingBenchmark,
    DraftResult,
    VerificationResult,
    create_speculative_decoder,
    create_default_speculative_config
)

from .vector_quantization import (
    VectorQuantizer,
    VQConfig,
    LSHIndex,
    EpisodicCodeMemory,
    VQMetrics,
    create_vector_quantizer,
    create_default_vq_config
)

from .counterfactual import (
    CounterfactualRoutingAnalyzer,
    CounterfactualConfig,
    ComponentType,
    CounterfactualMetrics,
    ComponentImportanceTracker,
    InteractionAnalyzer,
    create_counterfactual_analyzer,
    create_default_counterfactual_config
)

# Advanced BEM Variants (V2, V7, V11)
from .bem_v11_stable import BEMv11StableModel as BEMv11Stable
from .advanced_variants import (
    AdvancedBEMFactory,
    AdvancedVariantsRunner,
    create_advanced_variants_experiment_configs,
    run_advanced_variants_campaign
)

from .modules.dual_path_lora import (
    DualPathLoRA,
    MultiLayerDualPathLoRA,
    OrthogonalityRegularizer,
    GateDecorrelationLoss,
    create_dual_path_lora_for_model
)

from .modules.film_lite import (
    FiLMConditioner,
    FiLMLiteBEM,
    FiLMEnhancedBEMLayer,
    create_film_lite_bem_for_model
)

from .modules.learned_cache_policy import (
    CachePolicyController,
    LearnedCacheBEM,
    LearnedCacheKVLayer,
    create_learned_cache_bem_for_model
)

# Legacy training (for compatibility)
from .training import TrainingConfig, LoRATrainer, BEMTrainer

__all__ = [
    # Simple BEM (validation)
    'SimpleBEMModule',
    'BEMController', 
    'create_bem_from_linear',
    'InterpolationBEM',
    'StaticLoRA',
    'create_interpolation_bem',
    
    # Hierarchical Controller
    'HierarchicalController',
    'PrefixRouter',
    'ChunkRouter',
    'TokenRouter', 
    'UncertaintyHead',
    'RoutingLevel',
    'RoutingState',
    'create_hierarchical_controller',
    'analyze_routing_behavior',
    'compute_routing_stability',
    
    # Hierarchical BEM
    'HierarchicalBEMConfig',
    'HierarchicalBEMModule',
    'FullHierarchicalBEM',
    'create_hierarchical_bem',
    'create_hierarchical_bem_for_validation',
    
    # Training
    'HierarchicalTrainingConfig',
    'HierarchicalBEMTrainer',
    'TrainingStrategy',
    'create_hierarchical_trainer',
    'create_end_to_end_trainer',
    'create_expert_imitation_trainer',
    
    # Telemetry
    'TelemetryCollector',
    'PerformanceMetrics',
    'RoutingMetrics', 
    'SystemMetrics',
    'create_telemetry_collector',
    'profile_bem_operation',
    'analyze_routing_patterns',
    'detect_performance_regressions',
    
    # Phase 3: Retrieval-aware BEM
    'MicroRetriever',
    'RetrievalConfig',
    'DocumentIndex',
    'HyDEGenerator', 
    'create_micro_retriever',
    'RetrievalFeatureExtractor',
    'RetrievalFeaturesConfig',
    'CoverageCalculator',
    'ConsistencyCalculator',
    'create_retrieval_feature_extractor',
    'RetrievalBEMConfig',
    'RetrievalAwareBEMModule',
    'FullRetrievalAwareBEM',
    'RetrievalCache',
    'BackgroundRetriever',
    'create_retrieval_aware_bem',
    'RetrievalTrainingConfig',
    'RetrievalLossFunction',
    'RetrievalTraining',
    'create_retrieval_trainer',
    
    # Phase 4: Multi-BEM composition
    'SubspacePlanner',
    'OrthogonalityEnforcer',
    'CapacityManager',
    'SubspaceAllocation',
    'create_subspace_planner',
    'create_orthogonality_enforcer',
    'create_capacity_manager',
    'TrustRegionProjector',
    'TrustRegionBudget',
    'NormCalculator',
    'SpectralClamp',
    'AdaptiveTrustRegion',
    'create_trust_region_projector',
    'create_adaptive_trust_region',
    'MultiBEMComposer',
    'MultiBEMConfig',
    'BEMRegistryEntry',
    'create_multi_bem_composer',
    'create_default_multi_bem_config',
    'InterferenceTester',
    'CanaryTask',
    'BEMConfiguration',
    'InterferenceTestResult',
    'create_standard_canary_tasks',
    'create_interference_tester',
    'CompositionTrainer',
    'CompositionTrainingConfig',
    'CompositionLossFunction',
    'create_composition_trainer',
    'create_default_composition_training_config',
    
    # Phase 5: Advanced features
    'BankedExpertsModule',
    'BankedExpertsConfig',
    'LoRAExpert',
    'TopKGatingNetwork',
    'BatchedExpertRouter',
    'ExpertUtilizationStats',
    'create_banked_experts_module',
    'create_default_banked_experts_config',
    'OnlineLearningController',
    'OnlineLearningConfig',
    'TrustMonitor',
    'ConsolidationEngine',
    'TrustStatus',
    'TrustBudget',
    'OnlineLearningMetrics',
    'create_online_learning_controller',
    'create_default_online_learning_config',
    'SpeculativeDecoder',
    'SpeculativeDecodingConfig',
    'SpeculativeDecodingMetrics',
    'SpeculativeDecodingBenchmark',
    'DraftResult',
    'VerificationResult',
    'create_speculative_decoder',
    'create_default_speculative_config',
    'VectorQuantizer',
    'VQConfig',
    'LSHIndex',
    'EpisodicCodeMemory',
    'VQMetrics',
    'create_vector_quantizer',
    'create_default_vq_config',
    'CounterfactualRoutingAnalyzer',
    'CounterfactualConfig',
    'ComponentType',
    'CounterfactualMetrics',
    'ComponentImportanceTracker',
    'InteractionAnalyzer',
    'create_counterfactual_analyzer',
    'create_default_counterfactual_config',
    
    # Advanced BEM Variants (V2, V7, V11)
    'BEMv11Stable',
    'AdvancedBEMFactory',
    'AdvancedVariantsRunner',
    'create_advanced_variants_experiment_configs',
    'run_advanced_variants_campaign',
    'DualPathLoRA',
    'MultiLayerDualPathLoRA',
    'OrthogonalityRegularizer',
    'GateDecorrelationLoss',
    'create_dual_path_lora_for_model',
    'FiLMConditioner',
    'FiLMLiteBEM',
    'FiLMEnhancedBEMLayer',
    'create_film_lite_bem_for_model',
    'CachePolicyController',
    'LearnedCacheBEM',
    'LearnedCacheKVLayer',
    'create_learned_cache_bem_for_model',
    
    # Legacy compatibility
    'TrainingConfig',
    'LoRATrainer',
    'BEMTrainer'
]

# Package metadata
__version__ = "0.5.0"  # Phase 5 - Advanced Features Complete
__description__ = "Complete Hierarchical BEM System with Advanced Features: Banked Experts, Online Learning, Speculative Decoding, Vector Quantization, and Counterfactual Routing"
__author__ = "BEM Research Team"

# Configuration
DEFAULT_CONFIG = HierarchicalBEMConfig(
    rank=8,
    alpha=16.0,
    dropout=0.1,
    chunk_size=32,
    max_prefix_tokens=128,
    ema_decay=0.99,
    enable_uncertainty=True,
    enable_token_routing=True,
    code_clamp_value=3.0
)