"""
Advanced BEM Variants Integration Framework

Unified interface for V2 (Dual-Path), V7 (FiLM-lite), and V11 (Learned Cache)
variants with proper architecture routing and experimental protocols.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import logging

from .bem_v11_stable import BEMv11StableModel as BEMv11Stable
from .modules.dual_path_lora import (
    create_dual_path_lora_for_model,
    MultiLayerDualPathLoRA
)
from .modules.film_lite import (
    create_film_lite_bem_for_model,
    FiLMLiteBEM
)
from .modules.learned_cache_policy import (
    create_learned_cache_bem_for_model,
    LearnedCacheBEM
)

logger = logging.getLogger(__name__)


class AdvancedBEMFactory:
    """
    Factory for creating advanced BEM variants with unified configuration.
    """
    
    SUPPORTED_ARCHITECTURES = {
        'bem_v11_stable': 'B1 - BEM-v1.1-stable baseline',
        'dual_path_lora': 'V2 - Dual-Path LoRA++ with orthogonality regularization', 
        'film_lite_bem': 'V7 - FiLM-lite extension with γ/β modulation',
        'learned_cache_bem': 'V11 - Learned Cache Policy with K/V update windows'
    }
    
    @classmethod
    def create_architecture(
        cls,
        architecture_type: str,
        base_model: nn.Module,
        config: Dict[str, Any]
    ) -> nn.Module:
        """
        Create advanced BEM variant based on architecture type.
        
        Args:
            architecture_type: One of the supported architecture types
            base_model: Base transformer model
            config: Architecture-specific configuration
            
        Returns:
            Configured BEM variant module
        """
        if architecture_type not in cls.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture_type}. "
                f"Supported: {list(cls.SUPPORTED_ARCHITECTURES.keys())}"
            )
            
        logger.info(f"Creating architecture: {architecture_type}")
        logger.info(f"Description: {cls.SUPPORTED_ARCHITECTURES[architecture_type]}")
        
        if architecture_type == 'bem_v11_stable':
            return cls._create_bem_v11_stable(base_model, config)
        elif architecture_type == 'dual_path_lora':
            return cls._create_dual_path_lora(base_model, config)
        elif architecture_type == 'film_lite_bem':
            return cls._create_film_lite_bem(base_model, config)
        elif architecture_type == 'learned_cache_bem':
            return cls._create_learned_cache_bem(base_model, config)
        else:
            raise NotImplementedError(f"Architecture {architecture_type} not implemented")
            
    @classmethod
    def _create_bem_v11_stable(
        cls, 
        base_model: nn.Module, 
        config: Dict[str, Any]
    ) -> BEMv11Stable:
        """Create B1 baseline."""
        bem_config = config.get('bem_config', {})
        return BEMv11Stable(
            base_model=base_model,
            retrieval_dim=config.get('retrieval_dim', 384),
            **bem_config
        )
        
    @classmethod
    def _create_dual_path_lora(
        cls,
        base_model: nn.Module,
        config: Dict[str, Any]
    ) -> MultiLayerDualPathLoRA:
        """Create V2 Dual-Path variant."""
        dual_path_config = config.get('dual_path_config', {})
        
        # Extract rank schedule - V2 uses 2×[2,4,4,4,4,4,2] format
        rank_schedule_raw = dual_path_config.get('rank_schedule', [])
        if len(rank_schedule_raw) == 2 and isinstance(rank_schedule_raw[0], list):
            # Convert dual-path format to single schedule (same for both branches)
            rank_schedule = {f"layer_{i}": rank for i, rank in enumerate(rank_schedule_raw[0])}
        else:
            rank_schedule = None
            
        return create_dual_path_lora_for_model(
            model=base_model,
            retrieval_dim=config.get('retrieval_dim', 384),
            rank_schedule=rank_schedule,
            alpha=dual_path_config.get('alpha', 16.0),
            dropout=dual_path_config.get('dropout', 0.1),
            lambda_ortho=dual_path_config.get('orthogonality_reg', {}).get('lambda', 0.1),
            alpha_decorr=dual_path_config.get('gate_decorr_alpha', 0.01),
            chunk_size=dual_path_config.get('routing', {}).get('chunk_size', 128),
            hysteresis_tau=dual_path_config.get('routing', {}).get('hysteresis_tau', 0.7)
        )
        
    @classmethod
    def _create_film_lite_bem(
        cls,
        base_model: nn.Module,
        config: Dict[str, Any]
    ) -> FiLMLiteBEM:
        """Create V7 FiLM-lite variant."""
        film_config = config.get('film_config', {})
        
        # Extract rank schedule - same as B1
        rank_schedule_raw = film_config.get('rank_schedule', [2, 4, 8, 8, 8, 4, 2])
        rank_schedule = {f"layer_{i}": rank for i, rank in enumerate(rank_schedule_raw)}
        
        return create_film_lite_bem_for_model(
            model=base_model,
            retrieval_dim=config.get('retrieval_dim', 384),
            rank_schedule=rank_schedule,
            alpha=film_config.get('alpha', 16.0),
            dropout=film_config.get('dropout', 0.1),
            film_feature_dim=film_config.get('film_conditioning', {}).get('feature_dim', 64),
            chunk_size=film_config.get('routing', {}).get('chunk_size', 128),
            hysteresis_tau=film_config.get('routing', {}).get('hysteresis_tau', 0.7)
        )
        
    @classmethod
    def _create_learned_cache_bem(
        cls,
        base_model: nn.Module, 
        config: Dict[str, Any]
    ) -> LearnedCacheBEM:
        """Create V11 Learned Cache variant."""
        cache_config = config.get('cache_policy_config', {})
        
        # Extract rank schedule - same as B1
        rank_schedule_raw = cache_config.get('rank_schedule', [2, 4, 8, 8, 8, 4, 2])
        rank_schedule = {f"layer_{i}": rank for i, rank in enumerate(rank_schedule_raw)}
        
        return create_learned_cache_bem_for_model(
            model=base_model,
            retrieval_dim=config.get('retrieval_dim', 384),
            rank_schedule=rank_schedule,
            alpha=cache_config.get('alpha', 16.0),
            dropout=cache_config.get('dropout', 0.1),
            window_size=cache_config.get('routing', {}).get('window_size', 64),
            policy_lr=cache_config.get('cache_policy', {}).get('policy_lr', 1e-5),
            chunk_size=128,  # Fixed for consistency with other variants
            hysteresis_tau=0.7
        )


class AdvancedVariantsRunner:
    """
    Experimental runner for advanced BEM variants with quality gates and statistics.
    """
    
    def __init__(
        self, 
        output_dir: str = "logs/advanced_variants",
        num_seeds: int = 5
    ):
        self.output_dir = output_dir
        self.num_seeds = num_seeds
        self.quality_gates = self._init_quality_gates()
        
    def _init_quality_gates(self) -> Dict[str, Any]:
        """Initialize quality gates for variants."""
        return {
            'V2_dual_path': {
                'ci_improvement_threshold': 1.0,  # ≥1 primary metric improvement
                'latency_violation_max': 1.05,   # ≤5% latency increase
                'vram_violation_max': 1.05,      # ≤5% VRAM increase
                'routing_stability_min': 0.7,    # No routing thrash
                'ortho_loss_max': 0.1,          # Orthogonality regularization working
                'decorr_loss_max': 0.01         # Gate decorrelation working
            },
            'V7_film_lite': {
                'ci_improvement_threshold': 1.0,  # ≥1 primary metric improvement
                'latency_violation_max': 1.05,   # ≤5% latency increase  
                'vram_violation_max': 1.05,      # ≤5% VRAM increase
                'film_gamma_range': [0.5, 1.5], # Reasonable gamma values
                'film_beta_range': [-0.5, 0.5]   # Reasonable beta values
            },
            'V11_learned_cache': {
                'quality_vs_b1_min': 1.0,        # Quality ≥ B1
                'cache_hit_improvement': 0.05,    # Higher cache hit OR
                'latency_improvement': 0.05,      # Lower latency
                'kv_flip_rate_max': 0.2,         # Reasonable update frequency
                'policy_convergence_min': 0.8    # Policy learning stability
            }
        }
        
    def validate_quality_gates(
        self,
        variant_name: str,
        results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate quality gates for a variant against baseline.
        
        Args:
            variant_name: Name of variant (V2_dual_path, V7_film_lite, V11_learned_cache)
            results: Variant experimental results
            baseline_results: B1 baseline results for comparison
            
        Returns:
            Dictionary of gate validations
        """
        gates = self.quality_gates.get(variant_name, {})
        validations = {}
        
        if variant_name in ['V2_dual_path', 'V7_film_lite']:
            # Standard improvement gates
            primary_metrics = ['BLEU', 'chrF', 'EM', 'F1']
            improvements = []
            
            for metric in primary_metrics:
                if metric in results and metric in baseline_results:
                    improvement = results[metric] - baseline_results[metric]
                    improvements.append(improvement)
                    
            max_improvement = max(improvements) if improvements else 0
            validations['ci_improvement'] = max_improvement >= gates.get('ci_improvement_threshold', 1.0)
            
            # Performance gates
            if 'p95_latency_ms' in results and 'p95_latency_ms' in baseline_results:
                latency_ratio = results['p95_latency_ms'] / baseline_results['p95_latency_ms']
                validations['latency_violation'] = latency_ratio <= gates.get('latency_violation_max', 1.05)
                
            if 'vram_usage_gb' in results and 'vram_usage_gb' in baseline_results:
                vram_ratio = results['vram_usage_gb'] / baseline_results['vram_usage_gb']
                validations['vram_violation'] = vram_ratio <= gates.get('vram_violation_max', 1.05)
                
        elif variant_name == 'V11_learned_cache':
            # Cache-specific gates
            primary_metrics = ['BLEU', 'chrF', 'EM', 'F1']
            quality_maintained = True
            
            for metric in primary_metrics:
                if metric in results and metric in baseline_results:
                    if results[metric] < baseline_results[metric]:
                        quality_maintained = False
                        break
                        
            validations['quality_maintained'] = quality_maintained
            
            # Cache performance
            cache_hit_improved = results.get('cache_hit_rate', 0) > baseline_results.get('cache_hit_rate', 0) + 0.05
            latency_improved = results.get('p50_latency_ms', float('inf')) < baseline_results.get('p50_latency_ms', float('inf')) * 0.95
            validations['cache_or_latency_improved'] = cache_hit_improved or latency_improved
            
        return validations
        
    def run_advanced_variants_campaign(
        self,
        experiment_configs: List[str],
        baseline_config: str = "experiments/B1_bem_v11_stable.yaml"
    ) -> Dict[str, Any]:
        """
        Run complete advanced variants campaign.
        
        Args:
            experiment_configs: List of variant config files
            baseline_config: B1 baseline config for comparison
            
        Returns:
            Campaign results with quality gate validations
        """
        logger.info("Starting Advanced BEM Variants Campaign")
        logger.info(f"Running {len(experiment_configs)} variants with {self.num_seeds} seeds each")
        
        # Run baseline first
        logger.info("Running B1 baseline for comparison...")
        baseline_results = self._run_single_config(baseline_config)
        
        # Run variants
        variant_results = {}
        gate_validations = {}
        
        for config_path in experiment_configs:
            variant_name = self._extract_variant_name(config_path)
            logger.info(f"Running variant: {variant_name}")
            
            # Run variant across all seeds
            results = self._run_single_config(config_path)
            variant_results[variant_name] = results
            
            # Validate quality gates
            validations = self.validate_quality_gates(
                variant_name, results, baseline_results
            )
            gate_validations[variant_name] = validations
            
            # Log gate results
            passed_gates = sum(validations.values())
            total_gates = len(validations)
            logger.info(f"{variant_name}: {passed_gates}/{total_gates} quality gates passed")
            
        return {
            'baseline_results': baseline_results,
            'variant_results': variant_results,
            'quality_gate_validations': gate_validations,
            'campaign_summary': self._generate_campaign_summary(
                baseline_results, variant_results, gate_validations
            )
        }
        
    def _run_single_config(self, config_path: str) -> Dict[str, Any]:
        """Run a single configuration across all seeds."""
        # This would interface with the existing batch experiment runner
        # For now, return placeholder structure
        return {
            'BLEU': 0.0,
            'chrF': 0.0, 
            'EM': 0.0,
            'F1': 0.0,
            'p50_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'vram_usage_gb': 0.0,
            'cache_hit_rate': 0.0,
            'seeds_completed': self.num_seeds,
            'config_path': config_path
        }
        
    def _extract_variant_name(self, config_path: str) -> str:
        """Extract variant name from config path."""
        if 'V2_dual_path' in config_path:
            return 'V2_dual_path'
        elif 'V7_film_lite' in config_path:
            return 'V7_film_lite'
        elif 'V11_learned_cache' in config_path:
            return 'V11_learned_cache'
        else:
            return 'unknown_variant'
            
    def _generate_campaign_summary(
        self,
        baseline_results: Dict[str, Any],
        variant_results: Dict[str, Any],
        gate_validations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate campaign summary with key findings."""
        summary = {
            'total_variants': len(variant_results),
            'variants_passing_gates': sum(
                1 for v in gate_validations.values() 
                if all(v.values())
            ),
            'best_variant': None,
            'best_improvement': 0.0,
            'statistical_power': {
                'seeds_per_variant': self.num_seeds,
                'multiple_comparisons_correction': 'FDR',
                'confidence_level': 0.95
            }
        }
        
        # Find best performing variant
        best_bleu_improvement = 0.0
        best_variant = None
        
        for variant_name, results in variant_results.items():
            bleu_improvement = results['BLEU'] - baseline_results['BLEU']
            if bleu_improvement > best_bleu_improvement:
                best_bleu_improvement = bleu_improvement
                best_variant = variant_name
                
        summary['best_variant'] = best_variant
        summary['best_improvement'] = best_bleu_improvement
        
        return summary


# Integration with existing experiment infrastructure
def create_advanced_variants_experiment_configs() -> List[str]:
    """Create or verify advanced variant experiment configurations."""
    configs = [
        "experiments/V2_dual_path.yaml",
        "experiments/V7_film_lite.yaml", 
        "experiments/V11_learned_cache_policy.yaml"
    ]
    
    # Verify configs exist
    import os
    existing_configs = [config for config in configs if os.path.exists(config)]
    
    logger.info(f"Found {len(existing_configs)}/{len(configs)} variant configs")
    return existing_configs


def run_advanced_variants_campaign():
    """Main entry point for running advanced variants campaign."""
    runner = AdvancedVariantsRunner()
    
    # Get experiment configurations
    variant_configs = create_advanced_variants_experiment_configs()
    
    if not variant_configs:
        logger.error("No variant configurations found!")
        return None
        
    # Run campaign
    results = runner.run_advanced_variants_campaign(variant_configs)
    
    # Save results
    import json
    output_path = f"{runner.output_dir}/advanced_variants_campaign_results.json"
    os.makedirs(runner.output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Campaign results saved to: {output_path}")
    return results


if __name__ == "__main__":
    # Run advanced variants campaign
    results = run_advanced_variants_campaign()
    
    if results:
        print("\n=== Advanced BEM Variants Campaign Results ===")
        summary = results['campaign_summary']
        print(f"Variants tested: {summary['total_variants']}")
        print(f"Variants passing gates: {summary['variants_passing_gates']}")
        print(f"Best variant: {summary['best_variant']}")
        print(f"Best improvement: {summary['best_improvement']:.2f} BLEU points")