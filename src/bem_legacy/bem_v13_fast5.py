"""
BEM v1.3 Fast-5 Integration Module.

Integrates all Fast-5 variants with the existing v1.1-stable infrastructure
while maintaining cache safety and budget parity.

Fast-5 Variants:
- F5.1: Stateful Router (GRU/SSM over chunk summaries) 
- F5.2: Low-Rank + Diagonal (ΔW = U diag(c) V^T + diag(d))
- F5.3: SVD Warm-Start (initialization from strong static LoRA)
- F5.4: FP8 Generator (QAT for U,V quantization)
- F5.5: Counterfactual Hard-Negatives (robustness training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Import Fast-5 variant components
from .controller.stateful import (
    StatefulRouterConfig, 
    StatefulBEMRouter,
    create_stateful_router_config
)
from .generator.lowrank_diag import (
    LowRankDiagConfig,
    LowRankDiagModule,
    create_lowrank_diag_config
)
from .init.svd_warmstart import (
    SVDWarmStartConfig,
    SVDWarmStartTrainer,
    create_svd_warmstart_config
)
from .quant.fp8_qat import (
    FP8QATConfig,
    FP8BEMModule,
    create_fp8_qat_config
)

# Import existing v1.1 infrastructure
from .bem_v11_stable import (
    BEMv11Module,
    SpectralGovernance,
    AttentionLogitBias,
    validate_cache_safety
)

logger = logging.getLogger(__name__)


@dataclass
class BEMv13Config:
    """Configuration for BEM v1.3 Fast-5 system."""
    
    # Base v1.1 configuration
    sites: List[str] = None  # ["W_O", "W_down"] - cache-safe sites only
    rank_schedule: List[int] = None  # [2, 4, 8, 8, 8, 4, 2]
    num_experts: int = 2
    alpha: float = 16.0
    dropout: float = 0.1
    chunk_size: int = 128
    hysteresis_tau: float = 0.7
    
    # Spectral governance
    max_singular_value: float = 1.0
    fro_budget: float = 1.0
    
    # Fast-5 variant configurations
    fast5_variants: List[str] = None  # Which variants to enable
    
    # F5.1: Stateful Router
    stateful_router_config: Optional[StatefulRouterConfig] = None
    
    # F5.2: Low-Rank + Diagonal  
    lowrank_diag_config: Optional[LowRankDiagConfig] = None
    
    # F5.3: SVD Warm-Start
    svd_warmstart_config: Optional[SVDWarmStartConfig] = None
    
    # F5.4: FP8 Quantization
    fp8_qat_config: Optional[FP8QATConfig] = None
    
    # F5.5: Hard Negatives (handled in training, not architecture)
    
    # Integration settings
    validate_budget_parity: bool = True
    budget_tolerance: float = 0.05  # ±5%
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.sites is None:
            self.sites = ["W_O", "W_down"]  # Cache-safe sites only
            
        if self.rank_schedule is None:
            self.rank_schedule = [2, 4, 8, 8, 8, 4, 2]
            
        if self.fast5_variants is None:
            self.fast5_variants = []


class BEMv13Factory:
    """Factory for creating BEM v1.3 modules with Fast-5 variants."""
    
    @staticmethod
    def create_bem_module(
        base_layer: nn.Linear,
        config: BEMv13Config,
        variant_name: str = "baseline"
    ) -> nn.Module:
        """
        Create appropriate BEM module based on Fast-5 variant.
        
        Args:
            base_layer: Base linear layer to augment
            config: BEM v1.3 configuration
            variant_name: Which Fast-5 variant to create
            
        Returns:
            BEM module with requested Fast-5 enhancements
        """
        
        # Validate cache safety
        validate_cache_safety(config.sites)
        
        if variant_name == "f51_stateful" or "F5.1" in config.fast5_variants:
            return BEMv13Factory._create_f51_stateful(base_layer, config)
        elif variant_name == "f52_lowrank_diag" or "F5.2" in config.fast5_variants:
            return BEMv13Factory._create_f52_lowrank_diag(base_layer, config)
        elif variant_name == "f53_svd_warm" or "F5.3" in config.fast5_variants:
            return BEMv13Factory._create_f53_svd_warm(base_layer, config)
        elif variant_name == "f54_fp8" or "F5.4" in config.fast5_variants:
            return BEMv13Factory._create_f54_fp8(base_layer, config)
        elif variant_name == "f55_hardnegs" or "F5.5" in config.fast5_variants:
            # F5.5 uses standard architecture with enhanced training
            return BEMv13Factory._create_f55_hardnegs(base_layer, config)
        else:
            # Baseline v1.1 compatible module
            return BEMv13Factory._create_baseline(base_layer, config)
    
    @staticmethod
    def _create_f51_stateful(base_layer: nn.Linear, config: BEMv13Config) -> nn.Module:
        """Create F5.1 Stateful Router variant."""
        
        # Create stateful router config if not provided
        if config.stateful_router_config is None:
            config.stateful_router_config = create_stateful_router_config(
                d_feat=base_layer.in_features,
                d_state=64,
                code_dim=max(config.rank_schedule),
                chunk_size=config.chunk_size,
                flip_penalty_beta=0.01
            )
        
        return BEMv13StatefulModule(
            base_layer=base_layer,
            config=config
        )
    
    @staticmethod
    def _create_f52_lowrank_diag(base_layer: nn.Linear, config: BEMv13Config) -> nn.Module:
        """Create F5.2 Low-Rank + Diagonal variant."""
        
        # Create low-rank diagonal config if not provided
        if config.lowrank_diag_config is None:
            # Slightly reduce rank to accommodate diagonal terms
            adjusted_rank = max(1, max(config.rank_schedule) - 1)
            config.lowrank_diag_config = create_lowrank_diag_config(
                rank=adjusted_rank,
                d_max=0.2,
                diagonal_l2_penalty=0.01
            )
        
        return LowRankDiagModule(
            base_layer=base_layer,
            rank=max(config.rank_schedule),
            num_experts=config.num_experts,
            config=config.lowrank_diag_config,
            alpha=config.alpha,
            dropout=config.dropout,
            chunk_size=config.chunk_size,
            hysteresis_tau=config.hysteresis_tau
        )
    
    @staticmethod  
    def _create_f53_svd_warm(base_layer: nn.Linear, config: BEMv13Config) -> nn.Module:
        """Create F5.3 SVD Warm-Start variant."""
        
        # Create SVD warm-start config if not provided
        if config.svd_warmstart_config is None:
            config.svd_warmstart_config = create_svd_warmstart_config(
                rank_schedule=config.rank_schedule,
                controller_init_scale=0.01,
                freeze_bases_steps=1000
            )
        
        # Use standard BEM module but with special initialization
        module = BEMv11Module(
            base_layer=base_layer,
            rank=max(config.rank_schedule),
            num_experts=config.num_experts,
            alpha=config.alpha,
            dropout=config.dropout,
            chunk_size=config.chunk_size,
            hysteresis_tau=config.hysteresis_tau
        )
        
        # Wrap with SVD warm-start capability
        return BEMv13SVDWarmModule(module, config)
    
    @staticmethod
    def _create_f54_fp8(base_layer: nn.Linear, config: BEMv13Config) -> nn.Module:
        """Create F5.4 FP8 Quantized variant."""
        
        # Create FP8 QAT config if not provided
        if config.fp8_qat_config is None:
            config.fp8_qat_config = create_fp8_qat_config(
                numerical_tolerance=1e-3,
                calibration_steps=100
            )
        
        return FP8BEMModule(
            base_layer=base_layer,
            rank=max(config.rank_schedule),
            num_experts=config.num_experts,
            config=config.fp8_qat_config,
            alpha=config.alpha,
            dropout=config.dropout,
            chunk_size=config.chunk_size,
            hysteresis_tau=config.hysteresis_tau
        )
    
    @staticmethod
    def _create_f55_hardnegs(base_layer: nn.Linear, config: BEMv13Config) -> nn.Module:
        """Create F5.5 Hard Negatives variant."""
        
        # F5.5 uses standard architecture with enhanced controller features
        return BEMv13HardNegModule(
            base_layer=base_layer,
            config=config
        )
    
    @staticmethod
    def _create_baseline(base_layer: nn.Linear, config: BEMv13Config) -> nn.Module:
        """Create baseline v1.1 compatible module."""
        
        return BEMv11Module(
            base_layer=base_layer,
            rank=max(config.rank_schedule),
            num_experts=config.num_experts,
            alpha=config.alpha,
            dropout=config.dropout,
            chunk_size=config.chunk_size,
            hysteresis_tau=config.hysteresis_tau,
            max_singular_value=config.max_singular_value,
            fro_budget=config.fro_budget
        )


class BEMv13StatefulModule(nn.Module):
    """BEM v1.3 module with F5.1 Stateful Router."""
    
    def __init__(self, base_layer: nn.Linear, config: BEMv13Config):
        super().__init__()
        self.config = config
        self.base_layer = base_layer
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Create experts (standard LoRA for now)
        from .modules.parallel_lora import ParallelLoRAExpert
        
        self.experts = nn.ModuleList([
            ParallelLoRAExpert(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=max(config.rank_schedule),
                alpha=config.alpha,
                dropout=config.dropout
            )
            for _ in range(config.num_experts)
        ])
        
        # Stateful router
        self.router = StatefulBEMRouter(
            input_dim=base_layer.in_features,
            num_experts=config.num_experts,
            config=config.stateful_router_config,
            chunk_size=config.chunk_size,
            hysteresis_tau=config.hysteresis_tau
        )
        
        # Spectral governance
        self.governance = SpectralGovernance(
            max_singular_value=config.max_singular_value,
            fro_budget=config.fro_budget
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with stateful routing."""
        
        # Base output
        base_output = self.base_layer(x)
        
        # Stateful routing
        routing_weights, expert_indices, aux_info = self.router(x, hidden_state)
        codes = aux_info['codes']
        
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # Standard LoRA forward
            expert_outputs.append(expert_output)
            
        # Combine with routing weights
        expert_stack = torch.stack(expert_outputs, dim=-1)
        routed_output = torch.sum(
            expert_stack * routing_weights.unsqueeze(-2),
            dim=-1
        )
        
        # Final output
        output = base_output + routed_output
        
        result = {
            'output': output,
            'routing_weights': routing_weights,
            'expert_indices': expert_indices,
            'codes': codes,
            'hidden_state': aux_info['hidden_state'],
            'flip_penalty': aux_info['flip_penalty']
        }
        
        if return_details:
            result.update({
                'expert_outputs': expert_outputs,
                'base_output': base_output,
                'routed_output': routed_output,
                'router_metrics': aux_info.get('metrics', {})
            })
            
        return result


class BEMv13SVDWarmModule(nn.Module):
    """BEM v1.3 module with F5.3 SVD Warm-Start initialization."""
    
    def __init__(self, base_module: BEMv11Module, config: BEMv13Config):
        super().__init__()
        self.base_module = base_module
        self.config = config
        self.warmstart_trainer = None
        
        if config.svd_warmstart_config:
            self.warmstart_trainer = SVDWarmStartTrainer(config.svd_warmstart_config)
            
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass (delegates to base module)."""
        return self.base_module(*args, **kwargs)
        
    def initialize_from_lora(self, lora_checkpoint_path: Path):
        """Initialize from SVD decomposition of static LoRA."""
        if self.warmstart_trainer is None:
            raise ValueError("SVD warm-start trainer not configured")
            
        # Prepare SVD initialization
        svd_init_data = self.warmstart_trainer.prepare_from_lora_checkpoint(
            lora_checkpoint_path
        )
        
        # Initialize BEM module
        self.warmstart_trainer.initialize_bem_from_svd(
            self.base_module,
            svd_init_data
        )
        
        logger.info("BEM module initialized with SVD warm-start")
        
    def should_unfreeze_bases(self) -> bool:
        """Check if training phase allows base unfreezing."""
        if self.warmstart_trainer is None:
            return True
        return self.warmstart_trainer.should_unfreeze_bases()
        
    def unfreeze_bases(self):
        """Unfreeze base matrices for fine-tuning."""
        if self.warmstart_trainer is not None:
            self.warmstart_trainer.unfreeze_bases(self.base_module)


class BEMv13HardNegModule(nn.Module):
    """BEM v1.3 module with F5.5 Hard Negatives enhancement."""
    
    def __init__(self, base_layer: nn.Linear, config: BEMv13Config):
        super().__init__()
        self.config = config
        
        # Use standard BEM module as base
        self.base_bem = BEMv11Module(
            base_layer=base_layer,
            rank=max(config.rank_schedule),
            num_experts=config.num_experts,
            alpha=config.alpha,
            dropout=config.dropout,
            chunk_size=config.chunk_size,
            hysteresis_tau=config.hysteresis_tau
        )
        
        # Enhanced features for contradiction/consistency detection
        self.feature_dim_expansion = 64
        self.feature_enhancer = nn.Sequential(
            nn.Linear(base_layer.in_features, base_layer.in_features + self.feature_dim_expansion),
            nn.LayerNorm(base_layer.in_features + self.feature_dim_expansion),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Features for controller
        self.contradiction_score_head = nn.Linear(base_layer.in_features, 1)
        self.consistency_score_head = nn.Linear(base_layer.in_features, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        retrieval_features: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with hard negative features."""
        
        # Enhanced features for routing
        enhanced_x = self.feature_enhancer(x)
        
        # Get base BEM output
        bem_output = self.base_bem(enhanced_x[:, :, :x.size(-1)])  # Trim to original size
        
        # Additional features for hard negative training
        contradiction_scores = None
        consistency_scores = None
        
        if retrieval_features is not None:
            # Compute contradiction/consistency scores
            combined_features = torch.cat([x, retrieval_features], dim=-1) if retrieval_features.size(-1) != x.size(-1) else x * retrieval_features
            
            contradiction_scores = torch.sigmoid(self.contradiction_score_head(combined_features))
            consistency_scores = torch.sigmoid(self.consistency_score_head(combined_features))
            
        result = bem_output.copy()
        
        if return_details:
            result.update({
                'enhanced_features': enhanced_x,
                'contradiction_scores': contradiction_scores,
                'consistency_scores': consistency_scores
            })
            
        return result


def load_bem_v13_from_config(
    config_path: Path,
    base_layers: Dict[str, nn.Linear],
    variant_override: Optional[str] = None
) -> Dict[str, nn.Module]:
    """
    Load BEM v1.3 modules from experiment configuration.
    
    Args:
        config_path: Path to experiment YAML configuration
        base_layers: Dictionary of base layers to augment
        variant_override: Override variant detection from config
        
    Returns:
        Dictionary of BEM modules keyed by layer name
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    # Extract BEM configuration
    bem_config_dict = config_dict.get('model', {}).get('bem_config', {})
    
    # Create BEMv13Config
    config = BEMv13Config(
        sites=bem_config_dict.get('sites', ['W_O', 'W_down']),
        rank_schedule=bem_config_dict.get('rank_schedule', [2, 4, 8, 8, 8, 4, 2]),
        num_experts=bem_config_dict.get('num_experts', 2),
        alpha=bem_config_dict.get('alpha', 16.0),
        dropout=bem_config_dict.get('dropout', 0.1),
        chunk_size=bem_config_dict.get('routing', {}).get('chunk_size', 128),
        hysteresis_tau=bem_config_dict.get('routing', {}).get('hysteresis_tau', 0.7)
    )
    
    # Detect variant from config or use override
    if variant_override:
        variant_name = variant_override
    else:
        fast5_variant = config_dict.get('metadata', {}).get('fast5_variant', 'baseline')
        experiment_id = config_dict.get('metadata', {}).get('experiment_id', '')
        
        if fast5_variant != 'baseline':
            variant_name = fast5_variant.lower().replace('.', '_')
        elif 'f51' in experiment_id:
            variant_name = 'f51_stateful'
        elif 'f52' in experiment_id:
            variant_name = 'f52_lowrank_diag'
        elif 'f53' in experiment_id:
            variant_name = 'f53_svd_warm'
        elif 'f54' in experiment_id:
            variant_name = 'f54_fp8'
        elif 'f55' in experiment_id:
            variant_name = 'f55_hardnegs'
        else:
            variant_name = 'baseline'
    
    logger.info(f"Creating BEM v1.3 modules with variant: {variant_name}")
    
    # Create BEM modules for each base layer
    bem_modules = {}
    
    for layer_name, base_layer in base_layers.items():
        if any(site in layer_name for site in config.sites):
            bem_module = BEMv13Factory.create_bem_module(
                base_layer=base_layer,
                config=config,
                variant_name=variant_name
            )
            bem_modules[layer_name] = bem_module
            logger.info(f"Created {variant_name} BEM module for {layer_name}")
        else:
            logger.warning(f"Skipping {layer_name} - not in cache-safe sites {config.sites}")
            
    return bem_modules


def validate_bem_v13_budget(
    bem_modules: Dict[str, nn.Module],
    baseline_budget: Dict[str, int],
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Validate that BEM v1.3 modules stay within budget parity.
    
    Args:
        bem_modules: Dictionary of BEM modules
        baseline_budget: Baseline parameter/FLOP counts
        tolerance: Allowed deviation (±5% = 0.05)
        
    Returns:
        Validation results
    """
    
    # Count parameters and estimate FLOPs
    total_params = sum(
        sum(p.numel() for p in module.parameters() if p.requires_grad)
        for module in bem_modules.values()
    )
    
    # Rough FLOP estimation (forward pass)
    total_flops = 0
    for module in bem_modules.values():
        if hasattr(module, 'get_budget_info'):
            budget_info = module.get_budget_info()
            total_flops += budget_info.get('total_flops', 0)
        else:
            # Fallback estimation
            total_flops += total_params * 2  # Rough approximation
            
    # Check against baseline
    param_ratio = total_params / baseline_budget.get('params', 1)
    flop_ratio = total_flops / baseline_budget.get('flops', 1)
    
    param_delta = (param_ratio - 1.0) * 100
    flop_delta = (flop_ratio - 1.0) * 100
    
    passes_param_gate = abs(param_delta) <= (tolerance * 100)
    passes_flop_gate = abs(flop_delta) <= (tolerance * 100)
    
    return {
        'total_params': total_params,
        'total_flops': total_flops,
        'param_ratio': param_ratio,
        'flop_ratio': flop_ratio,
        'param_delta_pct': param_delta,
        'flop_delta_pct': flop_delta,
        'passes_param_gate': passes_param_gate,
        'passes_flop_gate': passes_flop_gate,
        'passes_overall': passes_param_gate and passes_flop_gate,
        'violations': [] if (passes_param_gate and passes_flop_gate) else [
            f"Param delta {param_delta:+.2f}% > ±{tolerance*100}%" if not passes_param_gate else "",
            f"FLOP delta {flop_delta:+.2f}% > ±{tolerance*100}%" if not passes_flop_gate else ""
        ]
    }


# Export public interface
__all__ = [
    'BEMv13Config',
    'BEMv13Factory', 
    'BEMv13StatefulModule',
    'BEMv13SVDWarmModule',
    'BEMv13HardNegModule',
    'load_bem_v13_from_config',
    'validate_bem_v13_budget'
]