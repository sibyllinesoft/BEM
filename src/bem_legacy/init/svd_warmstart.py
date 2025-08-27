"""
F5.3 - SVD Warm-Start initialization from strong static LoRA.

Mechanism: Train a strong static LoRA; SVD its ΔW per site/layer to initialize U,V;
initialize controller near zero and ramp.

Why: Better identifiability; faster convergence; controller learns *when* not *what*.
Budget: Same inference cost; training faster.
Gate: Same/↑ quality with fewer steps; smoother ΔW spectra; stability ↑.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SVDWarmStartConfig:
    """Configuration for SVD warm-start initialization."""
    rank_schedule: List[int] = None  # Rank per layer [2,4,8,8,8,4,2]
    truncation_threshold: float = 1e-6  # SVD singular value threshold
    scaling_preservation: bool = True    # Preserve LoRA scaling factors
    controller_init_scale: float = 0.01  # Initial controller output scale
    freeze_bases_steps: int = 1000      # Steps to freeze U,V while training controller
    ramp_controller_steps: int = 2000   # Steps to ramp up controller influence
    spectral_regularization: float = 0.1 # Spectral regularization during training
    save_decomposition: bool = True     # Save SVD decomposition for analysis
    verify_reconstruction: bool = True  # Verify reconstruction quality


class LoRACheckpointLoader:
    """Loads and analyzes LoRA checkpoints for SVD decomposition."""
    
    def __init__(self, checkpoint_path: Union[str, Path]):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = None
        self.lora_weights = {}
        
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load LoRA checkpoint and extract weights."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        logger.info(f"Loading LoRA checkpoint from {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract LoRA weights from state dict
        state_dict = self.checkpoint.get('model_state_dict', self.checkpoint.get('state_dict', self.checkpoint))
        
        self.lora_weights = self._extract_lora_weights(state_dict)
        logger.info(f"Extracted {len(self.lora_weights)} LoRA weight pairs")
        
        return self.checkpoint
        
    def _extract_lora_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract LoRA A and B weights from state dict."""
        lora_weights = {}
        
        # Group weights by layer/module name
        for key, tensor in state_dict.items():
            if 'lora_A' in key or 'lora_B' in key:
                # Extract base name (remove .lora_A or .lora_B suffix)
                if '.lora_A' in key:
                    base_name = key.replace('.lora_A.weight', '')
                    weight_type = 'A'
                elif '.lora_B' in key:
                    base_name = key.replace('.lora_B.weight', '')
                    weight_type = 'B'
                else:
                    continue
                    
                if base_name not in lora_weights:
                    lora_weights[base_name] = {}
                    
                lora_weights[base_name][weight_type] = tensor
                
        # Filter out incomplete pairs
        complete_pairs = {}
        for name, weights in lora_weights.items():
            if 'A' in weights and 'B' in weights:
                complete_pairs[name] = weights
            else:
                logger.warning(f"Incomplete LoRA pair for {name}, skipping")
                
        return complete_pairs
        
    def get_effective_weights(self) -> Dict[str, torch.Tensor]:
        """Compute effective ΔW = B @ A for each LoRA pair."""
        effective_weights = {}
        
        for name, weights in self.lora_weights.items():
            A = weights['A']  # [in_features, rank]
            B = weights['B']  # [out_features, rank]
            
            # Compute ΔW = B @ A^T
            delta_W = B @ A.T  # [out_features, in_features]
            effective_weights[name] = delta_W
            
            logger.debug(f"Layer {name}: ΔW shape {delta_W.shape}, norm {delta_W.norm():.4f}")
            
        return effective_weights


class SVDDecomposer:
    """Performs SVD decomposition of LoRA weight matrices."""
    
    def __init__(self, config: SVDWarmStartConfig):
        self.config = config
        self.decompositions = {}
        
    def decompose_weights(
        self, 
        effective_weights: Dict[str, torch.Tensor],
        rank_schedule: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Perform SVD decomposition on effective weights.
        
        Args:
            effective_weights: Dictionary of ΔW matrices
            rank_schedule: Optional rank per layer, defaults to config
            
        Returns:
            Dictionary containing U, Σ, V decompositions per layer
        """
        if rank_schedule is None:
            rank_schedule = self.config.rank_schedule
            
        if rank_schedule and len(rank_schedule) != len(effective_weights):
            logger.warning(
                f"Rank schedule length {len(rank_schedule)} doesn't match "
                f"number of layers {len(effective_weights)}. Using default ranks."
            )
            rank_schedule = None
            
        decompositions = {}
        layer_names = sorted(effective_weights.keys())
        
        for i, layer_name in enumerate(layer_names):
            delta_W = effective_weights[layer_name]
            
            # Determine target rank
            if rank_schedule:
                target_rank = rank_schedule[i]
            else:
                # Use 1/2 of minimum dimension as default
                target_rank = min(delta_W.shape) // 2
                
            # Perform SVD
            U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
            
            # Apply truncation threshold
            significant_sv = S > self.config.truncation_threshold
            n_significant = significant_sv.sum().item()
            
            # Use minimum of target rank and significant singular values
            effective_rank = min(target_rank, n_significant)
            
            if effective_rank == 0:
                logger.warning(f"No significant singular values for {layer_name}, using rank 1")
                effective_rank = 1
                
            # Truncate to effective rank
            U_trunc = U[:, :effective_rank]                    # [out_features, rank]
            S_trunc = S[:effective_rank]                       # [rank]
            V_trunc = Vt[:effective_rank, :].T                 # [in_features, rank]
            
            # Incorporate singular values into U for initialization
            # This preserves the magnitude information
            U_scaled = U_trunc * S_trunc.sqrt()               # [out_features, rank]
            V_scaled = V_trunc * S_trunc.sqrt()               # [in_features, rank]
            
            decomposition = {
                'U': U_scaled,
                'V': V_scaled, 
                'S': S_trunc,
                'rank': effective_rank,
                'original_shape': delta_W.shape,
                'reconstruction_error': self._compute_reconstruction_error(
                    delta_W, U_scaled, V_scaled
                )
            }
            
            decompositions[layer_name] = decomposition
            
            logger.info(
                f"Layer {layer_name}: rank {effective_rank}, "
                f"reconstruction error {decomposition['reconstruction_error']:.6f}"
            )
            
        self.decompositions = decompositions
        return decompositions
        
    def _compute_reconstruction_error(
        self, 
        original: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor
    ) -> float:
        """Compute relative reconstruction error."""
        reconstruction = U @ V.T
        error = torch.norm(original - reconstruction, p='fro')
        original_norm = torch.norm(original, p='fro')
        return (error / (original_norm + 1e-8)).item()
        
    def analyze_spectra(self) -> Dict[str, Dict[str, float]]:
        """Analyze spectral properties of decompositions."""
        spectra_analysis = {}
        
        for layer_name, decomp in self.decompositions.items():
            S = decomp['S']
            
            analysis = {
                'spectral_norm': S.max().item(),
                'spectral_gap': (S[0] - S[1]).item() if len(S) > 1 else S[0].item(),
                'effective_rank': (S.sum() / S.max()).item(),  # Participation ratio
                'condition_number': (S.max() / (S.min() + 1e-8)).item(),
                'singular_value_decay': self._compute_sv_decay_rate(S)
            }
            
            spectra_analysis[layer_name] = analysis
            
        return spectra_analysis
        
    def _compute_sv_decay_rate(self, S: torch.Tensor) -> float:
        """Compute singular value decay rate (slope in log space)."""
        if len(S) < 2:
            return 0.0
            
        log_S = torch.log(S + 1e-8)
        indices = torch.arange(len(S), dtype=torch.float32)
        
        # Linear fit in log space
        A = torch.stack([indices, torch.ones(len(S))], dim=1)
        slope, _ = torch.linalg.lstsq(A, log_S, rcond=None).solution
        
        return slope.item()


class ControllerInitializer:
    """Initializes controller networks for warm-start training."""
    
    def __init__(self, config: SVDWarmStartConfig):
        self.config = config
        
    def create_controller_schedule(self, num_steps: int) -> torch.Tensor:
        """
        Create ramping schedule for controller influence.
        
        Args:
            num_steps: Total training steps
            
        Returns:
            schedule: Controller scaling schedule [num_steps]
        """
        schedule = torch.zeros(num_steps)
        
        # Phase 1: Frozen (controller outputs near zero)
        freeze_steps = min(self.config.freeze_bases_steps, num_steps)
        schedule[:freeze_steps] = self.config.controller_init_scale
        
        # Phase 2: Ramp up
        ramp_steps = min(self.config.ramp_controller_steps, num_steps - freeze_steps)
        if ramp_steps > 0:
            ramp_end_idx = freeze_steps + ramp_steps
            ramp_values = torch.linspace(
                self.config.controller_init_scale, 1.0, ramp_steps
            )
            schedule[freeze_steps:ramp_end_idx] = ramp_values
            
        # Phase 3: Full controller
        schedule[freeze_steps + ramp_steps:] = 1.0
        
        return schedule
        
    def initialize_controller_near_zero(self, controller: nn.Module):
        """Initialize controller to output near-zero codes initially."""
        for module in controller.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with small std
                nn.init.normal_(module.weight, std=self.config.controller_init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
                # Initialize RNN weights
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.normal_(param, std=self.config.controller_init_scale)
                    elif 'bias' in name:
                        nn.init.zeros_(param)


class SVDWarmStartTrainer:
    """Manages SVD warm-start training protocol."""
    
    def __init__(self, config: SVDWarmStartConfig):
        self.config = config
        self.decomposer = SVDDecomposer(config)
        self.controller_init = ControllerInitializer(config)
        self.current_step = 0
        self.controller_schedule = None
        
    def prepare_from_lora_checkpoint(
        self, 
        checkpoint_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Prepare SVD initialization from LoRA checkpoint.
        
        Args:
            checkpoint_path: Path to LoRA checkpoint
            output_path: Optional path to save SVD initialization
            
        Returns:
            Dictionary containing initialization data
        """
        # Load LoRA checkpoint
        loader = LoRACheckpointLoader(checkpoint_path)
        loader.load_checkpoint()
        
        # Get effective weights
        effective_weights = loader.get_effective_weights()
        
        # Perform SVD decomposition
        decompositions = self.decomposer.decompose_weights(effective_weights)
        
        # Analyze spectra
        spectra_analysis = self.decomposer.analyze_spectra()
        
        # Prepare initialization data
        init_data = {
            'decompositions': decompositions,
            'spectra_analysis': spectra_analysis,
            'config': self.config,
            'source_checkpoint': str(checkpoint_path),
            'lora_weights': loader.lora_weights
        }
        
        # Verify reconstruction if requested
        if self.config.verify_reconstruction:
            verification_results = self._verify_reconstructions(
                effective_weights, decompositions
            )
            init_data['verification'] = verification_results
            
        # Save initialization data
        if output_path:
            self.save_initialization(init_data, output_path)
            
        logger.info("SVD warm-start preparation completed")
        return init_data
        
    def _verify_reconstructions(
        self,
        original_weights: Dict[str, torch.Tensor],
        decompositions: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Verify reconstruction quality."""
        verification = {}
        
        for layer_name in original_weights:
            if layer_name not in decompositions:
                continue
                
            original = original_weights[layer_name]
            decomp = decompositions[layer_name]
            
            U, V = decomp['U'], decomp['V']
            reconstruction = U @ V.T
            
            # Compute various error metrics
            l2_error = torch.norm(original - reconstruction, p=2).item()
            fro_error = torch.norm(original - reconstruction, p='fro').item()
            relative_error = fro_error / (torch.norm(original, p='fro').item() + 1e-8)
            
            verification[layer_name] = {
                'l2_error': l2_error,
                'frobenius_error': fro_error,
                'relative_error': relative_error,
                'max_element_error': torch.max(torch.abs(original - reconstruction)).item()
            }
            
        return verification
        
    def initialize_bem_from_svd(
        self,
        bem_module: nn.Module,
        svd_init_data: Dict[str, Any],
        freeze_bases: bool = True
    ):
        """
        Initialize BEM module using SVD decomposition.
        
        Args:
            bem_module: BEM module to initialize
            svd_init_data: SVD initialization data
            freeze_bases: Whether to freeze U,V initially
        """
        decompositions = svd_init_data['decompositions']
        
        # Map BEM expert parameters to SVD decompositions
        expert_mapping = self._map_experts_to_decompositions(bem_module, decompositions)
        
        # Initialize expert weights
        for expert_idx, expert in enumerate(bem_module.experts):
            if expert_idx < len(expert_mapping):
                layer_name, decomp = expert_mapping[expert_idx]
                
                with torch.no_grad():
                    # Initialize LoRA matrices from SVD
                    expert.lora_A.weight.copy_(decomp['V'])  # [in_features, rank]
                    expert.lora_B.weight.copy_(decomp['U'].T)  # [rank, out_features]
                    
                    # Freeze if requested
                    if freeze_bases:
                        expert.lora_A.weight.requires_grad_(False)
                        expert.lora_B.weight.requires_grad_(False)
                        
                logger.info(f"Initialized expert {expert_idx} from {layer_name}")
                
        # Initialize controller near zero
        if hasattr(bem_module, 'router') and hasattr(bem_module.router, 'stateful_router'):
            self.controller_init.initialize_controller_near_zero(
                bem_module.router.stateful_router
            )
            
        # Set up training schedule
        self.controller_schedule = self.controller_init.create_controller_schedule(10000)  # Default steps
        
        logger.info("BEM module initialized with SVD warm-start")
        
    def _map_experts_to_decompositions(
        self,
        bem_module: nn.Module,
        decompositions: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
        """Map BEM experts to SVD decompositions."""
        # Simple mapping: assign decompositions to experts in order
        decomp_items = list(decompositions.items())
        expert_mapping = []
        
        num_experts = len(bem_module.experts)
        for i in range(num_experts):
            if i < len(decomp_items):
                expert_mapping.append(decomp_items[i])
            else:
                # Reuse decompositions if more experts than layers
                expert_mapping.append(decomp_items[i % len(decomp_items)])
                
        return expert_mapping
        
    def get_current_controller_scale(self) -> float:
        """Get current controller scaling factor."""
        if self.controller_schedule is None:
            return 1.0
            
        step_idx = min(self.current_step, len(self.controller_schedule) - 1)
        return self.controller_schedule[step_idx].item()
        
    def step(self):
        """Advance training step counter."""
        self.current_step += 1
        
    def should_unfreeze_bases(self) -> bool:
        """Check if bases should be unfrozen."""
        return self.current_step >= self.config.freeze_bases_steps
        
    def unfreeze_bases(self, bem_module: nn.Module):
        """Unfreeze base matrices for fine-tuning."""
        for expert in bem_module.experts:
            expert.lora_A.weight.requires_grad_(True)
            expert.lora_B.weight.requires_grad_(True)
        logger.info("Unfroze base matrices for fine-tuning")
        
    def save_initialization(
        self,
        init_data: Dict[str, Any],
        output_path: Union[str, Path]
    ):
        """Save SVD initialization data."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to serializable format
        serializable_data = self._make_serializable(init_data)
        
        torch.save(serializable_data, output_path)
        logger.info(f"Saved SVD initialization to {output_path}")
        
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to serializable format."""
        if isinstance(data, torch.Tensor):
            return data.clone()
        elif isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_serializable(item) for item in data]
        else:
            return data


def load_svd_initialization(init_path: Union[str, Path]) -> Dict[str, Any]:
    """Load SVD initialization data."""
    return torch.load(init_path, map_location='cpu')


def create_svd_warmstart_config(**kwargs) -> SVDWarmStartConfig:
    """Factory function to create SVDWarmStartConfig with validation."""
    return SVDWarmStartConfig(**kwargs)


def main():
    """CLI interface for SVD warm-start preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SVD warm-start from LoRA checkpoint")
    parser.add_argument("--input", "-i", required=True, help="Input LoRA checkpoint path")
    parser.add_argument("--output", "-o", required=True, help="Output SVD initialization path")
    parser.add_argument("--rank-schedule", nargs="+", type=int, help="Rank schedule per layer")
    parser.add_argument("--threshold", type=float, default=1e-6, help="SVD truncation threshold")
    parser.add_argument("--verify", action="store_true", help="Verify reconstruction quality")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SVDWarmStartConfig(
        rank_schedule=args.rank_schedule,
        truncation_threshold=args.threshold,
        verify_reconstruction=args.verify
    )
    
    # Prepare SVD initialization
    trainer = SVDWarmStartTrainer(config)
    trainer.prepare_from_lora_checkpoint(args.input, args.output)
    
    print(f"SVD warm-start initialization saved to {args.output}")


if __name__ == "__main__":
    main()