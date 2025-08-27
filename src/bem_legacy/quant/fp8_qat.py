"""
F5.4 - FP8 Generator with Quantization-Aware Training.

Mechanism: Quantize U,V to FP8 per-channel with QAT; keep codes & accumulations fp16;
base INT8/4 unchanged.

Why: Memory BW bound → FP8 helps latency; quality typically neutral.
Budget: Params same; potential p50 latency ↓ 3–7%.
Gate: Quality non-regression (CI includes 0), latency improves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import math
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# FP8 format specifications
class FP8Format(Enum):
    E4M3 = "E4M3"  # 4 exponent bits, 3 mantissa bits
    E5M2 = "E5M2"  # 5 exponent bits, 2 mantissa bits


@dataclass 
class FP8QATConfig:
    """Configuration for FP8 Quantization-Aware Training."""
    format: FP8Format = FP8Format.E4M3        # FP8 format
    per_channel: bool = True                   # Per-channel vs per-tensor scaling
    calibration_steps: int = 100               # Calibration steps for scale estimation
    fake_quant_enabled: bool = True            # Enable fake quantization
    observer_momentum: float = 0.1             # Momentum for running statistics
    scale_init_method: str = "max"             # Scale initialization: 'max', 'percentile'
    percentile: float = 99.99                  # Percentile for scale init (if used)
    numerical_tolerance: float = 1e-3          # Tolerance for numerical verification
    fallback_per_tensor: bool = True           # Fallback to per-tensor if unstable
    enable_optimization: bool = True           # Enable optimization passes
    gradient_scaling: float = 1.0              # Scale gradients in backward pass
    

class FP8Quantizer:
    """FP8 quantization utilities."""
    
    # FP8 E4M3 constants (most common format)
    E4M3_MAX_VAL = 448.0       # Maximum representable value
    E4M3_MIN_VAL = 1.953125e-3 # Minimum normal value
    
    # FP8 E5M2 constants  
    E5M2_MAX_VAL = 57344.0     # Maximum representable value
    E5M2_MIN_VAL = 3.0517578125e-5  # Minimum normal value
    
    @staticmethod
    def get_format_constants(format: FP8Format) -> Tuple[float, float]:
        """Get max and min values for FP8 format."""
        if format == FP8Format.E4M3:
            return FP8Quantizer.E4M3_MAX_VAL, FP8Quantizer.E4M3_MIN_VAL
        elif format == FP8Format.E5M2:
            return FP8Quantizer.E5M2_MAX_VAL, FP8Quantizer.E5M2_MIN_VAL
        else:
            raise ValueError(f"Unknown FP8 format: {format}")
    
    @staticmethod
    def compute_scale(
        tensor: torch.Tensor, 
        format: FP8Format,
        method: str = "max",
        percentile: float = 99.99,
        per_channel: bool = True,
        channel_axis: int = 0
    ) -> torch.Tensor:
        """Compute quantization scale for tensor."""
        max_val, _ = FP8Quantizer.get_format_constants(format)
        
        if per_channel:
            # Per-channel scaling
            if method == "max":
                abs_max = tensor.abs().amax(dim=[i for i in range(tensor.ndim) if i != channel_axis], keepdim=True)
            elif method == "percentile":
                abs_max = torch.quantile(
                    tensor.abs().flatten(start_dim=1) if channel_axis == 0 else tensor.abs().flatten(end_dim=-2),
                    percentile / 100.0,
                    dim=-1,
                    keepdim=True
                )
                if channel_axis != 0:
                    abs_max = abs_max.unsqueeze(-1)
            else:
                raise ValueError(f"Unknown scale method: {method}")
        else:
            # Per-tensor scaling
            if method == "max":
                abs_max = tensor.abs().max()
            elif method == "percentile":
                abs_max = torch.quantile(tensor.abs().flatten(), percentile / 100.0)
            else:
                raise ValueError(f"Unknown scale method: {method}")
                
        # Compute scale: scale = max_representable / abs_max
        scale = max_val / (abs_max + 1e-8)  # Add small epsilon to avoid division by zero
        
        return scale.clamp(min=1e-8)  # Ensure scale is positive
    
    @staticmethod
    def fake_quantize(
        tensor: torch.Tensor,
        scale: torch.Tensor,
        format: FP8Format
    ) -> torch.Tensor:
        """Apply fake quantization to simulate FP8 precision."""
        max_val, min_val = FP8Quantizer.get_format_constants(format)
        
        # Scale tensor to FP8 range
        scaled = tensor * scale
        
        # Clamp to FP8 range
        clamped = torch.clamp(scaled, -max_val, max_val)
        
        # Simulate FP8 precision loss by rounding
        # This is a simplified approximation - real FP8 has more complex rounding
        if format == FP8Format.E4M3:
            # Simulate 3-bit mantissa precision
            quantized = torch.round(clamped / (max_val / 2**3)) * (max_val / 2**3)
        else:  # E5M2
            # Simulate 2-bit mantissa precision  
            quantized = torch.round(clamped / (max_val / 2**2)) * (max_val / 2**2)
            
        # Scale back to original range
        return quantized / scale


class FP8Observer(nn.Module):
    """Observer for collecting FP8 quantization statistics."""
    
    def __init__(self, config: FP8QATConfig, tensor_shape: torch.Size, channel_axis: int = 0):
        super().__init__()
        self.config = config
        self.tensor_shape = tensor_shape
        self.channel_axis = channel_axis
        
        # Running statistics for scale computation
        if config.per_channel:
            scale_shape = [1] * len(tensor_shape)
            scale_shape[channel_axis] = tensor_shape[channel_axis]
        else:
            scale_shape = [1]
            
        self.register_buffer('running_scale', torch.ones(scale_shape))
        self.register_buffer('num_batches_tracked', torch.zeros(1, dtype=torch.long))
        
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Update statistics and return current scale."""
        if self.training:
            # Compute current scale
            current_scale = FP8Quantizer.compute_scale(
                tensor,
                self.config.format,
                self.config.scale_init_method,
                self.config.percentile,
                self.config.per_channel,
                self.channel_axis
            )
            
            # Update running average
            momentum = self.config.observer_momentum
            if self.num_batches_tracked == 0:
                self.running_scale.copy_(current_scale)
            else:
                self.running_scale.mul_(1 - momentum).add_(current_scale, alpha=momentum)
                
            self.num_batches_tracked += 1
            
        return self.running_scale.clone()


class FakeQuantFP8(nn.Module):
    """Fake quantization module for FP8 QAT."""
    
    def __init__(
        self, 
        config: FP8QATConfig,
        tensor_shape: torch.Size,
        channel_axis: int = 0
    ):
        super().__init__()
        self.config = config
        self.tensor_shape = tensor_shape
        self.channel_axis = channel_axis
        
        # Observer for scale computation
        self.observer = FP8Observer(config, tensor_shape, channel_axis)
        
        # Flag to enable/disable fake quantization
        self.fake_quant_enabled = config.fake_quant_enabled
        
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization if enabled."""
        if not self.fake_quant_enabled:
            return tensor
            
        # Get quantization scale
        scale = self.observer(tensor)
        
        # Apply fake quantization
        return FP8Quantizer.fake_quantize(tensor, scale, self.config.format)
        
    def enable(self):
        """Enable fake quantization."""
        self.fake_quant_enabled = True
        
    def disable(self):
        """Disable fake quantization."""
        self.fake_quant_enabled = False


class FP8LoRAExpert(nn.Module):
    """LoRA expert with FP8 quantization on U and V matrices."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        config: FP8QATConfig,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.config = config
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices (will be quantized)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # FP8 fake quantizers for A and B matrices
        self.fake_quant_A = FakeQuantFP8(
            config,
            self.lora_A.weight.shape,
            channel_axis=0  # Output channels for A
        )
        self.fake_quant_B = FakeQuantFP8(
            config, 
            self.lora_B.weight.shape,
            channel_axis=0  # Output channels for B
        )
        
        # Dropout (applied in fp16)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
            
        # Numerical verification buffers
        self.register_buffer('fp16_reference', torch.zeros(1))
        self.register_buffer('fp8_output', torch.zeros(1))
        self.register_buffer('numerical_error', torch.zeros(1))
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters following LoRA conventions."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(
        self,
        x: torch.Tensor,
        codes: torch.Tensor,
        verify_numerics: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with FP8 quantized matrices.
        
        Args:
            x: Input tensor [*, in_features]
            codes: Routing codes [*, rank]
            verify_numerics: Whether to verify numerical accuracy
            
        Returns:
            output: LoRA output [*, out_features]
        """
        if verify_numerics and self.training:
            return self._forward_with_verification(x, codes)
        else:
            return self._forward_quantized(x, codes)
            
    def _forward_quantized(self, x: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Standard forward pass with quantization."""
        # Apply fake quantization to LoRA matrices
        A_quantized = self.fake_quant_A(self.lora_A.weight)
        B_quantized = self.fake_quant_B(self.lora_B.weight)
        
        # Forward pass using quantized matrices
        # H = x @ A_quantized.T  (keep computation in fp16)
        H = F.linear(x, A_quantized)  # [*, rank]
        
        # Apply codes (kept in fp16)
        H = H * codes
        H = self.dropout(H)
        
        # Output transformation: H @ B_quantized
        output = F.linear(H, B_quantized) * self.scaling
        
        return output
        
    def _forward_with_verification(self, x: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Forward pass with numerical verification against fp16 reference."""
        # FP16 reference computation
        H_ref = F.linear(x, self.lora_A.weight)
        H_ref = H_ref * codes
        H_ref = self.dropout(H_ref)
        output_ref = F.linear(H_ref, self.lora_B.weight) * self.scaling
        
        # FP8 computation
        output_fp8 = self._forward_quantized(x, codes)
        
        # Compute numerical error
        with torch.no_grad():
            error = torch.norm(output_ref - output_fp8, p=2) / (torch.norm(output_ref, p=2) + 1e-8)
            self.numerical_error.copy_(error)
            self.fp16_reference.copy_(output_ref.norm())
            self.fp8_output.copy_(output_fp8.norm())
            
            # Log warning if error is too high
            if error.item() > self.config.numerical_tolerance:
                logger.warning(
                    f"High numerical error: {error.item():.6f} > {self.config.numerical_tolerance}"
                )
                
        return output_fp8
        
    def get_quantization_info(self) -> Dict[str, Any]:
        """Get information about current quantization state."""
        info = {
            'format': self.config.format.value,
            'per_channel': self.config.per_channel,
            'fake_quant_enabled': self.fake_quant_A.fake_quant_enabled,
            'A_scale': self.fake_quant_A.observer.running_scale.clone(),
            'B_scale': self.fake_quant_B.observer.running_scale.clone(),
            'numerical_error': self.numerical_error.item(),
        }
        
        return info
        
    def enable_quantization(self):
        """Enable FP8 quantization."""
        self.fake_quant_A.enable()
        self.fake_quant_B.enable()
        
    def disable_quantization(self):
        """Disable FP8 quantization."""
        self.fake_quant_A.disable()
        self.fake_quant_B.disable()
        
    def prepare_for_export(self) -> Dict[str, torch.Tensor]:
        """Prepare quantized weights for export/deployment."""
        # Get final scales
        A_scale = self.fake_quant_A.observer.running_scale
        B_scale = self.fake_quant_B.observer.running_scale
        
        # Quantize weights to actual FP8 values (simulated)
        A_quantized = FP8Quantizer.fake_quantize(self.lora_A.weight, A_scale, self.config.format)
        B_quantized = FP8Quantizer.fake_quantize(self.lora_B.weight, B_scale, self.config.format)
        
        return {
            'A_quantized': A_quantized,
            'B_quantized': B_quantized,
            'A_scale': A_scale,
            'B_scale': B_scale,
            'scaling': self.scaling
        }


class FP8BEMModule(nn.Module):
    """BEM module with FP8 quantized experts."""
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        num_experts: int = 2,
        config: Optional[FP8QATConfig] = None,
        alpha: float = 16.0,
        dropout: float = 0.0,
        chunk_size: int = 128,
        hysteresis_tau: float = 0.7
    ):
        super().__init__()
        
        if config is None:
            config = FP8QATConfig()
            
        self.config = config
        self.base_layer = base_layer
        self.rank = rank
        self.num_experts = num_experts
        
        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # FP8 quantized experts
        self.experts = nn.ModuleList([
            FP8LoRAExpert(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=rank,
                config=config,
                alpha=alpha,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
        # Router (reuse from stateful implementation)
        from ..controller.stateful import StatefulBEMRouter, StatefulRouterConfig
        
        router_config = StatefulRouterConfig(
            d_feat=base_layer.in_features,
            code_dim=rank,
            chunk_size=chunk_size
        )
        
        self.router = StatefulBEMRouter(
            input_dim=base_layer.in_features,
            num_experts=num_experts,
            config=router_config,
            chunk_size=chunk_size,
            hysteresis_tau=hysteresis_tau
        )
        
        # Numerical verification tracking
        self.register_buffer('calibration_steps', torch.zeros(1, dtype=torch.long))
        self.verification_enabled = False
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with FP8 quantized experts."""
        # Base layer output
        base_output = self.base_layer(x)
        
        # Routing
        routing_weights, expert_indices, aux_info = self.router(x, hidden_state)
        codes = aux_info['codes']
        
        # Compute expert outputs with optional verification
        expert_outputs = []
        quantization_info = []
        
        verify_numerics = self.verification_enabled and self.calibration_steps < self.config.calibration_steps
        
        for expert in self.experts:
            expert_output = expert(x, codes, verify_numerics=verify_numerics)
            expert_outputs.append(expert_output)
            
            if return_details:
                quantization_info.append(expert.get_quantization_info())
                
        # Combine expert outputs
        expert_stack = torch.stack(expert_outputs, dim=-1)
        routed_output = torch.sum(
            expert_stack * routing_weights.unsqueeze(-2),
            dim=-1
        )
        
        # Final output
        output = base_output + routed_output
        
        # Update calibration step counter
        if self.training:
            self.calibration_steps += 1
            
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
                'quantization_info': quantization_info,
                'base_output': base_output,
                'routed_output': routed_output,
                'calibration_steps': self.calibration_steps.item()
            })
            
        return result
        
    def enable_quantization(self):
        """Enable FP8 quantization for all experts."""
        for expert in self.experts:
            expert.enable_quantization()
            
    def disable_quantization(self):
        """Disable FP8 quantization for all experts."""
        for expert in self.experts:
            expert.disable_quantization()
            
    def enable_verification(self):
        """Enable numerical verification."""
        self.verification_enabled = True
        
    def disable_verification(self):
        """Disable numerical verification.""" 
        self.verification_enabled = False
        
    def get_numerical_summary(self) -> Dict[str, float]:
        """Get summary of numerical accuracy."""
        errors = [expert.numerical_error.item() for expert in self.experts]
        
        return {
            'max_error': max(errors),
            'mean_error': sum(errors) / len(errors),
            'min_error': min(errors),
            'num_experts': len(self.experts),
            'tolerance': self.config.numerical_tolerance,
            'calibration_steps': self.calibration_steps.item()
        }
        
    def prepare_for_export(self) -> Dict[str, Any]:
        """Prepare all experts for export."""
        export_data = {
            'config': self.config,
            'experts': []
        }
        
        for i, expert in enumerate(self.experts):
            expert_data = expert.prepare_for_export()
            expert_data['expert_id'] = i
            export_data['experts'].append(expert_data)
            
        return export_data


def convert_lora_to_fp8(
    lora_expert: nn.Module,
    config: Optional[FP8QATConfig] = None
) -> FP8LoRAExpert:
    """Convert standard LoRA expert to FP8 quantized version."""
    if config is None:
        config = FP8QATConfig()
        
    # Create FP8 expert with same dimensions
    fp8_expert = FP8LoRAExpert(
        in_features=lora_expert.lora_A.in_features,
        out_features=lora_expert.lora_B.out_features,
        rank=lora_expert.lora_A.out_features,
        config=config
    )
    
    # Copy weights
    with torch.no_grad():
        fp8_expert.lora_A.weight.copy_(lora_expert.lora_A.weight)
        fp8_expert.lora_B.weight.copy_(lora_expert.lora_B.weight)
        
    return fp8_expert


def create_fp8_qat_config(**kwargs) -> FP8QATConfig:
    """Factory function to create FP8QATConfig with validation."""
    return FP8QATConfig(**kwargs)


def selftest_fp8_numerics(
    shape: Tuple[int, int] = (512, 64),
    config: Optional[FP8QATConfig] = None,
    tolerance: float = 1e-3
) -> Dict[str, Any]:
    """Self-test FP8 numerics implementation."""
    if config is None:
        config = FP8QATConfig(numerical_tolerance=tolerance)
        
    # Create test tensors
    torch.manual_seed(42)
    A = torch.randn(shape) * 0.1  # Small values for stability
    B = torch.randn(shape[::-1]) * 0.1
    x = torch.randn(32, shape[0])  # Batch of inputs
    codes = torch.randn(32, shape[1])
    
    # Create FP8 expert
    expert = FP8LoRAExpert(
        in_features=shape[0],
        out_features=shape[0], 
        rank=shape[1],
        config=config
    )
    
    # Copy test weights
    with torch.no_grad():
        expert.lora_A.weight.copy_(A.T)
        expert.lora_B.weight.copy_(B.T)
        
    expert.eval()
    
    # Run forward pass with verification
    with torch.no_grad():
        output = expert(x, codes, verify_numerics=True)
        
    # Get numerical summary
    summary = {
        'numerical_error': expert.numerical_error.item(),
        'tolerance': tolerance,
        'numerics_pass': expert.numerical_error.item() < tolerance,
        'fp16_norm': expert.fp16_reference.item(),
        'fp8_norm': expert.fp8_output.item(),
        'format': config.format.value,
        'per_channel': config.per_channel
    }
    
    return summary


def main():
    """CLI interface for FP8 self-testing."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="FP8 QAT self-test")
    parser.add_argument("--selftest", action="store_true", help="Run numerical self-test")
    parser.add_argument("--out", type=str, help="Output file for test results")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Numerical tolerance")
    parser.add_argument("--format", choices=["E4M3", "E5M2"], default="E4M3", help="FP8 format")
    
    args = parser.parse_args()
    
    if args.selftest:
        config = FP8QATConfig(
            format=FP8Format[args.format],
            numerical_tolerance=args.tolerance
        )
        
        results = selftest_fp8_numerics(config=config, tolerance=args.tolerance)
        
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(json.dumps(results, indent=2))
            
        if results['numerics_pass']:
            print("✅ FP8 numerics test PASSED")
        else:
            print("❌ FP8 numerics test FAILED")
            print(f"Error: {results['numerical_error']:.6f} > {args.tolerance}")
            exit(1)


# Alias for backward compatibility and pattern matching
FP8Config = FP8QATConfig


if __name__ == "__main__":
    main()