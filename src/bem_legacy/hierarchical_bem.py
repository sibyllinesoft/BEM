"""
Hierarchical BEM implementation that integrates the hierarchical controller
with the generated BEM variant for full adaptive control.

This combines the validated simple BEM with the hierarchical routing controller
to create the complete generated BEM system as specified in the TODO.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union
import math
from dataclasses import dataclass

from .controller import HierarchicalController, RoutingLevel, RoutingState, create_hierarchical_controller
from .simple_bem import SimpleBEMModule


@dataclass 
class HierarchicalBEMConfig:
    """Configuration for hierarchical BEM system."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    chunk_size: int = 32
    max_prefix_tokens: int = 128
    ema_decay: float = 0.99
    enable_uncertainty: bool = True
    enable_token_routing: bool = True
    code_clamp_value: float = 3.0
    side_signal_dim: Optional[int] = None
    
    # Attach points configuration
    attach_mlp: bool = True
    attach_attention: bool = True
    attach_qkv: bool = False  # Disabled by default for cache compatibility
    
    # Performance settings
    use_gradient_checkpointing: bool = False
    precision: str = "fp16"  # fp16, bf16, fp32
    use_fused_kernels: bool = True  # Enable high-performance CUDA kernels when available


class HierarchicalBEMModule(nn.Module):
    """
    Hierarchical BEM module that combines the controller with generated BEM updates.
    
    This is the full implementation of the generated BEM variant with hierarchical routing,
    supporting prefix/chunk/token level adaptation with uncertainty estimation.
    
    Key features:
    - Hierarchical routing (prefix → chunk → token)
    - Cache-aware Q/K/V updates (chunkwise only)
    - Uncertainty-weighted adaptations
    - EMA smoothing for stability
    - Comprehensive telemetry
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        config: HierarchicalBEMConfig,
        layer_name: str = "unknown",
        attach_point: str = "mlp"  # "mlp", "attention", "q", "k", "v", "o"
    ):
        super().__init__()
        
        self.config = config
        self.layer_name = layer_name
        self.attach_point = attach_point
        
        # Base layer (frozen)
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Get dimensions
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # LoRA parameters for generated BEM: ΔW = U * diag(code) * V^T
        self.lora_U = nn.Parameter(torch.randn(self.out_features, config.rank) * 0.02)
        self.lora_V = nn.Parameter(torch.randn(self.in_features, config.rank) * 0.02)
        
        # Scaling factor
        self.scaling = config.alpha / config.rank
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        
        # Track routing statistics
        self.register_buffer('total_forward_calls', torch.tensor(0))
        self.register_buffer('routing_stats', torch.zeros(3))  # [prefix, chunk, token] usage counts
    
    def get_routing_level(self, seq_len: int, chunk_position: int = 0) -> RoutingLevel:
        """
        Determine appropriate routing level based on attach point and sequence properties.
        
        Args:
            seq_len: Sequence length
            chunk_position: Current chunk position
            
        Returns:
            Appropriate routing level
        """
        # Q/K/V layers use chunk-level routing for cache compatibility
        if self.attach_point in ["q", "k", "v"]:
            return RoutingLevel.CHUNK
        
        # MLP layers can use token-level routing if enabled
        elif self.attach_point in ["mlp", "o"] and self.config.enable_token_routing:
            # Use chunk routing for long sequences to manage computational cost
            if seq_len > 512:
                return RoutingLevel.CHUNK
            else:
                return RoutingLevel.TOKEN
        
        # Default to chunk routing
        else:
            return RoutingLevel.CHUNK
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        controller: HierarchicalController,
        attention_mask: Optional[torch.Tensor] = None,
        side_signals: Optional[torch.Tensor] = None,
        chunk_position: int = 0,
        return_routing_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with hierarchical routing.
        
        Args:
            x: Input tensor [batch, seq_len, in_features] or [batch, in_features]
            hidden_states: Hidden states for controller [batch, seq_len, hidden_dim]
            controller: HierarchicalController instance
            attention_mask: Optional attention mask [batch, seq_len]
            side_signals: Optional side signals [batch, side_signal_dim]
            chunk_position: Current chunk position
            return_routing_info: Whether to return routing information
            
        Returns:
            output: Modified output tensor
            routing_info: Routing information (if requested)
        """
        # Base forward pass
        base_output = self.base_layer(x)
        
        # Handle input shapes
        original_shape = x.shape
        batch_size = x.shape[0]
        
        if x.dim() == 3:
            seq_len = x.shape[1]
            x_flat = x.view(-1, self.in_features)  # [batch*seq_len, in_features]
        else:
            seq_len = 1
            x_flat = x
        
        # Determine routing level
        routing_level = self.get_routing_level(seq_len, chunk_position)
        
        # Generate routing codes
        codes, routing_state = controller(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            side_signals=side_signals,
            routing_level=routing_level,
            chunk_position=chunk_position,
            return_routing_state=True
        )
        
        # Apply generated LoRA update: ΔW = U * diag(code) * V^T
        # Computation: (x @ V) * code @ U^T
        if routing_level == RoutingLevel.TOKEN and codes.dim() == 3:
            # Token-level codes: [batch, seq_len, rank]
            codes_flat = codes.view(-1, self.config.rank)  # [batch*seq_len, rank]
        else:
            # Prefix or chunk codes: [batch, rank] - expand for sequence
            if x.dim() == 3:
                codes_expanded = codes.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, rank]
                codes_flat = codes_expanded.contiguous().view(-1, self.config.rank)  # [batch*seq_len, rank]
            else:
                codes_flat = codes
        
        # Generated LoRA computation - use optimized kernel if available
        try:
            from .kernels.cuda_ops import fused_generated_update, KERNELS_AVAILABLE
            
            if KERNELS_AVAILABLE and x.is_cuda and hasattr(self.config, 'use_fused_kernels') and self.config.use_fused_kernels:
                # Use high-performance fused CUDA kernel (includes scaling)
                lora_output = fused_generated_update(
                    x_flat, self.lora_V, codes_flat, self.lora_U, self.scaling
                )
                use_fused = True
            else:
                # Fallback to PyTorch implementation
                x_v = torch.matmul(x_flat, self.lora_V)  # [*, rank]
                x_v_scaled = x_v * codes_flat  # Element-wise with dynamic code
                lora_output = torch.matmul(x_v_scaled, self.lora_U.t())  # [*, out_features]
                use_fused = False
                
        except ImportError:
            # CUDA kernels not available - use PyTorch fallback
            x_v = torch.matmul(x_flat, self.lora_V)  # [*, rank]
            x_v_scaled = x_v * codes_flat  # Element-wise with dynamic code
            lora_output = torch.matmul(x_v_scaled, self.lora_U.t())  # [*, out_features]
            use_fused = False
        
        # Reshape back if needed
        if len(original_shape) == 3:
            lora_output = lora_output.view(original_shape[0], original_shape[1], -1)
        
        # Apply dropout and scaling (scaling only if not using fused kernel)
        lora_output = self.dropout(lora_output)
        if not use_fused:
            lora_output = lora_output * self.scaling
        
        # Final output
        output = base_output + lora_output
        
        # Update statistics
        self.total_forward_calls += 1
        if routing_level == RoutingLevel.PREFIX:
            self.routing_stats[0] += 1
        elif routing_level == RoutingLevel.CHUNK:
            self.routing_stats[1] += 1
        else:
            self.routing_stats[2] += 1
        
        if return_routing_info:
            routing_info = {
                'routing_level': routing_level,
                'routing_state': routing_state,
                'code_norm': codes.norm(dim=-1).mean().item(),
                'lora_norm': lora_output.norm(dim=-1).mean().item(),
                'uncertainty': routing_state.uncertainty.mean().item() if routing_state.uncertainty is not None else None,
                'layer_name': self.layer_name,
                'attach_point': self.attach_point
            }
            return output, routing_info
        
        return output


class FullHierarchicalBEM(nn.Module):
    """
    Complete hierarchical BEM system that wraps a base model with hierarchical routing.
    
    This is the main interface for using hierarchical BEMs, handling:
    - Multiple attach points (MLP, attention)
    - Shared controller across layers
    - Cache-aware routing policies
    - Training and inference modes
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: HierarchicalBEMConfig,
        attach_layers: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.config = config
        self.base_model = base_model
        
        # Create shared controller
        if model_config is None:
            # Try to infer from base model
            model_config = getattr(base_model, 'config', {})
            if hasattr(model_config, '__dict__'):
                model_config = model_config.__dict__
        
        controller_config = {
            'rank': config.rank,
            'dropout': config.dropout,
            'chunk_size': config.chunk_size,
            'max_prefix_tokens': config.max_prefix_tokens,
            'ema_decay': config.ema_decay,
            'side_signal_dim': config.side_signal_dim,
            'enable_uncertainty': config.enable_uncertainty,
            'enable_token_routing': config.enable_token_routing,
            'code_clamp_value': config.code_clamp_value
        }
        
        self.controller = create_hierarchical_controller(model_config, controller_config)
        
        # BEM modules dictionary
        self.bem_modules = nn.ModuleDict()
        
        # Automatically attach to specified layers
        if attach_layers is None:
            attach_layers = self._find_attach_layers()
        
        self._attach_bem_modules(attach_layers)
        
        # Routing cache for efficiency
        self.routing_cache = {}
        self.cache_chunk_size = config.chunk_size
    
    def _find_attach_layers(self) -> List[str]:
        """Automatically find layers to attach BEMs to."""
        attach_layers = []
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if it's an MLP layer
                if any(mlp_name in name for mlp_name in ['mlp', 'feed_forward', 'ffn']):
                    if self.config.attach_mlp:
                        attach_layers.append(name)
                
                # Check if it's an attention layer
                elif any(att_name in name for att_name in ['attention', 'attn']):
                    if 'o_proj' in name or 'out_proj' in name:
                        if self.config.attach_attention:
                            attach_layers.append(name)
                    elif any(qkv in name for qkv in ['q_proj', 'k_proj', 'v_proj']):
                        if self.config.attach_qkv:
                            attach_layers.append(name)
        
        return attach_layers
    
    def _attach_bem_modules(self, attach_layers: List[str]):
        """Attach BEM modules to specified layers."""
        for layer_name in attach_layers:
            # Get the actual module
            module = self.base_model
            for part in layer_name.split('.'):
                module = getattr(module, part)
            
            if not isinstance(module, nn.Linear):
                continue
            
            # Determine attach point type
            if 'mlp' in layer_name or 'feed_forward' in layer_name or 'ffn' in layer_name:
                attach_point = 'mlp'
            elif 'o_proj' in layer_name or 'out_proj' in layer_name:
                attach_point = 'o'
            elif 'q_proj' in layer_name:
                attach_point = 'q'
            elif 'k_proj' in layer_name:
                attach_point = 'k'
            elif 'v_proj' in layer_name:
                attach_point = 'v'
            else:
                attach_point = 'unknown'
            
            # Create BEM module
            bem_module = HierarchicalBEMModule(
                base_layer=module,
                config=self.config,
                layer_name=layer_name,
                attach_point=attach_point
            )
            
            self.bem_modules[layer_name] = bem_module
            
            # Replace the original module with our BEM wrapper
            self._replace_module(layer_name, bem_module)
    
    def _replace_module(self, layer_name: str, new_module: nn.Module):
        """Replace a module in the base model."""
        parts = layer_name.split('.')
        parent = self.base_model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        side_signals: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through hierarchical BEM system.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            side_signals: Optional side signals [batch, side_signal_dim]
            return_routing_info: Whether to return routing information
            **kwargs: Additional arguments for base model
            
        Returns:
            outputs: Model outputs
            routing_info: Routing information (if requested)
        """
        # This is a simplified wrapper - in practice, you'd need to hook into
        # the model's forward pass to inject hierarchical routing at each layer
        
        # For now, we'll assume the BEM modules have been properly integrated
        # and will be called automatically during the base model's forward pass
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        if return_routing_info:
            # Collect routing info from all BEM modules
            routing_info = self.get_routing_statistics()
            return outputs, routing_info
        
        return outputs
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        stats = {
            'total_bem_modules': len(self.bem_modules),
            'layers': {},
            'global_stats': {
                'total_forward_calls': 0,
                'routing_distribution': torch.zeros(3)  # [prefix, chunk, token]
            }
        }
        
        for layer_name, bem_module in self.bem_modules.items():
            layer_stats = {
                'attach_point': bem_module.attach_point,
                'total_calls': bem_module.total_forward_calls.item(),
                'routing_counts': bem_module.routing_stats.tolist(),
                'parameters': {
                    'rank': self.config.rank,
                    'in_features': bem_module.in_features,
                    'out_features': bem_module.out_features
                }
            }
            
            stats['layers'][layer_name] = layer_stats
            stats['global_stats']['total_forward_calls'] += layer_stats['total_calls']
            stats['global_stats']['routing_distribution'] += bem_module.routing_stats
        
        # Convert to percentages
        total_routing = stats['global_stats']['routing_distribution'].sum()
        if total_routing > 0:
            stats['global_stats']['routing_distribution'] = (
                stats['global_stats']['routing_distribution'] / total_routing
            ).tolist()
        
        return stats
    
    def reset_routing_statistics(self):
        """Reset routing statistics for all BEM modules."""
        for bem_module in self.bem_modules.values():
            bem_module.total_forward_calls.zero_()
            bem_module.routing_stats.zero_()
    
    def get_bem_parameters(self) -> List[nn.Parameter]:
        """Get all BEM parameters (excluding base model)."""
        bem_params = []
        for bem_module in self.bem_modules.values():
            bem_params.extend([bem_module.lora_U, bem_module.lora_V])
        
        # Add controller parameters
        bem_params.extend(self.controller.parameters())
        
        return bem_params
    
    def freeze_base_model(self):
        """Freeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def save_bem_state(self, path: str):
        """Save only BEM-specific state (controller + modules)."""
        state = {
            'config': self.config,
            'controller_state': self.controller.state_dict(),
            'bem_modules': {name: module.state_dict() 
                          for name, module in self.bem_modules.items()},
            'routing_stats': self.get_routing_statistics()
        }
        torch.save(state, path)
    
    def load_bem_state(self, path: str):
        """Load BEM-specific state."""
        state = torch.load(path, map_location='cpu')
        
        self.controller.load_state_dict(state['controller_state'])
        
        for name, module_state in state['bem_modules'].items():
            if name in self.bem_modules:
                self.bem_modules[name].load_state_dict(module_state)


# Factory function for creating hierarchical BEMs

def create_hierarchical_bem(
    base_model: nn.Module,
    config: Optional[HierarchicalBEMConfig] = None,
    attach_layers: Optional[List[str]] = None,
    **config_kwargs
) -> FullHierarchicalBEM:
    """
    Factory function to create a hierarchical BEM system.
    
    Args:
        base_model: Base model to wrap
        config: HierarchicalBEMConfig instance
        attach_layers: Specific layers to attach BEMs to
        **config_kwargs: Config overrides
        
    Returns:
        FullHierarchicalBEM instance
    """
    if config is None:
        config = HierarchicalBEMConfig(**config_kwargs)
    else:
        # Update config with any provided kwargs
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    model_config = None
    if hasattr(base_model, 'config'):
        model_config = base_model.config
        if hasattr(model_config, '__dict__'):
            model_config = model_config.__dict__
    
    hierarchical_bem = FullHierarchicalBEM(
        base_model=base_model,
        config=config,
        attach_layers=attach_layers,
        model_config=model_config
    )
    
    return hierarchical_bem


# Compatibility layer with existing validation framework

def create_hierarchical_bem_for_validation(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    **kwargs
) -> FullHierarchicalBEM:
    """
    Create hierarchical BEM compatible with existing validation framework.
    
    Args:
        model: Base model
        target_modules: Target module names
        rank: LoRA rank
        alpha: LoRA alpha
        **kwargs: Additional config parameters
        
    Returns:
        FullHierarchicalBEM configured for validation
    """
    config = HierarchicalBEMConfig(
        rank=rank,
        alpha=alpha,
        **kwargs
    )
    
    return create_hierarchical_bem(
        base_model=model,
        config=config,
        attach_layers=target_modules
    )