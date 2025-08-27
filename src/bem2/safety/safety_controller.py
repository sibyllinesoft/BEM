"""
Safety Controller with Scalar Control Interface

Implements dynamic safety control with a scalar "safety knob" that allows
real-time adjustment of safety strength. Provides smooth interpolation between
safety and utility, with automatic adaptation based on context.

Key Features:
- Scalar safety knob [0,1] for dynamic adjustment
- Context-aware safety strength adaptation
- Smooth safety-utility trade-off curves
- Real-time safety strength monitoring
- Automatic safety escalation for high-risk content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import math
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyMode(Enum):
    """Safety operation modes."""
    DISABLED = 0.0      # Safety features disabled
    LOW = 0.3          # Low safety enforcement
    MEDIUM = 0.6       # Balanced safety-utility
    HIGH = 0.8         # High safety enforcement  
    MAXIMUM = 1.0      # Maximum safety enforcement


@dataclass
class ControlConfig:
    """Configuration for safety control interface."""
    
    # Core safety knob
    default_safety_level: float = 0.6      # Default safety strength [0,1]
    safety_knob_precision: int = 3          # Precision for safety knob values
    
    # Adaptation parameters
    context_adaptation: bool = True         # Enable context-aware adaptation
    adaptation_strength: float = 0.2       # Strength of context adaptation
    adaptation_smoothing: float = 0.9      # Smoothing factor for adaptation
    
    # Safety escalation
    auto_escalation: bool = True           # Enable automatic safety escalation
    escalation_threshold: float = 0.2      # Constitutional score threshold for escalation
    escalation_factor: float = 0.5         # How much to increase safety on escalation
    max_escalation: float = 1.0            # Maximum escalation level
    
    # Dynamic adjustment
    min_safety_level: float = 0.0          # Minimum allowed safety level
    max_safety_level: float = 1.0          # Maximum allowed safety level
    adjustment_rate: float = 0.1           # Rate of safety level adjustment
    
    # Monitoring
    track_safety_history: bool = True      # Track safety level history
    history_length: int = 1000             # Length of history tracking
    
    # User override
    allow_user_override: bool = True       # Allow user to override safety level
    override_decay_steps: int = 500        # Steps before override decays
    
    # Context sensitivity
    domain_specific_safety: bool = True    # Use domain-specific safety levels
    content_type_adaptation: bool = True   # Adapt based on content type
    
    # Performance optimization
    cache_safety_computations: bool = True # Cache safety strength computations
    batch_safety_processing: bool = True   # Process safety in batches


class SafetyController(nn.Module):
    """
    Dynamic safety controller with scalar interface.
    
    Provides a unified interface for controlling safety strength across
    the entire BEM 2.0 system with real-time adjustment capabilities.
    """
    
    def __init__(self, config: ControlConfig):
        super().__init__()
        self.config = config
        
        # Core safety knob parameter  
        self.register_buffer(
            'safety_knob', 
            torch.tensor(config.default_safety_level)
        )
        
        # Context adaptation network
        if config.context_adaptation:
            self.context_adapter = nn.Sequential(
                nn.Linear(768, 256),  # Assuming hidden_dim=768
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.context_adapter = None
        
        # Domain-specific safety levels
        if config.domain_specific_safety:
            self.domain_safety_levels = nn.ParameterDict({
                'general': nn.Parameter(torch.tensor(0.6)),
                'medical': nn.Parameter(torch.tensor(0.8)),
                'legal': nn.Parameter(torch.tensor(0.7)),
                'financial': nn.Parameter(torch.tensor(0.7)),
                'educational': nn.Parameter(torch.tensor(0.5)),
                'creative': nn.Parameter(torch.tensor(0.4)),
                'technical': nn.Parameter(torch.tensor(0.5)),
                'sensitive': nn.Parameter(torch.tensor(0.9))
            })
        else:
            self.domain_safety_levels = None
        
        # Safety level history tracking
        if config.track_safety_history:
            self.register_buffer(
                'safety_history',
                torch.zeros(config.history_length)
            )
            self.register_buffer('history_pointer', torch.tensor(0))
        
        # Adaptation state
        self.register_buffer(
            'adapted_safety_level',
            torch.tensor(config.default_safety_level)
        )
        self.register_buffer(
            'escalation_level',
            torch.tensor(0.0)
        )
        
        # User override state
        self.register_buffer('user_override_active', torch.tensor(False))
        self.register_buffer('user_override_value', torch.tensor(config.default_safety_level))
        self.register_buffer('override_steps_remaining', torch.tensor(0))
        
        # Safety computation cache
        if config.cache_safety_computations:
            self.safety_cache = {}
        else:
            self.safety_cache = None
        
        # Telemetry
        self.register_buffer('total_adjustments', torch.tensor(0))
        self.register_buffer('escalation_count', torch.tensor(0))
        self.register_buffer('adaptation_count', torch.tensor(0))
        
        # Initialize parameters
        self._initialize_parameters()
        
        logger.info(f"Initialized safety controller with default level {config.default_safety_level}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        constitutional_scores: torch.Tensor,
        domain: Optional[str] = None,
        user_override: Optional[float] = None,
        return_telemetry: bool = False
    ) -> Union[float, Tuple[float, Dict[str, any]]]:
        """
        Compute dynamic safety strength level.
        
        Args:
            hidden_states: Current model hidden states [batch, seq, hidden]
            constitutional_scores: Constitutional AI scores [batch]
            domain: Content domain for domain-specific safety
            user_override: Optional user override value
            return_telemetry: Whether to return detailed telemetry
            
        Returns:
            safety_strength: Current safety strength level [0,1]
            telemetry (optional): Detailed safety controller telemetry
        """
        batch_size = constitutional_scores.size(0)
        
        # Handle user override
        if user_override is not None:
            return self._handle_user_override(user_override, return_telemetry)
        
        # Check if user override is still active
        if self.user_override_active:
            return self._apply_active_override(return_telemetry)
        
        # Base safety level (scalar knob value)
        base_safety = self.safety_knob.clone()
        
        # Apply domain-specific adjustment
        if self.config.domain_specific_safety and domain is not None:
            domain_adjustment = self._get_domain_safety_level(domain)
            base_safety = base_safety * 0.7 + domain_adjustment * 0.3  # Weighted combination
        
        # Context-aware adaptation
        context_adjustment = torch.tensor(0.0, device=hidden_states.device)
        if self.config.context_adaptation and self.context_adapter is not None:
            context_adjustment = self._compute_context_adaptation(hidden_states)
        
        # Automatic safety escalation based on constitutional scores
        escalation_adjustment = torch.tensor(0.0, device=constitutional_scores.device)
        if self.config.auto_escalation:
            escalation_adjustment = self._compute_escalation_adjustment(constitutional_scores)
        
        # Combine all adjustments
        total_adjustment = (
            self.config.adaptation_strength * context_adjustment +
            escalation_adjustment
        )
        
        # Apply adjustment with smoothing
        current_safety = base_safety + total_adjustment
        
        # Apply smoothing to adapted safety level
        alpha = 1 - self.config.adaptation_smoothing
        self.adapted_safety_level = (
            (1 - alpha) * self.adapted_safety_level + 
            alpha * current_safety
        )
        
        # Clamp to valid range
        final_safety = torch.clamp(
            self.adapted_safety_level,
            self.config.min_safety_level,
            self.config.max_safety_level
        )
        
        # Update telemetry
        self._update_safety_telemetry(final_safety.item())
        
        # Return based on whether telemetry is requested
        if return_telemetry:
            telemetry = self._compute_safety_telemetry(
                base_safety.item(),
                context_adjustment.item(),
                escalation_adjustment.item(),
                final_safety.item(),
                domain
            )
            return final_safety.item(), telemetry
        else:
            return final_safety.item()
    
    def _get_domain_safety_level(self, domain: str) -> torch.Tensor:
        """Get domain-specific safety level."""
        if self.domain_safety_levels is None:
            return torch.tensor(self.config.default_safety_level)
        
        domain_key = domain.lower()
        if domain_key in self.domain_safety_levels:
            return torch.clamp(
                self.domain_safety_levels[domain_key],
                self.config.min_safety_level,
                self.config.max_safety_level
            )
        else:
            # Default to general domain
            return torch.clamp(
                self.domain_safety_levels['general'],
                self.config.min_safety_level,
                self.config.max_safety_level
            )
    
    def _compute_context_adaptation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute context-aware safety adaptation."""
        if self.context_adapter is None:
            return torch.tensor(0.0, device=hidden_states.device)
        
        # Pool hidden states for context analysis
        pooled_states = hidden_states.mean(dim=(0, 1))  # [hidden_dim]
        
        # Compute context-based safety adjustment
        context_safety = self.context_adapter(pooled_states.unsqueeze(0))
        context_adjustment = (context_safety.squeeze() - 0.5) * 0.4  # Center around 0, scale to [-0.2, 0.2]
        
        self.adaptation_count += 1
        
        return context_adjustment
    
    def _compute_escalation_adjustment(self, constitutional_scores: torch.Tensor) -> torch.Tensor:
        """Compute automatic safety escalation based on constitutional scores."""
        
        # Check if any scores are below escalation threshold
        min_score = constitutional_scores.min()
        
        if min_score < self.config.escalation_threshold:
            # Escalate safety based on how low the scores are
            escalation_factor = (self.config.escalation_threshold - min_score) / self.config.escalation_threshold
            escalation_adjustment = escalation_factor * self.config.escalation_factor
            
            # Update escalation level with momentum
            new_escalation = min(escalation_adjustment.item(), self.config.max_escalation)
            self.escalation_level = 0.9 * self.escalation_level + 0.1 * new_escalation
            
            self.escalation_count += 1
            
            return torch.tensor(self.escalation_level.item(), device=constitutional_scores.device)
        else:
            # Gradually decay escalation when scores are good
            self.escalation_level = self.escalation_level * 0.95
            return torch.tensor(self.escalation_level.item(), device=constitutional_scores.device)
    
    def _handle_user_override(self, user_override: float, return_telemetry: bool):
        """Handle user override of safety level."""
        if not self.config.allow_user_override:
            logger.warning("User override requested but not allowed by configuration")
            return self.safety_knob.item()
        
        # Clamp user override to valid range
        clamped_override = max(
            self.config.min_safety_level,
            min(user_override, self.config.max_safety_level)
        )
        
        # Set override state
        self.user_override_active = torch.tensor(True)
        self.user_override_value = torch.tensor(clamped_override)
        self.override_steps_remaining = torch.tensor(self.config.override_decay_steps)
        
        logger.info(f"User override activated: safety level set to {clamped_override:.3f}")
        
        if return_telemetry:
            telemetry = {
                'safety_strength': clamped_override,
                'user_override_active': True,
                'override_steps_remaining': self.config.override_decay_steps,
                'base_safety': self.safety_knob.item(),
                'domain_adjustment': 0.0,
                'context_adjustment': 0.0,
                'escalation_adjustment': 0.0
            }
            return clamped_override, telemetry
        else:
            return clamped_override
    
    def _apply_active_override(self, return_telemetry: bool):
        """Apply currently active user override."""
        # Decay override counter
        self.override_steps_remaining = self.override_steps_remaining - 1
        
        # Check if override has expired
        if self.override_steps_remaining <= 0:
            self.user_override_active = torch.tensor(False)
            logger.info("User override expired, returning to automatic safety control")
            
            # Fall back to normal computation (recursive call without override)
            return self.forward(
                torch.zeros(1, 1, 768),  # Dummy hidden states
                torch.tensor([0.7]),     # Dummy constitutional score
                return_telemetry=return_telemetry
            )
        
        current_override = self.user_override_value.item()
        
        if return_telemetry:
            telemetry = {
                'safety_strength': current_override,
                'user_override_active': True,
                'override_steps_remaining': self.override_steps_remaining.item(),
                'base_safety': self.safety_knob.item(),
                'domain_adjustment': 0.0,
                'context_adjustment': 0.0,
                'escalation_adjustment': 0.0
            }
            return current_override, telemetry
        else:
            return current_override
    
    def _update_safety_telemetry(self, safety_level: float):
        """Update safety level telemetry."""
        self.total_adjustments += 1
        
        # Update history tracking
        if self.config.track_safety_history:
            ptr = self.history_pointer.item()
            self.safety_history[ptr] = safety_level
            self.history_pointer = (ptr + 1) % self.safety_history.size(0)
    
    def _compute_safety_telemetry(
        self,
        base_safety: float,
        context_adjustment: float,
        escalation_adjustment: float,
        final_safety: float,
        domain: Optional[str]
    ) -> Dict[str, any]:
        """Compute detailed safety telemetry."""
        
        domain_adjustment = 0.0
        if domain and self.config.domain_specific_safety:
            domain_level = self._get_domain_safety_level(domain).item()
            domain_adjustment = domain_level - base_safety
        
        return {
            'safety_strength': final_safety,
            'base_safety': base_safety,
            'domain_adjustment': domain_adjustment,
            'context_adjustment': context_adjustment,
            'escalation_adjustment': escalation_adjustment,
            'adapted_safety_level': self.adapted_safety_level.item(),
            'escalation_level': self.escalation_level.item(),
            'user_override_active': self.user_override_active.item(),
            'total_adjustments': self.total_adjustments.item(),
            'escalation_count': self.escalation_count.item(),
            'adaptation_count': self.adaptation_count.item(),
            'domain': domain
        }
    
    def set_safety_level(self, new_level: float, user_override: bool = False):
        """Set the base safety level."""
        clamped_level = max(
            self.config.min_safety_level,
            min(new_level, self.config.max_safety_level)
        )
        
        if user_override and self.config.allow_user_override:
            # Set as user override
            self.user_override_active = torch.tensor(True)
            self.user_override_value = torch.tensor(clamped_level)
            self.override_steps_remaining = torch.tensor(self.config.override_decay_steps)
            logger.info(f"User override: safety level set to {clamped_level:.3f}")
        else:
            # Update base safety knob
            self.safety_knob = torch.tensor(clamped_level)
            logger.info(f"Base safety level updated to {clamped_level:.3f}")
    
    def get_safety_level(self) -> float:
        """Get current effective safety level."""
        if self.user_override_active:
            return self.user_override_value.item()
        else:
            return self.adapted_safety_level.item()
    
    def get_safety_statistics(self) -> Dict[str, float]:
        """Get safety controller statistics."""
        stats = {
            'current_safety_level': self.get_safety_level(),
            'base_safety_level': self.safety_knob.item(),
            'adapted_safety_level': self.adapted_safety_level.item(),
            'escalation_level': self.escalation_level.item(),
            'total_adjustments': self.total_adjustments.item(),
            'escalation_count': self.escalation_count.item(),
            'adaptation_count': self.adaptation_count.item(),
            'user_override_active': self.user_override_active.item()
        }
        
        # Add history statistics if available
        if self.config.track_safety_history:
            history = self.safety_history
            stats.update({
                'history_mean': history.mean().item(),
                'history_std': history.std().item(),
                'history_min': history.min().item(),
                'history_max': history.max().item()
            })
        
        # Add domain-specific levels if available
        if self.config.domain_specific_safety and self.domain_safety_levels:
            domain_stats = {}
            for domain_name, level_param in self.domain_safety_levels.items():
                domain_stats[f'domain_{domain_name}_level'] = level_param.item()
            stats.update(domain_stats)
        
        return stats
    
    def set_domain_safety_level(self, domain: str, level: float):
        """Set safety level for a specific domain."""
        if not self.config.domain_specific_safety or self.domain_safety_levels is None:
            logger.warning("Domain-specific safety levels not enabled")
            return
        
        domain_key = domain.lower()
        clamped_level = max(
            self.config.min_safety_level,
            min(level, self.config.max_safety_level)
        )
        
        if domain_key in self.domain_safety_levels:
            self.domain_safety_levels[domain_key].data = torch.tensor(clamped_level)
        else:
            # Add new domain
            self.domain_safety_levels[domain_key] = nn.Parameter(torch.tensor(clamped_level))
        
        logger.info(f"Domain '{domain}' safety level set to {clamped_level:.3f}")
    
    def clear_user_override(self):
        """Clear active user override."""
        self.user_override_active = torch.tensor(False)
        self.override_steps_remaining = torch.tensor(0)
        logger.info("User override cleared")
    
    def reset_telemetry(self):
        """Reset all telemetry counters."""
        self.total_adjustments.zero_()
        self.escalation_count.zero_()
        self.adaptation_count.zero_()
        self.escalation_level.zero_()
        
        if self.config.track_safety_history:
            self.safety_history.zero_()
            self.history_pointer.zero_()
        
        logger.info("Safety controller telemetry reset")
    
    def _initialize_parameters(self):
        """Initialize safety controller parameters."""
        if self.context_adapter is not None:
            for module in self.context_adapter:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        # Initialize domain safety levels to reasonable defaults
        if self.domain_safety_levels is not None:
            with torch.no_grad():
                # Clamp all domain levels to valid range
                for param in self.domain_safety_levels.values():
                    param.clamp_(
                        self.config.min_safety_level,
                        self.config.max_safety_level
                    )
    
    def export_safety_config(self, filepath: str):
        """Export current safety configuration."""
        import json
        
        config_data = {
            'base_safety_level': self.safety_knob.item(),
            'adapted_safety_level': self.adapted_safety_level.item(),
            'escalation_level': self.escalation_level.item(),
            'config': self.config.__dict__,
            'statistics': self.get_safety_statistics()
        }
        
        # Add domain levels if available
        if self.domain_safety_levels:
            config_data['domain_safety_levels'] = {
                domain: level.item() 
                for domain, level in self.domain_safety_levels.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Exported safety configuration to {filepath}")
    
    def load_safety_config(self, filepath: str):
        """Load safety configuration from file."""
        import json
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        # Update safety levels
        if 'base_safety_level' in config_data:
            self.safety_knob = torch.tensor(config_data['base_safety_level'])
        
        if 'adapted_safety_level' in config_data:
            self.adapted_safety_level = torch.tensor(config_data['adapted_safety_level'])
        
        if 'escalation_level' in config_data:
            self.escalation_level = torch.tensor(config_data['escalation_level'])
        
        # Update domain levels
        if 'domain_safety_levels' in config_data and self.domain_safety_levels:
            for domain, level in config_data['domain_safety_levels'].items():
                if domain in self.domain_safety_levels:
                    self.domain_safety_levels[domain].data = torch.tensor(level)
        
        logger.info(f"Loaded safety configuration from {filepath}")