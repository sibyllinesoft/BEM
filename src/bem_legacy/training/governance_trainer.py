"""
Governance-aware trainer with model parameter and resource constraints.
"""

import torch
from typing import Dict, Any, Optional
from .bem_v11_trainer import BEMv11Trainer


class GovernanceAwareTrainer(BEMv11Trainer):
    """Trainer with governance constraints and parameter monitoring."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_budget_limit = kwargs.get('param_budget_limit', 1.05)  # 5% increase max
        self.baseline_param_count = None
        
    def set_baseline_params(self, baseline_model):
        """Set baseline parameter count for governance."""
        self.baseline_param_count = sum(p.numel() for p in baseline_model.parameters())
        
    def validate_param_budget(self) -> bool:
        """Validate current model doesn't exceed parameter budget."""
        if self.baseline_param_count is None:
            return True
            
        current_params = sum(p.numel() for p in self.model.parameters())
        ratio = current_params / self.baseline_param_count
        
        if ratio > self.param_budget_limit:
            raise RuntimeError(
                f"Parameter budget exceeded: {ratio:.3f}x baseline "
                f"(limit: {self.param_budget_limit:.3f}x)"
            )
        return True
        
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Training step with governance validation."""
        self.validate_param_budget()
        return super().training_step(batch)
