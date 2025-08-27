"""
Agentic Router Module

Implements dynamic BEM composition with macro-policy routing:
- Synthetic trace generation for behavior cloning
- Macro-policy with trust-region constraints
- Action hysteresis and cache-safe routing
- Policy gradient training from task rewards

Key Components:
- MacroPolicy: Neural policy for BEM selection
- TraceGenerator: Synthetic data for behavior cloning  
- AgenticRouter: Main orchestration class
- CompositionEngine: Safe delta composition
"""

from .macro_policy import MacroPolicy, MacroAction
from .trace_generator import TraceGenerator
from .agentic_router import AgenticRouter
from .composition_engine import CompositionEngine
from .training import BCTrainer, PGTrainer

__all__ = [
    "MacroPolicy",
    "MacroAction", 
    "TraceGenerator",
    "AgenticRouter",
    "CompositionEngine",
    "BCTrainer",
    "PGTrainer"
]