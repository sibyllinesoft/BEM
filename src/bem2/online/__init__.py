"""
BEM 2.0 Online Learning System (OL0) - Lifelong Learner Implementation.

This module implements safe controller-only online updates using feedback signals 
while preserving system stability as specified in TODO.md.

Key Features:
- Controller-only updates: Only routing heads updated, BEM matrices unchanged
- EWC/Prox regularization to prevent catastrophic forgetting  
- Replay buffer for maintaining old knowledge
- Automatic rollback on canary failures
- Between-prompt updates only
- KL divergence and parameter norm monitoring with auto-rollback

Components:
- EWCRegularizer: Elastic Weight Consolidation with diagonal Fisher information
- ReplayBuffer: Experience replay for knowledge retention
- CanaryGate: Safety testing before applying updates  
- DriftMonitor: Continuous monitoring with auto-rollback
- OnlineLearner: Main controller orchestrating safe online learning
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import numpy as np

# Core online learning components
from .ewc_regularizer import EWCRegularizer, FisherInformationMatrix
from .replay_buffer import ReplayBuffer, Experience, ReplayConfig
from .canary_gate import CanaryGate, CanaryTest, CanaryResult
from .drift_monitor import DriftMonitor, DriftMetrics, DriftThresholds
from .online_learner import OnlineLearner, OnlineLearningConfig, LearningPhase
from .checkpointing import CheckpointManager, Checkpoint, CheckpointConfig

# Streaming and feedback processing
from .streaming import StreamProcessor, FeedbackSignal, StreamConfig
from .feedback_processor import FeedbackProcessor, FeedbackType, ProcessedFeedback

# Warmup and evaluation
from .warmup import WarmupManager, WarmupConfig, WarmupMetrics
from .evaluation import OnlineEvaluator, EvaluationMetrics, PerformanceTracker

# Utilities and interfaces
from .interfaces import (
    OnlineUpdateResult,
    SafetyStatus, 
    UpdateDecision,
    LearningState,
    CanaryStatus
)

__all__ = [
    # Core components
    'EWCRegularizer',
    'FisherInformationMatrix', 
    'ReplayBuffer',
    'Experience',
    'ReplayConfig',
    'CanaryGate',
    'CanaryTest',
    'CanaryResult',
    'DriftMonitor',
    'DriftMetrics',
    'DriftThresholds',
    'OnlineLearner',
    'OnlineLearningConfig',
    'LearningPhase',
    'CheckpointManager',
    'Checkpoint',
    'CheckpointConfig',
    
    # Streaming
    'StreamProcessor',
    'FeedbackSignal',
    'StreamConfig',
    'FeedbackProcessor',
    'FeedbackType',
    'ProcessedFeedback',
    
    # Warmup and evaluation
    'WarmupManager',
    'WarmupConfig', 
    'WarmupMetrics',
    'OnlineEvaluator',
    'EvaluationMetrics',
    'PerformanceTracker',
    
    # Interfaces
    'OnlineUpdateResult',
    'SafetyStatus',
    'UpdateDecision', 
    'LearningState',
    'CanaryStatus'
]

# Package version
__version__ = "2.0.0"