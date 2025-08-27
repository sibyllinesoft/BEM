"""Shared utilities for BEM experiments.

Provides common functionality for logging, checkpointing, data handling,
and system monitoring across all BEM components.
"""

from .logging_utils import setup_logger, get_logger
from .checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
from .data_utils import (
    load_dataset,
    create_dataloader,
    tokenize_batch,
    collate_fn,
)
from .model_utils import (
    count_parameters,
    get_model_size,
    freeze_parameters,
    unfreeze_parameters,
    load_pretrained_weights,
)
from .system_utils import (
    get_device_info,
    monitor_memory,
    monitor_gpu_usage,
    set_random_seed,
)

__all__ = [
    # Logging
    "setup_logger",
    "get_logger",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint", 
    "find_latest_checkpoint",
    # Data utilities
    "load_dataset",
    "create_dataloader",
    "tokenize_batch",
    "collate_fn",
    # Model utilities
    "count_parameters",
    "get_model_size",
    "freeze_parameters",
    "unfreeze_parameters",
    "load_pretrained_weights",
    # System utilities
    "get_device_info",
    "monitor_memory",
    "monitor_gpu_usage",
    "set_random_seed",
]