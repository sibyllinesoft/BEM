"""Unified logging utilities for BEM experiments.

Provides standardized logging configuration and utilities
across all BEM components and experiments.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        console_output: Whether to output to console
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging.
    
    Args:
        logger: Logger to use for output
    """
    import platform
    import torch
    
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.info("CUDA available: False")


def log_experiment_info(
    logger: logging.Logger,
    experiment_name: str,
    config: dict,
) -> None:
    """Log experiment configuration information.
    
    Args:
        logger: Logger to use
        experiment_name: Name of the experiment
        config: Configuration dictionary
    """
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info("Configuration:")
    
    def log_dict(d: dict, indent: int = 0) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)


def create_experiment_logger(
    experiment_name: str,
    output_dir: Union[str, Path],
    level: Union[str, int] = logging.INFO,
) -> logging.Logger:
    """Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for logs
        level: Logging level
        
    Returns:
        Configured experiment logger
    """
    output_path = Path(output_dir)
    log_file = output_path / "experiment.log"
    
    logger = setup_logger(
        name=f"bem_experiment_{experiment_name}",
        level=level,
        log_file=log_file,
        console_output=True,
    )
    
    return logger