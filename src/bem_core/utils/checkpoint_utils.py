"""Unified checkpoint utilities for BEM experiments.

Provides standardized checkpointing functionality for saving
and loading model states across all BEM components.
"""

import os
import glob
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def save_checkpoint(
    state_dict: Dict[str, Any],
    checkpoint_path: Union[str, Path],
    is_best: bool = False,
) -> None:
    """Save model checkpoint.
    
    Args:
        state_dict: State dictionary to save
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(state_dict, checkpoint_path)
    
    # Create best model symlink/copy if this is the best
    if is_best:
        best_path = checkpoint_path.parent / "best_model.pt"
        if best_path.exists():
            best_path.unlink()
        
        # Create hard link or copy
        try:
            best_path.hardlink_to(checkpoint_path)
        except OSError:
            # Fallback to copy if hard link fails
            import shutil
            shutil.copy2(checkpoint_path, best_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Loaded state dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    return checkpoint


def find_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    pattern: str = "checkpoint-*.pt",
) -> Optional[Path]:
    """Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        pattern: Glob pattern to match checkpoint files
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoints matching pattern
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return checkpoint_files[0]


def list_checkpoints(
    checkpoint_dir: Union[str, Path],
    pattern: str = "checkpoint-*.pt",
) -> List[Path]:
    """List all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern to match checkpoint files
        
    Returns:
        List of checkpoint paths sorted by modification time
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    # Find all checkpoints
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return checkpoint_files


def cleanup_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 3,
    pattern: str = "checkpoint-*.pt",
    preserve_best: bool = True,
) -> None:
    """Clean up old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        pattern: Glob pattern to match checkpoint files
        preserve_best: Whether to preserve best_model.pt
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoints
    checkpoints = list_checkpoints(checkpoint_dir, pattern)
    
    # Keep only the most recent N checkpoints
    checkpoints_to_remove = checkpoints[keep_last_n:]
    
    # Preserve best model if requested
    best_model_path = checkpoint_dir / "best_model.pt"
    
    for checkpoint_path in checkpoints_to_remove:
        # Don't delete if it's the best model (or linked to it)
        if preserve_best and best_model_path.exists():
            try:
                if checkpoint_path.samefile(best_model_path):
                    continue
            except OSError:
                pass
        
        try:
            checkpoint_path.unlink()
        except OSError:
            pass  # Ignore errors when deleting


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint metadata only
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    info = {
        "path": str(checkpoint_path),
        "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        "modified": checkpoint_path.stat().st_mtime,
    }
    
    # Extract metadata from checkpoint
    if "global_step" in checkpoint:
        info["global_step"] = checkpoint["global_step"]
    if "current_epoch" in checkpoint:
        info["current_epoch"] = checkpoint["current_epoch"]
    if "best_metric" in checkpoint:
        info["best_metric"] = checkpoint["best_metric"]
    if "config" in checkpoint:
        info["config_keys"] = list(checkpoint["config"].keys())
    
    # Count parameters if model state dict is present
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        total_params = sum(p.numel() for p in model_state.values() if p.dtype != torch.bool)
        info["total_parameters"] = total_params
    
    return info


def resume_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load on
        
    Returns:
        Dictionary with resumed training state
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Return training state
    training_state = {
        "global_step": checkpoint.get("global_step", 0),
        "current_epoch": checkpoint.get("current_epoch", 0),
        "best_metric": checkpoint.get("best_metric", float('-inf')),
    }
    
    return training_state