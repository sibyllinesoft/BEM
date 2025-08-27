"""
Checkpoint Manager for BEM 2.0 Online Learning.

Manages model checkpoints for rollback capability during online learning.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import time
import json
import copy
from collections import deque


@dataclass
class Checkpoint:
    """Single model checkpoint."""
    
    # Identification
    step: int
    timestamp: float
    checkpoint_id: str
    
    # Model state
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    
    # Metadata
    metrics: Dict[str, float] = field(default_factory=dict)
    safety_status: str = "unknown"
    performance_score: float = 0.0
    
    # Quality indicators
    is_safe: bool = True
    drift_score: float = 0.0
    canary_passed: bool = True
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        total_bytes = 0
        for tensor in self.model_state.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without tensors for JSON serialization)."""
        return {
            'step': self.step,
            'timestamp': self.timestamp,
            'checkpoint_id': self.checkpoint_id,
            'metrics': self.metrics,
            'safety_status': self.safety_status,
            'performance_score': self.performance_score,
            'is_safe': self.is_safe,
            'drift_score': self.drift_score,
            'canary_passed': self.canary_passed
        }


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint manager."""
    
    # Storage
    max_checkpoints: int = 20
    save_directory: Optional[str] = "checkpoints"
    
    # Creation frequency
    save_frequency: int = 100  # Save every N steps
    safe_checkpoint_frequency: int = 500  # Force safe checkpoint every N steps
    
    # Cleanup policy
    keep_safe_checkpoints: bool = True
    keep_recent_checkpoints: int = 5
    
    # Performance thresholds
    min_performance_for_safe: float = 0.8
    max_drift_for_safe: float = 0.5
    
    # Compression
    compress_checkpoints: bool = True
    
    # Persistence
    save_metadata: bool = True
    metadata_file: str = "checkpoint_metadata.json"


class CheckpointManager:
    """
    Manages model checkpoints for safe rollback during online learning.
    
    Features:
    - Circular buffer of recent checkpoints
    - Safe checkpoint identification and retention
    - Automatic cleanup based on policies
    - Metadata tracking for checkpoint selection
    - Rollback to best safe checkpoint
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self.checkpoints: deque = deque(maxlen=config.max_checkpoints)
        self.safe_checkpoints: List[Checkpoint] = []
        
        # Directory setup
        if config.save_directory:
            self.save_dir = Path(config.save_directory)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        
        # Statistics
        self.total_checkpoints_created = 0
        self.safe_checkpoints_created = 0
        self.rollbacks_performed = 0
        
        # Current state
        self.last_checkpoint_step = 0
        self.last_safe_checkpoint_step = 0
        
        self.logger.info(f"CheckpointManager initialized: max={config.max_checkpoints}")
    
    def create_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        safety_status: str = "unknown",
        performance_score: float = 0.0,
        drift_score: float = 0.0,
        canary_passed: bool = True,
        force_safe: bool = False
    ) -> Checkpoint:
        """Create new checkpoint."""
        
        # Generate checkpoint ID
        checkpoint_id = f"checkpoint_{step}_{int(time.time())}"
        
        # Deep copy model state
        model_state = copy.deepcopy(model.state_dict())
        
        # Copy optimizer state if provided
        optimizer_state = None
        if optimizer is not None:
            optimizer_state = copy.deepcopy(optimizer.state_dict())
        
        # Copy scheduler state if provided
        scheduler_state = None
        if scheduler is not None:
            scheduler_state = copy.deepcopy(scheduler.state_dict())
        
        # Determine if checkpoint is safe
        is_safe = self._is_checkpoint_safe(
            performance_score, drift_score, canary_passed, force_safe
        )
        
        # Create checkpoint
        checkpoint = Checkpoint(
            step=step,
            timestamp=time.time(),
            checkpoint_id=checkpoint_id,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            metrics=metrics or {},
            safety_status=safety_status,
            performance_score=performance_score,
            is_safe=is_safe,
            drift_score=drift_score,
            canary_passed=canary_passed
        )
        
        # Add to storage
        self.checkpoints.append(checkpoint)
        
        if is_safe:
            self.safe_checkpoints.append(checkpoint)
            self.safe_checkpoints_created += 1
            self.last_safe_checkpoint_step = step
            self.logger.debug(f"Safe checkpoint created at step {step}")
        
        # Cleanup old safe checkpoints
        self._cleanup_safe_checkpoints()
        
        # Save to disk if configured
        if self.save_dir:
            self._save_checkpoint_to_disk(checkpoint)
        
        # Update statistics
        self.total_checkpoints_created += 1
        self.last_checkpoint_step = step
        
        self.logger.debug(f"Checkpoint created: {checkpoint_id} (safe={is_safe})")
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None
    
    def get_best_safe_checkpoint(self) -> Optional[Checkpoint]:
        """Get best safe checkpoint based on performance score."""
        if not self.safe_checkpoints:
            return None
        
        # Sort by performance score (descending) then by recency
        sorted_safe = sorted(
            self.safe_checkpoints,
            key=lambda cp: (cp.performance_score, cp.step),
            reverse=True
        )
        
        return sorted_safe[0]
    
    def get_checkpoint_by_step(self, step: int) -> Optional[Checkpoint]:
        """Get checkpoint closest to specified step."""
        if not self.checkpoints:
            return None
        
        # Find checkpoint with closest step
        best_checkpoint = min(
            self.checkpoints,
            key=lambda cp: abs(cp.step - step)
        )
        
        return best_checkpoint
    
    def rollback_to_safe_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Optional[Checkpoint]:
        """Rollback model to best safe checkpoint."""
        
        safe_checkpoint = self.get_best_safe_checkpoint()
        if safe_checkpoint is None:
            self.logger.error("No safe checkpoints available for rollback")
            return None
        
        try:
            # Load model state
            model.load_state_dict(safe_checkpoint.model_state)
            
            # Load optimizer state if available
            if optimizer is not None and safe_checkpoint.optimizer_state is not None:
                optimizer.load_state_dict(safe_checkpoint.optimizer_state)
            
            # Load scheduler state if available
            if scheduler is not None and safe_checkpoint.scheduler_state is not None:
                scheduler.load_state_dict(safe_checkpoint.scheduler_state)
            
            self.rollbacks_performed += 1
            
            self.logger.info(f"Rolled back to safe checkpoint from step {safe_checkpoint.step}")
            return safe_checkpoint
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return None
    
    def should_create_checkpoint(self, step: int, force_safe: bool = False) -> bool:
        """Determine if checkpoint should be created."""
        
        # Regular checkpoint frequency
        steps_since_last = step - self.last_checkpoint_step
        if steps_since_last >= self.config.save_frequency:
            return True
        
        # Safe checkpoint frequency
        if force_safe:
            steps_since_safe = step - self.last_safe_checkpoint_step
            if steps_since_safe >= self.config.safe_checkpoint_frequency:
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        # Memory usage
        total_memory = sum(cp.get_memory_usage() for cp in self.checkpoints)
        safe_memory = sum(cp.get_memory_usage() for cp in self.safe_checkpoints)
        
        # Performance statistics
        performance_scores = [cp.performance_score for cp in self.checkpoints if cp.performance_score > 0]
        drift_scores = [cp.drift_score for cp in self.checkpoints]
        
        recent_checkpoints = list(self.checkpoints)[-5:]  # Last 5
        
        return {
            'total_checkpoints': len(self.checkpoints),
            'safe_checkpoints': len(self.safe_checkpoints),
            'total_created': self.total_checkpoints_created,
            'safe_created': self.safe_checkpoints_created,
            'rollbacks_performed': self.rollbacks_performed,
            'memory_usage': {
                'total_mb': total_memory / (1024 * 1024),
                'safe_mb': safe_memory / (1024 * 1024)
            },
            'performance': {
                'mean_score': np.mean(performance_scores) if performance_scores else 0.0,
                'best_score': np.max(performance_scores) if performance_scores else 0.0,
                'mean_drift': np.mean(drift_scores) if drift_scores else 0.0
            },
            'recent_checkpoints': [
                {
                    'step': cp.step,
                    'is_safe': cp.is_safe,
                    'performance_score': cp.performance_score,
                    'drift_score': cp.drift_score,
                    'canary_passed': cp.canary_passed
                }
                for cp in recent_checkpoints
            ]
        }
    
    def cleanup_old_checkpoints(self):
        """Manual cleanup of old checkpoints."""
        self._cleanup_safe_checkpoints()
        self.logger.info("Checkpoint cleanup completed")
    
    def save_metadata(self):
        """Save checkpoint metadata to disk."""
        if not self.save_dir or not self.config.save_metadata:
            return
        
        metadata = {
            'statistics': self.get_statistics(),
            'config': {
                'max_checkpoints': self.config.max_checkpoints,
                'save_frequency': self.config.save_frequency,
                'safe_checkpoint_frequency': self.config.safe_checkpoint_frequency
            },
            'checkpoints': [cp.to_dict() for cp in self.checkpoints]
        }
        
        metadata_path = self.save_dir / self.config.metadata_file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Metadata saved to {metadata_path}")
    
    def load_metadata(self):
        """Load checkpoint metadata from disk."""
        if not self.save_dir or not self.config.save_metadata:
            return
        
        metadata_path = self.save_dir / self.config.metadata_file
        if not metadata_path.exists():
            return
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Restore statistics
            if 'statistics' in metadata:
                stats = metadata['statistics']
                self.total_checkpoints_created = stats.get('total_created', 0)
                self.safe_checkpoints_created = stats.get('safe_created', 0)
                self.rollbacks_performed = stats.get('rollbacks_performed', 0)
            
            self.logger.info(f"Metadata loaded from {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load metadata: {e}")
    
    def _is_checkpoint_safe(
        self,
        performance_score: float,
        drift_score: float,
        canary_passed: bool,
        force_safe: bool
    ) -> bool:
        """Determine if checkpoint should be marked as safe."""
        
        if force_safe:
            return True
        
        # Check performance threshold
        if performance_score < self.config.min_performance_for_safe:
            return False
        
        # Check drift threshold  
        if drift_score > self.config.max_drift_for_safe:
            return False
        
        # Check canary status
        if not canary_passed:
            return False
        
        return True
    
    def _cleanup_safe_checkpoints(self):
        """Cleanup old safe checkpoints based on retention policy."""
        if not self.config.keep_safe_checkpoints:
            return
        
        # Keep only recent safe checkpoints beyond a minimum
        min_keep = max(3, self.config.keep_recent_checkpoints)
        if len(self.safe_checkpoints) > min_keep:
            # Sort by step (keep most recent)
            self.safe_checkpoints.sort(key=lambda cp: cp.step, reverse=True)
            self.safe_checkpoints = self.safe_checkpoints[:min_keep]
    
    def _save_checkpoint_to_disk(self, checkpoint: Checkpoint):
        """Save checkpoint to disk."""
        if not self.save_dir:
            return
        
        checkpoint_path = self.save_dir / f"{checkpoint.checkpoint_id}.pt"
        
        save_data = {
            'step': checkpoint.step,
            'timestamp': checkpoint.timestamp,
            'model_state_dict': checkpoint.model_state,
            'optimizer_state_dict': checkpoint.optimizer_state,
            'scheduler_state_dict': checkpoint.scheduler_state,
            'metrics': checkpoint.metrics,
            'safety_status': checkpoint.safety_status,
            'performance_score': checkpoint.performance_score,
            'is_safe': checkpoint.is_safe,
            'drift_score': checkpoint.drift_score,
            'canary_passed': checkpoint.canary_passed
        }
        
        try:
            torch.save(save_data, checkpoint_path)
            self.logger.debug(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to disk: {e}")
    
    def load_checkpoint_from_disk(self, checkpoint_path: str) -> Optional[Checkpoint]:
        """Load checkpoint from disk."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        try:
            data = torch.load(checkpoint_path, map_location='cpu')
            
            checkpoint = Checkpoint(
                step=data['step'],
                timestamp=data['timestamp'],
                checkpoint_id=checkpoint_path.stem,
                model_state=data['model_state_dict'],
                optimizer_state=data.get('optimizer_state_dict'),
                scheduler_state=data.get('scheduler_state_dict'),
                metrics=data.get('metrics', {}),
                safety_status=data.get('safety_status', 'unknown'),
                performance_score=data.get('performance_score', 0.0),
                is_safe=data.get('is_safe', True),
                drift_score=data.get('drift_score', 0.0),
                canary_passed=data.get('canary_passed', True)
            )
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None


# Utility functions
def create_checkpoint_manager(
    max_checkpoints: int = 20,
    save_frequency: int = 100,
    save_directory: Optional[str] = "checkpoints"
) -> CheckpointManager:
    """Create checkpoint manager with specified configuration."""
    config = CheckpointConfig(
        max_checkpoints=max_checkpoints,
        save_frequency=save_frequency,
        save_directory=save_directory
    )
    return CheckpointManager(config)


# Example usage
if __name__ == "__main__":
    # Create test model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    manager = create_checkpoint_manager(max_checkpoints=5)
    
    # Create some checkpoints
    for step in range(0, 500, 100):
        # Simulate varying performance and safety
        performance = 0.7 + 0.3 * np.random.random()
        drift = np.random.random() * 0.6
        canary = np.random.random() > 0.2  # 80% pass rate
        
        checkpoint = manager.create_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            performance_score=performance,
            drift_score=drift,
            canary_passed=canary,
            metrics={'loss': np.random.random()}
        )
        
        print(f"Step {step}: checkpoint created (safe={checkpoint.is_safe})")
    
    # Test rollback
    print(f"\nBefore rollback - current step: {step}")
    rollback_checkpoint = manager.rollback_to_safe_checkpoint(model, optimizer)
    if rollback_checkpoint:
        print(f"Rolled back to step: {rollback_checkpoint.step}")
    
    # Print statistics
    stats = manager.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total checkpoints: {stats['total_checkpoints']}")
    print(f"  Safe checkpoints: {stats['safe_checkpoints']}")
    print(f"  Best performance: {stats['performance']['best_score']:.3f}")
    print(f"  Memory usage: {stats['memory_usage']['total_mb']:.1f} MB")