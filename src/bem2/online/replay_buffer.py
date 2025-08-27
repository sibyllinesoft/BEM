"""
Experience Replay Buffer for BEM 2.0 Online Learning.

Implements replay buffer to maintain old knowledge during online updates.
Stores 10k samples as specified in TODO.md requirements.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
import random
import logging
from collections import deque
import pickle
import time
from pathlib import Path


@dataclass
class Experience:
    """Single experience/sample for replay buffer."""
    
    # Input data
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    
    # Target/label data
    targets: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    
    # Context information
    context_length: int = 0
    task_id: Optional[str] = None
    domain: Optional[str] = None
    
    # Feedback information
    feedback_score: Optional[float] = None
    feedback_type: Optional[str] = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0
    replay_count: int = 0
    
    def to_device(self, device: torch.device) -> 'Experience':
        """Move experience to specified device."""
        return Experience(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            targets=self.targets.to(device) if self.targets is not None else None,
            labels=self.labels.to(device) if self.labels is not None else None,
            context_length=self.context_length,
            task_id=self.task_id,
            domain=self.domain,
            feedback_score=self.feedback_score,
            feedback_type=self.feedback_type,
            timestamp=self.timestamp,
            priority=self.priority,
            replay_count=self.replay_count
        )
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        total_bytes = 0
        if self.input_ids is not None:
            total_bytes += self.input_ids.numel() * self.input_ids.element_size()
        if self.attention_mask is not None:
            total_bytes += self.attention_mask.numel() * self.attention_mask.element_size()
        if self.targets is not None:
            total_bytes += self.targets.numel() * self.targets.element_size()
        if self.labels is not None:
            total_bytes += self.labels.numel() * self.labels.element_size()
        return total_bytes


@dataclass
class ReplayConfig:
    """Configuration for replay buffer."""
    
    # Buffer size
    max_size: int = 10000  # As specified in TODO.md
    
    # Sampling configuration
    batch_size: int = 32
    min_replay_size: int = 1000  # Minimum samples before replay
    
    # Prioritization
    enable_prioritized_replay: bool = True
    priority_alpha: float = 0.6  # Prioritization strength
    priority_beta: float = 0.4  # Importance sampling correction
    priority_beta_increment: float = 0.001
    
    # Diversity
    enable_diversity_sampling: bool = True
    diversity_weight: float = 0.3
    
    # Storage efficiency
    compress_experiences: bool = True
    max_context_length: int = 512
    
    # Persistence
    save_frequency: int = 1000  # Save every N additions
    save_path: Optional[str] = None


class ReplayBuffer:
    """
    Experience replay buffer for maintaining old knowledge during online learning.
    
    Features:
    - Fixed size circular buffer (10k samples)
    - Prioritized sampling based on loss/feedback
    - Diversity sampling to avoid mode collapse
    - Compression to reduce memory usage
    - Persistence for recovery
    """
    
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self.buffer: deque = deque(maxlen=config.max_size)
        self.priorities: deque = deque(maxlen=config.max_size)
        
        # Indexing for efficient sampling
        self.domain_index: Dict[str, List[int]] = {}
        self.task_index: Dict[str, List[int]] = {}
        self.feedback_index: Dict[str, List[int]] = {}
        
        # Statistics
        self.total_added = 0
        self.total_sampled = 0
        self.last_save_count = 0
        
        # Current priority beta for importance sampling
        self.current_priority_beta = config.priority_beta
        
        # Memory tracking
        self.current_memory_usage = 0  # bytes
        
        self.logger.info(f"ReplayBuffer initialized with max_size={config.max_size}")
    
    def add(self, experience: Experience):
        """Add new experience to buffer."""
        # Update replay count if experience already exists
        experience.replay_count = 0
        
        # Compress if needed
        if self.config.compress_experiences:
            experience = self._compress_experience(experience)
        
        # Add to buffer
        if len(self.buffer) == self.config.max_size:
            # Remove oldest experience from indices
            oldest_idx = 0  # Will be overwritten
            self._remove_from_indices(oldest_idx)
        
        self.buffer.append(experience)
        new_idx = len(self.buffer) - 1
        
        # Add to indices
        self._add_to_indices(experience, new_idx)
        
        # Set initial priority
        initial_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(initial_priority)
        
        # Update statistics
        self.total_added += 1
        self.current_memory_usage += experience.get_memory_usage()
        
        # Save periodically
        if (self.config.save_path and 
            self.total_added - self.last_save_count >= self.config.save_frequency):
            self.save(self.config.save_path)
        
        if self.total_added % 1000 == 0:
            self.logger.debug(f"Buffer size: {len(self.buffer)}, total added: {self.total_added}")
    
    def sample(self, batch_size: Optional[int] = None) -> List[Experience]:
        """Sample batch of experiences for replay."""
        if len(self.buffer) < self.config.min_replay_size:
            self.logger.warning(f"Buffer too small for sampling: {len(self.buffer)} < {self.config.min_replay_size}")
            return []
        
        batch_size = batch_size or self.config.batch_size
        batch_size = min(batch_size, len(self.buffer))
        
        if self.config.enable_prioritized_replay:
            indices = self._sample_prioritized(batch_size)
        else:
            indices = random.sample(range(len(self.buffer)), batch_size)
        
        # Apply diversity sampling if enabled
        if self.config.enable_diversity_sampling:
            indices = self._apply_diversity_sampling(indices, batch_size)
        
        # Extract experiences and update replay counts
        experiences = []
        for idx in indices:
            exp = self.buffer[idx]
            exp.replay_count += 1
            experiences.append(exp)
        
        self.total_sampled += len(experiences)
        
        # Update priority beta for importance sampling
        self.current_priority_beta = min(
            1.0, 
            self.current_priority_beta + self.config.priority_beta_increment
        )
        
        return experiences
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay."""
        if not self.config.enable_prioritized_replay:
            return
        
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = max(priority, 1e-6)  # Avoid zero priority
    
    def sample_by_domain(self, domain: str, batch_size: int) -> List[Experience]:
        """Sample experiences from specific domain."""
        if domain not in self.domain_index or not self.domain_index[domain]:
            return []
        
        available_indices = self.domain_index[domain]
        sample_size = min(batch_size, len(available_indices))
        sampled_indices = random.sample(available_indices, sample_size)
        
        return [self.buffer[idx] for idx in sampled_indices]
    
    def sample_by_feedback(self, feedback_type: str, batch_size: int) -> List[Experience]:
        """Sample experiences with specific feedback type."""
        if feedback_type not in self.feedback_index or not self.feedback_index[feedback_type]:
            return []
        
        available_indices = self.feedback_index[feedback_type]
        sample_size = min(batch_size, len(available_indices))
        sampled_indices = random.sample(available_indices, sample_size)
        
        return [self.buffer[idx] for idx in sampled_indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        domain_counts = {domain: len(indices) for domain, indices in self.domain_index.items()}
        task_counts = {task: len(indices) for task, indices in self.task_index.items()}
        feedback_counts = {fb: len(indices) for fb, indices in self.feedback_index.items()}
        
        # Priority statistics
        priority_stats = {}
        if self.priorities:
            priorities_tensor = torch.tensor(list(self.priorities))
            priority_stats = {
                'mean': torch.mean(priorities_tensor).item(),
                'std': torch.std(priorities_tensor).item(),
                'min': torch.min(priorities_tensor).item(),
                'max': torch.max(priorities_tensor).item()
            }
        
        return {
            'buffer_size': len(self.buffer),
            'max_size': self.config.max_size,
            'utilization': len(self.buffer) / self.config.max_size,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
            'domain_counts': domain_counts,
            'task_counts': task_counts,
            'feedback_counts': feedback_counts,
            'priority_stats': priority_stats,
            'current_priority_beta': self.current_priority_beta
        }
    
    def save(self, filepath: str):
        """Save buffer to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            'buffer': list(self.buffer),
            'priorities': list(self.priorities),
            'config': self.config,
            'statistics': {
                'total_added': self.total_added,
                'total_sampled': self.total_sampled,
                'current_priority_beta': self.current_priority_beta
            }
        }
        
        # Save with pickle (handles torch tensors)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.last_save_count = self.total_added
        self.logger.info(f"Buffer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load buffer from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            self.logger.warning(f"Buffer file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore data
        self.buffer = deque(save_data['buffer'], maxlen=self.config.max_size)
        self.priorities = deque(save_data['priorities'], maxlen=self.config.max_size)
        
        # Restore statistics
        if 'statistics' in save_data:
            stats = save_data['statistics']
            self.total_added = stats.get('total_added', 0)
            self.total_sampled = stats.get('total_sampled', 0)
            self.current_priority_beta = stats.get('current_priority_beta', self.config.priority_beta)
        
        # Rebuild indices
        self._rebuild_indices()
        
        # Update memory usage
        self.current_memory_usage = sum(exp.get_memory_usage() for exp in self.buffer)
        
        self.logger.info(f"Buffer loaded from {filepath}, size: {len(self.buffer)}")
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.domain_index.clear()
        self.task_index.clear()
        self.feedback_index.clear()
        self.current_memory_usage = 0
        self.logger.info("Buffer cleared")
    
    def _sample_prioritized(self, batch_size: int) -> List[int]:
        """Sample indices using prioritized replay."""
        priorities = torch.tensor(list(self.priorities), dtype=torch.float32)
        
        # Apply alpha for prioritization
        priorities = priorities ** self.config.priority_alpha
        probabilities = priorities / torch.sum(priorities)
        
        # Sample indices
        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        return indices.tolist()
    
    def _apply_diversity_sampling(self, indices: List[int], target_size: int) -> List[int]:
        """Apply diversity sampling to reduce redundancy."""
        if len(indices) <= target_size:
            return indices
        
        # Group by domain and task for diversity
        domain_groups = {}
        for idx in indices:
            exp = self.buffer[idx]
            domain = exp.domain or "unknown"
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(idx)
        
        # Sample from each domain proportionally
        diverse_indices = []
        domains = list(domain_groups.keys())
        samples_per_domain = target_size // len(domains)
        remaining_samples = target_size % len(domains)
        
        for i, domain in enumerate(domains):
            domain_indices = domain_groups[domain]
            num_samples = samples_per_domain + (1 if i < remaining_samples else 0)
            num_samples = min(num_samples, len(domain_indices))
            
            sampled = random.sample(domain_indices, num_samples)
            diverse_indices.extend(sampled)
        
        return diverse_indices[:target_size]
    
    def _compress_experience(self, experience: Experience) -> Experience:
        """Compress experience to reduce memory usage."""
        # Truncate context if too long
        if (experience.input_ids is not None and 
            experience.input_ids.shape[-1] > self.config.max_context_length):
            
            max_len = self.config.max_context_length
            experience.input_ids = experience.input_ids[..., :max_len]
            
            if experience.attention_mask is not None:
                experience.attention_mask = experience.attention_mask[..., :max_len]
            
            if experience.targets is not None and experience.targets.shape[-1] > max_len:
                experience.targets = experience.targets[..., :max_len]
        
        return experience
    
    def _add_to_indices(self, experience: Experience, index: int):
        """Add experience to search indices."""
        if experience.domain:
            if experience.domain not in self.domain_index:
                self.domain_index[experience.domain] = []
            self.domain_index[experience.domain].append(index)
        
        if experience.task_id:
            if experience.task_id not in self.task_index:
                self.task_index[experience.task_id] = []
            self.task_index[experience.task_id].append(index)
        
        if experience.feedback_type:
            if experience.feedback_type not in self.feedback_index:
                self.feedback_index[experience.feedback_type] = []
            self.feedback_index[experience.feedback_type].append(index)
    
    def _remove_from_indices(self, index: int):
        """Remove experience from search indices."""
        # This is called when buffer is full and oldest experience is removed
        # Since we use deque with maxlen, we need to update all indices
        # For simplicity, we rebuild indices periodically
        if len(self.buffer) == self.config.max_size:
            self._rebuild_indices()
    
    def _rebuild_indices(self):
        """Rebuild all search indices."""
        self.domain_index.clear()
        self.task_index.clear()
        self.feedback_index.clear()
        
        for idx, experience in enumerate(self.buffer):
            self._add_to_indices(experience, idx)


# Utility functions
def create_replay_buffer(
    max_size: int = 10000,
    batch_size: int = 32,
    enable_prioritized_replay: bool = True,
    save_path: Optional[str] = None
) -> ReplayBuffer:
    """Create replay buffer with specified configuration."""
    config = ReplayConfig(
        max_size=max_size,
        batch_size=batch_size,
        enable_prioritized_replay=enable_prioritized_replay,
        save_path=save_path
    )
    return ReplayBuffer(config)


def create_experience_from_batch(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    domain: Optional[str] = None,
    feedback_score: Optional[float] = None
) -> List[Experience]:
    """Create list of experiences from a batch."""
    batch_size = input_ids.shape[0]
    experiences = []
    
    for i in range(batch_size):
        exp = Experience(
            input_ids=input_ids[i],
            attention_mask=attention_mask[i] if attention_mask is not None else None,
            targets=targets[i] if targets is not None else None,
            domain=domain,
            feedback_score=feedback_score,
            context_length=input_ids[i].shape[-1]
        )
        experiences.append(exp)
    
    return experiences


# Example usage
if __name__ == "__main__":
    # Create replay buffer
    buffer = create_replay_buffer(max_size=1000)
    
    # Create sample experiences
    for i in range(100):
        exp = Experience(
            input_ids=torch.randint(0, 1000, (50,)),
            attention_mask=torch.ones(50),
            targets=torch.randint(0, 10, (1,)),
            domain=f"domain_{i % 3}",
            task_id=f"task_{i % 5}",
            feedback_score=random.uniform(0, 1),
            feedback_type="positive" if random.random() > 0.5 else "negative"
        )
        buffer.add(exp)
    
    print("Buffer statistics:")
    stats = buffer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Sample batch
    batch = buffer.sample(batch_size=10)
    print(f"\nSampled batch of size: {len(batch)}")
    
    # Sample by domain
    domain_batch = buffer.sample_by_domain("domain_0", 5)
    print(f"Domain-specific batch size: {len(domain_batch)}")