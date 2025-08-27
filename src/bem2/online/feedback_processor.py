"""
Feedback Processor for BEM 2.0 Online Learning.

Processes and analyzes feedback signals to create training batches for online learning.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
import logging
from enum import Enum
import time
from collections import defaultdict, deque

from .streaming import FeedbackSignal, FeedbackType


class ProcessingStrategy(Enum):
    """Strategies for processing feedback into training signals."""
    IMMEDIATE = "immediate"  # Process each feedback immediately
    BATCHED = "batched"  # Batch feedback before processing
    WEIGHTED = "weighted"  # Weight by confidence and recency
    CONTRASTIVE = "contrastive"  # Create positive/negative pairs
    PREFERENCE = "preference"  # Learn from preferences


@dataclass
class ProcessedFeedback:
    """Processed feedback ready for training."""
    
    # Training data
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    target_reward: float = 0.0  # Target reward/preference
    
    # Context
    feedback_type: FeedbackType = FeedbackType.THUMBS_UP
    confidence: float = 1.0
    weight: float = 1.0  # Training weight
    
    # Metadata
    session_id: str = ""
    original_feedback: Optional[FeedbackSignal] = None
    processing_time: float = field(default_factory=time.time)
    
    def to_batch_dict(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert to batch dictionary for training."""
        batch = {
            'input_ids': self.input_ids.to(device),
            'target_reward': torch.tensor(self.target_reward, device=device),
            'weight': torch.tensor(self.weight, device=device)
        }
        
        if self.attention_mask is not None:
            batch['attention_mask'] = self.attention_mask.to(device)
        
        return batch


@dataclass
class FeedbackProcessorConfig:
    """Configuration for feedback processor."""
    
    # Processing strategy
    strategy: ProcessingStrategy = ProcessingStrategy.WEIGHTED
    
    # Reward mapping
    reward_mapping: Dict[FeedbackType, float] = field(default_factory=lambda: {
        FeedbackType.THUMBS_UP: 1.0,
        FeedbackType.THUMBS_DOWN: -1.0,
        FeedbackType.TOOL_SUCCESS: 0.8,
        FeedbackType.TOOL_FAILURE: -0.8,
        FeedbackType.IMPLICIT_POSITIVE: 0.5,
        FeedbackType.IMPLICIT_NEGATIVE: -0.5
    })
    
    # Weighting parameters
    recency_decay: float = 0.95  # Decay factor for older feedback
    confidence_power: float = 2.0  # Power for confidence weighting
    min_confidence: float = 0.3  # Minimum confidence to include
    
    # Batch processing
    batch_size: int = 32
    max_sequence_length: int = 512
    
    # Session aggregation
    enable_session_aggregation: bool = True
    session_weight_boost: float = 1.2  # Boost weight for consistent session feedback
    
    # Contrastive learning (for preference training)
    enable_contrastive_pairs: bool = False
    contrastive_margin: float = 0.5
    
    # Quality filtering
    filter_contradictory: bool = True
    contradiction_threshold: float = 0.7  # Threshold for detecting contradictions
    
    # Temporal considerations
    max_feedback_age: float = 3600.0  # 1 hour
    temporal_weighting: bool = True


class FeedbackProcessor:
    """
    Processes feedback signals into training data for online learning.
    
    Features:
    - Multiple processing strategies (immediate, batched, weighted, contrastive)
    - Temporal and confidence weighting
    - Session-level aggregation and consistency
    - Contradiction detection and resolution
    - Reward shaping for different feedback types
    """
    
    def __init__(self, config: FeedbackProcessorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.processed_feedback: deque = deque(maxlen=10000)
        self.session_feedback: Dict[str, List[ProcessedFeedback]] = {}
        
        # Statistics
        self.total_feedback_processed = 0
        self.feedback_by_type: Dict[FeedbackType, int] = defaultdict(int)
        self.contradictions_detected = 0
        self.session_aggregations = 0
        
        # Quality tracking
        self.quality_scores: deque = deque(maxlen=1000)
        self.processing_times: deque = deque(maxlen=1000)
        
        self.logger.info(f"FeedbackProcessor initialized with strategy: {config.strategy.value}")
    
    def process_feedback_signals(
        self, 
        signals: List[FeedbackSignal],
        tokenizer: Optional[Any] = None
    ) -> List[ProcessedFeedback]:
        """Process a batch of feedback signals."""
        
        start_time = time.time()
        processed = []
        
        # Filter by quality
        valid_signals = self._filter_signals(signals)
        
        # Process based on strategy
        if self.config.strategy == ProcessingStrategy.IMMEDIATE:
            processed = self._process_immediate(valid_signals, tokenizer)
        elif self.config.strategy == ProcessingStrategy.BATCHED:
            processed = self._process_batched(valid_signals, tokenizer)
        elif self.config.strategy == ProcessingStrategy.WEIGHTED:
            processed = self._process_weighted(valid_signals, tokenizer)
        elif self.config.strategy == ProcessingStrategy.CONTRASTIVE:
            processed = self._process_contrastive(valid_signals, tokenizer)
        elif self.config.strategy == ProcessingStrategy.PREFERENCE:
            processed = self._process_preference(valid_signals, tokenizer)
        
        # Post-processing
        processed = self._apply_post_processing(processed)
        
        # Update statistics
        self._update_statistics(processed, start_time)
        
        # Store processed feedback
        self.processed_feedback.extend(processed)
        
        return processed
    
    def process_single_feedback(
        self,
        signal: FeedbackSignal,
        tokenizer: Optional[Any] = None
    ) -> Optional[ProcessedFeedback]:
        """Process a single feedback signal."""
        processed = self.process_feedback_signals([signal], tokenizer)
        return processed[0] if processed else None
    
    def get_training_batch(
        self,
        batch_size: Optional[int] = None,
        device: torch.device = torch.device('cpu')
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get a batch of processed feedback for training."""
        batch_size = batch_size or self.config.batch_size
        
        if len(self.processed_feedback) < batch_size:
            return None
        
        # Sample batch
        batch_feedback = []
        for _ in range(batch_size):
            if self.processed_feedback:
                batch_feedback.append(self.processed_feedback.popleft())
        
        if not batch_feedback:
            return None
        
        # Convert to batch tensors
        return self._create_batch_tensors(batch_feedback, device)
    
    def get_session_feedback(self, session_id: str) -> List[ProcessedFeedback]:
        """Get processed feedback for a specific session."""
        return self.session_feedback.get(session_id, [])
    
    def get_session_reward_trend(self, session_id: str) -> Optional[Tuple[float, float]]:
        """Get reward trend for a session (mean, std)."""
        session_feedback = self.get_session_feedback(session_id)
        if not session_feedback:
            return None
        
        rewards = [fb.target_reward for fb in session_feedback]
        return float(np.mean(rewards)), float(np.std(rewards))
    
    def detect_contradictions(self, signals: List[FeedbackSignal]) -> List[Tuple[int, int, float]]:
        """Detect contradictory feedback signals. Returns list of (idx1, idx2, contradiction_score)."""
        contradictions = []
        
        for i, signal1 in enumerate(signals):
            for j, signal2 in enumerate(signals[i+1:], i+1):
                if self._are_contradictory(signal1, signal2):
                    contradiction_score = self._compute_contradiction_score(signal1, signal2)
                    if contradiction_score > self.config.contradiction_threshold:
                        contradictions.append((i, j, contradiction_score))
        
        return contradictions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        
        # Quality scores
        quality_stats = {}
        if self.quality_scores:
            quality_stats = {
                'mean': np.mean(list(self.quality_scores)),
                'std': np.std(list(self.quality_scores)),
                'min': np.min(list(self.quality_scores)),
                'max': np.max(list(self.quality_scores))
            }
        
        # Processing times
        timing_stats = {}
        if self.processing_times:
            timing_stats = {
                'mean_ms': np.mean(list(self.processing_times)) * 1000,
                'p95_ms': np.percentile(list(self.processing_times), 95) * 1000,
                'max_ms': np.max(list(self.processing_times)) * 1000
            }
        
        return {
            'processing': {
                'total_processed': self.total_feedback_processed,
                'feedback_by_type': dict(self.feedback_by_type),
                'contradictions_detected': self.contradictions_detected,
                'session_aggregations': self.session_aggregations,
                'pending_feedback': len(self.processed_feedback)
            },
            'sessions': {
                'active_sessions': len(self.session_feedback),
                'total_session_feedback': sum(len(fb) for fb in self.session_feedback.values())
            },
            'quality': quality_stats,
            'timing': timing_stats
        }
    
    def _filter_signals(self, signals: List[FeedbackSignal]) -> List[FeedbackSignal]:
        """Filter signals by quality criteria."""
        valid_signals = []
        
        for signal in signals:
            # Confidence filter
            if signal.confidence < self.config.min_confidence:
                continue
            
            # Age filter
            if self.config.max_feedback_age > 0:
                age = time.time() - signal.timestamp
                if age > self.config.max_feedback_age:
                    continue
            
            # Feedback type filter
            if signal.feedback_type not in self.config.reward_mapping:
                continue
            
            valid_signals.append(signal)
        
        return valid_signals
    
    def _process_immediate(
        self,
        signals: List[FeedbackSignal],
        tokenizer: Optional[Any]
    ) -> List[ProcessedFeedback]:
        """Process feedback immediately without aggregation."""
        processed = []
        
        for signal in signals:
            # Tokenize input
            input_ids, attention_mask = self._tokenize_signal(signal, tokenizer)
            if input_ids is None:
                continue
            
            # Map to reward
            target_reward = self.config.reward_mapping[signal.feedback_type]
            
            # Create processed feedback
            processed_fb = ProcessedFeedback(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_reward=target_reward,
                feedback_type=signal.feedback_type,
                confidence=signal.confidence,
                weight=signal.confidence,
                session_id=signal.session_id,
                original_feedback=signal
            )
            
            processed.append(processed_fb)
        
        return processed
    
    def _process_weighted(
        self,
        signals: List[FeedbackSignal],
        tokenizer: Optional[Any]
    ) -> List[ProcessedFeedback]:
        """Process feedback with temporal and confidence weighting."""
        processed = []
        current_time = time.time()
        
        for signal in signals:
            # Tokenize input
            input_ids, attention_mask = self._tokenize_signal(signal, tokenizer)
            if input_ids is None:
                continue
            
            # Compute weights
            confidence_weight = signal.confidence ** self.config.confidence_power
            
            if self.config.temporal_weighting:
                age = current_time - signal.timestamp
                temporal_weight = self.config.recency_decay ** (age / 3600.0)  # Hours
            else:
                temporal_weight = 1.0
            
            total_weight = confidence_weight * temporal_weight
            
            # Map to reward
            base_reward = self.config.reward_mapping[signal.feedback_type]
            target_reward = base_reward * signal.confidence  # Scale by confidence
            
            # Create processed feedback
            processed_fb = ProcessedFeedback(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_reward=target_reward,
                feedback_type=signal.feedback_type,
                confidence=signal.confidence,
                weight=total_weight,
                session_id=signal.session_id,
                original_feedback=signal
            )
            
            processed.append(processed_fb)
        
        return processed
    
    def _process_contrastive(
        self,
        signals: List[FeedbackSignal],
        tokenizer: Optional[Any]
    ) -> List[ProcessedFeedback]:
        """Process feedback to create contrastive pairs."""
        # First process individually
        processed = self._process_weighted(signals, tokenizer)
        
        # Then create contrastive pairs
        contrastive_pairs = []
        
        for i, fb1 in enumerate(processed):
            for j, fb2 in enumerate(processed[i+1:], i+1):
                # Check if suitable for contrastive learning
                if self._should_create_contrastive_pair(fb1, fb2):
                    pair = self._create_contrastive_pair(fb1, fb2)
                    contrastive_pairs.extend(pair)
        
        processed.extend(contrastive_pairs)
        return processed
    
    def _process_preference(
        self,
        signals: List[FeedbackSignal],
        tokenizer: Optional[Any]
    ) -> List[ProcessedFeedback]:
        """Process feedback for preference learning."""
        # Group by session for preference comparison
        session_groups = defaultdict(list)
        for signal in signals:
            session_groups[signal.session_id].append(signal)
        
        processed = []
        
        for session_id, session_signals in session_groups.items():
            # Sort by timestamp
            session_signals.sort(key=lambda x: x.timestamp)
            
            # Create preference pairs within session
            for i, signal1 in enumerate(session_signals):
                for j, signal2 in enumerate(session_signals[i+1:], i+1):
                    if self._should_create_preference_pair(signal1, signal2):
                        pair = self._create_preference_pair(signal1, signal2, tokenizer)
                        processed.extend(pair)
        
        return processed
    
    def _process_batched(
        self,
        signals: List[FeedbackSignal],
        tokenizer: Optional[Any]
    ) -> List[ProcessedFeedback]:
        """Process feedback in batches with aggregation."""
        # First process individually
        processed = self._process_weighted(signals, tokenizer)
        
        # Then apply session aggregation if enabled
        if self.config.enable_session_aggregation:
            processed = self._apply_session_aggregation(processed)
        
        return processed
    
    def _apply_session_aggregation(
        self,
        processed: List[ProcessedFeedback]
    ) -> List[ProcessedFeedback]:
        """Apply session-level aggregation to boost consistent feedback."""
        
        # Group by session
        session_groups = defaultdict(list)
        for fb in processed:
            session_groups[fb.session_id].append(fb)
        
        aggregated = []
        
        for session_id, session_feedback in session_groups.items():
            if len(session_feedback) <= 1:
                aggregated.extend(session_feedback)
                continue
            
            # Analyze session consistency
            rewards = [fb.target_reward for fb in session_feedback]
            reward_consistency = 1.0 - np.std(rewards) / (np.mean(np.abs(rewards)) + 1e-8)
            
            # Boost weights for consistent sessions
            if reward_consistency > 0.7:  # Consistent session
                boost_factor = self.config.session_weight_boost
                for fb in session_feedback:
                    fb.weight *= boost_factor
                self.session_aggregations += 1
            
            aggregated.extend(session_feedback)
        
        return aggregated
    
    def _apply_post_processing(self, processed: List[ProcessedFeedback]) -> List[ProcessedFeedback]:
        """Apply post-processing steps."""
        
        # Store session feedback
        for fb in processed:
            if fb.session_id:
                if fb.session_id not in self.session_feedback:
                    self.session_feedback[fb.session_id] = []
                self.session_feedback[fb.session_id].append(fb)
        
        # Filter contradictions if enabled
        if self.config.filter_contradictory:
            processed = self._filter_contradictory_feedback(processed)
        
        return processed
    
    def _filter_contradictory_feedback(self, processed: List[ProcessedFeedback]) -> List[ProcessedFeedback]:
        """Filter out contradictory feedback."""
        # Simple implementation: remove lower confidence feedback in contradictory pairs
        filtered = []
        
        for i, fb1 in enumerate(processed):
            keep = True
            
            for j, fb2 in enumerate(processed):
                if i == j:
                    continue
                
                if (fb1.session_id == fb2.session_id and 
                    fb1.original_feedback and fb2.original_feedback and
                    self._are_contradictory(fb1.original_feedback, fb2.original_feedback)):
                    
                    # Keep higher confidence feedback
                    if fb1.confidence < fb2.confidence:
                        keep = False
                        break
            
            if keep:
                filtered.append(fb1)
            else:
                self.contradictions_detected += 1
        
        return filtered
    
    def _tokenize_signal(
        self,
        signal: FeedbackSignal,
        tokenizer: Optional[Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Tokenize feedback signal."""
        
        # Combine input and output for context
        text = signal.input_text
        if signal.output_text:
            text = f"{signal.input_text} [SEP] {signal.output_text}"
        
        if not text:
            return None, None
        
        # Simple tokenization if no tokenizer provided
        if tokenizer is None:
            # Use character-level tokenization as fallback
            char_ids = [ord(c) % 1000 for c in text[:self.config.max_sequence_length]]
            input_ids = torch.tensor(char_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask
        
        # Use provided tokenizer
        try:
            encoding = tokenizer(
                text,
                max_length=self.config.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return encoding['input_ids'][0], encoding.get('attention_mask', [None])[0]
            
        except Exception as e:
            self.logger.warning(f"Tokenization failed: {e}")
            return None, None
    
    def _are_contradictory(self, signal1: FeedbackSignal, signal2: FeedbackSignal) -> bool:
        """Check if two signals are contradictory."""
        # Same session, similar input, opposite feedback
        if (signal1.session_id == signal2.session_id and
            abs(signal1.timestamp - signal2.timestamp) < 300):  # Within 5 minutes
            
            reward1 = self.config.reward_mapping.get(signal1.feedback_type, 0)
            reward2 = self.config.reward_mapping.get(signal2.feedback_type, 0)
            
            # Opposite signs
            return reward1 * reward2 < 0
        
        return False
    
    def _compute_contradiction_score(self, signal1: FeedbackSignal, signal2: FeedbackSignal) -> float:
        """Compute contradiction score between two signals."""
        # Time proximity factor
        time_diff = abs(signal1.timestamp - signal2.timestamp)
        time_factor = max(0, 1.0 - time_diff / 300.0)  # 5 minutes
        
        # Reward difference factor
        reward1 = self.config.reward_mapping.get(signal1.feedback_type, 0)
        reward2 = self.config.reward_mapping.get(signal2.feedback_type, 0)
        reward_diff = abs(reward1 - reward2)
        
        # Text similarity (simple check)
        text_sim = self._compute_text_similarity(signal1.input_text, signal2.input_text)
        
        return time_factor * reward_diff * text_sim
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity measure."""
        if not text1 or not text2:
            return 0.0
        
        # Simple character overlap
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _should_create_contrastive_pair(self, fb1: ProcessedFeedback, fb2: ProcessedFeedback) -> bool:
        """Check if two feedback items should form a contrastive pair."""
        # Different rewards but similar context
        return (abs(fb1.target_reward - fb2.target_reward) > self.config.contrastive_margin and
                fb1.session_id == fb2.session_id)
    
    def _create_contrastive_pair(self, fb1: ProcessedFeedback, fb2: ProcessedFeedback) -> List[ProcessedFeedback]:
        """Create contrastive pair from two feedback items."""
        # For now, just return the original feedback with adjusted weights
        # In full implementation, this would create specific contrastive loss targets
        
        if fb1.target_reward > fb2.target_reward:
            fb1.weight *= 1.2  # Boost positive example
            fb2.weight *= 0.8  # Reduce negative example
        else:
            fb1.weight *= 0.8
            fb2.weight *= 1.2
        
        return [fb1, fb2]
    
    def _should_create_preference_pair(self, signal1: FeedbackSignal, signal2: FeedbackSignal) -> bool:
        """Check if two signals should form a preference pair."""
        # Different feedback types in same session
        return (signal1.session_id == signal2.session_id and
                signal1.feedback_type != signal2.feedback_type)
    
    def _create_preference_pair(
        self,
        signal1: FeedbackSignal,
        signal2: FeedbackSignal,
        tokenizer: Optional[Any]
    ) -> List[ProcessedFeedback]:
        """Create preference pair from two signals."""
        processed = []
        
        for signal in [signal1, signal2]:
            input_ids, attention_mask = self._tokenize_signal(signal, tokenizer)
            if input_ids is not None:
                processed_fb = ProcessedFeedback(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_reward=self.config.reward_mapping[signal.feedback_type],
                    feedback_type=signal.feedback_type,
                    confidence=signal.confidence,
                    weight=signal.confidence,
                    session_id=signal.session_id,
                    original_feedback=signal
                )
                processed.append(processed_fb)
        
        return processed
    
    def _create_batch_tensors(
        self,
        batch_feedback: List[ProcessedFeedback],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Create batch tensors from processed feedback."""
        
        # Pad sequences to same length
        max_length = max(fb.input_ids.shape[-1] for fb in batch_feedback)
        
        batch_input_ids = []
        batch_attention_masks = []
        batch_rewards = []
        batch_weights = []
        
        for fb in batch_feedback:
            # Pad input_ids
            input_ids = fb.input_ids
            if input_ids.shape[-1] < max_length:
                padding = torch.zeros(max_length - input_ids.shape[-1], dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, padding])
            
            batch_input_ids.append(input_ids)
            
            # Pad attention_mask
            if fb.attention_mask is not None:
                attention_mask = fb.attention_mask
                if attention_mask.shape[-1] < max_length:
                    padding = torch.zeros(max_length - attention_mask.shape[-1], dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, padding])
            else:
                attention_mask = torch.ones_like(input_ids)
            
            batch_attention_masks.append(attention_mask)
            batch_rewards.append(fb.target_reward)
            batch_weights.append(fb.weight)
        
        return {
            'input_ids': torch.stack(batch_input_ids).to(device),
            'attention_mask': torch.stack(batch_attention_masks).to(device),
            'target_reward': torch.tensor(batch_rewards, dtype=torch.float32, device=device),
            'weight': torch.tensor(batch_weights, dtype=torch.float32, device=device)
        }
    
    def _update_statistics(self, processed: List[ProcessedFeedback], start_time: float):
        """Update processing statistics."""
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        self.total_feedback_processed += len(processed)
        
        for fb in processed:
            self.feedback_by_type[fb.feedback_type] += 1
            # Simple quality score based on confidence and weight
            quality_score = fb.confidence * min(1.0, fb.weight)
            self.quality_scores.append(quality_score)


# Utility functions
def create_feedback_processor(
    strategy: ProcessingStrategy = ProcessingStrategy.WEIGHTED,
    batch_size: int = 32,
    enable_session_aggregation: bool = True
) -> FeedbackProcessor:
    """Create feedback processor with specified configuration."""
    config = FeedbackProcessorConfig(
        strategy=strategy,
        batch_size=batch_size,
        enable_session_aggregation=enable_session_aggregation
    )
    return FeedbackProcessor(config)


# Example usage
if __name__ == "__main__":
    from .streaming import create_thumbs_feedback, create_tool_feedback
    
    # Create feedback processor
    processor = create_feedback_processor()
    
    # Create sample feedback signals
    signals = [
        create_thumbs_feedback(True, "session_1", "Hello", "Hi there!", 0.9),
        create_thumbs_feedback(False, "session_1", "Fix this", "I can't help", 0.8),
        create_tool_feedback(True, "session_2", "calculator", 1.0),
        create_tool_feedback(False, "session_2", "web_search", 0.7),
    ]
    
    # Process feedback
    processed = processor.process_feedback_signals(signals)
    print(f"Processed {len(processed)} feedback signals")
    
    # Get training batch
    batch = processor.get_training_batch(batch_size=2)
    if batch:
        print(f"Training batch shapes:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
    
    # Statistics
    stats = processor.get_statistics()
    print(f"Processing stats: {stats['processing']}")
    print(f"Session stats: {stats['sessions']}")