"""
Stream Processor for BEM 2.0 Online Learning.

Processes live feedback streams (thumbs up/down, tool success/failure) 
for online learning as specified in TODO.md.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple, Iterator
from dataclasses import dataclass, field
import torch
import numpy as np
import logging
import time
import json
import asyncio
from enum import Enum
from pathlib import Path
from collections import deque


class FeedbackType(Enum):
    """Types of feedback signals."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    TOOL_SUCCESS = "tool_success" 
    TOOL_FAILURE = "tool_failure"
    PERFORMANCE_SCORE = "performance_score"
    USER_CORRECTION = "user_correction"
    IMPLICIT_POSITIVE = "implicit_positive"
    IMPLICIT_NEGATIVE = "implicit_negative"


@dataclass
class FeedbackSignal:
    """Individual feedback signal from the stream."""
    
    # Core data
    feedback_type: FeedbackType
    value: float  # Normalized score [0, 1] where 1 is best
    confidence: float = 1.0  # Confidence in the feedback [0, 1]
    
    # Context
    session_id: str = ""
    user_id: str = ""
    task_type: str = ""
    
    # Input/output context
    input_text: str = ""
    output_text: str = ""
    expected_output: str = ""
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"  # Source of the feedback
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feedback_type': self.feedback_type.value,
            'value': self.value,
            'confidence': self.confidence,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'task_type': self.task_type,
            'input_text': self.input_text,
            'output_text': self.output_text,
            'expected_output': self.expected_output,
            'timestamp': self.timestamp,
            'source': self.source,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackSignal':
        """Create from dictionary."""
        return cls(
            feedback_type=FeedbackType(data['feedback_type']),
            value=data['value'],
            confidence=data.get('confidence', 1.0),
            session_id=data.get('session_id', ''),
            user_id=data.get('user_id', ''),
            task_type=data.get('task_type', ''),
            input_text=data.get('input_text', ''),
            output_text=data.get('output_text', ''),
            expected_output=data.get('expected_output', ''),
            timestamp=data.get('timestamp', time.time()),
            source=data.get('source', 'unknown'),
            metadata=data.get('metadata', {})
        )


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    
    # Stream sources
    stream_sources: List[str] = field(default_factory=list)
    
    # Processing
    batch_size: int = 32
    max_queue_size: int = 10000
    processing_interval: float = 1.0  # seconds
    
    # Filtering
    min_confidence: float = 0.5
    filter_duplicate_sessions: bool = True
    max_session_age: float = 3600.0  # 1 hour
    
    # Feedback aggregation
    enable_session_aggregation: bool = True
    session_timeout: float = 300.0  # 5 minutes
    
    # Quality control
    enable_spam_detection: bool = True
    max_feedback_per_user_per_hour: int = 100
    
    # Persistence
    save_stream_data: bool = True
    stream_data_path: Optional[str] = "stream_data"
    save_frequency: int = 1000  # Save every N signals
    
    # Real-time vs batch processing
    real_time_processing: bool = True
    batch_processing_interval: float = 60.0  # seconds


class StreamProcessor:
    """
    Processes live feedback streams for online learning.
    
    Handles:
    - Multiple feedback sources (thumbs, tool results, scores)
    - Real-time and batch processing
    - Quality filtering and spam detection
    - Session aggregation and context tracking
    - Persistence and recovery
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Stream state
        self.is_running = False
        self.signal_queue: deque = deque(maxlen=config.max_queue_size)
        self.processed_signals: deque = deque(maxlen=10000)
        
        # Session tracking
        self.active_sessions: Dict[str, List[FeedbackSignal]] = {}
        self.session_last_update: Dict[str, float] = {}
        
        # Quality control
        self.user_feedback_counts: Dict[str, Dict[str, int]] = {}  # user_id -> {hour -> count}
        self.seen_signals: set = set()  # For duplicate detection
        
        # Statistics
        self.total_signals_received = 0
        self.total_signals_processed = 0
        self.signals_filtered = 0
        self.active_sessions_count = 0
        
        # Persistence
        if config.save_stream_data and config.stream_data_path:
            self.stream_data_dir = Path(config.stream_data_path)
            self.stream_data_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.stream_data_dir = None
        
        self.last_save_count = 0
        
        self.logger.info("StreamProcessor initialized")
    
    def add_feedback_signal(self, signal: FeedbackSignal) -> bool:
        """Add feedback signal to processing queue."""
        self.total_signals_received += 1
        
        # Quality filtering
        if not self._passes_quality_filter(signal):
            self.signals_filtered += 1
            return False
        
        # Add to queue
        self.signal_queue.append(signal)
        
        # Real-time processing if enabled
        if self.config.real_time_processing and self.is_running:
            asyncio.create_task(self._process_signal_batch([signal]))
        
        return True
    
    def add_feedback_signals(self, signals: List[FeedbackSignal]) -> int:
        """Add multiple feedback signals. Returns count of accepted signals."""
        accepted = 0
        for signal in signals:
            if self.add_feedback_signal(signal):
                accepted += 1
        return accepted
    
    def start_processing(self):
        """Start stream processing."""
        if self.is_running:
            self.logger.warning("Stream processor already running")
            return
        
        self.is_running = True
        self.logger.info("Stream processing started")
        
        # Start background processing tasks
        if self.config.real_time_processing:
            asyncio.create_task(self._real_time_processing_loop())
        else:
            asyncio.create_task(self._batch_processing_loop())
        
        # Start maintenance tasks
        asyncio.create_task(self._maintenance_loop())
    
    def stop_processing(self):
        """Stop stream processing."""
        self.is_running = False
        self.logger.info("Stream processing stopped")
        
        # Save remaining data
        if self.stream_data_dir:
            self._save_stream_data()
    
    def get_processed_batch(self, batch_size: Optional[int] = None) -> List[FeedbackSignal]:
        """Get batch of processed feedback signals."""
        batch_size = batch_size or self.config.batch_size
        
        if len(self.processed_signals) < batch_size:
            return list(self.processed_signals)
        
        # Extract batch
        batch = []
        for _ in range(batch_size):
            if self.processed_signals:
                batch.append(self.processed_signals.popleft())
            else:
                break
        
        return batch
    
    def get_session_feedback(self, session_id: str) -> List[FeedbackSignal]:
        """Get all feedback for a specific session."""
        return self.active_sessions.get(session_id, [])
    
    def get_aggregated_session_score(self, session_id: str) -> Optional[float]:
        """Get aggregated feedback score for a session."""
        session_signals = self.active_sessions.get(session_id, [])
        if not session_signals:
            return None
        
        # Weight by confidence and recency
        weighted_scores = []
        current_time = time.time()
        
        for signal in session_signals:
            age_weight = max(0.1, 1.0 - (current_time - signal.timestamp) / self.config.session_timeout)
            weighted_score = signal.value * signal.confidence * age_weight
            weighted_scores.append(weighted_score)
        
        return np.mean(weighted_scores) if weighted_scores else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get stream processing statistics."""
        # Active sessions
        session_stats = {}
        if self.active_sessions:
            session_lengths = [len(signals) for signals in self.active_sessions.values()]
            session_stats = {
                'count': len(self.active_sessions),
                'avg_length': np.mean(session_lengths),
                'max_length': np.max(session_lengths),
                'total_signals': sum(session_lengths)
            }
        
        # Feedback type distribution
        type_counts = {}
        for signal in list(self.processed_signals)[-1000:]:  # Last 1000 signals
            signal_type = signal.feedback_type.value
            type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
        
        # Processing rates
        processing_rate = self.total_signals_processed / max(1, time.time() - self._start_time) if hasattr(self, '_start_time') else 0
        filter_rate = self.signals_filtered / max(1, self.total_signals_received)
        
        return {
            'processing': {
                'is_running': self.is_running,
                'total_received': self.total_signals_received,
                'total_processed': self.total_signals_processed,
                'signals_filtered': self.signals_filtered,
                'filter_rate': filter_rate,
                'processing_rate': processing_rate,
                'queue_size': len(self.signal_queue),
                'processed_queue_size': len(self.processed_signals)
            },
            'sessions': session_stats,
            'feedback_types': type_counts,
            'quality_control': {
                'active_users': len(self.user_feedback_counts),
                'unique_signals': len(self.seen_signals)
            }
        }
    
    async def _real_time_processing_loop(self):
        """Real-time processing loop."""
        self._start_time = time.time()
        
        while self.is_running:
            if self.signal_queue:
                # Process small batches frequently
                batch_size = min(self.config.batch_size, len(self.signal_queue))
                batch = []
                for _ in range(batch_size):
                    if self.signal_queue:
                        batch.append(self.signal_queue.popleft())
                
                if batch:
                    await self._process_signal_batch(batch)
            
            await asyncio.sleep(self.config.processing_interval)
    
    async def _batch_processing_loop(self):
        """Batch processing loop."""
        self._start_time = time.time()
        
        while self.is_running:
            await asyncio.sleep(self.config.batch_processing_interval)
            
            if self.signal_queue:
                # Process all queued signals
                batch = []
                while self.signal_queue and len(batch) < self.config.batch_size * 5:  # Larger batches
                    batch.append(self.signal_queue.popleft())
                
                if batch:
                    await self._process_signal_batch(batch)
    
    async def _process_signal_batch(self, batch: List[FeedbackSignal]):
        """Process a batch of feedback signals."""
        
        for signal in batch:
            # Update session tracking
            self._update_session_tracking(signal)
            
            # Add to processed signals
            self.processed_signals.append(signal)
            self.total_signals_processed += 1
        
        # Save periodically
        if (self.stream_data_dir and 
            self.total_signals_processed - self.last_save_count >= self.config.save_frequency):
            self._save_stream_data()
    
    async def _maintenance_loop(self):
        """Maintenance loop for cleanup and housekeeping."""
        while self.is_running:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Clean up old sessions
            self._cleanup_old_sessions()
            
            # Clean up user feedback counts
            self._cleanup_user_counts()
            
            # Clean up seen signals (memory management)
            if len(self.seen_signals) > 50000:
                # Keep only recent signals
                recent_cutoff = time.time() - 3600  # 1 hour
                self.seen_signals = {
                    sig_id for sig_id in self.seen_signals 
                    if float(sig_id.split('_')[-1]) > recent_cutoff
                }
    
    def _passes_quality_filter(self, signal: FeedbackSignal) -> bool:
        """Check if signal passes quality filters."""
        
        # Confidence threshold
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Duplicate detection (based on session, type, and timestamp)
        if self.config.filter_duplicate_sessions:
            signal_id = f"{signal.session_id}_{signal.feedback_type.value}_{signal.timestamp}"
            if signal_id in self.seen_signals:
                return False
            self.seen_signals.add(signal_id)
        
        # Spam detection
        if self.config.enable_spam_detection:
            if not self._passes_spam_filter(signal):
                return False
        
        return True
    
    def _passes_spam_filter(self, signal: FeedbackSignal) -> bool:
        """Check if signal passes spam detection."""
        if not signal.user_id:
            return True  # Can't filter without user ID
        
        current_hour = int(time.time() // 3600)
        
        # Initialize user tracking
        if signal.user_id not in self.user_feedback_counts:
            self.user_feedback_counts[signal.user_id] = {}
        
        # Count feedback in current hour
        hour_count = self.user_feedback_counts[signal.user_id].get(current_hour, 0)
        
        if hour_count >= self.config.max_feedback_per_user_per_hour:
            return False
        
        # Update count
        self.user_feedback_counts[signal.user_id][current_hour] = hour_count + 1
        return True
    
    def _update_session_tracking(self, signal: FeedbackSignal):
        """Update session tracking with new signal."""
        if not signal.session_id:
            return
        
        # Initialize session
        if signal.session_id not in self.active_sessions:
            self.active_sessions[signal.session_id] = []
        
        # Add signal to session
        self.active_sessions[signal.session_id].append(signal)
        self.session_last_update[signal.session_id] = time.time()
    
    def _cleanup_old_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, last_update in self.session_last_update.items():
            if current_time - last_update > self.config.max_session_age:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            del self.session_last_update[session_id]
        
        if expired_sessions:
            self.logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _cleanup_user_counts(self):
        """Clean up old user feedback counts."""
        current_hour = int(time.time() // 3600)
        cutoff_hour = current_hour - 24  # Keep last 24 hours
        
        for user_id in list(self.user_feedback_counts.keys()):
            user_counts = self.user_feedback_counts[user_id]
            # Remove old hour buckets
            old_hours = [hour for hour in user_counts.keys() if hour < cutoff_hour]
            for hour in old_hours:
                del user_counts[hour]
            
            # Remove user if no recent activity
            if not user_counts:
                del self.user_feedback_counts[user_id]
    
    def _save_stream_data(self):
        """Save processed stream data to disk."""
        if not self.stream_data_dir:
            return
        
        try:
            # Save recent processed signals
            signals_data = [signal.to_dict() for signal in list(self.processed_signals)[-1000:]]
            signals_file = self.stream_data_dir / f"signals_{int(time.time())}.jsonl"
            
            with open(signals_file, 'w') as f:
                for signal_data in signals_data:
                    f.write(json.dumps(signal_data) + '\n')
            
            # Save statistics
            stats_file = self.stream_data_dir / "stream_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.get_statistics(), f, indent=2)
            
            self.last_save_count = self.total_signals_processed
            self.logger.debug(f"Stream data saved: {len(signals_data)} signals")
            
        except Exception as e:
            self.logger.error(f"Failed to save stream data: {e}")


def create_stream_processor(
    batch_size: int = 32,
    real_time: bool = True,
    save_data: bool = True,
    stream_data_path: str = "stream_data"
) -> StreamProcessor:
    """Create stream processor with specified configuration."""
    config = StreamConfig(
        batch_size=batch_size,
        real_time_processing=real_time,
        save_stream_data=save_data,
        stream_data_path=stream_data_path
    )
    return StreamProcessor(config)


def create_thumbs_feedback(
    thumbs_up: bool,
    session_id: str = "",
    input_text: str = "",
    output_text: str = "",
    confidence: float = 1.0
) -> FeedbackSignal:
    """Create thumbs up/down feedback signal."""
    return FeedbackSignal(
        feedback_type=FeedbackType.THUMBS_UP if thumbs_up else FeedbackType.THUMBS_DOWN,
        value=1.0 if thumbs_up else 0.0,
        confidence=confidence,
        session_id=session_id,
        input_text=input_text,
        output_text=output_text,
        source="user_thumbs"
    )


def create_tool_feedback(
    success: bool,
    session_id: str = "",
    tool_name: str = "",
    confidence: float = 1.0
) -> FeedbackSignal:
    """Create tool success/failure feedback signal."""
    return FeedbackSignal(
        feedback_type=FeedbackType.TOOL_SUCCESS if success else FeedbackType.TOOL_FAILURE,
        value=1.0 if success else 0.0,
        confidence=confidence,
        session_id=session_id,
        source="tool_execution",
        metadata={'tool_name': tool_name}
    )


# Example usage
if __name__ == "__main__":
    # Create stream processor
    processor = create_stream_processor()
    
    # Create sample feedback signals
    signals = [
        create_thumbs_feedback(True, "session_1", "Hello", "Hi there!", 0.9),
        create_thumbs_feedback(False, "session_1", "Fix this", "I can't help with that", 0.8),
        create_tool_feedback(True, "session_2", "calculator", 1.0),
        create_tool_feedback(False, "session_2", "web_search", 0.7),
    ]
    
    # Add signals to processor
    accepted = processor.add_feedback_signals(signals)
    print(f"Accepted {accepted}/{len(signals)} signals")
    
    # Get processed batch
    batch = processor.get_processed_batch()
    print(f"Processed batch size: {len(batch)}")
    
    # Session aggregation
    session_score = processor.get_aggregated_session_score("session_1")
    print(f"Session 1 score: {session_score}")
    
    # Statistics
    stats = processor.get_statistics()
    print(f"Processing stats: {stats['processing']}")
    print(f"Feedback types: {stats['feedback_types']}")