"""
Task Environment for Policy Gradient Training

Provides task-specific environments for training the Agentic Router
with realistic rewards based on downstream task performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TaskState:
    """Current state in a task environment."""
    input_ids: torch.Tensor
    position: int
    context_length: int
    task_type: str
    metadata: Dict = None


@dataclass
class TaskReward:
    """Reward signal from task completion."""
    task_performance: float  # Primary task metric (0-1)
    efficiency_bonus: float  # Latency/compute efficiency (0-1)  
    cache_safety_bonus: float  # Cache safety compliance (0-1)
    total_reward: float  # Combined reward


class TaskEnvironment(ABC):
    """Abstract base class for task environments."""
    
    @abstractmethod
    def reset(self) -> TaskState:
        """Reset environment and return initial state."""
        pass
    
    @abstractmethod
    def step(self, state: TaskState, action: Dict) -> Tuple[TaskState, TaskReward, bool]:
        """Take a step and return next state, reward, and done flag."""
        pass
    
    @abstractmethod
    def get_task_metrics(self) -> Dict[str, float]:
        """Get task-specific performance metrics."""
        pass


class CodeCompletionEnvironment(TaskEnvironment):
    """Environment for code completion tasks."""
    
    def __init__(
        self,
        max_sequence_length: int = 2048,
        chunk_size: int = 128,
        vocab_size: int = 32000
    ):
        self.max_sequence_length = max_sequence_length
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        
        # Synthetic code patterns for training
        self.code_patterns = [
            "function definition with parameters",
            "class definition with methods", 
            "conditional logic with multiple branches",
            "loop constructs with iteration",
            "data structure manipulation",
            "error handling and exceptions",
            "async/await patterns",
            "type annotations and generics"
        ]
        
        self.current_episode = 0
        self.episode_metrics = {}
        
    def reset(self) -> TaskState:
        """Reset to a new code completion task."""
        self.current_episode += 1
        
        # Generate synthetic code completion task
        pattern = random.choice(self.code_patterns)
        sequence_length = random.randint(256, self.max_sequence_length)
        
        # Create synthetic input representing code context
        input_ids = torch.randint(
            0, self.vocab_size, 
            (1, sequence_length), 
            dtype=torch.long
        )
        
        # Add some structure to simulate real code tokens
        # Higher token values represent keywords, lower values represent identifiers
        keyword_positions = torch.rand(sequence_length) < 0.15
        input_ids[0, keyword_positions] = torch.randint(self.vocab_size - 1000, self.vocab_size, (keyword_positions.sum(),))
        
        return TaskState(
            input_ids=input_ids,
            position=0,
            context_length=sequence_length,
            task_type="code_completion",
            metadata={"pattern": pattern}
        )
    
    def step(
        self, 
        state: TaskState, 
        routing_result: Dict
    ) -> Tuple[TaskState, TaskReward, bool]:
        """Execute one routing step and compute rewards."""
        
        # Update position
        new_position = min(
            state.position + self.chunk_size,
            state.context_length
        )
        
        # Task performance simulation based on routing quality
        task_performance = self._evaluate_task_performance(state, routing_result)
        
        # Efficiency bonus based on routing decisions
        efficiency_bonus = self._evaluate_efficiency(routing_result)
        
        # Cache safety bonus
        cache_safety_bonus = self._evaluate_cache_safety(routing_result)
        
        # Combined reward
        total_reward = (
            0.6 * task_performance +
            0.2 * efficiency_bonus + 
            0.2 * cache_safety_bonus
        )
        
        reward = TaskReward(
            task_performance=task_performance,
            efficiency_bonus=efficiency_bonus,
            cache_safety_bonus=cache_safety_bonus,
            total_reward=total_reward
        )
        
        # Create next state
        next_state = TaskState(
            input_ids=state.input_ids,
            position=new_position,
            context_length=state.context_length,
            task_type=state.task_type,
            metadata=state.metadata
        )
        
        # Episode done when we've processed the full sequence
        done = new_position >= state.context_length
        
        return next_state, reward, done
    
    def _evaluate_task_performance(self, state: TaskState, routing_result: Dict) -> float:
        """Evaluate task-specific performance."""
        if not routing_result or 'steps' not in routing_result:
            return 0.3  # Poor performance for failed routing
            
        steps = routing_result['steps']
        if not steps:
            return 0.3
            
        # Simulate performance based on expert selection appropriateness
        pattern = state.metadata.get('pattern', '')
        
        performance_scores = []
        for step in steps:
            if hasattr(step, 'action'):
                expert_id = step.action.expert_id
                
                # Reward appropriate expert selection based on code pattern
                if 'function' in pattern and expert_id == 0:  # Code expert
                    performance_scores.append(0.9)
                elif 'class' in pattern and expert_id == 0:  # Code expert  
                    performance_scores.append(0.85)
                elif 'type' in pattern and expert_id == 1:  # Formal expert
                    performance_scores.append(0.8)
                elif 'error' in pattern and expert_id == 2:  # Safety expert
                    performance_scores.append(0.9)
                else:
                    performance_scores.append(0.5)  # Suboptimal but not terrible
        
        return float(np.mean(performance_scores)) if performance_scores else 0.3
    
    def _evaluate_efficiency(self, routing_result: Dict) -> float:
        """Evaluate routing efficiency."""
        if not routing_result:
            return 0.2
            
        # Penalize excessive expert switching (routing thrash)
        steps = routing_result.get('steps', [])
        if len(steps) < 2:
            return 0.8
            
        expert_changes = 0
        prev_expert = None
        
        for step in steps:
            if hasattr(step, 'action'):
                current_expert = step.action.expert_id
                if prev_expert is not None and current_expert != prev_expert:
                    expert_changes += 1
                prev_expert = current_expert
        
        # Lower expert switching rate = higher efficiency
        switch_rate = expert_changes / max(len(steps) - 1, 1)
        efficiency = max(0.0, 1.0 - 2.0 * switch_rate)  # Penalize switching
        
        return float(efficiency)
    
    def _evaluate_cache_safety(self, routing_result: Dict) -> float:
        """Evaluate cache safety compliance."""
        if not routing_result:
            return 0.1
            
        cache_metrics = routing_result.get('cache_metrics', {})
        
        # Check for cache safety violations
        cache_safety_rate = cache_metrics.get('cache_safety_rate', 0.5)
        
        # Check chunk alignment
        alignment_score = cache_metrics.get('chunk_alignment_rate', 0.5)
        
        # Combined cache safety score
        safety_score = 0.7 * cache_safety_rate + 0.3 * alignment_score
        
        return float(safety_score)
    
    def get_task_metrics(self) -> Dict[str, float]:
        """Get episode metrics."""
        return self.episode_metrics


class FormalReasoningEnvironment(TaskEnvironment):
    """Environment for formal reasoning tasks."""
    
    def __init__(
        self,
        max_sequence_length: int = 2048,
        chunk_size: int = 128,
        vocab_size: int = 32000
    ):
        self.max_sequence_length = max_sequence_length
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        
        self.reasoning_patterns = [
            "mathematical proof steps",
            "logical inference chains", 
            "theorem application",
            "symbolic manipulation",
            "constraint satisfaction",
            "formal verification steps"
        ]
        
        self.current_episode = 0
        self.episode_metrics = {}
        
    def reset(self) -> TaskState:
        """Reset to a new formal reasoning task."""
        pattern = random.choice(self.reasoning_patterns)
        sequence_length = random.randint(512, self.max_sequence_length)
        
        # Create input with formal structure
        input_ids = torch.randint(0, self.vocab_size, (1, sequence_length), dtype=torch.long)
        
        # Add formal tokens (higher values)
        formal_positions = torch.rand(sequence_length) < 0.3
        input_ids[0, formal_positions] = torch.randint(
            self.vocab_size - 2000, self.vocab_size, (formal_positions.sum(),)
        )
        
        return TaskState(
            input_ids=input_ids,
            position=0,
            context_length=sequence_length,
            task_type="formal_reasoning",
            metadata={"pattern": pattern}
        )
    
    def step(self, state: TaskState, routing_result: Dict) -> Tuple[TaskState, TaskReward, bool]:
        """Execute reasoning step."""
        new_position = min(state.position + self.chunk_size, state.context_length)
        
        # Formal reasoning rewards precise expert usage
        task_performance = self._evaluate_reasoning_quality(state, routing_result)
        efficiency_bonus = self._evaluate_efficiency(routing_result) 
        cache_safety_bonus = self._evaluate_cache_safety(routing_result)
        
        total_reward = (
            0.7 * task_performance +  # Higher weight on correctness
            0.15 * efficiency_bonus +
            0.15 * cache_safety_bonus
        )
        
        reward = TaskReward(
            task_performance=task_performance,
            efficiency_bonus=efficiency_bonus,
            cache_safety_bonus=cache_safety_bonus,
            total_reward=total_reward
        )
        
        next_state = TaskState(
            input_ids=state.input_ids,
            position=new_position,
            context_length=state.context_length,
            task_type=state.task_type,
            metadata=state.metadata
        )
        
        done = new_position >= state.context_length
        return next_state, reward, done
    
    def _evaluate_reasoning_quality(self, state: TaskState, routing_result: Dict) -> float:
        """Evaluate formal reasoning quality."""
        if not routing_result or 'steps' not in routing_result:
            return 0.2
            
        steps = routing_result['steps']
        pattern = state.metadata.get('pattern', '')
        
        quality_scores = []
        for step in steps:
            if hasattr(step, 'action'):
                expert_id = step.action.expert_id
                
                # Formal expert (id=1) should be heavily favored
                if expert_id == 1:  # Formal expert
                    quality_scores.append(0.95)
                elif expert_id == 2 and 'verification' in pattern:  # Safety for verification
                    quality_scores.append(0.8)  
                else:
                    quality_scores.append(0.4)  # Significant penalty
        
        return float(np.mean(quality_scores)) if quality_scores else 0.2
    
    def _evaluate_efficiency(self, routing_result: Dict) -> float:
        """Same efficiency evaluation as code completion."""
        # Reuse implementation
        return CodeCompletionEnvironment._evaluate_efficiency(self, routing_result)
    
    def _evaluate_cache_safety(self, routing_result: Dict) -> float:
        """Same cache safety evaluation."""
        # Reuse implementation  
        return CodeCompletionEnvironment._evaluate_cache_safety(self, routing_result)
    
    def get_task_metrics(self) -> Dict[str, float]:
        return self.episode_metrics


class MultiTaskEnvironment(TaskEnvironment):
    """Environment that combines multiple task types."""
    
    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        max_sequence_length: int = 2048,
        chunk_size: int = 128,
        vocab_size: int = 32000
    ):
        if task_weights is None:
            task_weights = {
                'code_completion': 0.5,
                'formal_reasoning': 0.3,
                'safety_analysis': 0.2
            }
        
        self.task_weights = task_weights
        self.environments = {
            'code_completion': CodeCompletionEnvironment(max_sequence_length, chunk_size, vocab_size),
            'formal_reasoning': FormalReasoningEnvironment(max_sequence_length, chunk_size, vocab_size)
        }
        
        self.current_env = None
        self.current_task_type = None
        
    def reset(self) -> TaskState:
        """Reset to a randomly selected task type."""
        task_type = np.random.choice(
            list(self.task_weights.keys()),
            p=list(self.task_weights.values())
        )
        
        self.current_task_type = task_type
        if task_type in self.environments:
            self.current_env = self.environments[task_type]
            return self.current_env.reset()
        else:
            # Fallback to code completion
            self.current_env = self.environments['code_completion']
            return self.current_env.reset()
    
    def step(self, state: TaskState, routing_result: Dict) -> Tuple[TaskState, TaskReward, bool]:
        """Delegate to current environment."""
        if self.current_env is None:
            raise ValueError("Environment not initialized - call reset() first")
        
        return self.current_env.step(state, routing_result)
    
    def get_task_metrics(self) -> Dict[str, float]:
        """Aggregate metrics from all environments."""
        all_metrics = {}
        for task_type, env in self.environments.items():
            task_metrics = env.get_task_metrics()
            for key, value in task_metrics.items():
                all_metrics[f"{task_type}_{key}"] = value
        
        return all_metrics


def create_task_environment(config: Dict) -> TaskEnvironment:
    """Factory function to create task environment."""
    env_type = config.get('type', 'multi_task')
    
    common_params = {
        'max_sequence_length': config.get('max_sequence_length', 2048),
        'chunk_size': config.get('chunk_size', 128),
        'vocab_size': config.get('vocab_size', 32000)
    }
    
    if env_type == 'code_completion':
        return CodeCompletionEnvironment(**common_params)
    elif env_type == 'formal_reasoning':
        return FormalReasoningEnvironment(**common_params)
    elif env_type == 'multi_task':
        task_weights = config.get('task_weights', {
            'code_completion': 0.5,
            'formal_reasoning': 0.3, 
            'safety_analysis': 0.2
        })
        return MultiTaskEnvironment(
            task_weights=task_weights,
            **common_params
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")