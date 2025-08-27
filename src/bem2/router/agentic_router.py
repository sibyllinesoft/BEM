"""
Agentic Router - Main Orchestration Class

Coordinates macro-policy, composition engine, and chunk-sticky routing:
- Processes input sequences chunk-by-chunk 
- Executes macro-policy decisions with hysteresis
- Applies composed BEM deltas safely
- Tracks routing statistics and cache metrics
- Supports both inference and training modes
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
from dataclasses import dataclass
import time

from .macro_policy import MacroPolicy, MacroAction, MacroPolicyState
from .composition_engine import CompositionEngine, CompositionResult
from .trace_generator import TraceGenerator
from ...bem.modules.chunk_sticky_routing import ChunkStickyRouter

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for Agentic Router."""
    chunk_size: int = 128
    num_experts: int = 3
    hysteresis_tau: float = 0.5
    trust_region_tau: float = 1.0
    max_sequence_length: int = 2048
    enable_routing_stats: bool = True
    enable_latency_tracking: bool = True


class RoutingStep(NamedTuple):
    """Single step in routing sequence."""
    chunk_index: int
    action: MacroAction
    state: MacroPolicyState
    composition_result: CompositionResult
    execution_time: float
    

class RoutingResult(NamedTuple):
    """Complete routing result for a sequence."""
    steps: List[RoutingStep]
    total_time: float
    routing_stats: Dict
    cache_metrics: Dict
    performance_metrics: Dict


class AgenticRouter(nn.Module):
    """
    Main Agentic Router orchestrating dynamic BEM composition.
    
    Key responsibilities:
    - Chunk-level routing decisions via macro-policy
    - Safe composition of BEM deltas with trust-region constraints
    - Cache-safe application maintaining KV window alignment
    - Action hysteresis to prevent routing thrash
    - Comprehensive telemetry for monitoring and evaluation
    """
    
    def __init__(
        self,
        config: RouterConfig,
        macro_policy: MacroPolicy,
        composition_engine: CompositionEngine,
        chunk_router: Optional[ChunkStickyRouter] = None
    ):
        super().__init__()
        self.config = config
        self.macro_policy = macro_policy
        self.composition_engine = composition_engine
        self.chunk_router = chunk_router
        
        # Routing state tracking
        self.routing_history = []
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.total_routing_steps = 0
        
        # Performance tracking
        self.step_times = []
        self.composition_times = []
        self.policy_times = []
        
        # Register buffers for persistent state
        self.register_buffer('flip_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        
        logger.info(f"Initialized AgenticRouter with {config.num_experts} experts")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        base_model: Optional[nn.Module] = None,
        return_routing_info: bool = True,
        training_mode: bool = False
    ) -> Tuple[torch.Tensor, Optional[RoutingResult]]:
        """
        Forward pass with dynamic BEM routing.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            base_model: Base model to apply deltas to
            return_routing_info: Whether to return detailed routing info
            training_mode: Whether in training mode (affects sampling)
            
        Returns:
            Tuple of (output_logits, routing_result)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        start_time = time.time()
        
        # Chunking
        chunk_size = self.config.chunk_size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        routing_steps = []
        prev_action = None
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            step_start_time = time.time()
            
            # Extract chunk
            start_pos = chunk_idx * chunk_size
            end_pos = min(start_pos + chunk_size, seq_len)
            chunk_input_ids = input_ids[:, start_pos:end_pos]
            
            # Generate chunk state
            state = self._generate_chunk_state(
                chunk_input_ids=chunk_input_ids,
                chunk_index=chunk_idx,
                prev_action=prev_action,
                base_model=base_model
            )
            
            # Execute macro-policy
            policy_start = time.time()
            policy_output = self.macro_policy(
                state=state,
                sample=training_mode,
                temperature=1.0 if training_mode else 0.1
            )
            policy_time = time.time() - policy_start
            
            # Extract action
            action_tensor = policy_output['actions'][0]  # First batch element
            action = MacroAction.from_tensor(action_tensor)
            
            # Compose deltas
            composition_start = time.time()
            composition_result = self.composition_engine.compose_action(
                action=action,
                chunk_index=chunk_idx,
                sequence_length=num_chunks
            )
            composition_time = time.time() - composition_start
            
            # Apply deltas to base model (if provided)
            original_weights = None
            if base_model is not None:
                original_weights = self.composition_engine.apply_deltas_to_model(
                    model=base_model,
                    composed_deltas=composition_result.composed_deltas
                )
            
            # Record routing step
            step_time = time.time() - step_start_time
            routing_step = RoutingStep(
                chunk_index=chunk_idx,
                action=action,
                state=state,
                composition_result=composition_result,
                execution_time=step_time
            )
            routing_steps.append(routing_step)
            
            # Update tracking
            self.step_times.append(step_time)
            self.policy_times.append(policy_time)
            self.composition_times.append(composition_time)
            prev_action = action
            
            # Rollback deltas if applied
            if original_weights is not None:
                self.composition_engine.rollback_deltas(base_model, original_weights)
        
        # Generate output (placeholder - would normally run base model)
        # For now, return dummy logits
        vocab_size = 32000  # Typical vocab size
        output_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        
        # Compile routing result
        total_time = time.time() - start_time
        routing_result = None
        
        if return_routing_info:
            routing_stats = self._compute_routing_stats(routing_steps)
            cache_metrics = self._compute_cache_metrics(routing_steps)
            performance_metrics = self._compute_performance_metrics(routing_steps, total_time)
            
            routing_result = RoutingResult(
                steps=routing_steps,
                total_time=total_time,
                routing_stats=routing_stats,
                cache_metrics=cache_metrics,
                performance_metrics=performance_metrics
            )
        
        # Update counters
        self.step_count += len(routing_steps)
        
        return output_logits, routing_result
    
    def _generate_chunk_state(
        self,
        chunk_input_ids: torch.Tensor,
        chunk_index: int,
        prev_action: Optional[MacroAction],
        base_model: Optional[nn.Module]
    ) -> MacroPolicyState:
        """Generate state representation for current chunk."""
        batch_size, chunk_len = chunk_input_ids.shape
        device = chunk_input_ids.device
        
        # Generate chunk summary (placeholder - would use actual encoder)
        chunk_summary_dim = self.macro_policy.chunk_proj.in_features
        chunk_summary = torch.randn(batch_size, chunk_summary_dim, device=device)
        
        # Generate retrieval features (placeholder)
        retrieval_dim = self.macro_policy.retrieval_proj.in_features  
        retrieval_features = torch.randn(batch_size, retrieval_dim, device=device)
        
        # Generate vision features (placeholder)
        vision_dim = self.macro_policy.vision_proj.in_features
        vision_features = torch.zeros(batch_size, vision_dim, device=device)  # No vision by default
        
        # Generate value features (placeholder)
        value_dim = self.macro_policy.value_proj.in_features
        value_features = torch.randn(batch_size, value_dim, device=device)
        
        # Previous action tensor
        prev_action_tensor = None
        if prev_action is not None:
            prev_action_tensor = prev_action.to_tensor(device).unsqueeze(0).expand(batch_size, -1)
        
        # Chunk index
        chunk_index_tensor = torch.full((batch_size,), chunk_index, device=device, dtype=torch.float32)
        
        return MacroPolicyState(
            chunk_summary=chunk_summary,
            retrieval_features=retrieval_features,
            vision_features=vision_features,
            value_features=value_features,
            prev_action=prev_action_tensor,
            chunk_index=chunk_index_tensor
        )
    
    def _compute_routing_stats(self, routing_steps: List[RoutingStep]) -> Dict:
        """Compute routing statistics."""
        if not routing_steps:
            return {}
        
        # Expert usage
        expert_counts = [0] * self.config.num_experts
        for step in routing_steps:
            expert_counts[step.action.expert_id] += 1
        
        expert_utilization = [count / len(routing_steps) for count in expert_counts]
        
        # Action statistics
        scope_counts = {'local': 0, 'global': 0}
        span_counts = {}
        rank_counts = {}
        bias_scales = []
        
        for step in routing_steps:
            action = step.action
            
            scope_counts[action.scope] += 1
            
            span_counts[action.span] = span_counts.get(action.span, 0) + 1
            rank_counts[action.rank_budget] = rank_counts.get(action.rank_budget, 0) + 1
            
            bias_scales.append(action.bias_scale)
        
        # Flip rate (transitions between different experts)
        flip_count = 0
        for i in range(1, len(routing_steps)):
            if routing_steps[i].action.expert_id != routing_steps[i-1].action.expert_id:
                flip_count += 1
        
        flip_rate = flip_count / max(1, len(routing_steps) - 1)
        
        return {
            'num_steps': len(routing_steps),
            'expert_utilization': expert_utilization,
            'flip_rate': flip_rate,
            'flip_count': flip_count,
            'scope_distribution': scope_counts,
            'span_distribution': span_counts,
            'rank_distribution': rank_counts,
            'avg_bias_scale': sum(bias_scales) / len(bias_scales) if bias_scales else 0,
            'bias_scale_std': torch.tensor(bias_scales).std().item() if len(bias_scales) > 1 else 0
        }
    
    def _compute_cache_metrics(self, routing_steps: List[RoutingStep]) -> Dict:
        """Compute KV cache-related metrics."""
        cache_safe_steps = 0
        total_violations = 0
        
        for step in routing_steps:
            if step.composition_result.cache_safety_report['cache_safe']:
                cache_safe_steps += 1
            
            total_violations += len(step.composition_result.cache_safety_report['violations'])
        
        cache_safety_rate = cache_safe_steps / len(routing_steps) if routing_steps else 1.0
        
        return {
            'cache_safety_rate': cache_safety_rate,
            'total_violations': total_violations,
            'cache_safe_steps': cache_safe_steps,
            'chunk_aligned': True,  # Always chunk-aligned by design
            'kv_hit_rate': self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count)
        }
    
    def _compute_performance_metrics(self, routing_steps: List[RoutingStep], total_time: float) -> Dict:
        """Compute performance metrics."""
        if not routing_steps:
            return {'total_time': total_time}
        
        step_times = [step.execution_time for step in routing_steps]
        
        return {
            'total_time': total_time,
            'avg_step_time': sum(step_times) / len(step_times),
            'max_step_time': max(step_times),
            'min_step_time': min(step_times),
            'p95_step_time': sorted(step_times)[int(0.95 * len(step_times))] if len(step_times) > 20 else max(step_times),
            'avg_policy_time': sum(self.policy_times[-len(routing_steps):]) / len(routing_steps),
            'avg_composition_time': sum(self.composition_times[-len(routing_steps):]) / len(routing_steps),
            'steps_per_second': len(routing_steps) / total_time
        }
    
    def generate_routing_trace(
        self,
        input_ids: torch.Tensor,
        target_outputs: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Generate a routing trace for training data.
        
        Args:
            input_ids: Input sequence
            target_outputs: Target outputs for reward computation
            
        Returns:
            Trace dictionary with states, actions, rewards
        """
        # Run forward pass to get routing steps
        _, routing_result = self.forward(
            input_ids=input_ids,
            return_routing_info=True,
            training_mode=True
        )
        
        if routing_result is None:
            return {'states': [], 'actions': [], 'rewards': []}
        
        # Extract states and actions
        states = []
        actions = []
        rewards = []
        
        for step in routing_result.steps:
            # Serialize state (convert tensors to lists)
            state_dict = {
                'chunk_summary': step.state.chunk_summary.cpu().tolist(),
                'retrieval_features': step.state.retrieval_features.cpu().tolist(),
                'vision_features': step.state.vision_features.cpu().tolist(),
                'value_features': step.state.value_features.cpu().tolist(),
                'prev_action': step.state.prev_action.cpu().tolist() if step.state.prev_action is not None else None,
                'chunk_index': step.state.chunk_index.cpu().tolist()
            }
            states.append(state_dict)
            
            # Serialize action
            action_dict = {
                'expert_id': step.action.expert_id,
                'scope': step.action.scope,
                'span': step.action.span,
                'rank_budget': step.action.rank_budget,
                'bias_scale': step.action.bias_scale
            }
            actions.append(action_dict)
            
            # Compute reward (placeholder - would use actual task performance)
            reward = self._compute_step_reward(step, target_outputs)
            rewards.append(reward)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'routing_stats': routing_result.routing_stats,
            'performance_metrics': routing_result.performance_metrics
        }
    
    def _compute_step_reward(
        self,
        step: RoutingStep,
        target_outputs: Optional[torch.Tensor]
    ) -> float:
        """
        Compute reward for a routing step.
        
        Based on composition quality, cache safety, and task performance.
        """
        reward = 0.0
        
        # Base reward for valid composition
        if step.composition_result.trust_region_violations:
            reward -= 0.2 * len(step.composition_result.trust_region_violations)
        else:
            reward += 0.1
        
        # Cache safety bonus
        if step.composition_result.cache_safety_report['cache_safe']:
            reward += 0.1
        else:
            reward -= 0.1
        
        # Efficiency bonus (faster steps get higher reward)
        max_time = 0.1  # 100ms threshold
        if step.execution_time < max_time:
            reward += 0.1 * (1 - step.execution_time / max_time)
        
        # Action hysteresis penalty (discourage too frequent expert switching)
        # This would be computed based on previous actions
        
        return max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]
    
    def get_routing_stats(self) -> Dict:
        """Get overall routing statistics."""
        hysteresis_stats = self.macro_policy.get_hysteresis_stats()
        composition_stats = self.composition_engine.get_composition_stats()
        
        return {
            'total_steps': self.step_count.item(),
            'total_flips': self.flip_count.item(),
            'avg_step_time': sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            'avg_policy_time': sum(self.policy_times) / len(self.policy_times) if self.policy_times else 0,
            'avg_composition_time': sum(self.composition_times) / len(self.composition_times) if self.composition_times else 0,
            **hysteresis_stats,
            **composition_stats
        }
    
    def reset_stats(self):
        """Reset tracking statistics."""
        self.routing_history.clear()
        self.step_times.clear()
        self.policy_times.clear()
        self.composition_times.clear()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.flip_count.zero_()
        self.step_count.zero_()


def create_agentic_router(
    config: Dict,
    macro_policy: MacroPolicy,
    composition_engine: CompositionEngine,
    chunk_router: Optional[ChunkStickyRouter] = None
) -> AgenticRouter:
    """
    Factory function to create AgenticRouter from configuration.
    
    Args:
        config: Configuration dictionary
        macro_policy: Configured macro policy
        composition_engine: Configured composition engine
        chunk_router: Optional chunk router
        
    Returns:
        Configured AgenticRouter instance
    """
    router_config = RouterConfig(
        chunk_size=config.get('chunk_size', 128),
        num_experts=config.get('num_experts', 3),
        hysteresis_tau=config.get('hysteresis_tau', 0.5),
        trust_region_tau=config.get('trust_region_tau', 1.0),
        max_sequence_length=config.get('max_sequence_length', 2048),
        enable_routing_stats=config.get('enable_routing_stats', True),
        enable_latency_tracking=config.get('enable_latency_tracking', True)
    )
    
    return AgenticRouter(
        config=router_config,
        macro_policy=macro_policy,
        composition_engine=composition_engine,
        chunk_router=chunk_router
    )