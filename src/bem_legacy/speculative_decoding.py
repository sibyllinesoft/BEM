"""
Speculative Decoding for BEM - Phase 5 Implementation.

Implements performance-neutral BEM application with base model drafting
and BEM verification as specified in TODO.md Phase 5.

Key Features:
- Base Drafting: Fast generation with base model
- BEM Verification: Check drafts against BEM predictions
- KL Threshold: Only apply BEM when divergence > ε
- Accept/Reject: Keep good drafts, apply BEM to poor ones
- Latency Neutral: Target zero or negative latency impact
"""

from typing import Dict, List, Optional, Tuple, Union, NamedTuple, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from collections import deque, defaultdict

from .telemetry import TelemetryCollector


class SpeculativeDecodingMetrics(NamedTuple):
    """Metrics for speculative decoding performance."""
    acceptance_rate: float
    average_draft_length: float
    base_tokens_per_second: float
    bem_tokens_per_second: float
    total_tokens_per_second: float
    kl_threshold_violations: int
    verification_overhead: float
    net_speedup: float


@dataclass
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding."""
    
    # Drafting parameters
    draft_length: int = 4  # Number of tokens to draft ahead
    max_draft_length: int = 8
    min_draft_length: int = 2
    
    # Acceptance thresholds
    kl_threshold: float = 0.1  # When to apply BEM verification
    acceptance_threshold: float = 0.8  # Accept if prob ratio > threshold
    temperature_scaling: float = 1.0
    
    # Adaptive parameters
    enable_adaptive_drafting: bool = True
    target_acceptance_rate: float = 0.7
    adaptation_window: int = 100
    draft_length_adjustment: float = 0.1
    
    # Performance optimization
    batch_verification: bool = True
    early_exit_threshold: float = 0.95  # Stop verification if confidence high
    parallel_speculation: bool = False
    
    # Safety parameters
    max_kl_divergence: float = 1.0  # Safety limit for KL divergence
    fallback_to_base: bool = True
    verification_timeout: float = 0.1  # Max time for verification (seconds)


class DraftResult(NamedTuple):
    """Result of drafting operation."""
    tokens: torch.Tensor
    logits: torch.Tensor
    draft_time: float
    num_tokens: int


class VerificationResult(NamedTuple):
    """Result of BEM verification."""
    accepted_tokens: torch.Tensor
    accepted_length: int
    kl_divergences: torch.Tensor
    verification_time: float
    bem_applied: bool


class SpeculativeDecoder:
    """Main speculative decoding system."""
    
    def __init__(
        self,
        base_model: nn.Module,
        bem_model: nn.Module,
        tokenizer,
        config: SpeculativeDecodingConfig,
        telemetry_collector: Optional[TelemetryCollector] = None
    ):
        self.base_model = base_model
        self.bem_model = bem_model
        self.tokenizer = tokenizer
        self.config = config
        self.telemetry = telemetry_collector
        
        # Performance tracking
        self.metrics_history = deque(maxlen=config.adaptation_window)
        self.acceptance_history = deque(maxlen=config.adaptation_window)
        
        # Adaptive parameters
        self.current_draft_length = config.draft_length
        self.current_kl_threshold = config.kl_threshold
        
        # Statistics
        self.total_drafted_tokens = 0
        self.total_accepted_tokens = 0
        self.total_base_time = 0.0
        self.total_bem_time = 0.0
        self.total_verification_time = 0.0
        
        self.logger = logging.getLogger(__name__)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **generation_kwargs
    ) -> Dict[str, any]:
        """
        Generate text using speculative decoding.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing generated tokens and metrics
        """
        generated_tokens = []
        current_input = input_ids.clone()
        
        generation_start_time = time.time()
        total_drafts = 0
        total_verifications = 0
        
        while len(generated_tokens) < max_new_tokens:
            # Draft tokens with base model
            draft_result = self._draft_tokens(
                current_input,
                draft_length=self.current_draft_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )
            total_drafts += 1
            
            # Verify draft with BEM
            verification_result = self._verify_draft(
                current_input,
                draft_result,
                temperature=temperature
            )
            total_verifications += 1
            
            # Accept tokens
            accepted_tokens = verification_result.accepted_tokens
            if len(accepted_tokens) > 0:
                generated_tokens.extend(accepted_tokens.tolist())
                current_input = torch.cat([
                    current_input,
                    accepted_tokens.unsqueeze(0) if accepted_tokens.dim() == 1 else accepted_tokens
                ], dim=1)
            else:
                # Fallback: generate one token with BEM
                fallback_token = self._fallback_generation(
                    current_input,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p
                )
                generated_tokens.append(fallback_token.item())
                current_input = torch.cat([current_input, fallback_token.unsqueeze(0)], dim=1)
            
            # Update acceptance tracking
            acceptance_rate = verification_result.accepted_length / draft_result.num_tokens
            self.acceptance_history.append(acceptance_rate)
            
            # Adaptive adjustment
            if self.config.enable_adaptive_drafting and len(self.acceptance_history) >= 10:
                self._adjust_draft_parameters()
            
            # Check for early termination
            if len(generated_tokens) >= max_new_tokens:
                break
        
        generation_time = time.time() - generation_start_time
        
        # Compute final metrics
        final_metrics = self._compute_generation_metrics(
            num_generated=len(generated_tokens),
            generation_time=generation_time,
            num_drafts=total_drafts,
            num_verifications=total_verifications
        )
        
        if self.telemetry:
            self.telemetry.log_speculative_decoding_metrics(final_metrics)
        
        return {
            'generated_tokens': torch.tensor(generated_tokens),
            'generated_text': self.tokenizer.decode(generated_tokens),
            'metrics': final_metrics,
            'generation_time': generation_time
        }
    
    def _draft_tokens(
        self,
        input_ids: torch.Tensor,
        draft_length: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> DraftResult:
        """Draft tokens using base model."""
        draft_start_time = time.time()
        
        with torch.no_grad():
            current_input = input_ids.clone()
            drafted_tokens = []
            all_logits = []
            
            for _ in range(draft_length):
                # Generate next token with base model
                outputs = self.base_model(current_input)
                logits = outputs.logits[:, -1, :]  # Last token logits
                
                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Sample next token
                if do_sample:
                    if top_p < 1.0:
                        logits = self._apply_top_p_filtering(logits, top_p)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                drafted_tokens.append(next_token)
                all_logits.append(logits)
                
                # Append to input for next iteration
                current_input = torch.cat([current_input, next_token], dim=1)
            
            draft_time = time.time() - draft_start_time
            self.total_base_time += draft_time
            self.total_drafted_tokens += len(drafted_tokens)
            
            return DraftResult(
                tokens=torch.cat(drafted_tokens, dim=1) if drafted_tokens else torch.empty(0, 0),
                logits=torch.stack(all_logits) if all_logits else torch.empty(0, 0, 0),
                draft_time=draft_time,
                num_tokens=len(drafted_tokens)
            )
    
    def _verify_draft(
        self,
        input_ids: torch.Tensor,
        draft_result: DraftResult,
        temperature: float = 1.0
    ) -> VerificationResult:
        """Verify draft tokens using BEM model."""
        if draft_result.num_tokens == 0:
            return VerificationResult(
                accepted_tokens=torch.empty(0, dtype=torch.long),
                accepted_length=0,
                kl_divergences=torch.empty(0),
                verification_time=0.0,
                bem_applied=False
            )
        
        verification_start_time = time.time()
        
        with torch.no_grad():
            # Get BEM predictions for the same context
            full_input = torch.cat([input_ids, draft_result.tokens], dim=1)
            bem_outputs = self.bem_model(full_input)
            
            # Extract logits for comparison
            bem_logits = bem_outputs.logits[:, input_ids.shape[1]-1:-1, :]  # BEM predictions for draft positions
            base_logits = draft_result.logits
            
            if temperature != 1.0:
                bem_logits = bem_logits / temperature
                base_logits = base_logits / temperature
            
            # Compute KL divergences between base and BEM predictions
            kl_divergences = self._compute_kl_divergences(base_logits, bem_logits)
            
            # Determine which tokens to accept
            accepted_tokens = []
            bem_applied = False
            
            for i, kl_div in enumerate(kl_divergences):
                if kl_div > self.current_kl_threshold:
                    # Use BEM prediction
                    bem_probs = F.softmax(bem_logits[i], dim=-1)
                    base_probs = F.softmax(base_logits[i], dim=-1)
                    
                    # Compute acceptance probability
                    drafted_token = draft_result.tokens[0, i]
                    acceptance_prob = (bem_probs[0, drafted_token] / 
                                     (base_probs[0, drafted_token] + 1e-8))
                    
                    if acceptance_prob >= self.config.acceptance_threshold:
                        # Accept base model's token
                        accepted_tokens.append(drafted_token)
                    else:
                        # Use BEM's prediction instead
                        bem_token = torch.multinomial(bem_probs, num_samples=1)
                        accepted_tokens.append(bem_token.squeeze())
                        bem_applied = True
                        break  # Stop after first rejection
                else:
                    # KL divergence is low, accept base model's token
                    accepted_tokens.append(draft_result.tokens[0, i])
        
        verification_time = time.time() - verification_start_time
        self.total_bem_time += verification_time
        self.total_verification_time += verification_time
        
        if accepted_tokens:
            accepted_tensor = torch.stack(accepted_tokens)
            self.total_accepted_tokens += len(accepted_tokens)
        else:
            accepted_tensor = torch.empty(0, dtype=torch.long)
        
        return VerificationResult(
            accepted_tokens=accepted_tensor,
            accepted_length=len(accepted_tokens),
            kl_divergences=kl_divergences,
            verification_time=verification_time,
            bem_applied=bem_applied
        )
    
    def _compute_kl_divergences(
        self,
        base_logits: torch.Tensor,
        bem_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergences between base and BEM predictions."""
        # Convert logits to probabilities
        base_probs = F.softmax(base_logits, dim=-1)
        bem_probs = F.softmax(bem_logits, dim=-1)
        
        # Compute KL divergence: KL(BEM || Base)
        kl_divs = F.kl_div(
            F.log_softmax(base_logits, dim=-1),
            bem_probs,
            reduction='none'
        ).sum(dim=-1)
        
        return kl_divs
    
    def _apply_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _fallback_generation(
        self,
        input_ids: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate single token using BEM model as fallback."""
        with torch.no_grad():
            outputs = self.bem_model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if do_sample:
                if top_p < 1.0:
                    logits = self._apply_top_p_filtering(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            return next_token.squeeze()
    
    def _adjust_draft_parameters(self):
        """Adjust draft length based on recent acceptance rates."""
        if len(self.acceptance_history) < 10:
            return
        
        recent_acceptance = np.mean(list(self.acceptance_history)[-10:])
        target_rate = self.config.target_acceptance_rate
        
        if recent_acceptance < target_rate - 0.1:
            # Low acceptance rate, reduce draft length
            self.current_draft_length = max(
                self.config.min_draft_length,
                int(self.current_draft_length * (1 - self.config.draft_length_adjustment))
            )
        elif recent_acceptance > target_rate + 0.1:
            # High acceptance rate, increase draft length
            self.current_draft_length = min(
                self.config.max_draft_length,
                int(self.current_draft_length * (1 + self.config.draft_length_adjustment))
            )
        
        self.logger.debug(f"Adjusted draft length to {self.current_draft_length} "
                         f"(acceptance rate: {recent_acceptance:.3f})")
    
    def _compute_generation_metrics(
        self,
        num_generated: int,
        generation_time: float,
        num_drafts: int,
        num_verifications: int
    ) -> SpeculativeDecodingMetrics:
        """Compute comprehensive generation metrics."""
        # Acceptance rate
        acceptance_rate = (self.total_accepted_tokens / 
                          max(self.total_drafted_tokens, 1))
        
        # Average draft length
        avg_draft_length = self.total_drafted_tokens / max(num_drafts, 1)
        
        # Tokens per second metrics
        base_tps = self.total_drafted_tokens / max(self.total_base_time, 1e-6)
        bem_tps = self.total_accepted_tokens / max(self.total_bem_time, 1e-6)
        total_tps = num_generated / max(generation_time, 1e-6)
        
        # Overhead calculation
        verification_overhead = self.total_verification_time / max(generation_time, 1e-6)
        
        # Net speedup (compared to pure BEM generation)
        # Estimate pure BEM time as: num_generated / bem_tps
        estimated_bem_only_time = num_generated / max(bem_tps, 1)
        net_speedup = estimated_bem_only_time / max(generation_time, 1e-6)
        
        return SpeculativeDecodingMetrics(
            acceptance_rate=acceptance_rate,
            average_draft_length=avg_draft_length,
            base_tokens_per_second=base_tps,
            bem_tokens_per_second=bem_tps,
            total_tokens_per_second=total_tps,
            kl_threshold_violations=sum(1 for hist in self.metrics_history 
                                      if hasattr(hist, 'kl_violations')),
            verification_overhead=verification_overhead,
            net_speedup=net_speedup
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive decoding statistics."""
        return {
            'total_drafted_tokens': self.total_drafted_tokens,
            'total_accepted_tokens': self.total_accepted_tokens,
            'overall_acceptance_rate': (self.total_accepted_tokens / 
                                       max(self.total_drafted_tokens, 1)),
            'current_draft_length': self.current_draft_length,
            'current_kl_threshold': self.current_kl_threshold,
            'total_base_time': self.total_base_time,
            'total_bem_time': self.total_bem_time,
            'total_verification_time': self.total_verification_time,
            'recent_acceptance_rate': (np.mean(list(self.acceptance_history)) 
                                      if self.acceptance_history else 0.0)
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.total_drafted_tokens = 0
        self.total_accepted_tokens = 0
        self.total_base_time = 0.0
        self.total_bem_time = 0.0
        self.total_verification_time = 0.0
        self.acceptance_history.clear()
        self.metrics_history.clear()


def create_speculative_decoder(
    base_model: nn.Module,
    bem_model: nn.Module,
    tokenizer,
    config: Optional[SpeculativeDecodingConfig] = None,
    telemetry_collector: Optional[TelemetryCollector] = None
) -> SpeculativeDecoder:
    """Create speculative decoder with default configuration."""
    if config is None:
        config = SpeculativeDecodingConfig()
    
    return SpeculativeDecoder(
        base_model=base_model,
        bem_model=bem_model,
        tokenizer=tokenizer,
        config=config,
        telemetry_collector=telemetry_collector
    )


def create_default_speculative_config(
    draft_length: int = 4,
    kl_threshold: float = 0.1,
    enable_adaptive_drafting: bool = True
) -> SpeculativeDecodingConfig:
    """Create default speculative decoding configuration."""
    return SpeculativeDecodingConfig(
        draft_length=draft_length,
        kl_threshold=kl_threshold,
        enable_adaptive_drafting=enable_adaptive_drafting,
        acceptance_threshold=0.8,
        target_acceptance_rate=0.7,
        batch_verification=True
    )


class SpeculativeDecodingBenchmark:
    """Benchmark tool for comparing speculative vs. standard decoding."""
    
    def __init__(
        self,
        base_model: nn.Module,
        bem_model: nn.Module,
        tokenizer,
        test_prompts: List[str]
    ):
        self.base_model = base_model
        self.bem_model = bem_model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
    
    def run_benchmark(
        self,
        config: SpeculativeDecodingConfig,
        max_new_tokens: int = 100,
        num_runs: int = 5
    ) -> Dict[str, any]:
        """Run comprehensive benchmark comparing decoding methods."""
        decoder = create_speculative_decoder(
            self.base_model, self.bem_model, self.tokenizer, config
        )
        
        speculative_times = []
        standard_times = []
        speculative_metrics = []
        
        for run in range(num_runs):
            for prompt in self.test_prompts:
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
                
                # Speculative decoding
                start_time = time.time()
                spec_result = decoder.generate(
                    input_ids, max_new_tokens=max_new_tokens
                )
                spec_time = time.time() - start_time
                speculative_times.append(spec_time)
                speculative_metrics.append(spec_result['metrics'])
                
                # Standard BEM decoding (for comparison)
                start_time = time.time()
                with torch.no_grad():
                    standard_result = self.bem_model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                standard_time = time.time() - start_time
                standard_times.append(standard_time)
        
        # Aggregate results
        avg_spec_time = np.mean(speculative_times)
        avg_standard_time = np.mean(standard_times)
        avg_acceptance_rate = np.mean([m.acceptance_rate for m in speculative_metrics])
        avg_speedup = np.mean([m.net_speedup for m in speculative_metrics])
        
        return {
            'speculative_avg_time': avg_spec_time,
            'standard_avg_time': avg_standard_time,
            'speedup_ratio': avg_standard_time / avg_spec_time,
            'average_acceptance_rate': avg_acceptance_rate,
            'average_net_speedup': avg_speedup,
            'num_test_cases': len(self.test_prompts) * num_runs,
            'detailed_metrics': speculative_metrics
        }


# Example usage and testing
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Mock models for testing (in practice, use real models)
    class MockModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(hidden_size, 8), 6
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            x = self.transformer(x, x)
            logits = self.lm_head(x)
            return type('Output', (), {'logits': logits})()
        
        def generate(self, input_ids, max_new_tokens=10, **kwargs):
            # Simple greedy generation for testing
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                outputs = self(generated)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            return generated
    
    # Create mock models
    base_model = MockModel()
    bem_model = MockModel()
    
    # Mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            
        def encode(self, text, return_tensors='pt'):
            # Simple character-based encoding for testing
            return torch.randint(0, 1000, (1, len(text) // 4 + 1))
            
        def decode(self, tokens):
            return f"Generated text with {len(tokens)} tokens"
    
    tokenizer = MockTokenizer()
    
    # Test speculative decoding
    config = create_default_speculative_config(draft_length=3, kl_threshold=0.1)
    decoder = create_speculative_decoder(
        base_model, bem_model, tokenizer, config
    )
    
    # Generate text
    input_ids = torch.randint(0, 1000, (1, 10))
    result = decoder.generate(input_ids, max_new_tokens=20)
    
    print("Speculative Decoding Test Results:")
    print(f"Generated tokens: {result['generated_tokens'].shape}")
    print(f"Generation time: {result['generation_time']:.3f}s")
    print(f"Metrics: {result['metrics']}")
    
    # Get statistics
    stats = decoder.get_statistics()
    print(f"\nDecoder Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")