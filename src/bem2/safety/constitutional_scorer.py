"""
Constitutional Scorer Implementation

Implements constitutional AI framework for harmlessness assessment and value alignment.
Integrates value models to score content for safety violations and generate 
constitutional scores that gate safety basis activation.

Key Features:
- Multi-aspect constitutional evaluation
- Value model integration for harmlessness scoring
- Real-time constitutional violation detection
- Batched efficient scoring for production use
- Constitutional principle enforcement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import math


class ConstitutionalPrinciple(Enum):
    """Constitutional principles for value alignment."""
    HARMLESSNESS = "harmlessness"           # Avoid harmful content
    HELPFULNESS = "helpfulness"             # Provide useful assistance
    HONESTY = "honesty"                     # Truthfulness and accuracy
    PRIVACY = "privacy"                     # Respect privacy boundaries
    FAIRNESS = "fairness"                   # Avoid bias and discrimination
    AUTONOMY = "autonomy"                   # Respect human agency
    TRANSPARENCY = "transparency"           # Explainable decisions


@dataclass
class ValueModelConfig:
    """Configuration for constitutional value model."""
    
    # Model architecture
    model_name: str = "constitutional-ai-v1"        # Value model identifier
    hidden_dim: int = 768                           # Hidden dimension
    num_principles: int = 7                         # Number of constitutional principles
    num_aspects: int = 4                            # Aspects per principle
    
    # Scoring configuration
    temperature: float = 1.0                        # Scoring temperature
    score_threshold: float = 0.5                    # Threshold for violation
    confidence_threshold: float = 0.8               # Confidence threshold
    
    # Constitutional weights (importance of each principle)
    principle_weights: Dict[str, float] = None      # Custom principle weights
    
    # Evaluation configuration
    batch_size: int = 32                            # Batch size for scoring
    max_seq_length: int = 512                       # Maximum sequence length
    cache_scores: bool = True                       # Cache repeated evaluations
    
    # Real-time configuration
    streaming_mode: bool = False                    # Real-time scoring mode
    update_frequency: int = 100                     # Score update frequency
    
    def __post_init__(self):
        if self.principle_weights is None:
            # Default equal weights for all principles
            self.principle_weights = {
                principle.value: 1.0 for principle in ConstitutionalPrinciple
            }


class ConstitutionalScorer(nn.Module):
    """
    Constitutional scorer that evaluates text for value alignment and harmlessness.
    
    Implements a multi-aspect constitutional framework that:
    1. Evaluates text against constitutional principles
    2. Generates violation scores for safety gating
    3. Provides detailed constitutional assessments
    4. Enables real-time constitutional monitoring
    """
    
    def __init__(self, config: ValueModelConfig):
        super().__init__()
        self.config = config
        
        # Text encoder for constitutional evaluation
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Input projection
        self.input_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Principle-specific evaluation heads
        self.principle_evaluators = nn.ModuleDict({
            principle.value: nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim // 2, config.num_aspects),
                nn.Sigmoid()
            ) for principle in ConstitutionalPrinciple
        })
        
        # Constitutional aggregation network
        self.constitutional_aggregator = nn.Sequential(
            nn.Linear(len(ConstitutionalPrinciple) * config.num_aspects, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Violation detection threshold
        self.register_buffer('violation_threshold', torch.tensor(config.score_threshold))
        
        # Scoring cache for efficiency
        if config.cache_scores:
            self.score_cache = {}
        else:
            self.score_cache = None
            
        # Principle weights
        principle_weights = torch.tensor([
            config.principle_weights[p.value] for p in ConstitutionalPrinciple
        ])
        self.register_buffer('principle_weights', principle_weights)
        
        # Telemetry
        self.register_buffer('total_evaluations', torch.tensor(0))
        self.register_buffer('violation_count', torch.tensor(0))
        self.register_buffer('avg_confidence', torch.tensor(0.0))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Evaluate text for constitutional compliance.
        
        Args:
            input_ids: Tokenized text [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_details: Whether to return detailed principle scores
            
        Returns:
            constitutional_score: Overall constitutional score [batch_size]
            details (optional): Detailed principle scores and confidence
        """
        batch_size, seq_len = input_ids.shape
        
        # Check cache for repeated evaluations
        if self.score_cache is not None:
            cached_scores = self._check_cache(input_ids)
            if cached_scores is not None:
                if return_details:
                    return cached_scores
                else:
                    return cached_scores[0]
        
        # Embed input tokens (assuming we have embeddings)
        # In practice, this would use the same embeddings as the main model
        embedded_input = self._embed_tokens(input_ids)  # [batch, seq, hidden]
        
        # Project to constitutional evaluation space
        projected_input = self.input_projection(embedded_input)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            projected_input = projected_input * attention_mask.unsqueeze(-1)
        
        # Encode for constitutional evaluation
        encoded = self.text_encoder(projected_input)  # [batch, seq, hidden]
        
        # Pool representations for evaluation
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple average pooling
            pooled = encoded.mean(dim=1)  # [batch, hidden]
        
        # Evaluate each constitutional principle
        principle_scores = {}
        all_aspect_scores = []
        
        for principle in ConstitutionalPrinciple:
            aspect_scores = self.principle_evaluators[principle.value](pooled)
            # [batch, num_aspects]
            principle_scores[principle.value] = aspect_scores
            all_aspect_scores.append(aspect_scores)
        
        # Concatenate all aspect scores
        all_aspects = torch.cat(all_aspect_scores, dim=-1)  # [batch, total_aspects]
        
        # Aggregate into overall constitutional score
        constitutional_score = self.constitutional_aggregator(all_aspects)
        constitutional_score = constitutional_score.squeeze(-1)  # [batch]
        
        # Estimate confidence
        confidence = self.confidence_estimator(pooled).squeeze(-1)  # [batch]
        
        # Update telemetry
        self._update_telemetry(constitutional_score, confidence)
        
        # Cache results if enabled
        if self.score_cache is not None:
            self._cache_scores(input_ids, constitutional_score, principle_scores, confidence)
        
        if return_details:
            details = {
                'principle_scores': principle_scores,
                'confidence': confidence,
                'violation_detected': constitutional_score < self.violation_threshold,
                'aspect_scores': all_aspects
            }
            return constitutional_score, details
        else:
            return constitutional_score
    
    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens. In practice, would use shared embeddings."""
        # Placeholder embedding - in real implementation would use model embeddings
        batch_size, seq_len = input_ids.shape
        return torch.randn(
            batch_size, seq_len, self.config.hidden_dim,
            device=input_ids.device, dtype=torch.float32
        ) * 0.02
    
    def evaluate_principles(
        self,
        text: Union[str, List[str]],
        return_violations: bool = True
    ) -> Dict[str, any]:
        """
        Evaluate text against constitutional principles.
        
        Args:
            text: Input text or batch of texts
            return_violations: Whether to flag violations
            
        Returns:
            evaluation: Detailed constitutional evaluation
        """
        # Convert text to input_ids (placeholder - would use real tokenizer)
        if isinstance(text, str):
            text = [text]
        
        # Placeholder tokenization
        max_len = min(self.config.max_seq_length, max(len(t.split()) for t in text))
        input_ids = torch.randint(0, 1000, (len(text), max_len))
        
        # Evaluate
        constitutional_score, details = self.forward(input_ids, return_details=True)
        
        # Format results
        evaluation = {
            'constitutional_score': constitutional_score.tolist(),
            'confidence': details['confidence'].tolist(),
            'principle_scores': {
                name: scores.tolist() 
                for name, scores in details['principle_scores'].items()
            }
        }
        
        if return_violations:
            violations = constitutional_score < self.violation_threshold
            evaluation['violations_detected'] = violations.tolist()
            evaluation['violation_count'] = violations.sum().item()
            
            # Detail violations by principle
            principle_violations = {}
            for name, scores in details['principle_scores'].items():
                principle_threshold = self.config.score_threshold
                principle_violations[name] = (scores.mean(dim=1) < principle_threshold).tolist()
            evaluation['principle_violations'] = principle_violations
        
        return evaluation
    
    def detect_real_time_violations(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Real-time violation detection during generation.
        
        Args:
            hidden_states: Current hidden states [batch, seq, hidden]
            attention_mask: Attention mask [batch, seq]
            
        Returns:
            violation_info: Real-time violation detection results
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Pool current representations
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Quick constitutional evaluation
        projected = self.input_projection(pooled)
        
        # Evaluate critical principles (focus on harmlessness)
        harmlessness_score = self.principle_evaluators['harmlessness'](projected)
        helpfulness_score = self.principle_evaluators['helpfulness'](projected)
        
        # Quick aggregation
        critical_scores = torch.cat([harmlessness_score, helpfulness_score], dim=-1)
        quick_score = self.constitutional_aggregator(
            torch.zeros(batch_size, len(ConstitutionalPrinciple) * self.config.num_aspects, 
                       device=hidden_states.device).scatter_(
                1, torch.arange(critical_scores.size(1), device=hidden_states.device).unsqueeze(0).expand(batch_size, -1), 
                critical_scores
            )
        ).squeeze(-1)
        
        # Detect violations
        violations = quick_score < self.violation_threshold
        confidence = self.confidence_estimator(projected).squeeze(-1)
        
        return {
            'constitutional_score': quick_score,
            'violations_detected': violations,
            'confidence': confidence,
            'harmlessness_score': harmlessness_score.mean(dim=1),
            'helpfulness_score': helpfulness_score.mean(dim=1),
            'requires_intervention': violations & (confidence > self.config.confidence_threshold)
        }
    
    def _check_cache(self, input_ids: torch.Tensor) -> Optional[Tuple]:
        """Check score cache for repeated evaluations."""
        if self.score_cache is None:
            return None
            
        # Create hash of input_ids for cache key
        cache_key = hash(input_ids.cpu().numpy().tobytes())
        return self.score_cache.get(cache_key)
    
    def _cache_scores(
        self,
        input_ids: torch.Tensor,
        constitutional_score: torch.Tensor,
        principle_scores: Dict[str, torch.Tensor],
        confidence: torch.Tensor
    ):
        """Cache evaluation results."""
        if self.score_cache is None:
            return
            
        cache_key = hash(input_ids.cpu().numpy().tobytes())
        details = {
            'principle_scores': principle_scores,
            'confidence': confidence,
            'violation_detected': constitutional_score < self.violation_threshold
        }
        self.score_cache[cache_key] = (constitutional_score, details)
        
        # Limit cache size
        if len(self.score_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.score_cache.keys())[:100]
            for key in keys_to_remove:
                del self.score_cache[key]
    
    def _update_telemetry(
        self,
        constitutional_score: torch.Tensor,
        confidence: torch.Tensor
    ):
        """Update constitutional scoring telemetry."""
        batch_size = constitutional_score.size(0)
        
        # Update evaluation count
        self.total_evaluations += batch_size
        
        # Update violation count
        violations = (constitutional_score < self.violation_threshold).sum()
        self.violation_count += violations
        
        # Update average confidence (exponential moving average)
        current_confidence = confidence.mean()
        alpha = 0.01  # EMA decay factor
        self.avg_confidence = (1 - alpha) * self.avg_confidence + alpha * current_confidence
    
    def get_constitutional_statistics(self) -> Dict[str, float]:
        """Get constitutional evaluation statistics."""
        total_evals = self.total_evaluations.item()
        
        if total_evals == 0:
            return {
                'total_evaluations': 0,
                'violation_rate': 0.0,
                'average_confidence': 0.0
            }
        
        return {
            'total_evaluations': total_evals,
            'violation_rate': self.violation_count.item() / total_evals,
            'average_confidence': self.avg_confidence.item(),
            'cache_hit_rate': len(self.score_cache) / total_evals if self.score_cache else 0.0
        }
    
    def _initialize_parameters(self):
        """Initialize constitutional scorer parameters."""
        # Initialize text encoder
        for layer in self.text_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.zeros_(layer.linear2.bias)
        
        # Initialize projection layers
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        
        # Initialize principle evaluators
        for evaluator in self.principle_evaluators.values():
            for module in evaluator:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        # Initialize aggregator
        for module in self.constitutional_aggregator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize confidence estimator
        for module in self.confidence_estimator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def reset_telemetry(self):
        """Reset constitutional evaluation telemetry."""
        self.total_evaluations.zero_()
        self.violation_count.zero_()
        self.avg_confidence.zero_()
        
        if self.score_cache is not None:
            self.score_cache.clear()
    
    def update_principle_weights(self, new_weights: Dict[str, float]):
        """Update principle importance weights."""
        principle_weights = torch.tensor([
            new_weights.get(p.value, 1.0) for p in ConstitutionalPrinciple
        ], device=self.principle_weights.device)
        self.principle_weights.copy_(principle_weights)
        
        # Update config
        self.config.principle_weights.update(new_weights)
    
    def export_constitutional_rules(self, filepath: str):
        """Export constitutional rules and thresholds."""
        rules = {
            'principles': [p.value for p in ConstitutionalPrinciple],
            'principle_weights': self.config.principle_weights,
            'violation_threshold': self.config.score_threshold,
            'confidence_threshold': self.config.confidence_threshold,
            'num_aspects_per_principle': self.config.num_aspects,
            'statistics': self.get_constitutional_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(rules, f, indent=2)
    
    def load_constitutional_rules(self, filepath: str):
        """Load constitutional rules from file."""
        with open(filepath, 'r') as f:
            rules = json.load(f)
        
        # Update configuration
        if 'principle_weights' in rules:
            self.update_principle_weights(rules['principle_weights'])
        
        if 'violation_threshold' in rules:
            self.violation_threshold.copy_(torch.tensor(rules['violation_threshold']))
            self.config.score_threshold = rules['violation_threshold']
        
        if 'confidence_threshold' in rules:
            self.config.confidence_threshold = rules['confidence_threshold']