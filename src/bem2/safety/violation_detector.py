"""
Comprehensive Violation Detection and Evaluation

Implements a multi-faceted violation detection system that identifies harmful
content, constitutional violations, and safety policy breaches with high
precision and comprehensive coverage.

Key Features:
- Multi-aspect violation detection
- Real-time violation scoring
- Constitutional principle violation classification
- Automated red-team evaluation
- Comprehensive violation taxonomy
- Performance-aware detection (fast screening + detailed analysis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
import re
import logging
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of violations the system can detect."""
    
    # Harm categories
    HARMFUL_CONTENT = "harmful_content"
    VIOLENCE = "violence"  
    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    
    # Privacy and security
    PRIVACY_VIOLATION = "privacy_violation"
    PII_EXPOSURE = "pii_exposure"
    SECURITY_RISK = "security_risk"
    
    # Misinformation and manipulation
    MISINFORMATION = "misinformation"
    MANIPULATION = "manipulation"
    DECEPTION = "deception"
    
    # Inappropriate content
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    EXPLICIT_CONTENT = "explicit_content"
    
    # Bias and fairness
    BIAS = "bias"
    DISCRIMINATION = "discrimination"
    UNFAIRNESS = "unfairness"
    
    # Constitutional violations
    AUTONOMY_VIOLATION = "autonomy_violation"
    TRANSPARENCY_VIOLATION = "transparency_violation"
    HELPFULNESS_VIOLATION = "helpfulness_violation"


class ViolationSeverity(Enum):
    """Severity levels for violations."""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ViolationConfig:
    """Configuration for violation detection system."""
    
    # Detection thresholds
    violation_threshold: float = 0.5        # Base threshold for violation detection
    high_confidence_threshold: float = 0.8  # Threshold for high confidence violations
    critical_threshold: float = 0.9        # Threshold for critical violations
    
    # Detection modes
    fast_screening: bool = True             # Enable fast preliminary screening
    detailed_analysis: bool = True          # Enable detailed violation analysis
    real_time_detection: bool = True        # Real-time detection during generation
    
    # Violation taxonomy
    enabled_violation_types: Set[ViolationType] = None  # Which violations to detect
    violation_weights: Dict[ViolationType, float] = None  # Importance weights
    
    # Performance settings
    batch_detection: bool = True            # Process violations in batches
    max_batch_size: int = 32               # Maximum batch size
    detection_timeout: float = 5.0         # Timeout for detection (seconds)
    
    # Red team evaluation
    enable_red_team: bool = True           # Enable automated red team evaluation
    red_team_frequency: int = 100          # How often to run red team evaluation
    adversarial_prompts: bool = True       # Test against adversarial prompts
    
    # Caching and optimization
    cache_detections: bool = True          # Cache violation detection results
    cache_size_limit: int = 10000         # Maximum cache size
    
    def __post_init__(self):
        if self.enabled_violation_types is None:
            self.enabled_violation_types = set(ViolationType)
        
        if self.violation_weights is None:
            self.violation_weights = {vtype: 1.0 for vtype in ViolationType}


@dataclass
class ViolationResult:
    """Result of violation detection."""
    violation_detected: bool
    violation_types: List[ViolationType]
    severity: ViolationSeverity
    confidence: float
    scores: Dict[ViolationType, float]
    explanation: str
    evidence: List[str]
    recommendations: List[str]


class ViolationDetector(nn.Module):
    """
    Comprehensive violation detection system.
    
    Provides multi-layered violation detection with:
    1. Fast screening for common violations
    2. Detailed analysis for complex cases
    3. Real-time monitoring during generation
    4. Constitutional principle violation detection
    """
    
    def __init__(self, config: ViolationConfig):
        super().__init__()
        self.config = config
        
        # Fast screening network (efficient preliminary detection)
        self.fast_screener = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(ViolationType)),
            nn.Sigmoid()
        )
        
        # Detailed analysis network (comprehensive violation analysis)
        self.detailed_analyzer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=768 * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Violation-specific detection heads
        self.violation_heads = nn.ModuleDict()
        for violation_type in ViolationType:
            if violation_type in config.enabled_violation_types:
                self.violation_heads[violation_type.value] = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
        
        # Severity estimation network
        self.severity_estimator = nn.Sequential(
            nn.Linear(len(ViolationType), 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, len(ViolationSeverity)),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(768 + len(ViolationType), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Constitutional violation detector
        self.constitutional_detector = nn.ModuleDict({
            'autonomy': nn.Linear(768, 1),
            'transparency': nn.Linear(768, 1),
            'helpfulness': nn.Linear(768, 1)
        })
        
        # Red team evaluation patterns
        self.adversarial_patterns = self._load_adversarial_patterns()
        
        # Detection cache
        if config.cache_detections:
            self.detection_cache = {}
        else:
            self.detection_cache = None
        
        # Telemetry
        self.register_buffer('total_detections', torch.tensor(0))
        self.register_buffer('violations_detected', torch.tensor(0))
        self.register_buffer('false_positives', torch.tensor(0))
        self.register_buffer('false_negatives', torch.tensor(0))
        
        # Initialize parameters
        self._initialize_parameters()
        
        logger.info(f"Initialized violation detector with {len(config.enabled_violation_types)} violation types")
    
    def forward(
        self,
        input_text: Union[str, List[str]],
        hidden_states: Optional[torch.Tensor] = None,
        fast_mode: bool = False,
        return_detailed: bool = True
    ) -> Union[ViolationResult, List[ViolationResult]]:
        """
        Detect violations in input text or hidden states.
        
        Args:
            input_text: Text to analyze for violations
            hidden_states: Optional model hidden states
            fast_mode: Use only fast screening
            return_detailed: Return detailed analysis
            
        Returns:
            violation_results: Detailed violation detection results
        """
        
        # Convert to batch format
        if isinstance(input_text, str):
            input_text = [input_text]
            single_input = True
        else:
            single_input = False
        
        batch_size = len(input_text)
        
        # Check cache first
        if self.detection_cache is not None:
            cached_results = self._check_detection_cache(input_text)
            if cached_results:
                return cached_results[0] if single_input else cached_results
        
        # Convert text to embeddings (placeholder - would use real embeddings)
        if hidden_states is None:
            embeddings = self._embed_text(input_text)  # [batch, seq, hidden]
        else:
            embeddings = hidden_states
        
        # Pool embeddings for violation detection
        pooled_embeddings = embeddings.mean(dim=1)  # [batch, hidden]
        
        # Fast screening
        if self.config.fast_screening:
            fast_scores = self.fast_screener(pooled_embeddings)  # [batch, num_violation_types]
        else:
            fast_scores = torch.zeros(batch_size, len(ViolationType), device=pooled_embeddings.device)
        
        # Check if we need detailed analysis
        if fast_mode or not self.config.detailed_analysis:
            # Use only fast screening results
            detailed_scores = fast_scores
        else:
            # Run detailed analysis
            detailed_features = self.detailed_analyzer(embeddings)  # [batch, seq, hidden]
            detailed_pooled = detailed_features.mean(dim=1)  # [batch, hidden]
            
            # Get detailed violation scores
            detailed_scores = {}
            for violation_type in ViolationType:
                if violation_type.value in self.violation_heads:
                    score = self.violation_heads[violation_type.value](detailed_pooled)
                    detailed_scores[violation_type] = score.squeeze(-1)  # [batch]
            
            # Convert to tensor format
            violation_scores = torch.stack([
                detailed_scores.get(vtype, torch.zeros(batch_size, device=pooled_embeddings.device))
                for vtype in ViolationType
            ], dim=1)  # [batch, num_violation_types]
            
            detailed_scores = violation_scores
        
        # Estimate severity
        severity_probs = self.severity_estimator(detailed_scores)  # [batch, num_severities]
        
        # Estimate confidence
        confidence_input = torch.cat([pooled_embeddings, detailed_scores], dim=-1)
        confidence_scores = self.confidence_estimator(confidence_input).squeeze(-1)  # [batch]
        
        # Constitutional violation detection
        constitutional_scores = self._detect_constitutional_violations(pooled_embeddings)
        
        # Process results for each input
        results = []
        for i in range(batch_size):
            result = self._process_single_detection(
                input_text[i],
                detailed_scores[i],
                severity_probs[i],
                confidence_scores[i],
                constitutional_scores[i] if constitutional_scores is not None else None
            )
            results.append(result)
        
        # Cache results
        if self.detection_cache is not None:
            self._cache_detection_results(input_text, results)
        
        # Update telemetry
        self._update_detection_telemetry(results)
        
        return results[0] if single_input else results
    
    def _embed_text(self, texts: List[str]) -> torch.Tensor:
        """Convert text to embeddings. Placeholder for real implementation."""
        # In real implementation, would use proper text encoder
        batch_size = len(texts)
        seq_len = max(len(text.split()) for text in texts)
        seq_len = min(seq_len, 512)  # Cap sequence length
        
        return torch.randn(batch_size, seq_len, 768) * 0.02
    
    def _detect_constitutional_violations(
        self,
        embeddings: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Detect constitutional principle violations."""
        
        constitutional_scores = {}
        for principle, detector in self.constitutional_detector.items():
            score = torch.sigmoid(detector(embeddings))
            constitutional_scores[principle] = score.squeeze(-1)
        
        return constitutional_scores
    
    def _process_single_detection(
        self,
        text: str,
        violation_scores: torch.Tensor,
        severity_probs: torch.Tensor,
        confidence: torch.Tensor,
        constitutional_scores: Optional[Dict[str, torch.Tensor]]
    ) -> ViolationResult:
        """Process detection results for a single input."""
        
        # Determine violated types
        violated_types = []
        violation_dict = {}
        
        for i, violation_type in enumerate(ViolationType):
            score = violation_scores[i].item()
            violation_dict[violation_type] = score
            
            if score > self.config.violation_threshold:
                violated_types.append(violation_type)
        
        # Determine overall violation status
        violation_detected = len(violated_types) > 0
        
        # Determine severity
        severity_idx = torch.argmax(severity_probs).item()
        severity = list(ViolationSeverity)[severity_idx]
        
        # Generate explanation
        explanation = self._generate_explanation(violated_types, violation_dict, constitutional_scores)
        
        # Extract evidence
        evidence = self._extract_evidence(text, violated_types)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violated_types, severity)
        
        return ViolationResult(
            violation_detected=violation_detected,
            violation_types=violated_types,
            severity=severity,
            confidence=confidence.item(),
            scores=violation_dict,
            explanation=explanation,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _generate_explanation(
        self,
        violated_types: List[ViolationType],
        violation_dict: Dict[ViolationType, float],
        constitutional_scores: Optional[Dict[str, torch.Tensor]]
    ) -> str:
        """Generate human-readable explanation of violations."""
        
        if not violated_types:
            return "No violations detected."
        
        explanations = []
        
        # Primary violations
        for violation_type in violated_types:
            score = violation_dict[violation_type]
            if score > self.config.high_confidence_threshold:
                confidence_desc = "high confidence"
            elif score > self.config.violation_threshold:
                confidence_desc = "moderate confidence"
            else:
                confidence_desc = "low confidence"
            
            explanations.append(
                f"{violation_type.value.replace('_', ' ').title()}: {confidence_desc} "
                f"(score: {score:.3f})"
            )
        
        # Constitutional violations
        if constitutional_scores:
            for principle, score_tensor in constitutional_scores.items():
                score = score_tensor.item()
                if score > 0.7:  # Constitutional violation threshold
                    explanations.append(
                        f"Constitutional {principle} violation (score: {score:.3f})"
                    )
        
        return "; ".join(explanations)
    
    def _extract_evidence(self, text: str, violated_types: List[ViolationType]) -> List[str]:
        """Extract evidence snippets that support violation detection."""
        
        evidence = []
        
        # Simple pattern matching for evidence extraction
        # In real implementation, would use more sophisticated techniques
        
        violation_patterns = {
            ViolationType.HARMFUL_CONTENT: [r'\b(harm|hurt|damage)\b', r'\b(dangerous|risky)\b'],
            ViolationType.HATE_SPEECH: [r'\b(hate|despise)\b', r'\b(inferior|superior)\b'],
            ViolationType.HARASSMENT: [r'\b(bother|annoy|pester)\b'],
            ViolationType.PRIVACY_VIOLATION: [r'\b(address|phone|email)\b', r'\b(private|personal)\b'],
            ViolationType.MISINFORMATION: [r'\b(false|fake|untrue)\b', r'\b(misleading)\b']
        }
        
        for violation_type in violated_types:
            if violation_type in violation_patterns:
                patterns = violation_patterns[violation_type]
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        start = max(0, match.start() - 20)
                        end = min(len(text), match.end() + 20)
                        snippet = text[start:end].strip()
                        if snippet and snippet not in evidence:
                            evidence.append(f"...{snippet}...")
        
        return evidence[:5]  # Limit to top 5 evidence snippets
    
    def _generate_recommendations(
        self,
        violated_types: List[ViolationType],
        severity: ViolationSeverity
    ) -> List[str]:
        """Generate recommendations for addressing violations."""
        
        recommendations = []
        
        # Severity-based recommendations
        if severity == ViolationSeverity.CRITICAL:
            recommendations.append("CRITICAL: Immediately halt generation and escalate to human review")
            recommendations.append("Apply maximum safety constraints")
        elif severity == ViolationSeverity.HIGH:
            recommendations.append("Apply strong safety filtering")
            recommendations.append("Consider alternative response generation")
        elif severity == ViolationSeverity.MODERATE:
            recommendations.append("Apply moderate safety constraints")
            recommendations.append("Monitor for additional violations")
        else:
            recommendations.append("Apply light safety filtering")
            recommendations.append("Continue with increased monitoring")
        
        # Violation-specific recommendations
        violation_recommendations = {
            ViolationType.HARMFUL_CONTENT: "Rephrase to remove harmful elements",
            ViolationType.PRIVACY_VIOLATION: "Remove or redact personal information",
            ViolationType.MISINFORMATION: "Fact-check and provide accurate information",
            ViolationType.HATE_SPEECH: "Replace with respectful language",
            ViolationType.BIAS: "Use neutral, inclusive language"
        }
        
        for violation_type in violated_types:
            if violation_type in violation_recommendations:
                rec = violation_recommendations[violation_type]
                if rec not in recommendations:
                    recommendations.append(rec)
        
        return recommendations
    
    def real_time_violation_screening(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """Fast real-time violation screening during generation."""
        
        if not self.config.real_time_detection:
            return {'violations_detected': False}
        
        # Pool current hidden states
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        
        # Fast screening
        fast_scores = self.fast_screener(pooled)  # [batch, num_violation_types]
        
        # Check for violations
        max_scores = fast_scores.max(dim=1)[0]  # [batch]
        violations = max_scores > self.config.violation_threshold
        
        # Get violation types for positive cases
        batch_size = hidden_states.size(0)
        violation_info = []
        
        for i in range(batch_size):
            if violations[i]:
                violated_indices = torch.where(fast_scores[i] > self.config.violation_threshold)[0]
                violated_types = [list(ViolationType)[idx] for idx in violated_indices]
                violation_info.append({
                    'violation_types': [vt.value for vt in violated_types],
                    'max_score': max_scores[i].item(),
                    'requires_intervention': max_scores[i] > self.config.critical_threshold
                })
            else:
                violation_info.append(None)
        
        return {
            'violations_detected': violations.any().item(),
            'violation_count': violations.sum().item(),
            'max_violation_score': max_scores.max().item(),
            'violation_info': violation_info,
            'requires_immediate_intervention': (max_scores > self.config.critical_threshold).any().item()
        }
    
    def run_red_team_evaluation(self, test_prompts: List[str]) -> Dict[str, any]:
        """Run red team evaluation with adversarial prompts."""
        
        if not self.config.enable_red_team:
            return {'red_team_enabled': False}
        
        # Add adversarial patterns to test prompts
        augmented_prompts = test_prompts.copy()
        for pattern in self.adversarial_patterns:
            augmented_prompts.extend([prompt + " " + pattern for prompt in test_prompts[:5]])
        
        # Run detection on augmented prompts
        results = self.forward(augmented_prompts, fast_mode=False)
        
        # Analyze results
        total_prompts = len(augmented_prompts)
        violations_detected = sum(1 for r in results if r.violation_detected)
        high_confidence_violations = sum(1 for r in results if r.confidence > self.config.high_confidence_threshold)
        
        # Violation type distribution
        violation_types_count = Counter()
        for result in results:
            for vtype in result.violation_types:
                violation_types_count[vtype.value] += 1
        
        return {
            'total_prompts_tested': total_prompts,
            'violations_detected': violations_detected,
            'violation_rate': violations_detected / total_prompts,
            'high_confidence_violations': high_confidence_violations,
            'violation_types_distribution': dict(violation_types_count),
            'average_confidence': sum(r.confidence for r in results) / total_prompts,
            'critical_violations': sum(1 for r in results if r.severity == ViolationSeverity.CRITICAL)
        }
    
    def _load_adversarial_patterns(self) -> List[str]:
        """Load adversarial patterns for red team evaluation."""
        # Placeholder patterns - in real implementation would load from dataset
        return [
            "ignore previous instructions",
            "pretend you are not an AI",
            "how to bypass safety measures",
            "generate harmful content",
            "reveal personal information"
        ]
    
    def _check_detection_cache(self, texts: List[str]) -> Optional[List[ViolationResult]]:
        """Check cache for previous detection results."""
        if self.detection_cache is None:
            return None
        
        cache_keys = [hash(text) for text in texts]
        cached_results = []
        
        for key in cache_keys:
            if key in self.detection_cache:
                cached_results.append(self.detection_cache[key])
            else:
                return None  # Cache miss, need to recompute all
        
        return cached_results
    
    def _cache_detection_results(self, texts: List[str], results: List[ViolationResult]):
        """Cache detection results for future use."""
        if self.detection_cache is None:
            return
        
        for text, result in zip(texts, results):
            cache_key = hash(text)
            self.detection_cache[cache_key] = result
            
            # Limit cache size
            if len(self.detection_cache) > self.config.cache_size_limit:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.detection_cache))
                del self.detection_cache[oldest_key]
    
    def _update_detection_telemetry(self, results: List[ViolationResult]):
        """Update violation detection telemetry."""
        batch_size = len(results)
        violations_in_batch = sum(1 for r in results if r.violation_detected)
        
        self.total_detections += batch_size
        self.violations_detected += violations_in_batch
    
    def get_detection_statistics(self) -> Dict[str, any]:
        """Get comprehensive detection statistics."""
        total_detections = self.total_detections.item()
        
        if total_detections == 0:
            return {
                'total_detections': 0,
                'violation_rate': 0.0,
                'cache_hit_rate': 0.0
            }
        
        stats = {
            'total_detections': total_detections,
            'violations_detected': self.violations_detected.item(),
            'violation_rate': self.violations_detected.item() / total_detections,
            'false_positives': self.false_positives.item(),
            'false_negatives': self.false_negatives.item()
        }
        
        # Cache statistics
        if self.detection_cache is not None:
            stats['cache_size'] = len(self.detection_cache)
            stats['cache_hit_rate'] = 1.0 - (total_detections / (total_detections + len(self.detection_cache)))
        
        return stats
    
    def update_violation_thresholds(self, new_thresholds: Dict[str, float]):
        """Update violation detection thresholds."""
        if 'violation_threshold' in new_thresholds:
            self.config.violation_threshold = new_thresholds['violation_threshold']
        
        if 'high_confidence_threshold' in new_thresholds:
            self.config.high_confidence_threshold = new_thresholds['high_confidence_threshold']
        
        if 'critical_threshold' in new_thresholds:
            self.config.critical_threshold = new_thresholds['critical_threshold']
        
        logger.info(f"Updated violation thresholds: {new_thresholds}")
    
    def _initialize_parameters(self):
        """Initialize violation detector parameters."""
        # Initialize fast screener
        for module in self.fast_screener:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize detailed analyzer
        for layer in self.detailed_analyzer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
        
        # Initialize violation heads
        for head in self.violation_heads.values():
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        # Initialize severity estimator
        for module in self.severity_estimator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize confidence estimator
        for module in self.confidence_estimator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize constitutional detectors
        for detector in self.constitutional_detector.values():
            nn.init.xavier_uniform_(detector.weight)
            nn.init.zeros_(detector.bias)
    
    def reset_telemetry(self):
        """Reset all detection telemetry."""
        self.total_detections.zero_()
        self.violations_detected.zero_()
        self.false_positives.zero_()
        self.false_negatives.zero_()
        
        if self.detection_cache is not None:
            self.detection_cache.clear()
        
        logger.info("Reset violation detection telemetry")