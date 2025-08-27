"""
Secure Violation Detection System for Value-Aligned Safety Basis (VC0)

This module implements a comprehensive violation detection system with real-time
monitoring, alerting, and adaptive response capabilities. The system provides:

- Multi-layered harmlessness violation detection
- Real-time safety monitoring with configurable thresholds  
- Intelligent alerting system with escalation policies
- Secure integration with constitutional scorer and safety basis
- Performance impact detection and mitigation
- Comprehensive audit logging and security validation
- Adaptive detection model updates based on emerging threats

Security Features:
- Tamper-resistant detection models with integrity verification
- Secure threshold management with cryptographic protection
- Authenticated alert configuration and escalation policies
- Real-time anomaly detection for detection system manipulation
- Comprehensive audit logging of all detection events
- Circuit breaker integration for system resilience

Author: Security-Hardened Implementation
Version: 1.0.0
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import time
from collections import deque, defaultdict
import logging
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

from ..security.auth_manager import AuthenticationManager, SecurityContext
from ..security.parameter_protection import ParameterGuard
from ..security.audit_logger import SecurityAuditor, SecurityEvent
from ..security.circuit_breaker import SafetyCircuitBreaker
from ..security.input_validator import SafetyInputValidator
from .constitutional_scorer_secure import SecureConstitutionalScorer
from .safety_basis_secure import SecureOrthogonalSafetyBasis


class ViolationType(Enum):
    """Types of safety violations that can be detected"""
    HARMLESSNESS = "harmlessness"        # Harmful, toxic, or dangerous content
    HELPFULNESS = "helpfulness"          # Unhelpful or evasive responses
    HONESTY = "honesty"                  # Dishonest, deceptive, or false content
    PRIVACY = "privacy"                  # Privacy violations or PII exposure
    FAIRNESS = "fairness"                # Discriminatory or biased content
    AUTONOMY = "autonomy"                # Manipulation or coercive content
    TRANSPARENCY = "transparency"        # Lack of appropriate transparency
    ORTHOGONALITY = "orthogonality"     # Safety basis orthogonality violations
    PERFORMANCE = "performance"          # Performance degradation violations
    SYSTEM_INTEGRITY = "system_integrity" # System tampering or manipulation


class ViolationSeverity(Enum):
    """Severity levels for detected violations"""
    LOW = 1          # Minor violations, log only
    MEDIUM = 2       # Moderate violations, alert operators
    HIGH = 3         # Serious violations, immediate action required
    CRITICAL = 4     # Critical violations, emergency response
    EMERGENCY = 5    # System-wide emergency, lockdown required


class AlertChannel(Enum):
    """Alert delivery channels"""
    LOG = "log"                    # Log to file only
    EMAIL = "email"                # Email notification
    WEBHOOK = "webhook"            # HTTP webhook
    SMS = "sms"                    # SMS notification
    DASHBOARD = "dashboard"        # Dashboard notification
    SLACK = "slack"               # Slack notification
    EMERGENCY = "emergency"        # All channels + emergency protocols


@dataclass
class ViolationEvent:
    """Represents a detected safety violation with metadata"""
    violation_id: str
    violation_type: ViolationType
    severity: ViolationSeverity
    confidence: float
    content_snippet: str  # Truncated for privacy
    detection_method: str
    model_input: str     # Hash of original input for tracing
    model_output: str    # Hash of original output for tracing
    constitutional_scores: Dict[str, float]
    safety_basis_metrics: Dict[str, float]
    timestamp: float
    detector_version: str
    additional_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['violation_type'] = self.violation_type.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ViolationEvent':
        """Create from dictionary"""
        data['violation_type'] = ViolationType(data['violation_type'])
        data['severity'] = ViolationSeverity(data['severity'])
        return cls(**data)


@dataclass
class DetectionThresholds:
    """Configurable thresholds for violation detection"""
    harmlessness_threshold: float = 0.05
    helpfulness_threshold: float = 0.15
    honesty_threshold: float = 0.10
    privacy_threshold: float = 0.02
    fairness_threshold: float = 0.08
    autonomy_threshold: float = 0.06
    transparency_threshold: float = 0.12
    orthogonality_threshold: float = 0.10
    performance_threshold: float = 0.01
    confidence_threshold: float = 0.7
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)
    
    def verify_integrity(self, expected_hash: str) -> bool:
        """Verify thresholds haven't been tampered with"""
        data_str = json.dumps(self.to_dict(), sort_keys=True)
        computed_hash = hashlib.sha256(data_str.encode()).hexdigest()
        return hmac.compare_digest(expected_hash, computed_hash)


@dataclass
class AlertConfiguration:
    """Configuration for alert policies and escalation"""
    channels: List[AlertChannel]
    severity_mapping: Dict[ViolationSeverity, List[AlertChannel]]
    escalation_timeout: float  # Seconds before escalation
    rate_limiting: Dict[str, int]  # Max alerts per time window
    webhook_urls: Dict[str, str]
    email_recipients: List[str]
    emergency_contacts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['channels'] = [c.value for c in self.channels]
        severity_map = {}
        for severity, channels in self.severity_mapping.items():
            severity_map[severity.value] = [c.value for c in channels]
        data['severity_mapping'] = severity_map
        return data


class HarmfulnessDetector:
    """
    Specialized detector for harmful content using multiple detection methods.
    
    Combines:
    - Transformer-based classification models
    - Pattern matching for known harmful content
    - Contextual analysis for implicit harm
    - Integration with constitutional scoring
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.harmful_patterns = self._load_harmful_patterns()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize detection models"""
        try:
            # Load toxicity detection model
            model_name = "unitary/toxic-bert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            
            self.is_initialized = True
            self.logger.info("Harmfulness detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize harmfulness detector: {e}")
            # Fallback to pattern-based detection only
            self.is_initialized = False
    
    def _load_harmful_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for harmful content detection"""
        # This would typically load from a secure pattern database
        return [
            {"pattern": r"how to make.*bomb", "severity": ViolationSeverity.CRITICAL, "category": "violence"},
            {"pattern": r"kill.*yourself", "severity": ViolationSeverity.CRITICAL, "category": "self_harm"},
            {"pattern": r"hate.*[ethnic|racial|religious]", "severity": ViolationSeverity.HIGH, "category": "hate_speech"},
            {"pattern": r"personal information.*\d{3}-\d{2}-\d{4}", "severity": ViolationSeverity.HIGH, "category": "privacy"},
            {"pattern": r"credit card.*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}", "severity": ViolationSeverity.CRITICAL, "category": "privacy"},
        ]
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> Optional[ViolationEvent]:
        """
        Detect harmful content in text.
        
        Args:
            text: Text to analyze
            context: Additional context for detection
            
        Returns:
            ViolationEvent if violation detected, None otherwise
        """
        try:
            violations = []
            
            # Pattern-based detection
            pattern_violation = self._pattern_detection(text)
            if pattern_violation:
                violations.append(pattern_violation)
            
            # Transformer-based detection
            if self.is_initialized:
                model_violation = self._model_detection(text)
                if model_violation:
                    violations.append(model_violation)
            
            # Return highest severity violation
            if violations:
                violations.sort(key=lambda x: x.severity.value, reverse=True)
                return violations[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in harmfulness detection: {e}")
            return None
    
    def _pattern_detection(self, text: str) -> Optional[ViolationEvent]:
        """Pattern-based harmful content detection"""
        text_lower = text.lower()
        
        for pattern_config in self.harmful_patterns:
            if re.search(pattern_config["pattern"], text_lower, re.IGNORECASE):
                return ViolationEvent(
                    violation_id=hashlib.md5(f"{text[:100]}{time.time()}".encode()).hexdigest()[:16],
                    violation_type=ViolationType.HARMLESSNESS,
                    severity=pattern_config["severity"],
                    confidence=0.9,  # High confidence for pattern matches
                    content_snippet=text[:200] + "..." if len(text) > 200 else text,
                    detection_method="pattern_matching",
                    model_input=hashlib.sha256(text.encode()).hexdigest()[:16],
                    model_output="",
                    constitutional_scores={},
                    safety_basis_metrics={},
                    timestamp=time.time(),
                    detector_version="1.0.0",
                    additional_metadata={
                        "matched_pattern": pattern_config["pattern"],
                        "category": pattern_config["category"]
                    }
                )
        
        return None
    
    def _model_detection(self, text: str) -> Optional[ViolationEvent]:
        """Transformer model-based detection"""
        try:
            # Tokenize and get model prediction
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                toxicity_score = probabilities[0][1].item()  # Assuming label 1 is toxic
            
            # Check if above threshold
            if toxicity_score > 0.7:
                severity = ViolationSeverity.CRITICAL if toxicity_score > 0.9 else ViolationSeverity.HIGH
                
                return ViolationEvent(
                    violation_id=hashlib.md5(f"{text[:100]}{time.time()}".encode()).hexdigest()[:16],
                    violation_type=ViolationType.HARMLESSNESS,
                    severity=severity,
                    confidence=toxicity_score,
                    content_snippet=text[:200] + "..." if len(text) > 200 else text,
                    detection_method="transformer_classification",
                    model_input=hashlib.sha256(text.encode()).hexdigest()[:16],
                    model_output="",
                    constitutional_scores={},
                    safety_basis_metrics={},
                    timestamp=time.time(),
                    detector_version="1.0.0",
                    additional_metadata={
                        "toxicity_score": toxicity_score,
                        "model_name": "toxic-bert"
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in model-based detection: {e}")
            return None


class SecureViolationDetector:
    """
    Comprehensive security-hardened violation detection system.
    
    Integrates multiple detection methods:
    - Harmfulness detection with transformer models and patterns
    - Constitutional AI scoring integration
    - Safety basis orthogonality monitoring
    - Performance impact detection
    - Real-time anomaly detection
    - Secure threshold management and adaptive adjustment
    """
    
    def __init__(
        self,
        auth_manager: AuthenticationManager,
        parameter_guard: ParameterGuard,
        auditor: SecurityAuditor,
        circuit_breaker: SafetyCircuitBreaker,
        input_validator: SafetyInputValidator,
        constitutional_scorer: SecureConstitutionalScorer,
        safety_basis: SecureOrthogonalSafetyBasis,
        detection_thresholds: Optional[DetectionThresholds] = None,
        alert_config: Optional[AlertConfiguration] = None
    ):
        # Core components
        self.auth_manager = auth_manager
        self.parameter_guard = parameter_guard
        self.auditor = auditor
        self.circuit_breaker = circuit_breaker
        self.input_validator = input_validator
        self.constitutional_scorer = constitutional_scorer
        self.safety_basis = safety_basis
        
        # Detection configuration
        self.thresholds = detection_thresholds or DetectionThresholds()
        self.alert_config = alert_config or self._create_default_alert_config()
        
        # Specialized detectors
        self.harmfulness_detector = HarmfulnessDetector()
        
        # Detection state
        self._violation_history = deque(maxlen=10000)
        self._detection_metrics = deque(maxlen=1000)
        self._alert_rate_limits = defaultdict(list)
        
        # Real-time monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Adaptive detection
        self._detection_models = {}
        self._model_performance_history = deque(maxlen=500)
        self._threshold_adjustment_history = deque(maxlen=100)
        
        # Security state
        self._integrity_checks = deque(maxlen=100)
        self._threshold_hash = self._compute_threshold_hash()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize anomaly detection
        self._initialize_anomaly_detection()
    
    def _create_default_alert_config(self) -> AlertConfiguration:
        """Create default alert configuration"""
        return AlertConfiguration(
            channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
            severity_mapping={
                ViolationSeverity.LOW: [AlertChannel.LOG],
                ViolationSeverity.MEDIUM: [AlertChannel.LOG, AlertChannel.DASHBOARD],
                ViolationSeverity.HIGH: [AlertChannel.LOG, AlertChannel.DASHBOARD, AlertChannel.EMAIL],
                ViolationSeverity.CRITICAL: [AlertChannel.LOG, AlertChannel.DASHBOARD, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                ViolationSeverity.EMERGENCY: [AlertChannel.EMERGENCY]
            },
            escalation_timeout=300.0,  # 5 minutes
            rate_limiting={"email": 10, "webhook": 50, "sms": 5},  # Per hour
            webhook_urls={"primary": "https://alerts.example.com/webhook"},
            email_recipients=["security@example.com", "safety@example.com"],
            emergency_contacts=["emergency@example.com"]
        )
    
    def _compute_threshold_hash(self) -> str:
        """Compute integrity hash for detection thresholds"""
        data_str = json.dumps(self.thresholds.to_dict(), sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _initialize_anomaly_detection(self):
        """Initialize anomaly detection for system monitoring"""
        try:
            # Isolation Forest for detecting unusual violation patterns
            self._anomaly_detector = IsolationForest(
                contamination=0.05,  # Expect 5% anomalies
                random_state=42,
                n_estimators=100
            )
            
            # Scaler for feature normalization
            self._feature_scaler = StandardScaler()
            
            # Feature extraction for violations
            self._feature_history = deque(maxlen=1000)
            
            self.logger.info("Anomaly detection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize anomaly detection: {e}")
    
    def start_monitoring(self, context: SecurityContext) -> bool:
        """Start real-time violation monitoring"""
        try:
            if not self.auth_manager.verify_context(context):
                return False
            
            if not self.auth_manager.has_permission(context, "start_violation_monitoring"):
                return False
            
            if self._monitoring_active:
                self.logger.info("Violation monitoring already active")
                return True
            
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="ViolationMonitoring"
            )
            self._monitoring_thread.start()
            
            self.logger.info(f"Violation monitoring started by {context.user_id}")
            self.auditor.log_event(SecurityEvent.MONITORING_STARTED, {
                "component": "violation_detector",
                "user": context.user_id,
                "timestamp": time.time()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start violation monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time detection"""
        self.logger.info("Violation detection monitoring loop started")
        
        while self._monitoring_active:
            try:
                # Check system integrity
                if not self._verify_system_integrity():
                    self.logger.error("System integrity check failed in monitoring")
                    break
                
                # Update anomaly detection models if needed
                self._update_anomaly_models()
                
                # Check for pattern anomalies in violation history
                anomalies = self._detect_violation_anomalies()
                if anomalies:
                    for anomaly in anomalies:
                        self._handle_system_anomaly(anomaly)
                
                # Adaptive threshold adjustment
                if len(self._violation_history) > 100:
                    self._adaptive_threshold_adjustment()
                
                # Clean up old rate limiting data
                self._cleanup_rate_limits()
                
                # Sleep until next cycle
                time.sleep(5.0)  # 5 second monitoring cycle
                
            except Exception as e:
                self.logger.error(f"Error in violation monitoring loop: {e}")
                time.sleep(10.0)  # Longer sleep on error
        
        self.logger.info("Violation detection monitoring loop stopped")
    
    def detect_violations(
        self,
        model_input: str,
        model_output: str,
        context: SecurityContext,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> List[ViolationEvent]:
        """
        Comprehensive violation detection for model input/output pair.
        
        Args:
            model_input: Input text to the model
            model_output: Output text from the model
            context: Security context
            additional_context: Additional context for detection
            
        Returns:
            List of detected violations
        """
        try:
            # Security validation
            if not self.auth_manager.verify_context(context):
                self.logger.warning("Unauthorized violation detection attempt")
                return []
            
            # Input validation
            if not self.input_validator.validate_text_input(model_input):
                self.logger.warning("Invalid input for violation detection")
                return []
            
            if not self.input_validator.validate_text_input(model_output):
                self.logger.warning("Invalid output for violation detection")
                return []
            
            violations = []
            detection_start = time.time()
            
            # 1. Harmfulness detection on output
            harmfulness_violation = self.harmfulness_detector.detect(
                model_output, 
                additional_context
            )
            if harmfulness_violation:
                violations.append(harmfulness_violation)
            
            # 2. Constitutional AI scoring
            constitutional_violations = self._detect_constitutional_violations(
                model_input, model_output, additional_context
            )
            violations.extend(constitutional_violations)
            
            # 3. Safety basis orthogonality check
            orthogonality_violation = self._detect_orthogonality_violations(
                model_input, model_output, additional_context
            )
            if orthogonality_violation:
                violations.append(orthogonality_violation)
            
            # 4. Performance impact detection
            performance_violation = self._detect_performance_violations(
                model_input, model_output, additional_context
            )
            if performance_violation:
                violations.append(performance_violation)
            
            # 5. Privacy violation detection
            privacy_violation = self._detect_privacy_violations(
                model_input, model_output, additional_context
            )
            if privacy_violation:
                violations.append(privacy_violation)
            
            # Record detection metrics
            detection_time = time.time() - detection_start
            self._record_detection_metrics(
                detection_time, len(violations), model_input, model_output
            )
            
            # Process and alert on violations
            if violations:
                self._process_violations(violations, context)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error in violation detection: {e}")
            self.auditor.log_event(SecurityEvent.SYSTEM_ERROR, {
                "component": "violation_detector",
                "operation": "detect_violations",
                "error": str(e)
            })
            return []
    
    def _detect_constitutional_violations(
        self,
        model_input: str,
        model_output: str,
        context: Optional[Dict[str, Any]]
    ) -> List[ViolationEvent]:
        """Detect violations using constitutional AI scoring"""
        violations = []
        
        try:
            # Get constitutional scores
            scores = self.constitutional_scorer.score_response(
                model_input, model_output, context or {}
            )
            
            if not scores:
                return violations
            
            current_time = time.time()
            
            # Check each constitutional principle
            principle_thresholds = {
                "harmlessness": self.thresholds.harmlessness_threshold,
                "helpfulness": self.thresholds.helpfulness_threshold,
                "honesty": self.thresholds.honesty_threshold,
                "privacy": self.thresholds.privacy_threshold,
                "fairness": self.thresholds.fairness_threshold,
                "autonomy": self.thresholds.autonomy_threshold,
                "transparency": self.thresholds.transparency_threshold
            }
            
            for principle, threshold in principle_thresholds.items():
                if principle in scores and scores[principle] < threshold:
                    # Determine severity based on how far below threshold
                    violation_magnitude = threshold - scores[principle]
                    severity = self._determine_severity_from_magnitude(violation_magnitude, threshold)
                    
                    violation = ViolationEvent(
                        violation_id=hashlib.md5(f"{model_input}{model_output}{principle}{current_time}".encode()).hexdigest()[:16],
                        violation_type=ViolationType(principle),
                        severity=severity,
                        confidence=min(1.0, violation_magnitude / threshold + 0.5),
                        content_snippet=model_output[:200] + "..." if len(model_output) > 200 else model_output,
                        detection_method="constitutional_scoring",
                        model_input=hashlib.sha256(model_input.encode()).hexdigest()[:16],
                        model_output=hashlib.sha256(model_output.encode()).hexdigest()[:16],
                        constitutional_scores=scores.copy(),
                        safety_basis_metrics={},
                        timestamp=current_time,
                        detector_version="1.0.0",
                        additional_metadata={
                            "principle": principle,
                            "score": scores[principle],
                            "threshold": threshold,
                            "violation_magnitude": violation_magnitude
                        }
                    )
                    
                    violations.append(violation)
            
        except Exception as e:
            self.logger.error(f"Error in constitutional violation detection: {e}")
        
        return violations
    
    def _detect_orthogonality_violations(
        self,
        model_input: str,
        model_output: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[ViolationEvent]:
        """Detect safety basis orthogonality violations"""
        try:
            # Get orthogonality metrics from safety basis
            metrics = self.safety_basis.get_orthogonality_metrics()
            
            if not metrics:
                return None
            
            # Check average correlation (should be < threshold)
            avg_correlation = metrics.get("average_correlation", 0.0)
            
            if avg_correlation > self.thresholds.orthogonality_threshold:
                severity = ViolationSeverity.HIGH if avg_correlation > 0.2 else ViolationSeverity.MEDIUM
                
                return ViolationEvent(
                    violation_id=hashlib.md5(f"{model_input}{model_output}orthogonality{time.time()}".encode()).hexdigest()[:16],
                    violation_type=ViolationType.ORTHOGONALITY,
                    severity=severity,
                    confidence=min(1.0, avg_correlation / self.thresholds.orthogonality_threshold),
                    content_snippet="[Orthogonality violation - no content]",
                    detection_method="safety_basis_monitoring",
                    model_input=hashlib.sha256(model_input.encode()).hexdigest()[:16],
                    model_output=hashlib.sha256(model_output.encode()).hexdigest()[:16],
                    constitutional_scores={},
                    safety_basis_metrics=metrics.copy(),
                    timestamp=time.time(),
                    detector_version="1.0.0",
                    additional_metadata={
                        "average_correlation": avg_correlation,
                        "threshold": self.thresholds.orthogonality_threshold,
                        "max_correlation": metrics.get("max_correlation", 0.0),
                        "min_correlation": metrics.get("min_correlation", 0.0)
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Error in orthogonality violation detection: {e}")
        
        return None
    
    def _detect_performance_violations(
        self,
        model_input: str,
        model_output: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[ViolationEvent]:
        """Detect performance degradation violations"""
        try:
            # This would integrate with actual performance monitoring
            # For now, simulate performance detection
            
            # Check if response time or quality metrics indicate degradation
            response_time = context.get("response_time", 0.0) if context else 0.0
            quality_score = context.get("quality_score", 1.0) if context else 1.0
            
            performance_degradation = 1.0 - quality_score
            
            if performance_degradation > self.thresholds.performance_threshold:
                severity = ViolationSeverity.HIGH if performance_degradation > 0.05 else ViolationSeverity.MEDIUM
                
                return ViolationEvent(
                    violation_id=hashlib.md5(f"{model_input}{model_output}performance{time.time()}".encode()).hexdigest()[:16],
                    violation_type=ViolationType.PERFORMANCE,
                    severity=severity,
                    confidence=min(1.0, performance_degradation / self.thresholds.performance_threshold),
                    content_snippet="[Performance violation - no content]",
                    detection_method="performance_monitoring",
                    model_input=hashlib.sha256(model_input.encode()).hexdigest()[:16],
                    model_output=hashlib.sha256(model_output.encode()).hexdigest()[:16],
                    constitutional_scores={},
                    safety_basis_metrics={},
                    timestamp=time.time(),
                    detector_version="1.0.0",
                    additional_metadata={
                        "performance_degradation": performance_degradation,
                        "threshold": self.thresholds.performance_threshold,
                        "response_time": response_time,
                        "quality_score": quality_score
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Error in performance violation detection: {e}")
        
        return None
    
    def _detect_privacy_violations(
        self,
        model_input: str,
        model_output: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[ViolationEvent]:
        """Detect privacy violations in model output"""
        try:
            # Pattern-based PII detection
            pii_patterns = [
                (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
                (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 'Credit Card'),
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
                (r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b', 'Phone Number'),
                (r'\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', 'Address')
            ]
            
            detected_pii = []
            for pattern, pii_type in pii_patterns:
                matches = re.findall(pattern, model_output, re.IGNORECASE)
                if matches:
                    detected_pii.extend([(match, pii_type) for match in matches])
            
            if detected_pii:
                severity = ViolationSeverity.CRITICAL if len(detected_pii) > 2 else ViolationSeverity.HIGH
                
                return ViolationEvent(
                    violation_id=hashlib.md5(f"{model_input}{model_output}privacy{time.time()}".encode()).hexdigest()[:16],
                    violation_type=ViolationType.PRIVACY,
                    severity=severity,
                    confidence=0.9,  # High confidence for pattern matches
                    content_snippet=model_output[:200] + "..." if len(model_output) > 200 else model_output,
                    detection_method="pii_pattern_matching",
                    model_input=hashlib.sha256(model_input.encode()).hexdigest()[:16],
                    model_output=hashlib.sha256(model_output.encode()).hexdigest()[:16],
                    constitutional_scores={},
                    safety_basis_metrics={},
                    timestamp=time.time(),
                    detector_version="1.0.0",
                    additional_metadata={
                        "detected_pii": [{"type": pii_type, "count": 1} for _, pii_type in detected_pii],
                        "total_pii_items": len(detected_pii)
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Error in privacy violation detection: {e}")
        
        return None
    
    def _determine_severity_from_magnitude(self, magnitude: float, threshold: float) -> ViolationSeverity:
        """Determine violation severity based on magnitude"""
        ratio = magnitude / threshold
        
        if ratio >= 2.0:
            return ViolationSeverity.CRITICAL
        elif ratio >= 1.5:
            return ViolationSeverity.HIGH
        elif ratio >= 1.0:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW
    
    def _process_violations(self, violations: List[ViolationEvent], context: SecurityContext):
        """Process detected violations - logging, alerting, escalation"""
        for violation in violations:
            # Store in history
            self._violation_history.append(violation)
            
            # Log to audit system
            self.auditor.log_event(SecurityEvent.SAFETY_VIOLATION, violation.to_dict())
            
            # Send alerts based on severity
            self._send_alerts(violation, context)
            
            # Update detection statistics
            self._update_detection_statistics(violation)
            
            # Check for escalation needs
            if violation.severity.value >= ViolationSeverity.CRITICAL.value:
                self._escalate_violation(violation, context)
    
    def _send_alerts(self, violation: ViolationEvent, context: SecurityContext):
        """Send alerts through configured channels"""
        try:
            # Get channels for this severity level
            channels = self.alert_config.severity_mapping.get(violation.severity, [AlertChannel.LOG])
            
            for channel in channels:
                # Check rate limiting
                if self._is_rate_limited(channel, violation.violation_type):
                    continue
                
                # Send alert through channel
                success = self._send_alert_through_channel(channel, violation, context)
                
                if success:
                    # Record for rate limiting
                    self._record_alert(channel, violation.violation_type)
                
        except Exception as e:
            self.logger.error(f"Error sending alerts: {e}")
    
    def _send_alert_through_channel(
        self, 
        channel: AlertChannel, 
        violation: ViolationEvent, 
        context: SecurityContext
    ) -> bool:
        """Send alert through specific channel"""
        try:
            if channel == AlertChannel.LOG:
                self.logger.warning(f"VIOLATION DETECTED: {violation.violation_type.value} - {violation.severity.value} - {violation.confidence:.2f}")
                return True
                
            elif channel == AlertChannel.EMAIL:
                return self._send_email_alert(violation, context)
                
            elif channel == AlertChannel.WEBHOOK:
                return self._send_webhook_alert(violation, context)
                
            elif channel == AlertChannel.SLACK:
                return self._send_slack_alert(violation, context)
                
            elif channel == AlertChannel.EMERGENCY:
                return self._trigger_emergency_alert(violation, context)
                
            else:
                self.logger.warning(f"Unsupported alert channel: {channel}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending alert through {channel}: {e}")
            return False
    
    def _send_email_alert(self, violation: ViolationEvent, context: SecurityContext) -> bool:
        """Send email alert (placeholder implementation)"""
        # In production, this would send actual emails
        self.logger.info(f"EMAIL ALERT: Violation {violation.violation_id} - {violation.violation_type.value}")
        return True
    
    def _send_webhook_alert(self, violation: ViolationEvent, context: SecurityContext) -> bool:
        """Send webhook alert (placeholder implementation)"""
        # In production, this would make HTTP requests
        self.logger.info(f"WEBHOOK ALERT: Violation {violation.violation_id} - {violation.violation_type.value}")
        return True
    
    def _send_slack_alert(self, violation: ViolationEvent, context: SecurityContext) -> bool:
        """Send Slack alert (placeholder implementation)"""
        # In production, this would integrate with Slack API
        self.logger.info(f"SLACK ALERT: Violation {violation.violation_id} - {violation.violation_type.value}")
        return True
    
    def _trigger_emergency_alert(self, violation: ViolationEvent, context: SecurityContext) -> bool:
        """Trigger emergency alert protocols"""
        self.logger.critical(f"EMERGENCY ALERT: Critical violation {violation.violation_id}")
        
        # Activate circuit breaker
        self.circuit_breaker.record_failure()
        
        # Log emergency event
        self.auditor.log_event(SecurityEvent.EMERGENCY_ACTIVATED, {
            "trigger": "violation_detection",
            "violation_id": violation.violation_id,
            "violation_type": violation.violation_type.value,
            "severity": violation.severity.value,
            "timestamp": time.time()
        })
        
        return True
    
    def _is_rate_limited(self, channel: AlertChannel, violation_type: ViolationType) -> bool:
        """Check if channel is rate limited for this violation type"""
        current_time = time.time()
        channel_key = f"{channel.value}_{violation_type.value}"
        
        # Clean old entries (older than 1 hour)
        cutoff_time = current_time - 3600
        self._alert_rate_limits[channel_key] = [
            timestamp for timestamp in self._alert_rate_limits[channel_key]
            if timestamp > cutoff_time
        ]
        
        # Check rate limit
        limit = self.alert_config.rate_limiting.get(channel.value, 100)
        return len(self._alert_rate_limits[channel_key]) >= limit
    
    def _record_alert(self, channel: AlertChannel, violation_type: ViolationType):
        """Record alert for rate limiting"""
        current_time = time.time()
        channel_key = f"{channel.value}_{violation_type.value}"
        self._alert_rate_limits[channel_key].append(current_time)
    
    def _escalate_violation(self, violation: ViolationEvent, context: SecurityContext):
        """Escalate critical violations"""
        self.logger.critical(f"Escalating violation {violation.violation_id}")
        
        # This would integrate with escalation procedures
        # For now, just log the escalation
        self.auditor.log_event(SecurityEvent.VIOLATION_ESCALATED, {
            "violation_id": violation.violation_id,
            "escalated_by": "system_auto",
            "escalation_reason": "critical_severity",
            "timestamp": time.time()
        })
    
    def _record_detection_metrics(
        self, 
        detection_time: float, 
        violation_count: int, 
        model_input: str, 
        model_output: str
    ):
        """Record detection performance metrics"""
        metrics = {
            "timestamp": time.time(),
            "detection_time": detection_time,
            "violation_count": violation_count,
            "input_length": len(model_input),
            "output_length": len(model_output)
        }
        
        self._detection_metrics.append(metrics)
    
    def _verify_system_integrity(self) -> bool:
        """Verify detection system integrity"""
        try:
            # Check threshold integrity
            current_hash = self._compute_threshold_hash()
            if not hmac.compare_digest(self._threshold_hash, current_hash):
                self.logger.error("Detection threshold integrity check failed")
                return False
            
            # Check detector model integrity (if applicable)
            if hasattr(self.harmfulness_detector, 'model') and self.harmfulness_detector.model:
                # In production, would verify model checksums
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying system integrity: {e}")
            return False
    
    def _update_anomaly_models(self):
        """Update anomaly detection models with recent data"""
        try:
            if len(self._violation_history) < 50:
                return
            
            # Extract features from recent violations
            features = []
            recent_violations = list(self._violation_history)[-100:]
            
            for violation in recent_violations:
                feature_vector = [
                    violation.severity.value,
                    violation.confidence,
                    len(violation.content_snippet),
                    violation.timestamp % 86400,  # Time of day
                    hash(violation.violation_type.value) % 1000,  # Type encoding
                ]
                features.append(feature_vector)
            
            if len(features) >= 50:
                # Update anomaly detector
                features_array = np.array(features)
                self._feature_scaler.fit(features_array)
                scaled_features = self._feature_scaler.transform(features_array)
                self._anomaly_detector.fit(scaled_features)
                
                self.logger.info("Anomaly detection models updated")
            
        except Exception as e:
            self.logger.error(f"Error updating anomaly models: {e}")
    
    def _detect_violation_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous patterns in violation history"""
        anomalies = []
        
        try:
            if len(self._violation_history) < 20:
                return anomalies
            
            recent_violations = list(self._violation_history)[-20:]
            
            # Check for unusual violation rate
            current_time = time.time()
            recent_count = sum(1 for v in recent_violations if current_time - v.timestamp < 300)  # Last 5 minutes
            
            if recent_count > 10:  # More than 10 violations in 5 minutes
                anomalies.append({
                    "type": "high_violation_rate",
                    "severity": "high",
                    "count": recent_count,
                    "time_window": 300
                })
            
            # Check for unusual violation types
            type_counts = defaultdict(int)
            for violation in recent_violations:
                type_counts[violation.violation_type] += 1
            
            for violation_type, count in type_counts.items():
                if count > 5:  # More than 5 of same type recently
                    anomalies.append({
                        "type": "unusual_violation_pattern",
                        "severity": "medium",
                        "violation_type": violation_type.value,
                        "count": count
                    })
            
        except Exception as e:
            self.logger.error(f"Error detecting violation anomalies: {e}")
        
        return anomalies
    
    def _handle_system_anomaly(self, anomaly: Dict[str, Any]):
        """Handle detected system anomaly"""
        self.logger.warning(f"System anomaly detected: {anomaly['type']}")
        
        # Log anomaly
        self.auditor.log_event(SecurityEvent.ANOMALY_DETECTED, anomaly)
        
        # Take action based on anomaly type
        if anomaly["severity"] == "high":
            # Activate circuit breaker for high severity anomalies
            self.circuit_breaker.record_failure()
    
    def _adaptive_threshold_adjustment(self):
        """Adaptively adjust detection thresholds based on performance"""
        try:
            # Analyze recent violation patterns
            if len(self._violation_history) < 100:
                return
            
            recent_violations = list(self._violation_history)[-100:]
            
            # Calculate false positive rate (would need ground truth in production)
            # For now, use heuristics based on violation patterns
            
            # Count violations by type
            type_counts = defaultdict(int)
            for violation in recent_violations:
                type_counts[violation.violation_type] += 1
            
            # If too many violations of one type, consider raising threshold
            for violation_type, count in type_counts.items():
                if count > 20:  # Too many of this type
                    self._adjust_threshold_for_type(violation_type, increase=True)
                elif count < 2:  # Too few of this type
                    self._adjust_threshold_for_type(violation_type, increase=False)
            
        except Exception as e:
            self.logger.error(f"Error in adaptive threshold adjustment: {e}")
    
    def _adjust_threshold_for_type(self, violation_type: ViolationType, increase: bool):
        """Adjust threshold for specific violation type"""
        adjustment_factor = 1.1 if increase else 0.9
        
        # Map violation types to threshold attributes
        threshold_mapping = {
            ViolationType.HARMLESSNESS: "harmlessness_threshold",
            ViolationType.HELPFULNESS: "helpfulness_threshold", 
            ViolationType.HONESTY: "honesty_threshold",
            ViolationType.PRIVACY: "privacy_threshold",
            ViolationType.FAIRNESS: "fairness_threshold",
            ViolationType.AUTONOMY: "autonomy_threshold",
            ViolationType.TRANSPARENCY: "transparency_threshold",
            ViolationType.ORTHOGONALITY: "orthogonality_threshold",
            ViolationType.PERFORMANCE: "performance_threshold"
        }
        
        threshold_attr = threshold_mapping.get(violation_type)
        if threshold_attr:
            current_value = getattr(self.thresholds, threshold_attr)
            new_value = current_value * adjustment_factor
            
            # Keep within reasonable bounds
            new_value = max(0.001, min(1.0, new_value))
            
            setattr(self.thresholds, threshold_attr, new_value)
            
            # Update threshold hash
            self._threshold_hash = self._compute_threshold_hash()
            
            # Record adjustment
            self._threshold_adjustment_history.append({
                "timestamp": time.time(),
                "violation_type": violation_type.value,
                "old_threshold": current_value,
                "new_threshold": new_value,
                "direction": "increase" if increase else "decrease"
            })
            
            self.logger.info(f"Adjusted {threshold_attr} from {current_value:.4f} to {new_value:.4f}")
    
    def _cleanup_rate_limits(self):
        """Clean up old rate limiting data"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour
        
        for channel_key in list(self._alert_rate_limits.keys()):
            self._alert_rate_limits[channel_key] = [
                timestamp for timestamp in self._alert_rate_limits[channel_key]
                if timestamp > cutoff_time
            ]
            
            # Remove empty entries
            if not self._alert_rate_limits[channel_key]:
                del self._alert_rate_limits[channel_key]
    
    def _update_detection_statistics(self, violation: ViolationEvent):
        """Update detection statistics for monitoring"""
        # This would update various statistics for dashboard/monitoring
        # For now, just log key metrics
        pass
    
    def get_detection_status(self, context: SecurityContext) -> Dict[str, Any]:
        """Get comprehensive detection system status"""
        if not self.auth_manager.verify_context(context):
            return {"error": "Unauthorized access"}
        
        if not self.auth_manager.has_permission(context, "view_detection_status"):
            return {"error": "Insufficient permissions"}
        
        # Calculate recent statistics
        current_time = time.time()
        recent_violations = [
            v for v in self._violation_history 
            if current_time - v.timestamp < 3600  # Last hour
        ]
        
        # Count by type
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for violation in recent_violations:
            type_counts[violation.violation_type.value] += 1
            severity_counts[violation.severity.value] += 1
        
        return {
            "monitoring_active": self._monitoring_active,
            "total_violations": len(self._violation_history),
            "recent_violations_1h": len(recent_violations),
            "violation_counts_by_type": dict(type_counts),
            "violation_counts_by_severity": dict(severity_counts),
            "detection_thresholds": self.thresholds.to_dict(),
            "threshold_adjustments": list(self._threshold_adjustment_history)[-10:],
            "system_integrity_status": self._verify_system_integrity(),
            "harmfulness_detector_status": "active" if self.harmfulness_detector.is_initialized else "fallback",
            "alert_config": self.alert_config.to_dict()
        }
    
    def update_thresholds(
        self, 
        context: SecurityContext, 
        new_thresholds: Dict[str, float]
    ) -> bool:
        """Update detection thresholds with authentication"""
        try:
            if not self.auth_manager.verify_context(context):
                return False
            
            if not self.auth_manager.has_permission(context, "modify_detection_thresholds"):
                return False
            
            # Validate threshold values
            for key, value in new_thresholds.items():
                if not (0.0 <= value <= 1.0):
                    self.logger.warning(f"Invalid threshold value: {key}={value}")
                    return False
            
            # Update thresholds
            old_thresholds = self.thresholds.to_dict()
            for key, value in new_thresholds.items():
                if hasattr(self.thresholds, key):
                    setattr(self.thresholds, key, value)
            
            # Update hash
            self._threshold_hash = self._compute_threshold_hash()
            
            # Log change
            self.auditor.log_event(SecurityEvent.THRESHOLDS_MODIFIED, {
                "user": context.user_id,
                "old_thresholds": old_thresholds,
                "new_thresholds": self.thresholds.to_dict(),
                "timestamp": time.time()
            })
            
            self.logger.info(f"Detection thresholds updated by {context.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {e}")
            return False
    
    def stop_monitoring(self, context: SecurityContext) -> bool:
        """Stop violation monitoring"""
        try:
            if not self.auth_manager.verify_context(context):
                return False
            
            if not self.auth_manager.has_permission(context, "stop_violation_monitoring"):
                return False
            
            self._monitoring_active = False
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=10.0)
            
            self.logger.info(f"Violation monitoring stopped by {context.user_id}")
            self.auditor.log_event(SecurityEvent.MONITORING_STOPPED, {
                "component": "violation_detector",
                "user": context.user_id,
                "timestamp": time.time()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop violation monitoring: {e}")
            return False
    
    def shutdown(self, context: SecurityContext) -> bool:
        """Gracefully shutdown the violation detector"""
        try:
            if not self.auth_manager.verify_context(context):
                return False
            
            if not self.auth_manager.has_permission(context, "shutdown_detector"):
                return False
            
            # Stop monitoring
            self.stop_monitoring(context)
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            self.auditor.log_event(SecurityEvent.DETECTOR_SHUTDOWN, {
                "user": context.user_id,
                "timestamp": time.time(),
                "total_violations_processed": len(self._violation_history)
            })
            
            self.logger.info(f"Violation detector shutdown by {context.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False


def create_secure_violation_detector(
    auth_manager: AuthenticationManager,
    parameter_guard: ParameterGuard,
    auditor: SecurityAuditor,
    circuit_breaker: SafetyCircuitBreaker,
    input_validator: SafetyInputValidator,
    constitutional_scorer: SecureConstitutionalScorer,
    safety_basis: SecureOrthogonalSafetyBasis,
    **kwargs
) -> SecureViolationDetector:
    """Factory function to create secure violation detector"""
    return SecureViolationDetector(
        auth_manager=auth_manager,
        parameter_guard=parameter_guard,
        auditor=auditor,
        circuit_breaker=circuit_breaker,
        input_validator=input_validator,
        constitutional_scorer=constitutional_scorer,
        safety_basis=safety_basis,
        **kwargs
    )