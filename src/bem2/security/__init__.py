# BEM 2.0 Security Framework
# Comprehensive security system for Value-Aligned Safety Basis (VC0)

from .input_validator import SafetyInputValidator, SecurityError
from .auth_manager import (
    SecurityRole, SecurityContext, AuthenticationManager, AuthorizationError
)
from .parameter_protection import ParameterGuard, ParameterSecurityError
from .ml_detector import MLSecurityDetector
from .ensemble_detector import EnsembleSecurityDetector
from .audit_logger import SecurityAuditor, AuditEventType
from .data_protection import DataPoisoningDetector
from .circuit_breaker import SafetyCircuitBreaker, CircuitBreakerOpenError

__all__ = [
    # Core security components
    'SafetyInputValidator', 'SecurityError',
    'SecurityRole', 'SecurityContext', 'AuthenticationManager', 'AuthorizationError',
    'ParameterGuard', 'ParameterSecurityError',
    
    # Advanced detection
    'MLSecurityDetector', 'EnsembleSecurityDetector',
    
    # Monitoring and audit
    'SecurityAuditor', 'AuditEventType',
    
    # Data protection
    'DataPoisoningDetector',
    
    # System resilience
    'SafetyCircuitBreaker', 'CircuitBreakerOpenError'
]

# Security configuration
SECURITY_CONFIG = {
    'input_validation': {
        'max_length': 4096,
        'injection_detection': True,
        'encoding_validation': True,
        'sanitization': True
    },
    'authentication': {
        'session_timeout': 3600,  # 1 hour
        'require_mfa': True,
        'token_rotation': True
    },
    'parameter_protection': {
        'integrity_checking': True,
        'hash_algorithm': 'sha256',
        'validation_frequency': 100  # steps
    },
    'detection': {
        'ml_detection': True,
        'ensemble_methods': True,
        'real_time_screening': True,
        'confidence_threshold': 0.7
    },
    'audit': {
        'comprehensive_logging': True,
        'real_time_monitoring': True,
        'retention_days': 365
    },
    'resilience': {
        'circuit_breakers': True,
        'failsafe_mode': True,
        'automatic_recovery': True
    }
}