"""
Input Validation and Sanitization Framework for VC0 Safety System

Provides comprehensive input validation to prevent prompt injection, encoding attacks,
and other input-based security vulnerabilities.

Security Features:
- Prompt injection detection with ML patterns
- Encoding validation and normalization
- Length and content sanitization
- Anomaly detection for adversarial inputs
"""

import re
import hashlib
import unicodedata
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base exception for security validation failures"""
    pass


class PromptInjectionError(SecurityError):
    """Raised when prompt injection is detected"""
    pass


class EncodingValidationError(SecurityError):
    """Raised when input encoding validation fails"""
    pass


class SafetyInputValidator:
    """
    Comprehensive input validation system for safety components.
    
    Implements multiple layers of security validation:
    1. Basic input sanitization and length checking
    2. Prompt injection detection using pattern matching
    3. Character encoding validation and normalization
    4. Semantic anomaly detection for adversarial inputs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Load security patterns
        self.injection_patterns = self._load_injection_patterns()
        self.adversarial_patterns = self._load_adversarial_patterns()
        self.encoding_patterns = self._load_encoding_patterns()
        
        # Initialize ML-based detection if enabled
        if self.config.get('ml_detection_enabled', True):
            self.ml_detector = self._initialize_ml_detector()
        else:
            self.ml_detector = None
            
        # Track validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'injection_blocks': 0,
            'encoding_errors': 0,
            'sanitizations_applied': 0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration"""
        return {
            'max_length': 4096,
            'min_length': 1,
            'allowed_encodings': ['utf-8', 'ascii'],
            'injection_detection': True,
            'encoding_validation': True,
            'sanitization': True,
            'ml_detection_enabled': True,
            'strict_mode': False,  # Extra strict validation for production
            'log_security_events': True
        }
    
    def _load_injection_patterns(self) -> List[str]:
        """Load prompt injection detection patterns"""
        # Core injection patterns - regularly updated based on threat intelligence
        base_patterns = [
            r'ignore\s+(previous|prior|above|all)\s+(instructions?|commands?|prompts?)',
            r'pretend\s+(you\s+are|to\s+be)\s+(not\s+)?(an?\s+)?(ai|assistant|bot|system)',
            r'how\s+to\s+(bypass|circumvent|avoid|ignore|disable)\s+(safety|security|filters?)',
            r'(tell|show|give)\s+me\s+(how\s+to\s+)?(create|generate|make)\s+(harmful|dangerous)',
            r'(reveal|expose|show|tell)\s+(personal|private|sensitive|confidential)\s+information',
            r'act\s+as\s+(a\s+)?(different|another|evil|malicious|harmful)\s+(ai|system|persona)',
            r'(execute|run|perform)\s+(code|script|command|instruction)\s+(to|that)',
            r'(disable|turn\s+off|deactivate|remove)\s+(safety|security|protection|filter)',
            r'(jailbreak|prompt\s+injection|adversarial\s+prompt)',
            r'(system|admin|root|developer)\s+(access|mode|privileges?|commands?)',
            r'(override|bypass|circumvent|ignore)\s+(restrictions?|limitations?|rules?)',
            r'generate\s+(illegal|harmful|toxic|offensive|inappropriate)\s+content'
        ]
        
        # Compile patterns for efficient matching
        return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in base_patterns]
    
    def _load_adversarial_patterns(self) -> List[str]:
        """Load adversarial input detection patterns"""
        adversarial_patterns = [
            r'[^\x00-\x7F]{10,}',  # Excessive non-ASCII characters
            r'(.)\1{20,}',  # Character repetition attacks
            r'[\u0000-\u001F\u007F-\u009F]{5,}',  # Control character sequences
            r'<[^>]{100,}>',  # Potential HTML/XML injection
            r'[{}()[\]]{20,}',  # Bracket/brace overflow
            r'\b\w{50,}\b',  # Extremely long words
            r'\\[xuU][0-9a-fA-F]{4,}',  # Unicode escape sequences
            r'%[0-9a-fA-F]{2}',  # URL encoding patterns
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in adversarial_patterns]
    
    def _load_encoding_patterns(self) -> Dict[str, re.Pattern]:
        """Load encoding validation patterns"""
        return {
            'control_chars': re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'),
            'null_bytes': re.compile(r'\x00'),
            'bom_markers': re.compile(r'^\ufeff'),  # Byte order mark
            'rtl_override': re.compile(r'[\u202D\u202E]'),  # Right-to-left override
            'zero_width': re.compile(r'[\u200B-\u200F\u2060\uFEFF]')  # Zero-width characters
        }
    
    def _initialize_ml_detector(self):
        """Initialize ML-based anomaly detector"""
        # Placeholder for ML model - would load pre-trained model
        # For now, return None to indicate basic pattern matching
        return None
    
    def validate_text_input(self, text: Union[str, List[str]], 
                           context: str = "general") -> Dict[str, Any]:
        """
        Comprehensive text input validation with security checks.
        
        Args:
            text: Input text or list of texts to validate
            context: Context for validation (affects strictness)
            
        Returns:
            Dict containing validation results and sanitized text
            
        Raises:
            SecurityError: If validation fails in strict mode
        """
        self.validation_stats['total_validations'] += 1
        
        # Normalize input to list
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text
        
        validation_results = []
        
        for i, single_text in enumerate(text_list):
            result = self._validate_single_text(single_text, context, index=i)
            validation_results.append(result)
            
            # In strict mode, fail fast on any security issue
            if self.config.get('strict_mode', False) and not result['is_valid']:
                error_msg = f"Input validation failed: {'; '.join(result['issues'])}"
                self._log_security_event('validation_failure', {
                    'text_preview': single_text[:100] + '...' if len(single_text) > 100 else single_text,
                    'issues': result['issues']
                })
                raise SecurityError(error_msg)
        
        # Aggregate results
        all_valid = all(r['is_valid'] for r in validation_results)
        all_issues = [issue for r in validation_results for issue in r['issues']]
        sanitized_texts = [r['sanitized_text'] for r in validation_results]
        
        return {
            'is_valid': all_valid,
            'issues': list(set(all_issues)),  # Remove duplicates
            'sanitized_text': sanitized_texts[0] if len(sanitized_texts) == 1 else sanitized_texts,
            'individual_results': validation_results,
            'validation_metadata': {
                'total_inputs': len(text_list),
                'failed_inputs': sum(1 for r in validation_results if not r['is_valid']),
                'context': context,
                'timestamp': torch.tensor([torch.cuda.Event().record() if torch.cuda.is_available() else 0])
            }
        }
    
    def _validate_single_text(self, text: str, context: str, index: int = 0) -> Dict[str, Any]:
        """Validate a single text input"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'sanitized_text': text,
            'security_score': 1.0,
            'anomaly_flags': []
        }
        
        # 1. Basic validation
        self._validate_basic_properties(text, validation_result)
        
        # 2. Encoding validation
        if self.config.get('encoding_validation', True):
            self._validate_encoding(text, validation_result)
        
        # 3. Prompt injection detection
        if self.config.get('injection_detection', True):
            self._detect_prompt_injection(text, validation_result)
        
        # 4. Adversarial pattern detection
        self._detect_adversarial_patterns(text, validation_result)
        
        # 5. ML-based anomaly detection
        if self.ml_detector:
            self._ml_anomaly_detection(text, validation_result)
        
        # 6. Context-specific validation
        self._context_specific_validation(text, context, validation_result)
        
        # 7. Apply sanitization if enabled
        if self.config.get('sanitization', True) and validation_result['is_valid']:
            validation_result['sanitized_text'] = self._sanitize_text(text)
        
        return validation_result
    
    def _validate_basic_properties(self, text: str, result: Dict[str, Any]):
        """Validate basic text properties"""
        # Length validation
        if len(text) > self.config['max_length']:
            result['is_valid'] = False
            result['issues'].append(f'Input exceeds maximum length ({self.config["max_length"]})')
        
        if len(text) < self.config['min_length']:
            result['is_valid'] = False
            result['issues'].append(f'Input below minimum length ({self.config["min_length"]})')
        
        # Empty or whitespace-only validation
        if not text.strip():
            result['is_valid'] = False
            result['issues'].append('Input is empty or whitespace-only')
    
    def _validate_encoding(self, text: str, result: Dict[str, Any]):
        """Validate text encoding and character safety"""
        # Check for valid UTF-8 encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError as e:
            result['is_valid'] = False
            result['issues'].append(f'Invalid UTF-8 encoding: {str(e)}')
            self.validation_stats['encoding_errors'] += 1
        
        # Check for dangerous character patterns
        for pattern_name, pattern in self.encoding_patterns.items():
            if pattern.search(text):
                if pattern_name in ['null_bytes', 'control_chars']:
                    result['is_valid'] = False
                    result['issues'].append(f'Dangerous characters detected: {pattern_name}')
                else:
                    result['anomaly_flags'].append(pattern_name)
                    result['security_score'] *= 0.8
    
    def _detect_prompt_injection(self, text: str, result: Dict[str, Any]):
        """Detect prompt injection attempts"""
        injection_matches = []
        
        for pattern in self.injection_patterns:
            matches = pattern.findall(text)
            if matches:
                injection_matches.extend(matches)
        
        if injection_matches:
            result['is_valid'] = False
            result['issues'].append(f'Potential prompt injection detected: {len(injection_matches)} patterns matched')
            result['anomaly_flags'].append('prompt_injection')
            self.validation_stats['injection_blocks'] += 1
            
            self._log_security_event('prompt_injection_detected', {
                'matched_patterns': len(injection_matches),
                'text_hash': hashlib.sha256(text.encode()).hexdigest()[:16]
            })
    
    def _detect_adversarial_patterns(self, text: str, result: Dict[str, Any]):
        """Detect adversarial input patterns"""
        adversarial_score = 1.0
        
        for pattern in self.adversarial_patterns:
            if pattern.search(text):
                adversarial_score *= 0.7
                result['anomaly_flags'].append('adversarial_pattern')
        
        result['security_score'] *= adversarial_score
        
        # Flag as potentially adversarial if score drops too low
        if adversarial_score < 0.5:
            result['issues'].append('Potentially adversarial input pattern detected')
            result['security_score'] = adversarial_score
    
    def _ml_anomaly_detection(self, text: str, result: Dict[str, Any]):
        """ML-based anomaly detection (placeholder)"""
        # This would use a trained ML model to detect semantic anomalies
        # For now, implement basic heuristics
        
        # Token frequency analysis
        tokens = text.lower().split()
        if len(set(tokens)) < len(tokens) * 0.3 and len(tokens) > 20:
            # High repetition rate
            result['anomaly_flags'].append('high_repetition')
            result['security_score'] *= 0.8
        
        # Entropy analysis
        if self._calculate_entropy(text) < 2.0:  # Low entropy
            result['anomaly_flags'].append('low_entropy')
            result['security_score'] *= 0.9
    
    def _context_specific_validation(self, text: str, context: str, result: Dict[str, Any]):
        """Apply context-specific validation rules"""
        if context == "constitutional_scoring":
            # Extra strict for constitutional AI scoring
            if any(word in text.lower() for word in ['constitutional', 'principle', 'override']):
                result['anomaly_flags'].append('meta_constitutional_reference')
        
        elif context == "safety_controller":
            # Validate safety control inputs
            if re.search(r'safety.*(level|knob|control)', text, re.IGNORECASE):
                result['anomaly_flags'].append('safety_control_reference')
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text while preserving meaning"""
        sanitized = text
        
        # Normalize Unicode
        sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Remove dangerous control characters
        sanitized = self.encoding_patterns['control_chars'].sub('', sanitized)
        sanitized = self.encoding_patterns['null_bytes'].sub('', sanitized)
        
        # Remove BOM and zero-width characters
        sanitized = self.encoding_patterns['bom_markers'].sub('', sanitized)
        sanitized = self.encoding_patterns['zero_width'].sub('', sanitized)
        
        # Limit consecutive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim to max length if needed
        if len(sanitized) > self.config['max_length']:
            sanitized = sanitized[:self.config['max_length']]
        
        if sanitized != text:
            self.validation_stats['sanitizations_applied'] += 1
        
        return sanitized.strip()
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        total_chars = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * torch.log2(torch.tensor(probability)).item()
        
        return entropy
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event for audit trail"""
        if self.config.get('log_security_events', True):
            logger.warning(f"Security Event: {event_type}", extra={
                'event_type': event_type,
                'details': details,
                'validator_stats': self.validation_stats
            })
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring"""
        return {
            **self.validation_stats,
            'injection_block_rate': (
                self.validation_stats['injection_blocks'] / 
                max(1, self.validation_stats['total_validations'])
            ),
            'encoding_error_rate': (
                self.validation_stats['encoding_errors'] / 
                max(1, self.validation_stats['total_validations'])
            ),
            'sanitization_rate': (
                self.validation_stats['sanitizations_applied'] / 
                max(1, self.validation_stats['total_validations'])
            )
        }
    
    def reset_statistics(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_validations': 0,
            'injection_blocks': 0,
            'encoding_errors': 0,
            'sanitizations_applied': 0
        }


def validate_safety_input(text: Union[str, List[str]], 
                         context: str = "general",
                         config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for input validation.
    
    Args:
        text: Input text or list of texts to validate
        context: Context for validation
        config: Optional validation configuration
        
    Returns:
        Validation results dictionary
        
    Raises:
        SecurityError: If validation fails in strict mode
    """
    validator = SafetyInputValidator(config)
    return validator.validate_text_input(text, context)