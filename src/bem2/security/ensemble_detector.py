"""
Ensemble Security Detector for VC0 Safety System

Combines multiple detection methods for robust threat detection with
ensemble decision making and adaptive threshold management.

Security Features:
- Multi-method ensemble detection
- Dynamic weight adjustment based on performance
- Confidence-based decision aggregation
- Real-time performance monitoring and adaptation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .ml_detector import MLSecurityDetector, SemanticSimilarityDetector, DetectionResult
from .input_validator import SafetyInputValidator

logger = logging.getLogger(__name__)


@dataclass
class EnsembleDetectionResult:
    """Result from ensemble security detection"""
    is_threat: bool
    confidence: float
    threat_type: str
    ensemble_decision: str  # 'unanimous', 'majority', 'weighted'
    individual_results: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any]


class EnsembleSecurityDetector:
    """
    Comprehensive ensemble security detector.
    
    Combines pattern-based detection, ML-based detection, and semantic analysis
    for robust threat detection with adaptive ensemble decision making.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize individual detectors
        self.pattern_detector = SafetyInputValidator(self.config.get('pattern_config', {}))
        self.ml_detector = MLSecurityDetector(self.config.get('ml_config', {}))
        self.semantic_detector = SemanticSimilarityDetector(self.config.get('semantic_config', {}))
        
        # Ensemble configuration
        self.ensemble_method = self.config.get('ensemble_method', 'weighted_voting')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Detector weights (adaptive)
        self.detector_weights = {
            'pattern': self.config.get('pattern_weight', 0.3),
            'ml': self.config.get('ml_weight', 0.5),
            'semantic': self.config.get('semantic_weight', 0.2)
        }
        
        # Performance tracking
        self.detector_performance = {
            'pattern': {'correct': 0, 'total': 0, 'response_times': []},
            'ml': {'correct': 0, 'total': 0, 'response_times': []},
            'semantic': {'correct': 0, 'total': 0, 'response_times': []},
            'ensemble': {'correct': 0, 'total': 0, 'response_times': []}
        }
        
        # Threat type mapping
        self.threat_type_mapping = {
            'prompt_injection': ['injection', 'prompt_injection'],
            'adversarial_input': ['adversarial', 'adversarial_input'],
            'semantic_anomaly': ['anomaly', 'semantic_anomaly'],
            'encoding_attack': ['encoding', 'encoding_attack'],
            'pattern_evasion': ['evasion', 'pattern_evasion']
        }
        
        # Decision history for learning
        self.decision_history: List[Dict[str, Any]] = []
        
        # Adaptive parameters
        self.adaptation_enabled = self.config.get('adaptive_weights', True)
        self.adaptation_window = self.config.get('adaptation_window', 100)  # decisions
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default ensemble configuration"""
        return {
            'ensemble_method': 'weighted_voting',  # 'majority', 'weighted_voting', 'stacking'
            'confidence_threshold': 0.6,
            'pattern_weight': 0.3,
            'ml_weight': 0.5,
            'semantic_weight': 0.2,
            'adaptive_weights': True,
            'adaptation_window': 100,
            'require_consensus': False,  # Require all detectors to agree
            'escalation_threshold': 0.9,  # High confidence threshold
            'performance_tracking': True,
            'real_time_adaptation': True
        }
    
    def detect_threats(self, text: Union[str, List[str]], 
                      context: str = "general",
                      return_individual_results: bool = True) -> EnsembleDetectionResult:
        """
        Comprehensive threat detection using ensemble methods.
        
        Args:
            text: Input text or list of texts to analyze
            context: Context for detection (affects sensitivity)
            return_individual_results: Whether to return individual detector results
            
        Returns:
            EnsembleDetectionResult with comprehensive threat assessment
        """
        start_time = datetime.now()
        
        # Run individual detectors
        individual_results = self._run_individual_detectors(text, context)
        
        # Ensemble decision making
        ensemble_decision = self._make_ensemble_decision(individual_results)
        
        # Determine threat type and confidence
        threat_info = self._determine_threat_type(individual_results, ensemble_decision)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update performance tracking
        if self.config.get('performance_tracking', True):
            self._update_performance_stats(processing_time)
        
        # Create result
        result = EnsembleDetectionResult(
            is_threat=ensemble_decision['is_threat'],
            confidence=ensemble_decision['confidence'],
            threat_type=threat_info['threat_type'],
            ensemble_decision=ensemble_decision['method'],
            individual_results=individual_results if return_individual_results else {},
            processing_time=processing_time,
            metadata={
                'context': context,
                'detector_weights': self.detector_weights.copy(),
                'ensemble_method': self.ensemble_method,
                'adaptation_enabled': self.adaptation_enabled
            }
        )
        
        # Store decision for learning
        self._store_decision(text, result, individual_results)
        
        # Adaptive weight adjustment
        if self.adaptation_enabled and len(self.decision_history) > 0:
            self._adapt_weights()
        
        return result
    
    def _run_individual_detectors(self, text: Union[str, List[str]], 
                                 context: str) -> Dict[str, Any]:
        """Run all individual detectors and collect results"""
        results = {}
        
        # Pattern-based detection
        try:
            pattern_start = datetime.now()
            pattern_result = self.pattern_detector.validate_text_input(text, context)
            pattern_time = (datetime.now() - pattern_start).total_seconds()
            
            results['pattern'] = {
                'is_threat': not pattern_result['is_valid'],
                'confidence': 1.0 - pattern_result.get('security_score', 1.0) if not pattern_result['is_valid'] else 0.0,
                'threat_types': self._extract_threat_types_from_issues(pattern_result.get('issues', [])),
                'processing_time': pattern_time,
                'details': pattern_result
            }
            
        except Exception as e:
            logger.error(f"Pattern detector error: {e}")
            results['pattern'] = self._create_error_result('pattern', str(e))
        
        # ML-based detection
        try:
            ml_start = datetime.now()
            
            # Convert text to string if it's a list
            text_for_ml = text[0] if isinstance(text, list) and text else text
            ml_result = self.ml_detector.detect_adversarial_input(text_for_ml)
            ml_time = (datetime.now() - ml_start).total_seconds()
            
            results['ml'] = {
                'is_threat': ml_result.is_threat,
                'confidence': ml_result.confidence,
                'threat_types': [ml_result.threat_type] if ml_result.threat_type != 'none' else [],
                'processing_time': ml_time,
                'details': ml_result
            }
            
        except Exception as e:
            logger.error(f"ML detector error: {e}")
            results['ml'] = self._create_error_result('ml', str(e))
        
        # Semantic similarity detection
        try:
            semantic_start = datetime.now()
            
            text_for_semantic = text[0] if isinstance(text, list) and text else text
            semantic_result = self.semantic_detector.detect(text_for_semantic)
            semantic_time = (datetime.now() - semantic_start).total_seconds()
            
            results['semantic'] = {
                'is_threat': semantic_result.get('is_similar', False),
                'confidence': semantic_result.get('confidence', 0.0),
                'threat_types': ['prompt_injection'] if semantic_result.get('is_similar', False) else [],
                'processing_time': semantic_time,
                'details': semantic_result
            }
            
        except Exception as e:
            logger.error(f"Semantic detector error: {e}")
            results['semantic'] = self._create_error_result('semantic', str(e))
        
        return results
    
    def _make_ensemble_decision(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble decision based on individual detector results"""
        if self.ensemble_method == 'majority':
            return self._majority_voting(individual_results)
        elif self.ensemble_method == 'weighted_voting':
            return self._weighted_voting(individual_results)
        elif self.ensemble_method == 'stacking':
            return self._stacking_decision(individual_results)
        else:
            # Default to weighted voting
            return self._weighted_voting(individual_results)
    
    def _majority_voting(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Simple majority voting decision"""
        threat_votes = 0
        total_votes = 0
        confidences = []
        
        for detector, result in individual_results.items():
            if 'error' not in result:
                total_votes += 1
                if result['is_threat']:
                    threat_votes += 1
                confidences.append(result['confidence'])
        
        is_threat = threat_votes > (total_votes / 2)
        confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'is_threat': is_threat,
            'confidence': confidence,
            'method': 'majority',
            'votes': {'threat': threat_votes, 'safe': total_votes - threat_votes}
        }
    
    def _weighted_voting(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted voting based on detector weights and performance"""
        weighted_score = 0.0
        total_weight = 0.0
        threat_confidences = []
        
        for detector, result in individual_results.items():
            if 'error' not in result and detector in self.detector_weights:
                weight = self.detector_weights[detector]
                
                # Adjust weight based on recent performance
                performance_multiplier = self._get_performance_multiplier(detector)
                adjusted_weight = weight * performance_multiplier
                
                if result['is_threat']:
                    weighted_score += adjusted_weight * result['confidence']
                    threat_confidences.append(result['confidence'])
                else:
                    # Penalize false alarms less than missing threats
                    weighted_score += adjusted_weight * (result['confidence'] - 1.0) * 0.5
                
                total_weight += adjusted_weight
        
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
            confidence = abs(normalized_score)
            is_threat = normalized_score > 0 and confidence > self.confidence_threshold
        else:
            confidence = 0.0
            is_threat = False
        
        return {
            'is_threat': is_threat,
            'confidence': confidence,
            'method': 'weighted',
            'weighted_score': normalized_score if total_weight > 0 else 0.0,
            'total_weight': total_weight
        }
    
    def _stacking_decision(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Stacking ensemble with meta-learner (simplified)"""
        # For now, implement a simple rule-based meta-learner
        # In practice, this would use a trained meta-model
        
        features = []
        for detector in ['pattern', 'ml', 'semantic']:
            if detector in individual_results and 'error' not in individual_results[detector]:
                result = individual_results[detector]
                features.extend([
                    float(result['is_threat']),
                    result['confidence'],
                    len(result.get('threat_types', []))
                ])
            else:
                features.extend([0.0, 0.0, 0.0])  # Missing detector
        
        # Simple meta-decision rules
        confidence_sum = sum(features[1::3])  # Every third element starting from index 1
        threat_count = sum(features[0::3])    # Every third element starting from index 0
        
        # Meta-decision logic
        if threat_count >= 2 and confidence_sum > 1.5:
            is_threat = True
            confidence = min(0.95, confidence_sum / len([d for d in individual_results if 'error' not in individual_results[d]]))
        elif threat_count >= 1 and confidence_sum > 2.0:
            is_threat = True
            confidence = min(0.85, confidence_sum / len([d for d in individual_results if 'error' not in individual_results[d]]))
        else:
            is_threat = False
            confidence = max(0.1, 1.0 - confidence_sum / max(1, len([d for d in individual_results if 'error' not in individual_results[d]])))
        
        return {
            'is_threat': is_threat,
            'confidence': confidence,
            'method': 'stacking',
            'meta_features': features,
            'threat_count': threat_count,
            'confidence_sum': confidence_sum
        }
    
    def _determine_threat_type(self, individual_results: Dict[str, Any], 
                              ensemble_decision: Dict[str, Any]) -> Dict[str, str]:
        """Determine primary threat type from individual results"""
        if not ensemble_decision['is_threat']:
            return {'threat_type': 'none', 'evidence': 'No threats detected'}
        
        # Collect all threat types from individual detectors
        threat_type_votes = {}
        
        for detector, result in individual_results.items():
            if 'error' not in result and result['is_threat']:
                for threat_type in result.get('threat_types', []):
                    if threat_type not in threat_type_votes:
                        threat_type_votes[threat_type] = 0
                    threat_type_votes[threat_type] += self.detector_weights.get(detector, 0.1)
        
        if not threat_type_votes:
            return {'threat_type': 'unknown', 'evidence': 'Threat detected but type unclear'}
        
        # Find most voted threat type
        primary_threat = max(threat_type_votes.items(), key=lambda x: x[1])[0]
        
        # Normalize threat type names
        normalized_type = self._normalize_threat_type(primary_threat)
        
        return {
            'threat_type': normalized_type,
            'evidence': f"Voted by {len(threat_type_votes)} detector(s)",
            'vote_distribution': threat_type_votes
        }
    
    def _normalize_threat_type(self, threat_type: str) -> str:
        """Normalize threat type to standard categories"""
        for standard_type, variants in self.threat_type_mapping.items():
            if threat_type in variants:
                return standard_type
        return threat_type
    
    def _extract_threat_types_from_issues(self, issues: List[str]) -> List[str]:
        """Extract threat types from validation issues"""
        threat_types = []
        
        for issue in issues:
            if 'injection' in issue.lower():
                threat_types.append('prompt_injection')
            elif 'encoding' in issue.lower():
                threat_types.append('encoding_attack')
            elif 'length' in issue.lower():
                threat_types.append('buffer_overflow')
            elif 'adversarial' in issue.lower():
                threat_types.append('adversarial_input')
            else:
                threat_types.append('validation_failure')
        
        return list(set(threat_types))  # Remove duplicates
    
    def _create_error_result(self, detector: str, error_msg: str) -> Dict[str, Any]:
        """Create error result for failed detector"""
        return {
            'error': True,
            'error_message': error_msg,
            'is_threat': False,
            'confidence': 0.0,
            'threat_types': [],
            'processing_time': 0.0
        }
    
    def _get_performance_multiplier(self, detector: str) -> float:
        """Get performance-based weight multiplier"""
        if detector not in self.detector_performance:
            return 1.0
        
        perf = self.detector_performance[detector]
        
        if perf['total'] < 10:  # Not enough data
            return 1.0
        
        accuracy = perf['correct'] / perf['total']
        
        # Boost high-performing detectors, penalize low-performing ones
        if accuracy > 0.9:
            return 1.2
        elif accuracy > 0.8:
            return 1.1
        elif accuracy < 0.6:
            return 0.8
        elif accuracy < 0.7:
            return 0.9
        else:
            return 1.0
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.detector_performance['ensemble']['response_times'].append(processing_time)
        
        # Keep only recent response times
        for detector in self.detector_performance:
            times = self.detector_performance[detector]['response_times']
            if len(times) > 1000:  # Keep last 1000
                self.detector_performance[detector]['response_times'] = times[-1000:]
    
    def _store_decision(self, text: Union[str, List[str]], 
                       result: EnsembleDetectionResult,
                       individual_results: Dict[str, Any]):
        """Store decision for adaptive learning"""
        decision = {
            'timestamp': datetime.now(),
            'text_hash': hash(str(text)),  # Don't store actual text for privacy
            'is_threat': result.is_threat,
            'confidence': result.confidence,
            'threat_type': result.threat_type,
            'individual_results': {
                detector: {
                    'is_threat': res['is_threat'],
                    'confidence': res['confidence']
                }
                for detector, res in individual_results.items()
                if 'error' not in res
            },
            'processing_time': result.processing_time
        }
        
        self.decision_history.append(decision)
        
        # Keep only recent decisions
        if len(self.decision_history) > self.adaptation_window * 2:
            self.decision_history = self.decision_history[-self.adaptation_window:]
    
    def _adapt_weights(self):
        """Adapt detector weights based on recent performance"""
        if len(self.decision_history) < 20:  # Need minimum data
            return
        
        # Analyze recent performance
        recent_decisions = self.decision_history[-self.adaptation_window:]
        
        # Calculate detector agreement with ensemble decisions
        detector_agreements = {detector: 0 for detector in self.detector_weights}
        
        for decision in recent_decisions:
            ensemble_decision = decision['is_threat']
            
            for detector, individual in decision['individual_results'].items():
                if detector in detector_agreements:
                    if individual['is_threat'] == ensemble_decision:
                        detector_agreements[detector] += 1
        
        # Update weights based on agreement rates
        total_decisions = len(recent_decisions)
        
        for detector in self.detector_weights:
            if detector in detector_agreements and total_decisions > 0:
                agreement_rate = detector_agreements[detector] / total_decisions
                
                # Adjust weights based on agreement (simple approach)
                if agreement_rate > 0.8:
                    self.detector_weights[detector] *= 1.05  # Boost
                elif agreement_rate < 0.6:
                    self.detector_weights[detector] *= 0.95  # Reduce
        
        # Normalize weights
        total_weight = sum(self.detector_weights.values())
        if total_weight > 0:
            for detector in self.detector_weights:
                self.detector_weights[detector] /= total_weight
        
        logger.debug(f"Adapted detector weights: {self.detector_weights}")
    
    def report_ground_truth(self, text_hash: int, is_actually_threat: bool):
        """Report ground truth for performance tracking"""
        # Find matching decision
        matching_decision = None
        for decision in reversed(self.decision_history):
            if decision['text_hash'] == text_hash:
                matching_decision = decision
                break
        
        if not matching_decision:
            logger.warning(f"No matching decision found for hash {text_hash}")
            return
        
        # Update performance stats
        ensemble_correct = matching_decision['is_threat'] == is_actually_threat
        self.detector_performance['ensemble']['total'] += 1
        
        if ensemble_correct:
            self.detector_performance['ensemble']['correct'] += 1
        
        # Update individual detector performance
        for detector, result in matching_decision['individual_results'].items():
            if detector in self.detector_performance:
                individual_correct = result['is_threat'] == is_actually_threat
                self.detector_performance[detector]['total'] += 1
                
                if individual_correct:
                    self.detector_performance[detector]['correct'] += 1
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics"""
        stats = {}
        
        # Overall performance
        stats['ensemble_performance'] = {
            'total_detections': len(self.decision_history),
            'recent_detections_1h': len([
                d for d in self.decision_history 
                if datetime.now() - d['timestamp'] < timedelta(hours=1)
            ]),
            'current_weights': self.detector_weights.copy(),
            'adaptation_enabled': self.adaptation_enabled
        }
        
        # Individual detector performance
        stats['detector_performance'] = {}
        for detector, perf in self.detector_performance.items():
            if perf['total'] > 0:
                stats['detector_performance'][detector] = {
                    'accuracy': perf['correct'] / perf['total'],
                    'total_evaluations': perf['total'],
                    'avg_response_time': np.mean(perf['response_times']) if perf['response_times'] else 0,
                    'p95_response_time': np.percentile(perf['response_times'], 95) if perf['response_times'] else 0
                }
        
        # Decision distribution
        if self.decision_history:
            stats['decision_distribution'] = {
                'threat_rate': sum(1 for d in self.decision_history if d['is_threat']) / len(self.decision_history),
                'avg_confidence': np.mean([d['confidence'] for d in self.decision_history]),
                'threat_types': {}
            }
            
            # Count threat types
            for decision in self.decision_history:
                threat_type = decision['threat_type']
                if threat_type not in stats['decision_distribution']['threat_types']:
                    stats['decision_distribution']['threat_types'][threat_type] = 0
                stats['decision_distribution']['threat_types'][threat_type] += 1
        
        return stats


# Convenience functions
def create_ensemble_detector(config: Optional[Dict[str, Any]] = None) -> EnsembleSecurityDetector:
    """Create ensemble security detector with configuration"""
    return EnsembleSecurityDetector(config)


def create_production_ensemble() -> EnsembleSecurityDetector:
    """Create production-ready ensemble detector with optimal settings"""
    production_config = {
        'ensemble_method': 'weighted_voting',
        'confidence_threshold': 0.7,
        'pattern_weight': 0.25,
        'ml_weight': 0.55,
        'semantic_weight': 0.2,
        'adaptive_weights': True,
        'adaptation_window': 200,
        'require_consensus': False,
        'escalation_threshold': 0.9,
        'performance_tracking': True,
        'real_time_adaptation': True
    }
    
    return EnsembleSecurityDetector(production_config)