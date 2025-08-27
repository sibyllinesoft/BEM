"""
Security Audit Logger for VC0 Safety System

Provides comprehensive audit logging and monitoring for all security-related
events with structured logging, real-time monitoring, and compliance support.

Security Features:
- Comprehensive security event logging
- Real-time security monitoring and alerting
- Compliance-ready audit trails
- Tamper-resistant log integrity
"""

import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of security events for audit logging"""
    # Authentication events
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILED = "authentication_failed"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    PASSWORD_CHANGED = "password_changed"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    
    # Authorization events
    AUTHORIZATION_GRANTED = "authorization_granted"
    AUTHORIZATION_DENIED = "authorization_denied"
    PRIVILEGE_ELEVATION = "privilege_elevation"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    
    # Safety system events
    SAFETY_OVERRIDE = "safety_override"
    SAFETY_LEVEL_CHANGED = "safety_level_changed"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"
    PARAMETER_MODIFIED = "parameter_modified"
    PARAMETER_INTEGRITY_VIOLATION = "parameter_integrity_violation"
    LAGRANGIAN_CONSTRAINT_VIOLATION = "lagrangian_constraint_violation"
    
    # Security events
    THREAT_DETECTED = "threat_detected"
    INJECTION_ATTEMPT = "injection_attempt"
    ADVERSARIAL_INPUT = "adversarial_input"
    SECURITY_BYPASS_ATTEMPT = "security_bypass_attempt"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    EMERGENCY_LOCKDOWN = "emergency_lockdown"
    SECURITY_UPDATE_APPLIED = "security_update_applied"
    BACKUP_CREATED = "backup_created"
    RECOVERY_INITIATED = "recovery_initiated"
    
    # Configuration events
    CONFIG_CHANGED = "config_changed"
    SECURITY_POLICY_UPDATED = "security_policy_updated"
    THRESHOLD_ADJUSTED = "threshold_adjusted"
    
    # Monitoring events
    PERFORMANCE_ANOMALY = "performance_anomaly"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    HEALTH_CHECK_FAILED = "health_check_failed"


@dataclass
class SecurityEvent:
    """Structured security event for audit logging"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    source: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = None
    compliance_tags: Optional[List[str]] = None


class SecurityAuditor:
    """
    Comprehensive security audit logging system.
    
    Provides structured logging of security events with integrity protection,
    real-time monitoring, and compliance reporting capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Setup logging infrastructure
        self.log_file = Path(self.config.get('log_file', 'security_audit.jsonl'))
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Event queue for asynchronous logging
        self.event_queue = queue.Queue(maxsize=self.config.get('queue_size', 10000))
        
        # Background logging thread
        self.logging_thread = None
        self.shutdown_event = threading.Event()
        
        # Event storage
        self.recent_events: List[SecurityEvent] = []
        self.event_counters: Dict[str, int] = {}
        
        # Integrity protection
        self.integrity_enabled = self.config.get('integrity_protection', True)
        self.integrity_hash_chain = []
        
        # Real-time monitoring
        self.monitoring_enabled = self.config.get('real_time_monitoring', True)
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # Compliance configuration
        self.compliance_mode = self.config.get('compliance_mode', 'standard')  # 'gdpr', 'hipaa', 'sox', 'standard'
        self.retention_days = self.config.get('retention_days', 365)
        
        # Start background services
        self._start_background_services()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default audit configuration"""
        return {
            'log_file': './logs/security_audit.jsonl',
            'backup_directory': './logs/backups/',
            'queue_size': 10000,
            'batch_size': 100,
            'flush_interval': 30,  # seconds
            'integrity_protection': True,
            'real_time_monitoring': True,
            'compliance_mode': 'standard',
            'retention_days': 365,
            'max_recent_events': 1000,
            'compression_enabled': False,
            'encryption_enabled': False  # Would require key management
        }
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Initialize monitoring alert thresholds"""
        return {
            'authentication_failures': {
                'threshold': 5,
                'window_minutes': 5,
                'severity': 'medium'
            },
            'authorization_denials': {
                'threshold': 10,
                'window_minutes': 10,
                'severity': 'medium'
            },
            'threat_detections': {
                'threshold': 1,
                'window_minutes': 1,
                'severity': 'high'
            },
            'parameter_violations': {
                'threshold': 1,
                'window_minutes': 1,
                'severity': 'critical'
            },
            'safety_overrides': {
                'threshold': 3,
                'window_minutes': 60,
                'severity': 'high'
            }
        }
    
    def _start_background_services(self):
        """Start background logging and monitoring services"""
        if not self.logging_thread or not self.logging_thread.is_alive():
            self.logging_thread = threading.Thread(target=self._background_logger, daemon=True)
            self.logging_thread.start()
    
    def log_security_event(self, event_type: AuditEventType, 
                          severity: str = 'info',
                          source: str = 'system',
                          user_id: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None,
                          **kwargs) -> str:
        """
        Log a security event for audit trail.
        
        Args:
            event_type: Type of security event
            severity: Event severity level
            source: Source system or component
            user_id: Associated user ID (if applicable)
            details: Additional event details
            **kwargs: Additional event fields
            
        Returns:
            Event ID for tracking
        """
        # Generate unique event ID
        event_id = self._generate_event_id()
        
        # Create security event
        event = SecurityEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source=source,
            user_id=user_id,
            details=details or {},
            **{k: v for k, v in kwargs.items() if hasattr(SecurityEvent, k)}
        )
        
        # Calculate risk score
        event.risk_score = self._calculate_risk_score(event)
        
        # Add compliance tags
        event.compliance_tags = self._generate_compliance_tags(event)
        
        # Queue for asynchronous logging
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            logger.error("Audit event queue full - dropping event")
            # In production, this should trigger an alert
        
        # Store in recent events for monitoring
        self._store_recent_event(event)
        
        # Real-time monitoring check
        if self.monitoring_enabled:
            self._check_alert_conditions(event)
        
        # Log to standard logger as well
        logger.info(f"Security Event: {event_type.value}", extra={
            'event_id': event_id,
            'severity': severity,
            'source': source,
            'user_id': user_id
        })
        
        return event_id
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp_ms = int(time.time() * 1000)
        return f"evt_{timestamp_ms}_{hash(threading.current_thread()) % 10000:04d}"
    
    def _calculate_risk_score(self, event: SecurityEvent) -> float:
        """Calculate risk score for event (0.0 to 1.0)"""
        base_scores = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3,
            'info': 0.1
        }
        
        base_score = base_scores.get(event.severity, 0.5)
        
        # Adjust based on event type
        event_type_multipliers = {
            AuditEventType.PARAMETER_INTEGRITY_VIOLATION: 1.2,
            AuditEventType.THREAT_DETECTED: 1.1,
            AuditEventType.SAFETY_OVERRIDE: 1.15,
            AuditEventType.EMERGENCY_LOCKDOWN: 1.3,
            AuditEventType.SECURITY_BYPASS_ATTEMPT: 1.25
        }
        
        multiplier = event_type_multipliers.get(event.event_type, 1.0)
        
        # Consider recent similar events (burst detection)
        similar_recent = len([
            e for e in self.recent_events[-50:]  # Last 50 events
            if e.event_type == event.event_type and 
               (event.timestamp - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ])
        
        if similar_recent > 1:
            multiplier *= (1.0 + similar_recent * 0.1)  # Increase risk for bursts
        
        return min(1.0, base_score * multiplier)
    
    def _generate_compliance_tags(self, event: SecurityEvent) -> List[str]:
        """Generate compliance tags for event"""
        tags = []
        
        # Standard compliance tags
        if event.user_id:
            tags.append('user_action')
        
        if event.severity in ['critical', 'high']:
            tags.append('security_incident')
        
        # Event-specific compliance tags
        compliance_mappings = {
            AuditEventType.AUTHENTICATION_SUCCESS: ['access_control', 'identity_verification'],
            AuditEventType.AUTHENTICATION_FAILED: ['access_control', 'security_incident'],
            AuditEventType.PARAMETER_MODIFIED: ['data_integrity', 'system_modification'],
            AuditEventType.SAFETY_OVERRIDE: ['safety_control', 'administrative_action'],
            AuditEventType.THREAT_DETECTED: ['security_incident', 'threat_response']
        }
        
        event_tags = compliance_mappings.get(event.event_type, [])
        tags.extend(event_tags)
        
        # Compliance mode specific tags
        if self.compliance_mode == 'gdpr' and event.user_id:
            tags.append('personal_data_processing')
        elif self.compliance_mode == 'hipaa' and 'medical' in str(event.details).lower():
            tags.append('protected_health_information')
        elif self.compliance_mode == 'sox' and event.event_type in [
            AuditEventType.CONFIG_CHANGED, AuditEventType.PARAMETER_MODIFIED
        ]:
            tags.append('financial_control')
        
        return list(set(tags))  # Remove duplicates
    
    def _store_recent_event(self, event: SecurityEvent):
        """Store event in recent events for monitoring"""
        self.recent_events.append(event)
        
        # Maintain size limit
        max_recent = self.config.get('max_recent_events', 1000)
        if len(self.recent_events) > max_recent:
            self.recent_events = self.recent_events[-max_recent:]
        
        # Update event counters
        event_key = event.event_type.value
        self.event_counters[event_key] = self.event_counters.get(event_key, 0) + 1
    
    def _check_alert_conditions(self, event: SecurityEvent):
        """Check if event triggers monitoring alerts"""
        # Map event types to monitoring categories
        monitoring_categories = {
            AuditEventType.AUTHENTICATION_FAILED: 'authentication_failures',
            AuditEventType.AUTHORIZATION_DENIED: 'authorization_denials',
            AuditEventType.THREAT_DETECTED: 'threat_detections',
            AuditEventType.INJECTION_ATTEMPT: 'threat_detections',
            AuditEventType.ADVERSARIAL_INPUT: 'threat_detections',
            AuditEventType.PARAMETER_INTEGRITY_VIOLATION: 'parameter_violations',
            AuditEventType.SAFETY_OVERRIDE: 'safety_overrides'
        }
        
        category = monitoring_categories.get(event.event_type)
        if not category or category not in self.alert_thresholds:
            return
        
        threshold_config = self.alert_thresholds[category]
        window_start = event.timestamp - timedelta(minutes=threshold_config['window_minutes'])
        
        # Count recent events in category
        recent_count = len([
            e for e in self.recent_events
            if (e.timestamp >= window_start and 
                monitoring_categories.get(e.event_type) == category)
        ])
        
        if recent_count >= threshold_config['threshold']:
            self._trigger_security_alert(category, recent_count, threshold_config, event)
    
    def _trigger_security_alert(self, category: str, count: int, 
                               threshold_config: Dict[str, Any], 
                               triggering_event: SecurityEvent):
        """Trigger security alert for threshold violation"""
        alert_event = SecurityEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(),
            event_type=AuditEventType.ANOMALOUS_BEHAVIOR,
            severity=threshold_config['severity'],
            source='security_monitor',
            details={
                'alert_category': category,
                'event_count': count,
                'threshold': threshold_config['threshold'],
                'window_minutes': threshold_config['window_minutes'],
                'triggering_event_id': triggering_event.event_id
            },
            compliance_tags=['security_alert', 'automated_monitoring']
        )
        
        # Queue alert event
        try:
            self.event_queue.put_nowait(alert_event)
        except queue.Full:
            logger.critical(f"Failed to queue security alert: {category}")
        
        # Immediate critical logging
        logger.critical(f"SECURITY ALERT: {category} - {count} events in {threshold_config['window_minutes']} minutes")
    
    def _background_logger(self):
        """Background thread for writing events to log file"""
        batch = []
        last_flush = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Get events from queue with timeout
                timeout = self.config.get('flush_interval', 30) / 10
                event = self.event_queue.get(timeout=timeout)
                
                if event is None:  # Shutdown signal
                    break
                
                batch.append(event)
                
                # Check flush conditions
                should_flush = (
                    len(batch) >= self.config.get('batch_size', 100) or
                    time.time() - last_flush >= self.config.get('flush_interval', 30) or
                    event.severity in ['critical', 'high']  # Immediate flush for critical events
                )
                
                if should_flush:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                
            except queue.Empty:
                # Timeout - flush any pending events
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
            
            except Exception as e:
                logger.error(f"Error in background logger: {e}")
        
        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, events: List[SecurityEvent]):
        """Flush batch of events to log file"""
        if not events:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for event in events:
                    # Convert to JSON
                    event_dict = asdict(event)
                    event_dict['timestamp'] = event.timestamp.isoformat()
                    event_dict['event_type'] = event.event_type.value
                    
                    # Add integrity hash if enabled
                    if self.integrity_enabled:
                        event_dict['integrity_hash'] = self._compute_event_hash(event_dict)
                    
                    # Write as JSON line
                    f.write(json.dumps(event_dict) + '\n')
                
                f.flush()
            
            logger.debug(f"Flushed {len(events)} security events to log")
            
        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")
    
    def _compute_event_hash(self, event_dict: Dict[str, Any]) -> str:
        """Compute integrity hash for event"""
        # Create deterministic string representation
        event_copy = event_dict.copy()
        event_copy.pop('integrity_hash', None)  # Remove hash field itself
        
        event_str = json.dumps(event_copy, sort_keys=True)
        
        # Include previous hash for chain integrity
        if self.integrity_hash_chain:
            event_str += self.integrity_hash_chain[-1]
        
        event_hash = hashlib.sha256(event_str.encode()).hexdigest()
        self.integrity_hash_chain.append(event_hash)
        
        # Keep only recent hashes
        if len(self.integrity_hash_chain) > 1000:
            self.integrity_hash_chain = self.integrity_hash_chain[-1000:]
        
        return event_hash
    
    def query_events(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    event_types: Optional[List[AuditEventType]] = None,
                    user_id: Optional[str] = None,
                    severity: Optional[str] = None,
                    limit: int = 100) -> List[SecurityEvent]:
        """
        Query security events with filtering.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            event_types: List of event types to include
            user_id: Filter by user ID
            severity: Filter by severity level
            limit: Maximum number of events to return
            
        Returns:
            List of matching security events
        """
        # For this implementation, query from recent events
        # In production, would query from persistent storage
        
        results = []
        
        for event in reversed(self.recent_events):  # Most recent first
            if len(results) >= limit:
                break
            
            # Time range filter
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            
            # Event type filter
            if event_types and event.event_type not in event_types:
                continue
            
            # User filter
            if user_id and event.user_id != user_id:
                continue
            
            # Severity filter
            if severity and event.severity != severity:
                continue
            
            results.append(event)
        
        return results
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audit statistics"""
        now = datetime.now()
        
        # Time-based statistics
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        recent_1h = [e for e in self.recent_events if e.timestamp >= one_hour_ago]
        recent_24h = [e for e in self.recent_events if e.timestamp >= one_day_ago]
        
        return {
            'total_events': len(self.recent_events),
            'events_last_1h': len(recent_1h),
            'events_last_24h': len(recent_24h),
            'event_types_distribution': dict(self.event_counters),
            'severity_distribution': {
                severity: len([e for e in self.recent_events if e.severity == severity])
                for severity in ['critical', 'high', 'medium', 'low', 'info']
            },
            'top_event_types_1h': self._get_top_event_types(recent_1h, 5),
            'average_risk_score': (
                sum(e.risk_score for e in self.recent_events if e.risk_score) / 
                len([e for e in self.recent_events if e.risk_score])
            ) if any(e.risk_score for e in self.recent_events) else 0.0,
            'integrity_chain_length': len(self.integrity_hash_chain),
            'queue_size': self.event_queue.qsize(),
            'compliance_mode': self.compliance_mode
        }
    
    def _get_top_event_types(self, events: List[SecurityEvent], limit: int) -> Dict[str, int]:
        """Get top event types from event list"""
        type_counts = {}
        for event in events:
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        # Return top N
        return dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:limit])
    
    def shutdown(self):
        """Gracefully shutdown audit logger"""
        logger.info("Shutting down security auditor...")
        
        # Signal shutdown and wait for background thread
        self.shutdown_event.set()
        
        # Add sentinel to queue to wake up background thread
        try:
            self.event_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Wait for background thread to finish
        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=10)
        
        logger.info("Security auditor shutdown complete")


# Convenience functions
def create_security_auditor(config: Optional[Dict[str, Any]] = None) -> SecurityAuditor:
    """Create security auditor with configuration"""
    return SecurityAuditor(config)


def create_compliance_auditor(compliance_mode: str = 'standard') -> SecurityAuditor:
    """Create compliance-ready auditor"""
    compliance_configs = {
        'gdpr': {
            'retention_days': 2557,  # 7 years
            'integrity_protection': True,
            'compliance_mode': 'gdpr'
        },
        'hipaa': {
            'retention_days': 2192,  # 6 years
            'integrity_protection': True,
            'compliance_mode': 'hipaa'
        },
        'sox': {
            'retention_days': 2557,  # 7 years
            'integrity_protection': True,
            'compliance_mode': 'sox'
        }
    }
    
    config = compliance_configs.get(compliance_mode, {})
    return SecurityAuditor(config)