"""
Circuit Breaker and Failsafe System for VC0 Safety Components

Implements circuit breaker patterns and failsafe mechanisms to ensure
system resilience and prevent cascade failures in safety-critical operations.

Security Features:
- Automatic failure detection and isolation
- Graceful degradation under load
- Circuit breaker patterns for safety components
- Emergency failsafe activation
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing - reject requests
    HALF_OPEN = "half_open" # Testing recovery


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_VIOLATION = "security_violation"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes before closing from half-open
    timeout_seconds: float = 60.0       # Timeout for open state
    half_open_max_calls: int = 3        # Max calls allowed in half-open
    sliding_window_size: int = 20       # Size of sliding window for failure tracking
    minimum_throughput: int = 10        # Minimum requests before evaluation


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class SafetyCircuitBreaker:
    """
    Circuit breaker for safety-critical components.
    
    Provides automatic failure detection, isolation, and recovery
    for safety system components to prevent cascade failures.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        
        # Sliding window for failure tracking
        self.call_history: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': 0,
            'last_state_change': None
        }
        
        # Failure tracking
        self.failure_types: Dict[FailureType, int] = {}
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_failure: Optional[Callable] = None
        self.on_success: Optional[Callable] = None
    
    def call(self, func: Callable, *args, timeout: Optional[float] = None, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            timeout: Optional timeout override
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            TimeoutError: If function times out
            Exception: If function raises exception
        """
        with self.lock:
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats['rejected_calls'] += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Next attempt allowed at {self.next_attempt_time}"
                    )
            
            # Check half-open state limits
            if self.state == CircuitState.HALF_OPEN:
                recent_half_open_calls = len([
                    call for call in self.call_history[-self.config.half_open_max_calls:]
                    if call.get('state') == CircuitState.HALF_OPEN.value
                ])
                
                if recent_half_open_calls >= self.config.half_open_max_calls:
                    self.stats['rejected_calls'] += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' in HALF_OPEN state has reached call limit"
                    )
        
        # Execute function with monitoring
        return self._execute_with_monitoring(func, args, kwargs, timeout)
    
    def _execute_with_monitoring(self, func: Callable, args: tuple, kwargs: dict, 
                                timeout: Optional[float]) -> Any:
        """Execute function with comprehensive monitoring"""
        start_time = time.time()
        call_timeout = timeout or 30.0  # Default timeout
        
        call_record = {
            'timestamp': datetime.now(),
            'state': self.state.value,
            'timeout': call_timeout
        }
        
        try:
            # Simple timeout implementation (in production, would use proper async timeout)
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            if execution_time > call_timeout:
                raise TimeoutError(f"Function execution exceeded timeout ({call_timeout}s)")
            
            # Record success
            call_record.update({
                'success': True,
                'execution_time': execution_time
            })
            
            self._record_success(call_record)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            failure_type = self._classify_failure(e)
            
            # Record failure
            call_record.update({
                'success': False,
                'execution_time': execution_time,
                'failure_type': failure_type.value,
                'error': str(e)
            })
            
            self._record_failure(call_record, failure_type)
            raise
    
    def _record_success(self, call_record: Dict[str, Any]):
        """Record successful call"""
        with self.lock:
            self.call_history.append(call_record)
            self._trim_call_history()
            
            self.stats['total_calls'] += 1
            self.stats['successful_calls'] += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            # Reset failure count on success in closed state
            if self.state == CircuitState.CLOSED:
                self.failure_count = 0
            
            if self.on_success:
                try:
                    self.on_success(self.name, call_record)
                except Exception as e:
                    logger.error(f"Error in success callback: {e}")
    
    def _record_failure(self, call_record: Dict[str, Any], failure_type: FailureType):
        """Record failed call"""
        with self.lock:
            self.call_history.append(call_record)
            self._trim_call_history()
            
            self.stats['total_calls'] += 1
            self.stats['failed_calls'] += 1
            
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Track failure types
            self.failure_types[failure_type] = self.failure_types.get(failure_type, 0) + 1
            
            # Check if should open circuit
            if self._should_open_circuit():
                self._transition_to_open()
            
            if self.on_failure:
                try:
                    self.on_failure(self.name, call_record, failure_type)
                except Exception as e:
                    logger.error(f"Error in failure callback: {e}")
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify type of failure"""
        if isinstance(exception, TimeoutError):
            return FailureType.TIMEOUT
        elif isinstance(exception, (ValueError, TypeError)):
            return FailureType.VALIDATION_ERROR
        elif isinstance(exception, (MemoryError, OSError)):
            return FailureType.RESOURCE_EXHAUSTION
        elif 'security' in str(exception).lower() or 'unauthorized' in str(exception).lower():
            return FailureType.SECURITY_VIOLATION
        elif 'constitutional' in str(exception).lower() or 'violation' in str(exception).lower():
            return FailureType.CONSTITUTIONAL_VIOLATION
        else:
            return FailureType.EXCEPTION
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        if len(self.call_history) < self.config.minimum_throughput:
            return False
        
        # Look at recent calls within sliding window
        recent_calls = self.call_history[-self.config.sliding_window_size:]
        recent_failures = [call for call in recent_calls if not call['success']]
        
        # Open if failure rate exceeds threshold
        if len(recent_failures) >= self.config.failure_threshold:
            return True
        
        # Special handling for critical failure types
        critical_failures = [
            call for call in recent_calls 
            if not call['success'] and call.get('failure_type') in [
                FailureType.SECURITY_VIOLATION.value,
                FailureType.CONSTITUTIONAL_VIOLATION.value
            ]
        ]
        
        # Lower threshold for critical failures
        if len(critical_failures) >= max(1, self.config.failure_threshold // 2):
            return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset from open state"""
        if not self.next_attempt_time:
            return False
        
        return time.time() >= self.next_attempt_time
    
    def _transition_to_open(self):
        """Transition circuit breaker to open state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.next_attempt_time = time.time() + self.config.timeout_seconds
        
        self._record_state_change(old_state, CircuitState.OPEN)
        
        logger.warning(
            f"Circuit breaker '{self.name}' OPENED due to {self.failure_count} failures. "
            f"Next attempt at {datetime.fromtimestamp(self.next_attempt_time)}"
        )
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        
        self._record_state_change(old_state, CircuitState.HALF_OPEN)
        
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN - testing recovery")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.next_attempt_time = None
        
        self._record_state_change(old_state, CircuitState.CLOSED)
        
        logger.info(f"Circuit breaker '{self.name}' CLOSED - normal operation resumed")
    
    def _record_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Record state change for monitoring"""
        self.stats['state_changes'] += 1
        self.stats['last_state_change'] = datetime.now()
        
        if self.on_state_change:
            try:
                self.on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def _trim_call_history(self):
        """Trim call history to maintain sliding window"""
        max_history = self.config.sliding_window_size * 2  # Keep extra for analysis
        if len(self.call_history) > max_history:
            self.call_history = self.call_history[-max_history:]
    
    def force_open(self, reason: str = "Manual override"):
        """Manually force circuit breaker to open"""
        with self.lock:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.config.timeout_seconds
            self.failure_count = self.config.failure_threshold  # Ensure it stays open
            
            self._record_state_change(old_state, CircuitState.OPEN)
            
            logger.warning(f"Circuit breaker '{self.name}' MANUALLY OPENED: {reason}")
    
    def force_closed(self, reason: str = "Manual override"):
        """Manually force circuit breaker to closed"""
        with self.lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.next_attempt_time = None
            
            self._record_state_change(old_state, CircuitState.CLOSED)
            
            logger.info(f"Circuit breaker '{self.name}' MANUALLY CLOSED: {reason}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        with self.lock:
            recent_calls = self.call_history[-self.config.sliding_window_size:]
            recent_failures = [call for call in recent_calls if not call['success']]
            
            status = {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'next_attempt_time': (
                    datetime.fromtimestamp(self.next_attempt_time).isoformat()
                    if self.next_attempt_time else None
                ),
                'statistics': self.stats.copy(),
                'recent_failure_rate': len(recent_failures) / max(1, len(recent_calls)),
                'failure_types': dict(self.failure_types),
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout_seconds': self.config.timeout_seconds
                }
            }
            
            return status
    
    def reset_statistics(self):
        """Reset all statistics"""
        with self.lock:
            self.stats = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'rejected_calls': 0,
                'state_changes': 0,
                'last_state_change': None
            }
            self.failure_types.clear()
            self.call_history.clear()


class SafetyFailsafeManager:
    """
    Manager for safety system failsafe mechanisms.
    
    Coordinates multiple circuit breakers and provides
    system-wide failsafe activation and monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Circuit breakers for safety components
        self.circuit_breakers: Dict[str, SafetyCircuitBreaker] = {}
        
        # System-wide failsafe state
        self.failsafe_active = False
        self.failsafe_reason = None
        self.failsafe_activated_at = None
        
        # Emergency thresholds
        self.emergency_thresholds = self.config.get('emergency_thresholds', {})
        
        # Monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_check = None
        
        # Initialize default circuit breakers
        self._initialize_default_circuit_breakers()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default failsafe manager configuration"""
        return {
            'emergency_thresholds': {
                'max_open_circuits': 3,           # Max circuit breakers that can be open
                'max_failure_rate': 0.5,          # 50% system-wide failure rate
                'critical_component_failure': True # Fail if critical component fails
            },
            'health_check_interval': 30,          # Seconds between health checks
            'automatic_recovery': True,           # Enable automatic failsafe recovery
            'recovery_delay': 300                 # Seconds before attempting recovery
        }
    
    def _initialize_default_circuit_breakers(self):
        """Initialize circuit breakers for key safety components"""
        # Constitutional scorer circuit breaker
        self.add_circuit_breaker(
            'constitutional_scorer',
            CircuitBreakerConfig(
                failure_threshold=3,  # Lower threshold for safety
                timeout_seconds=30.0,
                sliding_window_size=10
            )
        )
        
        # Safety controller circuit breaker
        self.add_circuit_breaker(
            'safety_controller',
            CircuitBreakerConfig(
                failure_threshold=2,  # Very low threshold for critical component
                timeout_seconds=60.0,
                sliding_window_size=5
            )
        )
        
        # Violation detector circuit breaker
        self.add_circuit_breaker(
            'violation_detector',
            CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=45.0,
                sliding_window_size=15
            )
        )
        
        # Parameter guard circuit breaker
        self.add_circuit_breaker(
            'parameter_guard',
            CircuitBreakerConfig(
                failure_threshold=2,  # Critical for parameter integrity
                timeout_seconds=30.0,
                sliding_window_size=5
            )
        )
    
    def add_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> SafetyCircuitBreaker:
        """Add circuit breaker for component"""
        circuit_breaker = SafetyCircuitBreaker(name, config)
        
        # Set up callbacks for system-wide monitoring
        circuit_breaker.on_state_change = self._on_circuit_state_change
        circuit_breaker.on_failure = self._on_circuit_failure
        
        self.circuit_breakers[name] = circuit_breaker
        
        logger.info(f"Added circuit breaker for component '{name}'")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[SafetyCircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def _on_circuit_state_change(self, name: str, old_state: CircuitState, new_state: CircuitState):
        """Handle circuit breaker state changes"""
        logger.info(f"Circuit breaker '{name}' changed state: {old_state.value} -> {new_state.value}")
        
        # Check if should activate system failsafe
        if new_state == CircuitState.OPEN:
            self._check_failsafe_conditions()
    
    def _on_circuit_failure(self, name: str, call_record: Dict[str, Any], failure_type: FailureType):
        """Handle circuit breaker failures"""
        # Check for critical failures
        if failure_type in [FailureType.SECURITY_VIOLATION, FailureType.CONSTITUTIONAL_VIOLATION]:
            logger.critical(f"Critical failure in circuit breaker '{name}': {failure_type.value}")
            
            # Consider immediate failsafe activation for critical components
            if name in ['safety_controller', 'parameter_guard']:
                self.activate_failsafe(f"Critical failure in {name}: {failure_type.value}")
    
    def _check_failsafe_conditions(self):
        """Check if system-wide failsafe should be activated"""
        if self.failsafe_active:
            return  # Already in failsafe mode
        
        # Count open circuit breakers
        open_circuits = [
            cb for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        ]
        
        # Check emergency thresholds
        max_open = self.emergency_thresholds.get('max_open_circuits', 3)
        
        if len(open_circuits) >= max_open:
            self.activate_failsafe(f"{len(open_circuits)} circuit breakers are open (max: {max_open})")
            return
        
        # Check for critical component failures
        critical_components = ['safety_controller', 'parameter_guard']
        critical_failures = [
            cb for cb in open_circuits
            if cb.name in critical_components
        ]
        
        if critical_failures and self.emergency_thresholds.get('critical_component_failure', True):
            self.activate_failsafe(f"Critical component failure: {[cb.name for cb in critical_failures]}")
    
    def activate_failsafe(self, reason: str):
        """Activate system-wide failsafe mode"""
        if self.failsafe_active:
            logger.warning(f"Failsafe already active, additional reason: {reason}")
            return
        
        self.failsafe_active = True
        self.failsafe_reason = reason
        self.failsafe_activated_at = datetime.now()
        
        # Force open all non-critical circuit breakers
        for name, cb in self.circuit_breakers.items():
            if name not in ['constitutional_scorer']:  # Keep some components for basic safety
                cb.force_open(f"System failsafe: {reason}")
        
        logger.critical(f"SYSTEM FAILSAFE ACTIVATED: {reason}")
        
        # In production, would trigger additional emergency procedures:
        # - Alert operations team
        # - Switch to safe mode
        # - Disable non-essential features
        # - Increase monitoring
    
    def deactivate_failsafe(self, reason: str = "Manual recovery"):
        """Deactivate system-wide failsafe mode"""
        if not self.failsafe_active:
            logger.warning("Attempted to deactivate failsafe, but it's not active")
            return
        
        self.failsafe_active = False
        previous_reason = self.failsafe_reason
        self.failsafe_reason = None
        
        # Reset circuit breakers to allow normal operation
        for cb in self.circuit_breakers.values():
            if cb.state == CircuitState.OPEN:
                cb.force_closed(f"Failsafe recovery: {reason}")
        
        logger.info(f"SYSTEM FAILSAFE DEACTIVATED: {reason} (was: {previous_reason})")
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.last_health_check = datetime.now()
        
        health_status = {
            'timestamp': self.last_health_check.isoformat(),
            'failsafe_active': self.failsafe_active,
            'failsafe_reason': self.failsafe_reason,
            'circuit_breakers': {},
            'overall_health': 'healthy'
        }
        
        unhealthy_count = 0
        
        # Check each circuit breaker
        for name, cb in self.circuit_breakers.items():
            cb_status = cb.get_status()
            health_status['circuit_breakers'][name] = cb_status
            
            if cb_status['state'] == CircuitState.OPEN.value:
                unhealthy_count += 1
        
        # Determine overall health
        if self.failsafe_active:
            health_status['overall_health'] = 'failsafe'
        elif unhealthy_count == 0:
            health_status['overall_health'] = 'healthy'
        elif unhealthy_count <= len(self.circuit_breakers) // 2:
            health_status['overall_health'] = 'degraded'
        else:
            health_status['overall_health'] = 'critical'
        
        # Run custom health checks
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                health_status[f'health_check_{check_name}'] = check_result
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                health_status[f'health_check_{check_name}'] = {'status': 'error', 'error': str(e)}
        
        return health_status
    
    def add_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Add custom health check function"""
        self.health_checks[name] = check_func
        logger.info(f"Added health check: {name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'failsafe_active': self.failsafe_active,
            'failsafe_reason': self.failsafe_reason,
            'failsafe_activated_at': (
                self.failsafe_activated_at.isoformat() 
                if self.failsafe_activated_at else None
            ),
            'circuit_breakers_count': len(self.circuit_breakers),
            'open_circuit_breakers': [
                name for name, cb in self.circuit_breakers.items()
                if cb.state == CircuitState.OPEN
            ],
            'last_health_check': (
                self.last_health_check.isoformat()
                if self.last_health_check else None
            ),
            'emergency_thresholds': self.emergency_thresholds
        }


# Convenience functions and decorators
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to functions"""
    cb = SafetyCircuitBreaker(name, config)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


def create_safety_circuit_breaker(name: str, **kwargs) -> SafetyCircuitBreaker:
    """Create safety circuit breaker with common configuration"""
    config = CircuitBreakerConfig(**kwargs)
    return SafetyCircuitBreaker(name, config)


def create_failsafe_manager() -> SafetyFailsafeManager:
    """Create failsafe manager with default configuration"""
    return SafetyFailsafeManager()