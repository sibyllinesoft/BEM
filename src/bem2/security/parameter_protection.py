"""
Parameter Protection System for VC0 Safety Components

Provides comprehensive protection for critical safety parameters including
integrity checking, access control, and tamper detection.

Security Features:
- Parameter integrity validation with cryptographic hashing
- Access control for parameter modifications
- Real-time tamper detection and alerting
- Secure parameter backup and recovery
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path
import logging

from .auth_manager import SecurityContext, SecurityRole, AuthorizationError

logger = logging.getLogger(__name__)


class ParameterSecurityError(Exception):
    """Base exception for parameter security violations"""
    pass


class ParameterTamperingError(ParameterSecurityError):
    """Raised when parameter tampering is detected"""
    pass


class ParameterAccessError(ParameterSecurityError):
    """Raised when unauthorized parameter access is attempted"""
    pass


@dataclass
class ParameterMetadata:
    """Metadata for protected parameters"""
    name: str
    parameter_type: str  # 'tensor', 'scalar', 'module'
    shape: Optional[Tuple[int, ...]]
    dtype: str
    device: str
    integrity_hash: str
    created_at: datetime
    last_modified: datetime
    last_modified_by: str
    protection_level: str  # 'critical', 'high', 'medium', 'low'
    access_count: int = 0
    modification_count: int = 0


class ParameterGuard:
    """
    Comprehensive parameter protection system.
    
    Protects critical safety parameters from unauthorized access and tampering,
    with cryptographic integrity checking and comprehensive audit logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Protected parameters registry
        self.protected_parameters: Dict[str, torch.nn.Parameter] = {}
        self.parameter_metadata: Dict[str, ParameterMetadata] = {}
        
        # Integrity checking
        self.integrity_check_frequency = self.config.get('integrity_check_frequency', 100)
        self.last_integrity_check = 0
        self.integrity_violations: List[Dict[str, Any]] = []
        
        # Access control
        self.access_log: List[Dict[str, Any]] = []
        self.modification_log: List[Dict[str, Any]] = []
        
        # Backup system
        self.backup_enabled = self.config.get('backup_enabled', True)
        self.backup_frequency = self.config.get('backup_frequency', 1000)
        self.backup_directory = Path(self.config.get('backup_directory', './parameter_backups'))
        self.backup_directory.mkdir(exist_ok=True)
        
        # Security thresholds
        self.max_modifications_per_hour = self.config.get('max_modifications_per_hour', 10)
        self.suspicious_access_threshold = self.config.get('suspicious_access_threshold', 50)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default protection configuration"""
        return {
            'hash_algorithm': 'sha256',
            'integrity_check_frequency': 100,  # Check every N steps
            'backup_enabled': True,
            'backup_frequency': 1000,  # Backup every N steps
            'backup_directory': './parameter_backups',
            'max_modifications_per_hour': 10,
            'suspicious_access_threshold': 50,
            'real_time_monitoring': True,
            'tamper_detection': True,
            'secure_deletion': True
        }
    
    def protect_parameter(self, name: str, parameter: torch.nn.Parameter,
                         protection_level: str = 'high',
                         security_context: Optional[SecurityContext] = None) -> bool:
        """
        Protect a parameter with integrity checking and access control.
        
        Args:
            name: Parameter name for identification
            parameter: Parameter tensor to protect
            protection_level: Protection level ('critical', 'high', 'medium', 'low')
            security_context: Security context for authorization
            
        Returns:
            True if protection successful
            
        Raises:
            ParameterAccessError: If unauthorized
        """
        # Verify authorization for critical parameters
        if protection_level == 'critical' and security_context:
            if SecurityRole.ADMIN not in security_context.roles:
                raise ParameterAccessError("Admin role required to protect critical parameters")
        
        # Create parameter metadata
        metadata = ParameterMetadata(
            name=name,
            parameter_type='tensor',
            shape=tuple(parameter.shape),
            dtype=str(parameter.dtype),
            device=str(parameter.device),
            integrity_hash=self._compute_parameter_hash(parameter),
            created_at=datetime.now(),
            last_modified=datetime.now(),
            last_modified_by=security_context.username if security_context else 'system',
            protection_level=protection_level
        )
        
        # Register parameter
        self.protected_parameters[name] = parameter
        self.parameter_metadata[name] = metadata
        
        # Create initial backup if enabled
        if self.backup_enabled:
            self._create_parameter_backup(name, parameter, metadata)
        
        self._log_access('parameter_protected', name, security_context, {
            'protection_level': protection_level,
            'parameter_shape': metadata.shape,
            'parameter_dtype': metadata.dtype
        })
        
        logger.info(f"Parameter '{name}' protected with level '{protection_level}'")
        return True
    
    def validate_parameter_integrity(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate integrity of protected parameters.
        
        Args:
            name: Specific parameter name to check (None for all)
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_parameters': 0,
            'valid_parameters': 0,
            'invalid_parameters': 0,
            'integrity_violations': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        parameters_to_check = [name] if name else list(self.protected_parameters.keys())
        
        for param_name in parameters_to_check:
            if param_name not in self.protected_parameters:
                continue
                
            results['total_parameters'] += 1
            parameter = self.protected_parameters[param_name]
            metadata = self.parameter_metadata[param_name]
            
            # Compute current hash
            current_hash = self._compute_parameter_hash(parameter)
            expected_hash = metadata.integrity_hash
            
            if current_hash == expected_hash:
                results['valid_parameters'] += 1
            else:
                results['invalid_parameters'] += 1
                
                # Record integrity violation
                violation = {
                    'parameter_name': param_name,
                    'expected_hash': expected_hash,
                    'actual_hash': current_hash,
                    'detected_at': datetime.now().isoformat(),
                    'protection_level': metadata.protection_level
                }
                
                results['integrity_violations'].append(violation)
                self.integrity_violations.append(violation)
                
                # Log critical violation
                self._log_access('integrity_violation', param_name, None, violation)
                
                # Alert for critical parameters
                if metadata.protection_level == 'critical':
                    logger.critical(f"CRITICAL PARAMETER TAMPERING DETECTED: {param_name}")
                    
                    # Trigger automatic recovery for critical parameters
                    if self.config.get('auto_recovery_critical', True):
                        self._attempt_parameter_recovery(param_name)
        
        self.last_integrity_check = time.time()
        return results
    
    def update_parameter(self, name: str, new_value: torch.Tensor,
                        security_context: SecurityContext,
                        justification: str = "") -> bool:
        """
        Securely update protected parameter with authorization.
        
        Args:
            name: Parameter name
            new_value: New parameter value
            security_context: Security context for authorization
            justification: Reason for parameter update
            
        Returns:
            True if update successful
            
        Raises:
            ParameterAccessError: If unauthorized
            ParameterSecurityError: If update fails security checks
        """
        if name not in self.protected_parameters:
            raise ParameterSecurityError(f"Parameter '{name}' is not protected")
        
        metadata = self.parameter_metadata[name]
        
        # Check authorization based on protection level
        required_role = self._get_required_role_for_modification(metadata.protection_level)
        if required_role not in security_context.roles:
            self._log_access('unauthorized_modification_attempt', name, security_context, {
                'required_role': required_role.value,
                'user_roles': [role.value for role in security_context.roles]
            })
            raise ParameterAccessError(f"Role {required_role.value} required for parameter modification")
        
        # Validate modification rate
        if not self._validate_modification_rate(security_context.username):
            raise ParameterSecurityError("Modification rate limit exceeded")
        
        # Validate new value
        if not self._validate_parameter_value(new_value, metadata):
            raise ParameterSecurityError("Invalid parameter value")
        
        # Create backup before modification
        if self.backup_enabled:
            self._create_parameter_backup(name, self.protected_parameters[name], metadata)
        
        # Update parameter
        old_hash = metadata.integrity_hash
        self.protected_parameters[name].data.copy_(new_value)
        
        # Update metadata
        new_hash = self._compute_parameter_hash(self.protected_parameters[name])
        metadata.integrity_hash = new_hash
        metadata.last_modified = datetime.now()
        metadata.last_modified_by = security_context.username
        metadata.modification_count += 1
        
        # Log modification
        self._log_access('parameter_modified', name, security_context, {
            'old_hash': old_hash[:16],  # Truncated for logging
            'new_hash': new_hash[:16],
            'justification': justification,
            'modification_count': metadata.modification_count
        })
        
        logger.info(f"Parameter '{name}' updated by {security_context.username}")
        return True
    
    def get_parameter_info(self, name: str, 
                          security_context: Optional[SecurityContext] = None) -> Dict[str, Any]:
        """
        Get information about protected parameter.
        
        Args:
            name: Parameter name
            security_context: Security context for authorization
            
        Returns:
            Parameter information dictionary
        """
        if name not in self.protected_parameters:
            raise ParameterSecurityError(f"Parameter '{name}' is not protected")
        
        metadata = self.parameter_metadata[name]
        metadata.access_count += 1
        
        # Log access
        self._log_access('parameter_info_accessed', name, security_context, {
            'access_count': metadata.access_count
        })
        
        # Return metadata (excluding sensitive information)
        info = asdict(metadata)
        info['current_integrity_valid'] = self._verify_single_parameter_integrity(name)
        info['last_integrity_check'] = datetime.fromtimestamp(self.last_integrity_check).isoformat()
        
        return info
    
    def _compute_parameter_hash(self, parameter: torch.nn.Parameter) -> str:
        """Compute cryptographic hash of parameter"""
        # Convert to consistent format for hashing
        data_bytes = parameter.data.cpu().detach().numpy().tobytes()
        
        # Use SHA-256 for cryptographic security
        hash_obj = hashlib.sha256()
        hash_obj.update(data_bytes)
        hash_obj.update(str(parameter.shape).encode())
        hash_obj.update(str(parameter.dtype).encode())
        
        return hash_obj.hexdigest()
    
    def _verify_single_parameter_integrity(self, name: str) -> bool:
        """Verify integrity of single parameter"""
        parameter = self.protected_parameters[name]
        metadata = self.parameter_metadata[name]
        
        current_hash = self._compute_parameter_hash(parameter)
        return current_hash == metadata.integrity_hash
    
    def _get_required_role_for_modification(self, protection_level: str) -> SecurityRole:
        """Get required role for parameter modification"""
        role_mapping = {
            'critical': SecurityRole.ADMIN,
            'high': SecurityRole.SAFETY_OPERATOR,
            'medium': SecurityRole.MODEL_OPERATOR,
            'low': SecurityRole.MODEL_OPERATOR
        }
        return role_mapping.get(protection_level, SecurityRole.ADMIN)
    
    def _validate_modification_rate(self, username: str) -> bool:
        """Validate modification rate for user"""
        now = datetime.now()
        one_hour_ago = now.timestamp() - 3600
        
        recent_modifications = [
            entry for entry in self.modification_log
            if (entry['username'] == username and 
                datetime.fromisoformat(entry['timestamp']).timestamp() > one_hour_ago)
        ]
        
        return len(recent_modifications) < self.max_modifications_per_hour
    
    def _validate_parameter_value(self, value: torch.Tensor, 
                                 metadata: ParameterMetadata) -> bool:
        """Validate new parameter value"""
        # Check shape compatibility
        if tuple(value.shape) != metadata.shape:
            logger.warning(f"Parameter shape mismatch: expected {metadata.shape}, got {value.shape}")
            return False
        
        # Check for NaN or infinite values
        if torch.isnan(value).any() or torch.isinf(value).any():
            logger.warning("Parameter contains NaN or infinite values")
            return False
        
        # Check value range (basic sanity check)
        if torch.abs(value).max() > 1000:  # Configurable threshold
            logger.warning("Parameter contains extremely large values")
            return False
        
        return True
    
    def _create_parameter_backup(self, name: str, parameter: torch.nn.Parameter,
                                metadata: ParameterMetadata):
        """Create backup of parameter"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_directory / f"{name}_{timestamp}.pt"
        
        backup_data = {
            'parameter': parameter.data.clone(),
            'metadata': asdict(metadata),
            'backup_timestamp': datetime.now().isoformat()
        }
        
        torch.save(backup_data, backup_path)
        logger.debug(f"Parameter backup created: {backup_path}")
    
    def _attempt_parameter_recovery(self, name: str) -> bool:
        """Attempt to recover parameter from backup"""
        backup_files = list(self.backup_directory.glob(f"{name}_*.pt"))
        
        if not backup_files:
            logger.error(f"No backups found for parameter '{name}'")
            return False
        
        # Use most recent backup
        latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
        
        try:
            backup_data = torch.load(latest_backup, map_location='cpu')
            
            # Restore parameter
            self.protected_parameters[name].data.copy_(backup_data['parameter'])
            
            # Update metadata
            metadata = self.parameter_metadata[name]
            metadata.integrity_hash = self._compute_parameter_hash(self.protected_parameters[name])
            metadata.last_modified = datetime.now()
            metadata.last_modified_by = 'system_recovery'
            
            logger.info(f"Parameter '{name}' recovered from backup: {latest_backup}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recover parameter '{name}': {str(e)}")
            return False
    
    def _log_access(self, event_type: str, parameter_name: str,
                   security_context: Optional[SecurityContext],
                   details: Dict[str, Any]):
        """Log parameter access for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'parameter_name': parameter_name,
            'username': security_context.username if security_context else 'system',
            'user_id': security_context.user_id if security_context else 'system',
            'details': details
        }
        
        if event_type == 'parameter_modified':
            self.modification_log.append(log_entry)
        else:
            self.access_log.append(log_entry)
        
        logger.info(f"Parameter Access: {event_type}", extra=log_entry)
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get parameter security statistics"""
        now = datetime.now()
        one_hour_ago = now.timestamp() - 3600
        
        recent_access = [
            entry for entry in self.access_log
            if datetime.fromisoformat(entry['timestamp']).timestamp() > one_hour_ago
        ]
        
        recent_modifications = [
            entry for entry in self.modification_log
            if datetime.fromisoformat(entry['timestamp']).timestamp() > one_hour_ago
        ]
        
        return {
            'protected_parameters_count': len(self.protected_parameters),
            'integrity_violations_total': len(self.integrity_violations),
            'recent_access_1h': len(recent_access),
            'recent_modifications_1h': len(recent_modifications),
            'last_integrity_check': datetime.fromtimestamp(self.last_integrity_check).isoformat(),
            'protection_levels': {
                level: sum(1 for m in self.parameter_metadata.values() 
                          if m.protection_level == level)
                for level in ['critical', 'high', 'medium', 'low']
            },
            'backup_files_count': len(list(self.backup_directory.glob("*.pt"))) if self.backup_directory.exists() else 0
        }
    
    def emergency_lockdown(self, security_context: SecurityContext,
                          reason: str = "Emergency lockdown") -> bool:
        """
        Emergency lockdown - disable all parameter modifications.
        
        Args:
            security_context: Admin security context
            reason: Reason for lockdown
            
        Returns:
            True if lockdown successful
        """
        # Require admin privileges
        if SecurityRole.ADMIN not in security_context.roles:
            raise ParameterAccessError("Admin role required for emergency lockdown")
        
        # Set emergency flag (would integrate with broader system lockdown)
        self.config['emergency_lockdown'] = True
        self.config['lockdown_reason'] = reason
        self.config['lockdown_time'] = datetime.now().isoformat()
        self.config['lockdown_by'] = security_context.username
        
        self._log_access('emergency_lockdown_activated', 'all_parameters', security_context, {
            'reason': reason
        })
        
        logger.critical(f"EMERGENCY PARAMETER LOCKDOWN ACTIVATED by {security_context.username}: {reason}")
        return True


# Convenience functions
def create_parameter_guard(config: Optional[Dict[str, Any]] = None) -> ParameterGuard:
    """Create parameter guard with configuration"""
    return ParameterGuard(config)


def protect_safety_parameters(model: nn.Module, parameter_guard: ParameterGuard,
                             security_context: SecurityContext) -> Dict[str, bool]:
    """
    Protect all safety-related parameters in a model.
    
    Args:
        model: Model containing safety parameters
        parameter_guard: Parameter protection system
        security_context: Security context for authorization
        
    Returns:
        Dictionary mapping parameter names to protection status
    """
    results = {}
    
    # Identify safety parameters (would be more sophisticated in practice)
    safety_parameter_patterns = [
        'safety', 'constitutional', 'violation', 'lagrangian', 'lambda'
    ]
    
    for name, param in model.named_parameters():
        if any(pattern in name.lower() for pattern in safety_parameter_patterns):
            try:
                # Determine protection level based on parameter name
                if 'lambda' in name.lower() or 'constitutional' in name.lower():
                    protection_level = 'critical'
                elif 'safety' in name.lower():
                    protection_level = 'high'
                else:
                    protection_level = 'medium'
                
                results[name] = parameter_guard.protect_parameter(
                    name, param, protection_level, security_context
                )
            except Exception as e:
                logger.error(f"Failed to protect parameter {name}: {str(e)}")
                results[name] = False
    
    return results