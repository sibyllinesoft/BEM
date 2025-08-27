"""
Authentication and Authorization Management for VC0 Safety System

Provides comprehensive authentication, authorization, and session management
with role-based access control for safety-critical operations.

Security Features:
- Multi-factor authentication support
- Role-based access control (RBAC)
- Session management with secure tokens
- Audit logging for all security events
"""

import jwt
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import torch
import logging

logger = logging.getLogger(__name__)


class SecurityRole(Enum):
    """Security roles for access control"""
    ADMIN = "admin"              # Full system access
    SAFETY_OPERATOR = "safety_operator"  # Safety system operations
    MODEL_OPERATOR = "model_operator"    # Model operations
    AUDITOR = "auditor"         # Read-only audit access
    USER = "user"               # Basic user access
    READONLY = "readonly"       # Read-only access


class AuthorizationError(Exception):
    """Raised when authorization fails"""
    pass


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


class SessionExpiredError(AuthenticationError):
    """Raised when session has expired"""
    pass


@dataclass
class SecurityContext:
    """Security context for authenticated operations"""
    user_id: str
    username: str
    roles: List[SecurityRole]
    session_token: str
    created_at: datetime
    expires_at: datetime
    mfa_verified: bool = False
    permissions: Set[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = set()
    
    def is_expired(self) -> bool:
        """Check if security context has expired"""
        return datetime.now() > self.expires_at
    
    def has_role(self, role: SecurityRole) -> bool:
        """Check if context has specific role"""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions
    
    def time_to_expiry(self) -> timedelta:
        """Get time until context expires"""
        return self.expires_at - datetime.now()


class AuthenticationManager:
    """
    Comprehensive authentication and authorization manager.
    
    Handles user authentication, session management, and access control
    with support for multi-factor authentication and audit logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # JWT secret key for token signing
        self.secret_key = self.config.get('secret_key') or self._generate_secret_key()
        
        # Active sessions
        self.active_sessions: Dict[str, SecurityContext] = {}
        
        # User database (in production, this would be external)
        self.users_db = self._initialize_user_database()
        
        # Role permissions mapping
        self.role_permissions = self._initialize_role_permissions()
        
        # Security event tracking
        self.security_events = []
        
        # Failed login attempts tracking
        self.failed_attempts: Dict[str, List[datetime]] = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default authentication configuration"""
        return {
            'session_timeout': 3600,  # 1 hour
            'max_failed_attempts': 5,
            'lockout_duration': 900,  # 15 minutes
            'require_mfa': False,  # Set to True for production
            'token_rotation': True,
            'password_min_length': 8,
            'jwt_algorithm': 'HS256',
            'audit_logging': True,
            'secure_cookies': True
        }
    
    def _generate_secret_key(self) -> str:
        """Generate cryptographically secure secret key"""
        return secrets.token_urlsafe(32)
    
    def _initialize_user_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize user database (placeholder - use external DB in production)"""
        # In production, this would connect to external user database
        return {
            'admin': {
                'password_hash': self._hash_password('admin_password_2024'),
                'roles': [SecurityRole.ADMIN],
                'mfa_enabled': True,
                'created_at': datetime.now(),
                'active': True
            },
            'safety_operator': {
                'password_hash': self._hash_password('safety_op_pass_2024'),
                'roles': [SecurityRole.SAFETY_OPERATOR, SecurityRole.MODEL_OPERATOR],
                'mfa_enabled': True,
                'created_at': datetime.now(),
                'active': True
            },
            'auditor': {
                'password_hash': self._hash_password('auditor_pass_2024'),
                'roles': [SecurityRole.AUDITOR, SecurityRole.READONLY],
                'mfa_enabled': False,
                'created_at': datetime.now(),
                'active': True
            }
        }
    
    def _initialize_role_permissions(self) -> Dict[SecurityRole, Set[str]]:
        """Initialize role-based permissions"""
        return {
            SecurityRole.ADMIN: {
                'safety.override', 'safety.configure', 'safety.monitor',
                'model.train', 'model.deploy', 'model.configure',
                'system.configure', 'system.restart', 'system.backup',
                'audit.read', 'audit.configure', 'users.manage'
            },
            SecurityRole.SAFETY_OPERATOR: {
                'safety.monitor', 'safety.configure',
                'model.evaluate', 'model.monitor',
                'audit.read'
            },
            SecurityRole.MODEL_OPERATOR: {
                'model.train', 'model.evaluate', 'model.monitor',
                'audit.read'
            },
            SecurityRole.AUDITOR: {
                'audit.read', 'audit.export',
                'system.monitor'
            },
            SecurityRole.USER: {
                'model.use', 'system.monitor'
            },
            SecurityRole.READONLY: {
                'system.monitor'
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{hash_obj.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split(':')
            hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hash_obj.hex() == hash_hex
        except ValueError:
            return False
    
    def authenticate_user(self, username: str, password: str, 
                         mfa_token: Optional[str] = None) -> SecurityContext:
        """
        Authenticate user and create security context.
        
        Args:
            username: Username for authentication
            password: Password for authentication
            mfa_token: Optional MFA token
            
        Returns:
            SecurityContext if authentication successful
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Check for account lockout
        if self._is_account_locked(username):
            self._log_security_event('authentication_blocked_locked_account', {
                'username': username,
                'lockout_remaining': self._get_lockout_remaining(username)
            })
            raise AuthenticationError("Account temporarily locked due to failed login attempts")
        
        # Validate user exists and is active
        if username not in self.users_db:
            self._record_failed_attempt(username)
            self._log_security_event('authentication_failed_unknown_user', {'username': username})
            raise AuthenticationError("Invalid credentials")
        
        user_data = self.users_db[username]
        if not user_data.get('active', True):
            self._log_security_event('authentication_failed_inactive_user', {'username': username})
            raise AuthenticationError("Account is disabled")
        
        # Verify password
        if not self._verify_password(password, user_data['password_hash']):
            self._record_failed_attempt(username)
            self._log_security_event('authentication_failed_wrong_password', {'username': username})
            raise AuthenticationError("Invalid credentials")
        
        # Check MFA if required
        mfa_verified = False
        if self.config.get('require_mfa', False) or user_data.get('mfa_enabled', False):
            if not mfa_token:
                raise AuthenticationError("MFA token required")
            
            if not self._verify_mfa_token(username, mfa_token):
                self._record_failed_attempt(username)
                self._log_security_event('authentication_failed_invalid_mfa', {'username': username})
                raise AuthenticationError("Invalid MFA token")
            
            mfa_verified = True
        
        # Create security context
        session_token = self._generate_session_token()
        expires_at = datetime.now() + timedelta(seconds=self.config['session_timeout'])
        
        # Get user permissions
        user_permissions = set()
        for role in user_data['roles']:
            user_permissions.update(self.role_permissions.get(role, set()))
        
        context = SecurityContext(
            user_id=f"user_{hash(username) % 10000}",
            username=username,
            roles=user_data['roles'],
            session_token=session_token,
            created_at=datetime.now(),
            expires_at=expires_at,
            mfa_verified=mfa_verified,
            permissions=user_permissions
        )
        
        # Store active session
        self.active_sessions[session_token] = context
        
        # Clear failed attempts
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        self._log_security_event('authentication_successful', {
            'username': username,
            'user_id': context.user_id,
            'roles': [role.value for role in context.roles],
            'mfa_verified': mfa_verified
        })
        
        return context
    
    def validate_session(self, session_token: str) -> SecurityContext:
        """
        Validate session token and return security context.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            SecurityContext if session is valid
            
        Raises:
            AuthenticationError: If session is invalid or expired
        """
        if not session_token or session_token not in self.active_sessions:
            raise AuthenticationError("Invalid session token")
        
        context = self.active_sessions[session_token]
        
        if context.is_expired():
            self._invalidate_session(session_token)
            self._log_security_event('session_expired', {
                'username': context.username,
                'user_id': context.user_id
            })
            raise SessionExpiredError("Session has expired")
        
        return context
    
    def authorize_action(self, context: SecurityContext, 
                        required_permission: str) -> bool:
        """
        Check if user is authorized for specific action.
        
        Args:
            context: Security context
            required_permission: Required permission string
            
        Returns:
            True if authorized, False otherwise
        """
        # Always allow if user has admin role
        if SecurityRole.ADMIN in context.roles:
            return True
        
        # Check specific permission
        is_authorized = context.has_permission(required_permission)
        
        self._log_security_event('authorization_check', {
            'username': context.username,
            'user_id': context.user_id,
            'permission': required_permission,
            'authorized': is_authorized
        })
        
        return is_authorized
    
    def require_authorization(self, context: SecurityContext, 
                            required_permission: str):
        """
        Require authorization for action (raises exception if not authorized).
        
        Args:
            context: Security context
            required_permission: Required permission string
            
        Raises:
            AuthorizationError: If not authorized
        """
        if not self.authorize_action(context, required_permission):
            self._log_security_event('authorization_denied', {
                'username': context.username,
                'user_id': context.user_id,
                'permission': required_permission
            })
            raise AuthorizationError(f"Permission denied: {required_permission}")
    
    def elevate_privileges(self, context: SecurityContext, 
                          target_role: SecurityRole,
                          justification: str) -> SecurityContext:
        """
        Temporarily elevate privileges for sensitive operations.
        
        Args:
            context: Current security context
            target_role: Role to elevate to
            justification: Reason for elevation
            
        Returns:
            New security context with elevated privileges
            
        Raises:
            AuthorizationError: If elevation not permitted
        """
        # Only admins can elevate to admin role
        if target_role == SecurityRole.ADMIN and SecurityRole.ADMIN not in context.roles:
            raise AuthorizationError("Cannot elevate to admin role")
        
        # Create elevated context with shorter expiry
        elevated_context = SecurityContext(
            user_id=context.user_id,
            username=context.username,
            roles=context.roles + [target_role],
            session_token=self._generate_session_token(),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=10),  # Short-lived
            mfa_verified=context.mfa_verified,
            permissions=context.permissions | self.role_permissions.get(target_role, set())
        )
        
        # Store elevated session
        self.active_sessions[elevated_context.session_token] = elevated_context
        
        self._log_security_event('privilege_elevation', {
            'username': context.username,
            'user_id': context.user_id,
            'target_role': target_role.value,
            'justification': justification,
            'elevated_session': elevated_context.session_token
        })
        
        return elevated_context
    
    def _generate_session_token(self) -> str:
        """Generate secure session token"""
        payload = {
            'token_id': secrets.token_urlsafe(16),
            'issued_at': datetime.now().timestamp(),
            'random': secrets.token_hex(8)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.config['jwt_algorithm'])
    
    def _verify_mfa_token(self, username: str, mfa_token: str) -> bool:
        """Verify MFA token (placeholder implementation)"""
        # In production, this would integrate with TOTP/SMS/hardware tokens
        # For demo purposes, accept any 6-digit numeric token
        return mfa_token.isdigit() and len(mfa_token) == 6
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if username not in self.failed_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_attempts[username]
            if datetime.now() - attempt < timedelta(seconds=self.config['lockout_duration'])
        ]
        
        return len(recent_attempts) >= self.config['max_failed_attempts']
    
    def _get_lockout_remaining(self, username: str) -> int:
        """Get remaining lockout time in seconds"""
        if username not in self.failed_attempts:
            return 0
        
        oldest_relevant_attempt = min([
            attempt for attempt in self.failed_attempts[username]
            if datetime.now() - attempt < timedelta(seconds=self.config['lockout_duration'])
        ])
        
        lockout_end = oldest_relevant_attempt + timedelta(seconds=self.config['lockout_duration'])
        remaining = lockout_end - datetime.now()
        
        return max(0, int(remaining.total_seconds()))
    
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.now())
        
        # Clean old attempts
        cutoff = datetime.now() - timedelta(seconds=self.config['lockout_duration'])
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff
        ]
    
    def _invalidate_session(self, session_token: str):
        """Invalidate session"""
        if session_token in self.active_sessions:
            del self.active_sessions[session_token]
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event for audit trail"""
        if self.config.get('audit_logging', True):
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'details': details,
                'source': 'authentication_manager'
            }
            
            self.security_events.append(event)
            
            logger.info(f"Security Event: {event_type}", extra=event)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions (for monitoring)"""
        return [
            {
                'username': context.username,
                'user_id': context.user_id,
                'roles': [role.value for role in context.roles],
                'created_at': context.created_at.isoformat(),
                'expires_at': context.expires_at.isoformat(),
                'mfa_verified': context.mfa_verified
            }
            for context in self.active_sessions.values()
        ]
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count removed"""
        expired_tokens = [
            token for token, context in self.active_sessions.items()
            if context.is_expired()
        ]
        
        for token in expired_tokens:
            del self.active_sessions[token]
        
        if expired_tokens:
            self._log_security_event('session_cleanup', {
                'expired_sessions_removed': len(expired_tokens)
            })
        
        return len(expired_tokens)
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics for monitoring"""
        now = datetime.now()
        recent_events = [
            event for event in self.security_events
            if (now - datetime.fromisoformat(event['timestamp'])) < timedelta(hours=1)
        ]
        
        return {
            'active_sessions': len(self.active_sessions),
            'recent_events_1h': len(recent_events),
            'failed_attempts_accounts': len(self.failed_attempts),
            'locked_accounts': sum(1 for username in self.failed_attempts.keys() 
                                 if self._is_account_locked(username)),
            'event_types_1h': {
                event_type: sum(1 for e in recent_events if e['event_type'] == event_type)
                for event_type in set(e['event_type'] for e in recent_events)
            }
        }


# Convenience functions
def create_auth_manager(config: Optional[Dict[str, Any]] = None) -> AuthenticationManager:
    """Create authentication manager with configuration"""
    return AuthenticationManager(config)


def require_auth(required_permission: str):
    """Decorator to require authentication for function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract security context from kwargs
            context = kwargs.get('security_context')
            if not context:
                raise AuthenticationError("Security context required")
            
            # Check authorization
            auth_manager = kwargs.get('auth_manager')
            if not auth_manager:
                raise AuthenticationError("Authentication manager required")
            
            auth_manager.require_authorization(context, required_permission)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator