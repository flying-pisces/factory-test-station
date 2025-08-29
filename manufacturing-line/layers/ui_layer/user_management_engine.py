"""
User Management Engine for Week 6: Advanced UI & Visualization

This module implements comprehensive user authentication, authorization, and session management
for the manufacturing line control system with role-based access control and security logging.

Performance Target: <200ms authentication and session management
Security Features: RBAC, audit logging, session management, security compliance
"""

import time
import logging
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from collections import defaultdict
import threading

# Week 5 integrations
try:
    from layers.control_layer.monitoring_engine import MonitoringEngine
    from layers.control_layer.orchestration_engine import OrchestrationEngine
except ImportError:
    MonitoringEngine = None
    OrchestrationEngine = None

# Week 4 integrations
try:
    from layers.optimization_layer.analytics_engine import AnalyticsEngine
except ImportError:
    AnalyticsEngine = None

# Core imports
from datetime import datetime
import uuid


class UserRole(Enum):
    """User role definitions for role-based access control"""
    ADMINISTRATOR = "administrator"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    ENGINEER = "engineer"
    MAINTENANCE = "maintenance"
    VIEWER = "viewer"
    AUDITOR = "auditor"


class Permission(Enum):
    """System permission definitions"""
    # System control permissions
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_RESET = "system_reset"
    EMERGENCY_STOP = "emergency_stop"
    
    # Configuration permissions
    CONFIG_READ = "config_read"
    CONFIG_WRITE = "config_write"
    CONFIG_DELETE = "config_delete"
    
    # Data access permissions
    DATA_READ = "data_read"
    DATA_EXPORT = "data_export"
    DATA_DELETE = "data_delete"
    
    # User management permissions
    USER_CREATE = "user_create"
    USER_MODIFY = "user_modify"
    USER_DELETE = "user_delete"
    
    # Monitoring permissions
    MONITOR_VIEW = "monitor_view"
    MONITOR_ALERTS = "monitor_alerts"
    
    # Maintenance permissions
    MAINTENANCE_MODE = "maintenance_mode"
    CALIBRATION = "calibration"
    
    # Audit permissions
    AUDIT_VIEW = "audit_view"
    AUDIT_EXPORT = "audit_export"


class SessionStatus(Enum):
    """Session status definitions"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    LOCKED = "locked"


class AuthenticationResult(Enum):
    """Authentication result types"""
    SUCCESS = "success"
    INVALID_CREDENTIALS = "invalid_credentials"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_DISABLED = "account_disabled"
    PASSWORD_EXPIRED = "password_expired"
    PERMISSION_DENIED = "permission_denied"


class UserManagementEngine:
    """User authentication, authorization, and session management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the user management engine."""
        self.config = config or {}
        
        # Performance targets
        self.auth_target_ms = self.config.get('auth_target_ms', 200)
        
        # Security configuration
        self.session_timeout_minutes = self.config.get('session_timeout_minutes', 30)
        self.max_login_attempts = self.config.get('max_login_attempts', 3)
        self.lockout_duration_minutes = self.config.get('lockout_duration_minutes', 15)
        self.password_min_length = self.config.get('password_min_length', 8)
        self.require_password_complexity = self.config.get('require_password_complexity', True)
        
        # Session management
        self._sessions = {}
        self._session_lock = threading.RLock()
        
        # User storage (in production, this would be a database)
        self._users = {}
        self._user_lock = threading.RLock()
        
        # Failed login tracking
        self._failed_attempts = defaultdict(lambda: {'count': 0, 'last_attempt': None})
        self._lockout_times = {}
        
        # Audit logging
        self._audit_log = []
        self._audit_lock = threading.RLock()
        
        # Role permissions mapping
        self._role_permissions = self._initialize_role_permissions()
        
        # Initialize monitoring and control integrations
        self._initialize_integrations()
        
        # Create default admin user if none exist
        self._create_default_admin()
        
        # Start session cleanup thread
        self._start_session_cleanup()
        
        # Performance metrics
        self.performance_metrics = {
            'total_authentications': 0,
            'successful_authentications': 0,
            'failed_authentications': 0,
            'average_auth_time_ms': 0.0,
            'active_sessions': 0,
            'max_concurrent_sessions': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("UserManagementEngine initialized")
    
    def _initialize_integrations(self):
        """Initialize integrations with other system engines."""
        try:
            monitoring_config = self.config.get('monitoring_config', {})
            self.monitoring_engine = MonitoringEngine(monitoring_config) if MonitoringEngine else None
            
            orchestration_config = self.config.get('orchestration_config', {})
            self.orchestration_engine = OrchestrationEngine(orchestration_config) if OrchestrationEngine else None
            
            analytics_config = self.config.get('analytics_config', {})
            self.analytics_engine = AnalyticsEngine(analytics_config) if AnalyticsEngine else None
            
        except Exception as e:
            self.logger.warning(f"Engine integration initialization failed: {e}")
            self.monitoring_engine = None
            self.orchestration_engine = None
            self.analytics_engine = None
    
    def _initialize_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Initialize the role-based permissions mapping."""
        return {
            UserRole.ADMINISTRATOR: list(Permission),  # All permissions
            
            UserRole.SUPERVISOR: [
                Permission.SYSTEM_START, Permission.SYSTEM_STOP, Permission.SYSTEM_RESET,
                Permission.EMERGENCY_STOP, Permission.CONFIG_READ, Permission.CONFIG_WRITE,
                Permission.DATA_READ, Permission.DATA_EXPORT, Permission.MONITOR_VIEW,
                Permission.MONITOR_ALERTS, Permission.MAINTENANCE_MODE, Permission.AUDIT_VIEW
            ],
            
            UserRole.ENGINEER: [
                Permission.SYSTEM_START, Permission.SYSTEM_STOP, Permission.SYSTEM_RESET,
                Permission.EMERGENCY_STOP, Permission.CONFIG_READ, Permission.CONFIG_WRITE,
                Permission.DATA_READ, Permission.DATA_EXPORT, Permission.MONITOR_VIEW,
                Permission.MONITOR_ALERTS, Permission.CALIBRATION
            ],
            
            UserRole.OPERATOR: [
                Permission.SYSTEM_START, Permission.SYSTEM_STOP, Permission.EMERGENCY_STOP,
                Permission.CONFIG_READ, Permission.DATA_READ, Permission.MONITOR_VIEW,
                Permission.MONITOR_ALERTS
            ],
            
            UserRole.MAINTENANCE: [
                Permission.SYSTEM_STOP, Permission.SYSTEM_RESET, Permission.EMERGENCY_STOP,
                Permission.CONFIG_READ, Permission.DATA_READ, Permission.MONITOR_VIEW,
                Permission.MAINTENANCE_MODE, Permission.CALIBRATION
            ],
            
            UserRole.VIEWER: [
                Permission.CONFIG_READ, Permission.DATA_READ, Permission.MONITOR_VIEW
            ],
            
            UserRole.AUDITOR: [
                Permission.CONFIG_READ, Permission.DATA_READ, Permission.MONITOR_VIEW,
                Permission.AUDIT_VIEW, Permission.AUDIT_EXPORT
            ]
        }
    
    def _create_default_admin(self):
        """Create a default administrator account if no users exist."""
        with self._user_lock:
            if not self._users:
                admin_user = {
                    'user_id': 'admin',
                    'username': 'admin',
                    'password_hash': self._hash_password('admin123'),
                    'role': UserRole.ADMINISTRATOR,
                    'email': 'admin@manufacturing.local',
                    'full_name': 'Default Administrator',
                    'created_date': datetime.now().isoformat(),
                    'last_login': None,
                    'is_active': True,
                    'password_expires': None,
                    'must_change_password': True
                }
                self._users['admin'] = admin_user
                self._log_audit_event('SYSTEM', 'user_created', 
                                    {'username': 'admin', 'role': UserRole.ADMINISTRATOR.value})
    
    def authenticate_users(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate users and establish secure sessions.
        
        Args:
            credentials: Dictionary containing username, password, and optional session info
            
        Returns:
            Authentication result with session information or error details
        """
        start_time = time.time()
        
        try:
            username = credentials.get('username', '').strip()
            password = credentials.get('password', '')
            
            if not username or not password:
                return self._auth_result(AuthenticationResult.INVALID_CREDENTIALS, 
                                       "Username and password are required")
            
            # Check if account is locked
            if self._is_account_locked(username):
                return self._auth_result(AuthenticationResult.ACCOUNT_LOCKED,
                                       f"Account locked due to too many failed attempts")
            
            # Get user from storage
            with self._user_lock:
                user = self._users.get(username)
            
            if not user or not self._verify_password(password, user['password_hash']):
                self._record_failed_attempt(username)
                return self._auth_result(AuthenticationResult.INVALID_CREDENTIALS,
                                       "Invalid username or password")
            
            # Check if account is active
            if not user.get('is_active', False):
                return self._auth_result(AuthenticationResult.ACCOUNT_DISABLED,
                                       "Account is disabled")
            
            # Check password expiration
            if self._is_password_expired(user):
                return self._auth_result(AuthenticationResult.PASSWORD_EXPIRED,
                                       "Password has expired")
            
            # Create session
            session = self._create_session(user)
            
            # Update user last login
            with self._user_lock:
                self._users[username]['last_login'] = datetime.now().isoformat()
            
            # Clear failed attempts
            if username in self._failed_attempts:
                del self._failed_attempts[username]
            if username in self._lockout_times:
                del self._lockout_times[username]
            
            # Log successful authentication
            self._log_audit_event(username, 'login_success', {
                'session_id': session['session_id'],
                'role': user['role'].value
            })
            
            # Update performance metrics
            auth_time = (time.time() - start_time) * 1000
            self._update_auth_metrics(True, auth_time)
            
            return {
                'success': True,
                'result': AuthenticationResult.SUCCESS,
                'session': session,
                'user': {
                    'username': user['username'],
                    'role': user['role'].value,
                    'full_name': user['full_name'],
                    'permissions': [p.value for p in self._role_permissions[user['role']]]
                },
                'must_change_password': user.get('must_change_password', False),
                'auth_time_ms': auth_time
            }
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            auth_time = (time.time() - start_time) * 1000
            self._update_auth_metrics(False, auth_time)
            return self._auth_result(AuthenticationResult.INVALID_CREDENTIALS, 
                                   "Authentication failed due to system error")
    
    def enforce_role_based_access(self, user_permissions: Dict[str, Any], 
                                 requested_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enforce role-based access control across system interfaces.
        
        Args:
            user_permissions: User session and permission information
            requested_actions: List of actions the user wants to perform
            
        Returns:
            Access control result with allowed/denied actions
        """
        try:
            session_id = user_permissions.get('session_id')
            if not session_id:
                return {'success': False, 'error': 'No session provided'}
            
            # Validate session
            session = self._get_session(session_id)
            if not session:
                return {'success': False, 'error': 'Invalid or expired session'}
            
            user = session['user']
            user_role = UserRole(user['role'])
            allowed_permissions = self._role_permissions.get(user_role, [])
            
            # Check each requested action
            results = []
            for action in requested_actions:
                action_type = action.get('action_type')
                required_permission = action.get('required_permission')
                
                if not required_permission:
                    results.append({
                        'action': action,
                        'allowed': False,
                        'reason': 'No permission specified for action'
                    })
                    continue
                
                try:
                    permission = Permission(required_permission)
                    allowed = permission in allowed_permissions
                    
                    results.append({
                        'action': action,
                        'allowed': allowed,
                        'reason': 'Permission granted' if allowed else 'Permission denied'
                    })
                    
                    # Log access attempt
                    self._log_audit_event(user['username'], 'permission_check', {
                        'action': action_type,
                        'permission': required_permission,
                        'result': 'allowed' if allowed else 'denied'
                    })
                    
                except ValueError:
                    results.append({
                        'action': action,
                        'allowed': False,
                        'reason': f'Invalid permission: {required_permission}'
                    })
            
            return {
                'success': True,
                'session_id': session_id,
                'user': user['username'],
                'role': user_role.value,
                'results': results,
                'all_allowed': all(r['allowed'] for r in results)
            }
            
        except Exception as e:
            self.logger.error(f"Access control error: {e}")
            return {'success': False, 'error': f'Access control failed: {str(e)}'}
    
    def manage_audit_logging(self, user_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Manage comprehensive audit logging for security and compliance.
        
        Args:
            user_actions: List of user actions to log
            
        Returns:
            Audit logging result and statistics
        """
        try:
            logged_actions = []
            
            for action in user_actions:
                username = action.get('username', 'unknown')
                action_type = action.get('action_type', 'unknown')
                details = action.get('details', {})
                
                # Add additional context
                audit_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'username': username,
                    'action_type': action_type,
                    'details': details,
                    'source_ip': action.get('source_ip', 'unknown'),
                    'session_id': action.get('session_id'),
                    'result': action.get('result', 'unknown')
                }
                
                with self._audit_lock:
                    self._audit_log.append(audit_entry)
                
                logged_actions.append(audit_entry)
                
                # Send to monitoring engine if available
                if self.monitoring_engine:
                    try:
                        self.monitoring_engine.process_alert_conditions({
                            'alert_type': 'audit_event',
                            'data': audit_entry
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to send audit event to monitoring: {e}")
            
            # Cleanup old audit entries (keep last 10000)
            with self._audit_lock:
                if len(self._audit_log) > 10000:
                    self._audit_log = self._audit_log[-10000:]
            
            return {
                'success': True,
                'logged_count': len(logged_actions),
                'total_audit_entries': len(self._audit_log),
                'actions': logged_actions
            }
            
        except Exception as e:
            self.logger.error(f"Audit logging error: {e}")
            return {'success': False, 'error': f'Audit logging failed: {str(e)}'}
    
    def create_user(self, user_data: Dict[str, Any], creating_user: str) -> Dict[str, Any]:
        """Create a new user account."""
        try:
            username = user_data.get('username', '').strip()
            password = user_data.get('password', '')
            role = user_data.get('role')
            
            # Validation
            if not username or not password:
                return {'success': False, 'error': 'Username and password are required'}
            
            if len(password) < self.password_min_length:
                return {'success': False, 'error': f'Password must be at least {self.password_min_length} characters'}
            
            if self.require_password_complexity and not self._validate_password_complexity(password):
                return {'success': False, 'error': 'Password does not meet complexity requirements'}
            
            try:
                user_role = UserRole(role) if role else UserRole.VIEWER
            except ValueError:
                return {'success': False, 'error': f'Invalid role: {role}'}
            
            with self._user_lock:
                if username in self._users:
                    return {'success': False, 'error': 'Username already exists'}
                
                user = {
                    'user_id': str(uuid.uuid4()),
                    'username': username,
                    'password_hash': self._hash_password(password),
                    'role': user_role,
                    'email': user_data.get('email', ''),
                    'full_name': user_data.get('full_name', username),
                    'created_date': datetime.now().isoformat(),
                    'created_by': creating_user,
                    'last_login': None,
                    'is_active': True,
                    'password_expires': None,
                    'must_change_password': user_data.get('must_change_password', False)
                }
                
                self._users[username] = user
            
            # Log user creation
            self._log_audit_event(creating_user, 'user_created', {
                'new_username': username,
                'role': user_role.value
            })
            
            return {
                'success': True,
                'user_id': user['user_id'],
                'username': username,
                'role': user_role.value
            }
            
        except Exception as e:
            self.logger.error(f"User creation error: {e}")
            return {'success': False, 'error': f'User creation failed: {str(e)}'}
    
    def logout_user(self, session_id: str) -> Dict[str, Any]:
        """Log out a user and terminate their session."""
        try:
            with self._session_lock:
                session = self._sessions.get(session_id)
                if not session:
                    return {'success': False, 'error': 'Invalid session'}
                
                username = session['user']['username']
                self._sessions[session_id]['status'] = SessionStatus.TERMINATED
                del self._sessions[session_id]
            
            # Log logout
            self._log_audit_event(username, 'logout', {'session_id': session_id})
            
            # Update metrics
            self.performance_metrics['active_sessions'] = len(self._sessions)
            
            return {'success': True, 'message': 'Logout successful'}
            
        except Exception as e:
            self.logger.error(f"Logout error: {e}")
            return {'success': False, 'error': f'Logout failed: {str(e)}'}
    
    def get_audit_log(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get audit log entries with optional filtering."""
        try:
            with self._audit_lock:
                logs = list(self._audit_log)
            
            if not filters:
                return logs
            
            # Apply filters
            username_filter = filters.get('username')
            action_filter = filters.get('action_type')
            start_time = filters.get('start_time')
            end_time = filters.get('end_time')
            
            filtered_logs = []
            for entry in logs:
                # Username filter
                if username_filter and entry['username'] != username_filter:
                    continue
                
                # Action type filter
                if action_filter and entry['action_type'] != action_filter:
                    continue
                
                # Time range filter
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if start_time and entry_time < datetime.fromisoformat(start_time):
                    continue
                if end_time and entry_time > datetime.fromisoformat(end_time):
                    continue
                
                filtered_logs.append(entry)
            
            return filtered_logs
            
        except Exception as e:
            self.logger.error(f"Audit log retrieval error: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get user management performance metrics."""
        with self._session_lock:
            active_sessions = len(self._sessions)
            
        return {
            **self.performance_metrics,
            'active_sessions': active_sessions,
            'total_users': len(self._users),
            'audit_log_entries': len(self._audit_log),
            'locked_accounts': len(self._lockout_times)
        }
    
    # Helper methods
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt."""
        salt = secrets.token_hex(32)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}${password_hash}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against the stored hash."""
        try:
            salt, password_hash = stored_hash.split('$')
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return password_hash == computed_hash
        except ValueError:
            return False
    
    def _validate_password_complexity(self, password: str) -> bool:
        """Validate password complexity requirements."""
        if len(password) < self.password_min_length:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if an account is locked due to failed login attempts."""
        if username not in self._lockout_times:
            return False
        
        lockout_time = self._lockout_times[username]
        unlock_time = lockout_time + timedelta(minutes=self.lockout_duration_minutes)
        
        if datetime.now() > unlock_time:
            # Unlock the account
            del self._lockout_times[username]
            if username in self._failed_attempts:
                del self._failed_attempts[username]
            return False
        
        return True
    
    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        self._failed_attempts[username]['count'] += 1
        self._failed_attempts[username]['last_attempt'] = datetime.now()
        
        if self._failed_attempts[username]['count'] >= self.max_login_attempts:
            self._lockout_times[username] = datetime.now()
            self._log_audit_event(username, 'account_locked', {
                'reason': 'too_many_failed_attempts',
                'attempts': self._failed_attempts[username]['count']
            })
    
    def _is_password_expired(self, user: Dict[str, Any]) -> bool:
        """Check if a user's password has expired."""
        password_expires = user.get('password_expires')
        if not password_expires:
            return False
        
        expiry_date = datetime.fromisoformat(password_expires)
        return datetime.now() > expiry_date
    
    def _create_session(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        session = {
            'session_id': session_id,
            'user': {
                'username': user['username'],
                'role': user['role'].value,
                'full_name': user['full_name']
            },
            'created_time': datetime.now(),
            'last_activity': datetime.now(),
            'expires_at': datetime.now() + timedelta(minutes=self.session_timeout_minutes),
            'status': SessionStatus.ACTIVE
        }
        
        with self._session_lock:
            self._sessions[session_id] = session
            
            # Update max concurrent sessions
            current_sessions = len(self._sessions)
            if current_sessions > self.performance_metrics['max_concurrent_sessions']:
                self.performance_metrics['max_concurrent_sessions'] = current_sessions
        
        return session
    
    def _get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get and validate a session."""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            # Check if session is expired
            if datetime.now() > session['expires_at']:
                session['status'] = SessionStatus.EXPIRED
                del self._sessions[session_id]
                return None
            
            # Update last activity and extend session
            session['last_activity'] = datetime.now()
            session['expires_at'] = datetime.now() + timedelta(minutes=self.session_timeout_minutes)
            
            return session
    
    def _auth_result(self, result: AuthenticationResult, message: str) -> Dict[str, Any]:
        """Create a standardized authentication result."""
        return {
            'success': result == AuthenticationResult.SUCCESS,
            'result': result,
            'message': message
        }
    
    def _log_audit_event(self, username: str, action_type: str, details: Dict[str, Any]):
        """Log an audit event."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'action_type': action_type,
            'details': details,
            'source': 'user_management_engine'
        }
        
        with self._audit_lock:
            self._audit_log.append(audit_entry)
    
    def _update_auth_metrics(self, success: bool, auth_time_ms: float):
        """Update authentication performance metrics."""
        self.performance_metrics['total_authentications'] += 1
        if success:
            self.performance_metrics['successful_authentications'] += 1
        else:
            self.performance_metrics['failed_authentications'] += 1
        
        # Update average authentication time
        total_auths = self.performance_metrics['total_authentications']
        current_avg = self.performance_metrics['average_auth_time_ms']
        new_avg = ((current_avg * (total_auths - 1)) + auth_time_ms) / total_auths
        self.performance_metrics['average_auth_time_ms'] = new_avg
    
    def _start_session_cleanup(self):
        """Start the session cleanup thread."""
        def cleanup_sessions():
            while True:
                try:
                    time.sleep(60)  # Check every minute
                    
                    with self._session_lock:
                        expired_sessions = []
                        for session_id, session in self._sessions.items():
                            if datetime.now() > session['expires_at']:
                                expired_sessions.append(session_id)
                        
                        for session_id in expired_sessions:
                            username = self._sessions[session_id]['user']['username']
                            self._sessions[session_id]['status'] = SessionStatus.EXPIRED
                            del self._sessions[session_id]
                            
                            # Log session expiry
                            self._log_audit_event(username, 'session_expired', {
                                'session_id': session_id
                            })
                
                except Exception as e:
                    self.logger.error(f"Session cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
        cleanup_thread.start()