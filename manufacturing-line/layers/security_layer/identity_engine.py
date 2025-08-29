#!/usr/bin/env python3
"""
IdentityEngine - Week 9 Security & Compliance Layer
Advanced identity management with zero-trust architecture
"""

import time
import json
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
import base64
import hmac
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthenticationMethod(Enum):
    """Supported authentication methods"""
    PASSWORD = "password"
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC = "biometric"
    HARDWARE_TOKEN = "hardware_token"

class AuthorizationDecision(Enum):
    """Authorization decisions"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"  # Requires additional verification

class PrivilegeLevel(Enum):
    """Privilege levels for access control"""
    READ_ONLY = "read_only"
    STANDARD = "standard"
    ELEVATED = "elevated"
    ADMINISTRATIVE = "administrative"
    SYSTEM = "system"

@dataclass
class User:
    """User identity and profile"""
    user_id: str
    username: str
    email: str
    full_name: str
    roles: List[str] = field(default_factory=list)
    privilege_level: PrivilegeLevel = PrivilegeLevel.STANDARD
    mfa_enabled: bool = False
    mfa_methods: List[AuthenticationMethod] = field(default_factory=list)
    last_login: Optional[str] = None
    failed_attempts: int = 0
    account_locked: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AuthenticationRequest:
    """Authentication request"""
    user_id: str
    primary_credential: str  # Password hash
    mfa_token: Optional[str] = None
    method: AuthenticationMethod = AuthenticationMethod.PASSWORD
    source_ip: str = "unknown"
    user_agent: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AuthorizationRequest:
    """Authorization request for zero-trust evaluation"""
    user_id: str
    resource: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AccessSession:
    """User access session"""
    session_id: str
    user_id: str
    created_at: str
    expires_at: str
    source_ip: str
    privileges: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())

class IdentityEngine:
    """Advanced identity management with zero-trust architecture
    
    Week 9 Performance Targets:
    - Authentication: <100ms
    - Authorization decisions: <50ms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize IdentityEngine with configuration"""
        self.config = config or {}
        
        # Performance targets
        self.auth_target_ms = 100
        self.authz_target_ms = 50
        
        # State management
        self.users = {}
        self.sessions = {}
        self.access_policies = {}
        self.mfa_secrets = {}
        self.privilege_requests = []
        
        # Initialize default policies
        self._initialize_default_policies()
        
        # Initialize sample users
        self._initialize_sample_users()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize data protection engine if available
        self.data_protection_engine = None
        try:
            from layers.security_layer.data_protection_engine import DataProtectionEngine
            self.data_protection_engine = DataProtectionEngine(config.get('data_protection_config', {}))
        except ImportError:
            logger.warning("DataProtectionEngine not available - using mock interface")
        
        logger.info("IdentityEngine initialized with MFA and zero-trust architecture")
    
    def authenticate_with_mfa(self, auth_request: AuthenticationRequest) -> Dict[str, Any]:
        """Authenticate users with multi-factor authentication
        
        Args:
            auth_request: Authentication request with credentials
            
        Returns:
            Authentication result with performance metrics
        """
        start_time = time.time()
        
        try:
            # Validate user exists
            user = self.users.get(auth_request.user_id)
            if not user:
                return self._create_auth_result("failed", "User not found", start_time)
            
            # Check if account is locked
            if user.account_locked:
                return self._create_auth_result("locked", "Account locked due to failed attempts", start_time)
            
            # Primary authentication (password)
            primary_auth_result = self._verify_primary_credential(user, auth_request.primary_credential)
            if not primary_auth_result:
                user.failed_attempts += 1
                if user.failed_attempts >= 5:
                    user.account_locked = True
                return self._create_auth_result("failed", "Invalid primary credential", start_time)
            
            # Multi-factor authentication if enabled
            if user.mfa_enabled:
                if not auth_request.mfa_token:
                    return self._create_auth_result("mfa_required", "MFA token required", start_time)
                
                mfa_result = self._verify_mfa_token(user, auth_request.mfa_token, auth_request.method)
                if not mfa_result:
                    return self._create_auth_result("failed", "Invalid MFA token", start_time)
            
            # Risk assessment
            risk_score = self._assess_authentication_risk(auth_request)
            
            # Create session if authentication successful
            session = self._create_access_session(user, auth_request.source_ip, risk_score)
            
            # Update user login information
            user.last_login = datetime.now().isoformat()
            user.failed_attempts = 0
            
            # Calculate authentication time
            auth_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'status': 'success',
                'user_id': user.user_id,
                'session_id': session.session_id,
                'auth_time_ms': round(auth_time_ms, 2),
                'target_met': auth_time_ms < self.auth_target_ms,
                'mfa_used': user.mfa_enabled,
                'risk_score': risk_score,
                'privileges': session.privileges,
                'expires_at': session.expires_at,
                'authenticated_at': datetime.now().isoformat()
            }
            
            logger.info(f"User {user.username} authenticated successfully in {auth_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return self._create_auth_result("error", str(e), start_time)
    
    def authorize_zero_trust(self, access_request: AuthorizationRequest) -> Dict[str, Any]:
        """Authorize access using zero-trust principles
        
        Args:
            access_request: Authorization request
            
        Returns:
            Authorization decision with performance metrics
        """
        start_time = time.time()
        
        try:
            # Get user and active session
            user = self.users.get(access_request.user_id)
            if not user:
                return self._create_authz_result(AuthorizationDecision.DENY, "User not found", start_time)
            
            active_session = self._get_active_session(access_request.user_id)
            if not active_session:
                return self._create_authz_result(AuthorizationDecision.DENY, "No active session", start_time)
            
            # Zero-trust evaluation factors
            evaluation_factors = {
                'user_context': self._evaluate_user_context(user, access_request),
                'resource_sensitivity': self._evaluate_resource_sensitivity(access_request.resource),
                'network_context': self._evaluate_network_context(access_request.context),
                'behavioral_analysis': self._evaluate_user_behavior(user, access_request),
                'risk_assessment': self._evaluate_access_risk(access_request)
            }
            
            # Calculate authorization score
            authz_score = self._calculate_authorization_score(evaluation_factors)
            
            # Make authorization decision
            decision = self._make_authorization_decision(authz_score, evaluation_factors)
            
            # Log authorization attempt
            self._log_authorization_attempt(access_request, decision, authz_score)
            
            # Calculate authorization time
            authz_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'decision': decision.value,
                'user_id': user.user_id,
                'resource': access_request.resource,
                'action': access_request.action,
                'authz_time_ms': round(authz_time_ms, 2),
                'target_met': authz_time_ms < self.authz_target_ms,
                'authorization_score': authz_score,
                'evaluation_factors': evaluation_factors,
                'additional_verification_required': decision == AuthorizationDecision.CONDITIONAL,
                'authorized_at': datetime.now().isoformat()
            }
            
            if decision == AuthorizationDecision.CONDITIONAL:
                result['verification_methods'] = self._get_additional_verification_methods(evaluation_factors)
            
            logger.info(f"Authorization decision for {user.username}: {decision.value} in {authz_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error during authorization: {e}")
            return self._create_authz_result(AuthorizationDecision.DENY, str(e), start_time)
    
    def manage_privileged_access(self, privilege_requests: Dict[str, Any]) -> Dict[str, Any]:
        """Manage privileged access with just-in-time provisioning
        
        Args:
            privilege_requests: Privileged access requests
            
        Returns:
            Privilege management results
        """
        start_time = time.time()
        
        try:
            # Parse privilege requests
            user_id = privilege_requests['user_id']
            requested_privileges = privilege_requests['privileges']
            justification = privilege_requests.get('justification', '')
            duration_hours = privilege_requests.get('duration_hours', 4)
            
            # Validate user
            user = self.users.get(user_id)
            if not user:
                return {'status': 'failed', 'reason': 'User not found'}
            
            # Evaluate privilege request
            privilege_evaluation = self._evaluate_privilege_request(user, requested_privileges, justification)
            
            if privilege_evaluation['approved']:
                # Grant temporary privileges
                session = self._get_active_session(user_id)
                if session:
                    # Add privileges to existing session
                    session.privileges.extend(requested_privileges)
                    session.expires_at = (datetime.now() + timedelta(hours=duration_hours)).isoformat()
                
                # Record privilege grant
                privilege_record = {
                    'request_id': f"PRIV_{int(time.time() * 1000)}",
                    'user_id': user_id,
                    'privileges_granted': requested_privileges,
                    'granted_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(hours=duration_hours)).isoformat(),
                    'justification': justification,
                    'approver': 'system',  # In real implementation, would be actual approver
                    'status': 'active'
                }
                
                self.privilege_requests.append(privilege_record)
                
                result = {
                    'status': 'approved',
                    'request_id': privilege_record['request_id'],
                    'privileges_granted': requested_privileges,
                    'duration_hours': duration_hours,
                    'expires_at': privilege_record['expires_at'],
                    'approval_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            else:
                result = {
                    'status': 'denied',
                    'reason': privilege_evaluation['reason'],
                    'approval_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            
            logger.info(f"Privilege request for {user.username}: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Error managing privileged access: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke user access session"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Session revoked: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error revoking session: {e}")
            return False
    
    def setup_mfa(self, user_id: str, method: AuthenticationMethod) -> Dict[str, Any]:
        """Setup multi-factor authentication for user"""
        try:
            user = self.users.get(user_id)
            if not user:
                return {'status': 'failed', 'reason': 'User not found'}
            
            # Generate MFA secret
            mfa_secret = secrets.token_hex(20)
            
            # Store MFA configuration
            self.mfa_secrets[user_id] = {
                'secret': mfa_secret,
                'method': method,
                'created_at': datetime.now().isoformat(),
                'backup_codes': [secrets.token_hex(8) for _ in range(5)]
            }
            
            # Update user MFA settings
            user.mfa_enabled = True
            if method not in user.mfa_methods:
                user.mfa_methods.append(method)
            
            result = {
                'status': 'success',
                'method': method.value,
                'secret': mfa_secret,
                'backup_codes': self.mfa_secrets[user_id]['backup_codes'],
                'qr_code_data': f"otpauth://totp/{user.email}?secret={mfa_secret}&issuer=ManufacturingLine"
            }
            
            logger.info(f"MFA setup completed for {user.username}")
            return result
            
        except Exception as e:
            logger.error(f"Error setting up MFA: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def _initialize_default_policies(self):
        """Initialize default access policies"""
        self.access_policies = {
            'admin_resources': {
                'resources': ['/admin/*', '/config/*', '/users/*'],
                'required_role': 'administrator',
                'mfa_required': True,
                'risk_threshold': 0.3
            },
            'sensitive_data': {
                'resources': ['/data/sensitive/*', '/reports/confidential/*'],
                'required_role': 'analyst',
                'mfa_required': True,
                'risk_threshold': 0.5
            },
            'standard_access': {
                'resources': ['/dashboard/*', '/reports/standard/*'],
                'required_role': 'user',
                'mfa_required': False,
                'risk_threshold': 0.8
            }
        }
    
    def _initialize_sample_users(self):
        """Initialize sample users for demonstration"""
        sample_users = [
            User(
                user_id="user_001",
                username="admin",
                email="admin@company.com",
                full_name="System Administrator",
                roles=["administrator", "user"],
                privilege_level=PrivilegeLevel.ADMINISTRATIVE,
                mfa_enabled=True,
                mfa_methods=[AuthenticationMethod.TOTP, AuthenticationMethod.EMAIL]
            ),
            User(
                user_id="user_002", 
                username="analyst",
                email="analyst@company.com",
                full_name="Security Analyst",
                roles=["analyst", "user"],
                privilege_level=PrivilegeLevel.ELEVATED,
                mfa_enabled=True,
                mfa_methods=[AuthenticationMethod.TOTP]
            ),
            User(
                user_id="user_003",
                username="operator",
                email="operator@company.com", 
                full_name="System Operator",
                roles=["user"],
                privilege_level=PrivilegeLevel.STANDARD,
                mfa_enabled=False
            )
        ]
        
        for user in sample_users:
            self.users[user.user_id] = user
    
    def _verify_primary_credential(self, user: User, credential: str) -> bool:
        """Verify primary authentication credential (password)"""
        # In real implementation, would hash and compare
        # For demo, accept any non-empty credential
        return len(credential) > 0
    
    def _verify_mfa_token(self, user: User, token: str, method: AuthenticationMethod) -> bool:
        """Verify MFA token"""
        # Simplified MFA verification for demonstration
        if method == AuthenticationMethod.TOTP:
            # For demo, accept 6-digit numeric tokens
            return token.isdigit() and len(token) == 6
        elif method == AuthenticationMethod.SMS:
            # For demo, accept 4-digit codes
            return token.isdigit() and len(token) == 4
        return False
    
    def _assess_authentication_risk(self, auth_request: AuthenticationRequest) -> float:
        """Assess risk factors for authentication"""
        risk_factors = []
        
        # Check source IP risk
        if auth_request.source_ip.startswith('10.'):
            risk_factors.append(0.1)  # Internal network - low risk
        else:
            risk_factors.append(0.3)  # External - higher risk
        
        # Check time of access
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            risk_factors.append(0.1)
        else:
            risk_factors.append(0.2)
        
        # Calculate overall risk score
        return min(sum(risk_factors), 1.0)
    
    def _create_access_session(self, user: User, source_ip: str, risk_score: float) -> AccessSession:
        """Create new access session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=8)  # 8-hour session
        
        session = AccessSession(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            source_ip=source_ip,
            privileges=user.roles.copy(),
            risk_score=risk_score
        )
        
        self.sessions[session_id] = session
        return session
    
    def _get_active_session(self, user_id: str) -> Optional[AccessSession]:
        """Get active session for user"""
        for session in self.sessions.values():
            if session.user_id == user_id:
                # Check if session is still valid
                expires_at = datetime.fromisoformat(session.expires_at)
                if expires_at > datetime.now():
                    return session
        return None
    
    def _evaluate_user_context(self, user: User, request: AuthorizationRequest) -> Dict[str, Any]:
        """Evaluate user context for zero-trust decision"""
        return {
            'privilege_level': user.privilege_level.value,
            'roles': user.roles,
            'mfa_enabled': user.mfa_enabled,
            'recent_activity': 'normal',  # Simplified for demo
            'risk_indicators': []
        }
    
    def _evaluate_resource_sensitivity(self, resource: str) -> Dict[str, Any]:
        """Evaluate resource sensitivity level"""
        if '/admin/' in resource or '/config/' in resource:
            sensitivity = 'high'
        elif '/sensitive/' in resource:
            sensitivity = 'medium'
        else:
            sensitivity = 'low'
        
        return {
            'sensitivity_level': sensitivity,
            'access_restrictions': sensitivity != 'low',
            'audit_required': sensitivity in ['high', 'medium']
        }
    
    def _evaluate_network_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate network context"""
        source_ip = context.get('source_ip', 'unknown')
        
        # Simple IP classification for demo
        if source_ip.startswith('192.168.') or source_ip.startswith('10.'):
            network_trust = 'trusted'
        else:
            network_trust = 'untrusted'
        
        return {
            'network_trust': network_trust,
            'geolocation_check': 'passed',  # Simplified
            'vpn_detected': False
        }
    
    def _evaluate_user_behavior(self, user: User, request: AuthorizationRequest) -> Dict[str, Any]:
        """Evaluate user behavior patterns"""
        return {
            'behavior_score': 0.8,  # Normal behavior
            'anomaly_detected': False,
            'access_pattern': 'consistent',
            'deviation_score': 0.1
        }
    
    def _evaluate_access_risk(self, request: AuthorizationRequest) -> Dict[str, Any]:
        """Evaluate overall access risk"""
        risk_factors = len(request.risk_factors)
        base_risk = 0.2 + (risk_factors * 0.1)
        
        return {
            'risk_score': min(base_risk, 1.0),
            'risk_factors': request.risk_factors,
            'mitigation_required': base_risk > 0.5
        }
    
    def _calculate_authorization_score(self, factors: Dict[str, Any]) -> float:
        """Calculate authorization score based on evaluation factors"""
        # Simplified scoring algorithm
        base_score = 0.7
        
        # Adjust based on user context
        if factors['user_context']['mfa_enabled']:
            base_score += 0.1
        
        # Adjust based on resource sensitivity
        if factors['resource_sensitivity']['sensitivity_level'] == 'high':
            base_score -= 0.2
        
        # Adjust based on network trust
        if factors['network_context']['network_trust'] == 'trusted':
            base_score += 0.1
        
        # Adjust based on behavior
        base_score += factors['behavioral_analysis']['behavior_score'] * 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _make_authorization_decision(self, score: float, factors: Dict[str, Any]) -> AuthorizationDecision:
        """Make authorization decision based on score and factors"""
        if score >= 0.8:
            return AuthorizationDecision.ALLOW
        elif score >= 0.5:
            return AuthorizationDecision.CONDITIONAL
        else:
            return AuthorizationDecision.DENY
    
    def _log_authorization_attempt(self, request: AuthorizationRequest, decision: AuthorizationDecision, score: float):
        """Log authorization attempt for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': request.user_id,
            'resource': request.resource,
            'action': request.action,
            'decision': decision.value,
            'authorization_score': score,
            'context': request.context
        }
        logger.debug(f"Authorization logged: {log_entry}")
    
    def _get_additional_verification_methods(self, factors: Dict[str, Any]) -> List[str]:
        """Get additional verification methods for conditional access"""
        methods = []
        
        if factors['resource_sensitivity']['sensitivity_level'] == 'high':
            methods.append('manager_approval')
        
        if factors['network_context']['network_trust'] == 'untrusted':
            methods.append('additional_mfa')
        
        if not methods:
            methods.append('security_question')
        
        return methods
    
    def _evaluate_privilege_request(self, user: User, privileges: List[str], justification: str) -> Dict[str, Any]:
        """Evaluate privileged access request"""
        # Simplified evaluation logic
        approval_score = 0.5
        
        # User's current privilege level affects approval
        if user.privilege_level in [PrivilegeLevel.ELEVATED, PrivilegeLevel.ADMINISTRATIVE]:
            approval_score += 0.3
        
        # Justification quality (simplified check)
        if len(justification) > 50:
            approval_score += 0.2
        
        # Check if privileges are reasonable for user role
        admin_privileges = ['user_management', 'system_config', 'security_admin']
        if any(priv in admin_privileges for priv in privileges):
            if 'administrator' not in user.roles:
                approval_score -= 0.4
        
        approved = approval_score >= 0.6
        
        return {
            'approved': approved,
            'score': approval_score,
            'reason': 'Request approved based on user role and justification' if approved else 'Insufficient justification or inappropriate privileges requested'
        }
    
    def _create_auth_result(self, status: str, message: str, start_time: float) -> Dict[str, Any]:
        """Create authentication result with timing"""
        auth_time_ms = (time.time() - start_time) * 1000
        return {
            'status': status,
            'message': message,
            'auth_time_ms': round(auth_time_ms, 2),
            'target_met': auth_time_ms < self.auth_target_ms,
            'authenticated_at': datetime.now().isoformat()
        }
    
    def _create_authz_result(self, decision: AuthorizationDecision, reason: str, start_time: float) -> Dict[str, Any]:
        """Create authorization result with timing"""
        authz_time_ms = (time.time() - start_time) * 1000
        return {
            'decision': decision.value,
            'reason': reason,
            'authz_time_ms': round(authz_time_ms, 2),
            'target_met': authz_time_ms < self.authz_target_ms,
            'authorization_score': 0.0,  # Default score for error cases
            'authorized_at': datetime.now().isoformat()
        }
    
    def demonstrate_identity_capabilities(self) -> Dict[str, Any]:
        """Demonstrate identity management capabilities"""
        print("\n👤 IDENTITY ENGINE DEMONSTRATION 👤")
        print("=" * 50)
        
        # Setup MFA for a user
        print("🔐 Setting up Multi-Factor Authentication...")
        mfa_setup = self.setup_mfa("user_001", AuthenticationMethod.TOTP)
        print(f"   ✅ MFA setup: {mfa_setup['status'].upper()}")
        print(f"   📱 Method: {mfa_setup['method']}")
        
        # Authenticate with MFA
        print("\n🔑 Authenticating with MFA...")
        auth_request = AuthenticationRequest(
            user_id="user_001",
            primary_credential="secure_password",
            mfa_token="123456",  # Valid 6-digit TOTP
            method=AuthenticationMethod.TOTP,
            source_ip="192.168.1.100"
        )
        auth_result = self.authenticate_with_mfa(auth_request)
        print(f"   ✅ Authentication: {auth_result['status'].upper()}")
        print(f"   ⏱️ Auth time: {auth_result['auth_time_ms']}ms")
        print(f"   🎯 Target: <{self.auth_target_ms}ms | {'✅ MET' if auth_result['target_met'] else '❌ MISSED'}")
        print(f"   🔒 MFA used: {'✅' if auth_result['mfa_used'] else '❌'}")
        
        # Zero-trust authorization
        print("\n🛡️ Zero-Trust Authorization...")
        authz_request = AuthorizationRequest(
            user_id="user_001",
            resource="/admin/user_management",
            action="READ",
            context={'source_ip': '192.168.1.100'},
            risk_factors=[]
        )
        authz_result = self.authorize_zero_trust(authz_request)
        print(f"   ✅ Authorization: {authz_result['decision'].upper()}")
        print(f"   ⏱️ Authz time: {authz_result['authz_time_ms']}ms")
        print(f"   🎯 Target: <{self.authz_target_ms}ms | {'✅ MET' if authz_result['target_met'] else '❌ MISSED'}")
        print(f"   📊 Authorization score: {authz_result['authorization_score']:.2f}")
        
        # Privileged access management
        print("\n🔐 Privileged Access Management...")
        privilege_request = {
            'user_id': 'user_002',
            'privileges': ['database_admin', 'system_restart'],
            'justification': 'Emergency maintenance required for critical system upgrade',
            'duration_hours': 2
        }
        privilege_result = self.manage_privileged_access(privilege_request)
        print(f"   ✅ Privilege request: {privilege_result['status'].upper()}")
        if privilege_result['status'] == 'approved':
            print(f"   ⏰ Duration: {privilege_result['duration_hours']} hours")
            print(f"   🔑 Privileges: {', '.join(privilege_result['privileges_granted'])}")
        
        # Session management
        print(f"\n📊 Active Sessions: {len(self.sessions)}")
        for session_id, session in list(self.sessions.items())[:3]:
            user = self.users[session.user_id]
            print(f"   👤 {user.username}: {session.session_id[:8]}... (Risk: {session.risk_score:.2f})")
        
        print("\n📈 DEMONSTRATION SUMMARY:")
        print(f"   Users Configured: {len(self.users)}")
        print(f"   MFA Setup Time: <1s")
        print(f"   Authentication Time: {auth_result['auth_time_ms']}ms") 
        print(f"   Authorization Time: {authz_result['authz_time_ms']}ms")
        print(f"   Active Sessions: {len(self.sessions)}")
        print(f"   Privilege Requests: {len(self.privilege_requests)}")
        print("=" * 50)
        
        return {
            'users_configured': len(self.users),
            'authentication_time_ms': auth_result['auth_time_ms'],
            'authorization_time_ms': authz_result['authz_time_ms'],
            'active_sessions': len(self.sessions),
            'privilege_requests': len(self.privilege_requests),
            'mfa_enabled_users': len([u for u in self.users.values() if u.mfa_enabled]),
            'performance_targets_met': auth_result['target_met'] and authz_result['target_met']
        }

def main():
    """Test IdentityEngine functionality"""
    engine = IdentityEngine()
    results = engine.demonstrate_identity_capabilities()
    
    print(f"\n🎯 Week 9 Identity Performance Targets:")
    print(f"   Authentication: <100ms ({'✅' if results['authentication_time_ms'] < 100 else '❌'})")
    print(f"   Authorization: <50ms ({'✅' if results['authorization_time_ms'] < 50 else '❌'})")
    print(f"   Overall Performance: {'🟢 EXCELLENT' if results['performance_targets_met'] else '🟡 NEEDS OPTIMIZATION'}")

if __name__ == "__main__":
    main()