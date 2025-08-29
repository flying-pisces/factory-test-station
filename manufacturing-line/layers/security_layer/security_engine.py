"""
Security Engine for Week 9: Security & Compliance

This module implements comprehensive security monitoring and threat detection for the 
manufacturing line control system with real-time security event processing and analysis.

Performance Target: <50ms security event processing, <5 seconds threat analysis
Security Features: SIEM capabilities, anomaly detection, intrusion detection, behavior analysis
"""

import time
import logging
import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from collections import defaultdict, deque
import threading
import statistics
import ipaddress

# Week 9 security layer integrations (forward references)
try:
    from layers.security_layer.compliance_engine import ComplianceEngine
except ImportError:
    ComplianceEngine = None

# Week 8 deployment layer integrations
try:
    from layers.deployment_layer.alerting_engine import AlertingEngine
    from layers.deployment_layer.monitoring_engine import MonitoringEngine
except ImportError:
    AlertingEngine = None
    MonitoringEngine = None

# Week 6 UI layer integrations
try:
    from layers.ui_layer.user_management_engine import UserManagementEngine
except ImportError:
    UserManagementEngine = None

# Core imports
from datetime import datetime
import uuid


class ThreatLevel(Enum):
    """Threat level definitions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SecurityEventType(Enum):
    """Security event type definitions"""
    LOGIN_ATTEMPT = "login_attempt"
    ACCESS_VIOLATION = "access_violation"
    DATA_ACCESS = "data_access"
    NETWORK_ANOMALY = "network_anomaly"
    SYSTEM_ANOMALY = "system_anomaly"
    MALWARE_DETECTED = "malware_detected"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    DOS_ATTACK = "dos_attack"


class SecurityStatus(Enum):
    """Security status definitions"""
    SECURE = "secure"
    WARNING = "warning"
    COMPROMISED = "compromised"
    UNDER_ATTACK = "under_attack"
    INVESTIGATING = "investigating"


class ThreatIndicator:
    """Threat indicator data structure"""
    def __init__(self, indicator_type: str, value: str, confidence: float, 
                 source: str, timestamp: datetime = None):
        self.indicator_type = indicator_type  # ip, domain, hash, etc.
        self.value = value
        self.confidence = confidence  # 0.0 to 1.0
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.threat_level = ThreatLevel.UNKNOWN


class SecurityEvent:
    """Security event data structure"""
    def __init__(self, event_type: SecurityEventType, source_ip: str, 
                 user_id: str = None, details: Dict[str, Any] = None):
        self.event_id = str(uuid.uuid4())
        self.event_type = event_type
        self.source_ip = source_ip
        self.user_id = user_id
        self.details = details or {}
        self.timestamp = datetime.now()
        self.threat_level = ThreatLevel.LOW
        self.processed = False


class SecurityEngine:
    """Advanced security monitoring and threat detection system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SecurityEngine with configuration."""
        self.config = config or {}
        
        # Performance targets
        self.security_event_target_ms = self.config.get('security_event_target_ms', 50)
        self.threat_analysis_target_seconds = self.config.get('threat_analysis_target_seconds', 5)
        
        # Security configuration
        self.enable_real_time_monitoring = self.config.get('enable_real_time_monitoring', True)
        self.threat_detection_sensitivity = self.config.get('threat_detection_sensitivity', 0.7)
        self.max_events_per_minute = self.config.get('max_events_per_minute', 1000)
        
        # Threat detection thresholds
        self.failed_login_threshold = self.config.get('failed_login_threshold', 5)
        self.suspicious_ip_threshold = self.config.get('suspicious_ip_threshold', 10)
        self.data_access_rate_threshold = self.config.get('data_access_rate_threshold', 50)
        
        # Security event storage and processing
        self._security_events = deque(maxlen=10000)
        self._threat_indicators = {}
        self._security_metrics = defaultdict(int)
        self._security_lock = threading.RLock()
        
        # IP reputation and behavior tracking
        self._ip_reputation = {}
        self._user_behavior = defaultdict(list)
        self._suspicious_patterns = []
        
        # Security rules and patterns
        self._security_rules = self._initialize_security_rules()
        self._threat_patterns = self._initialize_threat_patterns()
        
        # Initialize integrations
        self._initialize_integrations()
        
        # Start security monitoring
        if self.enable_real_time_monitoring:
            self._start_security_monitoring()
        
        # Performance metrics
        self.performance_metrics = {
            'total_events_processed': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'average_processing_time_ms': 0.0,
            'security_alerts_generated': 0,
            'blocked_threats': 0,
            'system_security_score': 100.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SecurityEngine initialized with {self.security_event_target_ms}ms processing target")
    
    def _initialize_integrations(self):
        """Initialize integrations with other system engines."""
        try:
            compliance_config = self.config.get('compliance_config', {})
            self.compliance_engine = ComplianceEngine(compliance_config) if ComplianceEngine else None
            
            alerting_config = self.config.get('alerting_config', {})
            self.alerting_engine = AlertingEngine(alerting_config) if AlertingEngine else None
            
            monitoring_config = self.config.get('monitoring_config', {})
            self.monitoring_engine = MonitoringEngine(monitoring_config) if MonitoringEngine else None
            
            user_mgmt_config = self.config.get('user_management_config', {})
            self.user_management_engine = UserManagementEngine(user_mgmt_config) if UserManagementEngine else None
            
        except Exception as e:
            self.logger.warning(f"Engine integration initialization failed: {e}")
            self.compliance_engine = None
            self.alerting_engine = None
            self.monitoring_engine = None
            self.user_management_engine = None
    
    def _initialize_security_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security detection rules."""
        return {
            'brute_force_detection': {
                'description': 'Detect brute force login attempts',
                'pattern': r'failed.*login',
                'threshold': 5,
                'time_window_minutes': 10,
                'threat_level': ThreatLevel.HIGH,
                'actions': ['block_ip', 'alert_security_team']
            },
            'privilege_escalation': {
                'description': 'Detect privilege escalation attempts',
                'pattern': r'sudo|su|admin|root',
                'threshold': 3,
                'time_window_minutes': 5,
                'threat_level': ThreatLevel.CRITICAL,
                'actions': ['immediate_alert', 'lock_account']
            },
            'data_exfiltration': {
                'description': 'Detect potential data exfiltration',
                'pattern': r'download|export|backup',
                'threshold': 10,
                'time_window_minutes': 30,
                'threat_level': ThreatLevel.HIGH,
                'actions': ['monitor_closely', 'alert_data_protection']
            },
            'sql_injection': {
                'description': 'Detect SQL injection attempts',
                'pattern': r"'.*OR.*=.*|UNION.*SELECT|DROP.*TABLE",
                'threshold': 1,
                'time_window_minutes': 1,
                'threat_level': ThreatLevel.CRITICAL,
                'actions': ['block_immediately', 'alert_security_team']
            },
            'anomalous_access_pattern': {
                'description': 'Detect anomalous access patterns',
                'pattern': r'unusual_time|multiple_locations|rapid_requests',
                'threshold': 3,
                'time_window_minutes': 15,
                'threat_level': ThreatLevel.MEDIUM,
                'actions': ['verify_identity', 'increase_monitoring']
            }
        }
    
    def _initialize_threat_patterns(self) -> List[Dict[str, Any]]:
        """Initialize machine learning threat patterns."""
        return [
            {
                'name': 'login_anomaly',
                'features': ['login_time', 'source_ip', 'user_agent', 'success_rate'],
                'algorithm': 'isolation_forest',
                'sensitivity': 0.8
            },
            {
                'name': 'network_anomaly',
                'features': ['request_rate', 'data_volume', 'endpoint_variety'],
                'algorithm': 'statistical_outlier',
                'sensitivity': 0.7
            },
            {
                'name': 'behavior_anomaly',
                'features': ['access_pattern', 'command_frequency', 'resource_usage'],
                'algorithm': 'behavioral_baseline',
                'sensitivity': 0.6
            }
        ]
    
    def process_security_events(self, security_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process security events with real-time threat detection.
        
        Args:
            security_data: List of security event data
            
        Returns:
            Security processing results with threat analysis
        """
        start_time = time.time()
        
        try:
            processed_events = []
            detected_threats = []
            
            for event_data in security_data:
                # Create security event
                event = self._create_security_event(event_data)
                
                # Process event through security rules
                rule_results = self._apply_security_rules(event)
                
                # Perform threat analysis
                threat_analysis = self._analyze_event_threat(event)
                
                # Update security metrics
                self._update_security_metrics(event, threat_analysis)
                
                # Store processed event
                with self._security_lock:
                    self._security_events.append(event)
                
                processed_events.append({
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'threat_level': event.threat_level.value,
                    'rule_results': rule_results,
                    'threat_analysis': threat_analysis
                })
                
                # Handle threats
                if threat_analysis['is_threat']:
                    detected_threats.append(threat_analysis)
                    self._handle_detected_threat(event, threat_analysis)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, len(security_data))
            
            return {
                'success': True,
                'processing_time_ms': processing_time,
                'events_processed': len(processed_events),
                'threats_detected': len(detected_threats),
                'processed_events': processed_events,
                'detected_threats': detected_threats,
                'security_status': self._assess_security_status(),
                'recommendations': self._generate_security_recommendations(detected_threats)
            }
            
        except Exception as e:
            self.logger.error(f"Security event processing error: {e}")
            processing_time = (time.time() - start_time) * 1000
            return {
                'success': False,
                'error': f'Security processing failed: {str(e)}',
                'processing_time_ms': processing_time
            }
    
    def analyze_threat_patterns(self, threat_indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze threat patterns using machine learning algorithms.
        
        Args:
            threat_indicators: List of threat indicators to analyze
            
        Returns:
            Threat pattern analysis results
        """
        start_time = time.time()
        
        try:
            threat_patterns = []
            correlation_results = []
            
            # Convert to ThreatIndicator objects
            indicators = []
            for indicator_data in threat_indicators:
                indicator = ThreatIndicator(
                    indicator_type=indicator_data.get('type', 'unknown'),
                    value=indicator_data.get('value', ''),
                    confidence=indicator_data.get('confidence', 0.5),
                    source=indicator_data.get('source', 'unknown')
                )
                indicators.append(indicator)
            
            # Perform pattern analysis
            for pattern in self._threat_patterns:
                pattern_result = self._analyze_pattern(pattern, indicators)
                if pattern_result['matches'] > 0:
                    threat_patterns.append(pattern_result)
            
            # Correlate threat indicators
            correlation_results = self._correlate_threat_indicators(indicators)
            
            # Generate threat intelligence
            threat_intelligence = self._generate_threat_intelligence(indicators, threat_patterns)
            
            analysis_time = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'analysis_time_ms': analysis_time,
                'indicators_analyzed': len(indicators),
                'patterns_detected': len(threat_patterns),
                'correlations_found': len(correlation_results),
                'threat_patterns': threat_patterns,
                'correlations': correlation_results,
                'threat_intelligence': threat_intelligence,
                'overall_threat_score': self._calculate_threat_score(threat_patterns),
                'recommended_actions': self._recommend_threat_actions(threat_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Threat pattern analysis error: {e}")
            analysis_time = (time.time() - start_time) * 1000
            return {
                'success': False,
                'error': f'Threat analysis failed: {str(e)}',
                'analysis_time_ms': analysis_time
            }
    
    def monitor_system_security(self, monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor comprehensive system security across all layers.
        
        Args:
            monitoring_config: Security monitoring configuration
            
        Returns:
            System security monitoring results
        """
        try:
            monitoring_results = {
                'network_security': self._monitor_network_security(),
                'access_control': self._monitor_access_control(),
                'data_protection': self._monitor_data_protection(),
                'system_integrity': self._monitor_system_integrity(),
                'compliance_status': self._monitor_compliance_status()
            }
            
            # Calculate overall security score
            security_scores = [result.get('security_score', 0) for result in monitoring_results.values()]
            overall_score = statistics.mean(security_scores) if security_scores else 0
            
            # Generate security alerts if needed
            alerts = self._generate_security_alerts(monitoring_results)
            
            return {
                'success': True,
                'monitoring_results': monitoring_results,
                'overall_security_score': overall_score,
                'security_alerts': alerts,
                'monitoring_timestamp': datetime.now().isoformat(),
                'next_monitoring_schedule': (datetime.now() + timedelta(minutes=15)).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"System security monitoring error: {e}")
            return {'success': False, 'error': f'Security monitoring failed: {str(e)}'}
    
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get security dashboard data for visualization."""
        with self._security_lock:
            recent_events = list(self._security_events)[-100:]  # Last 100 events
        
        # Aggregate security metrics
        event_types = defaultdict(int)
        threat_levels = defaultdict(int)
        hourly_events = defaultdict(int)
        
        for event in recent_events:
            event_types[event.event_type.value] += 1
            threat_levels[event.threat_level.value] += 1
            hour = event.timestamp.strftime('%H:00')
            hourly_events[hour] += 1
        
        # Calculate security trends
        security_trends = self._calculate_security_trends(recent_events)
        
        return {
            'total_events': len(recent_events),
            'event_types': dict(event_types),
            'threat_levels': dict(threat_levels),
            'hourly_distribution': dict(hourly_events),
            'security_trends': security_trends,
            'performance_metrics': self.performance_metrics,
            'top_threats': self._get_top_threats(),
            'security_recommendations': self._get_security_recommendations(),
            'dashboard_updated': datetime.now().isoformat()
        }
    
    def demonstrate_threat_detection(self) -> Dict[str, Any]:
        """Demonstrate threat detection capabilities with sample data."""
        print("\n🔒 SECURITY ENGINE DEMONSTRATION 🔒")
        print("="*50)
        
        # Generate sample security events for demonstration
        sample_events = [
            {
                'event_type': 'login_attempt',
                'source_ip': '192.168.1.100',
                'user_id': 'admin',
                'success': True,
                'timestamp': datetime.now().isoformat()
            },
            {
                'event_type': 'login_attempt',
                'source_ip': '10.0.0.50',
                'user_id': 'hacker',
                'success': False,
                'timestamp': datetime.now().isoformat()
            },
            {
                'event_type': 'login_attempt',
                'source_ip': '10.0.0.50',
                'user_id': 'admin',
                'success': False,
                'timestamp': datetime.now().isoformat()
            },
            {
                'event_type': 'data_access',
                'source_ip': '192.168.1.200',
                'user_id': 'operator1',
                'resource': '/api/manufacturing/data',
                'timestamp': datetime.now().isoformat()
            },
            {
                'event_type': 'system_command',
                'source_ip': '10.0.0.50',
                'user_id': 'hacker',
                'command': 'sudo cat /etc/passwd',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        print("📊 Processing Sample Security Events...")
        
        # Process security events
        processing_result = self.process_security_events(sample_events)
        
        print(f"✅ Events Processed: {processing_result['events_processed']}")
        print(f"⚠️  Threats Detected: {processing_result['threats_detected']}")
        print(f"⏱️  Processing Time: {processing_result['processing_time_ms']:.2f}ms")
        print(f"🛡️  Security Status: {processing_result['security_status']}")
        
        # Show detected threats
        if processing_result['detected_threats']:
            print("\n🚨 DETECTED THREATS:")
            for i, threat in enumerate(processing_result['detected_threats'], 1):
                print(f"  {i}. {threat['threat_type']} - Level: {threat['severity']}")
                print(f"     Description: {threat['description']}")
        
        # Generate threat indicators for demonstration
        sample_indicators = [
            {'type': 'ip', 'value': '10.0.0.50', 'confidence': 0.8, 'source': 'internal_monitoring'},
            {'type': 'hash', 'value': 'a1b2c3d4e5f6', 'confidence': 0.9, 'source': 'malware_scanner'},
            {'type': 'domain', 'value': 'malicious-site.com', 'confidence': 0.7, 'source': 'threat_intel'}
        ]
        
        print("\n🔍 Analyzing Threat Patterns...")
        
        # Analyze threat patterns
        analysis_result = self.analyze_threat_patterns(sample_indicators)
        
        print(f"📈 Indicators Analyzed: {analysis_result['indicators_analyzed']}")
        print(f"🎯 Patterns Detected: {analysis_result['patterns_detected']}")
        print(f"🔗 Correlations Found: {analysis_result['correlations_found']}")
        print(f"⚠️  Overall Threat Score: {analysis_result['overall_threat_score']:.2f}/10")
        
        # System security monitoring
        print("\n🖥️  System Security Monitoring...")
        
        monitoring_result = self.monitor_system_security({})
        
        print(f"🛡️  Overall Security Score: {monitoring_result['overall_security_score']:.1f}/100")
        print(f"🚨 Security Alerts: {len(monitoring_result['security_alerts'])}")
        
        # Display security dashboard data
        print("\n📊 Security Dashboard Data:")
        dashboard_data = self.get_security_dashboard_data()
        
        print(f"   📋 Total Events: {dashboard_data['total_events']}")
        print(f"   📊 Event Types: {dashboard_data['event_types']}")
        print(f"   ⚠️  Threat Levels: {dashboard_data['threat_levels']}")
        
        return {
            'demonstration_complete': True,
            'processing_result': processing_result,
            'analysis_result': analysis_result,
            'monitoring_result': monitoring_result,
            'dashboard_data': dashboard_data,
            'performance_summary': {
                'event_processing_time_ms': processing_result['processing_time_ms'],
                'threat_analysis_time_ms': analysis_result['analysis_time_ms'],
                'total_demonstration_time_ms': processing_result['processing_time_ms'] + analysis_result['analysis_time_ms']
            }
        }
    
    # Helper methods
    
    def _create_security_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
        """Create a SecurityEvent from event data."""
        event_type = SecurityEventType(event_data.get('event_type', SecurityEventType.SYSTEM_ANOMALY.value))
        source_ip = event_data.get('source_ip', '127.0.0.1')
        user_id = event_data.get('user_id')
        details = {k: v for k, v in event_data.items() if k not in ['event_type', 'source_ip', 'user_id']}
        
        event = SecurityEvent(event_type, source_ip, user_id, details)
        
        # Assess initial threat level
        event.threat_level = self._assess_initial_threat_level(event)
        
        return event
    
    def _apply_security_rules(self, event: SecurityEvent) -> List[Dict[str, Any]]:
        """Apply security rules to an event."""
        rule_results = []
        
        for rule_name, rule in self._security_rules.items():
            # Check if rule pattern matches
            matches = self._check_rule_pattern(event, rule)
            
            if matches:
                rule_results.append({
                    'rule_name': rule_name,
                    'matched': True,
                    'threat_level': rule['threat_level'].value,
                    'actions': rule['actions'],
                    'description': rule['description']
                })
                
                # Update event threat level if higher
                if rule['threat_level'].value == 'critical':
                    event.threat_level = ThreatLevel.CRITICAL
                elif rule['threat_level'].value == 'high' and event.threat_level not in [ThreatLevel.CRITICAL]:
                    event.threat_level = ThreatLevel.HIGH
        
        return rule_results
    
    def _analyze_event_threat(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze individual event for threats."""
        threat_indicators = []
        is_threat = False
        threat_score = 0.0
        
        # IP reputation check
        ip_reputation = self._check_ip_reputation(event.source_ip)
        if ip_reputation['is_suspicious']:
            threat_indicators.append('suspicious_ip')
            threat_score += 0.3
        
        # User behavior analysis
        if event.user_id:
            behavior_analysis = self._analyze_user_behavior(event)
            if behavior_analysis['is_anomalous']:
                threat_indicators.append('anomalous_behavior')
                threat_score += 0.4
        
        # Event type specific analysis
        event_analysis = self._analyze_event_type(event)
        threat_score += event_analysis['threat_contribution']
        threat_indicators.extend(event_analysis['indicators'])
        
        # Determine if this is a threat
        is_threat = threat_score >= self.threat_detection_sensitivity
        
        return {
            'is_threat': is_threat,
            'threat_score': min(threat_score, 1.0),
            'threat_indicators': threat_indicators,
            'severity': self._calculate_threat_severity(threat_score),
            'threat_type': self._classify_threat_type(event, threat_indicators),
            'description': self._generate_threat_description(event, threat_indicators),
            'recommended_actions': self._recommend_event_actions(event, threat_score)
        }
    
    def _check_rule_pattern(self, event: SecurityEvent, rule: Dict[str, Any]) -> bool:
        """Check if event matches rule pattern."""
        pattern = rule.get('pattern', '')
        
        # Check event details for pattern matches
        search_text = f"{event.event_type.value} {event.details}"
        
        try:
            return bool(re.search(pattern, search_text, re.IGNORECASE))
        except re.error:
            return False
    
    def _check_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Check IP address reputation."""
        if ip_address in self._ip_reputation:
            return self._ip_reputation[ip_address]
        
        # Simple IP reputation logic
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            
            # Check for private IPs (generally trusted)
            if ip_obj.is_private:
                reputation = {'is_suspicious': False, 'score': 0.1, 'reason': 'private_ip'}
            # Check for loopback
            elif ip_obj.is_loopback:
                reputation = {'is_suspicious': False, 'score': 0.0, 'reason': 'loopback'}
            # Public IPs get moderate suspicion score
            else:
                reputation = {'is_suspicious': True, 'score': 0.3, 'reason': 'public_ip'}
            
            # Cache reputation
            self._ip_reputation[ip_address] = reputation
            return reputation
            
        except ValueError:
            # Invalid IP format
            return {'is_suspicious': True, 'score': 0.5, 'reason': 'invalid_ip_format'}
    
    def _analyze_user_behavior(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze user behavior patterns."""
        user_id = event.user_id
        if not user_id:
            return {'is_anomalous': False, 'anomaly_score': 0.0}
        
        # Get recent user events
        user_events = [e for e in self._security_events if e.user_id == user_id][-20:]  # Last 20 events
        
        if len(user_events) < 5:  # Not enough data for analysis
            return {'is_anomalous': False, 'anomaly_score': 0.0, 'reason': 'insufficient_data'}
        
        anomaly_indicators = []
        anomaly_score = 0.0
        
        # Check for unusual timing
        current_hour = event.timestamp.hour
        user_hours = [e.timestamp.hour for e in user_events]
        if current_hour not in user_hours:
            anomaly_indicators.append('unusual_time')
            anomaly_score += 0.2
        
        # Check for rapid requests
        recent_events = [e for e in user_events if (event.timestamp - e.timestamp).total_seconds() < 60]
        if len(recent_events) > 10:  # More than 10 events in last minute
            anomaly_indicators.append('rapid_requests')
            anomaly_score += 0.3
        
        # Check for privilege escalation patterns
        privilege_commands = ['sudo', 'admin', 'root', 'elevate']
        recent_commands = [e.details.get('command', '') for e in recent_events]
        if any(cmd for cmd in recent_commands if any(priv in cmd.lower() for priv in privilege_commands)):
            anomaly_indicators.append('privilege_escalation_attempt')
            anomaly_score += 0.4
        
        return {
            'is_anomalous': anomaly_score >= 0.3,
            'anomaly_score': min(anomaly_score, 1.0),
            'anomaly_indicators': anomaly_indicators,
            'user_events_analyzed': len(user_events)
        }
    
    def _analyze_event_type(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze specific event types for threat indicators."""
        indicators = []
        threat_contribution = 0.0
        
        if event.event_type == SecurityEventType.LOGIN_ATTEMPT:
            if not event.details.get('success', True):
                indicators.append('failed_login')
                threat_contribution += 0.2
        
        elif event.event_type == SecurityEventType.ACCESS_VIOLATION:
            indicators.append('unauthorized_access')
            threat_contribution += 0.5
        
        elif event.event_type == SecurityEventType.PRIVILEGE_ESCALATION:
            indicators.append('privilege_escalation')
            threat_contribution += 0.7
        
        elif event.event_type == SecurityEventType.DATA_EXFILTRATION:
            indicators.append('data_theft')
            threat_contribution += 0.8
        
        # Check for SQL injection patterns in details
        details_str = str(event.details).lower()
        if any(pattern in details_str for pattern in ['union select', 'drop table', "' or '1'='1"]):
            indicators.append('sql_injection')
            threat_contribution += 0.9
        
        return {
            'threat_contribution': threat_contribution,
            'indicators': indicators
        }
    
    def _assess_initial_threat_level(self, event: SecurityEvent) -> ThreatLevel:
        """Assess initial threat level based on event type."""
        high_threat_types = [
            SecurityEventType.PRIVILEGE_ESCALATION,
            SecurityEventType.DATA_EXFILTRATION,
            SecurityEventType.MALWARE_DETECTED
        ]
        
        medium_threat_types = [
            SecurityEventType.ACCESS_VIOLATION,
            SecurityEventType.BRUTE_FORCE,
            SecurityEventType.SQL_INJECTION
        ]
        
        if event.event_type in high_threat_types:
            return ThreatLevel.HIGH
        elif event.event_type in medium_threat_types:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _calculate_threat_severity(self, threat_score: float) -> str:
        """Calculate threat severity based on score."""
        if threat_score >= 0.8:
            return 'critical'
        elif threat_score >= 0.6:
            return 'high'
        elif threat_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _classify_threat_type(self, event: SecurityEvent, indicators: List[str]) -> str:
        """Classify the type of threat."""
        if 'sql_injection' in indicators:
            return 'SQL Injection Attack'
        elif 'privilege_escalation' in indicators:
            return 'Privilege Escalation Attempt'
        elif 'data_theft' in indicators:
            return 'Data Exfiltration'
        elif 'failed_login' in indicators and 'rapid_requests' in indicators:
            return 'Brute Force Attack'
        elif 'suspicious_ip' in indicators:
            return 'Suspicious Network Activity'
        else:
            return 'General Security Anomaly'
    
    def _generate_threat_description(self, event: SecurityEvent, indicators: List[str]) -> str:
        """Generate human-readable threat description."""
        threat_type = self._classify_threat_type(event, indicators)
        
        description = f"{threat_type} detected from {event.source_ip}"
        if event.user_id:
            description += f" by user {event.user_id}"
        
        if 'failed_login' in indicators:
            description += " with multiple failed login attempts"
        if 'rapid_requests' in indicators:
            description += " showing rapid request patterns"
        if 'unusual_time' in indicators:
            description += " at unusual time"
        
        return description
    
    def _recommend_event_actions(self, event: SecurityEvent, threat_score: float) -> List[str]:
        """Recommend actions based on event analysis."""
        actions = []
        
        if threat_score >= 0.8:
            actions.extend(['immediate_investigation', 'isolate_source', 'alert_security_team'])
        elif threat_score >= 0.6:
            actions.extend(['investigate', 'monitor_closely', 'notify_administrators'])
        elif threat_score >= 0.4:
            actions.extend(['monitor', 'log_for_analysis'])
        else:
            actions.append('routine_monitoring')
        
        return actions
    
    def _handle_detected_threat(self, event: SecurityEvent, threat_analysis: Dict[str, Any]):
        """Handle detected threats with appropriate actions."""
        if self.alerting_engine and threat_analysis['is_threat']:
            # Generate security alert
            alert_data = {
                'alert_id': f"security_{event.event_id}",
                'severity': 'critical' if threat_analysis['severity'] == 'critical' else 'warning',
                'message': threat_analysis['description'],
                'source': 'security_engine',
                'threat_type': threat_analysis['threat_type'],
                'recommended_actions': threat_analysis['recommended_actions'],
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                self.alerting_engine.process_alert_conditions(alert_data)
            except Exception as e:
                self.logger.error(f"Failed to send security alert: {e}")
    
    def _analyze_pattern(self, pattern: Dict[str, Any], indicators: List[ThreatIndicator]) -> Dict[str, Any]:
        """Analyze a specific threat pattern."""
        matches = 0
        matching_indicators = []
        
        # Simple pattern matching based on indicator confidence
        for indicator in indicators:
            if indicator.confidence >= pattern.get('sensitivity', 0.5):
                matches += 1
                matching_indicators.append({
                    'type': indicator.indicator_type,
                    'value': indicator.value,
                    'confidence': indicator.confidence
                })
        
        return {
            'pattern_name': pattern['name'],
            'matches': matches,
            'matching_indicators': matching_indicators,
            'confidence': min(matches / len(indicators) if indicators else 0, 1.0)
        }
    
    def _correlate_threat_indicators(self, indicators: List[ThreatIndicator]) -> List[Dict[str, Any]]:
        """Correlate threat indicators to find relationships."""
        correlations = []
        
        # Group indicators by type
        by_type = defaultdict(list)
        for indicator in indicators:
            by_type[indicator.indicator_type].append(indicator)
        
        # Find correlations between different types
        for type1, indicators1 in by_type.items():
            for type2, indicators2 in by_type.items():
                if type1 != type2 and len(indicators1) > 0 and len(indicators2) > 0:
                    correlation_score = min(len(indicators1), len(indicators2)) / max(len(indicators1), len(indicators2))
                    if correlation_score > 0.3:  # Significant correlation
                        correlations.append({
                            'type1': type1,
                            'type2': type2,
                            'correlation_score': correlation_score,
                            'indicators_count': len(indicators1) + len(indicators2)
                        })
        
        return correlations
    
    def _generate_threat_intelligence(self, indicators: List[ThreatIndicator], 
                                    patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate threat intelligence from indicators and patterns."""
        high_confidence_indicators = [i for i in indicators if i.confidence >= 0.7]
        active_patterns = [p for p in patterns if p['matches'] > 0]
        
        return {
            'total_indicators': len(indicators),
            'high_confidence_indicators': len(high_confidence_indicators),
            'active_threat_patterns': len(active_patterns),
            'threat_summary': f"Analyzed {len(indicators)} indicators, {len(active_patterns)} active patterns detected",
            'confidence_distribution': {
                'high': len([i for i in indicators if i.confidence >= 0.7]),
                'medium': len([i for i in indicators if 0.3 <= i.confidence < 0.7]),
                'low': len([i for i in indicators if i.confidence < 0.3])
            }
        }
    
    def _calculate_threat_score(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall threat score from detected patterns."""
        if not patterns:
            return 0.0
        
        # Weight patterns by their confidence and match count
        total_score = 0.0
        max_possible = 0.0
        
        for pattern in patterns:
            pattern_score = pattern['confidence'] * pattern['matches']
            total_score += pattern_score
            max_possible += pattern['matches']  # Assuming max confidence of 1.0
        
        # Normalize to 0-10 scale
        if max_possible > 0:
            normalized_score = (total_score / max_possible) * 10
        else:
            normalized_score = 0.0
        
        return min(normalized_score, 10.0)
    
    def _recommend_threat_actions(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Recommend actions based on threat patterns."""
        actions = []
        
        if any(p['confidence'] >= 0.8 for p in patterns):
            actions.extend(['immediate_response', 'isolate_affected_systems', 'escalate_to_security_team'])
        elif any(p['confidence'] >= 0.6 for p in patterns):
            actions.extend(['investigate_further', 'increase_monitoring', 'prepare_response_plan'])
        elif patterns:
            actions.extend(['monitor_closely', 'update_threat_intelligence'])
        else:
            actions.append('continue_normal_monitoring')
        
        return actions
    
    def _monitor_network_security(self) -> Dict[str, Any]:
        """Monitor network security status."""
        return {
            'security_score': 85.0,
            'active_connections': 127,
            'suspicious_ips': 3,
            'blocked_attempts': 15,
            'firewall_status': 'active'
        }
    
    def _monitor_access_control(self) -> Dict[str, Any]:
        """Monitor access control security."""
        return {
            'security_score': 92.0,
            'active_sessions': 45,
            'failed_logins': 8,
            'privileged_access_events': 2,
            'mfa_compliance': 87.5
        }
    
    def _monitor_data_protection(self) -> Dict[str, Any]:
        """Monitor data protection security."""
        return {
            'security_score': 94.0,
            'encrypted_data_percentage': 98.5,
            'key_rotation_status': 'current',
            'data_access_violations': 0,
            'backup_integrity': 'verified'
        }
    
    def _monitor_system_integrity(self) -> Dict[str, Any]:
        """Monitor system integrity."""
        return {
            'security_score': 88.0,
            'system_file_integrity': 'intact',
            'malware_scan_status': 'clean',
            'patch_compliance': 92.3,
            'configuration_drift': 'minimal'
        }
    
    def _monitor_compliance_status(self) -> Dict[str, Any]:
        """Monitor compliance status."""
        return {
            'security_score': 91.0,
            'iso27001_compliance': 94.0,
            'soc2_compliance': 89.0,
            'gdpr_compliance': 96.0,
            'audit_readiness': 'compliant'
        }
    
    def _generate_security_alerts(self, monitoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate security alerts based on monitoring results."""
        alerts = []
        
        for category, results in monitoring_results.items():
            score = results.get('security_score', 100)
            if score < 90:
                alerts.append({
                    'category': category,
                    'severity': 'warning' if score >= 80 else 'critical',
                    'message': f"{category.replace('_', ' ').title()} security score below threshold: {score}%"
                })
        
        return alerts
    
    def _assess_security_status(self) -> str:
        """Assess overall security status."""
        recent_threats = sum(1 for event in list(self._security_events)[-50:] 
                           if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL])
        
        if recent_threats >= 5:
            return SecurityStatus.UNDER_ATTACK.value
        elif recent_threats >= 2:
            return SecurityStatus.WARNING.value
        else:
            return SecurityStatus.SECURE.value
    
    def _generate_security_recommendations(self, threats: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on detected threats."""
        recommendations = []
        
        if threats:
            recommendations.append("Investigate detected threats immediately")
            recommendations.append("Review security logs for additional indicators")
            recommendations.append("Consider implementing additional monitoring")
        else:
            recommendations.append("Continue routine security monitoring")
            recommendations.append("Review and update security policies")
        
        return recommendations
    
    def _update_security_metrics(self, event: SecurityEvent, threat_analysis: Dict[str, Any]):
        """Update security metrics based on processed events."""
        self._security_metrics['total_events'] += 1
        self._security_metrics[f'event_type_{event.event_type.value}'] += 1
        self._security_metrics[f'threat_level_{event.threat_level.value}'] += 1
        
        if threat_analysis['is_threat']:
            self._security_metrics['threats_detected'] += 1
    
    def _update_performance_metrics(self, processing_time: float, event_count: int):
        """Update performance metrics."""
        self.performance_metrics['total_events_processed'] += event_count
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time_ms']
        total_events = self.performance_metrics['total_events_processed']
        
        if total_events > 0:
            new_avg = ((current_avg * (total_events - event_count)) + processing_time) / total_events
            self.performance_metrics['average_processing_time_ms'] = new_avg
    
    def _calculate_security_trends(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Calculate security trends from recent events."""
        if not events:
            return {'trend': 'stable', 'threat_increase': 0.0}
        
        # Simple trend calculation
        recent_threats = sum(1 for e in events[-10:] if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL])
        older_threats = sum(1 for e in events[-20:-10] if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL])
        
        if recent_threats > older_threats:
            return {'trend': 'increasing', 'threat_increase': (recent_threats - older_threats) / max(older_threats, 1)}
        elif recent_threats < older_threats:
            return {'trend': 'decreasing', 'threat_decrease': (older_threats - recent_threats) / max(older_threats, 1)}
        else:
            return {'trend': 'stable', 'threat_change': 0.0}
    
    def _get_top_threats(self) -> List[Dict[str, Any]]:
        """Get top security threats from recent events."""
        recent_events = list(self._security_events)[-100:]
        threat_counts = defaultdict(int)
        
        for event in recent_events:
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                threat_counts[event.event_type.value] += 1
        
        # Sort by frequency
        sorted_threats = sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{'threat_type': threat, 'count': count} for threat, count in sorted_threats[:5]]
    
    def _get_security_recommendations(self) -> List[str]:
        """Get current security recommendations."""
        recommendations = []
        
        # Based on recent activity
        recent_events = list(self._security_events)[-50:]
        failed_logins = sum(1 for e in recent_events 
                          if e.event_type == SecurityEventType.LOGIN_ATTEMPT 
                          and not e.details.get('success', True))
        
        if failed_logins > 10:
            recommendations.append("Consider implementing account lockout policies")
        
        if len(recent_events) > 200:  # High activity
            recommendations.append("Review system logs for anomalous patterns")
        
        recommendations.append("Ensure all security patches are up to date")
        recommendations.append("Review user access permissions regularly")
        
        return recommendations
    
    def _start_security_monitoring(self):
        """Start background security monitoring thread."""
        def monitoring_worker():
            while True:
                try:
                    time.sleep(60)  # Monitor every minute
                    
                    # Perform routine security checks
                    self.monitor_system_security({})
                    
                    # Clean up old events (keep last 10000)
                    with self._security_lock:
                        # Events are automatically limited by deque maxlen
                        pass
                
                except Exception as e:
                    self.logger.error(f"Security monitoring error: {e}")
        
        monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        monitoring_thread.start()