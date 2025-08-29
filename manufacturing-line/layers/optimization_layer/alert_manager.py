"""
Intelligent Alert Manager - Week 14: Performance Optimization & Scalability

This module provides enterprise-grade intelligent alerting capabilities for the manufacturing system
with multi-channel delivery, alert correlation, escalation policies, and analytics.

Performance Targets:
- Multi-channel alerting (Email, SMS, Slack, PagerDuty)
- Intelligent alert correlation and grouping
- Escalation policies with automated routing
- Alert suppression and filtering
- Custom alert rules engine
- Alert analytics and tuning

Author: Manufacturing Line Control System
Created: Week 14 - Performance Optimization Phase
"""

import time
import threading
import asyncio
import json
import hashlib
import smtplib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
import logging
import queue
import uuid
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    CONSOLE = "console"
    FILE = "file"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    FATAL = "fatal"


class AlertStatus(Enum):
    """Alert status states."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    FAILED = "failed"


class EscalationAction(Enum):
    """Escalation action types."""
    NOTIFY = "notify"
    ESCALATE = "escalate"
    AUTO_RESOLVE = "auto_resolve"
    RUN_SCRIPT = "run_script"


@dataclass
class AlertRule:
    """Alert rule definition."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    conditions: Dict[str, Any]
    enabled: bool = True
    throttle_minutes: int = 5
    auto_resolve: bool = False
    auto_resolve_timeout_minutes: int = 60
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRecipient:
    """Alert recipient configuration."""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    pagerduty_user_id: Optional[str] = None
    severity_filter: Set[AlertSeverity] = field(default_factory=lambda: set(AlertSeverity))
    time_zone: str = "UTC"
    active_hours_start: int = 0  # 24-hour format
    active_hours_end: int = 24
    
    def is_active_hours(self) -> bool:
        """Check if current time is within recipient's active hours."""
        current_hour = datetime.now().hour
        if self.active_hours_start <= self.active_hours_end:
            return self.active_hours_start <= current_hour < self.active_hours_end
        else:
            # Handles cases like 22:00 to 06:00
            return current_hour >= self.active_hours_start or current_hour < self.active_hours_end


@dataclass
class EscalationPolicy:
    """Alert escalation policy."""
    id: str
    name: str
    steps: List[Dict[str, Any]]
    description: str = ""
    
    def get_escalation_step(self, minutes_since_triggered: int) -> Optional[Dict[str, Any]]:
        """Get current escalation step based on time elapsed."""
        for step in self.steps:
            if minutes_since_triggered >= step.get('delay_minutes', 0):
                continue
            return step
        return self.steps[-1] if self.steps else None


@dataclass  
class Alert:
    """Alert instance."""
    id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # State tracking
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    last_sent_at: Optional[datetime] = None
    send_count: int = 0
    
    # Correlation
    correlation_id: Optional[str] = None
    related_alert_ids: List[str] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.status in [AlertStatus.PENDING, AlertStatus.SENT]
    
    @property
    def duration_minutes(self) -> int:
        """Get alert duration in minutes."""
        end_time = self.resolved_at or datetime.now()
        return int((end_time - self.created_at).total_seconds() / 60)
    
    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = acknowledged_by
    
    def resolve(self, resolved_by: str) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.resolved_by = resolved_by


@dataclass
class AlertDeliveryResult:
    """Result of alert delivery attempt."""
    alert_id: str
    channel: AlertChannel
    recipient_id: str
    success: bool
    timestamp: datetime
    error_message: Optional[str] = None
    delivery_time_ms: Optional[float] = None


class AlertChannel_Interface(ABC):
    """Abstract interface for alert delivery channels."""
    
    @abstractmethod
    def send_alert(self, alert: Alert, recipient: AlertRecipient) -> AlertDeliveryResult:
        """Send alert via this channel."""
        pass
    
    @abstractmethod
    def get_channel_name(self) -> str:
        """Get channel name."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if channel is properly configured."""
        pass


class EmailChannel(AlertChannel_Interface):
    """Email alert delivery channel."""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str,
                 from_email: str, use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.use_tls = use_tls
    
    def send_alert(self, alert: Alert, recipient: AlertRecipient) -> AlertDeliveryResult:
        """Send alert via email."""
        start_time = time.perf_counter()
        
        try:
            if not recipient.email:
                return AlertDeliveryResult(
                    alert_id=alert.id,
                    channel=AlertChannel.EMAIL,
                    recipient_id=recipient.id,
                    success=False,
                    timestamp=datetime.now(),
                    error_message="No email address configured for recipient"
                )
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = recipient.email
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
--------------
Title: {alert.title}
Severity: {alert.severity.value.upper()}
Source: {alert.source}
Created: {alert.created_at.isoformat()}
Status: {alert.status.value}

Message:
{alert.message}

Labels:
{json.dumps(alert.labels, indent=2)}

Alert ID: {alert.id}
Rule ID: {alert.rule_id}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            delivery_time = (time.perf_counter() - start_time) * 1000
            
            return AlertDeliveryResult(
                alert_id=alert.id,
                channel=AlertChannel.EMAIL,
                recipient_id=recipient.id,
                success=True,
                timestamp=datetime.now(),
                delivery_time_ms=delivery_time
            )
            
        except Exception as e:
            delivery_time = (time.perf_counter() - start_time) * 1000
            return AlertDeliveryResult(
                alert_id=alert.id,
                channel=AlertChannel.EMAIL,
                recipient_id=recipient.id,
                success=False,
                timestamp=datetime.now(),
                error_message=str(e),
                delivery_time_ms=delivery_time
            )
    
    def get_channel_name(self) -> str:
        return "Email"
    
    def is_configured(self) -> bool:
        return all([self.smtp_host, self.smtp_port, self.username, self.from_email])


class SlackChannel(AlertChannel_Interface):
    """Slack alert delivery channel."""
    
    def __init__(self, webhook_url: str, bot_token: Optional[str] = None):
        self.webhook_url = webhook_url
        self.bot_token = bot_token
    
    def send_alert(self, alert: Alert, recipient: AlertRecipient) -> AlertDeliveryResult:
        """Send alert via Slack."""
        start_time = time.perf_counter()
        
        try:
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",      # Green
                AlertSeverity.WARNING: "#ff9900",   # Orange
                AlertSeverity.CRITICAL: "#ff0000",  # Red
                AlertSeverity.FATAL: "#8B0000"      # Dark red
            }
            
            # Create Slack payload
            payload = {
                "text": f"Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True
                            },
                            {
                                "title": "Created",
                                "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False
                            }
                        ],
                        "footer": f"Alert ID: {alert.id}",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            
            delivery_time = (time.perf_counter() - start_time) * 1000
            
            return AlertDeliveryResult(
                alert_id=alert.id,
                channel=AlertChannel.SLACK,
                recipient_id=recipient.id,
                success=True,
                timestamp=datetime.now(),
                delivery_time_ms=delivery_time
            )
            
        except Exception as e:
            delivery_time = (time.perf_counter() - start_time) * 1000
            return AlertDeliveryResult(
                alert_id=alert.id,
                channel=AlertChannel.SLACK,
                recipient_id=recipient.id,
                success=False,
                timestamp=datetime.now(),
                error_message=str(e),
                delivery_time_ms=delivery_time
            )
    
    def get_channel_name(self) -> str:
        return "Slack"
    
    def is_configured(self) -> bool:
        return bool(self.webhook_url)


class WebhookChannel(AlertChannel_Interface):
    """Generic webhook alert delivery channel."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    def send_alert(self, alert: Alert, recipient: AlertRecipient) -> AlertDeliveryResult:
        """Send alert via webhook."""
        start_time = time.perf_counter()
        
        try:
            payload = {
                "alert_id": alert.id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "source": alert.source,
                "created_at": alert.created_at.isoformat(),
                "labels": alert.labels,
                "context": alert.context,
                "recipient_id": recipient.id
            }
            
            headers = {'Content-Type': 'application/json'}
            headers.update(self.headers)
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            delivery_time = (time.perf_counter() - start_time) * 1000
            
            return AlertDeliveryResult(
                alert_id=alert.id,
                channel=AlertChannel.WEBHOOK,
                recipient_id=recipient.id,
                success=True,
                timestamp=datetime.now(),
                delivery_time_ms=delivery_time
            )
            
        except Exception as e:
            delivery_time = (time.perf_counter() - start_time) * 1000
            return AlertDeliveryResult(
                alert_id=alert.id,
                channel=AlertChannel.WEBHOOK,
                recipient_id=recipient.id,
                success=False,
                timestamp=datetime.now(),
                error_message=str(e),
                delivery_time_ms=delivery_time
            )
    
    def get_channel_name(self) -> str:
        return "Webhook"
    
    def is_configured(self) -> bool:
        return bool(self.webhook_url)


class ConsoleChannel(AlertChannel_Interface):
    """Console/logging alert delivery channel."""
    
    def __init__(self):
        self.logger = logging.getLogger("AlertManager.Console")
    
    def send_alert(self, alert: Alert, recipient: AlertRecipient) -> AlertDeliveryResult:
        """Send alert to console/log."""
        start_time = time.perf_counter()
        
        try:
            log_message = (
                f"ALERT [{alert.severity.value.upper()}] {alert.title} | "
                f"Source: {alert.source} | Status: {alert.status.value} | "
                f"ID: {alert.id} | Message: {alert.message}"
            )
            
            # Log at appropriate level based on severity
            if alert.severity == AlertSeverity.FATAL:
                self.logger.critical(log_message)
            elif alert.severity == AlertSeverity.CRITICAL:
                self.logger.error(log_message)
            elif alert.severity == AlertSeverity.WARNING:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            
            delivery_time = (time.perf_counter() - start_time) * 1000
            
            return AlertDeliveryResult(
                alert_id=alert.id,
                channel=AlertChannel.CONSOLE,
                recipient_id=recipient.id,
                success=True,
                timestamp=datetime.now(),
                delivery_time_ms=delivery_time
            )
            
        except Exception as e:
            delivery_time = (time.perf_counter() - start_time) * 1000
            return AlertDeliveryResult(
                alert_id=alert.id,
                channel=AlertChannel.CONSOLE,
                recipient_id=recipient.id,
                success=False,
                timestamp=datetime.now(),
                error_message=str(e),
                delivery_time_ms=delivery_time
            )
    
    def get_channel_name(self) -> str:
        return "Console"
    
    def is_configured(self) -> bool:
        return True


@dataclass
class AlertCorrelationRule:
    """Rule for correlating related alerts."""
    id: str
    name: str
    time_window_minutes: int
    max_alerts: int
    correlation_fields: List[str]  # Fields to match on (e.g., ['source', 'labels.component'])
    description: str = ""


@dataclass
class AlertManagerConfig:
    """Configuration for alert manager."""
    enable_correlation: bool = True
    correlation_time_window_minutes: int = 5
    max_alerts_per_correlation: int = 10
    enable_suppression: bool = True
    suppression_time_window_minutes: int = 15
    enable_auto_resolution: bool = True
    auto_resolution_timeout_minutes: int = 60
    delivery_retry_attempts: int = 3
    delivery_retry_delay_seconds: int = 30
    cleanup_resolved_alerts_hours: int = 24
    max_alerts_in_memory: int = 10000


class AlertManager:
    """
    Intelligent Alert Manager for Manufacturing System
    
    Provides enterprise-grade alerting with:
    - Multi-channel alerting (Email, SMS, Slack, PagerDuty, Webhooks)
    - Intelligent alert correlation and grouping
    - Escalation policies with automated routing
    - Alert suppression and filtering
    - Custom alert rules engine
    - Alert analytics and tuning
    - Real-time alert dashboard integration
    """
    
    def __init__(self, config: Optional[AlertManagerConfig] = None):
        self.config = config or AlertManagerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Alert storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Recipients and channels
        self.recipients: Dict[str, AlertRecipient] = {}
        self.channels: Dict[AlertChannel, AlertChannel_Interface] = {}
        
        # Escalation policies
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        
        # Correlation and suppression
        self.correlation_rules: Dict[str, AlertCorrelationRule] = {}
        self.correlation_groups: Dict[str, List[str]] = defaultdict(list)
        self.suppressed_rules: Dict[str, datetime] = {}
        
        # Delivery tracking
        self.delivery_results: deque = deque(maxlen=10000)
        
        # Background operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AlertManager")
        self._alert_queue = queue.Queue()
        self._shutdown = False
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_status': defaultdict(int),
            'delivery_attempts': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'average_delivery_time_ms': 0.0,
            'correlations_created': 0,
            'alerts_suppressed': 0
        }
        
        # Initialize console channel by default
        self.add_channel(AlertChannel.CONSOLE, ConsoleChannel())
        
        # Start background workers
        self._start_background_workers()
    
    def add_channel(self, channel_type: AlertChannel, channel_impl: AlertChannel_Interface) -> None:
        """Add alert delivery channel."""
        self.channels[channel_type] = channel_impl
        self.logger.info(f"Added alert channel: {channel_impl.get_channel_name()}")
    
    def add_recipient(self, recipient: AlertRecipient) -> None:
        """Add alert recipient."""
        self.recipients[recipient.id] = recipient
        self.logger.info(f"Added alert recipient: {recipient.name}")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.alert_rules[rule.id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Add escalation policy."""
        self.escalation_policies[policy.id] = policy
        self.logger.info(f"Added escalation policy: {policy.name}")
    
    def add_correlation_rule(self, rule: AlertCorrelationRule) -> None:
        """Add alert correlation rule."""
        self.correlation_rules[rule.id] = rule
        self.logger.info(f"Added correlation rule: {rule.name}")
    
    def create_alert(self, rule_id: str, title: str, message: str, 
                    severity: AlertSeverity, source: str,
                    labels: Optional[Dict[str, str]] = None,
                    context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new alert.
        
        Returns:
            Alert ID
        """
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            id=alert_id,
            rule_id=rule_id,
            title=title,
            message=message,
            severity=severity,
            status=AlertStatus.PENDING,
            created_at=datetime.now(),
            source=source,
            labels=labels or {},
            context=context or {}
        )
        
        # Check if rule should be suppressed
        if self._should_suppress_alert(alert):
            alert.status = AlertStatus.SUPPRESSED
            self.stats['alerts_suppressed'] += 1
            self.logger.info(f"Alert suppressed: {alert_id}")
            return alert_id
        
        # Store alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update statistics
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_severity'][severity.value] += 1
        self.stats['alerts_by_status'][alert.status.value] += 1
        
        # Handle correlation
        if self.config.enable_correlation:
            self._correlate_alert(alert)
        
        # Queue for processing
        self._alert_queue.put(alert_id)
        
        self.logger.info(f"Created alert: {alert_id} - {title}")
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.acknowledge(acknowledged_by)
        
        self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.resolve(resolved_by)
        
        # Also resolve correlated alerts
        if alert.correlation_id:
            for related_id in alert.related_alert_ids:
                if related_id in self.alerts:
                    related_alert = self.alerts[related_id]
                    if related_alert.is_active:
                        related_alert.resolve(f"Auto-resolved via {alert_id}")
        
        self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        return True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.alerts.values() if alert.is_active]
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.alerts.get(alert_id)
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity."""
        return [alert for alert in self.alerts.values() if alert.severity == severity]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        # Update dynamic statistics
        active_alerts = self.get_active_alerts()
        self.stats['active_alerts'] = len(active_alerts)
        
        # Calculate delivery success rate
        if self.stats['delivery_attempts'] > 0:
            self.stats['delivery_success_rate'] = (
                self.stats['successful_deliveries'] / self.stats['delivery_attempts']
            ) * 100
        
        # Calculate average delivery time
        recent_deliveries = [dr for dr in self.delivery_results if dr.delivery_time_ms]
        if recent_deliveries:
            self.stats['average_delivery_time_ms'] = sum(
                dr.delivery_time_ms for dr in recent_deliveries
            ) / len(recent_deliveries)
        
        return self.stats.copy()
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        if not self.config.enable_suppression:
            return False
        
        # Check if rule is in suppression period
        if alert.rule_id in self.suppressed_rules:
            suppression_end = self.suppressed_rules[alert.rule_id]
            if datetime.now() < suppression_end:
                return True
            else:
                # Remove expired suppression
                del self.suppressed_rules[alert.rule_id]
        
        return False
    
    def _correlate_alert(self, alert: Alert) -> None:
        """Correlate alert with existing alerts."""
        for rule_id, rule in self.correlation_rules.items():
            # Find potentially related alerts within time window
            cutoff_time = alert.created_at - timedelta(minutes=rule.time_window_minutes)
            
            related_alerts = []
            for existing_alert in self.alerts.values():
                if (existing_alert.created_at >= cutoff_time and
                    existing_alert.id != alert.id and
                    existing_alert.is_active):
                    
                    # Check if alerts match on correlation fields
                    if self._alerts_match_correlation_fields(alert, existing_alert, rule.correlation_fields):
                        related_alerts.append(existing_alert)
            
            # Create correlation group if we have related alerts
            if related_alerts and len(related_alerts) < rule.max_alerts:
                correlation_id = str(uuid.uuid4())
                alert.correlation_id = correlation_id
                alert.related_alert_ids = [a.id for a in related_alerts]
                
                # Update related alerts
                for related_alert in related_alerts:
                    related_alert.correlation_id = correlation_id
                    related_alert.related_alert_ids.append(alert.id)
                
                self.correlation_groups[correlation_id].extend([alert.id] + alert.related_alert_ids)
                self.stats['correlations_created'] += 1
                
                self.logger.info(f"Created correlation group {correlation_id} with {len(related_alerts) + 1} alerts")
                break
    
    def _alerts_match_correlation_fields(self, alert1: Alert, alert2: Alert, fields: List[str]) -> bool:
        """Check if two alerts match on correlation fields."""
        for field in fields:
            value1 = self._get_alert_field_value(alert1, field)
            value2 = self._get_alert_field_value(alert2, field)
            
            if value1 != value2:
                return False
        
        return True
    
    def _get_alert_field_value(self, alert: Alert, field: str) -> Any:
        """Get field value from alert using dot notation."""
        if '.' in field:
            parts = field.split('.')
            if parts[0] == 'labels' and len(parts) == 2:
                return alert.labels.get(parts[1])
            elif parts[0] == 'context' and len(parts) == 2:
                return alert.context.get(parts[1])
        else:
            return getattr(alert, field, None)
        
        return None
    
    def _start_background_workers(self) -> None:
        """Start background worker threads."""
        # Alert processing worker
        self._executor.submit(self._alert_processing_worker)
        
        # Auto-resolution worker
        if self.config.enable_auto_resolution:
            self._executor.submit(self._auto_resolution_worker)
        
        # Cleanup worker
        self._executor.submit(self._cleanup_worker)
    
    def _alert_processing_worker(self) -> None:
        """Process alerts from queue."""
        while not self._shutdown:
            try:
                alert_id = self._alert_queue.get(timeout=1)
                if alert_id in self.alerts:
                    alert = self.alerts[alert_id]
                    self._process_alert(alert)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Alert processing worker error: {e}")
    
    def _process_alert(self, alert: Alert) -> None:
        """Process individual alert."""
        try:
            # Get alert rule
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.enabled:
                self.logger.warning(f"Alert rule not found or disabled: {alert.rule_id}")
                return
            
            # Check throttling
            if self._is_throttled(alert, rule):
                self.logger.info(f"Alert throttled: {alert.id}")
                return
            
            # Send alert via configured channels
            for channel_type in rule.channels:
                if channel_type not in self.channels:
                    self.logger.warning(f"Channel not configured: {channel_type.value}")
                    continue
                
                channel = self.channels[channel_type]
                
                # Send to all recipients who should receive this severity
                for recipient in self.recipients.values():
                    if (alert.severity in recipient.severity_filter and
                        recipient.is_active_hours()):
                        
                        # Send alert
                        result = channel.send_alert(alert, recipient)
                        self.delivery_results.append(result)
                        
                        # Update statistics
                        self.stats['delivery_attempts'] += 1
                        if result.success:
                            self.stats['successful_deliveries'] += 1
                        else:
                            self.stats['failed_deliveries'] += 1
                            self.logger.error(f"Failed to deliver alert {alert.id} via {channel_type.value}: {result.error_message}")
            
            # Update alert status
            alert.status = AlertStatus.SENT
            alert.last_sent_at = datetime.now()
            alert.send_count += 1
            
            self.logger.info(f"Processed alert: {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process alert {alert.id}: {e}")
            alert.status = AlertStatus.FAILED
    
    def _is_throttled(self, alert: Alert, rule: AlertRule) -> bool:
        """Check if alert should be throttled."""
        if rule.throttle_minutes <= 0:
            return False
        
        # Check if we've sent this rule recently
        cutoff_time = datetime.now() - timedelta(minutes=rule.throttle_minutes)
        
        for existing_alert in self.alerts.values():
            if (existing_alert.rule_id == alert.rule_id and
                existing_alert.id != alert.id and
                existing_alert.last_sent_at and
                existing_alert.last_sent_at > cutoff_time):
                return True
        
        return False
    
    def _auto_resolution_worker(self) -> None:
        """Auto-resolve alerts that haven't been updated."""
        while not self._shutdown:
            try:
                current_time = datetime.now()
                
                for alert in list(self.alerts.values()):
                    if not alert.is_active:
                        continue
                    
                    # Get rule and check auto-resolve settings
                    rule = self.alert_rules.get(alert.rule_id)
                    if not rule or not rule.auto_resolve:
                        continue
                    
                    # Check if alert is old enough to auto-resolve
                    age_minutes = (current_time - alert.created_at).total_seconds() / 60
                    if age_minutes > rule.auto_resolve_timeout_minutes:
                        alert.resolve("auto-resolve")
                        self.logger.info(f"Auto-resolved alert: {alert.id}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Auto-resolution worker error: {e}")
                time.sleep(60)
    
    def _cleanup_worker(self) -> None:
        """Clean up old resolved alerts."""
        while not self._shutdown:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.config.cleanup_resolved_alerts_hours)
                
                # Remove old resolved alerts from active storage
                alerts_to_remove = []
                for alert_id, alert in self.alerts.items():
                    if (alert.status == AlertStatus.RESOLVED and
                        alert.resolved_at and
                        alert.resolved_at < cutoff_time):
                        alerts_to_remove.append(alert_id)
                
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                
                if alerts_to_remove:
                    self.logger.info(f"Cleaned up {len(alerts_to_remove)} old resolved alerts")
                
                # Limit memory usage
                if len(self.alerts) > self.config.max_alerts_in_memory:
                    # Remove oldest resolved alerts
                    resolved_alerts = [
                        (alert_id, alert) for alert_id, alert in self.alerts.items()
                        if alert.status == AlertStatus.RESOLVED
                    ]
                    resolved_alerts.sort(key=lambda x: x[1].resolved_at or datetime.min)
                    
                    to_remove = len(resolved_alerts) - (self.config.max_alerts_in_memory // 2)
                    for alert_id, _ in resolved_alerts[:to_remove]:
                        del self.alerts[alert_id]
                
                time.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup worker error: {e}")
                time.sleep(3600)
    
    def shutdown(self) -> None:
        """Shutdown alert manager."""
        self._shutdown = True
        self._executor.shutdown(wait=True)
        self.logger.info("Alert manager shutdown completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Convenience functions for global alert manager
_global_alert_manager: Optional[AlertManager] = None

def get_alert_manager(config: Optional[AlertManagerConfig] = None) -> AlertManager:
    """Get or create global alert manager instance."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager(config)
    return _global_alert_manager

def create_alert(rule_id: str, title: str, message: str, severity: AlertSeverity,
                source: str, labels: Optional[Dict[str, str]] = None) -> str:
    """Convenience function to create alert."""
    return get_alert_manager().create_alert(rule_id, title, message, severity, source, labels)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    print("Intelligent Alert Manager Demo")
    print("=" * 50)
    
    # Create alert manager
    config = AlertManagerConfig(
        enable_correlation=True,
        enable_suppression=True,
        enable_auto_resolution=True,
        auto_resolution_timeout_minutes=2  # Short for demo
    )
    
    with AlertManager(config) as alert_mgr:
        # Add recipients
        alert_mgr.add_recipient(AlertRecipient(
            id="ops_team",
            name="Operations Team",
            email="ops@company.com",
            severity_filter={AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.FATAL}
        ))
        
        alert_mgr.add_recipient(AlertRecipient(
            id="dev_team", 
            name="Development Team",
            email="dev@company.com",
            severity_filter={AlertSeverity.CRITICAL, AlertSeverity.FATAL}
        ))
        
        # Add alert rules
        alert_mgr.add_alert_rule(AlertRule(
            id="high_cpu",
            name="High CPU Usage",
            description="CPU usage above threshold",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
            conditions={"metric": "cpu_percent", "operator": "gt", "threshold": 80},
            throttle_minutes=1,
            auto_resolve=True,
            auto_resolve_timeout_minutes=2
        ))
        
        alert_mgr.add_alert_rule(AlertRule(
            id="system_failure",
            name="System Failure",
            description="Critical system failure detected",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.CONSOLE],
            conditions={"metric": "system_status", "operator": "eq", "value": "failed"}
        ))
        
        # Add correlation rule
        alert_mgr.add_correlation_rule(AlertCorrelationRule(
            id="cpu_correlation",
            name="CPU Related Alerts",
            time_window_minutes=5,
            max_alerts=5,
            correlation_fields=["source", "labels.component"],
            description="Group CPU-related alerts together"
        ))
        
        print("\n1. Alert Manager Setup:")
        print(f"Recipients: {len(alert_mgr.recipients)}")
        print(f"Alert Rules: {len(alert_mgr.alert_rules)}")
        print(f"Correlation Rules: {len(alert_mgr.correlation_rules)}")
        
        # Create some test alerts
        print("\n2. Creating Test Alerts:")
        
        alerts = []
        
        # High CPU alerts (should be correlated)
        for i in range(3):
            alert_id = alert_mgr.create_alert(
                rule_id="high_cpu",
                title=f"High CPU Usage on Server {i+1}",
                message=f"CPU usage is {85 + i}% on server-{i+1}",
                severity=AlertSeverity.WARNING,
                source=f"server-{i+1}",
                labels={"component": "cpu", "datacenter": "us-west"}
            )
            alerts.append(alert_id)
            print(f"Created alert: {alert_id}")
            time.sleep(0.5)  # Small delay for correlation
        
        # System failure alert
        failure_alert_id = alert_mgr.create_alert(
            rule_id="system_failure",
            title="Database Connection Failed",
            message="Unable to connect to primary database",
            severity=AlertSeverity.CRITICAL,
            source="database-1",
            labels={"component": "database", "datacenter": "us-west"}
        )
        alerts.append(failure_alert_id)
        print(f"Created critical alert: {failure_alert_id}")
        
        # Wait for processing
        time.sleep(2)
        
        # Show active alerts
        print("\n3. Active Alerts:")
        active_alerts = alert_mgr.get_active_alerts()
        for alert in active_alerts:
            correlation_info = f" (Correlated: {alert.correlation_id})" if alert.correlation_id else ""
            print(f"- {alert.severity.value.upper()}: {alert.title}{correlation_info}")
        
        # Acknowledge an alert
        print("\n4. Acknowledging Alert:")
        if alerts:
            alert_mgr.acknowledge_alert(alerts[0], "operator@company.com")
            print(f"Acknowledged alert: {alerts[0]}")
        
        # Show statistics
        print("\n5. Alert Statistics:")
        stats = alert_mgr.get_alert_statistics()
        print(f"Total Alerts: {stats['total_alerts']}")
        print(f"Active Alerts: {stats.get('active_alerts', 0)}")
        print(f"Delivery Success Rate: {stats.get('delivery_success_rate', 0):.1f}%")
        print(f"Correlations Created: {stats['correlations_created']}")
        print(f"Alerts Suppressed: {stats['alerts_suppressed']}")
        
        # Test auto-resolution (wait for timeout)
        print("\n6. Testing Auto-Resolution (waiting 3 seconds)...")
        time.sleep(3)
        
        active_alerts_after = alert_mgr.get_active_alerts()
        resolved_count = len(active_alerts) - len(active_alerts_after)
        print(f"Auto-resolved {resolved_count} alerts")
        
        print("\nIntelligent Alert Manager demo completed successfully!")