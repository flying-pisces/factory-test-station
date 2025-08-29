"""
Alerting Engine for Week 8: Deployment & Monitoring

This module implements comprehensive alerting system for the manufacturing line
control system with intelligent alerting, alert correlation, escalation policies,
and multi-channel notifications including email, Slack, PagerDuty, and webhooks.

Performance Target: <30 seconds for critical alert processing and delivery
Alerting Features: Intelligent alerting, alert correlation, escalation policies, multi-channel notifications
Integration: Email, Slack, PagerDuty, webhook notifications
"""

import time
import logging
import asyncio
import json
import os
import sys
import smtplib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import concurrent.futures
import traceback
from pathlib import Path
import uuid
import statistics
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Notification integrations
try:
    import slack_sdk
    from slack_sdk import WebClient as SlackClient
    SLACK_AVAILABLE = True
except ImportError:
    slack_sdk = None
    SlackClient = None
    SLACK_AVAILABLE = False

try:
    import twilio
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    twilio = None
    TwilioClient = None
    TWILIO_AVAILABLE = False

# Database for alert storage
try:
    import sqlite3
    import redis
    DATABASE_AVAILABLE = True
except ImportError:
    sqlite3 = None
    redis = None
    DATABASE_AVAILABLE = False

# Week 8 deployment layer integrations (forward references)
try:
    from layers.deployment_layer.deployment_engine import DeploymentEngine
    from layers.deployment_layer.monitoring_engine import MonitoringEngine
    from layers.deployment_layer.infrastructure_engine import InfrastructureEngine
except ImportError:
    DeploymentEngine = None
    MonitoringEngine = None
    InfrastructureEngine = None

# Week 7 testing layer integrations
try:
    from layers.testing_layer.quality_assurance_engine import QualityAssuranceEngine
    from layers.testing_layer.benchmarking_engine import BenchmarkingEngine
except ImportError:
    QualityAssuranceEngine = None
    BenchmarkingEngine = None

# Week 6 UI layer integrations
try:
    from layers.ui_layer.visualization_engine import VisualizationEngine
    from layers.ui_layer.webui_engine import WebUIEngine
except ImportError:
    VisualizationEngine = None
    WebUIEngine = None

# Common imports
try:
    from common.interfaces.layer_interface import LayerInterface
    from common.interfaces.data_interface import DataInterface
    from common.interfaces.communication_interface import CommunicationInterface
except ImportError:
    LayerInterface = None
    DataInterface = None
    CommunicationInterface = None


class AlertSeverity(Enum):
    """Alert severity enumeration"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status enumeration"""
    PENDING = "pending"
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    EXPIRED = "expired"


class NotificationChannel(Enum):
    """Notification channel enumeration"""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"
    TEAMS = "teams"


class EscalationAction(Enum):
    """Escalation action enumeration"""
    NOTIFY = "notify"
    ESCALATE = "escalate"
    AUTO_RESOLVE = "auto_resolve"
    RUN_SCRIPT = "run_script"
    SCALE_INFRASTRUCTURE = "scale_infrastructure"


class AlertGrouping(Enum):
    """Alert grouping enumeration"""
    BY_SERVICE = "by_service"
    BY_SEVERITY = "by_severity"
    BY_TEAM = "by_team"
    BY_DATACENTER = "by_datacenter"
    BY_ALERT_NAME = "by_alert_name"


@dataclass
class AlertDefinition:
    """Alert definition"""
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    duration: int  # seconds
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.annotations is None:
            self.annotations = {}


@dataclass
class Alert:
    """Alert instance"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    labels: Dict[str, str]
    annotations: Dict[str, str]
    started_at: datetime
    ended_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    fingerprint: str = ""
    
    def __post_init__(self):
        if not self.fingerprint:
            # Generate fingerprint for deduplication
            fingerprint_data = f"{self.name}:{json.dumps(self.labels, sort_keys=True)}"
            self.fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()


@dataclass
class NotificationConfig:
    """Notification configuration"""
    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit: Dict[str, int] = None  # {'count': 5, 'window': 300}
    
    def __post_init__(self):
        if self.rate_limit is None:
            self.rate_limit = {'count': 10, 'window': 300}  # 10 notifications per 5 minutes


@dataclass
class EscalationRule:
    """Escalation rule"""
    name: str
    condition: str
    delay: int  # seconds
    action: EscalationAction
    config: Dict[str, Any]
    enabled: bool = True


@dataclass
class EscalationPolicy:
    """Escalation policy"""
    name: str
    description: str
    rules: List[EscalationRule]
    repeat_interval: int = 3600  # seconds
    max_escalations: int = 3
    enabled: bool = True


@dataclass
class AlertGroup:
    """Alert group for correlation"""
    id: str
    name: str
    grouping_type: AlertGrouping
    alerts: List[Alert]
    created_at: datetime
    updated_at: datetime
    status: AlertStatus
    correlation_score: float = 0.0


@dataclass
class NotificationDelivery:
    """Notification delivery record"""
    id: str
    alert_id: str
    channel: NotificationChannel
    recipient: str
    status: str  # "sent", "failed", "pending"
    sent_at: datetime
    delivered_at: Optional[datetime] = None
    error: Optional[str] = None


class NotificationChannelManager:
    """Notification channel management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.channels = {}
        self.rate_limiters = defaultdict(deque)
        
    def register_channel(self, name: str, config: NotificationConfig):
        """Register notification channel"""
        self.channels[name] = config
        self.logger.info(f"Registered notification channel: {name} ({config.channel.value})")
    
    async def send_notification(self, channel_name: str, alert: Alert, message: str) -> NotificationDelivery:
        """Send notification through channel"""
        if channel_name not in self.channels:
            raise ValueError(f"Notification channel {channel_name} not found")
        
        config = self.channels[channel_name]
        
        if not config.enabled:
            raise RuntimeError(f"Notification channel {channel_name} is disabled")
        
        # Check rate limiting
        if not self._check_rate_limit(channel_name, config.rate_limit):
            raise RuntimeError(f"Rate limit exceeded for channel {channel_name}")
        
        # Create delivery record
        delivery = NotificationDelivery(
            id=str(uuid.uuid4()),
            alert_id=alert.id,
            channel=config.channel,
            recipient="",  # Will be filled by specific implementation
            status="pending",
            sent_at=datetime.now()
        )
        
        try:
            # Send through specific channel
            if config.channel == NotificationChannel.EMAIL:
                await self._send_email(config, alert, message, delivery)
            elif config.channel == NotificationChannel.SLACK:
                await self._send_slack(config, alert, message, delivery)
            elif config.channel == NotificationChannel.SMS:
                await self._send_sms(config, alert, message, delivery)
            elif config.channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(config, alert, message, delivery)
            elif config.channel == NotificationChannel.PAGERDUTY:
                await self._send_pagerduty(config, alert, message, delivery)
            else:
                raise ValueError(f"Unsupported notification channel: {config.channel}")
            
            delivery.status = "sent"
            delivery.delivered_at = datetime.now()
            
            # Update rate limiter
            self._update_rate_limiter(channel_name)
            
            self.logger.info(f"Notification sent through {channel_name}: {alert.name}")
            return delivery
            
        except Exception as e:
            delivery.status = "failed"
            delivery.error = str(e)
            self.logger.error(f"Notification failed for {channel_name}: {e}")
            raise
    
    def _check_rate_limit(self, channel_name: str, rate_limit: Dict[str, int]) -> bool:
        """Check rate limiting"""
        if not rate_limit:
            return True
        
        now = datetime.now()
        window_start = now - timedelta(seconds=rate_limit['window'])
        
        # Remove old entries
        limiter = self.rate_limiters[channel_name]
        while limiter and limiter[0] < window_start:
            limiter.popleft()
        
        # Check if we can send
        return len(limiter) < rate_limit['count']
    
    def _update_rate_limiter(self, channel_name: str):
        """Update rate limiter"""
        self.rate_limiters[channel_name].append(datetime.now())
    
    async def _send_email(self, config: NotificationConfig, alert: Alert, message: str, delivery: NotificationDelivery):
        """Send email notification"""
        email_config = config.config
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['from_address']
        msg['To'] = email_config['to_address']
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
        
        # Email body
        body = f"""
Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value}
Started: {alert.started_at.isoformat()}

Description:
{alert.description}

Details:
{message}

Labels:
{json.dumps(alert.labels, indent=2)}

Alert ID: {alert.id}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            server = smtplib.SMTP(email_config.get('smtp_host', 'localhost'), email_config.get('smtp_port', 587))
            
            if email_config.get('use_tls', True):
                server.starttls()
            
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            
            server.sendmail(email_config['from_address'], email_config['to_address'], msg.as_string())
            server.quit()
            
            delivery.recipient = email_config['to_address']
            
        except Exception as e:
            self.logger.error(f"Email sending failed: {e}")
            raise
    
    async def _send_slack(self, config: NotificationConfig, alert: Alert, message: str, delivery: NotificationDelivery):
        """Send Slack notification"""
        if not SLACK_AVAILABLE:
            raise RuntimeError("Slack SDK not available")
        
        slack_config = config.config
        client = SlackClient(token=slack_config['bot_token'])
        
        # Create Slack message
        color = {
            AlertSeverity.INFO: "good",
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "warning",
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.EMERGENCY: "danger"
        }.get(alert.severity, "warning")
        
        attachment = {
            "color": color,
            "title": f"{alert.severity.value.upper()}: {alert.name}",
            "text": alert.description,
            "fields": [
                {
                    "title": "Status",
                    "value": alert.status.value,
                    "short": True
                },
                {
                    "title": "Started",
                    "value": alert.started_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "short": True
                },
                {
                    "title": "Alert ID",
                    "value": alert.id,
                    "short": True
                }
            ],
            "footer": "Manufacturing Line Alerting",
            "ts": int(alert.started_at.timestamp())
        }
        
        if alert.labels:
            labels_text = "\n".join([f"*{k}*: {v}" for k, v in alert.labels.items()])
            attachment["fields"].append({
                "title": "Labels",
                "value": labels_text,
                "short": False
            })
        
        try:
            response = client.chat_postMessage(
                channel=slack_config['channel'],
                text=f"Alert: {alert.name}",
                attachments=[attachment]
            )
            
            delivery.recipient = slack_config['channel']
            
            if not response['ok']:
                raise RuntimeError(f"Slack API error: {response.get('error', 'Unknown error')}")
            
        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}")
            raise
    
    async def _send_sms(self, config: NotificationConfig, alert: Alert, message: str, delivery: NotificationDelivery):
        """Send SMS notification"""
        if not TWILIO_AVAILABLE:
            raise RuntimeError("Twilio SDK not available")
        
        sms_config = config.config
        client = TwilioClient(sms_config['account_sid'], sms_config['auth_token'])
        
        # Create SMS message
        sms_body = f"[{alert.severity.value.upper()}] {alert.name}\n{alert.description[:100]}...\nAlert ID: {alert.id}"
        
        try:
            message = client.messages.create(
                body=sms_body,
                from_=sms_config['from_number'],
                to=sms_config['to_number']
            )
            
            delivery.recipient = sms_config['to_number']
            
        except Exception as e:
            self.logger.error(f"SMS notification failed: {e}")
            raise
    
    async def _send_webhook(self, config: NotificationConfig, alert: Alert, message: str, delivery: NotificationDelivery):
        """Send webhook notification"""
        webhook_config = config.config
        
        # Create webhook payload
        payload = {
            "alert": {
                "id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "started_at": alert.started_at.isoformat(),
                "ended_at": alert.ended_at.isoformat() if alert.ended_at else None,
                "fingerprint": alert.fingerprint
            },
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Manufacturing-Line-Alerting/1.0'
        }
        
        # Add custom headers if specified
        if 'headers' in webhook_config:
            headers.update(webhook_config['headers'])
        
        try:
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=webhook_config.get('timeout', 10)
            )
            
            response.raise_for_status()
            delivery.recipient = webhook_config['url']
            
        except Exception as e:
            self.logger.error(f"Webhook notification failed: {e}")
            raise
    
    async def _send_pagerduty(self, config: NotificationConfig, alert: Alert, message: str, delivery: NotificationDelivery):
        """Send PagerDuty notification"""
        pagerduty_config = config.config
        
        # Create PagerDuty event
        event = {
            "routing_key": pagerduty_config['routing_key'],
            "event_action": "trigger" if alert.status == AlertStatus.FIRING else "resolve",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": f"{alert.name}: {alert.description}",
                "source": "manufacturing-line-alerting",
                "severity": {
                    AlertSeverity.INFO: "info",
                    AlertSeverity.LOW: "info",
                    AlertSeverity.MEDIUM: "warning",
                    AlertSeverity.HIGH: "error",
                    AlertSeverity.CRITICAL: "critical",
                    AlertSeverity.EMERGENCY: "critical"
                }.get(alert.severity, "error"),
                "component": alert.labels.get('component', 'unknown'),
                "group": alert.labels.get('service', 'manufacturing-line'),
                "class": alert.labels.get('alert_type', 'system'),
                "custom_details": {
                    "alert_id": alert.id,
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                    "started_at": alert.started_at.isoformat()
                }
            }
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=event,
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            delivery.recipient = pagerduty_config['routing_key']
            
        except Exception as e:
            self.logger.error(f"PagerDuty notification failed: {e}")
            raise


class AlertCorrelationEngine:
    """Alert correlation and grouping"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.correlation_rules = {}
        self.alert_groups = {}
        self.correlation_cache = {}
    
    def add_correlation_rule(self, name: str, rule_func: Callable[[List[Alert]], float]):
        """Add correlation rule"""
        self.correlation_rules[name] = rule_func
        self.logger.info(f"Added correlation rule: {name}")
    
    def correlate_alerts(self, alerts: List[Alert]) -> List[AlertGroup]:
        """Correlate alerts into groups"""
        if not alerts:
            return []
        
        # Initialize groups
        groups = []
        grouped_alerts = set()
        
        # Apply correlation rules
        for i, alert in enumerate(alerts):
            if alert.id in grouped_alerts:
                continue
            
            # Find correlated alerts
            correlated = [alert]
            grouped_alerts.add(alert.id)
            
            for j, other_alert in enumerate(alerts[i + 1:], i + 1):
                if other_alert.id in grouped_alerts:
                    continue
                
                correlation_score = self._calculate_correlation(alert, other_alert)
                
                if correlation_score > 0.7:  # Correlation threshold
                    correlated.append(other_alert)
                    grouped_alerts.add(other_alert.id)
            
            # Create group if multiple alerts are correlated
            if len(correlated) > 1:
                group = AlertGroup(
                    id=str(uuid.uuid4()),
                    name=f"Correlated alerts: {alert.name}",
                    grouping_type=AlertGrouping.BY_ALERT_NAME,
                    alerts=correlated,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status=max(a.status for a in correlated),
                    correlation_score=correlation_score
                )
                groups.append(group)
        
        return groups
    
    def _calculate_correlation(self, alert1: Alert, alert2: Alert) -> float:
        """Calculate correlation score between alerts"""
        cache_key = f"{alert1.fingerprint}:{alert2.fingerprint}"
        
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
        
        score = 0.0
        
        # Time correlation (alerts fired within 5 minutes)
        time_diff = abs((alert1.started_at - alert2.started_at).total_seconds())
        if time_diff <= 300:  # 5 minutes
            score += 0.3
        
        # Label correlation
        common_labels = set(alert1.labels.keys()) & set(alert2.labels.keys())
        if common_labels:
            matching_labels = sum(1 for label in common_labels 
                                if alert1.labels[label] == alert2.labels[label])
            score += 0.4 * (matching_labels / len(common_labels))
        
        # Severity correlation
        if alert1.severity == alert2.severity:
            score += 0.1
        
        # Service correlation
        service1 = alert1.labels.get('service', '')
        service2 = alert2.labels.get('service', '')
        if service1 and service2 and service1 == service2:
            score += 0.2
        
        # Apply custom correlation rules
        for rule_name, rule_func in self.correlation_rules.items():
            try:
                rule_score = rule_func([alert1, alert2])
                score += rule_score * 0.1  # Weight custom rules lower
            except Exception as e:
                self.logger.error(f"Correlation rule {rule_name} failed: {e}")
        
        # Cache result
        self.correlation_cache[cache_key] = score
        
        return score
    
    def group_alerts_by_criteria(self, alerts: List[Alert], criteria: AlertGrouping) -> List[AlertGroup]:
        """Group alerts by specific criteria"""
        groups_dict = defaultdict(list)
        
        for alert in alerts:
            if criteria == AlertGrouping.BY_SERVICE:
                key = alert.labels.get('service', 'unknown')
            elif criteria == AlertGrouping.BY_SEVERITY:
                key = alert.severity.value
            elif criteria == AlertGrouping.BY_TEAM:
                key = alert.labels.get('team', 'unknown')
            elif criteria == AlertGrouping.BY_DATACENTER:
                key = alert.labels.get('datacenter', 'unknown')
            elif criteria == AlertGrouping.BY_ALERT_NAME:
                key = alert.name
            else:
                key = 'default'
            
            groups_dict[key].append(alert)
        
        groups = []
        for key, group_alerts in groups_dict.items():
            if len(group_alerts) > 1:  # Only create groups with multiple alerts
                group = AlertGroup(
                    id=str(uuid.uuid4()),
                    name=f"{criteria.value}: {key}",
                    grouping_type=criteria,
                    alerts=group_alerts,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status=max(a.status for a in group_alerts)
                )
                groups.append(group)
        
        return groups


class EscalationManager:
    """Alert escalation management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.policies = {}
        self.escalation_states = {}
    
    def register_policy(self, policy: EscalationPolicy):
        """Register escalation policy"""
        self.policies[policy.name] = policy
        self.logger.info(f"Registered escalation policy: {policy.name}")
    
    async def start_escalation(self, alert: Alert, policy_name: str):
        """Start escalation for alert"""
        if policy_name not in self.policies:
            raise ValueError(f"Escalation policy {policy_name} not found")
        
        policy = self.policies[policy_name]
        
        if not policy.enabled:
            self.logger.warning(f"Escalation policy {policy_name} is disabled")
            return
        
        escalation_state = {
            'alert_id': alert.id,
            'policy_name': policy_name,
            'current_rule_index': 0,
            'escalations_count': 0,
            'started_at': datetime.now(),
            'next_escalation_at': datetime.now() + timedelta(seconds=policy.rules[0].delay),
            'status': 'active'
        }
        
        self.escalation_states[alert.id] = escalation_state
        
        # Schedule first escalation
        asyncio.create_task(self._schedule_escalation(alert.id))
        
        self.logger.info(f"Started escalation for alert {alert.id} with policy {policy_name}")
    
    async def _schedule_escalation(self, alert_id: str):
        """Schedule escalation execution"""
        while alert_id in self.escalation_states:
            state = self.escalation_states[alert_id]
            
            if state['status'] != 'active':
                break
            
            # Wait until next escalation time
            now = datetime.now()
            if now < state['next_escalation_at']:
                sleep_duration = (state['next_escalation_at'] - now).total_seconds()
                await asyncio.sleep(sleep_duration)
            
            # Execute escalation rule
            try:
                await self._execute_escalation_rule(alert_id)
            except Exception as e:
                self.logger.error(f"Escalation execution failed for alert {alert_id}: {e}")
                break
        
        # Cleanup
        if alert_id in self.escalation_states:
            del self.escalation_states[alert_id]
    
    async def _execute_escalation_rule(self, alert_id: str):
        """Execute escalation rule"""
        state = self.escalation_states[alert_id]
        policy = self.policies[state['policy_name']]
        
        if state['current_rule_index'] >= len(policy.rules):
            # Repeat escalation if configured
            if state['escalations_count'] < policy.max_escalations:
                state['current_rule_index'] = 0
                state['escalations_count'] += 1
                state['next_escalation_at'] = datetime.now() + timedelta(seconds=policy.repeat_interval)
            else:
                state['status'] = 'completed'
                self.logger.info(f"Escalation completed for alert {alert_id} (max escalations reached)")
            return
        
        rule = policy.rules[state['current_rule_index']]
        
        if not rule.enabled:
            # Skip disabled rule
            state['current_rule_index'] += 1
            return
        
        self.logger.info(f"Executing escalation rule {rule.name} for alert {alert_id}")
        
        # Execute rule action
        if rule.action == EscalationAction.NOTIFY:
            await self._execute_notify_action(alert_id, rule)
        elif rule.action == EscalationAction.ESCALATE:
            await self._execute_escalate_action(alert_id, rule)
        elif rule.action == EscalationAction.AUTO_RESOLVE:
            await self._execute_auto_resolve_action(alert_id, rule)
        elif rule.action == EscalationAction.RUN_SCRIPT:
            await self._execute_script_action(alert_id, rule)
        elif rule.action == EscalationAction.SCALE_INFRASTRUCTURE:
            await self._execute_scaling_action(alert_id, rule)
        
        # Schedule next rule
        state['current_rule_index'] += 1
        if state['current_rule_index'] < len(policy.rules):
            next_rule = policy.rules[state['current_rule_index']]
            state['next_escalation_at'] = datetime.now() + timedelta(seconds=next_rule.delay)
    
    async def _execute_notify_action(self, alert_id: str, rule: EscalationRule):
        """Execute notification action"""
        # This would integrate with the notification system
        self.logger.info(f"Escalation notification for alert {alert_id}: {rule.config}")
    
    async def _execute_escalate_action(self, alert_id: str, rule: EscalationRule):
        """Execute escalation action"""
        # This would escalate to higher tier support
        self.logger.info(f"Escalating alert {alert_id} to: {rule.config.get('team', 'unknown')}")
    
    async def _execute_auto_resolve_action(self, alert_id: str, rule: EscalationRule):
        """Execute auto-resolve action"""
        # This would automatically resolve the alert if conditions are met
        self.logger.info(f"Auto-resolving alert {alert_id}")
    
    async def _execute_script_action(self, alert_id: str, rule: EscalationRule):
        """Execute script action"""
        script_path = rule.config.get('script_path')
        if script_path and os.path.exists(script_path):
            try:
                import subprocess
                result = subprocess.run([script_path, alert_id], capture_output=True, text=True, timeout=60)
                self.logger.info(f"Script executed for alert {alert_id}: {result.stdout}")
            except Exception as e:
                self.logger.error(f"Script execution failed for alert {alert_id}: {e}")
    
    async def _execute_scaling_action(self, alert_id: str, rule: EscalationRule):
        """Execute infrastructure scaling action"""
        # This would integrate with infrastructure engine
        self.logger.info(f"Scaling infrastructure for alert {alert_id}: {rule.config}")
    
    def stop_escalation(self, alert_id: str):
        """Stop escalation for alert"""
        if alert_id in self.escalation_states:
            self.escalation_states[alert_id]['status'] = 'stopped'
            self.logger.info(f"Stopped escalation for alert {alert_id}")


class AlertingEngine:
    """
    Comprehensive alerting engine with intelligent alerting, alert correlation,
    escalation policies, and multi-channel notifications.
    """
    
    def __init__(self, db_path: str = "alerting.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Alert storage
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.silenced_alerts = {}
        
        # Performance tracking
        self.performance_metrics = {
            'alerts_processed': 0,
            'notifications_sent': 0,
            'escalations_triggered': 0,
            'avg_processing_time': deque(maxlen=100),
            'notification_failures': 0
        }
        
        # Initialize components
        self.notification_manager = NotificationChannelManager()
        self.correlation_engine = AlertCorrelationEngine()
        self.escalation_manager = EscalationManager()
        
        # Initialize database
        self._initialize_database()
        
        # Integration references
        self.monitoring_engine = None
        self.deployment_engine = None
        self.infrastructure_engine = None
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("AlertingEngine initialized successfully")
    
    def _initialize_database(self):
        """Initialize alert database"""
        if DATABASE_AVAILABLE and sqlite3:
            try:
                self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._create_tables()
                self.logger.info("Alert database initialized")
            except Exception as e:
                self.logger.error(f"Database initialization failed: {e}")
                self.connection = None
    
    def _create_tables(self):
        """Create database tables"""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                labels TEXT,
                annotations TEXT,
                started_at DATETIME NOT NULL,
                ended_at DATETIME,
                acknowledged_at DATETIME,
                acknowledged_by TEXT,
                resolved_at DATETIME,
                resolved_by TEXT,
                fingerprint TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Notifications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                alert_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                recipient TEXT NOT NULL,
                status TEXT NOT NULL,
                sent_at DATETIME NOT NULL,
                delivered_at DATETIME,
                error TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_fingerprint ON alerts(fingerprint)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_alert_id ON notifications(alert_id)")
        
        self.connection.commit()
    
    def set_integrations(self, monitoring_engine=None, deployment_engine=None, infrastructure_engine=None):
        """Set integration references"""
        self.monitoring_engine = monitoring_engine
        self.deployment_engine = deployment_engine
        self.infrastructure_engine = infrastructure_engine
    
    def _start_background_tasks(self):
        """Start background tasks"""
        # Start alert cleanup task
        threading.Thread(target=self._alert_cleanup_task, daemon=True).start()
        
        # Start correlation task
        threading.Thread(target=self._correlation_task, daemon=True).start()
    
    def _alert_cleanup_task(self):
        """Background task to clean up resolved alerts"""
        while True:
            try:
                now = datetime.now()
                cleanup_cutoff = now - timedelta(days=7)  # Keep alerts for 7 days
                
                # Clean up old resolved alerts from memory
                alerts_to_remove = []
                for alert_id, alert in self.active_alerts.items():
                    if (alert.status == AlertStatus.RESOLVED and 
                        alert.resolved_at and 
                        alert.resolved_at < cleanup_cutoff):
                        alerts_to_remove.append(alert_id)
                
                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]
                
                # Clean up silenced alerts
                silenced_to_remove = []
                for alert_id, silence_until in self.silenced_alerts.items():
                    if silence_until < now:
                        silenced_to_remove.append(alert_id)
                
                for alert_id in silenced_to_remove:
                    del self.silenced_alerts[alert_id]
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Alert cleanup task error: {e}")
                time.sleep(3600)
    
    def _correlation_task(self):
        """Background task for alert correlation"""
        while True:
            try:
                # Get active alerts
                firing_alerts = [alert for alert in self.active_alerts.values() 
                               if alert.status == AlertStatus.FIRING]
                
                if len(firing_alerts) > 1:
                    # Correlate alerts
                    correlated_groups = self.correlation_engine.correlate_alerts(firing_alerts)
                    
                    for group in correlated_groups:
                        self.logger.info(f"Correlated alert group: {group.name} ({len(group.alerts)} alerts)")
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Correlation task error: {e}")
                time.sleep(300)
    
    async def trigger_alert(self, name: str, severity: str, description: str, 
                           labels: Dict[str, str] = None, annotations: Dict[str, str] = None) -> str:
        """Trigger new alert"""
        start_time = time.time()
        
        try:
            # Create alert
            alert = Alert(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                severity=AlertSeverity(severity.lower()),
                status=AlertStatus.FIRING,
                labels=labels or {},
                annotations=annotations or {},
                started_at=datetime.now()
            )
            
            # Check for existing alert with same fingerprint (deduplication)
            existing_alert = None
            for existing in self.active_alerts.values():
                if existing.fingerprint == alert.fingerprint and existing.status == AlertStatus.FIRING:
                    existing_alert = existing
                    break
            
            if existing_alert:
                # Update existing alert timestamp
                existing_alert.started_at = datetime.now()
                self.logger.info(f"Updated existing alert: {existing_alert.name}")
                return existing_alert.id
            
            # Store alert
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Store in database
            self._store_alert_in_db(alert)
            
            # Send notifications
            await self._send_alert_notifications(alert)
            
            # Start escalation if configured
            escalation_policy = alert.annotations.get('escalation_policy')
            if escalation_policy:
                await self.escalation_manager.start_escalation(alert, escalation_policy)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['avg_processing_time'].append(processing_time)
            self.performance_metrics['alerts_processed'] += 1
            
            self.logger.info(f"Alert triggered: {alert.name} ({alert.id})")
            return alert.id
            
        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {e}")
            raise
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Resolve alert"""
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.active_alerts[alert_id]
        
        if alert.status == AlertStatus.RESOLVED:
            self.logger.warning(f"Alert {alert_id} already resolved")
            return
        
        # Update alert
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.resolved_by = resolved_by
        
        # Stop escalation
        self.escalation_manager.stop_escalation(alert_id)
        
        # Send resolution notifications
        await self._send_alert_notifications(alert, is_resolution=True)
        
        # Update in database
        self._update_alert_in_db(alert)
        
        self.logger.info(f"Alert resolved: {alert.name} ({alert_id}) by {resolved_by}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge alert"""
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.active_alerts[alert_id]
        
        if alert.status == AlertStatus.ACKNOWLEDGED:
            self.logger.warning(f"Alert {alert_id} already acknowledged")
            return
        
        # Update alert
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        
        # Stop escalation
        self.escalation_manager.stop_escalation(alert_id)
        
        # Update in database
        self._update_alert_in_db(alert)
        
        self.logger.info(f"Alert acknowledged: {alert.name} ({alert_id}) by {acknowledged_by}")
    
    def silence_alert(self, alert_id: str, duration: int):
        """Silence alert for specified duration (seconds)"""
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        silence_until = datetime.now() + timedelta(seconds=duration)
        self.silenced_alerts[alert_id] = silence_until
        
        # Update alert status
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SILENCED
        
        # Update in database
        self._update_alert_in_db(alert)
        
        self.logger.info(f"Alert silenced: {alert.name} ({alert_id}) until {silence_until}")
    
    async def _send_alert_notifications(self, alert: Alert, is_resolution: bool = False):
        """Send alert notifications"""
        if alert.id in self.silenced_alerts:
            self.logger.info(f"Alert {alert.id} is silenced, skipping notifications")
            return
        
        # Determine notification channels based on severity
        channels_to_notify = self._get_notification_channels_for_severity(alert.severity)
        
        # Create notification message
        message = self._create_notification_message(alert, is_resolution)
        
        # Send notifications
        for channel_name in channels_to_notify:
            try:
                delivery = await self.notification_manager.send_notification(channel_name, alert, message)
                
                # Store notification record
                self._store_notification_in_db(delivery)
                
                self.performance_metrics['notifications_sent'] += 1
                
            except Exception as e:
                self.logger.error(f"Notification failed for channel {channel_name}: {e}")
                self.performance_metrics['notification_failures'] += 1
    
    def _get_notification_channels_for_severity(self, severity: AlertSeverity) -> List[str]:
        """Get notification channels based on severity"""
        # This would be configured based on organization needs
        channel_mapping = {
            AlertSeverity.INFO: ['email'],
            AlertSeverity.LOW: ['email'],
            AlertSeverity.MEDIUM: ['email', 'slack'],
            AlertSeverity.HIGH: ['email', 'slack', 'sms'],
            AlertSeverity.CRITICAL: ['email', 'slack', 'sms', 'pagerduty'],
            AlertSeverity.EMERGENCY: ['email', 'slack', 'sms', 'pagerduty']
        }
        
        return channel_mapping.get(severity, ['email'])
    
    def _create_notification_message(self, alert: Alert, is_resolution: bool = False) -> str:
        """Create notification message"""
        action = "RESOLVED" if is_resolution else "TRIGGERED"
        
        message = f"""
Alert {action}: {alert.name}

Severity: {alert.severity.value.upper()}
Status: {alert.status.value}
Description: {alert.description}

Started: {alert.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        if is_resolution and alert.resolved_at:
            duration = alert.resolved_at - alert.started_at
            message += f"Duration: {duration}\n"
            message += f"Resolved: {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            if alert.resolved_by:
                message += f"Resolved by: {alert.resolved_by}\n"
        
        if alert.labels:
            message += f"\nLabels:\n"
            for key, value in alert.labels.items():
                message += f"  {key}: {value}\n"
        
        message += f"\nAlert ID: {alert.id}"
        
        return message
    
    def _store_alert_in_db(self, alert: Alert):
        """Store alert in database"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO alerts (id, name, description, severity, status, labels, annotations,
                                  started_at, fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id, alert.name, alert.description, alert.severity.value,
                alert.status.value, json.dumps(alert.labels), json.dumps(alert.annotations),
                alert.started_at, alert.fingerprint
            ))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store alert in database: {e}")
    
    def _update_alert_in_db(self, alert: Alert):
        """Update alert in database"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE alerts 
                SET status = ?, ended_at = ?, acknowledged_at = ?, acknowledged_by = ?,
                    resolved_at = ?, resolved_by = ?
                WHERE id = ?
            """, (
                alert.status.value, alert.ended_at, alert.acknowledged_at,
                alert.acknowledged_by, alert.resolved_at, alert.resolved_by, alert.id
            ))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to update alert in database: {e}")
    
    def _store_notification_in_db(self, delivery: NotificationDelivery):
        """Store notification delivery in database"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO notifications (id, alert_id, channel, recipient, status, sent_at, 
                                         delivered_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                delivery.id, delivery.alert_id, delivery.channel.value, delivery.recipient,
                delivery.status, delivery.sent_at, delivery.delivered_at, delivery.error
            ))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store notification in database: {e}")
    
    def register_notification_channel(self, name: str, config: NotificationConfig):
        """Register notification channel"""
        self.notification_manager.register_channel(name, config)
    
    def register_escalation_policy(self, policy: EscalationPolicy):
        """Register escalation policy"""
        self.escalation_manager.register_policy(policy)
    
    async def send_deployment_notification(self, deployment_id: str, status: str, message: str):
        """Send deployment notification"""
        severity = AlertSeverity.INFO if status == "success" else AlertSeverity.HIGH
        
        await self.trigger_alert(
            name=f"Deployment {status.capitalize()}",
            severity=severity.value,
            description=message,
            labels={
                'component': 'deployment',
                'deployment_id': deployment_id,
                'status': status
            }
        )
    
    def get_active_alerts(self, filters: Dict[str, Any] = None) -> List[Alert]:
        """Get active alerts with optional filters"""
        alerts = list(self.active_alerts.values())
        
        if filters:
            if 'severity' in filters:
                alerts = [a for a in alerts if a.severity.value == filters['severity']]
            if 'status' in filters:
                alerts = [a for a in alerts if a.status.value == filters['status']]
            if 'service' in filters:
                alerts = [a for a in alerts if a.labels.get('service') == filters['service']]
        
        return sorted(alerts, key=lambda x: x.started_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.active_alerts)
        
        if total_alerts == 0:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_status': {},
                'avg_processing_time': 0,
                'notifications_sent': self.performance_metrics['notifications_sent'],
                'notification_failures': self.performance_metrics['notification_failures']
            }
        
        # Count by severity
        by_severity = defaultdict(int)
        by_status = defaultdict(int)
        
        for alert in self.active_alerts.values():
            by_severity[alert.severity.value] += 1
            by_status[alert.status.value] += 1
        
        avg_processing_time = (
            statistics.mean(self.performance_metrics['avg_processing_time'])
            if self.performance_metrics['avg_processing_time'] else 0
        )
        
        return {
            'total_alerts': total_alerts,
            'by_severity': dict(by_severity),
            'by_status': dict(by_status),
            'avg_processing_time': avg_processing_time,
            'notifications_sent': self.performance_metrics['notifications_sent'],
            'notification_failures': self.performance_metrics['notification_failures'],
            'escalations_triggered': self.performance_metrics['escalations_triggered']
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close database connection
            if self.connection:
                self.connection.close()
            
            self.logger.info("AlertingEngine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"AlertingEngine cleanup error: {e}")


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create alerting engine
        engine = AlertingEngine()
        
        # Register email notification channel
        email_config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            config={
                'smtp_host': 'smtp.gmail.com',
                'smtp_port': 587,
                'from_address': 'alerts@company.com',
                'to_address': 'oncall@company.com',
                'username': 'alerts@company.com',
                'password': 'app-password'
            }
        )
        engine.register_notification_channel('email', email_config)
        
        # Register escalation policy
        escalation_policy = EscalationPolicy(
            name="critical_alerts",
            description="Escalation policy for critical alerts",
            rules=[
                EscalationRule(
                    name="notify_oncall",
                    condition="severity >= critical",
                    delay=0,
                    action=EscalationAction.NOTIFY,
                    config={'channel': 'email'}
                ),
                EscalationRule(
                    name="escalate_to_lead",
                    condition="not acknowledged after 15 minutes",
                    delay=900,
                    action=EscalationAction.ESCALATE,
                    config={'team': 'engineering_leads'}
                )
            ]
        )
        engine.register_escalation_policy(escalation_policy)
        
        try:
            # Trigger test alert
            alert_id = await engine.trigger_alert(
                name="High CPU Usage",
                severity="critical",
                description="CPU usage has exceeded 90% for more than 5 minutes",
                labels={
                    'service': 'manufacturing-line',
                    'component': 'line-controller',
                    'datacenter': 'dc-east-1'
                },
                annotations={
                    'escalation_policy': 'critical_alerts',
                    'runbook_url': 'https://docs.company.com/runbooks/high-cpu'
                }
            )
            
            print(f"Alert triggered: {alert_id}")
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Get alert statistics
            stats = engine.get_alert_statistics()
            print(f"Alert statistics: {stats}")
            
            # Get active alerts
            active_alerts = engine.get_active_alerts()
            print(f"Active alerts: {len(active_alerts)}")
            
            # Acknowledge alert
            await engine.acknowledge_alert(alert_id, "test-user")
            print(f"Alert acknowledged: {alert_id}")
            
            # Resolve alert
            await engine.resolve_alert(alert_id, "test-user")
            print(f"Alert resolved: {alert_id}")
            
        except Exception as e:
            print(f"Alerting test failed: {e}")
        
        finally:
            await engine.cleanup()
    
    asyncio.run(main())