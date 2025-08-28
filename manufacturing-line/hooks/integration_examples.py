"""Integration hook examples for external systems."""

import json
import requests
from typing import Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class HookEvent:
    """Structure for hook events."""
    event_type: str
    source: str
    timestamp: float
    data: Dict[str, Any]


class BaseHook(ABC):
    """Base class for integration hooks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
    
    @abstractmethod
    def process_event(self, event: HookEvent) -> bool:
        """Process incoming event."""
        pass
    
    def validate_event(self, event: HookEvent) -> bool:
        """Validate event structure."""
        required_fields = ['event_type', 'source', 'timestamp', 'data']
        return all(hasattr(event, field) for field in required_fields)


class MESIntegrationHook(BaseHook):
    """Hook for Manufacturing Execution System integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mes_endpoint = config['mes_endpoint']
        self.auth_token = config['auth_token']
    
    def process_event(self, event: HookEvent) -> bool:
        """Send manufacturing events to MES system."""
        if not self.enabled or not self.validate_event(event):
            return False
        
        # Map line events to MES format
        mes_data = self._transform_to_mes_format(event)
        
        try:
            response = requests.post(
                f"{self.mes_endpoint}/production-events",
                json=mes_data,
                headers={
                    'Authorization': f'Bearer {self.auth_token}',
                    'Content-Type': 'application/json'
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"MES integration error: {e}")
            return False
    
    def _transform_to_mes_format(self, event: HookEvent) -> Dict[str, Any]:
        """Transform line event to MES format."""
        return {
            'event_id': f"LINE_{event.timestamp}",
            'event_type': event.event_type,
            'production_line': event.data.get('line_id', 'UNKNOWN'),
            'station': event.data.get('station_id', 'UNKNOWN'),
            'part_number': event.data.get('dut_id', 'UNKNOWN'),
            'timestamp': event.timestamp,
            'result': event.data.get('result', 'UNKNOWN'),
            'measurements': event.data.get('measurements', [])
        }


class QualitySystemHook(BaseHook):
    """Hook for Quality Management System integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.qms_endpoint = config['qms_endpoint'] 
        self.api_key = config['api_key']
        self.quality_events = [
            'test_completed',
            'quality_failure',
            'retest_required',
            'inspection_completed'
        ]
    
    def process_event(self, event: HookEvent) -> bool:
        """Send quality events to QMS."""
        if event.event_type not in self.quality_events:
            return True  # Not a quality event, skip
        
        quality_data = self._extract_quality_data(event)
        
        try:
            response = requests.post(
                f"{self.qms_endpoint}/quality-events",
                json=quality_data,
                headers={'X-API-Key': self.api_key},
                timeout=10
            )
            return response.status_code in [200, 201]
        except Exception as e:
            print(f"QMS integration error: {e}")
            return False
    
    def _extract_quality_data(self, event: HookEvent) -> Dict[str, Any]:
        """Extract quality metrics from event."""
        return {
            'part_id': event.data.get('dut_id'),
            'station_id': event.data.get('station_id'),
            'test_result': event.data.get('result'),
            'defect_codes': event.data.get('defect_codes', []),
            'measurements': event.data.get('measurements', []),
            'operator_id': event.data.get('operator_id'),
            'timestamp': event.timestamp
        }


class AlertingSystemHook(BaseHook):
    """Hook for alerting and notification systems."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.slack_webhook = config.get('slack_webhook')
        self.email_config = config.get('email_config', {})
        self.alert_levels = {
            'critical': ['emergency_stop', 'line_down', 'safety_violation'],
            'warning': ['station_error', 'quality_alert', 'maintenance_due'],
            'info': ['shift_change', 'production_milestone']
        }
    
    def process_event(self, event: HookEvent) -> bool:
        """Send alerts based on event severity."""
        severity = self._determine_severity(event)
        
        if severity == 'critical':
            return self._send_critical_alert(event)
        elif severity == 'warning':
            return self._send_warning_alert(event)
        elif severity == 'info':
            return self._send_info_notification(event)
        
        return True
    
    def _determine_severity(self, event: HookEvent) -> str:
        """Determine alert severity level."""
        for level, event_types in self.alert_levels.items():
            if event.event_type in event_types:
                return level
        return 'info'
    
    def _send_critical_alert(self, event: HookEvent) -> bool:
        """Send critical alert via multiple channels."""
        success = True
        
        # Slack notification
        if self.slack_webhook:
            slack_data = {
                'text': f'🚨 CRITICAL ALERT: {event.event_type}',
                'attachments': [{
                    'color': 'danger',
                    'fields': [
                        {'title': 'Station', 'value': event.data.get('station_id', 'Unknown'), 'short': True},
                        {'title': 'Time', 'value': event.timestamp, 'short': True},
                        {'title': 'Details', 'value': str(event.data), 'short': False}
                    ]
                }]
            }
            try:
                requests.post(self.slack_webhook, json=slack_data, timeout=5)
            except:
                success = False
        
        # Email alert (implementation depends on email service)
        # self._send_email_alert(event, 'critical')
        
        return success
    
    def _send_warning_alert(self, event: HookEvent) -> bool:
        """Send warning alert."""
        # Implementation for warning alerts
        return True
    
    def _send_info_notification(self, event: HookEvent) -> bool:
        """Send info notification."""
        # Implementation for info notifications  
        return True


class DataWarehouseHook(BaseHook):
    """Hook for data warehouse/analytics integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.warehouse_endpoint = config['warehouse_endpoint']
        self.batch_size = config.get('batch_size', 100)
        self.batch_buffer = []
    
    def process_event(self, event: HookEvent) -> bool:
        """Buffer events for batch processing to data warehouse."""
        # Transform to warehouse format
        warehouse_record = {
            'event_id': f"{event.source}_{event.timestamp}",
            'event_type': event.event_type,
            'source_system': 'manufacturing_line',
            'timestamp': event.timestamp,
            'line_id': event.data.get('line_id'),
            'station_id': event.data.get('station_id'),
            'dut_id': event.data.get('dut_id'),
            'metrics': event.data.get('measurements', {}),
            'raw_data': event.data
        }
        
        self.batch_buffer.append(warehouse_record)
        
        # Send batch when full
        if len(self.batch_buffer) >= self.batch_size:
            return self._send_batch()
        
        return True
    
    def _send_batch(self) -> bool:
        """Send batch of records to data warehouse."""
        try:
            response = requests.post(
                f"{self.warehouse_endpoint}/batch-insert",
                json={'records': self.batch_buffer},
                timeout=30
            )
            
            if response.status_code == 200:
                self.batch_buffer.clear()
                return True
            else:
                print(f"Warehouse batch failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Warehouse error: {e}")
            return False
    
    def flush_batch(self):
        """Force send remaining buffered records."""
        if self.batch_buffer:
            self._send_batch()


class IoTPlatformHook(BaseHook):
    """Hook for IoT platform integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.iot_endpoint = config['iot_endpoint']
        self.device_mappings = config.get('device_mappings', {})
    
    def process_event(self, event: HookEvent) -> bool:
        """Send equipment telemetry to IoT platform."""
        device_id = self._get_device_id(event)
        if not device_id:
            return True  # Skip if no device mapping
        
        telemetry = self._extract_telemetry(event)
        
        try:
            response = requests.post(
                f"{self.iot_endpoint}/devices/{device_id}/telemetry",
                json=telemetry,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"IoT platform error: {e}")
            return False
    
    def _get_device_id(self, event: HookEvent) -> str:
        """Map station/component to IoT device ID."""
        station_id = event.data.get('station_id')
        component_id = event.data.get('component_id')
        
        return self.device_mappings.get(
            f"{station_id}:{component_id}",
            self.device_mappings.get(station_id)
        )
    
    def _extract_telemetry(self, event: HookEvent) -> Dict[str, Any]:
        """Extract telemetry data from event."""
        return {
            'timestamp': event.timestamp,
            'temperature': event.data.get('temperature'),
            'vibration': event.data.get('vibration'),
            'power_consumption': event.data.get('power'),
            'cycle_count': event.data.get('cycle_count'),
            'status': event.data.get('status')
        }


# Hook registry and dispatcher
class HookManager:
    """Manages all integration hooks."""
    
    def __init__(self):
        self.hooks: List[BaseHook] = []
    
    def register_hook(self, hook: BaseHook):
        """Register a new hook."""
        self.hooks.append(hook)
    
    def dispatch_event(self, event: HookEvent):
        """Send event to all registered hooks."""
        for hook in self.hooks:
            try:
                hook.process_event(event)
            except Exception as e:
                print(f"Hook error: {e}")
    
    def load_from_config(self, config_file: str):
        """Load hooks from configuration file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        for hook_config in config.get('hooks', []):
            hook_type = hook_config['type']
            
            if hook_type == 'mes':
                self.register_hook(MESIntegrationHook(hook_config))
            elif hook_type == 'quality':
                self.register_hook(QualitySystemHook(hook_config))
            elif hook_type == 'alerting':
                self.register_hook(AlertingSystemHook(hook_config))
            elif hook_type == 'warehouse':
                self.register_hook(DataWarehouseHook(hook_config))
            elif hook_type == 'iot':
                self.register_hook(IoTPlatformHook(hook_config))


# Example usage
if __name__ == "__main__":
    # Initialize hook manager
    hook_manager = HookManager()
    
    # Create sample hooks
    mes_hook = MESIntegrationHook({
        'enabled': True,
        'mes_endpoint': 'https://mes.factory.com/api',
        'auth_token': 'your_token_here'
    })
    
    quality_hook = QualitySystemHook({
        'enabled': True,
        'qms_endpoint': 'https://qms.factory.com/api',
        'api_key': 'your_api_key'
    })
    
    # Register hooks
    hook_manager.register_hook(mes_hook)
    hook_manager.register_hook(quality_hook)
    
    # Create sample event
    test_event = HookEvent(
        event_type='test_completed',
        source='ICT_01',
        timestamp=1706450400.0,
        data={
            'line_id': 'SMT_LINE_01',
            'station_id': 'ICT_01',
            'dut_id': 'DUT_001234',
            'result': 'pass',
            'measurements': [
                {'parameter': 'voltage_5v', 'value': 5.02, 'unit': 'V'}
            ]
        }
    )
    
    # Dispatch event to all hooks
    hook_manager.dispatch_event(test_event)