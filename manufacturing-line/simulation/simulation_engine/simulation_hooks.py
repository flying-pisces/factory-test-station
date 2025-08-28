"""Simulation hooks and integration with manufacturing line system."""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from .base_simulation import SimulationResult, SimulationStatus
from .digital_twin import BaseDigitalTwin, DigitalTwinManager, digital_twin_manager


class SimulationEventType(Enum):
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_COMPLETED = "simulation_completed"
    SIMULATION_ERROR = "simulation_error"
    PREDICTION_UPDATED = "prediction_updated"
    TWIN_SYNC = "twin_sync"
    BOTTLENECK_DETECTED = "bottleneck_detected"
    PERFORMANCE_ALERT = "performance_alert"


@dataclass
class SimulationEvent:
    """Event data structure for simulation events."""
    event_type: SimulationEventType
    component_id: str
    timestamp: float
    data: Dict[str, Any]
    severity: str = "info"  # info, warning, error, critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'component_id': self.component_id,
            'timestamp': self.timestamp,
            'data': self.data,
            'severity': self.severity
        }


class SimulationHookManager:
    """Manages hooks and integration between simulation and manufacturing line system."""
    
    def __init__(self):
        self.hooks: Dict[SimulationEventType, List[Callable]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_history: List[SimulationEvent] = []
        self.max_history_size = 1000
        
        # Integration endpoints
        self.external_webhooks: List[Dict[str, Any]] = []
        self.line_controller_callback: Optional[Callable] = None
        self.database_callback: Optional[Callable] = None
    
    def register_hook(self, event_type: SimulationEventType, callback: Callable):
        """Register a hook for specific simulation events."""
        if event_type not in self.hooks:
            self.hooks[event_type] = []
        
        self.hooks[event_type].append(callback)
        self.logger.info(f"Registered hook for {event_type.value}")
    
    def unregister_hook(self, event_type: SimulationEventType, callback: Callable):
        """Unregister a hook."""
        if event_type in self.hooks and callback in self.hooks[event_type]:
            self.hooks[event_type].remove(callback)
    
    def trigger_event(self, event: SimulationEvent):
        """Trigger all hooks for a specific event."""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        # Trigger registered hooks
        if event.event_type in self.hooks:
            for callback in self.hooks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Hook callback error: {e}")
        
        # Send to external systems
        self._send_to_external_systems(event)
        
        self.logger.info(f"Triggered event: {event.event_type.value} for {event.component_id}")
    
    def _send_to_external_systems(self, event: SimulationEvent):
        """Send event to external systems (MES, database, etc.)."""
        # Send to line controller
        if self.line_controller_callback:
            try:
                self.line_controller_callback(event)
            except Exception as e:
                self.logger.error(f"Line controller callback error: {e}")
        
        # Send to database
        if self.database_callback:
            try:
                self.database_callback(event)
            except Exception as e:
                self.logger.error(f"Database callback error: {e}")
        
        # Send to external webhooks
        for webhook in self.external_webhooks:
            self._send_webhook(event, webhook)
    
    def _send_webhook(self, event: SimulationEvent, webhook_config: Dict[str, Any]):
        """Send event to external webhook."""
        try:
            import requests
            
            payload = {
                'event': event.to_dict(),
                'source': 'manufacturing_line_simulation',
                'timestamp': event.timestamp
            }
            
            headers = {'Content-Type': 'application/json'}
            if 'auth_token' in webhook_config:
                headers['Authorization'] = f"Bearer {webhook_config['auth_token']}"
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code not in [200, 201, 202]:
                self.logger.warning(f"Webhook failed: {response.status_code}")
        
        except Exception as e:
            self.logger.error(f"Webhook error: {e}")
    
    def add_external_webhook(self, url: str, auth_token: Optional[str] = None, 
                           events: Optional[List[SimulationEventType]] = None):
        """Add external webhook endpoint."""
        webhook = {
            'url': url,
            'auth_token': auth_token,
            'events': events or []
        }
        self.external_webhooks.append(webhook)
    
    def set_line_controller_callback(self, callback: Callable):
        """Set callback for line controller integration."""
        self.line_controller_callback = callback
    
    def set_database_callback(self, callback: Callable):
        """Set callback for database integration."""
        self.database_callback = callback
    
    def get_event_history(self, 
                          event_type: Optional[SimulationEventType] = None,
                          component_id: Optional[str] = None,
                          limit: int = 100) -> List[SimulationEvent]:
        """Get event history with optional filtering."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if component_id:
            events = [e for e in events if e.component_id == component_id]
        
        return events[-limit:]


class SimulationIntegrationService:
    """Service for integrating simulation with manufacturing line components."""
    
    def __init__(self, hook_manager: SimulationHookManager, 
                 twin_manager: DigitalTwinManager):
        self.hook_manager = hook_manager
        self.twin_manager = twin_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Register default hooks
        self._register_default_hooks()
    
    def _register_default_hooks(self):
        """Register default integration hooks."""
        self.hook_manager.register_hook(
            SimulationEventType.SIMULATION_COMPLETED,
            self._handle_simulation_completed
        )
        
        self.hook_manager.register_hook(
            SimulationEventType.BOTTLENECK_DETECTED,
            self._handle_bottleneck_detected
        )
        
        self.hook_manager.register_hook(
            SimulationEventType.PERFORMANCE_ALERT,
            self._handle_performance_alert
        )
    
    def _handle_simulation_completed(self, event: SimulationEvent):
        """Handle simulation completion."""
        component_id = event.component_id
        result_data = event.data.get('simulation_result', {})
        
        # Update digital twin with results
        twin = self.twin_manager.get_twin(component_id)
        if twin and 'predictions' in result_data:
            twin.predicted_data.update(result_data['predictions'])
        
        # Trigger performance analysis
        self._analyze_performance(component_id, result_data)
    
    def _handle_bottleneck_detected(self, event: SimulationEvent):
        """Handle bottleneck detection."""
        bottleneck_data = event.data
        
        # Create performance alert
        alert_event = SimulationEvent(
            event_type=SimulationEventType.PERFORMANCE_ALERT,
            component_id=event.component_id,
            timestamp=time.time(),
            data={
                'alert_type': 'bottleneck',
                'bottleneck_station': bottleneck_data.get('station'),
                'predicted_impact': bottleneck_data.get('impact'),
                'recommendations': bottleneck_data.get('recommendations', [])
            },
            severity='warning'
        )
        
        self.hook_manager.trigger_event(alert_event)
    
    def _handle_performance_alert(self, event: SimulationEvent):
        """Handle performance alerts."""
        alert_data = event.data
        alert_type = alert_data.get('alert_type')
        
        if alert_type == 'bottleneck':
            self.logger.warning(f"Bottleneck detected at {alert_data.get('bottleneck_station')}")
        elif alert_type == 'low_efficiency':
            self.logger.warning(f"Low efficiency alert for {event.component_id}")
        elif alert_type == 'prediction_deviation':
            self.logger.warning(f"Prediction deviation for {event.component_id}")
    
    def _analyze_performance(self, component_id: str, result_data: Dict[str, Any]):
        """Analyze simulation performance and generate alerts."""
        predictions = result_data.get('predictions', {})
        
        # Check for efficiency issues
        efficiency = predictions.get('predicted_efficiency', 1.0)
        if efficiency < 0.7:  # Less than 70% efficiency
            alert_event = SimulationEvent(
                event_type=SimulationEventType.PERFORMANCE_ALERT,
                component_id=component_id,
                timestamp=time.time(),
                data={
                    'alert_type': 'low_efficiency',
                    'efficiency': efficiency,
                    'threshold': 0.7
                },
                severity='warning'
            )
            self.hook_manager.trigger_event(alert_event)
        
        # Check for bottleneck risk
        bottleneck_risk = predictions.get('bottleneck_risk', 'low')
        if bottleneck_risk == 'high':
            alert_event = SimulationEvent(
                event_type=SimulationEventType.BOTTLENECK_DETECTED,
                component_id=component_id,
                timestamp=time.time(),
                data={
                    'station': component_id,
                    'risk_level': bottleneck_risk,
                    'impact': 'high',
                    'recommendations': [
                        'Increase station capacity',
                        'Optimize process flow',
                        'Add parallel processing'
                    ]
                },
                severity='warning'
            )
            self.hook_manager.trigger_event(alert_event)
    
    def process_real_time_data(self, component_id: str, data: Dict[str, Any]):
        """Process real-time data from manufacturing line components."""
        # Update digital twin
        twin = self.twin_manager.get_twin(component_id)
        if twin:
            twin.update_real_data(data)
            
            # Check for prediction deviations
            if twin.predicted_data:
                deviation = self._calculate_prediction_deviation(twin)
                if deviation > 0.3:  # More than 30% deviation
                    alert_event = SimulationEvent(
                        event_type=SimulationEventType.PERFORMANCE_ALERT,
                        component_id=component_id,
                        timestamp=time.time(),
                        data={
                            'alert_type': 'prediction_deviation',
                            'deviation': deviation,
                            'threshold': 0.3,
                            'real_data': data,
                            'predicted_data': twin.predicted_data
                        },
                        severity='warning'
                    )
                    self.hook_manager.trigger_event(alert_event)
    
    def _calculate_prediction_deviation(self, twin: BaseDigitalTwin) -> float:
        """Calculate deviation between predictions and reality."""
        return twin.calculate_sync_accuracy()
    
    def trigger_scenario_simulation(self, scenario_name: str, 
                                  parameters: Dict[str, Any]) -> str:
        """Trigger what-if scenario simulation."""
        from .base_simulation import simulation_manager
        from ..jaamsim_integration.jaamsim_simulation import create_jaamsim_config
        
        # Create scenario configuration
        config = create_jaamsim_config(
            config_id=f"scenario_{scenario_name}",
            cfg_file_path=parameters.get('config_file', 
                                       'stations/fixture/simulation/cfg/1up/1-up-station-simulation.cfg'),
            parameters=parameters.get('simulation_params', {}),
            real_time_factor=parameters.get('real_time_factor', 16.0),
            max_runtime=parameters.get('max_runtime', 300.0)
        )
        
        # Run simulation
        simulation_id = simulation_manager.run_scenario(scenario_name, config)
        
        # Trigger event
        event = SimulationEvent(
            event_type=SimulationEventType.SIMULATION_STARTED,
            component_id=scenario_name,
            timestamp=time.time(),
            data={
                'simulation_id': simulation_id,
                'scenario_name': scenario_name,
                'parameters': parameters
            }
        )
        self.hook_manager.trigger_event(event)
        
        return simulation_id
    
    def get_line_performance_summary(self) -> Dict[str, Any]:
        """Get overall line performance summary from all digital twins."""
        predictions = self.twin_manager.get_line_predictions()
        sync_status = self.twin_manager.get_sync_status()
        
        # Calculate summary metrics
        summary = {
            'timestamp': time.time(),
            'line_predictions': predictions,
            'sync_status': sync_status,
            'alerts': {
                'critical': len([e for e in self.hook_manager.event_history[-50:] 
                               if e.severity == 'critical']),
                'warning': len([e for e in self.hook_manager.event_history[-50:] 
                              if e.severity == 'warning']),
                'info': len([e for e in self.hook_manager.event_history[-50:] 
                           if e.severity == 'info'])
            },
            'active_twins': len([t for t in self.twin_manager.twins.values() 
                               if t._is_active])
        }
        
        return summary


# Global instances
simulation_hook_manager = SimulationHookManager()
simulation_integration_service = SimulationIntegrationService(
    simulation_hook_manager, 
    digital_twin_manager
)


def setup_simulation_integration(line_controller_callback: Optional[Callable] = None,
                               database_callback: Optional[Callable] = None):
    """Setup simulation integration with manufacturing line system."""
    if line_controller_callback:
        simulation_hook_manager.set_line_controller_callback(line_controller_callback)
    
    if database_callback:
        simulation_hook_manager.set_database_callback(database_callback)
    
    logging.getLogger(__name__).info("Simulation integration setup complete")


def trigger_simulation_event(event_type: str, component_id: str, 
                           data: Dict[str, Any], severity: str = "info"):
    """Convenience function to trigger simulation events."""
    event = SimulationEvent(
        event_type=SimulationEventType(event_type),
        component_id=component_id,
        timestamp=time.time(),
        data=data,
        severity=severity
    )
    
    simulation_hook_manager.trigger_event(event)