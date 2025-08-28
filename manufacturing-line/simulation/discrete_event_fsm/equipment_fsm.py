"""Manufacturing Equipment Finite State Machine."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time
import random

from base_fsm import BaseFiniteStateMachine, FSMState, StateTransition, DiscreteEvent, global_scheduler


class EquipmentState(Enum):
    """Equipment states in the manufacturing process."""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    MEASURING = "measuring"
    CALIBRATING = "calibrating"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class EquipmentEvent(Enum):
    """Equipment discrete events with fixed durations."""
    CMD_INITIALIZE = "cmd_initialize"       # 10.0s - Initialize equipment
    CMD_MEASURE = "cmd_measure"             # Variable - Execute measurement
    CMD_CALIBRATE = "cmd_calibrate"         # 30.0s - Calibration sequence
    CMD_START = "cmd_start"                 # 2.0s - Start operation
    CMD_STOP = "cmd_stop"                   # 1.0s - Stop operation
    CMD_RESET = "cmd_reset"                 # 5.0s - Reset equipment
    CMD_SHUTDOWN = "cmd_shutdown"           # 8.0s - Shutdown sequence
    CMD_MAINTENANCE = "cmd_maintenance"     # 60.0s - Maintenance cycle


@dataclass
class EquipmentData:
    """Equipment-specific data structure."""
    equipment_id: str
    equipment_type: str  # DMM, Oscilloscope, Power Supply, etc.
    station_id: str
    model: str
    serial_number: str
    measurement_count: int = 0
    calibration_count: int = 0
    last_calibration: float = 0.0
    error_count: int = 0
    maintenance_hours: float = 0.0
    current_measurement: Optional[Dict[str, Any]] = None
    measurement_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.measurement_history is None:
            self.measurement_history = []


class EquipmentFSM(BaseFiniteStateMachine):
    """Equipment Finite State Machine with discrete event processing."""
    
    def __init__(self, equipment_data: EquipmentData, scheduler=None):
        if scheduler is None:
            scheduler = global_scheduler
        
        super().__init__(f"EQUIPMENT_{equipment_data.equipment_id}", scheduler)
        self.equipment_data = equipment_data
        self.state_data['equipment_data'] = equipment_data
        self.state_data['measurement_duration'] = 5.0  # Default measurement duration
        
    def _initialize_fsm(self):
        """Initialize Equipment FSM states, transitions, and event methods."""
        
        # Define states
        self.add_state(FSMState(
            state_name=EquipmentState.OFFLINE.value,
            entry_action=self._on_enter_offline
        ))
        
        self.add_state(FSMState(
            state_name=EquipmentState.INITIALIZING.value,
            entry_action=self._on_enter_initializing
        ))
        
        self.add_state(FSMState(
            state_name=EquipmentState.IDLE.value,
            entry_action=self._on_enter_idle
        ))
        
        self.add_state(FSMState(
            state_name=EquipmentState.BUSY.value,
            entry_action=self._on_enter_busy
        ))
        
        self.add_state(FSMState(
            state_name=EquipmentState.MEASURING.value,
            entry_action=self._on_enter_measuring
        ))
        
        self.add_state(FSMState(
            state_name=EquipmentState.CALIBRATING.value,
            entry_action=self._on_enter_calibrating
        ))
        
        self.add_state(FSMState(
            state_name=EquipmentState.ERROR.value,
            entry_action=self._on_enter_error
        ))
        
        self.add_state(FSMState(
            state_name=EquipmentState.MAINTENANCE.value,
            entry_action=self._on_enter_maintenance
        ))
        
        self.add_state(FSMState(
            state_name=EquipmentState.SHUTDOWN.value,
            entry_action=self._on_enter_shutdown
        ))
        
        # Define transitions
        self.add_transition(StateTransition(
            from_state=EquipmentState.OFFLINE.value,
            to_state=EquipmentState.INITIALIZING.value,
            trigger_event=EquipmentEvent.CMD_INITIALIZE.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.INITIALIZING.value,
            to_state=EquipmentState.IDLE.value,
            trigger_event=EquipmentEvent.CMD_START.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.IDLE.value,
            to_state=EquipmentState.BUSY.value,
            trigger_event=EquipmentEvent.CMD_START.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.BUSY.value,
            to_state=EquipmentState.MEASURING.value,
            trigger_event=EquipmentEvent.CMD_MEASURE.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.MEASURING.value,
            to_state=EquipmentState.IDLE.value,
            trigger_event=EquipmentEvent.CMD_STOP.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.IDLE.value,
            to_state=EquipmentState.CALIBRATING.value,
            trigger_event=EquipmentEvent.CMD_CALIBRATE.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.CALIBRATING.value,
            to_state=EquipmentState.IDLE.value,
            trigger_event=EquipmentEvent.CMD_STOP.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.ERROR.value,
            to_state=EquipmentState.MAINTENANCE.value,
            trigger_event=EquipmentEvent.CMD_MAINTENANCE.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.MAINTENANCE.value,
            to_state=EquipmentState.OFFLINE.value,
            trigger_event=EquipmentEvent.CMD_RESET.value
        ))
        
        self.add_transition(StateTransition(
            from_state=EquipmentState.IDLE.value,
            to_state=EquipmentState.SHUTDOWN.value,
            trigger_event=EquipmentEvent.CMD_SHUTDOWN.value
        ))
        
        # Add discrete event methods with fixed durations
        self.add_event_method(EquipmentEvent.CMD_INITIALIZE.value, self._cmd_initialize, 10.0)
        self.add_event_method(EquipmentEvent.CMD_MEASURE.value, self._cmd_measure, self.state_data['measurement_duration'])
        self.add_event_method(EquipmentEvent.CMD_CALIBRATE.value, self._cmd_calibrate, 30.0)
        self.add_event_method(EquipmentEvent.CMD_START.value, self._cmd_start, 2.0)
        self.add_event_method(EquipmentEvent.CMD_STOP.value, self._cmd_stop, 1.0)
        self.add_event_method(EquipmentEvent.CMD_RESET.value, self._cmd_reset, 5.0)
        self.add_event_method(EquipmentEvent.CMD_SHUTDOWN.value, self._cmd_shutdown, 8.0)
        self.add_event_method(EquipmentEvent.CMD_MAINTENANCE.value, self._cmd_maintenance, 60.0)
        
        # Set initial state
        self.current_state = EquipmentState.OFFLINE.value
    
    # State entry actions
    def _on_enter_offline(self, fsm, event):
        """Actions when entering OFFLINE state."""
        self.logger.info(f"Equipment {self.equipment_data.equipment_id} offline")
    
    def _on_enter_initializing(self, fsm, event):
        """Actions when entering INITIALIZING state."""
        self.logger.info(f"Equipment {self.equipment_data.equipment_id} initializing")
    
    def _on_enter_idle(self, fsm, event):
        """Actions when entering IDLE state."""
        self.logger.info(f"Equipment {self.equipment_data.equipment_id} idle and ready")
    
    def _on_enter_busy(self, fsm, event):
        """Actions when entering BUSY state."""
        self.logger.info(f"Equipment {self.equipment_data.equipment_id} busy")
    
    def _on_enter_measuring(self, fsm, event):
        """Actions when entering MEASURING state."""
        self.logger.info(f"Equipment {self.equipment_data.equipment_id} performing measurement")
    
    def _on_enter_calibrating(self, fsm, event):
        """Actions when entering CALIBRATING state."""
        self.logger.info(f"Equipment {self.equipment_data.equipment_id} calibrating")
    
    def _on_enter_error(self, fsm, event):
        """Actions when entering ERROR state."""
        self.equipment_data.error_count += 1
        self.logger.error(f"Equipment {self.equipment_data.equipment_id} in error state (count: {self.equipment_data.error_count})")
    
    def _on_enter_maintenance(self, fsm, event):
        """Actions when entering MAINTENANCE state."""
        self.logger.info(f"Equipment {self.equipment_data.equipment_id} undergoing maintenance")
    
    def _on_enter_shutdown(self, fsm, event):
        """Actions when entering SHUTDOWN state."""
        self.logger.info(f"Equipment {self.equipment_data.equipment_id} shutting down")
    
    # Discrete event methods (fixed duration)
    def _cmd_initialize(self, event: DiscreteEvent):
        """CMD_INITIALIZE event - 10.0s duration."""
        self.logger.debug(f"Equipment {self.equipment_data.equipment_id}: Initialization sequence")
        
        # Simulate initialization steps
        init_steps = [
            "Power-on self-test",
            "Hardware detection",
            "Firmware loading",
            "Interface configuration",
            "System ready"
        ]
        
        return {
            'initialized': True,
            'steps_completed': init_steps,
            'timestamp': self.scheduler.current_time
        }
    
    def _cmd_measure(self, event: DiscreteEvent):
        """CMD_MEASURE event - Variable duration."""
        measurement_type = event.conditions.get('measurement_type', 'default')
        target_value = event.conditions.get('target_value', 0.0)
        tolerance = event.conditions.get('tolerance', 0.1)
        
        # Simulate measurement based on equipment type
        if self.equipment_data.equipment_type == 'DMM':
            measured_value = self._simulate_dmm_measurement(target_value, tolerance)
        elif self.equipment_data.equipment_type == 'OSCILLOSCOPE':
            measured_value = self._simulate_scope_measurement(target_value, tolerance)
        elif self.equipment_data.equipment_type == 'POWER_SUPPLY':
            measured_value = self._simulate_power_measurement(target_value, tolerance)
        else:
            measured_value = target_value + random.uniform(-tolerance, tolerance)
        
        measurement_result = {
            'measurement_type': measurement_type,
            'measured_value': measured_value,
            'target_value': target_value,
            'tolerance': tolerance,
            'result': 'PASS' if abs(measured_value - target_value) <= tolerance else 'FAIL',
            'timestamp': self.scheduler.current_time,
            'equipment_id': self.equipment_data.equipment_id
        }
        
        self.equipment_data.measurement_count += 1
        self.equipment_data.current_measurement = measurement_result
        self.equipment_data.measurement_history.append(measurement_result)
        
        self.logger.debug(f"Equipment {self.equipment_data.equipment_id}: Measurement completed - {measurement_result['result']}")
        
        return measurement_result
    
    def _cmd_calibrate(self, event: DiscreteEvent):
        """CMD_CALIBRATE event - 30.0s duration."""
        calibration_type = event.conditions.get('calibration_type', 'standard')
        
        # Simulate calibration process
        calibration_points = []
        for i in range(5):  # 5 calibration points
            point = {
                'point': i + 1,
                'reference': 1.0 * (i + 1),
                'measured': 1.0 * (i + 1) + random.uniform(-0.01, 0.01),
                'error': 0.0
            }
            point['error'] = abs(point['measured'] - point['reference'])
            calibration_points.append(point)
        
        calibration_result = {
            'calibration_type': calibration_type,
            'points': calibration_points,
            'max_error': max(p['error'] for p in calibration_points),
            'result': 'PASS' if max(p['error'] for p in calibration_points) < 0.05 else 'FAIL',
            'timestamp': self.scheduler.current_time
        }
        
        self.equipment_data.calibration_count += 1
        self.equipment_data.last_calibration = self.scheduler.current_time
        
        self.logger.debug(f"Equipment {self.equipment_data.equipment_id}: Calibration completed - {calibration_result['result']}")
        
        return calibration_result
    
    def _cmd_start(self, event: DiscreteEvent):
        """CMD_START event - 2.0s duration."""
        self.logger.debug(f"Equipment {self.equipment_data.equipment_id}: Started")
        return {'started': True, 'timestamp': self.scheduler.current_time}
    
    def _cmd_stop(self, event: DiscreteEvent):
        """CMD_STOP event - 1.0s duration."""
        self.equipment_data.current_measurement = None
        self.logger.debug(f"Equipment {self.equipment_data.equipment_id}: Stopped")
        return {'stopped': True, 'timestamp': self.scheduler.current_time}
    
    def _cmd_reset(self, event: DiscreteEvent):
        """CMD_RESET event - 5.0s duration."""
        self.equipment_data.error_count = 0
        self.equipment_data.current_measurement = None
        self.logger.debug(f"Equipment {self.equipment_data.equipment_id}: Reset completed")
        return {'reset': True, 'timestamp': self.scheduler.current_time}
    
    def _cmd_shutdown(self, event: DiscreteEvent):
        """CMD_SHUTDOWN event - 8.0s duration."""
        self.logger.debug(f"Equipment {self.equipment_data.equipment_id}: Shutdown completed")
        return {'shutdown': True, 'timestamp': self.scheduler.current_time}
    
    def _cmd_maintenance(self, event: DiscreteEvent):
        """CMD_MAINTENANCE event - 60.0s duration."""
        maintenance_duration = event.conditions.get('duration', 60.0)
        self.equipment_data.maintenance_hours += maintenance_duration / 3600.0
        
        maintenance_result = {
            'maintenance_type': event.conditions.get('maintenance_type', 'preventive'),
            'duration': maintenance_duration,
            'total_maintenance_hours': self.equipment_data.maintenance_hours,
            'timestamp': self.scheduler.current_time
        }
        
        self.logger.debug(f"Equipment {self.equipment_data.equipment_id}: Maintenance completed")
        return maintenance_result
    
    # Measurement simulation methods
    def _simulate_dmm_measurement(self, target: float, tolerance: float) -> float:
        """Simulate DMM measurement with realistic accuracy."""
        # DMM typically has high accuracy but some noise
        noise = random.gauss(0, tolerance * 0.1)
        return target + noise
    
    def _simulate_scope_measurement(self, target: float, tolerance: float) -> float:
        """Simulate oscilloscope measurement."""
        # Scope measurements might have more variation
        noise = random.gauss(0, tolerance * 0.2)
        return target + noise
    
    def _simulate_power_measurement(self, target: float, tolerance: float) -> float:
        """Simulate power supply measurement."""
        # Power supply measurements with load regulation effects
        regulation_error = random.uniform(-tolerance * 0.3, tolerance * 0.3)
        return target + regulation_error
    
    def _schedule_next_events(self, completed_event: DiscreteEvent, result: Any):
        """Schedule next events based on current state."""
        if self.current_state == EquipmentState.INITIALIZING.value:
            # After initialization, start equipment
            self.trigger_event(EquipmentEvent.CMD_START.value, delay=1.0)
        
        elif self.current_state == EquipmentState.MEASURING.value:
            # After measurement, stop and return to idle
            self.trigger_event(EquipmentEvent.CMD_STOP.value, delay=0.5)
        
        elif self.current_state == EquipmentState.CALIBRATING.value:
            # After calibration, stop and return to idle
            self.trigger_event(EquipmentEvent.CMD_STOP.value, delay=0.5)
        
        # Schedule periodic calibration (every 100 measurements)
        if (self.equipment_data.measurement_count > 0 and 
            self.equipment_data.measurement_count % 100 == 0 and
            self.current_state == EquipmentState.IDLE.value):
            self.trigger_event(EquipmentEvent.CMD_CALIBRATE.value, delay=10.0)
    
    def start_equipment(self):
        """Start the equipment from offline state."""
        if self.current_state == EquipmentState.OFFLINE.value:
            self.trigger_event(EquipmentEvent.CMD_INITIALIZE.value)
    
    def perform_measurement(self, measurement_type: str, target_value: float, tolerance: float, duration: float = None):
        """Perform a measurement."""
        if duration:
            self.state_data['measurement_duration'] = duration
            # Update the measurement event method duration
            self.event_methods[EquipmentEvent.CMD_MEASURE.value]['duration'] = duration
        
        if self.current_state == EquipmentState.IDLE.value:
            self.trigger_event(EquipmentEvent.CMD_START.value)
        
        self.trigger_event(EquipmentEvent.CMD_MEASURE.value, 
                          delay=2.5,  # Wait for start to complete
                          measurement_type=measurement_type,
                          target_value=target_value,
                          tolerance=tolerance)
    
    def force_calibration(self, calibration_type: str = 'manual'):
        """Force equipment calibration."""
        self.trigger_event(EquipmentEvent.CMD_CALIBRATE.value, 
                          calibration_type=calibration_type,
                          delay=0.1)
    
    def get_equipment_status(self) -> Dict[str, Any]:
        """Get comprehensive equipment status."""
        base_info = self.get_state_info()
        base_info.update({
            'equipment_id': self.equipment_data.equipment_id,
            'equipment_type': self.equipment_data.equipment_type,
            'station_id': self.equipment_data.station_id,
            'model': self.equipment_data.model,
            'serial_number': self.equipment_data.serial_number,
            'measurement_count': self.equipment_data.measurement_count,
            'calibration_count': self.equipment_data.calibration_count,
            'last_calibration': self.equipment_data.last_calibration,
            'error_count': self.equipment_data.error_count,
            'maintenance_hours': self.equipment_data.maintenance_hours,
            'current_measurement': self.equipment_data.current_measurement,
            'latest_measurements': self.equipment_data.measurement_history[-5:] if self.equipment_data.measurement_history else []
        })
        return base_info