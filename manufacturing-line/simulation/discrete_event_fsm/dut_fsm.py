"""Device Under Test (DUT) Finite State Machine."""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time

from base_fsm import BaseFiniteStateMachine, FSMState, StateTransition, DiscreteEvent, global_scheduler


class DUTState(Enum):
    """DUT states in the manufacturing process."""
    CREATED = "created"
    LOADED = "loaded"
    IN_TRANSIT = "in_transit"
    AT_STATION = "at_station"
    PROCESSING = "processing"
    TESTED = "tested"
    PASSED = "passed"
    FAILED = "failed"
    UNLOADED = "unloaded"
    COMPLETED = "completed"
    SCRAPPED = "scrapped"


class DUTEvent(Enum):
    """DUT discrete events with fixed durations."""
    CMD_SIGNAL = "cmd_signal"           # 0.1s - Signal generation
    HANDLE_IN = "handle_in"             # 2.0s - Load into fixture
    HANDLE_OUT = "handle_out"           # 1.5s - Unload from fixture
    MOVE_NEXT = "move_next"             # 5.0s - Move to next station


@dataclass
class DUTData:
    """DUT-specific data structure."""
    serial_number: str
    batch_id: str
    creation_time: float
    material_cost: float = 10.0
    process_cost: float = 0.0
    test_results: Dict[str, Any] = None
    defects: list = None
    station_history: list = None
    
    def __post_init__(self):
        if self.test_results is None:
            self.test_results = {}
        if self.defects is None:
            self.defects = []
        if self.station_history is None:
            self.station_history = []


class DUTFSM(BaseFiniteStateMachine):
    """DUT Finite State Machine with discrete event processing."""
    
    def __init__(self, dut_data: DUTData, scheduler=None):
        if scheduler is None:
            scheduler = global_scheduler
        
        super().__init__(f"DUT_{dut_data.serial_number}", scheduler)
        self.dut_data = dut_data
        self.state_data['dut_data'] = dut_data
        
    def _initialize_fsm(self):
        """Initialize DUT FSM states, transitions, and event methods."""
        
        # Define states
        self.add_state(FSMState(
            state_name=DUTState.CREATED.value,
            entry_action=self._on_enter_created
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.LOADED.value,
            entry_action=self._on_enter_loaded
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.IN_TRANSIT.value,
            entry_action=self._on_enter_in_transit
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.AT_STATION.value,
            entry_action=self._on_enter_at_station
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.PROCESSING.value,
            entry_action=self._on_enter_processing
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.TESTED.value,
            entry_action=self._on_enter_tested
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.PASSED.value,
            entry_action=self._on_enter_passed
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.FAILED.value,
            entry_action=self._on_enter_failed
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.COMPLETED.value,
            entry_action=self._on_enter_completed,
            is_final=True
        ))
        
        self.add_state(FSMState(
            state_name=DUTState.SCRAPPED.value,
            entry_action=self._on_enter_scrapped,
            is_final=True
        ))
        
        # Define transitions
        self.add_transition(StateTransition(
            from_state=DUTState.CREATED.value,
            to_state=DUTState.LOADED.value,
            trigger_event=DUTEvent.HANDLE_IN.value
        ))
        
        self.add_transition(StateTransition(
            from_state=DUTState.LOADED.value,
            to_state=DUTState.IN_TRANSIT.value,
            trigger_event=DUTEvent.MOVE_NEXT.value
        ))
        
        self.add_transition(StateTransition(
            from_state=DUTState.IN_TRANSIT.value,
            to_state=DUTState.AT_STATION.value,
            trigger_event=DUTEvent.CMD_SIGNAL.value
        ))
        
        self.add_transition(StateTransition(
            from_state=DUTState.AT_STATION.value,
            to_state=DUTState.PROCESSING.value,
            trigger_event=DUTEvent.CMD_SIGNAL.value
        ))
        
        self.add_transition(StateTransition(
            from_state=DUTState.PROCESSING.value,
            to_state=DUTState.TESTED.value,
            trigger_event=DUTEvent.CMD_SIGNAL.value
        ))
        
        self.add_transition(StateTransition(
            from_state=DUTState.TESTED.value,
            to_state=DUTState.PASSED.value,
            trigger_event=DUTEvent.CMD_SIGNAL.value,
            condition=self._test_passed_condition
        ))
        
        self.add_transition(StateTransition(
            from_state=DUTState.TESTED.value,
            to_state=DUTState.FAILED.value,
            trigger_event=DUTEvent.CMD_SIGNAL.value,
            condition=self._test_failed_condition
        ))
        
        self.add_transition(StateTransition(
            from_state=DUTState.PASSED.value,
            to_state=DUTState.COMPLETED.value,
            trigger_event=DUTEvent.HANDLE_OUT.value
        ))
        
        self.add_transition(StateTransition(
            from_state=DUTState.FAILED.value,
            to_state=DUTState.SCRAPPED.value,
            trigger_event=DUTEvent.HANDLE_OUT.value
        ))
        
        # Add discrete event methods with fixed durations
        self.add_event_method(DUTEvent.CMD_SIGNAL.value, self._cmd_signal, 0.1)
        self.add_event_method(DUTEvent.HANDLE_IN.value, self._handle_in, 2.0)
        self.add_event_method(DUTEvent.HANDLE_OUT.value, self._handle_out, 1.5)
        self.add_event_method(DUTEvent.MOVE_NEXT.value, self._move_next, 5.0)
        
        # Set initial state
        self.current_state = DUTState.CREATED.value
    
    # State entry actions
    def _on_enter_created(self, fsm, event):
        """Actions when entering CREATED state."""
        self.dut_data.creation_time = self.scheduler.current_time
        self.logger.info(f"DUT {self.dut_data.serial_number} created at time {self.dut_data.creation_time}")
    
    def _on_enter_loaded(self, fsm, event):
        """Actions when entering LOADED state."""
        self.dut_data.station_history.append({
            'action': 'loaded',
            'timestamp': self.scheduler.current_time,
            'duration': event.duration
        })
        self.logger.info(f"DUT {self.dut_data.serial_number} loaded")
    
    def _on_enter_in_transit(self, fsm, event):
        """Actions when entering IN_TRANSIT state."""
        self.logger.info(f"DUT {self.dut_data.serial_number} in transit")
    
    def _on_enter_at_station(self, fsm, event):
        """Actions when entering AT_STATION state."""
        station_id = event.conditions.get('station_id', 'unknown')
        self.state_data['current_station'] = station_id
        self.logger.info(f"DUT {self.dut_data.serial_number} at station {station_id}")
    
    def _on_enter_processing(self, fsm, event):
        """Actions when entering PROCESSING state."""
        self.logger.info(f"DUT {self.dut_data.serial_number} processing started")
    
    def _on_enter_tested(self, fsm, event):
        """Actions when entering TESTED state."""
        self.logger.info(f"DUT {self.dut_data.serial_number} testing completed")
    
    def _on_enter_passed(self, fsm, event):
        """Actions when entering PASSED state."""
        self.logger.info(f"DUT {self.dut_data.serial_number} PASSED test")
    
    def _on_enter_failed(self, fsm, event):
        """Actions when entering FAILED state."""
        self.logger.warning(f"DUT {self.dut_data.serial_number} FAILED test")
    
    def _on_enter_completed(self, fsm, event):
        """Actions when entering COMPLETED state."""
        completion_time = self.scheduler.current_time
        cycle_time = completion_time - self.dut_data.creation_time
        self.state_data['completion_time'] = completion_time
        self.state_data['cycle_time'] = cycle_time
        self.logger.info(f"DUT {self.dut_data.serial_number} COMPLETED (cycle time: {cycle_time:.2f}s)")
    
    def _on_enter_scrapped(self, fsm, event):
        """Actions when entering SCRAPPED state."""
        self.logger.warning(f"DUT {self.dut_data.serial_number} SCRAPPED")
    
    # Transition conditions
    def _test_passed_condition(self, fsm, event):
        """Condition for test passing."""
        # Simulate 85% pass rate
        import random
        passed = random.random() < 0.85
        self.dut_data.test_results['last_test'] = {
            'result': 'PASS' if passed else 'FAIL',
            'timestamp': self.scheduler.current_time
        }
        return passed
    
    def _test_failed_condition(self, fsm, event):
        """Condition for test failure."""
        return not self._test_passed_condition(fsm, event)
    
    # Discrete event methods (fixed duration)
    def _cmd_signal(self, fsm, event: DiscreteEvent):
        """CMD_SIGNAL event - 0.1s duration."""
        self.logger.debug(f"DUT {self.dut_data.serial_number}: CMD_SIGNAL processed")
        return {'signal_sent': True, 'timestamp': self.scheduler.current_time}
    
    def _handle_in(self, fsm, event: DiscreteEvent):
        """HANDLE_IN event - 2.0s duration."""
        self.logger.debug(f"DUT {self.dut_data.serial_number}: HANDLE_IN processed")
        fixture_id = event.conditions.get('fixture_id', 'unknown')
        return {'loaded_to_fixture': fixture_id, 'timestamp': self.scheduler.current_time}
    
    def _handle_out(self, fsm, event: DiscreteEvent):
        """HANDLE_OUT event - 1.5s duration."""
        self.logger.debug(f"DUT {self.dut_data.serial_number}: HANDLE_OUT processed")
        return {'unloaded': True, 'timestamp': self.scheduler.current_time}
    
    def _move_next(self, fsm, event: DiscreteEvent):
        """MOVE_NEXT event - 5.0s duration."""
        self.logger.debug(f"DUT {self.dut_data.serial_number}: MOVE_NEXT processed")
        next_station = event.conditions.get('next_station', 'unknown')
        return {'moved_to': next_station, 'timestamp': self.scheduler.current_time}
    
    def _schedule_next_events(self, completed_event: DiscreteEvent, result: Any):
        """Schedule next events based on current state."""
        if self.current_state == DUTState.LOADED.value:
            # Schedule movement to next station
            self.trigger_event(DUTEvent.MOVE_NEXT.value, delay=0.5)
        
        elif self.current_state == DUTState.AT_STATION.value:
            # Schedule processing start
            self.trigger_event(DUTEvent.CMD_SIGNAL.value, delay=1.0)
        
        elif self.current_state == DUTState.PROCESSING.value:
            # Schedule test completion
            self.trigger_event(DUTEvent.CMD_SIGNAL.value, delay=10.0)  # 10s processing time
        
        elif self.current_state == DUTState.TESTED.value:
            # Schedule result evaluation
            self.trigger_event(DUTEvent.CMD_SIGNAL.value, delay=0.2)
        
        elif self.current_state in [DUTState.PASSED.value, DUTState.FAILED.value]:
            # Schedule unloading
            self.trigger_event(DUTEvent.HANDLE_OUT.value, delay=1.0)
    
    def start_manufacturing_process(self, fixture_id: str = None):
        """Start the DUT manufacturing process."""
        self.trigger_event(DUTEvent.HANDLE_IN.value, fixture_id=fixture_id)
    
    def get_dut_status(self) -> Dict[str, Any]:
        """Get comprehensive DUT status."""
        base_info = self.get_state_info()
        base_info.update({
            'serial_number': self.dut_data.serial_number,
            'batch_id': self.dut_data.batch_id,
            'material_cost': self.dut_data.material_cost,
            'process_cost': self.dut_data.process_cost,
            'test_results': self.dut_data.test_results,
            'defects': self.dut_data.defects,
            'station_history': self.dut_data.station_history
        })
        return base_info