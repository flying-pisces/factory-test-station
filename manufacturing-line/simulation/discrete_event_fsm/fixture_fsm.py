"""Manufacturing Fixture Finite State Machine."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time

from base_fsm import BaseFiniteStateMachine, FSMState, StateTransition, DiscreteEvent, global_scheduler


class FixtureState(Enum):
    """Fixture states in the manufacturing process."""
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    CLAMPED = "clamped"
    TESTING = "testing"
    UNLOADING = "unloading"
    CLEANING = "cleaning"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class FixtureEvent(Enum):
    """Fixture discrete events with fixed durations."""
    CMD_LOAD = "cmd_load"           # 3.0s - Load DUT into fixture
    CMD_UNLOAD = "cmd_unload"       # 2.5s - Unload DUT from fixture
    CMD_CLAMP = "cmd_clamp"         # 1.0s - Clamp DUT in position
    CMD_UNCLAMP = "cmd_unclamp"     # 0.8s - Release DUT clamp
    CMD_TEST = "cmd_test"           # Variable - Execute test sequence
    CMD_CLEAN = "cmd_clean"         # 5.0s - Clean fixture
    CMD_RESET = "cmd_reset"         # 2.0s - Reset to idle state
    CMD_MAINTENANCE = "cmd_maintenance"  # 30.0s - Maintenance cycle


@dataclass
class FixtureData:
    """Fixture-specific data structure."""
    fixture_id: str
    fixture_type: str
    station_id: str
    capacity: int = 1  # Number of DUTs it can hold
    current_duts: List[str] = None
    cycle_count: int = 0
    maintenance_cycles: int = 0
    last_maintenance: float = 0.0
    error_state: Optional[str] = None
    
    def __post_init__(self):
        if self.current_duts is None:
            self.current_duts = []


class FixtureFSM(BaseFiniteStateMachine):
    """Fixture Finite State Machine with discrete event processing."""
    
    def __init__(self, fixture_data: FixtureData, scheduler=None):
        if scheduler is None:
            scheduler = global_scheduler
        
        super().__init__(f"FIXTURE_{fixture_data.fixture_id}", scheduler)
        self.fixture_data = fixture_data
        self.state_data['fixture_data'] = fixture_data
        
    def _initialize_fsm(self):
        """Initialize Fixture FSM states, transitions, and event methods."""
        
        # Define states
        self.add_state(FSMState(
            state_name=FixtureState.IDLE.value,
            entry_action=self._on_enter_idle
        ))
        
        self.add_state(FSMState(
            state_name=FixtureState.LOADING.value,
            entry_action=self._on_enter_loading
        ))
        
        self.add_state(FSMState(
            state_name=FixtureState.LOADED.value,
            entry_action=self._on_enter_loaded
        ))
        
        self.add_state(FSMState(
            state_name=FixtureState.CLAMPED.value,
            entry_action=self._on_enter_clamped
        ))
        
        self.add_state(FSMState(
            state_name=FixtureState.TESTING.value,
            entry_action=self._on_enter_testing
        ))
        
        self.add_state(FSMState(
            state_name=FixtureState.UNLOADING.value,
            entry_action=self._on_enter_unloading
        ))
        
        self.add_state(FSMState(
            state_name=FixtureState.CLEANING.value,
            entry_action=self._on_enter_cleaning
        ))
        
        self.add_state(FSMState(
            state_name=FixtureState.ERROR.value,
            entry_action=self._on_enter_error
        ))
        
        self.add_state(FSMState(
            state_name=FixtureState.MAINTENANCE.value,
            entry_action=self._on_enter_maintenance
        ))
        
        # Define transitions
        self.add_transition(StateTransition(
            from_state=FixtureState.IDLE.value,
            to_state=FixtureState.LOADING.value,
            trigger_event=FixtureEvent.CMD_LOAD.value,
            condition=self._can_load_condition
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.LOADING.value,
            to_state=FixtureState.LOADED.value,
            trigger_event=FixtureEvent.CMD_LOAD.value
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.LOADED.value,
            to_state=FixtureState.CLAMPED.value,
            trigger_event=FixtureEvent.CMD_CLAMP.value
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.CLAMPED.value,
            to_state=FixtureState.TESTING.value,
            trigger_event=FixtureEvent.CMD_TEST.value
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.TESTING.value,
            to_state=FixtureState.LOADED.value,
            trigger_event=FixtureEvent.CMD_UNCLAMP.value
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.LOADED.value,
            to_state=FixtureState.UNLOADING.value,
            trigger_event=FixtureEvent.CMD_UNLOAD.value
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.UNLOADING.value,
            to_state=FixtureState.CLEANING.value,
            trigger_event=FixtureEvent.CMD_CLEAN.value
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.CLEANING.value,
            to_state=FixtureState.IDLE.value,
            trigger_event=FixtureEvent.CMD_RESET.value
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.ERROR.value,
            to_state=FixtureState.MAINTENANCE.value,
            trigger_event=FixtureEvent.CMD_MAINTENANCE.value
        ))
        
        self.add_transition(StateTransition(
            from_state=FixtureState.MAINTENANCE.value,
            to_state=FixtureState.IDLE.value,
            trigger_event=FixtureEvent.CMD_RESET.value
        ))
        
        # Add discrete event methods with fixed durations
        self.add_event_method(FixtureEvent.CMD_LOAD.value, self._cmd_load, 3.0)
        self.add_event_method(FixtureEvent.CMD_UNLOAD.value, self._cmd_unload, 2.5)
        self.add_event_method(FixtureEvent.CMD_CLAMP.value, self._cmd_clamp, 1.0)
        self.add_event_method(FixtureEvent.CMD_UNCLAMP.value, self._cmd_unclamp, 0.8)
        self.add_event_method(FixtureEvent.CMD_TEST.value, self._cmd_test, 15.0)  # Default test duration
        self.add_event_method(FixtureEvent.CMD_CLEAN.value, self._cmd_clean, 5.0)
        self.add_event_method(FixtureEvent.CMD_RESET.value, self._cmd_reset, 2.0)
        self.add_event_method(FixtureEvent.CMD_MAINTENANCE.value, self._cmd_maintenance, 30.0)
        
        # Set initial state
        self.current_state = FixtureState.IDLE.value
    
    # State entry actions
    def _on_enter_idle(self, fsm, event):
        """Actions when entering IDLE state."""
        self.logger.info(f"Fixture {self.fixture_data.fixture_id} idle and ready")
    
    def _on_enter_loading(self, fsm, event):
        """Actions when entering LOADING state."""
        self.logger.info(f"Fixture {self.fixture_data.fixture_id} loading DUT")
    
    def _on_enter_loaded(self, fsm, event):
        """Actions when entering LOADED state."""
        self.logger.info(f"Fixture {self.fixture_data.fixture_id} loaded with DUT")
    
    def _on_enter_clamped(self, fsm, event):
        """Actions when entering CLAMPED state."""
        self.logger.info(f"Fixture {self.fixture_data.fixture_id} DUT clamped and secured")
    
    def _on_enter_testing(self, fsm, event):
        """Actions when entering TESTING state."""
        self.logger.info(f"Fixture {self.fixture_data.fixture_id} executing test sequence")
    
    def _on_enter_unloading(self, fsm, event):
        """Actions when entering UNLOADING state."""
        self.logger.info(f"Fixture {self.fixture_data.fixture_id} unloading DUT")
    
    def _on_enter_cleaning(self, fsm, event):
        """Actions when entering CLEANING state."""
        self.logger.info(f"Fixture {self.fixture_data.fixture_id} cleaning cycle")
    
    def _on_enter_error(self, fsm, event):
        """Actions when entering ERROR state."""
        self.logger.error(f"Fixture {self.fixture_data.fixture_id} in error state")
    
    def _on_enter_maintenance(self, fsm, event):
        """Actions when entering MAINTENANCE state."""
        self.logger.info(f"Fixture {self.fixture_data.fixture_id} undergoing maintenance")
    
    # Transition conditions
    def _can_load_condition(self, fsm, event):
        """Condition for loading DUT."""
        return len(self.fixture_data.current_duts) < self.fixture_data.capacity
    
    # Discrete event methods (fixed duration)
    def _cmd_load(self, fsm, event: DiscreteEvent):
        """CMD_LOAD event - 3.0s duration."""
        dut_id = event.conditions.get('dut_id', f'DUT_{self.scheduler.current_time}')
        
        if self.current_state == FixtureState.IDLE.value:
            # Start loading process
            self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: Starting load of {dut_id}")
            return {'loading_started': True, 'dut_id': dut_id}
        else:
            # Complete loading process
            self.fixture_data.current_duts.append(dut_id)
            self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: Loaded {dut_id}")
            return {'loaded': True, 'dut_id': dut_id}
    
    def _cmd_unload(self, event: DiscreteEvent):
        """CMD_UNLOAD event - 2.5s duration."""
        if self.fixture_data.current_duts:
            dut_id = self.fixture_data.current_duts.pop(0)
            self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: Unloaded {dut_id}")
            return {'unloaded': True, 'dut_id': dut_id}
        return {'unloaded': False, 'error': 'No DUT to unload'}
    
    def _cmd_clamp(self, event: DiscreteEvent):
        """CMD_CLAMP event - 1.0s duration."""
        self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: DUT clamped")
        return {'clamped': True, 'timestamp': self.scheduler.current_time}
    
    def _cmd_unclamp(self, event: DiscreteEvent):
        """CMD_UNCLAMP event - 0.8s duration."""
        self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: DUT unclamped")
        return {'unclamped': True, 'timestamp': self.scheduler.current_time}
    
    def _cmd_test(self, event: DiscreteEvent):
        """CMD_TEST event - Variable duration."""
        test_type = event.conditions.get('test_type', 'default')
        
        # Simulate test execution
        import random
        test_result = {
            'test_type': test_type,
            'result': 'PASS' if random.random() < 0.85 else 'FAIL',
            'timestamp': self.scheduler.current_time,
            'duration': self.state_data['test_duration']
        }
        
        self.fixture_data.cycle_count += 1
        self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: Test completed - {test_result['result']}")
        
        return test_result
    
    def _cmd_clean(self, event: DiscreteEvent):
        """CMD_CLEAN event - 5.0s duration."""
        self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: Cleaning completed")
        return {'cleaned': True, 'timestamp': self.scheduler.current_time}
    
    def _cmd_reset(self, event: DiscreteEvent):
        """CMD_RESET event - 2.0s duration."""
        self.fixture_data.error_state = None
        self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: Reset completed")
        return {'reset': True, 'timestamp': self.scheduler.current_time}
    
    def _cmd_maintenance(self, event: DiscreteEvent):
        """CMD_MAINTENANCE event - 30.0s duration."""
        self.fixture_data.maintenance_cycles += 1
        self.fixture_data.last_maintenance = self.scheduler.current_time
        self.logger.debug(f"Fixture {self.fixture_data.fixture_id}: Maintenance completed")
        return {'maintenance_completed': True, 'cycle': self.fixture_data.maintenance_cycles}
    
    def _schedule_next_events(self, completed_event: DiscreteEvent, result: Any):
        """Schedule next events based on current state."""
        if self.current_state == FixtureState.LOADED.value and completed_event.event_name == FixtureEvent.CMD_LOAD.value:
            # After loading, automatically clamp
            self.trigger_event(FixtureEvent.CMD_CLAMP.value, delay=0.5)
        
        elif self.current_state == FixtureState.CLAMPED.value:
            # After clamping, start test
            self.trigger_event(FixtureEvent.CMD_TEST.value, delay=1.0, test_type='functional')
        
        elif self.current_state == FixtureState.TESTING.value:
            # After testing, unclamp
            self.trigger_event(FixtureEvent.CMD_UNCLAMP.value, delay=0.5)
        
        elif self.current_state == FixtureState.LOADED.value and completed_event.event_name == FixtureEvent.CMD_UNCLAMP.value:
            # After unclamping, unload
            self.trigger_event(FixtureEvent.CMD_UNLOAD.value, delay=1.0)
        
        elif self.current_state == FixtureState.UNLOADING.value:
            # After unloading, clean
            self.trigger_event(FixtureEvent.CMD_CLEAN.value, delay=0.5)
        
        elif self.current_state == FixtureState.CLEANING.value:
            # After cleaning, reset to idle
            self.trigger_event(FixtureEvent.CMD_RESET.value, delay=0.2)
    
    def start_dut_processing(self, dut_id: str, test_duration: float = None):
        """Start processing a DUT through the fixture."""
        if test_duration:
            self.state_data['test_duration'] = test_duration
            # Update the test event method duration
            self.event_methods[FixtureEvent.CMD_TEST.value]['duration'] = test_duration
        
        self.trigger_event(FixtureEvent.CMD_LOAD.value, dut_id=dut_id)
    
    def force_maintenance(self):
        """Force fixture into maintenance mode."""
        self.current_state = FixtureState.ERROR.value
        self.trigger_event(FixtureEvent.CMD_MAINTENANCE.value, delay=0.1)
    
    def get_fixture_status(self) -> Dict[str, Any]:
        """Get comprehensive fixture status."""
        base_info = self.get_state_info()
        base_info.update({
            'fixture_id': self.fixture_data.fixture_id,
            'fixture_type': self.fixture_data.fixture_type,
            'station_id': self.fixture_data.station_id,
            'capacity': self.fixture_data.capacity,
            'current_duts': self.fixture_data.current_duts.copy(),
            'cycle_count': self.fixture_data.cycle_count,
            'maintenance_cycles': self.fixture_data.maintenance_cycles,
            'last_maintenance': self.fixture_data.last_maintenance,
            'error_state': self.fixture_data.error_state,
            'utilization': len(self.fixture_data.current_duts) / self.fixture_data.capacity
        })
        return base_info