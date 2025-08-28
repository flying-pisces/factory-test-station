"""Discrete Event-Based Deterministic Finite State Machine Framework."""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import heapq
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class DiscreteEvent:
    """Discrete event with fixed execution duration."""
    event_id: str
    event_name: str
    duration: float  # Time in seconds
    target_state: str
    entry_action: Optional[Callable] = None
    exit_action: Optional[Callable] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StateTransition:
    """State transition definition."""
    from_state: str
    to_state: str
    trigger_event: str
    condition: Optional[Callable] = None
    action: Optional[Callable] = None


@dataclass
class FSMState:
    """Finite State Machine state definition."""
    state_name: str
    entry_action: Optional[Callable] = None
    exit_action: Optional[Callable] = None
    is_final: bool = False


class DiscreteEventScheduler:
    """Central scheduler for all discrete events in the system."""
    
    def __init__(self):
        self.event_queue = []  # Priority queue (heapq)
        self.current_time = 0.0
        self.running = False
        self.fsm_registry: Dict[str, 'BaseFiniteStateMachine'] = {}
        self.logger = logging.getLogger('DiscreteEventScheduler')
        self._lock = threading.Lock()
        
    def register_fsm(self, fsm: 'BaseFiniteStateMachine'):
        """Register an FSM with the scheduler."""
        with self._lock:
            self.fsm_registry[fsm.fsm_id] = fsm
            self.logger.info(f"Registered FSM: {fsm.fsm_id}")
    
    def schedule_event(self, event: DiscreteEvent, execution_time: float):
        """Schedule a discrete event for execution."""
        with self._lock:
            heapq.heappush(self.event_queue, (execution_time, event))
            self.logger.debug(f"Scheduled event {event.event_name} at time {execution_time}")
    
    def advance_time(self, delta_time: float = None):
        """Advance simulation time and execute due events."""
        if delta_time is None:
            # Advance to next event
            if self.event_queue:
                next_time, _ = self.event_queue[0]
                delta_time = max(0, next_time - self.current_time)
            else:
                delta_time = 1.0  # Default advance
        
        target_time = self.current_time + delta_time
        executed_events = []
        
        with self._lock:
            while self.event_queue and self.event_queue[0][0] <= target_time:
                event_time, event = heapq.heappop(self.event_queue)
                self.current_time = event_time
                executed_events.append(event)
        
        # Execute events outside the lock
        for event in executed_events:
            self._execute_event(event)
        
        self.current_time = target_time
        return executed_events
    
    def _execute_event(self, event: DiscreteEvent):
        """Execute a discrete event."""
        try:
            self.logger.debug(f"Executing event: {event.event_name} at time {self.current_time}")
            
            # Find the FSM that should handle this event
            target_fsm = None
            for fsm in self.fsm_registry.values():
                if fsm.can_handle_event(event):
                    target_fsm = fsm
                    break
            
            if target_fsm:
                target_fsm.process_event(event)
            else:
                self.logger.warning(f"No FSM found to handle event: {event.event_name}")
                
        except Exception as e:
            self.logger.error(f"Error executing event {event.event_name}: {e}")
    
    def run_simulation(self, duration: float = None, max_events: int = None):
        """Run the discrete event simulation."""
        self.running = True
        events_processed = 0
        start_time = self.current_time
        
        self.logger.info(f"Starting discrete event simulation")
        
        while self.running and self.event_queue:
            if duration and (self.current_time - start_time) >= duration:
                break
            if max_events and events_processed >= max_events:
                break
            
            executed = self.advance_time()
            events_processed += len(executed)
        
        self.logger.info(f"Simulation complete. Processed {events_processed} events in {self.current_time - start_time:.2f}s")
        self.running = False
        
        return {
            'events_processed': events_processed,
            'simulation_time': self.current_time - start_time,
            'final_time': self.current_time
        }


class BaseFiniteStateMachine(ABC):
    """Base class for all discrete event-based finite state machines."""
    
    def __init__(self, fsm_id: str, scheduler: DiscreteEventScheduler):
        self.fsm_id = fsm_id
        self.scheduler = scheduler
        self.current_state: Optional[str] = None
        self.states: Dict[str, FSMState] = {}
        self.transitions: List[StateTransition] = []
        self.event_methods: Dict[str, Callable] = {}
        self.state_data: Dict[str, Any] = {}
        self.logger = logging.getLogger(f'FSM_{fsm_id}')
        
        # Register with scheduler
        self.scheduler.register_fsm(self)
        
        # Initialize FSM-specific states and transitions
        self._initialize_fsm()
        
        # Set initial state
        if self.states and not self.current_state:
            self.current_state = list(self.states.keys())[0]
    
    @abstractmethod
    def _initialize_fsm(self):
        """Initialize FSM states, transitions, and event methods."""
        pass
    
    def add_state(self, state: FSMState):
        """Add a state to the FSM."""
        self.states[state.state_name] = state
        self.logger.debug(f"Added state: {state.state_name}")
    
    def add_transition(self, transition: StateTransition):
        """Add a transition to the FSM."""
        self.transitions.append(transition)
        self.logger.debug(f"Added transition: {transition.from_state} -> {transition.to_state}")
    
    def add_event_method(self, event_name: str, method: Callable, duration: float):
        """Add a discrete event method with fixed duration."""
        self.event_methods[event_name] = {
            'method': method,
            'duration': duration
        }
        self.logger.debug(f"Added event method: {event_name} (duration: {duration}s)")
    
    def can_handle_event(self, event: DiscreteEvent) -> bool:
        """Check if this FSM can handle the given event."""
        return event.event_name in self.event_methods
    
    def process_event(self, event: DiscreteEvent):
        """Process a discrete event."""
        if not self.can_handle_event(event):
            self.logger.warning(f"Cannot handle event: {event.event_name}")
            return
        
        # Find valid transition
        valid_transition = None
        for transition in self.transitions:
            if (transition.from_state == self.current_state and 
                transition.trigger_event == event.event_name):
                if not transition.condition or transition.condition(self, event):
                    valid_transition = transition
                    break
        
        if not valid_transition:
            self.logger.warning(f"No valid transition for event {event.event_name} from state {self.current_state}")
            return
        
        # Execute state transition
        self._execute_transition(valid_transition, event)
    
    def _execute_transition(self, transition: StateTransition, event: DiscreteEvent):
        """Execute a state transition."""
        old_state = self.current_state
        
        # Exit current state
        if self.current_state and self.current_state in self.states:
            current_state_obj = self.states[self.current_state]
            if current_state_obj.exit_action:
                current_state_obj.exit_action(self, event)
        
        # Execute transition action
        if transition.action:
            transition.action(self, event)
        
        # Execute event method with duration
        event_info = self.event_methods[event.event_name]
        start_time = self.scheduler.current_time
        
        try:
            result = event_info['method'](self, event)
            self.logger.info(f"Executed {event.event_name}: {old_state} -> {transition.to_state} (duration: {event_info['duration']}s)")
        except Exception as e:
            self.logger.error(f"Error executing event method {event.event_name}: {e}")
            return
        
        # Change state
        self.current_state = transition.to_state
        
        # Enter new state
        if self.current_state in self.states:
            new_state_obj = self.states[self.current_state]
            if new_state_obj.entry_action:
                new_state_obj.entry_action(self, event)
        
        # Schedule next events if needed
        self._schedule_next_events(event, result)
    
    def _schedule_next_events(self, completed_event: DiscreteEvent, result: Any):
        """Schedule next events based on current state and completed event result."""
        # Override in subclasses to implement specific event scheduling logic
        pass
    
    def trigger_event(self, event_name: str, delay: float = 0.0, **kwargs):
        """Trigger a new discrete event."""
        event = DiscreteEvent(
            event_id=str(uuid.uuid4()),
            event_name=event_name,
            duration=self.event_methods.get(event_name, {}).get('duration', 0.0),
            target_state=self.current_state,
            conditions=kwargs
        )
        
        execution_time = self.scheduler.current_time + delay
        self.scheduler.schedule_event(event, execution_time)
        
        return event
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current FSM state information."""
        return {
            'fsm_id': self.fsm_id,
            'current_state': self.current_state,
            'state_data': self.state_data.copy(),
            'available_events': list(self.event_methods.keys()),
            'scheduler_time': self.scheduler.current_time
        }


# Create global scheduler instance
global_scheduler = DiscreteEventScheduler()


def create_discrete_event(event_name: str, duration: float, target_state: str = None, **kwargs) -> DiscreteEvent:
    """Helper function to create discrete events."""
    return DiscreteEvent(
        event_id=str(uuid.uuid4()),
        event_name=event_name,
        duration=duration,
        target_state=target_state,
        conditions=kwargs
    )