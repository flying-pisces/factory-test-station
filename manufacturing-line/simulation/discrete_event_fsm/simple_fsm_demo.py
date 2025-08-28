#!/usr/bin/env python3
"""Simple FSM Demonstration - Discrete Event-Based Deterministic Finite State Machine."""

import logging
import time

from base_fsm import (
    DiscreteEventScheduler, BaseFiniteStateMachine, FSMState, StateTransition, 
    DiscreteEvent, global_scheduler, create_discrete_event
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SimpleFSMDemo')


class SimpleDoorFSM(BaseFiniteStateMachine):
    """Simple door FSM to demonstrate discrete event-based state machine."""
    
    def _initialize_fsm(self):
        """Initialize door FSM states, transitions, and event methods."""
        
        # Define states (matching your diagram)
        self.add_state(FSMState(
            state_name="opened",
            entry_action=self._on_enter_opened
        ))
        
        self.add_state(FSMState(
            state_name="closed", 
            entry_action=self._on_enter_closed
        ))
        
        # Define transitions (matching your diagram)
        self.add_transition(StateTransition(
            from_state="opened",
            to_state="closed",
            trigger_event="close"
        ))
        
        self.add_transition(StateTransition(
            from_state="closed",
            to_state="opened", 
            trigger_event="open"
        ))
        
        # Add discrete event methods with fixed durations
        self.add_event_method("open", self._open_door, 2.0)   # 2 seconds to open
        self.add_event_method("close", self._close_door, 1.5) # 1.5 seconds to close
        
        # Set initial state
        self.current_state = "closed"
    
    # State entry actions
    def _on_enter_opened(self, fsm, event):
        """Actions when entering OPENED state."""
        self.logger.info(f"Door {self.fsm_id} is now OPEN")
    
    def _on_enter_closed(self, fsm, event):
        """Actions when entering CLOSED state."""
        self.logger.info(f"Door {self.fsm_id} is now CLOSED")
    
    # Discrete event methods (fixed duration)
    def _open_door(self, fsm, event: DiscreteEvent):
        """OPEN event - 2.0s duration."""
        self.logger.info(f"Door {self.fsm_id}: Opening door (2.0s duration)")
        return {'action': 'opened', 'timestamp': self.scheduler.current_time}
    
    def _close_door(self, fsm, event: DiscreteEvent):
        """CLOSE event - 1.5s duration."""  
        self.logger.info(f"Door {self.fsm_id}: Closing door (1.5s duration)")
        return {'action': 'closed', 'timestamp': self.scheduler.current_time}


def demonstrate_basic_fsm():
    """Demonstrate basic FSM with discrete events."""
    print("\n" + "="*80)
    print("🚪 BASIC DISCRETE EVENT-BASED FSM DEMONSTRATION")
    print("="*80)
    
    # Reset scheduler
    global_scheduler.event_queue.clear()
    global_scheduler.current_time = 0.0
    global_scheduler.fsm_registry.clear()
    
    print("\n🔧 Step 1: Creating Door FSM")
    print("-" * 30)
    
    # Create door FSM
    door = SimpleDoorFSM("DOOR_001", global_scheduler)
    
    print(f"Door Created: {door.fsm_id}")
    print(f"Initial State: {door.current_state}")
    print(f"Available Events: {list(door.event_methods.keys())}")
    
    # Show event durations (matching your requirements)
    print("\nEvent Methods with Fixed Durations:")
    for event_name, event_info in door.event_methods.items():
        print(f"  {event_name}: {event_info['duration']:.1f}s duration")
    
    print("\n⚡ Step 2: Triggering Discrete Events")
    print("-" * 40)
    
    # Trigger events with specific timing
    print(f"Current time: {global_scheduler.current_time:.1f}s")
    
    # Schedule open event
    door.trigger_event("open", delay=1.0)
    print(f"Scheduled OPEN event at time: {global_scheduler.current_time + 1.0:.1f}s")
    
    # Schedule close event  
    door.trigger_event("close", delay=5.0)
    print(f"Scheduled CLOSE event at time: {global_scheduler.current_time + 5.0:.1f}s")
    
    print(f"Events in queue: {len(global_scheduler.event_queue)}")
    
    print("\n🕐 Step 3: Executing Discrete Events")
    print("-" * 40)
    
    # Execute events step by step
    for step in range(8):  # 8 time steps
        old_time = global_scheduler.current_time
        events = global_scheduler.advance_time(1.0)  # Advance 1 second
        new_time = global_scheduler.current_time
        
        print(f"\nTime {old_time:.1f}s -> {new_time:.1f}s:")
        
        if events:
            for event in events:
                print(f"  ✅ Executed: {event.event_name} (duration: {event.duration:.1f}s)")
        else:
            print(f"  ⏱️ No events executed")
        
        print(f"  🚪 Door State: {door.current_state}")
        
        if not global_scheduler.event_queue:
            print("  📭 No more events in queue")
            break
    
    print(f"\n📊 Step 4: Final Results")
    print("-" * 25)
    print(f"Final Door State: {door.current_state}")
    print(f"Total Simulation Time: {global_scheduler.current_time:.1f}s")
    
    return door


def demonstrate_concurrent_fsms():
    """Demonstrate multiple FSMs operating concurrently."""
    print("\n" + "="*80)  
    print("🔀 CONCURRENT FSM OPERATION DEMONSTRATION")
    print("="*80)
    
    # Reset scheduler
    global_scheduler.event_queue.clear()
    global_scheduler.current_time = 0.0
    global_scheduler.fsm_registry.clear()
    
    print("\n👥 Step 1: Creating Multiple Door FSMs")
    print("-" * 40)
    
    # Create multiple door FSMs
    doors = {}
    for i in range(3):
        door_id = f"DOOR_{i+1:03d}"
        door = SimpleDoorFSM(door_id, global_scheduler)
        doors[door_id] = door
        print(f"Created {door_id} (initial state: {door.current_state})")
    
    print(f"\nTotal FSMs: {len(doors)}")
    print(f"Registered in scheduler: {len(global_scheduler.fsm_registry)}")
    
    print("\n⚡ Step 2: Scheduling Concurrent Events")
    print("-" * 45)
    
    # Schedule events for different doors at different times
    doors["DOOR_001"].trigger_event("open", delay=1.0)
    print("DOOR_001: Scheduled OPEN at 1.0s")
    
    doors["DOOR_002"].trigger_event("open", delay=2.5) 
    print("DOOR_002: Scheduled OPEN at 2.5s")
    
    doors["DOOR_003"].trigger_event("open", delay=4.0)
    print("DOOR_003: Scheduled OPEN at 4.0s")
    
    # Schedule some close events
    doors["DOOR_001"].trigger_event("close", delay=6.0)
    print("DOOR_001: Scheduled CLOSE at 6.0s")
    
    doors["DOOR_002"].trigger_event("close", delay=7.5)
    print("DOOR_002: Scheduled CLOSE at 7.5s")
    
    print(f"\nTotal events scheduled: {len(global_scheduler.event_queue)}")
    
    print("\n🕐 Step 3: Concurrent Event Execution")
    print("-" * 40)
    
    # Execute all events
    for step in range(12):  # 12 time steps
        old_time = global_scheduler.current_time
        events = global_scheduler.advance_time(1.0)  # 1 second steps
        new_time = global_scheduler.current_time
        
        print(f"\nTime {old_time:.1f}s -> {new_time:.1f}s:")
        
        if events:
            for event in events:
                # Find which door handled this event
                door_id = "Unknown"
                for did, door in doors.items():
                    if door.can_handle_event(event):
                        door_id = did
                        break
                
                print(f"  ✅ {door_id}: {event.event_name} (duration: {event.duration:.1f}s)")
        
        # Show all door states
        states = [f"{did}:{door.current_state}" for did, door in doors.items()]
        print(f"  🚪 States: {', '.join(states)}")
        
        if not global_scheduler.event_queue:
            print("  📭 No more events in queue")
            break
    
    print(f"\n📊 Step 4: Final Concurrent States")
    print("-" * 40)
    
    for door_id, door in doors.items():
        print(f"{door_id}: {door.current_state}")
    
    print(f"\nTotal Simulation Time: {global_scheduler.current_time:.1f}s")
    
    return doors


if __name__ == "__main__":
    print("🔧 DISCRETE EVENT-BASED FINITE STATE MACHINE DEMONSTRATION")
    print("🎯 Demonstrating: Fixed Duration Events, State Transitions, Concurrent Execution")
    print("\n📋 Based on your diagram requirements:")
    print("  • Each object has fixed discrete event methods")
    print("  • Methods execute with specific durations (time-based)")
    print("  • State transitions are deterministic")
    print("  • Events are scheduled and executed chronologically")
    
    # Run basic FSM demo
    basic_door = demonstrate_basic_fsm()
    
    # Run concurrent FSM demo  
    concurrent_doors = demonstrate_concurrent_fsms()
    
    print("\n" + "="*80)
    print("🎉 DISCRETE EVENT FSM DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\n✅ Successfully demonstrated:")
    print("  • Discrete Event-Based Deterministic Finite State Machines")
    print("  • Fixed duration event methods (open: 2.0s, close: 1.5s)")
    print("  • State transitions with entry actions")
    print("  • Event scheduling with precise timing")
    print("  • Concurrent FSM operation")
    print("  • Central event scheduler coordination")
    
    print(f"\n📈 Key Architecture Features:")
    print(f"  ✓ Each object has fixed discrete event methods")
    print(f"  ✓ Methods execute with specific durations (time-based)")
    print(f"  ✓ State machines manage object behavior deterministically")
    print(f"  ✓ Events processed in chronological order")
    print(f"  ✓ Supports concurrent object operations")
    
    print(f"\n🔄 This architecture forms the backbone of the manufacturing system!")
    print(f"   Ready for integration with DUT, Fixture, and Equipment objects.")