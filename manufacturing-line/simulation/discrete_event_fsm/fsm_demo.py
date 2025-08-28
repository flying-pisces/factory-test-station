#!/usr/bin/env python3
"""Discrete Event-Based FSM Manufacturing Line Demonstration."""

import logging
import time
from pathlib import Path

from fsm_simulation_engine import (
    FSMSimulationEngine, StationConfiguration, create_sample_manufacturing_line
)
from dut_fsm import DUTData, DUTEvent
from fixture_fsm import FixtureData, FixtureEvent
from equipment_fsm import EquipmentData, EquipmentEvent
from base_fsm import global_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FSMDemo')


def demonstrate_individual_fsm_components():
    """Demonstrate individual FSM components with discrete events."""
    print("\n" + "="*80)
    print("🔧 DISCRETE EVENT-BASED FSM COMPONENTS DEMONSTRATION")
    print("="*80)
    
    # Reset scheduler
    global_scheduler.event_queue.clear()
    global_scheduler.current_time = 0.0
    global_scheduler.fsm_registry.clear()
    
    print("\n📱 Step 1: DUT Finite State Machine")
    print("-" * 40)
    
    # Create DUT FSM
    dut_data = DUTData(
        serial_number="DUT_DEMO_001",
        batch_id="DEMO_BATCH",
        creation_time=0.0
    )
    
    from dut_fsm import DUTFSM
    dut_fsm = DUTFSM(dut_data)
    
    print(f"DUT Created: {dut_data.serial_number}")
    print(f"Initial State: {dut_fsm.current_state}")
    print(f"Available Events: {list(dut_fsm.event_methods.keys())}")
    
    # Start DUT processing
    dut_fsm.start_manufacturing_process("FIXTURE_001")
    
    # Run simulation for DUT events
    print("\nExecuting DUT Events:")
    events_executed = global_scheduler.advance_time(50.0)  # Run for 50 seconds
    print(f"Events Executed: {len(events_executed)}")
    print(f"Final DUT State: {dut_fsm.current_state}")
    
    print("\n🔧 Step 2: Fixture Finite State Machine")
    print("-" * 40)
    
    # Create Fixture FSM
    fixture_data = FixtureData(
        fixture_id="FIXTURE_DEMO_001",
        fixture_type="TEST_FIXTURE",
        station_id="DEMO_STATION"
    )
    
    from fixture_fsm import FixtureFSM
    fixture_fsm = FixtureFSM(fixture_data)
    
    print(f"Fixture Created: {fixture_data.fixture_id}")
    print(f"Initial State: {fixture_fsm.current_state}")
    print(f"Available Events: {list(fixture_fsm.event_methods.keys())}")
    
    # Start fixture processing
    fixture_fsm.start_dut_processing("DUT_DEMO_002", test_duration=15.0)
    
    # Run simulation for fixture events
    print("\nExecuting Fixture Events:")
    events_executed = global_scheduler.advance_time(60.0)  # Run for 60 seconds
    print(f"Events Executed: {len(events_executed)}")
    print(f"Final Fixture State: {fixture_fsm.current_state}")
    
    print("\n⚙️ Step 3: Equipment Finite State Machine")
    print("-" * 40)
    
    # Create Equipment FSM
    equipment_data = EquipmentData(
        equipment_id="DMM_DEMO_001",
        equipment_type="DMM",
        station_id="DEMO_STATION",
        model="Agilent_34461A",
        serial_number="DEMO123456"
    )
    
    from equipment_fsm import EquipmentFSM
    equipment_fsm = EquipmentFSM(equipment_data)
    
    print(f"Equipment Created: {equipment_data.equipment_id}")
    print(f"Initial State: {equipment_fsm.current_state}")
    print(f"Available Events: {list(equipment_fsm.event_methods.keys())}")
    
    # Start equipment and perform measurement
    equipment_fsm.start_equipment()
    equipment_fsm.perform_measurement("voltage_dc", 5.0, 0.1, duration=8.0)
    
    # Run simulation for equipment events
    print("\nExecuting Equipment Events:")
    events_executed = global_scheduler.advance_time(30.0)  # Run for 30 seconds
    print(f"Events Executed: {len(events_executed)}")
    print(f"Final Equipment State: {equipment_fsm.current_state}")
    print(f"Measurements Performed: {equipment_data.measurement_count}")
    
    return {
        'dut_status': dut_fsm.get_dut_status(),
        'fixture_status': fixture_fsm.get_fixture_status(),
        'equipment_status': equipment_fsm.get_equipment_status()
    }


def demonstrate_complete_manufacturing_line():
    """Demonstrate complete manufacturing line with FSM integration."""
    print("\n" + "="*80)
    print("🏭 COMPLETE FSM-BASED MANUFACTURING LINE DEMONSTRATION")
    print("="*80)
    
    # Reset scheduler
    global_scheduler.event_queue.clear()
    global_scheduler.current_time = 0.0
    global_scheduler.fsm_registry.clear()
    
    print("\n📋 Step 1: Creating Manufacturing Line")
    print("-" * 40)
    
    # Create manufacturing line
    engine = create_sample_manufacturing_line()
    
    summary = engine.get_simulation_summary()
    print(f"Stations: {summary['station_count']}")
    print(f"Fixtures: {summary['fixture_count']}")
    print(f"Equipment: {summary['equipment_count']}")
    print(f"Registered FSMs: {summary['registered_fsms']}")
    
    print("\n🚀 Step 2: Starting Manufacturing Line")
    print("-" * 40)
    
    # Start all equipment
    engine.start_manufacturing_line()
    
    # Run initialization events
    init_events = global_scheduler.advance_time(15.0)
    print(f"Initialization Events: {len(init_events)}")
    
    print("\n📦 Step 3: Creating DUT Batch")
    print("-" * 40)
    
    # Create batch of DUTs
    batch_size = 5
    dut_ids = engine.create_dut_batch(batch_size, "FSM_DEMO_BATCH")
    print(f"Created {batch_size} DUTs: {dut_ids}")
    
    print("\n🔄 Step 4: Processing DUTs Through Line")
    print("-" * 40)
    
    # Process DUTs through the line
    engine.process_dut_batch(dut_ids, ["P0", "1"])
    
    print("\n⏱️ Step 5: Running Discrete Event Simulation")
    print("-" * 50)
    
    # Run complete simulation
    simulation_results = engine.run_simulation(duration=300.0)  # 5 minutes
    
    print("\n📊 Step 6: Simulation Results")
    print("-" * 35)
    
    # Display results
    metrics = simulation_results['metrics']
    sim_results = simulation_results['simulation_results']
    
    print(f"Simulation Duration: {sim_results['simulation_time']:.1f}s")
    print(f"Events Processed: {sim_results['events_processed']}")
    print(f"Final Simulation Time: {sim_results['final_time']:.1f}s")
    
    print(f"\nDUT Metrics:")
    print(f"  Created: {metrics['duts_created']}")
    print(f"  Completed: {metrics['duts_completed']}")
    print(f"  Failed: {metrics['duts_failed']}")
    print(f"  Overall Yield: {metrics.get('overall_yield', 0):.1%}")
    print(f"  Average Cycle Time: {metrics.get('average_cycle_time', 0):.1f}s")
    
    print(f"\nYield by Station:")
    for station_id, yield_rate in metrics.get('yield_by_station', {}).items():
        print(f"  {station_id}: {yield_rate:.1%}")
    
    return simulation_results


def demonstrate_discrete_event_details():
    """Demonstrate discrete event execution with timing details."""
    print("\n" + "="*80)
    print("⏱️ DISCRETE EVENT TIMING AND EXECUTION DEMONSTRATION")
    print("="*80)
    
    # Reset scheduler
    global_scheduler.event_queue.clear()
    global_scheduler.current_time = 0.0
    global_scheduler.fsm_registry.clear()
    
    print("\n🎯 Step 1: Fixed Duration Event Methods")
    print("-" * 45)
    
    # Create single DUT to demonstrate event timing
    dut_data = DUTData(
        serial_number="TIMING_DUT_001",
        batch_id="TIMING_DEMO",
        creation_time=0.0
    )
    
    from dut_fsm import DUTFSM
    dut_fsm = DUTFSM(dut_data)
    
    print("DUT Event Methods and Fixed Durations:")
    for event_name, event_info in dut_fsm.event_methods.items():
        print(f"  {event_name}: {event_info['duration']:.1f}s")
    
    print("\n📈 Step 2: Event Execution Timeline")
    print("-" * 40)
    
    # Track event execution with timestamps
    initial_time = global_scheduler.current_time
    print(f"Simulation Start Time: {initial_time:.1f}s")
    
    # Trigger manual events to show timing
    dut_fsm.trigger_event(DUTEvent.HANDLE_IN.value, delay=0.0, fixture_id="TIMING_FIXTURE")
    print(f"Scheduled HANDLE_IN at: {global_scheduler.current_time:.1f}s")
    
    # Execute events step by step
    print("\nEvent Execution Timeline:")
    for step in range(10):  # Execute 10 time steps
        old_time = global_scheduler.current_time
        events = global_scheduler.advance_time(5.0)  # Advance 5 seconds
        new_time = global_scheduler.current_time
        
        if events:
            for event in events:
                print(f"  {new_time:.1f}s: Executed {event.event_name} (duration: {event.duration:.1f}s)")
                print(f"           DUT State: {dut_fsm.current_state}")
        
        if not global_scheduler.event_queue:
            print(f"  {new_time:.1f}s: No more events in queue")
            break
    
    print(f"\nFinal State: {dut_fsm.current_state}")
    print(f"Total Simulation Time: {global_scheduler.current_time:.1f}s")
    
    return dut_fsm.get_dut_status()


def demonstrate_concurrent_fsm_operation():
    """Demonstrate multiple FSMs operating concurrently."""
    print("\n" + "="*80)
    print("🔀 CONCURRENT FSM OPERATION DEMONSTRATION")
    print("="*80)
    
    # Reset scheduler
    global_scheduler.event_queue.clear()
    global_scheduler.current_time = 0.0
    global_scheduler.fsm_registry.clear()
    
    print("\n👥 Step 1: Creating Multiple FSMs")
    print("-" * 35)
    
    # Create multiple FSMs that will operate concurrently
    fsms = {}
    
    # Create 3 DUT FSMs
    for i in range(3):
        dut_data = DUTData(
            serial_number=f"CONCURRENT_DUT_{i+1:03d}",
            batch_id="CONCURRENT_BATCH",
            creation_time=0.0
        )
        from dut_fsm import DUTFSM
        dut_fsm = DUTFSM(dut_data)
        fsms[f'dut_{i+1}'] = dut_fsm
    
    # Create 2 Fixture FSMs
    for i in range(2):
        fixture_data = FixtureData(
            fixture_id=f"CONCURRENT_FIXTURE_{i+1:03d}",
            fixture_type="CONCURRENT_FIXTURE",
            station_id=f"STATION_{i+1}"
        )
        from fixture_fsm import FixtureFSM
        fixture_fsm = FixtureFSM(fixture_data)
        fsms[f'fixture_{i+1}'] = fixture_fsm
    
    # Create 2 Equipment FSMs
    for i in range(2):
        equipment_data = EquipmentData(
            equipment_id=f"CONCURRENT_EQUIPMENT_{i+1:03d}",
            equipment_type="DMM" if i == 0 else "OSCILLOSCOPE",
            station_id=f"STATION_{i+1}",
            model=f"MODEL_{i+1}",
            serial_number=f"SN_{i+1:06d}"
        )
        from equipment_fsm import EquipmentFSM
        equipment_fsm = EquipmentFSM(equipment_data)
        fsms[f'equipment_{i+1}'] = equipment_fsm
    
    print(f"Created {len(fsms)} FSMs:")
    for name, fsm in fsms.items():
        print(f"  {name}: {fsm.fsm_id} (state: {fsm.current_state})")
    
    print(f"\nRegistered FSMs in scheduler: {len(global_scheduler.fsm_registry)}")
    
    print("\n🚀 Step 2: Starting Concurrent Operations")
    print("-" * 45)
    
    # Start operations on all FSMs with different delays
    delay = 0.0
    
    # Start DUTs
    for i, (name, fsm) in enumerate([item for item in fsms.items() if 'dut' in item[0]]):
        fsm.trigger_event('handle_in', delay=delay, fixture_id=f'fixture_{i+1}')
        print(f"Started {name} at delay {delay:.1f}s")
        delay += 3.0  # Stagger by 3 seconds
    
    # Start fixtures
    for i, (name, fsm) in enumerate([item for item in fsms.items() if 'fixture' in item[0]]):
        fsm.trigger_event('cmd_load', delay=delay, dut_id=f'dut_{i+1}')
        print(f"Started {name} at delay {delay:.1f}s")
        delay += 2.0  # Stagger by 2 seconds
    
    # Start equipment
    for name, fsm in [item for item in fsms.items() if 'equipment' in item[0]]:
        fsm.start_equipment()
        print(f"Started {name}")
    
    print("\n⚡ Step 3: Running Concurrent Simulation")
    print("-" * 45)
    
    # Run simulation and show concurrent execution
    total_events = 0
    for time_step in range(12):  # 12 time steps of 10 seconds each
        step_start_time = global_scheduler.current_time
        events = global_scheduler.advance_time(10.0)  # 10 second steps
        step_end_time = global_scheduler.current_time
        
        print(f"\nTime {step_start_time:.1f}s - {step_end_time:.1f}s:")
        print(f"  Events executed: {len(events)}")
        
        if events:
            # Show which FSMs are active
            active_fsms = set()
            for event in events:
                for fsm in global_scheduler.fsm_registry.values():
                    if fsm.can_handle_event(event):
                        active_fsms.add(fsm.fsm_id)
            
            print(f"  Active FSMs: {', '.join(sorted(active_fsms))}")
        
        total_events += len(events)
        
        if not global_scheduler.event_queue:
            print("  No more events in queue")
            break
    
    print(f"\n📊 Step 4: Final Concurrent States")
    print("-" * 40)
    
    for name, fsm in fsms.items():
        print(f"{name}: {fsm.current_state}")
    
    print(f"\nTotal Events Processed: {total_events}")
    print(f"Final Simulation Time: {global_scheduler.current_time:.1f}s")
    
    return {name: fsm.get_state_info() for name, fsm in fsms.items()}


if __name__ == "__main__":
    print("🔧 DISCRETE EVENT-BASED FINITE STATE MACHINE DEMONSTRATION")
    print("🎯 Demonstrating: Fixed Duration Events, State Transitions, Concurrent Execution")
    
    # Run individual component demos
    component_results = demonstrate_individual_fsm_components()
    
    # Run complete manufacturing line demo
    line_results = demonstrate_complete_manufacturing_line()
    
    # Run discrete event timing demo
    timing_results = demonstrate_discrete_event_details()
    
    # Run concurrent FSM demo
    concurrent_results = demonstrate_concurrent_fsm_operation()
    
    print("\n" + "="*80)
    print("🎉 FSM DISCRETE EVENT DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\n✅ Successfully demonstrated:")
    print("  • Discrete Event-Based Deterministic Finite State Machines")
    print("  • Fixed duration event methods (CMD_LOAD: 3.0s, CMD_MEASURE: Variable, etc.)")
    print("  • State transitions with entry/exit actions")
    print("  • Event scheduling and execution timing")
    print("  • Concurrent FSM operation with shared scheduler")
    print("  • Complete manufacturing line simulation")
    print("  • DUT, Fixture, and Equipment FSM integration")
    
    print(f"\n📈 Key Architecture Features:")
    print(f"  • Each object (DUT, Fixture, Equipment) has fixed discrete event methods")
    print(f"  • Methods execute with specific durations (time-based)")
    print(f"  • State machines manage object behavior deterministically")
    print(f"  • Central scheduler coordinates all discrete events")
    print(f"  • Events are processed in chronological order")
    print(f"  • Concurrent operations supported through event scheduling")
    
    print(f"\n🔄 This forms the backbone of the manufacturing simulation system!")