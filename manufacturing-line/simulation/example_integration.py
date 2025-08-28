#!/usr/bin/env python3
"""
Example integration demonstrating complete manufacturing line simulation system.

This example shows how to:
1. Set up digital twins for stations, conveyors, and operators
2. Integrate JAAMSIM simulation with real-time data
3. Use simulation hooks for predictive analytics
4. Run what-if scenarios for line optimization
"""

import time
import json
import logging
from pathlib import Path

# Simulation framework imports
from simulation_engine.base_simulation import simulation_manager, create_simulation_config
from simulation_engine.digital_twin import (
    StationDigitalTwin, ConveyorDigitalTwin, OperatorDigitalTwin,
    digital_twin_manager
)
from simulation_engine.simulation_hooks import (
    simulation_integration_service, simulation_hook_manager,
    SimulationEventType, trigger_simulation_event
)
from jaamsim_integration.jaamsim_simulation import create_jaamsim_config

# Manufacturing line component imports
from conveyors.belt_conveyor import BeltConveyor
from operators.digital_human import DigitalHuman

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_station_twin(station_id: str, fixture_type: str = "1-up") -> StationDigitalTwin:
    """Create a sample station digital twin."""
    twin = StationDigitalTwin(station_id, fixture_type)
    
    # Configure station parameters
    twin.station_config = {
        'good_dut_percentage': 85,
        'relit_dut_percentage': 10,
        'measurement_time': 9,
        'operator_load_time': 10,
        'operator_unload_time': 5,
        'ptb_litup_time': 5,
        'ptb_retry_count': 3
    }
    
    return twin


def create_sample_conveyor_twin(conveyor_id: str) -> ConveyorDigitalTwin:
    """Create a sample conveyor digital twin."""
    twin = ConveyorDigitalTwin(conveyor_id)
    
    # Configure conveyor segments
    twin.segments = [
        {'id': 'seg_001', 'from': 'ICT_01', 'to': 'FCT_01', 'length': 5.0},
        {'id': 'seg_002', 'from': 'FCT_01', 'to': 'CAMERA_01', 'length': 5.0}
    ]
    
    return twin


def create_sample_operator_twin(operator_id: str) -> OperatorDigitalTwin:
    """Create a sample operator digital twin."""
    twin = OperatorDigitalTwin(operator_id)
    
    # Configure operator behavior model
    twin.behavior_model = {
        'skill_level': 0.9,
        'attention_level': 0.95,
        'reaction_time': 2.0,
        'error_rate': 0.05
    }
    
    return twin


def setup_simulation_hooks():
    """Set up simulation event hooks for line integration."""
    
    def line_controller_callback(event):
        """Handle events from line controller perspective."""
        logger.info(f"Line Controller received: {event.event_type.value} from {event.component_id}")
        
        if event.event_type == SimulationEventType.BOTTLENECK_DETECTED:
            logger.warning(f"Bottleneck detected at {event.data.get('station')}")
            # In real implementation, would trigger line rebalancing
        
        elif event.event_type == SimulationEventType.PERFORMANCE_ALERT:
            alert_type = event.data.get('alert_type')
            logger.warning(f"Performance alert: {alert_type} for {event.component_id}")
    
    def database_callback(event):
        """Handle database storage of simulation events."""
        event_data = {
            'timestamp': event.timestamp,
            'event_type': event.event_type.value,
            'component_id': event.component_id,
            'data': event.data,
            'severity': event.severity
        }
        
        logger.info(f"Database: Storing event {event.event_type.value}")
        # In real implementation, would store to PocketBase
    
    # Register callbacks
    simulation_hook_manager.set_line_controller_callback(line_controller_callback)
    simulation_hook_manager.set_database_callback(database_callback)
    
    # Add external webhook (example)
    simulation_hook_manager.add_external_webhook(
        url="https://mes.factory.com/webhook",
        auth_token="your_token_here"
    )


def simulate_real_time_data():
    """Simulate real-time data from manufacturing line components."""
    
    # Simulate ICT station performance data
    ict_data = {
        'uph_actual': 118,
        'yield': 0.96,
        'cycle_time': 28.5,
        'efficiency': 0.92,
        'downtime_minutes': 2.5
    }
    
    # Simulate conveyor performance data  
    conveyor_data = {
        'current_speed': 0.45,
        'actual_throughput': 115,
        'dut_count': 8,
        'max_capacity': 10
    }
    
    # Simulate operator performance data
    operator_data = {
        'actions_completed': 156,
        'avg_response_time': 2.8,
        'success_rate': 0.94,
        'fatigue_level': 0.2
    }
    
    # Update digital twins with real data
    simulation_integration_service.process_real_time_data('ICT_01', ict_data)
    simulation_integration_service.process_real_time_data('MAIN_BELT', conveyor_data)
    simulation_integration_service.process_real_time_data('DH_001', operator_data)


def run_what_if_scenario():
    """Run a what-if scenario simulation."""
    logger.info("Running what-if scenario: Station downtime impact")
    
    # Scenario: ICT station down for 2 hours
    scenario_params = {
        'config_file': 'stations/fixture/simulation/cfg/1up/1-up-station-simulation.cfg',
        'simulation_params': {
            'GoodDUT': 0,  # Simulate station down
            'TotalDUT': 500,
            'StationTime_Input': 0  # No processing time
        },
        'real_time_factor': 32.0,
        'max_runtime': 120.0
    }
    
    # Run scenario
    simulation_id = simulation_integration_service.trigger_scenario_simulation(
        'ict_downtime_impact',
        scenario_params
    )
    
    logger.info(f"Scenario simulation started: {simulation_id}")
    
    # Wait for completion and get results
    simulation = simulation_manager.get_simulation(simulation_id)
    if simulation:
        result = simulation.wait_for_completion(timeout=180)
        if result:
            logger.info(f"Scenario results: {result.predictions}")
            
            # Trigger analysis event
            trigger_simulation_event(
                'simulation_completed',
                'ict_downtime_impact',
                {
                    'scenario': 'station_downtime',
                    'simulation_result': result.__dict__,
                    'impact_analysis': {
                        'predicted_line_impact': result.predictions.get('predicted_uph', 0),
                        'recommended_actions': [
                            'Activate backup station',
                            'Redirect DUTs to parallel line',
                            'Increase downstream buffer capacity'
                        ]
                    }
                },
                'warning'
            )


def demonstrate_full_integration():
    """Demonstrate complete simulation integration."""
    logger.info("=== Manufacturing Line Simulation Integration Demo ===")
    
    # Step 1: Set up simulation hooks
    logger.info("1. Setting up simulation hooks...")
    setup_simulation_hooks()
    
    # Step 2: Create and register digital twins
    logger.info("2. Creating digital twins...")
    
    # Create station twins
    ict_twin = create_sample_station_twin('ICT_01', '1-up')
    fct_twin = create_sample_station_twin('FCT_01', '3-up-turntable')
    
    # Create conveyor twin
    conveyor_twin = create_sample_conveyor_twin('MAIN_BELT')
    
    # Create operator twin
    operator_twin = create_sample_operator_twin('DH_001')
    
    # Register twins
    digital_twin_manager.register_twin(ict_twin)
    digital_twin_manager.register_twin(fct_twin)
    digital_twin_manager.register_twin(conveyor_twin)
    digital_twin_manager.register_twin(operator_twin)
    
    # Step 3: Start digital twin synchronization
    logger.info("3. Starting digital twin synchronization...")
    digital_twin_manager.start_all_twins()
    
    # Step 4: Simulate real-time data updates
    logger.info("4. Processing real-time data...")
    simulate_real_time_data()
    
    # Step 5: Run predictive simulation
    logger.info("5. Running predictive simulations...")
    
    # Run 1-up station prediction
    ict_config = create_jaamsim_config(
        config_id='ict_prediction',
        cfg_file_path='stations/fixture/simulation/cfg/1up/1-up-station-simulation.cfg',
        parameters={'GoodDUT': 85, 'TotalDUT': 500},
        real_time_factor=32.0,
        max_runtime=60.0
    )
    
    prediction_sim_id = simulation_manager.run_scenario('ict_prediction', ict_config)
    logger.info(f"ICT prediction simulation started: {prediction_sim_id}")
    
    # Step 6: Run what-if scenario
    logger.info("6. Running what-if scenario...")
    run_what_if_scenario()
    
    # Step 7: Get line performance summary
    logger.info("7. Getting line performance summary...")
    time.sleep(2)  # Let some events process
    
    summary = simulation_integration_service.get_line_performance_summary()
    logger.info("Line Performance Summary:")
    logger.info(f"- Active twins: {summary['active_twins']}")
    logger.info(f"- Alert counts: {summary['alerts']}")
    
    if 'line_summary' in summary['line_predictions']:
        line_summary = summary['line_predictions']['line_summary']
        logger.info(f"- Predicted line UPH: {line_summary.get('predicted_line_uph', 'N/A')}")
        logger.info(f"- Predicted efficiency: {line_summary.get('predicted_efficiency', 'N/A'):.2%}")
    
    # Step 8: Demonstrate real-time integration with physical components
    logger.info("8. Demonstrating physical component integration...")
    
    # Create sample conveyor with simulation integration
    conveyor_config = {
        'type': 'belt',
        'segments': [
            {'id': 'seg_001', 'from_station': 'ICT_01', 'to_station': 'FCT_01', 'length': 5.0, 'max_speed': 0.5}
        ],
        'station_stops': ['ICT_01', 'FCT_01'],
        'simulation_enabled': True
    }
    
    conveyor = BeltConveyor('MAIN_BELT', conveyor_config)
    conveyor.set_digital_twin(conveyor_twin)
    
    # Add data callback to demonstrate real-time integration
    def conveyor_data_callback(data):
        logger.info(f"Conveyor data update: Speed={data.get('current_speed'):.2f}, DUTs={data.get('dut_count')}")
    
    conveyor.add_data_callback(conveyor_data_callback)
    
    # Simulate conveyor operation
    conveyor.start()
    conveyor.load_dut('DUT_001', 'ICT_01', 'FCT_01')
    
    # Update positions to trigger callbacks
    conveyor.update_positions(1.0)
    
    conveyor.stop()
    
    # Step 9: Clean up
    logger.info("9. Cleaning up...")
    digital_twin_manager.stop_all_twins()
    simulation_manager.stop_all_simulations()
    
    logger.info("=== Demo Complete ===")


def create_example_config_files():
    """Create example configuration files for simulation scenarios."""
    
    # Example line configuration with simulation parameters
    line_config = {
        "line_id": "DEMO_LINE_01",
        "stations": [
            {
                "station_id": "ICT_01",
                "fixture_type": "1-up",
                "simulation": {
                    "enabled": True,
                    "config_template": "1up_station_template",
                    "parameters": {
                        "GoodDUT": 85,
                        "RelitDUT": 10,
                        "TotalDUT": 1000,
                        "StationTime_Input": 9
                    }
                }
            },
            {
                "station_id": "FCT_01", 
                "fixture_type": "3-up-turntable",
                "simulation": {
                    "enabled": True,
                    "config_template": "3up_turntable_template",
                    "parameters": {
                        "InputValue-GoodDUT": 95,
                        "InputValue-TotalDUT": 5000
                    }
                }
            }
        ],
        "conveyors": [
            {
                "conveyor_id": "MAIN_BELT",
                "simulation": {
                    "enabled": True,
                    "transport_model": "EntityConveyor",
                    "parameters": {
                        "belt_speed": 0.5,
                        "segment_count": 2
                    }
                }
            }
        ],
        "operators": [
            {
                "operator_id": "DH_001",
                "simulation": {
                    "enabled": True,
                    "behavior_model": "DigitalHumanAgent",
                    "parameters": {
                        "skill_level": 0.9,
                        "attention_level": 0.95
                    }
                }
            }
        ]
    }
    
    # Save example configuration
    config_path = Path("simulation/scenario_configs/demo_line_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(line_config, f, indent=2)
    
    logger.info(f"Created example configuration: {config_path}")


if __name__ == "__main__":
    # Create example configuration files
    create_example_config_files()
    
    # Run the complete integration demonstration
    demonstrate_full_integration()