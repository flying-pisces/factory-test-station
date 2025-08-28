# Manufacturing Line Simulation Framework

This simulation framework provides comprehensive discrete event simulation capabilities for manufacturing lines, integrating JAAMSIM with real-time digital twins for stations, conveyors, and operators.

## Architecture Overview

```
Manufacturing Line (Physical)
        ↕ ️(Real-time data sync)
Digital Twins (Python)
        ↕️ (Configuration & Results)
JAAMSIM Simulation (Java)
        ↕️ (Events & Predictions)
Integration Hooks (System)
```

## Key Features

### 🔄 Digital Twin Integration
- **Station Twins**: Predict performance, bottlenecks, and optimize fixture configurations
- **Conveyor Twins**: Model material flow and identify transport bottlenecks  
- **Operator Twins**: Simulate human behavior with fatigue, learning, and skill modeling
- **Real-time Synchronization**: Continuous calibration with actual performance data

### 🎯 JAAMSIM Integration  
- **Pre-configured Templates**: 1-up stations, 3-up turntables, custom fixtures
- **Parameterized Models**: Easy configuration updates without modifying .cfg files
- **Cross-platform Support**: macOS, Linux, Windows with native library management
- **Performance Optimized**: Fast execution with configurable real-time factors

### 🔗 Simulation Hooks
- **Event-driven Architecture**: React to simulation events in real-time
- **External Integration**: MES, database, webhook support
- **Predictive Analytics**: Bottleneck detection, performance alerts
- **What-if Scenarios**: Test changes before implementation

## Directory Structure

```
simulation/
├── simulation_engine/          # Core simulation framework
│   ├── base_simulation.py      # Abstract simulation interface
│   ├── digital_twin.py         # Digital twin implementations
│   └── simulation_hooks.py     # Event hooks and integration
├── jaamsim_integration/        # JAAMSIM-specific integration
│   └── jaamsim_simulation.py   # JAAMSIM execution engine
├── isaac_sim_integration/      # Future NVIDIA Isaac Sim integration
├── scenario_configs/           # Simulation scenario templates
│   ├── 1up_station_template.json
│   ├── 3up_turntable_template.json
│   └── demo_line_config.json
└── example_integration.py      # Complete integration example
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure JAAMSIM is available
# Place JaamSim2022-06.jar in stations/fixture/simulation/cfg/
# Or install system-wide
```

### 2. Basic Station Simulation

```python
from simulation_engine.base_simulation import simulation_manager
from jaamsim_integration.jaamsim_simulation import create_jaamsim_config

# Create 1-up station configuration
config = create_jaamsim_config(
    config_id='ict_test',
    cfg_file_path='stations/fixture/simulation/cfg/1up/1-up-station-simulation.cfg',
    parameters={
        'GoodDUT': 85,      # 85% good DUTs
        'RelitDUT': 10,     # 10% require re-lit
        'TotalDUT': 1000,   # Process 1000 DUTs
        'StationTime_Input': 9,    # 9s measurement time
        'Input_Load': 10,          # 10s operator load time
        'PTBTime_Input': 5         # 5s PTB litup time
    },
    real_time_factor=16.0,  # 16x speed
    max_runtime=300.0       # 5 minutes max
)

# Run simulation
simulation_id = simulation_manager.run_scenario('ict_performance_test', config)

# Get results
simulation = simulation_manager.get_simulation(simulation_id)
result = simulation.wait_for_completion()

print(f"Predicted UPH: {result.predictions.get('predicted_uph')}")
print(f"Efficiency: {result.performance.get('oee'):.2%}")
```

### 3. Digital Twin Integration

```python
from simulation_engine.digital_twin import StationDigitalTwin, digital_twin_manager

# Create station digital twin
twin = StationDigitalTwin('ICT_01', fixture_type='1-up')
twin.station_config = {
    'good_dut_percentage': 85,
    'measurement_time': 9,
    'operator_load_time': 10
}

# Register and start synchronization
digital_twin_manager.register_twin(twin)
twin.start_sync()

# Update with real performance data
real_data = {
    'uph_actual': 118,
    'yield': 0.96,
    'cycle_time': 28.5,
    'efficiency': 0.92
}
twin.update_real_data(real_data)

# Get predictions
predictions = twin.generate_predictions()
print(f"Predicted performance: {predictions}")
```

### 4. Event Hooks and Integration

```python
from simulation_engine.simulation_hooks import (
    simulation_hook_manager, SimulationEventType, 
    simulation_integration_service
)

# Set up callbacks for line controller
def line_callback(event):
    print(f"Line event: {event.event_type.value} from {event.component_id}")
    
    if event.event_type == SimulationEventType.BOTTLENECK_DETECTED:
        print(f"Bottleneck at {event.data.get('station')}")
        # Trigger line rebalancing logic

simulation_hook_manager.set_line_controller_callback(line_callback)

# Add external webhook
simulation_hook_manager.add_external_webhook(
    url="https://mes.factory.com/webhook",
    auth_token="your_token"
)

# Process real-time data
simulation_integration_service.process_real_time_data('ICT_01', real_data)
```

## Simulation Templates

### 1-up Station Template
- **Use Case**: Single DUT processing (ICT, FCT, Camera, Display tests)
- **Key Parameters**: GoodDUT%, StationTime, OperatorLoad/Unload times
- **Typical UPH**: 60-120 depending on test complexity

### 3-up Turntable Template  
- **Use Case**: Parallel processing with turntable (RF, Burn-in, Multi-step assembly)
- **Key Parameters**: Parallel processing count, container management, stage timing
- **Typical UPH**: 150-300 with 3x parallelization

### Custom Templates
Create custom templates by:
1. Copying existing .cfg files from JAAMSIM work
2. Creating parameter mapping in JSON template
3. Implementing specific prediction algorithms

## Integration Patterns

### Real-time Data Sync
```python
# Station performance monitoring
station_twin.update_real_data({
    'current_uph': 115,
    'yield': 0.97,
    'downtime_events': ['calibration_required']
})

# Automatic prediction updates
predictions = station_twin.predicted_data
if predictions['bottleneck_risk'] == 'high':
    # Trigger preventive action
    pass
```

### What-if Scenarios
```python
# Test impact of station downtime
scenario_params = {
    'simulation_params': {
        'GoodDUT': 0,  # Simulate station down
        'StationTime_Input': 0
    },
    'duration': '2h'  # 2 hour downtime
}

simulation_id = simulation_integration_service.trigger_scenario_simulation(
    'station_downtime_impact', 
    scenario_params
)
```

### Performance Alerts
```python
# Automatic bottleneck detection
@simulation_hook_manager.register_hook(SimulationEventType.BOTTLENECK_DETECTED)
def handle_bottleneck(event):
    station = event.data['station']
    impact = event.data['predicted_impact']
    
    # Send alert to production team
    # Trigger line rebalancing
    # Log for analysis
```

## Configuration Management

### Parameter Mapping
JAAMSIM configuration files use specific parameter names. The framework provides mapping:

```python
# Framework parameter → JAAMSIM config parameter
parameter_mapping = {
    'good_dut_percentage': 'GoodDUT Value',
    'measurement_time': 'StationTime_Input Value', 
    'operator_load_time': 'Input_Load Value',
    'total_dut_count': 'TotalDUT Value'
}
```

### Scenario Templates
JSON templates define reusable simulation scenarios:

```json
{
  "config_id": "high_volume_test",
  "parameters": {
    "TotalDUT": 10000,
    "GoodDUT": 95,
    "StationTime_Input": 8
  },
  "real_time_factor": 32.0,
  "expected_metrics": ["uph", "efficiency", "bottleneck_risk"]
}
```

## Performance Optimization

### Fast Simulation Execution
- **Real-time Factor**: Use 16-32x for quick feedback
- **DUT Count**: Use 500-1000 DUTs for balance of accuracy and speed  
- **Parallel Execution**: Run multiple scenarios simultaneously

### Memory Management
- **Result Caching**: Cache simulation results for repeated scenarios
- **History Limits**: Limit event history to prevent memory growth
- **Resource Cleanup**: Automatic cleanup of temporary files

## Troubleshooting

### Common Issues

**JAAMSIM JAR Not Found**
```bash
# Check JAAMSIM availability
java -jar stations/fixture/simulation/cfg/JaamSim2022-06.jar --help

# Or place JAR in simulation/jaamsim_integration/
```

**macOS Native Library Issues**
```bash
# Set up macOS environment
source stations/fixture/simulation/setup-macos.sh

# Or run with explicit flags
java -Djava.library.path=natives/macosx-universal -jar JaamSim.jar config.cfg
```

**Simulation Timeout**
```python
# Increase timeout for complex scenarios
config.max_runtime = 600.0  # 10 minutes

# Or use faster real-time factor
config.real_time_factor = 64.0  # 64x speed
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed simulation logging
simulation.logger.setLevel(logging.DEBUG)
```

## Examples

See `example_integration.py` for a complete demonstration of:
- Setting up digital twins for all component types
- Running predictive simulations
- Processing real-time data updates
- Handling simulation events and alerts
- Integration with physical manufacturing components

## Future Enhancements

### NVIDIA Isaac Sim Integration
- Physics-based 3D simulation
- Robot and automation modeling
- Computer vision integration
- AI training environments

### Advanced Analytics
- Machine learning-based predictions
- Anomaly detection algorithms
- Optimization recommendations
- Historical trend analysis

### Extended Integrations
- MES/ERP system connectors
- PLM system integration
- Supply chain modeling
- Quality system integration