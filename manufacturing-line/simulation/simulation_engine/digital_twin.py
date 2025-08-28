"""Digital Twin integration for manufacturing line components."""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import threading
import logging
import json
from enum import Enum

from .base_simulation import BaseSimulation, SimulationConfig, SimulationResult, create_simulation_config
from ..jaamsim_integration.jaamsim_simulation import create_jaamsim_config


class ComponentType(Enum):
    STATION = "station"
    CONVEYOR = "conveyor"
    OPERATOR = "operator"
    LINE = "line"


class TwinSyncStatus(Enum):
    SYNCED = "synced"
    OUT_OF_SYNC = "out_of_sync"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class DigitalTwinMetrics:
    """Metrics for digital twin performance."""
    prediction_accuracy: float = 0.0
    sync_latency: float = 0.0
    simulation_runtime: float = 0.0
    last_sync_time: float = 0.0
    sync_error_count: int = 0
    total_predictions: int = 0
    correct_predictions: int = 0


class BaseDigitalTwin(ABC):
    """Base class for digital twin implementations."""
    
    def __init__(self, component_id: str, component_type: ComponentType):
        self.component_id = component_id
        self.component_type = component_type
        self.logger = logging.getLogger(f"DigitalTwin_{component_id}")
        self.sync_status = TwinSyncStatus.DISABLED
        self.metrics = DigitalTwinMetrics()
        self.real_data: Dict[str, Any] = {}
        self.predicted_data: Dict[str, Any] = {}
        self.simulation: Optional[BaseSimulation] = None
        self._sync_callbacks: List[Callable] = []
        self._is_active = False
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_interval = 30.0  # seconds
    
    @abstractmethod
    def create_simulation_config(self) -> SimulationConfig:
        """Create simulation configuration for this component."""
        pass
    
    @abstractmethod
    def update_real_data(self, data: Dict[str, Any]):
        """Update with real-world data from the physical component."""
        pass
    
    @abstractmethod
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions based on current state."""
        pass
    
    @abstractmethod
    def calculate_sync_accuracy(self) -> float:
        """Calculate accuracy between predictions and reality."""
        pass
    
    def start_sync(self):
        """Start digital twin synchronization."""
        if self._is_active:
            return
        
        self._is_active = True
        self.sync_status = TwinSyncStatus.SYNCED
        self._sync_thread = threading.Thread(target=self._sync_loop)
        self._sync_thread.daemon = True
        self._sync_thread.start()
        self.logger.info(f"Digital twin sync started for {self.component_id}")
    
    def stop_sync(self):
        """Stop digital twin synchronization."""
        self._is_active = False
        self.sync_status = TwinSyncStatus.DISABLED
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        self.logger.info(f"Digital twin sync stopped for {self.component_id}")
    
    def _sync_loop(self):
        """Main synchronization loop."""
        while self._is_active:
            try:
                # Generate new predictions
                new_predictions = self.generate_predictions()
                
                # Update predicted data
                self.predicted_data.update(new_predictions)
                
                # Calculate accuracy if we have both real and predicted data
                if self.real_data and self.predicted_data:
                    accuracy = self.calculate_sync_accuracy()
                    self.metrics.prediction_accuracy = accuracy
                    self.metrics.last_sync_time = time.time()
                
                # Trigger callbacks
                self._trigger_sync_callbacks()
                
                # Update sync status
                if self.metrics.prediction_accuracy > 0.8:
                    self.sync_status = TwinSyncStatus.SYNCED
                elif self.metrics.prediction_accuracy > 0.6:
                    self.sync_status = TwinSyncStatus.OUT_OF_SYNC
                else:
                    self.sync_status = TwinSyncStatus.ERROR
                
            except Exception as e:
                self.logger.error(f"Sync error: {e}")
                self.sync_status = TwinSyncStatus.ERROR
                self.metrics.sync_error_count += 1
            
            time.sleep(self._sync_interval)
    
    def add_sync_callback(self, callback: Callable):
        """Add callback for sync events."""
        self._sync_callbacks.append(callback)
    
    def _trigger_sync_callbacks(self):
        """Trigger all registered sync callbacks."""
        for callback in self._sync_callbacks:
            try:
                callback(self)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get digital twin status."""
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'sync_status': self.sync_status.value,
            'is_active': self._is_active,
            'metrics': {
                'prediction_accuracy': self.metrics.prediction_accuracy,
                'sync_latency': self.metrics.sync_latency,
                'last_sync_time': self.metrics.last_sync_time,
                'error_count': self.metrics.sync_error_count
            },
            'real_data_keys': list(self.real_data.keys()),
            'predicted_data_keys': list(self.predicted_data.keys())
        }


class StationDigitalTwin(BaseDigitalTwin):
    """Digital twin for manufacturing stations."""
    
    def __init__(self, station_id: str, fixture_type: str = "1-up"):
        super().__init__(station_id, ComponentType.STATION)
        self.fixture_type = fixture_type
        self.station_config = {}
        self.performance_history: List[Dict[str, Any]] = []
        
    def create_simulation_config(self) -> SimulationConfig:
        """Create JAAMSIM simulation configuration for this station."""
        # Determine config file based on fixture type
        if self.fixture_type == "1-up":
            config_file = "stations/fixture/simulation/cfg/1up/1-up-station-simulation.cfg"
        elif self.fixture_type == "3-up-turntable":
            config_file = "stations/fixture/simulation/cfg/3upturntable/Turn-table-simulation.cfg"
        else:
            config_file = f"stations/fixture/simulation/cfg/{self.fixture_type}/{self.fixture_type}-simulation.cfg"
        
        # Default parameters based on current station configuration
        parameters = {
            'GoodDUT': self.station_config.get('good_dut_percentage', 85),
            'RelitDUT': self.station_config.get('relit_dut_percentage', 10),
            'TotalDUT': 500,  # Smaller runs for faster feedback
            'StationTime_Input': self.station_config.get('measurement_time', 9),
            'Input_Load': self.station_config.get('operator_load_time', 10),
            'Input_Unload': self.station_config.get('operator_unload_time', 5),
            'PTBTime_Input': self.station_config.get('ptb_litup_time', 5),
            'Input_PTB_Retry': self.station_config.get('ptb_retry_count', 3)
        }
        
        return create_jaamsim_config(
            config_id=f"station_{self.component_id}",
            cfg_file_path=config_file,
            parameters=parameters,
            real_time_factor=32.0,  # Faster simulation
            max_runtime=120.0  # 2 minutes max
        )
    
    def update_real_data(self, data: Dict[str, Any]):
        """Update with real station performance data."""
        self.real_data.update(data)
        
        # Track performance history
        performance_record = {
            'timestamp': time.time(),
            'uph': data.get('uph_actual', 0),
            'yield': data.get('yield', 0),
            'cycle_time': data.get('cycle_time', 0),
            'downtime': data.get('downtime_minutes', 0)
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only last 100 records
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions using JAAMSIM simulation."""
        try:
            # Update simulation config based on recent performance
            config = self.create_simulation_config()
            
            # Run quick simulation
            from ..jaamsim_integration.jaamsim_simulation import JaamSimSimulation
            simulation_id = f"{self.component_id}_prediction_{int(time.time())}"
            simulation = JaamSimSimulation(simulation_id, config)
            
            # Initialize and run
            if simulation.initialize():
                result = simulation.execute()
                
                if result.status.value == 'completed':
                    predictions = {
                        'predicted_uph': result.predictions.get('predicted_uph', 0),
                        'predicted_cycle_time': result.predictions.get('predicted_cycle_time', 0),
                        'predicted_efficiency': result.predictions.get('predicted_efficiency', 0),
                        'bottleneck_risk': result.predictions.get('bottleneck_risk', 'unknown'),
                        'simulation_runtime': result.duration,
                        'confidence': 0.85  # Base confidence
                    }
                    
                    return predictions
        
        except Exception as e:
            self.logger.error(f"Prediction generation error: {e}")
        
        # Return empty predictions on error
        return {}
    
    def calculate_sync_accuracy(self) -> float:
        """Calculate accuracy between predicted and actual performance."""
        if not self.real_data or not self.predicted_data:
            return 0.0
        
        accuracies = []
        
        # Compare UPH if available
        real_uph = self.real_data.get('uph_actual')
        pred_uph = self.predicted_data.get('predicted_uph')
        if real_uph and pred_uph and pred_uph > 0:
            uph_accuracy = 1.0 - abs(real_uph - pred_uph) / pred_uph
            accuracies.append(max(0.0, uph_accuracy))
        
        # Compare cycle time if available
        real_cycle = self.real_data.get('cycle_time')
        pred_cycle = self.predicted_data.get('predicted_cycle_time')
        if real_cycle and pred_cycle and pred_cycle > 0:
            cycle_accuracy = 1.0 - abs(real_cycle - pred_cycle) / pred_cycle
            accuracies.append(max(0.0, cycle_accuracy))
        
        # Compare efficiency if available
        real_eff = self.real_data.get('efficiency')
        pred_eff = self.predicted_data.get('predicted_efficiency')
        if real_eff and pred_eff and pred_eff > 0:
            eff_accuracy = 1.0 - abs(real_eff - pred_eff) / pred_eff
            accuracies.append(max(0.0, eff_accuracy))
        
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    def update_station_config(self, config: Dict[str, Any]):
        """Update station configuration."""
        self.station_config.update(config)


class ConveyorDigitalTwin(BaseDigitalTwin):
    """Digital twin for conveyor systems."""
    
    def __init__(self, conveyor_id: str):
        super().__init__(conveyor_id, ComponentType.CONVEYOR)
        self.segments = []
        self.dut_tracking_history = []
    
    def create_simulation_config(self) -> SimulationConfig:
        """Create simulation configuration for conveyor."""
        # Conveyors are typically part of larger station simulations
        # This could be a simplified conveyor-only simulation
        parameters = {
            'Belt_Speed': self.real_data.get('current_speed', 0.5),
            'Segment_Count': len(self.segments),
            'DUT_Buffer_Size': self.real_data.get('max_capacity', 10)
        }
        
        return create_simulation_config(
            config_id=f"conveyor_{self.component_id}",
            simulation_type="jaamsim",
            config_file="simulation/scenario_configs/conveyor_simulation.cfg",
            parameters=parameters
        )
    
    def update_real_data(self, data: Dict[str, Any]):
        """Update with real conveyor data."""
        self.real_data.update(data)
        
        # Track DUT movement
        if 'duts_on_conveyor' in data:
            self.dut_tracking_history.append({
                'timestamp': time.time(),
                'dut_count': len(data['duts_on_conveyor']),
                'average_speed': data.get('current_speed', 0)
            })
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate conveyor flow predictions."""
        # Simplified predictions based on current state
        predictions = {
            'predicted_throughput': 0,
            'predicted_bottleneck_segments': [],
            'predicted_dwell_time': 0
        }
        
        if self.real_data.get('current_speed'):
            speed = self.real_data['current_speed']
            segment_length = sum(seg.get('length', 0) for seg in self.segments)
            if segment_length > 0:
                predictions['predicted_dwell_time'] = segment_length / speed
        
        return predictions
    
    def calculate_sync_accuracy(self) -> float:
        """Calculate conveyor prediction accuracy."""
        # Simple accuracy based on throughput prediction
        real_throughput = self.real_data.get('actual_throughput', 0)
        pred_throughput = self.predicted_data.get('predicted_throughput', 0)
        
        if real_throughput > 0 and pred_throughput > 0:
            return 1.0 - abs(real_throughput - pred_throughput) / real_throughput
        
        return 0.0


class OperatorDigitalTwin(BaseDigitalTwin):
    """Digital twin for digital operators."""
    
    def __init__(self, operator_id: str):
        super().__init__(operator_id, ComponentType.OPERATOR)
        self.behavior_model = {}
        self.action_history = []
    
    def create_simulation_config(self) -> SimulationConfig:
        """Create simulation configuration for operator."""
        parameters = {
            'Skill_Level': self.behavior_model.get('skill_level', 0.9),
            'Attention_Level': self.behavior_model.get('attention_level', 0.95),
            'Reaction_Time': self.behavior_model.get('reaction_time', 2.0),
            'Error_Rate': self.behavior_model.get('error_rate', 0.05)
        }
        
        return create_simulation_config(
            config_id=f"operator_{self.component_id}",
            simulation_type="jaamsim",
            config_file="simulation/scenario_configs/operator_simulation.cfg",
            parameters=parameters
        )
    
    def update_real_data(self, data: Dict[str, Any]):
        """Update with real operator performance data."""
        self.real_data.update(data)
        
        # Track action history
        if 'last_action' in data:
            self.action_history.append({
                'timestamp': time.time(),
                'action': data['last_action'],
                'success': data.get('action_success', True),
                'duration': data.get('action_duration', 0)
            })
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate operator behavior predictions."""
        predictions = {
            'predicted_response_time': self.behavior_model.get('reaction_time', 2.0),
            'predicted_success_rate': 1.0 - self.behavior_model.get('error_rate', 0.05),
            'predicted_fatigue_level': self.real_data.get('fatigue_level', 0.0)
        }
        
        # Adjust predictions based on recent performance
        if self.action_history:
            recent_actions = self.action_history[-10:]  # Last 10 actions
            success_rate = sum(1 for a in recent_actions if a['success']) / len(recent_actions)
            predictions['predicted_success_rate'] = success_rate
        
        return predictions
    
    def calculate_sync_accuracy(self) -> float:
        """Calculate operator prediction accuracy."""
        if not self.action_history:
            return 0.0
        
        # Compare predicted vs actual response times
        real_avg_response = sum(a['duration'] for a in self.action_history[-10:]) / min(10, len(self.action_history))
        pred_response = self.predicted_data.get('predicted_response_time', 0)
        
        if real_avg_response > 0 and pred_response > 0:
            return 1.0 - abs(real_avg_response - pred_response) / pred_response
        
        return 0.0


class DigitalTwinManager:
    """Manages all digital twins in the manufacturing line."""
    
    def __init__(self):
        self.twins: Dict[str, BaseDigitalTwin] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_twin(self, twin: BaseDigitalTwin):
        """Register a digital twin."""
        self.twins[twin.component_id] = twin
        self.logger.info(f"Registered digital twin: {twin.component_id}")
    
    def unregister_twin(self, component_id: str):
        """Unregister a digital twin."""
        if component_id in self.twins:
            twin = self.twins[component_id]
            twin.stop_sync()
            del self.twins[component_id]
            self.logger.info(f"Unregistered digital twin: {component_id}")
    
    def get_twin(self, component_id: str) -> Optional[BaseDigitalTwin]:
        """Get digital twin by component ID."""
        return self.twins.get(component_id)
    
    def start_all_twins(self):
        """Start synchronization for all twins."""
        for twin in self.twins.values():
            twin.start_sync()
    
    def stop_all_twins(self):
        """Stop synchronization for all twins."""
        for twin in self.twins.values():
            twin.stop_sync()
    
    def get_line_predictions(self) -> Dict[str, Any]:
        """Get line-level predictions from all twins."""
        predictions = {
            'stations': {},
            'conveyors': {},
            'operators': {},
            'line_summary': {}
        }
        
        for twin_id, twin in self.twins.items():
            if twin.component_type == ComponentType.STATION:
                predictions['stations'][twin_id] = twin.predicted_data
            elif twin.component_type == ComponentType.CONVEYOR:
                predictions['conveyors'][twin_id] = twin.predicted_data
            elif twin.component_type == ComponentType.OPERATOR:
                predictions['operators'][twin_id] = twin.predicted_data
        
        # Calculate line-level summary
        station_uphs = [data.get('predicted_uph', 0) for data in predictions['stations'].values()]
        if station_uphs:
            predictions['line_summary']['predicted_line_uph'] = min(station_uphs)  # Bottleneck determines line UPH
            predictions['line_summary']['predicted_efficiency'] = sum(
                data.get('predicted_efficiency', 0) for data in predictions['stations'].values()
            ) / len(predictions['stations'])
        
        return predictions
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status for all twins."""
        return {
            twin_id: twin.get_status()
            for twin_id, twin in self.twins.items()
        }


# Global digital twin manager
digital_twin_manager = DigitalTwinManager()