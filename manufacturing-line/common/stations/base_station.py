"""Base station class for manufacturing line."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import time
import logging

from ..interfaces.manufacturing_interface import ManufacturingComponent, ComponentStatus, ComponentState
from ..interfaces.communication_interface import CommunicationProtocol, Message, MessageType
from ..interfaces.data_interface import DataLogger, MetricsCollector


class StationState(Enum):
    """Specific states for manufacturing stations."""
    OFFLINE = "offline"
    INITIALIZING = "initializing" 
    IDLE = "idle"
    LOADING = "loading"
    PROCESSING = "processing"
    UNLOADING = "unloading"
    TESTING = "testing"
    CLEANING = "cleaning"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


@dataclass
class StationMetrics:
    """Metrics collected by manufacturing station."""
    cycle_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    total_cycle_time: float = 0.0
    average_cycle_time: float = 0.0
    yield_rate: float = 0.0
    uptime: float = 0.0
    downtime: float = 0.0
    utilization: float = 0.0
    throughput_uph: float = 0.0  # Units per hour
    last_cycle_time: float = 0.0
    
    def update_cycle_metrics(self, cycle_time: float, passed: bool):
        """Update metrics after a cycle completion."""
        self.cycle_count += 1
        self.last_cycle_time = cycle_time
        self.total_cycle_time += cycle_time
        
        if passed:
            self.pass_count += 1
        else:
            self.fail_count += 1
        
        # Calculate derived metrics
        if self.cycle_count > 0:
            self.average_cycle_time = self.total_cycle_time / self.cycle_count
            self.yield_rate = self.pass_count / self.cycle_count
            if self.average_cycle_time > 0:
                self.throughput_uph = 3600.0 / self.average_cycle_time


class BaseStation(ManufacturingComponent):
    """Abstract base class for all manufacturing stations."""
    
    def __init__(self, station_id: str, station_type: str, 
                 position: int = 0, capacity: int = 1):
        super().__init__(station_id, station_type)
        
        # Station-specific attributes
        self.position = position  # Position in line
        self.capacity = capacity  # Number of DUTs it can handle simultaneously
        self.current_duts: List[str] = []
        self.station_state = StationState.OFFLINE
        
        # Metrics and performance tracking
        self.metrics = StationMetrics()
        self.cycle_start_time: Optional[float] = None
        
        # Components
        self.logger = logging.getLogger(f'Station_{station_id}')
        self.communication: Optional[CommunicationProtocol] = None
        self.data_logger: Optional[DataLogger] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Station configuration
        self.config = {
            'cycle_time_target': 30.0,  # Target cycle time in seconds
            'yield_target': 0.95,       # Target yield rate
            'auto_mode': True,          # Automatic operation mode
            'max_retry_attempts': 3     # Max retry attempts on failure
        }
    
    def initialize(self) -> bool:
        """Initialize the station."""
        try:
            self.station_state = StationState.INITIALIZING
            self.logger.info(f"Initializing station {self.component_id}")
            
            # Initialize station-specific components
            init_success = self._initialize_station_specific()
            
            if init_success:
                self.station_state = StationState.IDLE
                self.set_state(ComponentState.IDLE)
                self.logger.info(f"Station {self.component_id} initialized successfully")
                
                if self.data_logger:
                    self.data_logger.info("Station initialized", {
                        'station_type': self.component_type,
                        'position': self.position,
                        'capacity': self.capacity
                    })
                
                return True
            else:
                self.station_state = StationState.ERROR
                self.set_state(ComponentState.ERROR)
                self.logger.error(f"Failed to initialize station {self.component_id}")
                return False
                
        except Exception as e:
            self.station_state = StationState.ERROR
            self.set_state(ComponentState.ERROR)
            self.add_error(f"Initialization error: {str(e)}")
            self.logger.error(f"Exception during initialization: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the station safely."""
        try:
            self.logger.info(f"Shutting down station {self.component_id}")
            self.station_state = StationState.SHUTDOWN
            
            # Complete any ongoing operations
            self._complete_current_operations()
            
            # Shutdown station-specific components
            shutdown_success = self._shutdown_station_specific()
            
            if shutdown_success:
                self.set_state(ComponentState.SHUTDOWN)
                self.logger.info(f"Station {self.component_id} shutdown successfully")
                return True
            else:
                self.logger.error(f"Failed to shutdown station {self.component_id}")
                return False
                
        except Exception as e:
            self.add_error(f"Shutdown error: {str(e)}")
            self.logger.error(f"Exception during shutdown: {e}")
            return False
    
    def reset(self) -> bool:
        """Reset station to initial state."""
        try:
            self.logger.info(f"Resetting station {self.component_id}")
            
            # Clear current operations
            self.current_duts.clear()
            self.cycle_start_time = None
            
            # Reset station-specific components
            reset_success = self._reset_station_specific()
            
            if reset_success:
                self.station_state = StationState.IDLE
                self.set_state(ComponentState.IDLE)
                self.clear_errors()
                self.logger.info(f"Station {self.component_id} reset successfully")
                return True
            else:
                self.logger.error(f"Failed to reset station {self.component_id}")
                return False
                
        except Exception as e:
            self.add_error(f"Reset error: {str(e)}")
            self.logger.error(f"Exception during reset: {e}")
            return False
    
    def get_status(self) -> ComponentStatus:
        """Get current station status."""
        return ComponentStatus(
            component_id=self.component_id,
            component_type=self.component_type,
            state=self.state,
            timestamp=time.time(),
            metrics={
                'station_state': self.station_state.value,
                'position': self.position,
                'capacity': self.capacity,
                'current_duts': len(self.current_duts),
                'cycle_count': self.metrics.cycle_count,
                'yield_rate': self.metrics.yield_rate,
                'average_cycle_time': self.metrics.average_cycle_time,
                'throughput_uph': self.metrics.throughput_uph,
                'utilization': self.metrics.utilization
            },
            errors=[error['message'] for error in self.errors[-5:]]  # Last 5 errors
        )
    
    def handle_command(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle external commands."""
        if parameters is None:
            parameters = {}
        
        self.logger.info(f"Handling command: {command} with parameters: {parameters}")
        
        try:
            if command == "start_cycle":
                return self._handle_start_cycle(parameters)
            elif command == "stop_cycle":
                return self._handle_stop_cycle(parameters)
            elif command == "load_dut":
                return self._handle_load_dut(parameters)
            elif command == "unload_dut":
                return self._handle_unload_dut(parameters)
            elif command == "run_test":
                return self._handle_run_test(parameters)
            elif command == "clean_station":
                return self._handle_clean_station(parameters)
            elif command == "update_config":
                return self._handle_update_config(parameters)
            else:
                # Delegate to station-specific command handler
                return self._handle_station_specific_command(command, parameters)
        
        except Exception as e:
            error_msg = f"Command handling error: {str(e)}"
            self.add_error(error_msg)
            return {'success': False, 'error': error_msg}
    
    # Abstract methods for station-specific implementation
    @abstractmethod
    def _initialize_station_specific(self) -> bool:
        """Initialize station-specific components."""
        pass
    
    @abstractmethod
    def _shutdown_station_specific(self) -> bool:
        """Shutdown station-specific components."""
        pass
    
    @abstractmethod
    def _reset_station_specific(self) -> bool:
        """Reset station-specific components."""
        pass
    
    @abstractmethod
    def _handle_station_specific_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle station-specific commands."""
        pass
    
    @abstractmethod
    def _process_dut(self, dut_id: str) -> bool:
        """Process a DUT through this station."""
        pass
    
    # Station operation methods
    def _handle_start_cycle(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start cycle command."""
        if self.station_state != StationState.IDLE:
            return {'success': False, 'error': f'Station not in IDLE state: {self.station_state}'}
        
        self.cycle_start_time = time.time()
        self.station_state = StationState.PROCESSING
        self.set_state(ComponentState.BUSY)
        
        if self.metrics_collector:
            self.metrics_collector.counter('cycles_started')
        
        return {'success': True, 'message': 'Cycle started'}
    
    def _handle_stop_cycle(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stop cycle command."""
        if self.station_state != StationState.PROCESSING:
            return {'success': False, 'error': f'No active cycle to stop'}
        
        cycle_time = time.time() - (self.cycle_start_time or time.time())
        passed = parameters.get('passed', True)
        
        self.metrics.update_cycle_metrics(cycle_time, passed)
        self.station_state = StationState.IDLE
        self.set_state(ComponentState.IDLE)
        self.cycle_start_time = None
        
        if self.metrics_collector:
            self.metrics_collector.timer('cycle_time', cycle_time)
            self.metrics_collector.counter('cycles_completed')
            if passed:
                self.metrics_collector.counter('cycles_passed')
            else:
                self.metrics_collector.counter('cycles_failed')
        
        return {'success': True, 'cycle_time': cycle_time, 'passed': passed}
    
    def _handle_load_dut(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle load DUT command."""
        dut_id = parameters.get('dut_id')
        if not dut_id:
            return {'success': False, 'error': 'No DUT ID provided'}
        
        if len(self.current_duts) >= self.capacity:
            return {'success': False, 'error': 'Station at capacity'}
        
        self.current_duts.append(dut_id)
        self.station_state = StationState.LOADING
        
        if self.data_logger:
            self.data_logger.info(f"DUT loaded: {dut_id}")
        
        return {'success': True, 'dut_id': dut_id, 'current_count': len(self.current_duts)}
    
    def _handle_unload_dut(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unload DUT command."""
        dut_id = parameters.get('dut_id')
        
        if dut_id and dut_id in self.current_duts:
            self.current_duts.remove(dut_id)
        elif self.current_duts:
            dut_id = self.current_duts.pop(0)  # Remove first DUT
        else:
            return {'success': False, 'error': 'No DUT to unload'}
        
        self.station_state = StationState.UNLOADING
        
        if self.data_logger:
            self.data_logger.info(f"DUT unloaded: {dut_id}")
        
        return {'success': True, 'dut_id': dut_id, 'current_count': len(self.current_duts)}
    
    def _handle_run_test(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle run test command."""
        if not self.current_duts:
            return {'success': False, 'error': 'No DUT loaded for testing'}
        
        dut_id = parameters.get('dut_id', self.current_duts[0])
        test_type = parameters.get('test_type', 'default')
        
        self.station_state = StationState.TESTING
        
        # This would be implemented by specific station types
        test_result = self._run_station_test(dut_id, test_type)
        
        if self.metrics_collector:
            self.metrics_collector.counter(f'tests_{test_type}')
        
        return {'success': True, 'dut_id': dut_id, 'test_result': test_result}
    
    def _handle_clean_station(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clean station command."""
        self.station_state = StationState.CLEANING
        
        # Station-specific cleaning implementation
        clean_success = self._perform_station_cleaning()
        
        if clean_success:
            self.station_state = StationState.IDLE
            return {'success': True, 'message': 'Station cleaned successfully'}
        else:
            self.station_state = StationState.ERROR
            return {'success': False, 'error': 'Cleaning failed'}
    
    def _handle_update_config(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update configuration command."""
        try:
            for key, value in parameters.items():
                if key in self.config:
                    old_value = self.config[key]
                    self.config[key] = value
                    self.logger.info(f"Config updated: {key} = {value} (was {old_value})")
            
            return {'success': True, 'config': self.config.copy()}
        
        except Exception as e:
            return {'success': False, 'error': f'Config update failed: {str(e)}'}
    
    # Helper methods
    def _complete_current_operations(self):
        """Complete any ongoing operations before shutdown."""
        if self.station_state == StationState.PROCESSING:
            # Allow current cycle to complete
            if self.cycle_start_time:
                elapsed = time.time() - self.cycle_start_time
                timeout = self.config.get('cycle_time_target', 30.0) * 2  # 2x target time
                
                if elapsed < timeout:
                    self.logger.info("Waiting for current cycle to complete")
                    # In real implementation, would wait or force completion
    
    def _run_station_test(self, dut_id: str, test_type: str) -> Dict[str, Any]:
        """Run station-specific test (to be overridden)."""
        # Default implementation - always pass
        return {
            'result': 'PASS',
            'test_type': test_type,
            'duration': 1.0,
            'measurements': {}
        }
    
    def _perform_station_cleaning(self) -> bool:
        """Perform station cleaning (to be overridden)."""
        # Default implementation
        return True
    
    # Utility methods
    def is_available(self) -> bool:
        """Check if station is available for new DUT."""
        return (self.station_state == StationState.IDLE and 
                len(self.current_duts) < self.capacity)
    
    def get_utilization(self) -> float:
        """Get current station utilization percentage."""
        if self.capacity == 0:
            return 0.0
        return (len(self.current_duts) / self.capacity) * 100.0
    
    def set_communication(self, communication: CommunicationProtocol):
        """Set communication protocol."""
        self.communication = communication
    
    def set_data_logger(self, data_logger: DataLogger):
        """Set data logger."""
        self.data_logger = data_logger
    
    def set_metrics_collector(self, metrics_collector: MetricsCollector):
        """Set metrics collector."""
        self.metrics_collector = metrics_collector