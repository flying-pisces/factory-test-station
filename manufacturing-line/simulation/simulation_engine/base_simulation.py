"""Base simulation framework for manufacturing line digital twins."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging
import subprocess
import threading
from pathlib import Path


class SimulationStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class SimulationType(Enum):
    JAAMSIM = "jaamsim"
    ISAAC_SIM = "isaac_sim"
    CUSTOM = "custom"


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    simulation_id: str
    status: SimulationStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    config_id: str
    simulation_type: SimulationType
    config_file: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    real_time_factor: float = 1.0
    max_runtime: float = 3600.0  # 1 hour default
    output_metrics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_id': self.config_id,
            'simulation_type': self.simulation_type.value,
            'config_file': self.config_file,
            'parameters': self.parameters,
            'real_time_factor': self.real_time_factor,
            'max_runtime': self.max_runtime,
            'output_metrics': self.output_metrics
        }


class BaseSimulation(ABC):
    """Abstract base class for all simulation implementations."""
    
    def __init__(self, simulation_id: str, config: SimulationConfig):
        self.simulation_id = simulation_id
        self.config = config
        self.status = SimulationStatus.IDLE
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{simulation_id}")
        self.callbacks: List[Callable[[SimulationResult], None]] = []
        self.current_result: Optional[SimulationResult] = None
        self._execution_thread: Optional[threading.Thread] = None
        self._should_stop = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the simulation environment."""
        pass
    
    @abstractmethod
    def execute(self) -> SimulationResult:
        """Execute the simulation and return results."""
        pass
    
    @abstractmethod
    def pause(self) -> bool:
        """Pause the simulation."""
        pass
    
    @abstractmethod
    def resume(self) -> bool:
        """Resume the paused simulation."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the simulation."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics."""
        pass
    
    def add_callback(self, callback: Callable[[SimulationResult], None]):
        """Add callback to be called when simulation completes."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[SimulationResult], None]):
        """Remove callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def run_async(self) -> threading.Thread:
        """Run simulation asynchronously."""
        if self._execution_thread and self._execution_thread.is_alive():
            raise RuntimeError("Simulation is already running")
        
        self._should_stop = False
        self._execution_thread = threading.Thread(target=self._run_with_callbacks)
        self._execution_thread.start()
        return self._execution_thread
    
    def _run_with_callbacks(self):
        """Internal method to run simulation with callbacks."""
        try:
            if not self.initialize():
                self.logger.error("Simulation initialization failed")
                return
            
            self.current_result = self.execute()
            
            # Trigger callbacks
            for callback in self.callbacks:
                try:
                    callback(self.current_result)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
        
        except Exception as e:
            self.logger.error(f"Simulation execution error: {e}")
            if self.current_result:
                self.current_result.status = SimulationStatus.ERROR
                self.current_result.errors.append(str(e))
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Optional[SimulationResult]:
        """Wait for simulation to complete and return results."""
        if self._execution_thread:
            self._execution_thread.join(timeout)
            if not self._execution_thread.is_alive():
                return self.current_result
        return None
    
    def is_running(self) -> bool:
        """Check if simulation is currently running."""
        return self.status == SimulationStatus.RUNNING
    
    def get_progress(self) -> float:
        """Get simulation progress (0.0 to 1.0)."""
        return 0.0  # Override in subclasses
    
    def update_config(self, new_config: SimulationConfig):
        """Update simulation configuration."""
        if self.is_running():
            raise RuntimeError("Cannot update configuration while simulation is running")
        self.config = new_config
    
    def validate_config(self) -> List[str]:
        """Validate simulation configuration. Return list of errors."""
        errors = []
        
        if not self.config.config_file:
            errors.append("Configuration file not specified")
        elif not Path(self.config.config_file).exists():
            errors.append(f"Configuration file not found: {self.config.config_file}")
        
        if self.config.real_time_factor <= 0:
            errors.append("Real-time factor must be positive")
        
        if self.config.max_runtime <= 0:
            errors.append("Max runtime must be positive")
        
        return errors


class SimulationManager:
    """Manages multiple simulation instances."""
    
    def __init__(self):
        self.simulations: Dict[str, BaseSimulation] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.default_configs: Dict[str, SimulationConfig] = {}
    
    def register_simulation(self, simulation: BaseSimulation):
        """Register a simulation instance."""
        self.simulations[simulation.simulation_id] = simulation
        self.logger.info(f"Registered simulation: {simulation.simulation_id}")
    
    def unregister_simulation(self, simulation_id: str):
        """Unregister a simulation instance."""
        if simulation_id in self.simulations:
            simulation = self.simulations[simulation_id]
            if simulation.is_running():
                simulation.stop()
            del self.simulations[simulation_id]
            self.logger.info(f"Unregistered simulation: {simulation_id}")
    
    def get_simulation(self, simulation_id: str) -> Optional[BaseSimulation]:
        """Get simulation by ID."""
        return self.simulations.get(simulation_id)
    
    def list_simulations(self) -> List[str]:
        """List all registered simulation IDs."""
        return list(self.simulations.keys())
    
    def get_running_simulations(self) -> List[BaseSimulation]:
        """Get all currently running simulations."""
        return [sim for sim in self.simulations.values() if sim.is_running()]
    
    def stop_all_simulations(self):
        """Stop all running simulations."""
        for simulation in self.get_running_simulations():
            simulation.stop()
    
    def run_scenario(self, scenario_name: str, config: SimulationConfig) -> str:
        """Run a simulation scenario and return simulation ID."""
        simulation_id = f"{scenario_name}_{int(time.time())}"
        
        # Create appropriate simulation instance based on type
        if config.simulation_type == SimulationType.JAAMSIM:
            from .jaamsim_simulation import JaamSimSimulation
            simulation = JaamSimSimulation(simulation_id, config)
        else:
            raise ValueError(f"Unsupported simulation type: {config.simulation_type}")
        
        self.register_simulation(simulation)
        simulation.run_async()
        
        return simulation_id
    
    def load_default_configs(self, config_dir: str):
        """Load default simulation configurations from directory."""
        config_path = Path(config_dir)
        if not config_path.exists():
            self.logger.warning(f"Config directory not found: {config_dir}")
            return
        
        for config_file in config_path.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                config = SimulationConfig(
                    config_id=config_data['config_id'],
                    simulation_type=SimulationType(config_data['simulation_type']),
                    config_file=config_data['config_file'],
                    parameters=config_data.get('parameters', {}),
                    real_time_factor=config_data.get('real_time_factor', 1.0),
                    max_runtime=config_data.get('max_runtime', 3600.0),
                    output_metrics=config_data.get('output_metrics', [])
                )
                
                self.default_configs[config.config_id] = config
                self.logger.info(f"Loaded config: {config.config_id}")
                
            except Exception as e:
                self.logger.error(f"Error loading config {config_file}: {e}")
    
    def get_default_config(self, config_id: str) -> Optional[SimulationConfig]:
        """Get default configuration by ID."""
        return self.default_configs.get(config_id)
    
    def list_default_configs(self) -> List[str]:
        """List all default configuration IDs."""
        return list(self.default_configs.keys())


# Global simulation manager instance
simulation_manager = SimulationManager()


def create_simulation_config(
    config_id: str,
    simulation_type: str,
    config_file: str,
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> SimulationConfig:
    """Helper function to create simulation configuration."""
    return SimulationConfig(
        config_id=config_id,
        simulation_type=SimulationType(simulation_type),
        config_file=config_file,
        parameters=parameters or {},
        **kwargs
    )


def run_simulation(
    scenario_name: str,
    config: SimulationConfig,
    callback: Optional[Callable[[SimulationResult], None]] = None
) -> str:
    """Convenience function to run a simulation."""
    simulation_id = simulation_manager.run_scenario(scenario_name, config)
    
    if callback:
        simulation = simulation_manager.get_simulation(simulation_id)
        if simulation:
            simulation.add_callback(callback)
    
    return simulation_id