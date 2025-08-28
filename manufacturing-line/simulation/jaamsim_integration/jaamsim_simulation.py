"""JAAMSIM discrete event simulation integration."""

import os
import re
import json
import time
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from ..simulation_engine.base_simulation import (
    BaseSimulation, SimulationConfig, SimulationResult, SimulationStatus
)


class JaamSimSimulation(BaseSimulation):
    """JAAMSIM discrete event simulation implementation."""
    
    def __init__(self, simulation_id: str, config: SimulationConfig):
        super().__init__(simulation_id, config)
        self.jaamsim_jar_path = self._find_jaamsim_jar()
        self.java_executable = "java"
        self.process: Optional[subprocess.Popen] = None
        self.output_file: Optional[str] = None
        self.log_file: Optional[str] = None
        self._progress = 0.0
        self._metrics_cache: Dict[str, Any] = {}
    
    def _find_jaamsim_jar(self) -> str:
        """Find JAAMSIM JAR file in the project."""
        possible_paths = [
            "stations/fixture/simulation/cfg/JaamSim2022-06.jar",
            "simulation/jaamsim_integration/JaamSim2022-06.jar",
            "JaamSim.jar",
            "JaamSim2022-06.jar"
        ]
        
        for path in possible_paths:
            full_path = Path(path)
            if full_path.exists():
                return str(full_path.absolute())
        
        # Look in common system locations
        system_paths = [
            "/usr/local/bin/JaamSim.jar",
            "/opt/JaamSim/JaamSim.jar"
        ]
        
        for path in system_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError("JAAMSIM JAR file not found")
    
    def initialize(self) -> bool:
        """Initialize JAAMSIM simulation environment."""
        try:
            # Validate configuration
            errors = self.validate_config()
            if errors:
                self.logger.error(f"Configuration errors: {errors}")
                return False
            
            # Check Java availability
            result = subprocess.run([self.java_executable, "-version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("Java not found or not working")
                return False
            
            # Check JAAMSIM JAR
            if not Path(self.jaamsim_jar_path).exists():
                self.logger.error(f"JAAMSIM JAR not found: {self.jaamsim_jar_path}")
                return False
            
            # Prepare output files
            timestamp = int(time.time())
            output_dir = Path(f"simulation_outputs/{self.simulation_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.output_file = str(output_dir / f"output_{timestamp}.txt")
            self.log_file = str(output_dir / f"log_{timestamp}.log")
            
            self.logger.info("JAAMSIM simulation initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            return False
    
    def execute(self) -> SimulationResult:
        """Execute JAAMSIM simulation."""
        self.status = SimulationStatus.RUNNING
        start_time = time.time()
        
        result = SimulationResult(
            simulation_id=self.simulation_id,
            status=SimulationStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Prepare configuration file with parameters
            config_file = self._prepare_config_file()
            
            # Build JAAMSIM command
            cmd = self._build_jaamsim_command(config_file)
            
            self.logger.info(f"Starting JAAMSIM: {' '.join(cmd)}")
            
            # Start JAAMSIM process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(config_file).parent
            )
            
            # Monitor execution
            self._monitor_execution(result)
            
            # Wait for completion
            stdout, stderr = self.process.communicate(timeout=self.config.max_runtime)
            
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            
            if self.process.returncode == 0:
                result.status = SimulationStatus.COMPLETED
                result.metrics = self._extract_metrics(stdout, stderr)
                result.predictions = self._generate_predictions(result.metrics)
                result.performance = self._calculate_performance(result.metrics)
            else:
                result.status = SimulationStatus.ERROR
                result.errors.append(f"JAAMSIM exited with code {self.process.returncode}")
                if stderr:
                    result.errors.append(stderr)
            
        except subprocess.TimeoutExpired:
            result.status = SimulationStatus.ERROR
            result.errors.append("Simulation timed out")
            if self.process:
                self.process.kill()
        
        except Exception as e:
            result.status = SimulationStatus.ERROR
            result.errors.append(str(e))
            self.logger.error(f"Execution error: {e}")
        
        finally:
            self.status = result.status
            self.current_result = result
        
        return result
    
    def pause(self) -> bool:
        """Pause the simulation (not directly supported by JAAMSIM)."""
        self.logger.warning("JAAMSIM does not support pause/resume")
        return False
    
    def resume(self) -> bool:
        """Resume the paused simulation (not directly supported by JAAMSIM)."""
        self.logger.warning("JAAMSIM does not support pause/resume")
        return False
    
    def stop(self) -> bool:
        """Stop the JAAMSIM simulation."""
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                # Wait a bit for graceful termination
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                
                self.status = SimulationStatus.IDLE
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error stopping simulation: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics."""
        return self._metrics_cache.copy()
    
    def get_progress(self) -> float:
        """Get simulation progress (0.0 to 1.0)."""
        return self._progress
    
    def _prepare_config_file(self) -> str:
        """Prepare JAAMSIM configuration file with parameters."""
        config_path = Path(self.config.config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Read original config
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Apply parameter substitutions
        for param_name, param_value in self.config.parameters.items():
            # Replace parameter values in config
            pattern = rf'({param_name}\s+Value\s*{{\s*)[^}}]*}}'
            replacement = rf'\\g<1>{param_value} }}'
            content = re.sub(pattern, replacement, content)
        
        # Create temporary config file
        temp_config_path = config_path.parent / f"temp_{self.simulation_id}_{config_path.name}"
        with open(temp_config_path, 'w') as f:
            f.write(content)
        
        return str(temp_config_path)
    
    def _build_jaamsim_command(self, config_file: str) -> List[str]:
        """Build JAAMSIM execution command."""
        cmd = [self.java_executable]
        
        # Add Java options for macOS compatibility
        if os.uname().sysname == "Darwin":  # macOS
            native_lib_path = Path(self.jaamsim_jar_path).parent / "natives/macosx-universal"
            if native_lib_path.exists():
                cmd.extend([
                    f"-Djava.library.path={native_lib_path}",
                    "-Djogamp.gluegen.UseTempJarCache=false"
                ])
        
        # Memory settings
        cmd.extend(["-Xms512m", "-Xmx2g"])
        
        # JAR and config file
        cmd.extend(["-jar", self.jaamsim_jar_path, config_file])
        
        # Add simulation-specific options
        if self.config.real_time_factor != 1.0:
            # Note: JAAMSIM real-time factor is configured in the .cfg file
            pass
        
        return cmd
    
    def _monitor_execution(self, result: SimulationResult):
        """Monitor JAAMSIM execution progress."""
        def monitor():
            while self.process and self.process.poll() is None:
                # Check if output file exists and has content
                if self.output_file and Path(self.output_file).exists():
                    try:
                        with open(self.output_file, 'r') as f:
                            content = f.read()
                            # Parse progress indicators from output
                            self._update_progress_from_output(content)
                    except Exception:
                        pass
                
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def _update_progress_from_output(self, output_content: str):
        """Update progress based on JAAMSIM output."""
        # Look for simulation time progress indicators
        time_pattern = r'Simulation time:\\s*(\\d+\\.\\d+)\\s*s'
        matches = re.findall(time_pattern, output_content)
        if matches:
            current_time = float(matches[-1])
            # Estimate progress based on expected simulation time
            expected_time = self.config.parameters.get('simulation_duration', 1000.0)
            self._progress = min(current_time / expected_time, 1.0)
    
    def _extract_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract metrics from JAAMSIM output."""
        metrics = {}
        
        # Extract common metrics patterns from JAAMSIM output
        patterns = {
            'total_processed': r'Total Processed DUT:\\s*(\\d+)',
            'total_time': r'Total Time:\\s*(\\d+\\.\\d+)\\s*s',
            'measurement_utilization': r'Measure Utilisation\\s*(\\d+\\.\\d+)\\s*%',
            'downtime_percentage': r'Downtime Percentage\\s*(\\d+\\.\\d+)\\s*%',
            'completed_load': r'Completed load\\s*(\\d+\\.\\d+)',
            'completed_aoi': r'Completed AOI\\s*:\\s*(\\d+\\.\\d+)',
            'completed_logging': r'Completed logging:\\s*(\\d+\\.\\d+)',
            'completed_unload': r'Completed unload:\\s*(\\d+\\.\\d+)'
        }
        
        combined_output = stdout + stderr
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, combined_output)
            if matches:
                try:
                    value = float(matches[-1])  # Take last occurrence
                    metrics[metric_name] = value
                except ValueError:
                    continue
        
        # Cache metrics
        self._metrics_cache = metrics
        
        return metrics
    
    def _generate_predictions(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions based on simulation results."""
        predictions = {}
        
        if 'total_time' in metrics and 'total_processed' in metrics:
            total_time = metrics['total_time']
            total_processed = metrics['total_processed']
            
            if total_time > 0:
                # Calculate throughput predictions
                uph = (total_processed / total_time) * 3600  # Units per hour
                predictions['predicted_uph'] = uph
                
                # Calculate cycle time
                cycle_time = total_time / total_processed if total_processed > 0 else 0
                predictions['predicted_cycle_time'] = cycle_time
                
                # Efficiency predictions
                if 'measurement_utilization' in metrics:
                    predictions['predicted_efficiency'] = metrics['measurement_utilization'] / 100
                
                # Bottleneck prediction (simplified)
                if 'downtime_percentage' in metrics:
                    downtime = metrics['downtime_percentage']
                    if downtime > 10:
                        predictions['bottleneck_risk'] = 'high'
                    elif downtime > 5:
                        predictions['bottleneck_risk'] = 'medium'
                    else:
                        predictions['bottleneck_risk'] = 'low'
        
        return predictions
    
    def _calculate_performance(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance indicators from metrics."""
        performance = {}
        
        if 'measurement_utilization' in metrics:
            performance['utilization'] = metrics['measurement_utilization'] / 100
        
        if 'downtime_percentage' in metrics:
            performance['availability'] = 1.0 - (metrics['downtime_percentage'] / 100)
        
        # Calculate OEE (Overall Equipment Effectiveness)
        if 'utilization' in performance and 'availability' in performance:
            # Simplified OEE calculation (missing quality factor)
            performance['oee'] = performance['utilization'] * performance['availability']
        
        return performance


def create_jaamsim_config(
    config_id: str,
    cfg_file_path: str,
    parameters: Optional[Dict[str, Any]] = None,
    real_time_factor: float = 16.0,
    max_runtime: float = 300.0
) -> SimulationConfig:
    """Helper function to create JAAMSIM simulation configuration."""
    from ..simulation_engine.base_simulation import SimulationType, SimulationConfig
    
    default_params = {
        'GoodDUT': 85,
        'RelitDUT': 10,
        'TotalDUT': 1000,
        'StationTime_Input': 9,
        'Input_Load': 10,
        'Input_Unload': 5,
        'PTBTime_Input': 5,
        'Input_PTB_Retry': 3
    }
    
    if parameters:
        default_params.update(parameters)
    
    return SimulationConfig(
        config_id=config_id,
        simulation_type=SimulationType.JAAMSIM,
        config_file=cfg_file_path,
        parameters=default_params,
        real_time_factor=real_time_factor,
        max_runtime=max_runtime,
        output_metrics=['total_processed', 'total_time', 'measurement_utilization', 
                       'downtime_percentage', 'predicted_uph']
    )