"""Belt conveyor implementation for point-to-point transport."""

from typing import Dict, Any
import time
import threading
from .base_conveyor import BaseConveyor, ConveyorStatus


class BeltConveyor(BaseConveyor):
    """Belt conveyor with station stops."""
    
    def __init__(self, conveyor_id: str, config: Dict[str, Any]):
        super().__init__(conveyor_id, config)
        self.motor_enabled = False
        self.station_stops = config.get('station_stops', [])
        self._update_thread = None
        self._running = False
    
    def start(self) -> bool:
        """Start the belt conveyor."""
        if self.status == ConveyorStatus.RUNNING:
            return False
        
        self.motor_enabled = True
        self.status = ConveyorStatus.RUNNING
        self.set_speed(self.config.get('default_speed', 0.5))
        
        # Start position update thread
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        self.trigger_hook('conveyor_started', {'conveyor_id': self.conveyor_id})
        return True
    
    def stop(self) -> bool:
        """Stop the belt conveyor gracefully."""
        if self.status != ConveyorStatus.RUNNING:
            return False
        
        # Gradual slowdown
        for speed in [0.7, 0.5, 0.3, 0.1, 0.0]:
            self.set_speed(speed)
            time.sleep(0.2)
        
        self.motor_enabled = False
        self.status = ConveyorStatus.STOPPED
        self._running = False
        
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
        
        self.trigger_hook('conveyor_stopped', {'conveyor_id': self.conveyor_id})
        return True
    
    def emergency_stop(self) -> bool:
        """Emergency stop - immediate halt."""
        self.set_speed(0.0)
        self.motor_enabled = False
        self.status = ConveyorStatus.ERROR
        self._running = False
        
        self.trigger_hook('emergency_stop', {
            'conveyor_id': self.conveyor_id,
            'timestamp': time.time()
        })
        return True
    
    def _update_loop(self):
        """Background thread to update DUT positions."""
        last_update = time.time()
        
        while self._running:
            current_time = time.time()
            delta_time = current_time - last_update
            
            # Update positions
            self.update_positions(delta_time)
            
            # Check for DUTs at destination
            duts_at_destination = []
            for dut_id, dut in self.duts_on_conveyor.items():
                if dut.position >= 0.95:  # Close enough to destination
                    duts_at_destination.append(dut_id)
                    self.trigger_hook('dut_arrived', {
                        'dut_id': dut_id,
                        'station': dut.destination_station
                    })
            
            # Stop at stations if configured
            for dut_id in duts_at_destination:
                if dut_id in self.duts_on_conveyor:
                    dut = self.duts_on_conveyor[dut_id]
                    if dut.destination_station in self.station_stops:
                        # Pause conveyor for station operation
                        self._pause_for_station(dut.destination_station)
            
            last_update = current_time
            time.sleep(0.1)  # Update rate: 10Hz
    
    def _pause_for_station(self, station: str):
        """Pause conveyor for station operation."""
        pause_duration = self.config.get('station_pause_time', 2.0)
        original_speed = self.speed_multiplier
        
        # Stop for station operation
        self.set_speed(0.0)
        self.trigger_hook('station_pause', {'station': station})
        
        time.sleep(pause_duration)
        
        # Resume
        self.set_speed(original_speed)
        self.trigger_hook('station_resume', {'station': station})
    
    def get_motor_status(self) -> Dict[str, Any]:
        """Get motor and belt status."""
        return {
            'motor_enabled': self.motor_enabled,
            'belt_speed': self.speed_multiplier,
            'belt_tension': 'normal',  # Placeholder for sensor data
            'temperature': 25.0,  # Placeholder for sensor data
            'vibration': 0.1  # Placeholder for sensor data
        }