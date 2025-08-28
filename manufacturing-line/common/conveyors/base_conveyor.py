"""Base conveyor class for manufacturing line integration."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import time
import logging


class ConveyorStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ConveyorType(Enum):
    BELT = "belt"
    ROLLER = "roller"
    CHAIN = "chain"


@dataclass
class DUT:
    """Device Under Test tracking on conveyor."""
    dut_id: str
    position: float  # Position on conveyor (0.0 to 1.0)
    destination_station: str
    source_station: str
    timestamp: float


@dataclass
class ConveyorSegment:
    """Conveyor segment between stations."""
    segment_id: str
    from_station: str
    to_station: str
    length: float  # meters
    max_speed: float  # m/s
    current_speed: float  # m/s


class BaseConveyor(ABC):
    """Abstract base class for all conveyor implementations."""
    
    def __init__(self, conveyor_id: str, config: Dict[str, Any]):
        self.conveyor_id = conveyor_id
        self.config = config
        self.type = ConveyorType(config.get('type', 'belt'))
        self.status = ConveyorStatus.IDLE
        self.segments: List[ConveyorSegment] = []
        self.duts_on_conveyor: Dict[str, DUT] = {}
        self.speed_multiplier = 1.0
        self.logger = logging.getLogger(f'Conveyor_{conveyor_id}')
        
        # Digital twin integration
        self.digital_twin = None
        self.simulation_enabled = config.get('simulation_enabled', True)
        self.data_callbacks: List[Callable] = []
        
        self._initialize_segments()
    
    def _initialize_segments(self):
        """Initialize conveyor segments from configuration."""
        for seg_config in self.config.get('segments', []):
            segment = ConveyorSegment(
                segment_id=seg_config['id'],
                from_station=seg_config['from_station'],
                to_station=seg_config['to_station'],
                length=seg_config.get('length', 2.0),
                max_speed=seg_config.get('max_speed', 0.5),
                current_speed=0.0
            )
            self.segments.append(segment)
    
    @abstractmethod
    def start(self) -> bool:
        """Start the conveyor system."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the conveyor system."""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> bool:
        """Emergency stop - immediate halt."""
        pass
    
    def load_dut(self, dut_id: str, source_station: str, destination_station: str) -> bool:
        """Load a DUT onto the conveyor."""
        if dut_id in self.duts_on_conveyor:
            return False
        
        dut = DUT(
            dut_id=dut_id,
            position=0.0,
            destination_station=destination_station,
            source_station=source_station,
            timestamp=time.time()
        )
        self.duts_on_conveyor[dut_id] = dut
        return True
    
    def unload_dut(self, dut_id: str) -> Optional[DUT]:
        """Remove DUT from conveyor when it reaches destination."""
        return self.duts_on_conveyor.pop(dut_id, None)
    
    def get_dut_position(self, dut_id: str) -> Optional[float]:
        """Get current position of DUT on conveyor."""
        dut = self.duts_on_conveyor.get(dut_id)
        return dut.position if dut else None
    
    def update_positions(self, delta_time: float):
        """Update all DUT positions based on elapsed time."""
        if self.status != ConveyorStatus.RUNNING:
            return
        
        for dut in self.duts_on_conveyor.values():
            # Simple linear movement model
            segment = self._get_segment_for_dut(dut)
            if segment:
                distance = segment.current_speed * delta_time
                progress = distance / segment.length
                dut.position = min(1.0, dut.position + progress)
        
        # Update digital twin with new positions
        self._update_digital_twin()
    
    def _get_segment_for_dut(self, dut: DUT) -> Optional[ConveyorSegment]:
        """Find which segment the DUT is currently on."""
        for segment in self.segments:
            if segment.to_station == dut.destination_station:
                return segment
        return None
    
    def set_speed(self, speed_multiplier: float):
        """Set conveyor speed (0.0 to 1.0 of max speed)."""
        self.speed_multiplier = max(0.0, min(1.0, speed_multiplier))
        for segment in self.segments:
            segment.current_speed = segment.max_speed * self.speed_multiplier
    
    def get_status(self) -> Dict[str, Any]:
        """Get current conveyor status."""
        return {
            'conveyor_id': self.conveyor_id,
            'type': self.type.value,
            'status': self.status.value,
            'speed_multiplier': self.speed_multiplier,
            'segments': [
                {
                    'id': seg.segment_id,
                    'from': seg.from_station,
                    'to': seg.to_station,
                    'speed': seg.current_speed
                }
                for seg in self.segments
            ],
            'duts': [
                {
                    'id': dut.dut_id,
                    'position': dut.position,
                    'destination': dut.destination_station
                }
                for dut in self.duts_on_conveyor.values()
            ]
        }
    
    def self_test(self) -> Dict[str, Any]:
        """Run self-diagnostic test."""
        results = {
            'conveyor_id': self.conveyor_id,
            'timestamp': time.time(),
            'tests': {}
        }
        
        # Test segments
        results['tests']['segments'] = len(self.segments) > 0
        
        # Test speed control
        original_speed = self.speed_multiplier
        self.set_speed(0.5)
        results['tests']['speed_control'] = abs(self.speed_multiplier - 0.5) < 0.01
        self.set_speed(original_speed)
        
        results['passed'] = all(results['tests'].values())
        return results
    
    # Digital twin and simulation integration
    def set_digital_twin(self, digital_twin):
        """Set digital twin for this conveyor."""
        self.digital_twin = digital_twin
        self.logger.info(f"Digital twin configured for conveyor {self.conveyor_id}")
    
    def add_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for real-time data updates."""
        self.data_callbacks.append(callback)
    
    def _update_digital_twin(self):
        """Update digital twin with current conveyor state."""
        if self.digital_twin and self.simulation_enabled:
            status_data = self.get_status()
            performance_data = {
                'actual_throughput': len([dut for dut in self.duts_on_conveyor.values() 
                                        if dut.position > 0.9]),
                'current_speed': self.speed_multiplier,
                'dut_count': len(self.duts_on_conveyor),
                'segment_utilization': {seg.segment_id: len([d for d in self.duts_on_conveyor.values() 
                                                           if self._dut_in_segment(d, seg)]) / 3.0 
                                      for seg in self.segments}
            }
            
            try:
                self.digital_twin.update_real_data(performance_data)
                
                # Trigger data callbacks
                for callback in self.data_callbacks:
                    callback(performance_data)
                    
            except Exception as e:
                self.logger.error(f"Digital twin update error: {e}")
    
    def _dut_in_segment(self, dut: DUT, segment: ConveyorSegment) -> bool:
        """Check if DUT is currently in a specific segment."""
        # Simplified check - in practice would need more sophisticated logic
        return segment.to_station == dut.destination_station
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get configuration for conveyor simulation."""
        return {
            'conveyor_id': self.conveyor_id,
            'type': self.type.value,
            'segments': [{
                'id': seg.segment_id,
                'length': seg.length,
                'max_speed': seg.max_speed,
                'from': seg.from_station,
                'to': seg.to_station
            } for seg in self.segments],
            'parameters': {
                'default_speed': self.speed_multiplier,
                'capacity': len(self.duts_on_conveyor),
                'transport_time': sum(seg.length / seg.max_speed for seg in self.segments)
            }
        }
    
    # Hook points for external system integration
    def register_hook(self, event: str, callback):
        """Register callback for conveyor events."""
        # Implementation for hook system
        pass
    
    def trigger_hook(self, event: str, data: Dict[str, Any]):
        """Trigger registered hooks with event data."""
        # Implementation for hook system
        pass