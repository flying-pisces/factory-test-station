"""StationCoordinator - Multi-Station Communication and Synchronization."""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

from layers.station_layer.station_layer_engine import StationConfig


class CoordinationMessage(Enum):
    """Types of coordination messages between stations."""
    START_PRODUCTION = "start_production"
    STOP_PRODUCTION = "stop_production"
    PAUSE_PRODUCTION = "pause_production"
    UNIT_READY = "unit_ready"
    UNIT_RECEIVED = "unit_received"
    QUALITY_GATE = "quality_gate"
    MAINTENANCE_REQUEST = "maintenance_request"
    STATUS_UPDATE = "status_update"
    ERROR_ALERT = "error_alert"


class StationState(Enum):
    """Current state of a station in the line."""
    IDLE = "idle"
    RUNNING = "running"
    BLOCKED = "blocked"
    STARVED = "starved"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class Message:
    """Message structure for inter-station communication."""
    message_type: CoordinationMessage
    sender_station_id: str
    receiver_station_id: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=high, 2=medium, 3=low
    message_id: str = ""
    
    def __post_init__(self):
        """Generate message ID if not provided."""
        if not self.message_id:
            self.message_id = f"{self.sender_station_id}_{int(self.timestamp*1000)}"


@dataclass
class StationStatus:
    """Current status of a station."""
    station_id: str
    state: StationState
    current_units: int
    buffer_capacity: int
    cycle_time_s: float
    last_update: float
    error_message: str = ""
    maintenance_due: bool = False
    quality_issues: int = 0


class StationCoordinator:
    """Coordinates communication and synchronization between multiple stations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize StationCoordinator."""
        self.logger = logging.getLogger('StationCoordinator')
        self.config = config or {}
        
        # Communication parameters
        self.max_latency_ms = self.config.get('max_latency_ms', 10)  # Week 3 target
        self.message_timeout_s = self.config.get('message_timeout_s', 5.0)
        self.heartbeat_interval_s = self.config.get('heartbeat_interval_s', 1.0)
        
        # Station management
        self.stations: Dict[str, StationConfig] = {}
        self.station_status: Dict[str, StationStatus] = {}
        self.station_handlers: Dict[str, Callable] = {}
        
        # Message queues
        self.message_queues: Dict[str, queue.Queue] = {}
        self.message_history: List[Message] = []
        
        # Coordination state
        self.line_running = False
        self.coordination_active = False
        self.synchronization_points: Dict[str, float] = {}
        
        # Performance tracking
        self.message_latencies: List[float] = []
        
        # Threading
        self.coordination_thread = None
        self.stop_event = threading.Event()
        
        self.logger.info("StationCoordinator initialized")
    
    def register_station(self, station_config: StationConfig, 
                        message_handler: Optional[Callable] = None) -> bool:
        """Register a station for coordination."""
        try:
            station_id = station_config.station_id
            
            # Register station configuration
            self.stations[station_id] = station_config
            
            # Initialize station status
            self.station_status[station_id] = StationStatus(
                station_id=station_id,
                state=StationState.IDLE,
                current_units=0,
                buffer_capacity=10,  # Default buffer size
                cycle_time_s=station_config.cycle_time_s,
                last_update=time.time()
            )
            
            # Set up message queue
            self.message_queues[station_id] = queue.Queue()
            
            # Register message handler
            if message_handler:
                self.station_handlers[station_id] = message_handler
            
            self.logger.info(f"Station {station_id} registered for coordination")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register station {station_config.station_id}: {e}")
            return False
    
    def send_message(self, message: Message) -> bool:
        """Send a message between stations."""
        start_time = time.time()
        
        try:
            # Validate sender and receiver
            if message.sender_station_id not in self.stations:
                self.logger.warning(f"Unknown sender station: {message.sender_station_id}")
                return False
            
            if message.receiver_station_id not in self.stations and message.receiver_station_id != "broadcast":
                self.logger.warning(f"Unknown receiver station: {message.receiver_station_id}")
                return False
            
            # Handle broadcast messages
            if message.receiver_station_id == "broadcast":
                success = True
                for station_id in self.stations.keys():
                    if station_id != message.sender_station_id:
                        broadcast_msg = Message(
                            message_type=message.message_type,
                            sender_station_id=message.sender_station_id,
                            receiver_station_id=station_id,
                            timestamp=message.timestamp,
                            payload=message.payload.copy(),
                            priority=message.priority
                        )
                        success &= self._deliver_message(broadcast_msg)
                return success
            else:
                return self._deliver_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
        finally:
            # Track message latency
            latency_ms = (time.time() - start_time) * 1000
            self.message_latencies.append(latency_ms)
            self.message_latencies = self.message_latencies[-100:]  # Keep last 100
    
    def _deliver_message(self, message: Message) -> bool:
        """Deliver message to target station."""
        try:
            receiver_queue = self.message_queues.get(message.receiver_station_id)
            if receiver_queue is None:
                return False
            
            # Add to queue with timeout to prevent blocking
            receiver_queue.put(message, timeout=0.1)
            
            # Store in message history
            self.message_history.append(message)
            self.message_history = self.message_history[-1000:]  # Keep last 1000 messages
            
            # Call message handler if registered
            handler = self.station_handlers.get(message.receiver_station_id)
            if handler:
                try:
                    handler(message)
                except Exception as e:
                    self.logger.warning(f"Message handler for {message.receiver_station_id} failed: {e}")
            
            return True
            
        except queue.Full:
            self.logger.warning(f"Message queue full for station {message.receiver_station_id}")
            return False
        except Exception as e:
            self.logger.error(f"Message delivery failed: {e}")
            return False
    
    def get_messages(self, station_id: str, max_messages: int = 10) -> List[Message]:
        """Get pending messages for a station."""
        messages = []
        if station_id not in self.message_queues:
            return messages
        
        station_queue = self.message_queues[station_id]
        
        try:
            for _ in range(max_messages):
                try:
                    message = station_queue.get_nowait()
                    messages.append(message)
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f"Failed to get messages for {station_id}: {e}")
        
        return messages
    
    def update_station_status(self, station_id: str, status_update: Dict[str, Any]) -> bool:
        """Update status of a station."""
        if station_id not in self.station_status:
            return False
        
        try:
            status = self.station_status[station_id]
            
            # Update fields if provided
            if 'state' in status_update:
                status.state = StationState(status_update['state'])
            if 'current_units' in status_update:
                status.current_units = status_update['current_units']
            if 'error_message' in status_update:
                status.error_message = status_update['error_message']
            if 'maintenance_due' in status_update:
                status.maintenance_due = status_update['maintenance_due']
            if 'quality_issues' in status_update:
                status.quality_issues = status_update['quality_issues']
            
            status.last_update = time.time()
            
            # Send status update to other stations if significant change
            if 'state' in status_update:
                self._broadcast_status_change(station_id, status.state)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update status for {station_id}: {e}")
            return False
    
    def _broadcast_status_change(self, station_id: str, new_state: StationState) -> None:
        """Broadcast status change to other stations."""
        message = Message(
            message_type=CoordinationMessage.STATUS_UPDATE,
            sender_station_id=station_id,
            receiver_station_id="broadcast",
            timestamp=time.time(),
            payload={
                'new_state': new_state.value,
                'station_id': station_id
            },
            priority=2  # Medium priority
        )
        self.send_message(message)
    
    def synchronize_stations(self, synchronization_point: str) -> bool:
        """Synchronize multiple stations at a specific point."""
        try:
            self.synchronization_points[synchronization_point] = time.time()
            
            # Send synchronization message to all stations
            sync_message = Message(
                message_type=CoordinationMessage.START_PRODUCTION,
                sender_station_id="coordinator",
                receiver_station_id="broadcast",
                timestamp=time.time(),
                payload={
                    'sync_point': synchronization_point,
                    'action': 'synchronize'
                },
                priority=1  # High priority
            )
            
            return self.send_message(sync_message)
            
        except Exception as e:
            self.logger.error(f"Synchronization failed at {synchronization_point}: {e}")
            return False
    
    def start_line_coordination(self) -> bool:
        """Start coordinated line operation."""
        if self.coordination_active:
            self.logger.warning("Coordination already active")
            return False
        
        try:
            self.line_running = True
            self.coordination_active = True
            self.stop_event.clear()
            
            # Start coordination thread
            self.coordination_thread = threading.Thread(target=self._coordination_loop)
            self.coordination_thread.start()
            
            # Send start production to all stations
            start_message = Message(
                message_type=CoordinationMessage.START_PRODUCTION,
                sender_station_id="coordinator",
                receiver_station_id="broadcast",
                timestamp=time.time(),
                payload={'action': 'start_line'},
                priority=1
            )
            
            self.send_message(start_message)
            
            self.logger.info("Line coordination started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start line coordination: {e}")
            return False
    
    def stop_line_coordination(self) -> bool:
        """Stop coordinated line operation."""
        if not self.coordination_active:
            return False
        
        try:
            # Send stop production to all stations
            stop_message = Message(
                message_type=CoordinationMessage.STOP_PRODUCTION,
                sender_station_id="coordinator",
                receiver_station_id="broadcast",
                timestamp=time.time(),
                payload={'action': 'stop_line'},
                priority=1
            )
            
            self.send_message(stop_message)
            
            # Stop coordination thread
            self.stop_event.set()
            self.coordination_active = False
            self.line_running = False
            
            if self.coordination_thread and self.coordination_thread.is_alive():
                self.coordination_thread.join(timeout=2.0)
            
            self.logger.info("Line coordination stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop line coordination: {e}")
            return False
    
    def _coordination_loop(self) -> None:
        """Main coordination loop running in separate thread."""
        self.logger.info("Coordination loop started")
        
        while not self.stop_event.is_set():
            try:
                # Monitor station health
                self._monitor_station_health()
                
                # Handle line balancing
                self._balance_line_flow()
                
                # Process coordination logic
                self._process_coordination_logic()
                
                # Wait for next iteration
                self.stop_event.wait(self.heartbeat_interval_s)
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                time.sleep(0.1)  # Brief pause to prevent tight error loop
        
        self.logger.info("Coordination loop stopped")
    
    def _monitor_station_health(self) -> None:
        """Monitor health of all stations."""
        current_time = time.time()
        
        for station_id, status in self.station_status.items():
            # Check for stale status updates
            if current_time - status.last_update > 5.0:  # 5 second timeout
                self.logger.warning(f"Station {station_id} status is stale")
                status.state = StationState.ERROR
                status.error_message = "Communication timeout"
            
            # Check for blocked stations
            if status.state == StationState.BLOCKED and current_time - status.last_update > 2.0:
                self.logger.warning(f"Station {station_id} blocked for extended period")
                # Could trigger unblocking logic here
    
    def _balance_line_flow(self) -> None:
        """Balance flow between stations."""
        # Simple line balancing logic
        for station_id, status in self.station_status.items():
            # Check if station is starved
            if status.state == StationState.STARVED and status.current_units == 0:
                # Look for upstream stations with excess units
                self._request_units_from_upstream(station_id)
            
            # Check if station is blocked
            elif status.state == StationState.BLOCKED and status.current_units >= status.buffer_capacity:
                # Look for downstream stations with capacity
                self._push_units_to_downstream(station_id)
    
    def _request_units_from_upstream(self, station_id: str) -> None:
        """Request units from upstream stations."""
        # Simple implementation - in reality would be more sophisticated
        message = Message(
            message_type=CoordinationMessage.UNIT_READY,
            sender_station_id=station_id,
            receiver_station_id="broadcast",  # Would target specific upstream stations
            timestamp=time.time(),
            payload={'requesting_units': True},
            priority=2
        )
        self.send_message(message)
    
    def _push_units_to_downstream(self, station_id: str) -> None:
        """Push units to downstream stations."""
        # Simple implementation
        message = Message(
            message_type=CoordinationMessage.UNIT_READY,
            sender_station_id=station_id,
            receiver_station_id="broadcast",  # Would target specific downstream stations
            timestamp=time.time(),
            payload={'units_available': True},
            priority=2
        )
        self.send_message(message)
    
    def _process_coordination_logic(self) -> None:
        """Process general coordination logic."""
        # Quality gate coordination
        self._coordinate_quality_gates()
        
        # Maintenance coordination
        self._coordinate_maintenance()
        
        # Performance optimization
        self._optimize_line_performance()
    
    def _coordinate_quality_gates(self) -> None:
        """Coordinate quality gates across the line."""
        for station_id, status in self.station_status.items():
            if status.quality_issues > 0:
                # Trigger quality gate message
                message = Message(
                    message_type=CoordinationMessage.QUALITY_GATE,
                    sender_station_id="coordinator",
                    receiver_station_id=station_id,
                    timestamp=time.time(),
                    payload={'quality_issues': status.quality_issues},
                    priority=1
                )
                self.send_message(message)
    
    def _coordinate_maintenance(self) -> None:
        """Coordinate maintenance activities."""
        for station_id, status in self.station_status.items():
            if status.maintenance_due:
                # Schedule maintenance when line allows
                message = Message(
                    message_type=CoordinationMessage.MAINTENANCE_REQUEST,
                    sender_station_id="coordinator",
                    receiver_station_id=station_id,
                    timestamp=time.time(),
                    payload={'schedule_maintenance': True},
                    priority=2
                )
                self.send_message(message)
    
    def _optimize_line_performance(self) -> None:
        """Optimize line performance based on current conditions."""
        # Simple performance optimization
        blocked_count = sum(1 for s in self.station_status.values() if s.state == StationState.BLOCKED)
        starved_count = sum(1 for s in self.station_status.values() if s.state == StationState.STARVED)
        
        if blocked_count > 2 or starved_count > 2:
            self.logger.warning(f"Line imbalance detected: {blocked_count} blocked, {starved_count} starved")
            # Could trigger rebalancing algorithm here
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get summary of coordination performance."""
        recent_latencies = self.message_latencies[-50:] if self.message_latencies else []
        
        return {
            'stations_registered': len(self.stations),
            'coordination_active': self.coordination_active,
            'line_running': self.line_running,
            'message_count': len(self.message_history),
            'average_latency_ms': sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0,
            'max_latency_target_ms': self.max_latency_ms,
            'latency_target_met': max(recent_latencies) < self.max_latency_ms if recent_latencies else True,
            'station_states': {sid: status.state.value for sid, status in self.station_status.items()}
        }