"""Core manufacturing component interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import uuid
import time


class ComponentState(Enum):
    """Standard states for manufacturing components."""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


@dataclass
class ComponentStatus:
    """Status information for manufacturing component."""
    component_id: str
    component_type: str
    state: ComponentState
    timestamp: float
    metrics: Dict[str, Any]
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ManufacturingComponent(ABC):
    """Abstract base class for all manufacturing components."""
    
    def __init__(self, component_id: str, component_type: str):
        self.component_id = component_id
        self.component_type = component_type
        self.state = ComponentState.OFFLINE
        self.created_time = time.time()
        self.last_update = time.time()
        self.metrics = {}
        self.errors = []
        self.session_id = str(uuid.uuid4())
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the component."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the component safely."""
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset component to initial state."""
        pass
    
    @abstractmethod
    def get_status(self) -> ComponentStatus:
        """Get current component status."""
        pass
    
    @abstractmethod
    def handle_command(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle external commands."""
        pass
    
    def update_metrics(self, new_metrics: Dict[str, Any]):
        """Update component metrics."""
        self.metrics.update(new_metrics)
        self.last_update = time.time()
    
    def add_error(self, error_message: str):
        """Add error to component error log."""
        error_entry = {
            'timestamp': time.time(),
            'message': error_message
        }
        self.errors.append(error_entry)
        if len(self.errors) > 100:  # Keep only last 100 errors
            self.errors = self.errors[-100:]
    
    def clear_errors(self):
        """Clear error log."""
        self.errors.clear()
    
    def set_state(self, new_state: ComponentState):
        """Set component state with timestamp."""
        self.state = new_state
        self.last_update = time.time()
    
    def get_uptime(self) -> float:
        """Get component uptime in seconds."""
        return time.time() - self.created_time