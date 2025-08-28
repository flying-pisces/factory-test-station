"""Base digital operator class for manufacturing line integration."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import queue
import threading


class OperatorStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    INTERVENTION_REQUIRED = "intervention_required"
    OFFLINE = "offline"
    BREAK = "break"


class OperatorCapability(Enum):
    BUTTON_PRESS = "button_press"
    ITEM_PICKUP = "item_pickup"
    VISUAL_INSPECTION = "visual_inspection"
    ISSUE_MONITORING = "issue_monitoring"
    MANUAL_OVERRIDE = "manual_override"
    QUALITY_CHECK = "quality_check"


@dataclass
class OperatorAction:
    """Action to be performed by digital operator."""
    action_id: str
    action_type: OperatorCapability
    target_station: str
    parameters: Dict[str, Any]
    priority: int = 5  # 1-10, higher is more urgent
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class BaseOperator(ABC):
    """Abstract base class for digital operator implementations."""
    
    def __init__(self, operator_id: str, config: Dict[str, Any]):
        self.operator_id = operator_id
        self.name = config.get('name', f'Operator_{operator_id}')
        self.config = config
        self.status = OperatorStatus.IDLE
        self.assigned_station = config.get('assigned_station')
        self.capabilities = self._parse_capabilities(config.get('capabilities', []))
        self.action_queue = queue.PriorityQueue()
        self.action_history: List[OperatorAction] = []
        self.performance_metrics = {
            'actions_completed': 0,
            'avg_response_time': 0.0,
            'success_rate': 1.0
        }
        self._worker_thread = None
        self._running = False
        self._hooks: Dict[str, List[Callable]] = {}
    
    def _parse_capabilities(self, capability_list: List[str]) -> List[OperatorCapability]:
        """Parse capability strings to enum values."""
        capabilities = []
        for cap_str in capability_list:
            try:
                capabilities.append(OperatorCapability(cap_str))
            except ValueError:
                print(f"Unknown capability: {cap_str}")
        return capabilities
    
    def start(self):
        """Start the digital operator."""
        if self._running:
            return False
        
        self._running = True
        self.status = OperatorStatus.IDLE
        self._worker_thread = threading.Thread(target=self._action_processor)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        return True
    
    def stop(self):
        """Stop the digital operator."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        self.status = OperatorStatus.OFFLINE
    
    def add_action(self, action: OperatorAction) -> bool:
        """Add an action to the operator's queue."""
        if action.action_type not in self.capabilities:
            return False
        
        # Priority queue uses negative priority for higher urgency
        self.action_queue.put((-action.priority, action.timestamp, action))
        return True
    
    def _action_processor(self):
        """Background thread to process action queue."""
        while self._running:
            try:
                # Wait for action with timeout
                priority, timestamp, action = self.action_queue.get(timeout=0.5)
                
                self.status = OperatorStatus.BUSY
                start_time = time.time()
                
                # Execute action based on type
                success = self._execute_action(action)
                
                # Update metrics
                response_time = time.time() - start_time
                self._update_metrics(response_time, success)
                
                # Update digital twin
                self._update_digital_twin()
                
                # Add to history
                self.action_history.append(action)
                if len(self.action_history) > 100:
                    self.action_history.pop(0)
                
                # Trigger completion hook
                self.trigger_hook('action_completed', {
                    'action': action,
                    'success': success,
                    'duration': response_time
                })
                
                self.status = OperatorStatus.IDLE
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing action: {e}")
                self.status = OperatorStatus.INTERVENTION_REQUIRED
    
    def _execute_action(self, action: OperatorAction) -> bool:
        """Execute specific action based on type."""
        if action.action_type == OperatorCapability.BUTTON_PRESS:
            return self.press_button(action.target_station, action.parameters)
        elif action.action_type == OperatorCapability.ITEM_PICKUP:
            return self.pickup_item(action.target_station, action.parameters)
        elif action.action_type == OperatorCapability.VISUAL_INSPECTION:
            return self.visual_inspection(action.target_station, action.parameters)
        elif action.action_type == OperatorCapability.ISSUE_MONITORING:
            return self.monitor_issue(action.target_station, action.parameters)
        else:
            return False
    
    @abstractmethod
    def press_button(self, station: str, parameters: Dict[str, Any]) -> bool:
        """Simulate button press at station."""
        pass
    
    @abstractmethod
    def pickup_item(self, station: str, parameters: Dict[str, Any]) -> bool:
        """Pick up item missed by conveyor."""
        pass
    
    @abstractmethod
    def visual_inspection(self, station: str, parameters: Dict[str, Any]) -> bool:
        """Perform visual quality inspection."""
        pass
    
    @abstractmethod
    def monitor_issue(self, station: str, parameters: Dict[str, Any]) -> bool:
        """Monitor and report station issues."""
        pass
    
    def _update_metrics(self, response_time: float, success: bool):
        """Update performance metrics."""
        self.performance_metrics['actions_completed'] += 1
        
        # Update average response time
        n = self.performance_metrics['actions_completed']
        avg = self.performance_metrics['avg_response_time']
        self.performance_metrics['avg_response_time'] = (avg * (n-1) + response_time) / n
        
        # Update success rate
        if success:
            success_count = int(self.performance_metrics['success_rate'] * (n-1))
            self.performance_metrics['success_rate'] = (success_count + 1) / n
        else:
            success_count = int(self.performance_metrics['success_rate'] * (n-1))
            self.performance_metrics['success_rate'] = success_count / n
    
    def get_status(self) -> Dict[str, Any]:
        """Get current operator status."""
        return {
            'operator_id': self.operator_id,
            'name': self.name,
            'status': self.status.value,
            'assigned_station': self.assigned_station,
            'capabilities': [cap.value for cap in self.capabilities],
            'queue_size': self.action_queue.qsize(),
            'performance': self.performance_metrics
        }
    
    def reassign_station(self, new_station: str):
        """Reassign operator to different station."""
        self.assigned_station = new_station
        self.trigger_hook('station_reassigned', {
            'operator_id': self.operator_id,
            'new_station': new_station
        })
    
    def take_break(self, duration: float = 300):
        """Operator takes a break."""
        self.status = OperatorStatus.BREAK
        threading.Timer(duration, self._end_break).start()
    
    def _end_break(self):
        """End break and return to idle."""
        self.status = OperatorStatus.IDLE
    
    # Hook system for external integration
    def register_hook(self, event: str, callback: Callable):
        """Register callback for operator events."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def trigger_hook(self, event: str, data: Dict[str, Any]):
        """Trigger registered hooks with event data."""
        if event in self._hooks:
            for callback in self._hooks[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Hook callback error: {e}")
    
    def self_test(self) -> Dict[str, Any]:
        """Run self-diagnostic test."""
        results = {
            'operator_id': self.operator_id,
            'timestamp': time.time(),
            'tests': {}
        }
        
        # Test action queue
        test_action = OperatorAction(
            action_id='test_001',
            action_type=OperatorCapability.BUTTON_PRESS,
            target_station='test_station',
            parameters={'button': 'test'}
        )
        
        results['tests']['queue_add'] = self.add_action(test_action)
        results['tests']['status_check'] = self.status in OperatorStatus
        results['tests']['capabilities'] = len(self.capabilities) > 0
        
        results['passed'] = all(results['tests'].values())
        return results