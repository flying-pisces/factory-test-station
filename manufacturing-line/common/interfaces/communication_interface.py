"""Communication interfaces for manufacturing components."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import time
import json


class MessageType(Enum):
    """Types of messages in manufacturing system."""
    COMMAND = "command"
    STATUS = "status"
    ALARM = "alarm"
    DATA = "data"
    HEARTBEAT = "heartbeat"
    ACK = "acknowledgment"
    ERROR = "error"


@dataclass
class Message:
    """Standard message format for manufacturing communication."""
    message_id: str
    source_id: str
    target_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = None
    priority: int = 1  # 1=low, 5=high
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'message_type': self.message_type.value,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'priority': self.priority
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            message_id=data['message_id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            message_type=MessageType(data['message_type']),
            payload=data['payload'],
            timestamp=data.get('timestamp'),
            priority=data.get('priority', 1)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class CommunicationProtocol(ABC):
    """Abstract communication protocol interface."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.subscribers: List[str] = []
        self.message_queue: List[Message] = []
    
    @abstractmethod
    def send_message(self, message: Message) -> bool:
        """Send a message."""
        pass
    
    @abstractmethod
    def receive_messages(self) -> List[Message]:
        """Receive pending messages."""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish communication connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close communication connection."""
        pass
    
    def register_handler(self, message_type: MessageType, handler: Callable[[Message], None]):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    def process_messages(self):
        """Process all pending messages."""
        messages = self.receive_messages()
        for message in messages:
            if message.message_type in self.message_handlers:
                try:
                    self.message_handlers[message.message_type](message)
                except Exception as e:
                    self._handle_message_error(message, e)
    
    def _handle_message_error(self, message: Message, error: Exception):
        """Handle message processing errors."""
        error_msg = Message(
            message_id=f"error_{int(time.time())}",
            source_id=self.component_id,
            target_id=message.source_id,
            message_type=MessageType.ERROR,
            payload={
                'original_message_id': message.message_id,
                'error': str(error)
            }
        )
        self.send_message(error_msg)
    
    def broadcast(self, message_type: MessageType, payload: Dict[str, Any], priority: int = 1):
        """Broadcast message to all subscribers."""
        for subscriber in self.subscribers:
            message = Message(
                message_id=f"broadcast_{int(time.time() * 1000000)}",
                source_id=self.component_id,
                target_id=subscriber,
                message_type=message_type,
                payload=payload,
                priority=priority
            )
            self.send_message(message)
    
    def subscribe(self, subscriber_id: str):
        """Add subscriber to broadcast list."""
        if subscriber_id not in self.subscribers:
            self.subscribers.append(subscriber_id)
    
    def unsubscribe(self, subscriber_id: str):
        """Remove subscriber from broadcast list."""
        if subscriber_id in self.subscribers:
            self.subscribers.remove(subscriber_id)