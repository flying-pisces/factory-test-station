"""Common interfaces and protocols for manufacturing components."""

# Import with error handling during development
try:
    from .manufacturing_interface import ManufacturingComponent, ComponentState
except ImportError:
    ManufacturingComponent = ComponentState = None

try:
    from .communication_interface import CommunicationProtocol, MessageType
except ImportError:
    CommunicationProtocol = MessageType = None

try:
    from .data_interface import DataLogger, MetricsCollector
except ImportError:
    DataLogger = MetricsCollector = None

__all__ = [
    'ManufacturingComponent',
    'ComponentState',
    'CommunicationProtocol',
    'MessageType',
    'DataLogger',
    'MetricsCollector'
]