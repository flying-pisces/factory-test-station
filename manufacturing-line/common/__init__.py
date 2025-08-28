"""Common components for manufacturing line system.

This package contains shared components organized by type:
- stations/: Station-related components and base classes
- operators/: Operator (human/digital) components
- conveyors/: Conveyor belt and transport systems
- equipment/: Test and measurement equipment
- fixtures/: Manufacturing fixtures and tooling
- utils/: Utility functions and helpers
- interfaces/: Common interfaces and protocols
"""

# Import key components for easier access (with error handling during development)
try:
    from .interfaces.manufacturing_interface import ManufacturingComponent
except ImportError:
    ManufacturingComponent = None

try:
    from .stations.base_station import BaseStation
except ImportError:
    BaseStation = None

try:
    from .operators.base_operator import BaseOperator
except ImportError:
    BaseOperator = None

try:
    from .conveyors.base_conveyor import BaseConveyor
except ImportError:
    BaseConveyor = None

__all__ = [
    'ManufacturingComponent',
    'BaseStation',
    'BaseOperator', 
    'BaseConveyor'
]