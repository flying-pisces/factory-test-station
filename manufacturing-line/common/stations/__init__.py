"""Station components for manufacturing line system."""

# Import with error handling during development
try:
    from .base_station import BaseStation, StationState, StationMetrics
except ImportError:
    BaseStation = StationState = StationMetrics = None

try:
    from .smt_station import SMTStation
except ImportError:
    SMTStation = None

try:
    from .test_station import TestStation
except ImportError:
    TestStation = None

try:
    from .assembly_station import AssemblyStation
except ImportError:
    AssemblyStation = None

try:
    from .quality_station import QualityStation
except ImportError:
    QualityStation = None

try:
    from .station_manager import StationManager
except ImportError:
    StationManager = None

__all__ = [
    'BaseStation',
    'StationState', 
    'StationMetrics',
    'SMTStation',
    'TestStation',
    'AssemblyStation',
    'QualityStation',
    'StationManager'
]