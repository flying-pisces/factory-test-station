"""Manufacturing Layer Architecture.

This package implements the multi-layer manufacturing architecture with
standard data sockets for scalable and maintainable system design.

Layers:
- Component Layer: Raw vendor data → Structured components
- Station Layer: Component data → Station optimization  
- Line Layer: Station data → Line efficiency
- PM Layer: AI-enabled manufacturing optimization
"""

from .component_layer import ComponentLayerEngine
from .station_layer import StationLayerEngine  
from .line_layer import LineLayerEngine

try:
    from .pm_layer import PMLayerEngine
except ImportError:
    # PM Layer engine not yet implemented
    PMLayerEngine = None

__all__ = [
    'ComponentLayerEngine',
    'StationLayerEngine', 
    'LineLayerEngine',
    'PMLayerEngine'
]