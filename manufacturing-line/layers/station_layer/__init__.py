"""Station Layer Package.

Processes component data and station requirements into optimized
station configurations with cost and UPH metrics.
"""

try:
    from .station_engine import StationLayerEngine
except ImportError:
    # Handle import during development
    StationLayerEngine = None

__all__ = ['StationLayerEngine']