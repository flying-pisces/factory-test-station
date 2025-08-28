"""Line Layer Package.

Processes station configurations into optimized line layouts
with efficiency calculations and bottleneck analysis.
"""

try:
    from .line_engine import LineLayerEngine
except ImportError:
    # Handle import during development  
    LineLayerEngine = None

__all__ = ['LineLayerEngine']