"""Component Layer Package.

Processes raw vendor data (CAD, API, EE) into structured component data 
with discrete event profiles for manufacturing operations.
"""

try:
    from .component_engine import ComponentLayerEngine
except ImportError:
    # Handle import during development
    ComponentLayerEngine = None

__all__ = ['ComponentLayerEngine']