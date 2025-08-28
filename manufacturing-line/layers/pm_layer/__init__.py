"""PM (Product Management) Layer Package.

AI-enabled manufacturing optimization with genetic algorithms
for yield vs MVA trade-off analysis and Pareto optimal solutions.
"""

try:
    from .pm_engine import PMLayerEngine
except ImportError:
    # PM layer not yet implemented
    PMLayerEngine = None

__all__ = ['PMLayerEngine']