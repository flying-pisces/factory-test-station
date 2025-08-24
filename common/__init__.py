"""
Common Infrastructure Module

This module contains core infrastructure components:
- Test station framework
- DUT (Device Under Test) implementations
- Test equipment interfaces
- Test fixture components
- Utility functions (OS, retries, serial number handling)

Originally named 'hardware_station_common', now reorganized as 'common'
with GUI and log components moved to their respective modules.
"""

name = "common"

__version__ = "1.0.0"
__all__ = ['test_station', 'utils']

from . import test_station
from . import utils
