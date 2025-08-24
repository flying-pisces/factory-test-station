"""
Tests for common infrastructure components.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestCommonInfrastructure(unittest.TestCase):
    """Test cases for common infrastructure components."""
    
    def test_common_import(self):
        """Test that common module can be imported."""
        try:
            import common
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Could not import common module: {e}")
    
    def test_common_utils_import(self):
        """Test that common utilities can be imported."""
        try:
            from common import utils
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Could not import common utilities: {e}")
    
    def test_test_station_import(self):
        """Test that test_station module can be imported."""
        try:
            from common import test_station
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Could not import test_station module: {e}")


if __name__ == '__main__':
    unittest.main()