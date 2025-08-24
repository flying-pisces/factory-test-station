"""
Tests for GUI components.
"""

import unittest
import sys
import os

# Add project root to path  
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestGUIComponents(unittest.TestCase):
    """Test cases for GUI components."""
    
    def test_gui_import(self):
        """Test that GUI module can be imported."""
        try:
            import GUI
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Could not import GUI module: {e}")
    
    def test_gui_test_runner_import(self):
        """Test that GUI test runner can be imported."""
        try:
            from GUI.gui_test_runner import GuiTestRunner
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Could not import GuiTestRunner: {e}")
    
    def test_gui_dialogs_import(self):
        """Test that GUI dialogs can be imported."""
        try:
            from GUI.gui_dialogs import UpdateWorkorderDialog, UpdateStationIdDialog
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Could not import GUI dialogs: {e}")


class TestWPFAvailability(unittest.TestCase):
    """Test WPF availability detection."""
    
    def test_wpf_availability_detection(self):
        """Test that WPF availability can be detected without crashing."""
        try:
            from GUI.wpf_gui_core import WPF_AVAILABLE
            self.assertIsInstance(WPF_AVAILABLE, bool)
        except ImportError as e:
            self.fail(f"Could not import WPF availability flag: {e}")


if __name__ == '__main__':
    unittest.main()