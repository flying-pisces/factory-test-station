"""
GUI Components Module

This module contains all GUI-related components organized by functionality:
- WPF GUI core functionality
- Test runner logic
- Dialog components  
- Operator interface
- GUI utilities

The components are separated for better maintainability and testing.
"""

from .factory_test_gui_main import FactoryTestGui, FactoryTestGuiMain

__version__ = "1.0.0"
__all__ = ['FactoryTestGui', 'FactoryTestGuiMain']