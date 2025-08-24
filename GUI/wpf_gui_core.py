#!/usr/bin/env python
"""
Core WPF GUI functionality - separated from main factory_test_gui.py for better organization.
Contains the main GUI class and core initialization logic.
"""

__author__ = 'chuckyin'

import time
import sys
import os
import ctypes
import gc
import logging
import datetime

# WPF/CLR imports
try:
    import clr
    
    # Add WPF references
    clr.AddReference("PresentationFramework.Classic, Version=3.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35")
    clr.AddReference("PresentationCore, Version=3.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35")
    clr.AddReference('AgLib')
    clr.AddReference('CommonServiceLocator')
    clr.AddReference('ErrorHandler')
    clr.AddReference('GalaSoft.MvvmLight')
    clr.AddReference('GalaSoft.MvvmLight.Extras')
    clr.AddReference('GalaSoft.MvvmLight.Platform')
    clr.AddReference('log4net')
    clr.AddReference('Newtonsoft.Json')
    clr.AddReference('PresentationFramework.Aero2')
    clr.AddReference('System.Windows.Interactivity')
    clr.AddReference('Util')
    clr.AddReference('WPFMessageBox')
    clr.AddReference('Xceed.Wpf.DataGrid')
    clr.AddReference('Xceed.Wpf.Toolkit')
    clr.AddReference('Hsc')

    from System.Windows import Application, Window
    from System.Threading import Thread, ApartmentState, ThreadStart
    from System import Action, Delegate
    from Hsc import MainWindow, App, InputMsgBox
    from System.Collections.Generic import Dictionary
    from Hsc.ViewModel import ViewModelLocator, MainViewModel
    
    WPF_AVAILABLE = True
except ImportError:
    WPF_AVAILABLE = False
    print("WPF components not available - this GUI requires Windows with .NET Framework")

# Import GUI utilities
try:
    from gui_utils import MessageBox
except ImportError:
    try:
        from .gui_utils import MessageBox
    except ImportError:
        # Fallback message box for non-WPF environments
        class MessageBox:
            @staticmethod
            def error(title="Error", msg="An error occurred"):
                print(f"{title}: {msg}")


class WPFGuiCore:
    """Core WPF GUI functionality."""
    
    def __init__(self, station_config, test_station_init):
        if not WPF_AVAILABLE:
            raise RuntimeError("WPF GUI requires Windows with .NET Framework")
            
        self.station_config = station_config
        self._test_station_init = test_station_init
        self.station = None
        self._operator_interface = None
        self._vm_locator = None
        self._vm_main_view_model = None
        self.g_workorder = ""
        self.root = None

    def write(self, msg):
        """Write message to operator interface."""
        if not msg.isspace():
            msg = msg.strip('\n')
            msg = msg.replace('\r', '」') + '\n'
            self._operator_interface.print_to_console(msg)

    def create_station(self):
        """Create and initialize the test station."""
        try:
            self._operator_interface.print_to_console('Station Type: %s\n' % self.station_config.STATION_TYPE)
            self.station = self._test_station_init(self.station_config, self._operator_interface)
            station_id = (f'{self.station_config.STATION_TYPE}_{self.station_config.STATION_NUMBER}')
            self._operator_interface.update_root_config({'Title': f'Oculus HWTE {station_id}'})
        except:
            raise

    def get_free_space_mb(self, folder):
        """Get available disk space in MB."""
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024

    def check_free_space_ready(self):
        """Check if sufficient disk space is available."""
        if not hasattr(self.station_config, 'MIN_SPACE_REQUIRED'):
            return True
        if isinstance(self.station_config.MIN_SPACE_REQUIRED, list):
            req_dirs = [c for c in self.station_config.MIN_SPACE_REQUIRED if os.path.exists(c[0])]
            chk_free_space = [self.get_free_space_mb(req_dir) >= req_size for req_dir, req_size in req_dirs]
            if not all(chk_free_space):
                msg = f"Unable to start test, please check mini-space required: {req_dirs} "
                self._operator_interface.operator_input('WARN', msg=msg, msg_type='warning')
                return False
        else:
            self._operator_interface.print_to_console(f'Please update the configuration named min_space_required.\n')
        return True

    def on_app_exit(self, sender, e):
        """Handle application exit."""
        if self.station is not None:
            self.station.close()
        if self._operator_interface is not None:
            self._operator_interface.close()
        ViewModelLocator.Cleanup()
        time.sleep(0.5)
        self.root.Close()