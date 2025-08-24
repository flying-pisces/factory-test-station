#!/usr/bin/env python
"""
Main Factory Test GUI - reorganized version using separated components.
This is the main entry point that coordinates the various GUI components.
"""

__author__ = 'chuckyin'

import argparse
import os
import sys

# Import separated GUI components  
from .wpf_gui_core import WPFGuiCore, WPF_AVAILABLE
from .gui_test_runner import GuiTestRunner
from .gui_dialogs import UpdateWorkorderDialog, UpdateStationIdDialog

# Import operator interface
from .operator_interface.operator_interface import OperatorInterface

# Import common components
from common.test_station import test_station

# WPF-specific imports (only if available)
if WPF_AVAILABLE:
    from System.Threading import Thread, ApartmentState, ThreadStart
    from Hsc.ViewModel import ViewModelLocator
    from Hsc import MainWindow, App


class FactoryTestGui(WPFGuiCore):
    """Main Factory Test GUI - combines core GUI with test runner functionality."""
    
    def __init__(self, station_config, test_station_init):
        super().__init__(station_config, test_station_init)
        self.test_runner = None

    def initialize_test_runner(self):
        """Initialize the test runner component."""
        if self._operator_interface:
            self.test_runner = GuiTestRunner(self._operator_interface)
            self.test_runner.set_station(self.station, self.station_config)

    def setguistate_initializetester(self, station):
        """Initialize tester state."""
        self._operator_interface.print_to_console("Initializing Tester...\n", "grey")
        setup_ok = False
        
        if station:
            try:
                # Configure logging
                if (hasattr(self.station_config, 'IS_PRINT_TO_LOG')
                        and self.station_config.IS_PRINT_TO_LOG):
                    sys.stdout = self
                    sys.stderr = self
                    sys.stdin = None

                if (hasattr(self.station_config, 'IS_LOG_PROFILING')
                     and self.station_config.IS_LOG_PROFILING):
                    import logging
                    import datetime
                    logger = logging.getLogger('profiling')
                    logger.setLevel(logging.DEBUG)
                    fn = f'profile_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                    handler = logging.FileHandler(fn)
                    handler.setLevel(logging.DEBUG)
                    logger.addHandler(handler)

                station.initialize()

                # Handle console visibility (Windows only)
                if WPF_AVAILABLE:
                    import ctypes
                    show_console = 0
                    whnd = ctypes.windll.kernel32.GetConsoleWindow()
                    if whnd != 0:
                        if hasattr(self.station_config, 'SHOW_CONSOLE') and self.station_config.SHOW_CONSOLE:
                            show_console = 1
                    ctypes.windll.user32.ShowWindow(whnd, show_console)
                    self._vm_main_view_model.MovFocusToSn()
                    
                setup_ok = True
                
            except (test_station.TestStationError, Exception) as e:
                self._operator_interface.print_to_console(f"Error Initializing Test Station {str(e)}.\n", "red")
                setup_ok = False
                
        if setup_ok:
            self._operator_interface.print_to_console("Initialization complete.\n")
            self._operator_interface.update_root_config({'IsEnabled': 'True'})
            self._operator_interface.print_to_console("waiting for sn\n")
            self._operator_interface.prompt("Scan or type the DUT Serial Number", 'green')
            self.initialize_test_runner()

    def mu_action(self, sender, e):
        """Handle menu actions."""
        if sender == 'Browser':
            import subprocess
            subprocess.Popen(rf'explorer "{self.station_config.ROOT_DIR}"')
        elif sender == "Active":
            self.station_config.IS_STATION_ACTIVE = bool(e)
        elif sender == "WO":
            self.update_workorder_display()
        elif sender == 'Offline':
            self.station_config.FACEBOOK_IT_ENABLED = not bool(e)
        elif sender == 'AutoScan':
            self.station_config.AUTO_SCAN_CODE = bool(e)

    def start_loop(self, user_value):
        """Start test loop - delegates to test runner."""
        if not self.check_free_space_ready():
            return
        if self.test_runner:
            self.test_runner.start_loop(user_value)

    def update_workorder(self):
        """Update work order display."""
        if self.test_runner:
            self._operator_interface.update_root_config({'WorkOrder': self.test_runner.g_workorder})

    def update_workorder_display(self):
        """Show work order dialog."""
        sub_window_title = "Edit Work Order"
        UpdateWorkorderDialog(self, self.root, sub_window_title)

    def update_stationtype_display(self):
        """Show station type dialog.""" 
        sub_window_title = "Scan Station Label"
        UpdateStationIdDialog(self, self.root, sub_window_title)

    def _app_startup(self):
        """Application startup logic."""
        if not WPF_AVAILABLE:
            print("WPF not available - cannot start GUI")
            return
            
        self._vm_locator = ViewModelLocator.Instance
        self._vm_main_view_model = ViewModelLocator.Instance.Main
        self._vm_main_view_model.MuStartLoop += self.start_loop

        # Version info
        init_config = {}
        version_file = "VERSION.TXT"
        version_dir = os.getcwd()
        version_filename = os.path.join(version_dir, version_file)
        if os.path.isfile(version_filename):
            with open(version_filename, "r") as version_fileobj:
                version_data = version_fileobj.read()
                init_config['VersionData'] = version_data
            
        if hasattr(self.station_config, 'SW_TITLE'):
            init_config['SwTitle'] = self.station_config.SW_TITLE
        init_config['Offline'] = str(not self.station_config.FACEBOOK_IT_ENABLED)
        init_config['Active'] = 'True'
        init_config['IsEnabled'] = 'False'
        if hasattr(self.station_config, 'IS_STATION_ACTIVE'):
            init_config['Active'] = str(self.station_config.IS_STATION_ACTIVE)

        log_dir = os.path.join(self.station_config.ROOT_DIR, "factory-test_debug")
        self._operator_interface = OperatorInterface(self, self._vm_main_view_model, log_dir)
        self._operator_interface.update_root_config(init_config)

        self._vm_main_view_model.ShowWorkOrder = True if self.station_config.USE_WORKORDER_ENTRY else False
        if isinstance(self.station_config.STATION_NUMBER, int) and self.station_config.STATION_NUMBER == 0:
            self.update_stationtype_display()
        else:
            self.create_station()
            self.setguistate_initializetester(self.station)

    def AppStartUp(self, sender, e):
        """WPF application startup event."""
        if not WPF_AVAILABLE:
            return
        try:
            self.root = MainWindow()
            self.root.Title = "Please scan test_station id !!!!"
            self.root.Show()
            self.root.MuAction += self.mu_action
            self._app_startup()
        except Exception as ex:
            print(f"Error during app startup: {ex}")

    def STAMain(self):
        """Main WPF application thread."""
        if not WPF_AVAILABLE:
            return
        app = App()
        app.Startup += self.AppStartUp
        app.Exit += self.on_app_exit
        app.Run()

    def main_loop(self):
        """Main application loop."""
        if not WPF_AVAILABLE:
            print("WPF GUI not available on this platform")
            return
            
        try:
            t = Thread(ThreadStart(self.STAMain))
            t.ApartmentState = ApartmentState.STA
            t.Start()

            args = self.parse_arguments()

            # Initialize loop settings
            if self.test_runner:
                self.test_runner._g_num_loops_completed = 0
                self.test_runner._g_num_passing_loops = 0
                self.test_runner._g_loop_sn = None
                if args.numloops is not None:
                    self.test_runner._g_target_num_loops = args.numloops

            t.Join()
        except Exception as e:
            if self._operator_interface:
                self._operator_interface.print_to_console(f'Exception from station : {str(e)}')

    @staticmethod
    def parse_arguments():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='GUI for Programming/Test station.')
        parser.add_argument('-l', '--numloops', type=int, 
                          help='FOR STATION VERIFICATION ONLY: number of times to repeat the test without cycling the fixture.')
        return parser.parse_args()


# For backward compatibility
FactoryTestGuiMain = FactoryTestGui