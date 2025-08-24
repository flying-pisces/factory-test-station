#!/usr/bin/env python
"""
GUI Test Runner - handles test execution logic for GUI interfaces.
Separated from main GUI files for better organization.
"""

import time
import threading
import gc
from common.test_station import test_station


class GuiTestRunner:
    """Handles test execution logic for GUI interfaces."""
    
    def __init__(self, operator_interface):
        self._operator_interface = operator_interface
        self.station = None
        self.station_config = None
        self._g_loop_sn = None
        self._g_num_loops_completed = 0
        self._g_num_passing_loops = 0
        self._g_target_num_loops = None
        self.g_workorder = ""

    def set_station(self, station, station_config):
        """Set the test station and configuration."""
        self.station = station
        self.station_config = station_config

    def is_looping_enabled(self):
        """Check if loop testing is enabled."""
        return self._g_target_num_loops is not None

    def update_loop_counter(self, last_test_result=None):
        """Update loop counter display."""
        if self.is_looping_enabled():
            # Update counters
            if last_test_result is not None:
                self._g_num_loops_completed += 1
                if last_test_result:
                    self._g_num_passing_loops += 1

            # Report counts to user
            msg = "completed %d of %d loops.  %d passed" % (
                self._g_num_loops_completed,
                self._g_target_num_loops, 
                self._g_num_passing_loops
            )
            self._operator_interface.update_root_config({'Hint': msg})
            time.sleep(1)

            # Check if done looping
            if self._g_target_num_loops == 0:
                # 0 = infinite looping
                pass
            else:
                if self._g_num_loops_completed >= self._g_target_num_loops:
                    self._g_loop_sn = None  # Stop looping
                    self._g_num_loops_completed = 0
                    self._g_num_passing_loops = 0

            return self._g_num_loops_completed
        else:
            return None

    def test_iteration(self, serial_number):
        """Execute a single test iteration."""
        self._operator_interface.print_to_console("Running Test.\n")
        overall_result = False
        first_failed_test_result = None
        
        if self.station_config.USE_WORKORDER_ENTRY:
            self.station.workorder = self.g_workorder
            
        try:
            (overall_result, first_failed_test_result) = self.station.test_unit(serial_number.upper())
        except test_station.TestStationProcessControlError as e:
            msg = 'Process Control Error: Unit %s [  %s  ] ...\n' % (serial_number, e.message)
            self._operator_interface.print_to_console(msg, 'red')
            self._operator_interface.print_to_console('*******************\n')
            self._g_loop_sn = None  # Cancel loop on error
            return
        except Exception as e:
            self._operator_interface.print_to_console("Test Station Error:{0}\n".format(str(e)), "red")
            self._g_loop_sn = None

        self.gui_show_result(serial_number, overall_result, first_failed_test_result)

    def gui_show_result(self, serial_number, overall_result, first_failed_test_result=None):
        """Display test results in GUI."""
        self._operator_interface.print_to_console('-----------------------------------\n')
        did_pass = False
        
        if overall_result:
            did_pass = True
            error_code = '[0]'
        else:
            if first_failed_test_result is None:
                error_code = "(unknown)"
            else:
                error_code = "{0}.{1}".format(
                    first_failed_test_result.get_unique_id(),
                    first_failed_test_result.get_error_code_as_string())

        if did_pass:
            self._operator_interface.update_root_config({
                'FinalResult': 'OK',
                'ResultMsg': ''})
            self._operator_interface.print_to_console(f'Unit {serial_number} OK\n')
        else:
            self._operator_interface.update_root_config({
                'FinalResult': 'NG',
                'ResultMsg': error_code})
            self._operator_interface.print_to_console(f'Unit {serial_number} NG, Errcode = {error_code}\n')
            
        self.update_loop_counter(did_pass)

    def run_test(self, serial_number):
        """Run test with loop support."""
        self._operator_interface.update_root_config({'IsBusy': 'True'})
        self._operator_interface.print_to_console(f'SERIAL_NUMBER:{serial_number}\n')

        if self.is_looping_enabled():
            self._g_loop_sn = serial_number

        self.test_iteration(serial_number)
        
        # Loop testing
        while self._g_loop_sn is not None:
            time.sleep(1)
            self._operator_interface.clear_console()
            self._operator_interface.clear_test_values()
            self._operator_interface.update_root_config(
                {'FinalResult': '', 'ResultMsg': '', 'ResultMsgEx': ''})
            self.test_iteration(serial_number)
            
        gc.collect()
        self._operator_interface.update_root_config({'IsBusy': 'False', 'SN': ''})
        self._operator_interface.prompt('Scan or type the DUT Serial Number', 'green')

    def thread_it(self, func, *args):
        """Execute function in separate thread."""
        t = threading.Thread(target=func, args=args)
        t.setDaemon(True)
        t.start()
        return t

    def start_loop(self, user_value):
        """Start test loop with given serial number."""
        if not self.station_config.IS_STATION_ACTIVE:
            return
            
        try:
            self._operator_interface.clear_console()
            self._operator_interface.clear_test_values()
            self._operator_interface.update_root_config(
                {'FinalResult': '', 'Hint': '', 'ResultMsg': '', 'ResultMsgEx': ''})
            self._operator_interface.prompt('', '')

            self.station.validate_sn(user_value)
            if self.is_looping_enabled():
                self._g_num_loops_completed = 0
                self._g_num_passing_loops = 0
                self.update_loop_counter()
            self.station.is_ready()
            self.thread_it(self.run_test, user_value)
        except test_station.TestStationSerialNumberError as teststationerror:
            self._operator_interface.print_to_console(msg="%s" % str(teststationerror), color='red')
        except Exception as e:
            self._operator_interface.operator_input(msg="Exception: %s" % str(e), msg_type='error')
        finally:
            self._operator_interface.print_to_console("waiting for sn\n")
            self._g_loop_sn = None