import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
import test_station.dut.dut as base_dut

class projectDut(object):
    """
        class for project station DUT
            this is for doing all the specific things necessary to interface with the DUT
    """
    def __init__(self, serial_number, station_config, operator_interface):
        self._operator_interface = operator_interface
        self._station_config = station_config
        self._serial_number = serial_number
        
    @property 
    def serial_number(self):
        return self._serial_number

    def is_ready(self):
        return True

    def initialize(self):
        self._operator_interface.print_to_console("Initializing project station DUT\n")

    def close(self):
        self._operator_interface.print_to_console("Closing project station DUT\n")

    def __getattr__(self, item):
        def not_find(*args, **kwargs):
            pass
        if item in ['screen_on', 'screen_off', 'display_color', 'reboot', 'display_image', 'nvm_read_statistics',
                    'nvm_write_data', '_get_color_ext', 'render_image', 'nvm_read_data']:
            return not_find