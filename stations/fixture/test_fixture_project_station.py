import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from test_station.test_fixture.test_fixture import TestFixture


class projectstationFixture(TestFixture):
    """
        class for project station Fixture
            this is for doing all the specific things necessary to interface with instruments
    """
    def __init__(self, station_config, operator_interface):
        TestFixture.__init__(self, station_config, operator_interface)

    def is_ready(self):
        pass

    def initialize(self):
        self._operator_interface.print_to_console("Initializing project station Fixture\n")

    def close(self):
        self._operator_interface.print_to_console("Closing project station Fixture\n")

    def __getattr__(self, item):
        def not_find(*args, **kwargs):
            return
        if item in ['status', 'elminator_off', 'elminator_on', 'mov_abs_xy', 'unload',
                    'load', 'button_disable', 'button_enable', 'flush_data', 'mov_abs_xya',
                    'is_ready', 'power_on_button_status', 'start_button_status', 'query_temp',
                    'mov_abs_xy_wrt_alignment', 'mov_camera_z_wrt_alignment', 'query_probe_status',
                    'particle_counter_state', 'version', 'particle_counter_read_val', 'mov_abs_xy_wrt_dut',
                    'ca_postion_z',]:
            return not_find