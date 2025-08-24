
__author__ = 'chuckyin'

# pylint: disable=R0921
# Fix for the fact that nothing references this in here.


class TestEquipment(object):
    """
        abstract base class for Test Fixtures
    """
    def __init__(self, station_config, operator_interface):
        self._operator_interface = operator_interface
        self._station_config = station_config

    def initialize(self):
        """ Initialize test fixture
        """
        raise NotImplementedError()

    def close(self):
        """ close test fixture
        """
        raise NotImplementedError()

    def is_ready(self):
        raise NotImplementedError()
