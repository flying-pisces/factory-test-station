#!/usr/bin/env python
"""
GUI Dialog Components - handles dialog boxes and user input for GUI interfaces.
Separated from main GUI files for better organization.
"""

import re

try:
    from gui_utils import MessageBox
    from Hsc import InputMsgBox
    WPF_AVAILABLE = True
except ImportError:
    WPF_AVAILABLE = False
    # Fallback for non-WPF environments
    class MessageBox:
        @staticmethod
        def error(title="Error", msg="An error occurred"):
            print(f"{title}: {msg}")
    
    class InputMsgBox:
        def __init__(self):
            self.InputText = ""


class UpdateWorkorderDialog:
    """Dialog for updating work order information."""
    
    def __init__(self, factory_gui, parent, title):
        self._gui = factory_gui
        if WPF_AVAILABLE:
            self._wo_dlg = InputMsgBox()
            self._wo_dlg.Owner = parent
            self._wo_dlg.Title = title
            self._wo_dlg.MuAction += self._workorder_dlg_action
            self._wo_dlg.Show()
        else:
            # Console fallback
            self._wo_dlg = None
            self._console_workorder_input(title)

    def _workorder_dlg_action(self, sender, e):
        """Handle work order dialog action."""
        self._gui.g_workorder = self._wo_dlg.InputText
        self._wo_dlg.Hide()
        self._gui.update_workorder()
        return 1

    def _console_workorder_input(self, title):
        """Console fallback for work order input."""
        try:
            workorder = input(f"{title}: Enter work order: ")
            self._gui.g_workorder = workorder
            self._gui.update_workorder()
        except KeyboardInterrupt:
            pass


class UpdateStationIdDialog:
    """Dialog for updating station ID information."""
    
    def __init__(self, factory_gui, parent, title):
        self._gui = factory_gui
        self._station_id = None
        self._station_id_pattern = r'[\\/:*?"<>|\r\n]+'
        
        if WPF_AVAILABLE:
            self._station_type_dlg = InputMsgBox()
            self._station_type_dlg.Owner = parent
            self._station_type_dlg.Title = title
            self._station_type_dlg.MuAction += self._station_type_dlg_action
            self._station_type_dlg.Show()
        else:
            # Console fallback
            self._station_type_dlg = None
            self._console_station_id_input(title)

    def apply(self):
        """Apply the station ID changes."""
        self._gui.root.Title = f"Oculus HWTE {self._station_id}"
        self._gui.setguistate_initializetester(self._gui.station)
        if WPF_AVAILABLE and self._station_type_dlg:
            self._station_type_dlg.Hide()
        if self._gui.station_config.USE_WORKORDER_ENTRY:
            self._gui.update_workorder_display()

    def _station_type_dlg_action(self, sender, e):
        """Handle station type dialog action."""
        if WPF_AVAILABLE:
            self._station_id = self._station_type_dlg.InputText
        return self._process_station_id()

    def _console_station_id_input(self, title):
        """Console fallback for station ID input."""
        try:
            self._station_id = input(f"{title}: Enter station ID (format: stationId-stationNumber): ")
            self._process_station_id()
        except KeyboardInterrupt:
            pass

    def _process_station_id(self):
        """Process and validate the station ID."""
        try:
            if re.search(self._station_id_pattern, self._station_id, re.I | re.S):
                raise ValueError
            
            (self._gui.station_config.STATION_TYPE,
             self._gui.station_config.STATION_NUMBER) = re.split('-', self._station_id)
            self._gui.create_station()
            self.apply()
            return True
            
        except ValueError:
            MessageBox.error(
                title="Station Config Error",
                msg="Station ID is of the form stationId-stationNumber"
            )
            return False
        except Exception as e:
            from common.test_station import test_station
            if isinstance(e, test_station.TestStationError):
                MessageBox.error(
                    title="Station Config Error",
                    msg=f"{self._gui.station_config.STATION_TYPE} is not a valid station type!"
                )
                return False
            else:
                raise