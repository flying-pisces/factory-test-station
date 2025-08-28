"""Integration tests for standard data socket pipeline."""

import pytest
import json
from pathlib import Path

from common.interfaces.socket_manager import SocketManager
from common.interfaces.layer_interface import (
    RawComponentData,
    StructuredComponentData, 
    StructuredStationData,
    StructuredLineData
)


class TestSocketPipeline:
    """Test end-to-end data socket pipeline integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.socket_manager = SocketManager()
    
    def test_component_to_station_socket_transfer(self, sample_raw_component_data):
        """Test data transfer through component-to-station socket."""
        # Get the socket
        socket = self.socket_manager.get_socket('component_to_station')
        assert socket is not None
        
        # Create raw component data
        raw_component = RawComponentData(
            component_id=sample_raw_component_data['component_id'],
            component_type=sample_raw_component_data['component_type'],
            cad_data=sample_raw_component_data['cad_data'],
            api_data=sample_raw_component_data['api_data'],
            ee_data=sample_raw_component_data['ee_data'],
            vendor_id=sample_raw_component_data['vendor_id']
        )
        
        # Transfer through socket
        result = socket.transfer(raw_component)
        
        # Validate result
        assert isinstance(result, StructuredComponentData)
        assert result.component_id == sample_raw_component_data['component_id']
        assert result.discrete_event_profile is not None
    
    def test_station_to_line_socket_transfer(self, sample_raw_station_data):
        """Test data transfer through station-to-line socket."""
        # Get the socket
        socket = self.socket_manager.get_socket('station_to_line')
        assert socket is not None
        
        # Process raw station data
        structured_station = self.socket_manager.process_station_data(sample_raw_station_data)
        
        # Transfer through socket
        result = socket.transfer(structured_station)
        
        # Validate result (should be processed by LineLayerEngine)
        assert isinstance(result, StructuredLineData)
    
    def test_full_data_pipeline(self, sample_test_data_files):
        """Test complete data processing pipeline from components to line."""
        # Load test data
        components_file = sample_test_data_files['components']
        with open(components_file, 'r') as f:
            raw_data = json.load(f)
        
        # Process through full pipeline
        results = self.socket_manager.demonstrate_full_pipeline(str(components_file))
        
        # Validate pipeline results
        assert 'structured_components' in results
        assert 'processing_metadata' in results
        
        # Validate component processing
        structured_components = results['structured_components']
        assert len(structured_components) == 2  # R1_0603 and C1_0603
        
        # Validate component data
        r1_component = next(c for c in structured_components if c['component_id'] == 'R1_0603')
        assert r1_component['component_type'] == 'Resistor'
        assert r1_component['package_size'] == '0603'
        assert r1_component['price_usd'] == 0.050
        assert r1_component['discrete_event_profile']['event_name'] == 'smt_place_passive'
        
        c1_component = next(c for c in structured_components if c['component_id'] == 'C1_0603')
        assert c1_component['component_type'] == 'Capacitor'
        assert c1_component['package_size'] == '0603'
        assert c1_component['price_usd'] == 0.080
        assert c1_component['discrete_event_profile']['event_name'] == 'smt_place_passive'
    
    def test_complete_station_processing(self, sample_test_data_files):
        """Test complete station data processing."""
        # Load station test data
        stations_file = sample_test_data_files['stations']
        with open(stations_file, 'r') as f:
            raw_data = json.load(f)
        
        # Process through pipeline
        results = self.socket_manager.demonstrate_full_pipeline(str(stations_file))
        
        # Validate station processing
        assert 'structured_stations' in results
        
        structured_stations = results['structured_stations']
        assert len(structured_stations) == 1
        
        # Validate station data
        smt_station = structured_stations[0]
        assert smt_station['station_id'] == 'SMT_P0'
        assert smt_station['station_type'] == 'SMT'
        assert 'station_cost_usd' in smt_station
        assert 'uph_capacity' in smt_station
        assert 'discrete_event_profile' in smt_station
    
    def test_complete_line_processing(self, sample_test_data_files):
        """Test complete line data processing."""
        # Load line test data
        lines_file = sample_test_data_files['lines']
        with open(lines_file, 'r') as f:
            raw_data = json.load(f)
        
        # Process through pipeline
        results = self.socket_manager.demonstrate_full_pipeline(str(lines_file))
        
        # Validate line processing
        assert 'structured_lines' in results
        
        structured_lines = results['structured_lines']
        assert len(structured_lines) == 1
        
        # Validate line data
        fatp_line = structured_lines[0]
        assert fatp_line['line_id'] == 'SMT_FATP_LINE_01'
        assert 'line_cost_usd' in fatp_line
        assert 'line_uph' in fatp_line
        assert 'line_efficiency' in fatp_line
        assert 'footprint_sqm' in fatp_line
    
    def test_socket_manager_info(self):
        """Test socket manager information retrieval."""
        info = self.socket_manager.get_all_socket_info()
        
        assert 'sockets' in info
        assert 'layer_connections' in info
        assert 'total_sockets' in info
        
        # Check default sockets exist
        assert 'component_to_station' in info['sockets']
        assert 'station_to_line' in info['sockets']
        
        # Validate socket information
        comp_to_station = info['sockets']['component_to_station']
        assert comp_to_station['input_layer'] == 'COMPONENT'
        assert comp_to_station['output_layer'] == 'STATION'
        
        station_to_line = info['sockets']['station_to_line']
        assert station_to_line['input_layer'] == 'STATION'
        assert station_to_line['output_layer'] == 'LINE'
    
    def test_data_export_import(self, sample_raw_component_data, temp_data_dir):
        """Test structured data export and import functionality."""
        # Process component data
        components = [sample_raw_component_data]
        structured_components = self.socket_manager.process_component_data(components)
        
        # Export structured data
        export_file = temp_data_dir / 'exported_components.json'
        self.socket_manager.export_structured_data(structured_components, str(export_file))
        
        # Verify export file exists and is valid JSON
        assert export_file.exists()
        
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert isinstance(exported_data, list)
        assert len(exported_data) == 1
        
        # Validate exported structure
        exported_component = exported_data[0]
        assert exported_component['component_id'] == sample_raw_component_data['component_id']
        assert 'discrete_event_profile' in exported_component
    
    def test_layer_data_validation(self, sample_raw_component_data):
        """Test layer-specific data validation."""
        from common.interfaces.layer_interface import LayerType, RawComponentData
        
        # Test valid component data validation
        raw_component = RawComponentData(
            component_id=sample_raw_component_data['component_id'],
            component_type=sample_raw_component_data['component_type'],
            cad_data=sample_raw_component_data['cad_data'],
            api_data=sample_raw_component_data['api_data'],
            ee_data=sample_raw_component_data['ee_data'],
            vendor_id=sample_raw_component_data['vendor_id']
        )
        
        # Component layer validation (through station socket since it processes components)
        is_valid = self.socket_manager.validate_layer_data(LayerType.STATION, raw_component)
        assert is_valid is True
        
        # Test invalid data
        is_valid = self.socket_manager.validate_layer_data(LayerType.STATION, "invalid_data")
        assert is_valid is False
    
    def test_layer_schema_retrieval(self):
        """Test retrieval of layer output schemas."""
        from common.interfaces.layer_interface import LayerType
        
        # Get station layer schema
        station_schema = self.socket_manager.get_layer_schema(LayerType.STATION)
        assert isinstance(station_schema, dict)
        
        # Get line layer schema  
        line_schema = self.socket_manager.get_layer_schema(LayerType.LINE)
        assert isinstance(line_schema, dict)
        
        # Test non-existent layer
        pm_schema = self.socket_manager.get_layer_schema(LayerType.PM)
        assert pm_schema == {}  # PM layer not implemented yet
    
    def test_socket_error_handling(self):
        """Test socket error handling for invalid operations."""
        # Test non-existent socket
        with pytest.raises(ValueError, match="Socket non_existent not found"):
            self.socket_manager.transfer_data('non_existent', {})
        
        # Test invalid data transfer
        socket_id = 'component_to_station'
        with pytest.raises(Exception):  # Should raise processing error
            self.socket_manager.transfer_data(socket_id, "invalid_data")
    
    def test_socket_creation_and_management(self):
        """Test dynamic socket creation and management."""
        from common.interfaces.layer_interface import LayerType
        
        # Create custom socket
        custom_socket = self.socket_manager.create_socket(
            socket_id='test_socket',
            input_layer=LayerType.COMPONENT,
            output_layer=LayerType.LINE
        )
        
        assert custom_socket is not None
        assert self.socket_manager.get_socket('test_socket') is not None
        
        # Test duplicate socket creation
        with pytest.raises(ValueError, match="Socket test_socket already exists"):
            self.socket_manager.create_socket(
                socket_id='test_socket',
                input_layer=LayerType.STATION,
                output_layer=LayerType.PM
            )