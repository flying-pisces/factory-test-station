"""Unit tests for Component Layer Engine."""

import pytest
from typing import Dict, Any

from common.interfaces.layer_interface import (
    ComponentLayerEngine, 
    RawComponentData, 
    StructuredComponentData,
    DiscreteEventProfile
)


class TestComponentLayerEngine:
    """Test cases for Component Layer Engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ComponentLayerEngine()
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine is not None
        assert hasattr(self.engine, 'process')
        assert hasattr(self.engine, 'validate_input')
    
    def test_process_resistor_component(self, sample_raw_component_data):
        """Test processing resistor component data."""
        # Create RawComponentData instance
        raw_component = RawComponentData(
            component_id=sample_raw_component_data['component_id'],
            component_type=sample_raw_component_data['component_type'],
            cad_data=sample_raw_component_data['cad_data'],
            api_data=sample_raw_component_data['api_data'],
            ee_data=sample_raw_component_data['ee_data'],
            vendor_id=sample_raw_component_data['vendor_id']
        )
        
        # Process component
        result = self.engine.process(raw_component)
        
        # Validate result type
        assert isinstance(result, StructuredComponentData)
        
        # Validate structured data
        assert result.component_id == 'R1_TEST'
        assert result.component_type == 'Resistor'
        assert result.package_size == '0603'
        assert result.price_usd == 0.050
        assert result.lead_time_days == 14
        
        # Validate discrete event profile
        assert result.discrete_event_profile is not None
        assert isinstance(result.discrete_event_profile, DiscreteEventProfile)
        assert result.discrete_event_profile.event_name == 'smt_place_passive'
        assert result.discrete_event_profile.duration == 0.5
        assert result.discrete_event_profile.frequency == 7200
    
    def test_process_capacitor_component(self):
        """Test processing capacitor component data."""
        raw_component = RawComponentData(
            component_id='C1_TEST',
            component_type='Capacitor',
            cad_data={'package': '0603', 'dimensions': {'length': 1.6, 'width': 0.8}},
            api_data={'price_usd': 0.080, 'lead_time_days': 21},
            ee_data={'capacitance': 0.0000001, 'voltage_rating': 50},
            vendor_id='VENDOR_TEST_002'
        )
        
        result = self.engine.process(raw_component)
        
        assert result.component_id == 'C1_TEST'
        assert result.component_type == 'Capacitor'
        assert result.package_size == '0603'
        assert result.price_usd == 0.080
        assert result.lead_time_days == 21
        
        # Capacitors also use passive placement
        assert result.discrete_event_profile.event_name == 'smt_place_passive'
        assert result.discrete_event_profile.duration == 0.5
    
    def test_process_ic_component(self):
        """Test processing IC component data."""
        raw_component = RawComponentData(
            component_id='U1_TEST',
            component_type='IC',
            cad_data={'package': 'QFN32', 'dimensions': {'length': 5.0, 'width': 5.0}},
            api_data={'price_usd': 12.500, 'lead_time_days': 60},
            ee_data={'pin_count': 32, 'operating_voltage': 3.3},
            vendor_id='VENDOR_TEST_003'
        )
        
        result = self.engine.process(raw_component)
        
        assert result.component_id == 'U1_TEST'
        assert result.component_type == 'IC'
        assert result.package_size == 'QFN32'
        assert result.price_usd == 12.500
        assert result.lead_time_days == 60
        
        # ICs use different placement profile
        assert result.discrete_event_profile.event_name == 'smt_place_ic'
        assert result.discrete_event_profile.duration == 2.0
        assert result.discrete_event_profile.frequency == 1800
    
    def test_invalid_component_type(self):
        """Test processing invalid component type."""
        raw_component = RawComponentData(
            component_id='INVALID_TEST',
            component_type='InvalidType',
            cad_data={},
            api_data={'price_usd': 1.0, 'lead_time_days': 30},
            ee_data={},
            vendor_id='VENDOR_TEST'
        )
        
        # Should handle unknown types gracefully
        result = self.engine.process(raw_component)
        assert result.component_type == 'InvalidType'
        # Should default to generic placement
        assert result.discrete_event_profile.event_name == 'generic_placement'
    
    def test_missing_api_data(self):
        """Test processing component with missing API data."""
        raw_component = RawComponentData(
            component_id='R2_TEST',
            component_type='Resistor',
            cad_data={'package': '0805'},
            api_data={},  # Missing price and lead time
            ee_data={'resistance': 1000},
            vendor_id='VENDOR_TEST'
        )
        
        result = self.engine.process(raw_component)
        
        # Should use default values
        assert result.price_usd == 0.0  # Default price
        assert result.lead_time_days == 30  # Default lead time
    
    def test_validate_input_valid(self, sample_raw_component_data):
        """Test input validation with valid data."""
        raw_component = RawComponentData(
            component_id=sample_raw_component_data['component_id'],
            component_type=sample_raw_component_data['component_type'],
            cad_data=sample_raw_component_data['cad_data'],
            api_data=sample_raw_component_data['api_data'],
            ee_data=sample_raw_component_data['ee_data'],
            vendor_id=sample_raw_component_data['vendor_id']
        )
        
        assert self.engine.validate_input(raw_component) is True
    
    def test_validate_input_invalid(self):
        """Test input validation with invalid data."""
        # Test with non-RawComponentData type
        assert self.engine.validate_input("invalid_data") is False
        
        # Test with None
        assert self.engine.validate_input(None) is False
    
    def test_get_output_schema(self):
        """Test output schema retrieval."""
        schema = self.engine.get_output_schema()
        
        assert isinstance(schema, dict)
        assert 'component_id' in schema
        assert 'component_type' in schema
        assert 'package_size' in schema
        assert 'price_usd' in schema
        assert 'lead_time_days' in schema
        assert 'discrete_event_profile' in schema
    
    @pytest.mark.parametrize("component_type,expected_event", [
        ('Resistor', 'smt_place_passive'),
        ('Capacitor', 'smt_place_passive'),
        ('IC', 'smt_place_ic'),
        ('Inductor', 'smt_place_passive'),
        ('Unknown', 'generic_placement')
    ])
    def test_discrete_event_assignment(self, component_type, expected_event):
        """Test discrete event profile assignment for different component types."""
        raw_component = RawComponentData(
            component_id=f'{component_type}_TEST',
            component_type=component_type,
            cad_data={'package': '0603'},
            api_data={'price_usd': 1.0, 'lead_time_days': 30},
            ee_data={},
            vendor_id='VENDOR_TEST'
        )
        
        result = self.engine.process(raw_component)
        assert result.discrete_event_profile.event_name == expected_event