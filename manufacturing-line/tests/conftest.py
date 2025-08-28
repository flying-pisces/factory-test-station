"""Pytest configuration and fixtures for manufacturing line tests."""

import pytest
import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core components
from common.interfaces.layer_interface import (
    RawComponentData, StructuredComponentData,
    RawStationData, StructuredStationData,
    RawLineData, StructuredLineData
)


@pytest.fixture(scope="session")
def project_root_path():
    """Project root directory path."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_raw_component_data():
    """Sample raw component data for testing."""
    return {
        'component_id': 'R1_TEST',
        'component_type': 'Resistor',
        'cad_data': {'package': '0603', 'dimensions': {'length': 1.6, 'width': 0.8}},
        'api_data': {'price_usd': 0.050, 'lead_time_days': 14},
        'ee_data': {'resistance': 10000, 'tolerance': 0.05, 'power_rating': 0.1},
        'vendor_id': 'VENDOR_TEST_001'
    }


@pytest.fixture
def sample_raw_station_data():
    """Sample raw station data for testing."""
    return {
        'station_id': 'TEST_SMT_001',
        'station_type': 'SMT',
        'component_raw_data': [
            {
                'component_id': 'R1_TEST',
                'component_type': 'Resistor',
                'cad_data': {'package': '0603'},
                'api_data': {'price_usd': 0.050, 'lead_time_days': 14},
                'ee_data': {'resistance': 10000},
                'vendor_id': 'VENDOR_TEST_001'
            }
        ],
        'test_coverage': {'functional': 0.95, 'parametric': 0.80},
        'operator_requirements': {'count': 1, 'skill_level': 'intermediate'},
        'footprint_constraints': {'width': 2.0, 'depth': 1.5, 'height': 2.0}
    }


@pytest.fixture
def sample_raw_line_data():
    """Sample raw line data for testing."""
    return {
        'line_id': 'TEST_LINE_001',
        'station_raw_data': [
            {
                'station_id': 'TEST_SMT_001',
                'station_type': 'SMT',
                'component_raw_data': [
                    {
                        'component_id': 'R1_TEST',
                        'component_type': 'Resistor',
                        'cad_data': {'package': '0603'},
                        'api_data': {'price_usd': 0.050, 'lead_time_days': 14},
                        'ee_data': {'resistance': 10000},
                        'vendor_id': 'VENDOR_TEST_001'
                    }
                ],
                'test_coverage': {'functional': 0.95},
                'operator_requirements': {'count': 1},
                'footprint_constraints': {'width': 2.0, 'depth': 1.5}
            }
        ],
        'dut_raw_data': {'dut_type': 'PCB_ASSEMBLY', 'complexity': 'medium'},
        'operator_raw_data': {'total_operators': 2, 'shifts': 3},
        'retest_policy': 'AAB',
        'total_capacity': 100
    }


@pytest.fixture
def mock_socket_manager():
    """Mock socket manager for testing."""
    from common.interfaces.socket_manager import SocketManager
    return SocketManager()


@pytest.fixture
def sample_discrete_events():
    """Sample discrete events for FSM testing."""
    from simulation.simulation_engine.discrete_event import DiscreteEvent
    return [
        DiscreteEvent('load_component', 1.5),
        DiscreteEvent('place_component', 0.5),
        DiscreteEvent('inspect_placement', 2.0),
        DiscreteEvent('release_dut', 0.3)
    ]


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'simulation': {
            'time_step': 0.1,
            'max_time': 3600.0,
            'random_seed': 42
        },
        'optimization': {
            'population_size': 50,
            'generations': 100,
            'mutation_rate': 0.1
        },
        'database': {
            'url': 'sqlite:///:memory:',
            'echo': False
        }
    }


@pytest.fixture
def sample_test_data_files(temp_data_dir):
    """Create sample test data files."""
    # Component data file
    components_data = {
        'components': [
            {
                'component_id': 'R1_0603',
                'component_type': 'Resistor',
                'cad_data': {'package': '0603', 'dimensions': {'length': 1.6, 'width': 0.8}},
                'api_data': {'price_usd': 0.050, 'lead_time_days': 14},
                'ee_data': {'resistance': 10000, 'tolerance': 0.05},
                'vendor_id': 'VENDOR_A_001'
            },
            {
                'component_id': 'C1_0603',
                'component_type': 'Capacitor',
                'cad_data': {'package': '0603', 'dimensions': {'length': 1.6, 'width': 0.8}},
                'api_data': {'price_usd': 0.080, 'lead_time_days': 21},
                'ee_data': {'capacitance': 0.0000001, 'voltage_rating': 50},
                'vendor_id': 'VENDOR_B_002'
            }
        ]
    }
    
    components_file = temp_data_dir / 'components.json'
    with open(components_file, 'w') as f:
        json.dump(components_data, f, indent=2)
    
    # Station data file
    stations_data = {
        'stations': [
            {
                'station_id': 'SMT_P0',
                'station_type': 'SMT',
                'component_raw_data': components_data['components'],
                'test_coverage': {'functional': 0.95, 'parametric': 0.80},
                'operator_requirements': {'count': 1, 'skill_level': 'intermediate'},
                'footprint_constraints': {'width': 3.0, 'depth': 2.0, 'height': 2.5}
            }
        ]
    }
    
    stations_file = temp_data_dir / 'stations.json'
    with open(stations_file, 'w') as f:
        json.dump(stations_data, f, indent=2)
    
    # Line data file
    lines_data = {
        'lines': [
            {
                'line_id': 'SMT_FATP_LINE_01',
                'station_raw_data': stations_data['stations'],
                'dut_raw_data': {'dut_type': 'PCB_ASSEMBLY', 'complexity': 'medium'},
                'operator_raw_data': {'total_operators': 3, 'shifts': 2},
                'retest_policy': 'AAB',
                'total_capacity': 100
            }
        ]
    }
    
    lines_file = temp_data_dir / 'lines.json'
    with open(lines_file, 'w') as f:
        json.dump(lines_data, f, indent=2)
    
    return {
        'components': components_file,
        'stations': stations_file,
        'lines': lines_file
    }


# Pytest markers for organizing tests
pytest_plugins = []

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "system: System tests")
    config.addinivalue_line("markers", "acceptance: Acceptance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "simulation: Simulation tests")
    config.addinivalue_line("markers", "optimization: AI optimization tests")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        test_path = Path(item.fspath)
        
        if 'unit' in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif 'integration' in test_path.parts:
            item.add_marker(pytest.mark.integration)
        elif 'system' in test_path.parts:
            item.add_marker(pytest.mark.system)
        elif 'acceptance' in test_path.parts:
            item.add_marker(pytest.mark.acceptance)
        
        # Add specific markers for test content
        if 'simulation' in str(test_path):
            item.add_marker(pytest.mark.simulation)
        if 'optimization' in str(test_path):
            item.add_marker(pytest.mark.optimization)
        if 'slow' in item.name or 'performance' in item.name:
            item.add_marker(pytest.mark.slow)