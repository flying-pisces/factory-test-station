"""Unit Tests for ComponentLayerEngine - Week 2 Implementation."""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from layers.component_layer.component_layer_engine import ComponentLayerEngine, ComponentType, ProcessingMetrics
    from layers.component_layer.vendor_interfaces.cad_processor import CADProcessor
    from layers.component_layer.vendor_interfaces.api_processor import APIProcessor
    from layers.component_layer.vendor_interfaces.ee_processor import EEProcessor
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


@unittest.skipIf(not IMPORTS_AVAILABLE, "ComponentLayerEngine imports not available")
class TestComponentLayerEngine(unittest.TestCase):
    """Test ComponentLayerEngine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = ComponentLayerEngine()
        
        # Sample component data
        self.sample_component = {
            'component_id': 'R1_TEST',
            'component_type': 'Resistor',
            'cad_data': {
                'package': '0603',
                'footprint': 'RES_0603',
                'dimensions': {'length': 1.6, 'width': 0.8, 'height': 0.45}
            },
            'api_data': {
                'manufacturer': 'Yageo',
                'part_number': 'RC0603FR-0710KL',
                'price_usd': 0.050,
                'stock_quantity': 10000
            },
            'ee_data': {
                'resistance': 10000,
                'tolerance': 0.01,
                'power_rating': 0.25
            },
            'vendor_id': 'YAGEO_001'
        }
    
    def test_initialization(self):
        """Test ComponentLayerEngine initialization."""
        self.assertIsInstance(self.engine, ComponentLayerEngine)
        self.assertIsInstance(self.engine.cad_processor, CADProcessor)
        self.assertIsInstance(self.engine.api_processor, APIProcessor)
        self.assertIsInstance(self.engine.ee_processor, EEProcessor)
        self.assertEqual(self.engine.performance_target_ms, 100)
    
    def test_component_type_enum(self):
        """Test ComponentType enum values."""
        self.assertEqual(ComponentType.RESISTOR.value, "Resistor")
        self.assertEqual(ComponentType.CAPACITOR.value, "Capacitor")
        self.assertEqual(ComponentType.IC.value, "IC")
        self.assertEqual(ComponentType.INDUCTOR.value, "Inductor")
    
    def test_single_component_processing(self):
        """Test processing of a single component."""
        result = self.engine.process_raw_component_data([self.sample_component])
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['processed_components']), 1)
        self.assertIn('metrics', result)
        self.assertTrue(result['performance_target_met'])
        
        # Check processed component structure
        processed = result['processed_components'][0]
        self.assertEqual(processed['component_id'], 'R1_TEST')
        self.assertEqual(processed['component_type'], 'Resistor')
        self.assertIn('processed_cad_data', processed)
        self.assertIn('processed_api_data', processed)
        self.assertIn('processed_ee_data', processed)
        self.assertIn('discrete_event_profile', processed)
    
    def test_multiple_component_processing(self):
        """Test processing of multiple components."""
        components = [
            self.sample_component,
            {
                'component_id': 'C1_TEST',
                'component_type': 'Capacitor',
                'cad_data': {'package': '0402'},
                'api_data': {'price_usd': 0.03},
                'ee_data': {'capacitance': 1e-6}
            }
        ]
        
        result = self.engine.process_raw_component_data(components)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['processed_components']), 2)
        
        # Verify different component types processed
        component_types = [c['component_type'] for c in result['processed_components']]
        self.assertIn('Resistor', component_types)
        self.assertIn('Capacitor', component_types)
    
    def test_discrete_event_profile_generation(self):
        """Test discrete event profile generation."""
        result = self.engine.process_raw_component_data([self.sample_component])
        processed = result['processed_components'][0]
        
        profile = processed['discrete_event_profile']
        self.assertIn('event_type', profile)
        self.assertIn('placement_time_s', profile)
        self.assertIn('total_cycle_time_s', profile)
        self.assertEqual(profile['component_type'], 'Resistor')
        self.assertEqual(profile['event_type'], 'smt_place_passive')
        self.assertGreater(profile['placement_time_s'], 0)
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        # Process some components to generate metrics
        self.engine.process_raw_component_data([self.sample_component])
        
        summary = self.engine.get_performance_summary()
        
        self.assertIn('total_components_processed', summary)
        self.assertIn('overall_avg_component_time_ms', summary)
        self.assertIn('performance_target_met', summary)
        self.assertEqual(summary['total_components_processed'], 1)
        self.assertLess(summary['overall_avg_component_time_ms'], 100)  # Should meet target
    
    def test_week2_requirements_validation(self):
        """Test Week 2 specific requirements validation."""
        validation = self.engine.validate_week2_requirements()
        
        self.assertIn('vendor_interfaces_implemented', validation['validations'])
        self.assertIn('component_type_processors', validation['validations'])
        self.assertIn('performance_requirements', validation['validations'])
        
        # Check vendor interfaces
        vendor_interfaces = validation['validations']['vendor_interfaces_implemented']
        self.assertTrue(vendor_interfaces['cad_processor'])
        self.assertTrue(vendor_interfaces['api_processor'])
        self.assertTrue(vendor_interfaces['ee_processor'])
        
        # Check component processors
        component_processors = validation['validations']['component_type_processors']
        self.assertTrue(component_processors['resistor'])
        self.assertTrue(component_processors['capacitor'])
        self.assertTrue(component_processors['ic'])
        self.assertTrue(component_processors['inductor'])
    
    def test_error_handling(self):
        """Test error handling with malformed data."""
        malformed_component = {
            'component_id': 'BAD_TEST',
            # Missing required fields
        }
        
        result = self.engine.process_raw_component_data([malformed_component])
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('errors', result)
    
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        result = self.engine.process_raw_component_data([])
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['processed_components']), 0)
        self.assertEqual(result['metrics']['components_processed'], 0)
    
    @patch('layers.component_layer.component_layer_engine.time.time')
    def test_performance_measurement(self, mock_time):
        """Test performance measurement accuracy."""
        # Mock time to control timing
        mock_time.side_effect = [0.0, 0.05]  # 50ms processing time
        
        result = self.engine.process_raw_component_data([self.sample_component])
        
        self.assertAlmostEqual(result['metrics']['processing_time_ms'], 50.0, places=0)
        self.assertTrue(result['performance_target_met'])  # Should be under 100ms target


@unittest.skipIf(not IMPORTS_AVAILABLE, "Vendor processor imports not available")
class TestVendorProcessors(unittest.TestCase):
    """Test vendor interface processors."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cad_processor = CADProcessor()
        self.api_processor = APIProcessor()
        self.ee_processor = EEProcessor()
    
    def test_cad_processor(self):
        """Test CAD processor functionality."""
        cad_data = {
            'package': '0603',
            'footprint': 'RES_0603',
            'dimensions': {'length': 1.6, 'width': 0.8, 'height': 0.45}
        }
        
        result = self.cad_processor.process_cad_data(cad_data)
        
        self.assertIn('package_info', result)
        self.assertIn('processed_dimensions', result)
        self.assertIn('placement_requirements', result)
        self.assertEqual(result['package_info']['type'], 'smt_passive')
    
    def test_api_processor(self):
        """Test API processor functionality."""
        api_data = {
            'manufacturer': 'Yageo',
            'part_number': 'RC0603FR-0710KL',
            'price_usd': 0.050,
            'stock_quantity': 10000,
            'lead_time_days': 14
        }
        
        result = self.api_processor.process_api_data(api_data)
        
        self.assertIn('vendor_info', result)
        self.assertIn('pricing', result)
        self.assertIn('availability', result)
        self.assertEqual(result['pricing']['price_usd'], 0.050)
    
    def test_ee_processor(self):
        """Test EE processor functionality."""
        ee_data = {
            'resistance': 10000,
            'tolerance': 0.01,
            'power_rating': 0.25,
            'voltage_rating': 100
        }
        
        result = self.ee_processor.process_ee_data(ee_data)
        
        self.assertIn('validated_parameters', result)
        self.assertIn('test_requirements', result)
        self.assertIn('measurement_plan', result)
        
        # Check parameter validation
        resistance = result['validated_parameters']['resistance']
        self.assertEqual(resistance['value'], 10000)
        self.assertTrue(resistance['valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)