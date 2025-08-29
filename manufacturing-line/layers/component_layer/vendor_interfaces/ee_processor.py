"""EE (Electrical Engineering) Data Processor for Component Vendor Interfaces - Week 2 Implementation."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import math


class EEProcessor:
    """Processes electrical engineering data from component vendors (specifications, ratings, characteristics)."""
    
    def __init__(self):
        self.logger = logging.getLogger('EEProcessor')
        
        # Standard parameter ranges for validation
        self.parameter_ranges = {
            # Resistors
            'resistance': {
                'min': 0.001,  # 1mΩ
                'max': 1e12,   # 1TΩ
                'typical_unit': 'ohms'
            },
            'resistance_tolerance': {
                'min': 0.001,  # 0.1%
                'max': 0.20,   # 20%
                'typical_unit': 'percent'
            },
            
            # Capacitors
            'capacitance': {
                'min': 1e-15,  # 1fF
                'max': 10,     # 10F
                'typical_unit': 'farads'
            },
            
            # General electrical parameters
            'voltage_rating': {
                'min': 0.1,    # 0.1V
                'max': 50000,  # 50kV
                'typical_unit': 'volts'
            },
            'current_rating': {
                'min': 1e-12,  # 1pA
                'max': 10000,  # 10kA
                'typical_unit': 'amperes'
            },
            'power_rating': {
                'min': 1e-6,   # 1µW
                'max': 10000,  # 10kW
                'typical_unit': 'watts'
            },
            'temperature_coefficient': {
                'min': -5000,  # -5000ppm/°C
                'max': 5000,   # +5000ppm/°C
                'typical_unit': 'ppm_per_celsius'
            }
        }
        
        # Component-specific processing rules
        self.component_rules = {
            'resistor': {
                'required_params': ['resistance'],
                'optional_params': ['tolerance', 'power_rating', 'temperature_coefficient'],
                'derived_calculations': ['power_density', 'thermal_resistance']
            },
            'capacitor': {
                'required_params': ['capacitance', 'voltage_rating'],
                'optional_params': ['tolerance', 'esr', 'ripple_current'],
                'derived_calculations': ['energy_storage', 'charge_storage']
            },
            'inductor': {
                'required_params': ['inductance'],
                'optional_params': ['current_rating', 'dc_resistance', 'quality_factor'],
                'derived_calculations': ['energy_storage', 'time_constant']
            },
            'ic': {
                'required_params': ['voltage_rating'],
                'optional_params': ['current_consumption', 'frequency_max', 'pin_count'],
                'derived_calculations': ['power_consumption', 'thermal_characteristics']
            }
        }
        
        self.logger.info("EEProcessor initialized with component-specific processing rules")
    
    def process_ee_data(self, ee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process electrical engineering data and extract manufacturing-relevant information."""
        try:
            processed = {}
            
            # Validate and normalize parameters
            validated_params = self._validate_parameters(ee_data)
            processed['validated_parameters'] = validated_params
            
            # Perform derived calculations
            derived_data = self._calculate_derived_parameters(validated_params)
            processed['derived_parameters'] = derived_data
            
            # Generate test requirements
            test_requirements = self._generate_test_requirements(validated_params)
            processed['test_requirements'] = test_requirements
            
            # Assess reliability characteristics
            reliability = self._assess_reliability(validated_params)
            processed['reliability_assessment'] = reliability
            
            # Generate application constraints
            constraints = self._generate_application_constraints(validated_params)
            processed['application_constraints'] = constraints
            
            # Create measurement plan
            measurement_plan = self._create_measurement_plan(validated_params, test_requirements)
            processed['measurement_plan'] = measurement_plan
            
            return processed
            
        except Exception as e:
            self.logger.error(f"EE processing failed: {e}")
            return {
                'error': str(e),
                'processing_status': 'failed',
                'validated_parameters': {},
                'test_requirements': {'testable': False},
                'reliability_assessment': {'reliability_class': 'unknown'}
            }
    
    def _validate_parameters(self, ee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate electrical parameters against expected ranges."""
        validated = {}
        validation_warnings = []
        validation_errors = []
        
        for param, value in ee_data.items():
            if param in self.parameter_ranges:
                range_info = self.parameter_ranges[param]
                
                try:
                    numeric_value = float(value)
                    
                    # Range validation
                    if numeric_value < range_info['min']:
                        validation_warnings.append(f"{param} value {numeric_value} below expected minimum {range_info['min']}")
                    elif numeric_value > range_info['max']:
                        validation_warnings.append(f"{param} value {numeric_value} above expected maximum {range_info['max']}")
                    
                    validated[param] = {
                        'value': numeric_value,
                        'unit': range_info['typical_unit'],
                        'valid': True,
                        'in_typical_range': range_info['min'] <= numeric_value <= range_info['max']
                    }
                    
                except (ValueError, TypeError):
                    validation_errors.append(f"Cannot convert {param} value '{value}' to numeric")
                    validated[param] = {
                        'value': value,
                        'unit': 'unknown',
                        'valid': False,
                        'in_typical_range': False
                    }
            else:
                # Unknown parameter - keep as-is but mark as unvalidated
                validated[param] = {
                    'value': value,
                    'unit': 'unknown',
                    'valid': None,  # Unknown parameter
                    'in_typical_range': None
                }
        
        validated['_validation_summary'] = {
            'warnings': validation_warnings,
            'errors': validation_errors,
            'total_parameters': len(ee_data),
            'validated_parameters': len([v for v in validated.values() if isinstance(v, dict) and v.get('valid') == True])
        }
        
        return validated
    
    def _calculate_derived_parameters(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived electrical parameters for manufacturing analysis."""
        derived = {}
        
        # Resistor calculations
        if 'resistance' in validated_params:
            resistance = validated_params['resistance']['value']
            
            if 'power_rating' in validated_params:
                power_rating = validated_params['power_rating']['value']
                
                # Calculate maximum voltage and current
                derived['max_voltage_v'] = math.sqrt(power_rating * resistance)
                derived['max_current_a'] = math.sqrt(power_rating / resistance)
                derived['power_density_w_per_mm2'] = power_rating  # Simplified - needs package info
        
        # Capacitor calculations
        if 'capacitance' in validated_params:
            capacitance = validated_params['capacitance']['value']
            
            if 'voltage_rating' in validated_params:
                voltage_rating = validated_params['voltage_rating']['value']
                
                # Energy storage calculation
                derived['max_energy_storage_j'] = 0.5 * capacitance * (voltage_rating ** 2)
                derived['charge_storage_c'] = capacitance * voltage_rating
                
                # Ripple current analysis (if available)
                if 'ripple_current' in validated_params:
                    ripple_current = validated_params['ripple_current']['value']
                    derived['ripple_current_density'] = ripple_current / capacitance
        
        # Inductor calculations
        if 'inductance' in validated_params:
            inductance = validated_params['inductance']['value']
            
            if 'current_rating' in validated_params:
                current_rating = validated_params['current_rating']['value']
                
                # Energy storage calculation
                derived['max_energy_storage_j'] = 0.5 * inductance * (current_rating ** 2)
                
                # Time constant (requires resistance)
                if 'dc_resistance' in validated_params:
                    dc_resistance = validated_params['dc_resistance']['value']
                    derived['time_constant_s'] = inductance / dc_resistance
        
        # Temperature-related calculations
        if 'temperature_coefficient' in validated_params:
            temp_coeff = validated_params['temperature_coefficient']['value']
            
            # Operating temperature range from validated params
            temp_min = validated_params.get('operating_temperature_min', {}).get('value', -40)
            temp_max = validated_params.get('operating_temperature_max', {}).get('value', 85)
            
            # Parameter drift over temperature range
            temp_range = temp_max - temp_min
            derived['max_parameter_drift_percent'] = abs(temp_coeff) * temp_range / 10000  # ppm to percent
        
        # Power calculations for ICs
        if 'current_consumption' in validated_params and 'voltage_rating' in validated_params:
            current = validated_params['current_consumption']['value']
            voltage = validated_params['voltage_rating']['value']
            
            derived['power_consumption_w'] = current * voltage
            derived['thermal_power_w'] = derived['power_consumption_w']  # Simplified
        
        return derived
    
    def _generate_test_requirements(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test requirements for manufacturing validation."""
        test_req = {
            'required_tests': [],
            'optional_tests': [],
            'test_conditions': {},
            'measurement_accuracy_requirements': {}
        }
        
        # Basic parameter tests
        for param, data in validated_params.items():
            if param.startswith('_'):  # Skip internal fields
                continue
                
            if isinstance(data, dict) and data.get('valid') == True:
                if param == 'resistance':
                    test_req['required_tests'].append('dc_resistance_measurement')
                    test_req['measurement_accuracy_requirements']['resistance'] = '0.1%'
                    
                elif param == 'capacitance':
                    test_req['required_tests'].append('capacitance_measurement')
                    test_req['measurement_accuracy_requirements']['capacitance'] = '1%'
                    test_req['test_conditions']['capacitance_frequency'] = '1kHz'
                    
                elif param == 'inductance':
                    test_req['required_tests'].append('inductance_measurement')
                    test_req['measurement_accuracy_requirements']['inductance'] = '2%'
                    test_req['test_conditions']['inductance_frequency'] = '1kHz'
                    
                elif param == 'voltage_rating':
                    test_req['optional_tests'].append('dielectric_strength_test')
                    test_req['test_conditions']['dielectric_test_voltage'] = data['value'] * 1.5
                    
                elif param == 'current_rating':
                    test_req['optional_tests'].append('current_carrying_capacity_test')
                    test_req['test_conditions']['test_current'] = data['value'] * 1.1
        
        # Environmental tests based on temperature range
        if any('temperature' in param for param in validated_params.keys()):
            test_req['optional_tests'].append('temperature_cycling_test')
            test_req['test_conditions']['temperature_cycling'] = {
                'min_temp': -40,
                'max_temp': 85,
                'cycles': 5
            }
        
        # Determine testability
        test_req['testable'] = len(test_req['required_tests']) > 0
        test_req['test_complexity'] = 'low' if len(test_req['required_tests']) <= 2 else 'medium'
        
        return test_req
    
    def _assess_reliability(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess component reliability based on electrical characteristics."""
        reliability = {
            'reliability_factors': [],
            'risk_factors': [],
            'mtbf_estimate_hours': None
        }
        
        # Temperature stress analysis
        if 'operating_temperature_max' in validated_params:
            max_temp = validated_params['operating_temperature_max']['value']
            if max_temp > 125:
                reliability['risk_factors'].append('High temperature operation (>125°C)')
            elif max_temp > 85:
                reliability['risk_factors'].append('Elevated temperature operation (>85°C)')
            else:
                reliability['reliability_factors'].append('Standard temperature range')
        
        # Voltage stress analysis
        if 'voltage_rating' in validated_params:
            voltage_rating = validated_params['voltage_rating']['value']
            if voltage_rating > 1000:
                reliability['risk_factors'].append('High voltage component (>1000V)')
            elif voltage_rating > 100:
                reliability['risk_factors'].append('Medium voltage component (>100V)')
            else:
                reliability['reliability_factors'].append('Low voltage operation')
        
        # Power stress analysis
        if 'power_rating' in validated_params:
            power_rating = validated_params['power_rating']['value']
            if power_rating > 50:
                reliability['risk_factors'].append('High power dissipation (>50W)')
            elif power_rating > 5:
                reliability['risk_factors'].append('Medium power dissipation (>5W)')
            else:
                reliability['reliability_factors'].append('Low power dissipation')
        
        # Current stress analysis
        if 'current_rating' in validated_params:
            current_rating = validated_params['current_rating']['value']
            if current_rating > 10:
                reliability['risk_factors'].append('High current operation (>10A)')
            elif current_rating > 1:
                reliability['risk_factors'].append('Medium current operation (>1A)')
            else:
                reliability['reliability_factors'].append('Low current operation')
        
        # Overall reliability classification
        risk_count = len(reliability['risk_factors'])
        positive_count = len(reliability['reliability_factors'])
        
        if risk_count == 0 and positive_count > 2:
            reliability['reliability_class'] = 'high'
            reliability['mtbf_estimate_hours'] = 100000  # 11+ years
        elif risk_count <= 1 and positive_count >= 1:
            reliability['reliability_class'] = 'medium'
            reliability['mtbf_estimate_hours'] = 50000   # 5+ years
        elif risk_count <= 2:
            reliability['reliability_class'] = 'fair'
            reliability['mtbf_estimate_hours'] = 25000   # 3+ years
        else:
            reliability['reliability_class'] = 'poor'
            reliability['mtbf_estimate_hours'] = 10000   # 1+ year
        
        return reliability
    
    def _generate_application_constraints(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate application constraints for circuit design."""
        constraints = {
            'operating_constraints': {},
            'design_guidelines': [],
            'compatibility_warnings': []
        }
        
        # Operating voltage constraints
        if 'voltage_rating' in validated_params:
            voltage_rating = validated_params['voltage_rating']['value']
            constraints['operating_constraints']['max_voltage'] = voltage_rating
            constraints['operating_constraints']['recommended_voltage'] = voltage_rating * 0.8  # 80% derating
            
            if voltage_rating < 5:
                constraints['design_guidelines'].append('Suitable for low-voltage digital circuits')
            elif voltage_rating < 50:
                constraints['design_guidelines'].append('Suitable for analog and power circuits')
            else:
                constraints['design_guidelines'].append('High voltage component - special handling required')
        
        # Temperature constraints
        temp_max = validated_params.get('operating_temperature_max', {}).get('value')
        temp_min = validated_params.get('operating_temperature_min', {}).get('value')
        
        if temp_max is not None:
            constraints['operating_constraints']['max_temperature_c'] = temp_max
            if temp_max < 70:
                constraints['compatibility_warnings'].append('Limited temperature range - not suitable for automotive')
            elif temp_max >= 125:
                constraints['design_guidelines'].append('High temperature capable - suitable for harsh environments')
        
        if temp_min is not None:
            constraints['operating_constraints']['min_temperature_c'] = temp_min
            if temp_min > 0:
                constraints['compatibility_warnings'].append('Not suitable for outdoor/automotive applications')
        
        # Frequency constraints for reactive components
        if 'capacitance' in validated_params:
            capacitance = validated_params['capacitance']['value']
            # Self-resonant frequency estimate (simplified)
            if capacitance > 1e-6:  # >1µF
                constraints['design_guidelines'].append('Large capacitor - low frequency applications')
            elif capacitance < 1e-12:  # <1pF
                constraints['design_guidelines'].append('Small capacitor - high frequency applications')
        
        return constraints
    
    def _create_measurement_plan(self, validated_params: Dict[str, Any], test_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed measurement plan for manufacturing test."""
        plan = {
            'measurement_sequence': [],
            'required_equipment': [],
            'measurement_settings': {},
            'pass_fail_criteria': {}
        }
        
        # Sequence basic measurements first
        basic_measurements = ['resistance', 'capacitance', 'inductance']
        for measurement in basic_measurements:
            if measurement in validated_params:
                param_data = validated_params[measurement]
                plan['measurement_sequence'].append({
                    'parameter': measurement,
                    'measurement_type': f'dc_{measurement}' if measurement == 'resistance' else f'ac_{measurement}',
                    'expected_value': param_data['value'],
                    'tolerance': self._get_tolerance(measurement, validated_params),
                    'measurement_time_ms': self._estimate_measurement_time(measurement)
                })
        
        # Add required equipment
        if any('resistance' in seq['parameter'] for seq in plan['measurement_sequence']):
            plan['required_equipment'].append('digital_multimeter')
        
        if any(param in ['capacitance', 'inductance'] for seq in plan['measurement_sequence'] for param in [seq['parameter']]):
            plan['required_equipment'].append('lcr_meter')
        
        if any('voltage' in str(validated_params.get(param, {})) for param in validated_params):
            plan['required_equipment'].append('power_supply')
        
        # Set measurement settings
        plan['measurement_settings'] = {
            'measurement_frequency_hz': 1000,  # 1kHz for LCR measurements
            'measurement_voltage_v': 1.0,      # 1V for AC measurements
            'settling_time_ms': 100,           # Allow 100ms settling
            'averaging_samples': 10            # Average 10 samples
        }
        
        # Pass/fail criteria
        for seq in plan['measurement_sequence']:
            param = seq['parameter']
            expected = seq['expected_value']
            tolerance = seq['tolerance']
            
            plan['pass_fail_criteria'][param] = {
                'min_value': expected * (1 - tolerance/100),
                'max_value': expected * (1 + tolerance/100),
                'measurement_uncertainty': tolerance / 10  # Measurement should be 10x better than tolerance
            }
        
        return plan
    
    def _get_tolerance(self, parameter: str, validated_params: Dict[str, Any]) -> float:
        """Get tolerance for a parameter (in percent)."""
        tolerance_field = f"{parameter}_tolerance"
        if tolerance_field in validated_params:
            return validated_params[tolerance_field]['value'] * 100
        
        # Default tolerances if not specified
        default_tolerances = {
            'resistance': 5.0,      # 5%
            'capacitance': 10.0,    # 10%
            'inductance': 20.0,     # 20%
            'voltage_rating': 5.0   # 5%
        }
        
        return default_tolerances.get(parameter, 10.0)
    
    def _estimate_measurement_time(self, measurement_type: str) -> int:
        """Estimate measurement time in milliseconds."""
        measurement_times = {
            'resistance': 50,    # 50ms for DC resistance
            'capacitance': 100,  # 100ms for AC capacitance
            'inductance': 150,   # 150ms for AC inductance
            'voltage': 25,       # 25ms for DC voltage
            'current': 25        # 25ms for DC current
        }
        
        return measurement_times.get(measurement_type, 100)