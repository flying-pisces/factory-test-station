"""Resistor Component Type Processor - Week 2 Implementation."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import math


class ResistorProcessor:
    """Specialized processor for resistor components with manufacturing-specific analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger('ResistorProcessor')
        
        # Standard resistor series for value validation
        self.e_series = {
            'E12': [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
            'E24': [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1],
            'E96': [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24,
                    1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58]  # Truncated for brevity
        }
        
        # Power rating to package mapping for SMT resistors
        self.power_to_package = {
            0.0625: ['0201'],           # 1/16W
            0.1: ['0402'],              # 1/10W
            0.25: ['0603'],             # 1/4W
            0.5: ['0805', '1206'],      # 1/2W
            0.75: ['1210'],             # 3/4W
            1.0: ['2010', '2512'],      # 1W
            2.0: ['2512', '2725'],      # 2W
            3.0: ['2725']               # 3W
        }
        
        self.logger.info("ResistorProcessor initialized with E-series validation and power/package mapping")
    
    def process(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process resistor-specific manufacturing requirements."""
        try:
            result = {
                'component_type': 'Resistor',
                'processing_status': 'success'
            }
            
            # Extract resistor-specific parameters
            ee_data = processed_data.get('ee_data', {})
            cad_data = processed_data.get('cad_data', {})
            api_data = processed_data.get('api_data', {})
            
            # Process resistance value and tolerance
            resistance_analysis = self._analyze_resistance_value(ee_data)
            result['resistance_analysis'] = resistance_analysis
            
            # Power analysis and thermal considerations
            power_analysis = self._analyze_power_characteristics(ee_data, cad_data)
            result['power_analysis'] = power_analysis
            
            # Package validation and optimization
            package_analysis = self._analyze_package_selection(cad_data, power_analysis)
            result['package_analysis'] = package_analysis
            
            # Manufacturing process requirements
            manufacturing_req = self._determine_manufacturing_requirements(
                resistance_analysis, power_analysis, package_analysis
            )
            result['manufacturing_requirements'] = manufacturing_req
            
            # Test strategy for production
            test_strategy = self._create_test_strategy(resistance_analysis, power_analysis)
            result['test_strategy'] = test_strategy
            
            # Quality and reliability assessment
            quality_assessment = self._assess_quality_reliability(ee_data, cad_data, api_data)
            result['quality_assessment'] = quality_assessment
            
            return result
            
        except Exception as e:
            self.logger.error(f"Resistor processing failed: {e}")
            return {
                'component_type': 'Resistor',
                'processing_status': 'failed',
                'error': str(e),
                'manufacturing_requirements': {
                    'placement_profile': 'manual_place',
                    'estimated_time_s': 5.0
                }
            }
    
    def _analyze_resistance_value(self, ee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resistance value for standard series compliance."""
        analysis = {}
        
        resistance_param = ee_data.get('validated_parameters', {}).get('resistance', {})
        resistance_value = resistance_param.get('value', 0)
        
        if resistance_value <= 0:
            return {
                'value_ohms': 0,
                'series_compliance': 'invalid',
                'decade': 0,
                'normalized_value': 0,
                'standard_series': None
            }
        
        # Determine decade and normalized value
        decade = math.floor(math.log10(resistance_value))
        normalized_value = resistance_value / (10 ** decade)
        
        # Check E-series compliance
        series_compliance = self._check_e_series_compliance(normalized_value)
        
        analysis.update({
            'value_ohms': resistance_value,
            'decade': decade,
            'normalized_value': normalized_value,
            'series_compliance': series_compliance,
            'value_category': self._categorize_resistance_value(resistance_value)
        })
        
        # Tolerance analysis
        tolerance_param = ee_data.get('validated_parameters', {}).get('resistance_tolerance', {})
        if tolerance_param.get('valid'):
            tolerance_percent = tolerance_param['value'] * 100
            analysis['tolerance_percent'] = tolerance_percent
            analysis['tolerance_class'] = self._classify_tolerance(tolerance_percent)
        else:
            analysis['tolerance_percent'] = 5.0  # Default 5%
            analysis['tolerance_class'] = 'standard'
        
        return analysis
    
    def _analyze_power_characteristics(self, ee_data: Dict[str, Any], cad_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze power characteristics and thermal requirements."""
        analysis = {}
        
        # Extract power rating
        power_param = ee_data.get('validated_parameters', {}).get('power_rating', {})
        power_rating = power_param.get('value', 0.25)  # Default 1/4W
        
        analysis['power_rating_w'] = power_rating
        analysis['power_category'] = self._categorize_power_rating(power_rating)
        
        # Thermal analysis based on package dimensions
        dimensions = cad_data.get('processed_dimensions', {})
        if dimensions:
            footprint_area = dimensions.get('footprint_area_mm2', 1.0)
            
            # Power density calculation
            power_density = power_rating / footprint_area if footprint_area > 0 else 0
            analysis['power_density_w_per_mm2'] = power_density
            
            # Thermal resistance estimation (simplified)
            if power_density > 0.5:
                analysis['thermal_considerations'] = 'high_power_density'
                analysis['cooling_required'] = True
            elif power_density > 0.1:
                analysis['thermal_considerations'] = 'medium_power_density'
                analysis['cooling_required'] = False
            else:
                analysis['thermal_considerations'] = 'low_power_density'
                analysis['cooling_required'] = False
        
        # Temperature coefficient analysis
        temp_coeff_param = ee_data.get('validated_parameters', {}).get('temperature_coefficient', {})
        if temp_coeff_param.get('valid'):
            temp_coeff_ppm = temp_coeff_param['value']
            analysis['temp_coefficient_ppm_per_c'] = temp_coeff_ppm
            analysis['temp_stability'] = self._classify_temp_stability(abs(temp_coeff_ppm))
        else:
            analysis['temp_coefficient_ppm_per_c'] = 200  # Typical for standard resistors
            analysis['temp_stability'] = 'standard'
        
        return analysis
    
    def _analyze_package_selection(self, cad_data: Dict[str, Any], power_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze package selection and validate against power requirements."""
        analysis = {}
        
        package_info = cad_data.get('package_info', {})
        package_name = package_info.get('package_name', 'UNKNOWN')
        power_rating = power_analysis.get('power_rating_w', 0.25)
        
        analysis['current_package'] = package_name
        analysis['power_rating_w'] = power_rating
        
        # Find appropriate packages for power rating
        suitable_packages = []
        for power, packages in self.power_to_package.items():
            if power >= power_rating:
                suitable_packages.extend(packages)
        
        analysis['suitable_packages'] = list(set(suitable_packages))
        
        # Validate current package
        current_package_size = self._extract_package_size(package_name)
        if current_package_size in suitable_packages:
            analysis['package_validation'] = 'optimal'
        elif any(size in package_name for size in suitable_packages):
            analysis['package_validation'] = 'acceptable'
        else:
            analysis['package_validation'] = 'oversized' if suitable_packages and \
                self._package_size_comparison(current_package_size, suitable_packages[0]) > 0 else 'undersized'
        
        # Package-specific manufacturing considerations
        analysis['manufacturing_considerations'] = self._get_package_manufacturing_notes(current_package_size)
        
        return analysis
    
    def _determine_manufacturing_requirements(self, resistance_analysis: Dict[str, Any], 
                                           power_analysis: Dict[str, Any],
                                           package_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine specific manufacturing requirements for resistor placement."""
        requirements = {}
        
        # Basic placement profile
        requirements['placement_profile'] = 'smt_place_passive'
        
        # Package-based requirements
        current_package = package_analysis.get('current_package', '0603')
        package_size = self._extract_package_size(current_package)
        
        if package_size in ['0201', '0402']:
            requirements['placement_complexity'] = 'high'
            requirements['vision_alignment_required'] = True
            requirements['placement_accuracy_um'] = 15
            requirements['nozzle_type'] = 'micro'
            requirements['placement_force_n'] = 1.0
            requirements['placement_time_s'] = 0.8
        elif package_size in ['0603', '0805']:
            requirements['placement_complexity'] = 'medium'
            requirements['vision_alignment_required'] = False
            requirements['placement_accuracy_um'] = 25
            requirements['nozzle_type'] = 'standard'
            requirements['placement_force_n'] = 2.0
            requirements['placement_time_s'] = 0.5
        elif package_size in ['1206', '2010', '2512']:
            requirements['placement_complexity'] = 'low'
            requirements['vision_alignment_required'] = False
            requirements['placement_accuracy_um'] = 50
            requirements['nozzle_type'] = 'standard'
            requirements['placement_force_n'] = 3.0
            requirements['placement_time_s'] = 0.4
        else:
            # Default for unknown packages
            requirements['placement_complexity'] = 'medium'
            requirements['vision_alignment_required'] = True
            requirements['placement_accuracy_um'] = 25
            requirements['nozzle_type'] = 'standard'
            requirements['placement_force_n'] = 2.0
            requirements['placement_time_s'] = 0.6
        
        # Power-based thermal considerations
        if power_analysis.get('cooling_required', False):
            requirements['thermal_management'] = 'active_cooling'
            requirements['placement_considerations'] = ['thermal_via_proximity', 'heat_sink_access']
        else:
            requirements['thermal_management'] = 'passive'
            requirements['placement_considerations'] = ['standard_placement']
        
        # Precision requirements based on tolerance
        tolerance_class = resistance_analysis.get('tolerance_class', 'standard')
        if tolerance_class == 'precision':
            requirements['handling_precautions'] = ['esd_protection', 'humidity_control']
        else:
            requirements['handling_precautions'] = ['standard_handling']
        
        return requirements
    
    def _create_test_strategy(self, resistance_analysis: Dict[str, Any], 
                            power_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create manufacturing test strategy for resistors."""
        strategy = {
            'test_sequence': [],
            'test_equipment': ['digital_multimeter'],
            'test_settings': {}
        }
        
        # Primary resistance measurement
        resistance_value = resistance_analysis.get('value_ohms', 0)
        tolerance = resistance_analysis.get('tolerance_percent', 5.0)
        
        strategy['test_sequence'].append({
            'test_name': 'resistance_measurement',
            'test_type': 'dc_resistance',
            'expected_value': resistance_value,
            'tolerance_percent': tolerance,
            'test_current_ma': self._calculate_test_current(resistance_value),
            'measurement_time_ms': 50,
            'priority': 'critical'
        })
        
        # Power handling test for high-power resistors
        power_rating = power_analysis.get('power_rating_w', 0)
        if power_rating > 0.5:  # Test power handling for >0.5W resistors
            strategy['test_sequence'].append({
                'test_name': 'power_handling_test',
                'test_type': 'power_dissipation',
                'test_power_w': power_rating * 0.7,  # 70% of rated power
                'duration_s': 30,
                'temperature_monitoring': True,
                'priority': 'important'
            })
            strategy['test_equipment'].append('power_supply')
            strategy['test_equipment'].append('thermal_camera')
        
        # Temperature coefficient test for precision resistors
        if resistance_analysis.get('tolerance_class') == 'precision':
            strategy['test_sequence'].append({
                'test_name': 'temperature_coefficient_test',
                'test_type': 'temperature_sweep',
                'temp_range_c': [-25, 85],
                'measurement_points': 5,
                'expected_drift_ppm': power_analysis.get('temp_coefficient_ppm_per_c', 200),
                'priority': 'optional'
            })
            strategy['test_equipment'].append('temperature_chamber')
        
        # Test settings
        strategy['test_settings'] = {
            'measurement_frequency_hz': 0,  # DC measurement
            'settling_time_ms': 100,
            'averaging_samples': 5,
            'environmental_conditions': {
                'temperature_c': 23,
                'humidity_percent': 45
            }
        }
        
        return strategy
    
    def _assess_quality_reliability(self, ee_data: Dict[str, Any], cad_data: Dict[str, Any], 
                                  api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality and reliability characteristics."""
        assessment = {
            'quality_indicators': [],
            'reliability_factors': [],
            'risk_factors': []
        }
        
        # Tolerance-based quality assessment
        tolerance = ee_data.get('validated_parameters', {}).get('resistance_tolerance', {}).get('value', 0.05)
        if tolerance <= 0.01:  # ≤1%
            assessment['quality_indicators'].append('precision_tolerance')
        elif tolerance <= 0.05:  # ≤5%
            assessment['quality_indicators'].append('standard_tolerance')
        else:
            assessment['quality_indicators'].append('wide_tolerance')
        
        # Power rating assessment
        power_rating = ee_data.get('validated_parameters', {}).get('power_rating', {}).get('value', 0.25)
        if power_rating >= 1.0:
            assessment['reliability_factors'].append('high_power_rated')
        elif power_rating >= 0.25:
            assessment['reliability_factors'].append('standard_power_rated')
        else:
            assessment['risk_factors'].append('low_power_rating')
        
        # Temperature stability assessment
        temp_coeff = ee_data.get('validated_parameters', {}).get('temperature_coefficient', {}).get('value', 200)
        if abs(temp_coeff) <= 50:
            assessment['quality_indicators'].append('excellent_temp_stability')
        elif abs(temp_coeff) <= 200:
            assessment['quality_indicators'].append('good_temp_stability')
        else:
            assessment['risk_factors'].append('poor_temp_stability')
        
        # Package size reliability
        package_size = self._extract_package_size(cad_data.get('package_info', {}).get('package_name', '0603'))
        if package_size in ['0201', '0402']:
            assessment['risk_factors'].append('micro_component_handling_risk')
        elif package_size in ['2512', '2725']:
            assessment['reliability_factors'].append('robust_large_package')
        
        # Overall assessment
        quality_score = len(assessment['quality_indicators']) * 2 + len(assessment['reliability_factors'])
        risk_score = len(assessment['risk_factors'])
        
        if quality_score >= 4 and risk_score <= 1:
            assessment['overall_rating'] = 'excellent'
        elif quality_score >= 2 and risk_score <= 2:
            assessment['overall_rating'] = 'good'
        elif risk_score <= 3:
            assessment['overall_rating'] = 'acceptable'
        else:
            assessment['overall_rating'] = 'poor'
        
        return assessment
    
    def _check_e_series_compliance(self, normalized_value: float) -> Dict[str, Any]:
        """Check compliance with standard E-series values."""
        compliance = {'series': None, 'compliant': False, 'closest_value': None}
        
        # Check each E-series
        for series_name, values in self.e_series.items():
            closest_value = min(values, key=lambda x: abs(x - normalized_value))
            tolerance = abs(closest_value - normalized_value) / closest_value
            
            if tolerance <= 0.02:  # Within 2% is considered compliant
                compliance['series'] = series_name
                compliance['compliant'] = True
                compliance['closest_value'] = closest_value
                break
        
        if not compliance['compliant']:
            # Find overall closest value
            all_values = []
            for values in self.e_series.values():
                all_values.extend(values)
            compliance['closest_value'] = min(all_values, key=lambda x: abs(x - normalized_value))
        
        return compliance
    
    def _categorize_resistance_value(self, resistance: float) -> str:
        """Categorize resistance value for manufacturing considerations."""
        if resistance < 1:
            return 'milliohm'
        elif resistance < 100:
            return 'low_ohm'
        elif resistance < 10000:
            return 'medium_ohm' 
        elif resistance < 1e6:
            return 'high_ohm'
        else:
            return 'megaohm'
    
    def _categorize_power_rating(self, power: float) -> str:
        """Categorize power rating for thermal analysis."""
        if power <= 0.1:
            return 'low_power'
        elif power <= 0.5:
            return 'standard_power'
        elif power <= 2.0:
            return 'medium_power'
        else:
            return 'high_power'
    
    def _classify_tolerance(self, tolerance_percent: float) -> str:
        """Classify tolerance for precision requirements."""
        if tolerance_percent <= 1.0:
            return 'precision'
        elif tolerance_percent <= 5.0:
            return 'standard'
        else:
            return 'wide'
    
    def _classify_temp_stability(self, temp_coeff_ppm: float) -> str:
        """Classify temperature stability."""
        if temp_coeff_ppm <= 50:
            return 'excellent'
        elif temp_coeff_ppm <= 200:
            return 'good'
        elif temp_coeff_ppm <= 500:
            return 'standard'
        else:
            return 'poor'
    
    def _extract_package_size(self, package_name: str) -> str:
        """Extract package size from package name."""
        common_sizes = ['0201', '0402', '0603', '0805', '1206', '1210', '2010', '2512', '2725']
        for size in common_sizes:
            if size in package_name:
                return size
        return '0603'  # Default
    
    def _package_size_comparison(self, size1: str, size2: str) -> int:
        """Compare package sizes (-1: size1 < size2, 0: equal, 1: size1 > size2)."""
        size_order = ['0201', '0402', '0603', '0805', '1206', '1210', '2010', '2512', '2725']
        try:
            idx1 = size_order.index(size1)
            idx2 = size_order.index(size2)
            return (idx1 > idx2) - (idx1 < idx2)
        except ValueError:
            return 0
    
    def _get_package_manufacturing_notes(self, package_size: str) -> List[str]:
        """Get manufacturing considerations for package size."""
        notes = {
            '0201': ['extreme_precision_required', 'micro_nozzle_mandatory', 'vision_critical'],
            '0402': ['high_precision_required', 'micro_nozzle_recommended', 'careful_handling'],
            '0603': ['standard_placement', 'good_manufacturability'],
            '0805': ['standard_placement', 'excellent_manufacturability', 'robust_handling'],
            '1206': ['easy_placement', 'robust_package', 'good_thermal_mass'],
            '2512': ['large_component', 'excellent_power_handling', 'may_require_large_nozzle']
        }
        return notes.get(package_size, ['standard_considerations'])
    
    def _calculate_test_current(self, resistance: float) -> float:
        """Calculate appropriate test current for resistance measurement."""
        # Use 1mA for most resistors, adjust for very low/high values
        if resistance < 1:  # <1Ω
            return 10.0  # 10mA for low resistance
        elif resistance < 1000:  # <1kΩ
            return 1.0   # 1mA standard
        elif resistance < 1e6:  # <1MΩ
            return 0.1   # 100µA for high resistance
        else:  # >1MΩ
            return 0.01  # 10µA for very high resistance