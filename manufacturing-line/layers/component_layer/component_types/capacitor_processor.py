"""Capacitor Component Type Processor - Week 2 Implementation."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import math


class CapacitorProcessor:
    """Specialized processor for capacitor components with manufacturing-specific analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger('CapacitorProcessor')
        
        # Capacitor types and their characteristics
        self.capacitor_types = {
            'ceramic': {
                'dielectric_classes': ['C0G/NP0', 'X7R', 'X5R', 'Y5V', 'Z5U'],
                'typical_tolerance': [0.05, 0.10, 0.20],  # 5%, 10%, 20%
                'voltage_ratings': [6.3, 10, 16, 25, 50, 100, 200, 500, 1000],
                'temp_stability': 'good_to_poor'
            },
            'tantalum': {
                'typical_tolerance': [0.10, 0.20],  # 10%, 20%
                'voltage_ratings': [4, 6.3, 10, 16, 20, 25, 35, 50],
                'temp_stability': 'good',
                'polarity': 'polarized'
            },
            'aluminum_electrolytic': {
                'typical_tolerance': [0.20, 0.50],  # 20%, 50%
                'voltage_ratings': [6.3, 10, 16, 25, 35, 50, 63, 100, 200, 400, 450],
                'temp_stability': 'poor',
                'polarity': 'polarized'
            },
            'film': {
                'types': ['polyester', 'polypropylene', 'polystyrene'],
                'typical_tolerance': [0.05, 0.10],  # 5%, 10%
                'voltage_ratings': [50, 100, 250, 400, 630, 1000],
                'temp_stability': 'excellent'
            }
        }
        
        # Package size to capacitance mapping (ceramic capacitors)
        self.ceramic_capacitance_limits = {
            '0201': {'max_pf': 1000, 'max_voltage': 25},
            '0402': {'max_pf': 10000, 'max_voltage': 50},  # 10nF
            '0603': {'max_pf': 100000, 'max_voltage': 100},  # 100nF
            '0805': {'max_pf': 1000000, 'max_voltage': 200},  # 1µF
            '1206': {'max_pf': 10000000, 'max_voltage': 500},  # 10µF
            '1210': {'max_pf': 22000000, 'max_voltage': 100}  # 22µF
        }
        
        self.logger.info("CapacitorProcessor initialized with dielectric type analysis")
    
    def process(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process capacitor-specific manufacturing requirements."""
        try:
            result = {
                'component_type': 'Capacitor',
                'processing_status': 'success'
            }
            
            # Extract capacitor-specific parameters
            ee_data = processed_data.get('ee_data', {})
            cad_data = processed_data.get('cad_data', {})
            api_data = processed_data.get('api_data', {})
            
            # Analyze capacitance value and characteristics
            capacitance_analysis = self._analyze_capacitance_value(ee_data)
            result['capacitance_analysis'] = capacitance_analysis
            
            # Determine dielectric type and characteristics
            dielectric_analysis = self._analyze_dielectric_type(ee_data, api_data)
            result['dielectric_analysis'] = dielectric_analysis
            
            # Voltage rating and safety analysis
            voltage_analysis = self._analyze_voltage_characteristics(ee_data)
            result['voltage_analysis'] = voltage_analysis
            
            # Package validation and optimization
            package_analysis = self._analyze_package_selection(cad_data, capacitance_analysis, voltage_analysis)
            result['package_analysis'] = package_analysis
            
            # Manufacturing process requirements
            manufacturing_req = self._determine_manufacturing_requirements(
                capacitance_analysis, dielectric_analysis, package_analysis
            )
            result['manufacturing_requirements'] = manufacturing_req
            
            # Test strategy for production
            test_strategy = self._create_test_strategy(capacitance_analysis, voltage_analysis, dielectric_analysis)
            result['test_strategy'] = test_strategy
            
            # Quality and reliability assessment
            quality_assessment = self._assess_quality_reliability(
                capacitance_analysis, dielectric_analysis, voltage_analysis, api_data
            )
            result['quality_assessment'] = quality_assessment
            
            return result
            
        except Exception as e:
            self.logger.error(f"Capacitor processing failed: {e}")
            return {
                'component_type': 'Capacitor',
                'processing_status': 'failed',
                'error': str(e),
                'manufacturing_requirements': {
                    'placement_profile': 'smt_place_passive',
                    'estimated_time_s': 0.6
                }
            }
    
    def _analyze_capacitance_value(self, ee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capacitance value and classify the component."""
        analysis = {}
        
        capacitance_param = ee_data.get('validated_parameters', {}).get('capacitance', {})
        capacitance_value = capacitance_param.get('value', 0)
        
        if capacitance_value <= 0:
            return {
                'value_f': 0,
                'value_pf': 0,
                'category': 'invalid',
                'standard_notation': 'invalid'
            }
        
        # Convert to various units for analysis
        value_pf = capacitance_value * 1e12  # Convert to picofarads
        value_nf = capacitance_value * 1e9   # Convert to nanofarads
        value_uf = capacitance_value * 1e6   # Convert to microfarads
        
        analysis.update({
            'value_f': capacitance_value,
            'value_pf': value_pf,
            'value_nf': value_nf,
            'value_uf': value_uf
        })
        
        # Categorize by value range
        if value_pf < 100:
            analysis['category'] = 'ultra_low'
            analysis['typical_application'] = 'rf_tuning'
        elif value_pf < 1000:  # <1nF
            analysis['category'] = 'low'
            analysis['typical_application'] = 'coupling_filtering'
        elif value_pf < 100000:  # <100nF
            analysis['category'] = 'medium'
            analysis['typical_application'] = 'decoupling_bypass'
        elif value_pf < 10000000:  # <10µF
            analysis['category'] = 'high'
            analysis['typical_application'] = 'bulk_decoupling'
        else:  # ≥10µF
            analysis['category'] = 'very_high'
            analysis['typical_application'] = 'energy_storage'
        
        # Standard notation
        if value_pf < 1000:
            analysis['standard_notation'] = f"{value_pf:.0f}pF"
        elif value_nf < 1000:
            analysis['standard_notation'] = f"{value_nf:.1f}nF"
        else:
            analysis['standard_notation'] = f"{value_uf:.1f}µF"
        
        # Tolerance analysis
        tolerance_param = ee_data.get('validated_parameters', {}).get('tolerance', {})
        if tolerance_param.get('valid'):
            tolerance_percent = tolerance_param['value'] * 100
        else:
            # Infer typical tolerance based on value
            if value_pf < 1000:
                tolerance_percent = 5.0   # 5% for small values
            elif value_pf < 100000:
                tolerance_percent = 10.0  # 10% for medium values
            else:
                tolerance_percent = 20.0  # 20% for large values
        
        analysis['tolerance_percent'] = tolerance_percent
        analysis['tolerance_class'] = self._classify_tolerance(tolerance_percent)
        
        return analysis
    
    def _analyze_dielectric_type(self, ee_data: Dict[str, Any], api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine dielectric type and characteristics."""
        analysis = {
            'type': 'unknown',
            'temp_coefficient': 'unknown',
            'characteristics': []
        }
        
        # Try to determine from API data specifications
        specs = api_data.get('specifications', {})
        
        # Look for dielectric class indicators
        for spec_key, spec_value in specs.items():
            spec_str = str(spec_value).upper()
            
            if any(cls in spec_str for cls in ['C0G', 'NP0']):
                analysis['type'] = 'ceramic'
                analysis['dielectric_class'] = 'C0G/NP0'
                analysis['temp_coefficient'] = 'excellent'
                analysis['characteristics'] = ['stable', 'low_loss', 'precise']
                break
            elif any(cls in spec_str for cls in ['X7R', 'X5R']):
                analysis['type'] = 'ceramic'
                analysis['dielectric_class'] = spec_str
                analysis['temp_coefficient'] = 'good'
                analysis['characteristics'] = ['stable', 'general_purpose']
                break
            elif any(cls in spec_str for cls in ['Y5V', 'Z5U']):
                analysis['type'] = 'ceramic'
                analysis['dielectric_class'] = spec_str
                analysis['temp_coefficient'] = 'poor'
                analysis['characteristics'] = ['high_capacitance', 'temperature_sensitive']
                break
            elif 'TANTALUM' in spec_str:
                analysis['type'] = 'tantalum'
                analysis['temp_coefficient'] = 'good'
                analysis['characteristics'] = ['polarized', 'stable', 'low_esr']
                break
            elif any(term in spec_str for term in ['ELECTROLYTIC', 'ALUMINUM']):
                analysis['type'] = 'aluminum_electrolytic'
                analysis['temp_coefficient'] = 'poor'
                analysis['characteristics'] = ['polarized', 'high_capacitance', 'high_esr']
                break
            elif any(term in spec_str for term in ['FILM', 'POLYESTER', 'POLYPROPYLENE']):
                analysis['type'] = 'film'
                analysis['temp_coefficient'] = 'excellent'
                analysis['characteristics'] = ['stable', 'low_loss', 'high_voltage']
                break
        
        # If still unknown, infer from capacitance value and voltage rating
        if analysis['type'] == 'unknown':
            capacitance_pf = ee_data.get('validated_parameters', {}).get('capacitance', {}).get('value', 0) * 1e12
            voltage_rating = ee_data.get('validated_parameters', {}).get('voltage_rating', {}).get('value', 0)
            
            if capacitance_pf > 1e6 and voltage_rating < 50:  # >1µF and <50V
                analysis['type'] = 'ceramic'  # Likely high-K ceramic
                analysis['characteristics'] = ['high_capacitance']
            elif capacitance_pf > 1e7:  # >10µF
                if voltage_rating < 50:
                    analysis['type'] = 'tantalum'
                    analysis['characteristics'] = ['polarized']
                else:
                    analysis['type'] = 'aluminum_electrolytic'
                    analysis['characteristics'] = ['polarized', 'high_voltage']
            else:
                analysis['type'] = 'ceramic'
                analysis['characteristics'] = ['general_purpose']
        
        # Add manufacturing implications
        if 'polarized' in analysis['characteristics']:
            analysis['polarity_sensitive'] = True
            analysis['placement_orientation_critical'] = True
        else:
            analysis['polarity_sensitive'] = False
            analysis['placement_orientation_critical'] = False
        
        return analysis
    
    def _analyze_voltage_characteristics(self, ee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voltage rating and safety characteristics."""
        analysis = {}
        
        voltage_param = ee_data.get('validated_parameters', {}).get('voltage_rating', {})
        voltage_rating = voltage_param.get('value', 0)
        
        analysis['voltage_rating_v'] = voltage_rating
        analysis['voltage_category'] = self._categorize_voltage_rating(voltage_rating)
        
        # Derating recommendations
        analysis['recommended_operating_voltage_v'] = voltage_rating * 0.7  # 70% derating
        analysis['absolute_max_voltage_v'] = voltage_rating * 0.9  # 90% absolute max
        
        # Safety considerations
        if voltage_rating > 500:
            analysis['safety_class'] = 'high_voltage'
            analysis['safety_requirements'] = ['isolation', 'high_voltage_handling', 'safety_testing']
        elif voltage_rating > 100:
            analysis['safety_class'] = 'medium_voltage'
            analysis['safety_requirements'] = ['standard_handling', 'voltage_testing']
        else:
            analysis['safety_class'] = 'low_voltage'
            analysis['safety_requirements'] = ['standard_handling']
        
        # Ripple current analysis (if available)
        ripple_current_param = ee_data.get('validated_parameters', {}).get('ripple_current', {})
        if ripple_current_param.get('valid'):
            ripple_current = ripple_current_param['value']
            analysis['ripple_current_a'] = ripple_current
            analysis['ripple_capable'] = True
        else:
            analysis['ripple_current_a'] = 0
            analysis['ripple_capable'] = False
        
        return analysis
    
    def _analyze_package_selection(self, cad_data: Dict[str, Any], capacitance_analysis: Dict[str, Any], 
                                 voltage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze package selection and validate against electrical requirements."""
        analysis = {}
        
        package_info = cad_data.get('package_info', {})
        package_name = package_info.get('package_name', 'UNKNOWN')
        
        analysis['current_package'] = package_name
        
        # Extract package size
        package_size = self._extract_package_size(package_name)
        analysis['package_size'] = package_size
        
        # Validate package against electrical requirements (for ceramic capacitors)
        capacitance_pf = capacitance_analysis.get('value_pf', 0)
        voltage_rating = voltage_analysis.get('voltage_rating_v', 0)
        
        if package_size in self.ceramic_capacitance_limits:
            limits = self.ceramic_capacitance_limits[package_size]
            
            if capacitance_pf > limits['max_pf']:
                analysis['package_validation'] = 'too_small_for_capacitance'
                analysis['recommended_min_size'] = self._find_min_package_for_capacitance(capacitance_pf)
            elif voltage_rating > limits['max_voltage']:
                analysis['package_validation'] = 'too_small_for_voltage'
                analysis['recommended_min_size'] = self._find_min_package_for_voltage(voltage_rating)
            else:
                analysis['package_validation'] = 'optimal'
        else:
            analysis['package_validation'] = 'unknown_package'
        
        # Package-specific manufacturing considerations
        analysis['manufacturing_considerations'] = self._get_capacitor_package_notes(package_size, capacitance_analysis)
        
        return analysis
    
    def _determine_manufacturing_requirements(self, capacitance_analysis: Dict[str, Any],
                                           dielectric_analysis: Dict[str, Any],
                                           package_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine specific manufacturing requirements for capacitor placement."""
        requirements = {}
        
        # Basic placement profile
        requirements['placement_profile'] = 'smt_place_passive'
        
        # Package-based requirements
        package_size = package_analysis.get('package_size', '0603')
        
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
            requirements['placement_accuracy_um'] = 20
            requirements['nozzle_type'] = 'standard'
            requirements['placement_force_n'] = 2.0
            requirements['placement_time_s'] = 0.6
        elif package_size in ['1206', '1210']:
            requirements['placement_complexity'] = 'low'
            requirements['vision_alignment_required'] = False
            requirements['placement_accuracy_um'] = 25
            requirements['nozzle_type'] = 'standard'
            requirements['placement_force_n'] = 2.5
            requirements['placement_time_s'] = 0.5
        else:
            requirements['placement_complexity'] = 'medium'
            requirements['vision_alignment_required'] = True
            requirements['placement_accuracy_um'] = 20
            requirements['nozzle_type'] = 'standard'
            requirements['placement_force_n'] = 2.0
            requirements['placement_time_s'] = 0.6
        
        # Polarity-sensitive handling
        if dielectric_analysis.get('polarity_sensitive', False):
            requirements['polarity_sensitive'] = True
            requirements['orientation_verification'] = True
            requirements['placement_considerations'] = ['polarity_check', 'vision_verification']
            requirements['vision_alignment_required'] = True
            requirements['placement_time_s'] += 0.3  # Extra time for polarity verification
        else:
            requirements['polarity_sensitive'] = False
            requirements['placement_considerations'] = ['standard_placement']
        
        # High-voltage considerations
        voltage_category = capacitance_analysis.get('voltage_category', 'low')
        if voltage_category in ['high', 'very_high']:
            requirements['high_voltage_component'] = True
            requirements['handling_precautions'] = ['isolation_required', 'safety_clearance']
        else:
            requirements['high_voltage_component'] = False
            requirements['handling_precautions'] = ['standard_handling']
        
        # ESD sensitivity based on capacitance value
        if capacitance_analysis.get('value_pf', 0) < 1000:  # <1nF
            requirements['esd_sensitive'] = True
            requirements['esd_protection_required'] = True
        else:
            requirements['esd_sensitive'] = False
            requirements['esd_protection_required'] = False
        
        return requirements
    
    def _create_test_strategy(self, capacitance_analysis: Dict[str, Any],
                            voltage_analysis: Dict[str, Any],
                            dielectric_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create manufacturing test strategy for capacitors."""
        strategy = {
            'test_sequence': [],
            'test_equipment': ['lcr_meter'],
            'test_settings': {}
        }
        
        # Primary capacitance measurement
        capacitance_value = capacitance_analysis.get('value_f', 0)
        tolerance = capacitance_analysis.get('tolerance_percent', 10.0)
        
        strategy['test_sequence'].append({
            'test_name': 'capacitance_measurement',
            'test_type': 'ac_capacitance',
            'expected_value': capacitance_value,
            'tolerance_percent': tolerance,
            'test_frequency_hz': self._get_optimal_test_frequency(capacitance_value),
            'test_voltage_v': 1.0,
            'measurement_time_ms': 100,
            'priority': 'critical'
        })
        
        # Dielectric strength test for high-voltage capacitors
        voltage_rating = voltage_analysis.get('voltage_rating_v', 0)
        if voltage_rating > 100:
            strategy['test_sequence'].append({
                'test_name': 'dielectric_strength_test',
                'test_type': 'high_voltage_test',
                'test_voltage_v': voltage_rating * 0.8,  # 80% of rated voltage
                'duration_ms': 1000,
                'leakage_current_limit_ua': 1.0,
                'priority': 'important'
            })
            strategy['test_equipment'].append('high_voltage_tester')
        
        # ESR measurement for electrolytic capacitors
        if dielectric_analysis.get('type') in ['tantalum', 'aluminum_electrolytic']:
            strategy['test_sequence'].append({
                'test_name': 'esr_measurement',
                'test_type': 'equivalent_series_resistance',
                'test_frequency_hz': 100000,  # 100kHz
                'max_esr_ohm': self._estimate_max_esr(capacitance_value, dielectric_analysis['type']),
                'priority': 'important'
            })
        
        # Polarity test for polarized capacitors
        if dielectric_analysis.get('polarity_sensitive', False):
            strategy['test_sequence'].append({
                'test_name': 'polarity_verification',
                'test_type': 'polarity_check',
                'test_method': 'visual_inspection',
                'automated': True,
                'priority': 'critical'
            })
            strategy['test_equipment'].append('vision_system')
        
        # Temperature coefficient test for precision capacitors
        if capacitance_analysis.get('tolerance_class') == 'precision':
            strategy['test_sequence'].append({
                'test_name': 'temperature_coefficient_test',
                'test_type': 'temperature_sweep',
                'temp_range_c': [-25, 85],
                'measurement_points': 3,
                'max_drift_percent': 2.0,
                'priority': 'optional'
            })
            strategy['test_equipment'].append('temperature_chamber')
        
        # Test settings
        strategy['test_settings'] = {
            'default_measurement_frequency_hz': 1000,
            'settling_time_ms': 200,  # Longer for capacitors
            'averaging_samples': 10,
            'environmental_conditions': {
                'temperature_c': 23,
                'humidity_percent': 45
            }
        }
        
        return strategy
    
    def _assess_quality_reliability(self, capacitance_analysis: Dict[str, Any],
                                  dielectric_analysis: Dict[str, Any],
                                  voltage_analysis: Dict[str, Any],
                                  api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality and reliability characteristics."""
        assessment = {
            'quality_indicators': [],
            'reliability_factors': [],
            'risk_factors': []
        }
        
        # Tolerance-based quality assessment
        tolerance = capacitance_analysis.get('tolerance_percent', 10.0)
        if tolerance <= 5.0:
            assessment['quality_indicators'].append('tight_tolerance')
        elif tolerance <= 10.0:
            assessment['quality_indicators'].append('standard_tolerance')
        else:
            assessment['quality_indicators'].append('wide_tolerance')
        
        # Dielectric type reliability
        dielectric_type = dielectric_analysis.get('type', 'unknown')
        temp_coefficient = dielectric_analysis.get('temp_coefficient', 'unknown')
        
        if dielectric_type == 'ceramic' and temp_coefficient == 'excellent':
            assessment['reliability_factors'].append('stable_ceramic_dielectric')
        elif dielectric_type == 'tantalum':
            assessment['reliability_factors'].append('tantalum_reliability')
            assessment['risk_factors'].append('tantalum_failure_mode')  # Potential for shorts
        elif dielectric_type == 'aluminum_electrolytic':
            assessment['risk_factors'].append('electrolytic_aging')
            assessment['risk_factors'].append('temperature_sensitive')
        elif dielectric_type == 'film':
            assessment['reliability_factors'].append('excellent_film_stability')
        
        # Voltage derating assessment
        voltage_rating = voltage_analysis.get('voltage_rating_v', 0)
        if voltage_rating > 50:
            assessment['reliability_factors'].append('adequate_voltage_rating')
        elif voltage_rating > 25:
            assessment['quality_indicators'].append('standard_voltage_rating')
        else:
            assessment['risk_factors'].append('low_voltage_rating')
        
        # Capacitance value stability
        capacitance_category = capacitance_analysis.get('category', 'medium')
        if capacitance_category in ['ultra_low', 'low']:
            assessment['quality_indicators'].append('precision_capacitor')
        elif capacitance_category == 'very_high':
            assessment['risk_factors'].append('high_capacitance_drift')
        
        # Package size reliability
        package_size = self._extract_package_size(api_data.get('specifications', {}).get('package', '0603'))
        if package_size in ['0201', '0402']:
            assessment['risk_factors'].append('micro_component_handling_risk')
        elif package_size in ['1206', '1210']:
            assessment['reliability_factors'].append('robust_package_size')
        
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
    
    def _categorize_voltage_rating(self, voltage: float) -> str:
        """Categorize voltage rating for safety analysis."""
        if voltage <= 25:
            return 'low'
        elif voltage <= 100:
            return 'medium'
        elif voltage <= 500:
            return 'high'
        else:
            return 'very_high'
    
    def _classify_tolerance(self, tolerance_percent: float) -> str:
        """Classify tolerance for precision requirements."""
        if tolerance_percent <= 5.0:
            return 'precision'
        elif tolerance_percent <= 10.0:
            return 'standard'
        else:
            return 'wide'
    
    def _extract_package_size(self, package_name: str) -> str:
        """Extract package size from package name."""
        common_sizes = ['0201', '0402', '0603', '0805', '1206', '1210', '1812', '2220']
        for size in common_sizes:
            if size in package_name:
                return size
        return '0603'  # Default
    
    def _find_min_package_for_capacitance(self, capacitance_pf: float) -> str:
        """Find minimum package size for given capacitance."""
        for size, limits in sorted(self.ceramic_capacitance_limits.items()):
            if capacitance_pf <= limits['max_pf']:
                return size
        return '2220'  # Largest available
    
    def _find_min_package_for_voltage(self, voltage_rating: float) -> str:
        """Find minimum package size for given voltage rating."""
        for size, limits in sorted(self.ceramic_capacitance_limits.items()):
            if voltage_rating <= limits['max_voltage']:
                return size
        return '2220'  # Largest available for high voltage
    
    def _get_capacitor_package_notes(self, package_size: str, capacitance_analysis: Dict[str, Any]) -> List[str]:
        """Get manufacturing considerations for capacitor package size."""
        notes = []
        
        if package_size in ['0201', '0402']:
            notes.extend(['extreme_precision_required', 'micro_handling', 'esd_sensitive'])
        elif package_size in ['0603', '0805']:
            notes.extend(['standard_placement', 'good_manufacturability'])
        elif package_size in ['1206', '1210']:
            notes.extend(['robust_handling', 'excellent_manufacturability'])
        
        # Add capacitance-specific notes
        category = capacitance_analysis.get('category', 'medium')
        if category == 'very_high':
            notes.append('high_capacitance_handling')
        elif category == 'ultra_low':
            notes.append('precision_handling_required')
        
        return notes
    
    def _get_optimal_test_frequency(self, capacitance: float) -> int:
        """Get optimal test frequency for capacitance measurement."""
        # Higher frequencies for smaller capacitors, lower for larger
        if capacitance < 1e-9:  # <1nF
            return 10000  # 10kHz
        elif capacitance < 1e-6:  # <1µF
            return 1000   # 1kHz
        else:  # ≥1µF
            return 100    # 100Hz
    
    def _estimate_max_esr(self, capacitance: float, dielectric_type: str) -> float:
        """Estimate maximum allowable ESR for capacitor type."""
        if dielectric_type == 'tantalum':
            # Tantalum ESR roughly inversely proportional to capacitance
            return 10.0 / (capacitance * 1e6)  # Rough estimate in ohms
        elif dielectric_type == 'aluminum_electrolytic':
            # Aluminum electrolytics have higher ESR
            return 100.0 / (capacitance * 1e6)  # Rough estimate in ohms
        else:
            # Ceramic and film have very low ESR
            return 0.1  # 0.1 ohm maximum