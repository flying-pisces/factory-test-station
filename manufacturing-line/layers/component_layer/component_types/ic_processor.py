"""IC (Integrated Circuit) Component Type Processor - Week 2 Implementation."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re
import math


class ICProcessor:
    """Specialized processor for integrated circuit components with manufacturing-specific analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger('ICProcessor')
        
        # IC package types and their characteristics
        self.package_types = {
            'QFP': {
                'pin_pitches': [0.4, 0.5, 0.65, 0.8, 1.0],  # mm
                'typical_pin_counts': [32, 44, 64, 100, 144, 176, 208],
                'placement_accuracy_um': 10,
                'vision_required': True,
                'complexity': 'high'
            },
            'TQFP': {
                'pin_pitches': [0.4, 0.5, 0.65, 0.8],
                'typical_pin_counts': [32, 44, 64, 100, 144],
                'placement_accuracy_um': 8,
                'vision_required': True,
                'complexity': 'very_high'
            },
            'QFN': {
                'pin_pitches': [0.4, 0.5, 0.65],
                'typical_pin_counts': [16, 20, 24, 28, 32, 48, 64],
                'placement_accuracy_um': 5,
                'vision_required': True,
                'complexity': 'very_high'
            },
            'DFN': {
                'pin_pitches': [0.5, 0.65],
                'typical_pin_counts': [6, 8, 10, 12, 16, 20],
                'placement_accuracy_um': 5,
                'vision_required': True,
                'complexity': 'high'
            },
            'BGA': {
                'ball_pitches': [0.4, 0.5, 0.65, 0.8, 1.0, 1.27],
                'typical_ball_counts': [49, 81, 100, 144, 196, 256, 324, 400, 484],
                'placement_accuracy_um': 5,
                'vision_required': True,
                'complexity': 'extreme'
            },
            'SOIC': {
                'pin_pitches': [1.27],  # 50 mil
                'typical_pin_counts': [8, 14, 16, 20, 24, 28],
                'placement_accuracy_um': 15,
                'vision_required': False,
                'complexity': 'medium'
            },
            'SOT': {
                'pin_pitches': [0.95, 1.27],
                'typical_pin_counts': [3, 5, 6, 8],
                'placement_accuracy_um': 10,
                'vision_required': False,
                'complexity': 'medium'
            }
        }
        
        # IC categories by function
        self.ic_categories = {
            'microcontroller': {
                'typical_features': ['cpu', 'memory', 'gpio', 'peripherals'],
                'power_consumption': 'medium',
                'heat_generation': 'medium',
                'complexity_level': 'high'
            },
            'processor': {
                'typical_features': ['cpu', 'cache', 'high_speed'],
                'power_consumption': 'high',
                'heat_generation': 'high',
                'complexity_level': 'extreme'
            },
            'memory': {
                'typical_features': ['storage', 'high_density'],
                'power_consumption': 'low_to_medium',
                'heat_generation': 'low',
                'complexity_level': 'medium'
            },
            'analog': {
                'typical_features': ['amplification', 'conversion', 'precision'],
                'power_consumption': 'low_to_medium',
                'heat_generation': 'low_to_medium',
                'complexity_level': 'medium'
            },
            'power_management': {
                'typical_features': ['regulation', 'switching', 'efficiency'],
                'power_consumption': 'variable',
                'heat_generation': 'medium_to_high',
                'complexity_level': 'medium'
            },
            'interface': {
                'typical_features': ['communication', 'level_shifting', 'isolation'],
                'power_consumption': 'low',
                'heat_generation': 'low',
                'complexity_level': 'low_to_medium'
            }
        }
        
        self.logger.info("ICProcessor initialized with package type and functional category analysis")
    
    def process(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process IC-specific manufacturing requirements."""
        try:
            result = {
                'component_type': 'IC',
                'processing_status': 'success'
            }
            
            # Extract IC-specific parameters
            ee_data = processed_data.get('ee_data', {})
            cad_data = processed_data.get('cad_data', {})
            api_data = processed_data.get('api_data', {})
            
            # Analyze package type and characteristics
            package_analysis = self._analyze_package_type(cad_data, ee_data)
            result['package_analysis'] = package_analysis
            
            # Determine IC functional category
            functional_analysis = self._analyze_functional_category(api_data, ee_data)
            result['functional_analysis'] = functional_analysis
            
            # Power and thermal analysis
            thermal_analysis = self._analyze_thermal_characteristics(ee_data, package_analysis, functional_analysis)
            result['thermal_analysis'] = thermal_analysis
            
            # Pin/ball analysis for placement
            pin_analysis = self._analyze_pin_characteristics(package_analysis, cad_data)
            result['pin_analysis'] = pin_analysis
            
            # Manufacturing process requirements
            manufacturing_req = self._determine_manufacturing_requirements(
                package_analysis, pin_analysis, thermal_analysis
            )
            result['manufacturing_requirements'] = manufacturing_req
            
            # Test strategy for production
            test_strategy = self._create_test_strategy(package_analysis, functional_analysis, pin_analysis)
            result['test_strategy'] = test_strategy
            
            # Quality and reliability assessment
            quality_assessment = self._assess_quality_reliability(
                package_analysis, functional_analysis, thermal_analysis, api_data
            )
            result['quality_assessment'] = quality_assessment
            
            return result
            
        except Exception as e:
            self.logger.error(f"IC processing failed: {e}")
            return {
                'component_type': 'IC',
                'processing_status': 'failed',
                'error': str(e),
                'manufacturing_requirements': {
                    'placement_profile': 'smt_place_ic',
                    'estimated_time_s': 2.5
                }
            }
    
    def _analyze_package_type(self, cad_data: Dict[str, Any], ee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze IC package type and extract manufacturing characteristics."""
        analysis = {}
        
        package_info = cad_data.get('package_info', {})
        package_name = package_info.get('package_name', 'UNKNOWN')
        
        # Extract package family
        package_family = 'UNKNOWN'
        for family in self.package_types.keys():
            if family in package_name.upper():
                package_family = family
                break
        
        analysis['package_name'] = package_name
        analysis['package_family'] = package_family
        
        if package_family in self.package_types:
            package_specs = self.package_types[package_family]
            analysis.update({
                'placement_accuracy_um': package_specs['placement_accuracy_um'],
                'vision_required': package_specs['vision_required'],
                'complexity': package_specs['complexity']
            })
            
            # Extract pin/ball count
            pin_count = self._extract_pin_count(package_name, ee_data)
            analysis['pin_count'] = pin_count
            
            # Validate pin count against typical values
            if 'typical_pin_counts' in package_specs:
                if pin_count in package_specs['typical_pin_counts']:
                    analysis['pin_count_validation'] = 'standard'
                else:
                    closest = min(package_specs['typical_pin_counts'], key=lambda x: abs(x - pin_count))
                    analysis['pin_count_validation'] = 'non_standard'
                    analysis['closest_standard_pin_count'] = closest
            elif 'typical_ball_counts' in package_specs:
                if pin_count in package_specs['typical_ball_counts']:
                    analysis['pin_count_validation'] = 'standard'
                else:
                    closest = min(package_specs['typical_ball_counts'], key=lambda x: abs(x - pin_count))
                    analysis['pin_count_validation'] = 'non_standard'
                    analysis['closest_standard_ball_count'] = closest
            
            # Estimate pin pitch
            dimensions = cad_data.get('processed_dimensions', {})
            if dimensions and pin_count > 0:
                # Rough pitch estimation based on package size and pin count
                length_mm = dimensions.get('length_mm', 10)
                if package_family in ['QFP', 'TQFP']:
                    # 4-sided package
                    estimated_pitch = (length_mm * 4) / pin_count
                elif package_family in ['SOIC', 'SOT']:
                    # 2-sided package
                    estimated_pitch = (length_mm * 2) / pin_count
                else:
                    estimated_pitch = 0.5  # Default assumption
                
                analysis['estimated_pin_pitch_mm'] = estimated_pitch
                
                # Validate against typical pitches
                if package_family in self.package_types and 'pin_pitches' in package_specs:
                    closest_pitch = min(package_specs['pin_pitches'], key=lambda x: abs(x - estimated_pitch))
                    analysis['closest_standard_pitch_mm'] = closest_pitch
        else:
            # Unknown package type
            analysis.update({
                'placement_accuracy_um': 10,  # Conservative default
                'vision_required': True,
                'complexity': 'high',
                'pin_count': 0,
                'pin_count_validation': 'unknown'
            })
        
        return analysis
    
    def _analyze_functional_category(self, api_data: Dict[str, Any], ee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine IC functional category from specifications."""
        analysis = {
            'category': 'unknown',
            'subcategory': 'general_purpose',
            'functional_characteristics': []
        }
        
        # Analyze API specifications for category hints
        specs = api_data.get('specifications', {})
        part_number = api_data.get('part_number', '').upper()
        
        # Look for category indicators in specifications
        spec_text = ' '.join([str(v).upper() for v in specs.values()] + [part_number])
        
        if any(term in spec_text for term in ['MICROCONTROLLER', 'MCU', 'ARM', 'PIC', 'ATMEGA']):
            analysis['category'] = 'microcontroller'
            analysis['subcategory'] = self._determine_mcu_subcategory(spec_text)
            analysis['functional_characteristics'] = ['programmable', 'integrated_peripherals', 'gpio']
            
        elif any(term in spec_text for term in ['PROCESSOR', 'CPU', 'MPU', 'CORTEX']):
            analysis['category'] = 'processor'
            analysis['subcategory'] = self._determine_processor_subcategory(spec_text)
            analysis['functional_characteristics'] = ['high_performance', 'complex_instruction_set']
            
        elif any(term in spec_text for term in ['MEMORY', 'FLASH', 'SRAM', 'DRAM', 'EEPROM']):
            analysis['category'] = 'memory'
            analysis['subcategory'] = self._determine_memory_subcategory(spec_text)
            analysis['functional_characteristics'] = ['data_storage', 'high_density']
            
        elif any(term in spec_text for term in ['AMPLIFIER', 'OPAMP', 'ADC', 'DAC', 'ANALOG']):
            analysis['category'] = 'analog'
            analysis['subcategory'] = self._determine_analog_subcategory(spec_text)
            analysis['functional_characteristics'] = ['precision', 'low_noise']
            
        elif any(term in spec_text for term in ['REGULATOR', 'CONVERTER', 'SWITCHER', 'LDO', 'PMIC']):
            analysis['category'] = 'power_management'
            analysis['subcategory'] = self._determine_power_subcategory(spec_text)
            analysis['functional_characteristics'] = ['power_conversion', 'efficiency']
            
        elif any(term in spec_text for term in ['UART', 'SPI', 'I2C', 'USB', 'CAN', 'ETHERNET']):
            analysis['category'] = 'interface'
            analysis['subcategory'] = self._determine_interface_subcategory(spec_text)
            analysis['functional_characteristics'] = ['communication', 'protocol_support']
        
        # Add category-specific characteristics
        if analysis['category'] in self.ic_categories:
            category_info = self.ic_categories[analysis['category']]
            analysis.update({
                'expected_power_consumption': category_info['power_consumption'],
                'expected_heat_generation': category_info['heat_generation'],
                'complexity_level': category_info['complexity_level']
            })
        
        return analysis
    
    def _analyze_thermal_characteristics(self, ee_data: Dict[str, Any], 
                                       package_analysis: Dict[str, Any],
                                       functional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermal characteristics and requirements."""
        analysis = {}
        
        # Extract power consumption if available
        power_param = ee_data.get('validated_parameters', {}).get('power_consumption', {})
        if power_param.get('valid'):
            power_consumption_w = power_param['value']
        else:
            # Estimate based on functional category
            power_level = functional_analysis.get('expected_power_consumption', 'medium')
            power_consumption_w = self._estimate_power_consumption(power_level, package_analysis.get('pin_count', 32))
        
        analysis['power_consumption_w'] = power_consumption_w
        
        # Thermal resistance estimation based on package
        package_family = package_analysis.get('package_family', 'UNKNOWN')
        thermal_resistance = self._estimate_thermal_resistance(package_family)
        analysis['thermal_resistance_c_per_w'] = thermal_resistance
        
        # Junction temperature calculation
        ambient_temp = 25  # °C
        analysis['estimated_junction_temp_c'] = ambient_temp + (power_consumption_w * thermal_resistance)
        
        # Thermal management requirements
        if analysis['estimated_junction_temp_c'] > 85:
            analysis['thermal_management'] = 'active_cooling_required'
            analysis['cooling_recommendations'] = ['heat_sink', 'fan', 'thermal_vias']
        elif analysis['estimated_junction_temp_c'] > 70:
            analysis['thermal_management'] = 'passive_cooling_recommended'
            analysis['cooling_recommendations'] = ['thermal_vias', 'copper_pour']
        else:
            analysis['thermal_management'] = 'natural_convection'
            analysis['cooling_recommendations'] = ['standard_pcb_layout']
        
        # Package thermal characteristics
        if package_family == 'BGA':
            analysis['thermal_pad'] = True
            analysis['thermal_via_required'] = True
        elif package_family in ['QFN', 'DFN']:
            analysis['thermal_pad'] = True
            analysis['thermal_via_recommended'] = True
        else:
            analysis['thermal_pad'] = False
            analysis['thermal_via_recommended'] = False
        
        return analysis
    
    def _analyze_pin_characteristics(self, package_analysis: Dict[str, Any], cad_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pin characteristics for placement requirements."""
        analysis = {}
        
        pin_count = package_analysis.get('pin_count', 0)
        package_family = package_analysis.get('package_family', 'UNKNOWN')
        pin_pitch = package_analysis.get('estimated_pin_pitch_mm', 0.5)
        
        analysis.update({
            'pin_count': pin_count,
            'pin_pitch_mm': pin_pitch,
            'package_family': package_family
        })
        
        # Pin density analysis
        if pin_count > 0:
            analysis['pin_density'] = self._classify_pin_density(pin_count, pin_pitch)
        else:
            analysis['pin_density'] = 'unknown'
        
        # Placement difficulty based on pin characteristics
        if pin_pitch < 0.4:
            analysis['placement_difficulty'] = 'extreme'
            analysis['special_requirements'] = ['ultra_precision_placement', 'advanced_vision_system']
        elif pin_pitch < 0.5:
            analysis['placement_difficulty'] = 'very_high'
            analysis['special_requirements'] = ['precision_placement', 'high_resolution_vision']
        elif pin_pitch < 0.8:
            analysis['placement_difficulty'] = 'high'
            analysis['special_requirements'] = ['accurate_placement', 'vision_verification']
        else:
            analysis['placement_difficulty'] = 'medium'
            analysis['special_requirements'] = ['standard_placement']
        
        # Lead coplanarity requirements (for leaded packages)
        if package_family in ['QFP', 'TQFP', 'SOIC']:
            analysis['coplanarity_critical'] = True
            analysis['max_lead_coplanarity_um'] = 50  # 50µm typical requirement
        else:
            analysis['coplanarity_critical'] = False
        
        return analysis
    
    def _determine_manufacturing_requirements(self, package_analysis: Dict[str, Any],
                                           pin_analysis: Dict[str, Any],
                                           thermal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine specific manufacturing requirements for IC placement."""
        requirements = {}
        
        # Basic placement profile
        requirements['placement_profile'] = 'smt_place_ic'
        
        # Accuracy requirements
        requirements['placement_accuracy_um'] = package_analysis.get('placement_accuracy_um', 10)
        requirements['vision_alignment_required'] = package_analysis.get('vision_required', True)
        
        # Package-specific requirements
        package_family = package_analysis.get('package_family', 'UNKNOWN')
        complexity = package_analysis.get('complexity', 'high')
        
        if complexity == 'extreme':
            requirements['placement_complexity'] = 'extreme'
            requirements['nozzle_type'] = 'precision'
            requirements['placement_force_n'] = 0.5
            requirements['placement_time_s'] = 3.0
            requirements['special_equipment'] = ['high_precision_head', 'advanced_vision']
        elif complexity == 'very_high':
            requirements['placement_complexity'] = 'very_high'
            requirements['nozzle_type'] = 'precision'
            requirements['placement_force_n'] = 1.0
            requirements['placement_time_s'] = 2.5
            requirements['special_equipment'] = ['precision_head', 'vision_system']
        elif complexity == 'high':
            requirements['placement_complexity'] = 'high'
            requirements['nozzle_type'] = 'standard'
            requirements['placement_force_n'] = 1.5
            requirements['placement_time_s'] = 2.1
            requirements['special_equipment'] = ['vision_system']
        else:
            requirements['placement_complexity'] = 'medium'
            requirements['nozzle_type'] = 'standard'
            requirements['placement_force_n'] = 2.0
            requirements['placement_time_s'] = 1.5
            requirements['special_equipment'] = []
        
        # Thermal management requirements
        thermal_mgmt = thermal_analysis.get('thermal_management', 'natural_convection')
        if thermal_mgmt != 'natural_convection':
            requirements['thermal_considerations'] = thermal_mgmt
            requirements['placement_considerations'] = thermal_analysis.get('cooling_recommendations', [])
        
        # Pin-specific requirements
        pin_difficulty = pin_analysis.get('placement_difficulty', 'medium')
        if pin_difficulty in ['extreme', 'very_high']:
            requirements['pre_placement_inspection'] = True
            requirements['post_placement_verification'] = True
            requirements['handling_precautions'] = ['esd_protection', 'clean_environment']
        else:
            requirements['pre_placement_inspection'] = False
            requirements['post_placement_verification'] = True
            requirements['handling_precautions'] = ['esd_protection']
        
        # Coplanarity requirements
        if pin_analysis.get('coplanarity_critical', False):
            requirements['coplanarity_check_required'] = True
            requirements['max_coplanarity_deviation_um'] = pin_analysis.get('max_lead_coplanarity_um', 50)
        
        return requirements
    
    def _create_test_strategy(self, package_analysis: Dict[str, Any],
                            functional_analysis: Dict[str, Any],
                            pin_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create manufacturing test strategy for ICs."""
        strategy = {
            'test_sequence': [],
            'test_equipment': ['aoi_system'],  # Automated Optical Inspection is standard for ICs
            'test_settings': {}
        }
        
        # Visual inspection (always required for ICs)
        strategy['test_sequence'].append({
            'test_name': 'optical_inspection',
            'test_type': 'automated_optical_inspection',
            'checks': ['placement_accuracy', 'orientation', 'solder_joint_quality'],
            'measurement_time_ms': 500,
            'priority': 'critical'
        })
        
        # Electrical continuity test
        pin_count = pin_analysis.get('pin_count', 0)
        if pin_count > 0:
            strategy['test_sequence'].append({
                'test_name': 'continuity_test',
                'test_type': 'electrical_continuity',
                'pin_count': pin_count,
                'test_current_ma': 1.0,
                'measurement_time_ms': 50 * pin_count,  # 50ms per pin
                'priority': 'critical'
            })
            strategy['test_equipment'].append('in_circuit_tester')
        
        # Boundary scan test (if supported)
        if functional_analysis.get('category') in ['microcontroller', 'processor', 'memory']:
            strategy['test_sequence'].append({
                'test_name': 'boundary_scan_test',
                'test_type': 'jtag_boundary_scan',
                'test_coverage': 'structural',
                'measurement_time_ms': 1000,
                'priority': 'important'
            })
            strategy['test_equipment'].append('boundary_scan_tester')
        
        # Power consumption test for active components
        if functional_analysis.get('category') in ['microcontroller', 'processor']:
            strategy['test_sequence'].append({
                'test_name': 'power_consumption_test',
                'test_type': 'current_measurement',
                'test_conditions': ['idle', 'active'],
                'max_current_ma': self._estimate_max_current(functional_analysis),
                'measurement_time_ms': 200,
                'priority': 'important'
            })
            strategy['test_equipment'].append('current_meter')
        
        # Coplanarity test for leaded packages
        if pin_analysis.get('coplanarity_critical', False):
            strategy['test_sequence'].append({
                'test_name': 'coplanarity_test',
                'test_type': 'dimensional_measurement',
                'max_deviation_um': pin_analysis.get('max_lead_coplanarity_um', 50),
                'measurement_time_ms': 300,
                'priority': 'important'
            })
            strategy['test_equipment'].append('3d_measurement_system')
        
        # X-ray inspection for hidden solder joints (BGA, QFN)
        package_family = package_analysis.get('package_family', 'UNKNOWN')
        if package_family in ['BGA', 'QFN', 'DFN']:
            strategy['test_sequence'].append({
                'test_name': 'xray_inspection',
                'test_type': 'x_ray_imaging',
                'inspection_areas': ['hidden_solder_joints', 'void_detection'],
                'measurement_time_ms': 1000,
                'priority': 'important'
            })
            strategy['test_equipment'].append('x_ray_system')
        
        # Test settings
        strategy['test_settings'] = {
            'environmental_conditions': {
                'temperature_c': 23,
                'humidity_percent': 45,
                'esd_protection': True
            },
            'handling_requirements': {
                'clean_environment': True,
                'esd_safe_handling': True,
                'temperature_sensitive': functional_analysis.get('category') in ['processor', 'memory']
            }
        }
        
        return strategy
    
    def _assess_quality_reliability(self, package_analysis: Dict[str, Any],
                                  functional_analysis: Dict[str, Any],
                                  thermal_analysis: Dict[str, Any],
                                  api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality and reliability characteristics."""
        assessment = {
            'quality_indicators': [],
            'reliability_factors': [],
            'risk_factors': []
        }
        
        # Package type reliability
        package_family = package_analysis.get('package_family', 'UNKNOWN')
        complexity = package_analysis.get('complexity', 'high')
        
        if package_family in ['SOIC', 'SOT']:
            assessment['reliability_factors'].append('robust_leaded_package')
        elif package_family in ['QFP', 'TQFP']:
            assessment['reliability_factors'].append('proven_quad_package')
            assessment['risk_factors'].append('lead_coplanarity_sensitivity')
        elif package_family in ['QFN', 'DFN']:
            assessment['quality_indicators'].append('excellent_thermal_performance')
            assessment['risk_factors'].append('hidden_solder_joints')
        elif package_family == 'BGA':
            assessment['quality_indicators'].append('high_density_interconnect')
            assessment['risk_factors'].append('complex_solder_inspection')
            assessment['risk_factors'].append('thermal_cycling_stress')
        
        # Pin count and pitch reliability
        pin_count = package_analysis.get('pin_count', 0)
        pin_pitch = package_analysis.get('estimated_pin_pitch_mm', 0.5)
        
        if pin_pitch < 0.4:
            assessment['risk_factors'].append('ultra_fine_pitch_placement')
        elif pin_pitch < 0.5:
            assessment['risk_factors'].append('fine_pitch_placement')
        else:
            assessment['reliability_factors'].append('standard_pitch_robust')
        
        if pin_count > 200:
            assessment['risk_factors'].append('high_pin_count_complexity')
        elif pin_count > 100:
            assessment['quality_indicators'].append('high_functionality')
        
        # Thermal reliability
        junction_temp = thermal_analysis.get('estimated_junction_temp_c', 25)
        if junction_temp > 100:
            assessment['risk_factors'].append('high_operating_temperature')
        elif junction_temp > 85:
            assessment['risk_factors'].append('elevated_operating_temperature')
        else:
            assessment['reliability_factors'].append('safe_operating_temperature')
        
        # Functional category reliability
        category = functional_analysis.get('category', 'unknown')
        if category == 'processor':
            assessment['quality_indicators'].append('high_performance_capability')
            assessment['risk_factors'].append('complex_power_management')
        elif category == 'microcontroller':
            assessment['reliability_factors'].append('integrated_solution')
            assessment['quality_indicators'].append('proven_architecture')
        elif category == 'memory':
            assessment['reliability_factors'].append('data_integrity_features')
        elif category == 'analog':
            assessment['quality_indicators'].append('precision_performance')
        
        # Overall assessment
        quality_score = len(assessment['quality_indicators']) * 2 + len(assessment['reliability_factors'])
        risk_score = len(assessment['risk_factors']) * 2  # Risk factors weighted more heavily
        
        if quality_score >= 6 and risk_score <= 2:
            assessment['overall_rating'] = 'excellent'
        elif quality_score >= 3 and risk_score <= 4:
            assessment['overall_rating'] = 'good'
        elif risk_score <= 6:
            assessment['overall_rating'] = 'acceptable'
        else:
            assessment['overall_rating'] = 'challenging'
        
        return assessment
    
    # Helper methods
    def _extract_pin_count(self, package_name: str, ee_data: Dict[str, Any]) -> int:
        """Extract pin count from package name or EE data."""
        # First try to get from EE data
        pin_param = ee_data.get('validated_parameters', {}).get('pin_count', {})
        if pin_param.get('valid'):
            return int(pin_param['value'])
        
        # Extract from package name
        match = re.search(r'(\d+)', package_name)
        if match:
            return int(match.group(1))
        
        return 0
    
    def _determine_mcu_subcategory(self, spec_text: str) -> str:
        if '8BIT' in spec_text or 'PIC' in spec_text:
            return '8bit_microcontroller'
        elif '32BIT' in spec_text or 'ARM' in spec_text:
            return '32bit_microcontroller'
        else:
            return 'general_microcontroller'
    
    def _determine_processor_subcategory(self, spec_text: str) -> str:
        if 'DSP' in spec_text:
            return 'digital_signal_processor'
        elif 'FPGA' in spec_text:
            return 'fpga_processor'
        else:
            return 'general_processor'
    
    def _determine_memory_subcategory(self, spec_text: str) -> str:
        if 'FLASH' in spec_text:
            return 'flash_memory'
        elif 'SRAM' in spec_text:
            return 'sram_memory'
        elif 'DRAM' in spec_text:
            return 'dram_memory'
        else:
            return 'general_memory'
    
    def _determine_analog_subcategory(self, spec_text: str) -> str:
        if 'OPAMP' in spec_text or 'AMPLIFIER' in spec_text:
            return 'operational_amplifier'
        elif 'ADC' in spec_text:
            return 'analog_to_digital_converter'
        elif 'DAC' in spec_text:
            return 'digital_to_analog_converter'
        else:
            return 'general_analog'
    
    def _determine_power_subcategory(self, spec_text: str) -> str:
        if 'LDO' in spec_text or 'REGULATOR' in spec_text:
            return 'voltage_regulator'
        elif 'CONVERTER' in spec_text or 'SWITCHER' in spec_text:
            return 'switching_converter'
        else:
            return 'power_management'
    
    def _determine_interface_subcategory(self, spec_text: str) -> str:
        if 'USB' in spec_text:
            return 'usb_interface'
        elif 'ETHERNET' in spec_text:
            return 'ethernet_interface'
        elif 'CAN' in spec_text:
            return 'can_interface'
        else:
            return 'general_interface'
    
    def _estimate_power_consumption(self, power_level: str, pin_count: int) -> float:
        """Estimate power consumption based on category and pin count."""
        base_power = {
            'low': 0.001,      # 1mW
            'low_to_medium': 0.01,   # 10mW
            'medium': 0.1,     # 100mW
            'high': 1.0,       # 1W
            'variable': 0.5    # 500mW
        }
        
        power = base_power.get(power_level, 0.1)
        
        # Scale with pin count (more pins usually means more functionality)
        pin_factor = 1 + (pin_count - 32) * 0.01  # 1% increase per pin above 32
        return power * max(pin_factor, 1.0)
    
    def _estimate_thermal_resistance(self, package_family: str) -> float:
        """Estimate thermal resistance based on package type (°C/W)."""
        thermal_resistance = {
            'SOT': 200,    # Small outline packages have high thermal resistance
            'SOIC': 150,   # Medium thermal resistance
            'QFP': 100,    # Better thermal performance
            'TQFP': 80,    # Thin packages slightly better
            'QFN': 50,     # Excellent thermal performance with thermal pad
            'DFN': 60,     # Good thermal performance
            'BGA': 40      # Best thermal performance
        }
        
        return thermal_resistance.get(package_family, 100)
    
    def _classify_pin_density(self, pin_count: int, pin_pitch: float) -> str:
        """Classify pin density for manufacturing difficulty assessment."""
        density_factor = pin_count / pin_pitch
        
        if density_factor > 500:
            return 'ultra_high_density'
        elif density_factor > 200:
            return 'high_density'
        elif density_factor > 100:
            return 'medium_density'
        else:
            return 'low_density'
    
    def _estimate_max_current(self, functional_analysis: Dict[str, Any]) -> float:
        """Estimate maximum current consumption for test limits."""
        category = functional_analysis.get('category', 'unknown')
        
        current_limits = {
            'microcontroller': 100,  # 100mA
            'processor': 1000,       # 1A
            'memory': 200,           # 200mA
            'analog': 50,            # 50mA
            'power_management': 500, # 500mA
            'interface': 100         # 100mA
        }
        
        return current_limits.get(category, 100)