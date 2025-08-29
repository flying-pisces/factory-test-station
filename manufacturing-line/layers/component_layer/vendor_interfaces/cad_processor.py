"""CAD Data Processor for Component Vendor Interfaces - Week 2 Implementation."""

import logging
from typing import Dict, Any, List, Optional
import re
import math


class CADProcessor:
    """Processes CAD data from component vendors (STEP files, footprints, 3D models)."""
    
    def __init__(self):
        self.logger = logging.getLogger('CADProcessor')
        
        # Package type recognition patterns
        self.package_patterns = {
            # SMT packages
            r'0201': {'type': 'smt_passive', 'size': (0.6, 0.3), 'height': 0.13},
            r'0402': {'type': 'smt_passive', 'size': (1.0, 0.5), 'height': 0.2},
            r'0603': {'type': 'smt_passive', 'size': (1.6, 0.8), 'height': 0.45},
            r'0805': {'type': 'smt_passive', 'size': (2.0, 1.25), 'height': 0.6},
            r'1206': {'type': 'smt_passive', 'size': (3.2, 1.6), 'height': 0.65},
            
            # IC packages
            r'QFP-(\d+)': {'type': 'smt_ic_quad', 'pin_count': 'match_1'},
            r'TQFP-(\d+)': {'type': 'smt_ic_quad', 'pin_count': 'match_1'},
            r'QFN-(\d+)': {'type': 'smt_ic_quad_no_leads', 'pin_count': 'match_1'},
            r'DFN-(\d+)': {'type': 'smt_ic_dual_no_leads', 'pin_count': 'match_1'},
            r'SOIC-(\d+)': {'type': 'smt_ic_dual', 'pin_count': 'match_1'},
            r'SOT-(\d+)': {'type': 'smt_ic_small', 'package_variant': 'match_1'},
            r'BGA-(\d+)': {'type': 'smt_ic_ball_grid', 'ball_count': 'match_1'},
            
            # Through-hole
            r'DIP-(\d+)': {'type': 'th_ic_dual', 'pin_count': 'match_1'},
            r'TO-(\d+)': {'type': 'th_power', 'package_variant': 'match_1'}
        }
        
        self.logger.info("CADProcessor initialized with 15 package recognition patterns")
    
    def process_cad_data(self, cad_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process CAD data and extract manufacturing-relevant information."""
        try:
            processed = {}
            
            # Extract package information
            package_info = self._extract_package_info(cad_data)
            processed['package_info'] = package_info
            
            # Process dimensions
            dimensions = self._process_dimensions(cad_data.get('dimensions', {}))
            processed['processed_dimensions'] = dimensions
            
            # Analyze footprint
            footprint_analysis = self._analyze_footprint(cad_data.get('footprint', ''))
            processed['footprint_analysis'] = footprint_analysis
            
            # Process 3D model if available
            model_analysis = self._analyze_3d_model(cad_data.get('3d_model_path', ''))
            processed['model_analysis'] = model_analysis
            
            # Generate placement requirements
            placement_requirements = self._generate_placement_requirements(package_info, dimensions)
            processed['placement_requirements'] = placement_requirements
            
            # Assembly complexity assessment
            complexity = self._assess_assembly_complexity(package_info, dimensions)
            processed['assembly_complexity'] = complexity
            
            return processed
            
        except Exception as e:
            self.logger.error(f"CAD processing failed: {e}")
            return {
                'error': str(e),
                'processing_status': 'failed',
                'package_info': {'type': 'unknown'},
                'processed_dimensions': {},
                'placement_requirements': {'complexity': 'high'}
            }
    
    def _extract_package_info(self, cad_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract package type and characteristics from CAD data."""
        package = cad_data.get('package', 'UNKNOWN')
        footprint = cad_data.get('footprint', '')
        
        # Try to match package patterns
        for pattern, info in self.package_patterns.items():
            match = re.search(pattern, package, re.IGNORECASE)
            if match:
                package_info = info.copy()
                package_info['package_name'] = package
                package_info['footprint'] = footprint
                
                # Handle regex groups (pin counts, variants)
                for key, value in package_info.items():
                    if isinstance(value, str) and value.startswith('match_'):
                        group_num = int(value.split('_')[1])
                        if group_num <= len(match.groups()):
                            package_info[key] = int(match.group(group_num))
                
                return package_info
        
        # Default for unrecognized packages
        return {
            'type': 'unknown',
            'package_name': package,
            'footprint': footprint,
            'recognition_status': 'unrecognized'
        }
    
    def _process_dimensions(self, dimensions: Dict[str, Any]) -> Dict[str, Any]:
        """Process component dimensions for manufacturing planning."""
        processed_dims = {}
        
        # Extract basic dimensions
        length = dimensions.get('length', 0)
        width = dimensions.get('width', 0) 
        height = dimensions.get('height', 0)
        
        processed_dims.update({
            'length_mm': float(length),
            'width_mm': float(width),
            'height_mm': float(height),
            'volume_mm3': length * width * height,
            'footprint_area_mm2': length * width
        })
        
        # Size classification
        max_dim = max(length, width)
        if max_dim <= 1.0:
            size_class = 'micro'
            handling_difficulty = 'high'
        elif max_dim <= 2.0:
            size_class = 'small'
            handling_difficulty = 'medium'
        elif max_dim <= 5.0:
            size_class = 'standard'
            handling_difficulty = 'low'
        else:
            size_class = 'large'
            handling_difficulty = 'low'
        
        processed_dims.update({
            'size_classification': size_class,
            'handling_difficulty': handling_difficulty,
            'max_dimension_mm': max_dim
        })
        
        return processed_dims
    
    def _analyze_footprint(self, footprint: str) -> Dict[str, Any]:
        """Analyze footprint for placement requirements."""
        analysis = {
            'footprint_name': footprint,
            'pin_count': 0,
            'pad_type': 'unknown',
            'orientation_sensitive': False
        }
        
        # Extract pin count from footprint name
        pin_match = re.search(r'(\d+)', footprint)
        if pin_match:
            analysis['pin_count'] = int(pin_match.group(1))
        
        # Determine pad type
        if 'QFP' in footprint.upper() or 'TQFP' in footprint.upper():
            analysis['pad_type'] = 'gull_wing'
            analysis['orientation_sensitive'] = True
        elif 'QFN' in footprint.upper() or 'DFN' in footprint.upper():
            analysis['pad_type'] = 'no_lead'
            analysis['orientation_sensitive'] = True
        elif 'SOIC' in footprint.upper():
            analysis['pad_type'] = 'gull_wing'
            analysis['orientation_sensitive'] = True
        elif 'SOT' in footprint.upper():
            analysis['pad_type'] = 'small_outline'
            analysis['orientation_sensitive'] = True
        elif any(size in footprint for size in ['0201', '0402', '0603', '0805', '1206']):
            analysis['pad_type'] = 'rectangular'
            analysis['orientation_sensitive'] = False
        
        return analysis
    
    def _analyze_3d_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze 3D model file for additional manufacturing info."""
        analysis = {
            'model_available': bool(model_path),
            'model_path': model_path,
            'model_format': 'unknown'
        }
        
        if model_path:
            # Determine file format
            if model_path.endswith('.step') or model_path.endswith('.stp'):
                analysis['model_format'] = 'STEP'
                analysis['cad_compatible'] = True
            elif model_path.endswith('.iges') or model_path.endswith('.igs'):
                analysis['model_format'] = 'IGES'
                analysis['cad_compatible'] = True
            elif model_path.endswith('.stl'):
                analysis['model_format'] = 'STL'
                analysis['cad_compatible'] = False
            else:
                analysis['model_format'] = 'unknown'
                analysis['cad_compatible'] = False
        
        return analysis
    
    def _generate_placement_requirements(self, package_info: Dict[str, Any], dimensions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate placement requirements for manufacturing."""
        requirements = {}
        
        package_type = package_info.get('type', 'unknown')
        
        # Placement accuracy requirements
        if package_type.startswith('smt_ic'):
            requirements['placement_accuracy_um'] = 10  # ±10µm for ICs
            requirements['vision_required'] = True
        elif package_type == 'smt_passive':
            requirements['placement_accuracy_um'] = 20  # ±20µm for passives
            requirements['vision_required'] = False
        else:
            requirements['placement_accuracy_um'] = 50  # ±50µm for unknown
            requirements['vision_required'] = True
        
        # Force requirements
        component_area = dimensions.get('footprint_area_mm2', 1.0)
        if component_area < 1.0:
            requirements['placement_force_n'] = 1.0
        elif component_area < 5.0:
            requirements['placement_force_n'] = 2.0
        else:
            requirements['placement_force_n'] = 3.0
        
        # Nozzle requirements
        max_dim = dimensions.get('max_dimension_mm', 1.0)
        if max_dim < 1.0:
            requirements['nozzle_type'] = 'micro'
        elif max_dim < 5.0:
            requirements['nozzle_type'] = 'standard'
        else:
            requirements['nozzle_type'] = 'large'
        
        return requirements
    
    def _assess_assembly_complexity(self, package_info: Dict[str, Any], dimensions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess assembly complexity for process planning."""
        complexity_factors = []
        complexity_score = 0
        
        # Package type complexity
        package_type = package_info.get('type', 'unknown')
        if package_type.startswith('smt_ic'):
            complexity_score += 30
            complexity_factors.append('IC package requires precision')
        elif package_type == 'smt_passive':
            complexity_score += 10
            complexity_factors.append('Passive component - standard placement')
        else:
            complexity_score += 20
            complexity_factors.append('Unknown package type')
        
        # Size complexity
        max_dim = dimensions.get('max_dimension_mm', 1.0)
        if max_dim < 1.0:
            complexity_score += 20
            complexity_factors.append('Micro component - handling difficulty')
        elif max_dim > 10.0:
            complexity_score += 15
            complexity_factors.append('Large component - nozzle constraints')
        
        # Pin count complexity
        pin_count = package_info.get('pin_count', 0)
        if pin_count > 100:
            complexity_score += 25
            complexity_factors.append('High pin count IC')
        elif pin_count > 20:
            complexity_score += 10
            complexity_factors.append('Medium pin count IC')
        
        # Final complexity classification
        if complexity_score < 20:
            complexity_level = 'low'
        elif complexity_score < 40:
            complexity_level = 'medium'
        else:
            complexity_level = 'high'
        
        return {
            'complexity_level': complexity_level,
            'complexity_score': complexity_score,
            'complexity_factors': complexity_factors,
            'estimated_placement_time_s': 0.5 + (complexity_score * 0.02)
        }