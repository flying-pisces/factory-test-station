"""Enhanced ComponentLayerEngine with Vendor Data Processing - Week 2 Implementation."""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

# Import vendor interfaces
from .vendor_interfaces.cad_processor import CADProcessor
from .vendor_interfaces.api_processor import APIProcessor  
from .vendor_interfaces.ee_processor import EEProcessor

# Import component type processors
from .component_types.resistor_processor import ResistorProcessor
from .component_types.capacitor_processor import CapacitorProcessor
from .component_types.ic_processor import ICProcessor
from .component_types.inductor_processor import InductorProcessor


class ComponentType(Enum):
    """Component types supported by the system."""
    RESISTOR = "Resistor"
    CAPACITOR = "Capacitor" 
    IC = "IC"
    INDUCTOR = "Inductor"
    CONNECTOR = "Connector"
    CRYSTAL = "Crystal"
    DIODE = "Diode"
    TRANSISTOR = "Transistor"
    UNKNOWN = "Unknown"


@dataclass
class ProcessingMetrics:
    """Metrics for component processing performance."""
    processing_time_ms: float
    components_processed: int
    success_rate: float
    avg_component_time_ms: float
    cad_processing_time_ms: float
    api_processing_time_ms: float
    ee_processing_time_ms: float


@dataclass
class VendorProcessingResult:
    """Result from vendor data processing."""
    success: bool
    processed_data: Dict[str, Any]
    processing_time_ms: float
    errors: List[str]
    warnings: List[str]


class ComponentLayerEngine:
    """Enhanced Component Layer Engine with comprehensive vendor data processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ComponentLayerEngine with vendor processors."""
        self.logger = logging.getLogger('ComponentLayerEngine')
        self.config = config or {}
        
        # Initialize vendor processors
        self.cad_processor = CADProcessor()
        self.api_processor = APIProcessor()
        self.ee_processor = EEProcessor()
        
        # Initialize component type processors
        self.component_processors = {
            ComponentType.RESISTOR: ResistorProcessor(),
            ComponentType.CAPACITOR: CapacitorProcessor(),
            ComponentType.IC: ICProcessor(),
            ComponentType.INDUCTOR: InductorProcessor()
        }
        
        # Performance tracking
        self.processing_metrics: List[ProcessingMetrics] = []
        self.total_components_processed = 0
        self.performance_target_ms = 100  # Week 2 target: <100ms per component
        
        self.logger.info("ComponentLayerEngine initialized with vendor interfaces")
    
    def process_raw_component_data(self, raw_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw component data through vendor interfaces and component processors."""
        start_time = time.time()
        
        processed_components = []
        processing_errors = []
        processing_warnings = []
        
        for component_data in raw_components:
            try:
                # Process individual component
                result = self._process_single_component(component_data)
                
                if result['success']:
                    processed_components.append(result['component'])
                else:
                    processing_errors.extend(result['errors'])
                    
                processing_warnings.extend(result['warnings'])
                
            except Exception as e:
                self.logger.error(f"Failed to process component {component_data.get('component_id', 'unknown')}: {e}")
                processing_errors.append(str(e))
        
        # Calculate processing metrics
        total_time_ms = (time.time() - start_time) * 1000
        success_rate = len(processed_components) / len(raw_components) if raw_components else 0
        
        metrics = ProcessingMetrics(
            processing_time_ms=total_time_ms,
            components_processed=len(raw_components),
            success_rate=success_rate,
            avg_component_time_ms=total_time_ms / len(raw_components) if raw_components else 0,
            cad_processing_time_ms=0,  # Updated by individual processors
            api_processing_time_ms=0,
            ee_processing_time_ms=0
        )
        
        self.processing_metrics.append(metrics)
        self.total_components_processed += len(raw_components)
        
        # Performance validation
        performance_met = metrics.avg_component_time_ms < self.performance_target_ms
        
        return {
            'success': len(processing_errors) == 0,
            'processed_components': processed_components,
            'metrics': asdict(metrics),
            'performance_target_met': performance_met,
            'errors': processing_errors,
            'warnings': processing_warnings,
            'timestamp': time.time()
        }
    
    def _process_single_component(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single component through all vendor interfaces."""
        component_start_time = time.time()
        
        component_id = component_data.get('component_id', 'unknown')
        component_type_str = component_data.get('component_type', 'Unknown')
        
        try:
            component_type = ComponentType(component_type_str)
        except ValueError:
            component_type = ComponentType.UNKNOWN
        
        result = {
            'success': True,
            'component': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: CAD Data Processing
            cad_result = self._process_cad_data(component_data.get('cad_data', {}))
            if not cad_result.success:
                result['errors'].extend(cad_result.errors)
                result['warnings'].extend(cad_result.warnings)
            
            # Step 2: API Data Processing
            api_result = self._process_api_data(component_data.get('api_data', {}))
            if not api_result.success:
                result['errors'].extend(api_result.errors)
                result['warnings'].extend(api_result.warnings)
            
            # Step 3: EE Data Processing  
            ee_result = self._process_ee_data(component_data.get('ee_data', {}))
            if not ee_result.success:
                result['errors'].extend(ee_result.errors)
                result['warnings'].extend(ee_result.warnings)
            
            # Step 4: Component Type-Specific Processing
            type_result = self._process_component_type(component_type, {
                'cad_data': cad_result.processed_data,
                'api_data': api_result.processed_data,
                'ee_data': ee_result.processed_data,
                'raw_data': component_data
            })
            
            # Step 5: Generate Discrete Event Profile
            event_profile = self._generate_discrete_event_profile(component_type, type_result)
            
            # Assemble final processed component
            processed_component = {
                'component_id': component_id,
                'component_type': component_type.value,
                'processed_cad_data': cad_result.processed_data,
                'processed_api_data': api_result.processed_data,
                'processed_ee_data': ee_result.processed_data,
                'type_specific_data': type_result,
                'discrete_event_profile': event_profile,
                'processing_time_ms': (time.time() - component_start_time) * 1000,
                'processed_timestamp': time.time(),
                'vendor_id': component_data.get('vendor_id'),
                'validation_status': 'processed'
            }
            
            result['component'] = processed_component
            
            # Check for critical errors
            if result['errors']:
                result['success'] = False
                
        except Exception as e:
            result['success'] = False
            result['errors'].append(f"Component processing failed: {str(e)}")
            self.logger.error(f"Failed to process component {component_id}: {e}")
        
        return result
    
    def _process_cad_data(self, cad_data: Dict[str, Any]) -> VendorProcessingResult:
        """Process CAD data using CADProcessor."""
        start_time = time.time()
        
        try:
            processed_data = self.cad_processor.process_cad_data(cad_data)
            processing_time = (time.time() - start_time) * 1000
            
            return VendorProcessingResult(
                success=True,
                processed_data=processed_data,
                processing_time_ms=processing_time,
                errors=[],
                warnings=[]
            )
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return VendorProcessingResult(
                success=False,
                processed_data={},
                processing_time_ms=processing_time,
                errors=[f"CAD processing failed: {str(e)}"],
                warnings=[]
            )
    
    def _process_api_data(self, api_data: Dict[str, Any]) -> VendorProcessingResult:
        """Process API data using APIProcessor."""
        start_time = time.time()
        
        try:
            processed_data = self.api_processor.process_api_data(api_data)
            processing_time = (time.time() - start_time) * 1000
            
            return VendorProcessingResult(
                success=True,
                processed_data=processed_data,
                processing_time_ms=processing_time,
                errors=[],
                warnings=[]
            )
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return VendorProcessingResult(
                success=False,
                processed_data={},
                processing_time_ms=processing_time,
                errors=[f"API processing failed: {str(e)}"],
                warnings=[]
            )
    
    def _process_ee_data(self, ee_data: Dict[str, Any]) -> VendorProcessingResult:
        """Process EE data using EEProcessor."""
        start_time = time.time()
        
        try:
            processed_data = self.ee_processor.process_ee_data(ee_data)
            processing_time = (time.time() - start_time) * 1000
            
            return VendorProcessingResult(
                success=True,
                processed_data=processed_data,
                processing_time_ms=processing_time,
                errors=[],
                warnings=[]
            )
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return VendorProcessingResult(
                success=False,
                processed_data={},
                processing_time_ms=processing_time,
                errors=[f"EE processing failed: {str(e)}"],
                warnings=[]
            )
    
    def _process_component_type(self, component_type: ComponentType, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process component using type-specific processor."""
        if component_type in self.component_processors:
            processor = self.component_processors[component_type]
            return processor.process(processed_data)
        else:
            # Generic processing for unknown component types
            return {
                'type': component_type.value,
                'processing_method': 'generic',
                'placement_profile': 'manual_place',
                'estimated_time_s': 5.0,
                'complexity_rating': 'medium'
            }
    
    def _generate_discrete_event_profile(self, component_type: ComponentType, type_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate discrete event profile for manufacturing simulation."""
        
        # Component-specific event profiles
        event_profiles = {
            ComponentType.RESISTOR: {
                'event_type': 'smt_place_passive',
                'placement_time_s': 0.5,
                'pick_time_s': 0.2,
                'vision_time_s': 0.1,
                'placement_accuracy': 0.02,  # ±20µm
                'placement_force_n': 2.0
            },
            ComponentType.CAPACITOR: {
                'event_type': 'smt_place_passive',
                'placement_time_s': 0.6,
                'pick_time_s': 0.2,
                'vision_time_s': 0.1,
                'placement_accuracy': 0.02,
                'placement_force_n': 2.5
            },
            ComponentType.IC: {
                'event_type': 'smt_place_ic',
                'placement_time_s': 2.1,
                'pick_time_s': 0.3,
                'vision_time_s': 0.5,
                'placement_accuracy': 0.01,  # ±10µm for ICs
                'placement_force_n': 1.0,
                'requires_vision_alignment': True
            },
            ComponentType.INDUCTOR: {
                'event_type': 'smt_place_inductor', 
                'placement_time_s': 0.8,
                'pick_time_s': 0.2,
                'vision_time_s': 0.1,
                'placement_accuracy': 0.03,
                'placement_force_n': 3.0
            }
        }
        
        base_profile = event_profiles.get(component_type, {
            'event_type': 'manual_place',
            'placement_time_s': 5.0,
            'pick_time_s': 1.0,
            'vision_time_s': 0.5,
            'placement_accuracy': 0.1,
            'placement_force_n': 5.0
        })
        
        # Add component-specific modifications
        profile = base_profile.copy()
        profile.update({
            'component_type': component_type.value,
            'total_cycle_time_s': base_profile['placement_time_s'] + base_profile['pick_time_s'] + base_profile['vision_time_s'],
            'generated_timestamp': time.time(),
            'profile_version': '2.0'
        })
        
        return profile
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for Week 2 validation."""
        if not self.processing_metrics:
            return {'error': 'No processing metrics available'}
        
        latest_metrics = self.processing_metrics[-1]
        avg_processing_time = sum(m.avg_component_time_ms for m in self.processing_metrics) / len(self.processing_metrics)
        
        return {
            'total_components_processed': self.total_components_processed,
            'latest_processing_time_ms': latest_metrics.processing_time_ms,
            'latest_avg_component_time_ms': latest_metrics.avg_component_time_ms,
            'overall_avg_component_time_ms': avg_processing_time,
            'performance_target_ms': self.performance_target_ms,
            'performance_target_met': avg_processing_time < self.performance_target_ms,
            'success_rate': latest_metrics.success_rate,
            'vendor_processors_active': {
                'cad_processor': self.cad_processor is not None,
                'api_processor': self.api_processor is not None, 
                'ee_processor': self.ee_processor is not None
            },
            'component_types_supported': [ct.value for ct in ComponentType],
            'processing_sessions': len(self.processing_metrics)
        }
    
    def validate_week2_requirements(self) -> Dict[str, Any]:
        """Validate Week 2 specific requirements."""
        performance_summary = self.get_performance_summary()
        
        validations = {
            'vendor_interfaces_implemented': {
                'cad_processor': isinstance(self.cad_processor, CADProcessor),
                'api_processor': isinstance(self.api_processor, APIProcessor),
                'ee_processor': isinstance(self.ee_processor, EEProcessor)
            },
            'component_type_processors': {
                'resistor': ComponentType.RESISTOR in self.component_processors,
                'capacitor': ComponentType.CAPACITOR in self.component_processors,
                'ic': ComponentType.IC in self.component_processors,
                'inductor': ComponentType.INDUCTOR in self.component_processors
            },
            'performance_requirements': {
                'target_met': performance_summary.get('performance_target_met', False),
                'avg_processing_time_ms': performance_summary.get('overall_avg_component_time_ms', 999),
                'target_time_ms': self.performance_target_ms
            },
            'discrete_event_profiles': {
                'smt_place_passive': True,  # Resistor, Capacitor
                'smt_place_ic': True,       # IC
                'smt_place_inductor': True, # Inductor
                'manual_place': True        # Unknown types
            }
        }
        
        all_vendor_interfaces = all(validations['vendor_interfaces_implemented'].values())
        all_component_processors = all(validations['component_type_processors'].values())
        performance_met = validations['performance_requirements']['target_met']
        
        return {
            'overall_validation': all_vendor_interfaces and all_component_processors and performance_met,
            'validations': validations,
            'summary': {
                'vendor_interfaces_ready': all_vendor_interfaces,
                'component_processors_ready': all_component_processors,
                'performance_target_achieved': performance_met
            }
        }