"""Standard Layer Interface and Data Socket Architecture for Manufacturing System."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid


class LayerType(Enum):
    """Manufacturing system layer types."""
    COMPONENT = "component"
    STATION = "station" 
    LINE = "line"
    PM = "pm"  # Product Management


@dataclass
class DiscreteEventProfile:
    """Discrete event profile for components/stations/lines."""
    event_name: str
    duration: float
    frequency: float = 1.0  # Events per hour
    variability: float = 0.1  # ±10% duration variability
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_name': self.event_name,
            'duration': self.duration,
            'frequency': self.frequency,
            'variability': self.variability,
            'dependencies': self.dependencies
        }


@dataclass
class RawComponentData:
    """Raw component data from vendors (CAD, API, EE data)."""
    component_id: str
    component_type: str  # Resistor, Capacitor, IC, etc.
    cad_data: Dict[str, Any]  # Physical dimensions, footprint
    api_data: Dict[str, Any]  # Availability, pricing, lead time
    ee_data: Dict[str, Any]   # Electrical specifications
    vendor_id: str
    upload_timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'cad_data': self.cad_data,
            'api_data': self.api_data,
            'ee_data': self.ee_data,
            'vendor_id': self.vendor_id,
            'upload_timestamp': self.upload_timestamp
        }


@dataclass
class StructuredComponentData:
    """Structured component data output from Component Layer Engine."""
    component_id: str
    size: str
    price: float
    lead_time: int  # days
    discrete_event_profile: DiscreteEventProfile
    physical_properties: Dict[str, Any]
    electrical_properties: Dict[str, Any]
    availability: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'size': self.size,
            'price': self.price,
            'lead_time': self.lead_time,
            'discrete_event_profile': self.discrete_event_profile.to_dict(),
            'physical_properties': self.physical_properties,
            'electrical_properties': self.electrical_properties,
            'availability': self.availability
        }


@dataclass
class RawStationData:
    """Raw station data including component data and test coverage."""
    station_id: str
    station_type: str
    component_raw_data: List[RawComponentData]
    test_coverage: Dict[str, Any]
    operator_requirements: Dict[str, Any]
    footprint_constraints: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'station_id': self.station_id,
            'station_type': self.station_type,
            'component_raw_data': [comp.to_dict() for comp in self.component_raw_data],
            'test_coverage': self.test_coverage,
            'operator_requirements': self.operator_requirements,
            'footprint_constraints': self.footprint_constraints
        }


@dataclass
class StructuredStationData:
    """Structured station data output from Station Layer Engine."""
    station_id: str
    station_cost: float
    station_lead_time: int  # months
    station_operators: int
    station_amount: int
    station_footprint: float  # sqm^2
    discrete_event_profile: DiscreteEventProfile
    component_data: List[StructuredComponentData]
    efficiency_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'station_id': self.station_id,
            'station_cost': self.station_cost,
            'station_lead_time': self.station_lead_time,
            'station_operators': self.station_operators,
            'station_amount': self.station_amount,
            'station_footprint': self.station_footprint,
            'discrete_event_profile': self.discrete_event_profile.to_dict(),
            'component_data': [comp.to_dict() for comp in self.component_data],
            'efficiency_metrics': self.efficiency_metrics
        }


@dataclass
class RawLineData:
    """Raw line data including station data, DUT data, and policies."""
    line_id: str
    station_raw_data: List[RawStationData]
    dut_raw_data: Dict[str, Any]  # DUT pass/fail statistics
    operator_raw_data: Dict[str, Any]  # Operator constraints and capabilities
    retest_policy: str
    total_capacity: int  # DUTs per hour target
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'line_id': self.line_id,
            'station_raw_data': [station.to_dict() for station in self.station_raw_data],
            'dut_raw_data': self.dut_raw_data,
            'operator_raw_data': self.operator_raw_data,
            'retest_policy': self.retest_policy,
            'total_capacity': self.total_capacity
        }


@dataclass
class StructuredLineData:
    """Structured line data output from Line Layer Engine."""
    line_id: str
    line_cost: float
    line_lead_time: int  # months
    line_operators: int
    line_amount: int
    line_uph: int  # Units per hour
    line_footprint: float  # sqm^2
    line_efficiency: float
    station_data: List[StructuredStationData]
    optimization_parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'line_id': self.line_id,
            'line_cost': self.line_cost,
            'line_lead_time': self.line_lead_time,
            'line_operators': self.line_operators,
            'line_amount': self.line_amount,
            'line_uph': self.line_uph,
            'line_footprint': self.line_footprint,
            'line_efficiency': self.line_efficiency,
            'station_data': [station.to_dict() for station in self.station_data],
            'optimization_parameters': self.optimization_parameters
        }


class MOSAlgoEngine(ABC):
    """Abstract MOS Algorithm Engine for layer data processing."""
    
    def __init__(self, layer_type: LayerType):
        self.layer_type = layer_type
        self.engine_id = str(uuid.uuid4())
        self.processing_stats = {
            'total_processed': 0,
            'processing_time': 0.0,
            'last_update': time.time()
        }
    
    @abstractmethod
    def process(self, raw_data: Any) -> Any:
        """Process raw data into structured format."""
        pass
    
    @abstractmethod
    def validate_input(self, raw_data: Any) -> bool:
        """Validate input data format."""
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output data schema."""
        pass
    
    def update_stats(self, processing_time: float):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['processing_time'] += processing_time
        self.processing_stats['last_update'] = time.time()


class ComponentLayerEngine(MOSAlgoEngine):
    """Component Layer MOS Algorithm Engine."""
    
    def __init__(self):
        super().__init__(LayerType.COMPONENT)
    
    def process(self, raw_data: RawComponentData) -> StructuredComponentData:
        """Process raw component data into structured format."""
        start_time = time.time()
        
        if not self.validate_input(raw_data):
            raise ValueError("Invalid raw component data")
        
        # Extract key information from raw data
        size = self._extract_size(raw_data.cad_data)
        price = self._extract_price(raw_data.api_data)
        lead_time = self._extract_lead_time(raw_data.api_data)
        
        # Create discrete event profile
        discrete_event_profile = self._create_discrete_event_profile(raw_data)
        
        # Extract properties
        physical_properties = self._extract_physical_properties(raw_data.cad_data)
        electrical_properties = self._extract_electrical_properties(raw_data.ee_data)
        availability = self._extract_availability(raw_data.api_data)
        
        structured_data = StructuredComponentData(
            component_id=raw_data.component_id,
            size=size,
            price=price,
            lead_time=lead_time,
            discrete_event_profile=discrete_event_profile,
            physical_properties=physical_properties,
            electrical_properties=electrical_properties,
            availability=availability
        )
        
        processing_time = time.time() - start_time
        self.update_stats(processing_time)
        
        return structured_data
    
    def validate_input(self, raw_data: RawComponentData) -> bool:
        """Validate raw component data."""
        required_fields = ['component_id', 'component_type', 'cad_data', 'api_data', 'ee_data']
        return all(hasattr(raw_data, field) for field in required_fields)
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get component output schema."""
        return {
            'component_id': 'string',
            'size': 'string',
            'price': 'float',
            'lead_time': 'integer',
            'discrete_event_profile': 'object',
            'physical_properties': 'object',
            'electrical_properties': 'object',
            'availability': 'object'
        }
    
    def _extract_size(self, cad_data: Dict[str, Any]) -> str:
        """Extract component size from CAD data."""
        # Example extraction logic
        if 'dimensions' in cad_data:
            dims = cad_data['dimensions']
            if 'package' in dims:
                return dims['package']
        return "UNKNOWN"
    
    def _extract_price(self, api_data: Dict[str, Any]) -> float:
        """Extract price from API data."""
        return api_data.get('price_usd', 0.0)
    
    def _extract_lead_time(self, api_data: Dict[str, Any]) -> int:
        """Extract lead time from API data."""
        return api_data.get('lead_time_days', 30)
    
    def _create_discrete_event_profile(self, raw_data: RawComponentData) -> DiscreteEventProfile:
        """Create discrete event profile for component."""
        # Example: SMT placement event based on component type
        if raw_data.component_type.lower() in ['resistor', 'capacitor']:
            return DiscreteEventProfile(
                event_name="smt_place_passive",
                duration=0.5,  # 0.5 seconds per placement
                frequency=7200,  # 7200 placements per hour max
                variability=0.1
            )
        elif raw_data.component_type.lower() == 'ic':
            return DiscreteEventProfile(
                event_name="smt_place_ic",
                duration=2.0,  # 2 seconds per IC placement
                frequency=1800,  # 1800 placements per hour max
                variability=0.2
            )
        else:
            return DiscreteEventProfile(
                event_name="manual_place",
                duration=10.0,  # 10 seconds manual placement
                frequency=360,   # 360 placements per hour max
                variability=0.3
            )
    
    def _extract_physical_properties(self, cad_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract physical properties."""
        return {
            'length': cad_data.get('length_mm', 0.0),
            'width': cad_data.get('width_mm', 0.0),
            'height': cad_data.get('height_mm', 0.0),
            'weight': cad_data.get('weight_g', 0.0)
        }
    
    def _extract_electrical_properties(self, ee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract electrical properties."""
        return {
            'voltage_rating': ee_data.get('voltage_v', 0.0),
            'current_rating': ee_data.get('current_a', 0.0),
            'power_rating': ee_data.get('power_w', 0.0),
            'tolerance': ee_data.get('tolerance_percent', 0.0)
        }
    
    def _extract_availability(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract availability information."""
        return {
            'in_stock': api_data.get('in_stock', False),
            'stock_quantity': api_data.get('stock_qty', 0),
            'minimum_order': api_data.get('moq', 1),
            'supplier': api_data.get('supplier', 'UNKNOWN')
        }


class StationLayerEngine(MOSAlgoEngine):
    """Station Layer MOS Algorithm Engine."""
    
    def __init__(self):
        super().__init__(LayerType.STATION)
    
    def process(self, raw_data: RawStationData) -> StructuredStationData:
        """Process raw station data into structured format."""
        start_time = time.time()
        
        if not self.validate_input(raw_data):
            raise ValueError("Invalid raw station data")
        
        # Process component data through Component Layer Engine
        component_engine = ComponentLayerEngine()
        structured_components = []
        
        for raw_component in raw_data.component_raw_data:
            structured_component = component_engine.process(raw_component)
            structured_components.append(structured_component)
        
        # Calculate station-level parameters
        station_cost = self._calculate_station_cost(structured_components, raw_data)
        station_lead_time = self._calculate_station_lead_time(structured_components)
        station_operators = self._calculate_station_operators(raw_data)
        station_amount = self._calculate_station_amount(raw_data)
        station_footprint = self._calculate_station_footprint(raw_data)
        
        # Create station-level discrete event profile
        discrete_event_profile = self._create_station_discrete_event_profile(raw_data, structured_components)
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(raw_data, structured_components)
        
        structured_data = StructuredStationData(
            station_id=raw_data.station_id,
            station_cost=station_cost,
            station_lead_time=station_lead_time,
            station_operators=station_operators,
            station_amount=station_amount,
            station_footprint=station_footprint,
            discrete_event_profile=discrete_event_profile,
            component_data=structured_components,
            efficiency_metrics=efficiency_metrics
        )
        
        processing_time = time.time() - start_time
        self.update_stats(processing_time)
        
        return structured_data
    
    def validate_input(self, raw_data: RawStationData) -> bool:
        """Validate raw station data."""
        return (hasattr(raw_data, 'station_id') and 
                hasattr(raw_data, 'component_raw_data') and
                isinstance(raw_data.component_raw_data, list))
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get station output schema."""
        return {
            'station_id': 'string',
            'station_cost': 'float',
            'station_lead_time': 'integer',
            'station_operators': 'integer',
            'station_amount': 'integer',
            'station_footprint': 'float',
            'discrete_event_profile': 'object',
            'component_data': 'array',
            'efficiency_metrics': 'object'
        }
    
    def _calculate_station_cost(self, components: List[StructuredComponentData], raw_data: RawStationData) -> float:
        """Calculate total station cost."""
        component_cost = sum(comp.price for comp in components)
        equipment_cost = raw_data.test_coverage.get('equipment_cost', 100000.0)
        setup_cost = raw_data.footprint_constraints.get('setup_cost', 50000.0)
        return component_cost + equipment_cost + setup_cost
    
    def _calculate_station_lead_time(self, components: List[StructuredComponentData]) -> int:
        """Calculate station lead time in months."""
        max_component_lead_time = max((comp.lead_time for comp in components), default=30) / 30  # Convert to months
        equipment_lead_time = 4  # 4 months for equipment
        return int(max(max_component_lead_time, equipment_lead_time))
    
    def _calculate_station_operators(self, raw_data: RawStationData) -> int:
        """Calculate required operators."""
        return raw_data.operator_requirements.get('operators_required', 1)
    
    def _calculate_station_amount(self, raw_data: RawStationData) -> int:
        """Calculate station amount/quantity."""
        return raw_data.footprint_constraints.get('quantity', 1)
    
    def _calculate_station_footprint(self, raw_data: RawStationData) -> float:
        """Calculate station footprint in sqm^2."""
        return raw_data.footprint_constraints.get('area_sqm', 10.0)
    
    def _create_station_discrete_event_profile(self, raw_data: RawStationData, 
                                               components: List[StructuredComponentData]) -> DiscreteEventProfile:
        """Create station-level discrete event profile."""
        # Aggregate component processing times
        total_component_time = sum(comp.discrete_event_profile.duration for comp in components)
        test_time = raw_data.test_coverage.get('total_test_time', 30.0)
        
        total_cycle_time = total_component_time + test_time
        
        return DiscreteEventProfile(
            event_name=f"{raw_data.station_type}_process_cycle",
            duration=total_cycle_time,
            frequency=3600 / total_cycle_time,  # Cycles per hour
            variability=0.15
        )
    
    def _calculate_efficiency_metrics(self, raw_data: RawStationData,
                                      components: List[StructuredComponentData]) -> Dict[str, float]:
        """Calculate station efficiency metrics."""
        return {
            'theoretical_uph': 3600.0 / (sum(comp.discrete_event_profile.duration for comp in components) + 30.0),
            'yield_target': 0.95,
            'oee_target': 0.85,  # Overall Equipment Effectiveness
            'component_count': len(components)
        }


class LineLayerEngine(MOSAlgoEngine):
    """Line Layer MOS Algorithm Engine."""
    
    def __init__(self):
        super().__init__(LayerType.LINE)
    
    def process(self, raw_data: RawLineData) -> StructuredLineData:
        """Process raw line data into structured format."""
        start_time = time.time()
        
        if not self.validate_input(raw_data):
            raise ValueError("Invalid raw line data")
        
        # Process station data through Station Layer Engine
        station_engine = StationLayerEngine()
        structured_stations = []
        
        for raw_station in raw_data.station_raw_data:
            structured_station = station_engine.process(raw_station)
            structured_stations.append(structured_station)
        
        # Calculate line-level parameters
        line_cost = sum(station.station_cost for station in structured_stations)
        line_lead_time = max((station.station_lead_time for station in structured_stations), default=4)
        line_operators = sum(station.station_operators for station in structured_stations)
        line_amount = len(structured_stations)
        line_footprint = sum(station.station_footprint for station in structured_stations)
        
        # Calculate line UPH and efficiency
        line_uph = self._calculate_line_uph(structured_stations, raw_data)
        line_efficiency = self._calculate_line_efficiency(structured_stations, raw_data)
        
        # Create optimization parameters
        optimization_parameters = self._create_optimization_parameters(structured_stations, raw_data)
        
        structured_data = StructuredLineData(
            line_id=raw_data.line_id,
            line_cost=line_cost,
            line_lead_time=line_lead_time,
            line_operators=line_operators,
            line_amount=line_amount,
            line_uph=line_uph,
            line_footprint=line_footprint,
            line_efficiency=line_efficiency,
            station_data=structured_stations,
            optimization_parameters=optimization_parameters
        )
        
        processing_time = time.time() - start_time
        self.update_stats(processing_time)
        
        return structured_data
    
    def validate_input(self, raw_data: RawLineData) -> bool:
        """Validate raw line data."""
        return (hasattr(raw_data, 'line_id') and
                hasattr(raw_data, 'station_raw_data') and
                isinstance(raw_data.station_raw_data, list))
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get line output schema."""
        return {
            'line_id': 'string',
            'line_cost': 'float',
            'line_lead_time': 'integer',
            'line_operators': 'integer',
            'line_amount': 'integer',
            'line_uph': 'integer',
            'line_footprint': 'float',
            'line_efficiency': 'float',
            'station_data': 'array',
            'optimization_parameters': 'object'
        }
    
    def _calculate_line_uph(self, stations: List[StructuredStationData], raw_data: RawLineData) -> int:
        """Calculate line UPH (Units Per Hour)."""
        # Line UPH is limited by the slowest station (bottleneck)
        station_uphs = [station.discrete_event_profile.frequency for station in stations]
        bottleneck_uph = min(station_uphs) if station_uphs else 60
        
        # Apply yield losses
        overall_yield = 0.95 ** len(stations)  # Compound yield across stations
        effective_uph = int(bottleneck_uph * overall_yield)
        
        return min(effective_uph, raw_data.total_capacity)
    
    def _calculate_line_efficiency(self, stations: List[StructuredStationData], raw_data: RawLineData) -> float:
        """Calculate overall line efficiency."""
        station_efficiencies = [station.efficiency_metrics.get('oee_target', 0.85) for station in stations]
        overall_efficiency = 1.0
        for eff in station_efficiencies:
            overall_efficiency *= eff
        
        return overall_efficiency
    
    def _create_optimization_parameters(self, stations: List[StructuredStationData], 
                                        raw_data: RawLineData) -> Dict[str, Any]:
        """Create parameters for AI optimization."""
        return {
            'target_uph': raw_data.total_capacity,
            'yield_target': 0.90,
            'cost_target': sum(station.station_cost for station in stations) * 0.9,
            'bottleneck_stations': self._identify_bottlenecks(stations),
            'retest_policy': raw_data.retest_policy,
            'dut_profile': raw_data.dut_raw_data
        }
    
    def _identify_bottlenecks(self, stations: List[StructuredStationData]) -> List[str]:
        """Identify bottleneck stations."""
        if not stations:
            return []
        
        min_frequency = min(station.discrete_event_profile.frequency for station in stations)
        bottlenecks = [station.station_id for station in stations 
                       if station.discrete_event_profile.frequency <= min_frequency * 1.1]  # Within 10%
        
        return bottlenecks


class StandardDataSocket:
    """Standard socket interface for data exchange between layers."""
    
    def __init__(self, socket_id: str, input_layer: LayerType, output_layer: LayerType):
        self.socket_id = socket_id
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.created_time = time.time()
        self.transfer_count = 0
        
        # Select appropriate engine based on output layer
        self.engine = self._create_engine(output_layer)
    
    def _create_engine(self, layer_type: LayerType) -> MOSAlgoEngine:
        """Create appropriate MOS Algorithm Engine."""
        if layer_type == LayerType.COMPONENT:
            return ComponentLayerEngine()
        elif layer_type == LayerType.STATION:
            return StationLayerEngine()
        elif layer_type == LayerType.LINE:
            return LineLayerEngine()
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    def transfer(self, input_data: Any) -> Any:
        """Transfer data through the socket with processing."""
        self.transfer_count += 1
        
        # Validate and process data
        processed_data = self.engine.process(input_data)
        
        return processed_data
    
    def get_socket_info(self) -> Dict[str, Any]:
        """Get socket information."""
        return {
            'socket_id': self.socket_id,
            'input_layer': self.input_layer.value,
            'output_layer': self.output_layer.value,
            'engine_type': type(self.engine).__name__,
            'transfer_count': self.transfer_count,
            'created_time': self.created_time,
            'processing_stats': self.engine.processing_stats
        }