"""Socket Manager for Standard Data Exchange Between Manufacturing Layers."""

from typing import Dict, List, Any, Optional
import json
import time
import logging
from pathlib import Path

from .layer_interface import (
    StandardDataSocket, LayerType, MOSAlgoEngine,
    RawComponentData, StructuredComponentData,
    RawStationData, StructuredStationData,
    RawLineData, StructuredLineData,
    DiscreteEventProfile
)


class SocketManager:
    """Manages all standard data sockets between manufacturing layers."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.sockets: Dict[str, StandardDataSocket] = {}
        self.layer_connections: Dict[LayerType, List[str]] = {
            LayerType.COMPONENT: [],
            LayerType.STATION: [],
            LayerType.LINE: [],
            LayerType.PM: []
        }
        
        self.logger = logging.getLogger('SocketManager')
        self.config_file = config_file
        
        # Initialize default sockets
        self._initialize_default_sockets()
    
    def _initialize_default_sockets(self):
        """Initialize default manufacturing layer sockets."""
        
        # Component to Station socket
        self.create_socket(
            socket_id="component_to_station",
            input_layer=LayerType.COMPONENT,
            output_layer=LayerType.STATION
        )
        
        # Station to Line socket
        self.create_socket(
            socket_id="station_to_line", 
            input_layer=LayerType.STATION,
            output_layer=LayerType.LINE
        )
        
        # Note: PM layer socket would be created when PM layer engine is available
        # For now, we'll focus on Component -> Station -> Line pipeline
    
    def create_socket(self, socket_id: str, input_layer: LayerType, output_layer: LayerType) -> StandardDataSocket:
        """Create a new standard data socket."""
        if socket_id in self.sockets:
            raise ValueError(f"Socket {socket_id} already exists")
        
        socket = StandardDataSocket(socket_id, input_layer, output_layer)
        self.sockets[socket_id] = socket
        
        # Track layer connections
        self.layer_connections[output_layer].append(socket_id)
        
        self.logger.info(f"Created socket: {socket_id} ({input_layer.value} -> {output_layer.value})")
        return socket
    
    def get_socket(self, socket_id: str) -> Optional[StandardDataSocket]:
        """Get socket by ID."""
        return self.sockets.get(socket_id)
    
    def transfer_data(self, socket_id: str, input_data: Any) -> Any:
        """Transfer data through specified socket."""
        socket = self.get_socket(socket_id)
        if not socket:
            raise ValueError(f"Socket {socket_id} not found")
        
        try:
            result = socket.transfer(input_data)
            self.logger.debug(f"Data transferred through socket: {socket_id}")
            return result
        except Exception as e:
            self.logger.error(f"Transfer failed for socket {socket_id}: {e}")
            raise
    
    def process_component_data(self, raw_components: List[Dict[str, Any]]) -> List[StructuredComponentData]:
        """Process raw component data through Component Layer."""
        # Use ComponentLayerEngine directly since we need to process raw components
        from .layer_interface import ComponentLayerEngine
        component_engine = ComponentLayerEngine()
        
        structured_components = []
        
        for raw_comp_dict in raw_components:
            # Convert dict to RawComponentData
            raw_component = RawComponentData(
                component_id=raw_comp_dict['component_id'],
                component_type=raw_comp_dict['component_type'],
                cad_data=raw_comp_dict.get('cad_data', {}),
                api_data=raw_comp_dict.get('api_data', {}),
                ee_data=raw_comp_dict.get('ee_data', {}),
                vendor_id=raw_comp_dict.get('vendor_id', 'UNKNOWN')
            )
            
            # Process through Component Layer Engine
            structured_component = component_engine.process(raw_component)
            structured_components.append(structured_component)
        
        self.logger.info(f"Processed {len(structured_components)} components")
        return structured_components
    
    def process_station_data(self, raw_station_dict: Dict[str, Any]) -> StructuredStationData:
        """Process raw station data through Station Layer."""
        # Use StationLayerEngine directly
        from .layer_interface import StationLayerEngine
        station_engine = StationLayerEngine()
        
        # Convert dict to RawStationData
        raw_components = []
        for comp_data in raw_station_dict.get('component_raw_data', []):
            raw_component = RawComponentData(
                component_id=comp_data['component_id'],
                component_type=comp_data['component_type'],
                cad_data=comp_data.get('cad_data', {}),
                api_data=comp_data.get('api_data', {}),
                ee_data=comp_data.get('ee_data', {}),
                vendor_id=comp_data.get('vendor_id', 'UNKNOWN')
            )
            raw_components.append(raw_component)
        
        raw_station = RawStationData(
            station_id=raw_station_dict['station_id'],
            station_type=raw_station_dict['station_type'],
            component_raw_data=raw_components,
            test_coverage=raw_station_dict.get('test_coverage', {}),
            operator_requirements=raw_station_dict.get('operator_requirements', {}),
            footprint_constraints=raw_station_dict.get('footprint_constraints', {})
        )
        
        # Process through Station Layer Engine
        structured_station = station_engine.process(raw_station)
        
        self.logger.info(f"Processed station: {raw_station.station_id}")
        return structured_station
    
    def process_line_data(self, raw_line_dict: Dict[str, Any]) -> StructuredLineData:
        """Process raw line data through Line Layer."""
        # Use LineLayerEngine directly
        from .layer_interface import LineLayerEngine
        line_engine = LineLayerEngine()
        
        # Convert dict to RawLineData
        raw_stations = []
        for station_data in raw_line_dict.get('station_raw_data', []):
            raw_components = []
            for comp_data in station_data.get('component_raw_data', []):
                raw_component = RawComponentData(
                    component_id=comp_data['component_id'],
                    component_type=comp_data['component_type'],
                    cad_data=comp_data.get('cad_data', {}),
                    api_data=comp_data.get('api_data', {}),
                    ee_data=comp_data.get('ee_data', {}),
                    vendor_id=comp_data.get('vendor_id', 'UNKNOWN')
                )
                raw_components.append(raw_component)
            
            raw_station = RawStationData(
                station_id=station_data['station_id'],
                station_type=station_data['station_type'],
                component_raw_data=raw_components,
                test_coverage=station_data.get('test_coverage', {}),
                operator_requirements=station_data.get('operator_requirements', {}),
                footprint_constraints=station_data.get('footprint_constraints', {})
            )
            raw_stations.append(raw_station)
        
        raw_line = RawLineData(
            line_id=raw_line_dict['line_id'],
            station_raw_data=raw_stations,
            dut_raw_data=raw_line_dict.get('dut_raw_data', {}),
            operator_raw_data=raw_line_dict.get('operator_raw_data', {}),
            retest_policy=raw_line_dict.get('retest_policy', 'AAB'),
            total_capacity=raw_line_dict.get('total_capacity', 100)
        )
        
        # Process through Line Layer Engine
        structured_line = line_engine.process(raw_line)
        
        self.logger.info(f"Processed line: {raw_line.line_id}")
        return structured_line
    
    def get_all_socket_info(self) -> Dict[str, Any]:
        """Get information about all sockets."""
        socket_info = {}
        
        for socket_id, socket in self.sockets.items():
            socket_info[socket_id] = socket.get_socket_info()
        
        return {
            'sockets': socket_info,
            'layer_connections': {k.value: v for k, v in self.layer_connections.items()},
            'total_sockets': len(self.sockets)
        }
    
    def export_structured_data(self, structured_data: Any, output_file: str):
        """Export structured data to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(structured_data, 'to_dict'):
            data_dict = structured_data.to_dict()
        else:
            data_dict = structured_data
        
        with open(output_path, 'w') as f:
            json.dump(data_dict, f, indent=2, default=str)
        
        self.logger.info(f"Exported structured data to: {output_file}")
    
    def load_raw_data_from_file(self, input_file: str) -> Dict[str, Any]:
        """Load raw data from file."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        
        self.logger.info(f"Loaded raw data from: {input_file}")
        return raw_data
    
    def validate_layer_data(self, layer_type: LayerType, data: Any) -> bool:
        """Validate data for specific layer."""
        socket_id = f"validate_{layer_type.value}"
        
        # Get appropriate socket or create temporary one
        socket = None
        for sock_id, sock in self.sockets.items():
            if sock.output_layer == layer_type:
                socket = sock
                break
        
        if not socket:
            return False
        
        try:
            return socket.engine.validate_input(data)
        except Exception as e:
            self.logger.error(f"Validation failed for {layer_type.value}: {e}")
            return False
    
    def get_layer_schema(self, layer_type: LayerType) -> Dict[str, Any]:
        """Get output schema for specific layer."""
        # Find socket with matching output layer
        for socket in self.sockets.values():
            if socket.output_layer == layer_type:
                return socket.engine.get_output_schema()
        
        return {}
    
    def demonstrate_full_pipeline(self, raw_data_file: str) -> Dict[str, Any]:
        """Demonstrate full data processing pipeline."""
        self.logger.info("Starting full pipeline demonstration")
        
        # Load raw data
        raw_data = self.load_raw_data_from_file(raw_data_file)
        
        results = {}
        
        # Process components if present
        if 'components' in raw_data:
            structured_components = self.process_component_data(raw_data['components'])
            results['structured_components'] = [comp.to_dict() for comp in structured_components]
        
        # Process stations if present
        if 'stations' in raw_data:
            structured_stations = []
            for station_data in raw_data['stations']:
                structured_station = self.process_station_data(station_data)
                structured_stations.append(structured_station)
            results['structured_stations'] = [station.to_dict() for station in structured_stations]
        
        # Process lines if present
        if 'lines' in raw_data:
            structured_lines = []
            for line_data in raw_data['lines']:
                structured_line = self.process_line_data(line_data)
                structured_lines.append(structured_line)
            results['structured_lines'] = [line.to_dict() for line in structured_lines]
        
        # Add processing metadata
        results['processing_metadata'] = {
            'timestamp': time.time(),
            'sockets_used': len(self.sockets),
            'socket_info': self.get_all_socket_info()
        }
        
        self.logger.info("Full pipeline demonstration completed")
        return results


# Global socket manager instance
global_socket_manager = SocketManager()