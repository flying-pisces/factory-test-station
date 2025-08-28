#!/usr/bin/env python3
"""Standard Data Socket Architecture Demonstration."""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

from layer_interface import (
    LayerType, RawComponentData, RawStationData, RawLineData,
    DiscreteEventProfile, ComponentLayerEngine, StationLayerEngine, LineLayerEngine
)
from socket_manager import SocketManager, global_socket_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SocketDemo')


def create_sample_raw_data() -> Dict[str, Any]:
    """Create sample raw data for demonstration."""
    
    # Sample raw component data (from vendors: CAD, API, EE)
    raw_components = [
        {
            'component_id': 'R1_0603',
            'component_type': 'Resistor',
            'cad_data': {
                'dimensions': {'package': '0603', 'length_mm': 1.6, 'width_mm': 0.8, 'height_mm': 0.45},
                'footprint': 'RES_0603'
            },
            'api_data': {
                'price_usd': 0.05,
                'lead_time_days': 14,
                'in_stock': True,
                'stock_qty': 10000,
                'supplier': 'Vendor_A'
            },
            'ee_data': {
                'resistance_ohm': 1000,
                'tolerance_percent': 5.0,
                'power_w': 0.1,
                'voltage_v': 50
            },
            'vendor_id': 'VENDOR_A_001'
        },
        {
            'component_id': 'C1_0603',
            'component_type': 'Capacitor',
            'cad_data': {
                'dimensions': {'package': '0603', 'length_mm': 1.6, 'width_mm': 0.8, 'height_mm': 0.8},
                'footprint': 'CAP_0603'
            },
            'api_data': {
                'price_usd': 0.08,
                'lead_time_days': 21,
                'in_stock': True,
                'stock_qty': 5000,
                'supplier': 'Vendor_B'
            },
            'ee_data': {
                'capacitance_f': 1e-6,  # 1µF
                'tolerance_percent': 10.0,
                'voltage_v': 25,
                'esr_ohm': 0.1
            },
            'vendor_id': 'VENDOR_B_001'
        },
        {
            'component_id': 'U1_QFN32',
            'component_type': 'IC',
            'cad_data': {
                'dimensions': {'package': 'QFN32', 'length_mm': 5.0, 'width_mm': 5.0, 'height_mm': 0.9},
                'footprint': 'QFN32_5X5'
            },
            'api_data': {
                'price_usd': 12.50,
                'lead_time_days': 60,
                'in_stock': False,
                'stock_qty': 0,
                'supplier': 'Vendor_C'
            },
            'ee_data': {
                'voltage_v': 3.3,
                'current_a': 0.5,
                'power_w': 1.65,
                'operating_temp_c': [-40, 85]
            },
            'vendor_id': 'VENDOR_C_001'
        }
    ]
    
    # Sample raw station data
    raw_stations = [
        {
            'station_id': 'SMT_P0',
            'station_type': 'SMT',
            'component_raw_data': raw_components,
            'test_coverage': {
                'test_item1': {'type': 'continuity', 'duration': 5.0},
                'test_item2': {'type': 'insulation', 'duration': 3.0},
                'total_test_time': 8.0,
                'equipment_cost': 150000.0
            },
            'operator_requirements': {
                'operators_required': 1,
                'skill_level': 'intermediate',
                'training_hours': 40
            },
            'footprint_constraints': {
                'area_sqm': 12.0,
                'quantity': 1,
                'setup_cost': 25000.0
            }
        },
        {
            'station_id': 'TEST_1',
            'station_type': 'TEST',
            'component_raw_data': [raw_components[0]],  # Only resistor for test station
            'test_coverage': {
                'test_item1': {'type': 'functional', 'duration': 15.0},
                'test_item2': {'type': 'parametric', 'duration': 20.0},
                'total_test_time': 35.0,
                'equipment_cost': 200000.0
            },
            'operator_requirements': {
                'operators_required': 1,
                'skill_level': 'advanced',
                'training_hours': 80
            },
            'footprint_constraints': {
                'area_sqm': 15.0,
                'quantity': 1,
                'setup_cost': 30000.0
            }
        }
    ]
    
    # Sample raw line data
    raw_lines = [
        {
            'line_id': 'SMT_FATP_LINE_01',
            'station_raw_data': raw_stations,
            'dut_raw_data': {
                'DUT_pass': 900,
                'DUT_fail': 40,
                'DUT_retest_pass': 20,
                'DUT_retest_fail': 40,
                'target_yield': 0.95
            },
            'operator_raw_data': {
                'size': '0.5 by 0.5',
                'total_operators': 10,
                'operator_cost_per_hour': 25.0
            },
            'retest_policy': 'AAB',
            'total_capacity': 100  # Target UPH
        }
    ]
    
    return {
        'components': raw_components,
        'stations': raw_stations,
        'lines': raw_lines
    }


def demonstrate_component_layer():
    """Demonstrate Component Layer Engine processing."""
    print("\n" + "="*80)
    print("🔧 COMPONENT LAYER ENGINE DEMONSTRATION")
    print("="*80)
    
    # Create Component Layer Engine
    component_engine = ComponentLayerEngine()
    
    print(f"Engine Type: {type(component_engine).__name__}")
    print(f"Layer Type: {component_engine.layer_type.value}")
    print(f"Output Schema: {component_engine.get_output_schema()}")
    
    # Create sample raw component data
    raw_component_data = RawComponentData(
        component_id="R1_0603",
        component_type="Resistor",
        cad_data={
            'dimensions': {'package': '0603', 'length_mm': 1.6, 'width_mm': 0.8},
            'footprint': 'RES_0603'
        },
        api_data={
            'price_usd': 0.05,
            'lead_time_days': 14,
            'in_stock': True,
            'supplier': 'Vendor_A'
        },
        ee_data={
            'resistance_ohm': 1000,
            'tolerance_percent': 5.0,
            'power_w': 0.1
        },
        vendor_id="VENDOR_A_001"
    )
    
    print(f"\n📥 Input (Raw Component Data):")
    print(f"  Component ID: {raw_component_data.component_id}")
    print(f"  Type: {raw_component_data.component_type}")
    print(f"  Vendor: {raw_component_data.vendor_id}")
    
    # Process through engine
    print(f"\n⚙️ Processing through MOS Algo-Engine...")
    structured_data = component_engine.process(raw_component_data)
    
    print(f"\n📤 Output (Structured Component Data):")
    print(f"  Component ID: {structured_data.component_id}")
    print(f"  Size: {structured_data.size}")
    print(f"  Price: ${structured_data.price:.3f}")
    print(f"  Lead Time: {structured_data.lead_time} days")
    print(f"  Discrete Event Profile:")
    print(f"    Event: {structured_data.discrete_event_profile.event_name}")
    print(f"    Duration: {structured_data.discrete_event_profile.duration}s")
    print(f"    Frequency: {structured_data.discrete_event_profile.frequency:.0f}/hour")
    
    return structured_data


def demonstrate_station_layer():
    """Demonstrate Station Layer Engine processing."""
    print("\n" + "="*80)
    print("🏭 STATION LAYER ENGINE DEMONSTRATION")
    print("="*80)
    
    # Create Station Layer Engine
    station_engine = StationLayerEngine()
    
    print(f"Engine Type: {type(station_engine).__name__}")
    print(f"Layer Type: {station_engine.layer_type.value}")
    
    # Create sample raw station data
    raw_components = [
        RawComponentData(
            component_id="R1_0603",
            component_type="Resistor",
            cad_data={'dimensions': {'package': '0603'}},
            api_data={'price_usd': 0.05, 'lead_time_days': 14},
            ee_data={'resistance_ohm': 1000, 'power_w': 0.1},
            vendor_id="VENDOR_A"
        ),
        RawComponentData(
            component_id="C1_0603", 
            component_type="Capacitor",
            cad_data={'dimensions': {'package': '0603'}},
            api_data={'price_usd': 0.08, 'lead_time_days': 21},
            ee_data={'capacitance_f': 1e-6, 'voltage_v': 25},
            vendor_id="VENDOR_B"
        )
    ]
    
    raw_station_data = RawStationData(
        station_id="SMT_P0",
        station_type="SMT",
        component_raw_data=raw_components,
        test_coverage={
            'total_test_time': 30.0,
            'equipment_cost': 150000.0
        },
        operator_requirements={'operators_required': 1},
        footprint_constraints={'area_sqm': 12.0, 'setup_cost': 25000.0}
    )
    
    print(f"\n📥 Input (Raw Station Data):")
    print(f"  Station ID: {raw_station_data.station_id}")
    print(f"  Type: {raw_station_data.station_type}")
    print(f"  Components: {len(raw_station_data.component_raw_data)}")
    
    # Process through engine  
    print(f"\n⚙️ Processing through MOS Algo-Engine...")
    structured_data = station_engine.process(raw_station_data)
    
    print(f"\n📤 Output (Structured Station Data):")
    print(f"  Station ID: {structured_data.station_id}")
    print(f"  Station Cost: ${structured_data.station_cost:,.2f}")
    print(f"  Lead Time: {structured_data.station_lead_time} months")
    print(f"  Operators: {structured_data.station_operators}")
    print(f"  Footprint: {structured_data.station_footprint} sqm²")
    print(f"  Discrete Event Profile:")
    print(f"    Event: {structured_data.discrete_event_profile.event_name}")
    print(f"    Duration: {structured_data.discrete_event_profile.duration:.1f}s")
    print(f"    Frequency: {structured_data.discrete_event_profile.frequency:.0f}/hour")
    
    return structured_data


def demonstrate_line_layer():
    """Demonstrate Line Layer Engine processing."""
    print("\n" + "="*80)
    print("🏗️ LINE LAYER ENGINE DEMONSTRATION")
    print("="*80)
    
    # Create Line Layer Engine
    line_engine = LineLayerEngine()
    
    print(f"Engine Type: {type(line_engine).__name__}")
    print(f"Layer Type: {line_engine.layer_type.value}")
    
    # Create sample raw line data with stations
    raw_stations = []
    
    # Station 1
    station1_components = [
        RawComponentData("R1", "Resistor", {}, {'price_usd': 0.05}, {}, "VENDOR_A"),
        RawComponentData("C1", "Capacitor", {}, {'price_usd': 0.08}, {}, "VENDOR_B")
    ]
    
    raw_stations.append(RawStationData(
        station_id="SMT_P0",
        station_type="SMT",
        component_raw_data=station1_components,
        test_coverage={'total_test_time': 25.0, 'equipment_cost': 150000.0},
        operator_requirements={'operators_required': 1},
        footprint_constraints={'area_sqm': 12.0, 'setup_cost': 25000.0}
    ))
    
    # Station 2
    station2_components = [
        RawComponentData("U1", "IC", {}, {'price_usd': 12.50}, {}, "VENDOR_C")
    ]
    
    raw_stations.append(RawStationData(
        station_id="TEST_1",
        station_type="TEST",
        component_raw_data=station2_components,
        test_coverage={'total_test_time': 35.0, 'equipment_cost': 200000.0},
        operator_requirements={'operators_required': 1},
        footprint_constraints={'area_sqm': 15.0, 'setup_cost': 30000.0}
    ))
    
    raw_line_data = RawLineData(
        line_id="SMT_FATP_LINE_01",
        station_raw_data=raw_stations,
        dut_raw_data={'DUT_pass': 900, 'DUT_fail': 100, 'target_yield': 0.9},
        operator_raw_data={'total_operators': 10, 'operator_cost_per_hour': 25.0},
        retest_policy="AAB",
        total_capacity=100
    )
    
    print(f"\n📥 Input (Raw Line Data):")
    print(f"  Line ID: {raw_line_data.line_id}")
    print(f"  Stations: {len(raw_line_data.station_raw_data)}")
    print(f"  Target Capacity: {raw_line_data.total_capacity} UPH")
    print(f"  Retest Policy: {raw_line_data.retest_policy}")
    
    # Process through engine
    print(f"\n⚙️ Processing through MOS Algo-Engine...")
    structured_data = line_engine.process(raw_line_data)
    
    print(f"\n📤 Output (Structured Line Data):")
    print(f"  Line ID: {structured_data.line_id}")
    print(f"  Line Cost: ${structured_data.line_cost:,.2f}")
    print(f"  Lead Time: {structured_data.line_lead_time} months")
    print(f"  Operators: {structured_data.line_operators}")
    print(f"  Line UPH: {structured_data.line_uph}")
    print(f"  Footprint: {structured_data.line_footprint} sqm²")
    print(f"  Efficiency: {structured_data.line_efficiency:.1%}")
    
    return structured_data


def demonstrate_standard_sockets():
    """Demonstrate Standard Data Sockets between layers."""
    print("\n" + "="*80)
    print("🔌 STANDARD DATA SOCKETS DEMONSTRATION")
    print("="*80)
    
    # Use global socket manager
    socket_manager = global_socket_manager
    
    print(f"Socket Manager Initialized with {len(socket_manager.sockets)} default sockets:")
    
    for socket_id, socket in socket_manager.sockets.items():
        print(f"  • {socket_id}: {socket.input_layer.value} → {socket.output_layer.value}")
    
    # Create sample raw data
    sample_data = create_sample_raw_data()
    
    print(f"\n📊 Sample Data Created:")
    print(f"  Components: {len(sample_data['components'])}")
    print(f"  Stations: {len(sample_data['stations'])}")
    print(f"  Lines: {len(sample_data['lines'])}")
    
    # Save sample data to file
    sample_file = "sample_raw_data.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"\n🔄 Processing through Standard Sockets:")
    
    # Process components
    print(f"\n1️⃣ Component Layer Processing:")
    structured_components = socket_manager.process_component_data(sample_data['components'])
    print(f"   Processed {len(structured_components)} components")
    for comp in structured_components[:2]:  # Show first 2
        print(f"   • {comp.component_id}: {comp.size}, ${comp.price:.3f}, {comp.lead_time}d")
    
    # Process stations
    print(f"\n2️⃣ Station Layer Processing:")
    structured_stations = []
    for station_data in sample_data['stations']:
        structured_station = socket_manager.process_station_data(station_data)
        structured_stations.append(structured_station)
        print(f"   • {structured_station.station_id}: ${structured_station.station_cost:,.0f}, {structured_station.discrete_event_profile.frequency:.0f} UPH")
    
    # Process lines
    print(f"\n3️⃣ Line Layer Processing:")
    for line_data in sample_data['lines']:
        structured_line = socket_manager.process_line_data(line_data)
        print(f"   • {structured_line.line_id}: ${structured_line.line_cost:,.0f}, {structured_line.line_uph} UPH, {structured_line.line_efficiency:.1%} efficiency")
    
    # Show socket statistics
    print(f"\n📈 Socket Statistics:")
    socket_info = socket_manager.get_all_socket_info()
    
    for socket_id, info in socket_info['sockets'].items():
        stats = info['processing_stats']
        print(f"  • {socket_id}:")
        print(f"    Transfers: {stats['total_processed']}")
        print(f"    Total Processing Time: {stats['processing_time']:.3f}s")
    
    return socket_manager


def demonstrate_scalability_benefits():
    """Demonstrate scalability and separation benefits."""
    print("\n" + "="*80)
    print("📈 SCALABILITY & SEPARATION BENEFITS DEMONSTRATION")
    print("="*80)
    
    print(f"✅ Standard Data Socket Benefits:")
    print(f"")
    print(f"1️⃣ SCALABILITY:")
    print(f"   • Users can work at any layer independently")
    print(f"   • Component vendors upload raw data → structured format")
    print(f"   • Station designers use structured components → station data")
    print(f"   • Line engineers use structured stations → line data")
    print(f"   • No need to purchase full MOS at all layers")
    print(f"")
    print(f"2️⃣ LESS COORDINATION:")
    print(f"   • Standardized data format between layers")
    print(f"   • Clear separation of responsibilities")
    print(f"   • Independent development cycles")
    print(f"")
    print(f"3️⃣ SYSTEM STABILITY:")
    print(f"   • Well-defined interfaces prevent breaking changes")
    print(f"   • Version-controlled data schemas")
    print(f"   • Backward compatibility support")
    print(f"")
    print(f"4️⃣ UI SEPARATION:")
    print(f"   • Each layer can have specialized interface")
    print(f"   • Component vendors: CAD/API/EE upload interface")
    print(f"   • Station designers: Station configuration interface")
    print(f"   • Line engineers: Line optimization interface")
    print(f"")
    print(f"5️⃣ MATHEMATICAL EVOLUTION:")
    print(f"   • MOS Algo-Engine can be improved independently")
    print(f"   • Advanced algorithms without breaking compatibility")
    print(f"   • Machine learning integration capability")
    print(f"   • Discrete event profiles enable simulation accuracy")
    
    # Demonstrate artificial vs real data capability
    print(f"\n🔄 Artificial vs Real Data Capability:")
    
    # Create artificial data
    artificial_component = {
        'component_id': 'ARTIFICIAL_R1',
        'component_type': 'Resistor',
        'cad_data': {'dimensions': {'package': '0805'}},
        'api_data': {'price_usd': 0.10, 'lead_time_days': 7},
        'ee_data': {'resistance_ohm': 2200, 'power_w': 0.125},
        'vendor_id': 'SIMULATION'
    }
    
    print(f"   📝 Artificial Data Example:")
    print(f"      Component: {artificial_component['component_id']}")
    print(f"      Price: ${artificial_component['api_data']['price_usd']}")
    print(f"      Lead Time: {artificial_component['api_data']['lead_time_days']} days")
    
    # Process artificial data
    socket_manager = global_socket_manager
    structured_artificial = socket_manager.process_component_data([artificial_component])
    
    print(f"   ⚙️ Processed Artificial Data:")
    comp = structured_artificial[0]
    print(f"      Size: {comp.size}")
    print(f"      Discrete Event: {comp.discrete_event_profile.event_name}")
    print(f"      Duration: {comp.discrete_event_profile.duration}s")
    
    print(f"\n   💡 This enables:")
    print(f"      • Simulation with artificial data for planning")
    print(f"      • Real data integration for production")
    print(f"      • Hybrid artificial/real scenarios for optimization")
    print(f"      • Competitive advantage through mathematical modeling")


if __name__ == "__main__":
    print("🔌 STANDARD DATA SOCKET ARCHITECTURE DEMONSTRATION")
    print("🎯 Implementing: Layer-based processing with MOS Algo-Engine")
    
    # Demonstrate individual layer engines
    demonstrate_component_layer()
    demonstrate_station_layer()
    demonstrate_line_layer()
    
    # Demonstrate standard sockets
    socket_manager = demonstrate_standard_sockets()
    
    # Demonstrate scalability benefits
    demonstrate_scalability_benefits()
    
    print("\n" + "="*80)
    print("🎉 STANDARD DATA SOCKET DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\n✅ Successfully demonstrated:")
    print("  • Component Layer Engine: Raw vendor data → Structured component data")
    print("  • Station Layer Engine: Component data → Structured station data")
    print("  • Line Layer Engine: Station data → Structured line data")
    print("  • Standard Data Sockets: Seamless layer communication")
    print("  • MOS Algo-Engine: Raw data processing with discrete event profiles")
    print("  • Scalability benefits: Independent layer operation")
    print("  • Mathematical evolution: Competitive advantage through algorithms")
    
    print(f"\n🎯 Architecture Benefits Achieved:")
    print(f"  ✓ Scalability: Users can work at any layer independently")
    print(f"  ✓ Less coordination: Standardized interfaces reduce complexity")  
    print(f"  ✓ System stability: Well-defined data contracts")
    print(f"  ✓ UI separation: Specialized interfaces per layer")
    print(f"  ✓ Mathematical evolution: Advanced algorithms without breaking changes")
    print(f"  ✓ Artificial/Real data: Simulation and production ready")
    
    print(f"\n🔄 Ready for manufacturing system integration!")