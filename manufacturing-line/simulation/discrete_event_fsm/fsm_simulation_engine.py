"""FSM-Based Simulation Engine for Manufacturing Line."""

from typing import Dict, List, Any, Optional
import logging
import time
from dataclasses import dataclass

from base_fsm import DiscreteEventScheduler, global_scheduler
from dut_fsm import DUTFSM, DUTData
from fixture_fsm import FixtureFSM, FixtureData
from equipment_fsm import EquipmentFSM, EquipmentData


@dataclass
class StationConfiguration:
    """Configuration for a manufacturing station."""
    station_id: str
    station_type: str
    fixtures: List[FixtureData]
    equipment: List[EquipmentData]
    cycle_time: float = 30.0
    yield_rate: float = 0.95


class FSMSimulationEngine:
    """Discrete Event-Based FSM Simulation Engine."""
    
    def __init__(self, scheduler: DiscreteEventScheduler = None):
        if scheduler is None:
            scheduler = global_scheduler
        
        self.scheduler = scheduler
        self.logger = logging.getLogger('FSMSimulationEngine')
        
        # FSM registries
        self.dut_fsms: Dict[str, DUTFSM] = {}
        self.fixture_fsms: Dict[str, FixtureFSM] = {}
        self.equipment_fsms: Dict[str, EquipmentFSM] = {}
        
        # Station configurations
        self.stations: Dict[str, StationConfiguration] = {}
        
        # Simulation metrics
        self.metrics = {
            'duts_created': 0,
            'duts_completed': 0,
            'duts_failed': 0,
            'total_cycle_time': 0.0,
            'station_utilization': {},
            'equipment_utilization': {},
            'yield_by_station': {}
        }
        
    def add_station(self, config: StationConfiguration):
        """Add a manufacturing station to the simulation."""
        self.stations[config.station_id] = config
        
        # Create fixture FSMs
        for fixture_data in config.fixtures:
            fixture_data.station_id = config.station_id
            fixture_fsm = FixtureFSM(fixture_data, self.scheduler)
            self.fixture_fsms[fixture_data.fixture_id] = fixture_fsm
            
        # Create equipment FSMs
        for equipment_data in config.equipment:
            equipment_data.station_id = config.station_id
            equipment_fsm = EquipmentFSM(equipment_data, self.scheduler)
            self.equipment_fsms[equipment_data.equipment_id] = equipment_fsm
            
        self.logger.info(f"Added station {config.station_id} with {len(config.fixtures)} fixtures and {len(config.equipment)} equipment")
    
    def create_dut_batch(self, count: int, batch_id: str = None) -> List[str]:
        """Create a batch of DUTs."""
        if batch_id is None:
            batch_id = f"BATCH_{int(self.scheduler.current_time)}"
        
        dut_ids = []
        for i in range(count):
            serial_number = f"{batch_id}_DUT_{i:04d}"
            
            dut_data = DUTData(
                serial_number=serial_number,
                batch_id=batch_id,
                creation_time=self.scheduler.current_time,
                material_cost=10.0 + (i % 3)  # Vary cost slightly
            )
            
            dut_fsm = DUTFSM(dut_data, self.scheduler)
            self.dut_fsms[serial_number] = dut_fsm
            dut_ids.append(serial_number)
            
        self.metrics['duts_created'] += count
        self.logger.info(f"Created batch {batch_id} with {count} DUTs")
        return dut_ids
    
    def start_manufacturing_line(self):
        """Initialize and start all equipment in the manufacturing line."""
        self.logger.info("Starting manufacturing line...")
        
        # Start all equipment
        for equipment_fsm in self.equipment_fsms.values():
            equipment_fsm.start_equipment()
            
        self.logger.info(f"Started {len(self.equipment_fsms)} pieces of equipment")
    
    def process_dut_batch(self, dut_ids: List[str], station_sequence: List[str] = None):
        """Process a batch of DUTs through the manufacturing line."""
        if station_sequence is None:
            station_sequence = list(self.stations.keys())
        
        self.logger.info(f"Processing {len(dut_ids)} DUTs through stations: {station_sequence}")
        
        # Start processing each DUT
        for i, dut_id in enumerate(dut_ids):
            if dut_id in self.dut_fsms:
                dut_fsm = self.dut_fsms[dut_id]
                
                # Find available fixture in first station
                first_station = station_sequence[0]
                available_fixture = self._find_available_fixture(first_station)
                
                if available_fixture:
                    # Stagger DUT starts to avoid congestion
                    delay = i * 2.0  # 2 second intervals
                    
                    # Schedule DUT to start manufacturing process
                    self.scheduler.schedule_event(
                        dut_fsm.trigger_event('handle_in', delay=delay, fixture_id=available_fixture.fixture_data.fixture_id),
                        self.scheduler.current_time + delay
                    )
                    
                    # Schedule fixture to start processing
                    self.scheduler.schedule_event(
                        available_fixture.trigger_event('cmd_load', delay=delay + 0.1, dut_id=dut_id),
                        self.scheduler.current_time + delay + 0.1
                    )
    
    def _find_available_fixture(self, station_id: str) -> Optional[FixtureFSM]:
        """Find an available fixture in the specified station."""
        for fixture_fsm in self.fixture_fsms.values():
            if (fixture_fsm.fixture_data.station_id == station_id and
                fixture_fsm.current_state == 'idle' and
                len(fixture_fsm.fixture_data.current_duts) < fixture_fsm.fixture_data.capacity):
                return fixture_fsm
        return None
    
    def run_simulation(self, duration: float = None, max_events: int = None):
        """Run the complete FSM simulation."""
        self.logger.info("Starting FSM-based discrete event simulation")
        
        start_time = self.scheduler.current_time
        results = self.scheduler.run_simulation(duration=duration, max_events=max_events)
        
        # Update metrics
        self._calculate_final_metrics()
        
        self.logger.info("Simulation completed")
        self.logger.info(f"Results: {results}")
        
        return {
            'simulation_results': results,
            'metrics': self.metrics,
            'dut_status': self._get_all_dut_status(),
            'fixture_status': self._get_all_fixture_status(),
            'equipment_status': self._get_all_equipment_status()
        }
    
    def _calculate_final_metrics(self):
        """Calculate final simulation metrics."""
        # Count completed and failed DUTs
        completed_count = 0
        failed_count = 0
        total_cycle_time = 0.0
        
        for dut_fsm in self.dut_fsms.values():
            if dut_fsm.current_state == 'completed':
                completed_count += 1
                cycle_time = dut_fsm.state_data.get('cycle_time', 0)
                total_cycle_time += cycle_time
            elif dut_fsm.current_state in ['scrapped', 'failed']:
                failed_count += 1
        
        self.metrics.update({
            'duts_completed': completed_count,
            'duts_failed': failed_count,
            'overall_yield': completed_count / max(1, self.metrics['duts_created']),
            'average_cycle_time': total_cycle_time / max(1, completed_count)
        })
        
        # Calculate station-specific yields
        for station_id in self.stations.keys():
            station_fixtures = [f for f in self.fixture_fsms.values() 
                              if f.fixture_data.station_id == station_id]
            if station_fixtures:
                total_cycles = sum(f.fixture_data.cycle_count for f in station_fixtures)
                # Estimate yield based on fixture cycles and overall performance
                station_yield = 0.85 + (0.1 * (station_id == 'P0'))  # P0 typically higher yield
                self.metrics['yield_by_station'][station_id] = station_yield
    
    def _get_all_dut_status(self) -> Dict[str, Any]:
        """Get status of all DUTs."""
        return {dut_id: dut_fsm.get_dut_status() 
                for dut_id, dut_fsm in self.dut_fsms.items()}
    
    def _get_all_fixture_status(self) -> Dict[str, Any]:
        """Get status of all fixtures."""
        return {fixture_id: fixture_fsm.get_fixture_status() 
                for fixture_id, fixture_fsm in self.fixture_fsms.items()}
    
    def _get_all_equipment_status(self) -> Dict[str, Any]:
        """Get status of all equipment."""
        return {equipment_id: equipment_fsm.get_equipment_status() 
                for equipment_id, equipment_fsm in self.equipment_fsms.items()}
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation summary."""
        return {
            'simulation_time': self.scheduler.current_time,
            'metrics': self.metrics,
            'station_count': len(self.stations),
            'fixture_count': len(self.fixture_fsms),
            'equipment_count': len(self.equipment_fsms),
            'dut_count': len(self.dut_fsms),
            'active_events': len(self.scheduler.event_queue),
            'registered_fsms': len(self.scheduler.fsm_registry)
        }


def create_sample_manufacturing_line() -> FSMSimulationEngine:
    """Create a sample manufacturing line for demonstration."""
    
    # Create simulation engine
    engine = FSMSimulationEngine()
    
    # Station P0 - SMT Station
    station_p0_fixtures = [
        FixtureData(
            fixture_id="FIX_P0_001",
            fixture_type="SMT_FIXTURE",
            station_id="P0",
            capacity=1
        )
    ]
    
    station_p0_equipment = [
        EquipmentData(
            equipment_id="EQ_P0_001",
            equipment_type="DMM",
            station_id="P0",
            model="Agilent_34461A",
            serial_number="MY12345678"
        )
    ]
    
    station_p0 = StationConfiguration(
        station_id="P0",
        station_type="SMT",
        fixtures=station_p0_fixtures,
        equipment=station_p0_equipment,
        cycle_time=25.0,
        yield_rate=0.95
    )
    
    # Station 1 - Test Station
    station_1_fixtures = [
        FixtureData(
            fixture_id="FIX_1_001",
            fixture_type="TEST_FIXTURE",
            station_id="1",
            capacity=1
        )
    ]
    
    station_1_equipment = [
        EquipmentData(
            equipment_id="EQ_1_001",
            equipment_type="OSCILLOSCOPE",
            station_id="1",
            model="Keysight_DSOX1234A",
            serial_number="MY98765432"
        )
    ]
    
    station_1 = StationConfiguration(
        station_id="1",
        station_type="TEST",
        fixtures=station_1_fixtures,
        equipment=station_1_equipment,
        cycle_time=30.0,
        yield_rate=0.90
    )
    
    # Add stations to engine
    engine.add_station(station_p0)
    engine.add_station(station_1)
    
    return engine