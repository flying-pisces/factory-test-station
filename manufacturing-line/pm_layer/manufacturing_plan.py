"""Product Management Layer - Manufacturing Plan Optimization and DUT Flow Simulation."""

import uuid
import time
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from pathlib import Path


class StationType(Enum):
    SMT = "smt"
    FATP = "fatp"
    ASSEMBLY = "assembly"
    TEST = "test"
    QUALITY = "quality"


class DUTStatus(Enum):
    GENERATED = "generated"
    IN_TRANSIT = "in_transit"
    AT_STATION = "at_station"
    PROCESSING = "processing"
    PASSED = "passed"
    FAILED = "failed"
    REWORK = "rework"
    SCRAPPED = "scrapped"
    COMPLETED = "completed"


@dataclass
class DUT:
    """Device Under Test in the manufacturing line."""
    dut_id: str
    batch_id: str
    generation_time: float
    current_station: Optional[str] = None
    status: DUTStatus = DUTStatus.GENERATED
    process_history: List[Dict[str, Any]] = field(default_factory=list)
    defect_sources: List[str] = field(default_factory=list)
    material_cost: float = 0.0
    process_cost: float = 0.0
    yield_loss_reason: Optional[str] = None
    
    def add_process_record(self, station_id: str, result: str, duration: float, cost: float = 0.0):
        """Add processing record to DUT history."""
        self.process_history.append({
            'station_id': station_id,
            'result': result,
            'duration': duration,
            'cost': cost,
            'timestamp': time.time()
        })
        self.process_cost += cost


@dataclass
class StationConfig:
    """Configuration for a manufacturing station."""
    station_id: str
    station_type: StationType
    position: int  # Position in line (0, 1, 2, ...)
    yield_rate: float  # Passing ratio (0.0 to 1.0)
    process_time: float  # Average processing time in seconds
    process_cost: float  # Cost per DUT processed
    capacity: int  # Parallel processing capacity
    defect_sources: List[str] = field(default_factory=list)
    rework_capability: bool = False
    
    # Yield loss breakdown
    material_defect_rate: float = 0.02  # Incoming material defects
    test_induced_loss_rate: float = 0.03  # Test-caused failures
    process_variation_rate: float = 0.01  # Process variation losses


@dataclass 
class ManufacturingPlan:
    """Complete manufacturing plan with station sequence and parameters."""
    plan_id: str
    plan_name: str
    stations: List[StationConfig]
    target_volume: int  # Target DUT count to produce
    target_yield: float  # Overall line yield target
    target_mva: float  # Manufacturing Value Added target (¥)
    
    # Plan metrics (calculated)
    actual_yield: Optional[float] = None
    actual_mva: Optional[float] = None
    total_cost: Optional[float] = None
    cycle_time: Optional[float] = None
    throughput: Optional[float] = None  # DUTs per hour


class LineSimulation:
    """Simulate DUT flow through manufacturing line."""
    
    def __init__(self, plan: ManufacturingPlan):
        self.plan = plan
        self.logger = logging.getLogger(f'LineSimulation_{plan.plan_id}')
        
        # Simulation state
        self.duts: Dict[str, DUT] = {}
        self.current_batch_id = str(uuid.uuid4())[:8]
        self.simulation_time = 0.0
        self.completed_duts: List[DUT] = []
        self.failed_duts: List[DUT] = []
        
        # Performance tracking
        self.station_metrics: Dict[str, Dict[str, Any]] = {}
        self.line_metrics = {
            'total_generated': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_scrapped': 0,
            'yield_by_station': {},
            'cost_by_station': {},
            'bottleneck_station': None,
            'average_cycle_time': 0.0
        }
        
        self._initialize_station_metrics()
    
    def _initialize_station_metrics(self):
        """Initialize metrics tracking for each station."""
        for station in self.plan.stations:
            self.station_metrics[station.station_id] = {
                'processed_count': 0,
                'passed_count': 0,
                'failed_count': 0,
                'total_process_time': 0.0,
                'total_cost': 0.0,
                'current_queue': 0,
                'utilization': 0.0,
                'yield_rate': station.yield_rate
            }
    
    def generate_dut_batch(self, count: int) -> List[str]:
        """Generate a batch of DUTs."""
        dut_ids = []
        
        for i in range(count):
            dut_id = f"DUT_{self.current_batch_id}_{i:06d}"
            
            # Simulate material cost variation
            base_material_cost = 10.0  # Base cost in ¥
            material_cost = base_material_cost * random.uniform(0.9, 1.1)
            
            dut = DUT(
                dut_id=dut_id,
                batch_id=self.current_batch_id,
                generation_time=self.simulation_time,
                material_cost=material_cost
            )
            
            # Simulate incoming material defects
            if random.random() < 0.02:  # 2% incoming defects
                dut.defect_sources.append('incoming_material')
            
            self.duts[dut_id] = dut
            dut_ids.append(dut_id)
        
        self.line_metrics['total_generated'] += count
        self.logger.info(f"Generated {count} DUTs in batch {self.current_batch_id}")
        
        return dut_ids
    
    def simulate_station_processing(self, station: StationConfig, dut: DUT) -> Tuple[bool, str]:
        """Simulate processing at a specific station."""
        # Update DUT status
        dut.current_station = station.station_id
        dut.status = DUTStatus.PROCESSING
        
        # Simulate processing time
        process_time = station.process_time * random.uniform(0.8, 1.2)  # ±20% variation
        self.simulation_time += process_time
        
        # Determine if DUT passes based on various failure modes
        pass_probability = station.yield_rate
        
        # Adjust for existing defects
        if 'incoming_material' in dut.defect_sources:
            pass_probability *= 0.5  # 50% chance if material defect
        
        # Simulate different failure modes
        failure_reason = None
        passes = True
        
        if random.random() > pass_probability:
            passes = False
            
            # Determine failure reason
            failure_modes = [
                ('material_defect', station.material_defect_rate),
                ('test_induced', station.test_induced_loss_rate),
                ('process_variation', station.process_variation_rate)
            ]
            
            rand_val = random.random()
            cumulative = 0.0
            
            for mode, rate in failure_modes:
                cumulative += rate / (1 - station.yield_rate)  # Normalize
                if rand_val <= cumulative:
                    failure_reason = mode
                    break
            
            if failure_reason:
                dut.defect_sources.append(failure_reason)
                dut.yield_loss_reason = failure_reason
        
        # Record processing
        result = 'pass' if passes else 'fail'
        dut.add_process_record(station.station_id, result, process_time, station.process_cost)
        
        # Update station metrics
        metrics = self.station_metrics[station.station_id]
        metrics['processed_count'] += 1
        metrics['total_process_time'] += process_time
        metrics['total_cost'] += station.process_cost
        
        if passes:
            metrics['passed_count'] += 1
            dut.status = DUTStatus.PASSED
        else:
            metrics['failed_count'] += 1
            dut.status = DUTStatus.FAILED
        
        # Update actual yield rate
        metrics['yield_rate'] = metrics['passed_count'] / metrics['processed_count']
        
        return passes, failure_reason or 'pass'
    
    def simulate_line_flow(self, dut_ids: List[str]) -> Dict[str, Any]:
        """Simulate DUT flow through entire manufacturing line."""
        results = {
            'completed': [],
            'failed': [],
            'scrapped': [],
            'line_yield': 0.0,
            'total_cost': 0.0,
            'cycle_time': 0.0,
            'bottlenecks': []
        }
        
        for dut_id in dut_ids:
            dut = self.duts[dut_id]
            start_time = self.simulation_time
            
            # Process through each station in sequence
            for station in self.plan.stations:
                passed, reason = self.simulate_station_processing(station, dut)
                
                if not passed:
                    # Handle failure
                    if station.rework_capability and random.random() < 0.3:  # 30% rework chance
                        # Attempt rework
                        rework_passed, _ = self.simulate_station_processing(station, dut)
                        if not rework_passed:
                            dut.status = DUTStatus.SCRAPPED
                            results['scrapped'].append(dut_id)
                            self.failed_duts.append(dut)
                            break
                    else:
                        dut.status = DUTStatus.SCRAPPED
                        results['scrapped'].append(dut_id)
                        self.failed_duts.append(dut)
                        break
            
            # If DUT made it through all stations
            if dut.status == DUTStatus.PASSED:
                dut.status = DUTStatus.COMPLETED
                results['completed'].append(dut_id)
                self.completed_duts.append(dut)
            
            # Calculate cycle time for this DUT
            cycle_time = self.simulation_time - start_time
            dut.process_history.append({
                'event': 'cycle_complete',
                'cycle_time': cycle_time,
                'total_cost': dut.material_cost + dut.process_cost
            })
        
        # Calculate line-level metrics
        total_duts = len(dut_ids)
        completed_count = len(results['completed'])
        
        results['line_yield'] = completed_count / total_duts if total_duts > 0 else 0
        results['total_cost'] = sum(
            dut.material_cost + dut.process_cost 
            for dut in [self.duts[did] for did in dut_ids]
        )
        results['cycle_time'] = (
            sum(sum(record.get('duration', 0) for record in self.duts[did].process_history 
                   if 'duration' in record) for did in results['completed']) / 
            max(1, completed_count)
        )
        
        # Identify bottlenecks
        station_utilizations = {}
        for station in self.plan.stations:
            metrics = self.station_metrics[station.station_id]
            utilization = metrics['total_process_time'] / (self.simulation_time * station.capacity)
            station_utilizations[station.station_id] = utilization
            metrics['utilization'] = utilization
        
        # Find bottleneck (highest utilization)
        if station_utilizations:
            bottleneck = max(station_utilizations.items(), key=lambda x: x[1])
            results['bottlenecks'] = [bottleneck[0]]
            self.line_metrics['bottleneck_station'] = bottleneck[0]
        
        return results
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete line simulation with target volume."""
        self.logger.info(f"Starting simulation for plan {self.plan.plan_id}")
        
        # Generate DUTs
        dut_ids = self.generate_dut_batch(self.plan.target_volume)
        
        # Simulate line flow
        results = self.simulate_line_flow(dut_ids)
        
        # Update plan metrics
        self.plan.actual_yield = results['line_yield']
        self.plan.total_cost = results['total_cost']
        self.plan.cycle_time = results['cycle_time']
        
        # Calculate MVA (Manufacturing Value Added)
        if results['completed']:
            completed_duts = [self.duts[did] for did in results['completed']]
            total_material_cost = sum(dut.material_cost for dut in completed_duts)
            total_process_cost = sum(dut.process_cost for dut in completed_duts)
            
            # MVA = Value added through manufacturing process
            # Simplified: (selling price - material cost) * yield - process cost
            assumed_selling_price = 100.0  # ¥ per unit
            self.plan.actual_mva = (
                (assumed_selling_price * len(results['completed'])) -
                total_material_cost - total_process_cost
            ) / len(results['completed'])
        
        # Calculate throughput (DUTs/hour)
        if self.simulation_time > 0:
            self.plan.throughput = len(results['completed']) / (self.simulation_time / 3600)
        
        # Update line metrics
        self.line_metrics.update({
            'total_completed': len(results['completed']),
            'total_failed': len(results['scrapped']),
            'average_cycle_time': results['cycle_time'],
            'yield_by_station': {
                sid: metrics['yield_rate'] 
                for sid, metrics in self.station_metrics.items()
            },
            'cost_by_station': {
                sid: metrics['total_cost'] 
                for sid, metrics in self.station_metrics.items()
            }
        })
        
        self.logger.info(
            f"Simulation complete: {self.plan.actual_yield:.1%} yield, "
            f"{self.plan.actual_mva:.1f}¥ MVA, {self.plan.throughput:.0f} UPH"
        )
        
        return {
            'plan_metrics': {
                'plan_id': self.plan.plan_id,
                'actual_yield': self.plan.actual_yield,
                'actual_mva': self.plan.actual_mva,
                'total_cost': self.plan.total_cost,
                'cycle_time': self.plan.cycle_time,
                'throughput': self.plan.throughput
            },
            'line_metrics': self.line_metrics,
            'station_metrics': self.station_metrics,
            'dut_results': results
        }


def create_sample_manufacturing_plans() -> List[ManufacturingPlan]:
    """Create sample manufacturing plans for comparison."""
    
    # Manufacturing Plan 1: High Yield, Lower MVA
    plan1_stations = [
        StationConfig(
            station_id="P0", station_type=StationType.SMT, position=0,
            yield_rate=0.95, process_time=30, process_cost=5.0, capacity=1
        ),
        StationConfig(
            station_id="1", station_type=StationType.TEST, position=1,
            yield_rate=0.98, process_time=25, process_cost=3.0, capacity=1
        ),
        StationConfig(
            station_id="2", station_type=StationType.TEST, position=2,
            yield_rate=0.97, process_time=20, process_cost=2.0, capacity=1
        ),
        StationConfig(
            station_id="3", station_type=StationType.ASSEMBLY, position=3,
            yield_rate=0.99, process_time=35, process_cost=4.0, capacity=1
        ),
        StationConfig(
            station_id="4", station_type=StationType.QUALITY, position=4,
            yield_rate=0.985, process_time=15, process_cost=1.0, capacity=1
        )
    ]
    
    plan1 = ManufacturingPlan(
        plan_id="PLAN_001",
        plan_name="Manufacturing Plan 1",
        stations=plan1_stations,
        target_volume=10000,
        target_yield=0.90,
        target_mva=30.0
    )
    
    # Manufacturing Plan 2: Lower Yield, Higher MVA
    plan2_stations = [
        StationConfig(
            station_id="P0", station_type=StationType.SMT, position=0,
            yield_rate=0.85, process_time=45, process_cost=8.0, capacity=1
        ),
        StationConfig(
            station_id="1", station_type=StationType.TEST, position=1,
            yield_rate=0.80, process_time=40, process_cost=12.0, capacity=1
        ),
        StationConfig(
            station_id="2", station_type=StationType.TEST, position=2,
            yield_rate=0.90, process_time=35, process_cost=15.0, capacity=1
        ),
        StationConfig(
            station_id="3", station_type=StationType.ASSEMBLY, position=3,
            yield_rate=0.95, process_time=60, process_cost=20.0, capacity=1
        ),
        StationConfig(
            station_id="4", station_type=StationType.QUALITY, position=4,
            yield_rate=0.88, process_time=30, process_cost=10.0, capacity=1
        )
    ]
    
    plan2 = ManufacturingPlan(
        plan_id="PLAN_002", 
        plan_name="Manufacturing Plan 2",
        stations=plan2_stations,
        target_volume=10000,
        target_yield=0.45,
        target_mva=60.0
    )
    
    return [plan1, plan2]