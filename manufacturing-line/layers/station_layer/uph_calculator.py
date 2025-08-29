"""Station UPH Calculator - Modular Units Per Hour Analysis for Manufacturing Stations."""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class BottleneckType(Enum):
    """Types of manufacturing bottlenecks."""
    CYCLE_TIME = "cycle_time"
    SETUP_TIME = "setup_time"
    CHANGEOVER = "changeover"
    EQUIPMENT_UTILIZATION = "equipment_utilization"
    OPERATOR_EFFICIENCY = "operator_efficiency"


@dataclass
class UPHAnalysis:
    """Detailed UPH analysis for a station."""
    theoretical_uph: float
    practical_uph: float
    efficiency_percent: float
    bottleneck_type: BottleneckType
    bottleneck_station: str
    improvement_potential_uph: float


class StationUPHCalculator:
    """Modular UPH calculator for manufacturing stations."""
    
    def __init__(self):
        """Initialize UPH calculator with efficiency models."""
        self.logger = logging.getLogger('StationUPHCalculator')
        
        # Efficiency factors by station type
        self.station_efficiencies = {
            "smt_placement": {
                "base_efficiency": 0.85,  # 85% baseline
                "setup_loss": 0.05,       # 5% for setup/changeover
                "maintenance_loss": 0.03, # 3% for preventive maintenance
                "quality_loss": 0.02      # 2% for quality issues
            },
            "test_station": {
                "base_efficiency": 0.90,
                "setup_loss": 0.03,
                "maintenance_loss": 0.02,
                "quality_loss": 0.01
            },
            "assembly": {
                "base_efficiency": 0.75,  # More operator dependent
                "setup_loss": 0.10,
                "maintenance_loss": 0.05,
                "quality_loss": 0.05
            },
            "inspection": {
                "base_efficiency": 0.88,
                "setup_loss": 0.02,
                "maintenance_loss": 0.03,
                "quality_loss": 0.02
            },
            "packaging": {
                "base_efficiency": 0.92,  # Typically high efficiency
                "setup_loss": 0.02,
                "maintenance_loss": 0.02,
                "quality_loss": 0.01
            }
        }
        
        # Cycle time models for different operations
        self.operation_cycle_times = {
            # SMT Operations (seconds)
            "smt_place_passive": 0.5,
            "smt_place_ic": 2.1,
            "smt_place_inductor": 0.8,
            "smt_print_paste": 15.0,   # Per board
            "smt_reflow": 240.0,       # Per board
            
            # Test Operations (seconds)
            "basic_electrical_test": 5.0,
            "functional_test": 30.0,
            "burn_in_test": 3600.0,    # 1 hour
            "calibration_test": 300.0, # 5 minutes
            
            # Assembly Operations (seconds)
            "manual_assembly": 60.0,
            "screw_fastening": 10.0,
            "cable_connection": 15.0,
            "housing_assembly": 45.0,
            
            # Inspection Operations (seconds)
            "visual_inspection": 30.0,
            "dimensional_check": 120.0,
            "aoi_inspection": 10.0,
            "x_ray_inspection": 60.0
        }
        
        self.logger.info("StationUPHCalculator initialized with efficiency models")
    
    def calculate_station_uph(self, station_config: Dict[str, Any], 
                             component_requirements: Dict[str, Any] = None) -> UPHAnalysis:
        """Calculate comprehensive UPH analysis for a station."""
        try:
            station_type = station_config.get('station_type', 'smt_placement')
            cycle_time_s = station_config.get('cycle_time_s', 10.0)
            setup_time_s = station_config.get('setup_time_s', 300.0)
            batch_size = station_config.get('batch_size', 100)
            
            # Calculate theoretical UPH (perfect conditions)
            theoretical_uph = 3600.0 / cycle_time_s if cycle_time_s > 0 else 0
            
            # Apply efficiency factors
            efficiency_factors = self._calculate_efficiency_factors(station_type, setup_time_s, batch_size)
            practical_uph = theoretical_uph * efficiency_factors['total_efficiency']
            
            # Determine bottleneck
            bottleneck_type, bottleneck_impact = self._identify_bottleneck(
                station_config, efficiency_factors
            )
            
            # Calculate improvement potential
            improvement_potential = theoretical_uph - practical_uph
            
            return UPHAnalysis(
                theoretical_uph=theoretical_uph,
                practical_uph=practical_uph,
                efficiency_percent=efficiency_factors['total_efficiency'] * 100,
                bottleneck_type=bottleneck_type,
                bottleneck_station=station_config.get('station_id', 'unknown'),
                improvement_potential_uph=improvement_potential
            )
            
        except Exception as e:
            self.logger.error(f"UPH calculation failed: {e}")
            return UPHAnalysis(0, 0, 0, BottleneckType.CYCLE_TIME, "error", 0)
    
    def _calculate_efficiency_factors(self, station_type: str, setup_time_s: float, 
                                    batch_size: int) -> Dict[str, float]:
        """Calculate detailed efficiency factors."""
        # Get base efficiency model
        efficiency_model = self.station_efficiencies.get(
            station_type, self.station_efficiencies["smt_placement"]
        )
        
        base_efficiency = efficiency_model["base_efficiency"]
        
        # Setup time efficiency (varies with batch size)
        # Larger batches = higher efficiency due to setup amortization
        setup_efficiency = 1.0 - (setup_time_s / (batch_size * 60))  # Assume 60s average cycle
        setup_efficiency = max(0.7, min(1.0, setup_efficiency))  # Clamp between 70-100%
        
        # Other efficiency factors
        maintenance_efficiency = 1.0 - efficiency_model["maintenance_loss"]
        quality_efficiency = 1.0 - efficiency_model["quality_loss"]
        
        # Overall equipment effectiveness (OEE) calculation
        total_efficiency = (
            base_efficiency * 
            setup_efficiency * 
            maintenance_efficiency * 
            quality_efficiency
        )
        
        return {
            "base_efficiency": base_efficiency,
            "setup_efficiency": setup_efficiency,
            "maintenance_efficiency": maintenance_efficiency,
            "quality_efficiency": quality_efficiency,
            "total_efficiency": total_efficiency
        }
    
    def _identify_bottleneck(self, station_config: Dict[str, Any], 
                           efficiency_factors: Dict[str, float]) -> Tuple[BottleneckType, float]:
        """Identify the primary bottleneck limiting UPH."""
        bottlenecks = []
        
        # Cycle time bottleneck
        cycle_time_s = station_config.get('cycle_time_s', 10.0)
        if cycle_time_s > 30.0:  # Arbitrary threshold
            bottlenecks.append((BottleneckType.CYCLE_TIME, 1.0 - (30.0 / cycle_time_s)))
        
        # Setup time bottleneck
        setup_efficiency = efficiency_factors.get('setup_efficiency', 1.0)
        if setup_efficiency < 0.9:
            bottlenecks.append((BottleneckType.SETUP_TIME, 1.0 - setup_efficiency))
        
        # Equipment utilization bottleneck
        base_efficiency = efficiency_factors.get('base_efficiency', 1.0)
        if base_efficiency < 0.8:
            bottlenecks.append((BottleneckType.EQUIPMENT_UTILIZATION, 1.0 - base_efficiency))
        
        # Quality/maintenance bottleneck
        quality_eff = efficiency_factors.get('quality_efficiency', 1.0)
        maintenance_eff = efficiency_factors.get('maintenance_efficiency', 1.0)
        if quality_eff < 0.95 or maintenance_eff < 0.95:
            combined_loss = 1.0 - (quality_eff * maintenance_eff)
            bottlenecks.append((BottleneckType.OPERATOR_EFFICIENCY, combined_loss))
        
        # Return the most significant bottleneck
        if bottlenecks:
            return max(bottlenecks, key=lambda x: x[1])
        else:
            return BottleneckType.CYCLE_TIME, 0.0
    
    def calculate_line_uph(self, station_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall line UPH considering all stations."""
        if not station_configs:
            return {
                'line_uph': 0.0,
                'bottleneck_station': None,
                'station_analyses': [],
                'efficiency_balance': 0.0
            }
        
        station_analyses = []
        min_uph = float('inf')
        bottleneck_station = None
        
        # Analyze each station
        for config in station_configs:
            analysis = self.calculate_station_uph(config)
            station_analyses.append({
                'station_id': config.get('station_id', 'unknown'),
                'analysis': analysis
            })
            
            # Find bottleneck (station with lowest UPH)
            if analysis.practical_uph < min_uph:
                min_uph = analysis.practical_uph
                bottleneck_station = config.get('station_id', 'unknown')
        
        # Calculate efficiency balance (how well-balanced the line is)
        if station_analyses:
            uphs = [sa['analysis'].practical_uph for sa in station_analyses]
            avg_uph = sum(uphs) / len(uphs)
            efficiency_balance = min_uph / avg_uph if avg_uph > 0 else 0
        else:
            efficiency_balance = 0.0
        
        return {
            'line_uph': min_uph if min_uph != float('inf') else 0.0,
            'bottleneck_station': bottleneck_station,
            'station_analyses': station_analyses,
            'efficiency_balance': efficiency_balance,
            'balance_rating': self._rate_line_balance(efficiency_balance)
        }
    
    def optimize_uph(self, station_config: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest UPH optimization strategies for a station."""
        current_analysis = self.calculate_station_uph(station_config)
        
        optimizations = {
            'current_uph': current_analysis.practical_uph,
            'theoretical_max_uph': current_analysis.theoretical_uph,
            'optimization_suggestions': [],
            'potential_uph_gain': 0.0
        }
        
        # Cycle time optimization
        if current_analysis.bottleneck_type == BottleneckType.CYCLE_TIME:
            optimizations['optimization_suggestions'].append({
                'category': 'cycle_time',
                'suggestion': 'Optimize process steps and reduce non-value-added time',
                'potential_uph_gain': current_analysis.theoretical_uph * 0.15,
                'implementation_difficulty': 'medium'
            })
        
        # Setup time optimization
        if current_analysis.bottleneck_type == BottleneckType.SETUP_TIME:
            optimizations['optimization_suggestions'].append({
                'category': 'setup_time',
                'suggestion': 'Implement SMED (Single Minute Exchange of Die) techniques',
                'potential_uph_gain': current_analysis.theoretical_uph * 0.10,
                'implementation_difficulty': 'low'
            })
        
        # Equipment utilization optimization
        if current_analysis.bottleneck_type == BottleneckType.EQUIPMENT_UTILIZATION:
            optimizations['optimization_suggestions'].append({
                'category': 'equipment',
                'suggestion': 'Implement predictive maintenance and reduce unplanned downtime',
                'potential_uph_gain': current_analysis.theoretical_uph * 0.08,
                'implementation_difficulty': 'high'
            })
        
        # Operator efficiency optimization
        if current_analysis.bottleneck_type == BottleneckType.OPERATOR_EFFICIENCY:
            optimizations['optimization_suggestions'].append({
                'category': 'operator',
                'suggestion': 'Implement training programs and ergonomic improvements',
                'potential_uph_gain': current_analysis.theoretical_uph * 0.12,
                'implementation_difficulty': 'medium'
            })
        
        # Calculate total potential gain
        optimizations['potential_uph_gain'] = sum(
            suggestion['potential_uph_gain'] for suggestion in optimizations['optimization_suggestions']
        )
        
        # Cap at theoretical maximum
        max_practical_gain = current_analysis.theoretical_uph - current_analysis.practical_uph
        optimizations['potential_uph_gain'] = min(
            optimizations['potential_uph_gain'], max_practical_gain
        )
        
        return optimizations
    
    def calculate_takt_time(self, customer_demand_per_day: int, 
                          available_production_time_hours: float = 16.0) -> float:
        """Calculate takt time based on customer demand."""
        if customer_demand_per_day <= 0:
            return 0.0
        
        available_time_seconds = available_production_time_hours * 3600
        takt_time_seconds = available_time_seconds / customer_demand_per_day
        
        return takt_time_seconds
    
    def validate_takt_time_compliance(self, station_configs: List[Dict[str, Any]], 
                                    takt_time_s: float) -> Dict[str, Any]:
        """Validate if stations can meet takt time requirements."""
        line_analysis = self.calculate_line_uph(station_configs)
        line_cycle_time_s = 3600.0 / line_analysis['line_uph'] if line_analysis['line_uph'] > 0 else 0
        
        compliance = {
            'takt_time_s': takt_time_s,
            'actual_cycle_time_s': line_cycle_time_s,
            'compliant': line_cycle_time_s <= takt_time_s,
            'margin_s': takt_time_s - line_cycle_time_s,
            'margin_percent': ((takt_time_s - line_cycle_time_s) / takt_time_s * 100) if takt_time_s > 0 else 0,
            'bottleneck_station': line_analysis['bottleneck_station']
        }
        
        return compliance
    
    def _rate_line_balance(self, efficiency_balance: float) -> str:
        """Rate how well-balanced the production line is."""
        if efficiency_balance >= 0.95:
            return "excellent"
        elif efficiency_balance >= 0.85:
            return "good"
        elif efficiency_balance >= 0.70:
            return "fair"
        else:
            return "poor"
    
    def calculate_capacity_utilization(self, actual_uph: float, 
                                     theoretical_max_uph: float) -> Dict[str, Any]:
        """Calculate capacity utilization metrics."""
        if theoretical_max_uph <= 0:
            return {
                'utilization_percent': 0.0,
                'spare_capacity_uph': 0.0,
                'capacity_rating': 'unknown'
            }
        
        utilization_percent = (actual_uph / theoretical_max_uph) * 100
        spare_capacity = theoretical_max_uph - actual_uph
        
        if utilization_percent >= 85:
            rating = "high"
        elif utilization_percent >= 70:
            rating = "medium" 
        elif utilization_percent >= 50:
            rating = "low"
        else:
            rating = "very_low"
        
        return {
            'utilization_percent': utilization_percent,
            'spare_capacity_uph': spare_capacity,
            'capacity_rating': rating
        }