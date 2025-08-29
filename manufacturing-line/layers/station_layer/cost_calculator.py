"""Station Cost Calculator - Modular Cost Analysis for Manufacturing Stations."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class CostCategory(Enum):
    """Categories of manufacturing costs."""
    EQUIPMENT = "equipment"
    LABOR = "labor"
    FACILITIES = "facilities"
    MATERIALS = "materials"
    OVERHEAD = "overhead"


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a station."""
    equipment_cost_usd: float
    labor_cost_usd: float
    facilities_cost_usd: float
    materials_cost_usd: float
    overhead_cost_usd: float
    total_cost_usd: float


class StationCostCalculator:
    """Modular cost calculator for manufacturing stations."""
    
    def __init__(self):
        """Initialize cost calculator with standard cost models."""
        self.logger = logging.getLogger('StationCostCalculator')
        
        # Equipment cost database (simplified)
        self.equipment_costs = {
            # SMT Equipment
            "pick_and_place_machine": 150000,
            "smt_reflow_oven": 80000,
            "smt_printer": 45000,
            "vision_system": 25000,
            "aoi_system": 75000,
            
            # Test Equipment  
            "multimeter": 5000,
            "lcr_meter": 8000,
            "oscilloscope": 15000,
            "power_supply": 3000,
            "function_generator": 7000,
            "boundary_scan_tester": 35000,
            
            # Assembly Equipment
            "soldering_station": 2000,
            "hot_air_station": 1500,
            "microscope": 15000,
            "torque_screwdriver": 800,
            
            # Inspection Equipment
            "x_ray_system": 200000,
            "3d_measurement": 120000,
            "leak_tester": 25000
        }
        
        # Labor rates by skill level ($/hour)
        self.labor_rates = {
            "operator_level_1": 25.0,  # Basic operator
            "operator_level_2": 35.0,  # Skilled operator
            "technician": 45.0,        # Technician
            "engineer": 65.0           # Engineer oversight
        }
        
        # Facility costs ($/m²/year)
        self.facility_costs = {
            "clean_room": 2000,     # Clean room space
            "production_floor": 500, # Standard production
            "test_area": 800,       # Test/QA area
            "storage": 200          # Storage/inventory
        }
        
        self.logger.info("StationCostCalculator initialized with cost database")
    
    def calculate_station_cost(self, station_config: Dict[str, Any]) -> CostBreakdown:
        """Calculate comprehensive cost breakdown for a station."""
        try:
            # Extract station parameters
            equipment_list = station_config.get('equipment_list', [])
            operators_required = station_config.get('operators_required', 1)
            floor_space_m2 = station_config.get('floor_space_m2', 2.0)
            station_type = station_config.get('station_type', 'production_floor')
            annual_volume = station_config.get('annual_volume', 10000)  # units/year
            
            # Calculate equipment costs
            equipment_cost = self._calculate_equipment_cost(equipment_list)
            
            # Calculate labor costs
            labor_cost = self._calculate_labor_cost(operators_required, station_type)
            
            # Calculate facilities cost
            facilities_cost = self._calculate_facilities_cost(floor_space_m2, station_type)
            
            # Calculate materials cost (consumables, utilities)
            materials_cost = self._calculate_materials_cost(annual_volume, equipment_list)
            
            # Calculate overhead cost (typically 15-25% of direct costs)
            direct_costs = equipment_cost + labor_cost + facilities_cost + materials_cost
            overhead_cost = direct_costs * 0.20  # 20% overhead
            
            total_cost = direct_costs + overhead_cost
            
            return CostBreakdown(
                equipment_cost_usd=equipment_cost,
                labor_cost_usd=labor_cost,
                facilities_cost_usd=facilities_cost,
                materials_cost_usd=materials_cost,
                overhead_cost_usd=overhead_cost,
                total_cost_usd=total_cost
            )
            
        except Exception as e:
            self.logger.error(f"Cost calculation failed: {e}")
            return CostBreakdown(0, 0, 0, 0, 0, 0)
    
    def _calculate_equipment_cost(self, equipment_list: List[str]) -> float:
        """Calculate equipment costs including depreciation."""
        total_equipment_cost = 0.0
        
        for equipment in equipment_list:
            base_cost = self.equipment_costs.get(equipment, 10000)  # Default $10k
            
            # Apply depreciation (5-year depreciation schedule)
            annual_depreciation = base_cost / 5.0
            total_equipment_cost += annual_depreciation
        
        return total_equipment_cost
    
    def _calculate_labor_cost(self, operators_required: int, station_type: str) -> float:
        """Calculate annual labor costs."""
        # Determine skill level based on station type
        if 'test' in station_type.lower() or 'inspection' in station_type.lower():
            skill_level = "technician"
        elif 'smt' in station_type.lower():
            skill_level = "operator_level_2"
        else:
            skill_level = "operator_level_1"
        
        hourly_rate = self.labor_rates[skill_level]
        
        # Calculate annual cost (40 hours/week * 50 weeks/year)
        annual_hours = 40 * 50
        annual_cost_per_operator = hourly_rate * annual_hours
        
        # Add benefits multiplier (typically 1.3x base salary)
        total_labor_cost = annual_cost_per_operator * operators_required * 1.3
        
        return total_labor_cost
    
    def _calculate_facilities_cost(self, floor_space_m2: float, station_type: str) -> float:
        """Calculate annual facilities cost."""
        # Determine facility type
        if 'smt' in station_type.lower():
            facility_type = "clean_room"
        elif 'test' in station_type.lower() or 'inspection' in station_type.lower():
            facility_type = "test_area"
        else:
            facility_type = "production_floor"
        
        cost_per_m2 = self.facility_costs[facility_type]
        annual_facilities_cost = floor_space_m2 * cost_per_m2
        
        return annual_facilities_cost
    
    def _calculate_materials_cost(self, annual_volume: int, equipment_list: List[str]) -> float:
        """Calculate annual materials and consumables cost."""
        # Base materials cost per unit
        base_materials_cost_per_unit = 0.50  # $0.50/unit baseline
        
        # Adjust based on equipment complexity
        equipment_factor = 1.0
        if "pick_and_place_machine" in equipment_list:
            equipment_factor += 0.3  # SMT consumables
        if "aoi_system" in equipment_list or "x_ray_system" in equipment_list:
            equipment_factor += 0.2  # Inspection consumables
        if "reflow_oven" in equipment_list:
            equipment_factor += 0.4  # Energy costs
        
        annual_materials_cost = annual_volume * base_materials_cost_per_unit * equipment_factor
        
        return annual_materials_cost
    
    def calculate_cost_per_unit(self, total_annual_cost: float, annual_volume: int) -> float:
        """Calculate cost per unit produced."""
        if annual_volume <= 0:
            return 0.0
        
        return total_annual_cost / annual_volume
    
    def calculate_breakeven_volume(self, total_annual_cost: float, selling_price_per_unit: float) -> int:
        """Calculate breakeven volume for profitability."""
        if selling_price_per_unit <= 0:
            return 0
        
        return int(total_annual_cost / selling_price_per_unit)
    
    def compare_station_costs(self, station_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare costs across multiple station configurations."""
        comparison = {
            'stations': [],
            'total_cost': 0.0,
            'cost_rankings': []
        }
        
        station_costs = []
        
        for i, config in enumerate(station_configs):
            cost_breakdown = self.calculate_station_cost(config)
            
            station_info = {
                'station_id': config.get('station_id', f'Station_{i}'),
                'cost_breakdown': cost_breakdown,
                'cost_per_m2': cost_breakdown.total_cost_usd / config.get('floor_space_m2', 1),
                'cost_efficiency': cost_breakdown.total_cost_usd / config.get('operators_required', 1)
            }
            
            comparison['stations'].append(station_info)
            station_costs.append((station_info['station_id'], cost_breakdown.total_cost_usd))
            comparison['total_cost'] += cost_breakdown.total_cost_usd
        
        # Rank by cost efficiency
        comparison['cost_rankings'] = sorted(station_costs, key=lambda x: x[1])
        
        return comparison
    
    def optimize_station_cost(self, station_config: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest cost optimization strategies for a station."""
        cost_breakdown = self.calculate_station_cost(station_config)
        
        optimizations = {
            'current_cost': cost_breakdown.total_cost_usd,
            'optimization_suggestions': [],
            'potential_savings_usd': 0.0
        }
        
        # Equipment optimization
        if cost_breakdown.equipment_cost_usd > cost_breakdown.total_cost_usd * 0.4:
            optimizations['optimization_suggestions'].append({
                'category': 'equipment',
                'suggestion': 'Consider leasing high-cost equipment instead of purchasing',
                'potential_savings': cost_breakdown.equipment_cost_usd * 0.3
            })
        
        # Labor optimization
        if cost_breakdown.labor_cost_usd > cost_breakdown.total_cost_usd * 0.5:
            optimizations['optimization_suggestions'].append({
                'category': 'labor',
                'suggestion': 'Evaluate automation opportunities to reduce labor requirements',
                'potential_savings': cost_breakdown.labor_cost_usd * 0.25
            })
        
        # Facilities optimization
        if cost_breakdown.facilities_cost_usd > cost_breakdown.total_cost_usd * 0.2:
            optimizations['optimization_suggestions'].append({
                'category': 'facilities',
                'suggestion': 'Optimize floor space utilization and layout efficiency',
                'potential_savings': cost_breakdown.facilities_cost_usd * 0.15
            })
        
        # Calculate total potential savings
        optimizations['potential_savings_usd'] = sum(
            suggestion['potential_savings'] for suggestion in optimizations['optimization_suggestions']
        )
        
        return optimizations