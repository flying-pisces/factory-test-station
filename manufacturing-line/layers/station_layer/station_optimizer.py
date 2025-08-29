"""Station Optimization Framework - Modular Multi-Objective Optimization."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .cost_calculator import StationCostCalculator, CostBreakdown
from .uph_calculator import StationUPHCalculator, UPHAnalysis


class OptimizationObjective(Enum):
    """Optimization objectives for station configuration."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_UPH = "maximize_uph"
    MINIMIZE_COST_PER_UNIT = "minimize_cost_per_unit"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_LINE = "balance_line"


@dataclass
class OptimizationResult:
    """Result from station optimization."""
    optimized_config: Dict[str, Any]
    original_cost: float
    optimized_cost: float
    original_uph: float
    optimized_uph: float
    cost_savings_usd: float
    uph_improvement: float
    optimization_time_ms: float
    iterations: int


class StationOptimizer:
    """Modular optimization framework for manufacturing stations."""
    
    def __init__(self):
        """Initialize optimization framework."""
        self.logger = logging.getLogger('StationOptimizer')
        
        # Initialize calculators
        self.cost_calculator = StationCostCalculator()
        self.uph_calculator = StationUPHCalculator()
        
        # Optimization parameters
        self.max_iterations = 20
        self.convergence_threshold = 0.01  # 1% improvement threshold
        
        self.logger.info("StationOptimizer initialized with cost and UPH calculators")
    
    def optimize_single_station(self, station_config: Dict[str, Any],
                               objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST_PER_UNIT,
                               constraints: Dict[str, Any] = None) -> OptimizationResult:
        """Optimize a single station configuration."""
        start_time = time.time()
        constraints = constraints or {}
        
        try:
            # Calculate baseline metrics
            original_cost_breakdown = self.cost_calculator.calculate_station_cost(station_config)
            original_uph_analysis = self.uph_calculator.calculate_station_uph(station_config)
            
            original_cost = original_cost_breakdown.total_cost_usd
            original_uph = original_uph_analysis.practical_uph
            
            # Initialize optimization
            best_config = station_config.copy()
            best_score = self._calculate_objective_score(
                original_cost, original_uph, objective
            )
            
            iteration = 0
            
            # Simple optimization loop
            for iteration in range(self.max_iterations):
                # Generate candidate configurations
                candidates = self._generate_candidates(best_config, constraints)
                
                improved = False
                
                for candidate in candidates:
                    # Evaluate candidate
                    cost_breakdown = self.cost_calculator.calculate_station_cost(candidate)
                    uph_analysis = self.uph_calculator.calculate_station_uph(candidate)
                    
                    candidate_score = self._calculate_objective_score(
                        cost_breakdown.total_cost_usd, uph_analysis.practical_uph, objective
                    )
                    
                    # Check if candidate is better
                    if self._is_better_score(candidate_score, best_score, objective):
                        improvement = abs(candidate_score - best_score) / abs(best_score)
                        if improvement > self.convergence_threshold:
                            best_config = candidate.copy()
                            best_score = candidate_score
                            improved = True
                
                # Check convergence
                if not improved:
                    self.logger.info(f"Optimization converged after {iteration + 1} iterations")
                    break
            
            # Calculate final metrics
            final_cost_breakdown = self.cost_calculator.calculate_station_cost(best_config)
            final_uph_analysis = self.uph_calculator.calculate_station_uph(best_config)
            
            optimization_time_ms = (time.time() - start_time) * 1000
            
            return OptimizationResult(
                optimized_config=best_config,
                original_cost=original_cost,
                optimized_cost=final_cost_breakdown.total_cost_usd,
                original_uph=original_uph,
                optimized_uph=final_uph_analysis.practical_uph,
                cost_savings_usd=original_cost - final_cost_breakdown.total_cost_usd,
                uph_improvement=final_uph_analysis.practical_uph - original_uph,
                optimization_time_ms=optimization_time_ms,
                iterations=iteration + 1
            )
            
        except Exception as e:
            self.logger.error(f"Station optimization failed: {e}")
            optimization_time_ms = (time.time() - start_time) * 1000
            
            return OptimizationResult(
                optimized_config=station_config,
                original_cost=0, optimized_cost=0,
                original_uph=0, optimized_uph=0,
                cost_savings_usd=0, uph_improvement=0,
                optimization_time_ms=optimization_time_ms,
                iterations=0
            )
    
    def optimize_line_configuration(self, station_configs: List[Dict[str, Any]],
                                  objective: OptimizationObjective = OptimizationObjective.BALANCE_LINE,
                                  constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize entire line configuration for balanced throughput."""
        start_time = time.time()
        constraints = constraints or {}
        
        try:
            # Calculate baseline line metrics
            baseline_line_analysis = self.uph_calculator.calculate_line_uph(station_configs)
            baseline_costs = [
                self.cost_calculator.calculate_station_cost(config).total_cost_usd 
                for config in station_configs
            ]
            
            optimized_configs = []
            optimization_results = []
            
            # Optimize each station considering line balance
            for i, station_config in enumerate(station_configs):
                station_objective = self._determine_station_objective(
                    i, station_configs, baseline_line_analysis, objective
                )
                
                result = self.optimize_single_station(
                    station_config, station_objective, constraints
                )
                
                optimized_configs.append(result.optimized_config)
                optimization_results.append(result)
            
            # Calculate final line metrics
            final_line_analysis = self.uph_calculator.calculate_line_uph(optimized_configs)
            final_costs = [
                self.cost_calculator.calculate_station_cost(config).total_cost_usd 
                for config in optimized_configs
            ]
            
            optimization_time_ms = (time.time() - start_time) * 1000
            
            return {
                'optimized_station_configs': optimized_configs,
                'station_optimization_results': optimization_results,
                'baseline_line_uph': baseline_line_analysis['line_uph'],
                'optimized_line_uph': final_line_analysis['line_uph'],
                'baseline_total_cost': sum(baseline_costs),
                'optimized_total_cost': sum(final_costs),
                'line_uph_improvement': final_line_analysis['line_uph'] - baseline_line_analysis['line_uph'],
                'total_cost_savings': sum(baseline_costs) - sum(final_costs),
                'baseline_efficiency_balance': baseline_line_analysis['efficiency_balance'],
                'optimized_efficiency_balance': final_line_analysis['efficiency_balance'],
                'optimization_time_ms': optimization_time_ms,
                'bottleneck_shift': {
                    'original': baseline_line_analysis['bottleneck_station'],
                    'optimized': final_line_analysis['bottleneck_station']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Line optimization failed: {e}")
            optimization_time_ms = (time.time() - start_time) * 1000
            
            return {
                'optimized_station_configs': station_configs,
                'optimization_time_ms': optimization_time_ms,
                'success': False,
                'error': str(e)
            }
    
    def _generate_candidates(self, base_config: Dict[str, Any], 
                           constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate configurations for optimization."""
        candidates = []
        
        # Equipment optimization candidates
        equipment_list = base_config.get('equipment_list', [])
        if equipment_list:
            # Try reducing equipment (cost optimization)
            reduced_equipment = self._optimize_equipment_list(equipment_list, 'reduce')
            if reduced_equipment != equipment_list:
                candidate = base_config.copy()
                candidate['equipment_list'] = reduced_equipment
                candidates.append(candidate)
            
            # Try upgrading equipment (UPH optimization)
            upgraded_equipment = self._optimize_equipment_list(equipment_list, 'upgrade')
            if upgraded_equipment != equipment_list:
                candidate = base_config.copy()
                candidate['equipment_list'] = upgraded_equipment
                candidates.append(candidate)
        
        # Operator optimization candidates
        operators = base_config.get('operators_required', 1)
        if operators > 1:
            # Try reducing operators
            candidate = base_config.copy()
            candidate['operators_required'] = operators - 1
            candidates.append(candidate)
        
        # Batch size optimization candidates
        current_batch = base_config.get('batch_size', 100)
        batch_candidates = [current_batch * 0.8, current_batch * 1.2, current_batch * 1.5]
        
        for batch_size in batch_candidates:
            if batch_size != current_batch:
                candidate = base_config.copy()
                candidate['batch_size'] = int(batch_size)
                candidates.append(candidate)
        
        # Floor space optimization
        floor_space = base_config.get('floor_space_m2', 2.0)
        space_candidates = [floor_space * 0.9, floor_space * 1.1]
        
        for space in space_candidates:
            if abs(space - floor_space) > 0.1:  # Meaningful change
                candidate = base_config.copy()
                candidate['floor_space_m2'] = space
                candidates.append(candidate)
        
        return candidates[:10]  # Limit to 10 candidates per iteration
    
    def _optimize_equipment_list(self, equipment_list: List[str], mode: str) -> List[str]:
        """Optimize equipment list for cost or performance."""
        if mode == 'reduce':
            # Remove optional/redundant equipment
            optional_equipment = ['microscope', 'function_generator']
            return [eq for eq in equipment_list if eq not in optional_equipment]
        
        elif mode == 'upgrade':
            # Add performance-enhancing equipment
            upgrades = {
                'multimeter': 'boundary_scan_tester',
                'lcr_meter': 'network_analyzer'
            }
            
            upgraded_list = equipment_list.copy()
            for old, new in upgrades.items():
                if old in upgraded_list and new not in upgraded_list:
                    upgraded_list.append(new)
            
            return upgraded_list
        
        return equipment_list
    
    def _calculate_objective_score(self, cost: float, uph: float, 
                                 objective: OptimizationObjective) -> float:
        """Calculate optimization score based on objective."""
        if objective == OptimizationObjective.MINIMIZE_COST:
            return -cost  # Negative because we want to minimize
        
        elif objective == OptimizationObjective.MAXIMIZE_UPH:
            return uph
        
        elif objective == OptimizationObjective.MINIMIZE_COST_PER_UNIT:
            if uph <= 0:
                return -float('inf')
            return -(cost / uph)  # Negative cost per unit
        
        elif objective == OptimizationObjective.MAXIMIZE_EFFICIENCY:
            # Efficiency = UPH / Cost ratio
            if cost <= 0:
                return float('inf')
            return uph / cost
        
        else:  # Default to cost per unit
            if uph <= 0:
                return -float('inf')
            return -(cost / uph)
    
    def _is_better_score(self, new_score: float, current_best: float,
                        objective: OptimizationObjective) -> bool:
        """Determine if new score is better than current best."""
        return new_score > current_best
    
    def _determine_station_objective(self, station_index: int, 
                                   station_configs: List[Dict[str, Any]],
                                   line_analysis: Dict[str, Any],
                                   overall_objective: OptimizationObjective) -> OptimizationObjective:
        """Determine optimization objective for individual station in line context."""
        station_id = station_configs[station_index].get('station_id', f'station_{station_index}')
        
        if overall_objective == OptimizationObjective.BALANCE_LINE:
            # If this station is the bottleneck, optimize for UPH
            if station_id == line_analysis.get('bottleneck_station'):
                return OptimizationObjective.MAXIMIZE_UPH
            else:
                # Non-bottleneck stations optimize for cost
                return OptimizationObjective.MINIMIZE_COST
        else:
            # Use overall objective
            return overall_objective
    
    def recommend_improvements(self, station_config: Dict[str, Any]) -> Dict[str, Any]:
        """Provide improvement recommendations for a station."""
        cost_breakdown = self.cost_calculator.calculate_station_cost(station_config)
        uph_analysis = self.uph_calculator.calculate_station_uph(station_config)
        
        recommendations = {
            'cost_recommendations': [],
            'uph_recommendations': [],
            'overall_priority': 'balanced'
        }
        
        # Cost-based recommendations
        if cost_breakdown.equipment_cost_usd > cost_breakdown.total_cost_usd * 0.6:
            recommendations['cost_recommendations'].append({
                'type': 'equipment_cost',
                'priority': 'high',
                'suggestion': 'Consider equipment leasing or shared resources',
                'potential_savings_percent': 25
            })
        
        if cost_breakdown.labor_cost_usd > cost_breakdown.total_cost_usd * 0.4:
            recommendations['cost_recommendations'].append({
                'type': 'labor_cost',
                'priority': 'medium',
                'suggestion': 'Evaluate automation opportunities',
                'potential_savings_percent': 15
            })
        
        # UPH-based recommendations
        if uph_analysis.efficiency_percent < 70:
            recommendations['uph_recommendations'].append({
                'type': 'efficiency',
                'priority': 'high',
                'suggestion': f'Address {uph_analysis.bottleneck_type.value} bottleneck',
                'potential_improvement_percent': 20
            })
        
        if uph_analysis.practical_uph < uph_analysis.theoretical_uph * 0.8:
            recommendations['uph_recommendations'].append({
                'type': 'utilization',
                'priority': 'medium',
                'suggestion': 'Implement lean manufacturing principles',
                'potential_improvement_percent': 10
            })
        
        # Determine overall priority
        cost_issues = len(recommendations['cost_recommendations'])
        uph_issues = len(recommendations['uph_recommendations'])
        
        if cost_issues > uph_issues:
            recommendations['overall_priority'] = 'cost_reduction'
        elif uph_issues > cost_issues:
            recommendations['overall_priority'] = 'throughput_improvement'
        else:
            recommendations['overall_priority'] = 'balanced'
        
        return recommendations