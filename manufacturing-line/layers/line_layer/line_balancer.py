"""LineBalancer - Line Balancing and Optimization Algorithms."""

import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from layers.station_layer.station_layer_engine import StationConfig


class BalancingStrategy(Enum):
    """Strategies for line balancing."""
    MINIMIZE_CYCLE_TIME = "minimize_cycle_time"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCE_WORKLOAD = "balance_workload"
    MINIMIZE_WIP = "minimize_wip"
    OPTIMIZE_EFFICIENCY = "optimize_efficiency"


class BottleneckType(Enum):
    """Types of bottlenecks in the line."""
    CYCLE_TIME = "cycle_time"
    SETUP_TIME = "setup_time"
    EQUIPMENT = "equipment"
    OPERATOR = "operator"
    QUALITY = "quality"
    MATERIAL = "material"


@dataclass
class BalancingResult:
    """Result from line balancing optimization."""
    success: bool
    original_line_uph: float
    balanced_line_uph: float
    improvement_percent: float
    bottleneck_station: str
    bottleneck_type: BottleneckType
    optimization_iterations: int
    recommended_changes: List[str] = field(default_factory=list)
    balancing_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TaktTimeAnalysis:
    """Analysis of takt time requirements."""
    target_takt_time_s: float
    actual_takt_time_s: float
    station_cycle_times: Dict[str, float]
    takt_time_compliance: Dict[str, bool]
    required_adjustments: Dict[str, float]


@dataclass
class LineEfficiencyAnalysis:
    """Analysis of line efficiency."""
    overall_efficiency: float
    station_efficiencies: Dict[str, float]
    theoretical_max_uph: float
    practical_max_uph: float
    efficiency_losses: Dict[str, float]
    improvement_opportunities: List[str]


class LineBalancer:
    """Advanced line balancing and optimization algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LineBalancer."""
        self.logger = logging.getLogger('LineBalancer')
        self.config = config or {}
        
        # Optimization parameters
        self.max_optimization_iterations = self.config.get('max_iterations', 10)
        self.convergence_threshold = self.config.get('convergence_threshold', 0.01)  # 1%
        self.target_line_efficiency = self.config.get('target_efficiency', 0.85)  # 85%
        
        # Performance tracking
        self.balancing_history: List[BalancingResult] = []
        
        self.logger.info("LineBalancer initialized")
    
    def balance_line(self, station_configs: List[StationConfig],
                    target_uph: float,
                    strategy: BalancingStrategy = BalancingStrategy.MAXIMIZE_THROUGHPUT) -> BalancingResult:
        """Balance a manufacturing line for optimal performance."""
        start_time = time.time()
        
        try:
            # Calculate baseline performance
            baseline_metrics = self._calculate_baseline_metrics(station_configs, target_uph)
            original_line_uph = baseline_metrics['line_uph']
            
            # Identify bottlenecks
            bottleneck_info = self._identify_bottlenecks(station_configs, baseline_metrics)
            
            # Apply balancing strategy
            balanced_configs, iterations = self._apply_balancing_strategy(
                station_configs, target_uph, strategy, bottleneck_info
            )
            
            # Calculate improved performance
            improved_metrics = self._calculate_baseline_metrics(balanced_configs, target_uph)
            balanced_line_uph = improved_metrics['line_uph']
            
            # Calculate improvement
            improvement_percent = ((balanced_line_uph - original_line_uph) / original_line_uph * 100) if original_line_uph > 0 else 0
            
            # Generate recommendations
            recommendations = self._generate_balancing_recommendations(
                station_configs, balanced_configs, bottleneck_info
            )
            
            result = BalancingResult(
                success=True,
                original_line_uph=original_line_uph,
                balanced_line_uph=balanced_line_uph,
                improvement_percent=improvement_percent,
                bottleneck_station=bottleneck_info['station_id'],
                bottleneck_type=bottleneck_info['type'],
                optimization_iterations=iterations,
                recommended_changes=recommendations,
                balancing_metrics={
                    'original_efficiency': baseline_metrics['efficiency'],
                    'balanced_efficiency': improved_metrics['efficiency'],
                    'cycle_time_variance': self._calculate_cycle_time_variance(balanced_configs),
                    'throughput_improvement': balanced_line_uph - original_line_uph
                }
            )
            
            # Store in history
            self.balancing_history.append(result)
            self.balancing_history = self.balancing_history[-100:]  # Keep last 100
            
            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Line balancing completed in {processing_time_ms:.1f}ms, "
                           f"improvement: {improvement_percent:.1f}%")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Line balancing failed: {e}")
            return BalancingResult(
                success=False,
                original_line_uph=0.0,
                balanced_line_uph=0.0,
                improvement_percent=0.0,
                bottleneck_station="unknown",
                bottleneck_type=BottleneckType.EQUIPMENT,
                optimization_iterations=0
            )
    
    def _calculate_baseline_metrics(self, station_configs: List[StationConfig], target_uph: float) -> Dict[str, Any]:
        """Calculate baseline performance metrics for the line."""
        if not station_configs:
            return {'line_uph': 0.0, 'efficiency': 0.0, 'cycle_times': {}}
        
        # Calculate individual station UPH (Units Per Hour)
        station_uphs = {}
        station_cycle_times = {}
        
        for station in station_configs:
            # Station UPH is based on cycle time (3600 seconds per hour)
            station_uph = 3600.0 / station.cycle_time_s if station.cycle_time_s > 0 else 0
            station_uphs[station.station_id] = station_uph
            station_cycle_times[station.station_id] = station.cycle_time_s
        
        # Line UPH is limited by the slowest station (bottleneck)
        line_uph = min(station_uphs.values()) if station_uphs else 0.0
        
        # Line efficiency is actual vs target
        efficiency = min(line_uph / target_uph, 1.0) if target_uph > 0 else 0.0
        
        return {
            'line_uph': line_uph,
            'efficiency': efficiency,
            'station_uphs': station_uphs,
            'cycle_times': station_cycle_times
        }
    
    def _identify_bottlenecks(self, station_configs: List[StationConfig], 
                            baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Identify bottlenecks in the line."""
        station_uphs = baseline_metrics['station_uphs']
        
        if not station_uphs:
            return {'station_id': 'unknown', 'type': BottleneckType.EQUIPMENT}
        
        # Find station with lowest UPH (bottleneck)
        bottleneck_station_id = min(station_uphs.keys(), key=lambda k: station_uphs[k])
        bottleneck_station = next((s for s in station_configs if s.station_id == bottleneck_station_id), None)
        
        # Determine bottleneck type
        bottleneck_type = BottleneckType.CYCLE_TIME  # Default
        
        if bottleneck_station:
            # Analyze what makes this station the bottleneck
            if bottleneck_station.setup_time_s > 300:  # More than 5 minutes setup
                bottleneck_type = BottleneckType.SETUP_TIME
            elif bottleneck_station.operators_required > 2:
                bottleneck_type = BottleneckType.OPERATOR
            elif len(bottleneck_station.equipment_list) > 5:
                bottleneck_type = BottleneckType.EQUIPMENT
            else:
                bottleneck_type = BottleneckType.CYCLE_TIME
        
        return {
            'station_id': bottleneck_station_id,
            'station': bottleneck_station,
            'type': bottleneck_type,
            'uph': station_uphs[bottleneck_station_id],
            'cycle_time': bottleneck_station.cycle_time_s if bottleneck_station else 0
        }
    
    def _apply_balancing_strategy(self, station_configs: List[StationConfig],
                                target_uph: float,
                                strategy: BalancingStrategy,
                                bottleneck_info: Dict[str, Any]) -> Tuple[List[StationConfig], int]:
        """Apply specific balancing strategy to optimize the line."""
        balanced_configs = [self._copy_station_config(config) for config in station_configs]
        iterations = 0
        
        if strategy == BalancingStrategy.MAXIMIZE_THROUGHPUT:
            balanced_configs, iterations = self._maximize_throughput(balanced_configs, target_uph, bottleneck_info)
        
        elif strategy == BalancingStrategy.MINIMIZE_CYCLE_TIME:
            balanced_configs, iterations = self._minimize_cycle_time(balanced_configs, bottleneck_info)
        
        elif strategy == BalancingStrategy.BALANCE_WORKLOAD:
            balanced_configs, iterations = self._balance_workload(balanced_configs, target_uph)
        
        elif strategy == BalancingStrategy.MINIMIZE_WIP:
            balanced_configs, iterations = self._minimize_wip(balanced_configs, target_uph)
        
        elif strategy == BalancingStrategy.OPTIMIZE_EFFICIENCY:
            balanced_configs, iterations = self._optimize_efficiency(balanced_configs, target_uph, bottleneck_info)
        
        return balanced_configs, iterations
    
    def _maximize_throughput(self, station_configs: List[StationConfig],
                           target_uph: float,
                           bottleneck_info: Dict[str, Any]) -> Tuple[List[StationConfig], int]:
        """Maximize line throughput by optimizing bottleneck station."""
        iterations = 0
        max_iterations = self.max_optimization_iterations
        
        bottleneck_station_id = bottleneck_info['station_id']
        bottleneck_station = next((s for s in station_configs if s.station_id == bottleneck_station_id), None)
        
        if not bottleneck_station:
            return station_configs, 0
        
        current_uph = 3600.0 / bottleneck_station.cycle_time_s if bottleneck_station.cycle_time_s > 0 else 0
        
        while iterations < max_iterations and current_uph < target_uph * 0.95:  # Within 5% of target
            # Optimize bottleneck station
            if bottleneck_info['type'] == BottleneckType.CYCLE_TIME:
                # Reduce cycle time by optimizing process
                improvement_factor = min(1.1, target_uph / current_uph)  # Max 10% improvement per iteration
                bottleneck_station.cycle_time_s /= improvement_factor
                
            elif bottleneck_info['type'] == BottleneckType.SETUP_TIME:
                # Reduce setup time
                bottleneck_station.setup_time_s *= 0.9  # 10% reduction
                
            elif bottleneck_info['type'] == BottleneckType.OPERATOR:
                # Add operator if beneficial
                if bottleneck_station.operators_required < 3:  # Max 3 operators
                    bottleneck_station.operators_required += 1
                    bottleneck_station.cycle_time_s *= 0.8  # 20% improvement with additional operator
            
            # Recalculate UPH
            current_uph = 3600.0 / bottleneck_station.cycle_time_s if bottleneck_station.cycle_time_s > 0 else 0
            iterations += 1
        
        return station_configs, iterations
    
    def _minimize_cycle_time(self, station_configs: List[StationConfig],
                           bottleneck_info: Dict[str, Any]) -> Tuple[List[StationConfig], int]:
        """Minimize overall cycle time by optimizing all stations."""
        iterations = 1
        
        # Optimize each station's cycle time
        for station in station_configs:
            # Apply general cycle time improvements
            if station.cycle_time_s > 5.0:  # Only optimize if cycle time > 5 seconds
                optimization_factor = 0.95  # 5% improvement
                station.cycle_time_s *= optimization_factor
        
        return station_configs, iterations
    
    def _balance_workload(self, station_configs: List[StationConfig],
                         target_uph: float) -> Tuple[List[StationConfig], int]:
        """Balance workload across all stations."""
        iterations = 0
        max_iterations = 5
        
        target_cycle_time = 3600.0 / target_uph
        
        while iterations < max_iterations:
            # Calculate current variance
            cycle_times = [s.cycle_time_s for s in station_configs]
            variance_before = self._calculate_variance(cycle_times)
            
            # Adjust stations towards target cycle time
            for station in station_configs:
                if station.cycle_time_s > target_cycle_time * 1.1:  # 10% above target
                    station.cycle_time_s *= 0.95  # Reduce by 5%
                elif station.cycle_time_s < target_cycle_time * 0.9:  # 10% below target
                    # Could potentially increase cycle time or redistribute work
                    pass
            
            # Check for improvement
            cycle_times_after = [s.cycle_time_s for s in station_configs]
            variance_after = self._calculate_variance(cycle_times_after)
            
            if abs(variance_after - variance_before) / variance_before < self.convergence_threshold:
                break
                
            iterations += 1
        
        return station_configs, iterations
    
    def _minimize_wip(self, station_configs: List[StationConfig],
                     target_uph: float) -> Tuple[List[StationConfig], int]:
        """Minimize work-in-progress by balancing flow."""
        # Simple implementation - balance cycle times to minimize buffers
        return self._balance_workload(station_configs, target_uph)
    
    def _optimize_efficiency(self, station_configs: List[StationConfig],
                           target_uph: float,
                           bottleneck_info: Dict[str, Any]) -> Tuple[List[StationConfig], int]:
        """Optimize overall line efficiency."""
        iterations = 0
        
        # Focus on bottleneck and nearby stations
        bottleneck_station_id = bottleneck_info['station_id']
        
        for i, station in enumerate(station_configs):
            if station.station_id == bottleneck_station_id:
                # Optimize bottleneck station aggressively
                station.cycle_time_s *= 0.9  # 10% improvement
                
                # Optimize adjacent stations
                if i > 0:  # Previous station
                    station_configs[i-1].cycle_time_s *= 0.95  # 5% improvement
                if i < len(station_configs) - 1:  # Next station
                    station_configs[i+1].cycle_time_s *= 0.95  # 5% improvement
                
                break
        
        iterations = 1
        return station_configs, iterations
    
    def _copy_station_config(self, station_config: StationConfig) -> StationConfig:
        """Create a deep copy of station configuration."""
        return StationConfig(
            station_id=station_config.station_id,
            station_type=station_config.station_type,
            equipment_list=station_config.equipment_list.copy(),
            cycle_time_s=station_config.cycle_time_s,
            setup_time_s=station_config.setup_time_s,
            operators_required=station_config.operators_required,
            floor_space_m2=station_config.floor_space_m2
        )
    
    def _calculate_cycle_time_variance(self, station_configs: List[StationConfig]) -> float:
        """Calculate variance in cycle times across stations."""
        cycle_times = [station.cycle_time_s for station in station_configs]
        return self._calculate_variance(cycle_times)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance
    
    def _generate_balancing_recommendations(self, original_configs: List[StationConfig],
                                          balanced_configs: List[StationConfig],
                                          bottleneck_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations for line balancing improvements."""
        recommendations = []
        
        # Check bottleneck improvements
        bottleneck_id = bottleneck_info['station_id']
        original_station = next((s for s in original_configs if s.station_id == bottleneck_id), None)
        balanced_station = next((s for s in balanced_configs if s.station_id == bottleneck_id), None)
        
        if original_station and balanced_station:
            cycle_time_improvement = (original_station.cycle_time_s - balanced_station.cycle_time_s) / original_station.cycle_time_s
            
            if cycle_time_improvement > 0.05:  # 5% improvement
                recommendations.append(
                    f"Bottleneck station {bottleneck_id} cycle time improved by {cycle_time_improvement:.1%}"
                )
                
                if balanced_station.operators_required > original_station.operators_required:
                    recommendations.append(
                        f"Consider adding operator to station {bottleneck_id} for {cycle_time_improvement:.1%} improvement"
                    )
        
        # Check overall balance
        original_variance = self._calculate_cycle_time_variance(original_configs)
        balanced_variance = self._calculate_cycle_time_variance(balanced_configs)
        
        if balanced_variance < original_variance * 0.9:  # 10% reduction in variance
            recommendations.append("Line balance improved - reduced cycle time variance")
        
        # Check for extreme cycle times
        max_cycle_time = max(s.cycle_time_s for s in balanced_configs)
        min_cycle_time = min(s.cycle_time_s for s in balanced_configs)
        
        if max_cycle_time / min_cycle_time > 2.0:  # Ratio > 2:1
            recommendations.append("Consider redistributing work - large cycle time differences remain")
        
        # General recommendations
        if bottleneck_info['type'] == BottleneckType.SETUP_TIME:
            recommendations.append("Focus on setup time reduction techniques (SMED, quick changeover)")
        
        elif bottleneck_info['type'] == BottleneckType.EQUIPMENT:
            recommendations.append("Consider equipment capacity analysis and potential upgrades")
        
        elif bottleneck_info['type'] == BottleneckType.OPERATOR:
            recommendations.append("Analyze operator workload and consider ergonomic improvements")
        
        return recommendations
    
    def analyze_takt_time(self, station_configs: List[StationConfig], target_uph: float) -> TaktTimeAnalysis:
        """Analyze takt time requirements and compliance."""
        target_takt_time = 3600.0 / target_uph if target_uph > 0 else 0
        
        # Calculate actual takt time (slowest station)
        cycle_times = {s.station_id: s.cycle_time_s for s in station_configs}
        actual_takt_time = max(cycle_times.values()) if cycle_times else 0
        
        # Check compliance for each station
        takt_compliance = {}
        required_adjustments = {}
        
        for station in station_configs:
            compliant = station.cycle_time_s <= target_takt_time
            takt_compliance[station.station_id] = compliant
            
            if not compliant:
                required_adjustments[station.station_id] = station.cycle_time_s - target_takt_time
        
        return TaktTimeAnalysis(
            target_takt_time_s=target_takt_time,
            actual_takt_time_s=actual_takt_time,
            station_cycle_times=cycle_times,
            takt_time_compliance=takt_compliance,
            required_adjustments=required_adjustments
        )
    
    def analyze_line_efficiency(self, station_configs: List[StationConfig], target_uph: float) -> LineEfficiencyAnalysis:
        """Analyze overall line efficiency and improvement opportunities."""
        baseline_metrics = self._calculate_baseline_metrics(station_configs, target_uph)
        
        # Calculate theoretical maximum (fastest possible cycle time)
        theoretical_max_cycle_time = min(s.cycle_time_s for s in station_configs) if station_configs else 0
        theoretical_max_uph = 3600.0 / theoretical_max_cycle_time if theoretical_max_cycle_time > 0 else 0
        
        # Calculate practical maximum (considering setup times and efficiency)
        practical_efficiency = 0.85  # Assume 85% practical efficiency
        practical_max_uph = theoretical_max_uph * practical_efficiency
        
        # Calculate station efficiencies
        station_efficiencies = {}
        target_cycle_time = 3600.0 / target_uph if target_uph > 0 else float('inf')
        
        for station in station_configs:
            efficiency = min(target_cycle_time / station.cycle_time_s, 1.0) if station.cycle_time_s > 0 else 0
            station_efficiencies[station.station_id] = efficiency
        
        # Identify efficiency losses
        efficiency_losses = {}
        for station_id, efficiency in station_efficiencies.items():
            if efficiency < 0.9:  # Below 90% efficiency
                efficiency_losses[station_id] = (0.9 - efficiency) * 100  # Percentage loss
        
        # Generate improvement opportunities
        improvement_opportunities = []
        if baseline_metrics['efficiency'] < self.target_line_efficiency:
            improvement_opportunities.append(
                f"Overall line efficiency ({baseline_metrics['efficiency']:.1%}) "
                f"is below target ({self.target_line_efficiency:.1%})"
            )
        
        for station_id, loss in efficiency_losses.items():
            if loss > 10:  # More than 10% loss
                improvement_opportunities.append(f"Station {station_id} has {loss:.1f}% efficiency loss")
        
        return LineEfficiencyAnalysis(
            overall_efficiency=baseline_metrics['efficiency'],
            station_efficiencies=station_efficiencies,
            theoretical_max_uph=theoretical_max_uph,
            practical_max_uph=practical_max_uph,
            efficiency_losses=efficiency_losses,
            improvement_opportunities=improvement_opportunities
        )
    
    def get_balancing_summary(self) -> Dict[str, Any]:
        """Get summary of line balancing performance."""
        if not self.balancing_history:
            return {'no_data': True}
        
        recent_results = self.balancing_history[-10:]  # Last 10 balancing operations
        
        avg_improvement = sum(r.improvement_percent for r in recent_results) / len(recent_results)
        avg_iterations = sum(r.optimization_iterations for r in recent_results) / len(recent_results)
        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
        
        return {
            'total_balancing_operations': len(self.balancing_history),
            'average_improvement_percent': avg_improvement,
            'average_optimization_iterations': avg_iterations,
            'success_rate': success_rate,
            'target_iterations': self.max_optimization_iterations,
            'convergence_achieved_rate': sum(1 for r in recent_results if r.optimization_iterations < self.max_optimization_iterations) / len(recent_results)
        }