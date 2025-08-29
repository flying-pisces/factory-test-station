"""
Week 4: AnalyticsEngine - Advanced Analytics and KPI Calculation

This module implements advanced analytics capabilities for manufacturing line control,
including real-time KPI calculation, performance forecasting, trend analysis, and
optimization opportunity identification.

Key Features:
- Real-time advanced KPI calculation and monitoring
- Performance forecasting with statistical models
- Trend analysis and bottleneck prediction
- Optimization opportunity identification
- Dashboard data processing with <100ms performance target

Author: Claude Code
Date: 2024-08-28
Version: 1.0
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
import statistics
import math
from collections import defaultdict, deque

class KPIType(Enum):
    THROUGHPUT = "throughput"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    UTILIZATION = "utilization"
    COST = "cost"
    DELIVERY = "delivery"
    MAINTENANCE = "maintenance"

class TrendDirection(Enum):
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"

class BottleneckType(Enum):
    RESOURCE_BOTTLENECK = "resource_bottleneck"
    PROCESS_BOTTLENECK = "process_bottleneck"
    QUALITY_BOTTLENECK = "quality_bottleneck"
    MATERIAL_BOTTLENECK = "material_bottleneck"

@dataclass
class KPIMetric:
    kpi_name: str
    kpi_type: KPIType
    current_value: float
    target_value: Optional[float]
    unit: str
    timestamp: datetime
    confidence: float  # 0.0 to 1.0
    trend: TrendDirection
    variance: float
    percentile_rank: float  # 0.0 to 100.0

@dataclass
class PerformanceForecast:
    metric_name: str
    forecast_horizon_hours: int
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    forecast_accuracy: float
    trend_analysis: Dict[str, Any]
    seasonal_patterns: List[Dict[str, Any]]

@dataclass
class TrendAnalysis:
    metric_name: str
    trend_direction: TrendDirection
    trend_strength: float  # 0.0 to 1.0
    trend_duration_hours: int
    rate_of_change: float
    statistical_significance: float
    breakpoints: List[datetime]  # Significant trend changes

@dataclass
class BottleneckAnalysis:
    bottleneck_id: str
    bottleneck_type: BottleneckType
    affected_processes: List[str]
    severity: float  # 0.0 to 1.0
    estimated_impact: Dict[str, float]
    root_causes: List[str]
    improvement_opportunities: List[Dict[str, Any]]

@dataclass
class OptimizationOpportunity:
    opportunity_id: str
    category: str
    description: str
    potential_improvement: Dict[str, float]  # KPI -> improvement value
    implementation_effort: str  # "low", "medium", "high"
    estimated_roi: float
    priority_score: float
    recommended_actions: List[str]

class AnalyticsEngine:
    """Advanced analytics and KPI calculation engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AnalyticsEngine with configuration."""
        self.config = config or {}
        
        # Performance configuration
        self.performance_target_ms = self.config.get('performance_target_ms', 100)
        self.calculation_timeout_ms = self.config.get('calculation_timeout_ms', 500)
        
        # Analytics parameters
        self.historical_window_hours = self.config.get('historical_window_hours', 168)  # 1 week
        self.trend_detection_sensitivity = self.config.get('trend_detection_sensitivity', 0.05)
        self.forecast_horizon_hours = self.config.get('forecast_horizon_hours', 72)  # 3 days
        
        # Data storage
        self.kpi_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.trend_cache: Dict[str, TrendAnalysis] = {}
        self.bottleneck_cache: Dict[str, BottleneckAnalysis] = {}
        
        # KPI definitions and targets
        self.kpi_definitions = self._initialize_kpi_definitions()
        self.benchmark_data = self._initialize_benchmarks()
        
        # Performance monitoring
        self.performance_metrics = {
            'avg_calculation_time_ms': 0,
            'total_calculations': 0,
            'successful_calculations': 0,
            'kpis_calculated': 0,
            'forecasts_generated': 0,
            'opportunities_identified': 0
        }
        
        logging.info(f"AnalyticsEngine initialized with {self.performance_target_ms}ms target")

    def calculate_advanced_kpis(self, 
                               production_data: Dict[str, Any],
                               time_window_hours: int = 24) -> Dict[str, KPIMetric]:
        """Calculate advanced KPIs and performance metrics."""
        start_time = time.time()
        calculated_kpis = {}
        
        try:
            # Extract time window data
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            window_data = self._extract_time_window_data(production_data, cutoff_time)
            
            # Calculate each KPI type
            kpi_calculators = {
                KPIType.THROUGHPUT: self._calculate_throughput_kpis,
                KPIType.EFFICIENCY: self._calculate_efficiency_kpis,
                KPIType.QUALITY: self._calculate_quality_kpis,
                KPIType.UTILIZATION: self._calculate_utilization_kpis,
                KPIType.COST: self._calculate_cost_kpis,
                KPIType.DELIVERY: self._calculate_delivery_kpis,
                KPIType.MAINTENANCE: self._calculate_maintenance_kpis
            }
            
            for kpi_type, calculator in kpi_calculators.items():
                try:
                    kpi_results = calculator(window_data, time_window_hours)
                    calculated_kpis.update(kpi_results)
                except Exception as e:
                    logging.warning(f"Failed to calculate {kpi_type.value} KPIs: {e}")
            
            # Update KPI history
            self._update_kpi_history(calculated_kpis)
            
            # Calculate trends and percentiles
            self._enrich_kpis_with_analysis(calculated_kpis)
            
            # Update performance metrics
            computation_time = (time.time() - start_time) * 1000
            self._update_analytics_metrics(computation_time, True, len(calculated_kpis))
            
            logging.info(f"Calculated {len(calculated_kpis)} KPIs in {computation_time:.2f}ms")
            
            return calculated_kpis
            
        except Exception as e:
            logging.error(f"KPI calculation failed: {e}")
            computation_time = (time.time() - start_time) * 1000
            self._update_analytics_metrics(computation_time, False, 0)
            return {}

    def generate_performance_forecasts(self,
                                     historical_data: Dict[str, List[Dict[str, Any]]],
                                     forecast_horizon_hours: int = 72) -> Dict[str, PerformanceForecast]:
        """Generate performance and capacity forecasts."""
        start_time = time.time()
        forecasts = {}
        
        try:
            for metric_name, data_points in historical_data.items():
                if len(data_points) < 10:  # Need minimum data for forecasting
                    continue
                
                # Extract values and timestamps
                values = [point.get('value', 0) for point in data_points]
                timestamps = [point.get('timestamp', datetime.now()) for point in data_points]
                
                # Generate forecast
                forecast = self._generate_metric_forecast(
                    metric_name, values, timestamps, forecast_horizon_hours
                )
                
                if forecast:
                    forecasts[metric_name] = forecast
            
            # Update performance metrics
            self.performance_metrics['forecasts_generated'] += len(forecasts)
            
            computation_time = (time.time() - start_time) * 1000
            logging.info(f"Generated {len(forecasts)} forecasts in {computation_time:.2f}ms")
            
            return forecasts
            
        except Exception as e:
            logging.error(f"Forecast generation failed: {e}")
            return {}

    def identify_optimization_opportunities(self,
                                          current_metrics: Dict[str, KPIMetric],
                                          benchmarks: Dict[str, float]) -> List[OptimizationOpportunity]:
        """Identify opportunities for performance optimization."""
        start_time = time.time()
        opportunities = []
        
        try:
            # Compare current metrics with benchmarks
            performance_gaps = self._identify_performance_gaps(current_metrics, benchmarks)
            
            # Analyze bottlenecks
            bottlenecks = self._identify_bottlenecks(current_metrics)
            
            # Generate opportunities from gaps
            for metric_name, gap_info in performance_gaps.items():
                opportunity = self._create_gap_opportunity(metric_name, gap_info, current_metrics)
                if opportunity:
                    opportunities.append(opportunity)
            
            # Generate opportunities from bottlenecks
            for bottleneck in bottlenecks:
                bottleneck_opportunities = self._create_bottleneck_opportunities(bottleneck)
                opportunities.extend(bottleneck_opportunities)
            
            # Identify trend-based opportunities
            trend_opportunities = self._identify_trend_opportunities(current_metrics)
            opportunities.extend(trend_opportunities)
            
            # Score and prioritize opportunities
            opportunities = self._prioritize_opportunities(opportunities)
            
            # Update performance metrics
            self.performance_metrics['opportunities_identified'] += len(opportunities)
            
            computation_time = (time.time() - start_time) * 1000
            logging.info(f"Identified {len(opportunities)} optimization opportunities in {computation_time:.2f}ms")
            
            return opportunities
            
        except Exception as e:
            logging.error(f"Opportunity identification failed: {e}")
            return []

    def _initialize_kpi_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize KPI definitions and calculation methods."""
        return {
            'overall_throughput_uph': {
                'type': KPIType.THROUGHPUT,
                'unit': 'units/hour',
                'target': 120.0,
                'higher_is_better': True
            },
            'line_efficiency': {
                'type': KPIType.EFFICIENCY,
                'unit': 'percentage',
                'target': 85.0,
                'higher_is_better': True
            },
            'quality_yield': {
                'type': KPIType.QUALITY,
                'unit': 'percentage',
                'target': 99.0,
                'higher_is_better': True
            },
            'resource_utilization': {
                'type': KPIType.UTILIZATION,
                'unit': 'percentage',
                'target': 80.0,
                'higher_is_better': True
            },
            'cost_per_unit': {
                'type': KPIType.COST,
                'unit': 'currency',
                'target': 10.0,
                'higher_is_better': False
            },
            'on_time_delivery': {
                'type': KPIType.DELIVERY,
                'unit': 'percentage',
                'target': 95.0,
                'higher_is_better': True
            },
            'equipment_availability': {
                'type': KPIType.MAINTENANCE,
                'unit': 'percentage',
                'target': 95.0,
                'higher_is_better': True
            }
        }

    def _initialize_benchmarks(self) -> Dict[str, float]:
        """Initialize benchmark data for comparison."""
        return {
            'industry_throughput_uph': 100.0,
            'industry_efficiency': 75.0,
            'industry_quality_yield': 97.0,
            'industry_utilization': 70.0,
            'industry_cost_per_unit': 12.0,
            'industry_delivery': 90.0,
            'industry_availability': 90.0
        }

    def _extract_time_window_data(self, production_data: Dict[str, Any], cutoff_time: datetime) -> Dict[str, Any]:
        """Extract data within specified time window."""
        window_data = {}
        
        for key, value in production_data.items():
            if isinstance(value, list):
                # Filter time-series data
                filtered_data = []
                for item in value:
                    item_time = item.get('timestamp') if isinstance(item, dict) else datetime.now()
                    if isinstance(item_time, str):
                        try:
                            item_time = datetime.fromisoformat(item_time)
                        except:
                            item_time = datetime.now()
                    
                    if item_time >= cutoff_time:
                        filtered_data.append(item)
                
                window_data[key] = filtered_data
            else:
                window_data[key] = value
        
        return window_data

    def _calculate_throughput_kpis(self, window_data: Dict[str, Any], time_window_hours: int) -> Dict[str, KPIMetric]:
        """Calculate throughput-related KPIs."""
        kpis = {}
        
        # Overall throughput
        units_produced = window_data.get('units_produced', [])
        if units_produced:
            total_units = sum(item.get('quantity', 0) for item in units_produced if isinstance(item, dict))
            throughput_uph = total_units / time_window_hours if time_window_hours > 0 else 0
            
            kpis['overall_throughput_uph'] = KPIMetric(
                kpi_name='overall_throughput_uph',
                kpi_type=KPIType.THROUGHPUT,
                current_value=throughput_uph,
                target_value=self.kpi_definitions['overall_throughput_uph']['target'],
                unit='units/hour',
                timestamp=datetime.now(),
                confidence=0.9,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        # Line-specific throughput
        line_data = window_data.get('line_performance', {})
        for line_id, line_info in line_data.items():
            line_units = line_info.get('units_produced', 0)
            line_throughput = line_units / time_window_hours if time_window_hours > 0 else 0
            
            kpis[f'line_{line_id}_throughput'] = KPIMetric(
                kpi_name=f'line_{line_id}_throughput',
                kpi_type=KPIType.THROUGHPUT,
                current_value=line_throughput,
                target_value=100.0,  # Default target
                unit='units/hour',
                timestamp=datetime.now(),
                confidence=0.8,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        return kpis

    def _calculate_efficiency_kpis(self, window_data: Dict[str, Any], time_window_hours: int) -> Dict[str, KPIMetric]:
        """Calculate efficiency-related KPIs."""
        kpis = {}
        
        # Overall equipment efficiency (OEE)
        availability = window_data.get('availability', 95.0)
        performance = window_data.get('performance_rate', 90.0)
        quality = window_data.get('quality_rate', 98.0)
        
        oee = (availability * performance * quality) / 10000  # Convert to percentage
        
        kpis['overall_equipment_efficiency'] = KPIMetric(
            kpi_name='overall_equipment_efficiency',
            kpi_type=KPIType.EFFICIENCY,
            current_value=oee,
            target_value=85.0,
            unit='percentage',
            timestamp=datetime.now(),
            confidence=0.95,
            trend=TrendDirection.STABLE,
            variance=0.0,
            percentile_rank=0.0
        )
        
        # Energy efficiency
        energy_data = window_data.get('energy_consumption', [])
        if energy_data and 'units_produced' in window_data:
            total_energy = sum(item.get('kwh', 0) for item in energy_data if isinstance(item, dict))
            total_units = sum(item.get('quantity', 0) for item in window_data['units_produced'] if isinstance(item, dict))
            
            energy_per_unit = total_energy / total_units if total_units > 0 else 0
            
            kpis['energy_efficiency'] = KPIMetric(
                kpi_name='energy_efficiency',
                kpi_type=KPIType.EFFICIENCY,
                current_value=energy_per_unit,
                target_value=2.5,  # kWh per unit target
                unit='kwh/unit',
                timestamp=datetime.now(),
                confidence=0.85,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        return kpis

    def _calculate_quality_kpis(self, window_data: Dict[str, Any], time_window_hours: int) -> Dict[str, KPIMetric]:
        """Calculate quality-related KPIs."""
        kpis = {}
        
        # First-pass yield
        quality_data = window_data.get('quality_results', [])
        if quality_data:
            passed_units = sum(1 for item in quality_data if isinstance(item, dict) and item.get('result') == 'pass')
            total_tested = len(quality_data)
            yield_rate = (passed_units / total_tested * 100) if total_tested > 0 else 0
            
            kpis['first_pass_yield'] = KPIMetric(
                kpi_name='first_pass_yield',
                kpi_type=KPIType.QUALITY,
                current_value=yield_rate,
                target_value=99.0,
                unit='percentage',
                timestamp=datetime.now(),
                confidence=0.9,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        # Defect rate
        defect_data = window_data.get('defects', [])
        if defect_data and 'units_produced' in window_data:
            total_defects = len(defect_data)
            total_units = sum(item.get('quantity', 0) for item in window_data['units_produced'] if isinstance(item, dict))
            defect_rate = (total_defects / total_units * 1000) if total_units > 0 else 0  # Defects per 1000 units
            
            kpis['defect_rate'] = KPIMetric(
                kpi_name='defect_rate',
                kpi_type=KPIType.QUALITY,
                current_value=defect_rate,
                target_value=10.0,  # 10 defects per 1000 units
                unit='defects/1000',
                timestamp=datetime.now(),
                confidence=0.8,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        return kpis

    def _calculate_utilization_kpis(self, window_data: Dict[str, Any], time_window_hours: int) -> Dict[str, KPIMetric]:
        """Calculate utilization-related KPIs."""
        kpis = {}
        
        # Resource utilization
        resource_data = window_data.get('resource_utilization', {})
        if resource_data:
            total_utilization = 0
            resource_count = 0
            
            for resource_id, utilization in resource_data.items():
                if isinstance(utilization, (int, float)):
                    total_utilization += utilization
                    resource_count += 1
            
            avg_utilization = (total_utilization / resource_count) if resource_count > 0 else 0
            
            kpis['average_resource_utilization'] = KPIMetric(
                kpi_name='average_resource_utilization',
                kpi_type=KPIType.UTILIZATION,
                current_value=avg_utilization,
                target_value=80.0,
                unit='percentage',
                timestamp=datetime.now(),
                confidence=0.9,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        # Capacity utilization
        capacity_data = window_data.get('capacity_usage', {})
        if capacity_data:
            used_capacity = capacity_data.get('used', 0)
            available_capacity = capacity_data.get('available', 100)
            capacity_utilization = (used_capacity / available_capacity * 100) if available_capacity > 0 else 0
            
            kpis['capacity_utilization'] = KPIMetric(
                kpi_name='capacity_utilization',
                kpi_type=KPIType.UTILIZATION,
                current_value=capacity_utilization,
                target_value=85.0,
                unit='percentage',
                timestamp=datetime.now(),
                confidence=0.85,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        return kpis

    def _calculate_cost_kpis(self, window_data: Dict[str, Any], time_window_hours: int) -> Dict[str, KPIMetric]:
        """Calculate cost-related KPIs."""
        kpis = {}
        
        # Cost per unit
        cost_data = window_data.get('costs', {})
        units_produced = window_data.get('units_produced', [])
        
        if cost_data and units_produced:
            total_cost = sum(cost_data.values()) if isinstance(cost_data, dict) else 0
            total_units = sum(item.get('quantity', 0) for item in units_produced if isinstance(item, dict))
            
            cost_per_unit = total_cost / total_units if total_units > 0 else 0
            
            kpis['cost_per_unit'] = KPIMetric(
                kpi_name='cost_per_unit',
                kpi_type=KPIType.COST,
                current_value=cost_per_unit,
                target_value=10.0,
                unit='currency',
                timestamp=datetime.now(),
                confidence=0.8,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        # Labor cost efficiency
        labor_cost = cost_data.get('labor', 0) if isinstance(cost_data, dict) else 0
        if labor_cost > 0 and units_produced:
            total_units = sum(item.get('quantity', 0) for item in units_produced if isinstance(item, dict))
            labor_cost_per_unit = labor_cost / total_units if total_units > 0 else 0
            
            kpis['labor_cost_per_unit'] = KPIMetric(
                kpi_name='labor_cost_per_unit',
                kpi_type=KPIType.COST,
                current_value=labor_cost_per_unit,
                target_value=3.0,
                unit='currency',
                timestamp=datetime.now(),
                confidence=0.85,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        return kpis

    def _calculate_delivery_kpis(self, window_data: Dict[str, Any], time_window_hours: int) -> Dict[str, KPIMetric]:
        """Calculate delivery-related KPIs."""
        kpis = {}
        
        # On-time delivery rate
        delivery_data = window_data.get('deliveries', [])
        if delivery_data:
            on_time_deliveries = sum(1 for item in delivery_data 
                                   if isinstance(item, dict) and item.get('on_time', False))
            total_deliveries = len(delivery_data)
            
            on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
            
            kpis['on_time_delivery_rate'] = KPIMetric(
                kpi_name='on_time_delivery_rate',
                kpi_type=KPIType.DELIVERY,
                current_value=on_time_rate,
                target_value=95.0,
                unit='percentage',
                timestamp=datetime.now(),
                confidence=0.9,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        # Lead time
        order_data = window_data.get('completed_orders', [])
        if order_data:
            lead_times = []
            for order in order_data:
                if isinstance(order, dict) and 'start_time' in order and 'end_time' in order:
                    start = order['start_time']
                    end = order['end_time']
                    if isinstance(start, str):
                        try:
                            start = datetime.fromisoformat(start)
                        except:
                            continue
                    if isinstance(end, str):
                        try:
                            end = datetime.fromisoformat(end)
                        except:
                            continue
                    
                    lead_time = (end - start).total_seconds() / 3600  # Hours
                    lead_times.append(lead_time)
            
            if lead_times:
                avg_lead_time = statistics.mean(lead_times)
                
                kpis['average_lead_time'] = KPIMetric(
                    kpi_name='average_lead_time',
                    kpi_type=KPIType.DELIVERY,
                    current_value=avg_lead_time,
                    target_value=48.0,  # 48 hours target
                    unit='hours',
                    timestamp=datetime.now(),
                    confidence=0.85,
                    trend=TrendDirection.STABLE,
                    variance=statistics.stdev(lead_times) if len(lead_times) > 1 else 0,
                    percentile_rank=0.0
                )
        
        return kpis

    def _calculate_maintenance_kpis(self, window_data: Dict[str, Any], time_window_hours: int) -> Dict[str, KPIMetric]:
        """Calculate maintenance-related KPIs."""
        kpis = {}
        
        # Equipment availability
        downtime_data = window_data.get('downtime_events', [])
        if downtime_data:
            total_downtime = sum(item.get('duration_hours', 0) 
                               for item in downtime_data if isinstance(item, dict))
            availability = ((time_window_hours - total_downtime) / time_window_hours * 100) if time_window_hours > 0 else 100
            
            kpis['equipment_availability'] = KPIMetric(
                kpi_name='equipment_availability',
                kpi_type=KPIType.MAINTENANCE,
                current_value=availability,
                target_value=95.0,
                unit='percentage',
                timestamp=datetime.now(),
                confidence=0.9,
                trend=TrendDirection.STABLE,
                variance=0.0,
                percentile_rank=0.0
            )
        
        # Mean time between failures (MTBF)
        failure_data = window_data.get('failure_events', [])
        if failure_data and len(failure_data) > 1:
            failure_times = []
            for failure in failure_data:
                if isinstance(failure, dict) and 'timestamp' in failure:
                    timestamp = failure['timestamp']
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp)
                        except:
                            continue
                    failure_times.append(timestamp)
            
            if len(failure_times) > 1:
                failure_times.sort()
                intervals = []
                for i in range(1, len(failure_times)):
                    interval = (failure_times[i] - failure_times[i-1]).total_seconds() / 3600
                    intervals.append(interval)
                
                mtbf = statistics.mean(intervals) if intervals else 0
                
                kpis['mean_time_between_failures'] = KPIMetric(
                    kpi_name='mean_time_between_failures',
                    kpi_type=KPIType.MAINTENANCE,
                    current_value=mtbf,
                    target_value=720.0,  # 30 days target
                    unit='hours',
                    timestamp=datetime.now(),
                    confidence=0.8,
                    trend=TrendDirection.STABLE,
                    variance=statistics.stdev(intervals) if len(intervals) > 1 else 0,
                    percentile_rank=0.0
                )
        
        return kpis

    def _update_kpi_history(self, kpis: Dict[str, KPIMetric]):
        """Update KPI historical data."""
        for kpi_name, kpi in kpis.items():
            self.kpi_history[kpi_name].append({
                'timestamp': kpi.timestamp,
                'value': kpi.current_value,
                'target': kpi.target_value,
                'confidence': kpi.confidence
            })

    def _enrich_kpis_with_analysis(self, kpis: Dict[str, KPIMetric]):
        """Enrich KPIs with trend analysis and percentile rankings."""
        for kpi_name, kpi in kpis.items():
            # Calculate trend
            if kpi_name in self.kpi_history:
                history = list(self.kpi_history[kpi_name])
                if len(history) >= 5:
                    kpi.trend = self._calculate_trend_direction(history)
                    kpi.variance = self._calculate_variance(history)
            
            # Calculate percentile rank
            if kpi.target_value:
                kpi.percentile_rank = self._calculate_percentile_rank(kpi.current_value, kpi.target_value)

    def _calculate_trend_direction(self, history: List[Dict[str, Any]]) -> TrendDirection:
        """Calculate trend direction from historical data."""
        if len(history) < 3:
            return TrendDirection.STABLE
        
        recent_values = [item['value'] for item in history[-5:]]
        
        # Calculate linear regression slope
        n = len(recent_values)
        x_values = list(range(n))
        
        if n < 2:
            return TrendDirection.STABLE
        
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(recent_values)
        
        numerator = sum((x_values[i] - x_mean) * (recent_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return TrendDirection.STABLE
        
        slope = numerator / denominator
        
        # Classify trend based on slope and variance
        variance = statistics.variance(recent_values) if n > 1 else 0
        relative_variance = (math.sqrt(variance) / abs(y_mean)) if y_mean != 0 else 0
        
        if relative_variance > 0.2:  # High variance indicates volatility
            return TrendDirection.VOLATILE
        elif slope > self.trend_detection_sensitivity:
            return TrendDirection.IMPROVING
        elif slope < -self.trend_detection_sensitivity:
            return TrendDirection.DEGRADING
        else:
            return TrendDirection.STABLE

    def _calculate_variance(self, history: List[Dict[str, Any]]) -> float:
        """Calculate variance from historical data."""
        if len(history) < 2:
            return 0.0
        
        values = [item['value'] for item in history]
        return statistics.variance(values)

    def _calculate_percentile_rank(self, current_value: float, target_value: float) -> float:
        """Calculate percentile rank relative to target."""
        if target_value == 0:
            return 50.0
        
        performance_ratio = current_value / target_value
        
        if performance_ratio >= 1.0:
            return min(100.0, 50.0 + (performance_ratio - 1.0) * 50.0)
        else:
            return max(0.0, 50.0 * performance_ratio)

    def _generate_metric_forecast(self,
                                metric_name: str,
                                values: List[float],
                                timestamps: List[datetime],
                                forecast_horizon_hours: int) -> Optional[PerformanceForecast]:
        """Generate forecast for a specific metric."""
        try:
            if len(values) < 10:
                return None
            
            # Simple moving average forecast with trend
            window_size = min(10, len(values) // 2)
            recent_values = values[-window_size:]
            
            # Calculate trend
            trend_slope = self._calculate_simple_trend(recent_values)
            
            # Generate forecast points
            forecast_points = forecast_horizon_hours // 4  # Every 4 hours
            predicted_values = []
            confidence_intervals = []
            
            last_value = values[-1]
            noise_level = statistics.stdev(recent_values) if len(recent_values) > 1 else abs(last_value) * 0.1
            
            for i in range(1, forecast_points + 1):
                predicted_value = last_value + (trend_slope * i * 4)  # 4-hour intervals
                
                # Add confidence intervals (simplified)
                margin = 1.96 * noise_level * math.sqrt(i)  # Expanding uncertainty
                confidence_intervals.append((predicted_value - margin, predicted_value + margin))
                predicted_values.append(predicted_value)
            
            # Calculate forecast accuracy (simplified)
            accuracy = max(0.3, 0.9 - (noise_level / abs(last_value)) if last_value != 0 else 0.5)
            
            return PerformanceForecast(
                metric_name=metric_name,
                forecast_horizon_hours=forecast_horizon_hours,
                predicted_values=predicted_values,
                confidence_intervals=confidence_intervals,
                forecast_accuracy=accuracy,
                trend_analysis={'slope': trend_slope, 'direction': 'increasing' if trend_slope > 0 else 'decreasing'},
                seasonal_patterns=[]  # Would be more complex in real implementation
            )
            
        except Exception as e:
            logging.warning(f"Forecast generation failed for {metric_name}: {e}")
            return None

    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calculate simple trend slope."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0

    def _identify_performance_gaps(self,
                                 current_metrics: Dict[str, KPIMetric],
                                 benchmarks: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Identify performance gaps compared to benchmarks."""
        gaps = {}
        
        for kpi_name, kpi in current_metrics.items():
            benchmark_key = f"industry_{kpi_name.replace('_kpi', '').replace('overall_', '')}"
            benchmark_value = benchmarks.get(benchmark_key)
            
            if benchmark_value and kpi.target_value:
                gap_to_target = ((kpi.target_value - kpi.current_value) / kpi.target_value) * 100
                gap_to_benchmark = ((benchmark_value - kpi.current_value) / benchmark_value) * 100
                
                if abs(gap_to_target) > 10 or abs(gap_to_benchmark) > 15:  # Significant gaps
                    gaps[kpi_name] = {
                        'current_value': kpi.current_value,
                        'target_value': kpi.target_value,
                        'benchmark_value': benchmark_value,
                        'gap_to_target_percent': gap_to_target,
                        'gap_to_benchmark_percent': gap_to_benchmark,
                        'priority': 'high' if abs(gap_to_target) > 20 else 'medium'
                    }
        
        return gaps

    def _identify_bottlenecks(self, current_metrics: Dict[str, KPIMetric]) -> List[BottleneckAnalysis]:
        """Identify system bottlenecks from current metrics."""
        bottlenecks = []
        
        # Identify resource utilization bottlenecks
        utilization_metrics = {name: kpi for name, kpi in current_metrics.items() 
                             if kpi.kpi_type == KPIType.UTILIZATION}
        
        for name, kpi in utilization_metrics.items():
            if kpi.current_value > 95:  # Over-utilized resource
                bottleneck = BottleneckAnalysis(
                    bottleneck_id=f"resource_{name}",
                    bottleneck_type=BottleneckType.RESOURCE_BOTTLENECK,
                    affected_processes=[name],
                    severity=min(1.0, (kpi.current_value - 80) / 20),
                    estimated_impact={'throughput': -10.0, 'efficiency': -5.0},
                    root_causes=['Resource over-utilization', 'Insufficient capacity'],
                    improvement_opportunities=[
                        {'action': 'Add capacity', 'impact': 'High'},
                        {'action': 'Optimize scheduling', 'impact': 'Medium'}
                    ]
                )
                bottlenecks.append(bottleneck)
        
        # Identify quality bottlenecks
        quality_metrics = {name: kpi for name, kpi in current_metrics.items() 
                          if kpi.kpi_type == KPIType.QUALITY}
        
        for name, kpi in quality_metrics.items():
            if kpi.target_value and kpi.current_value < kpi.target_value * 0.95:
                bottleneck = BottleneckAnalysis(
                    bottleneck_id=f"quality_{name}",
                    bottleneck_type=BottleneckType.QUALITY_BOTTLENECK,
                    affected_processes=[name],
                    severity=(kpi.target_value - kpi.current_value) / kpi.target_value,
                    estimated_impact={'quality': -5.0, 'cost': 3.0},
                    root_causes=['Process variation', 'Equipment calibration'],
                    improvement_opportunities=[
                        {'action': 'Process optimization', 'impact': 'High'},
                        {'action': 'Equipment maintenance', 'impact': 'Medium'}
                    ]
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks

    def _create_gap_opportunity(self,
                              metric_name: str,
                              gap_info: Dict[str, Any],
                              current_metrics: Dict[str, KPIMetric]) -> Optional[OptimizationOpportunity]:
        """Create optimization opportunity from performance gap."""
        kpi = current_metrics.get(metric_name)
        if not kpi:
            return None
        
        gap_percent = gap_info['gap_to_target_percent']
        potential_improvement = abs(gap_percent) / 100 * kpi.current_value
        
        return OptimizationOpportunity(
            opportunity_id=f"gap_{metric_name}_{int(time.time())}",
            category="Performance Gap",
            description=f"Close {abs(gap_percent):.1f}% gap in {metric_name}",
            potential_improvement={metric_name: potential_improvement},
            implementation_effort="medium",
            estimated_roi=potential_improvement * 2,  # Simplified ROI calculation
            priority_score=min(100, abs(gap_percent)),
            recommended_actions=[
                f"Analyze root causes of {metric_name} underperformance",
                f"Implement targeted improvements for {metric_name}",
                "Monitor progress towards target"
            ]
        )

    def _create_bottleneck_opportunities(self, bottleneck: BottleneckAnalysis) -> List[OptimizationOpportunity]:
        """Create optimization opportunities from bottleneck analysis."""
        opportunities = []
        
        for improvement in bottleneck.improvement_opportunities:
            opportunity = OptimizationOpportunity(
                opportunity_id=f"bottleneck_{bottleneck.bottleneck_id}_{int(time.time())}",
                category="Bottleneck Resolution",
                description=f"Address {bottleneck.bottleneck_type.value}: {improvement['action']}",
                potential_improvement=bottleneck.estimated_impact,
                implementation_effort=improvement['impact'].lower(),
                estimated_roi=sum(abs(v) for v in bottleneck.estimated_impact.values()) * 1.5,
                priority_score=bottleneck.severity * 100,
                recommended_actions=[
                    improvement['action'],
                    f"Monitor {bottleneck.bottleneck_type.value} metrics",
                    "Validate improvement effectiveness"
                ]
            )
            opportunities.append(opportunity)
        
        return opportunities

    def _identify_trend_opportunities(self, current_metrics: Dict[str, KPIMetric]) -> List[OptimizationOpportunity]:
        """Identify opportunities based on trend analysis."""
        opportunities = []
        
        degrading_metrics = [kpi for kpi in current_metrics.values() 
                           if kpi.trend == TrendDirection.DEGRADING]
        
        for kpi in degrading_metrics:
            opportunity = OptimizationOpportunity(
                opportunity_id=f"trend_{kpi.kpi_name}_{int(time.time())}",
                category="Trend Correction",
                description=f"Reverse degrading trend in {kpi.kpi_name}",
                potential_improvement={kpi.kpi_name: kpi.current_value * 0.1},  # 10% improvement
                implementation_effort="medium",
                estimated_roi=kpi.current_value * 0.2,
                priority_score=75,  # High priority for degrading trends
                recommended_actions=[
                    f"Investigate causes of {kpi.kpi_name} degradation",
                    "Implement corrective measures",
                    "Establish monitoring alerts"
                ]
            )
            opportunities.append(opportunity)
        
        return opportunities

    def _prioritize_opportunities(self, opportunities: List[OptimizationOpportunity]) -> List[OptimizationOpportunity]:
        """Score and prioritize optimization opportunities."""
        for opportunity in opportunities:
            # Calculate priority score based on multiple factors
            roi_score = min(50, opportunity.estimated_roi / 100)  # Normalize ROI
            
            effort_scores = {"low": 30, "medium": 20, "high": 10}
            effort_score = effort_scores.get(opportunity.implementation_effort, 15)
            
            impact_score = sum(abs(v) for v in opportunity.potential_improvement.values())
            impact_score = min(30, impact_score)
            
            opportunity.priority_score = roi_score + effort_score + impact_score
        
        # Sort by priority score (highest first)
        return sorted(opportunities, key=lambda o: o.priority_score, reverse=True)

    def _update_analytics_metrics(self, computation_time: float, success: bool, kpis_calculated: int):
        """Update analytics performance metrics."""
        self.performance_metrics['total_calculations'] += 1
        
        if success:
            self.performance_metrics['successful_calculations'] += 1
        
        self.performance_metrics['kpis_calculated'] += kpis_calculated
        
        # Update average computation time
        total_calcs = self.performance_metrics['total_calculations']
        current_avg = self.performance_metrics['avg_calculation_time_ms']
        self.performance_metrics['avg_calculation_time_ms'] = (
            (current_avg * (total_calcs - 1) + computation_time) / total_calcs
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics['kpis_in_history'] = sum(len(history) for history in self.kpi_history.values())
        metrics['trend_cache_size'] = len(self.trend_cache)
        return metrics

    def clear_analytics_history(self):
        """Clear analytics history to free memory."""
        self.kpi_history.clear()
        self.performance_data.clear()
        self.trend_cache.clear()
        self.bottleneck_cache.clear()
        logging.info("Analytics history cleared")

    def __str__(self) -> str:
        return f"AnalyticsEngine(target={self.performance_target_ms}ms, kpis_tracked={len(self.kpi_history)})"

    def __repr__(self) -> str:
        return self.__str__()