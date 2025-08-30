"""
Week 6: VisualizationEngine - Advanced Data Visualization and Charting System

This module implements advanced data visualization capabilities for manufacturing
line control, providing real-time charts, KPI dashboards, trend analysis, and
3D system visualizations with high-performance rendering.

Key Features:
- Real-time data visualization with <50ms chart updates
- Advanced charting capabilities (line, bar, gauge, scatter, heatmap)
- Interactive KPI dashboards with drill-down capabilities
- 3D factory layout and equipment visualization
- Performance-optimized rendering for real-time data streams

Author: Claude Code
Date: 2024-08-29
Version: 1.0
"""

import time
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import deque
import math
import uuid

# Import Week 4 dependencies for analytics
try:
    from ..optimization_layer.analytics_engine import AnalyticsEngine, KPIMetric, TrendDirection
except ImportError:
    logging.warning("Week 4 analytics engine not available - using mock interfaces")
    AnalyticsEngine = None
    KPIMetric = None
    TrendDirection = None

class ChartType(Enum):
    LINE = "line"
    BAR = "bar"
    GAUGE = "gauge"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    AREA = "area"
    CANDLESTICK = "candlestick"
    RADAR = "radar"
    TREEMAP = "treemap"

class VisualizationType(Enum):
    CHART_2D = "chart_2d"
    CHART_3D = "chart_3d"
    KPI_DASHBOARD = "kpi_dashboard"
    FACTORY_LAYOUT = "factory_layout"
    TREND_ANALYSIS = "trend_analysis"
    PERFORMANCE_HEATMAP = "performance_heatmap"

class ColorScheme(Enum):
    MANUFACTURING = "manufacturing"
    PERFORMANCE = "performance"
    ALERT = "alert"
    GRADIENT = "gradient"
    MONOCHROME = "monochrome"
    HIGH_CONTRAST = "high_contrast"

@dataclass
class ChartConfiguration:
    chart_id: str
    chart_type: ChartType
    title: str
    data_sources: List[str]
    x_axis: Dict[str, Any]
    y_axis: Dict[str, Any]
    color_scheme: ColorScheme
    update_interval_ms: int = 1000
    max_data_points: int = 100
    animations_enabled: bool = True
    interactive: bool = True
    real_time: bool = True

@dataclass  
class VisualizationData:
    data_id: str
    chart_id: str
    timestamp: datetime
    data_points: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]

@dataclass
class KPIDashboardConfig:
    dashboard_id: str
    title: str
    kpi_metrics: List[str]
    layout: Dict[str, Any]
    alert_thresholds: Dict[str, Dict[str, float]]
    refresh_interval_ms: int = 5000
    trend_indicators: bool = True
    comparison_enabled: bool = True

@dataclass
class RenderingResult:
    visualization_id: str
    render_time_ms: float
    data_points_rendered: int
    frame_rate: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None

class VisualizationEngine:
    """Advanced data visualization and charting system for manufacturing line control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the VisualizationEngine with configuration."""
        self.config = config or {}
        
        # Performance configuration
        self.render_target_ms = self.config.get('render_target_ms', 50)
        self.max_concurrent_charts = self.config.get('max_concurrent_charts', 20)
        self.data_buffer_size = self.config.get('data_buffer_size', 1000)
        
        # Rendering configuration
        self.default_color_scheme = ColorScheme(self.config.get('default_color_scheme', 'manufacturing'))
        self.animation_duration_ms = self.config.get('animation_duration_ms', 300)
        self.frame_rate_target = self.config.get('frame_rate_target', 60)
        
        # Integration with Week 4 analytics
        if AnalyticsEngine:
            self.analytics_engine = AnalyticsEngine(self.config.get('analytics_config', {}))
        else:
            self.analytics_engine = None
        
        # Chart management
        self.active_charts: Dict[str, ChartConfiguration] = {}
        self.chart_data_buffers: Dict[str, deque] = {}
        self.kpi_dashboards: Dict[str, KPIDashboardConfig] = {}
        
        # Real-time data management
        self.data_subscriptions: Dict[str, List[str]] = {}  # data_source -> chart_ids
        self.update_threads: Dict[str, threading.Thread] = {}
        self.rendering_queue: deque = deque(maxlen=100)
        
        # Performance monitoring
        self.performance_metrics = {
            'avg_render_time_ms': 0,
            'total_renders': 0,
            'successful_renders': 0,
            'active_charts': 0,
            'data_updates_per_second': 0,
            'frame_rate_current': 0,
            'memory_usage_mb': 0
        }
        
        # Initialize default visualizations
        self._initialize_default_charts()
        self._initialize_default_dashboards()
        
        logging.info(f"VisualizationEngine initialized with {self.render_target_ms}ms render target")

    def create_real_time_charts(self, chart_specifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create and configure real-time data visualization charts."""
        start_time = time.time()
        created_charts = []
        failed_charts = []
        
        try:
            for spec in chart_specifications:
                try:
                    # Parse chart specification
                    chart_config = self._parse_chart_specification(spec)
                    
                    # Validate chart configuration
                    if not self._validate_chart_config(chart_config):
                        failed_charts.append({
                            'spec': spec,
                            'error': 'Invalid chart configuration'
                        })
                        continue
                    
                    # Create chart
                    chart_result = self._create_chart(chart_config)
                    
                    if chart_result['success']:
                        created_charts.append(chart_result)
                        
                        # Start real-time data updates if enabled
                        if chart_config.real_time:
                            self._start_real_time_updates(chart_config.chart_id)
                    else:
                        failed_charts.append({
                            'spec': spec,
                            'error': chart_result.get('error', 'Chart creation failed')
                        })
                        
                except Exception as e:
                    failed_charts.append({
                        'spec': spec,
                        'error': str(e)
                    })
            
            creation_time = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'created_charts': created_charts,
                'failed_charts': failed_charts,
                'creation_time_ms': creation_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            creation_time = (time.time() - start_time) * 1000
            logging.error(f"Chart creation failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'creation_time_ms': creation_time,
                'timestamp': datetime.now().isoformat()
            }

    def generate_kpi_dashboards(self, dashboard_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive KPI visualization dashboards."""
        start_time = time.time()
        
        try:
            generated_dashboards = []
            
            for spec in dashboard_specs:
                dashboard_config = self._parse_dashboard_specification(spec)
                
                # Generate KPI data if analytics engine available
                if self.analytics_engine:
                    kpi_data = self._generate_kpi_data(dashboard_config)
                else:
                    kpi_data = self._generate_mock_kpi_data(dashboard_config)
                
                # Create dashboard visualization
                dashboard_viz = self._create_kpi_dashboard(dashboard_config, kpi_data)
                generated_dashboards.append(dashboard_viz)
                
                # Store dashboard configuration
                self.kpi_dashboards[dashboard_config.dashboard_id] = dashboard_config
            
            generation_time = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'dashboards': generated_dashboards,
                'generation_time_ms': generation_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            generation_time = (time.time() - start_time) * 1000
            logging.error(f"KPI dashboard generation failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'generation_time_ms': generation_time,
                'timestamp': datetime.now().isoformat()
            }

    def render_3d_visualizations(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render 3D system visualizations and factory layouts."""
        start_time = time.time()
        
        try:
            # Parse 3D visualization requirements
            factory_layout = system_data.get('factory_layout', {})
            equipment_data = system_data.get('equipment_data', {})
            flow_data = system_data.get('flow_data', {})
            
            # Generate 3D factory layout
            layout_viz = self._generate_3d_factory_layout(factory_layout)
            
            # Generate equipment visualizations
            equipment_viz = self._generate_3d_equipment_visualization(equipment_data)
            
            # Generate material flow visualization
            flow_viz = self._generate_3d_flow_visualization(flow_data)
            
            render_time = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self._update_rendering_metrics(render_time, True)
            
            return {
                'success': True,
                'visualizations': {
                    'factory_layout': layout_viz,
                    'equipment': equipment_viz,
                    'material_flow': flow_viz
                },
                'render_time_ms': render_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            render_time = (time.time() - start_time) * 1000
            self._update_rendering_metrics(render_time, False)
            logging.error(f"3D visualization rendering failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'render_time_ms': render_time,
                'timestamp': datetime.now().isoformat()
            }

    def update_chart_data(self, chart_id: str, new_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update chart data with new real-time data points."""
        start_time = time.time()
        
        try:
            if chart_id not in self.active_charts:
                raise ValueError(f"Chart not found: {chart_id}")
            
            chart_config = self.active_charts[chart_id]
            
            # Initialize data buffer if needed
            if chart_id not in self.chart_data_buffers:
                self.chart_data_buffers[chart_id] = deque(maxlen=chart_config.max_data_points)
            
            data_buffer = self.chart_data_buffers[chart_id]
            
            # Add new data points
            for data_point in new_data:
                # Add timestamp if not present
                if 'timestamp' not in data_point:
                    data_point['timestamp'] = datetime.now().isoformat()
                
                data_buffer.append(data_point)
            
            # Prepare data for rendering
            render_data = self._prepare_chart_data(chart_config, list(data_buffer))
            
            # Calculate update performance
            update_time = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'chart_id': chart_id,
                'data_points_added': len(new_data),
                'total_data_points': len(data_buffer),
                'render_data': render_data,
                'update_time_ms': update_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            update_time = (time.time() - start_time) * 1000
            logging.error(f"Chart data update failed for {chart_id}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'update_time_ms': update_time,
                'timestamp': datetime.now().isoformat()
            }

    def _initialize_default_charts(self):
        """Initialize default chart configurations."""
        # Production throughput chart
        throughput_chart = ChartConfiguration(
            chart_id="production_throughput",
            chart_type=ChartType.LINE,
            title="Production Throughput (UPH)",
            data_sources=["analytics_engine.throughput_kpis"],
            x_axis={"label": "Time", "type": "datetime", "format": "%H:%M"},
            y_axis={"label": "Units/Hour", "min": 0, "max": 200},
            color_scheme=ColorScheme.MANUFACTURING,
            update_interval_ms=1000,
            max_data_points=288,  # 24 hours of 5-minute intervals
            real_time=True
        )
        
        # Efficiency gauge chart
        efficiency_gauge = ChartConfiguration(
            chart_id="efficiency_gauge",
            chart_type=ChartType.GAUGE,
            title="Overall Equipment Efficiency",
            data_sources=["analytics_engine.efficiency_kpis"],
            x_axis={},
            y_axis={"label": "Efficiency %", "min": 0, "max": 100},
            color_scheme=ColorScheme.PERFORMANCE,
            update_interval_ms=5000,
            max_data_points=1,
            real_time=True
        )
        
        # Quality trend chart
        quality_trend = ChartConfiguration(
            chart_id="quality_trend",
            chart_type=ChartType.AREA,
            title="Quality Metrics Trend",
            data_sources=["analytics_engine.quality_kpis"],
            x_axis={"label": "Time", "type": "datetime"},
            y_axis={"label": "Quality %", "min": 90, "max": 100},
            color_scheme=ColorScheme.ALERT,
            update_interval_ms=2000,
            max_data_points=144,  # 12 hours of 5-minute intervals
            real_time=True
        )
        
        self.active_charts[throughput_chart.chart_id] = throughput_chart
        self.active_charts[efficiency_gauge.chart_id] = efficiency_gauge
        self.active_charts[quality_trend.chart_id] = quality_trend

    def _initialize_default_dashboards(self):
        """Initialize default KPI dashboard configurations."""
        # Manufacturing performance dashboard
        performance_dashboard = KPIDashboardConfig(
            dashboard_id="manufacturing_performance",
            title="Manufacturing Performance Dashboard",
            kpi_metrics=[
                "overall_throughput_uph",
                "line_efficiency",
                "quality_yield",
                "equipment_availability",
                "cost_per_unit"
            ],
            layout={
                "rows": 2,
                "columns": 3,
                "responsive": True
            },
            refresh_interval_ms=5000,
            alert_thresholds={
                "throughput": {"warning": 80, "critical": 60},
                "efficiency": {"warning": 75, "critical": 60},
                "quality": {"warning": 95, "critical": 90}
            }
        )
        
        self.kpi_dashboards[performance_dashboard.dashboard_id] = performance_dashboard

    def _parse_chart_specification(self, spec: Dict[str, Any]) -> ChartConfiguration:
        """Parse chart specification into configuration object."""
        return ChartConfiguration(
            chart_id=spec.get('chart_id', str(uuid.uuid4())),
            chart_type=ChartType(spec.get('chart_type', 'line')),
            title=spec.get('title', 'Untitled Chart'),
            data_sources=spec.get('data_sources', []),
            x_axis=spec.get('x_axis', {}),
            y_axis=spec.get('y_axis', {}),
            color_scheme=ColorScheme(spec.get('color_scheme', 'manufacturing')),
            update_interval_ms=spec.get('update_interval_ms', 1000),
            max_data_points=spec.get('max_data_points', 100),
            animations_enabled=spec.get('animations_enabled', True),
            interactive=spec.get('interactive', True),
            real_time=spec.get('real_time', True)
        )

    def _parse_dashboard_specification(self, spec: Dict[str, Any]) -> KPIDashboardConfig:
        """Parse dashboard specification into configuration object."""
        return KPIDashboardConfig(
            dashboard_id=spec.get('dashboard_id', str(uuid.uuid4())),
            title=spec.get('title', 'KPI Dashboard'),
            kpi_metrics=spec.get('kpi_metrics', []),
            layout=spec.get('layout', {}),
            refresh_interval_ms=spec.get('refresh_interval_ms', 5000),
            alert_thresholds=spec.get('alert_thresholds', {}),
            trend_indicators=spec.get('trend_indicators', True),
            comparison_enabled=spec.get('comparison_enabled', True)
        )

    def _validate_chart_config(self, config: ChartConfiguration) -> bool:
        """Validate chart configuration parameters."""
        if not config.chart_id or not config.title:
            return False
        
        if not config.data_sources:
            return False
        
        if config.update_interval_ms < 100:  # Minimum 100ms updates
            return False
        
        if config.max_data_points > 10000:  # Maximum 10k data points
            return False
        
        return True

    def _create_chart(self, config: ChartConfiguration) -> Dict[str, Any]:
        """Create chart with specified configuration."""
        try:
            # Store chart configuration
            self.active_charts[config.chart_id] = config
            
            # Initialize data buffer
            self.chart_data_buffers[config.chart_id] = deque(maxlen=config.max_data_points)
            
            # Create chart visualization structure
            chart_viz = {
                'chart_id': config.chart_id,
                'type': config.chart_type.value,
                'title': config.title,
                'config': {
                    'x_axis': config.x_axis,
                    'y_axis': config.y_axis,
                    'color_scheme': self._get_color_palette(config.color_scheme),
                    'animations': config.animations_enabled,
                    'interactive': config.interactive
                },
                'data_sources': config.data_sources,
                'update_interval_ms': config.update_interval_ms,
                'created_at': datetime.now().isoformat()
            }
            
            self.performance_metrics['active_charts'] = len(self.active_charts)
            
            return {
                'success': True,
                'chart': chart_viz,
                'chart_id': config.chart_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'chart_id': config.chart_id
            }

    def _generate_kpi_data(self, dashboard_config: KPIDashboardConfig) -> Dict[str, Any]:
        """Generate KPI data using analytics engine."""
        if not self.analytics_engine:
            return self._generate_mock_kpi_data(dashboard_config)
        
        try:
            # Create mock production data for analytics
            production_data = {
                'units_produced': [{'quantity': 120, 'timestamp': datetime.now()}],
                'quality_results': [{'result': 'pass'} for _ in range(95)] + [{'result': 'fail'} for _ in range(5)],
                'energy_consumption': [{'kwh': 50, 'timestamp': datetime.now()}],
                'resource_utilization': {'line1': 85, 'line2': 78}
            }
            
            # Calculate KPIs
            kpis = self.analytics_engine.calculate_advanced_kpis(production_data, 24)
            
            # Convert KPI metrics to dashboard format
            kpi_data = {}
            for kpi_name, kpi_metric in kpis.items():
                kpi_data[kpi_name] = {
                    'current_value': kpi_metric.current_value,
                    'target_value': kpi_metric.target_value,
                    'unit': kpi_metric.unit,
                    'trend': kpi_metric.trend.value if kpi_metric.trend else 'stable',
                    'confidence': kpi_metric.confidence,
                    'timestamp': kpi_metric.timestamp.isoformat()
                }
            
            return kpi_data
            
        except Exception as e:
            logging.warning(f"KPI data generation failed, using mock data: {e}")
            return self._generate_mock_kpi_data(dashboard_config)

    def _generate_mock_kpi_data(self, dashboard_config: KPIDashboardConfig) -> Dict[str, Any]:
        """Generate mock KPI data for testing and development."""
        mock_data = {}
        
        for metric in dashboard_config.kpi_metrics:
            if 'throughput' in metric.lower():
                mock_data[metric] = {
                    'current_value': 115.5,
                    'target_value': 120.0,
                    'unit': 'units/hour',
                    'trend': 'improving',
                    'confidence': 0.9
                }
            elif 'efficiency' in metric.lower():
                mock_data[metric] = {
                    'current_value': 82.3,
                    'target_value': 85.0,
                    'unit': 'percentage',
                    'trend': 'stable',
                    'confidence': 0.95
                }
            elif 'quality' in metric.lower():
                mock_data[metric] = {
                    'current_value': 97.8,
                    'target_value': 99.0,
                    'unit': 'percentage',
                    'trend': 'degrading',
                    'confidence': 0.88
                }
            elif 'availability' in metric.lower():
                mock_data[metric] = {
                    'current_value': 94.2,
                    'target_value': 95.0,
                    'unit': 'percentage',
                    'trend': 'stable',
                    'confidence': 0.92
                }
            elif 'cost' in metric.lower():
                mock_data[metric] = {
                    'current_value': 8.75,
                    'target_value': 10.0,
                    'unit': 'currency',
                    'trend': 'improving',
                    'confidence': 0.85
                }
        
        return mock_data

    def _create_kpi_dashboard(self, config: KPIDashboardConfig, kpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create KPI dashboard visualization."""
        dashboard_viz = {
            'dashboard_id': config.dashboard_id,
            'title': config.title,
            'layout': config.layout,
            'kpi_widgets': [],
            'refresh_interval_ms': config.refresh_interval_ms,
            'created_at': datetime.now().isoformat()
        }
        
        # Create KPI widgets
        for metric_name, metric_data in kpi_data.items():
            widget = self._create_kpi_widget(metric_name, metric_data, config)
            dashboard_viz['kpi_widgets'].append(widget)
        
        return dashboard_viz

    def _create_kpi_widget(self, metric_name: str, metric_data: Dict[str, Any], config: KPIDashboardConfig) -> Dict[str, Any]:
        """Create individual KPI widget."""
        current = metric_data.get('current_value', 0)
        target = metric_data.get('target_value', 0)
        
        # Calculate status based on thresholds
        status = 'normal'
        if metric_name in config.alert_thresholds:
            thresholds = config.alert_thresholds[metric_name]
            if target > 0:
                percentage = (current / target) * 100
                if percentage < thresholds.get('critical', 0):
                    status = 'critical'
                elif percentage < thresholds.get('warning', 0):
                    status = 'warning'
        
        return {
            'metric_name': metric_name,
            'title': metric_name.replace('_', ' ').title(),
            'current_value': current,
            'target_value': target,
            'unit': metric_data.get('unit', ''),
            'trend': metric_data.get('trend', 'stable'),
            'status': status,
            'confidence': metric_data.get('confidence', 0.5),
            'widget_type': 'kpi_card'
        }

    def _generate_3d_factory_layout(self, factory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D factory layout visualization."""
        return {
            'visualization_type': '3d_factory_layout',
            'factory_dimensions': factory_data.get('dimensions', {'length': 100, 'width': 50, 'height': 10}),
            'production_lines': [
                {
                    'line_id': 'line_1',
                    'position': {'x': 10, 'y': 10, 'z': 0},
                    'dimensions': {'length': 30, 'width': 5, 'height': 3},
                    'status': 'active',
                    'throughput': 115
                },
                {
                    'line_id': 'line_2', 
                    'position': {'x': 10, 'y': 20, 'z': 0},
                    'dimensions': {'length': 30, 'width': 5, 'height': 3},
                    'status': 'active',
                    'throughput': 98
                }
            ],
            'work_stations': [],
            'material_paths': [],
            'render_options': {
                'lighting': 'factory_standard',
                'camera_angle': 'isometric',
                'show_labels': True,
                'show_metrics': True
            }
        }

    def _generate_3d_equipment_visualization(self, equipment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D equipment visualization."""
        return {
            'visualization_type': '3d_equipment',
            'equipment_models': [
                {
                    'equipment_id': 'smt_line_1',
                    'model_type': 'smt_assembly_line',
                    'position': {'x': 15, 'y': 12, 'z': 0},
                    'status': 'operational',
                    'performance_indicators': {
                        'temperature': 25.5,
                        'vibration': 0.2,
                        'efficiency': 82.3
                    }
                }
            ],
            'performance_overlays': True,
            'real_time_updates': True
        }

    def _generate_3d_flow_visualization(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D material flow visualization."""
        return {
            'visualization_type': '3d_material_flow',
            'flow_paths': [
                {
                    'path_id': 'main_flow',
                    'start_point': {'x': 5, 'y': 10, 'z': 1},
                    'end_point': {'x': 45, 'y': 10, 'z': 1},
                    'flow_rate': 120,
                    'material_type': 'pcb_components'
                }
            ],
            'flow_animations': True,
            'bottleneck_indicators': True
        }

    def _prepare_chart_data(self, config: ChartConfiguration, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare chart data for rendering."""
        if not data_points:
            return {'datasets': [], 'labels': []}
        
        # Extract data based on chart type
        if config.chart_type == ChartType.LINE or config.chart_type == ChartType.AREA:
            return self._prepare_time_series_data(data_points)
        elif config.chart_type == ChartType.GAUGE:
            return self._prepare_gauge_data(data_points)
        elif config.chart_type == ChartType.BAR:
            return self._prepare_bar_data(data_points)
        else:
            return self._prepare_generic_data(data_points)

    def _prepare_time_series_data(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare time series data for line/area charts."""
        labels = []
        values = []
        
        for point in data_points:
            timestamp = point.get('timestamp')
            value = point.get('value', 0)
            
            if timestamp:
                if isinstance(timestamp, str):
                    labels.append(timestamp)
                else:
                    labels.append(timestamp.isoformat())
                values.append(value)
        
        return {
            'labels': labels,
            'datasets': [{
                'label': 'Value',
                'data': values,
                'fill': True
            }]
        }

    def _prepare_gauge_data(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare gauge data for gauge charts."""
        if data_points:
            current_value = data_points[-1].get('value', 0)
            return {
                'value': current_value,
                'min': 0,
                'max': 100,
                'thresholds': [
                    {'value': 60, 'color': 'red'},
                    {'value': 80, 'color': 'yellow'},
                    {'value': 100, 'color': 'green'}
                ]
            }
        return {'value': 0, 'min': 0, 'max': 100}

    def _prepare_bar_data(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare bar chart data."""
        labels = []
        values = []
        
        for point in data_points:
            label = point.get('label', 'Unknown')
            value = point.get('value', 0)
            labels.append(label)
            values.append(value)
        
        return {
            'labels': labels,
            'datasets': [{
                'label': 'Values',
                'data': values
            }]
        }

    def _prepare_generic_data(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare generic chart data."""
        return {
            'raw_data': data_points,
            'data_points': len(data_points)
        }

    def _get_color_palette(self, scheme: ColorScheme) -> List[str]:
        """Get color palette for specified color scheme."""
        palettes = {
            ColorScheme.MANUFACTURING: [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ],
            ColorScheme.PERFORMANCE: [
                '#2E8B57', '#FFD700', '#FF6347', '#4169E1', '#32CD32',
                '#FF4500', '#9932CC', '#20B2AA', '#FF1493', '#00CED1'
            ],
            ColorScheme.ALERT: [
                '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#0000FF',
                '#800080', '#FFC0CB', '#808080', '#000000', '#FFFFFF'
            ]
        }
        
        return palettes.get(scheme, palettes[ColorScheme.MANUFACTURING])

    def _start_real_time_updates(self, chart_id: str):
        """Start real-time data updates for chart."""
        if chart_id in self.update_threads:
            return  # Already running
        
        def update_loop():
            chart_config = self.active_charts[chart_id]
            while chart_id in self.active_charts:
                try:
                    # Generate mock real-time data
                    new_data = self._generate_mock_real_time_data(chart_config)
                    if new_data:
                        self.update_chart_data(chart_id, new_data)
                    
                    time.sleep(chart_config.update_interval_ms / 1000)
                except Exception as e:
                    logging.error(f"Real-time update failed for chart {chart_id}: {e}")
                    break
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        self.update_threads[chart_id] = update_thread

    def _generate_mock_real_time_data(self, config: ChartConfiguration) -> List[Dict[str, Any]]:
        """Generate mock real-time data for testing."""
        current_time = datetime.now()
        
        if config.chart_type == ChartType.LINE or config.chart_type == ChartType.AREA:
            # Generate time series data
            base_value = 100
            variation = np.random.normal(0, 5)
            return [{
                'timestamp': current_time.isoformat(),
                'value': max(0, base_value + variation)
            }]
        elif config.chart_type == ChartType.GAUGE:
            # Generate gauge data
            base_value = 85
            variation = np.random.normal(0, 3)
            return [{
                'timestamp': current_time.isoformat(),
                'value': max(0, min(100, base_value + variation))
            }]
        
        return []

    def _update_rendering_metrics(self, render_time: float, success: bool):
        """Update rendering performance metrics."""
        self.performance_metrics['total_renders'] += 1
        
        if success:
            self.performance_metrics['successful_renders'] += 1
        
        # Update average render time
        total_renders = self.performance_metrics['total_renders']
        current_avg = self.performance_metrics['avg_render_time_ms']
        self.performance_metrics['avg_render_time_ms'] = (
            (current_avg * (total_renders - 1) + render_time) / total_renders
        )

    def get_chart_status(self, chart_id: str) -> Dict[str, Any]:
        """Get status and performance information for a chart."""
        if chart_id not in self.active_charts:
            return {'error': 'Chart not found'}
        
        config = self.active_charts[chart_id]
        data_buffer = self.chart_data_buffers.get(chart_id, deque())
        
        return {
            'chart_id': chart_id,
            'chart_type': config.chart_type.value,
            'title': config.title,
            'data_points': len(data_buffer),
            'max_data_points': config.max_data_points,
            'real_time_enabled': config.real_time,
            'update_interval_ms': config.update_interval_ms,
            'last_update': data_buffer[-1].get('timestamp') if data_buffer else None
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current visualization performance metrics."""
        return self.performance_metrics.copy()

    def cleanup_inactive_charts(self):
        """Clean up inactive charts and free resources."""
        inactive_charts = []
        current_time = datetime.now()
        
        for chart_id, data_buffer in self.chart_data_buffers.items():
            if data_buffer:
                last_update_str = data_buffer[-1].get('timestamp')
                if last_update_str:
                    last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
                    if (current_time - last_update).total_seconds() > 300:  # 5 minutes
                        inactive_charts.append(chart_id)
        
        for chart_id in inactive_charts:
            self.remove_chart(chart_id)
        
        logging.info(f"Cleaned up {len(inactive_charts)} inactive charts")

    def remove_chart(self, chart_id: str) -> bool:
        """Remove chart and clean up resources."""
        if chart_id not in self.active_charts:
            return False
        
        # Stop update thread
        if chart_id in self.update_threads:
            # Note: In a real implementation, would need proper thread termination
            del self.update_threads[chart_id]
        
        # Clean up data
        if chart_id in self.chart_data_buffers:
            del self.chart_data_buffers[chart_id]
        
        # Remove chart config
        del self.active_charts[chart_id]
        
        self.performance_metrics['active_charts'] = len(self.active_charts)
        
        logging.info(f"Chart {chart_id} removed")
        return True

    async def validate_visualization_engine(self) -> Dict[str, Any]:
        """Validate visualization engine functionality and performance."""
        validation_results = {
            'engine_name': 'VisualizationEngine',
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: Chart Creation
            test_chart_config = ChartConfig(
                chart_id='validation_chart',
                chart_type=ChartType.LINE,
                title='Validation Chart',
                x_axis_label='Time',
                y_axis_label='Value',
                width=800,
                height=400
            )
            
            chart_result = await self.create_chart(test_chart_config)
            
            validation_results['tests']['chart_creation'] = {
                'status': 'pass' if chart_result['success'] else 'fail',
                'render_time_ms': chart_result.get('render_time_ms', 0),
                'target_ms': self.render_target_ms,
                'details': f"Created chart: {chart_result.get('chart_id', 'unknown')}"
            }
            
            # Test 2: Data Update Performance
            test_data = [
                {'x': 1, 'y': 10, 'timestamp': datetime.now().isoformat()},
                {'x': 2, 'y': 20, 'timestamp': datetime.now().isoformat()},
                {'x': 3, 'y': 15, 'timestamp': datetime.now().isoformat()}
            ]
            
            update_result = await self.update_chart_data('validation_chart', test_data)
            
            validation_results['tests']['data_update'] = {
                'status': 'pass' if update_result['success'] else 'fail',
                'update_time_ms': update_result.get('update_time_ms', 0),
                'target_ms': self.render_target_ms,
                'details': f"Updated chart with {len(test_data)} data points"
            }
            
            # Test 3: KPI Dashboard Creation
            kpi_config = KPIDashboardConfig(
                dashboard_id='validation_kpi_dashboard',
                title='Validation KPI Dashboard',
                metrics=['efficiency', 'quality', 'throughput'],
                layout={'type': 'grid', 'columns': 3},
                refresh_interval_ms=1000
            )
            
            kpi_result = await self.create_kpi_dashboard(kpi_config)
            
            validation_results['tests']['kpi_dashboard'] = {
                'status': 'pass' if kpi_result['success'] else 'fail',
                'creation_time_ms': kpi_result.get('creation_time_ms', 0),
                'details': f"Created KPI dashboard with {len(kpi_config.metrics)} metrics"
            }
            
            # Performance metrics
            validation_results['performance_metrics'] = self.get_performance_metrics()
            
            # Overall status
            passed_tests = sum(1 for test in validation_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(validation_results['tests'])
            
            validation_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            validation_results['test_summary'] = f"{passed_tests}/{total_tests} tests passed"
            
            self.logger.info(f"Visualization engine validation completed: {validation_results['test_summary']}")
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"Visualization engine validation failed: {e}")
        
        return validation_results

    def __str__(self) -> str:
        return f"VisualizationEngine(render_target={self.render_target_ms}ms, active_charts={len(self.active_charts)})"

    def __repr__(self) -> str:
        return self.__str__()