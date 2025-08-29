"""
Operations Dashboard Engine for Week 8: Deployment & Monitoring

This module implements comprehensive operations dashboard system for the manufacturing line
control system with real-time operations view, historical analytics, performance trends,
operational reports, and comprehensive visibility across all system layers.

Performance Target: <500ms dashboard rendering, <2 seconds analytics queries
Dashboard Features: Real-time operations view, historical analytics, performance trends, operational reports
Integration: All system layers for comprehensive visibility
"""

import time
import logging
import asyncio
import json
import os
import sys
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import concurrent.futures
import traceback
from pathlib import Path
import uuid
import statistics
import base64

# Analytics and visualization
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    ANALYTICS_AVAILABLE = True
except ImportError:
    np = None
    pd = None
    plt = None
    sns = None
    BytesIO = None
    ANALYTICS_AVAILABLE = False

# Web framework for dashboard
try:
    from flask import Flask, render_template, jsonify, request, send_file
    from flask_socketio import SocketIO, emit
    import jinja2
    WEB_AVAILABLE = True
except ImportError:
    Flask = None
    render_template = None
    jsonify = None
    request = None
    send_file = None
    SocketIO = None
    emit = None
    jinja2 = None
    WEB_AVAILABLE = False

# Database for analytics
try:
    import sqlite3
    import redis
    DATABASE_AVAILABLE = True
except ImportError:
    sqlite3 = None
    redis = None
    DATABASE_AVAILABLE = False

# Week 8 deployment layer integrations
try:
    from layers.deployment_layer.deployment_engine import DeploymentEngine
    from layers.deployment_layer.monitoring_engine import MonitoringEngine
    from layers.deployment_layer.alerting_engine import AlertingEngine
    from layers.deployment_layer.infrastructure_engine import InfrastructureEngine
except ImportError:
    DeploymentEngine = None
    MonitoringEngine = None
    AlertingEngine = None
    InfrastructureEngine = None

# Week 7 testing layer integrations
try:
    from layers.testing_layer.ci_engine import CIEngine
    from layers.testing_layer.quality_assurance_engine import QualityAssuranceEngine
    from layers.testing_layer.benchmarking_engine import BenchmarkingEngine
except ImportError:
    CIEngine = None
    QualityAssuranceEngine = None
    BenchmarkingEngine = None

# Week 6 UI layer integrations
try:
    from layers.ui_layer.visualization_engine import VisualizationEngine
    from layers.ui_layer.webui_engine import WebUIEngine
    from layers.ui_layer.mobile_interface_engine import MobileInterfaceEngine
except ImportError:
    VisualizationEngine = None
    WebUIEngine = None
    MobileInterfaceEngine = None

# Week 5 control layer integrations
try:
    from layers.line_layer.line_layer_engine import LineLayerEngine
    from layers.station_layer.station_layer_engine import StationLayerEngine
    from layers.optimization_layer.optimization_layer_engine import OptimizationLayerEngine
except ImportError:
    LineLayerEngine = None
    StationLayerEngine = None
    OptimizationLayerEngine = None

# Common imports
try:
    from common.interfaces.layer_interface import LayerInterface
    from common.interfaces.data_interface import DataInterface
    from common.interfaces.communication_interface import CommunicationInterface
except ImportError:
    LayerInterface = None
    DataInterface = None
    CommunicationInterface = None


class DashboardType(Enum):
    """Dashboard type enumeration"""
    SYSTEM_OVERVIEW = "system_overview"
    OPERATIONS_REAL_TIME = "operations_real_time"
    DEPLOYMENT_STATUS = "deployment_status"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    QUALITY_METRICS = "quality_metrics"
    INFRASTRUCTURE_HEALTH = "infrastructure_health"
    BUSINESS_METRICS = "business_metrics"
    CUSTOM_DASHBOARD = "custom_dashboard"


class WidgetType(Enum):
    """Widget type enumeration"""
    METRIC_CARD = "metric_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    MAP = "map"
    LOG_VIEWER = "log_viewer"
    ALERT_LIST = "alert_list"


class TimeRange(Enum):
    """Time range enumeration"""
    LAST_5_MINUTES = "5m"
    LAST_15_MINUTES = "15m"
    LAST_HOUR = "1h"
    LAST_4_HOURS = "4h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    CUSTOM = "custom"


class RefreshMode(Enum):
    """Dashboard refresh mode enumeration"""
    REAL_TIME = "real_time"
    AUTO_REFRESH = "auto_refresh"
    MANUAL = "manual"
    ON_DEMAND = "on_demand"


class ExportFormat(Enum):
    """Export format enumeration"""
    PDF = "pdf"
    PNG = "png"
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    title: str
    type: WidgetType
    data_source: str
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 30
    enabled: bool = True
    
    def __post_init__(self):
        if 'x' not in self.position or 'y' not in self.position:
            self.position.update({'x': 0, 'y': 0})
        if 'width' not in self.position or 'height' not in self.position:
            self.position.update({'width': 4, 'height': 4})


@dataclass
class Dashboard:
    """Dashboard configuration"""
    id: str
    name: str
    title: str
    type: DashboardType
    widgets: List[DashboardWidget]
    layout: Dict[str, Any]
    refresh_mode: RefreshMode = RefreshMode.AUTO_REFRESH
    refresh_interval: int = 30
    time_range: TimeRange = TimeRange.LAST_HOUR
    filters: Dict[str, Any] = None
    permissions: Dict[str, List[str]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.permissions is None:
            self.permissions = {'view': ['*'], 'edit': ['admin']}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class ReportDefinition:
    """Report definition"""
    id: str
    name: str
    description: str
    template: str
    data_sources: List[str]
    parameters: Dict[str, Any]
    schedule: Dict[str, Any] = None  # cron-like schedule
    output_format: ExportFormat = ExportFormat.PDF
    recipients: List[str] = None
    
    def __post_init__(self):
        if self.recipients is None:
            self.recipients = []


@dataclass
class AnalyticsQuery:
    """Analytics query definition"""
    id: str
    name: str
    query: str
    data_source: str
    parameters: Dict[str, Any] = None
    cache_ttl: int = 300
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class DataAggregator:
    """Data aggregation and processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_sources = {}
        self.aggregation_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def register_data_source(self, name: str, source_func: Callable):
        """Register data source"""
        self.data_sources[name] = source_func
        self.logger.info(f"Registered data source: {name}")
    
    async def get_data(self, source_name: str, params: Dict[str, Any] = None) -> Any:
        """Get data from source"""
        if source_name not in self.data_sources:
            raise ValueError(f"Data source {source_name} not found")
        
        # Check cache
        cache_key = f"{source_name}:{json.dumps(params or {}, sort_keys=True)}"
        cached_data = self.aggregation_cache.get(cache_key)
        
        if cached_data and (datetime.now() - cached_data['timestamp']).seconds < self.cache_ttl:
            return cached_data['data']
        
        # Fetch fresh data
        try:
            source_func = self.data_sources[source_name]
            data = await source_func(params or {})
            
            # Cache result
            self.aggregation_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data source {source_name} failed: {e}")
            raise
    
    def aggregate_time_series(self, data: List[Dict[str, Any]], 
                            time_field: str, value_field: str,
                            interval: str = "1h") -> List[Dict[str, Any]]:
        """Aggregate time series data"""
        if not ANALYTICS_AVAILABLE or not data:
            return data
        
        try:
            df = pd.DataFrame(data)
            df[time_field] = pd.to_datetime(df[time_field])
            df = df.set_index(time_field)
            
            # Resample based on interval
            if interval == "1m":
                resampled = df.resample('1T').mean()
            elif interval == "5m":
                resampled = df.resample('5T').mean()
            elif interval == "1h":
                resampled = df.resample('1H').mean()
            elif interval == "1d":
                resampled = df.resample('1D').mean()
            else:
                return data
            
            # Convert back to list of dicts
            result = []
            for timestamp, row in resampled.iterrows():
                result.append({
                    time_field: timestamp.isoformat(),
                    value_field: row[value_field] if not pd.isna(row[value_field]) else 0
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Time series aggregation failed: {e}")
            return data
    
    def calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate statistics for data"""
        if not data:
            return {}
        
        try:
            return {
                'count': len(data),
                'min': min(data),
                'max': max(data),
                'mean': statistics.mean(data),
                'median': statistics.median(data),
                'std': statistics.stdev(data) if len(data) > 1 else 0,
                'p95': self._percentile(data, 0.95),
                'p99': self._percentile(data, 0.99)
            }
        except Exception as e:
            self.logger.error(f"Statistics calculation failed: {e}")
            return {}
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        
        if index >= len(sorted_data):
            return sorted_data[-1]
        
        return sorted_data[index]


class ChartGenerator:
    """Chart generation for dashboard widgets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_line_chart(self, data: List[Dict[str, Any]], 
                           x_field: str, y_field: str,
                           title: str = "", width: int = 800, height: int = 400) -> str:
        """Generate line chart"""
        if not ANALYTICS_AVAILABLE:
            return ""
        
        try:
            plt.figure(figsize=(width/100, height/100))
            
            # Extract data
            x_data = [item[x_field] for item in data]
            y_data = [item[y_field] for item in data]
            
            # Convert timestamps if needed
            if x_data and isinstance(x_data[0], str):
                try:
                    x_data = [pd.to_datetime(x) for x in x_data]
                except:
                    pass
            
            plt.plot(x_data, y_data, linewidth=2)
            plt.title(title)
            plt.xlabel(x_field.replace('_', ' ').title())
            plt.ylabel(y_field.replace('_', ' ').title())
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Line chart generation failed: {e}")
            return ""
    
    def generate_bar_chart(self, data: List[Dict[str, Any]], 
                          x_field: str, y_field: str,
                          title: str = "", width: int = 800, height: int = 400) -> str:
        """Generate bar chart"""
        if not ANALYTICS_AVAILABLE:
            return ""
        
        try:
            plt.figure(figsize=(width/100, height/100))
            
            x_data = [item[x_field] for item in data]
            y_data = [item[y_field] for item in data]
            
            plt.bar(x_data, y_data)
            plt.title(title)
            plt.xlabel(x_field.replace('_', ' ').title())
            plt.ylabel(y_field.replace('_', ' ').title())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Bar chart generation failed: {e}")
            return ""
    
    def generate_pie_chart(self, data: List[Dict[str, Any]], 
                          label_field: str, value_field: str,
                          title: str = "", width: int = 600, height: int = 400) -> str:
        """Generate pie chart"""
        if not ANALYTICS_AVAILABLE:
            return ""
        
        try:
            plt.figure(figsize=(width/100, height/100))
            
            labels = [item[label_field] for item in data]
            values = [item[value_field] for item in data]
            
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(title)
            plt.axis('equal')
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Pie chart generation failed: {e}")
            return ""
    
    def generate_heatmap(self, data: List[List[float]], 
                        x_labels: List[str], y_labels: List[str],
                        title: str = "", width: int = 800, height: int = 600) -> str:
        """Generate heatmap"""
        if not ANALYTICS_AVAILABLE:
            return ""
        
        try:
            plt.figure(figsize=(width/100, height/100))
            
            sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, 
                       annot=True, cmap='viridis')
            plt.title(title)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Heatmap generation failed: {e}")
            return ""


class ReportGenerator:
    """Report generation and scheduling"""
    
    def __init__(self, data_aggregator: DataAggregator):
        self.logger = logging.getLogger(__name__)
        self.data_aggregator = data_aggregator
        self.report_definitions = {}
        self.report_cache = {}
    
    def register_report(self, report: ReportDefinition):
        """Register report definition"""
        self.report_definitions[report.id] = report
        self.logger.info(f"Registered report: {report.name}")
    
    async def generate_report(self, report_id: str, parameters: Dict[str, Any] = None) -> bytes:
        """Generate report"""
        if report_id not in self.report_definitions:
            raise ValueError(f"Report {report_id} not found")
        
        report = self.report_definitions[report_id]
        
        try:
            # Collect data from sources
            report_data = {}
            for source in report.data_sources:
                data = await self.data_aggregator.get_data(source, parameters or {})
                report_data[source] = data
            
            # Generate report based on format
            if report.output_format == ExportFormat.JSON:
                return self._generate_json_report(report, report_data)
            elif report.output_format == ExportFormat.CSV:
                return self._generate_csv_report(report, report_data)
            elif report.output_format == ExportFormat.PDF:
                return self._generate_pdf_report(report, report_data)
            else:
                raise ValueError(f"Unsupported report format: {report.output_format}")
            
        except Exception as e:
            self.logger.error(f"Report generation failed for {report_id}: {e}")
            raise
    
    def _generate_json_report(self, report: ReportDefinition, data: Dict[str, Any]) -> bytes:
        """Generate JSON report"""
        report_content = {
            'report_id': report.id,
            'report_name': report.name,
            'generated_at': datetime.now().isoformat(),
            'data': data
        }
        
        return json.dumps(report_content, indent=2).encode('utf-8')
    
    def _generate_csv_report(self, report: ReportDefinition, data: Dict[str, Any]) -> bytes:
        """Generate CSV report"""
        if not ANALYTICS_AVAILABLE:
            raise RuntimeError("Pandas not available for CSV generation")
        
        try:
            # Combine all data sources into single DataFrame
            combined_data = []
            for source, source_data in data.items():
                if isinstance(source_data, list):
                    for item in source_data:
                        if isinstance(item, dict):
                            item['source'] = source
                            combined_data.append(item)
            
            if combined_data:
                df = pd.DataFrame(combined_data)
                return df.to_csv(index=False).encode('utf-8')
            else:
                return b"No data available\n"
                
        except Exception as e:
            self.logger.error(f"CSV report generation failed: {e}")
            return str(e).encode('utf-8')
    
    def _generate_pdf_report(self, report: ReportDefinition, data: Dict[str, Any]) -> bytes:
        """Generate PDF report"""
        # This would require a PDF library like reportlab
        # For now, return a simple text representation
        
        report_content = f"""
Manufacturing Line Operations Report
{report.name}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Description: {report.description}

Data Sources:
"""
        
        for source, source_data in data.items():
            report_content += f"\n{source}:\n"
            if isinstance(source_data, list):
                report_content += f"  Records: {len(source_data)}\n"
            elif isinstance(source_data, dict):
                report_content += f"  Fields: {len(source_data)}\n"
            else:
                report_content += f"  Data: {str(source_data)[:100]}...\n"
        
        return report_content.encode('utf-8')


class WebDashboardServer:
    """Web server for dashboard interface"""
    
    def __init__(self, operations_engine, port: int = 5000):
        self.logger = logging.getLogger(__name__)
        self.operations_engine = operations_engine
        self.port = port
        self.app = None
        self.socketio = None
        self.server_thread = None
        
        if WEB_AVAILABLE:
            self._initialize_server()
    
    def _initialize_server(self):
        """Initialize Flask server"""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'manufacturing-line-dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard routes
        @self.app.route('/')
        def dashboard_index():
            return render_template('dashboard.html') if render_template else "Dashboard Interface"
        
        @self.app.route('/api/dashboards')
        def list_dashboards():
            dashboards = self.operations_engine.list_dashboards()
            return jsonify(dashboards)
        
        @self.app.route('/api/dashboards/<dashboard_id>')
        async def get_dashboard(dashboard_id):
            try:
                dashboard_data = await self.operations_engine.render_dashboard(dashboard_id)
                return jsonify(dashboard_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/api/widgets/<widget_id>/data')
        async def get_widget_data(widget_id):
            try:
                widget_data = await self.operations_engine.get_widget_data(widget_id)
                return jsonify(widget_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/api/reports/<report_id>/generate', methods=['POST'])
        async def generate_report(report_id):
            try:
                parameters = request.get_json() or {}
                report_data = await self.operations_engine.generate_report(report_id, parameters)
                return send_file(BytesIO(report_data), as_attachment=True, 
                               download_name=f"report_{report_id}.pdf")
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        # Real-time updates via WebSocket
        @self.socketio.on('subscribe_dashboard')
        def handle_subscribe(data):
            dashboard_id = data.get('dashboard_id')
            if dashboard_id:
                # Join room for dashboard updates
                from flask_socketio import join_room
                join_room(f"dashboard_{dashboard_id}")
                self.logger.info(f"Client subscribed to dashboard {dashboard_id}")
        
        @self.socketio.on('unsubscribe_dashboard')
        def handle_unsubscribe(data):
            dashboard_id = data.get('dashboard_id')
            if dashboard_id:
                from flask_socketio import leave_room
                leave_room(f"dashboard_{dashboard_id}")
    
    def start_server(self):
        """Start web server"""
        if not self.app:
            self.logger.warning("Flask not available, cannot start dashboard server")
            return
        
        def run_server():
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        self.logger.info(f"Dashboard server started on port {self.port}")
    
    def broadcast_dashboard_update(self, dashboard_id: str, data: Dict[str, Any]):
        """Broadcast dashboard update to connected clients"""
        if self.socketio:
            self.socketio.emit('dashboard_update', data, room=f"dashboard_{dashboard_id}")


class OperationsDashboardEngine:
    """
    Comprehensive operations dashboard engine with real-time operations view,
    historical analytics, performance trends, and operational reports.
    """
    
    def __init__(self, db_path: str = "operations.db", dashboard_port: int = 5000):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.dashboard_port = dashboard_port
        
        # Dashboard storage
        self.dashboards = {}
        self.widgets = {}
        self.widget_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            'dashboard_renders': deque(maxlen=100),
            'widget_updates': deque(maxlen=100),
            'query_times': deque(maxlen=100),
            'active_users': 0,
            'total_views': 0
        }
        
        # Initialize components
        self.data_aggregator = DataAggregator()
        self.chart_generator = ChartGenerator()
        self.report_generator = ReportGenerator(self.data_aggregator)
        self.web_server = WebDashboardServer(self, dashboard_port)
        
        # Initialize database
        self._initialize_database()
        
        # Integration references
        self.monitoring_engine = None
        self.alerting_engine = None
        self.deployment_engine = None
        self.infrastructure_engine = None
        self.ui_engine = None
        
        # Register default data sources
        self._register_default_data_sources()
        
        # Create default dashboards
        self._create_default_dashboards()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("OperationsDashboardEngine initialized successfully")
    
    def _initialize_database(self):
        """Initialize dashboard database"""
        if DATABASE_AVAILABLE and sqlite3:
            try:
                self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._create_tables()
                self.logger.info("Operations database initialized")
            except Exception as e:
                self.logger.error(f"Database initialization failed: {e}")
                self.connection = None
    
    def _create_tables(self):
        """Create database tables"""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        
        # Dashboards table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dashboards (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Dashboard views table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dashboard_id TEXT NOT NULL,
                user_id TEXT,
                viewed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                duration INTEGER
            )
        """)
        
        self.connection.commit()
    
    def set_integrations(self, monitoring_engine=None, alerting_engine=None, 
                        deployment_engine=None, infrastructure_engine=None, ui_engine=None):
        """Set integration references"""
        self.monitoring_engine = monitoring_engine
        self.alerting_engine = alerting_engine
        self.deployment_engine = deployment_engine
        self.infrastructure_engine = infrastructure_engine
        self.ui_engine = ui_engine
    
    def _register_default_data_sources(self):
        """Register default data sources"""
        # System metrics data source
        self.data_aggregator.register_data_source("system_metrics", self._get_system_metrics)
        
        # Deployment metrics data source
        self.data_aggregator.register_data_source("deployment_metrics", self._get_deployment_metrics)
        
        # Alert metrics data source
        self.data_aggregator.register_data_source("alert_metrics", self._get_alert_metrics)
        
        # Performance metrics data source
        self.data_aggregator.register_data_source("performance_metrics", self._get_performance_metrics)
        
        # Quality metrics data source
        self.data_aggregator.register_data_source("quality_metrics", self._get_quality_metrics)
        
        # Business metrics data source
        self.data_aggregator.register_data_source("business_metrics", self._get_business_metrics)
    
    async def _get_system_metrics(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get system metrics"""
        if not self.monitoring_engine:
            return []
        
        try:
            time_range = params.get('time_range', 'last_hour')
            metrics = await self.monitoring_engine.get_metrics([
                'cpu_usage_total',
                'memory_usage_percent',
                'disk_usage_percent',
                'network_bytes_total'
            ])
            
            return self._format_metrics_for_dashboard(metrics)
            
        except Exception as e:
            self.logger.error(f"System metrics retrieval failed: {e}")
            return []
    
    async def _get_deployment_metrics(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get deployment metrics"""
        if not self.deployment_engine:
            return []
        
        try:
            deployment_metrics = self.deployment_engine.get_deployment_metrics()
            
            return [
                {
                    'timestamp': datetime.now().isoformat(),
                    'total_deployments': deployment_metrics.get('total_deployments', 0),
                    'success_rate': deployment_metrics.get('success_rate', 0),
                    'avg_deployment_time': deployment_metrics.get('avg_deployment_time', 0),
                    'rollback_rate': deployment_metrics.get('rollback_rate', 0)
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Deployment metrics retrieval failed: {e}")
            return []
    
    async def _get_alert_metrics(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get alert metrics"""
        if not self.alerting_engine:
            return []
        
        try:
            alert_stats = self.alerting_engine.get_alert_statistics()
            
            return [
                {
                    'timestamp': datetime.now().isoformat(),
                    'total_alerts': alert_stats.get('total_alerts', 0),
                    'critical_alerts': alert_stats.get('by_severity', {}).get('critical', 0),
                    'active_alerts': alert_stats.get('by_status', {}).get('firing', 0),
                    'notifications_sent': alert_stats.get('notifications_sent', 0)
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Alert metrics retrieval failed: {e}")
            return []
    
    async def _get_performance_metrics(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        # Aggregate performance metrics from all engines
        performance_data = []
        
        try:
            current_time = datetime.now().isoformat()
            
            # Dashboard performance
            avg_render_time = (
                statistics.mean(self.performance_metrics['dashboard_renders'])
                if self.performance_metrics['dashboard_renders'] else 0
            )
            
            performance_data.append({
                'timestamp': current_time,
                'component': 'dashboard',
                'avg_render_time': avg_render_time,
                'total_views': self.performance_metrics['total_views'],
                'active_users': self.performance_metrics['active_users']
            })
            
            # Monitoring engine performance
            if self.monitoring_engine:
                monitoring_metrics = self.monitoring_engine.get_monitoring_metrics()
                performance_data.append({
                    'timestamp': current_time,
                    'component': 'monitoring',
                    'metrics_collected': monitoring_metrics.get('metrics_collected', 0),
                    'avg_collection_time': monitoring_metrics.get('avg_collection_time', 0)
                })
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Performance metrics retrieval failed: {e}")
            return []
    
    async def _get_quality_metrics(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get quality metrics"""
        # This would integrate with quality assurance systems
        try:
            return [
                {
                    'timestamp': datetime.now().isoformat(),
                    'test_pass_rate': 0.95,
                    'code_coverage': 0.85,
                    'defect_density': 0.02,
                    'quality_score': 8.5
                }
            ]
        except Exception as e:
            self.logger.error(f"Quality metrics retrieval failed: {e}")
            return []
    
    async def _get_business_metrics(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get business metrics"""
        # This would integrate with business systems
        try:
            return [
                {
                    'timestamp': datetime.now().isoformat(),
                    'production_volume': 1250,
                    'efficiency': 0.92,
                    'downtime_minutes': 15,
                    'cost_per_unit': 12.50
                }
            ]
        except Exception as e:
            self.logger.error(f"Business metrics retrieval failed: {e}")
            return []
    
    def _format_metrics_for_dashboard(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format metrics for dashboard display"""
        formatted_metrics = []
        
        for metric_name, metric_values in metrics.items():
            for value in metric_values:
                formatted_metrics.append({
                    'timestamp': value.timestamp.isoformat(),
                    'metric': metric_name,
                    'value': value.value,
                    'labels': value.labels
                })
        
        return formatted_metrics
    
    def _create_default_dashboards(self):
        """Create default dashboards"""
        # System Overview Dashboard
        system_dashboard = Dashboard(
            id="system_overview",
            name="system_overview",
            title="System Overview",
            type=DashboardType.SYSTEM_OVERVIEW,
            widgets=[
                DashboardWidget(
                    id="cpu_usage",
                    title="CPU Usage",
                    type=WidgetType.GAUGE,
                    data_source="system_metrics",
                    config={'metric': 'cpu_usage_total', 'unit': '%', 'max': 100},
                    position={'x': 0, 'y': 0, 'width': 3, 'height': 3}
                ),
                DashboardWidget(
                    id="memory_usage",
                    title="Memory Usage",
                    type=WidgetType.GAUGE,
                    data_source="system_metrics",
                    config={'metric': 'memory_usage_percent', 'unit': '%', 'max': 100},
                    position={'x': 3, 'y': 0, 'width': 3, 'height': 3}
                ),
                DashboardWidget(
                    id="system_metrics_chart",
                    title="System Metrics Timeline",
                    type=WidgetType.LINE_CHART,
                    data_source="system_metrics",
                    config={'metrics': ['cpu_usage_total', 'memory_usage_percent']},
                    position={'x': 0, 'y': 3, 'width': 6, 'height': 4}
                )
            ],
            layout={'columns': 12, 'rows': 12}
        )
        self.register_dashboard(system_dashboard)
        
        # Operations Real-time Dashboard
        operations_dashboard = Dashboard(
            id="operations_real_time",
            name="operations_real_time",
            title="Real-time Operations",
            type=DashboardType.OPERATIONS_REAL_TIME,
            widgets=[
                DashboardWidget(
                    id="active_alerts",
                    title="Active Alerts",
                    type=WidgetType.METRIC_CARD,
                    data_source="alert_metrics",
                    config={'metric': 'active_alerts'},
                    position={'x': 0, 'y': 0, 'width': 2, 'height': 2}
                ),
                DashboardWidget(
                    id="deployment_status",
                    title="Recent Deployments",
                    type=WidgetType.TABLE,
                    data_source="deployment_metrics",
                    config={'columns': ['timestamp', 'status', 'duration']},
                    position={'x': 2, 'y': 0, 'width': 4, 'height': 4}
                ),
                DashboardWidget(
                    id="alert_timeline",
                    title="Alert Timeline",
                    type=WidgetType.LINE_CHART,
                    data_source="alert_metrics",
                    config={'x_field': 'timestamp', 'y_field': 'total_alerts'},
                    position={'x': 0, 'y': 4, 'width': 6, 'height': 3}
                )
            ],
            layout={'columns': 12, 'rows': 12},
            refresh_mode=RefreshMode.REAL_TIME
        )
        self.register_dashboard(operations_dashboard)
        
        # Performance Analytics Dashboard
        performance_dashboard = Dashboard(
            id="performance_analytics",
            name="performance_analytics", 
            title="Performance Analytics",
            type=DashboardType.PERFORMANCE_ANALYTICS,
            widgets=[
                DashboardWidget(
                    id="performance_overview",
                    title="Performance Overview",
                    type=WidgetType.BAR_CHART,
                    data_source="performance_metrics",
                    config={'x_field': 'component', 'y_field': 'avg_render_time'},
                    position={'x': 0, 'y': 0, 'width': 6, 'height': 4}
                ),
                DashboardWidget(
                    id="quality_score",
                    title="Quality Score",
                    type=WidgetType.GAUGE,
                    data_source="quality_metrics",
                    config={'metric': 'quality_score', 'min': 0, 'max': 10},
                    position={'x': 6, 'y': 0, 'width': 3, 'height': 3}
                )
            ],
            layout={'columns': 12, 'rows': 12}
        )
        self.register_dashboard(performance_dashboard)
    
    def _start_background_tasks(self):
        """Start background tasks"""
        # Start web server
        self.web_server.start_server()
        
        # Start dashboard update task
        threading.Thread(target=self._dashboard_update_task, daemon=True).start()
        
        # Start cache cleanup task
        threading.Thread(target=self._cache_cleanup_task, daemon=True).start()
    
    def _dashboard_update_task(self):
        """Background task to update real-time dashboards"""
        while True:
            try:
                # Update real-time dashboards
                for dashboard_id, dashboard in self.dashboards.items():
                    if dashboard.refresh_mode == RefreshMode.REAL_TIME:
                        asyncio.create_task(self._update_dashboard(dashboard_id))
                
                time.sleep(5)  # Update every 5 seconds for real-time dashboards
                
            except Exception as e:
                self.logger.error(f"Dashboard update task error: {e}")
                time.sleep(5)
    
    async def _update_dashboard(self, dashboard_id: str):
        """Update dashboard data"""
        if dashboard_id not in self.dashboards:
            return
        
        dashboard = self.dashboards[dashboard_id]
        
        try:
            # Update each widget
            for widget in dashboard.widgets:
                widget_data = await self.get_widget_data(widget.id)
                
                # Broadcast update to web clients
                self.web_server.broadcast_dashboard_update(dashboard_id, {
                    'widget_id': widget.id,
                    'data': widget_data
                })
            
        except Exception as e:
            self.logger.error(f"Dashboard {dashboard_id} update failed: {e}")
    
    def _cache_cleanup_task(self):
        """Background task to clean up cached data"""
        while True:
            try:
                now = datetime.now()
                
                # Clean up widget cache
                expired_keys = []
                for cache_key, cache_data in self.widget_cache.items():
                    if (now - cache_data['timestamp']).seconds > 300:  # 5 minutes
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self.widget_cache[key]
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cache cleanup task error: {e}")
                time.sleep(300)
    
    def register_dashboard(self, dashboard: Dashboard):
        """Register dashboard"""
        self.dashboards[dashboard.id] = dashboard
        
        # Register widgets
        for widget in dashboard.widgets:
            self.widgets[widget.id] = widget
        
        # Store in database
        self._store_dashboard_in_db(dashboard)
        
        self.logger.info(f"Registered dashboard: {dashboard.name}")
    
    async def render_dashboard(self, dashboard_id: str, time_range: TimeRange = None) -> Dict[str, Any]:
        """Render dashboard"""
        start_time = time.time()
        
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.dashboards[dashboard_id]
        
        try:
            # Render widgets
            rendered_widgets = []
            for widget in dashboard.widgets:
                widget_data = await self.get_widget_data(widget.id, time_range)
                rendered_widgets.append({
                    'id': widget.id,
                    'title': widget.title,
                    'type': widget.type.value,
                    'config': widget.config,
                    'position': widget.position,
                    'data': widget_data
                })
            
            rendered_dashboard = {
                'id': dashboard.id,
                'name': dashboard.name,
                'title': dashboard.title,
                'type': dashboard.type.value,
                'widgets': rendered_widgets,
                'layout': dashboard.layout,
                'refresh_mode': dashboard.refresh_mode.value,
                'refresh_interval': dashboard.refresh_interval,
                'time_range': (time_range or dashboard.time_range).value,
                'generated_at': datetime.now().isoformat()
            }
            
            # Update performance metrics
            render_time = time.time() - start_time
            self.performance_metrics['dashboard_renders'].append(render_time)
            self.performance_metrics['total_views'] += 1
            
            # Log dashboard view
            self._log_dashboard_view(dashboard_id)
            
            return rendered_dashboard
            
        except Exception as e:
            self.logger.error(f"Dashboard rendering failed for {dashboard_id}: {e}")
            raise
    
    async def get_widget_data(self, widget_id: str, time_range: TimeRange = None) -> Dict[str, Any]:
        """Get widget data"""
        if widget_id not in self.widgets:
            raise ValueError(f"Widget {widget_id} not found")
        
        widget = self.widgets[widget_id]
        
        # Check cache
        cache_key = f"{widget_id}:{time_range.value if time_range else 'default'}"
        cached_data = self.widget_cache.get(cache_key)
        
        if cached_data and (datetime.now() - cached_data['timestamp']).seconds < 30:
            return cached_data['data']
        
        try:
            # Get data from source
            params = {
                'time_range': time_range.value if time_range else TimeRange.LAST_HOUR.value,
                **widget.config
            }
            
            raw_data = await self.data_aggregator.get_data(widget.data_source, params)
            
            # Format data based on widget type
            widget_data = await self._format_widget_data(widget, raw_data)
            
            # Cache result
            self.widget_cache[cache_key] = {
                'data': widget_data,
                'timestamp': datetime.now()
            }
            
            # Update performance metrics
            self.performance_metrics['widget_updates'].append(time.time())
            
            return widget_data
            
        except Exception as e:
            self.logger.error(f"Widget data retrieval failed for {widget_id}: {e}")
            return {'error': str(e)}
    
    async def _format_widget_data(self, widget: DashboardWidget, raw_data: Any) -> Dict[str, Any]:
        """Format widget data based on type"""
        try:
            if widget.type == WidgetType.METRIC_CARD:
                return self._format_metric_card_data(widget, raw_data)
            elif widget.type == WidgetType.LINE_CHART:
                return self._format_line_chart_data(widget, raw_data)
            elif widget.type == WidgetType.BAR_CHART:
                return self._format_bar_chart_data(widget, raw_data)
            elif widget.type == WidgetType.PIE_CHART:
                return self._format_pie_chart_data(widget, raw_data)
            elif widget.type == WidgetType.GAUGE:
                return self._format_gauge_data(widget, raw_data)
            elif widget.type == WidgetType.TABLE:
                return self._format_table_data(widget, raw_data)
            elif widget.type == WidgetType.HEATMAP:
                return self._format_heatmap_data(widget, raw_data)
            else:
                return {'raw_data': raw_data}
                
        except Exception as e:
            self.logger.error(f"Widget data formatting failed: {e}")
            return {'error': str(e)}
    
    def _format_metric_card_data(self, widget: DashboardWidget, data: Any) -> Dict[str, Any]:
        """Format metric card data"""
        if not data or not isinstance(data, list):
            return {'value': 0, 'status': 'no_data'}
        
        metric_name = widget.config.get('metric', 'value')
        
        # Get latest value
        latest_value = 0
        for item in reversed(data):
            if isinstance(item, dict) and metric_name in item:
                latest_value = item[metric_name]
                break
        
        # Determine status based on thresholds
        status = 'ok'
        if 'thresholds' in widget.config:
            thresholds = widget.config['thresholds']
            if latest_value >= thresholds.get('critical', float('inf')):
                status = 'critical'
            elif latest_value >= thresholds.get('warning', float('inf')):
                status = 'warning'
        
        return {
            'value': latest_value,
            'status': status,
            'unit': widget.config.get('unit', ''),
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_line_chart_data(self, widget: DashboardWidget, data: Any) -> Dict[str, Any]:
        """Format line chart data"""
        if not data or not isinstance(data, list):
            return {'chart_data': [], 'chart_image': ''}
        
        x_field = widget.config.get('x_field', 'timestamp')
        y_field = widget.config.get('y_field', 'value')
        
        # Generate chart image
        chart_image = self.chart_generator.generate_line_chart(
            data, x_field, y_field, widget.title
        )
        
        return {
            'chart_data': data,
            'chart_image': chart_image,
            'x_field': x_field,
            'y_field': y_field
        }
    
    def _format_bar_chart_data(self, widget: DashboardWidget, data: Any) -> Dict[str, Any]:
        """Format bar chart data"""
        if not data or not isinstance(data, list):
            return {'chart_data': [], 'chart_image': ''}
        
        x_field = widget.config.get('x_field', 'category')
        y_field = widget.config.get('y_field', 'value')
        
        # Generate chart image
        chart_image = self.chart_generator.generate_bar_chart(
            data, x_field, y_field, widget.title
        )
        
        return {
            'chart_data': data,
            'chart_image': chart_image,
            'x_field': x_field,
            'y_field': y_field
        }
    
    def _format_pie_chart_data(self, widget: DashboardWidget, data: Any) -> Dict[str, Any]:
        """Format pie chart data"""
        if not data or not isinstance(data, list):
            return {'chart_data': [], 'chart_image': ''}
        
        label_field = widget.config.get('label_field', 'label')
        value_field = widget.config.get('value_field', 'value')
        
        # Generate chart image
        chart_image = self.chart_generator.generate_pie_chart(
            data, label_field, value_field, widget.title
        )
        
        return {
            'chart_data': data,
            'chart_image': chart_image,
            'label_field': label_field,
            'value_field': value_field
        }
    
    def _format_gauge_data(self, widget: DashboardWidget, data: Any) -> Dict[str, Any]:
        """Format gauge data"""
        if not data or not isinstance(data, list):
            return {'value': 0, 'min': 0, 'max': 100}
        
        metric_name = widget.config.get('metric', 'value')
        
        # Get latest value
        latest_value = 0
        for item in reversed(data):
            if isinstance(item, dict) and metric_name in item:
                latest_value = item[metric_name]
                break
        
        return {
            'value': latest_value,
            'min': widget.config.get('min', 0),
            'max': widget.config.get('max', 100),
            'unit': widget.config.get('unit', '')
        }
    
    def _format_table_data(self, widget: DashboardWidget, data: Any) -> Dict[str, Any]:
        """Format table data"""
        if not data or not isinstance(data, list):
            return {'columns': [], 'rows': []}
        
        columns = widget.config.get('columns', [])
        if not columns and data:
            # Auto-detect columns from first item
            if isinstance(data[0], dict):
                columns = list(data[0].keys())
        
        return {
            'columns': columns,
            'rows': data[:100],  # Limit to 100 rows
            'total_rows': len(data)
        }
    
    def _format_heatmap_data(self, widget: DashboardWidget, data: Any) -> Dict[str, Any]:
        """Format heatmap data"""
        # This would require more complex data processing
        return {'heatmap_data': data}
    
    async def create_custom_dashboard(self, dashboard: Dashboard):
        """Create custom dashboard"""
        self.register_dashboard(dashboard)
        self.logger.info(f"Created custom dashboard: {dashboard.name}")
    
    async def generate_report(self, report_id: str, parameters: Dict[str, Any] = None) -> bytes:
        """Generate report"""
        return await self.report_generator.generate_report(report_id, parameters)
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List available dashboards"""
        dashboard_list = []
        
        for dashboard in self.dashboards.values():
            dashboard_list.append({
                'id': dashboard.id,
                'name': dashboard.name,
                'title': dashboard.title,
                'type': dashboard.type.value,
                'widget_count': len(dashboard.widgets),
                'refresh_mode': dashboard.refresh_mode.value,
                'created_at': dashboard.created_at.isoformat() if dashboard.created_at else None,
                'updated_at': dashboard.updated_at.isoformat() if dashboard.updated_at else None
            })
        
        return dashboard_list
    
    def _store_dashboard_in_db(self, dashboard: Dashboard):
        """Store dashboard in database"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dashboards (id, name, title, type, config)
                VALUES (?, ?, ?, ?, ?)
            """, (
                dashboard.id,
                dashboard.name,
                dashboard.title,
                dashboard.type.value,
                json.dumps(asdict(dashboard))
            ))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store dashboard in database: {e}")
    
    def _log_dashboard_view(self, dashboard_id: str, user_id: str = None):
        """Log dashboard view"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO dashboard_views (dashboard_id, user_id)
                VALUES (?, ?)
            """, (dashboard_id, user_id))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to log dashboard view: {e}")
    
    def get_dashboard_analytics(self) -> Dict[str, Any]:
        """Get dashboard analytics"""
        try:
            return {
                'total_dashboards': len(self.dashboards),
                'total_widgets': len(self.widgets),
                'avg_render_time': statistics.mean(self.performance_metrics['dashboard_renders']) if self.performance_metrics['dashboard_renders'] else 0,
                'total_views': self.performance_metrics['total_views'],
                'active_users': self.performance_metrics['active_users'],
                'cache_hit_rate': len(self.widget_cache) / max(len(self.widgets), 1)
            }
        except Exception as e:
            self.logger.error(f"Dashboard analytics failed: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close database connection
            if self.connection:
                self.connection.close()
            
            self.logger.info("OperationsDashboardEngine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"OperationsDashboardEngine cleanup error: {e}")


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create operations dashboard engine
        engine = OperationsDashboardEngine(dashboard_port=5001)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        try:
            # Get dashboard list
            dashboards = engine.list_dashboards()
            print(f"Available dashboards: {len(dashboards)}")
            
            for dashboard in dashboards:
                print(f"  - {dashboard['title']} ({dashboard['type']})")
            
            # Render system overview dashboard
            if dashboards:
                dashboard_data = await engine.render_dashboard("system_overview")
                print(f"System overview dashboard rendered with {len(dashboard_data['widgets'])} widgets")
            
            # Get dashboard analytics
            analytics = engine.get_dashboard_analytics()
            print(f"Dashboard analytics: {analytics}")
            
            # Create custom dashboard
            custom_dashboard = Dashboard(
                id="custom_test",
                name="custom_test",
                title="Custom Test Dashboard",
                type=DashboardType.CUSTOM_DASHBOARD,
                widgets=[
                    DashboardWidget(
                        id="test_metric",
                        title="Test Metric",
                        type=WidgetType.METRIC_CARD,
                        data_source="system_metrics",
                        config={'metric': 'test_value'},
                        position={'x': 0, 'y': 0, 'width': 2, 'height': 2}
                    )
                ],
                layout={'columns': 12, 'rows': 8}
            )
            
            await engine.create_custom_dashboard(custom_dashboard)
            print(f"Created custom dashboard: {custom_dashboard.title}")
            
        except Exception as e:
            print(f"Operations dashboard test failed: {e}")
        
        finally:
            await engine.cleanup()
    
    asyncio.run(main())