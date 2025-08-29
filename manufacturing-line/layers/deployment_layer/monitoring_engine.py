"""
Enhanced Monitoring Engine for Week 8: Deployment & Monitoring

This module implements comprehensive monitoring system for the manufacturing line
control system with real-time metrics collection, Prometheus-style metrics,
Grafana-style dashboards, time-series data, and performance monitoring.

Performance Target: <100ms metrics collection, <1 second dashboard updates
Monitoring Features: System metrics, application metrics, custom KPIs, trend analysis, real-time monitoring
Integration: Prometheus-style metrics, Grafana-style dashboards, time-series data
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
import psutil
import platform
import socket

# Time series and monitoring
try:
    import numpy as np
    import pandas as pd
    ANALYTICS_AVAILABLE = True
except ImportError:
    np = None
    pd = None
    ANALYTICS_AVAILABLE = False

# Web framework for metrics endpoint
try:
    from flask import Flask, jsonify, request, Response
    import websocket
    WEB_AVAILABLE = True
except ImportError:
    Flask = None
    jsonify = None
    request = None
    Response = None
    websocket = None
    WEB_AVAILABLE = False

# Database for metrics storage
try:
    import sqlite3
    import redis
    DATABASE_AVAILABLE = True
except ImportError:
    sqlite3 = None
    redis = None
    DATABASE_AVAILABLE = False

# Week 8 deployment layer integrations (forward references)
try:
    from layers.deployment_layer.deployment_engine import DeploymentEngine
    from layers.deployment_layer.alerting_engine import AlertingEngine
    from layers.deployment_layer.infrastructure_engine import InfrastructureEngine
except ImportError:
    DeploymentEngine = None
    AlertingEngine = None
    InfrastructureEngine = None

# Week 7 testing layer integrations
try:
    from layers.testing_layer.benchmarking_engine import BenchmarkingEngine
    from layers.testing_layer.quality_assurance_engine import QualityAssuranceEngine
except ImportError:
    BenchmarkingEngine = None
    QualityAssuranceEngine = None

# Week 6 UI layer integrations
try:
    from layers.ui_layer.visualization_engine import VisualizationEngine
    from layers.ui_layer.webui_engine import WebUIEngine
except ImportError:
    VisualizationEngine = None
    WebUIEngine = None

# Common imports
try:
    from common.interfaces.layer_interface import LayerInterface
    from common.interfaces.data_interface import DataInterface
    from common.interfaces.communication_interface import CommunicationInterface
except ImportError:
    LayerInterface = None
    DataInterface = None
    CommunicationInterface = None


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class MetricStatus(Enum):
    """Metric status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AggregationType(Enum):
    """Aggregation type enumeration"""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


class TimeInterval(Enum):
    """Time interval enumeration"""
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"


class DashboardType(Enum):
    """Dashboard type enumeration"""
    SYSTEM_OVERVIEW = "system_overview"
    APPLICATION_METRICS = "application_metrics"
    DEPLOYMENT_METRICS = "deployment_metrics"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    CUSTOM_DASHBOARD = "custom_dashboard"


@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: Dict[str, str] = None
    thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.thresholds is None:
            self.thresholds = {}


@dataclass
class MetricValue:
    """Metric value"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TimeSeriesData:
    """Time series data"""
    metric_name: str
    timestamps: List[datetime]
    values: List[float]
    labels: Dict[str, str] = None
    aggregation: AggregationType = AggregationType.AVERAGE
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    metric_name: str
    condition: str  # ">" | "<" | "==" | "!="
    threshold: float
    duration: int  # seconds
    severity: str = "warning"
    description: str = ""
    enabled: bool = True


@dataclass
class DashboardPanel:
    """Dashboard panel"""
    title: str
    type: str  # "graph", "singlestat", "table", "heatmap"
    metrics: List[str]
    time_range: TimeInterval = TimeInterval.HOUR
    aggregation: AggregationType = AggregationType.AVERAGE
    thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {}


@dataclass
class Dashboard:
    """Dashboard definition"""
    name: str
    title: str
    type: DashboardType
    panels: List[DashboardPanel]
    refresh_interval: int = 30  # seconds
    time_range: TimeInterval = TimeInterval.HOUR
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MetricsCollector:
    """Metrics collection system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_registry = {}
        self.metric_values = defaultdict(deque)
        self.collectors = {}
        self.collection_threads = {}
        self.collection_intervals = {}
        
        # Initialize system collectors
        self._initialize_system_collectors()
    
    def register_metric(self, definition: MetricDefinition):
        """Register metric definition"""
        self.metrics_registry[definition.name] = definition
        self.logger.info(f"Registered metric: {definition.name}")
    
    def register_collector(self, name: str, collector_func: Callable, interval: int = 60):
        """Register metric collector"""
        self.collectors[name] = collector_func
        self.collection_intervals[name] = interval
        
        # Start collection thread
        self._start_collector_thread(name)
        self.logger.info(f"Registered collector: {name} (interval: {interval}s)")
    
    def _start_collector_thread(self, name: str):
        """Start collector thread"""
        def collector_thread():
            while name in self.collectors:
                try:
                    collector_func = self.collectors[name]
                    metrics = collector_func()
                    
                    if isinstance(metrics, list):
                        for metric in metrics:
                            self._store_metric_value(metric)
                    elif isinstance(metrics, MetricValue):
                        self._store_metric_value(metrics)
                    
                    time.sleep(self.collection_intervals[name])
                    
                except Exception as e:
                    self.logger.error(f"Collector {name} error: {e}")
                    time.sleep(self.collection_intervals[name])
        
        thread = threading.Thread(target=collector_thread, daemon=True)
        thread.start()
        self.collection_threads[name] = thread
    
    def _store_metric_value(self, metric: MetricValue):
        """Store metric value"""
        metric_key = f"{metric.name}:{json.dumps(metric.labels, sort_keys=True)}"
        
        # Store with timestamp-based deque (keep last 1000 values)
        values = self.metric_values[metric_key]
        values.append(metric)
        
        if len(values) > 1000:
            values.popleft()
    
    def _initialize_system_collectors(self):
        """Initialize system metric collectors"""
        # CPU metrics
        self.register_collector("cpu_collector", self._collect_cpu_metrics, 15)
        
        # Memory metrics
        self.register_collector("memory_collector", self._collect_memory_metrics, 15)
        
        # Disk metrics
        self.register_collector("disk_collector", self._collect_disk_metrics, 30)
        
        # Network metrics
        self.register_collector("network_collector", self._collect_network_metrics, 30)
        
        # System metrics
        self.register_collector("system_collector", self._collect_system_metrics, 60)
    
    def _collect_cpu_metrics(self) -> List[MetricValue]:
        """Collect CPU metrics"""
        metrics = []
        now = datetime.now()
        
        # CPU usage per core
        cpu_percents = psutil.cpu_percent(percpu=True)
        for i, percent in enumerate(cpu_percents):
            metrics.append(MetricValue(
                name="cpu_usage_percent",
                value=percent,
                timestamp=now,
                labels={"core": str(i)}
            ))
        
        # Overall CPU usage
        metrics.append(MetricValue(
            name="cpu_usage_total",
            value=psutil.cpu_percent(),
            timestamp=now
        ))
        
        # Load averages
        if hasattr(os, 'getloadavg'):
            load1, load5, load15 = os.getloadavg()
            metrics.extend([
                MetricValue(name="load_average_1m", value=load1, timestamp=now),
                MetricValue(name="load_average_5m", value=load5, timestamp=now),
                MetricValue(name="load_average_15m", value=load15, timestamp=now)
            ])
        
        return metrics
    
    def _collect_memory_metrics(self) -> List[MetricValue]:
        """Collect memory metrics"""
        metrics = []
        now = datetime.now()
        
        # Virtual memory
        vmem = psutil.virtual_memory()
        metrics.extend([
            MetricValue(name="memory_total_bytes", value=vmem.total, timestamp=now),
            MetricValue(name="memory_available_bytes", value=vmem.available, timestamp=now),
            MetricValue(name="memory_used_bytes", value=vmem.used, timestamp=now),
            MetricValue(name="memory_usage_percent", value=vmem.percent, timestamp=now)
        ])
        
        # Swap memory
        swap = psutil.swap_memory()
        metrics.extend([
            MetricValue(name="swap_total_bytes", value=swap.total, timestamp=now),
            MetricValue(name="swap_used_bytes", value=swap.used, timestamp=now),
            MetricValue(name="swap_usage_percent", value=swap.percent, timestamp=now)
        ])
        
        return metrics
    
    def _collect_disk_metrics(self) -> List[MetricValue]:
        """Collect disk metrics"""
        metrics = []
        now = datetime.now()
        
        # Disk usage per partition
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                device = partition.device
                
                metrics.extend([
                    MetricValue(
                        name="disk_total_bytes",
                        value=usage.total,
                        timestamp=now,
                        labels={"device": device, "mountpoint": partition.mountpoint}
                    ),
                    MetricValue(
                        name="disk_used_bytes",
                        value=usage.used,
                        timestamp=now,
                        labels={"device": device, "mountpoint": partition.mountpoint}
                    ),
                    MetricValue(
                        name="disk_usage_percent",
                        value=(usage.used / usage.total) * 100,
                        timestamp=now,
                        labels={"device": device, "mountpoint": partition.mountpoint}
                    )
                ])
            except PermissionError:
                continue
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.extend([
                MetricValue(name="disk_read_bytes_total", value=disk_io.read_bytes, timestamp=now),
                MetricValue(name="disk_write_bytes_total", value=disk_io.write_bytes, timestamp=now),
                MetricValue(name="disk_read_ops_total", value=disk_io.read_count, timestamp=now),
                MetricValue(name="disk_write_ops_total", value=disk_io.write_count, timestamp=now)
            ])
        
        return metrics
    
    def _collect_network_metrics(self) -> List[MetricValue]:
        """Collect network metrics"""
        metrics = []
        now = datetime.now()
        
        # Network I/O per interface
        net_io = psutil.net_io_counters(pernic=True)
        for interface, stats in net_io.items():
            metrics.extend([
                MetricValue(
                    name="network_bytes_sent_total",
                    value=stats.bytes_sent,
                    timestamp=now,
                    labels={"interface": interface}
                ),
                MetricValue(
                    name="network_bytes_recv_total",
                    value=stats.bytes_recv,
                    timestamp=now,
                    labels={"interface": interface}
                ),
                MetricValue(
                    name="network_packets_sent_total",
                    value=stats.packets_sent,
                    timestamp=now,
                    labels={"interface": interface}
                ),
                MetricValue(
                    name="network_packets_recv_total",
                    value=stats.packets_recv,
                    timestamp=now,
                    labels={"interface": interface}
                )
            ])
        
        return metrics
    
    def _collect_system_metrics(self) -> List[MetricValue]:
        """Collect system metrics"""
        metrics = []
        now = datetime.now()
        
        # System information
        metrics.extend([
            MetricValue(name="system_uptime_seconds", value=time.time() - psutil.boot_time(), timestamp=now),
            MetricValue(name="system_processes_total", value=len(psutil.pids()), timestamp=now)
        ])
        
        # Connection counts
        try:
            connections = psutil.net_connections()
            connection_counts = defaultdict(int)
            for conn in connections:
                if conn.status:
                    connection_counts[conn.status] += 1
            
            for status, count in connection_counts.items():
                metrics.append(MetricValue(
                    name="network_connections_total",
                    value=count,
                    timestamp=now,
                    labels={"status": status}
                ))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        
        return metrics
    
    def get_metric_values(self, metric_name: str, labels: Dict[str, str] = None, 
                         time_range: TimeInterval = TimeInterval.HOUR) -> List[MetricValue]:
        """Get metric values"""
        metric_key = f"{metric_name}:{json.dumps(labels or {}, sort_keys=True)}"
        
        if metric_key not in self.metric_values:
            return []
        
        values = list(self.metric_values[metric_key])
        
        # Filter by time range
        now = datetime.now()
        time_delta = self._get_time_delta(time_range)
        cutoff_time = now - time_delta
        
        filtered_values = [v for v in values if v.timestamp >= cutoff_time]
        return sorted(filtered_values, key=lambda x: x.timestamp)
    
    def _get_time_delta(self, interval: TimeInterval) -> timedelta:
        """Get time delta from interval"""
        mapping = {
            TimeInterval.SECOND: timedelta(seconds=1),
            TimeInterval.MINUTE: timedelta(minutes=1),
            TimeInterval.FIVE_MINUTES: timedelta(minutes=5),
            TimeInterval.FIFTEEN_MINUTES: timedelta(minutes=15),
            TimeInterval.HOUR: timedelta(hours=1),
            TimeInterval.DAY: timedelta(days=1),
            TimeInterval.WEEK: timedelta(weeks=1)
        }
        return mapping.get(interval, timedelta(hours=1))
    
    def record_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record custom metric value"""
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self._store_metric_value(metric)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        all_metrics = {}
        
        for metric_key, values in self.metric_values.items():
            if values:
                latest = values[-1]
                all_metrics[metric_key] = {
                    'name': latest.name,
                    'value': latest.value,
                    'timestamp': latest.timestamp.isoformat(),
                    'labels': latest.labels
                }
        
        return all_metrics


class TimeSeriesDatabase:
    """Time series database for metrics storage"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.connection = None
        self.redis_client = None
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database"""
        if DATABASE_AVAILABLE and sqlite3:
            try:
                self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._create_tables()
                self.logger.info("SQLite database initialized")
            except Exception as e:
                self.logger.error(f"SQLite initialization failed: {e}")
        
        # Try Redis for faster access
        if DATABASE_AVAILABLE and redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                self.logger.info("Redis client initialized")
            except Exception as e:
                self.logger.warning(f"Redis initialization failed: {e}")
    
    def _create_tables(self):
        """Create database tables"""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                labels TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
        
        self.connection.commit()
    
    def store_metric(self, metric: MetricValue):
        """Store metric in database"""
        # Store in Redis for fast access
        if self.redis_client:
            try:
                redis_key = f"metric:{metric.name}:{json.dumps(metric.labels, sort_keys=True)}"
                redis_value = {
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'labels': json.dumps(metric.labels),
                    'metadata': json.dumps(metric.metadata)
                }
                self.redis_client.hset(redis_key, mapping=redis_value)
                self.redis_client.expire(redis_key, 3600)  # 1 hour TTL
            except Exception as e:
                self.logger.warning(f"Redis storage failed: {e}")
        
        # Store in SQLite for persistence
        if self.connection:
            try:
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO metrics (name, value, timestamp, labels, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.value,
                    metric.timestamp,
                    json.dumps(metric.labels),
                    json.dumps(metric.metadata)
                ))
                self.connection.commit()
            except Exception as e:
                self.logger.error(f"SQLite storage failed: {e}")
    
    def query_metrics(self, metric_name: str, labels: Dict[str, str] = None,
                     start_time: datetime = None, end_time: datetime = None,
                     limit: int = 1000) -> List[MetricValue]:
        """Query metrics from database"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        
        # Build query
        query = "SELECT name, value, timestamp, labels, metadata FROM metrics WHERE name = ?"
        params = [metric_name]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                name, value, timestamp, labels_json, metadata_json = row
                
                # Filter by labels if specified
                if labels:
                    row_labels = json.loads(labels_json) if labels_json else {}
                    if not all(row_labels.get(k) == v for k, v in labels.items()):
                        continue
                
                metrics.append(MetricValue(
                    name=name,
                    value=value,
                    timestamp=datetime.fromisoformat(timestamp),
                    labels=json.loads(labels_json) if labels_json else {},
                    metadata=json.loads(metadata_json) if metadata_json else {}
                ))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return []
    
    def aggregate_metrics(self, metric_name: str, aggregation: AggregationType,
                         time_interval: TimeInterval, labels: Dict[str, str] = None,
                         start_time: datetime = None, end_time: datetime = None) -> List[Tuple[datetime, float]]:
        """Aggregate metrics by time interval"""
        metrics = self.query_metrics(metric_name, labels, start_time, end_time)
        
        if not metrics:
            return []
        
        # Group by time interval
        interval_seconds = self._get_interval_seconds(time_interval)
        grouped = defaultdict(list)
        
        for metric in metrics:
            # Round timestamp to interval
            timestamp_seconds = metric.timestamp.timestamp()
            interval_key = int(timestamp_seconds // interval_seconds) * interval_seconds
            interval_time = datetime.fromtimestamp(interval_key)
            grouped[interval_time].append(metric.value)
        
        # Apply aggregation
        result = []
        for interval_time, values in sorted(grouped.items()):
            if aggregation == AggregationType.SUM:
                aggregated_value = sum(values)
            elif aggregation == AggregationType.AVERAGE:
                aggregated_value = statistics.mean(values)
            elif aggregation == AggregationType.MIN:
                aggregated_value = min(values)
            elif aggregation == AggregationType.MAX:
                aggregated_value = max(values)
            elif aggregation == AggregationType.COUNT:
                aggregated_value = len(values)
            elif aggregation == AggregationType.PERCENTILE_50:
                aggregated_value = statistics.median(values)
            elif aggregation == AggregationType.PERCENTILE_95:
                aggregated_value = self._percentile(values, 0.95)
            elif aggregation == AggregationType.PERCENTILE_99:
                aggregated_value = self._percentile(values, 0.99)
            else:
                aggregated_value = statistics.mean(values)
            
            result.append((interval_time, aggregated_value))
        
        return result
    
    def _get_interval_seconds(self, interval: TimeInterval) -> int:
        """Get interval in seconds"""
        mapping = {
            TimeInterval.SECOND: 1,
            TimeInterval.MINUTE: 60,
            TimeInterval.FIVE_MINUTES: 300,
            TimeInterval.FIFTEEN_MINUTES: 900,
            TimeInterval.HOUR: 3600,
            TimeInterval.DAY: 86400,
            TimeInterval.WEEK: 604800
        }
        return mapping.get(interval, 60)
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        
        if index >= len(sorted_values):
            return sorted_values[-1]
        
        return sorted_values[index]


class DashboardEngine:
    """Dashboard rendering and management"""
    
    def __init__(self, metrics_collector: MetricsCollector, time_series_db: TimeSeriesDatabase):
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = metrics_collector
        self.time_series_db = time_series_db
        self.dashboards = {}
        self.dashboard_cache = {}
        self.cache_ttl = 30  # seconds
        
        self._initialize_default_dashboards()
    
    def _initialize_default_dashboards(self):
        """Initialize default dashboards"""
        # System Overview Dashboard
        system_dashboard = Dashboard(
            name="system_overview",
            title="System Overview",
            type=DashboardType.SYSTEM_OVERVIEW,
            panels=[
                DashboardPanel(
                    title="CPU Usage",
                    type="graph",
                    metrics=["cpu_usage_total"],
                    time_range=TimeInterval.HOUR,
                    thresholds={"warning": 70, "critical": 90}
                ),
                DashboardPanel(
                    title="Memory Usage",
                    type="graph",
                    metrics=["memory_usage_percent"],
                    time_range=TimeInterval.HOUR,
                    thresholds={"warning": 80, "critical": 95}
                ),
                DashboardPanel(
                    title="Disk Usage",
                    type="singlestat",
                    metrics=["disk_usage_percent"],
                    thresholds={"warning": 80, "critical": 90}
                ),
                DashboardPanel(
                    title="Network I/O",
                    type="graph",
                    metrics=["network_bytes_sent_total", "network_bytes_recv_total"],
                    time_range=TimeInterval.HOUR
                )
            ]
        )
        self.register_dashboard(system_dashboard)
        
        # Application Metrics Dashboard
        app_dashboard = Dashboard(
            name="application_metrics",
            title="Application Metrics",
            type=DashboardType.APPLICATION_METRICS,
            panels=[
                DashboardPanel(
                    title="Request Rate",
                    type="graph",
                    metrics=["http_requests_total"],
                    time_range=TimeInterval.HOUR,
                    aggregation=AggregationType.SUM
                ),
                DashboardPanel(
                    title="Response Time",
                    type="graph",
                    metrics=["http_request_duration_seconds"],
                    time_range=TimeInterval.HOUR,
                    aggregation=AggregationType.PERCENTILE_95
                ),
                DashboardPanel(
                    title="Error Rate",
                    type="singlestat",
                    metrics=["http_errors_total"],
                    thresholds={"warning": 0.01, "critical": 0.05}
                ),
                DashboardPanel(
                    title="Active Connections",
                    type="singlestat",
                    metrics=["active_connections"],
                    time_range=TimeInterval.MINUTE
                )
            ]
        )
        self.register_dashboard(app_dashboard)
    
    def register_dashboard(self, dashboard: Dashboard):
        """Register dashboard"""
        self.dashboards[dashboard.name] = dashboard
        self.logger.info(f"Registered dashboard: {dashboard.name}")
    
    async def render_dashboard(self, dashboard_name: str, time_range: TimeInterval = None) -> Dict[str, Any]:
        """Render dashboard"""
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_name} not found")
        
        dashboard = self.dashboards[dashboard_name]
        
        # Check cache
        cache_key = f"{dashboard_name}:{time_range or dashboard.time_range}"
        cached_result = self.dashboard_cache.get(cache_key)
        
        if cached_result and (datetime.now() - cached_result['timestamp']).seconds < self.cache_ttl:
            return cached_result['data']
        
        # Render dashboard
        rendered_dashboard = {
            'name': dashboard.name,
            'title': dashboard.title,
            'type': dashboard.type.value,
            'refresh_interval': dashboard.refresh_interval,
            'time_range': (time_range or dashboard.time_range).value,
            'panels': []
        }
        
        # Render panels
        for panel in dashboard.panels:
            rendered_panel = await self._render_panel(panel, time_range or dashboard.time_range)
            rendered_dashboard['panels'].append(rendered_panel)
        
        # Cache result
        self.dashboard_cache[cache_key] = {
            'data': rendered_dashboard,
            'timestamp': datetime.now()
        }
        
        return rendered_dashboard
    
    async def _render_panel(self, panel: DashboardPanel, time_range: TimeInterval) -> Dict[str, Any]:
        """Render dashboard panel"""
        rendered_panel = {
            'title': panel.title,
            'type': panel.type,
            'time_range': time_range.value,
            'aggregation': panel.aggregation.value,
            'thresholds': panel.thresholds,
            'data': {}
        }
        
        # Get data for each metric
        for metric_name in panel.metrics:
            if panel.type == "singlestat":
                # Single value
                values = self.metrics_collector.get_metric_values(metric_name, time_range=TimeInterval.MINUTE)
                if values:
                    current_value = values[-1].value
                    rendered_panel['data'][metric_name] = {
                        'current_value': current_value,
                        'status': self._get_metric_status(current_value, panel.thresholds)
                    }
            
            elif panel.type == "graph":
                # Time series data
                end_time = datetime.now()
                start_time = end_time - self.time_series_db._get_time_delta(time_range)
                
                aggregated_data = self.time_series_db.aggregate_metrics(
                    metric_name,
                    panel.aggregation,
                    TimeInterval.MINUTE,
                    start_time=start_time,
                    end_time=end_time
                )
                
                rendered_panel['data'][metric_name] = {
                    'timestamps': [t.isoformat() for t, v in aggregated_data],
                    'values': [v for t, v in aggregated_data]
                }
            
            elif panel.type == "table":
                # Table data
                values = self.metrics_collector.get_metric_values(metric_name, time_range=time_range)
                table_data = []
                
                for value in values[-10:]:  # Last 10 values
                    table_data.append({
                        'timestamp': value.timestamp.isoformat(),
                        'value': value.value,
                        'labels': value.labels
                    })
                
                rendered_panel['data'][metric_name] = table_data
        
        return rendered_panel
    
    def _get_metric_status(self, value: float, thresholds: Dict[str, float]) -> str:
        """Get metric status based on thresholds"""
        if not thresholds:
            return MetricStatus.HEALTHY.value
        
        critical_threshold = thresholds.get('critical')
        warning_threshold = thresholds.get('warning')
        
        if critical_threshold is not None and value >= critical_threshold:
            return MetricStatus.CRITICAL.value
        elif warning_threshold is not None and value >= warning_threshold:
            return MetricStatus.WARNING.value
        else:
            return MetricStatus.HEALTHY.value
    
    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards"""
        dashboard_list = []
        
        for dashboard in self.dashboards.values():
            dashboard_list.append({
                'name': dashboard.name,
                'title': dashboard.title,
                'type': dashboard.type.value,
                'panel_count': len(dashboard.panels),
                'refresh_interval': dashboard.refresh_interval,
                'tags': dashboard.tags
            })
        
        return dashboard_list


class MetricsServer:
    """Metrics HTTP server for Prometheus-style endpoints"""
    
    def __init__(self, metrics_collector: MetricsCollector, dashboard_engine: DashboardEngine, port: int = 9090):
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = metrics_collector
        self.dashboard_engine = dashboard_engine
        self.port = port
        self.app = None
        self.server_thread = None
        
        if WEB_AVAILABLE and Flask:
            self._initialize_server()
    
    def _initialize_server(self):
        """Initialize Flask server"""
        self.app = Flask(__name__)
        
        # Metrics endpoint (Prometheus format)
        @self.app.route('/metrics')
        def metrics_endpoint():
            return Response(self._generate_prometheus_metrics(), mimetype='text/plain')
        
        # JSON metrics endpoint
        @self.app.route('/api/metrics')
        def json_metrics_endpoint():
            return jsonify(self.metrics_collector.get_all_metrics())
        
        # Dashboard list endpoint
        @self.app.route('/api/dashboards')
        def dashboards_endpoint():
            return jsonify(self.dashboard_engine.get_dashboard_list())
        
        # Dashboard render endpoint
        @self.app.route('/api/dashboards/<dashboard_name>')
        async def dashboard_endpoint(dashboard_name):
            time_range = request.args.get('time_range', 'hour')
            try:
                time_interval = TimeInterval(time_range)
                dashboard_data = await self.dashboard_engine.render_dashboard(dashboard_name, time_interval)
                return jsonify(dashboard_data)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
        
        # Health check endpoint
        @self.app.route('/health')
        def health_endpoint():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'metrics_count': len(self.metrics_collector.get_all_metrics())
            })
    
    def _generate_prometheus_metrics(self) -> str:
        """Generate Prometheus-format metrics"""
        output = []
        all_metrics = self.metrics_collector.get_all_metrics()
        
        for metric_key, metric_data in all_metrics.items():
            metric_name = metric_data['name']
            value = metric_data['value']
            labels = metric_data['labels']
            
            # Format labels
            label_str = ""
            if labels:
                label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
                label_str = "{" + ",".join(label_pairs) + "}"
            
            # Add metric line
            output.append(f"{metric_name}{label_str} {value}")
        
        return "\n".join(output) + "\n"
    
    def start_server(self):
        """Start metrics server"""
        if not self.app:
            self.logger.warning("Flask not available, cannot start metrics server")
            return
        
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        self.logger.info(f"Metrics server started on port {self.port}")
    
    def stop_server(self):
        """Stop metrics server"""
        # Flask server will stop when main thread exits (daemon=True)
        self.logger.info("Metrics server stopped")


class MonitoringEngine:
    """
    Enhanced monitoring engine with real-time metrics collection,
    Prometheus-style metrics, Grafana-style dashboards, and comprehensive monitoring.
    """
    
    def __init__(self, db_path: str = "monitoring.db", metrics_port: int = 9090):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.metrics_port = metrics_port
        
        # Performance tracking
        self.performance_metrics = {
            'collection_times': deque(maxlen=100),
            'dashboard_render_times': deque(maxlen=100),
            'metrics_count': 0,
            'dashboards_rendered': 0,
            'alerts_triggered': 0
        }
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.time_series_db = TimeSeriesDatabase(db_path)
        self.dashboard_engine = DashboardEngine(self.metrics_collector, self.time_series_db)
        self.metrics_server = MetricsServer(self.metrics_collector, self.dashboard_engine, metrics_port)
        
        # Alert rules
        self.alert_rules = {}
        self.alert_states = {}
        
        # Integration references
        self.deployment_engine = None
        self.alerting_engine = None
        self.infrastructure_engine = None
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("MonitoringEngine initialized successfully")
    
    def set_integrations(self, deployment_engine=None, alerting_engine=None, infrastructure_engine=None):
        """Set integration references"""
        self.deployment_engine = deployment_engine
        self.alerting_engine = alerting_engine
        self.infrastructure_engine = infrastructure_engine
        
        # Register deployment-specific collectors if available
        if deployment_engine:
            self.metrics_collector.register_collector(
                "deployment_collector",
                self._collect_deployment_metrics,
                30
            )
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Start metrics server
        self.metrics_server.start_server()
        
        # Start database storage task
        threading.Thread(target=self._metrics_storage_task, daemon=True).start()
        
        # Start alert evaluation task
        threading.Thread(target=self._alert_evaluation_task, daemon=True).start()
    
    def _metrics_storage_task(self):
        """Background task to store metrics in database"""
        while True:
            try:
                # Store all current metrics in database
                all_metrics = self.metrics_collector.get_all_metrics()
                
                for metric_key, metric_data in all_metrics.items():
                    metric = MetricValue(
                        name=metric_data['name'],
                        value=metric_data['value'],
                        timestamp=datetime.fromisoformat(metric_data['timestamp']),
                        labels=metric_data['labels']
                    )
                    self.time_series_db.store_metric(metric)
                
                # Update performance metrics
                self.performance_metrics['metrics_count'] = len(all_metrics)
                
                time.sleep(60)  # Store every minute
                
            except Exception as e:
                self.logger.error(f"Metrics storage task error: {e}")
                time.sleep(60)
    
    def _alert_evaluation_task(self):
        """Background task to evaluate alert rules"""
        while True:
            try:
                for rule_name, rule in self.alert_rules.items():
                    if rule.enabled:
                        self._evaluate_alert_rule(rule_name, rule)
                
                time.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Alert evaluation task error: {e}")
                time.sleep(30)
    
    async def _evaluate_alert_rule(self, rule_name: str, rule: AlertRule):
        """Evaluate alert rule"""
        try:
            # Get recent metric values
            values = self.metrics_collector.get_metric_values(
                rule.metric_name,
                time_range=TimeInterval.FIVE_MINUTES
            )
            
            if not values:
                return
            
            # Check condition
            current_value = values[-1].value
            condition_met = False
            
            if rule.condition == ">" and current_value > rule.threshold:
                condition_met = True
            elif rule.condition == "<" and current_value < rule.threshold:
                condition_met = True
            elif rule.condition == "==" and current_value == rule.threshold:
                condition_met = True
            elif rule.condition == "!=" and current_value != rule.threshold:
                condition_met = True
            
            # Update alert state
            current_state = self.alert_states.get(rule_name, {
                'active': False,
                'triggered_at': None,
                'last_notification': None
            })
            
            if condition_met and not current_state['active']:
                # Check duration
                if current_state.get('first_trigger'):
                    time_since_first = (datetime.now() - current_state['first_trigger']).seconds
                    if time_since_first >= rule.duration:
                        # Trigger alert
                        current_state['active'] = True
                        current_state['triggered_at'] = datetime.now()
                        
                        if self.alerting_engine:
                            await self.alerting_engine.trigger_alert(
                                rule_name,
                                rule.severity,
                                f"{rule.description}: {rule.metric_name} {rule.condition} {rule.threshold} (current: {current_value})"
                            )
                        
                        self.performance_metrics['alerts_triggered'] += 1
                        self.logger.warning(f"Alert triggered: {rule_name}")
                else:
                    current_state['first_trigger'] = datetime.now()
            
            elif not condition_met and current_state['active']:
                # Resolve alert
                current_state['active'] = False
                current_state['first_trigger'] = None
                
                if self.alerting_engine:
                    await self.alerting_engine.resolve_alert(rule_name)
                
                self.logger.info(f"Alert resolved: {rule_name}")
            
            self.alert_states[rule_name] = current_state
            
        except Exception as e:
            self.logger.error(f"Alert evaluation failed for {rule_name}: {e}")
    
    def _collect_deployment_metrics(self) -> List[MetricValue]:
        """Collect deployment-specific metrics"""
        if not self.deployment_engine:
            return []
        
        metrics = []
        now = datetime.now()
        
        try:
            # Get deployment metrics
            deployment_metrics = self.deployment_engine.get_deployment_metrics()
            
            metrics.extend([
                MetricValue(
                    name="deployments_total",
                    value=deployment_metrics.get('total_deployments', 0),
                    timestamp=now
                ),
                MetricValue(
                    name="deployment_success_rate",
                    value=deployment_metrics.get('success_rate', 0),
                    timestamp=now
                ),
                MetricValue(
                    name="deployment_avg_duration",
                    value=deployment_metrics.get('avg_deployment_time', 0),
                    timestamp=now
                ),
                MetricValue(
                    name="deployment_rollback_rate",
                    value=deployment_metrics.get('rollback_rate', 0),
                    timestamp=now
                )
            ])
            
        except Exception as e:
            self.logger.error(f"Deployment metrics collection failed: {e}")
        
        return metrics
    
    async def register_alert_rule(self, rule: AlertRule):
        """Register alert rule"""
        self.alert_rules[rule.name] = rule
        self.alert_states[rule.name] = {
            'active': False,
            'triggered_at': None,
            'first_trigger': None
        }
        
        self.logger.info(f"Registered alert rule: {rule.name}")
    
    def register_custom_metric(self, definition: MetricDefinition):
        """Register custom metric definition"""
        self.metrics_collector.register_metric(definition)
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record metric value"""
        self.metrics_collector.record_custom_metric(name, value, labels)
    
    async def get_metrics(self, metric_names: List[str] = None, 
                         time_range: TimeInterval = TimeInterval.HOUR,
                         labels: Dict[str, str] = None) -> Dict[str, List[MetricValue]]:
        """Get metrics data"""
        start_time = time.time()
        
        if metric_names is None:
            # Get all available metrics
            all_metrics = self.metrics_collector.get_all_metrics()
            metric_names = list(set(m['name'] for m in all_metrics.values()))
        
        result = {}
        for metric_name in metric_names:
            values = self.metrics_collector.get_metric_values(metric_name, labels, time_range)
            result[metric_name] = values
        
        # Update performance metrics
        collection_time = time.time() - start_time
        self.performance_metrics['collection_times'].append(collection_time)
        
        return result
    
    async def get_dashboard(self, dashboard_name: str, time_range: TimeInterval = None) -> Dict[str, Any]:
        """Get dashboard data"""
        start_time = time.time()
        
        dashboard_data = await self.dashboard_engine.render_dashboard(dashboard_name, time_range)
        
        # Update performance metrics
        render_time = time.time() - start_time
        self.performance_metrics['dashboard_render_times'].append(render_time)
        self.performance_metrics['dashboards_rendered'] += 1
        
        return dashboard_data
    
    async def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get metrics for specific deployment"""
        if not self.deployment_engine:
            return {}
        
        try:
            deployment_status = self.deployment_engine.get_deployment_status(deployment_id)
            
            # Get deployment-related metrics
            deployment_metrics = {}
            
            # Basic deployment info
            deployment_metrics.update({
                'deployment_id': deployment_id,
                'status': deployment_status.get('status'),
                'duration': deployment_status.get('metrics', {}).get('duration', 0),
                'success_rate': deployment_status.get('metrics', {}).get('success_rate', 0)
            })
            
            # Get application metrics if available
            app_name = deployment_status.get('config', {}).get('name')
            if app_name:
                app_metrics = await self.get_metrics([
                    f"{app_name}_cpu_usage",
                    f"{app_name}_memory_usage",
                    f"{app_name}_request_count",
                    f"{app_name}_error_count"
                ], TimeInterval.HOUR)
                
                deployment_metrics['application_metrics'] = app_metrics
            
            return deployment_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get deployment metrics: {e}")
            return {}
    
    async def create_custom_dashboard(self, dashboard: Dashboard):
        """Create custom dashboard"""
        self.dashboard_engine.register_dashboard(dashboard)
        self.logger.info(f"Created custom dashboard: {dashboard.name}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        try:
            # Get recent metrics
            cpu_metrics = self.metrics_collector.get_metric_values(
                "cpu_usage_total", 
                time_range=TimeInterval.MINUTE
            )
            memory_metrics = self.metrics_collector.get_metric_values(
                "memory_usage_percent",
                time_range=TimeInterval.MINUTE
            )
            disk_metrics = self.metrics_collector.get_metric_values(
                "disk_usage_percent",
                time_range=TimeInterval.MINUTE
            )
            
            health_status = {
                'overall_status': MetricStatus.HEALTHY.value,
                'timestamp': datetime.now().isoformat(),
                'components': {}
            }
            
            # CPU health
            if cpu_metrics:
                cpu_value = cpu_metrics[-1].value
                if cpu_value > 90:
                    health_status['components']['cpu'] = {'status': MetricStatus.CRITICAL.value, 'value': cpu_value}
                elif cpu_value > 70:
                    health_status['components']['cpu'] = {'status': MetricStatus.WARNING.value, 'value': cpu_value}
                else:
                    health_status['components']['cpu'] = {'status': MetricStatus.HEALTHY.value, 'value': cpu_value}
            
            # Memory health
            if memory_metrics:
                memory_value = memory_metrics[-1].value
                if memory_value > 95:
                    health_status['components']['memory'] = {'status': MetricStatus.CRITICAL.value, 'value': memory_value}
                elif memory_value > 80:
                    health_status['components']['memory'] = {'status': MetricStatus.WARNING.value, 'value': memory_value}
                else:
                    health_status['components']['memory'] = {'status': MetricStatus.HEALTHY.value, 'value': memory_value}
            
            # Disk health
            if disk_metrics:
                disk_value = max(m.value for m in disk_metrics)  # Worst disk usage
                if disk_value > 90:
                    health_status['components']['disk'] = {'status': MetricStatus.CRITICAL.value, 'value': disk_value}
                elif disk_value > 80:
                    health_status['components']['disk'] = {'status': MetricStatus.WARNING.value, 'value': disk_value}
                else:
                    health_status['components']['disk'] = {'status': MetricStatus.HEALTHY.value, 'value': disk_value}
            
            # Overall status based on components
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            if MetricStatus.CRITICAL.value in component_statuses:
                health_status['overall_status'] = MetricStatus.CRITICAL.value
            elif MetricStatus.WARNING.value in component_statuses:
                health_status['overall_status'] = MetricStatus.WARNING.value
            
            # Active alerts
            active_alerts = [name for name, state in self.alert_states.items() if state['active']]
            health_status['active_alerts'] = len(active_alerts)
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {
                'overall_status': MetricStatus.UNKNOWN.value,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get monitoring engine performance metrics"""
        return {
            'metrics_collected': self.performance_metrics['metrics_count'],
            'dashboards_rendered': self.performance_metrics['dashboards_rendered'],
            'alerts_triggered': self.performance_metrics['alerts_triggered'],
            'avg_collection_time': statistics.mean(self.performance_metrics['collection_times']) if self.performance_metrics['collection_times'] else 0,
            'avg_dashboard_render_time': statistics.mean(self.performance_metrics['dashboard_render_times']) if self.performance_metrics['dashboard_render_times'] else 0,
            'active_alert_rules': len(self.alert_rules),
            'active_alerts': len([s for s in self.alert_states.values() if s['active']]),
            'registered_dashboards': len(self.dashboard_engine.dashboards)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop metrics server
            self.metrics_server.stop_server()
            
            # Close database connections
            if self.time_series_db.connection:
                self.time_series_db.connection.close()
            
            if self.time_series_db.redis_client:
                self.time_series_db.redis_client.close()
            
            self.logger.info("MonitoringEngine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"MonitoringEngine cleanup error: {e}")


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create monitoring engine
        engine = MonitoringEngine()
        
        # Register custom metrics
        custom_metric = MetricDefinition(
            name="manufacturing_line_throughput",
            type=MetricType.GAUGE,
            description="Manufacturing line throughput in units per hour",
            unit="units/hour",
            thresholds={'warning': 800, 'critical': 500}
        )
        engine.register_custom_metric(custom_metric)
        
        # Register alert rule
        alert_rule = AlertRule(
            name="high_cpu_usage",
            metric_name="cpu_usage_total",
            condition=">",
            threshold=80.0,
            duration=120,  # 2 minutes
            severity="warning",
            description="High CPU usage detected"
        )
        await engine.register_alert_rule(alert_rule)
        
        # Record some custom metrics
        engine.record_metric("manufacturing_line_throughput", 950.0, {"line": "line_1"})
        engine.record_metric("manufacturing_line_throughput", 880.0, {"line": "line_2"})
        
        # Wait for metrics collection
        await asyncio.sleep(5)
        
        try:
            # Get system health
            health = engine.get_system_health()
            print(f"System health: {health}")
            
            # Get dashboard
            dashboard = await engine.get_dashboard("system_overview")
            print(f"Dashboard panels: {len(dashboard['panels'])}")
            
            # Get monitoring metrics
            metrics = engine.get_monitoring_metrics()
            print(f"Monitoring metrics: {metrics}")
            
            # Get custom metrics
            custom_metrics = await engine.get_metrics(["manufacturing_line_throughput"])
            print(f"Custom metrics: {len(custom_metrics['manufacturing_line_throughput'])} values")
            
        except Exception as e:
            print(f"Monitoring test failed: {e}")
        
        finally:
            await engine.cleanup()
    
    asyncio.run(main())