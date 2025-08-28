"""Data logging and metrics collection interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import json
from pathlib import Path


@dataclass
class LogEntry:
    """Standard log entry format."""
    timestamp: float
    component_id: str
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'component_id': self.component_id,
            'level': self.level,
            'message': self.message,
            'data': self.data
        }


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    component_id: str
    metric_name: str
    value: Any
    unit: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'component_id': self.component_id,
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'tags': self.tags
        }


class DataLogger(ABC):
    """Abstract data logging interface."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.log_buffer: List[LogEntry] = []
        self.max_buffer_size = 1000
    
    @abstractmethod
    def write_log(self, entry: LogEntry) -> bool:
        """Write log entry to storage."""
        pass
    
    @abstractmethod
    def read_logs(self, start_time: Optional[float] = None, 
                  end_time: Optional[float] = None,
                  level: Optional[str] = None) -> List[LogEntry]:
        """Read log entries from storage."""
        pass
    
    def log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Add log entry."""
        entry = LogEntry(
            timestamp=time.time(),
            component_id=self.component_id,
            level=level,
            message=message,
            data=data
        )
        
        self.log_buffer.append(entry)
        if len(self.log_buffer) >= self.max_buffer_size:
            self.flush_logs()
        
        self.write_log(entry)
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.log(level='INFO', message=message, data=data)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.log(level='WARNING', message=message, data=data)
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self.log(level='ERROR', message=message, data=data)
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self.log(level='DEBUG', message=message, data=data)
    
    def flush_logs(self):
        """Flush log buffer to storage."""
        for entry in self.log_buffer:
            self.write_log(entry)
        self.log_buffer.clear()


class MetricsCollector(ABC):
    """Abstract metrics collection interface."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.metrics_buffer: List[MetricPoint] = []
        self.max_buffer_size = 1000
    
    @abstractmethod
    def write_metric(self, metric: MetricPoint) -> bool:
        """Write metric point to storage."""
        pass
    
    @abstractmethod
    def read_metrics(self, metric_name: str,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> List[MetricPoint]:
        """Read metric points from storage."""
        pass
    
    def record(self, metric_name: str, value: Any, 
               unit: Optional[str] = None,
               tags: Optional[Dict[str, str]] = None):
        """Record a metric point."""
        metric = MetricPoint(
            timestamp=time.time(),
            component_id=self.component_id,
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags
        )
        
        self.metrics_buffer.append(metric)
        if len(self.metrics_buffer) >= self.max_buffer_size:
            self.flush_metrics()
        
        self.write_metric(metric)
    
    def counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        self.record(name, value, unit='count', tags=tags)
    
    def gauge(self, name: str, value: float, unit: Optional[str] = None,
              tags: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        self.record(name, value, unit=unit, tags=tags)
    
    def timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric."""
        self.record(name, duration, unit='seconds', tags=tags)
    
    def flush_metrics(self):
        """Flush metrics buffer to storage."""
        for metric in self.metrics_buffer:
            self.write_metric(metric)
        self.metrics_buffer.clear()


class FileDataLogger(DataLogger):
    """File-based data logger implementation."""
    
    def __init__(self, component_id: str, log_directory: str = "logs"):
        super().__init__(component_id)
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_directory / f"{component_id}.log"
    
    def write_log(self, entry: LogEntry) -> bool:
        """Write log entry to file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
            return True
        except Exception:
            return False
    
    def read_logs(self, start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  level: Optional[str] = None) -> List[LogEntry]:
        """Read log entries from file."""
        entries = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    entry = LogEntry(**data)
                    
                    # Apply filters
                    if start_time and entry.timestamp < start_time:
                        continue
                    if end_time and entry.timestamp > end_time:
                        continue
                    if level and entry.level != level:
                        continue
                    
                    entries.append(entry)
        except Exception:
            pass
        
        return entries


class FileMetricsCollector(MetricsCollector):
    """File-based metrics collector implementation."""
    
    def __init__(self, component_id: str, metrics_directory: str = "metrics"):
        super().__init__(component_id)
        self.metrics_directory = Path(metrics_directory)
        self.metrics_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_directory / f"{component_id}_metrics.json"
    
    def write_metric(self, metric: MetricPoint) -> bool:
        """Write metric point to file."""
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metric.to_dict()) + '\n')
            return True
        except Exception:
            return False
    
    def read_metrics(self, metric_name: str,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> List[MetricPoint]:
        """Read metric points from file."""
        metrics = []
        try:
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    metric = MetricPoint(**data)
                    
                    # Apply filters
                    if metric.metric_name != metric_name:
                        continue
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    
                    metrics.append(metric)
        except Exception:
            pass
        
        return metrics