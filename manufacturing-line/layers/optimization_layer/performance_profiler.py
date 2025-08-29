"""
Performance Profiler Engine - Week 14: Performance Optimization & Scalability

Comprehensive performance profiling system for CPU, memory, I/O, and network
analysis with real-time bottleneck detection and optimization recommendations.
"""

import asyncio
import logging
import time
import psutil
import threading
import gc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
import functools
import tracemalloc
import sys


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    network_sent_mb_s: float
    network_recv_mb_s: float
    process_count: int
    thread_count: int
    open_files: int
    response_time_ms: float = 0.0
    throughput_ops_s: float = 0.0


@dataclass
class BottleneckAlert:
    """Bottleneck detection alert."""
    alert_id: str
    component: str
    bottleneck_type: str
    severity: str  # low, medium, high, critical
    description: str
    metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: str
    resolved: bool = False


@dataclass
class ProfilerConfig:
    """Profiler configuration."""
    sampling_interval_ms: int = 1000
    history_retention_minutes: int = 60
    bottleneck_threshold_cpu: float = 85.0
    bottleneck_threshold_memory: float = 80.0
    bottleneck_threshold_disk_io: float = 100.0  # MB/s
    bottleneck_threshold_response_time: float = 200.0  # ms
    enable_memory_tracing: bool = True
    enable_function_profiling: bool = True
    max_bottleneck_alerts: int = 100


class PerformanceProfiler:
    """
    Enterprise-grade performance profiler for manufacturing systems.
    
    Provides comprehensive system profiling with real-time bottleneck detection,
    performance analytics, and automated optimization recommendations.
    """
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        """
        Initialize Performance Profiler.
        
        Args:
            config: Profiler configuration settings
        """
        self.config = config or ProfilerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance targets
        self.profile_target_ms = 500  # Complete system profile in <500ms
        self.bottleneck_detection_target_ms = 100  # Detect bottlenecks in <100ms
        self.recommendation_target_ms = 200  # Generate recommendations in <200ms
        
        # Profiling state
        self.is_profiling = False
        self.profiling_thread = None
        self.metrics_history = deque(maxlen=int(self.config.history_retention_minutes * 60))
        self.bottleneck_alerts = deque(maxlen=self.config.max_bottleneck_alerts)
        
        # Function profiling
        self.function_profiles = defaultdict(list)
        self.profiled_functions = {}
        
        # System baseline
        self.baseline_metrics = None
        self.performance_trends = defaultdict(list)
        
        # Resource monitoring
        self.system_resources = psutil.virtual_memory()
        self.cpu_count = psutil.cpu_count()
        self.disk_io_baseline = None
        self.network_io_baseline = None
        
        # Threading
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="profiler")
        
        # Performance statistics
        self.profiler_stats = {
            'profiles_completed': 0,
            'bottlenecks_detected': 0,
            'recommendations_generated': 0,
            'avg_profile_time_ms': 0.0,
            'avg_detection_time_ms': 0.0,
            'uptime_seconds': 0
        }
        
        self.start_time = datetime.now()
        
        # Initialize memory tracing if enabled
        if self.config.enable_memory_tracing:
            tracemalloc.start()
        
        self.logger.info("PerformanceProfiler initialized successfully")
    
    def start_profiling(self) -> Dict[str, Any]:
        """Start continuous performance profiling."""
        if self.is_profiling:
            return {'success': False, 'message': 'Profiling already running'}
        
        try:
            self.is_profiling = True
            self.profiling_thread = threading.Thread(
                target=self._profiling_worker,
                daemon=True,
                name="performance_profiler"
            )
            self.profiling_thread.start()
            
            # Establish baseline
            self._establish_baseline()
            
            self.logger.info("Performance profiling started")
            
            return {
                'success': True,
                'message': 'Performance profiling started',
                'sampling_interval_ms': self.config.sampling_interval_ms,
                'retention_minutes': self.config.history_retention_minutes
            }
            
        except Exception as e:
            self.is_profiling = False
            self.logger.error(f"Failed to start profiling: {e}")
            return {'success': False, 'error': str(e)}
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop performance profiling."""
        try:
            self.is_profiling = False
            
            if self.profiling_thread and self.profiling_thread.is_alive():
                self.profiling_thread.join(timeout=5.0)
            
            self.logger.info("Performance profiling stopped")
            
            return {
                'success': True,
                'message': 'Performance profiling stopped',
                'total_profiles': self.profiler_stats['profiles_completed'],
                'total_bottlenecks': self.profiler_stats['bottlenecks_detected']
            }
            
        except Exception as e:
            self.logger.error(f"Error stopping profiling: {e}")
            return {'success': False, 'error': str(e)}
    
    def _profiling_worker(self):
        """Background worker for continuous profiling."""
        while self.is_profiling:
            try:
                start_time = time.time()
                
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                with self.lock:
                    self.metrics_history.append(metrics)
                
                # Detect bottlenecks
                self._detect_bottlenecks(metrics)
                
                # Update statistics
                profile_time_ms = (time.time() - start_time) * 1000
                self._update_profiler_stats(profile_time_ms)
                
                # Sleep until next sampling
                sleep_time = self.config.sampling_interval_ms / 1000.0
                time.sleep(max(0, sleep_time - (profile_time_ms / 1000.0)))
                
            except Exception as e:
                self.logger.error(f"Profiling worker error: {e}")
                time.sleep(1.0)  # Brief pause before retrying
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_read_mb_s = 0.0
            disk_write_mb_s = 0.0
            
            if self.disk_io_baseline:
                time_delta = time.time() - self.disk_io_baseline['timestamp']
                if time_delta > 0:
                    read_bytes_delta = disk_io.read_bytes - self.disk_io_baseline['read_bytes']
                    write_bytes_delta = disk_io.write_bytes - self.disk_io_baseline['write_bytes']
                    disk_read_mb_s = (read_bytes_delta / time_delta) / (1024 * 1024)
                    disk_write_mb_s = (write_bytes_delta / time_delta) / (1024 * 1024)
            
            self.disk_io_baseline = {
                'timestamp': time.time(),
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes
            }
            
            # Network I/O metrics
            network_io = psutil.net_io_counters()
            network_sent_mb_s = 0.0
            network_recv_mb_s = 0.0
            
            if self.network_io_baseline:
                time_delta = time.time() - self.network_io_baseline['timestamp']
                if time_delta > 0:
                    sent_bytes_delta = network_io.bytes_sent - self.network_io_baseline['bytes_sent']
                    recv_bytes_delta = network_io.bytes_recv - self.network_io_baseline['bytes_recv']
                    network_sent_mb_s = (sent_bytes_delta / time_delta) / (1024 * 1024)
                    network_recv_mb_s = (recv_bytes_delta / time_delta) / (1024 * 1024)
            
            self.network_io_baseline = {
                'timestamp': time.time(),
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv
            }
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Current process metrics
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            try:
                open_files = len(current_process.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                open_files = 0
            
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_read_mb_s=disk_read_mb_s,
                disk_write_mb_s=disk_write_mb_s,
                network_sent_mb_s=network_sent_mb_s,
                network_recv_mb_s=network_recv_mb_s,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics on error
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_read_mb_s=0.0,
                disk_write_mb_s=0.0,
                network_sent_mb_s=0.0,
                network_recv_mb_s=0.0,
                process_count=0,
                thread_count=0,
                open_files=0
            )
    
    def _detect_bottlenecks(self, metrics: PerformanceMetrics):
        """Detect performance bottlenecks from metrics."""
        try:
            start_time = time.time()
            
            # CPU bottleneck detection
            if metrics.cpu_percent > self.config.bottleneck_threshold_cpu:
                self._create_bottleneck_alert(
                    'cpu',
                    'high_cpu_usage',
                    'high',
                    f'CPU usage at {metrics.cpu_percent:.1f}% exceeds threshold',
                    {'cpu_percent': metrics.cpu_percent},
                    [
                        'Identify CPU-intensive processes and optimize',
                        'Consider horizontal scaling to distribute load',
                        'Profile code for optimization opportunities',
                        'Enable CPU caching for frequent operations'
                    ]
                )
            
            # Memory bottleneck detection
            if metrics.memory_percent > self.config.bottleneck_threshold_memory:
                self._create_bottleneck_alert(
                    'memory',
                    'high_memory_usage',
                    'high',
                    f'Memory usage at {metrics.memory_percent:.1f}% exceeds threshold',
                    {'memory_percent': metrics.memory_percent, 'available_mb': metrics.memory_available_mb},
                    [
                        'Review memory usage patterns and optimize',
                        'Implement memory caching strategies',
                        'Consider increasing available memory',
                        'Profile for memory leaks and optimize garbage collection'
                    ]
                )
            
            # Disk I/O bottleneck detection
            total_disk_io = metrics.disk_read_mb_s + metrics.disk_write_mb_s
            if total_disk_io > self.config.bottleneck_threshold_disk_io:
                self._create_bottleneck_alert(
                    'disk_io',
                    'high_disk_io',
                    'medium',
                    f'Disk I/O at {total_disk_io:.1f}MB/s exceeds threshold',
                    {'disk_read_mb_s': metrics.disk_read_mb_s, 'disk_write_mb_s': metrics.disk_write_mb_s},
                    [
                        'Optimize database queries to reduce disk access',
                        'Implement disk caching for frequently accessed data',
                        'Consider SSD upgrade for better I/O performance',
                        'Optimize file access patterns and batch operations'
                    ]
                )
            
            # Response time bottleneck detection
            if metrics.response_time_ms > self.config.bottleneck_threshold_response_time:
                self._create_bottleneck_alert(
                    'response_time',
                    'slow_response',
                    'high',
                    f'Response time at {metrics.response_time_ms:.1f}ms exceeds threshold',
                    {'response_time_ms': metrics.response_time_ms},
                    [
                        'Profile slow endpoints and optimize',
                        'Implement response caching',
                        'Optimize database queries and indexing',
                        'Consider load balancing to distribute requests'
                    ]
                )
            
            # Update detection statistics
            detection_time_ms = (time.time() - start_time) * 1000
            if detection_time_ms > self.bottleneck_detection_target_ms:
                self.logger.warning(f"Bottleneck detection took {detection_time_ms:.1f}ms (target: {self.bottleneck_detection_target_ms}ms)")
            
        except Exception as e:
            self.logger.error(f"Error detecting bottlenecks: {e}")
    
    def _create_bottleneck_alert(self, component: str, bottleneck_type: str, severity: str,
                                description: str, metrics: Dict[str, float], recommendations: List[str]):
        """Create a bottleneck alert."""
        try:
            alert = BottleneckAlert(
                alert_id=f"{component}_{bottleneck_type}_{int(time.time())}",
                component=component,
                bottleneck_type=bottleneck_type,
                severity=severity,
                description=description,
                metrics=metrics,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
            # Check if similar alert already exists (avoid spam)
            existing_alert = None
            with self.lock:
                for existing in self.bottleneck_alerts:
                    if (existing.component == component and 
                        existing.bottleneck_type == bottleneck_type and 
                        not existing.resolved):
                        existing_alert = existing
                        break
                
                if not existing_alert:
                    self.bottleneck_alerts.append(alert)
                    self.profiler_stats['bottlenecks_detected'] += 1
                    self.logger.warning(f"Bottleneck detected: {description}")
            
        except Exception as e:
            self.logger.error(f"Error creating bottleneck alert: {e}")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.config.enable_function_profiling:
                return func(*args, **kwargs)
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                execution_time_ms = (end_time - start_time) * 1000
                memory_delta_mb = (end_memory - start_memory) / (1024 * 1024)
                
                # Store function profile
                profile_data = {
                    'function_name': func.__name__,
                    'execution_time_ms': execution_time_ms,
                    'memory_delta_mb': memory_delta_mb,
                    'success': success,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                
                with self.lock:
                    self.function_profiles[func.__name__].append(profile_data)
                    # Keep only last 100 profiles per function
                    if len(self.function_profiles[func.__name__]) > 100:
                        self.function_profiles[func.__name__] = self.function_profiles[func.__name__][-100:]
            
            return result
        
        # Store reference for analysis
        self.profiled_functions[func.__name__] = wrapper
        return wrapper
    
    async def profile_async_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Profile an async function execution."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = await func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            profile_data = {
                'execution_time_ms': (end_time - start_time) * 1000,
                'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
                'success': success,
                'error': error
            }
        
        return result, profile_data
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            if self.config.enable_memory_tracing and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                return current
            else:
                process = psutil.Process()
                return process.memory_info().rss
        except:
            return 0
    
    def _establish_baseline(self):
        """Establish performance baseline for comparison."""
        try:
            # Collect initial metrics for baseline
            baseline_samples = []
            for _ in range(10):  # Take 10 samples over 10 seconds
                metrics = self._collect_system_metrics()
                baseline_samples.append(metrics)
                time.sleep(1.0)
            
            # Calculate baseline averages
            self.baseline_metrics = {
                'cpu_percent': statistics.mean([m.cpu_percent for m in baseline_samples]),
                'memory_percent': statistics.mean([m.memory_percent for m in baseline_samples]),
                'disk_io_mb_s': statistics.mean([m.disk_read_mb_s + m.disk_write_mb_s for m in baseline_samples]),
                'network_io_mb_s': statistics.mean([m.network_sent_mb_s + m.network_recv_mb_s for m in baseline_samples])
            }
            
            self.logger.info(f"Performance baseline established: {self.baseline_metrics}")
            
        except Exception as e:
            self.logger.error(f"Error establishing baseline: {e}")
    
    def _update_profiler_stats(self, profile_time_ms: float):
        """Update profiler statistics."""
        with self.lock:
            self.profiler_stats['profiles_completed'] += 1
            
            # Update average profile time
            current_avg = self.profiler_stats['avg_profile_time_ms']
            profile_count = self.profiler_stats['profiles_completed']
            new_avg = ((current_avg * (profile_count - 1)) + profile_time_ms) / profile_count
            self.profiler_stats['avg_profile_time_ms'] = new_avg
            
            # Update uptime
            self.profiler_stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
            
            # Performance validation
            if profile_time_ms > self.profile_target_ms:
                self.logger.warning(f"Profile time {profile_time_ms:.1f}ms exceeded target {self.profile_target_ms}ms")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 30) -> List[PerformanceMetrics]:
        """Get performance metrics history for specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        cutoff_timestamp = cutoff_time.isoformat()
        
        with self.lock:
            return [
                metrics for metrics in self.metrics_history
                if metrics.timestamp >= cutoff_timestamp
            ]
    
    def get_bottleneck_alerts(self, resolved: bool = False) -> List[BottleneckAlert]:
        """Get bottleneck alerts."""
        with self.lock:
            if resolved:
                return list(self.bottleneck_alerts)
            else:
                return [alert for alert in self.bottleneck_alerts if not alert.resolved]
    
    def resolve_bottleneck(self, alert_id: str) -> Dict[str, Any]:
        """Mark a bottleneck alert as resolved."""
        try:
            with self.lock:
                for alert in self.bottleneck_alerts:
                    if alert.alert_id == alert_id:
                        alert.resolved = True
                        self.logger.info(f"Bottleneck alert {alert_id} marked as resolved")
                        return {'success': True, 'message': 'Alert resolved'}
                
                return {'success': False, 'message': 'Alert not found'}
                
        except Exception as e:
            self.logger.error(f"Error resolving bottleneck: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_function_profiles(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get function profiling results."""
        with self.lock:
            if function_name:
                return {
                    function_name: self.function_profiles.get(function_name, [])
                }
            else:
                return dict(self.function_profiles)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            current_metrics = self.get_current_metrics()
            recent_metrics = self.get_metrics_history(minutes=30)
            active_alerts = self.get_bottleneck_alerts(resolved=False)
            
            summary = {
                'profiler_status': 'running' if self.is_profiling else 'stopped',
                'profiler_stats': self.profiler_stats.copy(),
                'current_performance': asdict(current_metrics) if current_metrics else None,
                'baseline_comparison': self._compare_to_baseline(current_metrics) if current_metrics else None,
                'active_bottlenecks': len(active_alerts),
                'performance_trends': self._analyze_trends(recent_metrics),
                'system_health_score': self._calculate_health_score(current_metrics, active_alerts),
                'recommendations': self._generate_recommendations(current_metrics, active_alerts)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}
    
    def _compare_to_baseline(self, current_metrics: PerformanceMetrics) -> Dict[str, float]:
        """Compare current metrics to baseline."""
        if not self.baseline_metrics:
            return {}
        
        return {
            'cpu_change_percent': ((current_metrics.cpu_percent - self.baseline_metrics['cpu_percent']) / 
                                  self.baseline_metrics['cpu_percent'] * 100) if self.baseline_metrics['cpu_percent'] > 0 else 0,
            'memory_change_percent': ((current_metrics.memory_percent - self.baseline_metrics['memory_percent']) / 
                                    self.baseline_metrics['memory_percent'] * 100) if self.baseline_metrics['memory_percent'] > 0 else 0,
            'disk_io_change_percent': (((current_metrics.disk_read_mb_s + current_metrics.disk_write_mb_s) - 
                                      self.baseline_metrics['disk_io_mb_s']) / 
                                     self.baseline_metrics['disk_io_mb_s'] * 100) if self.baseline_metrics['disk_io_mb_s'] > 0 else 0
        }
    
    def _analyze_trends(self, metrics_history: List[PerformanceMetrics]) -> Dict[str, str]:
        """Analyze performance trends."""
        if len(metrics_history) < 5:
            return {'trend': 'insufficient_data'}
        
        # Calculate trends for key metrics
        cpu_values = [m.cpu_percent for m in metrics_history]
        memory_values = [m.memory_percent for m in metrics_history]
        
        cpu_trend = 'stable'
        memory_trend = 'stable'
        
        if len(cpu_values) >= 10:
            # Simple trend analysis
            first_half_cpu = statistics.mean(cpu_values[:len(cpu_values)//2])
            second_half_cpu = statistics.mean(cpu_values[len(cpu_values)//2:])
            
            if second_half_cpu > first_half_cpu * 1.1:
                cpu_trend = 'increasing'
            elif second_half_cpu < first_half_cpu * 0.9:
                cpu_trend = 'decreasing'
            
            first_half_memory = statistics.mean(memory_values[:len(memory_values)//2])
            second_half_memory = statistics.mean(memory_values[len(memory_values)//2:])
            
            if second_half_memory > first_half_memory * 1.05:
                memory_trend = 'increasing'
            elif second_half_memory < first_half_memory * 0.95:
                memory_trend = 'decreasing'
        
        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'data_points': len(metrics_history)
        }
    
    def _calculate_health_score(self, current_metrics: Optional[PerformanceMetrics], 
                               active_alerts: List[BottleneckAlert]) -> float:
        """Calculate overall system health score (0-100)."""
        if not current_metrics:
            return 50.0  # Default score when no data
        
        score = 100.0
        
        # CPU penalty
        if current_metrics.cpu_percent > 80:
            score -= (current_metrics.cpu_percent - 80) * 2
        
        # Memory penalty
        if current_metrics.memory_percent > 75:
            score -= (current_metrics.memory_percent - 75) * 1.5
        
        # Active alerts penalty
        for alert in active_alerts:
            if alert.severity == 'critical':
                score -= 20
            elif alert.severity == 'high':
                score -= 10
            elif alert.severity == 'medium':
                score -= 5
            else:
                score -= 2
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, current_metrics: Optional[PerformanceMetrics],
                                 active_alerts: List[BottleneckAlert]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not current_metrics:
            recommendations.append("Enable performance monitoring to get detailed recommendations")
            return recommendations
        
        # CPU recommendations
        if current_metrics.cpu_percent > 70:
            recommendations.append("Consider optimizing CPU-intensive operations or scaling horizontally")
        
        # Memory recommendations
        if current_metrics.memory_percent > 70:
            recommendations.append("Review memory usage patterns and implement caching strategies")
        
        # I/O recommendations
        total_disk_io = current_metrics.disk_read_mb_s + current_metrics.disk_write_mb_s
        if total_disk_io > 50:
            recommendations.append("Optimize database queries and implement disk caching")
        
        # Alert-specific recommendations
        for alert in active_alerts:
            if alert.recommendations:
                recommendations.extend(alert.recommendations[:1])  # Add top recommendation
        
        # General recommendations
        if not active_alerts and current_metrics.cpu_percent < 50 and current_metrics.memory_percent < 50:
            recommendations.append("System performance is optimal - consider enabling advanced monitoring")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def validate_performance_profiler(self) -> Dict[str, Any]:
        """Validate performance profiler functionality."""
        validation_results = {
            'component': 'PerformanceProfiler',
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: System Metrics Collection
            start_time = time.time()
            metrics = self._collect_system_metrics()
            collection_time_ms = (time.time() - start_time) * 1000
            
            validation_results['tests']['metrics_collection'] = {
                'status': 'pass',
                'collection_time_ms': collection_time_ms,
                'target_ms': 100,
                'details': f'Collected system metrics in {collection_time_ms:.1f}ms'
            }
            
            # Test 2: Bottleneck Detection
            start_time = time.time()
            self._detect_bottlenecks(metrics)
            detection_time_ms = (time.time() - start_time) * 1000
            
            validation_results['tests']['bottleneck_detection'] = {
                'status': 'pass',
                'detection_time_ms': detection_time_ms,
                'target_ms': self.bottleneck_detection_target_ms,
                'details': f'Bottleneck detection completed in {detection_time_ms:.1f}ms'
            }
            
            # Test 3: Function Profiling
            @self.profile_function
            def test_function():
                time.sleep(0.01)  # 10ms test function
                return "test_result"
            
            result = test_function()
            
            validation_results['tests']['function_profiling'] = {
                'status': 'pass' if result == "test_result" else 'fail',
                'details': 'Function profiling decorator working correctly'
            }
            
            # Test 4: Performance Summary Generation
            start_time = time.time()
            summary = self.get_performance_summary()
            summary_time_ms = (time.time() - start_time) * 1000
            
            validation_results['tests']['performance_summary'] = {
                'status': 'pass' if 'profiler_status' in summary else 'fail',
                'generation_time_ms': summary_time_ms,
                'details': f'Performance summary generated in {summary_time_ms:.1f}ms'
            }
            
            # Performance metrics
            validation_results['performance_metrics'] = {
                'profiler_stats': self.profiler_stats.copy(),
                'current_metrics': asdict(metrics),
                'health_score': self._calculate_health_score(metrics, []),
                'memory_tracing_enabled': self.config.enable_memory_tracing,
                'function_profiling_enabled': self.config.enable_function_profiling
            }
            
            # Overall status
            passed_tests = sum(1 for test in validation_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(validation_results['tests'])
            
            validation_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            validation_results['test_summary'] = f"{passed_tests}/{total_tests} tests passed"
            
            self.logger.info(f"Performance profiler validation completed: {validation_results['test_summary']}")
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"Performance profiler validation failed: {e}")
        
        return validation_results
    
    def shutdown(self):
        """Shutdown the performance profiler."""
        try:
            self.is_profiling = False
            
            if self.profiling_thread and self.profiling_thread.is_alive():
                self.profiling_thread.join(timeout=5.0)
            
            self.executor.shutdown(wait=True)
            
            if self.config.enable_memory_tracing and tracemalloc.is_tracing():
                tracemalloc.stop()
            
            self.logger.info("PerformanceProfiler shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during performance profiler shutdown: {e}")


# Utility functions
def profile_performance(config: Optional[ProfilerConfig] = None) -> PerformanceProfiler:
    """Create and configure a performance profiler instance."""
    return PerformanceProfiler(config)