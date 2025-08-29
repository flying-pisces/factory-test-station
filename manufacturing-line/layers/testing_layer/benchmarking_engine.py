"""
Benchmarking Engine for Week 7: Testing & Integration

This module implements comprehensive performance benchmarking and analysis system for the 
manufacturing line control system with load testing, stress testing, performance profiling, 
and optimization recommendations.

Performance Target: <100ms for comprehensive performance benchmark suite
Benchmarking Features: Load testing, stress testing, performance profiling, optimization recommendations
"""

import time
import logging
import asyncio
import json
import psutil
import statistics
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import concurrent.futures
import traceback
import gc
import sys
import os
from pathlib import Path
import uuid
import subprocess
import socket
import multiprocessing

# Week 7 testing layer integrations (forward references)
try:
    from layers.testing_layer.testing_engine import TestingEngine
    from layers.testing_layer.integration_engine import IntegrationEngine
    from layers.testing_layer.quality_assurance_engine import QualityAssuranceEngine
except ImportError:
    TestingEngine = None
    IntegrationEngine = None
    QualityAssuranceEngine = None

# Week 6 UI layer integrations
try:
    from layers.ui_layer.webui_engine import WebUIEngine
    from layers.ui_layer.visualization_engine import VisualizationEngine
    from layers.ui_layer.control_interface_engine import ControlInterfaceEngine
    from layers.ui_layer.user_management_engine import UserManagementEngine
    from layers.ui_layer.mobile_interface_engine import MobileInterfaceEngine
except ImportError:
    WebUIEngine = None
    VisualizationEngine = None
    ControlInterfaceEngine = None
    UserManagementEngine = None
    MobileInterfaceEngine = None

# Week 5 control layer integrations
try:
    from layers.control_layer.realtime_control_engine import RealTimeControlEngine
    from layers.control_layer.monitoring_engine import MonitoringEngine
    from layers.control_layer.orchestration_engine import OrchestrationEngine
    from layers.control_layer.data_stream_engine import DataStreamEngine
except ImportError:
    RealTimeControlEngine = None
    MonitoringEngine = None
    OrchestrationEngine = None
    DataStreamEngine = None


class BenchmarkType(Enum):
    """Benchmark test type definitions"""
    PERFORMANCE = "performance"
    LOAD = "load"
    STRESS = "stress"
    ENDURANCE = "endurance"
    SCALABILITY = "scalability"
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    DISK_IO = "disk_io"
    CONCURRENCY = "concurrency"


class BenchmarkStatus(Enum):
    """Benchmark execution status definitions"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    TIMEOUT = "timeout"


class LoadPattern(Enum):
    """Load test pattern definitions"""
    RAMP_UP = "ramp_up"
    CONSTANT = "constant"
    SPIKE = "spike"
    STEP = "step"
    SAWTOOTH = "sawtooth"
    RANDOM = "random"


class MetricType(Enum):
    """Performance metric type definitions"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_USAGE = "network_usage"
    DISK_USAGE = "disk_usage"
    CONCURRENT_USERS = "concurrent_users"


@dataclass
class BenchmarkMetrics:
    """Container for benchmark performance metrics"""
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_usage: float = 0.0
    disk_usage: float = 0.0
    concurrent_users: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BenchmarkConfiguration:
    """Benchmark test configuration"""
    benchmark_id: str
    benchmark_type: BenchmarkType
    duration: int  # seconds
    load_pattern: LoadPattern
    max_concurrent_users: int = 100
    ramp_up_time: int = 30  # seconds
    target_throughput: float = 1000.0  # requests/second
    resource_limits: Dict[str, float] = None
    custom_parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                'cpu_limit': 80.0,
                'memory_limit': 85.0,
                'network_limit': 100.0,
                'disk_limit': 90.0
            }
        if self.custom_parameters is None:
            self.custom_parameters = {}


@dataclass
class BenchmarkResult:
    """Container for benchmark test results"""
    benchmark_id: str
    configuration: BenchmarkConfiguration
    status: BenchmarkStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: float
    metrics_history: List[BenchmarkMetrics]
    summary_metrics: BenchmarkMetrics
    recommendations: List[str]
    performance_grade: str
    bottlenecks: List[str]
    resource_utilization: Dict[str, float]


class BenchmarkingEngine:
    """
    Comprehensive performance benchmarking and analysis system for manufacturing line control.
    
    Provides load testing, stress testing, performance profiling, and optimization recommendations
    with <100ms target for comprehensive performance benchmark suite.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the benchmarking engine with configuration."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Performance tracking
        self.benchmark_start_time = None
        self.benchmark_end_time = None
        self.metrics_history = deque(maxlen=10000)
        
        # Benchmark management
        self.active_benchmarks = {}
        self.benchmark_results = {}
        self.benchmark_queue = deque()
        
        # Resource monitoring
        self.resource_monitor = None
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Load generation
        self.load_generators = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=50)
        
        # Statistics tracking
        self.performance_stats = defaultdict(list)
        self.baseline_metrics = {}
        
        # Integration engines
        self.testing_engine = None
        self.integration_engine = None
        self.quality_engine = None
        
        # Layer engines
        self.ui_engines = {}
        self.control_engines = {}
        
        self.logger.info("BenchmarkingEngine initialized successfully")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the benchmarking engine."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def initialize_benchmarking(self) -> bool:
        """
        Initialize the benchmarking system and all components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            start_time = time.perf_counter()
            self.benchmark_start_time = start_time
            
            # Initialize resource monitoring
            await self._initialize_resource_monitoring()
            
            # Initialize load generators
            await self._initialize_load_generators()
            
            # Set up baseline metrics
            await self._establish_baseline_metrics()
            
            # Initialize integration engines
            await self._initialize_integration_engines()
            
            # Start monitoring threads
            await self._start_monitoring_threads()
            
            # Validate system readiness
            system_ready = await self._validate_system_readiness()
            
            end_time = time.perf_counter()
            initialization_time = (end_time - start_time) * 1000
            
            self.logger.info(f"BenchmarkingEngine initialized in {initialization_time:.2f}ms")
            
            return system_ready
            
        except Exception as e:
            self.logger.error(f"Failed to initialize benchmarking engine: {str(e)}")
            return False

    async def run_benchmark_suite(
        self, 
        configurations: List[BenchmarkConfiguration],
        parallel: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """
        Run a comprehensive benchmark suite.
        
        Args:
            configurations: List of benchmark configurations to run
            parallel: Whether to run benchmarks in parallel
            
        Returns:
            Dict mapping benchmark IDs to results
        """
        try:
            start_time = time.perf_counter()
            results = {}
            
            if parallel and len(configurations) > 1:
                # Run benchmarks in parallel
                tasks = []
                for config in configurations:
                    task = asyncio.create_task(self._run_single_benchmark(config))
                    tasks.append(task)
                
                benchmark_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(benchmark_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Benchmark {configurations[i].benchmark_id} failed: {result}")
                        continue
                    results[configurations[i].benchmark_id] = result
            else:
                # Run benchmarks sequentially
                for config in configurations:
                    result = await self._run_single_benchmark(config)
                    results[config.benchmark_id] = result
            
            # Generate suite summary
            suite_summary = await self._generate_suite_summary(results)
            
            end_time = time.perf_counter()
            suite_duration = (end_time - start_time) * 1000
            
            self.logger.info(f"Benchmark suite completed in {suite_duration:.2f}ms")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Benchmark suite execution failed: {str(e)}")
            return {}

    async def run_load_test(
        self,
        target_function: Callable,
        configuration: BenchmarkConfiguration
    ) -> BenchmarkResult:
        """
        Run a comprehensive load test on target function.
        
        Args:
            target_function: Function to load test
            configuration: Load test configuration
            
        Returns:
            BenchmarkResult containing test results
        """
        try:
            start_time = time.perf_counter()
            
            # Initialize load test
            result = await self._initialize_load_test(configuration)
            
            # Start resource monitoring
            self._start_load_monitoring(configuration.benchmark_id)
            
            # Generate load according to pattern
            load_metrics = await self._generate_load(
                target_function, 
                configuration
            )
            
            # Collect and analyze results
            await self._collect_load_results(configuration.benchmark_id, load_metrics)
            
            # Generate recommendations
            recommendations = await self._generate_load_recommendations(load_metrics)
            
            # Stop monitoring
            self._stop_load_monitoring(configuration.benchmark_id)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Build result
            result.end_time = datetime.now()
            result.duration = duration
            result.recommendations = recommendations
            result.status = BenchmarkStatus.COMPLETED
            
            self.logger.info(f"Load test completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Load test failed: {str(e)}")
            raise

    async def run_stress_test(
        self,
        target_system: Any,
        stress_level: float = 1.5,
        duration: int = 300
    ) -> BenchmarkResult:
        """
        Run stress test to determine system breaking points.
        
        Args:
            target_system: System to stress test
            stress_level: Multiplier for normal load (1.5 = 150% of normal)
            duration: Test duration in seconds
            
        Returns:
            BenchmarkResult containing stress test results
        """
        try:
            start_time = time.perf_counter()
            
            # Create stress test configuration
            config = BenchmarkConfiguration(
                benchmark_id=f"stress_test_{uuid.uuid4().hex[:8]}",
                benchmark_type=BenchmarkType.STRESS,
                duration=duration,
                load_pattern=LoadPattern.RAMP_UP,
                max_concurrent_users=int(100 * stress_level),
                target_throughput=1000.0 * stress_level
            )
            
            # Initialize stress test
            result = await self._initialize_stress_test(config, target_system)
            
            # Run progressive stress levels
            stress_results = await self._run_progressive_stress(
                target_system, 
                stress_level, 
                duration
            )
            
            # Analyze breaking points
            breaking_points = await self._analyze_breaking_points(stress_results)
            
            # Generate stress recommendations
            recommendations = await self._generate_stress_recommendations(
                breaking_points
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            result.end_time = datetime.now()
            result.duration = duration
            result.recommendations = recommendations
            result.bottlenecks = breaking_points
            result.status = BenchmarkStatus.COMPLETED
            
            self.logger.info(f"Stress test completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {str(e)}")
            raise

    async def profile_performance(
        self,
        target_function: Callable,
        iterations: int = 1000,
        profile_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Profile performance of target function with detailed analysis.
        
        Args:
            target_function: Function to profile
            iterations: Number of iterations to run
            profile_memory: Whether to include memory profiling
            
        Returns:
            Dict containing detailed performance profile
        """
        try:
            start_time = time.perf_counter()
            
            # Initialize profiling
            profile_data = {
                'execution_times': [],
                'memory_usage': [],
                'cpu_usage': [],
                'function_calls': 0,
                'exceptions': 0
            }
            
            # Run profiling iterations
            for i in range(iterations):
                iteration_start = time.perf_counter()
                
                try:
                    # Measure memory before
                    if profile_memory:
                        memory_before = psutil.Process().memory_info().rss
                    
                    # Execute function
                    result = await self._execute_profiled_function(target_function)
                    
                    # Measure memory after
                    if profile_memory:
                        memory_after = psutil.Process().memory_info().rss
                        memory_delta = memory_after - memory_before
                        profile_data['memory_usage'].append(memory_delta)
                    
                    iteration_end = time.perf_counter()
                    execution_time = (iteration_end - iteration_start) * 1000
                    profile_data['execution_times'].append(execution_time)
                    profile_data['function_calls'] += 1
                    
                except Exception as e:
                    profile_data['exceptions'] += 1
                    self.logger.warning(f"Exception in profiling iteration {i}: {e}")
            
            # Analyze profiling results
            analysis = await self._analyze_performance_profile(profile_data)
            
            end_time = time.perf_counter()
            profiling_duration = (end_time - start_time) * 1000
            
            self.logger.info(f"Performance profiling completed in {profiling_duration:.2f}ms")
            
            return {
                'raw_data': profile_data,
                'analysis': analysis,
                'profiling_duration': profiling_duration
            }
            
        except Exception as e:
            self.logger.error(f"Performance profiling failed: {str(e)}")
            return {}

    async def generate_optimization_recommendations(
        self,
        benchmark_results: Dict[str, BenchmarkResult]
    ) -> List[str]:
        """
        Generate optimization recommendations based on benchmark results.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            
        Returns:
            List of optimization recommendations
        """
        try:
            recommendations = []
            
            # Analyze performance patterns
            performance_issues = await self._identify_performance_issues(benchmark_results)
            
            # CPU optimization recommendations
            cpu_recommendations = await self._generate_cpu_recommendations(performance_issues)
            recommendations.extend(cpu_recommendations)
            
            # Memory optimization recommendations
            memory_recommendations = await self._generate_memory_recommendations(performance_issues)
            recommendations.extend(memory_recommendations)
            
            # Network optimization recommendations
            network_recommendations = await self._generate_network_recommendations(performance_issues)
            recommendations.extend(network_recommendations)
            
            # Concurrency optimization recommendations
            concurrency_recommendations = await self._generate_concurrency_recommendations(performance_issues)
            recommendations.extend(concurrency_recommendations)
            
            # Architecture optimization recommendations
            architecture_recommendations = await self._generate_architecture_recommendations(performance_issues)
            recommendations.extend(architecture_recommendations)
            
            self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization recommendations: {str(e)}")
            return []

    async def compare_benchmarks(
        self,
        baseline_results: Dict[str, BenchmarkResult],
        current_results: Dict[str, BenchmarkResult]
    ) -> Dict[str, Any]:
        """
        Compare benchmark results to identify performance regressions or improvements.
        
        Args:
            baseline_results: Baseline benchmark results
            current_results: Current benchmark results
            
        Returns:
            Dict containing comparison analysis
        """
        try:
            comparison = {
                'performance_changes': {},
                'regressions': [],
                'improvements': [],
                'overall_summary': {}
            }
            
            # Compare common benchmarks
            common_benchmarks = set(baseline_results.keys()) & set(current_results.keys())
            
            for benchmark_id in common_benchmarks:
                baseline = baseline_results[benchmark_id]
                current = current_results[benchmark_id]
                
                # Compare key metrics
                metric_comparison = await self._compare_benchmark_metrics(
                    baseline.summary_metrics,
                    current.summary_metrics
                )
                
                comparison['performance_changes'][benchmark_id] = metric_comparison
                
                # Identify regressions
                regressions = await self._identify_regressions(metric_comparison)
                if regressions:
                    comparison['regressions'].extend(regressions)
                
                # Identify improvements
                improvements = await self._identify_improvements(metric_comparison)
                if improvements:
                    comparison['improvements'].extend(improvements)
            
            # Generate overall summary
            comparison['overall_summary'] = await self._generate_comparison_summary(comparison)
            
            self.logger.info(f"Benchmark comparison completed for {len(common_benchmarks)} benchmarks")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Benchmark comparison failed: {str(e)}")
            return {}

    # Helper methods for internal operations

    async def _initialize_resource_monitoring(self):
        """Initialize resource monitoring systems."""
        try:
            # Set up system resource monitoring
            self.resource_monitor = {
                'cpu': psutil.cpu_percent(interval=None),
                'memory': psutil.virtual_memory(),
                'disk': psutil.disk_usage('/'),
                'network': psutil.net_io_counters()
            }
            
            self.logger.debug("Resource monitoring initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize resource monitoring: {str(e)}")
            raise

    async def _initialize_load_generators(self):
        """Initialize load generation systems."""
        try:
            # Set up different types of load generators
            self.load_generators = {
                'http': self._create_http_load_generator(),
                'websocket': self._create_websocket_load_generator(),
                'api': self._create_api_load_generator(),
                'database': self._create_database_load_generator()
            }
            
            self.logger.debug("Load generators initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize load generators: {str(e)}")
            raise

    async def _establish_baseline_metrics(self):
        """Establish baseline performance metrics."""
        try:
            # Collect baseline system metrics
            baseline = BenchmarkMetrics(
                cpu_usage=psutil.cpu_percent(interval=1),
                memory_usage=psutil.virtual_memory().percent,
                network_usage=0.0,
                disk_usage=psutil.disk_usage('/').percent,
                timestamp=datetime.now()
            )
            
            self.baseline_metrics = baseline
            self.logger.debug("Baseline metrics established")
            
        except Exception as e:
            self.logger.error(f"Failed to establish baseline metrics: {str(e)}")
            raise

    async def _initialize_integration_engines(self):
        """Initialize integration with other engines."""
        try:
            # Initialize testing engine integration
            if TestingEngine:
                self.testing_engine = TestingEngine()
            
            # Initialize integration engine integration
            if IntegrationEngine:
                self.integration_engine = IntegrationEngine()
            
            # Initialize UI engines
            if WebUIEngine:
                self.ui_engines['webui'] = WebUIEngine()
            
            # Initialize control engines
            if MonitoringEngine:
                self.control_engines['monitoring'] = MonitoringEngine()
            
            self.logger.debug("Integration engines initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration engines: {str(e)}")

    async def _start_monitoring_threads(self):
        """Start background monitoring threads."""
        try:
            self.monitoring_thread = threading.Thread(
                target=self._resource_monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
            self.logger.debug("Monitoring threads started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring threads: {str(e)}")
            raise

    async def _validate_system_readiness(self) -> bool:
        """Validate that the system is ready for benchmarking."""
        try:
            # Check resource availability
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            if cpu_usage > 80:
                self.logger.warning(f"High CPU usage detected: {cpu_usage}%")
                return False
            
            if memory_usage > 85:
                self.logger.warning(f"High memory usage detected: {memory_usage}%")
                return False
            
            # Check network connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
            except OSError:
                self.logger.warning("Network connectivity issues detected")
                return False
            
            self.logger.info("System readiness validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"System readiness validation failed: {str(e)}")
            return False

    async def _run_single_benchmark(self, config: BenchmarkConfiguration) -> BenchmarkResult:
        """Run a single benchmark test."""
        try:
            start_time = datetime.now()
            
            # Initialize result object
            result = BenchmarkResult(
                benchmark_id=config.benchmark_id,
                configuration=config,
                status=BenchmarkStatus.RUNNING,
                start_time=start_time,
                end_time=None,
                duration=0.0,
                metrics_history=[],
                summary_metrics=BenchmarkMetrics(),
                recommendations=[],
                performance_grade="",
                bottlenecks=[],
                resource_utilization={}
            )
            
            # Execute benchmark based on type
            if config.benchmark_type == BenchmarkType.PERFORMANCE:
                await self._run_performance_benchmark(config, result)
            elif config.benchmark_type == BenchmarkType.LOAD:
                await self._run_load_benchmark(config, result)
            elif config.benchmark_type == BenchmarkType.STRESS:
                await self._run_stress_benchmark(config, result)
            elif config.benchmark_type == BenchmarkType.ENDURANCE:
                await self._run_endurance_benchmark(config, result)
            else:
                await self._run_generic_benchmark(config, result)
            
            # Calculate final metrics
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.status = BenchmarkStatus.COMPLETED
            
            return result
            
        except Exception as e:
            self.logger.error(f"Benchmark {config.benchmark_id} failed: {str(e)}")
            result.status = BenchmarkStatus.FAILED
            return result

    def _create_http_load_generator(self):
        """Create HTTP load generator."""
        class HTTPLoadGenerator:
            def __init__(self, logger):
                self.logger = logger
                self.session = None
            
            async def generate_load(self, url, concurrent_users, duration):
                # Implementation for HTTP load generation
                pass
        
        return HTTPLoadGenerator(self.logger)

    def _create_websocket_load_generator(self):
        """Create WebSocket load generator."""
        class WebSocketLoadGenerator:
            def __init__(self, logger):
                self.logger = logger
            
            async def generate_load(self, endpoint, concurrent_connections, duration):
                # Implementation for WebSocket load generation
                pass
        
        return WebSocketLoadGenerator(self.logger)

    def _create_api_load_generator(self):
        """Create API load generator."""
        class APILoadGenerator:
            def __init__(self, logger):
                self.logger = logger
            
            async def generate_load(self, api_calls, rate, duration):
                # Implementation for API load generation
                pass
        
        return APILoadGenerator(self.logger)

    def _create_database_load_generator(self):
        """Create database load generator."""
        class DatabaseLoadGenerator:
            def __init__(self, logger):
                self.logger = logger
            
            async def generate_load(self, queries, connections, duration):
                # Implementation for database load generation
                pass
        
        return DatabaseLoadGenerator(self.logger)

    def _resource_monitoring_loop(self):
        """Background resource monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Collect resource metrics
                metrics = BenchmarkMetrics(
                    cpu_usage=psutil.cpu_percent(interval=None),
                    memory_usage=psutil.virtual_memory().percent,
                    network_usage=self._get_network_usage(),
                    disk_usage=psutil.disk_usage('/').percent,
                    timestamp=datetime.now()
                )
                
                self.metrics_history.append(metrics)
                
                # Sleep for monitoring interval
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {str(e)}")
                time.sleep(5)

    def _get_network_usage(self) -> float:
        """Calculate network usage percentage."""
        try:
            net_io = psutil.net_io_counters()
            # Calculate network usage based on bytes sent/received
            # This is a simplified calculation
            return min(100.0, (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024 * 100))
        except:
            return 0.0

    async def _generate_suite_summary(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary for benchmark suite."""
        try:
            total_tests = len(results)
            passed_tests = sum(1 for r in results.values() if r.status == BenchmarkStatus.COMPLETED)
            failed_tests = total_tests - passed_tests
            
            # Calculate average metrics
            avg_duration = statistics.mean([r.duration for r in results.values() if r.duration > 0])
            
            summary = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                'average_duration': avg_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate suite summary: {str(e)}")
            return {}

    async def shutdown(self):
        """Shutdown the benchmarking engine and cleanup resources."""
        try:
            self.benchmark_end_time = time.perf_counter()
            
            # Stop monitoring
            self.stop_monitoring.set()
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear data structures
            self.active_benchmarks.clear()
            self.load_generators.clear()
            
            total_time = (self.benchmark_end_time - self.benchmark_start_time) * 1000
            self.logger.info(f"BenchmarkingEngine shutdown completed. Total runtime: {total_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")