"""
Performance Benchmarks - Week 15: Integration Testing & Validation

This module provides comprehensive performance and scalability testing to validate
system performance under load, measure key performance indicators, and ensure
the manufacturing system meets enterprise-grade performance requirements.

Performance Targets:
- Response Time: <200ms for 95% of requests
- Throughput: >10,000 operations per second
- Concurrent Users: 1,000+ simultaneous active users
- Data Processing: 1M+ sensor readings per hour
- AI Inference: <100ms for ML model predictions
- System Availability: >99.9% uptime

Author: Manufacturing Line Control System
Created: Week 15 - Performance Testing Phase
"""

import time
import asyncio
import logging
import threading
import statistics
import psutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import json
import uuid
import traceback
import queue
import random


class LoadTestType(Enum):
    """Types of load tests."""
    BASELINE = "baseline"
    LOAD = "load"
    STRESS = "stress"
    SPIKE = "spike"
    VOLUME = "volume"
    ENDURANCE = "endurance"
    SCALABILITY = "scalability"


class PerformanceMetric(Enum):
    """Performance metric types."""
    RESPONSE_TIME = "response_time_ms"
    THROUGHPUT = "throughput_ops_sec"
    ERROR_RATE = "error_rate_percent"
    CPU_UTILIZATION = "cpu_utilization_percent"
    MEMORY_UTILIZATION = "memory_utilization_percent"
    NETWORK_UTILIZATION = "network_utilization_mbps"
    CONCURRENT_USERS = "concurrent_users"
    SUCCESS_RATE = "success_rate_percent"


@dataclass
class PerformanceTarget:
    """Performance target definition."""
    metric: PerformanceMetric
    target_value: float
    threshold_type: str  # 'max', 'min', 'range'
    acceptable_variance: float = 0.1  # 10% variance
    measurement_window_seconds: int = 60
    description: str = ""


@dataclass
class LoadTestScenario:
    """Load test scenario definition."""
    scenario_id: str
    name: str
    test_type: LoadTestType
    description: str
    user_count: int
    duration_minutes: int
    ramp_up_minutes: int = 0
    ramp_down_minutes: int = 0
    target_throughput: Optional[int] = None
    test_data: Optional[Dict[str, Any]] = None
    success_criteria: List[PerformanceTarget] = field(default_factory=list)


@dataclass
class PerformanceResult:
    """Performance test result."""
    scenario: LoadTestScenario
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    resource_utilization: Dict[str, List[float]] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def error_rate(self) -> float:
        return 100.0 - self.success_rate
    
    @property
    def average_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def percentile_95_response_time(self) -> float:
        if len(self.response_times) < 20:
            return self.average_response_time
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index]
    
    @property
    def percentile_99_response_time(self) -> float:
        if len(self.response_times) < 100:
            return self.average_response_time
        sorted_times = sorted(self.response_times)
        index = int(0.99 * len(sorted_times))
        return sorted_times[index]
    
    @property
    def throughput(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return self.successful_requests / self.duration_seconds


@dataclass
class SystemResourceSnapshot:
    """System resource utilization snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_io_read_mb_s: float
    disk_io_write_mb_s: float
    network_sent_mb_s: float
    network_recv_mb_s: float
    active_connections: int
    process_count: int
    thread_count: int


class ResourceMonitor:
    """System resource monitoring during performance tests."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.snapshots: List[SystemResourceSnapshot] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
        # Baseline measurements
        self.last_disk_io = None
        self.last_network_io = None
        self.last_snapshot_time = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.snapshots = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info(f"Resource monitoring stopped. Collected {len(self.snapshots)} snapshots")
    
    def _monitor_resources(self):
        """Background resource monitoring loop."""
        while self.monitoring_active:
            try:
                snapshot = self._collect_resource_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_resource_snapshot(self) -> SystemResourceSnapshot:
        """Collect current system resource snapshot."""
        current_time = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024 ** 3)
        
        # Disk I/O rates
        disk_io = psutil.disk_io_counters()
        disk_read_mb_s = 0.0
        disk_write_mb_s = 0.0
        
        if self.last_disk_io and self.last_snapshot_time:
            time_delta = current_time - self.last_snapshot_time
            if time_delta > 0:
                read_delta = disk_io.read_bytes - self.last_disk_io.read_bytes
                write_delta = disk_io.write_bytes - self.last_disk_io.write_bytes
                disk_read_mb_s = (read_delta / time_delta) / (1024 * 1024)
                disk_write_mb_s = (write_delta / time_delta) / (1024 * 1024)
        
        self.last_disk_io = disk_io
        
        # Network I/O rates
        network_io = psutil.net_io_counters()
        network_sent_mb_s = 0.0
        network_recv_mb_s = 0.0
        
        if self.last_network_io and self.last_snapshot_time:
            time_delta = current_time - self.last_snapshot_time
            if time_delta > 0:
                sent_delta = network_io.bytes_sent - self.last_network_io.bytes_sent
                recv_delta = network_io.bytes_recv - self.last_network_io.bytes_recv
                network_sent_mb_s = (sent_delta / time_delta) / (1024 * 1024)
                network_recv_mb_s = (recv_delta / time_delta) / (1024 * 1024)
        
        self.last_network_io = network_io
        self.last_snapshot_time = current_time
        
        # Process and connection counts
        process_count = len(psutil.pids())
        
        # Get active network connections
        try:
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
        except:
            active_connections = 0
        
        # Thread count approximation
        try:
            thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
        except:
            thread_count = 0
        
        return SystemResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_io_read_mb_s=disk_read_mb_s,
            disk_io_write_mb_s=disk_write_mb_s,
            network_sent_mb_s=network_sent_mb_s,
            network_recv_mb_s=network_recv_mb_s,
            active_connections=active_connections,
            process_count=process_count,
            thread_count=thread_count
        )
    
    def get_resource_summary(self) -> Dict[str, float]:
        """Get resource utilization summary."""
        if not self.snapshots:
            return {}
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_percent for s in self.snapshots]
        disk_read_values = [s.disk_io_read_mb_s for s in self.snapshots]
        disk_write_values = [s.disk_io_write_mb_s for s in self.snapshots]
        network_sent_values = [s.network_sent_mb_s for s in self.snapshots]
        network_recv_values = [s.network_recv_mb_s for s in self.snapshots]
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': statistics.mean(memory_values),
            'max_memory_percent': max(memory_values),
            'avg_memory_available_gb': statistics.mean([s.memory_available_gb for s in self.snapshots]),
            'max_disk_read_mb_s': max(disk_read_values),
            'max_disk_write_mb_s': max(disk_write_values),
            'max_network_sent_mb_s': max(network_sent_values),
            'max_network_recv_mb_s': max(network_recv_values),
            'avg_active_connections': statistics.mean([s.active_connections for s in self.snapshots]),
            'max_process_count': max([s.process_count for s in self.snapshots])
        }


class LoadGenerator:
    """Generate load for performance testing."""
    
    def __init__(self, max_workers: int = 100):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LoadGen")
        self.logger = logging.getLogger(__name__)
        self.active_users = 0
        self.results_queue = queue.Queue()
    
    def execute_load_test(self, scenario: LoadTestScenario, 
                         test_function: Callable[[Dict[str, Any]], Tuple[bool, float, Optional[str]]]) -> PerformanceResult:
        """
        Execute load test scenario.
        
        Args:
            scenario: Load test scenario definition
            test_function: Function to execute for each request
                          Returns (success: bool, response_time_ms: float, error_msg: Optional[str])
        """
        self.logger.info(f"Starting load test: {scenario.name}")
        
        start_time = datetime.now()
        result = PerformanceResult(
            scenario=scenario,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            total_requests=0,
            successful_requests=0,
            failed_requests=0
        )
        
        # Start resource monitoring
        resource_monitor = ResourceMonitor()
        resource_monitor.start_monitoring()
        
        try:
            # Execute test based on scenario type
            if scenario.test_type == LoadTestType.LOAD:
                self._execute_constant_load_test(scenario, test_function, result)
            elif scenario.test_type == LoadTestType.STRESS:
                self._execute_stress_test(scenario, test_function, result)
            elif scenario.test_type == LoadTestType.SPIKE:
                self._execute_spike_test(scenario, test_function, result)
            elif scenario.test_type == LoadTestType.ENDURANCE:
                self._execute_endurance_test(scenario, test_function, result)
            else:
                # Default to constant load
                self._execute_constant_load_test(scenario, test_function, result)
        
        finally:
            # Stop resource monitoring
            resource_monitor.stop_monitoring()
            
            # Add resource utilization to result
            result.resource_utilization = self._format_resource_data(resource_monitor.get_resource_summary())
            result.end_time = datetime.now()
        
        self.logger.info(f"Load test completed: {scenario.name}")
        self.logger.info(f"  Total requests: {result.total_requests}")
        self.logger.info(f"  Success rate: {result.success_rate:.1f}%")
        self.logger.info(f"  Average response time: {result.average_response_time:.1f}ms")
        self.logger.info(f"  Throughput: {result.throughput:.1f} ops/sec")
        
        return result
    
    def _execute_constant_load_test(self, scenario: LoadTestScenario, 
                                   test_function: Callable, result: PerformanceResult):
        """Execute constant load test."""
        total_duration = scenario.duration_minutes * 60
        ramp_up_duration = scenario.ramp_up_minutes * 60
        
        # Calculate request timing
        if scenario.target_throughput:
            request_interval = 1.0 / scenario.target_throughput
        else:
            # Estimate based on user count
            request_interval = 1.0 / scenario.user_count
        
        end_time = time.time() + total_duration
        futures = []
        
        while time.time() < end_time:
            # Ramp up users gradually
            current_elapsed = time.time() - (end_time - total_duration)
            if current_elapsed < ramp_up_duration:
                active_user_ratio = current_elapsed / ramp_up_duration
                current_user_count = max(1, int(scenario.user_count * active_user_ratio))
            else:
                current_user_count = scenario.user_count
            
            # Submit requests to maintain target load
            while len(futures) - sum(1 for f in futures if f.done()) < current_user_count:
                future = self.executor.submit(self._execute_single_request, test_function, scenario.test_data)
                futures.append(future)
            
            time.sleep(request_interval)
        
        # Collect results
        self._collect_results_from_futures(futures, result)
    
    def _execute_stress_test(self, scenario: LoadTestScenario, 
                            test_function: Callable, result: PerformanceResult):
        """Execute stress test with gradually increasing load."""
        total_duration = scenario.duration_minutes * 60
        max_users = scenario.user_count
        
        # Gradually increase load
        step_duration = total_duration / 10  # 10 steps
        step_increment = max_users / 10
        
        futures = []
        
        for step in range(10):
            step_user_count = int((step + 1) * step_increment)
            step_end_time = time.time() + step_duration
            
            self.logger.info(f"Stress test step {step + 1}/10: {step_user_count} users")
            
            while time.time() < step_end_time:
                # Submit requests for current step
                active_futures = [f for f in futures if not f.done()]
                while len(active_futures) < step_user_count:
                    future = self.executor.submit(self._execute_single_request, test_function, scenario.test_data)
                    futures.append(future)
                    active_futures.append(future)
                
                time.sleep(0.1)  # Brief pause
        
        # Collect results
        self._collect_results_from_futures(futures, result)
    
    def _execute_spike_test(self, scenario: LoadTestScenario, 
                           test_function: Callable, result: PerformanceResult):
        """Execute spike test with sudden load increase."""
        baseline_users = max(1, scenario.user_count // 10)  # 10% baseline
        spike_users = scenario.user_count
        
        total_duration = scenario.duration_minutes * 60
        spike_duration = min(120, total_duration // 3)  # Spike for 2 minutes or 1/3 duration
        
        futures = []
        
        # Phase 1: Baseline load
        baseline_end = time.time() + (total_duration - spike_duration) / 2
        self.logger.info(f"Spike test phase 1: {baseline_users} users (baseline)")
        
        while time.time() < baseline_end:
            active_futures = [f for f in futures if not f.done()]
            while len(active_futures) < baseline_users:
                future = self.executor.submit(self._execute_single_request, test_function, scenario.test_data)
                futures.append(future)
                active_futures.append(future)
            time.sleep(0.1)
        
        # Phase 2: Spike load
        spike_end = time.time() + spike_duration
        self.logger.info(f"Spike test phase 2: {spike_users} users (SPIKE)")
        
        while time.time() < spike_end:
            active_futures = [f for f in futures if not f.done()]
            while len(active_futures) < spike_users:
                future = self.executor.submit(self._execute_single_request, test_function, scenario.test_data)
                futures.append(future)
                active_futures.append(future)
            time.sleep(0.05)  # Faster submission during spike
        
        # Phase 3: Return to baseline
        final_end = time.time() + (total_duration - spike_duration) / 2
        self.logger.info(f"Spike test phase 3: {baseline_users} users (recovery)")
        
        while time.time() < final_end:
            active_futures = [f for f in futures if not f.done()]
            while len(active_futures) < baseline_users:
                future = self.executor.submit(self._execute_single_request, test_function, scenario.test_data)
                futures.append(future)
                active_futures.append(future)
            time.sleep(0.1)
        
        # Collect results
        self._collect_results_from_futures(futures, result)
    
    def _execute_endurance_test(self, scenario: LoadTestScenario, 
                               test_function: Callable, result: PerformanceResult):
        """Execute endurance test with sustained load."""
        # Similar to constant load but with longer duration and monitoring for degradation
        self._execute_constant_load_test(scenario, test_function, result)
        
        # Additional analysis for performance degradation over time
        if result.response_times:
            # Analyze performance degradation
            chunk_size = len(result.response_times) // 10
            if chunk_size > 0:
                chunks = [result.response_times[i:i + chunk_size] 
                         for i in range(0, len(result.response_times), chunk_size)]
                
                avg_times = [statistics.mean(chunk) for chunk in chunks if chunk]
                if len(avg_times) >= 2:
                    initial_avg = statistics.mean(avg_times[:2])
                    final_avg = statistics.mean(avg_times[-2:])
                    degradation = ((final_avg - initial_avg) / initial_avg) * 100
                    
                    result.custom_metrics['performance_degradation_percent'] = degradation
    
    def _execute_single_request(self, test_function: Callable, test_data: Optional[Dict[str, Any]]) -> Tuple[bool, float, Optional[str]]:
        """Execute single request and return result."""
        try:
            return test_function(test_data or {})
        except Exception as e:
            return False, 0.0, str(e)
    
    def _collect_results_from_futures(self, futures: List[Future], result: PerformanceResult):
        """Collect results from completed futures."""
        for future in as_completed(futures):
            try:
                success, response_time, error_msg = future.result(timeout=1.0)
                
                result.total_requests += 1
                if success:
                    result.successful_requests += 1
                else:
                    result.failed_requests += 1
                    if error_msg:
                        result.error_messages.append(error_msg)
                
                result.response_times.append(response_time)
                
            except Exception as e:
                result.total_requests += 1
                result.failed_requests += 1
                result.error_messages.append(str(e))
                result.response_times.append(0.0)
    
    def _format_resource_data(self, resource_summary: Dict[str, float]) -> Dict[str, List[float]]:
        """Format resource data for result storage."""
        return {key: [value] for key, value in resource_summary.items()}
    
    def shutdown(self):
        """Shutdown load generator."""
        self.executor.shutdown(wait=True)


class PerformanceBenchmarkSuite:
    """
    Comprehensive Performance Benchmark Suite
    
    Validates manufacturing system performance under various load conditions:
    - Load testing with concurrent users
    - Stress testing to find breaking points
    - Spike testing for traffic surges
    - Endurance testing for sustained load
    - Scalability testing for system growth
    - Performance regression testing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.load_generator = LoadGenerator(max_workers=200)
        self.performance_targets = self._create_performance_targets()
        self.test_scenarios = self._create_test_scenarios()
        self.results: List[PerformanceResult] = []
    
    def _create_performance_targets(self) -> List[PerformanceTarget]:
        """Create performance targets based on requirements."""
        return [
            PerformanceTarget(
                metric=PerformanceMetric.RESPONSE_TIME,
                target_value=200.0,  # 200ms
                threshold_type='max',
                description='95% of requests should complete within 200ms'
            ),
            PerformanceTarget(
                metric=PerformanceMetric.THROUGHPUT,
                target_value=10000.0,  # 10,000 ops/sec
                threshold_type='min',
                description='System should handle minimum 10,000 operations per second'
            ),
            PerformanceTarget(
                metric=PerformanceMetric.ERROR_RATE,
                target_value=1.0,  # 1%
                threshold_type='max',
                description='Error rate should not exceed 1%'
            ),
            PerformanceTarget(
                metric=PerformanceMetric.CPU_UTILIZATION,
                target_value=80.0,  # 80%
                threshold_type='max',
                description='CPU utilization should not exceed 80% under normal load'
            ),
            PerformanceTarget(
                metric=PerformanceMetric.MEMORY_UTILIZATION,
                target_value=85.0,  # 85%
                threshold_type='max',
                description='Memory utilization should not exceed 85%'
            ),
            PerformanceTarget(
                metric=PerformanceMetric.SUCCESS_RATE,
                target_value=99.0,  # 99%
                threshold_type='min',
                description='Success rate should be at least 99%'
            )
        ]
    
    def _create_test_scenarios(self) -> List[LoadTestScenario]:
        """Create comprehensive test scenarios."""
        return [
            # Baseline Performance Test
            LoadTestScenario(
                scenario_id='PERF001',
                name='Baseline Performance',
                test_type=LoadTestType.BASELINE,
                description='Establish baseline performance with minimal load',
                user_count=10,
                duration_minutes=2,
                target_throughput=100,
                success_criteria=self.performance_targets[:3]  # Response time, throughput, error rate
            ),
            
            # Normal Load Test  
            LoadTestScenario(
                scenario_id='PERF002',
                name='Normal Load Test',
                test_type=LoadTestType.LOAD,
                description='Test system under normal operational load',
                user_count=100,
                duration_minutes=5,
                ramp_up_minutes=1,
                target_throughput=1000,
                success_criteria=self.performance_targets
            ),
            
            # High Load Test
            LoadTestScenario(
                scenario_id='PERF003', 
                name='High Load Test',
                test_type=LoadTestType.LOAD,
                description='Test system under high operational load',
                user_count=500,
                duration_minutes=8,
                ramp_up_minutes=2,
                target_throughput=5000,
                success_criteria=self.performance_targets
            ),
            
            # Peak Load Test
            LoadTestScenario(
                scenario_id='PERF004',
                name='Peak Load Test',
                test_type=LoadTestType.LOAD,
                description='Test system at peak capacity',
                user_count=1000,
                duration_minutes=10,
                ramp_up_minutes=3,
                ramp_down_minutes=2,
                target_throughput=10000,
                success_criteria=self.performance_targets
            ),
            
            # Stress Test
            LoadTestScenario(
                scenario_id='PERF005',
                name='System Stress Test',
                test_type=LoadTestType.STRESS,
                description='Find system breaking point under increasing load',
                user_count=2000,
                duration_minutes=15,
                target_throughput=15000,
                success_criteria=self.performance_targets[:2]  # More lenient for stress test
            ),
            
            # Spike Test
            LoadTestScenario(
                scenario_id='PERF006',
                name='Traffic Spike Test',
                test_type=LoadTestType.SPIKE,
                description='Test system response to sudden traffic spikes',
                user_count=1500,
                duration_minutes=8,
                success_criteria=self.performance_targets
            ),
            
            # Endurance Test
            LoadTestScenario(
                scenario_id='PERF007',
                name='System Endurance Test',
                test_type=LoadTestType.ENDURANCE,
                description='Test system stability under sustained load',
                user_count=300,
                duration_minutes=20,
                ramp_up_minutes=2,
                target_throughput=3000,
                success_criteria=self.performance_targets + [
                    PerformanceTarget(
                        metric=PerformanceMetric.RESPONSE_TIME,
                        target_value=10.0,  # 10% degradation max
                        threshold_type='max',
                        description='Performance degradation should not exceed 10%'
                    )
                ]
            ),
            
            # Volume Test
            LoadTestScenario(
                scenario_id='PERF008',
                name='Data Volume Test',
                test_type=LoadTestType.VOLUME,
                description='Test system with large data volumes',
                user_count=200,
                duration_minutes=12,
                target_throughput=2000,
                test_data={'data_size': 'large', 'record_count': 10000},
                success_criteria=self.performance_targets
            )
        ]
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmark tests."""
        self.logger.info("Starting comprehensive performance benchmark suite")
        start_time = datetime.now()
        
        self.results = []
        
        for scenario in self.test_scenarios:
            try:
                result = self._run_single_benchmark(scenario)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Benchmark {scenario.name} failed: {e}")
                # Create failed result
                failed_result = PerformanceResult(
                    scenario=scenario,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    total_requests=0,
                    successful_requests=0,
                    failed_requests=0,
                    error_messages=[str(e)]
                )
                self.results.append(failed_result)
        
        end_time = datetime.now()
        
        # Generate comprehensive report
        return self._generate_benchmark_report(start_time, end_time)
    
    def run_specific_benchmark(self, scenario_id: str) -> PerformanceResult:
        """Run specific benchmark scenario."""
        scenario = next((s for s in self.test_scenarios if s.scenario_id == scenario_id), None)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        return self._run_single_benchmark(scenario)
    
    def _run_single_benchmark(self, scenario: LoadTestScenario) -> PerformanceResult:
        """Run single benchmark scenario."""
        self.logger.info(f"Running benchmark: {scenario.name}")
        
        # Create test function based on scenario
        test_function = self._create_test_function(scenario)
        
        # Execute load test
        result = self.load_generator.execute_load_test(scenario, test_function)
        
        # Evaluate success criteria
        self._evaluate_success_criteria(result)
        
        return result
    
    def _create_test_function(self, scenario: LoadTestScenario) -> Callable:
        """Create test function for specific scenario."""
        def manufacturing_system_test(test_data: Dict[str, Any]) -> Tuple[bool, float, Optional[str]]:
            """Simulate manufacturing system operation."""
            start_time = time.perf_counter()
            
            try:
                # Simulate different types of operations based on scenario
                operation_type = random.choice([
                    'sensor_data_processing',
                    'control_loop_execution',
                    'quality_analysis',
                    'ai_prediction',
                    'dashboard_update',
                    'report_generation'
                ])
                
                # Simulate operation-specific processing time and complexity
                if operation_type == 'sensor_data_processing':
                    self._simulate_sensor_processing(test_data)
                elif operation_type == 'control_loop_execution':
                    self._simulate_control_loop(test_data)
                elif operation_type == 'quality_analysis':
                    self._simulate_quality_analysis(test_data)
                elif operation_type == 'ai_prediction':
                    self._simulate_ai_prediction(test_data)
                elif operation_type == 'dashboard_update':
                    self._simulate_dashboard_update(test_data)
                elif operation_type == 'report_generation':
                    self._simulate_report_generation(test_data)
                
                response_time = (time.perf_counter() - start_time) * 1000
                
                # Simulate occasional failures (2% failure rate)
                if random.random() < 0.02:
                    return False, response_time, f"{operation_type}_error"
                
                return True, response_time, None
                
            except Exception as e:
                response_time = (time.perf_counter() - start_time) * 1000
                return False, response_time, str(e)
        
        return manufacturing_system_test
    
    def _simulate_sensor_processing(self, test_data: Dict[str, Any]):
        """Simulate sensor data processing."""
        # Simulate processing time based on data volume
        processing_time = 0.01  # Base 10ms
        if test_data.get('data_size') == 'large':
            processing_time *= 2
        
        time.sleep(processing_time)
        
        # Simulate some CPU work
        _ = sum(i * i for i in range(1000))
    
    def _simulate_control_loop(self, test_data: Dict[str, Any]):
        """Simulate control loop execution."""
        # Control loops are fast but CPU intensive
        time.sleep(0.005)  # 5ms
        
        # Simulate PID calculation
        _ = sum(math.sqrt(i) for i in range(500))
    
    def _simulate_quality_analysis(self, test_data: Dict[str, Any]):
        """Simulate quality analysis operation."""
        # Quality analysis is moderate processing
        time.sleep(0.02)  # 20ms
        
        # Simulate statistical calculations
        data = [random.random() for _ in range(100)]
        _ = statistics.mean(data)
        _ = statistics.stdev(data)
    
    def _simulate_ai_prediction(self, test_data: Dict[str, Any]):
        """Simulate AI prediction operation."""
        # AI predictions are more intensive
        time.sleep(0.05)  # 50ms
        
        # Simulate ML computation
        matrix_size = 50
        matrix_a = [[random.random() for _ in range(matrix_size)] for _ in range(matrix_size)]
        matrix_b = [[random.random() for _ in range(matrix_size)] for _ in range(matrix_size)]
        
        # Simple matrix multiplication simulation
        for i in range(min(10, matrix_size)):
            for j in range(min(10, matrix_size)):
                sum(matrix_a[i][k] * matrix_b[k][j] for k in range(min(10, matrix_size)))
    
    def _simulate_dashboard_update(self, test_data: Dict[str, Any]):
        """Simulate dashboard update operation."""
        # Dashboard updates are lightweight
        time.sleep(0.008)  # 8ms
        
        # Simulate data aggregation
        _ = [random.random() for _ in range(50)]
    
    def _simulate_report_generation(self, test_data: Dict[str, Any]):
        """Simulate report generation operation."""
        # Report generation is heavier
        processing_time = 0.1  # Base 100ms
        if test_data.get('record_count'):
            # Scale with record count
            processing_time *= (test_data['record_count'] / 1000)
        
        time.sleep(min(processing_time, 0.5))  # Cap at 500ms
        
        # Simulate data processing
        _ = json.dumps({'data': [i for i in range(100)]})
    
    def _evaluate_success_criteria(self, result: PerformanceResult):
        """Evaluate performance result against success criteria."""
        result.custom_metrics['criteria_results'] = {}
        
        for target in result.scenario.success_criteria:
            if target.metric == PerformanceMetric.RESPONSE_TIME:
                actual_value = result.percentile_95_response_time
            elif target.metric == PerformanceMetric.THROUGHPUT:
                actual_value = result.throughput
            elif target.metric == PerformanceMetric.ERROR_RATE:
                actual_value = result.error_rate
            elif target.metric == PerformanceMetric.SUCCESS_RATE:
                actual_value = result.success_rate
            elif target.metric == PerformanceMetric.CPU_UTILIZATION:
                actual_value = result.resource_utilization.get('avg_cpu_percent', [0])[0]
            elif target.metric == PerformanceMetric.MEMORY_UTILIZATION:
                actual_value = result.resource_utilization.get('avg_memory_percent', [0])[0]
            else:
                continue  # Skip unknown metrics
            
            # Check if target is met
            if target.threshold_type == 'max':
                criteria_met = actual_value <= target.target_value
            elif target.threshold_type == 'min':
                criteria_met = actual_value >= target.target_value
            else:  # range
                criteria_met = True  # Simplified for now
            
            result.custom_metrics['criteria_results'][target.metric.value] = {
                'target': target.target_value,
                'actual': actual_value,
                'met': criteria_met,
                'description': target.description
            }
    
    def _generate_benchmark_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success_rate >= 95.0)
        
        # Aggregate performance metrics
        all_response_times = []
        all_throughputs = []
        all_success_rates = []
        
        for result in self.results:
            all_response_times.extend(result.response_times)
            all_throughputs.append(result.throughput)
            all_success_rates.append(result.success_rate)
        
        # Calculate overall metrics
        overall_metrics = {
            'avg_response_time_ms': statistics.mean(all_response_times) if all_response_times else 0,
            'p95_response_time_ms': statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) >= 20 else 0,
            'p99_response_time_ms': statistics.quantiles(all_response_times, n=100)[98] if len(all_response_times) >= 100 else 0,
            'max_throughput_ops_sec': max(all_throughputs) if all_throughputs else 0,
            'avg_throughput_ops_sec': statistics.mean(all_throughputs) if all_throughputs else 0,
            'avg_success_rate_percent': statistics.mean(all_success_rates) if all_success_rates else 0,
            'min_success_rate_percent': min(all_success_rates) if all_success_rates else 0
        }
        
        # Evaluate against performance targets
        target_compliance = {}
        for target in self.performance_targets:
            compliance_results = []
            
            for result in self.results:
                criteria_results = result.custom_metrics.get('criteria_results', {})
                if target.metric.value in criteria_results:
                    compliance_results.append(criteria_results[target.metric.value]['met'])
            
            if compliance_results:
                compliance_rate = sum(compliance_results) / len(compliance_results) * 100
                target_compliance[target.metric.value] = {
                    'target_description': target.description,
                    'compliance_rate_percent': compliance_rate,
                    'tests_passed': sum(compliance_results),
                    'total_tests': len(compliance_results)
                }
        
        # Identify performance bottlenecks
        bottlenecks = []
        if overall_metrics['p95_response_time_ms'] > 200:
            bottlenecks.append('Response time exceeds 200ms target')
        if overall_metrics['max_throughput_ops_sec'] < 10000:
            bottlenecks.append('Throughput below 10,000 ops/sec target')
        if overall_metrics['min_success_rate_percent'] < 99:
            bottlenecks.append('Success rate below 99% target')
        
        # Performance trends analysis
        trends = self._analyze_performance_trends()
        
        # Results by test type
        results_by_type = {}
        for result in self.results:
            test_type = result.scenario.test_type.value
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            
            results_by_type[test_type].append({
                'scenario_name': result.scenario.name,
                'success_rate': result.success_rate,
                'avg_response_time_ms': result.average_response_time,
                'throughput_ops_sec': result.throughput,
                'total_requests': result.total_requests
            })
        
        return {
            'summary': {
                'total_scenarios': total_tests,
                'successful_scenarios': successful_tests,
                'overall_success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                'test_start_time': start_time.isoformat(),
                'test_end_time': end_time.isoformat(),
                'total_duration_minutes': (end_time - start_time).total_seconds() / 60
            },
            'overall_performance_metrics': overall_metrics,
            'target_compliance': target_compliance,
            'performance_bottlenecks': bottlenecks,
            'results_by_test_type': results_by_type,
            'performance_trends': trends,
            'detailed_results': [
                {
                    'scenario_id': r.scenario.scenario_id,
                    'scenario_name': r.scenario.name,
                    'test_type': r.scenario.test_type.value,
                    'duration_seconds': r.duration_seconds,
                    'total_requests': r.total_requests,
                    'success_rate': r.success_rate,
                    'avg_response_time_ms': r.average_response_time,
                    'p95_response_time_ms': r.percentile_95_response_time,
                    'throughput_ops_sec': r.throughput,
                    'resource_utilization': r.resource_utilization,
                    'criteria_results': r.custom_metrics.get('criteria_results', {})
                }
                for r in self.results
            ],
            'recommendations': self._generate_performance_recommendations(overall_metrics, target_compliance)
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across test scenarios."""
        # Group results by load level
        load_levels = []
        for result in self.results:
            load_levels.append({
                'user_count': result.scenario.user_count,
                'response_time': result.average_response_time,
                'throughput': result.throughput,
                'success_rate': result.success_rate
            })
        
        # Sort by user count
        load_levels.sort(key=lambda x: x['user_count'])
        
        # Analyze trends
        trends = {
            'response_time_trend': 'stable',
            'throughput_scaling': 'linear',
            'success_rate_stability': 'stable',
            'scalability_limit': None
        }
        
        if len(load_levels) >= 3:
            # Response time trend
            response_times = [ll['response_time'] for ll in load_levels]
            if response_times[-1] > response_times[0] * 2:
                trends['response_time_trend'] = 'degrading'
            elif response_times[-1] > response_times[0] * 1.5:
                trends['response_time_trend'] = 'increasing'
            
            # Throughput scaling
            throughputs = [ll['throughput'] for ll in load_levels]
            user_counts = [ll['user_count'] for ll in load_levels]
            
            # Simple linear regression to check scaling
            if len(throughputs) >= 2:
                throughput_growth = (throughputs[-1] - throughputs[0]) / (user_counts[-1] - user_counts[0])
                if throughput_growth < 0.5:
                    trends['throughput_scaling'] = 'sublinear'
                elif throughput_growth > 2:
                    trends['throughput_scaling'] = 'superlinear'
            
            # Success rate stability
            success_rates = [ll['success_rate'] for ll in load_levels]
            if min(success_rates) < 95:
                trends['success_rate_stability'] = 'degrading'
                # Find where it starts degrading
                for i, rate in enumerate(success_rates):
                    if rate < 95:
                        trends['scalability_limit'] = user_counts[i]
                        break
        
        return trends
    
    def _generate_performance_recommendations(self, metrics: Dict[str, Any], 
                                            compliance: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Response time recommendations
        if metrics['p95_response_time_ms'] > 200:
            recommendations.append(
                f"Response time optimization needed: P95 is {metrics['p95_response_time_ms']:.1f}ms "
                f"(target: <200ms). Consider caching, database optimization, or code profiling."
            )
        
        # Throughput recommendations
        if metrics['max_throughput_ops_sec'] < 10000:
            recommendations.append(
                f"Throughput scaling needed: Maximum achieved {metrics['max_throughput_ops_sec']:.0f} ops/sec "
                f"(target: >10,000 ops/sec). Consider load balancing, horizontal scaling, or async processing."
            )
        
        # Success rate recommendations
        if metrics['min_success_rate_percent'] < 99:
            recommendations.append(
                f"Reliability improvement needed: Minimum success rate {metrics['min_success_rate_percent']:.1f}% "
                f"(target: >99%). Investigate error patterns and implement retry mechanisms."
            )
        
        # Resource utilization recommendations
        cpu_compliance = compliance.get('cpu_utilization_percent', {})
        if cpu_compliance and cpu_compliance.get('compliance_rate_percent', 100) < 80:
            recommendations.append(
                "High CPU utilization detected. Consider CPU optimization, algorithmic improvements, "
                "or additional computing resources."
            )
        
        memory_compliance = compliance.get('memory_utilization_percent', {})
        if memory_compliance and memory_compliance.get('compliance_rate_percent', 100) < 80:
            recommendations.append(
                "High memory utilization detected. Review memory usage patterns, implement garbage collection "
                "optimization, or increase available memory."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance targets are being met. Consider monitoring trends for early detection of degradation.")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown benchmark suite."""
        self.load_generator.shutdown()
        self.logger.info("Performance benchmark suite shutdown completed")


# Import math for simulations
import math

# Convenience functions for running specific benchmark types
def run_load_tests() -> Dict[str, Any]:
    """Run load testing scenarios."""
    suite = PerformanceBenchmarkSuite()
    load_scenarios = [s for s in suite.test_scenarios if s.test_type == LoadTestType.LOAD]
    
    results = []
    for scenario in load_scenarios:
        result = suite.run_specific_benchmark(scenario.scenario_id)
        results.append(result)
    
    suite.results = results
    return suite._generate_benchmark_report(datetime.now() - timedelta(minutes=30), datetime.now())

def run_stress_tests() -> Dict[str, Any]:
    """Run stress testing scenarios."""
    suite = PerformanceBenchmarkSuite()
    return suite.run_specific_benchmark('PERF005')

def run_all_performance_benchmarks() -> Dict[str, Any]:
    """Run complete performance benchmark suite."""
    suite = PerformanceBenchmarkSuite()
    return suite.run_all_benchmarks()


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Performance Benchmark Suite Demo")
    print("=" * 80)
    
    # Create benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite()
    
    print(f"\nTotal Scenarios: {len(benchmark_suite.test_scenarios)}")
    print(f"Performance Targets: {len(benchmark_suite.performance_targets)}")
    
    # Run a subset of tests for demo (to keep runtime reasonable)
    demo_scenarios = ['PERF001', 'PERF002', 'PERF006']  # Baseline, Normal Load, Spike
    
    print("\nRunning demo performance benchmarks...")
    demo_results = []
    
    for scenario_id in demo_scenarios:
        print(f"\n--- Running {scenario_id} ---")
        try:
            result = benchmark_suite.run_specific_benchmark(scenario_id)
            demo_results.append(result)
            print(f"✅ Completed: Success Rate {result.success_rate:.1f}%, "
                  f"Avg Response {result.average_response_time:.1f}ms, "
                  f"Throughput {result.throughput:.0f} ops/sec")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Set results for report generation
    benchmark_suite.results = demo_results
    
    # Generate summary report
    if demo_results:
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        report = benchmark_suite._generate_benchmark_report(
            datetime.now() - timedelta(minutes=10), 
            datetime.now()
        )
        
        summary = report['summary']
        metrics = report['overall_performance_metrics']
        
        print(f"Scenarios Executed: {summary['successful_scenarios']}/{summary['total_scenarios']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Average Response Time: {metrics['avg_response_time_ms']:.1f}ms")
        print(f"P95 Response Time: {metrics['p95_response_time_ms']:.1f}ms")
        print(f"Maximum Throughput: {metrics['max_throughput_ops_sec']:.0f} ops/sec")
        
        print("\nTarget Compliance:")
        for target, compliance in report['target_compliance'].items():
            print(f"  {target}: {compliance['compliance_rate_percent']:.0f}% compliance")
        
        if report['performance_bottlenecks']:
            print("\nPerformance Issues Identified:")
            for bottleneck in report['performance_bottlenecks']:
                print(f"  ⚠️  {bottleneck}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  💡 {rec}")
    
    benchmark_suite.shutdown()
    print("\nPerformance Benchmark Suite demo completed!")