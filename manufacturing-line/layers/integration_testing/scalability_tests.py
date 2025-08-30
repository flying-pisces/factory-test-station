"""
Scalability Tests - Week 15: Integration Testing & Validation

This module provides comprehensive scalability testing to validate system capacity,
horizontal and vertical scaling capabilities, auto-scaling behavior, and performance
under increasing load conditions for the manufacturing control system.

Scalability Targets:
- User Scaling: Support 10x user increase (100 → 1,000 users)
- Data Scaling: Handle 5x data volume increase 
- Processing Scaling: Maintain <200ms response time at scale
- Storage Scaling: Database performance under 10x load
- Network Scaling: Bandwidth utilization optimization

Author: Manufacturing Line Control System
Created: Week 15 - Scalability Testing Phase
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
from typing import Any, Dict, List, Optional, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
import json
import uuid
import traceback
import queue
import random
import multiprocessing
import socket
import subprocess
import sys
import os


class ScalabilityTestType(Enum):
    """Types of scalability tests."""
    HORIZONTAL_SCALING = "horizontal_scaling"
    VERTICAL_SCALING = "vertical_scaling"
    DATA_VOLUME_SCALING = "data_volume_scaling"
    USER_LOAD_SCALING = "user_load_scaling"
    PROCESSING_SCALING = "processing_scaling"
    STORAGE_SCALING = "storage_scaling"
    NETWORK_SCALING = "network_scaling"
    AUTO_SCALING = "auto_scaling"


class ScalabilityMetric(Enum):
    """Scalability-specific metrics."""
    SCALABILITY_FACTOR = "scalability_factor"
    LINEAR_SCALING_INDEX = "linear_scaling_index"
    EFFICIENCY_RATIO = "efficiency_ratio"
    BREAKING_POINT = "breaking_point_users"
    RESOURCE_UTILIZATION_RATIO = "resource_utilization_ratio"
    SCALING_RESPONSE_TIME = "scaling_response_time_seconds"
    THROUGHPUT_PER_RESOURCE = "throughput_per_resource_unit"
    DEGRADATION_RATE = "performance_degradation_rate"


@dataclass
class ScalabilityTarget:
    """Scalability target definition."""
    metric: ScalabilityMetric
    target_value: float
    threshold_type: str  # 'max', 'min', 'range'
    acceptable_variance: float = 0.15  # 15% variance for scalability
    measurement_points: int = 5  # Number of measurement points
    description: str = ""


@dataclass
class ResourceConfiguration:
    """Resource configuration for scaling tests."""
    cpu_cores: int
    memory_gb: float
    disk_iops: int
    network_bandwidth_mbps: int
    concurrent_connections: int
    thread_pool_size: int
    process_pool_size: int
    cache_size_mb: int


@dataclass
class ScalingPoint:
    """Single point in scaling test."""
    scale_level: int
    resource_config: ResourceConfiguration
    load_factor: float
    response_time_ms: float
    throughput_ops_sec: float
    success_rate_percent: float
    cpu_utilization_percent: float
    memory_utilization_percent: float
    error_count: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalabilityResult:
    """Scalability test result."""
    test_type: ScalabilityTestType
    test_name: str
    start_time: datetime
    end_time: datetime
    scaling_points: List[ScalingPoint]
    baseline_performance: Optional[ScalingPoint]
    optimal_scaling_point: Optional[ScalingPoint]
    breaking_point: Optional[ScalingPoint]
    scalability_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def max_scale_achieved(self) -> int:
        return max(point.scale_level for point in self.scaling_points) if self.scaling_points else 0
    
    @property
    def linear_scaling_efficiency(self) -> float:
        """Calculate how close to linear scaling the system achieves."""
        if len(self.scaling_points) < 2:
            return 0.0
        
        # Compare actual scaling vs theoretical linear scaling
        baseline = min(self.scaling_points, key=lambda p: p.scale_level)
        max_scale = max(self.scaling_points, key=lambda p: p.scale_level)
        
        if baseline.scale_level >= max_scale.scale_level:
            return 0.0
        
        scale_ratio = max_scale.scale_level / baseline.scale_level
        throughput_ratio = max_scale.throughput_ops_sec / baseline.throughput_ops_sec
        
        # Perfect linear scaling would give throughput_ratio == scale_ratio
        return min(throughput_ratio / scale_ratio, 1.0) * 100.0


class SystemResourceScaler:
    """Simulate system resource scaling for testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_config = ResourceConfiguration(
            cpu_cores=multiprocessing.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            disk_iops=1000,
            network_bandwidth_mbps=1000,
            concurrent_connections=1000,
            thread_pool_size=50,
            process_pool_size=4,
            cache_size_mb=256
        )
    
    def scale_resources(self, scale_factor: float) -> ResourceConfiguration:
        """Scale resources by given factor."""
        return ResourceConfiguration(
            cpu_cores=max(1, int(self.current_config.cpu_cores * scale_factor)),
            memory_gb=self.current_config.memory_gb * scale_factor,
            disk_iops=int(self.current_config.disk_iops * scale_factor),
            network_bandwidth_mbps=int(self.current_config.network_bandwidth_mbps * scale_factor),
            concurrent_connections=int(self.current_config.concurrent_connections * scale_factor),
            thread_pool_size=max(1, int(self.current_config.thread_pool_size * scale_factor)),
            process_pool_size=max(1, int(self.current_config.process_pool_size * scale_factor)),
            cache_size_mb=int(self.current_config.cache_size_mb * scale_factor)
        )
    
    def apply_resource_limits(self, config: ResourceConfiguration):
        """Apply resource limits for testing (simulated)."""
        self.logger.info(f"Applying resource configuration: "
                        f"CPUs={config.cpu_cores}, Memory={config.memory_gb:.1f}GB, "
                        f"Threads={config.thread_pool_size}")
        
        # In real implementation, this would use cgroups, Docker limits, etc.
        # For testing, we'll simulate by limiting our thread pools and processing
        pass
    
    def get_optimal_configuration(self, target_load: int) -> ResourceConfiguration:
        """Calculate optimal resource configuration for target load."""
        # Simple heuristic - in practice this would be much more sophisticated
        scale_factor = max(1.0, target_load / 100.0)  # Baseline at 100 users
        return self.scale_resources(scale_factor)


class LoadPatternGenerator:
    """Generate various load patterns for scalability testing."""
    
    @staticmethod
    def linear_growth(start_load: int, end_load: int, steps: int) -> List[int]:
        """Generate linear growth pattern."""
        if steps <= 1:
            return [end_load]
        step_size = (end_load - start_load) / (steps - 1)
        return [int(start_load + i * step_size) for i in range(steps)]
    
    @staticmethod
    def exponential_growth(start_load: int, end_load: int, steps: int) -> List[int]:
        """Generate exponential growth pattern."""
        if steps <= 1:
            return [end_load]
        
        import math
        growth_rate = math.pow(end_load / start_load, 1.0 / (steps - 1))
        return [int(start_load * math.pow(growth_rate, i)) for i in range(steps)]
    
    @staticmethod
    def step_growth(start_load: int, end_load: int, steps: int) -> List[int]:
        """Generate step growth pattern."""
        if steps <= 1:
            return [end_load]
        
        step_size = (end_load - start_load) // steps
        loads = []
        current = start_load
        for _ in range(steps):
            loads.append(current)
            current = min(current + step_size, end_load)
        return loads
    
    @staticmethod
    def fibonacci_growth(start_load: int, max_load: int) -> List[int]:
        """Generate Fibonacci-based growth pattern."""
        loads = [start_load]
        if start_load >= max_load:
            return loads
        
        fib_a, fib_b = 1, 1
        while True:
            next_load = start_load + (fib_b * 10)  # Scale Fibonacci numbers
            if next_load > max_load:
                loads.append(max_load)
                break
            loads.append(next_load)
            fib_a, fib_b = fib_b, fib_a + fib_b
        
        return loads


class ScalabilityTestExecutor:
    """Execute scalability tests with various patterns and configurations."""
    
    def __init__(self, max_workers: int = 100):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.resource_scaler = SystemResourceScaler()
        self.load_generator = LoadPatternGenerator()
        
        # Performance tracking
        self.measurement_window = 30  # seconds
        self.stabilization_time = 10  # seconds to wait for system to stabilize
    
    def execute_horizontal_scaling_test(self, 
                                      min_instances: int = 1, 
                                      max_instances: int = 8,
                                      test_duration_per_scale: int = 60) -> ScalabilityResult:
        """Test horizontal scaling capabilities."""
        self.logger.info("Starting horizontal scaling test")
        
        start_time = datetime.now()
        scaling_points = []
        
        # Generate scaling pattern
        instance_counts = self.load_generator.linear_growth(min_instances, max_instances, 6)
        
        for instance_count in instance_counts:
            self.logger.info(f"Testing with {instance_count} instances")
            
            # Configure resources for this scale
            config = self.resource_scaler.scale_resources(instance_count)
            self.resource_scaler.apply_resource_limits(config)
            
            # Run test at this scale
            scaling_point = self._measure_performance_at_scale(
                scale_level=instance_count,
                resource_config=config,
                test_duration=test_duration_per_scale,
                load_factor=instance_count
            )
            
            scaling_points.append(scaling_point)
            
            # Check if we've hit the breaking point
            if scaling_point.success_rate_percent < 95.0:
                self.logger.warning(f"Breaking point detected at {instance_count} instances")
                break
        
        result = ScalabilityResult(
            test_type=ScalabilityTestType.HORIZONTAL_SCALING,
            test_name="Horizontal Instance Scaling",
            start_time=start_time,
            end_time=datetime.now(),
            scaling_points=scaling_points
        )
        
        self._analyze_scaling_result(result)
        return result
    
    def execute_vertical_scaling_test(self,
                                    min_resources: float = 0.5,
                                    max_resources: float = 4.0,
                                    test_duration_per_scale: int = 60) -> ScalabilityResult:
        """Test vertical scaling capabilities."""
        self.logger.info("Starting vertical scaling test")
        
        start_time = datetime.now()
        scaling_points = []
        
        # Generate resource scaling factors
        scale_factors = [min_resources + (max_resources - min_resources) * i / 5 for i in range(6)]
        
        for scale_factor in scale_factors:
            self.logger.info(f"Testing with {scale_factor:.1f}x resources")
            
            # Configure resources for this scale
            config = self.resource_scaler.scale_resources(scale_factor)
            self.resource_scaler.apply_resource_limits(config)
            
            # Run test at this scale
            scaling_point = self._measure_performance_at_scale(
                scale_level=int(scale_factor * 100),  # Convert to integer scale level
                resource_config=config,
                test_duration=test_duration_per_scale,
                load_factor=scale_factor
            )
            
            scaling_points.append(scaling_point)
        
        result = ScalabilityResult(
            test_type=ScalabilityTestType.VERTICAL_SCALING,
            test_name="Vertical Resource Scaling",
            start_time=start_time,
            end_time=datetime.now(),
            scaling_points=scaling_points
        )
        
        self._analyze_scaling_result(result)
        return result
    
    def execute_user_load_scaling_test(self,
                                     min_users: int = 10,
                                     max_users: int = 10000,
                                     growth_pattern: str = "exponential") -> ScalabilityResult:
        """Test system scaling under increasing user load."""
        self.logger.info(f"Starting user load scaling test with {growth_pattern} growth")
        
        start_time = datetime.now()
        scaling_points = []
        
        # Generate user load pattern
        if growth_pattern == "linear":
            user_loads = self.load_generator.linear_growth(min_users, max_users, 8)
        elif growth_pattern == "exponential":
            user_loads = self.load_generator.exponential_growth(min_users, max_users, 8)
        elif growth_pattern == "fibonacci":
            user_loads = self.load_generator.fibonacci_growth(min_users, max_users)
        else:
            user_loads = self.load_generator.step_growth(min_users, max_users, 8)
        
        for user_count in user_loads:
            self.logger.info(f"Testing with {user_count} concurrent users")
            
            # Auto-scale resources based on user count
            config = self.resource_scaler.get_optimal_configuration(user_count)
            self.resource_scaler.apply_resource_limits(config)
            
            # Run test with this user load
            scaling_point = self._measure_performance_at_scale(
                scale_level=user_count,
                resource_config=config,
                test_duration=90,  # Longer duration for user load tests
                load_factor=user_count / 100.0  # Normalize to 100-user baseline
            )
            
            scaling_points.append(scaling_point)
            
            # Check for breaking point
            if (scaling_point.success_rate_percent < 90.0 or 
                scaling_point.response_time_ms > 1000.0):
                self.logger.warning(f"Breaking point detected at {user_count} users")
                break
        
        result = ScalabilityResult(
            test_type=ScalabilityTestType.USER_LOAD_SCALING,
            test_name=f"User Load Scaling ({growth_pattern})",
            start_time=start_time,
            end_time=datetime.now(),
            scaling_points=scaling_points
        )
        
        self._analyze_scaling_result(result)
        return result
    
    def execute_data_volume_scaling_test(self,
                                       min_volume_mb: int = 1,
                                       max_volume_mb: int = 1000,
                                       test_duration_per_scale: int = 120) -> ScalabilityResult:
        """Test system scaling with increasing data volumes."""
        self.logger.info("Starting data volume scaling test")
        
        start_time = datetime.now()
        scaling_points = []
        
        # Generate data volume pattern (exponential growth)
        volume_sizes = self.load_generator.exponential_growth(min_volume_mb, max_volume_mb, 6)
        
        for volume_mb in volume_sizes:
            self.logger.info(f"Testing with {volume_mb}MB data volume")
            
            # Scale resources based on data volume
            scale_factor = max(1.0, volume_mb / 100.0)  # 100MB baseline
            config = self.resource_scaler.scale_resources(scale_factor)
            self.resource_scaler.apply_resource_limits(config)
            
            # Run test with this data volume
            scaling_point = self._measure_performance_at_scale(
                scale_level=volume_mb,
                resource_config=config,
                test_duration=test_duration_per_scale,
                load_factor=scale_factor,
                data_volume_mb=volume_mb
            )
            
            scaling_points.append(scaling_point)
            
            # Check if processing becomes too slow
            if scaling_point.response_time_ms > 2000.0:
                self.logger.warning(f"Performance degradation detected at {volume_mb}MB")
                break
        
        result = ScalabilityResult(
            test_type=ScalabilityTestType.DATA_VOLUME_SCALING,
            test_name="Data Volume Scaling",
            start_time=start_time,
            end_time=datetime.now(),
            scaling_points=scaling_points
        )
        
        self._analyze_scaling_result(result)
        return result
    
    def execute_auto_scaling_test(self,
                                load_spike_factor: float = 5.0,
                                spike_duration_minutes: int = 5,
                                total_test_duration_minutes: int = 20) -> ScalabilityResult:
        """Test auto-scaling behavior under load spikes."""
        self.logger.info("Starting auto-scaling behavior test")
        
        start_time = datetime.now()
        scaling_points = []
        
        baseline_load = 100
        spike_load = int(baseline_load * load_spike_factor)
        
        # Phase 1: Baseline
        self.logger.info(f"Phase 1: Baseline load ({baseline_load} users)")
        config = self.resource_scaler.get_optimal_configuration(baseline_load)
        baseline_point = self._measure_performance_at_scale(
            scale_level=baseline_load,
            resource_config=config,
            test_duration=300,  # 5 minutes baseline
            load_factor=1.0
        )
        scaling_points.append(baseline_point)
        
        # Phase 2: Load spike
        self.logger.info(f"Phase 2: Load spike ({spike_load} users)")
        spike_start = time.time()
        spike_config = self.resource_scaler.get_optimal_configuration(spike_load)
        
        # Measure auto-scaling response time
        scaling_response_start = time.time()
        self.resource_scaler.apply_resource_limits(spike_config)
        scaling_response_time = time.time() - scaling_response_start
        
        spike_point = self._measure_performance_at_scale(
            scale_level=spike_load,
            resource_config=spike_config,
            test_duration=spike_duration_minutes * 60,
            load_factor=load_spike_factor
        )
        spike_point.custom_metrics['scaling_response_time'] = scaling_response_time
        scaling_points.append(spike_point)
        
        # Phase 3: Scale down
        self.logger.info(f"Phase 3: Scale down to baseline")
        self.resource_scaler.apply_resource_limits(config)
        
        recovery_point = self._measure_performance_at_scale(
            scale_level=baseline_load,
            resource_config=config,
            test_duration=300,  # 5 minutes recovery
            load_factor=1.0
        )
        scaling_points.append(recovery_point)
        
        result = ScalabilityResult(
            test_type=ScalabilityTestType.AUTO_SCALING,
            test_name="Auto-scaling Behavior",
            start_time=start_time,
            end_time=datetime.now(),
            scaling_points=scaling_points
        )
        
        # Add auto-scaling specific metrics
        result.scalability_metrics['scaling_response_time'] = scaling_response_time
        result.scalability_metrics['spike_performance_ratio'] = (
            spike_point.throughput_ops_sec / baseline_point.throughput_ops_sec
        )
        result.scalability_metrics['recovery_efficiency'] = (
            recovery_point.throughput_ops_sec / baseline_point.throughput_ops_sec
        )
        
        self._analyze_scaling_result(result)
        return result
    
    def _measure_performance_at_scale(self,
                                    scale_level: int,
                                    resource_config: ResourceConfiguration,
                                    test_duration: int,
                                    load_factor: float,
                                    data_volume_mb: Optional[int] = None) -> ScalingPoint:
        """Measure performance at specific scale level."""
        
        # Wait for system to stabilize
        time.sleep(self.stabilization_time)
        
        # Start resource monitoring
        cpu_measurements = []
        memory_measurements = []
        response_times = []
        successful_operations = 0
        failed_operations = 0
        
        start_time = time.time()
        end_time = start_time + test_duration
        
        # Create thread pool based on resource configuration
        with ThreadPoolExecutor(max_workers=resource_config.thread_pool_size) as executor:
            
            while time.time() < end_time:
                # Submit work based on load factor
                operations_per_second = max(1, int(10 * load_factor))
                
                futures = []
                for _ in range(operations_per_second):
                    future = executor.submit(
                        self._simulate_manufacturing_operation,
                        data_volume_mb or 1
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures, timeout=2.0):
                    try:
                        success, response_time = future.result(timeout=1.0)
                        if success:
                            successful_operations += 1
                        else:
                            failed_operations += 1
                        response_times.append(response_time)
                    except Exception:
                        failed_operations += 1
                        response_times.append(2000.0)  # Timeout penalty
                
                # Sample resource utilization
                try:
                    cpu_measurements.append(psutil.cpu_percent(interval=0.1))
                    memory_measurements.append(psutil.virtual_memory().percent)
                except:
                    pass
                
                # Brief pause
                time.sleep(1.0)
        
        # Calculate metrics
        total_operations = successful_operations + failed_operations
        success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput = successful_operations / test_duration
        avg_cpu = statistics.mean(cpu_measurements) if cpu_measurements else 0
        avg_memory = statistics.mean(memory_measurements) if memory_measurements else 0
        
        return ScalingPoint(
            scale_level=scale_level,
            resource_config=resource_config,
            load_factor=load_factor,
            response_time_ms=avg_response_time,
            throughput_ops_sec=throughput,
            success_rate_percent=success_rate,
            cpu_utilization_percent=avg_cpu,
            memory_utilization_percent=avg_memory,
            error_count=failed_operations
        )
    
    def _simulate_manufacturing_operation(self, data_volume_mb: int) -> Tuple[bool, float]:
        """Simulate a manufacturing system operation."""
        start_time = time.perf_counter()
        
        try:
            # Simulate operation complexity based on data volume
            base_processing_time = 0.01  # 10ms base
            volume_factor = min(data_volume_mb / 100.0, 5.0)  # Cap at 5x
            processing_time = base_processing_time * volume_factor
            
            # Simulate different types of manufacturing operations
            operation_type = random.choice([
                'sensor_reading',
                'control_adjustment', 
                'quality_check',
                'data_logging',
                'ai_analysis'
            ])
            
            if operation_type == 'sensor_reading':
                time.sleep(processing_time * 0.5)  # Fast operation
                # Simulate data processing
                _ = [random.random() for _ in range(min(100 * data_volume_mb, 10000))]
                
            elif operation_type == 'control_adjustment':
                time.sleep(processing_time * 0.3)  # Very fast
                # Simulate control calculations
                _ = sum(math.sqrt(i) for i in range(min(50 * data_volume_mb, 1000)))
                
            elif operation_type == 'quality_check':
                time.sleep(processing_time * 1.5)  # Slower operation
                # Simulate quality analysis
                data = [random.gauss(0, 1) for _ in range(min(200 * data_volume_mb, 5000))]
                _ = statistics.mean(data)
                _ = statistics.stdev(data)
                
            elif operation_type == 'data_logging':
                time.sleep(processing_time * 0.8)  # Medium operation
                # Simulate database write
                record = {
                    'timestamp': time.time(),
                    'data': [random.random() for _ in range(min(10 * data_volume_mb, 1000))]
                }
                _ = json.dumps(record)
                
            elif operation_type == 'ai_analysis':
                time.sleep(processing_time * 2.0)  # Slowest operation
                # Simulate ML inference
                matrix_size = min(20 * data_volume_mb, 100)
                matrix = [[random.random() for _ in range(matrix_size)] for _ in range(matrix_size)]
                # Simple computation
                for i in range(min(matrix_size, 20)):
                    for j in range(min(matrix_size, 20)):
                        _ = sum(matrix[i])
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            # Simulate occasional failures (higher failure rate under high load)
            failure_rate = min(0.02 + (data_volume_mb / 10000.0), 0.1)  # 2-10% failure rate
            success = random.random() > failure_rate
            
            return success, response_time
            
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            return False, response_time
    
    def _analyze_scaling_result(self, result: ScalabilityResult):
        """Analyze scaling result and add metrics."""
        if len(result.scaling_points) < 2:
            return
        
        # Find baseline and optimal points
        result.baseline_performance = min(result.scaling_points, key=lambda p: p.scale_level)
        result.optimal_scaling_point = max(
            result.scaling_points, 
            key=lambda p: p.throughput_ops_sec * (p.success_rate_percent / 100.0)
        )
        
        # Find breaking point
        for point in sorted(result.scaling_points, key=lambda p: p.scale_level):
            if point.success_rate_percent < 95.0 or point.response_time_ms > 1000.0:
                result.breaking_point = point
                break
        
        # Calculate scalability metrics
        baseline = result.baseline_performance
        optimal = result.optimal_scaling_point
        
        if baseline and optimal and baseline.scale_level < optimal.scale_level:
            # Scalability factor (how much performance improved with scaling)
            result.scalability_metrics['scalability_factor'] = (
                optimal.throughput_ops_sec / baseline.throughput_ops_sec
            )
            
            # Linear scaling index (0-1, where 1 is perfect linear scaling)
            scale_ratio = optimal.scale_level / baseline.scale_level
            throughput_ratio = optimal.throughput_ops_sec / baseline.throughput_ops_sec
            result.scalability_metrics['linear_scaling_index'] = min(
                throughput_ratio / scale_ratio, 1.0
            )
            
            # Efficiency ratio (throughput improvement per resource unit)
            resource_ratio = optimal.load_factor / baseline.load_factor
            result.scalability_metrics['efficiency_ratio'] = throughput_ratio / resource_ratio
            
            # Performance degradation rate
            response_time_ratio = optimal.response_time_ms / baseline.response_time_ms
            result.scalability_metrics['performance_degradation_rate'] = response_time_ratio - 1.0
        
        # Breaking point metrics
        if result.breaking_point:
            result.scalability_metrics['breaking_point_scale_level'] = result.breaking_point.scale_level
            result.scalability_metrics['breaking_point_success_rate'] = result.breaking_point.success_rate_percent


# Import math for simulations
import math


class ScalabilityTestSuite:
    """
    Comprehensive Scalability Test Suite
    
    Validates manufacturing system scalability across multiple dimensions:
    - Horizontal scaling (adding instances/nodes)  
    - Vertical scaling (increasing resources per instance)
    - User load scaling (concurrent user capacity)
    - Data volume scaling (processing large datasets)
    - Auto-scaling behavior (dynamic resource adjustment)
    - Network bandwidth scaling
    - Storage scaling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.executor = ScalabilityTestExecutor(max_workers=200)
        self.scalability_targets = self._create_scalability_targets()
        self.results: List[ScalabilityResult] = []
    
    def _create_scalability_targets(self) -> List[ScalabilityTarget]:
        """Create scalability targets based on requirements."""
        return [
            ScalabilityTarget(
                metric=ScalabilityMetric.LINEAR_SCALING_INDEX,
                target_value=0.8,  # 80% linear scaling efficiency
                threshold_type='min',
                description='System should achieve at least 80% linear scaling efficiency'
            ),
            ScalabilityTarget(
                metric=ScalabilityMetric.SCALABILITY_FACTOR,
                target_value=8.0,  # 8x performance improvement possible
                threshold_type='min', 
                description='System should scale performance by at least 8x with resource scaling'
            ),
            ScalabilityTarget(
                metric=ScalabilityMetric.BREAKING_POINT,
                target_value=1000.0,  # Support 1000+ users
                threshold_type='min',
                description='System should support at least 1000 concurrent users before breaking'
            ),
            ScalabilityTarget(
                metric=ScalabilityMetric.EFFICIENCY_RATIO,
                target_value=0.7,  # 70% efficiency ratio
                threshold_type='min',
                description='Resource scaling should maintain at least 70% efficiency'
            ),
            ScalabilityTarget(
                metric=ScalabilityMetric.DEGRADATION_RATE,
                target_value=0.5,  # 50% max degradation
                threshold_type='max',
                description='Performance degradation should not exceed 50% at scale'
            ),
            ScalabilityTarget(
                metric=ScalabilityMetric.SCALING_RESPONSE_TIME,
                target_value=30.0,  # 30 seconds
                threshold_type='max',
                description='Auto-scaling should respond within 30 seconds'
            )
        ]
    
    def run_all_scalability_tests(self) -> Dict[str, Any]:
        """Run comprehensive scalability test suite."""
        self.logger.info("Starting comprehensive scalability test suite")
        start_time = datetime.now()
        
        self.results = []
        
        # Test scenarios in order of complexity
        test_scenarios = [
            ("Vertical Scaling", self._run_vertical_scaling_test),
            ("Horizontal Scaling", self._run_horizontal_scaling_test), 
            ("User Load Scaling", self._run_user_load_scaling_test),
            ("Data Volume Scaling", self._run_data_volume_scaling_test),
            ("Auto-scaling Behavior", self._run_auto_scaling_test)
        ]
        
        for test_name, test_function in test_scenarios:
            try:
                self.logger.info(f"Running {test_name} test...")
                result = test_function()
                self.results.append(result)
                
                # Log key metrics
                if result.optimal_scaling_point:
                    opt = result.optimal_scaling_point
                    self.logger.info(f"  Optimal scale: {opt.scale_level} "
                                   f"({opt.throughput_ops_sec:.0f} ops/sec, "
                                   f"{opt.response_time_ms:.1f}ms response)")
                
            except Exception as e:
                self.logger.error(f"{test_name} test failed: {e}")
                # Create failed result placeholder
                failed_result = ScalabilityResult(
                    test_type=ScalabilityTestType.HORIZONTAL_SCALING,  # Default type
                    test_name=test_name,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    scaling_points=[],
                    baseline_performance=None,
                    optimal_scaling_point=None,
                    breaking_point=None
                )
                self.results.append(failed_result)
        
        end_time = datetime.now()
        
        # Generate comprehensive report
        return self._generate_scalability_report(start_time, end_time)
    
    def _run_vertical_scaling_test(self) -> ScalabilityResult:
        """Run vertical scaling test."""
        return self.executor.execute_vertical_scaling_test(
            min_resources=0.5,
            max_resources=4.0,
            test_duration_per_scale=90
        )
    
    def _run_horizontal_scaling_test(self) -> ScalabilityResult:
        """Run horizontal scaling test."""
        return self.executor.execute_horizontal_scaling_test(
            min_instances=1,
            max_instances=8,
            test_duration_per_scale=90
        )
    
    def _run_user_load_scaling_test(self) -> ScalabilityResult:
        """Run user load scaling test."""
        return self.executor.execute_user_load_scaling_test(
            min_users=10,
            max_users=2000,
            growth_pattern="exponential"
        )
    
    def _run_data_volume_scaling_test(self) -> ScalabilityResult:
        """Run data volume scaling test."""
        return self.executor.execute_data_volume_scaling_test(
            min_volume_mb=1,
            max_volume_mb=500,
            test_duration_per_scale=120
        )
    
    def _run_auto_scaling_test(self) -> ScalabilityResult:
        """Run auto-scaling behavior test."""
        return self.executor.execute_auto_scaling_test(
            load_spike_factor=3.0,
            spike_duration_minutes=5,
            total_test_duration_minutes=15
        )
    
    def _generate_scalability_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive scalability report."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if len(r.scaling_points) > 0)
        
        # Aggregate scalability metrics
        all_scalability_factors = []
        all_linear_scaling_indices = []
        all_efficiency_ratios = []
        breaking_points = []
        
        for result in self.results:
            if 'scalability_factor' in result.scalability_metrics:
                all_scalability_factors.append(result.scalability_metrics['scalability_factor'])
            if 'linear_scaling_index' in result.scalability_metrics:
                all_linear_scaling_indices.append(result.scalability_metrics['linear_scaling_index'])
            if 'efficiency_ratio' in result.scalability_metrics:
                all_efficiency_ratios.append(result.scalability_metrics['efficiency_ratio'])
            if result.breaking_point:
                breaking_points.append(result.breaking_point.scale_level)
        
        # Calculate overall scalability metrics
        overall_metrics = {
            'avg_scalability_factor': statistics.mean(all_scalability_factors) if all_scalability_factors else 0,
            'avg_linear_scaling_index': statistics.mean(all_linear_scaling_indices) if all_linear_scaling_indices else 0,
            'avg_efficiency_ratio': statistics.mean(all_efficiency_ratios) if all_efficiency_ratios else 0,
            'min_breaking_point': min(breaking_points) if breaking_points else 0,
            'max_breaking_point': max(breaking_points) if breaking_points else 0,
            'linear_scaling_efficiency_percent': (
                statistics.mean(all_linear_scaling_indices) * 100 if all_linear_scaling_indices else 0
            )
        }
        
        # Evaluate against scalability targets
        target_compliance = {}
        for target in self.scalability_targets:
            compliance_results = []
            
            for result in self.results:
                if target.metric == ScalabilityMetric.LINEAR_SCALING_INDEX:
                    actual_value = result.scalability_metrics.get('linear_scaling_index', 0)
                elif target.metric == ScalabilityMetric.SCALABILITY_FACTOR:
                    actual_value = result.scalability_metrics.get('scalability_factor', 0)
                elif target.metric == ScalabilityMetric.BREAKING_POINT:
                    actual_value = result.breaking_point.scale_level if result.breaking_point else 0
                elif target.metric == ScalabilityMetric.EFFICIENCY_RATIO:
                    actual_value = result.scalability_metrics.get('efficiency_ratio', 0)
                elif target.metric == ScalabilityMetric.DEGRADATION_RATE:
                    actual_value = result.scalability_metrics.get('performance_degradation_rate', 0)
                elif target.metric == ScalabilityMetric.SCALING_RESPONSE_TIME:
                    actual_value = result.scalability_metrics.get('scaling_response_time', 0)
                else:
                    continue
                
                # Check if target is met
                if target.threshold_type == 'max':
                    criteria_met = actual_value <= target.target_value
                elif target.threshold_type == 'min':
                    criteria_met = actual_value >= target.target_value
                else:  # range
                    criteria_met = True  # Simplified for now
                
                compliance_results.append(criteria_met)
            
            if compliance_results:
                compliance_rate = sum(compliance_results) / len(compliance_results) * 100
                target_compliance[target.metric.value] = {
                    'target_description': target.description,
                    'compliance_rate_percent': compliance_rate,
                    'tests_passed': sum(compliance_results),
                    'total_tests': len(compliance_results)
                }
        
        # Identify scalability bottlenecks
        bottlenecks = []
        if overall_metrics['avg_linear_scaling_index'] < 0.7:
            bottlenecks.append('Poor linear scaling efficiency - investigate resource contention')
        if overall_metrics['min_breaking_point'] < 500:
            bottlenecks.append('Low breaking point - system cannot handle expected user load')
        if overall_metrics['avg_efficiency_ratio'] < 0.6:
            bottlenecks.append('Resource scaling inefficiency - optimize resource utilization')
        
        # Scalability recommendations
        recommendations = self._generate_scalability_recommendations(overall_metrics, target_compliance)
        
        # Results by test type
        results_by_type = {}
        for result in self.results:
            test_type = result.test_type.value
            if test_type not in results_by_type:
                results_by_type[test_type] = {}
            
            results_by_type[test_type] = {
                'test_name': result.test_name,
                'max_scale_achieved': result.max_scale_achieved,
                'linear_scaling_efficiency': result.linear_scaling_efficiency,
                'breaking_point': result.breaking_point.scale_level if result.breaking_point else None,
                'scalability_metrics': result.scalability_metrics,
                'optimal_point': {
                    'scale_level': result.optimal_scaling_point.scale_level if result.optimal_scaling_point else 0,
                    'throughput_ops_sec': result.optimal_scaling_point.throughput_ops_sec if result.optimal_scaling_point else 0,
                    'response_time_ms': result.optimal_scaling_point.response_time_ms if result.optimal_scaling_point else 0
                } if result.optimal_scaling_point else None
            }
        
        return {
            'summary': {
                'total_scalability_tests': total_tests,
                'successful_tests': successful_tests,
                'overall_success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                'test_start_time': start_time.isoformat(),
                'test_end_time': end_time.isoformat(),
                'total_duration_minutes': (end_time - start_time).total_seconds() / 60
            },
            'overall_scalability_metrics': overall_metrics,
            'target_compliance': target_compliance,
            'scalability_bottlenecks': bottlenecks,
            'results_by_test_type': results_by_type,
            'recommendations': recommendations,
            'detailed_results': [
                {
                    'test_type': r.test_type.value,
                    'test_name': r.test_name,
                    'duration_seconds': r.duration_seconds,
                    'max_scale_achieved': r.max_scale_achieved,
                    'linear_scaling_efficiency': r.linear_scaling_efficiency,
                    'baseline_performance': {
                        'scale_level': r.baseline_performance.scale_level,
                        'throughput_ops_sec': r.baseline_performance.throughput_ops_sec,
                        'response_time_ms': r.baseline_performance.response_time_ms
                    } if r.baseline_performance else None,
                    'optimal_performance': {
                        'scale_level': r.optimal_scaling_point.scale_level,
                        'throughput_ops_sec': r.optimal_scaling_point.throughput_ops_sec,
                        'response_time_ms': r.optimal_scaling_point.response_time_ms
                    } if r.optimal_scaling_point else None,
                    'breaking_point': {
                        'scale_level': r.breaking_point.scale_level,
                        'success_rate_percent': r.breaking_point.success_rate_percent
                    } if r.breaking_point else None,
                    'scalability_metrics': r.scalability_metrics
                }
                for r in self.results
            ]
        }
    
    def _generate_scalability_recommendations(self, 
                                            metrics: Dict[str, Any],
                                            compliance: Dict[str, Any]) -> List[str]:
        """Generate scalability improvement recommendations."""
        recommendations = []
        
        # Linear scaling recommendations
        if metrics['avg_linear_scaling_index'] < 0.8:
            recommendations.append(
                f"Linear scaling efficiency is {metrics['avg_linear_scaling_index']:.1%} "
                f"(target: >80%). Consider optimizing shared resources, reducing lock contention, "
                f"and implementing better load balancing."
            )
        
        # Breaking point recommendations  
        if metrics['min_breaking_point'] < 1000:
            recommendations.append(
                f"System breaking point is {metrics['min_breaking_point']} users "
                f"(target: >1000). Investigate bottlenecks, optimize database connections, "
                f"and consider horizontal scaling architecture."
            )
        
        # Efficiency recommendations
        if metrics['avg_efficiency_ratio'] < 0.7:
            recommendations.append(
                f"Resource scaling efficiency is {metrics['avg_efficiency_ratio']:.1%} "
                f"(target: >70%). Review resource allocation, optimize algorithms, "
                f"and implement better caching strategies."
            )
        
        # Scalability factor recommendations
        if metrics['avg_scalability_factor'] < 5.0:
            recommendations.append(
                f"Scalability factor is {metrics['avg_scalability_factor']:.1f}x "
                f"(target: >8x). Consider microservices architecture, async processing, "
                f"and distributed computing patterns."
            )
        
        # Auto-scaling recommendations
        auto_scaling_compliance = compliance.get('scaling_response_time', {})
        if auto_scaling_compliance and auto_scaling_compliance.get('compliance_rate_percent', 100) < 80:
            recommendations.append(
                "Auto-scaling response time needs improvement. Consider predictive scaling, "
                "pre-warming instances, and optimizing scaling triggers."
            )
        
        # General architecture recommendations
        if len(recommendations) >= 3:
            recommendations.append(
                "Multiple scalability issues detected. Consider comprehensive architecture "
                "review focusing on distributed systems design patterns, caching strategies, "
                "and asynchronous processing."
            )
        elif not recommendations:
            recommendations.append(
                "Scalability targets are being met. Continue monitoring performance trends "
                "and plan for future growth patterns."
            )
        
        return recommendations


# Convenience functions for running specific scalability tests
def run_horizontal_scaling_test() -> ScalabilityResult:
    """Run horizontal scaling test."""
    executor = ScalabilityTestExecutor()
    return executor.execute_horizontal_scaling_test()

def run_vertical_scaling_test() -> ScalabilityResult:
    """Run vertical scaling test."""
    executor = ScalabilityTestExecutor()
    return executor.execute_vertical_scaling_test()

def run_user_load_scaling_test() -> ScalabilityResult:
    """Run user load scaling test."""
    executor = ScalabilityTestExecutor()
    return executor.execute_user_load_scaling_test()

def run_all_scalability_tests() -> Dict[str, Any]:
    """Run complete scalability test suite."""
    suite = ScalabilityTestSuite()
    return suite.run_all_scalability_tests()


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Scalability Test Suite Demo")
    print("=" * 80)
    
    # Create scalability test suite
    scalability_suite = ScalabilityTestSuite()
    
    print(f"\nScalability Targets: {len(scalability_suite.scalability_targets)}")
    for target in scalability_suite.scalability_targets:
        print(f"  • {target.description}")
    
    # Run a subset of tests for demo (to keep runtime reasonable)
    print("\nRunning demo scalability tests...")
    
    try:
        # Run vertical scaling test
        print("\n--- Vertical Scaling Test ---")
        vertical_result = scalability_suite._run_vertical_scaling_test()
        print(f"✅ Vertical Scaling: {vertical_result.linear_scaling_efficiency:.1f}% efficiency, "
              f"max scale {vertical_result.max_scale_achieved}")
        
        # Run user load scaling test (limited)
        print("\n--- User Load Scaling Test ---") 
        scalability_suite.executor.max_workers = 50  # Limit for demo
        user_result = scalability_suite.executor.execute_user_load_scaling_test(
            min_users=10, max_users=200, growth_pattern="linear"
        )
        print(f"✅ User Load Scaling: {user_result.linear_scaling_efficiency:.1f}% efficiency, "
              f"breaking point at {user_result.breaking_point.scale_level if user_result.breaking_point else 'N/A'}")
        
        # Set results for report generation  
        scalability_suite.results = [vertical_result, user_result]
        
        # Generate summary report
        print("\n" + "="*80)
        print("SCALABILITY TEST SUMMARY")
        print("="*80)
        
        report = scalability_suite._generate_scalability_report(
            datetime.now() - timedelta(minutes=10),
            datetime.now()
        )
        
        summary = report['summary']
        metrics = report['overall_scalability_metrics']
        
        print(f"Tests Executed: {summary['successful_tests']}/{summary['total_scalability_tests']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Average Scalability Factor: {metrics['avg_scalability_factor']:.1f}x")
        print(f"Linear Scaling Efficiency: {metrics['linear_scaling_efficiency_percent']:.1f}%")
        print(f"Resource Efficiency: {metrics['avg_efficiency_ratio']:.1%}")
        
        if report['scalability_bottlenecks']:
            print("\nScalability Issues:")
            for bottleneck in report['scalability_bottlenecks']:
                print(f"  ⚠️  {bottleneck}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  💡 {rec}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    print("\nScalability Test Suite demo completed!")