#!/usr/bin/env python3
"""
Test Suite for Week 14 Optimization Layer Components

This script tests all the major components of the optimization layer:
- Performance Profiler
- Cache Manager
- Load Balancer
- Performance Monitor
- Alert Manager

Author: Manufacturing Line Control System
Created: Week 14 - Testing Phase
"""

import time
import threading
import logging
import sys
import traceback
from typing import Dict, Any, List
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_performance_profiler():
    """Test the Performance Profiler component."""
    print("\n" + "="*50)
    print("TESTING PERFORMANCE PROFILER")
    print("="*50)
    
    try:
        from performance_profiler import PerformanceProfiler, ProfilerConfig
        
        # Create profiler with test configuration
        config = ProfilerConfig(
            sampling_interval_ms=500,  # Fast for testing
            history_retention_minutes=5,
            enable_function_profiling=True,
            enable_memory_tracing=True
        )
        
        profiler = PerformanceProfiler(config)
        
        # Test basic profiling
        print("1. Testing basic system profiling...")
        metrics = profiler.collect_system_metrics()
        print(f"   CPU: {metrics.cpu_percent:.1f}%")
        print(f"   Memory: {metrics.memory_percent:.1f}%")
        print(f"   Processes: {metrics.process_count}")
        
        # Test function profiling
        print("2. Testing function profiling...")
        
        @profiler.profile_function
        def test_function():
            time.sleep(0.1)
            return sum(range(1000))
        
        result = test_function()
        print(f"   Function result: {result}")
        
        # Test bottleneck detection
        print("3. Testing bottleneck detection...")
        bottlenecks = profiler.detect_bottlenecks()
        print(f"   Found {len(bottlenecks)} potential bottlenecks")
        
        # Test background profiling
        print("4. Testing background profiling...")
        profiler.start_profiling()
        time.sleep(2)
        profiler.stop_profiling()
        
        baseline = profiler.get_performance_baseline()
        print(f"   Baseline established with {len(baseline)} metrics")
        
        print("✅ Performance Profiler: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Performance Profiler: FAILED - {e}")
        traceback.print_exc()
        return False

def test_cache_manager():
    """Test the Cache Manager component."""
    print("\n" + "="*50)
    print("TESTING CACHE MANAGER")
    print("="*50)
    
    try:
        from cache_manager import CacheManager, CacheConfiguration
        
        # Create cache manager with test configuration
        config = CacheConfiguration(
            l1_max_size=100,
            l1_default_ttl=60,
            l2_max_size=500,
            cache_warming_enabled=True,
            auto_optimization_enabled=True,
            background_cleanup_enabled=False  # Disable for testing
        )
        
        cache_mgr = CacheManager(config)
        
        # Test basic cache operations
        print("1. Testing basic cache operations...")
        cache_mgr.set("test_key", "test_value", ttl_seconds=30)
        value = cache_mgr.get("test_key")
        print(f"   Cached value: {value}")
        assert value == "test_value", "Cache get/set failed"
        
        exists = cache_mgr.exists("test_key")
        print(f"   Key exists: {exists}")
        assert exists, "Cache exists check failed"
        
        # Test cache warming
        print("2. Testing cache warming...")
        warm_data = [
            ("sensor_1", {"temperature": 25.5, "reading_time": time.time()}),
            ("sensor_2", {"temperature": 26.1, "reading_time": time.time()}),
            ("production_rate", 98.5),
            ("quality_score", 94.2)
        ]
        
        warmed_count = cache_mgr.warm_cache(warm_data, ttl_seconds=60)
        print(f"   Warmed {warmed_count} cache entries")
        
        # Test cache statistics
        print("3. Testing cache statistics...")
        stats = cache_mgr.get_cache_stats()
        for level, stat in stats.items():
            if level != "Global":
                print(f"   {level}: Hit Rate: {stat.hit_rate:.1f}%, Entries: {stat.cache_size}")
        
        # Test cache optimization
        print("4. Testing cache optimization...")
        optimization = cache_mgr.optimize_cache()
        print(f"   Optimization actions: {len(optimization.get('actions_taken', []))}")
        print(f"   Recommendations: {len(optimization.get('recommendations', []))}")
        
        cache_mgr.shutdown()
        print("✅ Cache Manager: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Cache Manager: FAILED - {e}")
        traceback.print_exc()
        return False

def test_load_balancer():
    """Test the Load Balancer component."""
    print("\n" + "="*50)
    print("TESTING LOAD BALANCER")
    print("="*50)
    
    try:
        from load_balancer import LoadBalancer, LoadBalancerConfig, Server, LoadBalancingAlgorithm, ServerStatus, HealthCheckConfig
        
        # Create load balancer with test configuration
        health_check_config = HealthCheckConfig(enabled=False)  # Disable health checks for testing
        config = LoadBalancerConfig(
            algorithm=LoadBalancingAlgorithm.HEALTH_WEIGHTED,
            enable_circuit_breaker=True,
            health_check=health_check_config
        )
        
        lb = LoadBalancer(config)
        
        # Add test servers
        print("1. Testing server management...")
        servers = [
            Server("server1", "localhost", 8001, weight=1.0),
            Server("server2", "localhost", 8002, weight=1.5),
            Server("server3", "localhost", 8003, weight=2.0),
        ]
        
        # Set servers as healthy for testing
        for server in servers:
            server.status = ServerStatus.HEALTHY
            lb.add_server(server, group="web_servers")
        
        print(f"   Added {len(servers)} servers")
        
        # Test server selection
        print("2. Testing server selection...")
        for i in range(5):
            server = lb.get_server(
                request_context={'client_ip': f"192.168.1.{i+1}", 'session_id': f"session_{i}"},
                group="web_servers"
            )
            if server:
                print(f"   Request {i+1}: Selected {server.id}")
            else:
                print(f"   Request {i+1}: No server selected")
        
        # Test load balancer statistics
        print("3. Testing load balancer statistics...")
        lb_stats = lb.get_load_balancer_stats()
        print(f"   Total servers: {lb_stats.server_count}")
        print(f"   Healthy servers: {lb_stats.healthy_server_count}")
        
        server_stats = lb.get_server_stats()
        for server_id, stats in server_stats.items():
            print(f"   {server_id}: Status={stats['status']}, Weight={stats['weight']}")
        
        # Test request execution with mock function
        print("4. Testing request execution...")
        
        def mock_request(server: Server) -> str:
            # Simulate request processing time
            time.sleep(random.uniform(0.01, 0.05))
            if random.random() < 0.9:  # 90% success rate
                return f"Success from {server.id}"
            else:
                raise Exception("Simulated request failure")
        
        successful_requests = 0
        for i in range(10):
            try:
                result = lb.execute_request(
                    mock_request, 
                    {'client_ip': f"192.168.1.{i % 3 + 1}"},
                    group="web_servers"
                )
                successful_requests += 1
            except Exception as e:
                pass  # Expected some failures
        
        print(f"   Successful requests: {successful_requests}/10")
        
        lb.shutdown()
        print("✅ Load Balancer: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Load Balancer: FAILED - {e}")
        traceback.print_exc()
        return False

def test_performance_monitor():
    """Test the Performance Monitor component."""
    print("\n" + "="*50)
    print("TESTING PERFORMANCE MONITOR")
    print("="*50)
    
    try:
        from performance_monitor import (
            PerformanceMonitor, PerformanceMonitorConfig, 
            MetricType, AlertSeverity, ThresholdRule, SLADefinition
        )
        
        # Create performance monitor with test configuration
        config = PerformanceMonitorConfig(
            collection_interval_seconds=1.0,
            anomaly_detection_enabled=True,
            alert_evaluation_interval_seconds=1,
            enable_system_metrics=True,
            enable_application_metrics=True
        )
        
        monitor = PerformanceMonitor(config)
        
        # Add custom metrics
        print("1. Testing custom metrics...")
        monitor.add_metric("manufacturing_throughput", MetricType.GAUGE, "Production throughput", "units/hour")
        monitor.add_metric("quality_score", MetricType.GAUGE, "Quality score", "percent")
        monitor.add_metric("equipment_temperature", MetricType.GAUGE, "Equipment temperature", "celsius")
        
        print(f"   Added {len(monitor.metrics)} metrics")
        
        # Record some test data
        print("2. Testing metric recording...")
        for i in range(5):
            throughput = 95 + random.uniform(-5, 5)
            quality = 94 + random.uniform(-2, 2)
            temperature = 65 + random.uniform(-5, 5)
            
            monitor.record_metric("manufacturing_throughput", throughput)
            monitor.record_metric("quality_score", quality)
            monitor.record_metric("equipment_temperature", temperature)
            
            # Record application requests
            response_time = random.uniform(100, 200)
            is_error = random.random() < 0.05
            monitor.record_application_request(response_time, is_error)
        
        print("   Recorded test data")
        
        # Add alerting rules
        print("3. Testing alerting rules...")
        monitor.add_threshold_rule(ThresholdRule(
            metric_name="manufacturing_throughput",
            operator="lt",
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            description="Low manufacturing throughput"
        ))
        
        # Add SLA definition
        monitor.add_sla_definition(SLADefinition(
            name="response_time_sla",
            metric_name="app_avg_response_time_ms",
            target_value=200.0,
            operator="lt",
            measurement_window_minutes=1,
            description="Response time SLA"
        ))
        
        print("   Added alerting rules and SLA")
        
        # Wait for some monitoring cycles
        print("4. Testing monitoring cycles...")
        time.sleep(3)
        
        # Get current metrics
        print("5. Testing metric retrieval...")
        for metric_name in ["manufacturing_throughput", "quality_score", "system_cpu_percent"]:
            value = monitor.get_metric_value(metric_name)
            if value is not None:
                print(f"   {metric_name}: {value:.2f}")
        
        # Get monitoring statistics
        stats = monitor.get_monitoring_statistics()
        print(f"   Total collections: {stats['collection_stats']['total_collections']}")
        print(f"   Failed collections: {stats['collection_stats']['failed_collections']}")
        print(f"   Average collection time: {stats['collection_stats']['average_collection_time_ms']:.2f}ms")
        
        monitor.stop_monitoring()
        print("✅ Performance Monitor: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Performance Monitor: FAILED - {e}")
        traceback.print_exc()
        return False

def test_alert_manager():
    """Test the Alert Manager component."""
    print("\n" + "="*50)
    print("TESTING ALERT MANAGER")
    print("="*50)
    
    try:
        from alert_manager import (
            AlertManager, AlertManagerConfig, AlertRule, AlertRecipient,
            AlertChannel, AlertSeverity, AlertCorrelationRule
        )
        
        # Create alert manager with test configuration
        config = AlertManagerConfig(
            enable_correlation=True,
            enable_suppression=False,  # Disable for testing
            enable_auto_resolution=False,  # Disable for testing
        )
        
        alert_mgr = AlertManager(config)
        
        # Add recipients
        print("1. Testing recipient management...")
        alert_mgr.add_recipient(AlertRecipient(
            id="test_ops",
            name="Test Operations",
            email="ops@test.com",
            severity_filter={AlertSeverity.WARNING, AlertSeverity.CRITICAL}
        ))
        
        alert_mgr.add_recipient(AlertRecipient(
            id="test_dev",
            name="Test Development", 
            email="dev@test.com",
            severity_filter={AlertSeverity.CRITICAL}
        ))
        
        print(f"   Added {len(alert_mgr.recipients)} recipients")
        
        # Add alert rules
        print("2. Testing alert rules...")
        alert_mgr.add_alert_rule(AlertRule(
            id="test_warning",
            name="Test Warning Alert",
            description="Test warning alert rule",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.CONSOLE],
            conditions={"test": True}
        ))
        
        alert_mgr.add_alert_rule(AlertRule(
            id="test_critical",
            name="Test Critical Alert", 
            description="Test critical alert rule",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.CONSOLE],
            conditions={"test": True}
        ))
        
        print(f"   Added {len(alert_mgr.alert_rules)} alert rules")
        
        # Create test alerts
        print("3. Testing alert creation...")
        alerts = []
        
        for i in range(3):
            alert_id = alert_mgr.create_alert(
                rule_id="test_warning",
                title=f"Test Warning Alert {i+1}",
                message=f"This is test warning alert #{i+1}",
                severity=AlertSeverity.WARNING,
                source=f"test-component-{i+1}",
                labels={"component": "test", "instance": str(i+1)}
            )
            alerts.append(alert_id)
        
        # Create a critical alert
        critical_alert_id = alert_mgr.create_alert(
            rule_id="test_critical",
            title="Test Critical Alert",
            message="This is a test critical alert",
            severity=AlertSeverity.CRITICAL,
            source="test-system",
            labels={"component": "system", "severity": "high"}
        )
        alerts.append(critical_alert_id)
        
        print(f"   Created {len(alerts)} test alerts")
        
        # Wait for alert processing
        time.sleep(1)
        
        # Test alert retrieval
        print("4. Testing alert retrieval...")
        active_alerts = alert_mgr.get_active_alerts()
        print(f"   Active alerts: {len(active_alerts)}")
        
        for alert in active_alerts[:3]:  # Show first 3
            print(f"   - {alert.severity.value.upper()}: {alert.title}")
        
        # Test alert acknowledgment
        print("5. Testing alert acknowledgment...")
        if alerts:
            success = alert_mgr.acknowledge_alert(alerts[0], "test@operator.com")
            print(f"   Acknowledged alert: {success}")
        
        # Test alert resolution
        print("6. Testing alert resolution...")
        if len(alerts) > 1:
            success = alert_mgr.resolve_alert(alerts[1], "test@operator.com")
            print(f"   Resolved alert: {success}")
        
        # Get alert statistics
        print("7. Testing alert statistics...")
        stats = alert_mgr.get_alert_statistics()
        print(f"   Total alerts: {stats['total_alerts']}")
        print(f"   Active alerts: {stats.get('active_alerts', 0)}")
        print(f"   Delivery attempts: {stats['delivery_attempts']}")
        print(f"   Successful deliveries: {stats['successful_deliveries']}")
        
        alert_mgr.shutdown()
        print("✅ Alert Manager: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Alert Manager: FAILED - {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Week 14 Optimization Layer Component Testing")
    print("=" * 80)
    
    test_results = []
    
    # Run all component tests
    test_results.append(("Performance Profiler", test_performance_profiler()))
    test_results.append(("Cache Manager", test_cache_manager()))
    test_results.append(("Load Balancer", test_load_balancer()))
    test_results.append(("Performance Monitor", test_performance_monitor()))
    test_results.append(("Alert Manager", test_alert_manager()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for component, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{component:20} : {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 80)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Week 14 optimization layer is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please review and fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())