#!/usr/bin/env python3
"""
Week 8 Comprehensive Test Runner - Deployment & Monitoring

This script runs comprehensive tests for all Week 8 deployment and monitoring components:
- DeploymentEngine: Zero-downtime deployments with <5 minutes target
- MonitoringEngine: Real-time monitoring with <100ms metrics collection, <1s dashboard updates
- AlertingEngine: Intelligent alerting with <30 seconds processing and delivery
- OperationsDashboardEngine: Operations analytics with <500ms rendering, <2s queries
- InfrastructureEngine: Auto-scaling with <2 minutes scaling operations

Performance Validation:
- DeploymentEngine: <5 minutes complete production deployment
- MonitoringEngine: <100ms metrics collection, <1 second dashboard updates
- AlertingEngine: <30 seconds alert processing and delivery
- OperationsDashboardEngine: <500ms dashboard rendering, <2 seconds analytics queries
- InfrastructureEngine: <2 minutes scaling operations and resource provisioning
"""

import os
import sys
import time
import unittest
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Week 8 Deployment Layer imports
try:
    from layers.deployment_layer.deployment_engine import DeploymentEngine
    from layers.deployment_layer.monitoring_engine import MonitoringEngine
    from layers.deployment_layer.alerting_engine import AlertingEngine
    from layers.deployment_layer.operations_dashboard_engine import OperationsDashboardEngine
    from layers.deployment_layer.infrastructure_engine import InfrastructureEngine
except ImportError as e:
    print(f"Warning: Could not import Week 8 deployment engines: {e}")
    DeploymentEngine = None
    MonitoringEngine = None
    AlertingEngine = None
    OperationsDashboardEngine = None
    InfrastructureEngine = None


class Week8DeploymentTestSuite(unittest.TestCase):
    """Comprehensive test suite for Week 8 Deployment & Monitoring layer components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment for Week 8 deployment and monitoring."""
        cls.test_config = {
            'deployment_config': {
                'deployment_target_minutes': 5,
                'enable_blue_green': True,
                'enable_canary': True,
                'enable_rollback': True,
                'health_check_timeout_seconds': 30
            },
            'monitoring_config': {
                'metrics_target_ms': 100,
                'dashboard_target_ms': 1000,
                'enable_real_time': True,
                'metrics_retention_hours': 24
            },
            'alerting_config': {
                'alert_target_seconds': 30,
                'enable_correlation': True,
                'enable_escalation': True,
                'notification_channels': ['email', 'webhook']
            },
            'operations_config': {
                'dashboard_target_ms': 500,
                'analytics_target_seconds': 2,
                'enable_real_time_updates': True,
                'cache_enabled': True
            },
            'infrastructure_config': {
                'scaling_target_minutes': 2,
                'enable_auto_scaling': True,
                'enable_cost_optimization': True,
                'providers': ['kubernetes', 'local']
            }
        }
        
        # Initialize engines for testing
        cls.deployment_engine = DeploymentEngine(cls.test_config['deployment_config']) if DeploymentEngine else None
        cls.monitoring_engine = MonitoringEngine(cls.test_config['monitoring_config']) if MonitoringEngine else None
        cls.alerting_engine = AlertingEngine(cls.test_config['alerting_config']) if AlertingEngine else None
        cls.operations_engine = OperationsDashboardEngine(cls.test_config['operations_config']) if OperationsDashboardEngine else None
        cls.infrastructure_engine = InfrastructureEngine(cls.test_config['infrastructure_config']) if InfrastructureEngine else None
        
        cls.test_results = {
            'deployment_engine': [],
            'monitoring_engine': [],
            'alerting_engine': [],
            'operations_engine': [],
            'infrastructure_engine': [],
            'performance': []
        }
        
        # Test data configurations
        cls.sample_deployment_config = {
            'deployment_id': 'test_deployment_001',
            'strategy': 'blue_green',
            'environment': 'testing',
            'services': [
                {
                    'name': 'webui-service',
                    'image': 'manufacturing-ui:latest',
                    'replicas': 2,
                    'health_check': '/health'
                },
                {
                    'name': 'control-service',
                    'image': 'control-engine:latest',
                    'replicas': 3,
                    'health_check': '/api/health'
                }
            ],
            'rollback_enabled': True,
            'health_check_timeout': 30
        }
        
        cls.sample_monitoring_config = {
            'metrics_config': [
                {
                    'name': 'system_cpu_usage',
                    'type': 'gauge',
                    'collection_interval_seconds': 10,
                    'alert_threshold': 80.0
                },
                {
                    'name': 'manufacturing_throughput',
                    'type': 'counter',
                    'collection_interval_seconds': 5,
                    'alert_threshold': 1000
                }
            ],
            'dashboard_config': {
                'refresh_interval_seconds': 5,
                'panels': ['system_overview', 'manufacturing_kpis', 'alerts']
            }
        }
        
        cls.sample_alert_config = {
            'alert_rules': [
                {
                    'name': 'high_cpu_usage',
                    'severity': 'critical',
                    'condition': 'cpu_usage > 90',
                    'duration': '5m',
                    'channels': ['email', 'slack']
                },
                {
                    'name': 'low_throughput',
                    'severity': 'warning',
                    'condition': 'throughput < 800',
                    'duration': '10m',
                    'channels': ['webhook']
                }
            ],
            'notification_config': {
                'email': {'smtp_server': 'localhost', 'port': 587},
                'slack': {'webhook_url': 'https://hooks.slack.com/test'},
                'webhook': {'url': 'https://api.test.com/alerts'}
            }
        }
    
    def test_01_deployment_engine_functionality(self):
        """Test DeploymentEngine core functionality and performance."""
        if not self.deployment_engine:
            self.skipTest("DeploymentEngine not available")
        
        print("\n=== Testing DeploymentEngine Functionality ===")
        
        # Test production deployment execution
        start_time = time.time()
        deployment_result = self.deployment_engine.execute_production_deployment(self.sample_deployment_config)
        deployment_time = (time.time() - start_time) * 1000
        
        self.assertTrue(deployment_result['success'])
        self.assertIn('deployment_id', deployment_result)
        # Convert minutes to milliseconds for comparison
        target_ms = self.test_config['deployment_config']['deployment_target_minutes'] * 60 * 1000
        self.assertLess(deployment_time, target_ms)
        
        # Test blue-green deployment strategy
        blue_green_config = dict(self.sample_deployment_config)
        blue_green_config['strategy'] = 'blue_green'
        blue_green_config['traffic_shift_percentage'] = 50
        
        start_time = time.time()
        bg_result = self.deployment_engine.manage_blue_green_deployment(blue_green_config)
        bg_time = (time.time() - start_time) * 1000
        
        self.assertTrue(bg_result['success'])
        self.assertEqual(bg_result['strategy'], 'blue_green')
        self.assertLess(bg_time, 60000)  # Should complete within 1 minute
        
        # Test rolling update orchestration
        rolling_config = dict(self.sample_deployment_config)
        rolling_config['strategy'] = 'rolling'
        rolling_config['max_surge'] = 1
        rolling_config['max_unavailable'] = 0
        
        start_time = time.time()
        rolling_result = self.deployment_engine.orchestrate_rolling_updates(rolling_config)
        rolling_time = (time.time() - start_time) * 1000
        
        self.assertTrue(rolling_result['success'])
        self.assertEqual(rolling_result['strategy'], 'rolling')
        self.assertLess(rolling_time, 90000)  # Should complete within 1.5 minutes
        
        # Test deployment rollback
        rollback_config = {
            'deployment_id': deployment_result.get('deployment_id', 'test_deployment_001'),
            'rollback_to_version': 'previous',
            'force_rollback': False
        }
        
        start_time = time.time()
        rollback_result = self.deployment_engine.execute_deployment_rollback(rollback_config)
        rollback_time = (time.time() - start_time) * 1000
        
        self.assertTrue(rollback_result['success'])
        self.assertLess(rollback_time, 120000)  # Should complete within 2 minutes
        
        self.test_results['deployment_engine'].extend([
            {'test': 'production_deployment', 'time_ms': deployment_time, 'passed': True},
            {'test': 'blue_green_deployment', 'time_ms': bg_time, 'passed': True},
            {'test': 'rolling_updates', 'time_ms': rolling_time, 'passed': True},
            {'test': 'deployment_rollback', 'time_ms': rollback_time, 'passed': True}
        ])
        
        print(f"DeploymentEngine tests passed - Deployment: {deployment_time:.2f}ms, Blue-Green: {bg_time:.2f}ms")
    
    def test_02_monitoring_engine_functionality(self):
        """Test MonitoringEngine core functionality and performance."""
        if not self.monitoring_engine:
            self.skipTest("MonitoringEngine not available")
        
        print("\n=== Testing MonitoringEngine Functionality ===")
        
        # Test metrics collection
        start_time = time.time()
        metrics_result = self.monitoring_engine.collect_system_metrics(self.sample_monitoring_config['metrics_config'])
        metrics_time = (time.time() - start_time) * 1000
        
        self.assertTrue(metrics_result['success'])
        self.assertLess(metrics_time, self.test_config['monitoring_config']['metrics_target_ms'])
        self.assertGreater(len(metrics_result.get('metrics', [])), 0)
        
        # Test real-time monitoring processing
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': [
                {'name': 'cpu_usage', 'value': 45.2, 'tags': {'host': 'server-01'}},
                {'name': 'memory_usage', 'value': 68.5, 'tags': {'host': 'server-01'}},
                {'name': 'throughput', 'value': 1250, 'tags': {'line': 'line-01'}}
            ]
        }
        
        start_time = time.time()
        processing_result = self.monitoring_engine.process_real_time_monitoring(monitoring_data)
        processing_time = (time.time() - start_time) * 1000
        
        self.assertTrue(processing_result['success'])
        self.assertLess(processing_time, 500)  # Should process within 500ms
        
        # Test dashboard generation
        start_time = time.time()
        dashboard_result = self.monitoring_engine.generate_monitoring_dashboards(
            self.sample_monitoring_config['dashboard_config']
        )
        dashboard_time = (time.time() - start_time) * 1000
        
        self.assertTrue(dashboard_result['success'])
        self.assertLess(dashboard_time, self.test_config['monitoring_config']['dashboard_target_ms'])
        self.assertIn('dashboard_data', dashboard_result)
        
        # Test health monitoring
        health_config = {
            'services': ['webui-service', 'control-service'],
            'check_interval_seconds': 10,
            'timeout_seconds': 5
        }
        
        start_time = time.time()
        health_result = self.monitoring_engine.monitor_system_health(health_config)
        health_time = (time.time() - start_time) * 1000
        
        self.assertTrue(health_result['success'])
        self.assertLess(health_time, 10000)  # Should complete within 10 seconds
        
        self.test_results['monitoring_engine'].extend([
            {'test': 'metrics_collection', 'time_ms': metrics_time, 'passed': True},
            {'test': 'real_time_processing', 'time_ms': processing_time, 'passed': True},
            {'test': 'dashboard_generation', 'time_ms': dashboard_time, 'passed': True},
            {'test': 'health_monitoring', 'time_ms': health_time, 'passed': True}
        ])
        
        print(f"MonitoringEngine tests passed - Metrics: {metrics_time:.2f}ms, Dashboard: {dashboard_time:.2f}ms")
    
    def test_03_alerting_engine_functionality(self):
        """Test AlertingEngine core functionality and performance."""
        if not self.alerting_engine:
            self.skipTest("AlertingEngine not available")
        
        print("\n=== Testing AlertingEngine Functionality ===")
        
        # Test alert processing
        alert_data = {
            'alert_id': 'test_alert_001',
            'severity': 'critical',
            'message': 'High CPU usage detected',
            'source': 'monitoring_system',
            'tags': {'host': 'server-01', 'service': 'manufacturing'},
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = time.time()
        alert_result = self.alerting_engine.process_alert_conditions(alert_data)
        alert_time = (time.time() - start_time) * 1000
        
        self.assertTrue(alert_result['success'])
        self.assertLess(alert_time, self.test_config['alerting_config']['alert_target_seconds'] * 1000)
        self.assertIn('alert_id', alert_result)
        
        # Test escalation policy management
        escalation_config = {
            'policy_name': 'critical_alerts',
            'escalation_levels': [
                {'level': 1, 'timeout_minutes': 5, 'channels': ['email']},
                {'level': 2, 'timeout_minutes': 15, 'channels': ['email', 'slack']},
                {'level': 3, 'timeout_minutes': 30, 'channels': ['email', 'slack', 'pagerduty']}
            ]
        }
        
        start_time = time.time()
        escalation_result = self.alerting_engine.manage_escalation_policies(escalation_config)
        escalation_time = (time.time() - start_time) * 1000
        
        self.assertTrue(escalation_result['success'])
        self.assertLess(escalation_time, 5000)  # Should complete within 5 seconds
        
        # Test multi-channel notifications
        notification_requests = [
            {
                'alert_id': 'test_alert_001',
                'channels': ['email', 'webhook'],
                'message': 'Critical alert: High CPU usage detected',
                'severity': 'critical'
            }
        ]
        
        start_time = time.time()
        notification_result = self.alerting_engine.deliver_multi_channel_notifications(notification_requests)
        notification_time = (time.time() - start_time) * 1000
        
        self.assertTrue(notification_result['success'])
        self.assertLess(notification_time, 15000)  # Should complete within 15 seconds
        self.assertGreater(notification_result.get('delivered_count', 0), 0)
        
        # Test alert correlation
        correlated_alerts = [
            {
                'alert_id': 'alert_001',
                'message': 'High CPU usage on server-01',
                'tags': {'host': 'server-01', 'metric': 'cpu'}
            },
            {
                'alert_id': 'alert_002', 
                'message': 'High memory usage on server-01',
                'tags': {'host': 'server-01', 'metric': 'memory'}
            }
        ]
        
        start_time = time.time()
        correlation_result = self.alerting_engine.correlate_alerts(correlated_alerts)
        correlation_time = (time.time() - start_time) * 1000
        
        self.assertTrue(correlation_result['success'])
        self.assertLess(correlation_time, 2000)  # Should complete within 2 seconds
        
        self.test_results['alerting_engine'].extend([
            {'test': 'alert_processing', 'time_ms': alert_time, 'passed': True},
            {'test': 'escalation_management', 'time_ms': escalation_time, 'passed': True},
            {'test': 'multi_channel_notifications', 'time_ms': notification_time, 'passed': True},
            {'test': 'alert_correlation', 'time_ms': correlation_time, 'passed': True}
        ])
        
        print(f"AlertingEngine tests passed - Processing: {alert_time:.2f}ms, Notifications: {notification_time:.2f}ms")
    
    def test_04_operations_dashboard_engine_functionality(self):
        """Test OperationsDashboardEngine core functionality and performance."""
        if not self.operations_engine:
            self.skipTest("OperationsDashboardEngine not available")
        
        print("\n=== Testing OperationsDashboardEngine Functionality ===")
        
        # Test operations dashboard rendering
        dashboard_config = {
            'dashboard_id': 'manufacturing_operations',
            'panels': [
                {'type': 'kpi', 'metric': 'throughput', 'title': 'Production Throughput'},
                {'type': 'chart', 'metric': 'efficiency', 'title': 'Overall Efficiency'},
                {'type': 'table', 'data_source': 'alerts', 'title': 'Active Alerts'}
            ],
            'refresh_interval_seconds': 10,
            'auto_refresh': True
        }
        
        start_time = time.time()
        dashboard_result = self.operations_engine.render_operations_dashboard(dashboard_config)
        dashboard_time = (time.time() - start_time) * 1000
        
        self.assertTrue(dashboard_result['success'])
        self.assertLess(dashboard_time, self.test_config['operations_config']['dashboard_target_ms'])
        self.assertIn('dashboard_data', dashboard_result)
        
        # Test analytics queries
        analytics_requests = [
            {
                'query_id': 'throughput_trend',
                'metric': 'manufacturing_throughput',
                'time_range': '24h',
                'aggregation': 'avg',
                'group_by': 'hour'
            },
            {
                'query_id': 'efficiency_analysis',
                'metric': 'overall_efficiency',
                'time_range': '7d',
                'aggregation': 'max',
                'group_by': 'day'
            }
        ]
        
        start_time = time.time()
        analytics_result = self.operations_engine.process_analytics_queries(analytics_requests)
        analytics_time = (time.time() - start_time) * 1000
        
        self.assertTrue(analytics_result['success'])
        self.assertLess(analytics_time, self.test_config['operations_config']['analytics_target_seconds'] * 1000)
        self.assertGreater(len(analytics_result.get('query_results', [])), 0)
        
        # Test operational report generation
        report_specs = [
            {
                'report_id': 'daily_production_report',
                'type': 'summary',
                'time_range': '24h',
                'format': 'json',
                'metrics': ['throughput', 'efficiency', 'quality']
            }
        ]
        
        start_time = time.time()
        report_result = self.operations_engine.generate_operational_reports(report_specs)
        report_time = (time.time() - start_time) * 1000
        
        self.assertTrue(report_result['success'])
        self.assertLess(report_time, 10000)  # Should complete within 10 seconds
        self.assertGreater(len(report_result.get('reports', [])), 0)
        
        # Test real-time data aggregation
        aggregation_config = {
            'data_sources': ['monitoring', 'deployment', 'infrastructure'],
            'metrics': ['system_health', 'deployment_status', 'resource_utilization'],
            'aggregation_window_seconds': 60
        }
        
        start_time = time.time()
        aggregation_result = self.operations_engine.aggregate_real_time_data(aggregation_config)
        aggregation_time = (time.time() - start_time) * 1000
        
        self.assertTrue(aggregation_result['success'])
        self.assertLess(aggregation_time, 3000)  # Should complete within 3 seconds
        
        self.test_results['operations_engine'].extend([
            {'test': 'dashboard_rendering', 'time_ms': dashboard_time, 'passed': True},
            {'test': 'analytics_queries', 'time_ms': analytics_time, 'passed': True},
            {'test': 'report_generation', 'time_ms': report_time, 'passed': True},
            {'test': 'data_aggregation', 'time_ms': aggregation_time, 'passed': True}
        ])
        
        print(f"OperationsDashboardEngine tests passed - Dashboard: {dashboard_time:.2f}ms, Analytics: {analytics_time:.2f}ms")
    
    def test_05_infrastructure_engine_functionality(self):
        """Test InfrastructureEngine core functionality and performance."""
        if not self.infrastructure_engine:
            self.skipTest("InfrastructureEngine not available")
        
        print("\n=== Testing InfrastructureEngine Functionality ===")
        
        # Test auto-scaling management
        scaling_policies = [
            {
                'name': 'cpu_based_scaling',
                'metric': 'cpu_utilization',
                'target_value': 70.0,
                'scale_up_threshold': 80.0,
                'scale_down_threshold': 30.0,
                'min_replicas': 2,
                'max_replicas': 10
            }
        ]
        
        start_time = time.time()
        scaling_result = self.infrastructure_engine.manage_auto_scaling(scaling_policies)
        scaling_time = (time.time() - start_time) * 1000
        
        self.assertTrue(scaling_result['success'])
        self.assertLess(scaling_time, self.test_config['infrastructure_config']['scaling_target_minutes'] * 60 * 1000)
        self.assertIn('scaling_decisions', scaling_result)
        
        # Test resource optimization
        resource_requirements = {
            'services': [
                {
                    'name': 'webui-service',
                    'current_resources': {'cpu': '500m', 'memory': '512Mi'},
                    'usage_metrics': {'cpu_avg': 45.0, 'memory_avg': 320}
                },
                {
                    'name': 'control-service',
                    'current_resources': {'cpu': '1000m', 'memory': '1Gi'},
                    'usage_metrics': {'cpu_avg': 78.0, 'memory_avg': 850}
                }
            ],
            'optimization_goals': ['cost', 'performance']
        }
        
        start_time = time.time()
        optimization_result = self.infrastructure_engine.optimize_resource_allocation(resource_requirements)
        optimization_time = (time.time() - start_time) * 1000
        
        self.assertTrue(optimization_result['success'])
        self.assertLess(optimization_time, 30000)  # Should complete within 30 seconds
        self.assertIn('optimization_recommendations', optimization_result)
        
        # Test infrastructure health monitoring
        infrastructure_data = {
            'clusters': ['production', 'staging'],
            'services': ['webui-service', 'control-service', 'monitoring-service'],
            'metrics': ['node_health', 'pod_status', 'resource_usage']
        }
        
        start_time = time.time()
        health_result = self.infrastructure_engine.monitor_infrastructure_health(infrastructure_data)
        health_time = (time.time() - start_time) * 1000
        
        self.assertTrue(health_result['success'])
        self.assertLess(health_time, 15000)  # Should complete within 15 seconds
        self.assertIn('health_status', health_result)
        
        # Test cost analysis and optimization
        cost_analysis_config = {
            'time_range': '30d',
            'resources': ['compute', 'storage', 'network'],
            'optimization_target': 'cost_efficiency'
        }
        
        start_time = time.time()
        cost_result = self.infrastructure_engine.analyze_cost_optimization(cost_analysis_config)
        cost_time = (time.time() - start_time) * 1000
        
        self.assertTrue(cost_result['success'])
        self.assertLess(cost_time, 20000)  # Should complete within 20 seconds
        
        self.test_results['infrastructure_engine'].extend([
            {'test': 'auto_scaling_management', 'time_ms': scaling_time, 'passed': True},
            {'test': 'resource_optimization', 'time_ms': optimization_time, 'passed': True},
            {'test': 'infrastructure_health', 'time_ms': health_time, 'passed': True},
            {'test': 'cost_optimization', 'time_ms': cost_time, 'passed': True}
        ])
        
        print(f"InfrastructureEngine tests passed - Scaling: {scaling_time:.2f}ms, Optimization: {optimization_time:.2f}ms")
    
    def test_06_week8_performance_benchmarks(self):
        """Test performance benchmarks for all Week 8 engines."""
        print("\n=== Running Week 8 Performance Benchmarks ===")
        
        performance_results = {}
        
        # DeploymentEngine performance test
        if self.deployment_engine:
            deployment_configs = [self.sample_deployment_config] * 3  # Multiple deployments
            
            start_time = time.time()
            for i, config in enumerate(deployment_configs):
                config_copy = dict(config)
                config_copy['deployment_id'] = f'perf_deployment_{i}'
                result = self.deployment_engine.execute_production_deployment(config_copy)
                if not result['success']:
                    break
            total_time = (time.time() - start_time) * 1000
            
            avg_deployment_time = total_time / len(deployment_configs)
            target_ms = self.test_config['deployment_config']['deployment_target_minutes'] * 60 * 1000
            self.assertLess(avg_deployment_time, target_ms)
            
            performance_results['deployment_engine'] = {
                'average_deployment_time_ms': avg_deployment_time,
                'target_ms': target_ms,
                'passed': True
            }
        
        # MonitoringEngine performance test
        if self.monitoring_engine:
            metrics_configs = self.sample_monitoring_config['metrics_config'] * 5  # Multiple metrics
            
            start_time = time.time()
            result = self.monitoring_engine.collect_system_metrics(metrics_configs)
            metrics_time = (time.time() - start_time) * 1000
            
            self.assertTrue(result['success'])
            per_metric_time = metrics_time / len(metrics_configs)
            self.assertLess(per_metric_time, self.test_config['monitoring_config']['metrics_target_ms'])
            
            performance_results['monitoring_engine'] = {
                'metrics_collection_time_ms': metrics_time,
                'per_metric_time_ms': per_metric_time,
                'target_ms': self.test_config['monitoring_config']['metrics_target_ms'],
                'passed': True
            }
        
        # AlertingEngine performance test
        if self.alerting_engine:
            alert_data_list = []
            for i in range(10):
                alert_data_list.append({
                    'alert_id': f'perf_alert_{i}',
                    'severity': 'warning' if i % 2 == 0 else 'critical',
                    'message': f'Performance test alert {i}',
                    'timestamp': datetime.now().isoformat()
                })
            
            start_time = time.time()
            for alert_data in alert_data_list:
                result = self.alerting_engine.process_alert_conditions(alert_data)
                if not result['success']:
                    break
            total_time = (time.time() - start_time) * 1000
            
            avg_alert_time = total_time / len(alert_data_list)
            target_ms = self.test_config['alerting_config']['alert_target_seconds'] * 1000
            self.assertLess(avg_alert_time, target_ms)
            
            performance_results['alerting_engine'] = {
                'average_alert_processing_time_ms': avg_alert_time,
                'target_ms': target_ms,
                'passed': True
            }
        
        self.test_results['performance'].append({
            'test': 'week8_performance_benchmarks',
            'results': performance_results,
            'passed': True
        })
        
        print(f"Week 8 Performance Benchmarks completed - All engines meeting targets")
    
    def test_07_week8_integration_testing(self):
        """Test integration between Week 8 engines."""
        print("\n=== Running Week 8 Integration Tests ===")
        
        if not all([self.deployment_engine, self.monitoring_engine]):
            self.skipTest("Required engines not available for integration testing")
        
        # Test Deployment and Monitoring integration
        # Deploy a service and monitor its health
        deployment_config = dict(self.sample_deployment_config)
        deployment_result = self.deployment_engine.execute_production_deployment(deployment_config)
        self.assertTrue(deployment_result['success'])
        
        # Monitor the deployed services
        if self.monitoring_engine:
            health_config = {
                'services': [service['name'] for service in deployment_config['services']],
                'check_interval_seconds': 5,
                'timeout_seconds': 30
            }
            
            health_result = self.monitoring_engine.monitor_system_health(health_config)
            self.assertTrue(health_result['success'])
        
        # Test Monitoring and Alerting integration
        if self.alerting_engine:
            # Create an alert based on monitoring data
            alert_data = {
                'alert_id': 'integration_test_alert',
                'severity': 'warning',
                'message': 'Integration test alert from monitoring',
                'source': 'monitoring_engine',
                'timestamp': datetime.now().isoformat()
            }
            
            alert_result = self.alerting_engine.process_alert_conditions(alert_data)
            self.assertTrue(alert_result['success'])
        
        # Test Operations Dashboard integration
        if self.operations_engine:
            # Create a dashboard that shows deployment and monitoring data
            dashboard_config = {
                'dashboard_id': 'integration_test_dashboard',
                'panels': [
                    {'type': 'deployment_status', 'title': 'Deployment Status'},
                    {'type': 'system_health', 'title': 'System Health'},
                    {'type': 'active_alerts', 'title': 'Active Alerts'}
                ]
            }
            
            dashboard_result = self.operations_engine.render_operations_dashboard(dashboard_config)
            self.assertTrue(dashboard_result['success'])
        
        print("Week 8 Integration tests passed")
    
    def test_08_week8_error_handling_and_resilience(self):
        """Test error handling and system resilience for Week 8 engines."""
        print("\n=== Testing Week 8 Error Handling and Resilience ===")
        
        # Test DeploymentEngine error handling
        if self.deployment_engine:
            invalid_deployment_config = {
                'deployment_id': 'invalid_deployment',
                'strategy': 'nonexistent_strategy',
                'services': []  # Empty services
            }
            
            result = self.deployment_engine.execute_production_deployment(invalid_deployment_config)
            # Should handle gracefully
            self.assertIsInstance(result, dict)
        
        # Test MonitoringEngine error handling
        if self.monitoring_engine:
            invalid_metrics_config = [
                {
                    'name': '',  # Empty name
                    'type': 'invalid_type',
                    'collection_interval_seconds': -1  # Invalid interval
                }
            ]
            
            result = self.monitoring_engine.collect_system_metrics(invalid_metrics_config)
            # Should handle gracefully
            self.assertIsInstance(result, dict)
        
        # Test AlertingEngine error handling
        if self.alerting_engine:
            invalid_alert_data = {
                'alert_id': '',  # Empty ID
                'severity': 'invalid_severity',
                'message': None  # Null message
            }
            
            result = self.alerting_engine.process_alert_conditions(invalid_alert_data)
            # Should handle gracefully
            self.assertIsInstance(result, dict)
        
        print("Week 8 Error handling tests completed")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after testing."""
        # Generate test report
        cls._generate_test_report()
        
        print("\n=== Week 8 Deployment & Monitoring Complete ===")
    
    @classmethod
    def _generate_test_report(cls):
        """Generate comprehensive test report."""
        report = {
            'test_suite': 'Week 8 Deployment & Monitoring',
            'execution_time': datetime.now().isoformat(),
            'engines_tested': {
                'DeploymentEngine': cls.deployment_engine is not None,
                'MonitoringEngine': cls.monitoring_engine is not None,
                'AlertingEngine': cls.alerting_engine is not None,
                'OperationsDashboardEngine': cls.operations_engine is not None,
                'InfrastructureEngine': cls.infrastructure_engine is not None
            },
            'performance_targets': {
                'DeploymentEngine Production Deployment': '< 5 minutes',
                'MonitoringEngine Metrics Collection': '< 100ms',
                'MonitoringEngine Dashboard Updates': '< 1 second',
                'AlertingEngine Alert Processing': '< 30 seconds',
                'OperationsDashboardEngine Dashboard Rendering': '< 500ms',
                'OperationsDashboardEngine Analytics Queries': '< 2 seconds',
                'InfrastructureEngine Scaling Operations': '< 2 minutes'
            },
            'test_results': cls.test_results
        }
        
        # Save report
        report_dir = os.path.join(project_root, 'testing', 'results')
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f'week8_deployment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTest report saved to: {report_file}")


def run_week8_tests(verbose=False):
    """Run Week 8 comprehensive tests."""
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("Week 8 Deployment & Monitoring - Comprehensive Test Suite")
    print("="*60)
    print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    
    # Check engine availability
    available_engines = []
    if DeploymentEngine:
        available_engines.append("DeploymentEngine")
    if MonitoringEngine:
        available_engines.append("MonitoringEngine")
    if AlertingEngine:
        available_engines.append("AlertingEngine")
    if OperationsDashboardEngine:
        available_engines.append("OperationsDashboardEngine")
    if InfrastructureEngine:
        available_engines.append("InfrastructureEngine")
    
    print(f"Available engines: {', '.join(available_engines)}")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(Week8DeploymentTestSuite)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("WEEK 8 TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")
    
    if result.errors:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"- {test}: {error}")
    
    print("\nWeek 8 Performance Summary:")
    print("- DeploymentEngine: Zero-downtime deployments with blue-green and canary strategies")
    print("- MonitoringEngine: Real-time metrics collection and dashboard generation")
    print("- AlertingEngine: Intelligent alerting with multi-channel notifications")
    print("- OperationsDashboardEngine: Production operations analytics and reporting")
    print("- InfrastructureEngine: Auto-scaling and resource optimization")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Week 8 Deployment & Monitoring Comprehensive Test Suite')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--engines', nargs='*', 
                       choices=['deployment', 'monitoring', 'alerting', 'operations', 'infrastructure'],
                       help='Specific engines to test')
    
    args = parser.parse_args()
    
    success = run_week8_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)