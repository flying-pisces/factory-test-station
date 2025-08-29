#!/usr/bin/env python3
"""
Week 6 Comprehensive Test Runner - Advanced UI & Visualization

This script runs comprehensive tests for all Week 6 UI layer components:
- WebUIEngine: Web-based user interface testing
- VisualizationEngine: Data visualization and charting testing
- ControlInterfaceEngine: Interactive control interface testing
- UserManagementEngine: Authentication and authorization testing
- MobileInterfaceEngine: Mobile interface and offline capability testing

Performance Targets:
- WebUIEngine: <100ms UI response times
- VisualizationEngine: <50ms chart rendering
- ControlInterfaceEngine: <75ms control execution
- UserManagementEngine: <200ms authentication
- MobileInterfaceEngine: <150ms mobile responsiveness
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

# Week 6 UI Layer imports
try:
    from layers.ui_layer.webui_engine import WebUIEngine
    from layers.ui_layer.visualization_engine import VisualizationEngine
    from layers.ui_layer.control_interface_engine import ControlInterfaceEngine
    from layers.ui_layer.user_management_engine import UserManagementEngine
    from layers.ui_layer.mobile_interface_engine import MobileInterfaceEngine
except ImportError as e:
    print(f"Warning: Could not import Week 6 UI engines: {e}")
    WebUIEngine = None
    VisualizationEngine = None
    ControlInterfaceEngine = None
    UserManagementEngine = None
    MobileInterfaceEngine = None

# Testing framework imports (using unittest instead of custom framework)


class Week6UITestSuite(unittest.TestCase):
    """Comprehensive test suite for Week 6 UI layer components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment for Week 6 UI testing."""
        cls.test_config = {
            'webui_config': {
                'response_target_ms': 100,
                'max_concurrent_users': 50,
                'enable_websockets': True
            },
            'visualization_config': {
                'render_target_ms': 50,
                'max_concurrent_charts': 20,
                'enable_real_time': True
            },
            'control_config': {
                'control_target_ms': 75,
                'emergency_response_ms': 25,
                'enable_safety_interlocks': True
            },
            'user_management_config': {
                'auth_target_ms': 200,
                'session_timeout_minutes': 30,
                'max_login_attempts': 3
            },
            'mobile_config': {
                'mobile_target_ms': 150,
                'offline_cache_size': 1000,
                'sync_interval_seconds': 30
            }
        }
        
        # Initialize engines for testing
        cls.webui_engine = WebUIEngine(cls.test_config['webui_config']) if WebUIEngine else None
        cls.visualization_engine = VisualizationEngine(cls.test_config['visualization_config']) if VisualizationEngine else None
        cls.control_engine = ControlInterfaceEngine(cls.test_config['control_config']) if ControlInterfaceEngine else None
        cls.user_engine = UserManagementEngine(cls.test_config['user_management_config']) if UserManagementEngine else None
        cls.mobile_engine = MobileInterfaceEngine(cls.test_config['mobile_config']) if MobileInterfaceEngine else None
        
        cls.test_results = {
            'webui': [],
            'visualization': [],
            'control': [],
            'user_management': [],
            'mobile': [],
            'performance': []
        }
        
        # Test data
        cls.sample_dashboard_config = {
            'dashboard_id': 'test_dashboard',
            'layout': 'responsive',
            'components': [
                {'type': 'kpi_card', 'metric': 'production_rate'},
                {'type': 'chart', 'chart_type': 'line', 'data_source': 'throughput'},
                {'type': 'control_panel', 'scope': 'system'}
            ]
        }
        
        cls.sample_chart_specs = [
            {
                'chart_id': 'throughput_chart',
                'chart_type': 'line',
                'data_source': 'production_throughput',
                'update_interval_ms': 1000
            },
            {
                'chart_id': 'efficiency_gauge',
                'chart_type': 'gauge',
                'data_source': 'overall_efficiency',
                'thresholds': [60, 80, 95]
            }
        ]
        
        cls.sample_control_commands = [
            {
                'command_id': 'start_production',
                'action_type': 'START',
                'scope': 'system',
                'parameters': {'mode': 'auto'}
            },
            {
                'command_id': 'emergency_stop',
                'action_type': 'EMERGENCY_STOP',
                'scope': 'system',
                'priority': 'critical'
            }
        ]
    
    def test_01_webui_engine_functionality(self):
        """Test WebUIEngine core functionality."""
        if not self.webui_engine:
            self.skipTest("WebUIEngine not available")
        
        print("\n=== Testing WebUIEngine Functionality ===")
        
        # Test dashboard rendering
        start_time = time.time()
        dashboard_result = self.webui_engine.render_real_time_dashboard(self.sample_dashboard_config)
        render_time = (time.time() - start_time) * 1000
        
        self.assertTrue(dashboard_result['success'])
        self.assertLess(render_time, self.test_config['webui_config']['response_target_ms'])
        
        # Test user interactions
        interaction_data = {
            'type': 'button_click',
            'component_id': 'start_button',
            'user_id': 'test_user'
        }
        
        start_time = time.time()
        interaction_result = self.webui_engine.handle_user_interactions(interaction_data)
        interaction_time = (time.time() - start_time) * 1000
        
        self.assertTrue(interaction_result['success'])
        self.assertLess(interaction_time, self.test_config['webui_config']['response_target_ms'])
        
        # Test real-time updates
        update_data = {
            'component_updates': [
                {'component_id': 'throughput_display', 'value': 1250.5},
                {'component_id': 'status_indicator', 'status': 'running'}
            ]
        }
        
        start_time = time.time()
        update_result = self.webui_engine.update_ui_components(update_data)
        update_time = (time.time() - start_time) * 1000
        
        self.assertTrue(update_result['success'])
        self.assertLess(update_time, self.test_config['webui_config']['response_target_ms'])
        
        self.test_results['webui'].extend([
            {'test': 'dashboard_render', 'time_ms': render_time, 'passed': True},
            {'test': 'user_interaction', 'time_ms': interaction_time, 'passed': True},
            {'test': 'real_time_update', 'time_ms': update_time, 'passed': True}
        ])
        
        print(f"WebUIEngine tests passed - Dashboard: {render_time:.2f}ms, Interaction: {interaction_time:.2f}ms")
    
    def test_02_visualization_engine_functionality(self):
        """Test VisualizationEngine core functionality."""
        if not self.visualization_engine:
            self.skipTest("VisualizationEngine not available")
        
        print("\n=== Testing VisualizationEngine Functionality ===")
        
        # Test chart creation
        start_time = time.time()
        chart_result = self.visualization_engine.create_real_time_charts(self.sample_chart_specs)
        chart_time = (time.time() - start_time) * 1000
        
        self.assertTrue(chart_result['success'])
        self.assertLess(chart_time, self.test_config['visualization_config']['render_target_ms'])
        self.assertGreater(len(chart_result['charts']), 0)
        
        # Test KPI dashboard generation
        kpi_specs = [
            {
                'dashboard_id': 'production_kpis',
                'kpis': [
                    {'name': 'Throughput', 'value': 1250, 'unit': 'units/hr'},
                    {'name': 'Efficiency', 'value': 87.5, 'unit': '%'},
                    {'name': 'Quality', 'value': 99.2, 'unit': '%'}
                ]
            }
        ]
        
        start_time = time.time()
        kpi_result = self.visualization_engine.generate_kpi_dashboards(kpi_specs)
        kpi_time = (time.time() - start_time) * 1000
        
        self.assertTrue(kpi_result['success'])
        self.assertLess(kpi_time, self.test_config['visualization_config']['render_target_ms'])
        
        # Test 3D visualization
        system_data = {
            'factory_layout': {
                'stations': [
                    {'id': 'station_1', 'position': [0, 0, 0], 'status': 'running'},
                    {'id': 'station_2', 'position': [10, 0, 0], 'status': 'idle'}
                ]
            }
        }
        
        start_time = time.time()
        viz_3d_result = self.visualization_engine.render_3d_visualizations(system_data)
        viz_3d_time = (time.time() - start_time) * 1000
        
        self.assertTrue(viz_3d_result['success'])
        self.assertLess(viz_3d_time, self.test_config['visualization_config']['render_target_ms'])
        
        self.test_results['visualization'].extend([
            {'test': 'chart_creation', 'time_ms': chart_time, 'passed': True},
            {'test': 'kpi_dashboard', 'time_ms': kpi_time, 'passed': True},
            {'test': '3d_visualization', 'time_ms': viz_3d_time, 'passed': True}
        ])
        
        print(f"VisualizationEngine tests passed - Charts: {chart_time:.2f}ms, KPI: {kpi_time:.2f}ms")
    
    def test_03_control_interface_engine_functionality(self):
        """Test ControlInterfaceEngine core functionality."""
        if not self.control_engine:
            self.skipTest("ControlInterfaceEngine not available")
        
        print("\n=== Testing ControlInterfaceEngine Functionality ===")
        
        # Test control command processing
        user_context = {
            'user_id': 'test_operator',
            'role': 'operator',
            'permissions': ['system_start', 'system_stop', 'emergency_stop']
        }
        
        start_time = time.time()
        control_result = self.control_engine.process_control_commands(
            self.sample_control_commands, user_context
        )
        control_time = (time.time() - start_time) * 1000
        
        self.assertTrue(control_result['success'])
        self.assertLess(control_time, self.test_config['control_config']['control_target_ms'])
        
        # Test emergency interface handling
        emergency_data = {
            'emergency_type': 'equipment_failure',
            'severity': 'high',
            'affected_equipment': ['station_1', 'conveyor_2'],
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = time.time()
        emergency_result = self.control_engine.handle_emergency_interfaces(emergency_data)
        emergency_time = (time.time() - start_time) * 1000
        
        self.assertTrue(emergency_result['success'])
        self.assertLess(emergency_time, self.test_config['control_config']['emergency_response_ms'])
        
        # Test system configuration management
        config_changes = [
            {
                'parameter': 'production_rate_limit',
                'value': 1500,
                'scope': 'system'
            }
        ]
        
        start_time = time.time()
        config_result = self.control_engine.manage_system_configuration(
            config_changes, user_context
        )
        config_time = (time.time() - start_time) * 1000
        
        self.assertTrue(config_result['success'])
        self.assertLess(config_time, self.test_config['control_config']['control_target_ms'])
        
        self.test_results['control'].extend([
            {'test': 'control_commands', 'time_ms': control_time, 'passed': True},
            {'test': 'emergency_handling', 'time_ms': emergency_time, 'passed': True},
            {'test': 'configuration_mgmt', 'time_ms': config_time, 'passed': True}
        ])
        
        print(f"ControlInterfaceEngine tests passed - Control: {control_time:.2f}ms, Emergency: {emergency_time:.2f}ms")
    
    def test_04_user_management_engine_functionality(self):
        """Test UserManagementEngine core functionality."""
        if not self.user_engine:
            self.skipTest("UserManagementEngine not available")
        
        print("\n=== Testing UserManagementEngine Functionality ===")
        
        # Test user authentication
        credentials = {
            'username': 'admin',
            'password': 'admin123'
        }
        
        start_time = time.time()
        auth_result = self.user_engine.authenticate_users(credentials)
        auth_time = (time.time() - start_time) * 1000
        
        self.assertTrue(auth_result['success'])
        self.assertLess(auth_time, self.test_config['user_management_config']['auth_target_ms'])
        
        session_id = auth_result['session']['session_id']
        user_permissions = {
            'session_id': session_id,
            'username': 'admin',
            'role': 'administrator'
        }
        
        # Test role-based access control
        requested_actions = [
            {'action_type': 'system_start', 'required_permission': 'system_start'},
            {'action_type': 'user_create', 'required_permission': 'user_create'},
            {'action_type': 'data_export', 'required_permission': 'data_export'}
        ]
        
        start_time = time.time()
        rbac_result = self.user_engine.enforce_role_based_access(
            user_permissions, requested_actions
        )
        rbac_time = (time.time() - start_time) * 1000
        
        self.assertTrue(rbac_result['success'])
        self.assertTrue(rbac_result['all_allowed'])  # Admin should have all permissions
        
        # Test audit logging
        user_actions = [
            {
                'username': 'admin',
                'action_type': 'login',
                'details': {'session_id': session_id},
                'result': 'success'
            }
        ]
        
        start_time = time.time()
        audit_result = self.user_engine.manage_audit_logging(user_actions)
        audit_time = (time.time() - start_time) * 1000
        
        self.assertTrue(audit_result['success'])
        self.assertEqual(audit_result['logged_count'], len(user_actions))
        
        # Test user creation
        new_user_data = {
            'username': 'test_operator',
            'password': 'TestPass123!',
            'role': 'operator',
            'full_name': 'Test Operator',
            'email': 'operator@test.com'
        }
        
        start_time = time.time()
        create_result = self.user_engine.create_user(new_user_data, 'admin')
        create_time = (time.time() - start_time) * 1000
        
        self.assertTrue(create_result['success'])
        
        # Test logout
        logout_result = self.user_engine.logout_user(session_id)
        self.assertTrue(logout_result['success'])
        
        self.test_results['user_management'].extend([
            {'test': 'authentication', 'time_ms': auth_time, 'passed': True},
            {'test': 'access_control', 'time_ms': rbac_time, 'passed': True},
            {'test': 'audit_logging', 'time_ms': audit_time, 'passed': True},
            {'test': 'user_creation', 'time_ms': create_time, 'passed': True}
        ])
        
        print(f"UserManagementEngine tests passed - Auth: {auth_time:.2f}ms, RBAC: {rbac_time:.2f}ms")
    
    def test_05_mobile_interface_engine_functionality(self):
        """Test MobileInterfaceEngine core functionality."""
        if not self.mobile_engine:
            self.skipTest("MobileInterfaceEngine not available")
        
        print("\n=== Testing MobileInterfaceEngine Functionality ===")
        
        # Register a test device
        device_info = {
            'device_type': 'smartphone',
            'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
            'screen_size': {'width': 375, 'height': 812},
            'capabilities': {'touch': True, 'offline': True, 'push_notifications': True}
        }
        
        register_result = self.mobile_engine.register_device(device_info)
        self.assertTrue(register_result['success'])
        device_id = register_result['device_id']
        
        # Test mobile dashboard rendering
        mobile_specs = [
            {
                'device_id': device_id,
                'device_type': 'smartphone',
                'orientation': 'portrait',
                'screen_size': {'width': 375, 'height': 812},
                'components': [
                    {'type': 'status_card', 'title': 'System Status'},
                    {'type': 'chart', 'chart_type': 'line', 'simplified': True}
                ]
            }
        ]
        
        start_time = time.time()
        dashboard_result = self.mobile_engine.render_mobile_dashboards(mobile_specs)
        dashboard_time = (time.time() - start_time) * 1000
        
        self.assertTrue(dashboard_result['success'])
        self.assertLess(dashboard_time, self.test_config['mobile_config']['mobile_target_ms'])
        
        # Test offline capabilities
        offline_data = {
            'operation_type': 'cache_data',
            'device_id': device_id,
            'cache_items': [
                {'type': 'kpi_data', 'data': {'throughput': 1250, 'efficiency': 87.5}},
                {'type': 'alert_data', 'data': {'active_alerts': 2}}
            ]
        }
        
        start_time = time.time()
        offline_result = self.mobile_engine.handle_offline_capabilities(offline_data)
        offline_time = (time.time() - start_time) * 1000
        
        self.assertTrue(offline_result['success'])
        
        # Test push notifications
        notification_data = {
            'type': 'alert',
            'title': 'Production Alert',
            'message': 'Station 1 efficiency below threshold',
            'target_type': 'specific',
            'device_ids': [device_id]
        }
        
        start_time = time.time()
        notification_result = self.mobile_engine.manage_push_notifications(notification_data)
        notification_time = (time.time() - start_time) * 1000
        
        self.assertTrue(notification_result['success'])
        self.assertEqual(notification_result['targets'], 1)
        
        # Test touch interactions
        interaction_data = {
            'device_id': device_id,
            'touch_events': [
                {
                    'type': 'tap',
                    'duration': 150,
                    'distance': 5,
                    'position': {'x': 100, 'y': 200}
                }
            ]
        }
        
        start_time = time.time()
        touch_result = self.mobile_engine.handle_touch_interaction(interaction_data)
        touch_time = (time.time() - start_time) * 1000
        
        self.assertTrue(touch_result['success'])
        self.assertGreater(touch_result['gestures_recognized'], 0)
        
        self.test_results['mobile'].extend([
            {'test': 'dashboard_render', 'time_ms': dashboard_time, 'passed': True},
            {'test': 'offline_handling', 'time_ms': offline_time, 'passed': True},
            {'test': 'push_notifications', 'time_ms': notification_time, 'passed': True},
            {'test': 'touch_interaction', 'time_ms': touch_time, 'passed': True}
        ])
        
        print(f"MobileInterfaceEngine tests passed - Dashboard: {dashboard_time:.2f}ms, Offline: {offline_time:.2f}ms")
    
    def test_06_performance_benchmarks(self):
        """Test performance benchmarks for all Week 6 engines."""
        print("\n=== Running Performance Benchmarks ===")
        
        # Concurrent user simulation for WebUI
        if self.webui_engine:
            concurrent_users = 10
            results = []
            
            def simulate_user():
                start = time.time()
                self.webui_engine.render_real_time_dashboard(self.sample_dashboard_config)
                return (time.time() - start) * 1000
            
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(simulate_user) for _ in range(concurrent_users)]
                results = [f.result() for f in as_completed(futures)]
            
            avg_concurrent_time = sum(results) / len(results)
            max_concurrent_time = max(results)
            
            self.assertLess(avg_concurrent_time, self.test_config['webui_config']['response_target_ms'])
            print(f"WebUI Concurrent Performance - Avg: {avg_concurrent_time:.2f}ms, Max: {max_concurrent_time:.2f}ms")
        
        # Chart rendering performance for Visualization
        if self.visualization_engine:
            chart_counts = [1, 5, 10, 20]
            for count in chart_counts:
                charts = [self.sample_chart_specs[0]] * count
                
                start_time = time.time()
                result = self.visualization_engine.create_real_time_charts(charts)
                render_time = (time.time() - start_time) * 1000
                
                if result['success']:
                    per_chart_time = render_time / count
                    self.assertLess(per_chart_time, self.test_config['visualization_config']['render_target_ms'])
                    print(f"Visualization Performance ({count} charts) - Total: {render_time:.2f}ms, Per chart: {per_chart_time:.2f}ms")
        
        # Authentication load testing for UserManagement
        if self.user_engine:
            auth_attempts = 50
            success_count = 0
            total_time = 0
            
            for i in range(auth_attempts):
                start_time = time.time()
                result = self.user_engine.authenticate_users({
                    'username': 'admin',
                    'password': 'admin123'
                })
                auth_time = (time.time() - start_time) * 1000
                total_time += auth_time
                
                if result['success']:
                    success_count += 1
                    # Logout immediately
                    self.user_engine.logout_user(result['session']['session_id'])
            
            avg_auth_time = total_time / auth_attempts
            success_rate = (success_count / auth_attempts) * 100
            
            self.assertLess(avg_auth_time, self.test_config['user_management_config']['auth_target_ms'])
            self.assertGreaterEqual(success_rate, 95.0)  # At least 95% success rate
            print(f"Authentication Performance - Avg: {avg_auth_time:.2f}ms, Success rate: {success_rate:.1f}%")
        
        self.test_results['performance'].append({
            'test': 'performance_benchmarks',
            'webui_concurrent_avg_ms': avg_concurrent_time if self.webui_engine else None,
            'visualization_per_chart_ms': per_chart_time if self.visualization_engine else None,
            'auth_avg_ms': avg_auth_time if self.user_engine else None,
            'passed': True
        })
    
    def test_07_integration_testing(self):
        """Test integration between Week 6 UI engines."""
        print("\n=== Running Integration Tests ===")
        
        if not all([self.webui_engine, self.user_engine]):
            self.skipTest("Required engines not available for integration testing")
        
        # Test user authentication with WebUI
        auth_result = self.user_engine.authenticate_users({
            'username': 'admin',
            'password': 'admin123'
        })
        self.assertTrue(auth_result['success'])
        
        # Test WebUI with authenticated user context
        dashboard_config = dict(self.sample_dashboard_config)
        dashboard_config['user_context'] = {
            'session_id': auth_result['session']['session_id'],
            'username': 'admin',
            'role': 'administrator'
        }
        
        dashboard_result = self.webui_engine.render_real_time_dashboard(dashboard_config)
        self.assertTrue(dashboard_result['success'])
        
        # Test visualization integration with control
        if self.visualization_engine and self.control_engine:
            # Simulate control action triggering visualization update
            control_result = self.control_engine.process_control_commands([{
                'command_id': 'update_display',
                'action_type': 'CONFIGURE',
                'scope': 'visualization',
                'parameters': {'chart_id': 'throughput_chart', 'refresh': True}
            }], {'user_id': 'admin', 'role': 'administrator'})
            
            if control_result['success']:
                # Update visualization based on control action
                viz_result = self.visualization_engine.create_real_time_charts([{
                    'chart_id': 'throughput_chart',
                    'chart_type': 'line',
                    'data_source': 'updated_throughput'
                }])
                self.assertTrue(viz_result['success'])
        
        # Clean up
        self.user_engine.logout_user(auth_result['session']['session_id'])
        
        print("Integration tests passed")
    
    def test_08_error_handling_and_resilience(self):
        """Test error handling and system resilience."""
        print("\n=== Testing Error Handling and Resilience ===")
        
        # Test invalid authentication
        if self.user_engine:
            invalid_auth = self.user_engine.authenticate_users({
                'username': 'invalid_user',
                'password': 'wrong_password'
            })
            self.assertFalse(invalid_auth['success'])
        
        # Test invalid control commands
        if self.control_engine:
            invalid_control = self.control_engine.process_control_commands([{
                'command_id': 'invalid_command',
                'action_type': 'INVALID_ACTION',
                'scope': 'unknown'
            }], {'user_id': 'test', 'role': 'viewer'})
            # Should handle gracefully
            self.assertIsInstance(invalid_control, dict)
        
        # Test invalid visualization requests
        if self.visualization_engine:
            invalid_viz = self.visualization_engine.create_real_time_charts([{
                'chart_id': 'invalid_chart',
                'chart_type': 'nonexistent_type'
            }])
            # Should handle gracefully
            self.assertIsInstance(invalid_viz, dict)
        
        print("Error handling tests completed")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after testing."""
        # Generate test report
        cls._generate_test_report()
        
        print("\n=== Week 6 UI Testing Complete ===")
    
    @classmethod
    def _generate_test_report(cls):
        """Generate comprehensive test report."""
        report = {
            'test_suite': 'Week 6 UI Layer',
            'execution_time': datetime.now().isoformat(),
            'engines_tested': {
                'WebUIEngine': cls.webui_engine is not None,
                'VisualizationEngine': cls.visualization_engine is not None,
                'ControlInterfaceEngine': cls.control_engine is not None,
                'UserManagementEngine': cls.user_engine is not None,
                'MobileInterfaceEngine': cls.mobile_engine is not None
            },
            'performance_targets': {
                'WebUI Response': '< 100ms',
                'Visualization Render': '< 50ms',
                'Control Execution': '< 75ms',
                'Authentication': '< 200ms',
                'Mobile Response': '< 150ms'
            },
            'test_results': cls.test_results
        }
        
        # Save report
        report_dir = os.path.join(project_root, 'testing', 'results')
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f'week6_ui_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTest report saved to: {report_file}")


def run_week6_tests(verbose=False):
    """Run Week 6 comprehensive tests."""
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("Week 6 Advanced UI & Visualization - Comprehensive Test Suite")
    print("="*60)
    print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    
    # Check engine availability
    available_engines = []
    if WebUIEngine:
        available_engines.append("WebUIEngine")
    if VisualizationEngine:
        available_engines.append("VisualizationEngine")
    if ControlInterfaceEngine:
        available_engines.append("ControlInterfaceEngine")
    if UserManagementEngine:
        available_engines.append("UserManagementEngine")
    if MobileInterfaceEngine:
        available_engines.append("MobileInterfaceEngine")
    
    print(f"Available engines: {', '.join(available_engines)}")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(Week6UITestSuite)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("WEEK 6 TEST SUMMARY")
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
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Week 6 UI Layer Comprehensive Test Suite')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--engines', nargs='*', 
                       choices=['webui', 'visualization', 'control', 'user', 'mobile'],
                       help='Specific engines to test')
    
    args = parser.parse_args()
    
    success = run_week6_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)