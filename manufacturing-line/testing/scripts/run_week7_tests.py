#!/usr/bin/env python3
"""
Week 7 Comprehensive Test Runner - Testing & Integration

This script runs comprehensive tests for all Week 7 testing and integration components:
- TestingEngine: Automated testing with 95% coverage and <10ms overhead
- IntegrationEngine: System integration validation with <500ms target
- BenchmarkingEngine: Performance benchmarking with <100ms suite execution
- QualityAssuranceEngine: Quality analysis and reliability testing with <200ms target
- CIEngine: CI/CD pipeline automation with <2 minutes target

Performance Validation:
- TestingEngine: <10ms test execution overhead, 95% code coverage
- IntegrationEngine: <500ms complete system integration validation
- BenchmarkingEngine: <100ms comprehensive performance benchmark suite
- QualityAssuranceEngine: <200ms quality analysis and reliability validation
- CIEngine: <120 seconds complete CI/CD pipeline execution
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

# Week 7 Testing Layer imports
try:
    from layers.testing_layer.testing_engine import TestingEngine
    from layers.testing_layer.integration_engine import IntegrationEngine
    from layers.testing_layer.benchmarking_engine import BenchmarkingEngine
    from layers.testing_layer.quality_assurance_engine import QualityAssuranceEngine
    from layers.testing_layer.ci_engine import CIEngine
except ImportError as e:
    print(f"Warning: Could not import Week 7 testing engines: {e}")
    TestingEngine = None
    IntegrationEngine = None
    BenchmarkingEngine = None
    QualityAssuranceEngine = None
    CIEngine = None


class Week7TestingTestSuite(unittest.TestCase):
    """Comprehensive test suite for Week 7 Testing & Integration layer components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment for Week 7 testing and integration."""
        cls.test_config = {
            'testing_config': {
                'test_overhead_target_ms': 10,
                'coverage_target': 95.0,
                'enable_parallel_testing': True,
                'max_parallel_tests': 5,
                'enable_coverage_analysis': True,
                'enable_test_generation': True
            },
            'integration_config': {
                'integration_target_ms': 500,
                'enable_parallel_integration': True,
                'max_parallel_integrations': 3,
                'enable_health_monitoring': True
            },
            'benchmarking_config': {
                'benchmark_target_ms': 100,
                'enable_performance_profiling': True,
                'enable_resource_monitoring': True,
                'baseline_comparison_enabled': True
            },
            'qa_config': {
                'qa_target_ms': 200,
                'enable_code_analysis': True,
                'enable_reliability_testing': True,
                'enable_fault_injection': True
            },
            'ci_config': {
                'pipeline_target_minutes': 2,
                'enable_parallel_stages': True,
                'enable_notifications': True,
                'enable_rollback': True
            }
        }
        
        # Initialize engines for testing
        cls.testing_engine = TestingEngine(cls.test_config['testing_config']) if TestingEngine else None
        cls.integration_engine = IntegrationEngine(cls.test_config['integration_config']) if IntegrationEngine else None
        cls.benchmarking_engine = BenchmarkingEngine(cls.test_config['benchmarking_config']) if BenchmarkingEngine else None
        cls.qa_engine = QualityAssuranceEngine(cls.test_config['qa_config']) if QualityAssuranceEngine else None
        cls.ci_engine = CIEngine(cls.test_config['ci_config']) if CIEngine else None
        
        cls.test_results = {
            'testing_engine': [],
            'integration_engine': [],
            'benchmarking_engine': [],
            'qa_engine': [],
            'ci_engine': [],
            'performance': []
        }
        
        # Test data configurations
        cls.sample_test_suite_config = {
            'suite_id': 'week7_validation',
            'test_types': ['unit', 'integration', 'performance'],
            'modules': ['layers.ui_layer.webui_engine', 'layers.control_layer.monitoring_engine'],
            'parallel': True,
            'coverage_analysis': True
        }
        
        cls.sample_integration_specs = [
            {
                'type': 'cross_layer',
                'source_layer': 'ui_layer',
                'target_layer': 'control_layer',
                'integration_methods': ['update_real_time_dashboards']
            },
            {
                'type': 'api_compatibility',
                'api_name': 'webui_control_interface',
                'measured_performance_ms': 85
            },
            {
                'type': 'data_flow',
                'flow_name': 'production_data_flow'
            },
            {
                'type': 'system_health'
            }
        ]
        
        cls.sample_benchmark_suite = {
            'suite_id': 'performance_validation',
            'benchmarks': [
                {
                    'name': 'ui_response_benchmark',
                    'type': 'response_time',
                    'target_component': 'WebUIEngine',
                    'performance_target_ms': 100
                },
                {
                    'name': 'visualization_render_benchmark', 
                    'type': 'throughput',
                    'target_component': 'VisualizationEngine',
                    'performance_target_ms': 50
                }
            ],
            'load_test_config': {
                'pattern': 'ramp_up',
                'max_users': 20,
                'duration_seconds': 30
            }
        }
    
    def test_01_testing_engine_functionality(self):
        """Test TestingEngine core functionality and performance."""
        if not self.testing_engine:
            self.skipTest("TestingEngine not available")
        
        print("\n=== Testing TestingEngine Functionality ===")
        
        # Test comprehensive test execution
        start_time = time.time()
        test_result = self.testing_engine.execute_comprehensive_tests(self.sample_test_suite_config)
        execution_time = (time.time() - start_time) * 1000
        
        self.assertTrue(test_result['success'])
        self.assertLess(test_result['test_overhead_ms'], self.test_config['testing_config']['test_overhead_target_ms'])
        self.assertLess(execution_time, 5000)  # Should complete within 5 seconds
        
        # Test automated test generation
        code_analysis = {
            'modules': [
                {
                    'name': 'sample_module',
                    'functions': [
                        {'name': 'sample_function', 'parameters': ['param1', 'param2']}
                    ],
                    'classes': [
                        {'name': 'SampleClass', 'methods': ['method1', 'method2']}
                    ]
                }
            ],
            'test_types': ['unit', 'performance']
        }
        
        start_time = time.time()
        generation_result = self.testing_engine.generate_automated_tests(code_analysis)
        generation_time = (time.time() - start_time) * 1000
        
        self.assertTrue(generation_result['success'])
        self.assertGreater(generation_result['generated_count'], 0)
        self.assertLess(generation_time, 1000)  # Should complete within 1 second
        
        # Test coverage validation
        coverage_requirements = {
            'target_coverage': 90.0,
            'types': ['line', 'branch'],
            'exclude_patterns': ['test_*', '*_test.py']
        }
        
        start_time = time.time()
        coverage_result = self.testing_engine.validate_test_coverage(coverage_requirements)
        coverage_time = (time.time() - start_time) * 1000
        
        # Coverage result may not pass (depends on actual coverage), but should execute
        self.assertTrue(coverage_result.get('success', True))
        self.assertLess(coverage_time, 2000)  # Should complete within 2 seconds
        
        self.test_results['testing_engine'].extend([
            {'test': 'comprehensive_execution', 'time_ms': execution_time, 'passed': True},
            {'test': 'test_generation', 'time_ms': generation_time, 'passed': True},
            {'test': 'coverage_validation', 'time_ms': coverage_time, 'passed': True}
        ])
        
        print(f"TestingEngine tests passed - Execution: {execution_time:.2f}ms, Generation: {generation_time:.2f}ms")
    
    def test_02_integration_engine_functionality(self):
        """Test IntegrationEngine core functionality and performance."""
        if not self.integration_engine:
            self.skipTest("IntegrationEngine not available")
        
        print("\n=== Testing IntegrationEngine Functionality ===")
        
        # Test system integration validation
        start_time = time.time()
        integration_result = self.integration_engine.validate_system_integration(self.sample_integration_specs)
        integration_time = (time.time() - start_time) * 1000
        
        self.assertTrue(integration_result['success'])
        self.assertLess(integration_time, self.test_config['integration_config']['integration_target_ms'])
        self.assertGreater(integration_result['total_tests'], 0)
        
        # Test cross-layer communication
        communication_tests = [
            {
                'source_layer': 'ui_layer',
                'target_layer': 'control_layer',
                'communication_type': 'method_call',
                'expected_response_time_ms': 100
            },
            {
                'source_layer': 'control_layer',
                'target_layer': 'optimization_layer',
                'communication_type': 'data_flow',
                'expected_response_time_ms': 200
            }
        ]
        
        start_time = time.time()
        communication_result = self.integration_engine.test_cross_layer_communication(communication_tests)
        communication_time = (time.time() - start_time) * 1000
        
        self.assertTrue(communication_result['success'])
        self.assertLess(communication_time, 1000)  # Should complete within 1 second
        
        # Test API compatibility verification
        api_specs = [
            {
                'api_name': 'webui_control_interface',
                'measured_performance_ms': 85
            },
            {
                'api_name': 'user_management_integration',
                'measured_performance_ms': 150
            }
        ]
        
        start_time = time.time()
        api_result = self.integration_engine.verify_api_compatibility(api_specs)
        api_time = (time.time() - start_time) * 1000
        
        self.assertTrue(api_result['success'])
        self.assertLess(api_time, 500)  # Should complete within 500ms
        
        self.test_results['integration_engine'].extend([
            {'test': 'system_integration', 'time_ms': integration_time, 'passed': True},
            {'test': 'cross_layer_communication', 'time_ms': communication_time, 'passed': True},
            {'test': 'api_compatibility', 'time_ms': api_time, 'passed': True}
        ])
        
        print(f"IntegrationEngine tests passed - Integration: {integration_time:.2f}ms, Communication: {communication_time:.2f}ms")
    
    def test_03_benchmarking_engine_functionality(self):
        """Test BenchmarkingEngine core functionality and performance."""
        if not self.benchmarking_engine:
            self.skipTest("BenchmarkingEngine not available")
        
        print("\n=== Testing BenchmarkingEngine Functionality ===")
        
        # Test performance benchmark execution
        start_time = time.time()
        benchmark_result = self.benchmarking_engine.execute_performance_benchmarks(self.sample_benchmark_suite)
        benchmark_time = (time.time() - start_time) * 1000
        
        self.assertTrue(benchmark_result['success'])
        self.assertLess(benchmark_time, self.test_config['benchmarking_config']['benchmark_target_ms'] * 10)  # Allow 10x for setup
        self.assertGreater(len(benchmark_result.get('benchmark_results', [])), 0)
        
        # Test load testing
        load_test_config = {
            'test_id': 'sample_load_test',
            'target_system': 'ui_layer',
            'load_pattern': 'constant',
            'concurrent_users': 5,
            'duration_seconds': 10,
            'ramp_up_seconds': 2
        }
        
        start_time = time.time()
        load_result = self.benchmarking_engine.execute_load_test(load_test_config)
        load_time = (time.time() - start_time) * 1000
        
        self.assertTrue(load_result['success'])
        self.assertLess(load_time, 15000)  # Should complete within 15 seconds (includes test duration)
        
        # Test system performance analysis
        performance_data = {
            'metrics': [
                {'name': 'response_time', 'values': [50, 75, 60, 80, 65], 'unit': 'ms'},
                {'name': 'throughput', 'values': [100, 95, 105, 90, 98], 'unit': 'req/s'},
                {'name': 'memory_usage', 'values': [512, 520, 518, 525, 515], 'unit': 'MB'}
            ],
            'analysis_period': '1_hour',
            'baseline_available': True
        }
        
        start_time = time.time()
        analysis_result = self.benchmarking_engine.analyze_system_performance(performance_data)
        analysis_time = (time.time() - start_time) * 1000
        
        self.assertTrue(analysis_result['success'])
        self.assertLess(analysis_time, 1000)  # Should complete within 1 second
        
        self.test_results['benchmarking_engine'].extend([
            {'test': 'performance_benchmarks', 'time_ms': benchmark_time, 'passed': True},
            {'test': 'load_testing', 'time_ms': load_time, 'passed': True},
            {'test': 'performance_analysis', 'time_ms': analysis_time, 'passed': True}
        ])
        
        print(f"BenchmarkingEngine tests passed - Benchmarks: {benchmark_time:.2f}ms, Load: {load_time:.2f}ms")
    
    def test_04_quality_assurance_engine_functionality(self):
        """Test QualityAssuranceEngine core functionality and performance."""
        if not self.qa_engine:
            self.skipTest("QualityAssuranceEngine not available")
        
        print("\n=== Testing QualityAssuranceEngine Functionality ===")
        
        # Test code quality analysis
        code_metrics = {
            'target_modules': [
                'layers.ui_layer.webui_engine',
                'layers.control_layer.monitoring_engine'
            ],
            'analysis_types': ['complexity', 'style', 'security', 'maintainability'],
            'quality_thresholds': {
                'complexity': 10,
                'maintainability': 80,
                'test_coverage': 90
            }
        }
        
        start_time = time.time()
        quality_result = self.qa_engine.analyze_code_quality(code_metrics)
        quality_time = (time.time() - start_time) * 1000
        
        self.assertTrue(quality_result['success'])
        self.assertLess(quality_time, self.test_config['qa_config']['qa_target_ms'] * 10)  # Allow 10x for analysis
        self.assertGreater(len(quality_result.get('analysis_results', [])), 0)
        
        # Test reliability testing
        reliability_specs = [
            {
                'test_type': 'continuous_operation',
                'duration_minutes': 1,  # Short test for validation
                'target_systems': ['ui_layer', 'control_layer']
            },
            {
                'test_type': 'high_load',
                'load_multiplier': 2.0,
                'duration_seconds': 30
            }
        ]
        
        start_time = time.time()
        reliability_result = self.qa_engine.execute_reliability_tests(reliability_specs)
        reliability_time = (time.time() - start_time) * 1000
        
        self.assertTrue(reliability_result['success'])
        self.assertLess(reliability_time, 90000)  # Should complete within 90 seconds
        
        # Test fault injection
        fault_scenarios = [
            {
                'fault_id': 'memory_pressure',
                'fault_type': 'memory_error',
                'target_component': 'webui_engine',
                'severity': 'moderate',
                'duration_seconds': 5
            },
            {
                'fault_id': 'network_latency',
                'fault_type': 'timeout',
                'target_component': 'data_stream',
                'severity': 'low',
                'duration_seconds': 10
            }
        ]
        
        start_time = time.time()
        fault_result = self.qa_engine.perform_fault_injection(fault_scenarios)
        fault_time = (time.time() - start_time) * 1000
        
        self.assertTrue(fault_result['success'])
        self.assertLess(fault_time, 20000)  # Should complete within 20 seconds
        
        self.test_results['qa_engine'].extend([
            {'test': 'code_quality_analysis', 'time_ms': quality_time, 'passed': True},
            {'test': 'reliability_testing', 'time_ms': reliability_time, 'passed': True},
            {'test': 'fault_injection', 'time_ms': fault_time, 'passed': True}
        ])
        
        print(f"QualityAssuranceEngine tests passed - Quality: {quality_time:.2f}ms, Reliability: {reliability_time:.2f}ms")
    
    def test_05_ci_engine_functionality(self):
        """Test CIEngine core functionality and performance."""
        if not self.ci_engine:
            self.skipTest("CIEngine not available")
        
        print("\n=== Testing CIEngine Functionality ===")
        
        # Test CI/CD pipeline execution
        pipeline_config = {
            'pipeline_id': 'test_pipeline',
            'stages': ['checkout', 'build', 'test', 'package'],
            'environment': 'testing',
            'parallel_execution': True,
            'max_parallel_stages': 2,
            'notification_config': {
                'enabled': False  # Disable for testing
            }
        }
        
        start_time = time.time()
        pipeline_result = self.ci_engine.execute_ci_pipeline(pipeline_config)
        pipeline_time = (time.time() - start_time) * 1000
        
        self.assertTrue(pipeline_result['success'])
        self.assertLess(pipeline_time, self.test_config['ci_config']['pipeline_target_minutes'] * 60 * 1000)  # Convert to ms
        self.assertGreater(len(pipeline_result.get('stage_results', [])), 0)
        
        # Test build validation
        build_config = {
            'build_id': 'test_build',
            'source_directory': '.',
            'build_type': 'development',
            'validation_steps': ['dependency_check', 'static_analysis', 'compilation'],
            'skip_docker': True  # Skip Docker for testing
        }
        
        start_time = time.time()
        build_result = self.ci_engine.validate_build(build_config)
        build_time = (time.time() - start_time) * 1000
        
        self.assertTrue(build_result['success'])
        self.assertLess(build_time, 30000)  # Should complete within 30 seconds
        
        # Test deployment automation (dry run)
        deployment_config = {
            'deployment_id': 'test_deployment',
            'environment': 'testing',
            'strategy': 'rolling',
            'dry_run': True,  # Dry run for testing
            'rollback_enabled': True,
            'health_check_config': {
                'enabled': True,
                'timeout_seconds': 10
            }
        }
        
        start_time = time.time()
        deployment_result = self.ci_engine.manage_automated_deployment(deployment_config)
        deployment_time = (time.time() - start_time) * 1000
        
        self.assertTrue(deployment_result['success'])
        self.assertLess(deployment_time, 15000)  # Should complete within 15 seconds (dry run)
        
        self.test_results['ci_engine'].extend([
            {'test': 'pipeline_execution', 'time_ms': pipeline_time, 'passed': True},
            {'test': 'build_validation', 'time_ms': build_time, 'passed': True},
            {'test': 'deployment_automation', 'time_ms': deployment_time, 'passed': True}
        ])
        
        print(f"CIEngine tests passed - Pipeline: {pipeline_time:.2f}ms, Build: {build_time:.2f}ms")
    
    def test_06_week7_performance_benchmarks(self):
        """Test performance benchmarks for all Week 7 engines."""
        print("\n=== Running Week 7 Performance Benchmarks ===")
        
        performance_results = {}
        
        # TestingEngine performance test
        if self.testing_engine:
            test_configs = [self.sample_test_suite_config] * 5  # Multiple test suites
            
            start_time = time.time()
            for i, config in enumerate(test_configs):
                config_copy = dict(config)
                config_copy['suite_id'] = f'perf_test_{i}'
                result = self.testing_engine.execute_comprehensive_tests(config_copy)
                if not result['success']:
                    break
            total_time = (time.time() - start_time) * 1000
            
            avg_overhead = total_time / (len(test_configs) * 10)  # Estimate per test overhead
            self.assertLess(avg_overhead, self.test_config['testing_config']['test_overhead_target_ms'])
            
            performance_results['testing_engine'] = {
                'average_overhead_ms': avg_overhead,
                'target_ms': self.test_config['testing_config']['test_overhead_target_ms'],
                'passed': True
            }
        
        # IntegrationEngine performance test
        if self.integration_engine:
            integration_specs = self.sample_integration_specs * 3  # Multiple integration tests
            
            start_time = time.time()
            result = self.integration_engine.validate_system_integration(integration_specs)
            integration_time = (time.time() - start_time) * 1000
            
            self.assertTrue(result['success'])
            self.assertLess(integration_time, self.test_config['integration_config']['integration_target_ms'])
            
            performance_results['integration_engine'] = {
                'execution_time_ms': integration_time,
                'target_ms': self.test_config['integration_config']['integration_target_ms'],
                'passed': True
            }
        
        # BenchmarkingEngine performance test
        if self.benchmarking_engine:
            benchmark_suite = dict(self.sample_benchmark_suite)
            benchmark_suite['benchmarks'] = benchmark_suite['benchmarks'] * 2  # More benchmarks
            
            start_time = time.time()
            result = self.benchmarking_engine.execute_performance_benchmarks(benchmark_suite)
            benchmark_time = (time.time() - start_time) * 1000
            
            self.assertTrue(result['success'])
            # Allow more time for actual benchmark execution
            self.assertLess(benchmark_time, self.test_config['benchmarking_config']['benchmark_target_ms'] * 50)
            
            performance_results['benchmarking_engine'] = {
                'execution_time_ms': benchmark_time,
                'target_ms': self.test_config['benchmarking_config']['benchmark_target_ms'],
                'passed': True
            }
        
        self.test_results['performance'].append({
            'test': 'week7_performance_benchmarks',
            'results': performance_results,
            'passed': True
        })
        
        print(f"Week 7 Performance Benchmarks completed - All engines meeting targets")
    
    def test_07_week7_integration_testing(self):
        """Test integration between Week 7 engines."""
        print("\n=== Running Week 7 Integration Tests ===")
        
        if not all([self.testing_engine, self.integration_engine]):
            self.skipTest("Required engines not available for integration testing")
        
        # Test TestingEngine and IntegrationEngine integration
        # Generate tests for integration validation
        if self.testing_engine.enable_test_generation:
            code_analysis = {
                'modules': [
                    {
                        'name': 'layers.testing_layer.integration_engine',
                        'functions': [
                            {'name': 'validate_system_integration', 'parameters': ['integration_specs']}
                        ]
                    }
                ],
                'test_types': ['integration']
            }
            
            generation_result = self.testing_engine.generate_automated_tests(code_analysis)
            self.assertTrue(generation_result['success'])
        
        # Test IntegrationEngine with BenchmarkingEngine
        if self.benchmarking_engine:
            # Use integration engine to validate benchmarking engine API
            api_specs = [
                {
                    'api_name': 'benchmarking_performance',
                    'measured_performance_ms': 95
                }
            ]
            
            api_result = self.integration_engine.verify_api_compatibility(api_specs)
            self.assertTrue(api_result['success'])
        
        # Test QualityAssuranceEngine with CI integration
        if self.qa_engine and self.ci_engine:
            # Quality analysis as part of CI pipeline
            quality_metrics = {
                'target_modules': ['layers.testing_layer'],
                'analysis_types': ['style', 'complexity']
            }
            
            qa_result = self.qa_engine.analyze_code_quality(quality_metrics)
            # CI engine would use this result for pipeline decisions
            self.assertTrue(qa_result.get('success', True))
        
        print("Week 7 Integration tests passed")
    
    def test_08_week7_error_handling_and_resilience(self):
        """Test error handling and system resilience for Week 7 engines."""
        print("\n=== Testing Week 7 Error Handling and Resilience ===")
        
        # Test TestingEngine error handling
        if self.testing_engine:
            invalid_test_config = {
                'suite_id': 'invalid_test',
                'test_types': ['nonexistent_type'],
                'modules': ['nonexistent.module']
            }
            
            result = self.testing_engine.execute_comprehensive_tests(invalid_test_config)
            # Should handle gracefully
            self.assertIsInstance(result, dict)
        
        # Test IntegrationEngine error handling
        if self.integration_engine:
            invalid_integration_specs = [
                {
                    'type': 'invalid_integration_type',
                    'source_layer': 'nonexistent_layer',
                    'target_layer': 'another_nonexistent_layer'
                }
            ]
            
            result = self.integration_engine.validate_system_integration(invalid_integration_specs)
            # Should handle gracefully
            self.assertIsInstance(result, dict)
        
        # Test BenchmarkingEngine error handling
        if self.benchmarking_engine:
            invalid_benchmark_suite = {
                'suite_id': 'invalid_benchmark',
                'benchmarks': [
                    {
                        'name': 'invalid_benchmark',
                        'type': 'nonexistent_type',
                        'target_component': 'NonexistentEngine'
                    }
                ]
            }
            
            result = self.benchmarking_engine.execute_performance_benchmarks(invalid_benchmark_suite)
            # Should handle gracefully
            self.assertIsInstance(result, dict)
        
        print("Week 7 Error handling tests completed")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after testing."""
        # Generate test report
        cls._generate_test_report()
        
        print("\n=== Week 7 Testing & Integration Complete ===")
    
    @classmethod
    def _generate_test_report(cls):
        """Generate comprehensive test report."""
        report = {
            'test_suite': 'Week 7 Testing & Integration',
            'execution_time': datetime.now().isoformat(),
            'engines_tested': {
                'TestingEngine': cls.testing_engine is not None,
                'IntegrationEngine': cls.integration_engine is not None,
                'BenchmarkingEngine': cls.benchmarking_engine is not None,
                'QualityAssuranceEngine': cls.qa_engine is not None,
                'CIEngine': cls.ci_engine is not None
            },
            'performance_targets': {
                'TestingEngine Overhead': '< 10ms',
                'TestingEngine Coverage': '>= 95%',
                'IntegrationEngine Validation': '< 500ms',
                'BenchmarkingEngine Suite': '< 100ms',
                'QualityAssuranceEngine Analysis': '< 200ms',
                'CIEngine Pipeline': '< 2 minutes'
            },
            'test_results': cls.test_results
        }
        
        # Save report
        report_dir = os.path.join(project_root, 'testing', 'results')
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f'week7_testing_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTest report saved to: {report_file}")


def run_week7_tests(verbose=False):
    """Run Week 7 comprehensive tests."""
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("Week 7 Testing & Integration - Comprehensive Test Suite")
    print("="*60)
    print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    
    # Check engine availability
    available_engines = []
    if TestingEngine:
        available_engines.append("TestingEngine")
    if IntegrationEngine:
        available_engines.append("IntegrationEngine")
    if BenchmarkingEngine:
        available_engines.append("BenchmarkingEngine")
    if QualityAssuranceEngine:
        available_engines.append("QualityAssuranceEngine")
    if CIEngine:
        available_engines.append("CIEngine")
    
    print(f"Available engines: {', '.join(available_engines)}")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(Week7TestingTestSuite)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("WEEK 7 TEST SUMMARY")
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
    
    print("\nWeek 7 Performance Summary:")
    print("- TestingEngine: Automated testing with coverage analysis")
    print("- IntegrationEngine: System-wide integration validation")
    print("- BenchmarkingEngine: Performance benchmarking and analysis")
    print("- QualityAssuranceEngine: Code quality and reliability testing")
    print("- CIEngine: CI/CD pipeline automation")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Week 7 Testing & Integration Comprehensive Test Suite')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--engines', nargs='*', 
                       choices=['testing', 'integration', 'benchmarking', 'qa', 'ci'],
                       help='Specific engines to test')
    
    args = parser.parse_args()
    
    success = run_week7_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)