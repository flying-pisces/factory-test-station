#!/usr/bin/env python3
"""
Week 13 Validation Script - UI & Visualization Layer

Comprehensive validation of all Week 13 UI and visualization components
before commit to ensure functionality and integration.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Week13Validator:
    """Comprehensive validator for Week 13 UI & Visualization Layer."""
    
    def __init__(self):
        self.validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'week': 13,
            'layer': 'UI & Visualization Layer',
            'components': {},
            'overall_status': 'unknown',
            'summary': {},
            'errors': []
        }
        
        self.total_tests = 0
        self.passed_tests = 0
        
        logger.info("🎯 Week 13 Validator initialized")
    
    async def validate_all_components(self) -> Dict[str, Any]:
        """Validate all Week 13 UI components."""
        try:
            logger.info("🚀 Starting Week 13 comprehensive validation...")
            
            # Test 1: Visualization Engine
            await self._validate_visualization_engine()
            
            # Test 2: Dashboard Manager
            await self._validate_dashboard_manager()
            
            # Test 3: Real-Time Data Pipeline
            await self._validate_data_pipeline()
            
            # Test 4: UI Controller
            await self._validate_ui_controller()
            
            # Test 5: Operator Dashboard
            await self._validate_operator_dashboard()
            
            # Test 6: Management Dashboard
            await self._validate_management_dashboard()
            
            # Test 7: Mobile Interface
            await self._validate_mobile_interface()
            
            # Test 8: Integration Tests
            await self._validate_ui_integration()
            
            # Calculate overall status
            self._calculate_overall_status()
            
            logger.info("✅ Week 13 validation completed")
            return self.validation_results
            
        except Exception as e:
            logger.error(f"❌ Critical error in validation: {e}")
            self.validation_results['overall_status'] = 'error'
            self.validation_results['critical_error'] = str(e)
            return self.validation_results
    
    async def _validate_visualization_engine(self):
        """Validate visualization engine functionality."""
        logger.info("🎨 Validating Visualization Engine...")
        
        try:
            from layers.ui_layer.visualization_engine import VisualizationEngine
            
            # Initialize engine
            viz_engine = VisualizationEngine()
            
            # Run validation
            results = await viz_engine.validate_visualization_engine()
            
            self.validation_results['components']['visualization_engine'] = results
            self._update_test_counts(results)
            
            logger.info(f"✓ Visualization Engine: {results['test_summary']}")
            
        except Exception as e:
            error_msg = f"Visualization Engine validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.validation_results['errors'].append(error_msg)
            self.validation_results['components']['visualization_engine'] = {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _validate_dashboard_manager(self):
        """Validate dashboard manager functionality."""
        logger.info("📊 Validating Dashboard Manager...")
        
        try:
            from layers.ui_layer.dashboard_manager import DashboardManager
            
            # Initialize manager
            dashboard_manager = DashboardManager()
            
            # Run validation
            results = await dashboard_manager.validate_dashboard_manager()
            
            self.validation_results['components']['dashboard_manager'] = results
            self._update_test_counts(results)
            
            logger.info(f"✓ Dashboard Manager: {results['test_summary']}")
            
        except Exception as e:
            error_msg = f"Dashboard Manager validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.validation_results['errors'].append(error_msg)
            self.validation_results['components']['dashboard_manager'] = {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _validate_data_pipeline(self):
        """Validate real-time data pipeline functionality."""
        logger.info("🔄 Validating Real-Time Data Pipeline...")
        
        try:
            from layers.ui_layer.real_time_data_pipeline import RealTimeDataPipeline
            
            # Initialize pipeline
            data_pipeline = RealTimeDataPipeline()
            
            # Run validation
            results = await data_pipeline.validate_data_pipeline()
            
            self.validation_results['components']['data_pipeline'] = results
            self._update_test_counts(results)
            
            logger.info(f"✓ Data Pipeline: {results['test_summary']}")
            
        except Exception as e:
            error_msg = f"Data Pipeline validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.validation_results['errors'].append(error_msg)
            self.validation_results['components']['data_pipeline'] = {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _validate_ui_controller(self):
        """Validate UI controller functionality."""
        logger.info("🎮 Validating UI Controller...")
        
        try:
            from layers.ui_layer.ui_controller import UIController
            
            # Initialize controller
            ui_controller = UIController()
            
            # Run validation
            results = await ui_controller.validate_ui_controller()
            
            self.validation_results['components']['ui_controller'] = results
            self._update_test_counts(results)
            
            logger.info(f"✓ UI Controller: {results['test_summary']}")
            
        except Exception as e:
            error_msg = f"UI Controller validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.validation_results['errors'].append(error_msg)
            self.validation_results['components']['ui_controller'] = {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _validate_operator_dashboard(self):
        """Validate operator dashboard functionality."""
        logger.info("👷 Validating Operator Dashboard...")
        
        try:
            from layers.ui_layer.operator_dashboard import OperatorDashboard
            
            # Initialize dashboard
            operator_dashboard = OperatorDashboard()
            
            # Run validation
            results = await operator_dashboard.validate_operator_dashboard()
            
            self.validation_results['components']['operator_dashboard'] = results
            self._update_test_counts(results)
            
            logger.info(f"✓ Operator Dashboard: {results['test_summary']}")
            
        except Exception as e:
            error_msg = f"Operator Dashboard validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.validation_results['errors'].append(error_msg)
            self.validation_results['components']['operator_dashboard'] = {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _validate_management_dashboard(self):
        """Validate management dashboard functionality."""
        logger.info("👔 Validating Management Dashboard...")
        
        try:
            from layers.ui_layer.management_dashboard import ManagementDashboard
            
            # Initialize dashboard
            mgmt_dashboard = ManagementDashboard()
            
            # Run validation
            results = await mgmt_dashboard.validate_management_dashboard()
            
            self.validation_results['components']['management_dashboard'] = results
            self._update_test_counts(results)
            
            logger.info(f"✓ Management Dashboard: {results['test_summary']}")
            
        except Exception as e:
            error_msg = f"Management Dashboard validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.validation_results['errors'].append(error_msg)
            self.validation_results['components']['management_dashboard'] = {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _validate_mobile_interface(self):
        """Validate mobile interface functionality."""
        logger.info("📱 Validating Mobile Interface...")
        
        try:
            from layers.ui_layer.mobile_interface import MobileInterface
            
            # Initialize interface
            mobile_interface = MobileInterface()
            
            # Run validation
            results = await mobile_interface.validate_mobile_interface()
            
            self.validation_results['components']['mobile_interface'] = results
            self._update_test_counts(results)
            
            logger.info(f"✓ Mobile Interface: {results['test_summary']}")
            
        except Exception as e:
            error_msg = f"Mobile Interface validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.validation_results['errors'].append(error_msg)
            self.validation_results['components']['mobile_interface'] = {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _validate_ui_integration(self):
        """Validate UI layer integration."""
        logger.info("🔗 Validating UI Integration...")
        
        try:
            # Test 1: Import all UI components
            integration_results = {
                'validation_timestamp': datetime.now().isoformat(),
                'tests': {},
                'overall_status': 'unknown'
            }
            
            # Import test
            try:
                from layers.ui_layer import (
                    VisualizationEngine, DashboardManager, RealTimeDataPipeline,
                    UIController, OperatorDashboard, ManagementDashboard, MobileInterface
                )
                integration_results['tests']['import_all_components'] = {
                    'status': 'pass',
                    'details': 'All UI components imported successfully'
                }
            except Exception as e:
                integration_results['tests']['import_all_components'] = {
                    'status': 'fail',
                    'error': str(e)
                }
            
            # Test 2: Component initialization
            try:
                components = {
                    'viz_engine': VisualizationEngine(),
                    'dashboard_manager': DashboardManager(),
                    'data_pipeline': RealTimeDataPipeline(),
                    'ui_controller': UIController(),
                    'operator_dashboard': OperatorDashboard(),
                    'management_dashboard': ManagementDashboard(),
                    'mobile_interface': MobileInterface()
                }
                
                integration_results['tests']['component_initialization'] = {
                    'status': 'pass',
                    'details': f'Initialized {len(components)} components successfully'
                }
            except Exception as e:
                integration_results['tests']['component_initialization'] = {
                    'status': 'fail',
                    'error': str(e)
                }
            
            # Test 3: Cross-component communication
            try:
                # Test data pipeline to dashboard integration
                test_data = {'test': 'data', 'timestamp': datetime.now().isoformat()}
                
                # Test real-time pipeline data flow
                pipeline_result = await components['data_pipeline'].push_data_to_pipeline('test_source', test_data)
                
                integration_results['tests']['cross_component_communication'] = {
                    'status': 'pass' if pipeline_result['success'] else 'fail',
                    'details': f"Data pipeline test: {pipeline_result}"
                }
            except Exception as e:
                integration_results['tests']['cross_component_communication'] = {
                    'status': 'fail',
                    'error': str(e)
                }
            
            # Test 4: Template file existence
            import os
            template_files = [
                'layers/ui_layer/templates/operator_dashboard.html',
                'layers/ui_layer/templates/management_dashboard.html',
                'layers/ui_layer/templates/mobile_interface.html'
            ]
            
            missing_templates = [f for f in template_files if not os.path.exists(f)]
            
            integration_results['tests']['template_files'] = {
                'status': 'pass' if not missing_templates else 'fail',
                'details': f"All template files present" if not missing_templates 
                          else f"Missing templates: {missing_templates}"
            }
            
            # Calculate overall integration status
            passed_tests = sum(1 for test in integration_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(integration_results['tests'])
            
            integration_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            integration_results['test_summary'] = f"{passed_tests}/{total_tests} integration tests passed"
            
            self.validation_results['components']['ui_integration'] = integration_results
            self._update_test_counts(integration_results)
            
            logger.info(f"✓ UI Integration: {integration_results['test_summary']}")
            
        except Exception as e:
            error_msg = f"UI Integration validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.validation_results['errors'].append(error_msg)
            self.validation_results['components']['ui_integration'] = {
                'overall_status': 'error',
                'error': str(e)
            }
    
    def _update_test_counts(self, component_results: Dict[str, Any]):
        """Update overall test counts from component results."""
        if 'tests' in component_results:
            component_total = len(component_results['tests'])
            component_passed = sum(1 for test in component_results['tests'].values() 
                                 if test.get('status') == 'pass')
            
            self.total_tests += component_total
            self.passed_tests += component_passed
    
    def _calculate_overall_status(self):
        """Calculate overall validation status."""
        if self.validation_results['errors']:
            self.validation_results['overall_status'] = 'error'
        elif self.passed_tests == self.total_tests:
            self.validation_results['overall_status'] = 'pass'
        else:
            self.validation_results['overall_status'] = 'fail'
        
        # Create summary
        self.validation_results['summary'] = {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.total_tests - self.passed_tests,
            'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            'components_validated': len(self.validation_results['components']),
            'errors_count': len(self.validation_results['errors'])
        }
    
    def print_validation_report(self):
        """Print formatted validation report."""
        print("\n" + "="*80)
        print("🏭 WEEK 13 - UI & VISUALIZATION LAYER VALIDATION REPORT")
        print("="*80)
        
        # Overall status
        status_emoji = "✅" if self.validation_results['overall_status'] == 'pass' else "❌"
        print(f"\n{status_emoji} Overall Status: {self.validation_results['overall_status'].upper()}")
        
        # Summary
        summary = self.validation_results['summary']
        print(f"\n📊 Summary:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Components: {summary['components_validated']}")
        
        # Component results
        print(f"\n🔍 Component Results:")
        for component, results in self.validation_results['components'].items():
            status = results.get('overall_status', 'unknown')
            emoji = "✅" if status == 'pass' else "❌" if status == 'error' else "⚠️"
            
            component_name = component.replace('_', ' ').title()
            print(f"   {emoji} {component_name}: {status}")
            
            if 'test_summary' in results:
                print(f"      {results['test_summary']}")
        
        # Errors
        if self.validation_results['errors']:
            print(f"\n🚨 Errors ({len(self.validation_results['errors'])}):")
            for error in self.validation_results['errors']:
                print(f"   • {error}")
        
        # Performance highlights
        print(f"\n🎯 Week 13 Highlights:")
        print(f"   • Comprehensive UI layer with 3 specialized dashboards")
        print(f"   • Real-time data pipeline with WebSocket communication")
        print(f"   • Mobile-optimized interface for field operations")
        print(f"   • Executive management dashboard with strategic insights")
        print(f"   • Cross-platform visualization engine")
        
        print("\n" + "="*80)


async def main():
    """Main validation function."""
    print("🚀 Starting Week 13 UI & Visualization Layer Validation...")
    
    start_time = time.time()
    
    # Create validator
    validator = Week13Validator()
    
    try:
        # Run comprehensive validation
        results = await validator.validate_all_components()
        
        # Print report
        validator.print_validation_report()
        
        # Calculate validation time
        validation_time = time.time() - start_time
        print(f"\n⏱️  Validation completed in {validation_time:.2f} seconds")
        
        # Return appropriate exit code
        if results['overall_status'] == 'pass':
            print("\n🎉 Week 13 validation PASSED! Ready for commit.")
            return 0
        else:
            print("\n💥 Week 13 validation FAILED! Please fix issues before commit.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Critical validation error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run validation
    exit_code = asyncio.run(main())
    sys.exit(exit_code)