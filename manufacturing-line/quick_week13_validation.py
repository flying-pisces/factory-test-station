#!/usr/bin/env python3
"""
Quick Week 13 Validation - UI & Visualization Layer

Fast validation of Week 13 components for commit readiness check.
"""

import asyncio
import sys
import time
from datetime import datetime

def test_imports():
    """Test that all Week 13 components can be imported."""
    print("🔍 Testing imports...")
    try:
        from layers.ui_layer import (
            VisualizationEngine, DashboardManager, RealTimeDataPipeline,
            UIController, OperatorDashboard, ManagementDashboard, MobileInterface
        )
        print("✅ All UI components imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_component_initialization():
    """Test that components can be initialized."""
    print("🚀 Testing component initialization...")
    try:
        from layers.ui_layer import (
            VisualizationEngine, DashboardManager, RealTimeDataPipeline,
            UIController, OperatorDashboard, ManagementDashboard, MobileInterface
        )
        
        # Initialize core components
        viz_engine = VisualizationEngine()
        dashboard_manager = DashboardManager() 
        data_pipeline = RealTimeDataPipeline()
        ui_controller = UIController()
        
        # Initialize dashboard components (without starting servers)
        operator_dash = OperatorDashboard({'debug': False})
        mgmt_dash = ManagementDashboard({'debug': False})
        mobile_interface = MobileInterface({'debug': False})
        
        print("✅ All components initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_template_files():
    """Test that HTML template files exist."""
    print("📄 Testing template files...")
    import os
    
    template_files = [
        'layers/ui_layer/templates/operator_dashboard.html',
        'layers/ui_layer/templates/management_dashboard.html', 
        'layers/ui_layer/templates/mobile_interface.html'
    ]
    
    missing_files = []
    for file_path in template_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing template files: {missing_files}")
        return False
    else:
        print("✅ All template files present")
        return True

async def test_basic_functionality():
    """Test basic functionality of core components."""
    print("⚙️  Testing basic functionality...")
    try:
        from layers.ui_layer import RealTimeDataPipeline
        
        # Test data pipeline basic operations
        pipeline = RealTimeDataPipeline()
        
        # Test data push (should not fail)
        test_data = {'test': 'value', 'timestamp': datetime.now().isoformat()}
        result = await pipeline.push_data_to_pipeline('production_system', test_data)
        
        if result['success']:
            print("✅ Basic functionality test passed")
            return True
        else:
            print(f"❌ Basic functionality test failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_week13_structure():
    """Test Week 13 specific structure and components."""
    print("🏗️  Testing Week 13 structure...")
    try:
        from layers.ui_layer.dashboard_manager import DashboardType, DashboardRole
        from layers.ui_layer.visualization_engine import ChartType
        
        # Test enum values
        assert DashboardType.OPERATOR.value == "operator"
        assert DashboardType.MANAGER.value == "manager"
        assert DashboardRole.OPERATOR == "operator"
        assert ChartType.LINE.value == "line"
        
        print("✅ Week 13 structure validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Week 13 structure validation failed: {e}")
        return False

async def main():
    """Run quick validation."""
    print("🏭 WEEK 13 - QUICK UI LAYER VALIDATION")
    print("="*50)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 5
    
    # Run validation tests
    test_results = [
        test_imports(),
        test_component_initialization(),
        test_template_files(),
        await test_basic_functionality(),
        test_week13_structure()
    ]
    
    tests_passed = sum(test_results)
    validation_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%") 
    print(f"Validation Time: {validation_time:.2f} seconds")
    
    if tests_passed == total_tests:
        print("\n🎉 Week 13 quick validation PASSED! Components are ready.")
        return 0
    else:
        print("\n💥 Week 13 quick validation FAILED! Please fix issues.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)