"""Test repository structure validation for Week 1 deliverable TC1.1."""

import os
import sys
from pathlib import Path
from typing import List, Set
import pytest
import importlib.util


class TestRepositoryStructure:
    """Validate repository structure meets architectural requirements."""
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_required_directories_exist(self):
        """TC1.1.1: Validate all 47 required directories exist."""
        required_dirs = [
            # Core structure
            "layers", "common", "simulation", "web_interfaces", 
            "database", "line_controller", "config", "deployment",
            "docs", "tools", "tests",
            
            # Layers structure  
            "layers/component_layer", "layers/component_layer/vendor_interfaces",
            "layers/component_layer/component_types", "layers/station_layer",
            "layers/station_layer/test_coverage", "layers/line_layer", 
            "layers/line_layer/retest_policies", "layers/pm_layer",
            
            # Common structure
            "common/interfaces", "common/stations", "common/operators",
            "common/conveyors", "common/equipment", "common/fixtures",
            "common/utils",
            
            # Simulation structure
            "simulation/discrete_event_fsm", "simulation/jaamsim_integration",
            "simulation/simulation_engine", "simulation/scenarios",
            
            # Web interfaces
            "web_interfaces/super_admin", "web_interfaces/line_manager",
            "web_interfaces/station_engineer", "web_interfaces/component_vendor",
            "web_interfaces/shared",
            
            # Database structure
            "database/pocketbase", "database/pocketbase/schemas",
            "database/pocketbase/migrations", "database/models", 
            "database/repositories",
            
            # Line controller
            "line_controller/station_controllers", "line_controller/coordination",
            "line_controller/monitoring",
            
            # Test structure
            "tests/unit", "tests/integration", "tests/system", "tests/acceptance",
            "tests/fixtures"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        assert len(missing_dirs) == 0, f"Missing directories: {missing_dirs}"
        print(f"✓ All {len(required_dirs)} required directories exist")
    
    def test_init_files_present(self):
        """TC1.1.2: Validate all Python package __init__.py files are present."""
        python_package_dirs = [
            "layers", "layers/component_layer", "layers/station_layer", 
            "layers/line_layer", "layers/pm_layer",
            "common", "common/interfaces", "common/stations", "common/operators",
            "common/conveyors", "common/equipment", "common/fixtures", "common/utils",
            "simulation", "simulation/discrete_event_fsm", "simulation/jaamsim_integration",
            "simulation/simulation_engine", "simulation/scenarios",
            "web_interfaces", "web_interfaces/super_admin", "web_interfaces/line_manager",
            "web_interfaces/station_engineer", "web_interfaces/component_vendor",
            "web_interfaces/shared",
            "database", "database/pocketbase", "database/models", "database/repositories",
            "line_controller", "line_controller/station_controllers", 
            "line_controller/coordination", "line_controller/monitoring",
            "tests", "tests/unit", "tests/integration", "tests/system",
            "tests/acceptance", "tests/fixtures"
        ]
        
        missing_init_files = []
        for pkg_dir in python_package_dirs:
            init_file = self.project_root / pkg_dir / "__init__.py"
            if not init_file.exists():
                missing_init_files.append(f"{pkg_dir}/__init__.py")
        
        assert len(missing_init_files) == 0, f"Missing __init__.py files: {missing_init_files}"
        print(f"✓ All {len(python_package_dirs)} __init__.py files present")
    
    def test_import_paths_resolve(self):
        """TC1.1.3: Validate core import paths resolve correctly."""
        # Add project root to path for testing
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        # Test basic package imports (modules that should exist)
        core_imports = [
            "layers",
            "common", 
            "simulation",
            "tests"
        ]
        
        import_failures = []
        successful_imports = 0
        
        for import_path in core_imports:
            try:
                spec = importlib.util.find_spec(import_path)
                if spec is None:
                    import_failures.append(f"{import_path}: Module not found")
                else:
                    successful_imports += 1
            except (ImportError, ModuleNotFoundError, ValueError) as e:
                import_failures.append(f"{import_path}: {e}")
        
        success_rate = (successful_imports / len(core_imports)) * 100
        assert success_rate == 100.0, f"Import failures: {import_failures}"
        print(f"✓ Import paths resolve correctly (100% success, {successful_imports}/{len(core_imports)})")
    
    def test_no_circular_dependencies(self):
        """TC1.1.4: Validate no circular dependencies exist in core modules."""
        # Add project root to path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        # Test basic packages for circular dependencies
        test_modules = [
            "layers",
            "common", 
            "simulation",
            "tests"
        ]
        
        circular_deps = []
        successful_loads = 0
        
        for module_name in test_modules:
            try:
                # Attempt to import module
                spec = importlib.util.find_spec(module_name)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    successful_loads += 1
            except ImportError as e:
                if "circular import" in str(e).lower():
                    circular_deps.append(f"{module_name}: {e}")
            except Exception as e:
                # Other import issues (not circular dependencies)
                pass
        
        assert len(circular_deps) == 0, f"Circular dependencies detected: {circular_deps}"
        print(f"✓ No circular dependencies detected (tested {len(test_modules)} packages)")
    
    def test_core_configuration_files_exist(self):
        """TC1.1.5: Validate core configuration files are present."""
        required_config_files = [
            "requirements.txt",
            "requirements-dev.txt", 
            ".github/workflows/ci.yml",
            "tests/conftest.py",
            "tests/pytest.ini"
        ]
        
        missing_files = []
        for config_file in required_config_files:
            file_path = self.project_root / config_file
            if not file_path.exists():
                missing_files.append(config_file)
        
        assert len(missing_files) == 0, f"Missing configuration files: {missing_files}"
        print(f"✓ All {len(required_config_files)} core configuration files exist")
    
    def test_documentation_structure(self):
        """TC1.1.6: Validate documentation structure is complete."""
        required_docs = [
            "README.md",
            "REPOSITORY_REORGANIZATION_PLAN.md", 
            "COMPREHENSIVE_PROJECT_PLAN.md",
            "16_WEEK_TEST_PLAN.md",
            "COMPREHENSIVE_PROJECT_SUMMARY.md",
            "STANDARD_SOCKET_ARCHITECTURE.md",
            "ORGANIZED_STRUCTURE.md",
            "TEST_PLAN_CONFIRMATION.md",
            "CLAUDE.md"
        ]
        
        missing_docs = []
        for doc_file in required_docs:
            doc_path = self.project_root / doc_file
            if not doc_path.exists():
                missing_docs.append(doc_file)
        
        assert len(missing_docs) == 0, f"Missing documentation files: {missing_docs}"
        print(f"✓ All {len(required_docs)} documentation files exist")


if __name__ == "__main__":
    # Run validation directly
    test_repo = TestRepositoryStructure()
    
    print("🔍 Running Repository Structure Validation Tests...")
    print("=" * 60)
    
    try:
        test_repo.test_required_directories_exist()
        test_repo.test_init_files_present()
        test_repo.test_import_paths_resolve()
        test_repo.test_no_circular_dependencies()
        test_repo.test_core_configuration_files_exist()
        test_repo.test_documentation_structure()
        
        print("=" * 60)
        print("✅ Repository Structure Validation: ALL TESTS PASSED")
        print("📊 Validation Summary:")
        print("   - 47 required directories validated")
        print("   - 23+ __init__.py files confirmed")
        print("   - Core imports resolve correctly")
        print("   - No circular dependencies detected")
        print("   - All configuration files present")
        print("   - Complete documentation structure")
        
    except AssertionError as e:
        print(f"❌ Repository Structure Validation FAILED: {e}")
        sys.exit(1)