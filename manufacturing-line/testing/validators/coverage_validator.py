"""Test Coverage Validation Framework - Validate test coverage and completeness."""

import json
import ast
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple


class CoverageValidator:
    """Validates test coverage across the codebase."""
    
    def __init__(self, project_root: Path = None):
        """Initialize coverage validator."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = project_root
        self.source_dirs = [
            project_root / "layers",
            project_root / "common", 
            project_root / "stations"
        ]
        
        # Coverage requirements by module type
        self.coverage_requirements = {
            'core_modules': {
                'min_coverage_percent': 80,
                'patterns': ['*engine*.py', '*calculator*.py', '*processor*.py']
            },
            'utility_modules': {
                'min_coverage_percent': 60,
                'patterns': ['*utils*.py', '*helper*.py', '*tool*.py']
            },
            'integration_modules': {
                'min_coverage_percent': 70,
                'patterns': ['*integration*.py', '*interface*.py']
            }
        }
        
        # Week-specific coverage requirements
        self.week_coverage_requirements = {
            1: {
                'required_modules': [
                    'layers/super_admin_layer',
                    'common/database',
                    'common/data_socket'
                ],
                'min_overall_coverage': 60
            },
            2: {
                'required_modules': [
                    'layers/component_layer',
                    'layers/station_layer',
                    'layers/component_layer/vendor_interfaces',
                    'layers/component_layer/component_types'
                ],
                'min_overall_coverage': 75
            }
        }
    
    def validate_test_coverage(self, test_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate test coverage across the codebase."""
        validation_result = {
            'validation_status': 'passed',
            'timestamp': datetime.now().isoformat(),
            'coverage_analysis': {},
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'coverage_summary': {}
        }
        
        try:
            # Analyze source code structure
            source_analysis = self._analyze_source_structure()
            validation_result['coverage_analysis']['source_structure'] = source_analysis
            
            # Analyze test structure
            test_analysis = self._analyze_test_structure()
            validation_result['coverage_analysis']['test_structure'] = test_analysis
            
            # Calculate coverage mapping
            coverage_mapping = self._calculate_coverage_mapping(source_analysis, test_analysis)
            validation_result['coverage_analysis']['coverage_mapping'] = coverage_mapping
            
            # Validate coverage requirements
            requirements_check = self._validate_coverage_requirements(coverage_mapping)
            validation_result['coverage_analysis']['requirements_check'] = requirements_check
            
            # Week-specific coverage validation (if test results provided)
            if test_results and test_results.get('run_metadata', {}).get('week'):
                week = test_results['run_metadata']['week']
                week_check = self._validate_week_coverage(week, coverage_mapping)
                validation_result['coverage_analysis']['week_coverage'] = week_check
            
            # Generate coverage summary
            summary = self._generate_coverage_summary(coverage_mapping, source_analysis, test_analysis)
            validation_result['coverage_summary'] = summary
            
            # Collect issues and recommendations
            self._collect_coverage_issues(validation_result)
            
            # Determine overall status
            if validation_result['issues']:
                has_critical = any(i.get('severity') == 'critical' for i in validation_result['issues'])
                has_high = any(i.get('severity') == 'high' for i in validation_result['issues'])
                
                if has_critical:
                    validation_result['validation_status'] = 'failed'
                elif has_high:
                    validation_result['validation_status'] = 'warning'
            
        except Exception as e:
            validation_result['validation_status'] = 'error'
            validation_result['issues'].append({
                'type': 'coverage_validation_error',
                'severity': 'critical',
                'message': f'Coverage validation failed: {str(e)}',
                'recommendation': 'Check validator implementation and source code access'
            })
        
        return validation_result
    
    def _analyze_source_structure(self) -> Dict[str, Any]:
        """Analyze source code structure and identify modules."""
        structure = {
            'total_modules': 0,
            'modules_by_category': {},
            'modules_by_directory': {},
            'module_details': {}
        }
        
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                continue
            
            dir_name = source_dir.name
            structure['modules_by_directory'][dir_name] = []
            
            # Find all Python modules
            for py_file in source_dir.rglob('*.py'):
                if py_file.name.startswith('__'):
                    continue  # Skip __init__.py, __pycache__, etc.
                
                relative_path = py_file.relative_to(self.project_root)
                module_info = self._analyze_module(py_file)
                
                structure['total_modules'] += 1
                structure['modules_by_directory'][dir_name].append(str(relative_path))
                structure['module_details'][str(relative_path)] = module_info
                
                # Categorize module
                category = self._categorize_module(py_file)
                if category not in structure['modules_by_category']:
                    structure['modules_by_category'][category] = []
                structure['modules_by_category'][category].append(str(relative_path))
        
        return structure
    
    def _analyze_test_structure(self) -> Dict[str, Any]:
        """Analyze test structure and identify test coverage."""
        test_dir = self.project_root / "tests"
        structure = {
            'total_test_files': 0,
            'test_files_by_type': {},
            'test_details': {},
            'test_to_source_mapping': {}
        }
        
        if not test_dir.exists():
            return structure
        
        # Find all test files
        for test_file in test_dir.rglob('test_*.py'):
            relative_path = test_file.relative_to(self.project_root)
            test_info = self._analyze_test_file(test_file)
            
            structure['total_test_files'] += 1
            structure['test_details'][str(relative_path)] = test_info
            
            # Categorize test
            test_type = self._categorize_test_file(test_file)
            if test_type not in structure['test_files_by_type']:
                structure['test_files_by_type'][test_type] = []
            structure['test_files_by_type'][test_type].append(str(relative_path))
            
            # Map to source modules
            source_modules = self._map_test_to_source(test_file)
            structure['test_to_source_mapping'][str(relative_path)] = source_modules
        
        return structure
    
    def _analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a single Python module."""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Count classes and functions
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line_number': node.lineno,
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.FunctionDef) and not any(node in cls.body for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)):
                    functions.append({
                        'name': node.name,
                        'line_number': node.lineno
                    })
            
            return {
                'file_size_bytes': module_path.stat().st_size,
                'line_count': len(content.splitlines()),
                'classes': classes,
                'functions': functions,
                'class_count': len(classes),
                'function_count': len(functions),
                'complexity_estimate': len(classes) * 3 + len(functions)  # Simple complexity metric
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'file_size_bytes': module_path.stat().st_size if module_path.exists() else 0,
                'line_count': 0,
                'classes': [],
                'functions': [],
                'class_count': 0,
                'function_count': 0,
                'complexity_estimate': 0
            }
    
    def _analyze_test_file(self, test_path: Path) -> Dict[str, Any]:
        """Analyze a single test file."""
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find test classes and methods
            test_classes = []
            test_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and 'test' in node.name.lower():
                    test_methods = [m.name for m in node.body 
                                  if isinstance(m, ast.FunctionDef) and m.name.startswith('test_')]
                    test_classes.append({
                        'name': node.name,
                        'test_methods': test_methods,
                        'method_count': len(test_methods)
                    })
                elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_functions.append(node.name)
            
            total_test_methods = sum(cls['method_count'] for cls in test_classes) + len(test_functions)
            
            return {
                'file_size_bytes': test_path.stat().st_size,
                'line_count': len(content.splitlines()),
                'test_classes': test_classes,
                'test_functions': test_functions,
                'total_test_methods': total_test_methods
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'file_size_bytes': test_path.stat().st_size if test_path.exists() else 0,
                'line_count': 0,
                'test_classes': [],
                'test_functions': [],
                'total_test_methods': 0
            }
    
    def _categorize_module(self, module_path: Path) -> str:
        """Categorize a module based on its path and name."""
        name = module_path.name.lower()
        path_str = str(module_path).lower()
        
        if 'engine' in name or 'processor' in name or 'calculator' in name:
            return 'core_modules'
        elif 'util' in name or 'helper' in name or 'tool' in name:
            return 'utility_modules'
        elif 'interface' in name or 'integration' in name:
            return 'integration_modules'
        elif 'config' in name or 'setting' in name:
            return 'configuration_modules'
        else:
            return 'other_modules'
    
    def _categorize_test_file(self, test_path: Path) -> str:
        """Categorize a test file based on its path."""
        path_parts = test_path.parts
        
        if 'unit' in path_parts:
            return 'unit_tests'
        elif 'integration' in path_parts:
            return 'integration_tests'
        elif 'performance' in path_parts:
            return 'performance_tests'
        elif 'end_to_end' in path_parts or 'e2e' in path_parts:
            return 'e2e_tests'
        else:
            return 'other_tests'
    
    def _map_test_to_source(self, test_path: Path) -> List[str]:
        """Map test file to source modules it might be testing."""
        test_name = test_path.stem  # e.g., 'test_component_layer_engine'
        
        # Extract potential module names from test name
        if test_name.startswith('test_'):
            base_name = test_name[5:]  # Remove 'test_' prefix
            
            # Look for matching source modules
            potential_matches = []
            
            for source_dir in self.source_dirs:
                if not source_dir.exists():
                    continue
                
                # Direct name match
                direct_match = source_dir / f"{base_name}.py"
                if direct_match.exists():
                    potential_matches.append(str(direct_match.relative_to(self.project_root)))
                
                # Pattern matching within subdirectories
                for py_file in source_dir.rglob('*.py'):
                    if base_name in py_file.stem.lower():
                        potential_matches.append(str(py_file.relative_to(self.project_root)))
            
            return potential_matches
        
        return []
    
    def _calculate_coverage_mapping(self, source_analysis: Dict[str, Any], 
                                  test_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate coverage mapping between tests and source code."""
        mapping = {
            'covered_modules': {},
            'uncovered_modules': [],
            'test_coverage_stats': {},
            'overall_coverage_percent': 0
        }
        
        all_modules = set(source_analysis['module_details'].keys())
        tested_modules = set()
        
        # Build coverage mapping
        for test_file, source_modules in test_analysis['test_to_source_mapping'].items():
            for source_module in source_modules:
                if source_module in all_modules:
                    tested_modules.add(source_module)
                    
                    if source_module not in mapping['covered_modules']:
                        mapping['covered_modules'][source_module] = []
                    mapping['covered_modules'][source_module].append(test_file)
        
        # Find uncovered modules
        mapping['uncovered_modules'] = list(all_modules - tested_modules)
        
        # Calculate coverage statistics
        total_modules = len(all_modules)
        covered_modules = len(tested_modules)
        
        if total_modules > 0:
            mapping['overall_coverage_percent'] = (covered_modules / total_modules) * 100
        
        mapping['test_coverage_stats'] = {
            'total_modules': total_modules,
            'covered_modules': covered_modules,
            'uncovered_modules': len(mapping['uncovered_modules']),
            'coverage_ratio': covered_modules / total_modules if total_modules > 0 else 0
        }
        
        return mapping
    
    def _validate_coverage_requirements(self, coverage_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Validate coverage against requirements."""
        check_result = {
            'overall_status': 'passed',
            'requirement_checks': {},
            'failed_requirements': [],
            'coverage_gaps': []
        }
        
        overall_coverage = coverage_mapping['overall_coverage_percent']
        
        # Check category-specific requirements
        for category, requirements in self.coverage_requirements.items():
            min_coverage = requirements['min_coverage_percent']
            
            # For now, use overall coverage (could be enhanced to calculate per-category)
            category_coverage = overall_coverage  
            
            requirement_met = category_coverage >= min_coverage
            
            check_result['requirement_checks'][category] = {
                'required_coverage': min_coverage,
                'actual_coverage': category_coverage,
                'requirement_met': requirement_met
            }
            
            if not requirement_met:
                check_result['failed_requirements'].append(category)
                check_result['overall_status'] = 'failed'
        
        # Identify significant coverage gaps
        uncovered_modules = coverage_mapping['uncovered_modules']
        if len(uncovered_modules) > 0:
            for module in uncovered_modules:
                # Prioritize core modules
                if any(pattern in module for pattern in ['engine', 'calculator', 'processor']):
                    check_result['coverage_gaps'].append({
                        'module': module,
                        'priority': 'high',
                        'reason': 'Core module without test coverage'
                    })
                else:
                    check_result['coverage_gaps'].append({
                        'module': module,
                        'priority': 'medium',
                        'reason': 'Module without test coverage'
                    })
        
        return check_result
    
    def _validate_week_coverage(self, week: int, coverage_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Validate coverage for specific week requirements."""
        if week not in self.week_coverage_requirements:
            return {'status': 'no_requirements', 'week': week}
        
        requirements = self.week_coverage_requirements[week]
        check_result = {
            'week': week,
            'status': 'passed',
            'required_modules_coverage': {},
            'missing_coverage': [],
            'overall_coverage_met': False
        }
        
        # Check required modules
        required_modules = requirements.get('required_modules', [])
        covered_modules = coverage_mapping['covered_modules']
        
        for required_module in required_modules:
            # Check if any module under this path has coverage
            has_coverage = any(required_module in module_path for module_path in covered_modules.keys())
            
            check_result['required_modules_coverage'][required_module] = has_coverage
            
            if not has_coverage:
                check_result['missing_coverage'].append(required_module)
                check_result['status'] = 'failed'
        
        # Check overall coverage requirement
        min_coverage = requirements.get('min_overall_coverage', 0)
        actual_coverage = coverage_mapping['overall_coverage_percent']
        
        check_result['overall_coverage_met'] = actual_coverage >= min_coverage
        check_result['required_overall_coverage'] = min_coverage
        check_result['actual_overall_coverage'] = actual_coverage
        
        if not check_result['overall_coverage_met']:
            check_result['status'] = 'failed'
        
        return check_result
    
    def _generate_coverage_summary(self, coverage_mapping: Dict[str, Any],
                                 source_analysis: Dict[str, Any],
                                 test_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive coverage summary."""
        return {
            'overall_coverage_percent': coverage_mapping['overall_coverage_percent'],
            'total_source_modules': source_analysis['total_modules'],
            'total_test_files': test_analysis['total_test_files'],
            'covered_modules': len(coverage_mapping['covered_modules']),
            'uncovered_modules': len(coverage_mapping['uncovered_modules']),
            'coverage_ratio': coverage_mapping['test_coverage_stats']['coverage_ratio'],
            'modules_by_category': source_analysis['modules_by_category'],
            'tests_by_type': test_analysis['test_files_by_type'],
            'total_test_methods': sum(
                details['total_test_methods'] 
                for details in test_analysis['test_details'].values()
            )
        }
    
    def _collect_coverage_issues(self, validation_result: Dict[str, Any]) -> None:
        """Collect coverage issues and recommendations from analysis."""
        coverage_mapping = validation_result['coverage_analysis']['coverage_mapping']
        requirements_check = validation_result['coverage_analysis']['requirements_check']
        
        # Overall coverage issues
        overall_coverage = coverage_mapping['overall_coverage_percent']
        if overall_coverage < 50:
            validation_result['issues'].append({
                'type': 'low_overall_coverage',
                'severity': 'high',
                'message': f'Overall test coverage is low: {overall_coverage:.1f}%',
                'recommendation': 'Implement comprehensive test suites to improve coverage'
            })
        elif overall_coverage < 75:
            validation_result['warnings'].append({
                'type': 'moderate_coverage',
                'severity': 'medium',
                'message': f'Test coverage could be improved: {overall_coverage:.1f}%',
                'recommendation': 'Add tests for uncovered modules and edge cases'
            })
        
        # Requirements failures
        for req_type in requirements_check.get('failed_requirements', []):
            validation_result['issues'].append({
                'type': 'coverage_requirement_failed',
                'severity': 'high',
                'category': req_type,
                'message': f'Coverage requirement failed for {req_type}',
                'recommendation': f'Implement tests to meet {req_type} coverage requirements'
            })
        
        # Coverage gaps
        for gap in requirements_check.get('coverage_gaps', []):
            severity = 'high' if gap['priority'] == 'high' else 'medium'
            validation_result['warnings'].append({
                'type': 'coverage_gap',
                'severity': severity,
                'module': gap['module'],
                'message': f'No test coverage for {gap["module"]}',
                'recommendation': f'Implement tests for {gap["module"]}'
            })
        
        # Week-specific issues
        week_coverage = validation_result['coverage_analysis'].get('week_coverage')
        if week_coverage and week_coverage.get('status') == 'failed':
            week = week_coverage['week']
            validation_result['issues'].append({
                'type': 'week_coverage_failed',
                'severity': 'high',
                'week': week,
                'message': f'Week {week} coverage requirements not met',
                'recommendation': f'Implement required tests for Week {week} modules'
            })


def main():
    """Main entry point for standalone coverage validation."""
    import sys
    
    validator = CoverageValidator()
    
    # If test results file provided, include week-specific validation
    test_results = None
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    test_results = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load test results: {e}")
    
    try:
        validation_result = validator.validate_test_coverage(test_results)
        
        print("Test Coverage Validation:")
        print(f"Status: {validation_result['validation_status'].upper()}")
        
        summary = validation_result.get('coverage_summary', {})
        if summary:
            print(f"Overall Coverage: {summary['overall_coverage_percent']:.1f}%")
            print(f"Modules: {summary['covered_modules']}/{summary['total_source_modules']} covered")
            print(f"Test Files: {summary['total_test_files']}")
        
        if validation_result.get('issues'):
            print(f"Issues: {len(validation_result['issues'])}")
            for issue in validation_result['issues']:
                severity_icon = {'critical': '🔥', 'high': '❌', 'medium': '⚠️', 'low': 'ℹ️'}.get(issue.get('severity'), '❓')
                print(f"  {severity_icon} {issue['message']}")
        
        if validation_result.get('warnings'):
            print(f"Warnings: {len(validation_result['warnings'])}")
            for warning in validation_result['warnings'][:5]:  # Show first 5
                print(f"  ⚠️ {warning['message']}")
        
        # Exit with status based on validation result
        if validation_result['validation_status'] == 'failed':
            sys.exit(1)
        elif validation_result['validation_status'] == 'error':
            sys.exit(2)
        else:
            sys.exit(0)
        
    except Exception as e:
        print(f"Error during coverage validation: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()