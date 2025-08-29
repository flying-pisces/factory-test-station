"""Test Result Validation Framework - Validate test results and outcomes."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set


class ResultValidator:
    """Validates test results for completeness, accuracy, and compliance."""
    
    def __init__(self, project_root: Path = None):
        """Initialize result validator."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = project_root
        
        # Validation rules and requirements
        self.required_fields = {
            'run_metadata': ['run_id', 'start_time', 'git_info'],
            'test_results': [],  # Dynamic based on content
            'summary': ['overall_success', 'total_tests', 'execution_time_seconds']
        }
        
        # Week-specific test requirements
        self.week_test_requirements = {
            1: {
                'required_test_suites': [
                    'Super Admin Layer Core Tests',
                    'Database Integration Tests',
                    'Standard Data Socket Tests'
                ],
                'minimum_tests': 5,
                'required_objectives': [
                    'super_admin_layer',
                    'data_socket_architecture', 
                    'database_integration'
                ]
            },
            2: {
                'required_test_suites': [
                    'Component Layer Engine Tests',
                    'Station Layer Engine Tests'
                ],
                'minimum_tests': 10,
                'required_objectives': [
                    'enhanced_component_layer',
                    'station_layer_optimization',
                    'component_type_processors'
                ],
                'performance_requirements': {
                    'target_ms': 100.0,
                    'must_meet_target': True
                }
            }
        }
    
    def validate_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate comprehensive test results."""
        validation_result = {
            'validation_status': 'passed',
            'run_id': test_results.get('run_metadata', {}).get('run_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'validation_checks': {},
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'compliance_status': {}
        }
        
        try:
            # Structure validation
            structure_check = self._validate_structure(test_results)
            validation_result['validation_checks']['structure'] = structure_check
            
            # Content validation
            content_check = self._validate_content(test_results)
            validation_result['validation_checks']['content'] = content_check
            
            # Git tracking validation
            git_check = self._validate_git_tracking(test_results)
            validation_result['validation_checks']['git_tracking'] = git_check
            
            # Week-specific validation (if applicable)
            week_check = self._validate_week_requirements(test_results)
            if week_check:
                validation_result['validation_checks']['week_requirements'] = week_check
            
            # Test quality validation
            quality_check = self._validate_test_quality(test_results)
            validation_result['validation_checks']['test_quality'] = quality_check
            
            # Aggregate issues and warnings from all checks
            for check_name, check_result in validation_result['validation_checks'].items():
                if check_result.get('issues'):
                    validation_result['issues'].extend(check_result['issues'])
                if check_result.get('warnings'):
                    validation_result['warnings'].extend(check_result['warnings'])
                if check_result.get('recommendations'):
                    validation_result['recommendations'].extend(check_result['recommendations'])
            
            # Determine overall validation status
            if any(check.get('status') == 'failed' for check in validation_result['validation_checks'].values()):
                validation_result['validation_status'] = 'failed'
            elif validation_result['warnings']:
                validation_result['validation_status'] = 'warning'
            else:
                validation_result['validation_status'] = 'passed'
            
            # Compliance assessment
            validation_result['compliance_status'] = self._assess_compliance(validation_result)
            
        except Exception as e:
            validation_result['validation_status'] = 'error'
            validation_result['issues'].append({
                'type': 'validation_error',
                'severity': 'critical',
                'message': f'Validation failed with error: {str(e)}',
                'recommendation': 'Check validator implementation and input data format'
            })
        
        return validation_result
    
    def _validate_structure(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the structure of test results."""
        check_result = {
            'status': 'passed',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check required top-level fields
        for field, subfields in self.required_fields.items():
            if field not in test_results:
                check_result['issues'].append({
                    'type': 'missing_field',
                    'severity': 'high',
                    'field': field,
                    'message': f'Required field "{field}" is missing',
                    'recommendation': f'Ensure test runner includes "{field}" in output'
                })
                check_result['status'] = 'failed'
                continue
            
            # Check required subfields
            for subfield in subfields:
                if subfield not in test_results[field]:
                    check_result['issues'].append({
                        'type': 'missing_subfield',
                        'severity': 'medium',
                        'field': f'{field}.{subfield}',
                        'message': f'Required subfield "{field}.{subfield}" is missing',
                        'recommendation': f'Ensure test runner includes "{subfield}" in "{field}"'
                    })
        
        # Check test results structure
        test_results_data = test_results.get('test_results', {})
        if isinstance(test_results_data, dict):
            for suite_name, suite_result in test_results_data.items():
                required_suite_fields = ['success', 'tests_run', 'passed', 'failed', 'errors']
                for field in required_suite_fields:
                    if field not in suite_result:
                        check_result['warnings'].append({
                            'type': 'missing_suite_field',
                            'severity': 'low',
                            'suite': suite_name,
                            'field': field,
                            'message': f'Suite "{suite_name}" missing field "{field}"',
                            'recommendation': 'Ensure test suite results include all standard fields'
                        })
        
        return check_result
    
    def _validate_content(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the content and values of test results."""
        check_result = {
            'status': 'passed',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        summary = test_results.get('summary', {})
        test_results_data = test_results.get('test_results', {})
        
        # Check numeric consistency
        summary_total = summary.get('total_tests', 0)
        summary_passed = summary.get('total_passed', 0)
        summary_failed = summary.get('total_failed', 0)
        summary_errors = summary.get('total_errors', 0)
        
        # Calculate actual totals from test results
        actual_total = sum(suite.get('tests_run', 0) for suite in test_results_data.values())
        actual_passed = sum(suite.get('passed', 0) for suite in test_results_data.values())
        actual_failed = sum(suite.get('failed', 0) for suite in test_results_data.values())
        actual_errors = sum(suite.get('errors', 0) for suite in test_results_data.values())
        
        # Check for inconsistencies
        if summary_total != actual_total:
            check_result['issues'].append({
                'type': 'total_mismatch',
                'severity': 'high',
                'message': f'Summary total ({summary_total}) does not match actual total ({actual_total})',
                'recommendation': 'Check test result aggregation logic'
            })
            check_result['status'] = 'failed'
        
        if summary_passed != actual_passed:
            check_result['warnings'].append({
                'type': 'passed_mismatch',
                'severity': 'medium',
                'message': f'Summary passed ({summary_passed}) does not match actual passed ({actual_passed})',
                'recommendation': 'Verify test result counting accuracy'
            })
        
        # Check for logical consistency
        if summary_passed + summary_failed + summary_errors != summary_total:
            check_result['issues'].append({
                'type': 'logical_inconsistency',
                'severity': 'high',
                'message': f'Passed ({summary_passed}) + Failed ({summary_failed}) + Errors ({summary_errors}) != Total ({summary_total})',
                'recommendation': 'Ensure all test outcomes are properly counted'
            })
            check_result['status'] = 'failed'
        
        # Check overall success logic
        overall_success = summary.get('overall_success', False)
        should_be_successful = summary_failed == 0 and summary_errors == 0
        
        if overall_success != should_be_successful:
            check_result['issues'].append({
                'type': 'success_status_mismatch',
                'severity': 'high',
                'message': f'Overall success ({overall_success}) inconsistent with failed ({summary_failed}) and errors ({summary_errors})',
                'recommendation': 'Check overall success calculation logic'
            })
            check_result['status'] = 'failed'
        
        # Check execution time reasonableness
        execution_time = summary.get('execution_time_seconds', 0)
        if execution_time <= 0:
            check_result['warnings'].append({
                'type': 'invalid_execution_time',
                'severity': 'low',
                'message': f'Execution time ({execution_time}s) is not positive',
                'recommendation': 'Ensure execution time is measured correctly'
            })
        elif execution_time > 3600:  # > 1 hour
            check_result['warnings'].append({
                'type': 'long_execution_time',
                'severity': 'low',
                'message': f'Execution time ({execution_time:.1f}s) is unusually long',
                'recommendation': 'Consider optimizing test execution or checking for timeouts'
            })
        
        return check_result
    
    def _validate_git_tracking(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate git tracking information."""
        check_result = {
            'status': 'passed',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        git_info = test_results.get('run_metadata', {}).get('git_info', {})
        
        if not git_info:
            check_result['issues'].append({
                'type': 'missing_git_info',
                'severity': 'high',
                'message': 'Git tracking information is missing',
                'recommendation': 'Ensure test runner captures git information'
            })
            check_result['status'] = 'failed'
            return check_result
        
        # Check required git fields
        required_git_fields = ['commit_hash', 'branch', 'commit_message', 'commit_date']
        for field in required_git_fields:
            if not git_info.get(field) or git_info[field] == 'unknown':
                check_result['warnings'].append({
                    'type': 'incomplete_git_info',
                    'severity': 'medium',
                    'field': field,
                    'message': f'Git field "{field}" is missing or unknown',
                    'recommendation': 'Check git repository status and access'
                })
        
        # Check for uncommitted changes
        has_uncommitted = git_info.get('has_uncommitted_changes', False)
        if has_uncommitted:
            uncommitted_files = git_info.get('uncommitted_files', [])
            check_result['warnings'].append({
                'type': 'uncommitted_changes',
                'severity': 'medium',
                'message': f'Tests run with {len(uncommitted_files)} uncommitted changes',
                'recommendation': 'Consider running tests on clean commits for reproducibility'
            })
        
        # Validate commit hash format
        commit_hash = git_info.get('commit_hash', '')
        if commit_hash and commit_hash != 'unknown':
            if not re.match(r'^[a-f0-9]{40}$', commit_hash):
                check_result['warnings'].append({
                    'type': 'invalid_commit_hash',
                    'severity': 'low',
                    'message': f'Commit hash format appears invalid: {commit_hash[:20]}...',
                    'recommendation': 'Verify git command output formatting'
                })
        
        return check_result
    
    def _validate_week_requirements(self, test_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate week-specific requirements."""
        metadata = test_results.get('run_metadata', {})
        week = metadata.get('week')
        
        if not week or week not in self.week_test_requirements:
            return None  # No week-specific validation needed
        
        check_result = {
            'status': 'passed',
            'week': week,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        requirements = self.week_test_requirements[week]
        
        # Check required test suites
        test_results_data = test_results.get('test_results', {})
        required_suites = requirements.get('required_test_suites', [])
        
        for required_suite in required_suites:
            if not any(required_suite.lower() in suite_name.lower() 
                      for suite_name in test_results_data.keys()):
                check_result['issues'].append({
                    'type': 'missing_required_suite',
                    'severity': 'high',
                    'suite': required_suite,
                    'message': f'Required test suite "{required_suite}" not found',
                    'recommendation': f'Implement and run "{required_suite}" for Week {week}'
                })
                check_result['status'] = 'failed'
        
        # Check minimum test count
        minimum_tests = requirements.get('minimum_tests', 0)
        total_tests = test_results.get('summary', {}).get('total_tests', 0)
        
        if total_tests < minimum_tests:
            check_result['warnings'].append({
                'type': 'insufficient_test_coverage',
                'severity': 'medium',
                'message': f'Total tests ({total_tests}) below minimum ({minimum_tests}) for Week {week}',
                'recommendation': f'Add more comprehensive tests to meet Week {week} requirements'
            })
        
        # Check objectives status
        summary = test_results.get('summary', {})
        objectives_status = summary.get('objectives_status', {})
        required_objectives = requirements.get('required_objectives', [])
        
        for objective in required_objectives:
            status = objectives_status.get(objective, 'not_tested')
            if status == 'failed':
                check_result['issues'].append({
                    'type': 'objective_failed',
                    'severity': 'high',
                    'objective': objective,
                    'message': f'Week {week} objective "{objective}" failed',
                    'recommendation': f'Address failures in "{objective}" implementation'
                })
                check_result['status'] = 'failed'
            elif status == 'not_tested':
                check_result['warnings'].append({
                    'type': 'objective_not_tested',
                    'severity': 'medium',
                    'objective': objective,
                    'message': f'Week {week} objective "{objective}" not tested',
                    'recommendation': f'Implement tests for "{objective}" objective'
                })
        
        # Performance requirements (if applicable)
        perf_requirements = requirements.get('performance_requirements')
        if perf_requirements:
            performance_success = summary.get('performance_success', False)
            if perf_requirements.get('must_meet_target', False) and not performance_success:
                check_result['issues'].append({
                    'type': 'performance_requirement_failed',
                    'severity': 'high',
                    'message': f'Week {week} performance requirements not met',
                    'recommendation': 'Optimize implementation to meet performance targets'
                })
                check_result['status'] = 'failed'
        
        return check_result
    
    def _validate_test_quality(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test quality metrics."""
        check_result = {
            'status': 'passed',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        test_results_data = test_results.get('test_results', {})
        summary = test_results.get('summary', {})
        
        # Check for test failures
        total_failed = summary.get('total_failed', 0)
        total_errors = summary.get('total_errors', 0)
        total_tests = summary.get('total_tests', 0)
        
        if total_tests == 0:
            check_result['issues'].append({
                'type': 'no_tests_run',
                'severity': 'critical',
                'message': 'No tests were executed',
                'recommendation': 'Ensure test suites are properly configured and runnable'
            })
            check_result['status'] = 'failed'
        
        # Check failure rate
        if total_tests > 0:
            failure_rate = (total_failed + total_errors) / total_tests
            if failure_rate > 0.2:  # >20% failure rate
                check_result['warnings'].append({
                    'type': 'high_failure_rate',
                    'severity': 'medium',
                    'failure_rate': failure_rate,
                    'message': f'High test failure rate: {failure_rate:.1%}',
                    'recommendation': 'Investigate and fix failing tests to improve reliability'
                })
        
        # Check for empty test suites
        for suite_name, suite_result in test_results_data.items():
            tests_run = suite_result.get('tests_run', 0)
            if tests_run == 0:
                status = suite_result.get('status', 'unknown')
                if status == 'not_implemented':
                    check_result['recommendations'].append({
                        'type': 'suite_not_implemented',
                        'severity': 'low',
                        'suite': suite_name,
                        'message': f'Test suite "{suite_name}" not yet implemented',
                        'recommendation': f'Implement test cases for "{suite_name}"'
                    })
                else:
                    check_result['warnings'].append({
                        'type': 'empty_test_suite',
                        'severity': 'low',
                        'suite': suite_name,
                        'message': f'Test suite "{suite_name}" ran no tests',
                        'recommendation': f'Verify test discovery for "{suite_name}"'
                    })
        
        return check_result
    
    def _assess_compliance(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall compliance with testing standards."""
        issues = validation_result.get('issues', [])
        warnings = validation_result.get('warnings', [])
        
        # Count issues by severity
        critical_issues = sum(1 for i in issues if i.get('severity') == 'critical')
        high_issues = sum(1 for i in issues if i.get('severity') == 'high')
        medium_issues = sum(1 for i in issues if i.get('severity') == 'medium')
        
        # Determine compliance level
        if critical_issues > 0:
            compliance_level = 'non_compliant'
        elif high_issues > 0:
            compliance_level = 'partially_compliant'
        elif medium_issues > 0 or len(warnings) > 5:
            compliance_level = 'mostly_compliant'
        else:
            compliance_level = 'fully_compliant'
        
        return {
            'compliance_level': compliance_level,
            'critical_issues': critical_issues,
            'high_issues': high_issues,
            'medium_issues': medium_issues,
            'warnings': len(warnings),
            'overall_score': self._calculate_compliance_score(critical_issues, high_issues, medium_issues, len(warnings))
        }
    
    def _calculate_compliance_score(self, critical: int, high: int, medium: int, warnings: int) -> float:
        """Calculate a compliance score (0-100)."""
        # Penalty system
        penalty = (critical * 25) + (high * 10) + (medium * 5) + (warnings * 2)
        score = max(0, 100 - penalty)
        return round(score, 1)


def main():
    """Main entry point for standalone validation."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python result_validator.py <test_results.json>")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    if not results_file.exists():
        print(f"Error: File {results_file} not found")
        sys.exit(1)
    
    try:
        with open(results_file, 'r') as f:
            test_results = json.load(f)
        
        validator = ResultValidator()
        validation_result = validator.validate_test_results(test_results)
        
        print("Test Result Validation:")
        print(f"Status: {validation_result['validation_status'].upper()}")
        print(f"Run ID: {validation_result['run_id']}")
        
        compliance = validation_result.get('compliance_status', {})
        if compliance:
            print(f"Compliance: {compliance['compliance_level']} (Score: {compliance['overall_score']}/100)")
        
        if validation_result.get('issues'):
            print(f"Issues: {len(validation_result['issues'])}")
            for issue in validation_result['issues']:
                severity_icon = {'critical': '🔥', 'high': '❌', 'medium': '⚠️', 'low': 'ℹ️'}.get(issue.get('severity'), '❓')
                print(f"  {severity_icon} {issue['message']}")
        
        if validation_result.get('warnings'):
            print(f"Warnings: {len(validation_result['warnings'])}")
            for warning in validation_result['warnings']:
                print(f"  ⚠️ {warning['message']}")
        
        # Exit with status based on validation result
        if validation_result['validation_status'] == 'failed':
            sys.exit(1)
        elif validation_result['validation_status'] == 'error':
            sys.exit(2)
        else:
            sys.exit(0)
        
    except Exception as e:
        print(f"Error validating test results: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()