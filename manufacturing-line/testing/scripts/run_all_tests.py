#!/usr/bin/env python3
"""Master Test Runner - Centralized test execution with git tracking."""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestRunner:
    """Centralized test execution with git commit tracking."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize test runner."""
        self.project_root = project_root
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "testing" / "logs" / "test_runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get git information
        self.git_info = self._get_git_info()
        
        # Test run metadata
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_results = {}
        
    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit information."""
        try:
            # Get current commit hash
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Get commit message
            commit_message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%B'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Get branch name
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Get commit date
            commit_date = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%ci'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Check for uncommitted changes
            status_output = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            has_uncommitted = bool(status_output)
            
            return {
                'commit_hash': commit_hash,
                'commit_message': commit_message,
                'branch': branch,
                'commit_date': commit_date,
                'has_uncommitted_changes': has_uncommitted,
                'uncommitted_files': status_output.split('\n') if has_uncommitted else []
            }
            
        except subprocess.CalledProcessError:
            return {
                'commit_hash': 'unknown',
                'commit_message': 'Git not available',
                'branch': 'unknown',
                'commit_date': 'unknown',
                'has_uncommitted_changes': False,
                'uncommitted_files': []
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites with comprehensive logging."""
        print(f"Starting test execution - Run ID: {self.run_id}")
        print(f"Git Commit: {self.git_info['commit_hash'][:8]} ({self.git_info['branch']})")
        print(f"Commit Message: {self.git_info['commit_message']}")
        
        if self.git_info['has_uncommitted_changes']:
            print("⚠️  Warning: Uncommitted changes detected")
        
        start_time = time.time()
        
        # Test suites to run
        test_suites = [
            {
                'name': 'Week 2 Component Layer Tests',
                'module': 'tests.unit.test_component_layer_engine',
                'category': 'unit'
            },
            {
                'name': 'Week 2 Station Layer Tests',
                'module': 'tests.unit.test_station_layer_engine', 
                'category': 'unit'
            }
        ]
        
        results = {
            'run_metadata': {
                'run_id': self.run_id,
                'start_time': datetime.now().isoformat(),
                'git_info': self.git_info,
                'test_environment': {
                    'python_version': sys.version,
                    'platform': sys.platform,
                    'working_directory': str(self.project_root)
                }
            },
            'test_results': {},
            'summary': {}
        }
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        for suite in test_suites:
            print(f"\n🧪 Running: {suite['name']}")
            print("-" * 50)
            
            suite_result = self._run_test_suite(suite)
            results['test_results'][suite['name']] = suite_result
            
            # Aggregate results
            total_tests += suite_result['tests_run']
            total_passed += suite_result['passed']
            total_failed += suite_result['failed']
            total_errors += suite_result['errors']
            
            # Print suite summary
            status = "✅ PASSED" if suite_result['success'] else "❌ FAILED"
            print(f"Suite Status: {status}")
            print(f"Tests: {suite_result['tests_run']}, "
                  f"Passed: {suite_result['passed']}, "
                  f"Failed: {suite_result['failed']}, "
                  f"Errors: {suite_result['errors']}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Overall summary
        overall_success = total_failed == 0 and total_errors == 0
        results['summary'] = {
            'overall_success': overall_success,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'execution_time_seconds': execution_time,
            'end_time': datetime.now().isoformat()
        }
        
        # Print final summary
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"Run ID: {self.run_id}")
        print(f"Git Commit: {self.git_info['commit_hash'][:8]} ({self.git_info['branch']})")
        print(f"Overall Status: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Errors: {total_errors}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        # Save results
        self._save_results(results)
        
        # Create git tracking entry
        self._create_git_tracking_entry(results)
        
        return results
    
    def _run_test_suite(self, suite: Dict[str, str]) -> Dict[str, Any]:
        """Run a single test suite."""
        try:
            # Run the test module
            cmd = [sys.executable, '-m', 'unittest', suite['module'], '-v']
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per suite
            )
            
            # Parse output for test counts
            output_lines = result.stderr.split('\n')
            
            tests_run = 0
            passed = 0
            failed = 0
            errors = 0
            
            for line in output_lines:
                if 'Ran' in line and 'test' in line:
                    # Extract number of tests run
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            tests_run = int(parts[1])
                        except ValueError:
                            pass
                elif 'FAILED' in line and 'failures=' in line:
                    # Parse failures and errors
                    if 'failures=' in line:
                        try:
                            failures_part = line.split('failures=')[1].split(',')[0].split(')')[0]
                            failed = int(failures_part)
                        except (ValueError, IndexError):
                            pass
                    if 'errors=' in line:
                        try:
                            errors_part = line.split('errors=')[1].split(',')[0].split(')')[0]
                            errors = int(errors_part)
                        except (ValueError, IndexError):
                            pass
            
            # Calculate passed tests
            passed = tests_run - failed - errors
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'tests_run': tests_run,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': time.time()
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'tests_run': 0,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test suite timed out',
                'execution_time': 300
            }
        except Exception as e:
            return {
                'success': False,
                'tests_run': 0,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'return_code': -1,
                'stdout': '',
                'stderr': f'Exception running test suite: {str(e)}',
                'execution_time': 0
            }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save test results to JSON file."""
        output_file = self.output_dir / f"test_run_{self.run_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to: {output_file}")
    
    def _create_git_tracking_entry(self, results: Dict[str, Any]) -> None:
        """Create git tracking entry for this test run."""
        git_tracking_dir = self.project_root / "testing" / "logs" / "git_tracking"
        git_tracking_dir.mkdir(parents=True, exist_ok=True)
        
        tracking_entry = {
            'test_run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self.git_info['commit_hash'],
            'git_branch': self.git_info['branch'],
            'commit_message': self.git_info['commit_message'],
            'commit_date': self.git_info['commit_date'],
            'has_uncommitted_changes': self.git_info['has_uncommitted_changes'],
            'test_summary': results['summary'],
            'test_status': 'PASSED' if results['summary']['overall_success'] else 'FAILED'
        }
        
        # Save individual tracking entry
        tracking_file = git_tracking_dir / f"commit_{self.git_info['commit_hash'][:8]}_{self.run_id}.json"
        with open(tracking_file, 'w') as f:
            json.dump(tracking_entry, f, indent=2, default=str)
        
        # Update master tracking log
        master_log = git_tracking_dir / "test_history.json"
        
        if master_log.exists():
            with open(master_log, 'r') as f:
                history = json.load(f)
        else:
            history = {'test_runs': []}
        
        history['test_runs'].append(tracking_entry)
        
        # Keep only last 100 entries
        history['test_runs'] = history['test_runs'][-100:]
        
        with open(master_log, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        print(f"🔗 Git tracking entry created: {tracking_file}")
        print(f"📈 Master log updated: {master_log}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = None
    
    runner = TestRunner(output_dir)
    results = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results['summary']['overall_success'] else 1)


if __name__ == '__main__':
    main()