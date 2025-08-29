"""Performance Validation Framework - Validate performance requirements."""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class PerformanceValidator:
    """Validates performance requirements and tracks trends."""
    
    def __init__(self, project_root: Path = None):
        """Initialize performance validator."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = project_root
        self.performance_dir = project_root / "testing" / "logs" / "performance"
        self.history_file = self.performance_dir / "performance_history.json"
        
        # Performance requirements by week
        self.week_requirements = {
            1: {'target_ms': 500.0, 'description': 'Super Admin Layer Foundation'},
            2: {'target_ms': 100.0, 'description': 'Component & Station Layer Implementation'},
            3: {'target_ms': 80.0, 'description': 'Line & PM Layer Foundation'},
            4: {'target_ms': 60.0, 'description': 'Advanced Optimization Algorithms'}
        }
    
    def validate_performance_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single performance test run."""
        validation_result = {
            'run_id': run_data.get('run_metadata', {}).get('run_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'validation_status': 'passed',
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'performance_metrics': {}
        }
        
        try:
            metadata = run_data.get('run_metadata', {})
            summary = run_data.get('summary', {})
            benchmarks = run_data.get('benchmarks', {})
            
            # Validate against target
            target_ms = metadata.get('target_ms', summary.get('target_ms', 100.0))
            avg_performance = summary.get('average_performance_ms', 0)
            
            validation_result['performance_metrics'] = {
                'target_ms': target_ms,
                'actual_avg_ms': avg_performance,
                'performance_ratio': avg_performance / target_ms if target_ms > 0 else 0,
                'meets_target': avg_performance < target_ms if avg_performance > 0 else False
            }
            
            # Check overall performance
            if not summary.get('overall_success', False):
                validation_result['validation_status'] = 'failed'
                validation_result['issues'].append({
                    'type': 'performance_failure',
                    'severity': 'high',
                    'message': f'Overall performance failed. Average: {avg_performance:.1f}ms, Target: <{target_ms}ms',
                    'recommendation': 'Investigate performance bottlenecks in failing test suites'
                })
            
            # Validate individual benchmarks
            for suite_name, benchmark_data in benchmarks.items():
                suite_validation = self._validate_benchmark_suite(suite_name, benchmark_data, target_ms)
                
                if suite_validation['issues']:
                    validation_result['issues'].extend(suite_validation['issues'])
                if suite_validation['warnings']:
                    validation_result['warnings'].extend(suite_validation['warnings'])
                if suite_validation['recommendations']:
                    validation_result['recommendations'].extend(suite_validation['recommendations'])
            
            # Check for performance regression
            regression_check = self._check_performance_regression(avg_performance, target_ms)
            if regression_check:
                validation_result['warnings'].append(regression_check)
            
            # Performance trend analysis
            trend_analysis = self._analyze_performance_trend()
            if trend_analysis:
                validation_result['trend_analysis'] = trend_analysis
            
            # Set final validation status
            if validation_result['issues']:
                validation_result['validation_status'] = 'failed'
            elif validation_result['warnings']:
                validation_result['validation_status'] = 'warning'
            else:
                validation_result['validation_status'] = 'passed'
            
        except Exception as e:
            validation_result['validation_status'] = 'error'
            validation_result['issues'].append({
                'type': 'validation_error',
                'severity': 'high',
                'message': f'Validation failed with error: {str(e)}',
                'recommendation': 'Check validation logic and input data format'
            })
        
        return validation_result
    
    def _validate_benchmark_suite(self, suite_name: str, benchmark_data: Dict[str, Any], 
                                target_ms: float) -> Dict[str, Any]:
        """Validate individual benchmark suite."""
        result = {'issues': [], 'warnings': [], 'recommendations': []}
        
        avg_time = benchmark_data.get('avg_time_ms', 0)
        max_time = benchmark_data.get('max_time_ms', 0)
        std_dev = benchmark_data.get('std_dev_ms', 0)
        runs = benchmark_data.get('runs', 1)
        
        # Check if suite meets target
        if avg_time >= target_ms:
            result['issues'].append({
                'type': 'performance_target_miss',
                'severity': 'high',
                'suite': suite_name,
                'message': f'{suite_name}: {avg_time:.1f}ms exceeds target of {target_ms}ms',
                'recommendation': f'Optimize {suite_name.lower()} implementation'
            })
        
        # Check for high variability
        if runs > 1 and std_dev > (avg_time * 0.2):  # >20% variability
            result['warnings'].append({
                'type': 'high_variability',
                'severity': 'medium',
                'suite': suite_name,
                'message': f'{suite_name}: High performance variability (σ={std_dev:.1f}ms)',
                'recommendation': 'Investigate sources of performance inconsistency'
            })
        
        # Check for slow maximum time
        if max_time > (target_ms * 1.5):
            result['warnings'].append({
                'type': 'slow_maximum',
                'severity': 'medium', 
                'suite': suite_name,
                'message': f'{suite_name}: Slow maximum time ({max_time:.1f}ms)',
                'recommendation': 'Investigate worst-case performance scenarios'
            })
        
        # Check for synthetic data usage
        if benchmark_data.get('source') == 'synthetic':
            result['warnings'].append({
                'type': 'synthetic_data',
                'severity': 'low',
                'suite': suite_name,
                'message': f'{suite_name}: Using synthetic benchmark data',
                'recommendation': 'Implement dedicated performance tests for accurate measurement'
            })
        
        return result
    
    def _check_performance_regression(self, current_avg: float, target_ms: float) -> Optional[Dict[str, Any]]:
        """Check for performance regression compared to recent runs."""
        try:
            if not self.history_file.exists():
                return None
            
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            runs = history.get('performance_runs', [])
            if len(runs) < 2:
                return None
            
            # Get recent runs (last 5)
            recent_runs = runs[-5:-1]  # Exclude current run
            if not recent_runs:
                return None
            
            recent_averages = [run['average_performance_ms'] for run in recent_runs 
                             if run.get('average_performance_ms', 0) > 0]
            
            if not recent_averages:
                return None
            
            recent_avg = statistics.mean(recent_averages)
            
            # Check for significant regression (>20% slower)
            if current_avg > recent_avg * 1.2:
                return {
                    'type': 'performance_regression',
                    'severity': 'high',
                    'message': f'Performance regression detected: {current_avg:.1f}ms vs recent average {recent_avg:.1f}ms',
                    'recommendation': 'Investigate recent code changes that may have impacted performance'
                }
            
        except Exception:
            return None
        
        return None
    
    def _analyze_performance_trend(self) -> Optional[Dict[str, Any]]:
        """Analyze performance trends over time."""
        try:
            if not self.history_file.exists():
                return None
            
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            runs = history.get('performance_runs', [])
            if len(runs) < 10:  # Need at least 10 runs for trend analysis
                return None
            
            # Get performance data over time
            timestamps = []
            performances = []
            
            for run in runs[-20:]:  # Last 20 runs
                if run.get('average_performance_ms', 0) > 0:
                    try:
                        timestamp = datetime.fromisoformat(run['timestamp'])
                        timestamps.append(timestamp)
                        performances.append(run['average_performance_ms'])
                    except (ValueError, KeyError):
                        continue
            
            if len(performances) < 10:
                return None
            
            # Simple trend analysis
            first_half = performances[:len(performances)//2]
            second_half = performances[len(performances)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            trend_direction = 'improving' if second_avg < first_avg else 'degrading'
            trend_magnitude = abs(second_avg - first_avg) / first_avg
            
            return {
                'trend_direction': trend_direction,
                'trend_magnitude_percent': trend_magnitude * 100,
                'first_half_avg': first_avg,
                'second_half_avg': second_avg,
                'analysis_period_runs': len(performances),
                'significance': 'high' if trend_magnitude > 0.15 else 'medium' if trend_magnitude > 0.05 else 'low'
            }
            
        except Exception:
            return None
    
    def validate_week_requirements(self, week: int, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance against specific week requirements."""
        if week not in self.week_requirements:
            return {
                'validation_status': 'error',
                'message': f'No requirements defined for week {week}'
            }
        
        requirements = self.week_requirements[week]
        target_ms = requirements['target_ms']
        description = requirements['description']
        
        avg_performance = performance_data.get('average_performance_ms', 0)
        
        meets_requirements = avg_performance > 0 and avg_performance < target_ms
        
        result = {
            'week': week,
            'week_description': description,
            'target_ms': target_ms,
            'actual_avg_ms': avg_performance,
            'meets_requirements': meets_requirements,
            'validation_status': 'passed' if meets_requirements else 'failed',
            'performance_margin_ms': target_ms - avg_performance if avg_performance > 0 else 0,
            'performance_margin_percent': ((target_ms - avg_performance) / target_ms * 100) if avg_performance > 0 else 0
        }
        
        if meets_requirements:
            if result['performance_margin_percent'] > 50:
                result['grade'] = 'excellent'
            elif result['performance_margin_percent'] > 20:
                result['grade'] = 'good'
            else:
                result['grade'] = 'acceptable'
        else:
            result['grade'] = 'failing'
        
        return result
    
    def generate_performance_report(self, validation_results: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive performance validation report."""
        report_lines = []
        
        report_lines.append("# Performance Validation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        total_runs = len(validation_results)
        passed_runs = sum(1 for r in validation_results if r.get('validation_status') == 'passed')
        failed_runs = sum(1 for r in validation_results if r.get('validation_status') == 'failed')
        warning_runs = sum(1 for r in validation_results if r.get('validation_status') == 'warning')
        
        report_lines.append("## Summary")
        report_lines.append(f"- Total Runs Validated: {total_runs}")
        report_lines.append(f"- Passed: {passed_runs}")
        report_lines.append(f"- Failed: {failed_runs}")
        report_lines.append(f"- Warnings: {warning_runs}")
        report_lines.append("")
        
        # Individual run details
        report_lines.append("## Validation Results")
        
        for result in validation_results:
            run_id = result.get('run_id', 'unknown')
            status = result.get('validation_status', 'unknown')
            status_icon = {'passed': '✅', 'failed': '❌', 'warning': '⚠️', 'error': '💥'}.get(status, '❓')
            
            report_lines.append(f"### {status_icon} Run {run_id}")
            
            metrics = result.get('performance_metrics', {})
            if metrics:
                report_lines.append(f"- Target: <{metrics.get('target_ms', 0):.1f}ms")
                report_lines.append(f"- Actual: {metrics.get('actual_avg_ms', 0):.1f}ms")
                report_lines.append(f"- Performance Ratio: {metrics.get('performance_ratio', 0):.2f}x")
            
            # Issues
            issues = result.get('issues', [])
            if issues:
                report_lines.append("#### Issues:")
                for issue in issues:
                    report_lines.append(f"- **{issue.get('type', 'unknown')}**: {issue.get('message', 'No message')}")
                    if issue.get('recommendation'):
                        report_lines.append(f"  - *Recommendation*: {issue['recommendation']}")
            
            # Warnings
            warnings = result.get('warnings', [])
            if warnings:
                report_lines.append("#### Warnings:")
                for warning in warnings:
                    report_lines.append(f"- **{warning.get('type', 'unknown')}**: {warning.get('message', 'No message')}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main entry point for standalone validation."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python performance_validator.py <performance_results.json>")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    if not results_file.exists():
        print(f"Error: File {results_file} not found")
        sys.exit(1)
    
    try:
        with open(results_file, 'r') as f:
            performance_data = json.load(f)
        
        validator = PerformanceValidator()
        validation_result = validator.validate_performance_run(performance_data)
        
        print("Performance Validation Result:")
        print(f"Status: {validation_result['validation_status'].upper()}")
        print(f"Run ID: {validation_result['run_id']}")
        
        if validation_result.get('performance_metrics'):
            metrics = validation_result['performance_metrics']
            print(f"Performance: {metrics['actual_avg_ms']:.1f}ms (target: <{metrics['target_ms']}ms)")
        
        if validation_result.get('issues'):
            print(f"Issues: {len(validation_result['issues'])}")
            for issue in validation_result['issues']:
                print(f"  - {issue['message']}")
        
        if validation_result.get('warnings'):
            print(f"Warnings: {len(validation_result['warnings'])}")
            for warning in validation_result['warnings']:
                print(f"  - {warning['message']}")
        
        # Exit with status based on validation result
        if validation_result['validation_status'] == 'failed':
            sys.exit(1)
        elif validation_result['validation_status'] == 'error':
            sys.exit(2)
        else:
            sys.exit(0)
        
    except Exception as e:
        print(f"Error validating performance data: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()