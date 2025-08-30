"""
Integration Testing Layer - Week 15: Comprehensive System Integration Testing & Validation

This layer provides comprehensive testing and validation capabilities for the manufacturing system
with cross-layer integration testing, user acceptance validation, performance benchmarking,
security auditing, and compliance verification.

Performance Targets:
- 100% integration test pass rate
- Complete user acceptance validation (57 user stories)
- Performance targets: <200ms response, >10,000 users, >99.9% uptime
- Security compliance: Zero critical vulnerabilities
- Business validation: 100% requirements traceability

Author: Manufacturing Line Control System
Created: Week 15 - Integration Testing & Validation Phase
"""

from .cross_layer_tests import CrossLayerTestSuite
from .end_to_end_tests import EndToEndTestSuite
from .user_acceptance_tests import UserAcceptanceTestSuite
from .performance_benchmarks import PerformanceBenchmarkSuite
from .scalability_tests import ScalabilityTestSuite
from .security_audit_tests import SecurityAuditSuite
from .requirements_traceability import RequirementsTraceabilityMatrix
from .test_dashboard import TestExecutionDashboard
from .quality_metrics import QualityMetricsReporter

__all__ = [
    'CrossLayerTestSuite',
    'EndToEndTestSuite', 
    'UserAcceptanceTestSuite',
    'PerformanceBenchmarkSuite',
    'ScalabilityTestSuite',
    'SecurityAuditSuite',
    'RequirementsTraceabilityMatrix',
    'TestExecutionDashboard',
    'QualityMetricsReporter'
]