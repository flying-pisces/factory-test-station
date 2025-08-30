"""
Requirements Traceability Matrix - Week 15: Integration Testing & Validation

This module provides comprehensive requirements traceability to ensure all business requirements,
functional requirements, and user stories are properly implemented, tested, and validated 
across the manufacturing control system.

Traceability Coverage:
- Business Requirements: Map business needs to system features
- Functional Requirements: Validate feature implementation completeness  
- Non-Functional Requirements: Performance, security, usability validation
- User Story Coverage: Ensure all user stories are tested and validated
- Test Case Mapping: Link test cases to requirements and user stories
- Gap Analysis: Identify untested or missing requirements

Author: Manufacturing Line Control System
Created: Week 15 - Requirements Traceability Phase
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import os


class RequirementType(Enum):
    """Types of requirements."""
    BUSINESS = "business"
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    USER_STORY = "user_story"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"


class RequirementPriority(Enum):
    """Requirement priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RequirementStatus(Enum):
    """Implementation and testing status."""
    DRAFT = "draft"
    APPROVED = "approved"
    IN_DEVELOPMENT = "in_development"
    IMPLEMENTED = "implemented"
    IN_TESTING = "in_testing"
    TESTED = "tested"
    VERIFIED = "verified"
    ACCEPTED = "accepted"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class TestStatus(Enum):
    """Test execution status."""
    NOT_TESTED = "not_tested"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class Requirement:
    """Individual requirement definition."""
    requirement_id: str
    title: str
    description: str
    requirement_type: RequirementType
    priority: RequirementPriority
    status: RequirementStatus
    source: str  # Who requested it
    rationale: str  # Why it's needed
    acceptance_criteria: List[str] = field(default_factory=list)
    related_requirements: List[str] = field(default_factory=list)
    implementation_notes: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    updated_date: datetime = field(default_factory=datetime.now)
    assigned_to: str = ""
    estimated_effort_hours: Optional[int] = None
    actual_effort_hours: Optional[int] = None
    business_value: str = ""
    risk_level: str = "medium"
    
    def __post_init__(self):
        if not self.requirement_id:
            self.requirement_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"


@dataclass 
class TestCase:
    """Test case definition."""
    test_case_id: str
    title: str
    description: str
    test_type: str  # unit, integration, system, acceptance
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[str] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    test_data: Dict[str, Any] = field(default_factory=dict)
    automated: bool = False
    priority: RequirementPriority = RequirementPriority.MEDIUM
    status: TestStatus = TestStatus.NOT_TESTED
    execution_notes: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    executed_by: str = ""
    execution_time_minutes: Optional[int] = None
    
    def __post_init__(self):
        if not self.test_case_id:
            self.test_case_id = f"TC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class TraceabilityLink:
    """Link between requirement and test case."""
    requirement_id: str
    test_case_id: str
    coverage_type: str  # "direct", "indirect", "partial"
    verification_method: str  # "test", "review", "demo", "analysis"
    coverage_percentage: float = 100.0  # How well test covers requirement
    notes: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    verified_by: str = ""
    verified_date: Optional[datetime] = None


@dataclass
class UserStory:
    """User story definition."""
    story_id: str
    title: str
    description: str
    user_role: str
    user_goal: str
    user_benefit: str
    acceptance_criteria: List[str] = field(default_factory=list)
    story_points: Optional[int] = None
    priority: RequirementPriority = RequirementPriority.MEDIUM
    status: RequirementStatus = RequirementStatus.DRAFT
    epic: str = ""
    sprint: str = ""
    related_requirements: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.story_id:
            self.story_id = f"US-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class CoverageReport:
    """Requirement coverage analysis report."""
    total_requirements: int
    covered_requirements: int
    uncovered_requirements: int
    partially_covered_requirements: int
    coverage_percentage: float
    coverage_by_type: Dict[RequirementType, float] = field(default_factory=dict)
    coverage_by_priority: Dict[RequirementPriority, float] = field(default_factory=dict)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_date: datetime = field(default_factory=datetime.now)


class RequirementsTraceabilityMatrix:
    """
    Comprehensive Requirements Traceability Matrix
    
    Provides full traceability between business requirements, functional specifications,
    user stories, test cases, and implementation artifacts to ensure complete coverage
    and validation of all system requirements.
    """
    
    def __init__(self, project_name: str = "Manufacturing Line Control System"):
        self.project_name = project_name
        self.logger = logging.getLogger(__name__)
        
        # Core data structures
        self.requirements: Dict[str, Requirement] = {}
        self.test_cases: Dict[str, TestCase] = {}
        self.user_stories: Dict[str, UserStory] = {}
        self.traceability_links: List[TraceabilityLink] = []
        
        # Initialize with manufacturing system requirements
        self._initialize_manufacturing_requirements()
        self._initialize_user_stories()
        self._initialize_test_cases()
        self._establish_traceability_links()
    
    def _initialize_manufacturing_requirements(self):
        """Initialize core manufacturing system requirements."""
        
        # Business Requirements
        business_requirements = [
            {
                'id': 'BR001',
                'title': 'Real-time Manufacturing Control',
                'description': 'System shall provide real-time control and monitoring of manufacturing processes',
                'type': RequirementType.BUSINESS,
                'priority': RequirementPriority.CRITICAL,
                'source': 'Manufacturing Operations',
                'rationale': 'Essential for production efficiency and quality control',
                'acceptance_criteria': [
                    'Control loop response time < 10ms',
                    'Real-time data updates every 100ms',
                    'Zero data loss during normal operations'
                ],
                'business_value': 'Enables precision manufacturing and quality assurance'
            },
            {
                'id': 'BR002', 
                'title': 'Production Efficiency Optimization',
                'description': 'System shall optimize production efficiency through AI-driven insights and automation',
                'type': RequirementType.BUSINESS,
                'priority': RequirementPriority.HIGH,
                'source': 'Production Management',
                'rationale': 'Reduce waste, increase throughput, minimize downtime',
                'acceptance_criteria': [
                    'Achieve >95% Overall Equipment Effectiveness (OEE)',
                    'Reduce setup times by 30%',
                    'Predict maintenance needs with >90% accuracy'
                ],
                'business_value': 'Increases profitability through operational excellence'
            },
            {
                'id': 'BR003',
                'title': 'Quality Management System',
                'description': 'System shall implement comprehensive quality management with automated inspection',
                'type': RequirementType.BUSINESS,
                'priority': RequirementPriority.CRITICAL,
                'source': 'Quality Assurance',
                'rationale': 'Ensure product quality meets specifications and regulations',
                'acceptance_criteria': [
                    'Defect detection rate >99.5%',
                    'False positive rate <1%',
                    'Complete traceability of all products'
                ],
                'business_value': 'Maintains brand reputation and customer satisfaction'
            },
            {
                'id': 'BR004',
                'title': 'Predictive Maintenance',
                'description': 'System shall provide predictive maintenance capabilities to prevent unplanned downtime',
                'type': RequirementType.BUSINESS,
                'priority': RequirementPriority.HIGH,
                'source': 'Maintenance Department',
                'rationale': 'Reduce maintenance costs and prevent production disruptions',
                'acceptance_criteria': [
                    'Predict failures 72 hours in advance',
                    'Reduce unplanned downtime by 50%',
                    'Optimize maintenance schedules'
                ],
                'business_value': 'Reduces operational costs and increases availability'
            }
        ]
        
        # Functional Requirements
        functional_requirements = [
            {
                'id': 'FR001',
                'title': 'Sensor Data Processing',
                'description': 'System shall process sensor data from multiple manufacturing stations',
                'type': RequirementType.FUNCTIONAL,
                'priority': RequirementPriority.CRITICAL,
                'source': 'System Architecture',
                'rationale': 'Foundation for all control and monitoring functions',
                'acceptance_criteria': [
                    'Process 1M+ sensor readings per hour',
                    'Support 500+ concurrent sensors',
                    'Data validation and error detection'
                ],
                'related_requirements': ['BR001']
            },
            {
                'id': 'FR002',
                'title': 'Control Algorithm Execution',
                'description': 'System shall execute control algorithms for manufacturing process control',
                'type': RequirementType.FUNCTIONAL,
                'priority': RequirementPriority.CRITICAL,
                'source': 'Control Engineering',
                'rationale': 'Direct manufacturing process control',
                'acceptance_criteria': [
                    'PID control loops with <10ms response time',
                    'Multi-variable control support',
                    'Safety interlocks and emergency stops'
                ],
                'related_requirements': ['BR001', 'BR002']
            },
            {
                'id': 'FR003',
                'title': 'AI-Based Analytics',
                'description': 'System shall provide AI-based analytics for process optimization and prediction',
                'type': RequirementType.FUNCTIONAL,
                'priority': RequirementPriority.HIGH,
                'source': 'Data Science Team',
                'rationale': 'Enable intelligent decision making and optimization',
                'acceptance_criteria': [
                    'Real-time ML model inference <100ms',
                    'Continuous model training and improvement',
                    'Pattern recognition and anomaly detection'
                ],
                'related_requirements': ['BR002', 'BR004']
            },
            {
                'id': 'FR004',
                'title': 'User Interface and Dashboards',
                'description': 'System shall provide intuitive user interfaces and real-time dashboards',
                'type': RequirementType.FUNCTIONAL,
                'priority': RequirementPriority.HIGH,
                'source': 'Operations Team',
                'rationale': 'Enable effective human-machine interaction',
                'acceptance_criteria': [
                    'Response time <200ms for UI interactions',
                    'Real-time data visualization',
                    'Role-based access control'
                ],
                'related_requirements': ['BR001', 'BR003']
            }
        ]
        
        # Non-Functional Requirements
        non_functional_requirements = [
            {
                'id': 'NFR001',
                'title': 'System Performance',
                'description': 'System shall meet specified performance benchmarks under full operational load',
                'type': RequirementType.NON_FUNCTIONAL,
                'priority': RequirementPriority.CRITICAL,
                'source': 'System Requirements',
                'rationale': 'Ensure system can handle production workloads',
                'acceptance_criteria': [
                    'Support 1000+ concurrent users',
                    'Process 10,000+ operations per second',
                    'Maintain <200ms response time at 95th percentile'
                ],
                'related_requirements': ['FR001', 'FR002', 'FR004']
            },
            {
                'id': 'NFR002',
                'title': 'System Security',
                'description': 'System shall implement comprehensive security controls and data protection',
                'type': RequirementType.NON_FUNCTIONAL,
                'priority': RequirementPriority.CRITICAL,
                'source': 'Security Team',
                'rationale': 'Protect sensitive manufacturing data and prevent cyber attacks',
                'acceptance_criteria': [
                    'Multi-factor authentication for all users',
                    'End-to-end encryption for all data transmission',
                    'Regular security audits and penetration testing'
                ],
                'related_requirements': ['FR004']
            },
            {
                'id': 'NFR003',
                'title': 'System Availability',
                'description': 'System shall provide high availability with minimal downtime',
                'type': RequirementType.NON_FUNCTIONAL,
                'priority': RequirementPriority.CRITICAL,
                'source': 'Operations',
                'rationale': 'Manufacturing operations require continuous system availability',
                'acceptance_criteria': [
                    '99.9% uptime availability',
                    'Automatic failover within 30 seconds',
                    'Complete disaster recovery capability'
                ],
                'related_requirements': ['BR001', 'BR002']
            },
            {
                'id': 'NFR004',
                'title': 'Scalability',
                'description': 'System shall scale to support growing manufacturing operations',
                'type': RequirementType.NON_FUNCTIONAL,
                'priority': RequirementPriority.HIGH,
                'source': 'Business Planning',
                'rationale': 'Support business growth and expansion',
                'acceptance_criteria': [
                    'Horizontal scaling to 10x current capacity',
                    'Linear performance scaling with resources',
                    'Auto-scaling based on demand'
                ],
                'related_requirements': ['NFR001', 'FR001']
            }
        ]
        
        # Create requirement objects
        all_requirements = business_requirements + functional_requirements + non_functional_requirements
        
        for req_data in all_requirements:
            requirement = Requirement(
                requirement_id=req_data['id'],
                title=req_data['title'],
                description=req_data['description'],
                requirement_type=req_data['type'],
                priority=req_data['priority'],
                status=RequirementStatus.APPROVED,
                source=req_data['source'],
                rationale=req_data['rationale'],
                acceptance_criteria=req_data['acceptance_criteria'],
                related_requirements=req_data.get('related_requirements', []),
                business_value=req_data.get('business_value', ''),
                estimated_effort_hours=req_data.get('effort_hours', None)
            )
            self.requirements[requirement.requirement_id] = requirement
    
    def _initialize_user_stories(self):
        """Initialize user stories for different roles."""
        
        user_stories_data = [
            # Production Operator Stories
            {
                'id': 'US001',
                'title': 'Monitor Real-time Production Metrics',
                'description': 'As a production operator, I want to monitor real-time production metrics so that I can ensure optimal manufacturing performance',
                'user_role': 'Production Operator',
                'user_goal': 'Monitor production metrics in real-time',
                'user_benefit': 'Ensure optimal manufacturing performance and quick issue detection',
                'acceptance_criteria': [
                    'Dashboard displays current production rate, quality metrics, and equipment status',
                    'Metrics update in real-time (every 5 seconds or less)',
                    'Clear visual indicators for normal/warning/alarm conditions',
                    'Historical trending available for last 24 hours'
                ],
                'priority': RequirementPriority.CRITICAL,
                'epic': 'Production Monitoring',
                'related_requirements': ['BR001', 'FR004']
            },
            {
                'id': 'US002',
                'title': 'Control Equipment Settings',
                'description': 'As a production operator, I want to adjust equipment settings so that I can optimize production parameters',
                'user_role': 'Production Operator',
                'user_goal': 'Adjust equipment operational parameters',
                'user_benefit': 'Optimize production based on current conditions',
                'acceptance_criteria': [
                    'Secure interface for adjusting operational parameters',
                    'Real-time validation of parameter ranges',
                    'Automatic logging of all parameter changes',
                    'Immediate feedback on parameter effects'
                ],
                'priority': RequirementPriority.CRITICAL,
                'epic': 'Production Control',
                'related_requirements': ['BR001', 'FR002']
            },
            
            # Production Manager Stories
            {
                'id': 'US003',
                'title': 'Review Production Performance KPIs',
                'description': 'As a production manager, I want to review production KPIs so that I can make informed decisions about operations',
                'user_role': 'Production Manager',
                'user_goal': 'Access comprehensive production performance data',
                'user_benefit': 'Make data-driven decisions to improve operations',
                'acceptance_criteria': [
                    'Executive dashboard with key performance indicators',
                    'OEE calculations and trending',
                    'Comparison against targets and historical data',
                    'Export capabilities for reporting'
                ],
                'priority': RequirementPriority.HIGH,
                'epic': 'Performance Management',
                'related_requirements': ['BR002', 'FR003']
            },
            {
                'id': 'US004',
                'title': 'Schedule Production Runs',
                'description': 'As a production manager, I want to schedule production runs so that I can optimize resource utilization',
                'user_role': 'Production Manager',
                'user_goal': 'Create and manage production schedules',
                'user_benefit': 'Optimize resource utilization and meet delivery commitments',
                'acceptance_criteria': [
                    'Visual scheduling interface with drag-and-drop capability',
                    'Resource availability checking',
                    'Automatic conflict detection and resolution suggestions',
                    'Integration with material requirements planning'
                ],
                'priority': RequirementPriority.HIGH,
                'epic': 'Production Planning',
                'related_requirements': ['BR002', 'FR003']
            },
            
            # Maintenance Technician Stories
            {
                'id': 'US005',
                'title': 'Monitor Equipment Health',
                'description': 'As a maintenance technician, I want to monitor equipment health so that I can perform proactive maintenance',
                'user_role': 'Maintenance Technician',
                'user_goal': 'Track equipment condition and health metrics',
                'user_benefit': 'Prevent equipment failures through proactive maintenance',
                'acceptance_criteria': [
                    'Equipment health dashboard with predictive indicators',
                    'Trend analysis for key health parameters',
                    'Automatic alerts for maintenance recommendations',
                    'Maintenance history and work order integration'
                ],
                'priority': RequirementPriority.HIGH,
                'epic': 'Predictive Maintenance',
                'related_requirements': ['BR004', 'FR003']
            },
            
            # Quality Controller Stories
            {
                'id': 'US006',
                'title': 'Monitor Quality Metrics and Control Charts',
                'description': 'As a quality controller, I want to monitor quality metrics so that I can ensure product quality standards',
                'user_role': 'Quality Controller',
                'user_goal': 'Track and control product quality',
                'user_benefit': 'Ensure products meet quality specifications and standards',
                'acceptance_criteria': [
                    'Real-time quality metrics and statistical process control charts',
                    'Automatic out-of-control detection and alerts',
                    'Quality trend analysis and reporting',
                    'Integration with inspection and testing equipment'
                ],
                'priority': RequirementPriority.CRITICAL,
                'epic': 'Quality Management',
                'related_requirements': ['BR003', 'FR001']
            }
        ]
        
        # Create user story objects
        for story_data in user_stories_data:
            user_story = UserStory(
                story_id=story_data['id'],
                title=story_data['title'],
                description=story_data['description'],
                user_role=story_data['user_role'],
                user_goal=story_data['user_goal'],
                user_benefit=story_data['user_benefit'],
                acceptance_criteria=story_data['acceptance_criteria'],
                priority=story_data['priority'],
                status=RequirementStatus.APPROVED,
                epic=story_data['epic'],
                related_requirements=story_data['related_requirements']
            )
            self.user_stories[user_story.story_id] = user_story
    
    def _initialize_test_cases(self):
        """Initialize comprehensive test cases."""
        
        test_cases_data = [
            # Performance Test Cases
            {
                'id': 'TC001',
                'title': 'Load Test - 1000 Concurrent Users',
                'description': 'Verify system performance with 1000 concurrent users',
                'type': 'performance',
                'preconditions': ['System deployed in test environment', 'Load testing tools configured'],
                'test_steps': [
                    'Configure load testing tool for 1000 concurrent users',
                    'Execute load test for 30 minutes',
                    'Monitor system resources and response times',
                    'Collect and analyze performance metrics'
                ],
                'expected_results': [
                    'Response time <200ms for 95% of requests',
                    'System maintains stability throughout test',
                    'No critical errors or failures',
                    'Resource utilization within acceptable limits'
                ],
                'automated': True,
                'priority': RequirementPriority.CRITICAL
            },
            {
                'id': 'TC002',
                'title': 'Real-time Control Loop Performance',
                'description': 'Verify control loop response time meets requirements',
                'type': 'performance',
                'preconditions': ['Control system configured', 'Test hardware connected'],
                'test_steps': [
                    'Configure control loop with test parameters',
                    'Apply step input to control system',
                    'Measure control loop response time',
                    'Record settling time and overshoot'
                ],
                'expected_results': [
                    'Control loop response time <10ms',
                    'Settling time within specifications',
                    'No oscillations or instability'
                ],
                'automated': True,
                'priority': RequirementPriority.CRITICAL
            },
            
            # Functional Test Cases
            {
                'id': 'TC003',
                'title': 'Sensor Data Processing Accuracy',
                'description': 'Verify accurate processing of sensor data inputs',
                'type': 'functional',
                'preconditions': ['Sensor simulation system available', 'Data processing module deployed'],
                'test_steps': [
                    'Configure sensor simulators with known values',
                    'Send test data to processing system',
                    'Verify processed data accuracy',
                    'Test error handling for invalid data'
                ],
                'expected_results': [
                    'Processed data matches expected values within tolerance',
                    'Invalid data properly rejected with error logging',
                    'All sensor channels processed correctly'
                ],
                'automated': True,
                'priority': RequirementPriority.CRITICAL
            },
            {
                'id': 'TC004',
                'title': 'AI Model Inference Performance',
                'description': 'Verify AI model inference meets performance requirements',
                'type': 'performance',
                'preconditions': ['AI models deployed', 'Test data available'],
                'test_steps': [
                    'Load AI models in inference engine',
                    'Submit test data for inference',
                    'Measure inference response time',
                    'Verify inference accuracy'
                ],
                'expected_results': [
                    'Inference response time <100ms',
                    'Model accuracy meets specifications',
                    'System handles concurrent inference requests'
                ],
                'automated': True,
                'priority': RequirementPriority.HIGH
            },
            
            # Security Test Cases
            {
                'id': 'TC005',
                'title': 'Authentication and Authorization',
                'description': 'Verify user authentication and role-based access control',
                'type': 'security',
                'preconditions': ['Security system configured', 'Test user accounts created'],
                'test_steps': [
                    'Attempt login with valid credentials',
                    'Attempt login with invalid credentials',
                    'Test role-based access to different features',
                    'Verify session timeout and security'
                ],
                'expected_results': [
                    'Valid users authenticate successfully',
                    'Invalid login attempts are rejected',
                    'Users can only access authorized features',
                    'Sessions timeout appropriately'
                ],
                'automated': True,
                'priority': RequirementPriority.CRITICAL
            },
            
            # User Acceptance Test Cases
            {
                'id': 'TC006',
                'title': 'Production Operator Dashboard Usability',
                'description': 'Verify production operator dashboard meets usability requirements',
                'type': 'acceptance',
                'preconditions': ['Production operator dashboard deployed', 'Test user available'],
                'test_steps': [
                    'Login as production operator',
                    'Navigate through dashboard features',
                    'Monitor real-time production metrics',
                    'Perform typical operational tasks',
                    'Collect user feedback on usability'
                ],
                'expected_results': [
                    'Dashboard loads within 3 seconds',
                    'Real-time data updates are clearly visible',
                    'All operational tasks can be completed intuitively',
                    'User satisfaction score >4.0/5.0'
                ],
                'automated': False,
                'priority': RequirementPriority.HIGH
            }
        ]
        
        # Create test case objects
        for tc_data in test_cases_data:
            test_case = TestCase(
                test_case_id=tc_data['id'],
                title=tc_data['title'],
                description=tc_data['description'],
                test_type=tc_data['type'],
                preconditions=tc_data['preconditions'],
                test_steps=tc_data['test_steps'],
                expected_results=tc_data['expected_results'],
                automated=tc_data['automated'],
                priority=tc_data['priority'],
                status=TestStatus.PLANNED
            )
            self.test_cases[test_case.test_case_id] = test_case
    
    def _establish_traceability_links(self):
        """Establish traceability links between requirements and test cases."""
        
        # Define traceability mappings
        traceability_mappings = [
            # Performance Requirements
            ('NFR001', 'TC001', 'direct', 'test', 100.0, 'Load testing verifies concurrent user performance'),
            ('FR002', 'TC002', 'direct', 'test', 100.0, 'Control loop performance test'),
            ('FR001', 'TC003', 'direct', 'test', 100.0, 'Sensor data processing accuracy test'),
            ('FR003', 'TC004', 'direct', 'test', 100.0, 'AI model inference performance test'),
            
            # Security Requirements
            ('NFR002', 'TC005', 'direct', 'test', 80.0, 'Authentication and authorization testing'),
            
            # User Stories to Test Cases
            ('US001', 'TC006', 'direct', 'test', 100.0, 'Production operator dashboard usability'),
            
            # Business Requirements to Test Cases
            ('BR001', 'TC002', 'indirect', 'test', 80.0, 'Real-time control supports business requirement'),
            ('BR001', 'TC003', 'indirect', 'test', 60.0, 'Sensor processing supports real-time control'),
            ('BR002', 'TC004', 'indirect', 'test', 70.0, 'AI analytics supports efficiency optimization'),
            ('BR003', 'TC003', 'indirect', 'test', 50.0, 'Accurate data processing supports quality management'),
            
            # Cross-requirement coverage
            ('FR004', 'TC006', 'direct', 'test', 90.0, 'UI functionality directly tested through usability testing')
        ]
        
        # Create traceability link objects
        for req_id, tc_id, coverage_type, verification_method, coverage_percentage, notes in traceability_mappings:
            if req_id in self.requirements or req_id in self.user_stories:
                if tc_id in self.test_cases:
                    link = TraceabilityLink(
                        requirement_id=req_id,
                        test_case_id=tc_id,
                        coverage_type=coverage_type,
                        verification_method=verification_method,
                        coverage_percentage=coverage_percentage,
                        notes=notes
                    )
                    self.traceability_links.append(link)
    
    def add_requirement(self, requirement: Requirement) -> bool:
        """Add new requirement to the matrix."""
        try:
            if requirement.requirement_id in self.requirements:
                self.logger.warning(f"Requirement {requirement.requirement_id} already exists")
                return False
            
            self.requirements[requirement.requirement_id] = requirement
            self.logger.info(f"Added requirement: {requirement.requirement_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding requirement: {e}")
            return False
    
    def add_test_case(self, test_case: TestCase) -> bool:
        """Add new test case to the matrix."""
        try:
            if test_case.test_case_id in self.test_cases:
                self.logger.warning(f"Test case {test_case.test_case_id} already exists")
                return False
                
            self.test_cases[test_case.test_case_id] = test_case
            self.logger.info(f"Added test case: {test_case.test_case_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding test case: {e}")
            return False
    
    def add_user_story(self, user_story: UserStory) -> bool:
        """Add new user story to the matrix."""
        try:
            if user_story.story_id in self.user_stories:
                self.logger.warning(f"User story {user_story.story_id} already exists")
                return False
                
            self.user_stories[user_story.story_id] = user_story
            self.logger.info(f"Added user story: {user_story.story_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding user story: {e}")
            return False
    
    def link_requirement_to_test(self, requirement_id: str, test_case_id: str,
                                coverage_type: str = "direct", 
                                verification_method: str = "test",
                                coverage_percentage: float = 100.0,
                                notes: str = "") -> bool:
        """Create traceability link between requirement and test case."""
        try:
            # Verify requirement exists (could be requirement or user story)
            if requirement_id not in self.requirements and requirement_id not in self.user_stories:
                self.logger.error(f"Requirement/Story {requirement_id} not found")
                return False
            
            # Verify test case exists
            if test_case_id not in self.test_cases:
                self.logger.error(f"Test case {test_case_id} not found")
                return False
            
            # Check if link already exists
            existing_link = next(
                (link for link in self.traceability_links 
                 if link.requirement_id == requirement_id and link.test_case_id == test_case_id),
                None
            )
            
            if existing_link:
                self.logger.warning(f"Traceability link already exists: {requirement_id} -> {test_case_id}")
                return False
            
            # Create new link
            link = TraceabilityLink(
                requirement_id=requirement_id,
                test_case_id=test_case_id,
                coverage_type=coverage_type,
                verification_method=verification_method,
                coverage_percentage=coverage_percentage,
                notes=notes
            )
            
            self.traceability_links.append(link)
            self.logger.info(f"Created traceability link: {requirement_id} -> {test_case_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating traceability link: {e}")
            return False
    
    def update_test_status(self, test_case_id: str, status: TestStatus, 
                          execution_notes: str = "", executed_by: str = "") -> bool:
        """Update test case execution status."""
        try:
            if test_case_id not in self.test_cases:
                self.logger.error(f"Test case {test_case_id} not found")
                return False
            
            test_case = self.test_cases[test_case_id]
            test_case.status = status
            test_case.execution_notes = execution_notes
            test_case.executed_by = executed_by
            test_case.last_executed = datetime.now()
            
            self.logger.info(f"Updated test case {test_case_id} status to {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating test status: {e}")
            return False
    
    def update_requirement_status(self, requirement_id: str, status: RequirementStatus) -> bool:
        """Update requirement implementation status."""
        try:
            if requirement_id not in self.requirements:
                self.logger.error(f"Requirement {requirement_id} not found")
                return False
            
            requirement = self.requirements[requirement_id]
            requirement.status = status
            requirement.updated_date = datetime.now()
            
            self.logger.info(f"Updated requirement {requirement_id} status to {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating requirement status: {e}")
            return False
    
    def generate_coverage_report(self) -> CoverageReport:
        """Generate comprehensive coverage analysis report."""
        try:
            # Calculate basic coverage statistics
            total_requirements = len(self.requirements) + len(self.user_stories)
            covered_requirements = set()
            partially_covered_requirements = set()
            
            # Analyze traceability links
            coverage_details = {}
            for link in self.traceability_links:
                req_id = link.requirement_id
                
                if req_id not in coverage_details:
                    coverage_details[req_id] = {
                        'total_coverage': 0.0,
                        'test_cases': [],
                        'verification_methods': set()
                    }
                
                coverage_details[req_id]['total_coverage'] += link.coverage_percentage
                coverage_details[req_id]['test_cases'].append(link.test_case_id)
                coverage_details[req_id]['verification_methods'].add(link.verification_method)
            
            # Classify requirements by coverage
            for req_id, details in coverage_details.items():
                total_coverage = min(details['total_coverage'], 100.0)  # Cap at 100%
                
                if total_coverage >= 80.0:
                    covered_requirements.add(req_id)
                elif total_coverage >= 30.0:
                    partially_covered_requirements.add(req_id)
            
            # Calculate coverage percentages
            covered_count = len(covered_requirements)
            partially_covered_count = len(partially_covered_requirements)
            uncovered_count = total_requirements - covered_count - partially_covered_count
            
            overall_coverage_percentage = (covered_count / total_requirements * 100) if total_requirements > 0 else 0
            
            # Coverage by requirement type
            coverage_by_type = {}
            for req_type in RequirementType:
                type_requirements = [req for req in self.requirements.values() if req.requirement_type == req_type]
                type_covered = sum(1 for req in type_requirements if req.requirement_id in covered_requirements)
                
                if type_requirements:
                    coverage_by_type[req_type] = (type_covered / len(type_requirements)) * 100
                else:
                    coverage_by_type[req_type] = 0.0
            
            # Coverage by priority
            coverage_by_priority = {}
            for priority in RequirementPriority:
                priority_requirements = [req for req in self.requirements.values() if req.priority == priority]
                priority_covered = sum(1 for req in priority_requirements if req.requirement_id in covered_requirements)
                
                if priority_requirements:
                    coverage_by_priority[priority] = (priority_covered / len(priority_requirements)) * 100
                else:
                    coverage_by_priority[priority] = 0.0
            
            # Identify gaps and generate recommendations
            gaps = self._identify_coverage_gaps(coverage_details)
            recommendations = self._generate_coverage_recommendations(
                overall_coverage_percentage, coverage_by_type, coverage_by_priority, gaps
            )
            
            return CoverageReport(
                total_requirements=total_requirements,
                covered_requirements=covered_count,
                uncovered_requirements=uncovered_count,
                partially_covered_requirements=partially_covered_count,
                coverage_percentage=overall_coverage_percentage,
                coverage_by_type=coverage_by_type,
                coverage_by_priority=coverage_by_priority,
                gaps=gaps,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error generating coverage report: {e}")
            return CoverageReport(
                total_requirements=0,
                covered_requirements=0,
                uncovered_requirements=0,
                partially_covered_requirements=0,
                coverage_percentage=0.0
            )
    
    def _identify_coverage_gaps(self, coverage_details: Dict[str, Any]) -> List[str]:
        """Identify specific coverage gaps."""
        gaps = []
        
        # Find uncovered requirements
        all_requirement_ids = set(self.requirements.keys()) | set(self.user_stories.keys())
        covered_requirement_ids = set(coverage_details.keys())
        uncovered_requirements = all_requirement_ids - covered_requirement_ids
        
        for req_id in uncovered_requirements:
            if req_id in self.requirements:
                req = self.requirements[req_id]
                gaps.append(f"Requirement {req_id} ({req.title}) has no test coverage")
            elif req_id in self.user_stories:
                story = self.user_stories[req_id]
                gaps.append(f"User Story {req_id} ({story.title}) has no test coverage")
        
        # Find critical requirements with insufficient coverage
        for req_id, details in coverage_details.items():
            if details['total_coverage'] < 50.0:
                if req_id in self.requirements:
                    req = self.requirements[req_id]
                    if req.priority == RequirementPriority.CRITICAL:
                        gaps.append(f"Critical requirement {req_id} has insufficient test coverage ({details['total_coverage']:.1f}%)")
        
        # Find missing verification methods
        critical_requirements = [req for req in self.requirements.values() if req.priority == RequirementPriority.CRITICAL]
        for req in critical_requirements:
            if req.requirement_id in coverage_details:
                methods = coverage_details[req.requirement_id]['verification_methods']
                if 'test' not in methods and req.requirement_type == RequirementType.FUNCTIONAL:
                    gaps.append(f"Functional requirement {req.requirement_id} lacks direct testing verification")
        
        return gaps
    
    def _generate_coverage_recommendations(self, 
                                         overall_coverage: float,
                                         coverage_by_type: Dict[RequirementType, float],
                                         coverage_by_priority: Dict[RequirementPriority, float],
                                         gaps: List[str]) -> List[str]:
        """Generate recommendations for improving coverage."""
        recommendations = []
        
        # Overall coverage recommendations
        if overall_coverage < 80.0:
            recommendations.append(
                f"Overall requirement coverage is {overall_coverage:.1f}% (target: >80%). "
                f"Focus on creating test cases for uncovered requirements."
            )
        
        # Critical requirement coverage
        if coverage_by_priority.get(RequirementPriority.CRITICAL, 0) < 95.0:
            recommendations.append(
                f"Critical requirement coverage is {coverage_by_priority.get(RequirementPriority.CRITICAL, 0):.1f}% "
                f"(target: >95%). All critical requirements must have comprehensive test coverage."
            )
        
        # Functional requirement coverage
        if coverage_by_type.get(RequirementType.FUNCTIONAL, 0) < 90.0:
            recommendations.append(
                f"Functional requirement coverage is {coverage_by_type.get(RequirementType.FUNCTIONAL, 0):.1f}% "
                f"(target: >90%). Add automated test cases for functional requirements."
            )
        
        # Security requirement coverage
        if coverage_by_type.get(RequirementType.NON_FUNCTIONAL, 0) < 85.0:
            recommendations.append(
                f"Non-functional requirement coverage is {coverage_by_type.get(RequirementType.NON_FUNCTIONAL, 0):.1f}% "
                f"(target: >85%). Implement performance, security, and scalability tests."
            )
        
        # User story coverage
        user_story_coverage = self._calculate_user_story_coverage()
        if user_story_coverage < 90.0:
            recommendations.append(
                f"User story coverage is {user_story_coverage:.1f}% (target: >90%). "
                f"Create acceptance tests for all user stories."
            )
        
        # Test automation recommendations
        automated_test_count = sum(1 for tc in self.test_cases.values() if tc.automated)
        automation_percentage = (automated_test_count / len(self.test_cases)) * 100 if self.test_cases else 0
        
        if automation_percentage < 70.0:
            recommendations.append(
                f"Test automation is at {automation_percentage:.1f}% (target: >70%). "
                f"Automate more test cases for regression testing efficiency."
            )
        
        # Gap-specific recommendations
        if len(gaps) > 10:
            recommendations.append(
                f"Found {len(gaps)} coverage gaps. Prioritize addressing critical and high-priority requirement gaps first."
            )
        
        return recommendations
    
    def _calculate_user_story_coverage(self) -> float:
        """Calculate user story test coverage percentage."""
        if not self.user_stories:
            return 100.0
        
        covered_stories = set()
        for link in self.traceability_links:
            if link.requirement_id in self.user_stories:
                covered_stories.add(link.requirement_id)
        
        return (len(covered_stories) / len(self.user_stories)) * 100.0
    
    def generate_traceability_matrix_report(self) -> Dict[str, Any]:
        """Generate comprehensive traceability matrix report."""
        try:
            coverage_report = self.generate_coverage_report()
            
            # Requirements summary
            requirements_summary = {
                'total_requirements': len(self.requirements),
                'by_type': {},
                'by_priority': {},
                'by_status': {}
            }
            
            for req_type in RequirementType:
                requirements_summary['by_type'][req_type.value] = len([
                    req for req in self.requirements.values() if req.requirement_type == req_type
                ])
            
            for priority in RequirementPriority:
                requirements_summary['by_priority'][priority.value] = len([
                    req for req in self.requirements.values() if req.priority == priority
                ])
            
            for status in RequirementStatus:
                requirements_summary['by_status'][status.value] = len([
                    req for req in self.requirements.values() if req.status == status
                ])
            
            # Test cases summary
            test_cases_summary = {
                'total_test_cases': len(self.test_cases),
                'by_type': {},
                'by_status': {},
                'automation_rate': 0.0
            }
            
            test_types = set(tc.test_type for tc in self.test_cases.values())
            for test_type in test_types:
                test_cases_summary['by_type'][test_type] = len([
                    tc for tc in self.test_cases.values() if tc.test_type == test_type
                ])
            
            for status in TestStatus:
                test_cases_summary['by_status'][status.value] = len([
                    tc for tc in self.test_cases.values() if tc.status == status
                ])
            
            automated_count = sum(1 for tc in self.test_cases.values() if tc.automated)
            test_cases_summary['automation_rate'] = (
                (automated_count / len(self.test_cases)) * 100 if self.test_cases else 0
            )
            
            # User stories summary
            user_stories_summary = {
                'total_user_stories': len(self.user_stories),
                'by_role': {},
                'by_priority': {},
                'by_epic': {}
            }
            
            user_roles = set(story.user_role for story in self.user_stories.values())
            for role in user_roles:
                user_stories_summary['by_role'][role] = len([
                    story for story in self.user_stories.values() if story.user_role == role
                ])
            
            for priority in RequirementPriority:
                user_stories_summary['by_priority'][priority.value] = len([
                    story for story in self.user_stories.values() if story.priority == priority
                ])
            
            epics = set(story.epic for story in self.user_stories.values() if story.epic)
            for epic in epics:
                user_stories_summary['by_epic'][epic] = len([
                    story for story in self.user_stories.values() if story.epic == epic
                ])
            
            # Traceability links summary
            traceability_summary = {
                'total_links': len(self.traceability_links),
                'by_coverage_type': {},
                'by_verification_method': {},
                'average_coverage_percentage': 0.0
            }
            
            coverage_types = set(link.coverage_type for link in self.traceability_links)
            for coverage_type in coverage_types:
                traceability_summary['by_coverage_type'][coverage_type] = len([
                    link for link in self.traceability_links if link.coverage_type == coverage_type
                ])
            
            verification_methods = set(link.verification_method for link in self.traceability_links)
            for method in verification_methods:
                traceability_summary['by_verification_method'][method] = len([
                    link for link in self.traceability_links if link.verification_method == method
                ])
            
            if self.traceability_links:
                traceability_summary['average_coverage_percentage'] = (
                    sum(link.coverage_percentage for link in self.traceability_links) / 
                    len(self.traceability_links)
                )
            
            return {
                'report_generated': datetime.now().isoformat(),
                'project_name': self.project_name,
                'coverage_analysis': {
                    'overall_coverage_percentage': coverage_report.coverage_percentage,
                    'covered_requirements': coverage_report.covered_requirements,
                    'uncovered_requirements': coverage_report.uncovered_requirements,
                    'partially_covered_requirements': coverage_report.partially_covered_requirements,
                    'coverage_by_type': {k.value: v for k, v in coverage_report.coverage_by_type.items()},
                    'coverage_by_priority': {k.value: v for k, v in coverage_report.coverage_by_priority.items()}
                },
                'requirements_summary': requirements_summary,
                'test_cases_summary': test_cases_summary,
                'user_stories_summary': user_stories_summary,
                'traceability_summary': traceability_summary,
                'coverage_gaps': coverage_report.gaps,
                'recommendations': coverage_report.recommendations,
                'detailed_requirements': [
                    {
                        'requirement_id': req.requirement_id,
                        'title': req.title,
                        'type': req.requirement_type.value,
                        'priority': req.priority.value,
                        'status': req.status.value,
                        'source': req.source,
                        'business_value': req.business_value,
                        'test_coverage': self._get_requirement_coverage(req.requirement_id)
                    }
                    for req in self.requirements.values()
                ],
                'detailed_user_stories': [
                    {
                        'story_id': story.story_id,
                        'title': story.title,
                        'user_role': story.user_role,
                        'priority': story.priority.value,
                        'status': story.status.value,
                        'epic': story.epic,
                        'test_coverage': self._get_requirement_coverage(story.story_id)
                    }
                    for story in self.user_stories.values()
                ],
                'detailed_test_cases': [
                    {
                        'test_case_id': tc.test_case_id,
                        'title': tc.title,
                        'test_type': tc.test_type,
                        'automated': tc.automated,
                        'status': tc.status.value,
                        'priority': tc.priority.value,
                        'covers_requirements': self._get_test_case_requirements(tc.test_case_id)
                    }
                    for tc in self.test_cases.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating traceability matrix report: {e}")
            return {'error': str(e), 'report_generated': datetime.now().isoformat()}
    
    def _get_requirement_coverage(self, requirement_id: str) -> Dict[str, Any]:
        """Get coverage details for a specific requirement."""
        coverage_info = {
            'total_coverage_percentage': 0.0,
            'test_cases': [],
            'verification_methods': [],
            'coverage_type': 'none'
        }
        
        for link in self.traceability_links:
            if link.requirement_id == requirement_id:
                coverage_info['total_coverage_percentage'] += link.coverage_percentage
                coverage_info['test_cases'].append(link.test_case_id)
                if link.verification_method not in coverage_info['verification_methods']:
                    coverage_info['verification_methods'].append(link.verification_method)
                coverage_info['coverage_type'] = link.coverage_type
        
        # Cap coverage at 100%
        coverage_info['total_coverage_percentage'] = min(coverage_info['total_coverage_percentage'], 100.0)
        
        return coverage_info
    
    def _get_test_case_requirements(self, test_case_id: str) -> List[str]:
        """Get requirements covered by a specific test case."""
        return [link.requirement_id for link in self.traceability_links if link.test_case_id == test_case_id]
    
    def export_to_json(self, filepath: str) -> bool:
        """Export traceability matrix to JSON file."""
        try:
            report = self.generate_traceability_matrix_report()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Traceability matrix exported to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting traceability matrix: {e}")
            return False
    
    def import_requirements_from_json(self, filepath: str) -> bool:
        """Import requirements from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Import requirements
            if 'requirements' in data:
                for req_data in data['requirements']:
                    requirement = Requirement(
                        requirement_id=req_data['requirement_id'],
                        title=req_data['title'],
                        description=req_data['description'],
                        requirement_type=RequirementType(req_data['requirement_type']),
                        priority=RequirementPriority(req_data['priority']),
                        status=RequirementStatus(req_data['status']),
                        source=req_data['source'],
                        rationale=req_data['rationale'],
                        acceptance_criteria=req_data.get('acceptance_criteria', []),
                        related_requirements=req_data.get('related_requirements', [])
                    )
                    self.add_requirement(requirement)
            
            # Import test cases
            if 'test_cases' in data:
                for tc_data in data['test_cases']:
                    test_case = TestCase(
                        test_case_id=tc_data['test_case_id'],
                        title=tc_data['title'],
                        description=tc_data['description'],
                        test_type=tc_data['test_type'],
                        preconditions=tc_data.get('preconditions', []),
                        test_steps=tc_data.get('test_steps', []),
                        expected_results=tc_data.get('expected_results', []),
                        automated=tc_data.get('automated', False),
                        priority=RequirementPriority(tc_data.get('priority', 'medium')),
                        status=TestStatus(tc_data.get('status', 'not_tested'))
                    )
                    self.add_test_case(test_case)
            
            self.logger.info(f"Successfully imported requirements from: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing requirements: {e}")
            return False


# Convenience functions for common operations
def create_manufacturing_traceability_matrix() -> RequirementsTraceabilityMatrix:
    """Create pre-configured traceability matrix for manufacturing system."""
    return RequirementsTraceabilityMatrix("Manufacturing Line Control System")

def generate_coverage_analysis(matrix: RequirementsTraceabilityMatrix) -> CoverageReport:
    """Generate coverage analysis report."""
    return matrix.generate_coverage_report()

def export_traceability_report(matrix: RequirementsTraceabilityMatrix, filepath: str) -> bool:
    """Export comprehensive traceability report."""
    return matrix.export_to_json(filepath)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Requirements Traceability Matrix Demo")
    print("=" * 80)
    
    # Create traceability matrix
    rtm = RequirementsTraceabilityMatrix()
    
    print(f"\nProject: {rtm.project_name}")
    print(f"Total Requirements: {len(rtm.requirements)}")
    print(f"Total User Stories: {len(rtm.user_stories)}")
    print(f"Total Test Cases: {len(rtm.test_cases)}")
    print(f"Total Traceability Links: {len(rtm.traceability_links)}")
    
    # Generate coverage report
    print("\n" + "="*80)
    print("COVERAGE ANALYSIS")
    print("="*80)
    
    coverage_report = rtm.generate_coverage_report()
    
    print(f"Overall Coverage: {coverage_report.coverage_percentage:.1f}%")
    print(f"Covered Requirements: {coverage_report.covered_requirements}")
    print(f"Uncovered Requirements: {coverage_report.uncovered_requirements}")
    print(f"Partially Covered: {coverage_report.partially_covered_requirements}")
    
    print(f"\nCoverage by Type:")
    for req_type, coverage in coverage_report.coverage_by_type.items():
        print(f"  {req_type.value}: {coverage:.1f}%")
    
    print(f"\nCoverage by Priority:")
    for priority, coverage in coverage_report.coverage_by_priority.items():
        print(f"  {priority.value}: {coverage:.1f}%")
    
    if coverage_report.gaps:
        print(f"\nCoverage Gaps ({len(coverage_report.gaps)}):")
        for i, gap in enumerate(coverage_report.gaps[:5], 1):  # Show first 5 gaps
            print(f"  {i}. {gap}")
        if len(coverage_report.gaps) > 5:
            print(f"  ... and {len(coverage_report.gaps) - 5} more gaps")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(coverage_report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Generate full traceability report
    print(f"\n" + "="*80)
    print("GENERATING TRACEABILITY MATRIX REPORT")
    print("="*80)
    
    report = rtm.generate_traceability_matrix_report()
    
    print(f"Requirements Summary:")
    req_summary = report['requirements_summary']
    print(f"  Total: {req_summary['total_requirements']}")
    print(f"  By Priority: {req_summary['by_priority']}")
    
    print(f"\nTest Cases Summary:")
    tc_summary = report['test_cases_summary']
    print(f"  Total: {tc_summary['total_test_cases']}")
    print(f"  Automation Rate: {tc_summary['automation_rate']:.1f}%")
    print(f"  By Type: {tc_summary['by_type']}")
    
    print(f"\nUser Stories Summary:")
    us_summary = report['user_stories_summary']
    print(f"  Total: {us_summary['total_user_stories']}")
    print(f"  By Role: {us_summary['by_role']}")
    
    # Export to file (demo)
    try:
        export_filename = "traceability_matrix_demo.json"
        success = rtm.export_to_json(export_filename)
        if success:
            print(f"\n✅ Traceability matrix exported to: {export_filename}")
        else:
            print(f"\n❌ Failed to export traceability matrix")
    except Exception as e:
        print(f"\n⚠️  Export demo skipped: {e}")
    
    print(f"\nRequirements Traceability Matrix demo completed!")