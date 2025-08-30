"""
Production Deployment - Week 16: Production Deployment & Documentation

This layer provides comprehensive production deployment capabilities, documentation systems,
training infrastructure, and ongoing support procedures to ensure successful system 
operation and user adoption in manufacturing environments.

Production Deployment Features:
- Zero-downtime blue-green deployment system
- Production infrastructure with auto-scaling and monitoring
- Comprehensive user documentation for all roles
- Interactive training system with assessments
- Multi-tier support infrastructure with knowledge management
- API documentation with interactive examples
- Maintenance procedures and operational runbooks

Deployment Targets:
- 100% zero-downtime deployment success rate
- >99.9% system availability post-deployment
- 100% documentation coverage of system functionality
- >90% user training competency achievement
- >95% support issue resolution without escalation

Author: Manufacturing Line Control System
Created: Week 16 - Production Deployment & Documentation Phase
"""

from .infrastructure_setup import ProductionInfrastructure
from .blue_green_deployment import BlueGreenDeploymentSystem
from .monitoring_system import ProductionMonitoringSystem
from .user_documentation import UserDocumentationSystem
from .api_documentation import APIDocumentationSystem
from .training_system import InteractiveTrainingSystem
from .support_system import SupportTicketSystem
from .maintenance_procedures import MaintenanceProcedureManager

__all__ = [
    'ProductionInfrastructure',
    'BlueGreenDeploymentSystem', 
    'ProductionMonitoringSystem',
    'UserDocumentationSystem',
    'APIDocumentationSystem',
    'InteractiveTrainingSystem',
    'SupportTicketSystem',
    'MaintenanceProcedureManager'
]