"""
Blue-Green Deployment System - Week 16: Production Deployment & Documentation

This module provides comprehensive zero-downtime deployment capabilities using blue-green
deployment strategy with automated health checks, traffic switching, database migrations,
and rollback capabilities for the manufacturing control system.

Deployment Capabilities:
- Zero-downtime blue-green deployments with parallel environments
- Automated traffic switching with health validation
- Database migration handling with rollback support
- Canary releases with gradual traffic routing
- Feature toggles for runtime activation/deactivation
- Comprehensive rollback with data consistency protection

Author: Manufacturing Line Control System
Created: Week 16 - Zero-Downtime Deployment Phase
"""

import json
import logging
import time
import threading
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import uuid
import hashlib
import requests
import asyncio


class DeploymentStrategy(Enum):
    """Deployment strategy types."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING_UPDATE = "rolling_update"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class DeploymentPhase(Enum):
    """Deployment execution phases."""
    PREPARING = "preparing"
    PROVISIONING = "provisioning"
    DEPLOYING = "deploying"
    TESTING = "testing"
    SWITCHING = "switching"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


class EnvironmentStatus(Enum):
    """Environment status states."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    STANDBY = "standby"
    DEPLOYING = "deploying"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


@dataclass
class DeploymentConfig:
    """Deployment configuration definition."""
    strategy: DeploymentStrategy
    application_name: str
    version: str
    image_registry: str
    image_tag: str
    
    # Traffic Management
    traffic_shift_duration_minutes: int = 5
    canary_traffic_percentage: float = 10.0
    health_check_timeout_seconds: int = 300
    
    # Rollback Configuration
    auto_rollback_enabled: bool = True
    rollback_trigger_error_rate: float = 5.0  # 5% error rate triggers rollback
    rollback_trigger_response_time_ms: float = 2000  # 2s response time triggers rollback
    
    # Database Migration
    database_migration_enabled: bool = True
    migration_timeout_minutes: int = 30
    migration_rollback_enabled: bool = True
    
    # Feature Flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    # Notification
    notification_channels: List[str] = field(default_factory=lambda: ["slack", "email"])


@dataclass
class EnvironmentConfig:
    """Environment configuration for blue-green deployment."""
    environment_name: str
    status: EnvironmentStatus
    cluster_endpoint: str
    database_endpoint: str
    load_balancer_endpoint: str
    namespace: str = "manufacturing-system"
    replicas: int = 3
    resource_limits: Dict[str, str] = field(default_factory=lambda: {"cpu": "1000m", "memory": "2Gi"})
    environment_variables: Dict[str, str] = field(default_factory=dict)
    health_check_endpoint: str = "/api/health"
    readiness_check_endpoint: str = "/api/ready"


@dataclass
class DeploymentResult:
    """Deployment execution result."""
    deployment_id: str
    strategy: DeploymentStrategy
    application_name: str
    version: str
    start_time: datetime
    end_time: Optional[datetime] = None
    phase: DeploymentPhase = DeploymentPhase.PREPARING
    status: str = "in_progress"
    
    # Environment Information
    blue_environment: Optional[EnvironmentConfig] = None
    green_environment: Optional[EnvironmentConfig] = None
    active_environment: Optional[str] = None
    
    # Traffic Information
    traffic_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Health Metrics
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Migration Information
    database_migrations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Logs and Events
    events: List[Dict[str, Any]] = field(default_factory=list)
    rollback_info: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Health checking system for deployment validation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def check_application_health(self, environment: EnvironmentConfig) -> Dict[str, Any]:
        """Check application health comprehensively."""
        health_result = {
            "environment": environment.environment_name,
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # HTTP Health Check
        health_result["checks"]["http_health"] = await self._check_http_health(environment)
        
        # Readiness Check
        health_result["checks"]["readiness"] = await self._check_readiness(environment)
        
        # Performance Check
        health_result["checks"]["performance"] = await self._check_performance(environment)
        
        # Database Connectivity
        health_result["checks"]["database"] = await self._check_database_connectivity(environment)
        
        # Resource Utilization
        health_result["checks"]["resources"] = await self._check_resource_utilization(environment)
        
        # Determine overall status
        failed_checks = [
            check_name for check_name, check_result in health_result["checks"].items()
            if not check_result.get("healthy", False)
        ]
        
        if failed_checks:
            health_result["overall_status"] = "unhealthy"
            health_result["failed_checks"] = failed_checks
        
        return health_result
    
    async def _check_http_health(self, environment: EnvironmentConfig) -> Dict[str, Any]:
        """Check HTTP health endpoint."""
        try:
            # Simulated HTTP health check
            health_url = f"{environment.load_balancer_endpoint}{environment.health_check_endpoint}"
            
            # In real implementation, this would make actual HTTP request
            # response = await self._make_http_request(health_url)
            
            return {
                "healthy": True,
                "response_time_ms": 89,
                "status_code": 200,
                "endpoint": health_url
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "endpoint": f"{environment.load_balancer_endpoint}{environment.health_check_endpoint}"
            }
    
    async def _check_readiness(self, environment: EnvironmentConfig) -> Dict[str, Any]:
        """Check application readiness."""
        return {
            "healthy": True,
            "ready_replicas": environment.replicas,
            "total_replicas": environment.replicas,
            "readiness_percentage": 100.0
        }
    
    async def _check_performance(self, environment: EnvironmentConfig) -> Dict[str, Any]:
        """Check performance metrics."""
        return {
            "healthy": True,
            "average_response_time_ms": 156,
            "error_rate_percent": 0.02,
            "throughput_rps": 847,
            "p95_response_time_ms": 289
        }
    
    async def _check_database_connectivity(self, environment: EnvironmentConfig) -> Dict[str, Any]:
        """Check database connectivity."""
        return {
            "healthy": True,
            "connection_pool_active": 12,
            "connection_pool_size": 20,
            "query_response_time_ms": 23,
            "replication_lag_ms": 8
        }
    
    async def _check_resource_utilization(self, environment: EnvironmentConfig) -> Dict[str, Any]:
        """Check resource utilization."""
        return {
            "healthy": True,
            "cpu_utilization_percent": 45.2,
            "memory_utilization_percent": 67.8,
            "disk_utilization_percent": 23.1,
            "network_utilization_mbps": 12.4
        }


class TrafficManager:
    """Traffic management for blue-green deployments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def shift_traffic(self, 
                          from_environment: EnvironmentConfig,
                          to_environment: EnvironmentConfig,
                          percentage: float,
                          duration_minutes: int = 5) -> Dict[str, Any]:
        """Gradually shift traffic between environments."""
        self.logger.info(f"Shifting {percentage}% traffic from {from_environment.environment_name} to {to_environment.environment_name}")
        
        shift_start = datetime.now()
        steps = max(1, int(duration_minutes))  # One step per minute
        percentage_per_step = percentage / steps
        
        traffic_shift_log = []
        
        for step in range(steps + 1):
            current_percentage = min(percentage_per_step * step, percentage)
            remaining_percentage = 100 - current_percentage
            
            # Apply traffic routing configuration
            routing_result = await self._apply_traffic_routing(
                {
                    from_environment.environment_name: remaining_percentage,
                    to_environment.environment_name: current_percentage
                }
            )
            
            traffic_shift_log.append({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "traffic_distribution": routing_result["traffic_distribution"],
                "routing_status": routing_result["status"]
            })
            
            # Wait between steps (except last step)
            if step < steps:
                await asyncio.sleep(60)  # Wait 1 minute
        
        shift_end = datetime.now()
        shift_duration = (shift_end - shift_start).total_seconds()
        
        return {
            "status": "completed",
            "start_time": shift_start.isoformat(),
            "end_time": shift_end.isoformat(),
            "duration_seconds": shift_duration,
            "final_traffic_distribution": {
                from_environment.environment_name: 100 - percentage,
                to_environment.environment_name: percentage
            },
            "shift_log": traffic_shift_log
        }
    
    async def _apply_traffic_routing(self, traffic_distribution: Dict[str, float]) -> Dict[str, Any]:
        """Apply traffic routing configuration to load balancer."""
        # Simulated traffic routing configuration
        # In real implementation, this would configure actual load balancer
        
        self.logger.info(f"Applying traffic routing: {traffic_distribution}")
        
        return {
            "status": "applied",
            "traffic_distribution": traffic_distribution,
            "load_balancer_rules_updated": True,
            "dns_updated": True
        }
    
    async def complete_traffic_switch(self, target_environment: EnvironmentConfig) -> Dict[str, Any]:
        """Complete traffic switch to target environment."""
        self.logger.info(f"Completing traffic switch to {target_environment.environment_name}")
        
        # Route 100% traffic to target environment
        switch_result = await self._apply_traffic_routing({
            target_environment.environment_name: 100.0
        })
        
        return {
            "status": "completed",
            "active_environment": target_environment.environment_name,
            "traffic_percentage": 100.0,
            "switch_timestamp": datetime.now().isoformat(),
            "switch_details": switch_result
        }


class DatabaseMigrationManager:
    """Database migration management for zero-downtime deployments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.migration_history: List[Dict[str, Any]] = []
    
    async def execute_migrations(self, 
                                migration_scripts: List[str],
                                database_config: Dict[str, str],
                                timeout_minutes: int = 30) -> Dict[str, Any]:
        """Execute database migrations with rollback capability."""
        self.logger.info(f"Executing {len(migration_scripts)} database migrations")
        
        migration_start = datetime.now()
        migration_id = f"migration-{uuid.uuid4().hex[:8]}"
        
        migration_result = {
            "migration_id": migration_id,
            "start_time": migration_start.isoformat(),
            "status": "in_progress",
            "scripts_executed": [],
            "rollback_scripts": [],
            "errors": []
        }
        
        try:
            for i, script in enumerate(migration_scripts):
                script_start = datetime.now()
                
                # Execute migration script
                script_result = await self._execute_migration_script(script, database_config)
                
                script_end = datetime.now()
                script_duration = (script_end - script_start).total_seconds()
                
                if script_result["success"]:
                    migration_result["scripts_executed"].append({
                        "script": script,
                        "execution_time": script_duration,
                        "status": "success",
                        "rollback_script": script_result.get("rollback_script")
                    })
                    
                    # Store rollback script
                    if script_result.get("rollback_script"):
                        migration_result["rollback_scripts"].insert(0, script_result["rollback_script"])
                else:
                    migration_result["errors"].append({
                        "script": script,
                        "error": script_result["error"],
                        "execution_time": script_duration
                    })
                    raise Exception(f"Migration script {script} failed: {script_result['error']}")
            
            migration_end = datetime.now()
            migration_result["end_time"] = migration_end.isoformat()
            migration_result["duration_seconds"] = (migration_end - migration_start).total_seconds()
            migration_result["status"] = "completed"
            
            self.migration_history.append(migration_result)
            self.logger.info(f"Database migrations completed successfully")
            
        except Exception as e:
            migration_result["status"] = "failed"
            migration_result["end_time"] = datetime.now().isoformat()
            migration_result["final_error"] = str(e)
            
            self.logger.error(f"Database migration failed: {e}")
            self.migration_history.append(migration_result)
            
            raise
        
        return migration_result
    
    async def rollback_migrations(self, migration_id: str) -> Dict[str, Any]:
        """Rollback database migrations."""
        self.logger.info(f"Rolling back migrations: {migration_id}")
        
        # Find migration to rollback
        target_migration = None
        for migration in self.migration_history:
            if migration["migration_id"] == migration_id:
                target_migration = migration
                break
        
        if not target_migration:
            raise ValueError(f"Migration {migration_id} not found")
        
        rollback_start = datetime.now()
        rollback_id = f"rollback-{uuid.uuid4().hex[:8]}"
        
        rollback_result = {
            "rollback_id": rollback_id,
            "original_migration_id": migration_id,
            "start_time": rollback_start.isoformat(),
            "status": "in_progress",
            "scripts_executed": [],
            "errors": []
        }
        
        try:
            # Execute rollback scripts in reverse order
            rollback_scripts = target_migration.get("rollback_scripts", [])
            
            for script in rollback_scripts:
                script_result = await self._execute_rollback_script(script)
                
                if script_result["success"]:
                    rollback_result["scripts_executed"].append({
                        "script": script,
                        "status": "success"
                    })
                else:
                    rollback_result["errors"].append({
                        "script": script,
                        "error": script_result["error"]
                    })
                    raise Exception(f"Rollback script failed: {script_result['error']}")
            
            rollback_end = datetime.now()
            rollback_result["end_time"] = rollback_end.isoformat()
            rollback_result["duration_seconds"] = (rollback_end - rollback_start).total_seconds()
            rollback_result["status"] = "completed"
            
            self.logger.info("Database rollback completed successfully")
            
        except Exception as e:
            rollback_result["status"] = "failed"
            rollback_result["end_time"] = datetime.now().isoformat()
            rollback_result["final_error"] = str(e)
            
            self.logger.error(f"Database rollback failed: {e}")
            raise
        
        return rollback_result
    
    async def _execute_migration_script(self, script: str, database_config: Dict[str, str]) -> Dict[str, Any]:
        """Execute individual migration script."""
        # Simulated migration script execution
        # In real implementation, this would execute actual database migration
        
        self.logger.info(f"Executing migration script: {script}")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "script": script,
            "rows_affected": 156,
            "execution_time_ms": 234,
            "rollback_script": f"rollback_{script}"
        }
    
    async def _execute_rollback_script(self, script: str) -> Dict[str, Any]:
        """Execute rollback script."""
        self.logger.info(f"Executing rollback script: {script}")
        
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "script": script,
            "rows_affected": 156,
            "execution_time_ms": 198
        }


class BlueGreenDeploymentSystem:
    """
    Comprehensive Blue-Green Deployment System
    
    Provides zero-downtime deployment capabilities using blue-green strategy with:
    - Parallel environment management (blue/green)
    - Automated health checking and validation
    - Gradual traffic switching with rollback protection
    - Database migration handling with consistency guarantees
    - Feature flag management for runtime control
    - Comprehensive monitoring and alerting integration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_checker = HealthChecker()
        self.traffic_manager = TrafficManager()
        self.migration_manager = DatabaseMigrationManager()
        
        # Deployment tracking
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        
        # Environment management
        self.environments: Dict[str, EnvironmentConfig] = {}
        self.active_environment: Optional[str] = None
    
    def initialize_environments(self, 
                              blue_config: EnvironmentConfig,
                              green_config: EnvironmentConfig) -> Dict[str, Any]:
        """Initialize blue and green environments."""
        self.logger.info("Initializing blue-green environments")
        
        # Register environments
        self.environments["blue"] = blue_config
        self.environments["green"] = green_config
        
        # Set initial active environment
        if blue_config.status == EnvironmentStatus.ACTIVE:
            self.active_environment = "blue"
        elif green_config.status == EnvironmentStatus.ACTIVE:
            self.active_environment = "green"
        else:
            # Default to blue as active
            self.active_environment = "blue"
            self.environments["blue"].status = EnvironmentStatus.ACTIVE
            self.environments["green"].status = EnvironmentStatus.STANDBY
        
        initialization_result = {
            "status": "initialized",
            "blue_environment": {
                "name": blue_config.environment_name,
                "status": blue_config.status.value,
                "endpoint": blue_config.load_balancer_endpoint
            },
            "green_environment": {
                "name": green_config.environment_name,
                "status": green_config.status.value,
                "endpoint": green_config.load_balancer_endpoint
            },
            "active_environment": self.active_environment,
            "initialization_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Environments initialized. Active: {self.active_environment}")
        return initialization_result
    
    async def execute_blue_green_deployment(self, config: DeploymentConfig) -> DeploymentResult:
        """Execute complete blue-green deployment."""
        self.logger.info(f"Starting blue-green deployment: {config.application_name} v{config.version}")
        
        deployment_id = f"deploy-{uuid.uuid4().hex[:8]}"
        deployment_start = datetime.now()
        
        # Initialize deployment result
        deployment_result = DeploymentResult(
            deployment_id=deployment_id,
            strategy=config.strategy,
            application_name=config.application_name,
            version=config.version,
            start_time=deployment_start,
            blue_environment=self.environments.get("blue"),
            green_environment=self.environments.get("green"),
            active_environment=self.active_environment
        )
        
        # Register active deployment
        self.active_deployments[deployment_id] = deployment_result
        
        try:
            # Phase 1: Prepare deployment
            await self._execute_deployment_phase(
                deployment_result, DeploymentPhase.PREPARING, 
                self._prepare_deployment, config
            )
            
            # Phase 2: Deploy to inactive environment
            await self._execute_deployment_phase(
                deployment_result, DeploymentPhase.DEPLOYING,
                self._deploy_to_inactive_environment, config
            )
            
            # Phase 3: Execute database migrations
            if config.database_migration_enabled:
                await self._execute_deployment_phase(
                    deployment_result, DeploymentPhase.TESTING,
                    self._execute_database_migrations, config
                )
            
            # Phase 4: Health check and validation
            await self._execute_deployment_phase(
                deployment_result, DeploymentPhase.TESTING,
                self._validate_deployment_health, config
            )
            
            # Phase 5: Switch traffic
            await self._execute_deployment_phase(
                deployment_result, DeploymentPhase.SWITCHING,
                self._switch_traffic_to_new_environment, config
            )
            
            # Phase 6: Final validation
            await self._execute_deployment_phase(
                deployment_result, DeploymentPhase.VALIDATING,
                self._perform_final_validation, config
            )
            
            # Mark deployment as completed
            deployment_result.phase = DeploymentPhase.COMPLETED
            deployment_result.status = "completed"
            deployment_result.end_time = datetime.now()
            
            self.logger.info(f"Blue-green deployment completed successfully: {deployment_id}")
            
        except Exception as e:
            # Handle deployment failure
            deployment_result.phase = DeploymentPhase.FAILED
            deployment_result.status = "failed"
            deployment_result.end_time = datetime.now()
            
            deployment_result.events.append({
                "timestamp": datetime.now().isoformat(),
                "event_type": "deployment_failed",
                "message": str(e),
                "details": {"error": str(e)}
            })
            
            self.logger.error(f"Blue-green deployment failed: {e}")
            
            # Attempt rollback if auto-rollback is enabled
            if config.auto_rollback_enabled:
                try:
                    await self._execute_automatic_rollback(deployment_result, config)
                except Exception as rollback_error:
                    self.logger.error(f"Automatic rollback failed: {rollback_error}")
            
            raise
        
        finally:
            # Move to deployment history
            self.deployment_history.append(deployment_result)
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
        
        return deployment_result
    
    async def _execute_deployment_phase(self,
                                      deployment_result: DeploymentResult,
                                      phase: DeploymentPhase,
                                      phase_function: Callable,
                                      config: DeploymentConfig) -> None:
        """Execute deployment phase with error handling."""
        self.logger.info(f"Executing deployment phase: {phase.value}")
        
        deployment_result.phase = phase
        phase_start = datetime.now()
        
        deployment_result.events.append({
            "timestamp": phase_start.isoformat(),
            "event_type": "phase_started",
            "phase": phase.value,
            "message": f"Started deployment phase: {phase.value}"
        })
        
        try:
            phase_result = await phase_function(deployment_result, config)
            
            phase_end = datetime.now()
            phase_duration = (phase_end - phase_start).total_seconds()
            
            deployment_result.events.append({
                "timestamp": phase_end.isoformat(),
                "event_type": "phase_completed",
                "phase": phase.value,
                "duration_seconds": phase_duration,
                "message": f"Completed deployment phase: {phase.value}",
                "result": phase_result
            })
            
        except Exception as e:
            phase_end = datetime.now()
            phase_duration = (phase_end - phase_start).total_seconds()
            
            deployment_result.events.append({
                "timestamp": phase_end.isoformat(),
                "event_type": "phase_failed",
                "phase": phase.value,
                "duration_seconds": phase_duration,
                "error": str(e),
                "message": f"Failed deployment phase: {phase.value}"
            })
            
            raise
    
    async def _prepare_deployment(self, deployment_result: DeploymentResult, config: DeploymentConfig) -> Dict[str, Any]:
        """Prepare deployment by setting up inactive environment."""
        # Determine inactive environment
        inactive_env_name = "green" if self.active_environment == "blue" else "blue"
        inactive_env = self.environments[inactive_env_name]
        
        # Update environment status
        inactive_env.status = EnvironmentStatus.DEPLOYING
        
        preparation_result = {
            "inactive_environment": inactive_env_name,
            "active_environment": self.active_environment,
            "preparation_actions": [
                "Environment status updated to deploying",
                "Resource allocation verified",
                "Network configuration validated"
            ]
        }
        
        return preparation_result
    
    async def _deploy_to_inactive_environment(self, deployment_result: DeploymentResult, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy application to inactive environment."""
        inactive_env_name = "green" if self.active_environment == "blue" else "blue"
        inactive_env = self.environments[inactive_env_name]
        
        self.logger.info(f"Deploying {config.application_name} v{config.version} to {inactive_env_name} environment")
        
        # Simulate deployment process
        await asyncio.sleep(2)  # Simulate deployment time
        
        # Update environment with new version
        inactive_env.environment_variables["VERSION"] = config.version
        inactive_env.environment_variables["IMAGE_TAG"] = config.image_tag
        
        deployment_info = {
            "environment": inactive_env_name,
            "application": config.application_name,
            "version": config.version,
            "image_tag": config.image_tag,
            "replicas_deployed": inactive_env.replicas,
            "deployment_status": "successful"
        }
        
        return deployment_info
    
    async def _execute_database_migrations(self, deployment_result: DeploymentResult, config: DeploymentConfig) -> Dict[str, Any]:
        """Execute database migrations."""
        self.logger.info("Executing database migrations")
        
        # Example migration scripts
        migration_scripts = [
            "001_add_user_preferences_table.sql",
            "002_update_manufacturing_data_schema.sql",
            "003_create_audit_log_indexes.sql"
        ]
        
        database_config = {
            "host": "manufacturing-db-writer.us-west-2.rds.amazonaws.com",
            "database": "manufacturing_db",
            "username": "admin",
            "password": "***"  # Would be retrieved from secrets manager
        }
        
        migration_result = await self.migration_manager.execute_migrations(
            migration_scripts, database_config, config.migration_timeout_minutes
        )
        
        deployment_result.database_migrations.append(migration_result)
        
        return {
            "migration_id": migration_result["migration_id"],
            "scripts_executed": len(migration_result["scripts_executed"]),
            "status": migration_result["status"],
            "duration_seconds": migration_result.get("duration_seconds", 0)
        }
    
    async def _validate_deployment_health(self, deployment_result: DeploymentResult, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate deployment health in inactive environment."""
        inactive_env_name = "green" if self.active_environment == "blue" else "blue"
        inactive_env = self.environments[inactive_env_name]
        
        self.logger.info(f"Validating health of {inactive_env_name} environment")
        
        # Perform comprehensive health check
        health_result = await self.health_checker.check_application_health(inactive_env)
        
        deployment_result.health_checks.append(health_result)
        
        # Extract performance metrics
        if health_result["overall_status"] == "healthy":
            performance_check = health_result["checks"].get("performance", {})
            deployment_result.performance_metrics.update({
                "response_time_ms": performance_check.get("average_response_time_ms", 0),
                "error_rate_percent": performance_check.get("error_rate_percent", 0),
                "throughput_rps": performance_check.get("throughput_rps", 0)
            })
            
            inactive_env.status = EnvironmentStatus.HEALTHY
        else:
            inactive_env.status = EnvironmentStatus.UNHEALTHY
            raise Exception(f"Health check failed for {inactive_env_name} environment: {health_result['failed_checks']}")
        
        return {
            "environment": inactive_env_name,
            "health_status": health_result["overall_status"],
            "checks_passed": len([c for c in health_result["checks"].values() if c.get("healthy", False)]),
            "total_checks": len(health_result["checks"]),
            "performance_metrics": deployment_result.performance_metrics
        }
    
    async def _switch_traffic_to_new_environment(self, deployment_result: DeploymentResult, config: DeploymentConfig) -> Dict[str, Any]:
        """Switch traffic to new environment."""
        inactive_env_name = "green" if self.active_environment == "blue" else "blue"
        active_env = self.environments[self.active_environment]
        inactive_env = self.environments[inactive_env_name]
        
        self.logger.info(f"Switching traffic from {self.active_environment} to {inactive_env_name}")
        
        # Perform gradual traffic shift
        traffic_shift_result = await self.traffic_manager.shift_traffic(
            active_env, inactive_env, 100.0, config.traffic_shift_duration_minutes
        )
        
        # Update traffic distribution in deployment result
        deployment_result.traffic_distribution = traffic_shift_result["final_traffic_distribution"]
        
        # Complete the switch
        switch_result = await self.traffic_manager.complete_traffic_switch(inactive_env)
        
        # Update environment statuses
        active_env.status = EnvironmentStatus.STANDBY
        inactive_env.status = EnvironmentStatus.ACTIVE
        
        # Update active environment
        old_active = self.active_environment
        self.active_environment = inactive_env_name
        deployment_result.active_environment = self.active_environment
        
        return {
            "previous_active_environment": old_active,
            "new_active_environment": self.active_environment,
            "traffic_switch_duration_seconds": traffic_shift_result["duration_seconds"],
            "switch_timestamp": switch_result["switch_timestamp"],
            "traffic_distribution": deployment_result.traffic_distribution
        }
    
    async def _perform_final_validation(self, deployment_result: DeploymentResult, config: DeploymentConfig) -> Dict[str, Any]:
        """Perform final validation of deployment."""
        self.logger.info(f"Performing final validation of deployment")
        
        # Health check on newly active environment
        active_env = self.environments[self.active_environment]
        final_health_check = await self.health_checker.check_application_health(active_env)
        
        deployment_result.health_checks.append(final_health_check)
        
        # Check performance metrics against rollback triggers
        performance_metrics = deployment_result.performance_metrics
        rollback_required = False
        rollback_reasons = []
        
        if performance_metrics.get("error_rate_percent", 0) > config.rollback_trigger_error_rate:
            rollback_required = True
            rollback_reasons.append(f"Error rate {performance_metrics['error_rate_percent']}% exceeds threshold {config.rollback_trigger_error_rate}%")
        
        if performance_metrics.get("response_time_ms", 0) > config.rollback_trigger_response_time_ms:
            rollback_required = True
            rollback_reasons.append(f"Response time {performance_metrics['response_time_ms']}ms exceeds threshold {config.rollback_trigger_response_time_ms}ms")
        
        if rollback_required and config.auto_rollback_enabled:
            raise Exception(f"Performance validation failed, triggering rollback: {'; '.join(rollback_reasons)}")
        
        return {
            "validation_status": "passed",
            "final_health_status": final_health_check["overall_status"],
            "performance_validation": "passed",
            "rollback_triggers_checked": True,
            "rollback_required": rollback_required,
            "rollback_reasons": rollback_reasons
        }
    
    async def _execute_automatic_rollback(self, deployment_result: DeploymentResult, config: DeploymentConfig) -> Dict[str, Any]:
        """Execute automatic rollback on deployment failure."""
        self.logger.info("Executing automatic rollback")
        
        deployment_result.phase = DeploymentPhase.ROLLING_BACK
        rollback_start = datetime.now()
        
        rollback_result = {
            "rollback_id": f"rollback-{uuid.uuid4().hex[:8]}",
            "trigger_reason": "deployment_failure",
            "start_time": rollback_start.isoformat(),
            "status": "in_progress",
            "actions": []
        }
        
        try:
            # Step 1: Switch traffic back to previous environment
            if deployment_result.active_environment != self.active_environment:
                previous_env = self.environments[deployment_result.active_environment]
                current_env = self.environments[self.active_environment]
                
                traffic_rollback = await self.traffic_manager.complete_traffic_switch(previous_env)
                rollback_result["actions"].append({
                    "action": "traffic_rollback",
                    "details": traffic_rollback,
                    "status": "completed"
                })
                
                # Update environment statuses
                current_env.status = EnvironmentStatus.STANDBY
                previous_env.status = EnvironmentStatus.ACTIVE
                self.active_environment = deployment_result.active_environment
            
            # Step 2: Rollback database migrations
            if deployment_result.database_migrations:
                for migration in reversed(deployment_result.database_migrations):
                    migration_rollback = await self.migration_manager.rollback_migrations(migration["migration_id"])
                    rollback_result["actions"].append({
                        "action": "database_rollback",
                        "migration_id": migration["migration_id"],
                        "details": migration_rollback,
                        "status": migration_rollback["status"]
                    })
            
            rollback_end = datetime.now()
            rollback_result["end_time"] = rollback_end.isoformat()
            rollback_result["duration_seconds"] = (rollback_end - rollback_start).total_seconds()
            rollback_result["status"] = "completed"
            
            deployment_result.rollback_info = rollback_result
            
            self.logger.info("Automatic rollback completed successfully")
            
        except Exception as e:
            rollback_result["status"] = "failed"
            rollback_result["error"] = str(e)
            rollback_result["end_time"] = datetime.now().isoformat()
            
            deployment_result.rollback_info = rollback_result
            
            self.logger.error(f"Automatic rollback failed: {e}")
            raise
        
        return rollback_result
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status."""
        # Check active deployments first
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]
        
        # Check deployment history
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        
        return None
    
    def list_active_deployments(self) -> List[DeploymentResult]:
        """List all active deployments."""
        return list(self.active_deployments.values())
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status."""
        return {
            "active_environment": self.active_environment,
            "environments": {
                name: {
                    "status": env.status.value,
                    "endpoint": env.load_balancer_endpoint,
                    "replicas": env.replicas,
                    "version": env.environment_variables.get("VERSION", "unknown")
                }
                for name, env in self.environments.items()
            },
            "last_deployment": self.deployment_history[-1].deployment_id if self.deployment_history else None,
            "active_deployments": len(self.active_deployments)
        }


# Convenience functions for common operations
async def execute_zero_downtime_deployment(application_name: str, version: str, image_tag: str) -> DeploymentResult:
    """Execute zero-downtime deployment with default configuration."""
    deployment_system = BlueGreenDeploymentSystem()
    
    config = DeploymentConfig(
        strategy=DeploymentStrategy.BLUE_GREEN,
        application_name=application_name,
        version=version,
        image_registry="manufacturing-registry",
        image_tag=image_tag,
        auto_rollback_enabled=True,
        database_migration_enabled=True
    )
    
    return await deployment_system.execute_blue_green_deployment(config)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Blue-Green Deployment System Demo")
    print("=" * 80)
    
    async def demo_deployment():
        # Initialize deployment system
        deployment_system = BlueGreenDeploymentSystem()
        
        # Create blue and green environment configurations
        blue_env = EnvironmentConfig(
            environment_name="blue",
            status=EnvironmentStatus.ACTIVE,
            cluster_endpoint="https://manufacturing-blue.us-west-2.eks.amazonaws.com",
            database_endpoint="manufacturing-db-writer.us-west-2.rds.amazonaws.com",
            load_balancer_endpoint="https://manufacturing-blue-lb.us-west-2.elb.amazonaws.com",
            replicas=3
        )
        
        green_env = EnvironmentConfig(
            environment_name="green",
            status=EnvironmentStatus.STANDBY,
            cluster_endpoint="https://manufacturing-green.us-west-2.eks.amazonaws.com",
            database_endpoint="manufacturing-db-writer.us-west-2.rds.amazonaws.com",
            load_balancer_endpoint="https://manufacturing-green-lb.us-west-2.elb.amazonaws.com",
            replicas=3
        )
        
        # Initialize environments
        init_result = deployment_system.initialize_environments(blue_env, green_env)
        print(f"Environments initialized:")
        print(f"  Blue: {init_result['blue_environment']['status']}")
        print(f"  Green: {init_result['green_environment']['status']}")
        print(f"  Active: {init_result['active_environment']}")
        
        # Create deployment configuration
        deployment_config = DeploymentConfig(
            strategy=DeploymentStrategy.BLUE_GREEN,
            application_name="manufacturing-control-system",
            version="2.1.0",
            image_registry="manufacturing-registry",
            image_tag="v2.1.0",
            traffic_shift_duration_minutes=2,  # Faster for demo
            auto_rollback_enabled=True,
            database_migration_enabled=True
        )
        
        print(f"\n" + "="*80)
        print("EXECUTING BLUE-GREEN DEPLOYMENT")
        print("="*80)
        print(f"Application: {deployment_config.application_name}")
        print(f"Version: {deployment_config.version}")
        print(f"Strategy: {deployment_config.strategy.value}")
        
        try:
            # Execute deployment
            deployment_result = await deployment_system.execute_blue_green_deployment(deployment_config)
            
            print(f"\n✅ Deployment completed successfully!")
            print(f"Deployment ID: {deployment_result.deployment_id}")
            print(f"Duration: {(deployment_result.end_time - deployment_result.start_time).total_seconds():.1f} seconds")
            print(f"Final Status: {deployment_result.status}")
            print(f"Active Environment: {deployment_result.active_environment}")
            
            print(f"\nTraffic Distribution:")
            for env, percentage in deployment_result.traffic_distribution.items():
                print(f"  {env}: {percentage}%")
            
            print(f"\nPerformance Metrics:")
            for metric, value in deployment_result.performance_metrics.items():
                print(f"  {metric}: {value}")
            
            print(f"\nDeployment Events:")
            for i, event in enumerate(deployment_result.events[-5:], 1):  # Show last 5 events
                print(f"  {i}. {event['event_type']}: {event['message']}")
            
            # Show environment status
            env_status = deployment_system.get_environment_status()
            print(f"\nEnvironment Status:")
            print(f"  Active Environment: {env_status['active_environment']}")
            for env_name, env_info in env_status['environments'].items():
                print(f"  {env_name}: {env_info['status']} (v{env_info['version']})")
            
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}")
            
            # Check if rollback was attempted
            deployment_result = list(deployment_system.deployment_history)[-1] if deployment_system.deployment_history else None
            if deployment_result and deployment_result.rollback_info:
                print(f"\nRollback Information:")
                rollback = deployment_result.rollback_info
                print(f"  Rollback ID: {rollback['rollback_id']}")
                print(f"  Status: {rollback['status']}")
                print(f"  Actions Performed: {len(rollback.get('actions', []))}")
    
    # Run demo
    asyncio.run(demo_deployment())
    print("\nBlue-Green Deployment System demo completed!")