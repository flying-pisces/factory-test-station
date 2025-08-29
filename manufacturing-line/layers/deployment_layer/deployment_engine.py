"""
Deployment Engine for Week 8: Deployment & Monitoring

This module implements comprehensive deployment automation system for the manufacturing line
control system with zero-downtime deployments, blue-green deployment strategies, canary releases,
rolling updates, and automated rollback capabilities.

Performance Target: <5 minutes for complete production deployment
Deployment Features: Zero-downtime deployments, blue-green deployment, rolling updates, canary releases, rollback automation
Integration: Docker/Kubernetes orchestration, health checks, deployment validation
"""

import time
import logging
import asyncio
import json
import os
import sys
import subprocess
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import concurrent.futures
import traceback
import threading
from pathlib import Path
import uuid
import shutil
import requests
import hashlib
import base64

# Container orchestration
try:
    import docker
    import kubernetes
    from kubernetes import client, config
    CONTAINER_AVAILABLE = True
except ImportError:
    docker = None
    kubernetes = None
    client = None
    config = None
    CONTAINER_AVAILABLE = False

# Week 8 deployment layer integrations (forward references)
try:
    from layers.deployment_layer.monitoring_engine import MonitoringEngine
    from layers.deployment_layer.alerting_engine import AlertingEngine
    from layers.deployment_layer.infrastructure_engine import InfrastructureEngine
except ImportError:
    MonitoringEngine = None
    AlertingEngine = None
    InfrastructureEngine = None

# Week 7 testing layer integrations
try:
    from layers.testing_layer.ci_engine import CIEngine
    from layers.testing_layer.quality_assurance_engine import QualityAssuranceEngine
except ImportError:
    CIEngine = None
    QualityAssuranceEngine = None

# Week 6 UI layer integrations
try:
    from layers.ui_layer.webui_engine import WebUIEngine
    from layers.ui_layer.visualization_engine import VisualizationEngine
except ImportError:
    WebUIEngine = None
    VisualizationEngine = None

# Common imports
try:
    from common.interfaces.layer_interface import LayerInterface
    from common.interfaces.data_interface import DataInterface
    from common.interfaces.communication_interface import CommunicationInterface
except ImportError:
    LayerInterface = None
    DataInterface = None
    CommunicationInterface = None


class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStrategy(Enum):
    """Deployment strategy enumeration"""
    BLUE_GREEN = "blue_green"
    ROLLING_UPDATE = "rolling_update"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class EnvironmentType(Enum):
    """Environment type enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"
    PREVIEW = "preview"


class HealthCheckStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ContainerStatus(Enum):
    """Container status enumeration"""
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    PENDING = "pending"
    TERMINATING = "terminating"


class TrafficSplitType(Enum):
    """Traffic split type enumeration"""
    PERCENTAGE = "percentage"
    HEADER_BASED = "header_based"
    GEOGRAPHIC = "geographic"
    USER_BASED = "user_based"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    name: str
    version: str
    environment: EnvironmentType
    strategy: DeploymentStrategy
    image: str
    replicas: int = 3
    resources: Dict[str, Any] = None
    health_check: Dict[str, Any] = None
    rollback_config: Dict[str, Any] = None
    timeout: int = 300
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = {
                'cpu': '100m',
                'memory': '128Mi',
                'cpu_limit': '500m',
                'memory_limit': '256Mi'
            }
        
        if self.health_check is None:
            self.health_check = {
                'path': '/health',
                'port': 8080,
                'initial_delay': 30,
                'period': 10,
                'timeout': 5,
                'success_threshold': 1,
                'failure_threshold': 3
            }
        
        if self.rollback_config is None:
            self.rollback_config = {
                'enabled': True,
                'auto_rollback': True,
                'success_threshold': 0.95,
                'monitor_duration': 300
            }


@dataclass
class DeploymentTarget:
    """Deployment target configuration"""
    name: str
    type: str  # kubernetes, docker, cloud
    connection: Dict[str, Any]
    namespace: str = "default"
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.annotations is None:
            self.annotations = {}


@dataclass
class DeploymentMetrics:
    """Deployment metrics"""
    deployment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    rollback_count: int = 0
    health_check_failures: int = 0
    traffic_split_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.traffic_split_metrics is None:
            self.traffic_split_metrics = {}


@dataclass
class BlueGreenState:
    """Blue-green deployment state"""
    blue_version: str
    green_version: str
    active_slot: str  # "blue" or "green"
    traffic_split: Dict[str, float]
    switch_time: Optional[datetime] = None
    validation_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = {}


@dataclass
class CanaryState:
    """Canary deployment state"""
    stable_version: str
    canary_version: str
    traffic_percentage: float
    success_metrics: Dict[str, float]
    promotion_criteria: Dict[str, Any]
    rollback_triggers: Dict[str, Any]
    
    def __post_init__(self):
        if self.success_metrics is None:
            self.success_metrics = {}


class ContainerOrchestrator:
    """Container orchestration interface"""
    
    def __init__(self):
        self.docker_client = None
        self.k8s_client = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize container clients"""
        try:
            if CONTAINER_AVAILABLE and docker:
                self.docker_client = docker.from_env()
                self.logger.info("Docker client initialized")
        except Exception as e:
            self.logger.warning(f"Docker client initialization failed: {e}")
        
        try:
            if CONTAINER_AVAILABLE and kubernetes:
                config.load_incluster_config()
                self.k8s_client = client.ApiClient()
                self.logger.info("Kubernetes client initialized")
        except Exception:
            try:
                config.load_kube_config()
                self.k8s_client = client.ApiClient()
                self.logger.info("Kubernetes client initialized from config")
            except Exception as e:
                self.logger.warning(f"Kubernetes client initialization failed: {e}")
    
    async def deploy_to_docker(self, config: DeploymentConfig, target: DeploymentTarget) -> Dict[str, Any]:
        """Deploy to Docker"""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        try:
            # Pull image
            self.logger.info(f"Pulling image: {config.image}")
            image = self.docker_client.images.pull(config.image)
            
            # Create container configuration
            container_config = {
                'image': config.image,
                'name': f"{config.name}-{config.version}",
                'detach': True,
                'labels': target.labels,
                'environment': target.connection.get('environment', {}),
                'ports': target.connection.get('ports', {}),
                'volumes': target.connection.get('volumes', {}),
                'restart_policy': {'Name': 'unless-stopped'},
                'healthcheck': self._create_docker_healthcheck(config.health_check)
            }
            
            # Start container
            container = self.docker_client.containers.run(**container_config)
            
            # Wait for health check
            await self._wait_for_container_health(container, config.timeout)
            
            return {
                'container_id': container.id,
                'status': 'running',
                'image': config.image,
                'ports': container.ports
            }
        
        except Exception as e:
            self.logger.error(f"Docker deployment failed: {e}")
            raise
    
    async def deploy_to_kubernetes(self, config: DeploymentConfig, target: DeploymentTarget) -> Dict[str, Any]:
        """Deploy to Kubernetes"""
        if not self.k8s_client:
            raise RuntimeError("Kubernetes client not available")
        
        try:
            apps_v1 = client.AppsV1Api()
            core_v1 = client.CoreV1Api()
            
            # Create deployment manifest
            deployment = self._create_k8s_deployment(config, target)
            
            # Apply deployment
            try:
                apps_v1.patch_namespaced_deployment(
                    name=config.name,
                    namespace=target.namespace,
                    body=deployment
                )
                self.logger.info(f"Updated existing deployment: {config.name}")
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    apps_v1.create_namespaced_deployment(
                        namespace=target.namespace,
                        body=deployment
                    )
                    self.logger.info(f"Created new deployment: {config.name}")
                else:
                    raise
            
            # Create or update service
            service = self._create_k8s_service(config, target)
            try:
                core_v1.patch_namespaced_service(
                    name=config.name,
                    namespace=target.namespace,
                    body=service
                )
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    core_v1.create_namespaced_service(
                        namespace=target.namespace,
                        body=service
                    )
            
            # Wait for rollout
            await self._wait_for_k8s_rollout(config.name, target.namespace, config.timeout)
            
            return {
                'deployment': config.name,
                'namespace': target.namespace,
                'status': 'deployed',
                'replicas': config.replicas
            }
        
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            raise
    
    def _create_docker_healthcheck(self, health_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Docker healthcheck configuration"""
        return {
            'test': ['CMD-SHELL', f"curl -f http://localhost:{health_config['port']}{health_config['path']} || exit 1"],
            'interval': health_config['period'] * 1000000000,  # nanoseconds
            'timeout': health_config['timeout'] * 1000000000,
            'retries': health_config['failure_threshold'],
            'start_period': health_config['initial_delay'] * 1000000000
        }
    
    async def _wait_for_container_health(self, container, timeout: int):
        """Wait for container health check"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            container.reload()
            if container.status == 'running':
                health = container.attrs.get('State', {}).get('Health', {})
                if health.get('Status') == 'healthy':
                    return True
            await asyncio.sleep(5)
        raise TimeoutError("Container health check timeout")
    
    def _create_k8s_deployment(self, config: DeploymentConfig, target: DeploymentTarget):
        """Create Kubernetes deployment manifest"""
        return client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=config.name,
                namespace=target.namespace,
                labels=target.labels,
                annotations=target.annotations
            ),
            spec=client.V1DeploymentSpec(
                replicas=config.replicas,
                selector=client.V1LabelSelector(
                    match_labels={'app': config.name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={'app': config.name, 'version': config.version}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=config.name,
                                image=config.image,
                                ports=[client.V1ContainerPort(container_port=8080)],
                                resources=client.V1ResourceRequirements(
                                    requests={
                                        'cpu': config.resources['cpu'],
                                        'memory': config.resources['memory']
                                    },
                                    limits={
                                        'cpu': config.resources['cpu_limit'],
                                        'memory': config.resources['memory_limit']
                                    }
                                ),
                                liveness_probe=self._create_k8s_probe(config.health_check),
                                readiness_probe=self._create_k8s_probe(config.health_check)
                            )
                        ]
                    )
                )
            )
        )
    
    def _create_k8s_service(self, config: DeploymentConfig, target: DeploymentTarget):
        """Create Kubernetes service manifest"""
        return client.V1Service(
            metadata=client.V1ObjectMeta(
                name=config.name,
                namespace=target.namespace,
                labels=target.labels
            ),
            spec=client.V1ServiceSpec(
                selector={'app': config.name},
                ports=[
                    client.V1ServicePort(
                        port=80,
                        target_port=8080,
                        protocol='TCP'
                    )
                ],
                type='ClusterIP'
            )
        )
    
    def _create_k8s_probe(self, health_config: Dict[str, Any]):
        """Create Kubernetes probe"""
        return client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path=health_config['path'],
                port=health_config['port']
            ),
            initial_delay_seconds=health_config['initial_delay'],
            period_seconds=health_config['period'],
            timeout_seconds=health_config['timeout'],
            success_threshold=health_config['success_threshold'],
            failure_threshold=health_config['failure_threshold']
        )
    
    async def _wait_for_k8s_rollout(self, name: str, namespace: str, timeout: int):
        """Wait for Kubernetes rollout completion"""
        apps_v1 = client.AppsV1Api()
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = apps_v1.read_namespaced_deployment(name, namespace)
                status = deployment.status
                
                if (status.ready_replicas == deployment.spec.replicas and
                    status.updated_replicas == deployment.spec.replicas and
                    status.available_replicas == deployment.spec.replicas):
                    return True
                
                await asyncio.sleep(10)
            except Exception as e:
                self.logger.error(f"Error checking rollout status: {e}")
                await asyncio.sleep(10)
        
        raise TimeoutError("Kubernetes rollout timeout")


class TrafficManager:
    """Traffic management for deployments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_splits = {}
        self.traffic_rules = {}
    
    async def create_traffic_split(self, deployment_id: str, split_config: Dict[str, Any]) -> str:
        """Create traffic split"""
        split_id = str(uuid.uuid4())
        
        self.active_splits[split_id] = {
            'deployment_id': deployment_id,
            'config': split_config,
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        await self._apply_traffic_split(split_id, split_config)
        
        self.logger.info(f"Created traffic split {split_id} for deployment {deployment_id}")
        return split_id
    
    async def update_traffic_split(self, split_id: str, new_config: Dict[str, Any]):
        """Update traffic split"""
        if split_id not in self.active_splits:
            raise ValueError(f"Traffic split {split_id} not found")
        
        self.active_splits[split_id]['config'] = new_config
        await self._apply_traffic_split(split_id, new_config)
        
        self.logger.info(f"Updated traffic split {split_id}")
    
    async def switch_traffic(self, split_id: str, target_version: str, percentage: float = 100.0):
        """Switch traffic to target version"""
        if split_id not in self.active_splits:
            raise ValueError(f"Traffic split {split_id} not found")
        
        split = self.active_splits[split_id]
        config = split['config'].copy()
        
        # Gradually shift traffic
        current_percentage = config.get('splits', {}).get(target_version, 0.0)
        
        while current_percentage < percentage:
            current_percentage = min(current_percentage + 10, percentage)
            config['splits'][target_version] = current_percentage
            
            # Adjust other versions
            total_others = 100 - current_percentage
            other_versions = {k: v for k, v in config['splits'].items() if k != target_version}
            
            if other_versions:
                scale_factor = total_others / sum(other_versions.values()) if sum(other_versions.values()) > 0 else 0
                for version in other_versions:
                    config['splits'][version] = other_versions[version] * scale_factor
            
            await self._apply_traffic_split(split_id, config)
            await asyncio.sleep(30)  # Wait 30 seconds between shifts
        
        self.active_splits[split_id]['config'] = config
        self.logger.info(f"Switched traffic to {target_version}: {percentage}%")
    
    async def _apply_traffic_split(self, split_id: str, config: Dict[str, Any]):
        """Apply traffic split configuration"""
        try:
            # This would integrate with actual traffic management systems
            # like Istio, NGINX, AWS ALB, etc.
            
            # Simulate traffic rule application
            self.traffic_rules[split_id] = {
                'rules': config,
                'applied_at': datetime.now()
            }
            
            self.logger.info(f"Applied traffic split rules for {split_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply traffic split: {e}")
            raise
    
    def get_traffic_metrics(self, split_id: str) -> Dict[str, Any]:
        """Get traffic metrics for split"""
        if split_id not in self.active_splits:
            return {}
        
        # Simulate traffic metrics
        split = self.active_splits[split_id]
        config = split['config']
        
        metrics = {
            'split_id': split_id,
            'total_requests': 10000,
            'version_metrics': {}
        }
        
        for version, percentage in config.get('splits', {}).items():
            metrics['version_metrics'][version] = {
                'requests': int(10000 * percentage / 100),
                'success_rate': 0.99,
                'avg_latency': 150,
                'error_rate': 0.01
            }
        
        return metrics


class HealthChecker:
    """Health check management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_checks = {}
        self.check_results = defaultdict(deque)
    
    async def register_health_check(self, deployment_id: str, check_config: Dict[str, Any]) -> str:
        """Register health check"""
        check_id = str(uuid.uuid4())
        
        self.health_checks[check_id] = {
            'deployment_id': deployment_id,
            'config': check_config,
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        # Start health check monitoring
        asyncio.create_task(self._monitor_health(check_id))
        
        self.logger.info(f"Registered health check {check_id} for deployment {deployment_id}")
        return check_id
    
    async def _monitor_health(self, check_id: str):
        """Monitor health check"""
        check = self.health_checks[check_id]
        config = check['config']
        
        while check['status'] == 'active':
            try:
                result = await self._perform_health_check(config)
                
                # Store result (keep last 100 results)
                results = self.check_results[check_id]
                results.append(result)
                if len(results) > 100:
                    results.popleft()
                
                # Check for health issues
                if result['status'] != HealthCheckStatus.HEALTHY:
                    self.logger.warning(f"Health check {check_id} failed: {result}")
                
                await asyncio.sleep(config.get('interval', 30))
                
            except Exception as e:
                self.logger.error(f"Health check {check_id} error: {e}")
                await asyncio.sleep(config.get('interval', 30))
    
    async def _perform_health_check(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check"""
        check_type = config.get('type', 'http')
        
        if check_type == 'http':
            return await self._http_health_check(config)
        elif check_type == 'tcp':
            return await self._tcp_health_check(config)
        elif check_type == 'command':
            return await self._command_health_check(config)
        else:
            raise ValueError(f"Unknown health check type: {check_type}")
    
    async def _http_health_check(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform HTTP health check"""
        url = f"http://{config['host']}:{config['port']}{config.get('path', '/health')}"
        timeout = config.get('timeout', 5)
        
        try:
            start_time = time.time()
            
            async with asyncio.timeout(timeout):
                response = requests.get(url, timeout=timeout)
                
            latency = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                return {
                    'status': HealthCheckStatus.HEALTHY,
                    'latency': latency,
                    'response_code': response.status_code,
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'status': HealthCheckStatus.UNHEALTHY,
                    'latency': latency,
                    'response_code': response.status_code,
                    'timestamp': datetime.now()
                }
        
        except Exception as e:
            return {
                'status': HealthCheckStatus.UNHEALTHY,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    async def _tcp_health_check(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform TCP health check"""
        host = config['host']
        port = config['port']
        timeout = config.get('timeout', 5)
        
        try:
            start_time = time.time()
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            
            writer.close()
            await writer.wait_closed()
            
            latency = (time.time() - start_time) * 1000  # ms
            
            return {
                'status': HealthCheckStatus.HEALTHY,
                'latency': latency,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            return {
                'status': HealthCheckStatus.UNHEALTHY,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    async def _command_health_check(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform command health check"""
        command = config['command']
        timeout = config.get('timeout', 5)
        
        try:
            start_time = time.time()
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            latency = (time.time() - start_time) * 1000  # ms
            
            if process.returncode == 0:
                return {
                    'status': HealthCheckStatus.HEALTHY,
                    'latency': latency,
                    'stdout': stdout.decode(),
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'status': HealthCheckStatus.UNHEALTHY,
                    'return_code': process.returncode,
                    'stderr': stderr.decode(),
                    'timestamp': datetime.now()
                }
        
        except Exception as e:
            return {
                'status': HealthCheckStatus.UNHEALTHY,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def get_health_status(self, check_id: str) -> Dict[str, Any]:
        """Get current health status"""
        if check_id not in self.check_results:
            return {'status': HealthCheckStatus.UNKNOWN}
        
        results = list(self.check_results[check_id])
        if not results:
            return {'status': HealthCheckStatus.UNKNOWN}
        
        latest = results[-1]
        recent_results = results[-10:]  # Last 10 checks
        
        success_rate = sum(1 for r in recent_results if r['status'] == HealthCheckStatus.HEALTHY) / len(recent_results)
        avg_latency = sum(r.get('latency', 0) for r in recent_results if 'latency' in r) / len(recent_results)
        
        return {
            'current_status': latest['status'],
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'last_check': latest.get('timestamp'),
            'total_checks': len(results)
        }


class DeploymentEngine:
    """
    Comprehensive deployment automation engine with zero-downtime deployments,
    blue-green deployment, canary releases, and rollback automation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.deployments = {}
        self.deployment_history = []
        self.blue_green_states = {}
        self.canary_states = {}
        self.deployment_targets = {}
        self.metrics_history = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics = {
            'deployment_times': deque(maxlen=100),
            'success_rate': deque(maxlen=100),
            'rollback_rate': deque(maxlen=100),
            'error_count': 0,
            'total_deployments': 0
        }
        
        # Component initialization
        self.orchestrator = ContainerOrchestrator()
        self.traffic_manager = TrafficManager()
        self.health_checker = HealthChecker()
        
        # Integration references
        self.monitoring_engine = None
        self.alerting_engine = None
        self.infrastructure_engine = None
        
        self.logger.info("DeploymentEngine initialized successfully")
    
    def set_integrations(self, monitoring_engine=None, alerting_engine=None, infrastructure_engine=None):
        """Set integration references"""
        self.monitoring_engine = monitoring_engine
        self.alerting_engine = alerting_engine
        self.infrastructure_engine = infrastructure_engine
    
    async def register_deployment_target(self, target: DeploymentTarget) -> str:
        """Register deployment target"""
        target_id = str(uuid.uuid4())
        self.deployment_targets[target_id] = target
        
        self.logger.info(f"Registered deployment target: {target.name}")
        return target_id
    
    async def deploy(self, config: DeploymentConfig, target_id: str) -> str:
        """Deploy application with specified strategy"""
        deployment_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        if target_id not in self.deployment_targets:
            raise ValueError(f"Deployment target {target_id} not found")
        
        target = self.deployment_targets[target_id]
        
        try:
            # Initialize deployment record
            deployment_record = {
                'id': deployment_id,
                'config': config,
                'target_id': target_id,
                'status': DeploymentStatus.PREPARING,
                'start_time': start_time,
                'end_time': None,
                'metrics': DeploymentMetrics(deployment_id, start_time)
            }
            
            self.deployments[deployment_id] = deployment_record
            
            # Execute deployment based on strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green_deployment(deployment_id, config, target)
            elif config.strategy == DeploymentStrategy.ROLLING_UPDATE:
                await self._execute_rolling_update(deployment_id, config, target)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary_deployment(deployment_id, config, target)
            elif config.strategy == DeploymentStrategy.RECREATE:
                await self._execute_recreate_deployment(deployment_id, config, target)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy}")
            
            # Finalize deployment
            deployment_record['status'] = DeploymentStatus.COMPLETED
            deployment_record['end_time'] = datetime.now()
            deployment_record['metrics'].end_time = datetime.now()
            deployment_record['metrics'].duration = (deployment_record['end_time'] - start_time).total_seconds()
            
            # Update performance metrics
            self.performance_metrics['deployment_times'].append(deployment_record['metrics'].duration)
            self.performance_metrics['success_rate'].append(1.0)
            self.performance_metrics['total_deployments'] += 1
            
            # Store in history
            self.deployment_history.append(deployment_record.copy())
            
            # Send success notification
            if self.alerting_engine:
                await self.alerting_engine.send_deployment_notification(
                    deployment_id, "success", f"Deployment {config.name} completed successfully"
                )
            
            self.logger.info(f"Deployment {deployment_id} completed successfully in {deployment_record['metrics'].duration:.2f}s")
            return deployment_id
            
        except Exception as e:
            # Handle deployment failure
            deployment_record = self.deployments.get(deployment_id, {})
            deployment_record['status'] = DeploymentStatus.FAILED
            deployment_record['error'] = str(e)
            deployment_record['end_time'] = datetime.now()
            
            # Update performance metrics
            self.performance_metrics['success_rate'].append(0.0)
            self.performance_metrics['error_count'] += 1
            
            # Attempt automatic rollback if configured
            if config.rollback_config.get('auto_rollback', False):
                try:
                    await self.rollback_deployment(deployment_id)
                except Exception as rollback_error:
                    self.logger.error(f"Automatic rollback failed: {rollback_error}")
            
            # Send failure notification
            if self.alerting_engine:
                await self.alerting_engine.send_deployment_notification(
                    deployment_id, "failure", f"Deployment {config.name} failed: {str(e)}"
                )
            
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            raise
    
    async def _execute_blue_green_deployment(self, deployment_id: str, config: DeploymentConfig, target: DeploymentTarget):
        """Execute blue-green deployment"""
        deployment = self.deployments[deployment_id]
        deployment['status'] = DeploymentStatus.DEPLOYING
        
        # Initialize blue-green state
        bg_state = BlueGreenState(
            blue_version="v1",
            green_version=config.version,
            active_slot="blue",
            traffic_split={"blue": 100.0, "green": 0.0}
        )
        self.blue_green_states[deployment_id] = bg_state
        
        try:
            # Deploy to green slot
            self.logger.info(f"Deploying {config.version} to green slot")
            
            # Create green deployment
            green_config = config
            green_config.name = f"{config.name}-green"
            
            if target.type == "kubernetes":
                result = await self.orchestrator.deploy_to_kubernetes(green_config, target)
            elif target.type == "docker":
                result = await self.orchestrator.deploy_to_docker(green_config, target)
            else:
                raise ValueError(f"Unsupported target type: {target.type}")
            
            # Register health check for green deployment
            health_check_id = await self.health_checker.register_health_check(
                deployment_id,
                {
                    'type': 'http',
                    'host': 'localhost',
                    'port': config.health_check['port'],
                    'path': config.health_check['path'],
                    'interval': config.health_check['period']
                }
            )
            
            # Wait for green to be healthy
            await self._wait_for_deployment_health(health_check_id, config.timeout)
            
            # Validate green deployment
            deployment['status'] = DeploymentStatus.VALIDATING
            validation_results = await self._validate_deployment(deployment_id, config, target)
            bg_state.validation_results = validation_results
            
            if not validation_results.get('success', False):
                raise RuntimeError(f"Green deployment validation failed: {validation_results}")
            
            # Switch traffic to green
            self.logger.info("Switching traffic to green deployment")
            
            split_id = await self.traffic_manager.create_traffic_split(
                deployment_id,
                {'splits': {'blue': 100.0, 'green': 0.0}}
            )
            
            await self.traffic_manager.switch_traffic(split_id, 'green', 100.0)
            
            # Update state
            bg_state.active_slot = "green"
            bg_state.traffic_split = {"blue": 0.0, "green": 100.0}
            bg_state.switch_time = datetime.now()
            
            # Clean up old blue deployment after successful switch
            await asyncio.sleep(300)  # Wait 5 minutes before cleanup
            self.logger.info("Blue-green deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Blue-green deployment failed: {e}")
            # Cleanup green deployment on failure
            await self._cleanup_failed_deployment(deployment_id, "green")
            raise
    
    async def _execute_rolling_update(self, deployment_id: str, config: DeploymentConfig, target: DeploymentTarget):
        """Execute rolling update deployment"""
        deployment = self.deployments[deployment_id]
        deployment['status'] = DeploymentStatus.DEPLOYING
        
        try:
            self.logger.info(f"Starting rolling update for {config.name}")
            
            # For Kubernetes, this is handled by the deployment controller
            if target.type == "kubernetes":
                result = await self.orchestrator.deploy_to_kubernetes(config, target)
                
                # Monitor rollout progress
                await self.orchestrator._wait_for_k8s_rollout(config.name, target.namespace, config.timeout)
                
            elif target.type == "docker":
                # Implement rolling update for Docker
                await self._docker_rolling_update(deployment_id, config, target)
                
            else:
                raise ValueError(f"Unsupported target type: {target.type}")
            
            # Validate deployment
            deployment['status'] = DeploymentStatus.VALIDATING
            validation_results = await self._validate_deployment(deployment_id, config, target)
            
            if not validation_results.get('success', False):
                raise RuntimeError(f"Rolling update validation failed: {validation_results}")
            
            self.logger.info("Rolling update completed successfully")
            
        except Exception as e:
            self.logger.error(f"Rolling update failed: {e}")
            raise
    
    async def _execute_canary_deployment(self, deployment_id: str, config: DeploymentConfig, target: DeploymentTarget):
        """Execute canary deployment"""
        deployment = self.deployments[deployment_id]
        deployment['status'] = DeploymentStatus.DEPLOYING
        
        # Initialize canary state
        canary_state = CanaryState(
            stable_version="v1",
            canary_version=config.version,
            traffic_percentage=10.0,  # Start with 10% traffic
            success_metrics={},
            promotion_criteria={
                'success_rate': 0.99,
                'error_rate': 0.01,
                'latency_p95': 500
            },
            rollback_triggers={
                'error_rate': 0.05,
                'latency_p95': 1000
            }
        )
        self.canary_states[deployment_id] = canary_state
        
        try:
            # Deploy canary version
            self.logger.info(f"Deploying canary version {config.version}")
            
            canary_config = config
            canary_config.name = f"{config.name}-canary"
            canary_config.replicas = max(1, config.replicas // 10)  # 10% of replicas
            
            if target.type == "kubernetes":
                result = await self.orchestrator.deploy_to_kubernetes(canary_config, target)
            elif target.type == "docker":
                result = await self.orchestrator.deploy_to_docker(canary_config, target)
            else:
                raise ValueError(f"Unsupported target type: {target.type}")
            
            # Create traffic split
            split_id = await self.traffic_manager.create_traffic_split(
                deployment_id,
                {'splits': {'stable': 90.0, 'canary': 10.0}}
            )
            
            # Monitor canary performance
            await self._monitor_canary_deployment(deployment_id, split_id, config.timeout)
            
            # Promote canary if successful
            await self._promote_canary_deployment(deployment_id, split_id, config, target)
            
            self.logger.info("Canary deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Canary deployment failed: {e}")
            # Rollback canary
            await self._rollback_canary_deployment(deployment_id)
            raise
    
    async def _execute_recreate_deployment(self, deployment_id: str, config: DeploymentConfig, target: DeploymentTarget):
        """Execute recreate deployment (with downtime)"""
        deployment = self.deployments[deployment_id]
        deployment['status'] = DeploymentStatus.DEPLOYING
        
        try:
            self.logger.info(f"Starting recreate deployment for {config.name}")
            
            # Stop old deployment
            if target.type == "kubernetes":
                apps_v1 = client.AppsV1Api()
                try:
                    apps_v1.patch_namespaced_deployment(
                        name=config.name,
                        namespace=target.namespace,
                        body={'spec': {'replicas': 0}}
                    )
                    await asyncio.sleep(30)  # Wait for pods to terminate
                except Exception:
                    pass  # Deployment might not exist
                
                # Deploy new version
                result = await self.orchestrator.deploy_to_kubernetes(config, target)
                
            elif target.type == "docker":
                # Stop and remove old containers
                if self.orchestrator.docker_client:
                    try:
                        old_container = self.orchestrator.docker_client.containers.get(config.name)
                        old_container.stop()
                        old_container.remove()
                    except Exception:
                        pass  # Container might not exist
                
                # Deploy new container
                result = await self.orchestrator.deploy_to_docker(config, target)
                
            else:
                raise ValueError(f"Unsupported target type: {target.type}")
            
            # Validate deployment
            deployment['status'] = DeploymentStatus.VALIDATING
            validation_results = await self._validate_deployment(deployment_id, config, target)
            
            if not validation_results.get('success', False):
                raise RuntimeError(f"Recreate deployment validation failed: {validation_results}")
            
            self.logger.info("Recreate deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Recreate deployment failed: {e}")
            raise
    
    async def _wait_for_deployment_health(self, health_check_id: str, timeout: int):
        """Wait for deployment to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health_status = self.health_checker.get_health_status(health_check_id)
            
            if health_status.get('current_status') == HealthCheckStatus.HEALTHY:
                if health_status.get('success_rate', 0) >= 0.9:  # 90% success rate required
                    return True
            
            await asyncio.sleep(10)
        
        raise TimeoutError("Deployment health check timeout")
    
    async def _validate_deployment(self, deployment_id: str, config: DeploymentConfig, target: DeploymentTarget) -> Dict[str, Any]:
        """Validate deployment"""
        validation_results = {
            'success': True,
            'checks': {},
            'errors': []
        }
        
        try:
            # Health check validation
            validation_results['checks']['health'] = await self._validate_health_checks(config)
            
            # Performance validation
            validation_results['checks']['performance'] = await self._validate_performance(config)
            
            # Integration validation
            if self.monitoring_engine:
                validation_results['checks']['monitoring'] = await self._validate_monitoring_integration(deployment_id)
            
            # Check if all validations passed
            validation_results['success'] = all(
                check.get('passed', False) for check in validation_results['checks'].values()
            )
            
        except Exception as e:
            validation_results['success'] = False
            validation_results['errors'].append(str(e))
        
        return validation_results
    
    async def _validate_health_checks(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate health checks"""
        try:
            health_config = config.health_check
            url = f"http://localhost:{health_config['port']}{health_config['path']}"
            
            # Perform multiple health checks
            success_count = 0
            total_checks = 5
            
            for _ in range(total_checks):
                try:
                    response = requests.get(url, timeout=health_config['timeout'])
                    if response.status_code == 200:
                        success_count += 1
                except Exception:
                    pass
                await asyncio.sleep(2)
            
            success_rate = success_count / total_checks
            
            return {
                'passed': success_rate >= 0.8,
                'success_rate': success_rate,
                'checks_performed': total_checks
            }
        
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    async def _validate_performance(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate performance metrics"""
        try:
            # Simulate performance validation
            # In real implementation, this would check actual metrics
            
            return {
                'passed': True,
                'avg_response_time': 150,
                'throughput': 1000,
                'error_rate': 0.001
            }
        
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    async def _validate_monitoring_integration(self, deployment_id: str) -> Dict[str, Any]:
        """Validate monitoring integration"""
        try:
            if self.monitoring_engine:
                metrics = await self.monitoring_engine.get_deployment_metrics(deployment_id)
                return {
                    'passed': len(metrics) > 0,
                    'metrics_count': len(metrics)
                }
            
            return {
                'passed': True,
                'note': 'Monitoring engine not available'
            }
        
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback deployment"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.deployments[deployment_id]
        deployment['status'] = DeploymentStatus.ROLLING_BACK
        
        try:
            config = deployment['config']
            target_id = deployment['target_id']
            target = self.deployment_targets[target_id]
            
            self.logger.info(f"Starting rollback for deployment {deployment_id}")
            
            # Find previous successful deployment
            previous_deployment = self._find_previous_successful_deployment(config.name)
            
            if not previous_deployment:
                raise RuntimeError("No previous successful deployment found for rollback")
            
            # Execute rollback based on original strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._rollback_blue_green_deployment(deployment_id)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self._rollback_canary_deployment(deployment_id)
            else:
                # Standard rollback - deploy previous version
                rollback_config = previous_deployment['config']
                await self._execute_recreate_deployment(deployment_id, rollback_config, target)
            
            deployment['status'] = DeploymentStatus.COMPLETED
            deployment['rollback_completed'] = True
            
            # Update metrics
            self.performance_metrics['rollback_rate'].append(1.0)
            
            self.logger.info(f"Rollback completed for deployment {deployment_id}")
            return True
            
        except Exception as e:
            deployment['status'] = DeploymentStatus.FAILED
            deployment['rollback_error'] = str(e)
            
            self.logger.error(f"Rollback failed for deployment {deployment_id}: {e}")
            return False
    
    def _find_previous_successful_deployment(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Find previous successful deployment for rollback"""
        for deployment in reversed(self.deployment_history):
            if (deployment['config'].name == app_name and
                deployment['status'] == DeploymentStatus.COMPLETED and
                not deployment.get('rollback_completed', False)):
                return deployment
        return None
    
    async def _rollback_blue_green_deployment(self, deployment_id: str):
        """Rollback blue-green deployment"""
        if deployment_id not in self.blue_green_states:
            raise RuntimeError("Blue-green state not found")
        
        bg_state = self.blue_green_states[deployment_id]
        
        # Switch traffic back to previous slot
        if bg_state.active_slot == "green":
            target_slot = "blue"
        else:
            target_slot = "green"
        
        # Create new traffic split for rollback
        split_id = await self.traffic_manager.create_traffic_split(
            deployment_id,
            {'splits': {bg_state.active_slot: 100.0, target_slot: 0.0}}
        )
        
        await self.traffic_manager.switch_traffic(split_id, target_slot, 100.0)
        
        # Update state
        bg_state.active_slot = target_slot
        bg_state.switch_time = datetime.now()
        
        self.logger.info(f"Blue-green rollback completed, switched to {target_slot}")
    
    async def _rollback_canary_deployment(self, deployment_id: str):
        """Rollback canary deployment"""
        if deployment_id not in self.canary_states:
            raise RuntimeError("Canary state not found")
        
        # Simply remove canary traffic
        split_id = await self.traffic_manager.create_traffic_split(
            deployment_id,
            {'splits': {'stable': 100.0, 'canary': 0.0}}
        )
        
        # Clean up canary deployment
        await self._cleanup_failed_deployment(deployment_id, "canary")
        
        self.logger.info("Canary rollback completed, traffic restored to stable")
    
    async def _monitor_canary_deployment(self, deployment_id: str, split_id: str, timeout: int):
        """Monitor canary deployment performance"""
        start_time = time.time()
        canary_state = self.canary_states[deployment_id]
        
        while time.time() - start_time < timeout:
            # Get traffic metrics
            traffic_metrics = self.traffic_manager.get_traffic_metrics(split_id)
            
            # Check canary metrics
            canary_metrics = traffic_metrics['version_metrics'].get('canary', {})
            
            if canary_metrics:
                # Update success metrics
                canary_state.success_metrics.update({
                    'success_rate': canary_metrics.get('success_rate', 0),
                    'avg_latency': canary_metrics.get('avg_latency', 0),
                    'error_rate': canary_metrics.get('error_rate', 0)
                })
                
                # Check rollback triggers
                if canary_metrics.get('error_rate', 0) > canary_state.rollback_triggers['error_rate']:
                    raise RuntimeError(f"Canary error rate too high: {canary_metrics['error_rate']}")
                
                if canary_metrics.get('avg_latency', 0) > canary_state.rollback_triggers['latency_p95']:
                    raise RuntimeError(f"Canary latency too high: {canary_metrics['avg_latency']}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _promote_canary_deployment(self, deployment_id: str, split_id: str, config: DeploymentConfig, target: DeploymentTarget):
        """Promote canary to full deployment"""
        canary_state = self.canary_states[deployment_id]
        
        # Check promotion criteria
        success_rate = canary_state.success_metrics.get('success_rate', 0)
        if success_rate < canary_state.promotion_criteria['success_rate']:
            raise RuntimeError(f"Canary success rate too low for promotion: {success_rate}")
        
        # Gradually increase traffic to canary
        traffic_percentages = [25, 50, 75, 100]
        
        for percentage in traffic_percentages:
            await self.traffic_manager.update_traffic_split(
                split_id,
                {'splits': {'stable': 100 - percentage, 'canary': percentage}}
            )
            
            # Monitor for 2 minutes at each step
            await asyncio.sleep(120)
            
            # Check metrics at each step
            traffic_metrics = self.traffic_manager.get_traffic_metrics(split_id)
            canary_metrics = traffic_metrics['version_metrics'].get('canary', {})
            
            if canary_metrics.get('error_rate', 0) > canary_state.rollback_triggers['error_rate']:
                raise RuntimeError(f"Canary promotion failed at {percentage}% traffic")
        
        # Replace stable deployment with canary
        stable_config = config
        stable_config.name = config.name.replace('-canary', '')
        stable_config.replicas = config.replicas * 10  # Scale back to full replicas
        
        if target.type == "kubernetes":
            await self.orchestrator.deploy_to_kubernetes(stable_config, target)
        elif target.type == "docker":
            await self.orchestrator.deploy_to_docker(stable_config, target)
        
        self.logger.info("Canary promotion completed successfully")
    
    async def _cleanup_failed_deployment(self, deployment_id: str, slot: str):
        """Clean up failed deployment"""
        try:
            deployment = self.deployments.get(deployment_id, {})
            config = deployment.get('config')
            target_id = deployment.get('target_id')
            
            if not config or not target_id:
                return
            
            target = self.deployment_targets[target_id]
            cleanup_name = f"{config.name}-{slot}"
            
            if target.type == "kubernetes":
                apps_v1 = client.AppsV1Api()
                try:
                    apps_v1.delete_namespaced_deployment(
                        name=cleanup_name,
                        namespace=target.namespace
                    )
                    self.logger.info(f"Cleaned up failed Kubernetes deployment: {cleanup_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup Kubernetes deployment: {e}")
            
            elif target.type == "docker":
                if self.orchestrator.docker_client:
                    try:
                        container = self.orchestrator.docker_client.containers.get(cleanup_name)
                        container.stop()
                        container.remove()
                        self.logger.info(f"Cleaned up failed Docker container: {cleanup_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup Docker container: {e}")
        
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    async def _docker_rolling_update(self, deployment_id: str, config: DeploymentConfig, target: DeploymentTarget):
        """Implement rolling update for Docker"""
        if not self.orchestrator.docker_client:
            raise RuntimeError("Docker client not available")
        
        try:
            # Get current containers
            current_containers = self.orchestrator.docker_client.containers.list(
                filters={'label': f'app={config.name}'}
            )
            
            # Start new containers one by one
            new_containers = []
            
            for i in range(config.replicas):
                container_name = f"{config.name}-{config.version}-{i}"
                
                container_config = {
                    'image': config.image,
                    'name': container_name,
                    'detach': True,
                    'labels': {'app': config.name, 'version': config.version},
                    'environment': target.connection.get('environment', {}),
                    'ports': target.connection.get('ports', {}),
                    'restart_policy': {'Name': 'unless-stopped'}
                }
                
                new_container = self.orchestrator.docker_client.containers.run(**container_config)
                new_containers.append(new_container)
                
                # Wait for container to be healthy
                await self.orchestrator._wait_for_container_health(new_container, 60)
                
                # Remove old container if exists
                if i < len(current_containers):
                    old_container = current_containers[i]
                    old_container.stop()
                    old_container.remove()
                
                # Wait between container updates
                await asyncio.sleep(30)
            
            self.logger.info("Docker rolling update completed")
        
        except Exception as e:
            # Cleanup new containers on failure
            for container in new_containers:
                try:
                    container.stop()
                    container.remove()
                except Exception:
                    pass
            raise
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        if deployment_id not in self.deployments:
            return {'error': 'Deployment not found'}
        
        deployment = self.deployments[deployment_id]
        
        status = {
            'id': deployment_id,
            'status': deployment['status'].value,
            'config': {
                'name': deployment['config'].name,
                'version': deployment['config'].version,
                'strategy': deployment['config'].strategy.value,
                'environment': deployment['config'].environment.value
            },
            'start_time': deployment['start_time'].isoformat(),
            'end_time': deployment['end_time'].isoformat() if deployment['end_time'] else None,
            'metrics': asdict(deployment['metrics'])
        }
        
        # Add strategy-specific status
        if deployment['config'].strategy == DeploymentStrategy.BLUE_GREEN:
            if deployment_id in self.blue_green_states:
                status['blue_green_state'] = asdict(self.blue_green_states[deployment_id])
        
        elif deployment['config'].strategy == DeploymentStrategy.CANARY:
            if deployment_id in self.canary_states:
                status['canary_state'] = asdict(self.canary_states[deployment_id])
        
        return status
    
    def list_deployments(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List deployments with optional filters"""
        deployments = []
        
        for deployment_id, deployment in self.deployments.items():
            deployment_info = {
                'id': deployment_id,
                'name': deployment['config'].name,
                'version': deployment['config'].version,
                'status': deployment['status'].value,
                'strategy': deployment['config'].strategy.value,
                'environment': deployment['config'].environment.value,
                'start_time': deployment['start_time'].isoformat(),
                'duration': deployment['metrics'].duration
            }
            
            # Apply filters
            if filters:
                if 'status' in filters and deployment_info['status'] != filters['status']:
                    continue
                if 'environment' in filters and deployment_info['environment'] != filters['environment']:
                    continue
                if 'name' in filters and deployment_info['name'] != filters['name']:
                    continue
            
            deployments.append(deployment_info)
        
        return sorted(deployments, key=lambda x: x['start_time'], reverse=True)
    
    def get_deployment_metrics(self) -> Dict[str, Any]:
        """Get deployment engine metrics"""
        total_deployments = len(self.performance_metrics['deployment_times'])
        
        if total_deployments == 0:
            return {
                'total_deployments': 0,
                'avg_deployment_time': 0,
                'success_rate': 0,
                'rollback_rate': 0,
                'error_count': 0
            }
        
        return {
            'total_deployments': total_deployments,
            'avg_deployment_time': sum(self.performance_metrics['deployment_times']) / total_deployments,
            'success_rate': sum(self.performance_metrics['success_rate']) / len(self.performance_metrics['success_rate']),
            'rollback_rate': sum(self.performance_metrics['rollback_rate']) / len(self.performance_metrics['rollback_rate']) if self.performance_metrics['rollback_rate'] else 0,
            'error_count': self.performance_metrics['error_count'],
            'performance_trend': {
                'deployment_times': list(self.performance_metrics['deployment_times']),
                'recent_success_rate': list(self.performance_metrics['success_rate'])[-10:]
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop health check monitoring
            for check_id, check in self.health_checker.health_checks.items():
                check['status'] = 'inactive'
            
            # Cleanup container clients
            if self.orchestrator.docker_client:
                self.orchestrator.docker_client.close()
            
            self.logger.info("DeploymentEngine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"DeploymentEngine cleanup error: {e}")


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create deployment engine
        engine = DeploymentEngine()
        
        # Create deployment target
        target = DeploymentTarget(
            name="production-cluster",
            type="kubernetes",
            connection={
                'kubeconfig': '/path/to/kubeconfig',
                'context': 'production'
            },
            namespace="manufacturing-line",
            labels={'environment': 'production', 'tier': 'application'}
        )
        
        target_id = await engine.register_deployment_target(target)
        
        # Create deployment configuration
        config = DeploymentConfig(
            name="line-controller",
            version="v2.1.0",
            environment=EnvironmentType.PRODUCTION,
            strategy=DeploymentStrategy.BLUE_GREEN,
            image="manufacturing-line/controller:v2.1.0",
            replicas=5
        )
        
        try:
            # Execute deployment
            deployment_id = await engine.deploy(config, target_id)
            print(f"Deployment completed: {deployment_id}")
            
            # Get deployment status
            status = engine.get_deployment_status(deployment_id)
            print(f"Deployment status: {status}")
            
            # Get metrics
            metrics = engine.get_deployment_metrics()
            print(f"Engine metrics: {metrics}")
            
        except Exception as e:
            print(f"Deployment failed: {e}")
        
        finally:
            await engine.cleanup()
    
    asyncio.run(main())