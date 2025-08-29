"""
Infrastructure Engine for Week 8: Deployment & Monitoring

This module implements comprehensive infrastructure management system for the manufacturing line
control system with auto-scaling, resource optimization, infrastructure monitoring,
cloud integration across AWS/Azure/GCP, and hybrid infrastructure support.

Performance Target: <2 minutes for scaling operations and resource provisioning
Infrastructure Features: Auto-scaling, resource optimization, infrastructure monitoring, cloud integration
Integration: AWS/Azure/GCP, Kubernetes HPA, resource management
"""

import time
import logging
import asyncio
import json
import os
import sys
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import concurrent.futures
import traceback
from pathlib import Path
import uuid
import statistics
import hashlib
import yaml

# Cloud provider SDKs
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    AWS_AVAILABLE = True
except ImportError:
    boto3 = None
    ClientError = None
    BotoCoreError = None
    AWS_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.resource import ResourceManagementClient
    from azure.mgmt.compute import ComputeManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    DefaultAzureCredential = None
    ResourceManagementClient = None
    ComputeManagementClient = None
    AZURE_AVAILABLE = False

try:
    from google.cloud import compute_v1
    from google.cloud import monitoring_v3
    GCP_AVAILABLE = True
except ImportError:
    compute_v1 = None
    monitoring_v3 = None
    GCP_AVAILABLE = False

# Container orchestration
try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    kubernetes = None
    client = None
    config = None
    KUBERNETES_AVAILABLE = False

# Database for infrastructure data
try:
    import sqlite3
    import redis
    DATABASE_AVAILABLE = True
except ImportError:
    sqlite3 = None
    redis = None
    DATABASE_AVAILABLE = False

# Week 8 deployment layer integrations
try:
    from layers.deployment_layer.deployment_engine import DeploymentEngine
    from layers.deployment_layer.monitoring_engine import MonitoringEngine
    from layers.deployment_layer.alerting_engine import AlertingEngine
except ImportError:
    DeploymentEngine = None
    MonitoringEngine = None
    AlertingEngine = None

# Week 7 testing layer integrations
try:
    from layers.testing_layer.benchmarking_engine import BenchmarkingEngine
    from layers.testing_layer.quality_assurance_engine import QualityAssuranceEngine
except ImportError:
    BenchmarkingEngine = None
    QualityAssuranceEngine = None

# Week 6 UI layer integrations
try:
    from layers.ui_layer.visualization_engine import VisualizationEngine
except ImportError:
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


class CloudProvider(Enum):
    """Cloud provider enumeration"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Resource type enumeration"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    LOAD_BALANCER = "load_balancer"
    CONTAINER = "container"
    FUNCTION = "function"


class ScalingPolicy(Enum):
    """Scaling policy enumeration"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_COUNT = "request_count"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"
    SCHEDULE_BASED = "schedule_based"


class ScalingDirection(Enum):
    """Scaling direction enumeration"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"


class ResourceStatus(Enum):
    """Resource status enumeration"""
    RUNNING = "running"
    STOPPED = "stopped"
    PENDING = "pending"
    TERMINATING = "terminating"
    ERROR = "error"
    UNKNOWN = "unknown"


class OptimizationStrategy(Enum):
    """Optimization strategy enumeration"""
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    AVAILABILITY_OPTIMIZATION = "availability_optimization"
    SUSTAINABILITY_OPTIMIZATION = "sustainability_optimization"
    BALANCED = "balanced"


@dataclass
class CloudCredentials:
    """Cloud provider credentials"""
    provider: CloudProvider
    credentials: Dict[str, str]
    region: str = "us-east-1"
    project_id: Optional[str] = None
    
    def __post_init__(self):
        # Mask sensitive data in logs
        self.masked_credentials = {k: "***" for k in self.credentials.keys()}


@dataclass
class ResourceSpec:
    """Resource specification"""
    name: str
    type: ResourceType
    provider: CloudProvider
    config: Dict[str, Any]
    tags: Dict[str, str] = None
    min_capacity: int = 1
    max_capacity: int = 10
    desired_capacity: int = 2
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class ScalingRule:
    """Auto-scaling rule"""
    name: str
    policy: ScalingPolicy
    direction: ScalingDirection
    threshold: float
    duration: int  # seconds
    cooldown: int  # seconds
    adjustment: Union[int, float]  # absolute or percentage
    adjustment_type: str = "absolute"  # "absolute" or "percentage"
    enabled: bool = True


@dataclass
class ResourceInstance:
    """Resource instance"""
    id: str
    name: str
    type: ResourceType
    provider: CloudProvider
    status: ResourceStatus
    config: Dict[str, Any]
    metrics: Dict[str, float] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class ScalingEvent:
    """Scaling event record"""
    id: str
    resource_id: str
    rule_name: str
    direction: ScalingDirection
    old_capacity: int
    new_capacity: int
    trigger_value: float
    threshold: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None


@dataclass
class CostReport:
    """Cost report"""
    period_start: datetime
    period_end: datetime
    total_cost: float
    cost_by_service: Dict[str, float]
    cost_by_region: Dict[str, float]
    cost_optimization_recommendations: List[str]
    currency: str = "USD"


class CloudProviderInterface:
    """Base interface for cloud providers"""
    
    def __init__(self, credentials: CloudCredentials):
        self.credentials = credentials
        self.logger = logging.getLogger(__name__)
    
    async def create_resource(self, spec: ResourceSpec) -> ResourceInstance:
        """Create resource"""
        raise NotImplementedError
    
    async def delete_resource(self, resource_id: str) -> bool:
        """Delete resource"""
        raise NotImplementedError
    
    async def scale_resource(self, resource_id: str, new_capacity: int) -> bool:
        """Scale resource"""
        raise NotImplementedError
    
    async def get_resource_metrics(self, resource_id: str) -> Dict[str, float]:
        """Get resource metrics"""
        raise NotImplementedError
    
    async def list_resources(self, resource_type: ResourceType = None) -> List[ResourceInstance]:
        """List resources"""
        raise NotImplementedError
    
    async def get_cost_data(self, start_date: datetime, end_date: datetime) -> CostReport:
        """Get cost data"""
        raise NotImplementedError


class AWSProvider(CloudProviderInterface):
    """AWS cloud provider implementation"""
    
    def __init__(self, credentials: CloudCredentials):
        super().__init__(credentials)
        self.session = None
        self.ec2_client = None
        self.autoscaling_client = None
        self.cloudwatch_client = None
        self.cost_explorer_client = None
        
        if AWS_AVAILABLE:
            self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS clients"""
        try:
            if 'aws_access_key_id' in self.credentials.credentials:
                self.session = boto3.Session(
                    aws_access_key_id=self.credentials.credentials['aws_access_key_id'],
                    aws_secret_access_key=self.credentials.credentials['aws_secret_access_key'],
                    region_name=self.credentials.region
                )
            else:
                self.session = boto3.Session(region_name=self.credentials.region)
            
            self.ec2_client = self.session.client('ec2')
            self.autoscaling_client = self.session.client('autoscaling')
            self.cloudwatch_client = self.session.client('cloudwatch')
            self.cost_explorer_client = self.session.client('ce')
            
            self.logger.info("AWS clients initialized")
            
        except Exception as e:
            self.logger.error(f"AWS client initialization failed: {e}")
            raise
    
    async def create_resource(self, spec: ResourceSpec) -> ResourceInstance:
        """Create AWS resource"""
        if not self.ec2_client:
            raise RuntimeError("AWS client not initialized")
        
        try:
            if spec.type == ResourceType.COMPUTE:
                response = self.ec2_client.run_instances(
                    ImageId=spec.config.get('image_id', 'ami-0abcdef1234567890'),
                    MinCount=spec.min_capacity,
                    MaxCount=spec.max_capacity,
                    InstanceType=spec.config.get('instance_type', 't3.micro'),
                    KeyName=spec.config.get('key_name'),
                    SecurityGroupIds=spec.config.get('security_groups', []),
                    SubnetId=spec.config.get('subnet_id'),
                    TagSpecifications=[{
                        'ResourceType': 'instance',
                        'Tags': [{'Key': k, 'Value': v} for k, v in spec.tags.items()]
                    }]
                )
                
                instance_id = response['Instances'][0]['InstanceId']
                
                return ResourceInstance(
                    id=instance_id,
                    name=spec.name,
                    type=spec.type,
                    provider=spec.provider,
                    status=ResourceStatus.PENDING,
                    config=spec.config
                )
            
            else:
                raise ValueError(f"Resource type {spec.type} not supported for AWS")
                
        except Exception as e:
            self.logger.error(f"AWS resource creation failed: {e}")
            raise
    
    async def delete_resource(self, resource_id: str) -> bool:
        """Delete AWS resource"""
        if not self.ec2_client:
            return False
        
        try:
            self.ec2_client.terminate_instances(InstanceIds=[resource_id])
            self.logger.info(f"AWS resource {resource_id} termination initiated")
            return True
            
        except Exception as e:
            self.logger.error(f"AWS resource deletion failed: {e}")
            return False
    
    async def scale_resource(self, resource_id: str, new_capacity: int) -> bool:
        """Scale AWS resource"""
        if not self.autoscaling_client:
            return False
        
        try:
            # Update auto scaling group capacity
            self.autoscaling_client.update_auto_scaling_group(
                AutoScalingGroupName=resource_id,
                DesiredCapacity=new_capacity
            )
            
            self.logger.info(f"AWS resource {resource_id} scaled to {new_capacity}")
            return True
            
        except Exception as e:
            self.logger.error(f"AWS resource scaling failed: {e}")
            return False
    
    async def get_resource_metrics(self, resource_id: str) -> Dict[str, float]:
        """Get AWS resource metrics"""
        if not self.cloudwatch_client:
            return {}
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=10)
            
            # Get CPU utilization
            cpu_response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': resource_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            
            cpu_utilization = 0
            if cpu_response['Datapoints']:
                cpu_utilization = cpu_response['Datapoints'][-1]['Average']
            
            return {
                'cpu_utilization': cpu_utilization,
                'timestamp': end_time.timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"AWS metrics retrieval failed: {e}")
            return {}
    
    async def list_resources(self, resource_type: ResourceType = None) -> List[ResourceInstance]:
        """List AWS resources"""
        if not self.ec2_client:
            return []
        
        try:
            response = self.ec2_client.describe_instances()
            resources = []
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    status_mapping = {
                        'running': ResourceStatus.RUNNING,
                        'stopped': ResourceStatus.STOPPED,
                        'pending': ResourceStatus.PENDING,
                        'terminating': ResourceStatus.TERMINATING,
                        'terminated': ResourceStatus.TERMINATING
                    }
                    
                    resources.append(ResourceInstance(
                        id=instance['InstanceId'],
                        name=self._get_tag_value(instance.get('Tags', []), 'Name', instance['InstanceId']),
                        type=ResourceType.COMPUTE,
                        provider=CloudProvider.AWS,
                        status=status_mapping.get(instance['State']['Name'], ResourceStatus.UNKNOWN),
                        config={
                            'instance_type': instance['InstanceType'],
                            'image_id': instance['ImageId'],
                            'availability_zone': instance.get('Placement', {}).get('AvailabilityZone')
                        },
                        created_at=instance['LaunchTime']
                    ))
            
            return resources
            
        except Exception as e:
            self.logger.error(f"AWS resource listing failed: {e}")
            return []
    
    def _get_tag_value(self, tags: List[Dict], key: str, default: str = "") -> str:
        """Get tag value by key"""
        for tag in tags:
            if tag['Key'] == key:
                return tag['Value']
        return default
    
    async def get_cost_data(self, start_date: datetime, end_date: datetime) -> CostReport:
        """Get AWS cost data"""
        if not self.cost_explorer_client:
            return CostReport(
                period_start=start_date,
                period_end=end_date,
                total_cost=0,
                cost_by_service={},
                cost_by_region={},
                cost_optimization_recommendations=[]
            )
        
        try:
            response = self.cost_explorer_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            )
            
            total_cost = 0
            cost_by_service = {}
            
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service = group['Keys'][0]
                    amount = float(group['Metrics']['BlendedCost']['Amount'])
                    
                    cost_by_service[service] = cost_by_service.get(service, 0) + amount
                    total_cost += amount
            
            return CostReport(
                period_start=start_date,
                period_end=end_date,
                total_cost=total_cost,
                cost_by_service=cost_by_service,
                cost_by_region={},
                cost_optimization_recommendations=[
                    "Consider using Reserved Instances for steady workloads",
                    "Review and terminate unused resources",
                    "Use Spot Instances for fault-tolerant workloads"
                ]
            )
            
        except Exception as e:
            self.logger.error(f"AWS cost data retrieval failed: {e}")
            return CostReport(
                period_start=start_date,
                period_end=end_date,
                total_cost=0,
                cost_by_service={},
                cost_by_region={},
                cost_optimization_recommendations=[]
            )


class KubernetesProvider(CloudProviderInterface):
    """Kubernetes provider implementation"""
    
    def __init__(self, credentials: CloudCredentials):
        super().__init__(credentials)
        self.api_client = None
        self.apps_v1 = None
        self.core_v1 = None
        self.autoscaling_v1 = None
        
        if KUBERNETES_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Kubernetes client"""
        try:
            if 'kubeconfig_path' in self.credentials.credentials:
                config.load_kube_config(self.credentials.credentials['kubeconfig_path'])
            else:
                config.load_incluster_config()
            
            self.api_client = client.ApiClient()
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            self.autoscaling_v1 = client.AutoscalingV1Api()
            
            self.logger.info("Kubernetes client initialized")
            
        except Exception as e:
            self.logger.error(f"Kubernetes client initialization failed: {e}")
    
    async def create_resource(self, spec: ResourceSpec) -> ResourceInstance:
        """Create Kubernetes resource"""
        if not self.apps_v1:
            raise RuntimeError("Kubernetes client not initialized")
        
        try:
            if spec.type == ResourceType.COMPUTE:
                # Create deployment
                deployment = client.V1Deployment(
                    metadata=client.V1ObjectMeta(
                        name=spec.name,
                        labels=spec.tags
                    ),
                    spec=client.V1DeploymentSpec(
                        replicas=spec.desired_capacity,
                        selector=client.V1LabelSelector(
                            match_labels={'app': spec.name}
                        ),
                        template=client.V1PodTemplateSpec(
                            metadata=client.V1ObjectMeta(
                                labels={'app': spec.name}
                            ),
                            spec=client.V1PodSpec(
                                containers=[
                                    client.V1Container(
                                        name=spec.name,
                                        image=spec.config.get('image', 'nginx:latest'),
                                        ports=[client.V1ContainerPort(container_port=80)],
                                        resources=client.V1ResourceRequirements(
                                            requests={
                                                'cpu': spec.config.get('cpu_request', '100m'),
                                                'memory': spec.config.get('memory_request', '128Mi')
                                            },
                                            limits={
                                                'cpu': spec.config.get('cpu_limit', '500m'),
                                                'memory': spec.config.get('memory_limit', '256Mi')
                                            }
                                        )
                                    )
                                ]
                            )
                        )
                    )
                )
                
                result = self.apps_v1.create_namespaced_deployment(
                    namespace=spec.config.get('namespace', 'default'),
                    body=deployment
                )
                
                return ResourceInstance(
                    id=result.metadata.name,
                    name=spec.name,
                    type=spec.type,
                    provider=CloudProvider.KUBERNETES,
                    status=ResourceStatus.PENDING,
                    config=spec.config
                )
            
            else:
                raise ValueError(f"Resource type {spec.type} not supported for Kubernetes")
                
        except Exception as e:
            self.logger.error(f"Kubernetes resource creation failed: {e}")
            raise
    
    async def delete_resource(self, resource_id: str) -> bool:
        """Delete Kubernetes resource"""
        if not self.apps_v1:
            return False
        
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=resource_id,
                namespace='default'
            )
            
            self.logger.info(f"Kubernetes resource {resource_id} deletion initiated")
            return True
            
        except Exception as e:
            self.logger.error(f"Kubernetes resource deletion failed: {e}")
            return False
    
    async def scale_resource(self, resource_id: str, new_capacity: int) -> bool:
        """Scale Kubernetes resource"""
        if not self.apps_v1:
            return False
        
        try:
            # Scale deployment
            self.apps_v1.patch_namespaced_deployment_scale(
                name=resource_id,
                namespace='default',
                body={'spec': {'replicas': new_capacity}}
            )
            
            self.logger.info(f"Kubernetes resource {resource_id} scaled to {new_capacity}")
            return True
            
        except Exception as e:
            self.logger.error(f"Kubernetes resource scaling failed: {e}")
            return False
    
    async def get_resource_metrics(self, resource_id: str) -> Dict[str, float]:
        """Get Kubernetes resource metrics"""
        # This would require metrics-server or custom metrics API
        # For now, return simulated metrics
        return {
            'cpu_utilization': 50.0,
            'memory_utilization': 60.0,
            'replica_count': 3,
            'timestamp': datetime.now().timestamp()
        }
    
    async def list_resources(self, resource_type: ResourceType = None) -> List[ResourceInstance]:
        """List Kubernetes resources"""
        if not self.apps_v1:
            return []
        
        try:
            deployments = self.apps_v1.list_deployment_for_all_namespaces()
            resources = []
            
            for deployment in deployments.items:
                status = ResourceStatus.RUNNING
                if deployment.status.available_replicas != deployment.status.replicas:
                    status = ResourceStatus.PENDING
                
                resources.append(ResourceInstance(
                    id=deployment.metadata.name,
                    name=deployment.metadata.name,
                    type=ResourceType.COMPUTE,
                    provider=CloudProvider.KUBERNETES,
                    status=status,
                    config={
                        'namespace': deployment.metadata.namespace,
                        'replicas': deployment.spec.replicas,
                        'available_replicas': deployment.status.available_replicas or 0
                    },
                    created_at=deployment.metadata.creation_timestamp
                ))
            
            return resources
            
        except Exception as e:
            self.logger.error(f"Kubernetes resource listing failed: {e}")
            return []
    
    async def get_cost_data(self, start_date: datetime, end_date: datetime) -> CostReport:
        """Get Kubernetes cost data"""
        # Kubernetes doesn't have built-in cost tracking
        # This would require integration with cloud provider billing APIs
        # or third-party tools like Kubecost
        return CostReport(
            period_start=start_date,
            period_end=end_date,
            total_cost=0,
            cost_by_service={'kubernetes': 0},
            cost_by_region={},
            cost_optimization_recommendations=[
                "Right-size resource requests and limits",
                "Use horizontal pod autoscaling",
                "Consider using spot/preemptible nodes"
            ]
        )


class AutoScaler:
    """Auto-scaling engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaling_rules = {}
        self.scaling_history = deque(maxlen=1000)
        self.cooldown_periods = {}
    
    def add_scaling_rule(self, resource_id: str, rule: ScalingRule):
        """Add scaling rule"""
        if resource_id not in self.scaling_rules:
            self.scaling_rules[resource_id] = []
        
        self.scaling_rules[resource_id].append(rule)
        self.logger.info(f"Added scaling rule {rule.name} for resource {resource_id}")
    
    def remove_scaling_rule(self, resource_id: str, rule_name: str):
        """Remove scaling rule"""
        if resource_id in self.scaling_rules:
            self.scaling_rules[resource_id] = [
                rule for rule in self.scaling_rules[resource_id] 
                if rule.name != rule_name
            ]
            self.logger.info(f"Removed scaling rule {rule_name} for resource {resource_id}")
    
    async def evaluate_scaling(self, resource_id: str, metrics: Dict[str, float], 
                             current_capacity: int) -> Optional[ScalingEvent]:
        """Evaluate scaling rules"""
        if resource_id not in self.scaling_rules:
            return None
        
        # Check cooldown period
        if self._is_in_cooldown(resource_id):
            return None
        
        for rule in self.scaling_rules[resource_id]:
            if not rule.enabled:
                continue
            
            should_scale, new_capacity = await self._evaluate_rule(
                rule, metrics, current_capacity
            )
            
            if should_scale:
                scaling_event = ScalingEvent(
                    id=str(uuid.uuid4()),
                    resource_id=resource_id,
                    rule_name=rule.name,
                    direction=rule.direction,
                    old_capacity=current_capacity,
                    new_capacity=new_capacity,
                    trigger_value=metrics.get(rule.policy.value, 0),
                    threshold=rule.threshold,
                    timestamp=datetime.now(),
                    success=False  # Will be updated after scaling attempt
                )
                
                # Update cooldown
                self.cooldown_periods[resource_id] = datetime.now() + timedelta(seconds=rule.cooldown)
                
                return scaling_event
        
        return None
    
    async def _evaluate_rule(self, rule: ScalingRule, metrics: Dict[str, float], 
                           current_capacity: int) -> Tuple[bool, int]:
        """Evaluate individual scaling rule"""
        metric_value = metrics.get(rule.policy.value, 0)
        
        should_scale = False
        new_capacity = current_capacity
        
        if rule.direction in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
            if metric_value > rule.threshold:
                should_scale = True
                if rule.adjustment_type == "percentage":
                    adjustment = int(current_capacity * rule.adjustment / 100)
                else:
                    adjustment = int(rule.adjustment)
                new_capacity = current_capacity + adjustment
        
        elif rule.direction in [ScalingDirection.SCALE_DOWN, ScalingDirection.SCALE_IN]:
            if metric_value < rule.threshold:
                should_scale = True
                if rule.adjustment_type == "percentage":
                    adjustment = int(current_capacity * rule.adjustment / 100)
                else:
                    adjustment = int(rule.adjustment)
                new_capacity = max(1, current_capacity - adjustment)
        
        return should_scale, new_capacity
    
    def _is_in_cooldown(self, resource_id: str) -> bool:
        """Check if resource is in cooldown period"""
        if resource_id not in self.cooldown_periods:
            return False
        
        return datetime.now() < self.cooldown_periods[resource_id]
    
    def record_scaling_event(self, event: ScalingEvent):
        """Record scaling event"""
        self.scaling_history.append(event)
        self.logger.info(f"Scaling event recorded: {event.resource_id} {event.direction.value}")
    
    def get_scaling_history(self, resource_id: str = None, 
                           limit: int = 100) -> List[ScalingEvent]:
        """Get scaling history"""
        history = list(self.scaling_history)
        
        if resource_id:
            history = [event for event in history if event.resource_id == resource_id]
        
        return sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]


class ResourceOptimizer:
    """Resource optimization engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_strategies = {}
        self.recommendations = defaultdict(list)
    
    def register_optimization_strategy(self, name: str, strategy_func: Callable):
        """Register optimization strategy"""
        self.optimization_strategies[name] = strategy_func
        self.logger.info(f"Registered optimization strategy: {name}")
    
    async def analyze_resources(self, resources: List[ResourceInstance], 
                              strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> Dict[str, Any]:
        """Analyze resources for optimization opportunities"""
        analysis = {
            'total_resources': len(resources),
            'optimization_opportunities': [],
            'estimated_savings': 0,
            'recommendations': []
        }
        
        try:
            # Cost optimization
            if strategy in [OptimizationStrategy.COST_OPTIMIZATION, OptimizationStrategy.BALANCED]:
                cost_opportunities = await self._analyze_cost_optimization(resources)
                analysis['optimization_opportunities'].extend(cost_opportunities)
            
            # Performance optimization
            if strategy in [OptimizationStrategy.PERFORMANCE_OPTIMIZATION, OptimizationStrategy.BALANCED]:
                perf_opportunities = await self._analyze_performance_optimization(resources)
                analysis['optimization_opportunities'].extend(perf_opportunities)
            
            # Availability optimization
            if strategy in [OptimizationStrategy.AVAILABILITY_OPTIMIZATION, OptimizationStrategy.BALANCED]:
                avail_opportunities = await self._analyze_availability_optimization(resources)
                analysis['optimization_opportunities'].extend(avail_opportunities)
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(
                analysis['optimization_opportunities']
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Resource analysis failed: {e}")
            return analysis
    
    async def _analyze_cost_optimization(self, resources: List[ResourceInstance]) -> List[Dict[str, Any]]:
        """Analyze cost optimization opportunities"""
        opportunities = []
        
        # Check for oversized resources
        for resource in resources:
            if resource.metrics.get('cpu_utilization', 0) < 20:  # Low CPU utilization
                opportunities.append({
                    'type': 'cost_optimization',
                    'resource_id': resource.id,
                    'issue': 'low_cpu_utilization',
                    'current_utilization': resource.metrics.get('cpu_utilization', 0),
                    'recommendation': 'Downsize instance or use auto-scaling',
                    'estimated_savings': 30  # percentage
                })
            
            if resource.status == ResourceStatus.STOPPED:
                opportunities.append({
                    'type': 'cost_optimization',
                    'resource_id': resource.id,
                    'issue': 'unused_resource',
                    'recommendation': 'Terminate unused resource',
                    'estimated_savings': 100
                })
        
        return opportunities
    
    async def _analyze_performance_optimization(self, resources: List[ResourceInstance]) -> List[Dict[str, Any]]:
        """Analyze performance optimization opportunities"""
        opportunities = []
        
        for resource in resources:
            if resource.metrics.get('cpu_utilization', 0) > 80:  # High CPU utilization
                opportunities.append({
                    'type': 'performance_optimization',
                    'resource_id': resource.id,
                    'issue': 'high_cpu_utilization',
                    'current_utilization': resource.metrics.get('cpu_utilization', 0),
                    'recommendation': 'Upgrade instance or scale horizontally'
                })
            
            if resource.metrics.get('memory_utilization', 0) > 85:  # High memory utilization
                opportunities.append({
                    'type': 'performance_optimization',
                    'resource_id': resource.id,
                    'issue': 'high_memory_utilization',
                    'current_utilization': resource.metrics.get('memory_utilization', 0),
                    'recommendation': 'Increase memory or optimize application'
                })
        
        return opportunities
    
    async def _analyze_availability_optimization(self, resources: List[ResourceInstance]) -> List[Dict[str, Any]]:
        """Analyze availability optimization opportunities"""
        opportunities = []
        
        # Check for single points of failure
        resource_counts_by_type = defaultdict(int)
        for resource in resources:
            if resource.status == ResourceStatus.RUNNING:
                resource_counts_by_type[resource.type] += 1
        
        for resource_type, count in resource_counts_by_type.items():
            if count == 1:
                opportunities.append({
                    'type': 'availability_optimization',
                    'resource_type': resource_type.value,
                    'issue': 'single_point_of_failure',
                    'current_count': count,
                    'recommendation': 'Deploy multiple instances across availability zones'
                })
        
        return opportunities
    
    def _generate_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Group opportunities by type
        by_type = defaultdict(list)
        for opp in opportunities:
            by_type[opp['type']].append(opp)
        
        # Cost recommendations
        if 'cost_optimization' in by_type:
            cost_opps = by_type['cost_optimization']
            total_savings = sum(opp.get('estimated_savings', 0) for opp in cost_opps)
            recommendations.append(f"Implement cost optimizations to save approximately {total_savings}% on infrastructure costs")
        
        # Performance recommendations
        if 'performance_optimization' in by_type:
            perf_opps = by_type['performance_optimization']
            recommendations.append(f"Address {len(perf_opps)} performance bottlenecks to improve system responsiveness")
        
        # Availability recommendations
        if 'availability_optimization' in by_type:
            avail_opps = by_type['availability_optimization']
            recommendations.append(f"Improve availability by addressing {len(avail_opps)} single points of failure")
        
        return recommendations


class InfrastructureEngine:
    """
    Comprehensive infrastructure management engine with auto-scaling,
    resource optimization, cloud integration, and hybrid infrastructure support.
    """
    
    def __init__(self, db_path: str = "infrastructure.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Provider management
        self.cloud_providers = {}
        self.active_resources = {}
        self.resource_metrics = defaultdict(dict)
        
        # Performance tracking
        self.performance_metrics = {
            'scaling_operations': deque(maxlen=100),
            'resource_provisions': deque(maxlen=100),
            'optimization_runs': deque(maxlen=100),
            'total_resources': 0,
            'active_scaling_rules': 0
        }
        
        # Initialize components
        self.auto_scaler = AutoScaler()
        self.resource_optimizer = ResourceOptimizer()
        
        # Initialize database
        self._initialize_database()
        
        # Integration references
        self.monitoring_engine = None
        self.alerting_engine = None
        self.deployment_engine = None
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("InfrastructureEngine initialized successfully")
    
    def _initialize_database(self):
        """Initialize infrastructure database"""
        if DATABASE_AVAILABLE and sqlite3:
            try:
                self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._create_tables()
                self.logger.info("Infrastructure database initialized")
            except Exception as e:
                self.logger.error(f"Database initialization failed: {e}")
                self.connection = None
    
    def _create_tables(self):
        """Create database tables"""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        
        # Resources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                provider TEXT NOT NULL,
                status TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Scaling events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scaling_events (
                id TEXT PRIMARY KEY,
                resource_id TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                old_capacity INTEGER NOT NULL,
                new_capacity INTEGER NOT NULL,
                trigger_value REAL NOT NULL,
                threshold REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Cost data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                service TEXT NOT NULL,
                cost REAL NOT NULL,
                currency TEXT NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
    
    def set_integrations(self, monitoring_engine=None, alerting_engine=None, deployment_engine=None):
        """Set integration references"""
        self.monitoring_engine = monitoring_engine
        self.alerting_engine = alerting_engine
        self.deployment_engine = deployment_engine
    
    def register_cloud_provider(self, name: str, provider: CloudProviderInterface):
        """Register cloud provider"""
        self.cloud_providers[name] = provider
        self.logger.info(f"Registered cloud provider: {name}")
    
    def _start_background_tasks(self):
        """Start background infrastructure tasks"""
        # Start auto-scaling task
        threading.Thread(target=self._auto_scaling_task, daemon=True).start()
        
        # Start resource monitoring task
        threading.Thread(target=self._resource_monitoring_task, daemon=True).start()
        
        # Start optimization task
        threading.Thread(target=self._optimization_task, daemon=True).start()
        
        # Start cost tracking task
        threading.Thread(target=self._cost_tracking_task, daemon=True).start()
    
    def _auto_scaling_task(self):
        """Background auto-scaling task"""
        while True:
            try:
                # Check all resources for scaling opportunities
                for resource_id, resource in self.active_resources.items():
                    if resource.status != ResourceStatus.RUNNING:
                        continue
                    
                    metrics = self.resource_metrics.get(resource_id, {})
                    if not metrics:
                        continue
                    
                    current_capacity = resource.config.get('current_capacity', 1)
                    
                    scaling_event = await self.auto_scaler.evaluate_scaling(
                        resource_id, metrics, current_capacity
                    )
                    
                    if scaling_event:
                        success = await self._execute_scaling(resource_id, scaling_event.new_capacity)
                        scaling_event.success = success
                        
                        if not success:
                            scaling_event.error = "Scaling operation failed"
                        
                        self.auto_scaler.record_scaling_event(scaling_event)
                        self._store_scaling_event(scaling_event)
                        
                        # Send alert if scaling failed
                        if not success and self.alerting_engine:
                            await self.alerting_engine.trigger_alert(
                                name="Infrastructure Scaling Failed",
                                severity="high",
                                description=f"Failed to scale resource {resource_id}",
                                labels={'component': 'infrastructure', 'resource_id': resource_id}
                            )
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Auto-scaling task error: {e}")
                time.sleep(60)
    
    def _resource_monitoring_task(self):
        """Background resource monitoring task"""
        while True:
            try:
                # Update resource metrics from providers
                for provider_name, provider in self.cloud_providers.items():
                    resources = await provider.list_resources()
                    
                    for resource in resources:
                        # Update resource status
                        if resource.id in self.active_resources:
                            self.active_resources[resource.id].status = resource.status
                            self.active_resources[resource.id].updated_at = datetime.now()
                        
                        # Get fresh metrics
                        metrics = await provider.get_resource_metrics(resource.id)
                        if metrics:
                            self.resource_metrics[resource.id] = metrics
                
                # Update performance metrics
                self.performance_metrics['total_resources'] = len(self.active_resources)
                self.performance_metrics['active_scaling_rules'] = sum(
                    len(rules) for rules in self.auto_scaler.scaling_rules.values()
                )
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Resource monitoring task error: {e}")
                time.sleep(300)
    
    def _optimization_task(self):
        """Background optimization task"""
        while True:
            try:
                # Run optimization analysis
                all_resources = list(self.active_resources.values())
                
                if all_resources:
                    analysis = await self.resource_optimizer.analyze_resources(all_resources)
                    
                    # Log optimization opportunities
                    if analysis['optimization_opportunities']:
                        self.logger.info(f"Found {len(analysis['optimization_opportunities'])} optimization opportunities")
                        
                        # Send optimization recommendations if alerting is available
                        if self.alerting_engine and analysis['recommendations']:
                            for recommendation in analysis['recommendations']:
                                await self.alerting_engine.trigger_alert(
                                    name="Infrastructure Optimization Opportunity",
                                    severity="info",
                                    description=recommendation,
                                    labels={'component': 'infrastructure', 'type': 'optimization'}
                                )
                    
                    self.performance_metrics['optimization_runs'].append(time.time())
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Optimization task error: {e}")
                time.sleep(3600)
    
    def _cost_tracking_task(self):
        """Background cost tracking task"""
        while True:
            try:
                # Get cost data from all providers
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)  # Last 24 hours
                
                for provider_name, provider in self.cloud_providers.items():
                    cost_report = await provider.get_cost_data(start_date, end_date)
                    
                    if cost_report.total_cost > 0:
                        self._store_cost_data(provider_name, cost_report)
                        
                        self.logger.info(f"Cost data updated for {provider_name}: ${cost_report.total_cost:.2f}")
                
                time.sleep(86400)  # Run daily
                
            except Exception as e:
                self.logger.error(f"Cost tracking task error: {e}")
                time.sleep(86400)
    
    async def create_resource(self, spec: ResourceSpec, provider_name: str) -> str:
        """Create infrastructure resource"""
        start_time = time.time()
        
        if provider_name not in self.cloud_providers:
            raise ValueError(f"Cloud provider {provider_name} not found")
        
        provider = self.cloud_providers[provider_name]
        
        try:
            resource = await provider.create_resource(spec)
            
            # Store resource
            self.active_resources[resource.id] = resource
            self._store_resource_in_db(resource)
            
            # Update performance metrics
            provision_time = time.time() - start_time
            self.performance_metrics['resource_provisions'].append(provision_time)
            
            self.logger.info(f"Resource created: {resource.id} ({spec.name})")
            return resource.id
            
        except Exception as e:
            self.logger.error(f"Resource creation failed: {e}")
            raise
    
    async def delete_resource(self, resource_id: str) -> bool:
        """Delete infrastructure resource"""
        if resource_id not in self.active_resources:
            raise ValueError(f"Resource {resource_id} not found")
        
        resource = self.active_resources[resource_id]
        
        # Find appropriate provider
        provider = None
        for p in self.cloud_providers.values():
            if p.credentials.provider == resource.provider:
                provider = p
                break
        
        if not provider:
            self.logger.error(f"No provider found for resource {resource_id}")
            return False
        
        try:
            success = await provider.delete_resource(resource_id)
            
            if success:
                # Remove from active resources
                del self.active_resources[resource_id]
                
                # Remove metrics
                if resource_id in self.resource_metrics:
                    del self.resource_metrics[resource_id]
                
                # Remove scaling rules
                if resource_id in self.auto_scaler.scaling_rules:
                    del self.auto_scaler.scaling_rules[resource_id]
                
                self.logger.info(f"Resource deleted: {resource_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Resource deletion failed: {e}")
            return False
    
    async def _execute_scaling(self, resource_id: str, new_capacity: int) -> bool:
        """Execute scaling operation"""
        if resource_id not in self.active_resources:
            return False
        
        resource = self.active_resources[resource_id]
        
        # Find appropriate provider
        provider = None
        for p in self.cloud_providers.values():
            if p.credentials.provider == resource.provider:
                provider = p
                break
        
        if not provider:
            return False
        
        try:
            start_time = time.time()
            success = await provider.scale_resource(resource_id, new_capacity)
            
            if success:
                # Update resource configuration
                resource.config['current_capacity'] = new_capacity
                resource.updated_at = datetime.now()
                
                # Update performance metrics
                scaling_time = time.time() - start_time
                self.performance_metrics['scaling_operations'].append(scaling_time)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Scaling operation failed: {e}")
            return False
    
    def add_auto_scaling_rule(self, resource_id: str, rule: ScalingRule):
        """Add auto-scaling rule"""
        if resource_id not in self.active_resources:
            raise ValueError(f"Resource {resource_id} not found")
        
        self.auto_scaler.add_scaling_rule(resource_id, rule)
    
    def remove_auto_scaling_rule(self, resource_id: str, rule_name: str):
        """Remove auto-scaling rule"""
        self.auto_scaler.remove_scaling_rule(resource_id, rule_name)
    
    async def get_resource_status(self, resource_id: str) -> Dict[str, Any]:
        """Get resource status"""
        if resource_id not in self.active_resources:
            raise ValueError(f"Resource {resource_id} not found")
        
        resource = self.active_resources[resource_id]
        metrics = self.resource_metrics.get(resource_id, {})
        
        return {
            'id': resource.id,
            'name': resource.name,
            'type': resource.type.value,
            'provider': resource.provider.value,
            'status': resource.status.value,
            'config': resource.config,
            'metrics': metrics,
            'created_at': resource.created_at.isoformat(),
            'updated_at': resource.updated_at.isoformat()
        }
    
    def list_resources(self, provider: CloudProvider = None, 
                      resource_type: ResourceType = None) -> List[Dict[str, Any]]:
        """List infrastructure resources"""
        resources = []
        
        for resource in self.active_resources.values():
            # Apply filters
            if provider and resource.provider != provider:
                continue
            if resource_type and resource.type != resource_type:
                continue
            
            resources.append({
                'id': resource.id,
                'name': resource.name,
                'type': resource.type.value,
                'provider': resource.provider.value,
                'status': resource.status.value,
                'created_at': resource.created_at.isoformat(),
                'updated_at': resource.updated_at.isoformat()
            })
        
        return sorted(resources, key=lambda x: x['created_at'], reverse=True)
    
    async def get_optimization_analysis(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> Dict[str, Any]:
        """Get infrastructure optimization analysis"""
        resources = list(self.active_resources.values())
        return await self.resource_optimizer.analyze_resources(resources, strategy)
    
    async def get_cost_report(self, provider_name: str = None, 
                            days: int = 30) -> Dict[str, Any]:
        """Get cost report"""
        if not self.connection:
            return {'error': 'Database not available'}
        
        try:
            cursor = self.connection.cursor()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = """
                SELECT provider, service, SUM(cost) as total_cost, currency
                FROM cost_data 
                WHERE period_start >= ? AND period_end <= ?
            """
            params = [start_date, end_date]
            
            if provider_name:
                query += " AND provider = ?"
                params.append(provider_name)
            
            query += " GROUP BY provider, service, currency"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            cost_by_provider = defaultdict(float)
            cost_by_service = defaultdict(float)
            total_cost = 0
            
            for row in rows:
                provider, service, cost, currency = row
                cost_by_provider[provider] += cost
                cost_by_service[service] += cost
                total_cost += cost
            
            return {
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_cost': total_cost,
                'cost_by_provider': dict(cost_by_provider),
                'cost_by_service': dict(cost_by_service),
                'currency': 'USD'
            }
            
        except Exception as e:
            self.logger.error(f"Cost report generation failed: {e}")
            return {'error': str(e)}
    
    def get_scaling_history(self, resource_id: str = None) -> List[Dict[str, Any]]:
        """Get auto-scaling history"""
        events = self.auto_scaler.get_scaling_history(resource_id)
        
        return [
            {
                'id': event.id,
                'resource_id': event.resource_id,
                'rule_name': event.rule_name,
                'direction': event.direction.value,
                'old_capacity': event.old_capacity,
                'new_capacity': event.new_capacity,
                'trigger_value': event.trigger_value,
                'threshold': event.threshold,
                'timestamp': event.timestamp.isoformat(),
                'success': event.success,
                'error': event.error
            }
            for event in events
        ]
    
    def get_infrastructure_metrics(self) -> Dict[str, Any]:
        """Get infrastructure engine metrics"""
        avg_scaling_time = (
            statistics.mean(self.performance_metrics['scaling_operations'])
            if self.performance_metrics['scaling_operations'] else 0
        )
        
        avg_provision_time = (
            statistics.mean(self.performance_metrics['resource_provisions'])
            if self.performance_metrics['resource_provisions'] else 0
        )
        
        return {
            'total_resources': self.performance_metrics['total_resources'],
            'active_scaling_rules': self.performance_metrics['active_scaling_rules'],
            'registered_providers': len(self.cloud_providers),
            'avg_scaling_time': avg_scaling_time,
            'avg_provision_time': avg_provision_time,
            'scaling_operations_count': len(self.performance_metrics['scaling_operations']),
            'resource_provisions_count': len(self.performance_metrics['resource_provisions']),
            'optimization_runs_count': len(self.performance_metrics['optimization_runs'])
        }
    
    def _store_resource_in_db(self, resource: ResourceInstance):
        """Store resource in database"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO resources (id, name, type, provider, status, config)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                resource.id,
                resource.name,
                resource.type.value,
                resource.provider.value,
                resource.status.value,
                json.dumps(resource.config)
            ))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store resource in database: {e}")
    
    def _store_scaling_event(self, event: ScalingEvent):
        """Store scaling event in database"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO scaling_events (id, resource_id, rule_name, direction, old_capacity, 
                                          new_capacity, trigger_value, threshold, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.resource_id,
                event.rule_name,
                event.direction.value,
                event.old_capacity,
                event.new_capacity,
                event.trigger_value,
                event.threshold,
                event.success,
                event.error
            ))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store scaling event in database: {e}")
    
    def _store_cost_data(self, provider_name: str, cost_report: CostReport):
        """Store cost data in database"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            
            for service, cost in cost_report.cost_by_service.items():
                cursor.execute("""
                    INSERT INTO cost_data (provider, service, cost, currency, period_start, period_end)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    provider_name,
                    service,
                    cost,
                    cost_report.currency,
                    cost_report.period_start,
                    cost_report.period_end
                ))
            
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store cost data in database: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close database connection
            if self.connection:
                self.connection.close()
            
            self.logger.info("InfrastructureEngine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"InfrastructureEngine cleanup error: {e}")


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create infrastructure engine
        engine = InfrastructureEngine()
        
        # Register Kubernetes provider (if available)
        if KUBERNETES_AVAILABLE:
            k8s_credentials = CloudCredentials(
                provider=CloudProvider.KUBERNETES,
                credentials={'kubeconfig_path': '~/.kube/config'}
            )
            k8s_provider = KubernetesProvider(k8s_credentials)
            engine.register_cloud_provider('kubernetes', k8s_provider)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        try:
            # Create a test resource spec
            test_spec = ResourceSpec(
                name="test-deployment",
                type=ResourceType.COMPUTE,
                provider=CloudProvider.KUBERNETES,
                config={
                    'image': 'nginx:latest',
                    'namespace': 'default',
                    'cpu_request': '100m',
                    'memory_request': '128Mi'
                },
                tags={'environment': 'test', 'application': 'manufacturing-line'},
                min_capacity=1,
                max_capacity=5,
                desired_capacity=2
            )
            
            # Create resource (if Kubernetes is available)
            if KUBERNETES_AVAILABLE:
                resource_id = await engine.create_resource(test_spec, 'kubernetes')
                print(f"Test resource created: {resource_id}")
                
                # Add auto-scaling rule
                scaling_rule = ScalingRule(
                    name="cpu_scale_up",
                    policy=ScalingPolicy.CPU_UTILIZATION,
                    direction=ScalingDirection.SCALE_OUT,
                    threshold=70.0,
                    duration=300,
                    cooldown=600,
                    adjustment=1,
                    adjustment_type="absolute"
                )
                
                engine.add_auto_scaling_rule(resource_id, scaling_rule)
                print(f"Added scaling rule to resource: {resource_id}")
            
            # Get infrastructure metrics
            metrics = engine.get_infrastructure_metrics()
            print(f"Infrastructure metrics: {metrics}")
            
            # List resources
            resources = engine.list_resources()
            print(f"Total resources: {len(resources)}")
            
            # Get optimization analysis
            optimization = await engine.get_optimization_analysis()
            print(f"Optimization opportunities: {len(optimization['optimization_opportunities'])}")
            
            if optimization['recommendations']:
                print("Optimization recommendations:")
                for rec in optimization['recommendations']:
                    print(f"  - {rec}")
            
            # Get cost report
            cost_report = await engine.get_cost_report()
            print(f"Total infrastructure cost: ${cost_report.get('total_cost', 0):.2f}")
            
        except Exception as e:
            print(f"Infrastructure engine test failed: {e}")
        
        finally:
            await engine.cleanup()
    
    asyncio.run(main())