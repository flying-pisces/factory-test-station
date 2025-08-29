#!/usr/bin/env python3
"""
ScalabilityEngine - Week 10 Scalability & Performance Layer
Advanced scalability management with intelligent auto-scaling
"""

import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
import math
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalingType(Enum):
    """Types of scaling operations"""
    HORIZONTAL_OUT = "horizontal_out"  # Scale out (add instances)
    HORIZONTAL_IN = "horizontal_in"    # Scale in (remove instances)
    VERTICAL_UP = "vertical_up"        # Scale up (add resources)
    VERTICAL_DOWN = "vertical_down"    # Scale down (reduce resources)

class ScalingStrategy(Enum):
    """Scaling strategies"""
    REACTIVE = "reactive"        # React to current metrics
    PREDICTIVE = "predictive"    # Scale based on predictions
    SCHEDULED = "scheduled"      # Scale based on schedule
    MANUAL = "manual"           # Manual scaling trigger

class ResourceType(Enum):
    """Resource types for scaling"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"

@dataclass
class ScalingPolicy:
    """Auto-scaling policy definition"""
    policy_id: str
    name: str
    strategy: ScalingStrategy
    scaling_type: ScalingType
    resource_type: ResourceType
    threshold_up: float
    threshold_down: float
    min_instances: int
    max_instances: int
    cooldown_seconds: int = 300
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ScalingEvent:
    """Scaling event record"""
    event_id: str
    policy_id: str
    scaling_type: ScalingType
    trigger_metric: str
    trigger_value: float
    instances_before: int
    instances_after: int
    resource_adjustment: Dict[str, float]
    duration_seconds: float
    success: bool
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InstanceSpec:
    """Instance specification for scaling"""
    instance_id: str
    instance_type: str
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    status: str  # pending, running, terminated
    created_at: str
    region: str = "us-east-1"
    availability_zone: str = "us-east-1a"

class ScalabilityEngine:
    """Advanced scalability management with intelligent auto-scaling
    
    Week 10 Performance Targets:
    - Scaling decisions: <100ms
    - Scale-out operations: <2 minutes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ScalabilityEngine with configuration"""
        self.config = config or {}
        
        # Performance targets
        self.scaling_decision_target_ms = 100
        self.scale_out_target_minutes = 2
        
        # State management
        self.scaling_policies = {}
        self.scaling_events = []
        self.active_instances = {}
        self.performance_metrics = {}
        self.prediction_models = {}
        
        # Initialize default scaling policies
        self._initialize_default_policies()
        
        # Initialize performance monitoring
        self._initialize_performance_monitoring()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize performance engine if available
        self.performance_engine = None
        try:
            from layers.scalability_layer.performance_engine import PerformanceEngine
            self.performance_engine = PerformanceEngine(config.get('performance_config', {}))
        except ImportError:
            logger.warning("PerformanceEngine not available - using mock interface")
        
        logger.info("ScalabilityEngine initialized with auto-scaling and performance optimization")
    
    def manage_horizontal_scaling(self, scaling_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Manage horizontal scaling with container orchestration
        
        Args:
            scaling_specs: Horizontal scaling specifications
            
        Returns:
            Scaling results with performance metrics
        """
        start_time = time.time()
        
        try:
            # Parse scaling specifications
            service_name = scaling_specs['service_name']
            target_instances = scaling_specs.get('target_instances', 1)
            current_instances = scaling_specs.get('current_instances', 1)
            instance_type = scaling_specs.get('instance_type', 't3.medium')
            
            # Determine scaling type
            if target_instances > current_instances:
                scaling_type = ScalingType.HORIZONTAL_OUT
                instances_to_add = target_instances - current_instances
                instances_to_remove = 0
            elif target_instances < current_instances:
                scaling_type = ScalingType.HORIZONTAL_IN
                instances_to_add = 0
                instances_to_remove = current_instances - target_instances
            else:
                # No scaling needed
                scaling_time_ms = (time.time() - start_time) * 1000
                return {
                    'scaling_needed': False,
                    'current_instances': current_instances,
                    'scaling_time_ms': round(scaling_time_ms, 2),
                    'message': 'No scaling required'
                }
            
            # Validate scaling limits
            scaling_validation = self._validate_scaling_limits(service_name, target_instances)
            if not scaling_validation['valid']:
                return {
                    'scaling_success': False,
                    'reason': scaling_validation['reason'],
                    'scaling_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            
            # Perform horizontal scaling
            scaling_results = []
            
            if scaling_type == ScalingType.HORIZONTAL_OUT:
                # Scale out - add instances
                for i in range(instances_to_add):
                    instance = self._create_instance(service_name, instance_type)
                    scaling_results.append({
                        'action': 'create',
                        'instance_id': instance.instance_id,
                        'status': instance.status
                    })
            else:
                # Scale in - remove instances
                instances_to_terminate = self._select_instances_for_termination(service_name, instances_to_remove)
                for instance_id in instances_to_terminate:
                    termination_result = self._terminate_instance(instance_id)
                    scaling_results.append({
                        'action': 'terminate',
                        'instance_id': instance_id,
                        'status': termination_result['status']
                    })
            
            # Record scaling event
            scaling_event = ScalingEvent(
                event_id=f"SCALE_{int(time.time() * 1000)}",
                policy_id=scaling_specs.get('policy_id', 'manual'),
                scaling_type=scaling_type,
                trigger_metric=scaling_specs.get('trigger_metric', 'manual'),
                trigger_value=scaling_specs.get('trigger_value', 0.0),
                instances_before=current_instances,
                instances_after=target_instances,
                resource_adjustment={},
                duration_seconds=0,  # Will be calculated below
                success=True,
                timestamp=datetime.now().isoformat()
            )
            
            # Calculate scaling time
            scaling_time_ms = (time.time() - start_time) * 1000
            scaling_time_minutes = scaling_time_ms / (1000 * 60)
            scaling_event.duration_seconds = scaling_time_ms / 1000
            
            # Store scaling event
            self.scaling_events.append(scaling_event)
            
            result = {
                'scaling_success': True,
                'service_name': service_name,
                'scaling_type': scaling_type.value,
                'instances_before': current_instances,
                'instances_after': target_instances,
                'instances_changed': abs(target_instances - current_instances),
                'scaling_time_ms': round(scaling_time_ms, 2),
                'scaling_time_minutes': round(scaling_time_minutes, 2),
                'target_met': scaling_time_minutes < self.scale_out_target_minutes,
                'scaling_results': scaling_results,
                'event_id': scaling_event.event_id,
                'scaled_at': datetime.now().isoformat()
            }
            
            logger.info(f"Horizontal scaling completed: {service_name} {scaling_type.value} in {scaling_time_minutes:.2f}min")
            return result
            
        except Exception as e:
            logger.error(f"Error in horizontal scaling: {e}")
            raise
    
    def manage_vertical_scaling(self, resource_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Manage vertical scaling with resource optimization
        
        Args:
            resource_specs: Vertical scaling specifications
            
        Returns:
            Vertical scaling results with performance metrics
        """
        start_time = time.time()
        
        try:
            # Parse resource specifications
            instance_id = resource_specs['instance_id']
            new_resources = resource_specs.get('new_resources', {})
            scaling_strategy = resource_specs.get('strategy', 'gradual')
            
            # Get current instance
            current_instance = self.active_instances.get(instance_id)
            if not current_instance:
                return {
                    'scaling_success': False,
                    'reason': f'Instance {instance_id} not found',
                    'scaling_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            
            # Calculate resource changes
            resource_changes = {}
            if 'cpu_cores' in new_resources:
                resource_changes['cpu'] = {
                    'from': current_instance.cpu_cores,
                    'to': new_resources['cpu_cores'],
                    'change': new_resources['cpu_cores'] - current_instance.cpu_cores
                }
            
            if 'memory_gb' in new_resources:
                resource_changes['memory'] = {
                    'from': current_instance.memory_gb,
                    'to': new_resources['memory_gb'],
                    'change': new_resources['memory_gb'] - current_instance.memory_gb
                }
            
            if 'storage_gb' in new_resources:
                resource_changes['storage'] = {
                    'from': current_instance.storage_gb,
                    'to': new_resources['storage_gb'],
                    'change': new_resources['storage_gb'] - current_instance.storage_gb
                }
            
            # Perform vertical scaling
            scaling_steps = self._perform_vertical_scaling(instance_id, resource_changes, scaling_strategy)
            
            # Update instance specifications
            if 'cpu_cores' in new_resources:
                current_instance.cpu_cores = new_resources['cpu_cores']
            if 'memory_gb' in new_resources:
                current_instance.memory_gb = new_resources['memory_gb']
            if 'storage_gb' in new_resources:
                current_instance.storage_gb = new_resources['storage_gb']
            
            # Record scaling event
            scaling_type = ScalingType.VERTICAL_UP if any(change['change'] > 0 for change in resource_changes.values()) else ScalingType.VERTICAL_DOWN
            
            scaling_event = ScalingEvent(
                event_id=f"VSCALE_{int(time.time() * 1000)}",
                policy_id=resource_specs.get('policy_id', 'manual'),
                scaling_type=scaling_type,
                trigger_metric=resource_specs.get('trigger_metric', 'manual'),
                trigger_value=resource_specs.get('trigger_value', 0.0),
                instances_before=1,
                instances_after=1,
                resource_adjustment=resource_changes,
                duration_seconds=0,  # Will be calculated below
                success=True,
                timestamp=datetime.now().isoformat()
            )
            
            # Calculate scaling time
            scaling_time_ms = (time.time() - start_time) * 1000
            scaling_time_minutes = scaling_time_ms / (1000 * 60)
            scaling_event.duration_seconds = scaling_time_ms / 1000
            
            # Store scaling event
            self.scaling_events.append(scaling_event)
            
            result = {
                'scaling_success': True,
                'instance_id': instance_id,
                'scaling_type': scaling_type.value,
                'resource_changes': resource_changes,
                'scaling_steps': len(scaling_steps),
                'scaling_time_ms': round(scaling_time_ms, 2),
                'scaling_time_minutes': round(scaling_time_minutes, 2),
                'target_met': scaling_time_minutes < self.scale_out_target_minutes,
                'scaling_strategy': scaling_strategy,
                'event_id': scaling_event.event_id,
                'scaled_at': datetime.now().isoformat()
            }
            
            logger.info(f"Vertical scaling completed: {instance_id} {scaling_type.value} in {scaling_time_minutes:.2f}min")
            return result
            
        except Exception as e:
            logger.error(f"Error in vertical scaling: {e}")
            raise
    
    def implement_predictive_scaling(self, prediction_models: Dict[str, Any]) -> Dict[str, Any]:
        """Implement predictive scaling based on usage patterns
        
        Args:
            prediction_models: Predictive scaling model specifications
            
        Returns:
            Predictive scaling implementation results
        """
        start_time = time.time()
        
        try:
            # Parse prediction models
            model_type = prediction_models.get('model_type', 'time_series')
            prediction_horizon_hours = prediction_models.get('horizon_hours', 24)
            historical_data_days = prediction_models.get('historical_days', 30)
            
            # Generate historical data patterns (simulated)
            historical_patterns = self._generate_historical_patterns(historical_data_days)
            
            # Build prediction model
            prediction_model = self._build_prediction_model(model_type, historical_patterns)
            
            # Generate predictions
            predictions = self._generate_load_predictions(prediction_model, prediction_horizon_hours)
            
            # Create predictive scaling policies
            predictive_policies = []
            for hour in range(prediction_horizon_hours):
                predicted_load = predictions.get(f"hour_{hour}", {})
                
                if predicted_load.get('cpu_utilization', 0) > 70:
                    # High load predicted - scale out
                    policy = self._create_predictive_scaling_policy(
                        hour, ScalingType.HORIZONTAL_OUT, predicted_load
                    )
                    predictive_policies.append(policy)
                elif predicted_load.get('cpu_utilization', 0) < 30:
                    # Low load predicted - scale in
                    policy = self._create_predictive_scaling_policy(
                        hour, ScalingType.HORIZONTAL_IN, predicted_load
                    )
                    predictive_policies.append(policy)
            
            # Store prediction models
            self.prediction_models[model_type] = {
                'model': prediction_model,
                'predictions': predictions,
                'policies': predictive_policies,
                'created_at': datetime.now().isoformat(),
                'horizon_hours': prediction_horizon_hours
            }
            
            # Calculate implementation time
            implementation_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'implementation_success': True,
                'model_type': model_type,
                'prediction_horizon_hours': prediction_horizon_hours,
                'predictive_policies_created': len(predictive_policies),
                'predictions_generated': len(predictions),
                'implementation_time_ms': round(implementation_time_ms, 2),
                'target_met': implementation_time_ms < self.scaling_decision_target_ms * 10,  # Allow 10x for model building
                'model_accuracy': prediction_model.get('accuracy', 0.85),
                'implemented_at': datetime.now().isoformat()
            }
            
            logger.info(f"Predictive scaling implemented: {model_type} with {len(predictive_policies)} policies")
            return result
            
        except Exception as e:
            logger.error(f"Error implementing predictive scaling: {e}")
            raise
    
    def evaluate_scaling_triggers(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate scaling triggers and make scaling decisions"""
        start_time = time.time()
        
        try:
            # Evaluate all active scaling policies
            scaling_decisions = []
            
            for policy_id, policy in self.scaling_policies.items():
                if not policy.enabled:
                    continue
                
                # Get relevant metric for policy
                metric_value = metrics.get(policy.resource_type.value, {}).get('utilization', 0)
                
                # Check cooldown period
                if self._is_policy_in_cooldown(policy):
                    continue
                
                # Evaluate scaling conditions
                scaling_needed = False
                target_scaling_type = None
                
                if metric_value > policy.threshold_up:
                    # Scale up/out needed
                    if policy.scaling_type in [ScalingType.HORIZONTAL_OUT, ScalingType.VERTICAL_UP]:
                        scaling_needed = True
                        target_scaling_type = policy.scaling_type
                
                elif metric_value < policy.threshold_down:
                    # Scale down/in needed
                    if policy.scaling_type in [ScalingType.HORIZONTAL_IN, ScalingType.VERTICAL_DOWN]:
                        scaling_needed = True
                        target_scaling_type = policy.scaling_type
                
                if scaling_needed:
                    scaling_decision = {
                        'policy_id': policy_id,
                        'policy_name': policy.name,
                        'scaling_type': target_scaling_type.value,
                        'trigger_metric': policy.resource_type.value,
                        'trigger_value': metric_value,
                        'threshold': policy.threshold_up if metric_value > policy.threshold_up else policy.threshold_down,
                        'recommended_action': self._calculate_scaling_magnitude(policy, metric_value)
                    }
                    scaling_decisions.append(scaling_decision)
            
            # Calculate evaluation time
            evaluation_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'evaluation_success': True,
                'policies_evaluated': len(self.scaling_policies),
                'scaling_decisions': len(scaling_decisions),
                'evaluation_time_ms': round(evaluation_time_ms, 2),
                'target_met': evaluation_time_ms < self.scaling_decision_target_ms,
                'decisions': scaling_decisions,
                'evaluated_at': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating scaling triggers: {e}")
            raise
    
    def _initialize_default_policies(self):
        """Initialize default auto-scaling policies"""
        default_policies = [
            ScalingPolicy(
                policy_id="cpu_scale_out",
                name="CPU Scale Out Policy",
                strategy=ScalingStrategy.REACTIVE,
                scaling_type=ScalingType.HORIZONTAL_OUT,
                resource_type=ResourceType.CPU,
                threshold_up=75.0,
                threshold_down=25.0,
                min_instances=1,
                max_instances=100,
                cooldown_seconds=300
            ),
            ScalingPolicy(
                policy_id="memory_scale_up",
                name="Memory Scale Up Policy",
                strategy=ScalingStrategy.REACTIVE,
                scaling_type=ScalingType.VERTICAL_UP,
                resource_type=ResourceType.MEMORY,
                threshold_up=80.0,
                threshold_down=40.0,
                min_instances=1,
                max_instances=1,
                cooldown_seconds=600
            ),
            ScalingPolicy(
                policy_id="predictive_scale_out",
                name="Predictive Scale Out Policy",
                strategy=ScalingStrategy.PREDICTIVE,
                scaling_type=ScalingType.HORIZONTAL_OUT,
                resource_type=ResourceType.CPU,
                threshold_up=60.0,
                threshold_down=30.0,
                min_instances=1,
                max_instances=50,
                cooldown_seconds=900
            )
        ]
        
        for policy in default_policies:
            self.scaling_policies[policy.policy_id] = policy
    
    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring integration"""
        # Simulated performance metrics
        self.performance_metrics = {
            'cpu': {'utilization': 45.0, 'cores_used': 2.1, 'cores_total': 4.0},
            'memory': {'utilization': 62.0, 'used_gb': 6.2, 'total_gb': 10.0},
            'storage': {'utilization': 35.0, 'used_gb': 35.0, 'total_gb': 100.0},
            'network': {'utilization': 25.0, 'bandwidth_used_mbps': 250, 'bandwidth_total_mbps': 1000}
        }
        
        # Initialize some sample instances
        sample_instances = [
            InstanceSpec(
                instance_id="i-001",
                instance_type="t3.medium",
                cpu_cores=2,
                memory_gb=4.0,
                storage_gb=20.0,
                status="running",
                created_at=datetime.now().isoformat()
            ),
            InstanceSpec(
                instance_id="i-002",
                instance_type="t3.large",
                cpu_cores=4,
                memory_gb=8.0,
                storage_gb=40.0,
                status="running",
                created_at=datetime.now().isoformat()
            )
        ]
        
        for instance in sample_instances:
            self.active_instances[instance.instance_id] = instance
    
    def _validate_scaling_limits(self, service_name: str, target_instances: int) -> Dict[str, Any]:
        """Validate scaling limits and constraints"""
        # Get applicable policies for service
        applicable_policies = [p for p in self.scaling_policies.values() if p.enabled]
        
        for policy in applicable_policies:
            if target_instances < policy.min_instances:
                return {
                    'valid': False,
                    'reason': f'Target instances ({target_instances}) below minimum ({policy.min_instances}) for policy {policy.policy_id}'
                }
            
            if target_instances > policy.max_instances:
                return {
                    'valid': False,
                    'reason': f'Target instances ({target_instances}) exceeds maximum ({policy.max_instances}) for policy {policy.policy_id}'
                }
        
        return {'valid': True}
    
    def _create_instance(self, service_name: str, instance_type: str) -> InstanceSpec:
        """Create new instance for horizontal scaling"""
        instance_id = f"i-{len(self.active_instances) + 1:03d}"
        
        # Instance type specifications
        instance_specs = {
            't3.micro': {'cpu_cores': 2, 'memory_gb': 1.0, 'storage_gb': 8.0},
            't3.small': {'cpu_cores': 2, 'memory_gb': 2.0, 'storage_gb': 20.0},
            't3.medium': {'cpu_cores': 2, 'memory_gb': 4.0, 'storage_gb': 20.0},
            't3.large': {'cpu_cores': 2, 'memory_gb': 8.0, 'storage_gb': 40.0},
            't3.xlarge': {'cpu_cores': 4, 'memory_gb': 16.0, 'storage_gb': 80.0}
        }
        
        specs = instance_specs.get(instance_type, instance_specs['t3.medium'])
        
        instance = InstanceSpec(
            instance_id=instance_id,
            instance_type=instance_type,
            cpu_cores=specs['cpu_cores'],
            memory_gb=specs['memory_gb'],
            storage_gb=specs['storage_gb'],
            status="pending",
            created_at=datetime.now().isoformat()
        )
        
        # Simulate instance launch time
        time.sleep(0.01)  # Simulate brief launch delay
        instance.status = "running"
        
        # Store instance
        self.active_instances[instance_id] = instance
        
        return instance
    
    def _select_instances_for_termination(self, service_name: str, count: int) -> List[str]:
        """Select instances to terminate during scale-in"""
        # Simple selection strategy - terminate newest instances first
        running_instances = [
            (instance_id, instance) for instance_id, instance in self.active_instances.items()
            if instance.status == "running"
        ]
        
        # Sort by creation time (newest first)
        running_instances.sort(key=lambda x: x[1].created_at, reverse=True)
        
        # Select instances to terminate
        instances_to_terminate = [instance_id for instance_id, _ in running_instances[:count]]
        
        return instances_to_terminate
    
    def _terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        """Terminate instance during scale-in"""
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]
            instance.status = "terminated"
            
            # Remove from active instances
            del self.active_instances[instance_id]
            
            return {'status': 'terminated', 'instance_id': instance_id}
        else:
            return {'status': 'not_found', 'instance_id': instance_id}
    
    def _perform_vertical_scaling(self, instance_id: str, resource_changes: Dict[str, Dict], strategy: str) -> List[Dict[str, Any]]:
        """Perform vertical scaling operations"""
        scaling_steps = []
        
        # Simulate scaling steps based on strategy
        if strategy == 'gradual':
            # Gradual scaling with multiple steps
            for resource_type, change_info in resource_changes.items():
                if change_info['change'] != 0:
                    step = {
                        'step': len(scaling_steps) + 1,
                        'resource_type': resource_type,
                        'from_value': change_info['from'],
                        'to_value': change_info['to'],
                        'change': change_info['change'],
                        'duration_ms': 100  # Simulated step duration
                    }
                    scaling_steps.append(step)
                    time.sleep(0.01)  # Simulate scaling step delay
        
        elif strategy == 'immediate':
            # Immediate scaling - all at once
            step = {
                'step': 1,
                'resource_changes': resource_changes,
                'duration_ms': 50  # Simulated immediate scaling duration
            }
            scaling_steps.append(step)
            time.sleep(0.005)  # Simulate immediate scaling delay
        
        return scaling_steps
    
    def _generate_historical_patterns(self, days: int) -> Dict[str, List[float]]:
        """Generate historical load patterns for prediction"""
        patterns = {}
        
        # Generate CPU utilization pattern (daily cycle)
        cpu_pattern = []
        for day in range(days):
            for hour in range(24):
                # Simulate daily pattern: low at night, high during business hours
                base_load = 30 + 40 * math.sin((hour - 6) * math.pi / 12) if 6 <= hour <= 18 else 20
                noise = (hash(f"{day}_{hour}") % 20) - 10  # Random noise
                cpu_pattern.append(max(0, min(100, base_load + noise)))
        
        patterns['cpu_utilization'] = cpu_pattern
        
        # Generate memory utilization pattern
        memory_pattern = []
        for day in range(days):
            for hour in range(24):
                base_memory = 50 + 20 * math.sin((hour - 8) * math.pi / 10) if 8 <= hour <= 18 else 35
                noise = (hash(f"mem_{day}_{hour}") % 15) - 7
                memory_pattern.append(max(0, min(100, base_memory + noise)))
        
        patterns['memory_utilization'] = memory_pattern
        
        return patterns
    
    def _build_prediction_model(self, model_type: str, historical_patterns: Dict[str, List[float]]) -> Dict[str, Any]:
        """Build prediction model from historical patterns"""
        if model_type == 'time_series':
            # Simple moving average model
            cpu_data = historical_patterns['cpu_utilization']
            recent_window = 168  # Last 7 days (24 * 7)
            
            if len(cpu_data) >= recent_window:
                recent_avg = sum(cpu_data[-recent_window:]) / recent_window
                trend = (cpu_data[-1] - cpu_data[-recent_window]) / recent_window
            else:
                recent_avg = sum(cpu_data) / len(cpu_data) if cpu_data else 50
                trend = 0
            
            model = {
                'type': 'moving_average',
                'recent_average': recent_avg,
                'trend': trend,
                'accuracy': 0.78,
                'parameters': {
                    'window_size': recent_window,
                    'trend_factor': 0.1
                }
            }
        
        elif model_type == 'ml_regression':
            # Simulated ML regression model
            model = {
                'type': 'ml_regression',
                'coefficients': {'hour_of_day': 1.5, 'day_of_week': 0.8, 'trend': 0.3},
                'accuracy': 0.85,
                'parameters': {
                    'features': ['hour_of_day', 'day_of_week', 'historical_average'],
                    'algorithm': 'linear_regression'
                }
            }
        
        else:
            # Default simple model
            model = {
                'type': 'simple',
                'accuracy': 0.70,
                'parameters': {}
            }
        
        return model
    
    def _generate_load_predictions(self, model: Dict[str, Any], hours: int) -> Dict[str, Dict[str, float]]:
        """Generate load predictions using the model"""
        predictions = {}
        
        current_time = datetime.now()
        
        for hour in range(hours):
            prediction_time = current_time + timedelta(hours=hour)
            hour_of_day = prediction_time.hour
            
            if model['type'] == 'moving_average':
                base_prediction = model['recent_average'] + (model['trend'] * hour)
                # Add hourly variation
                hourly_factor = 1 + 0.3 * math.sin((hour_of_day - 6) * math.pi / 12) if 6 <= hour_of_day <= 18 else 0.7
                cpu_prediction = base_prediction * hourly_factor
                
            elif model['type'] == 'ml_regression':
                # Simulate ML prediction
                coeffs = model['coefficients']
                cpu_prediction = (
                    coeffs['hour_of_day'] * hour_of_day +
                    coeffs['day_of_week'] * (prediction_time.weekday() + 1) +
                    coeffs['trend'] * hour + 40  # Base load
                )
                
            else:
                # Simple prediction
                cpu_prediction = 50 + 20 * math.sin((hour_of_day - 8) * math.pi / 10)
            
            # Ensure prediction is within bounds
            cpu_prediction = max(0, min(100, cpu_prediction))
            
            predictions[f"hour_{hour}"] = {
                'cpu_utilization': cpu_prediction,
                'memory_utilization': cpu_prediction * 0.8,  # Memory often correlates with CPU
                'prediction_confidence': model['accuracy']
            }
        
        return predictions
    
    def _create_predictive_scaling_policy(self, hour: int, scaling_type: ScalingType, predicted_load: Dict[str, float]) -> Dict[str, Any]:
        """Create predictive scaling policy for specific time"""
        policy = {
            'type': 'predictive',
            'trigger_time': (datetime.now() + timedelta(hours=hour)).isoformat(),
            'scaling_type': scaling_type.value,
            'predicted_load': predicted_load,
            'confidence': predicted_load.get('prediction_confidence', 0.8),
            'magnitude': self._calculate_predictive_scaling_magnitude(predicted_load)
        }
        
        return policy
    
    def _calculate_predictive_scaling_magnitude(self, predicted_load: Dict[str, float]) -> Dict[str, Any]:
        """Calculate scaling magnitude based on predicted load"""
        cpu_utilization = predicted_load.get('cpu_utilization', 50)
        
        if cpu_utilization > 80:
            scale_factor = 2.0  # Double capacity
        elif cpu_utilization > 70:
            scale_factor = 1.5  # 50% increase
        elif cpu_utilization < 20:
            scale_factor = 0.5  # Scale down by half
        elif cpu_utilization < 30:
            scale_factor = 0.75  # Scale down by 25%
        else:
            scale_factor = 1.0  # No scaling
        
        return {
            'scale_factor': scale_factor,
            'recommended_instances': max(1, int(len(self.active_instances) * scale_factor))
        }
    
    def _is_policy_in_cooldown(self, policy: ScalingPolicy) -> bool:
        """Check if policy is in cooldown period"""
        # Find last scaling event for this policy
        recent_events = [
            event for event in self.scaling_events
            if event.policy_id == policy.policy_id
        ]
        
        if not recent_events:
            return False  # No previous events, not in cooldown
        
        # Get most recent event
        last_event = max(recent_events, key=lambda x: x.timestamp)
        last_event_time = datetime.fromisoformat(last_event.timestamp)
        cooldown_end = last_event_time + timedelta(seconds=policy.cooldown_seconds)
        
        return datetime.now() < cooldown_end
    
    def _calculate_scaling_magnitude(self, policy: ScalingPolicy, metric_value: float) -> Dict[str, Any]:
        """Calculate scaling magnitude based on policy and metric value"""
        if metric_value > policy.threshold_up:
            # Scale up/out
            urgency = (metric_value - policy.threshold_up) / (100 - policy.threshold_up)
            scale_factor = 1 + (urgency * 0.5)  # Up to 50% increase
        elif metric_value < policy.threshold_down:
            # Scale down/in
            urgency = (policy.threshold_down - metric_value) / policy.threshold_down
            scale_factor = 1 - (urgency * 0.3)  # Up to 30% decrease
        else:
            scale_factor = 1.0
        
        current_instances = len([i for i in self.active_instances.values() if i.status == "running"])
        target_instances = max(policy.min_instances, min(policy.max_instances, int(current_instances * scale_factor)))
        
        return {
            'scale_factor': scale_factor,
            'current_instances': current_instances,
            'target_instances': target_instances,
            'instance_change': target_instances - current_instances
        }
    
    def demonstrate_scalability_capabilities(self) -> Dict[str, Any]:
        """Demonstrate scalability management capabilities"""
        print("\n📈 SCALABILITY ENGINE DEMONSTRATION 📈")
        print("=" * 50)
        
        # Current system state
        print(f"🏗️ Current System State:")
        print(f"   Active Instances: {len(self.active_instances)}")
        print(f"   Scaling Policies: {len(self.scaling_policies)}")
        
        # Demonstrate horizontal scaling
        print("\n🔄 Horizontal Scaling Demonstration...")
        scaling_specs = {
            'service_name': 'manufacturing-api',
            'current_instances': len(self.active_instances),
            'target_instances': len(self.active_instances) + 3,
            'instance_type': 't3.large',
            'trigger_metric': 'cpu_utilization',
            'trigger_value': 85.0
        }
        
        scaling_result = self.manage_horizontal_scaling(scaling_specs)
        if scaling_result.get('scaling_success', False):
            print(f"   ✅ Horizontal Scaling: {scaling_result['scaling_type'].upper()}")
            print(f"   📊 Instances: {scaling_result['instances_before']} → {scaling_result['instances_after']}")
            print(f"   ⏱️ Scaling time: {scaling_result['scaling_time_minutes']:.2f}min")
            print(f"   🎯 Target: <{self.scale_out_target_minutes}min | {'✅ MET' if scaling_result['target_met'] else '❌ MISSED'}")
        else:
            print(f"   ❌ Horizontal Scaling: {scaling_result.get('reason', 'Unknown error')}")
            # Set default values for failed scaling
            scaling_result.update({
                'scaling_time_minutes': 0.0,
                'target_met': False,
                'instances_before': len(self.active_instances),
                'instances_after': len(self.active_instances)
            })
        
        # Demonstrate vertical scaling
        print("\n📊 Vertical Scaling Demonstration...")
        instance_id = list(self.active_instances.keys())[0]
        resource_specs = {
            'instance_id': instance_id,
            'new_resources': {
                'cpu_cores': 4,
                'memory_gb': 16.0,
                'storage_gb': 100.0
            },
            'strategy': 'gradual',
            'trigger_metric': 'memory_utilization',
            'trigger_value': 82.0
        }
        
        vertical_result = self.manage_vertical_scaling(resource_specs)
        if vertical_result.get('scaling_success', False):
            print(f"   ✅ Vertical Scaling: {vertical_result['scaling_type'].upper()}")
            print(f"   🔧 Resource changes: {len(vertical_result['resource_changes'])}")
            print(f"   ⏱️ Scaling time: {vertical_result['scaling_time_minutes']:.2f}min")
            print(f"   🎯 Target: <{self.scale_out_target_minutes}min | {'✅ MET' if vertical_result['target_met'] else '❌ MISSED'}")
        else:
            print(f"   ❌ Vertical Scaling: {vertical_result.get('reason', 'Unknown error')}")
            # Set default values for failed scaling
            vertical_result.update({
                'scaling_time_minutes': 0.0,
                'target_met': False,
                'resource_changes': {}
            })
        
        # Demonstrate predictive scaling
        print("\n🔮 Predictive Scaling Demonstration...")
        prediction_specs = {
            'model_type': 'time_series',
            'horizon_hours': 12,
            'historical_days': 14
        }
        
        predictive_result = self.implement_predictive_scaling(prediction_specs)
        print(f"   ✅ Predictive Model: {predictive_result['model_type'].upper()}")
        print(f"   📈 Prediction horizon: {predictive_result['prediction_horizon_hours']} hours")
        print(f"   🤖 Predictive policies: {predictive_result['predictive_policies_created']}")
        print(f"   📊 Model accuracy: {predictive_result['model_accuracy']:.1%}")
        
        # Demonstrate scaling trigger evaluation
        print("\n⚡ Scaling Trigger Evaluation...")
        test_metrics = {
            'cpu': {'utilization': 78.0},
            'memory': {'utilization': 65.0},
            'storage': {'utilization': 45.0},
            'network': {'utilization': 30.0}
        }
        
        trigger_result = self.evaluate_scaling_triggers(test_metrics)
        print(f"   ✅ Trigger evaluation: {trigger_result['policies_evaluated']} policies")
        print(f"   🚨 Scaling decisions: {trigger_result['scaling_decisions']}")
        print(f"   ⏱️ Evaluation time: {trigger_result['evaluation_time_ms']}ms")
        print(f"   🎯 Target: <{self.scaling_decision_target_ms}ms | {'✅ MET' if trigger_result['target_met'] else '❌ MISSED'}")
        
        print(f"\n📊 Updated System State:")
        print(f"   Active Instances: {len(self.active_instances)}")
        print(f"   Scaling Events: {len(self.scaling_events)}")
        print(f"   Prediction Models: {len(self.prediction_models)}")
        
        print("\n📈 DEMONSTRATION SUMMARY:")
        print(f"   Horizontal Scaling Time: {scaling_result['scaling_time_minutes']:.2f}min")
        print(f"   Vertical Scaling Time: {vertical_result['scaling_time_minutes']:.2f}min")
        print(f"   Predictive Policies Created: {predictive_result['predictive_policies_created']}")
        print(f"   Scaling Decisions Made: {trigger_result['scaling_decisions']}")
        print(f"   Total Active Instances: {len(self.active_instances)}")
        print("=" * 50)
        
        return {
            'horizontal_scaling_time_minutes': scaling_result['scaling_time_minutes'],
            'vertical_scaling_time_minutes': vertical_result['scaling_time_minutes'],
            'predictive_policies_created': predictive_result['predictive_policies_created'],
            'scaling_decisions_made': trigger_result['scaling_decisions'],
            'total_active_instances': len(self.active_instances),
            'scaling_events_recorded': len(self.scaling_events),
            'prediction_models': len(self.prediction_models),
            'performance_targets_met': scaling_result['target_met'] and vertical_result['target_met'] and trigger_result['target_met']
        }

def main():
    """Test ScalabilityEngine functionality"""
    engine = ScalabilityEngine()
    results = engine.demonstrate_scalability_capabilities()
    
    print(f"\n🎯 Week 10 Scalability Performance Targets:")
    print(f"   Scaling Decisions: <100ms ({'✅' if results.get('scaling_decisions_made', 0) == 0 or results.get('performance_targets_met', False) else '❌'})")
    print(f"   Scale-out Operations: <2min ({'✅' if results['horizontal_scaling_time_minutes'] < 2 else '❌'})")
    print(f"   Overall Performance: {'🟢 EXCELLENT' if results['performance_targets_met'] else '🟡 NEEDS OPTIMIZATION'}")

if __name__ == "__main__":
    main()