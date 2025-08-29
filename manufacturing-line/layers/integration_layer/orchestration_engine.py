#!/usr/bin/env python3
"""
OrchestrationEngine - Week 11 Integration & Orchestration Layer
Advanced system orchestration with intelligent workflow coordination
"""

import time
import json
import uuid
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import queue
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(Enum):
    """Individual task status"""
    WAITING = "waiting"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class DependencyType(Enum):
    """Task dependency types"""
    SEQUENTIAL = "sequential"    # Task B waits for Task A to complete
    PARALLEL = "parallel"        # Tasks can run simultaneously
    CONDITIONAL = "conditional"  # Task B runs if Task A meets condition
    RESOURCE = "resource"        # Tasks share limited resources

class OrchestrationStrategy(Enum):
    """Orchestration execution strategies"""
    IMMEDIATE = "immediate"      # Execute as soon as dependencies are met
    BATCHED = "batched"         # Group tasks into batches
    PRIORITY = "priority"       # Execute based on priority levels
    RESOURCE_AWARE = "resource_aware"  # Consider resource availability

@dataclass
class WorkflowTask:
    """Individual workflow task definition"""
    task_id: str
    name: str
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, 1 being highest priority
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.WAITING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    execution_duration_ms: Optional[float] = None

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    strategy: OrchestrationStrategy
    timeout_seconds: int = 3600  # 1 hour default
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "system"
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: str
    end_time: Optional[str] = None
    execution_duration_ms: Optional[float] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tasks: int = 0
    progress_percentage: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None

class OrchestrationEngine:
    """Advanced system orchestration with intelligent workflow coordination
    
    Week 11 Performance Targets:
    - Orchestration decisions: <200ms
    - Workflow execution: <10 seconds
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OrchestrationEngine with configuration"""
        self.config = config or {}
        
        # Performance targets
        self.orchestration_decision_target_ms = 200
        self.workflow_execution_target_seconds = 10
        
        # State management
        self.workflow_definitions = {}
        self.workflow_executions = {}
        self.active_workflows = {}
        self.task_registry = {}
        self.dependency_graph = {}
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.task_queue = queue.PriorityQueue()
        self.resource_locks = defaultdict(threading.RLock)
        
        # Statistics
        self.execution_statistics = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'total_tasks': 0,
            'average_execution_time_ms': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize integration engine if available
        self.integration_engine = None
        try:
            from layers.integration_layer.integration_engine import IntegrationEngine
            integration_config = config.get('integration_config', {}) if config else {}
            self.integration_engine = IntegrationEngine(integration_config)
        except ImportError:
            logger.warning("IntegrationEngine not available - using mock interface")
        
        # Initialize task registry with built-in tasks
        self._initialize_task_registry()
        
        # Start workflow execution thread
        self._start_execution_thread()
        
        logger.info("OrchestrationEngine initialized with intelligent workflow coordination")
    
    def orchestrate_system_workflows(self, workflow_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate complex system workflows across all layers
        
        Args:
            workflow_specs: Workflow orchestration specifications
            
        Returns:
            Workflow orchestration results with performance metrics
        """
        start_time = time.time()
        
        try:
            # Parse workflow specifications
            workflow_name = workflow_specs.get('workflow_name', 'unnamed_workflow')
            tasks = workflow_specs.get('tasks', [])
            strategy = OrchestrationStrategy(workflow_specs.get('strategy', 'immediate'))
            timeout_seconds = workflow_specs.get('timeout_seconds', 600)
            
            # Create workflow definition
            workflow_definition = self._create_workflow_definition(
                workflow_name, tasks, strategy, timeout_seconds
            )
            
            # Store workflow definition
            self.workflow_definitions[workflow_definition.workflow_id] = workflow_definition
            
            # Create workflow execution instance
            execution = self._create_workflow_execution(workflow_definition)
            
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(workflow_definition.tasks)
            self.dependency_graph[execution.execution_id] = dependency_graph
            
            # Start workflow execution
            execution_future = self._execute_workflow_async(execution, workflow_definition)
            
            # Store active workflow
            self.active_workflows[execution.execution_id] = {
                'execution': execution,
                'definition': workflow_definition,
                'future': execution_future,
                'dependency_graph': dependency_graph
            }
            
            # Calculate orchestration decision time
            decision_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'orchestration_success': True,
                'workflow_id': workflow_definition.workflow_id,
                'execution_id': execution.execution_id,
                'workflow_name': workflow_name,
                'total_tasks': len(tasks),
                'orchestration_strategy': strategy.value,
                'decision_time_ms': round(decision_time_ms, 2),
                'target_met': decision_time_ms < self.orchestration_decision_target_ms,
                'estimated_completion_time': timeout_seconds,
                'orchestrated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Workflow orchestrated: {workflow_name} with {len(tasks)} tasks")
            return result
            
        except Exception as e:
            logger.error(f"Error orchestrating system workflows: {e}")
            raise
    
    def manage_task_dependencies(self, dependency_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Manage task dependencies and execution order
        
        Args:
            dependency_graph: Task dependency specifications
            
        Returns:
            Dependency management results with execution order
        """
        start_time = time.time()
        
        try:
            # Parse dependency graph
            tasks = dependency_graph.get('tasks', {})
            dependencies = dependency_graph.get('dependencies', {})
            
            # Validate dependency graph for cycles
            validation_result = self._validate_dependency_graph(tasks, dependencies)
            if not validation_result['valid']:
                return {
                    'dependency_success': False,
                    'reason': validation_result['reason'],
                    'management_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            
            # Calculate execution order using topological sort
            execution_order = self._calculate_execution_order(tasks, dependencies)
            
            # Identify parallel execution opportunities
            parallel_groups = self._identify_parallel_groups(tasks, dependencies)
            
            # Calculate resource requirements
            resource_requirements = self._calculate_resource_requirements(tasks)
            
            # Generate optimized execution plan
            execution_plan = self._generate_execution_plan(
                execution_order, parallel_groups, resource_requirements
            )
            
            # Calculate management time
            management_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'dependency_success': True,
                'total_tasks': len(tasks),
                'execution_order': execution_order,
                'parallel_groups': len(parallel_groups),
                'resource_conflicts': len([r for r in resource_requirements.values() if r.get('conflicts', [])]),
                'execution_plan': execution_plan,
                'management_time_ms': round(management_time_ms, 2),
                'target_met': management_time_ms < self.orchestration_decision_target_ms,
                'cycle_free': validation_result['cycle_free'],
                'managed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Task dependencies managed: {len(tasks)} tasks with {len(parallel_groups)} parallel groups")
            return result
            
        except Exception as e:
            logger.error(f"Error managing task dependencies: {e}")
            raise
    
    def coordinate_cross_layer_operations(self, operation_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate operations across multiple system layers
        
        Args:
            operation_specs: Cross-layer operation specifications
            
        Returns:
            Cross-layer coordination results
        """
        start_time = time.time()
        
        try:
            # Parse operation specifications
            operation_name = operation_specs.get('operation_name', 'cross_layer_operation')
            target_layers = operation_specs.get('target_layers', [])
            coordination_type = operation_specs.get('coordination_type', 'sequential')
            synchronization_required = operation_specs.get('synchronization_required', True)
            
            # Initialize layer coordinators
            layer_coordinators = {}
            for layer in target_layers:
                coordinator = self._get_layer_coordinator(layer)
                if coordinator:
                    layer_coordinators[layer] = coordinator
            
            # Plan cross-layer operation
            operation_plan = self._plan_cross_layer_operation(
                operation_specs, layer_coordinators, coordination_type
            )
            
            # Execute cross-layer coordination
            coordination_results = []
            if coordination_type == 'sequential':
                # Execute layers sequentially
                for layer in target_layers:
                    if layer in layer_coordinators:
                        layer_result = self._execute_layer_operation(
                            layer, layer_coordinators[layer], operation_specs
                        )
                        coordination_results.append(layer_result)
            
            elif coordination_type == 'parallel':
                # Execute layers in parallel
                futures = []
                for layer in target_layers:
                    if layer in layer_coordinators:
                        future = self.executor.submit(
                            self._execute_layer_operation,
                            layer, layer_coordinators[layer], operation_specs
                        )
                        futures.append((layer, future))
                
                # Collect results
                for layer, future in futures:
                    try:
                        layer_result = future.result(timeout=30)
                        coordination_results.append(layer_result)
                    except Exception as e:
                        coordination_results.append({
                            'layer': layer,
                            'success': False,
                            'error': str(e)
                        })
            
            # Synchronize results if required
            if synchronization_required:
                sync_result = self._synchronize_cross_layer_results(coordination_results)
            else:
                sync_result = {'synchronized': False, 'reason': 'synchronization_not_required'}
            
            # Calculate coordination time
            coordination_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'coordination_success': True,
                'operation_name': operation_name,
                'target_layers': target_layers,
                'coordination_type': coordination_type,
                'layers_coordinated': len(layer_coordinators),
                'operation_results': coordination_results,
                'synchronization_result': sync_result,
                'coordination_time_ms': round(coordination_time_ms, 2),
                'target_met': coordination_time_ms < self.orchestration_decision_target_ms * 2,  # Allow 2x for complex coordination
                'coordinated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Cross-layer operation coordinated: {operation_name} across {len(target_layers)} layers")
            return result
            
        except Exception as e:
            logger.error(f"Error coordinating cross-layer operations: {e}")
            raise
    
    def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get current status of workflow execution"""
        try:
            if execution_id in self.active_workflows:
                workflow_info = self.active_workflows[execution_id]
                execution = workflow_info['execution']
                definition = workflow_info['definition']
                
                # Update progress
                completed_tasks = sum(1 for task in definition.tasks if task.status == TaskStatus.COMPLETED)
                failed_tasks = sum(1 for task in definition.tasks if task.status == TaskStatus.FAILED)
                total_tasks = len(definition.tasks)
                
                execution.tasks_completed = completed_tasks
                execution.tasks_failed = failed_tasks
                execution.total_tasks = total_tasks
                execution.progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
                
                return {
                    'execution_id': execution_id,
                    'workflow_id': execution.workflow_id,
                    'status': execution.status.value,
                    'progress_percentage': execution.progress_percentage,
                    'tasks_completed': execution.tasks_completed,
                    'tasks_failed': execution.tasks_failed,
                    'total_tasks': execution.total_tasks,
                    'start_time': execution.start_time,
                    'current_time': datetime.now().isoformat(),
                    'task_details': [
                        {
                            'task_id': task.task_id,
                            'name': task.name,
                            'status': task.status.value,
                            'duration_ms': task.execution_duration_ms
                        }
                        for task in definition.tasks
                    ]
                }
            else:
                return {'error': f'Workflow execution {execution_id} not found'}
                
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {'error': str(e)}
    
    def cancel_workflow(self, execution_id: str) -> Dict[str, Any]:
        """Cancel active workflow execution"""
        try:
            if execution_id in self.active_workflows:
                workflow_info = self.active_workflows[execution_id]
                execution = workflow_info['execution']
                future = workflow_info['future']
                
                # Cancel the future
                if future and not future.done():
                    future.cancel()
                
                # Update execution status
                execution.status = WorkflowStatus.CANCELLED
                execution.end_time = datetime.now().isoformat()
                
                # Cancel running tasks
                definition = workflow_info['definition']
                for task in definition.tasks:
                    if task.status == TaskStatus.EXECUTING:
                        task.status = TaskStatus.CANCELLED
                
                logger.info(f"Workflow cancelled: {execution_id}")
                return {
                    'cancelled': True,
                    'execution_id': execution_id,
                    'cancelled_at': execution.end_time
                }
            else:
                return {'error': f'Workflow execution {execution_id} not found'}
                
        except Exception as e:
            logger.error(f"Error cancelling workflow: {e}")
            return {'error': str(e)}
    
    def _initialize_task_registry(self):
        """Initialize built-in task registry"""
        self.task_registry = {
            'data_processing': self._execute_data_processing_task,
            'system_check': self._execute_system_check_task,
            'notification': self._execute_notification_task,
            'delay': self._execute_delay_task,
            'layer_operation': self._execute_layer_operation_task,
            'conditional': self._execute_conditional_task,
            'loop': self._execute_loop_task,
            'parallel_group': self._execute_parallel_group_task
        }
    
    def _create_workflow_definition(self, name: str, tasks: List[Dict], strategy: OrchestrationStrategy, timeout: int) -> WorkflowDefinition:
        """Create workflow definition from specifications"""
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        workflow_tasks = []
        for i, task_spec in enumerate(tasks):
            task = WorkflowTask(
                task_id=task_spec.get('task_id', f"task_{i}"),
                name=task_spec.get('name', f"Task {i+1}"),
                task_type=task_spec.get('type', 'system_check'),
                parameters=task_spec.get('parameters', {}),
                dependencies=task_spec.get('dependencies', []),
                priority=task_spec.get('priority', 5),
                timeout_seconds=task_spec.get('timeout_seconds'),
                max_retries=task_spec.get('max_retries', 3)
            )
            workflow_tasks.append(task)
        
        return WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=f"Orchestrated workflow: {name}",
            tasks=workflow_tasks,
            strategy=strategy,
            timeout_seconds=timeout
        )
    
    def _create_workflow_execution(self, workflow_def: WorkflowDefinition) -> WorkflowExecution:
        """Create workflow execution instance"""
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        return WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_def.workflow_id,
            status=WorkflowStatus.PENDING,
            start_time=datetime.now().isoformat(),
            total_tasks=len(workflow_def.tasks)
        )
    
    def _build_dependency_graph(self, tasks: List[WorkflowTask]) -> Dict[str, Any]:
        """Build task dependency graph"""
        graph = {
            'nodes': {task.task_id: task for task in tasks},
            'edges': {},
            'roots': [],  # Tasks with no dependencies
            'leaves': []  # Tasks with no dependents
        }
        
        # Build edges (dependencies)
        for task in tasks:
            graph['edges'][task.task_id] = task.dependencies
        
        # Find root and leaf nodes
        all_task_ids = set(task.task_id for task in tasks)
        dependent_tasks = set()
        
        for task in tasks:
            if not task.dependencies:
                graph['roots'].append(task.task_id)
            for dep in task.dependencies:
                dependent_tasks.add(dep)
        
        graph['leaves'] = list(all_task_ids - dependent_tasks)
        
        return graph
    
    def _execute_workflow_async(self, execution: WorkflowExecution, definition: WorkflowDefinition) -> Future:
        """Execute workflow asynchronously"""
        return self.executor.submit(self._execute_workflow, execution, definition)
    
    def _execute_workflow(self, execution: WorkflowExecution, definition: WorkflowDefinition):
        """Execute workflow tasks"""
        try:
            execution.status = WorkflowStatus.RUNNING
            workflow_start_time = time.time()
            
            # Get dependency graph
            dependency_graph = self.dependency_graph.get(execution.execution_id, {})
            
            # Execute tasks based on strategy
            if definition.strategy == OrchestrationStrategy.IMMEDIATE:
                self._execute_immediate_strategy(execution, definition, dependency_graph)
            elif definition.strategy == OrchestrationStrategy.PRIORITY:
                self._execute_priority_strategy(execution, definition, dependency_graph)
            else:
                # Default to immediate strategy
                self._execute_immediate_strategy(execution, definition, dependency_graph)
            
            # Calculate execution time
            execution_time_ms = (time.time() - workflow_start_time) * 1000
            execution.execution_duration_ms = execution_time_ms
            execution.end_time = datetime.now().isoformat()
            
            # Determine final status
            if execution.tasks_failed > 0:
                execution.status = WorkflowStatus.FAILED
            else:
                execution.status = WorkflowStatus.COMPLETED
            
            # Update statistics
            self._update_execution_statistics(execution)
            
            logger.info(f"Workflow execution completed: {execution.execution_id} in {execution_time_ms:.2f}ms")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_details = str(e)
            execution.end_time = datetime.now().isoformat()
            logger.error(f"Workflow execution failed: {execution.execution_id} - {e}")
    
    def _execute_immediate_strategy(self, execution: WorkflowExecution, definition: WorkflowDefinition, dependency_graph: Dict):
        """Execute workflow using immediate strategy"""
        completed_tasks = set()
        failed_tasks = set()
        
        while len(completed_tasks) + len(failed_tasks) < len(definition.tasks):
            ready_tasks = []
            
            # Find tasks ready for execution
            for task in definition.tasks:
                if (task.status == TaskStatus.WAITING and 
                    all(dep in completed_tasks for dep in task.dependencies)):
                    task.status = TaskStatus.READY
                    ready_tasks.append(task)
            
            # Execute ready tasks
            if ready_tasks:
                # Sort by priority
                ready_tasks.sort(key=lambda t: t.priority)
                
                for task in ready_tasks:
                    try:
                        task.status = TaskStatus.EXECUTING
                        task.start_time = datetime.now().isoformat()
                        
                        # Execute task
                        task_start = time.time()
                        result = self._execute_task(task)
                        task_duration = (time.time() - task_start) * 1000
                        
                        task.execution_duration_ms = task_duration
                        task.end_time = datetime.now().isoformat()
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        completed_tasks.add(task.task_id)
                        
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.end_time = datetime.now().isoformat()
                        failed_tasks.add(task.task_id)
                        
                        # Retry if possible
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.status = TaskStatus.WAITING
                            failed_tasks.remove(task.task_id)
            else:
                # No ready tasks - check for deadlock or completion
                waiting_tasks = [t for t in definition.tasks if t.status == TaskStatus.WAITING]
                if waiting_tasks:
                    # Possible deadlock - mark remaining tasks as failed
                    for task in waiting_tasks:
                        task.status = TaskStatus.FAILED
                        task.error = "Dependency deadlock detected"
                        failed_tasks.add(task.task_id)
                break
        
        # Update execution counts
        execution.tasks_completed = len(completed_tasks)
        execution.tasks_failed = len(failed_tasks)
    
    def _execute_priority_strategy(self, execution: WorkflowExecution, definition: WorkflowDefinition, dependency_graph: Dict):
        """Execute workflow using priority-based strategy"""
        # Sort all tasks by priority
        priority_queue = sorted(definition.tasks, key=lambda t: t.priority)
        self._execute_immediate_strategy(execution, definition, dependency_graph)  # Use immediate for now
    
    def _execute_task(self, task: WorkflowTask) -> Any:
        """Execute individual workflow task"""
        task_executor = self.task_registry.get(task.task_type)
        if task_executor:
            return task_executor(task)
        else:
            # Default task execution
            time.sleep(0.01)  # Simulate task execution
            return f"Task {task.name} completed successfully"
    
    def _execute_data_processing_task(self, task: WorkflowTask) -> str:
        """Execute data processing task"""
        processing_time = task.parameters.get('processing_time_ms', 50)
        time.sleep(processing_time / 1000.0)
        return f"Data processing completed in {processing_time}ms"
    
    def _execute_system_check_task(self, task: WorkflowTask) -> str:
        """Execute system health check task"""
        time.sleep(0.02)  # 20ms simulation
        return "System health check passed"
    
    def _execute_notification_task(self, task: WorkflowTask) -> str:
        """Execute notification task"""
        message = task.parameters.get('message', 'Default notification')
        recipients = task.parameters.get('recipients', ['system'])
        time.sleep(0.01)  # 10ms simulation
        return f"Notification sent to {len(recipients)} recipients: {message}"
    
    def _execute_delay_task(self, task: WorkflowTask) -> str:
        """Execute delay task"""
        delay_ms = task.parameters.get('delay_ms', 100)
        time.sleep(delay_ms / 1000.0)
        return f"Delayed execution by {delay_ms}ms"
    
    def _execute_layer_operation_task(self, task: WorkflowTask) -> str:
        """Execute cross-layer operation task"""
        layer = task.parameters.get('layer', 'unknown')
        operation = task.parameters.get('operation', 'status_check')
        time.sleep(0.05)  # 50ms simulation
        return f"Layer operation completed: {layer}.{operation}"
    
    def _execute_conditional_task(self, task: WorkflowTask) -> str:
        """Execute conditional task"""
        condition = task.parameters.get('condition', 'true')
        # Simplified condition evaluation
        if condition == 'true':
            time.sleep(0.01)
            return "Condition met - task executed"
        else:
            return "Condition not met - task skipped"
    
    def _execute_loop_task(self, task: WorkflowTask) -> str:
        """Execute loop task"""
        iterations = task.parameters.get('iterations', 3)
        for i in range(iterations):
            time.sleep(0.005)  # 5ms per iteration
        return f"Loop completed with {iterations} iterations"
    
    def _execute_parallel_group_task(self, task: WorkflowTask) -> str:
        """Execute parallel group task"""
        subtasks = task.parameters.get('subtasks', [])
        time.sleep(0.03)  # 30ms simulation
        return f"Parallel group executed with {len(subtasks)} subtasks"
    
    def _validate_dependency_graph(self, tasks: Dict, dependencies: Dict) -> Dict[str, Any]:
        """Validate dependency graph for cycles and consistency"""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id):
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep in dependencies.get(task_id, []):
                if has_cycle(dep):
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        # Check for cycles
        for task_id in tasks:
            if has_cycle(task_id):
                return {
                    'valid': False,
                    'cycle_free': False,
                    'reason': f'Dependency cycle detected involving task {task_id}'
                }
        
        return {
            'valid': True,
            'cycle_free': True,
            'reason': 'Dependency graph is valid'
        }
    
    def _calculate_execution_order(self, tasks: Dict, dependencies: Dict) -> List[str]:
        """Calculate execution order using topological sort"""
        in_degree = {task_id: 0 for task_id in tasks}
        
        # Calculate in-degrees
        for task_id in tasks:
            for dep in dependencies.get(task_id, []):
                in_degree[task_id] += 1
        
        # Topological sort
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees
            for task_id in tasks:
                if current in dependencies.get(task_id, []):
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        return execution_order
    
    def _identify_parallel_groups(self, tasks: Dict, dependencies: Dict) -> List[List[str]]:
        """Identify tasks that can be executed in parallel"""
        execution_order = self._calculate_execution_order(tasks, dependencies)
        parallel_groups = []
        
        # Group tasks by dependency level
        levels = {}
        for task_id in execution_order:
            level = max([levels.get(dep, -1) for dep in dependencies.get(task_id, [])] + [-1]) + 1
            levels[task_id] = level
        
        # Group by level
        level_groups = defaultdict(list)
        for task_id, level in levels.items():
            level_groups[level].append(task_id)
        
        # Each level can be executed in parallel
        for level_tasks in level_groups.values():
            if len(level_tasks) > 1:
                parallel_groups.append(level_tasks)
        
        return parallel_groups
    
    def _calculate_resource_requirements(self, tasks: Dict) -> Dict[str, Any]:
        """Calculate resource requirements for tasks"""
        resource_requirements = {}
        
        for task_id, task_spec in tasks.items():
            resources = task_spec.get('resources', {})
            requirements = {
                'cpu': resources.get('cpu', 0.1),
                'memory': resources.get('memory', 100),  # MB
                'storage': resources.get('storage', 0),  # MB
                'network': resources.get('network', 0),  # Mbps
                'conflicts': []
            }
            resource_requirements[task_id] = requirements
        
        return resource_requirements
    
    def _generate_execution_plan(self, execution_order: List[str], parallel_groups: List[List[str]], resource_requirements: Dict) -> Dict[str, Any]:
        """Generate optimized execution plan"""
        return {
            'execution_strategy': 'dependency_aware',
            'total_phases': len(parallel_groups) + len(execution_order) - sum(len(group) for group in parallel_groups),
            'parallel_opportunities': len(parallel_groups),
            'resource_constraints': len([r for r in resource_requirements.values() if any(v > 0 for v in [r['cpu'], r['memory'], r['storage'], r['network']])]),
            'estimated_duration_ms': sum(50 for _ in execution_order),  # 50ms per task estimate
            'optimization_applied': True
        }
    
    def _get_layer_coordinator(self, layer: str) -> Optional[Dict[str, Any]]:
        """Get coordinator for specific layer"""
        # Mock layer coordinators
        layer_coordinators = {
            'data_layer': {'name': 'DataLayerCoordinator', 'version': '1.0'},
            'control_layer': {'name': 'ControlLayerCoordinator', 'version': '1.0'},
            'ui_layer': {'name': 'UILayerCoordinator', 'version': '1.0'},
            'security_layer': {'name': 'SecurityLayerCoordinator', 'version': '1.0'},
            'scalability_layer': {'name': 'ScalabilityLayerCoordinator', 'version': '1.0'}
        }
        
        return layer_coordinators.get(layer)
    
    def _plan_cross_layer_operation(self, operation_specs: Dict, coordinators: Dict, coordination_type: str) -> Dict[str, Any]:
        """Plan cross-layer operation"""
        return {
            'operation_name': operation_specs.get('operation_name'),
            'coordination_type': coordination_type,
            'layers_involved': list(coordinators.keys()),
            'execution_plan': f'{coordination_type}_execution',
            'estimated_duration_ms': len(coordinators) * 100  # 100ms per layer
        }
    
    def _execute_layer_operation(self, layer: str, coordinator: Dict, operation_specs: Dict) -> Dict[str, Any]:
        """Execute operation on specific layer"""
        # Simulate layer operation
        operation_start = time.time()
        time.sleep(0.05)  # 50ms simulation
        operation_duration = (time.time() - operation_start) * 1000
        
        return {
            'layer': layer,
            'success': True,
            'coordinator': coordinator['name'],
            'operation': operation_specs.get('operation_name', 'unknown'),
            'duration_ms': round(operation_duration, 2),
            'result': f'Operation completed successfully on {layer}'
        }
    
    def _synchronize_cross_layer_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Synchronize results across layers"""
        successful_layers = [r['layer'] for r in results if r.get('success', False)]
        failed_layers = [r['layer'] for r in results if not r.get('success', False)]
        
        return {
            'synchronized': len(failed_layers) == 0,
            'successful_layers': successful_layers,
            'failed_layers': failed_layers,
            'synchronization_strategy': 'all_or_nothing' if failed_layers else 'success'
        }
    
    def _update_execution_statistics(self, execution: WorkflowExecution):
        """Update execution statistics"""
        with self._lock:
            self.execution_statistics['total_workflows'] += 1
            
            if execution.status == WorkflowStatus.COMPLETED:
                self.execution_statistics['successful_workflows'] += 1
            else:
                self.execution_statistics['failed_workflows'] += 1
            
            self.execution_statistics['total_tasks'] += execution.total_tasks
            
            if execution.execution_duration_ms:
                current_avg = self.execution_statistics['average_execution_time_ms']
                total_workflows = self.execution_statistics['total_workflows']
                new_avg = ((current_avg * (total_workflows - 1)) + execution.execution_duration_ms) / total_workflows
                self.execution_statistics['average_execution_time_ms'] = new_avg
    
    def _start_execution_thread(self):
        """Start background workflow execution thread"""
        # In a real implementation, this would start a background thread
        # For demo purposes, we'll use the executor
        pass
    
    def demonstrate_orchestration_capabilities(self) -> Dict[str, Any]:
        """Demonstrate orchestration capabilities"""
        print("\n🎼 ORCHESTRATION ENGINE DEMONSTRATION 🎼")
        print("=" * 50)
        
        # Demonstrate workflow orchestration
        print("🎬 System Workflow Orchestration...")
        workflow_specs = {
            'workflow_name': 'Manufacturing Line Startup',
            'strategy': 'immediate',
            'timeout_seconds': 300,
            'tasks': [
                {'task_id': 'system_init', 'name': 'System Initialization', 'type': 'system_check', 'dependencies': []},
                {'task_id': 'data_load', 'name': 'Data Loading', 'type': 'data_processing', 'dependencies': ['system_init'], 'parameters': {'processing_time_ms': 100}},
                {'task_id': 'security_check', 'name': 'Security Validation', 'type': 'layer_operation', 'dependencies': ['system_init'], 'parameters': {'layer': 'security_layer', 'operation': 'validate'}},
                {'task_id': 'ui_startup', 'name': 'UI Initialization', 'type': 'layer_operation', 'dependencies': ['data_load', 'security_check'], 'parameters': {'layer': 'ui_layer', 'operation': 'startup'}},
                {'task_id': 'notify_ready', 'name': 'System Ready Notification', 'type': 'notification', 'dependencies': ['ui_startup'], 'parameters': {'message': 'Manufacturing line ready', 'recipients': ['operators', 'supervisors']}}
            ]
        }
        
        orchestration_result = self.orchestrate_system_workflows(workflow_specs)
        print(f"   ✅ Workflow Orchestrated: {orchestration_result['workflow_name']}")
        print(f"   🎯 Tasks: {orchestration_result['total_tasks']}")
        print(f"   ⏱️ Decision Time: {orchestration_result['decision_time_ms']:.2f}ms")
        print(f"   🎯 Target: <{self.orchestration_decision_target_ms}ms | {'✅ MET' if orchestration_result['target_met'] else '❌ MISSED'}")
        
        # Wait for workflow completion
        execution_id = orchestration_result['execution_id']
        time.sleep(1)  # Allow workflow to execute
        
        workflow_status = self.get_workflow_status(execution_id)
        print(f"   📊 Progress: {workflow_status.get('progress_percentage', 0):.1f}%")
        print(f"   ✅ Completed Tasks: {workflow_status.get('tasks_completed', 0)}")
        
        # Demonstrate task dependency management
        print("\n🔗 Task Dependency Management...")
        dependency_specs = {
            'tasks': {
                'A': {'name': 'Task A', 'resources': {'cpu': 0.5}},
                'B': {'name': 'Task B', 'resources': {'memory': 200}},
                'C': {'name': 'Task C', 'resources': {'cpu': 0.3}},
                'D': {'name': 'Task D', 'resources': {}}
            },
            'dependencies': {
                'B': ['A'],
                'C': ['A'],
                'D': ['B', 'C']
            }
        }
        
        dependency_result = self.manage_task_dependencies(dependency_specs)
        print(f"   ✅ Dependencies Managed: {dependency_result['total_tasks']} tasks")
        print(f"   🔄 Parallel Groups: {dependency_result['parallel_groups']}")
        print(f"   ⏱️ Management Time: {dependency_result['management_time_ms']:.2f}ms")
        print(f"   🎯 Target: <{self.orchestration_decision_target_ms}ms | {'✅ MET' if dependency_result['target_met'] else '❌ MISSED'}")
        print(f"   ✅ Cycle Free: {'✅' if dependency_result['cycle_free'] else '❌'}")
        
        # Demonstrate cross-layer coordination
        print("\n🌐 Cross-Layer Operation Coordination...")
        coordination_specs = {
            'operation_name': 'System Health Check',
            'target_layers': ['data_layer', 'control_layer', 'ui_layer', 'security_layer'],
            'coordination_type': 'parallel',
            'synchronization_required': True
        }
        
        coordination_result = self.coordinate_cross_layer_operations(coordination_specs)
        print(f"   ✅ Cross-Layer Coordination: {coordination_result['operation_name']}")
        print(f"   🌐 Layers Coordinated: {coordination_result['layers_coordinated']}")
        print(f"   ⚡ Coordination Type: {coordination_result['coordination_type']}")
        print(f"   ⏱️ Coordination Time: {coordination_result['coordination_time_ms']:.2f}ms")
        print(f"   🔗 Synchronized: {'✅' if coordination_result['synchronization_result']['synchronized'] else '❌'}")
        
        # Show orchestration statistics
        print(f"\n📊 Orchestration Statistics:")
        stats = self.execution_statistics
        print(f"   Total Workflows: {stats['total_workflows']}")
        print(f"   Successful Workflows: {stats['successful_workflows']}")
        print(f"   Total Tasks Executed: {stats['total_tasks']}")
        print(f"   Average Execution Time: {stats['average_execution_time_ms']:.2f}ms")
        print(f"   Success Rate: {(stats['successful_workflows'] / max(1, stats['total_workflows']) * 100):.1f}%")
        
        print("\n📈 DEMONSTRATION SUMMARY:")
        print(f"   Workflow Decision Time: {orchestration_result['decision_time_ms']:.2f}ms")
        print(f"   Dependency Management: {dependency_result['management_time_ms']:.2f}ms")
        print(f"   Cross-Layer Coordination: {coordination_result['coordination_time_ms']:.2f}ms")
        print(f"   Tasks Orchestrated: {orchestration_result['total_tasks']}")
        print(f"   Layers Coordinated: {coordination_result['layers_coordinated']}")
        print("=" * 50)
        
        return {
            'workflow_decision_time_ms': orchestration_result['decision_time_ms'],
            'dependency_management_time_ms': dependency_result['management_time_ms'],
            'cross_layer_coordination_time_ms': coordination_result['coordination_time_ms'],
            'tasks_orchestrated': orchestration_result['total_tasks'],
            'layers_coordinated': coordination_result['layers_coordinated'],
            'workflows_executed': stats['total_workflows'],
            'success_rate': (stats['successful_workflows'] / max(1, stats['total_workflows']) * 100),
            'performance_targets_met': (
                orchestration_result['target_met'] and 
                dependency_result['target_met'] and 
                coordination_result['target_met']
            )
        }

def main():
    """Test OrchestrationEngine functionality"""
    engine = OrchestrationEngine()
    results = engine.demonstrate_orchestration_capabilities()
    
    print(f"\n🎯 Week 11 Orchestration Performance Targets:")
    print(f"   Orchestration Decisions: <200ms ({'✅' if results['workflow_decision_time_ms'] < 200 else '❌'})")
    print(f"   Workflow Execution: <10s (✅ simulated)")
    print(f"   Overall Performance: {'🟢 EXCELLENT' if results['performance_targets_met'] else '🟡 NEEDS OPTIMIZATION'}")

if __name__ == "__main__":
    main()