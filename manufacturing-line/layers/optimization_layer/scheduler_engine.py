"""
Week 4: SchedulerEngine - Intelligent Production Scheduling with Constraint Optimization

This module implements intelligent production scheduling capabilities with AI-powered
constraint optimization, dynamic rescheduling, and resource conflict resolution.

Key Features:
- AI-powered production scheduling with constraint satisfaction
- Dynamic rescheduling for production disruptions
- Resource conflict resolution with optimization algorithms
- Real-time schedule optimization with <300ms performance target
- Integration with OptimizationLayerEngine for advanced algorithms

Author: Claude Code
Date: 2024-08-28
Version: 1.0
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
import queue
import heapq
from collections import defaultdict
import random

from .optimization_layer_engine import OptimizationLayerEngine, OptimizationObjective, OptimizationConstraint

class ScheduleStatus(Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"

class ResourceType(Enum):
    EQUIPMENT = "equipment"
    LABOR = "labor"
    MATERIAL = "material"
    TOOL = "tool"
    UTILITY = "utility"

class DisruptionType(Enum):
    EQUIPMENT_FAILURE = "equipment_failure"
    MATERIAL_SHORTAGE = "material_shortage"
    QUALITY_ISSUE = "quality_issue"
    LABOR_SHORTAGE = "labor_shortage"
    RUSH_ORDER = "rush_order"
    SCHEDULE_CHANGE = "schedule_change"

@dataclass
class ScheduledTask:
    task_id: str
    order_id: str
    operation_name: str
    estimated_duration_hours: float
    planned_start_time: datetime
    planned_end_time: datetime
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    status: ScheduleStatus = ScheduleStatus.PLANNED
    priority: int = 3  # 1-5, 5 being highest
    required_resources: Dict[ResourceType, List[str]] = None
    dependencies: List[str] = None  # Task IDs this task depends on
    constraints: List[str] = None
    progress: float = 0.0  # 0.0 to 1.0

@dataclass
class Resource:
    resource_id: str
    resource_type: ResourceType
    capacity: float
    availability_windows: List[Tuple[datetime, datetime]]
    current_allocation: float = 0.0
    scheduled_tasks: List[str] = None  # Task IDs using this resource
    maintenance_windows: List[Tuple[datetime, datetime]] = None

@dataclass
class ScheduleConflict:
    conflict_id: str
    conflict_type: str
    affected_tasks: List[str]
    affected_resources: List[str]
    severity: float  # 0.0 to 1.0
    detected_at: datetime
    resolution_options: List[Dict[str, Any]]

@dataclass
class DisruptionEvent:
    disruption_id: str
    disruption_type: DisruptionType
    affected_resources: List[str]
    affected_tasks: List[str]
    impact_severity: float  # 0.0 to 1.0
    estimated_duration_hours: float
    occurred_at: datetime
    description: str

@dataclass
class ScheduleResult:
    schedule_id: str
    tasks: List[ScheduledTask]
    resource_utilization: Dict[str, float]
    total_duration_hours: float
    schedule_efficiency: float
    conflicts_resolved: int
    optimization_time_ms: float
    success: bool

class SchedulerEngine:
    """Intelligent production scheduling with constraint optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SchedulerEngine with configuration."""
        self.config = config or {}
        
        # Performance configuration
        self.performance_target_ms = self.config.get('performance_target_ms', 300)
        self.max_scheduling_time_ms = self.config.get('max_scheduling_time_ms', 10000)
        
        # Scheduling parameters
        self.scheduling_horizon_hours = self.config.get('scheduling_horizon_hours', 168)  # 1 week
        self.rescheduling_buffer_hours = self.config.get('rescheduling_buffer_hours', 2)
        self.resource_utilization_target = self.config.get('resource_utilization_target', 0.85)
        
        # Integration with optimization engine
        self.optimization_engine = OptimizationLayerEngine(self.config.get('optimization_config', {}))
        
        # Schedule storage
        self.current_schedules: Dict[str, ScheduleResult] = {}
        self.schedule_history: List[ScheduleResult] = []
        self.active_disruptions: Dict[str, DisruptionEvent] = {}
        self.resolved_conflicts: List[ScheduleConflict] = []
        
        # Resource tracking
        self.available_resources: Dict[str, Resource] = {}
        self.resource_conflicts: Dict[str, List[ScheduleConflict]] = defaultdict(list)
        
        # Performance monitoring
        self.performance_metrics = {
            'avg_scheduling_time_ms': 0,
            'successful_schedules': 0,
            'total_schedules': 0,
            'conflicts_resolved': 0,
            'disruptions_handled': 0,
            'schedule_efficiency_avg': 0.0
        }
        
        logging.info(f"SchedulerEngine initialized with {self.performance_target_ms}ms target")

    def create_optimal_schedule(self,
                              orders: List[Dict[str, Any]],
                              available_resources: Dict[str, Resource],
                              constraints: List[Dict[str, Any]],
                              objectives: List[OptimizationObjective] = None) -> ScheduleResult:
        """Generate optimal production schedule using constraint optimization."""
        start_time = time.time()
        schedule_id = f"schedule_{int(time.time())}"
        
        try:
            # Validate inputs
            if not orders:
                raise ValueError("Orders list cannot be empty")
            
            # Set default objectives
            if objectives is None:
                objectives = [OptimizationObjective.MAXIMIZE_THROUGHPUT, OptimizationObjective.MAXIMIZE_EFFICIENCY]
            
            # Update available resources
            self.available_resources.update(available_resources)
            
            # Convert orders to tasks
            tasks = self._convert_orders_to_tasks(orders)
            
            # Build constraint graph
            task_constraints = self._build_constraint_graph(tasks, constraints)
            
            # Initial scheduling using greedy algorithm
            initial_schedule = self._greedy_scheduling(tasks, available_resources, task_constraints)
            
            # Optimize schedule using optimization engine
            optimized_schedule = self._optimize_schedule(initial_schedule, objectives, constraints)
            
            # Resolve resource conflicts
            final_schedule = self._resolve_resource_conflicts(optimized_schedule)
            
            # Calculate performance metrics
            resource_utilization = self._calculate_resource_utilization(final_schedule, available_resources)
            total_duration = self._calculate_total_duration(final_schedule)
            efficiency = self._calculate_schedule_efficiency(final_schedule, resource_utilization)
            
            # Create result
            computation_time = (time.time() - start_time) * 1000
            result = ScheduleResult(
                schedule_id=schedule_id,
                tasks=final_schedule,
                resource_utilization=resource_utilization,
                total_duration_hours=total_duration,
                schedule_efficiency=efficiency,
                conflicts_resolved=len(self.resolved_conflicts),
                optimization_time_ms=computation_time,
                success=computation_time < self.max_scheduling_time_ms
            )
            
            # Store result
            self.current_schedules[schedule_id] = result
            self.schedule_history.append(result)
            
            # Update performance metrics
            self._update_scheduling_metrics(computation_time, result.success, efficiency)
            
            return result
            
        except Exception as e:
            logging.error(f"Schedule creation failed: {e}")
            computation_time = (time.time() - start_time) * 1000
            
            return ScheduleResult(
                schedule_id=schedule_id,
                tasks=[],
                resource_utilization={},
                total_duration_hours=0.0,
                schedule_efficiency=0.0,
                conflicts_resolved=0,
                optimization_time_ms=computation_time,
                success=False
            )

    def handle_dynamic_rescheduling(self,
                                  disruption_event: DisruptionEvent,
                                  current_schedule: ScheduleResult) -> ScheduleResult:
        """Handle dynamic rescheduling for production disruptions."""
        start_time = time.time()
        
        try:
            # Store disruption event
            self.active_disruptions[disruption_event.disruption_id] = disruption_event
            
            # Identify affected tasks
            affected_tasks = self._identify_affected_tasks(disruption_event, current_schedule)
            
            # Calculate disruption impact
            impact_analysis = self._analyze_disruption_impact(disruption_event, affected_tasks)
            
            # Determine rescheduling strategy
            strategy = self._determine_rescheduling_strategy(disruption_event, impact_analysis)
            
            # Execute rescheduling
            if strategy == "full_reschedule":
                rescheduled_tasks = self._full_reschedule(current_schedule, disruption_event)
            elif strategy == "partial_reschedule":
                rescheduled_tasks = self._partial_reschedule(current_schedule, affected_tasks, disruption_event)
            elif strategy == "delay_propagation":
                rescheduled_tasks = self._propagate_delays(current_schedule, affected_tasks, disruption_event)
            else:  # "no_action"
                rescheduled_tasks = current_schedule.tasks
            
            # Create new schedule result
            new_schedule_id = f"{current_schedule.schedule_id}_rescheduled_{int(time.time())}"
            resource_utilization = self._calculate_resource_utilization(rescheduled_tasks, self.available_resources)
            total_duration = self._calculate_total_duration(rescheduled_tasks)
            efficiency = self._calculate_schedule_efficiency(rescheduled_tasks, resource_utilization)
            
            computation_time = (time.time() - start_time) * 1000
            result = ScheduleResult(
                schedule_id=new_schedule_id,
                tasks=rescheduled_tasks,
                resource_utilization=resource_utilization,
                total_duration_hours=total_duration,
                schedule_efficiency=efficiency,
                conflicts_resolved=len(self.resolved_conflicts),
                optimization_time_ms=computation_time,
                success=True
            )
            
            # Update current schedule
            self.current_schedules[new_schedule_id] = result
            
            # Update performance metrics
            self.performance_metrics['disruptions_handled'] += 1
            
            logging.info(f"Dynamic rescheduling completed in {computation_time:.2f}ms for disruption {disruption_event.disruption_id}")
            
            return result
            
        except Exception as e:
            logging.error(f"Dynamic rescheduling failed: {e}")
            return current_schedule  # Return original schedule if rescheduling fails

    def resolve_resource_conflicts(self,
                                 conflicting_tasks: List[ScheduledTask],
                                 available_resources: Dict[str, Resource]) -> List[ScheduledTask]:
        """Resolve resource conflicts using intelligent optimization."""
        start_time = time.time()
        
        try:
            # Detect all conflicts
            conflicts = self._detect_resource_conflicts(conflicting_tasks, available_resources)
            
            if not conflicts:
                return conflicting_tasks  # No conflicts to resolve
            
            # Sort conflicts by severity
            conflicts.sort(key=lambda c: c.severity, reverse=True)
            
            resolved_tasks = conflicting_tasks.copy()
            
            # Resolve each conflict
            for conflict in conflicts:
                resolution_strategy = self._select_conflict_resolution_strategy(conflict)
                
                if resolution_strategy == "time_shift":
                    resolved_tasks = self._resolve_by_time_shift(resolved_tasks, conflict)
                elif resolution_strategy == "resource_substitution":
                    resolved_tasks = self._resolve_by_resource_substitution(resolved_tasks, conflict, available_resources)
                elif resolution_strategy == "task_splitting":
                    resolved_tasks = self._resolve_by_task_splitting(resolved_tasks, conflict)
                elif resolution_strategy == "priority_based":
                    resolved_tasks = self._resolve_by_priority(resolved_tasks, conflict)
                
                # Record resolved conflict
                self.resolved_conflicts.append(conflict)
                self.performance_metrics['conflicts_resolved'] += 1
            
            computation_time = (time.time() - start_time) * 1000
            logging.info(f"Resolved {len(conflicts)} resource conflicts in {computation_time:.2f}ms")
            
            return resolved_tasks
            
        except Exception as e:
            logging.error(f"Resource conflict resolution failed: {e}")
            return conflicting_tasks

    def _convert_orders_to_tasks(self, orders: List[Dict[str, Any]]) -> List[ScheduledTask]:
        """Convert production orders to scheduled tasks."""
        tasks = []
        
        for order in orders:
            order_id = order.get('order_id', f"order_{len(tasks)}")
            operations = order.get('operations', [{'name': 'default_operation', 'duration': 4}])
            priority = order.get('priority', 3)
            due_date = order.get('due_date')
            
            for i, operation in enumerate(operations):
                task_id = f"{order_id}_op_{i}"
                duration = operation.get('duration_hours', operation.get('estimated_hours', 4))
                
                # Calculate planned times
                if i == 0:
                    start_time = datetime.now()
                else:
                    # Dependent on previous task
                    start_time = datetime.now() + timedelta(hours=i * duration)
                
                end_time = start_time + timedelta(hours=duration)
                
                # Build required resources
                required_resources = {}
                if 'line_id' in order:
                    required_resources[ResourceType.EQUIPMENT] = [order['line_id']]
                if 'operators_required' in operation:
                    required_resources[ResourceType.LABOR] = [f"operator_{j}" for j in range(operation['operators_required'])]
                
                # Build dependencies
                dependencies = []
                if i > 0:
                    dependencies.append(f"{order_id}_op_{i-1}")
                
                task = ScheduledTask(
                    task_id=task_id,
                    order_id=order_id,
                    operation_name=operation.get('name', f'operation_{i}'),
                    estimated_duration_hours=duration,
                    planned_start_time=start_time,
                    planned_end_time=end_time,
                    priority=priority,
                    required_resources=required_resources,
                    dependencies=dependencies,
                    constraints=operation.get('constraints', [])
                )
                
                tasks.append(task)
        
        return tasks

    def _build_constraint_graph(self, tasks: List[ScheduledTask], constraints: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build constraint dependency graph."""
        constraint_graph = defaultdict(list)
        
        # Add task dependencies
        for task in tasks:
            if task.dependencies:
                for dependency in task.dependencies:
                    constraint_graph[task.task_id].append(dependency)
        
        # Add explicit constraints
        for constraint in constraints:
            if constraint.get('type') == 'precedence':
                predecessor = constraint.get('predecessor_task')
                successor = constraint.get('successor_task')
                if predecessor and successor:
                    constraint_graph[successor].append(predecessor)
        
        return dict(constraint_graph)

    def _greedy_scheduling(self,
                          tasks: List[ScheduledTask],
                          available_resources: Dict[str, Resource],
                          constraints: Dict[str, List[str]]) -> List[ScheduledTask]:
        """Initial greedy scheduling algorithm."""
        # Sort tasks by priority and earliest start time
        sorted_tasks = sorted(tasks, key=lambda t: (-t.priority, t.planned_start_time))
        
        scheduled_tasks = []
        resource_schedules = {rid: [] for rid in available_resources.keys()}
        
        for task in sorted_tasks:
            # Find earliest possible start time considering dependencies and resources
            earliest_start = self._find_earliest_start_time(task, scheduled_tasks, constraints, resource_schedules)
            
            # Update task timing
            task.planned_start_time = earliest_start
            task.planned_end_time = earliest_start + timedelta(hours=task.estimated_duration_hours)
            
            # Reserve resources
            self._reserve_resources(task, resource_schedules)
            
            scheduled_tasks.append(task)
        
        return scheduled_tasks

    def _optimize_schedule(self,
                          initial_schedule: List[ScheduledTask],
                          objectives: List[OptimizationObjective],
                          constraints: List[Dict[str, Any]]) -> List[ScheduledTask]:
        """Optimize schedule using optimization engine."""
        try:
            # Convert schedule to optimization problem
            orders_data = self._convert_schedule_to_orders(initial_schedule)
            resources_data = {rid: {'capacity': res.capacity} for rid, res in self.available_resources.items()}
            
            # Create optimization constraints
            opt_constraints = []
            for constraint in constraints:
                if constraint.get('type') == 'time_window':
                    opt_constraints.append(OptimizationConstraint(
                        name=f"time_window_{constraint.get('task_id')}",
                        constraint_type="time_limit",
                        value=constraint.get('max_hours', 24),
                        operator="<=",
                        priority=constraint.get('priority', 1)
                    ))
            
            # Run optimization
            optimization_result = self.optimization_engine.optimize_production_schedule(
                orders_data, resources_data, opt_constraints, objectives
            )
            
            if optimization_result.success and optimization_result.best_solution:
                # Apply optimization results to schedule
                return self._apply_optimization_to_schedule(initial_schedule, optimization_result)
            else:
                return initial_schedule
                
        except Exception as e:
            logging.warning(f"Schedule optimization failed, using initial schedule: {e}")
            return initial_schedule

    def _resolve_resource_conflicts(self, schedule: List[ScheduledTask]) -> List[ScheduledTask]:
        """Resolve resource conflicts in the schedule."""
        return self.resolve_resource_conflicts(schedule, self.available_resources)

    def _find_earliest_start_time(self,
                                 task: ScheduledTask,
                                 scheduled_tasks: List[ScheduledTask],
                                 constraints: Dict[str, List[str]],
                                 resource_schedules: Dict[str, List[Tuple[datetime, datetime]]]) -> datetime:
        """Find earliest possible start time for a task."""
        earliest_start = task.planned_start_time
        
        # Check dependency constraints
        if task.task_id in constraints:
            for dependency_id in constraints[task.task_id]:
                dependency_task = next((t for t in scheduled_tasks if t.task_id == dependency_id), None)
                if dependency_task:
                    earliest_start = max(earliest_start, dependency_task.planned_end_time)
        
        # Check resource availability
        if task.required_resources:
            for resource_type, resource_list in task.required_resources.items():
                for resource_id in resource_list:
                    if resource_id in resource_schedules:
                        # Find next available slot
                        duration = timedelta(hours=task.estimated_duration_hours)
                        available_slot = self._find_next_available_slot(
                            resource_schedules[resource_id], earliest_start, duration
                        )
                        earliest_start = max(earliest_start, available_slot)
        
        return earliest_start

    def _find_next_available_slot(self,
                                 resource_schedule: List[Tuple[datetime, datetime]],
                                 earliest_start: datetime,
                                 duration: timedelta) -> datetime:
        """Find next available slot in resource schedule."""
        if not resource_schedule:
            return earliest_start
        
        # Sort schedule by start time
        sorted_schedule = sorted(resource_schedule)
        
        # Check if we can fit before first task
        if earliest_start + duration <= sorted_schedule[0][0]:
            return earliest_start
        
        # Find gap between tasks
        for i in range(len(sorted_schedule) - 1):
            gap_start = max(earliest_start, sorted_schedule[i][1])
            gap_end = sorted_schedule[i + 1][0]
            
            if gap_end - gap_start >= duration:
                return gap_start
        
        # Schedule after last task
        return max(earliest_start, sorted_schedule[-1][1])

    def _reserve_resources(self, task: ScheduledTask, resource_schedules: Dict[str, List[Tuple[datetime, datetime]]]):
        """Reserve resources for a task."""
        if not task.required_resources:
            return
        
        time_slot = (task.planned_start_time, task.planned_end_time)
        
        for resource_type, resource_list in task.required_resources.items():
            for resource_id in resource_list:
                if resource_id not in resource_schedules:
                    resource_schedules[resource_id] = []
                resource_schedules[resource_id].append(time_slot)

    def _detect_resource_conflicts(self,
                                  tasks: List[ScheduledTask],
                                  available_resources: Dict[str, Resource]) -> List[ScheduleConflict]:
        """Detect resource conflicts in the schedule."""
        conflicts = []
        resource_usage = defaultdict(list)
        
        # Build resource usage map
        for task in tasks:
            if task.required_resources:
                for resource_type, resource_list in task.required_resources.items():
                    for resource_id in resource_list:
                        resource_usage[resource_id].append(task)
        
        # Check for conflicts
        for resource_id, using_tasks in resource_usage.items():
            if len(using_tasks) <= 1:
                continue
            
            # Check for time overlaps
            for i in range(len(using_tasks)):
                for j in range(i + 1, len(using_tasks)):
                    task1, task2 = using_tasks[i], using_tasks[j]
                    
                    if self._tasks_overlap(task1, task2):
                        conflict = ScheduleConflict(
                            conflict_id=f"conflict_{resource_id}_{int(time.time())}",
                            conflict_type="resource_overlap",
                            affected_tasks=[task1.task_id, task2.task_id],
                            affected_resources=[resource_id],
                            severity=self._calculate_conflict_severity(task1, task2),
                            detected_at=datetime.now(),
                            resolution_options=self._generate_resolution_options(task1, task2, resource_id)
                        )
                        conflicts.append(conflict)
        
        return conflicts

    def _tasks_overlap(self, task1: ScheduledTask, task2: ScheduledTask) -> bool:
        """Check if two tasks have overlapping time windows."""
        return (task1.planned_start_time < task2.planned_end_time and
                task2.planned_start_time < task1.planned_end_time)

    def _calculate_conflict_severity(self, task1: ScheduledTask, task2: ScheduledTask) -> float:
        """Calculate severity of conflict between two tasks."""
        # Base severity on priority and overlap duration
        priority_factor = (task1.priority + task2.priority) / 10.0
        
        # Calculate overlap duration
        overlap_start = max(task1.planned_start_time, task2.planned_start_time)
        overlap_end = min(task1.planned_end_time, task2.planned_end_time)
        overlap_hours = max(0, (overlap_end - overlap_start).total_seconds() / 3600)
        
        duration_factor = overlap_hours / max(task1.estimated_duration_hours, task2.estimated_duration_hours)
        
        return min(1.0, priority_factor * 0.5 + duration_factor * 0.5)

    def _generate_resolution_options(self, task1: ScheduledTask, task2: ScheduledTask, resource_id: str) -> List[Dict[str, Any]]:
        """Generate resolution options for resource conflict."""
        options = []
        
        # Time shift options
        options.append({
            'type': 'time_shift',
            'description': f'Delay lower priority task',
            'impact': 'schedule_delay'
        })
        
        # Resource substitution option
        options.append({
            'type': 'resource_substitution',
            'description': f'Use alternative resource for {resource_id}',
            'impact': 'resource_change'
        })
        
        # Priority-based resolution
        if task1.priority != task2.priority:
            higher_priority = task1 if task1.priority > task2.priority else task2
            options.append({
                'type': 'priority_based',
                'description': f'Give priority to {higher_priority.task_id}',
                'impact': 'task_delay'
            })
        
        return options

    def _select_conflict_resolution_strategy(self, conflict: ScheduleConflict) -> str:
        """Select best conflict resolution strategy."""
        if conflict.severity > 0.8:
            return "priority_based"
        elif len(conflict.affected_resources) == 1:
            return "time_shift"
        else:
            return "resource_substitution"

    def _resolve_by_time_shift(self, tasks: List[ScheduledTask], conflict: ScheduleConflict) -> List[ScheduledTask]:
        """Resolve conflict by shifting task timing."""
        affected_tasks = [t for t in tasks if t.task_id in conflict.affected_tasks]
        
        if len(affected_tasks) >= 2:
            # Sort by priority (lower priority gets shifted)
            affected_tasks.sort(key=lambda t: t.priority)
            task_to_shift = affected_tasks[0]
            reference_task = affected_tasks[1]
            
            # Shift to after reference task
            new_start = reference_task.planned_end_time + timedelta(minutes=30)  # Buffer time
            time_shift = new_start - task_to_shift.planned_start_time
            
            # Update task timing
            for task in tasks:
                if task.task_id == task_to_shift.task_id:
                    task.planned_start_time = new_start
                    task.planned_end_time = new_start + timedelta(hours=task.estimated_duration_hours)
                    break
        
        return tasks

    def _resolve_by_resource_substitution(self,
                                        tasks: List[ScheduledTask],
                                        conflict: ScheduleConflict,
                                        available_resources: Dict[str, Resource]) -> List[ScheduledTask]:
        """Resolve conflict by substituting resources."""
        # Simplified resource substitution logic
        for task in tasks:
            if task.task_id in conflict.affected_tasks:
                if task.required_resources:
                    # Try to find alternative resources
                    for resource_type, resource_list in task.required_resources.items():
                        for resource_id in resource_list:
                            if resource_id in conflict.affected_resources:
                                # Find alternative resource of same type
                                alternatives = [rid for rid, res in available_resources.items()
                                              if res.resource_type == resource_type and rid != resource_id]
                                if alternatives:
                                    # Use first available alternative
                                    resource_list[resource_list.index(resource_id)] = alternatives[0]
                                    break
        
        return tasks

    def _resolve_by_task_splitting(self, tasks: List[ScheduledTask], conflict: ScheduleConflict) -> List[ScheduledTask]:
        """Resolve conflict by splitting tasks into smaller parts."""
        # Simplified task splitting - this would be more complex in real implementation
        affected_tasks = [t for t in tasks if t.task_id in conflict.affected_tasks]
        
        for task in affected_tasks:
            if task.estimated_duration_hours > 2:  # Only split longer tasks
                # Split into two parts
                split_duration = task.estimated_duration_hours / 2
                
                # Create second part
                second_part = ScheduledTask(
                    task_id=f"{task.task_id}_part2",
                    order_id=task.order_id,
                    operation_name=f"{task.operation_name}_part2",
                    estimated_duration_hours=split_duration,
                    planned_start_time=task.planned_end_time + timedelta(hours=1),  # Buffer
                    planned_end_time=task.planned_end_time + timedelta(hours=1 + split_duration),
                    priority=task.priority,
                    required_resources=task.required_resources,
                    dependencies=[task.task_id]
                )
                
                # Update first part
                task.estimated_duration_hours = split_duration
                task.planned_end_time = task.planned_start_time + timedelta(hours=split_duration)
                task.operation_name = f"{task.operation_name}_part1"
                
                tasks.append(second_part)
                break
        
        return tasks

    def _resolve_by_priority(self, tasks: List[ScheduledTask], conflict: ScheduleConflict) -> List[ScheduledTask]:
        """Resolve conflict based on task priorities."""
        affected_tasks = [t for t in tasks if t.task_id in conflict.affected_tasks]
        
        if len(affected_tasks) >= 2:
            # Sort by priority (highest first)
            affected_tasks.sort(key=lambda t: t.priority, reverse=True)
            high_priority_task = affected_tasks[0]
            
            # Delay other tasks
            for task in affected_tasks[1:]:
                new_start = high_priority_task.planned_end_time + timedelta(minutes=30)
                time_shift = new_start - task.planned_start_time
                
                # Update task timing
                for t in tasks:
                    if t.task_id == task.task_id:
                        t.planned_start_time = new_start
                        t.planned_end_time = new_start + timedelta(hours=t.estimated_duration_hours)
                        break
        
        return tasks

    def _identify_affected_tasks(self, disruption: DisruptionEvent, schedule: ScheduleResult) -> List[ScheduledTask]:
        """Identify tasks affected by disruption event."""
        affected_tasks = []
        
        for task in schedule.tasks:
            # Check if task uses affected resources
            if task.required_resources:
                for resource_type, resource_list in task.required_resources.items():
                    if any(res_id in disruption.affected_resources for res_id in resource_list):
                        affected_tasks.append(task)
                        break
            
            # Check if task is directly listed as affected
            if task.task_id in disruption.affected_tasks:
                affected_tasks.append(task)
        
        return affected_tasks

    def _analyze_disruption_impact(self, disruption: DisruptionEvent, affected_tasks: List[ScheduledTask]) -> Dict[str, Any]:
        """Analyze impact of disruption on schedule."""
        total_affected_duration = sum(task.estimated_duration_hours for task in affected_tasks)
        high_priority_affected = len([task for task in affected_tasks if task.priority >= 4])
        
        return {
            'affected_task_count': len(affected_tasks),
            'total_affected_duration_hours': total_affected_duration,
            'high_priority_tasks_affected': high_priority_affected,
            'estimated_delay_hours': disruption.estimated_duration_hours,
            'severity_score': disruption.impact_severity
        }

    def _determine_rescheduling_strategy(self, disruption: DisruptionEvent, impact: Dict[str, Any]) -> str:
        """Determine appropriate rescheduling strategy."""
        if impact['severity_score'] > 0.8 or impact['high_priority_tasks_affected'] > 2:
            return "full_reschedule"
        elif impact['affected_task_count'] > 5:
            return "partial_reschedule"
        elif impact['estimated_delay_hours'] > 4:
            return "delay_propagation"
        else:
            return "no_action"

    def _full_reschedule(self, current_schedule: ScheduleResult, disruption: DisruptionEvent) -> List[ScheduledTask]:
        """Perform full rescheduling."""
        # Extract unaffected tasks that haven't started
        unaffected_tasks = []
        for task in current_schedule.tasks:
            if (task.status == ScheduleStatus.PLANNED and 
                task.task_id not in disruption.affected_tasks):
                unaffected_tasks.append(task)
        
        # Create new orders from unaffected tasks
        orders_data = self._convert_schedule_to_orders(unaffected_tasks)
        
        # Reschedule from current time
        new_schedule = self.create_optimal_schedule(orders_data, self.available_resources, [])
        return new_schedule.tasks

    def _partial_reschedule(self, current_schedule: ScheduleResult, affected_tasks: List[ScheduledTask], disruption: DisruptionEvent) -> List[ScheduledTask]:
        """Perform partial rescheduling of affected tasks."""
        updated_tasks = []
        
        for task in current_schedule.tasks:
            if task in affected_tasks:
                # Delay affected task
                delay_hours = disruption.estimated_duration_hours
                task.planned_start_time += timedelta(hours=delay_hours)
                task.planned_end_time += timedelta(hours=delay_hours)
                task.status = ScheduleStatus.RESCHEDULED
            
            updated_tasks.append(task)
        
        return updated_tasks

    def _propagate_delays(self, current_schedule: ScheduleResult, affected_tasks: List[ScheduledTask], disruption: DisruptionEvent) -> List[ScheduledTask]:
        """Propagate delays through dependent tasks."""
        updated_tasks = current_schedule.tasks.copy()
        delay_hours = disruption.estimated_duration_hours
        
        # Build dependency graph
        task_map = {task.task_id: task for task in updated_tasks}
        
        # Propagate delay through dependencies
        delayed_tasks = set(task.task_id for task in affected_tasks)
        
        while True:
            new_delayed = set()
            for task in updated_tasks:
                if task.task_id not in delayed_tasks and task.dependencies:
                    if any(dep_id in delayed_tasks for dep_id in task.dependencies):
                        new_delayed.add(task.task_id)
                        task.planned_start_time += timedelta(hours=delay_hours)
                        task.planned_end_time += timedelta(hours=delay_hours)
                        task.status = ScheduleStatus.DELAYED
            
            if not new_delayed:
                break
            
            delayed_tasks.update(new_delayed)
        
        return updated_tasks

    def _calculate_resource_utilization(self, tasks: List[ScheduledTask], resources: Dict[str, Resource]) -> Dict[str, float]:
        """Calculate resource utilization for the schedule."""
        utilization = {}
        
        for resource_id, resource in resources.items():
            total_scheduled_hours = 0.0
            
            for task in tasks:
                if task.required_resources:
                    for resource_type, resource_list in task.required_resources.items():
                        if resource_id in resource_list:
                            total_scheduled_hours += task.estimated_duration_hours
                            break
            
            # Calculate utilization (assuming 24/7 availability for simplification)
            available_hours = self.scheduling_horizon_hours
            utilization[resource_id] = min(1.0, total_scheduled_hours / available_hours) if available_hours > 0 else 0.0
        
        return utilization

    def _calculate_total_duration(self, tasks: List[ScheduledTask]) -> float:
        """Calculate total schedule duration."""
        if not tasks:
            return 0.0
        
        earliest_start = min(task.planned_start_time for task in tasks)
        latest_end = max(task.planned_end_time for task in tasks)
        
        return (latest_end - earliest_start).total_seconds() / 3600  # Convert to hours

    def _calculate_schedule_efficiency(self, tasks: List[ScheduledTask], resource_utilization: Dict[str, float]) -> float:
        """Calculate overall schedule efficiency."""
        if not tasks or not resource_utilization:
            return 0.0
        
        # Efficiency based on resource utilization and task priority completion
        avg_utilization = sum(resource_utilization.values()) / len(resource_utilization)
        
        # Priority completion efficiency
        total_priority_value = sum(task.priority for task in tasks)
        completed_on_time = len([task for task in tasks if task.status == ScheduleStatus.COMPLETED])
        completion_efficiency = completed_on_time / len(tasks) if tasks else 0
        
        return (avg_utilization * 0.6 + completion_efficiency * 0.4)

    def _convert_schedule_to_orders(self, tasks: List[ScheduledTask]) -> List[Dict[str, Any]]:
        """Convert schedule back to orders format for optimization."""
        orders = []
        order_tasks = defaultdict(list)
        
        # Group tasks by order
        for task in tasks:
            order_tasks[task.order_id].append(task)
        
        # Create orders
        for order_id, task_list in order_tasks.items():
            operations = []
            for task in task_list:
                operations.append({
                    'name': task.operation_name,
                    'duration_hours': task.estimated_duration_hours,
                    'constraints': task.constraints or []
                })
            
            orders.append({
                'order_id': order_id,
                'operations': operations,
                'priority': task_list[0].priority if task_list else 3
            })
        
        return orders

    def _apply_optimization_to_schedule(self, original_schedule: List[ScheduledTask], optimization_result) -> List[ScheduledTask]:
        """Apply optimization results to schedule."""
        # Simplified application of optimization results
        # In real implementation, would parse optimization solution and update task timings
        optimized_schedule = original_schedule.copy()
        
        if optimization_result.best_solution and 'schedule' in optimization_result.best_solution.parameters:
            schedule_params = optimization_result.best_solution.parameters['schedule']
            
            for task in optimized_schedule:
                task_schedule = schedule_params.get(task.task_id)
                if task_schedule:
                    # Update task timing based on optimization
                    start_hours = task_schedule.get('start_time', 0)
                    task.planned_start_time = datetime.now() + timedelta(hours=start_hours)
                    task.planned_end_time = task.planned_start_time + timedelta(hours=task.estimated_duration_hours)
        
        return optimized_schedule

    def _update_scheduling_metrics(self, computation_time: float, success: bool, efficiency: float):
        """Update scheduling performance metrics."""
        self.performance_metrics['total_schedules'] += 1
        
        if success:
            self.performance_metrics['successful_schedules'] += 1
        
        # Update average computation time
        total_schedules = self.performance_metrics['total_schedules']
        current_avg = self.performance_metrics['avg_scheduling_time_ms']
        self.performance_metrics['avg_scheduling_time_ms'] = (
            (current_avg * (total_schedules - 1) + computation_time) / total_schedules
        )
        
        # Update efficiency average
        current_eff_avg = self.performance_metrics['schedule_efficiency_avg']
        self.performance_metrics['schedule_efficiency_avg'] = (
            (current_eff_avg * (total_schedules - 1) + efficiency) / total_schedules
        )

    def get_schedule_status(self, schedule_id: str) -> Dict[str, Any]:
        """Get status of a specific schedule."""
        if schedule_id in self.current_schedules:
            schedule = self.current_schedules[schedule_id]
            return {
                'schedule_id': schedule_id,
                'status': 'active',
                'total_tasks': len(schedule.tasks),
                'completed_tasks': len([t for t in schedule.tasks if t.status == ScheduleStatus.COMPLETED]),
                'efficiency': schedule.schedule_efficiency,
                'conflicts_resolved': schedule.conflicts_resolved
            }
        else:
            return {
                'schedule_id': schedule_id,
                'status': 'not_found',
                'total_tasks': 0,
                'completed_tasks': 0,
                'efficiency': 0.0,
                'conflicts_resolved': 0
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics['active_schedules'] = len(self.current_schedules)
        metrics['active_disruptions'] = len(self.active_disruptions)
        return metrics

    def clear_schedule_history(self):
        """Clear schedule history to free memory."""
        self.schedule_history.clear()
        self.resolved_conflicts.clear()
        logging.info("Schedule history cleared")

    def __str__(self) -> str:
        return f"SchedulerEngine(target={self.performance_target_ms}ms, active_schedules={len(self.current_schedules)})"

    def __repr__(self) -> str:
        return self.__str__()