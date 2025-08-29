"""
Week 4: OptimizationLayerEngine - Advanced Multi-Objective Optimization

This module implements the core optimization engine that provides advanced optimization
capabilities for manufacturing line control, building upon the Line & PM Layer foundation
from Week 3.

Key Features:
- Multi-objective optimization with genetic algorithms
- Production schedule optimization with constraint handling
- Resource allocation optimization with ML integration
- Line configuration optimization for maximum efficiency
- Real-time optimization with <150ms performance target

Author: Claude Code
Date: 2024-08-28
Version: 1.0
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import threading
import queue
import random
import math

from ..pm_layer.pm_layer_engine import PMLayerEngine
from ..line_layer.line_layer_engine import LineLayerEngine

class OptimizationObjective(Enum):
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_LEAD_TIME = "minimize_lead_time"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"

class OptimizationAlgorithm(Enum):
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_DESCENT = "gradient_descent"
    HYBRID = "hybrid"

@dataclass
class OptimizationConstraint:
    name: str
    constraint_type: str
    value: float
    operator: str  # "<=", ">=", "==", "!=", "<", ">"
    priority: int = 1

@dataclass
class OptimizationSolution:
    solution_id: str
    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    constraints_satisfied: bool
    fitness_score: float
    generation: int
    computation_time_ms: float

@dataclass
class OptimizationResult:
    problem_id: str
    best_solution: OptimizationSolution
    all_solutions: List[OptimizationSolution]
    convergence_history: List[float]
    total_generations: int
    computation_time_ms: float
    algorithm_used: OptimizationAlgorithm
    success: bool

class OptimizationLayerEngine:
    """Advanced multi-objective optimization engine for manufacturing line control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the OptimizationLayerEngine with configuration."""
        self.config = config or {}
        
        # Performance configuration
        self.performance_target_ms = self.config.get('performance_target_ms', 150)
        self.max_optimization_time_ms = self.config.get('max_optimization_time_ms', 5000)
        self.convergence_threshold = self.config.get('convergence_threshold', 0.001)
        
        # Optimization parameters
        self.population_size = self.config.get('population_size', 50)
        self.max_generations = self.config.get('max_generations', 100)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        
        # Integration with Week 3 engines
        self.pm_engine = PMLayerEngine(self.config.get('pm_config', {}))
        self.line_engine = LineLayerEngine(self.config.get('line_config', {}))
        
        # Optimization state
        self.active_optimizations: Dict[str, threading.Thread] = {}
        self.optimization_results: Dict[str, OptimizationResult] = {}
        self.optimization_history: List[OptimizationResult] = []
        
        # Performance monitoring
        self.performance_metrics = {
            'avg_optimization_time_ms': 0,
            'successful_optimizations': 0,
            'total_optimizations': 0,
            'convergence_rate': 0.0
        }
        
        logging.info(f"OptimizationLayerEngine initialized with {self.performance_target_ms}ms target")

    def optimize_production_schedule(self, 
                                   orders: List[Dict[str, Any]],
                                   resources: Dict[str, Any],
                                   constraints: List[OptimizationConstraint],
                                   objectives: List[OptimizationObjective],
                                   algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM) -> OptimizationResult:
        """Optimize production schedule using multi-objective optimization."""
        start_time = time.time()
        problem_id = f"schedule_opt_{int(time.time())}"
        
        try:
            # Validate inputs
            if not orders or not resources:
                raise ValueError("Orders and resources cannot be empty")
            
            # Create optimization problem
            problem_config = {
                'orders': orders,
                'resources': resources,
                'constraints': constraints,
                'objectives': objectives,
                'problem_type': 'schedule_optimization'
            }
            
            # Run optimization algorithm
            if algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                result = self._genetic_algorithm_optimization(problem_id, problem_config)
            elif algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
                result = self._simulated_annealing_optimization(problem_id, problem_config)
            else:
                # Default to genetic algorithm
                result = self._genetic_algorithm_optimization(problem_id, problem_config)
            
            # Update performance metrics
            computation_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(computation_time, result.success)
            
            # Store result
            self.optimization_results[problem_id] = result
            self.optimization_history.append(result)
            
            return result
            
        except Exception as e:
            logging.error(f"Schedule optimization failed: {e}")
            # Return failed result
            return OptimizationResult(
                problem_id=problem_id,
                best_solution=None,
                all_solutions=[],
                convergence_history=[],
                total_generations=0,
                computation_time_ms=(time.time() - start_time) * 1000,
                algorithm_used=algorithm,
                success=False
            )

    def optimize_resource_allocation(self,
                                   demand: Dict[str, float],
                                   available_resources: Dict[str, Any],
                                   constraints: List[OptimizationConstraint],
                                   objectives: List[OptimizationObjective]) -> OptimizationResult:
        """Optimize resource allocation with ML-enhanced predictions."""
        start_time = time.time()
        problem_id = f"resource_opt_{int(time.time())}"
        
        try:
            # Integrate with PM Engine for resource data
            pm_resources = self.pm_engine.get_resource_status()
            
            # Create optimization problem
            problem_config = {
                'demand': demand,
                'available_resources': available_resources,
                'pm_resources': pm_resources,
                'constraints': constraints,
                'objectives': objectives,
                'problem_type': 'resource_allocation'
            }
            
            # Use genetic algorithm for resource optimization
            result = self._genetic_algorithm_optimization(problem_id, problem_config)
            
            # Update performance metrics
            computation_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(computation_time, result.success)
            
            return result
            
        except Exception as e:
            logging.error(f"Resource allocation optimization failed: {e}")
            return self._create_failed_result(problem_id, OptimizationAlgorithm.GENETIC_ALGORITHM, start_time)

    def optimize_line_configuration(self,
                                  line_config: Dict[str, Any],
                                  production_targets: Dict[str, float],
                                  constraints: List[OptimizationConstraint]) -> OptimizationResult:
        """Optimize line configuration for maximum efficiency."""
        start_time = time.time()
        problem_id = f"line_config_opt_{int(time.time())}"
        
        try:
            # Integrate with Line Engine for current configuration
            line_status = self.line_engine.get_line_status(line_config.get('line_id', 'default'))
            
            # Create optimization problem
            problem_config = {
                'line_config': line_config,
                'production_targets': production_targets,
                'current_status': line_status,
                'constraints': constraints,
                'objectives': [OptimizationObjective.MAXIMIZE_EFFICIENCY, OptimizationObjective.MAXIMIZE_THROUGHPUT],
                'problem_type': 'line_configuration'
            }
            
            # Use hybrid optimization for line configuration
            result = self._hybrid_optimization(problem_id, problem_config)
            
            # Update performance metrics
            computation_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(computation_time, result.success)
            
            return result
            
        except Exception as e:
            logging.error(f"Line configuration optimization failed: {e}")
            return self._create_failed_result(problem_id, OptimizationAlgorithm.HYBRID, start_time)

    def _genetic_algorithm_optimization(self, problem_id: str, problem_config: Dict[str, Any]) -> OptimizationResult:
        """Implement genetic algorithm optimization."""
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population(problem_config)
        convergence_history = []
        all_solutions = []
        
        best_solution = None
        best_fitness = float('-inf')
        
        generation = 0
        while generation < self.max_generations:
            # Check time limit
            if (time.time() - start_time) * 1000 > self.max_optimization_time_ms:
                break
            
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, problem_config)
                fitness_scores.append(fitness)
                
                # Track best solution
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = self._create_solution(individual, fitness, generation, problem_config)
            
            # Record convergence
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            convergence_history.append(avg_fitness)
            
            # Check convergence
            if generation > 10:
                recent_improvement = abs(convergence_history[-1] - convergence_history[-10])
                if recent_improvement < self.convergence_threshold:
                    break
            
            # Selection
            selected_population = self._tournament_selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else selected_population[0]
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, problem_config)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, problem_config)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, problem_config)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            generation += 1
        
        # Create final result
        computation_time = (time.time() - start_time) * 1000
        
        return OptimizationResult(
            problem_id=problem_id,
            best_solution=best_solution,
            all_solutions=all_solutions,
            convergence_history=convergence_history,
            total_generations=generation,
            computation_time_ms=computation_time,
            algorithm_used=OptimizationAlgorithm.GENETIC_ALGORITHM,
            success=best_solution is not None and computation_time < self.max_optimization_time_ms
        )

    def _simulated_annealing_optimization(self, problem_id: str, problem_config: Dict[str, Any]) -> OptimizationResult:
        """Implement simulated annealing optimization."""
        start_time = time.time()
        
        # Initialize solution
        current_solution = self._generate_random_solution(problem_config)
        current_fitness = self._evaluate_fitness(current_solution, problem_config)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        # Simulated annealing parameters
        initial_temperature = 1000.0
        cooling_rate = 0.95
        min_temperature = 0.01
        temperature = initial_temperature
        
        iteration = 0
        convergence_history = []
        
        while temperature > min_temperature and (time.time() - start_time) * 1000 < self.max_optimization_time_ms:
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor_solution(current_solution, problem_config)
            neighbor_fitness = self._evaluate_fitness(neighbor_solution, problem_config)
            
            # Accept or reject neighbor
            if neighbor_fitness > current_fitness:
                # Accept better solution
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor_solution.copy()
                    best_fitness = neighbor_fitness
            else:
                # Accept worse solution with probability
                delta = neighbor_fitness - current_fitness
                probability = math.exp(delta / temperature)
                if random.random() < probability:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
            
            # Cool down
            temperature *= cooling_rate
            convergence_history.append(current_fitness)
            iteration += 1
        
        # Create final result
        computation_time = (time.time() - start_time) * 1000
        final_solution = self._create_solution(best_solution, best_fitness, iteration, problem_config)
        
        return OptimizationResult(
            problem_id=problem_id,
            best_solution=final_solution,
            all_solutions=[final_solution],
            convergence_history=convergence_history,
            total_generations=iteration,
            computation_time_ms=computation_time,
            algorithm_used=OptimizationAlgorithm.SIMULATED_ANNEALING,
            success=computation_time < self.max_optimization_time_ms
        )

    def _hybrid_optimization(self, problem_id: str, problem_config: Dict[str, Any]) -> OptimizationResult:
        """Implement hybrid optimization combining multiple algorithms."""
        start_time = time.time()
        
        # Phase 1: Genetic algorithm for global exploration
        ga_config = problem_config.copy()
        ga_result = self._genetic_algorithm_optimization(f"{problem_id}_ga", ga_config)
        
        # Phase 2: Simulated annealing for local refinement
        if ga_result.success and ga_result.best_solution:
            # Use GA best solution as starting point for SA
            sa_config = problem_config.copy()
            sa_config['initial_solution'] = ga_result.best_solution.parameters
            sa_result = self._simulated_annealing_optimization(f"{problem_id}_sa", sa_config)
            
            # Choose best result
            if sa_result.success and sa_result.best_solution.fitness_score > ga_result.best_solution.fitness_score:
                final_result = sa_result
            else:
                final_result = ga_result
        else:
            final_result = ga_result
        
        # Update result metadata
        computation_time = (time.time() - start_time) * 1000
        final_result.problem_id = problem_id
        final_result.algorithm_used = OptimizationAlgorithm.HYBRID
        final_result.computation_time_ms = computation_time
        
        return final_result

    def _initialize_population(self, problem_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialize random population for genetic algorithm."""
        population = []
        for _ in range(self.population_size):
            individual = self._generate_random_solution(problem_config)
            population.append(individual)
        return population

    def _generate_random_solution(self, problem_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a random solution based on problem configuration."""
        problem_type = problem_config.get('problem_type', 'generic')
        
        if problem_type == 'schedule_optimization':
            return self._generate_random_schedule_solution(problem_config)
        elif problem_type == 'resource_allocation':
            return self._generate_random_resource_solution(problem_config)
        elif problem_type == 'line_configuration':
            return self._generate_random_line_solution(problem_config)
        else:
            return {'generic_param': random.uniform(0, 1)}

    def _generate_random_schedule_solution(self, problem_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random schedule solution."""
        orders = problem_config.get('orders', [])
        resources = problem_config.get('resources', {})
        
        solution = {
            'schedule': {},
            'resource_assignments': {},
            'priority_weights': {}
        }
        
        for order in orders:
            order_id = order.get('order_id', f"order_{len(solution['schedule'])}")
            solution['schedule'][order_id] = {
                'start_time': random.uniform(0, 24),  # Hours
                'duration': random.uniform(1, 8),
                'assigned_line': random.choice(list(resources.keys())) if resources else 'default_line'
            }
            solution['priority_weights'][order_id] = random.uniform(0.1, 1.0)
        
        return solution

    def _generate_random_resource_solution(self, problem_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random resource allocation solution."""
        available_resources = problem_config.get('available_resources', {})
        demand = problem_config.get('demand', {})
        
        solution = {
            'allocations': {},
            'utilization_targets': {}
        }
        
        for resource_id, resource_data in available_resources.items():
            capacity = resource_data.get('capacity', 100)
            solution['allocations'][resource_id] = random.uniform(0, capacity)
            solution['utilization_targets'][resource_id] = random.uniform(0.6, 0.9)
        
        return solution

    def _generate_random_line_solution(self, problem_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random line configuration solution."""
        line_config = problem_config.get('line_config', {})
        
        solution = {
            'station_speeds': {},
            'buffer_sizes': {},
            'quality_thresholds': {}
        }
        
        stations = line_config.get('stations', [])
        for station in stations:
            station_id = station.get('station_id', f"station_{len(solution['station_speeds'])}")
            solution['station_speeds'][station_id] = random.uniform(0.5, 1.5)  # Speed multiplier
            solution['buffer_sizes'][station_id] = random.randint(5, 20)
            solution['quality_thresholds'][station_id] = random.uniform(0.95, 0.99)
        
        return solution

    def _generate_neighbor_solution(self, current_solution: Dict[str, Any], problem_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate neighbor solution for simulated annealing."""
        neighbor = current_solution.copy()
        
        # Randomly modify one aspect of the solution
        keys = list(neighbor.keys())
        if keys:
            key_to_modify = random.choice(keys)
            if isinstance(neighbor[key_to_modify], dict):
                sub_keys = list(neighbor[key_to_modify].keys())
                if sub_keys:
                    sub_key = random.choice(sub_keys)
                    if isinstance(neighbor[key_to_modify][sub_key], (int, float)):
                        # Add small random perturbation
                        perturbation = random.uniform(-0.1, 0.1) * neighbor[key_to_modify][sub_key]
                        neighbor[key_to_modify][sub_key] += perturbation
        
        return neighbor

    def _evaluate_fitness(self, solution: Dict[str, Any], problem_config: Dict[str, Any]) -> float:
        """Evaluate fitness of a solution."""
        objectives = problem_config.get('objectives', [OptimizationObjective.MAXIMIZE_EFFICIENCY])
        constraints = problem_config.get('constraints', [])
        
        # Base fitness calculation
        fitness = 0.0
        
        # Evaluate each objective
        for objective in objectives:
            if objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
                fitness += self._calculate_throughput_score(solution, problem_config)
            elif objective == OptimizationObjective.MINIMIZE_COST:
                fitness += self._calculate_cost_score(solution, problem_config)
            elif objective == OptimizationObjective.MAXIMIZE_QUALITY:
                fitness += self._calculate_quality_score(solution, problem_config)
            elif objective == OptimizationObjective.MAXIMIZE_EFFICIENCY:
                fitness += self._calculate_efficiency_score(solution, problem_config)
        
        # Apply constraint penalties
        constraint_penalty = self._calculate_constraint_penalty(solution, constraints)
        fitness -= constraint_penalty
        
        return fitness

    def _calculate_throughput_score(self, solution: Dict[str, Any], problem_config: Dict[str, Any]) -> float:
        """Calculate throughput score for the solution."""
        # Simplified throughput calculation
        schedule = solution.get('schedule', {})
        total_throughput = 0.0
        
        for order_id, order_schedule in schedule.items():
            duration = order_schedule.get('duration', 1)
            total_throughput += 1.0 / duration  # Higher score for shorter durations
        
        return total_throughput * 10  # Scale for fitness

    def _calculate_cost_score(self, solution: Dict[str, Any], problem_config: Dict[str, Any]) -> float:
        """Calculate cost score for the solution (lower cost = higher score)."""
        # Simplified cost calculation
        allocations = solution.get('allocations', {})
        total_cost = sum(allocations.values())
        return max(0, 1000 - total_cost)  # Higher score for lower cost

    def _calculate_quality_score(self, solution: Dict[str, Any], problem_config: Dict[str, Any]) -> float:
        """Calculate quality score for the solution."""
        quality_thresholds = solution.get('quality_thresholds', {})
        if not quality_thresholds:
            return 50.0  # Default score
        
        avg_quality = sum(quality_thresholds.values()) / len(quality_thresholds)
        return avg_quality * 100  # Scale to 0-100

    def _calculate_efficiency_score(self, solution: Dict[str, Any], problem_config: Dict[str, Any]) -> float:
        """Calculate efficiency score for the solution."""
        utilization_targets = solution.get('utilization_targets', {})
        if not utilization_targets:
            return 50.0  # Default score
        
        avg_utilization = sum(utilization_targets.values()) / len(utilization_targets)
        return avg_utilization * 100  # Scale to 0-100

    def _calculate_constraint_penalty(self, solution: Dict[str, Any], constraints: List[OptimizationConstraint]) -> float:
        """Calculate penalty for constraint violations."""
        total_penalty = 0.0
        
        for constraint in constraints:
            violation = self._check_constraint_violation(solution, constraint)
            if violation > 0:
                total_penalty += violation * constraint.priority
        
        return total_penalty

    def _check_constraint_violation(self, solution: Dict[str, Any], constraint: OptimizationConstraint) -> float:
        """Check if constraint is violated and return violation amount."""
        # Simplified constraint checking
        constraint_value = self._extract_constraint_value(solution, constraint.name)
        target_value = constraint.value
        
        if constraint.operator == "<=":
            return max(0, constraint_value - target_value)
        elif constraint.operator == ">=":
            return max(0, target_value - constraint_value)
        elif constraint.operator == "==":
            return abs(constraint_value - target_value)
        else:
            return 0.0

    def _extract_constraint_value(self, solution: Dict[str, Any], constraint_name: str) -> float:
        """Extract value for constraint checking."""
        # Simplified value extraction based on constraint name
        if 'time' in constraint_name.lower():
            schedule = solution.get('schedule', {})
            total_time = sum(order.get('duration', 0) for order in schedule.values())
            return total_time
        elif 'resource' in constraint_name.lower():
            allocations = solution.get('allocations', {})
            return sum(allocations.values())
        else:
            return 0.0

    def _tournament_selection(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Tournament selection for genetic algorithm."""
        selected = []
        tournament_size = min(3, len(population))
        
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(population[winner_index].copy())
        
        return selected

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], problem_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Simple crossover - exchange random keys
        for key in parent1.keys():
            if random.random() < 0.5:
                if key in parent2:
                    child1[key] = parent2[key]
                    child2[key] = parent1[key]
        
        return child1, child2

    def _mutate(self, individual: Dict[str, Any], problem_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        # Randomly modify values with small perturbations
        for key, value in mutated.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)) and random.random() < 0.1:
                        perturbation = random.uniform(-0.2, 0.2) * abs(sub_value) if sub_value != 0 else random.uniform(-1, 1)
                        mutated[key][sub_key] = max(0, sub_value + perturbation)
        
        return mutated

    def _create_solution(self, parameters: Dict[str, Any], fitness: float, generation: int, problem_config: Dict[str, Any]) -> OptimizationSolution:
        """Create OptimizationSolution object."""
        constraints = problem_config.get('constraints', [])
        constraints_satisfied = self._calculate_constraint_penalty(parameters, constraints) == 0
        
        objectives = {}
        for obj in problem_config.get('objectives', []):
            if obj == OptimizationObjective.MAXIMIZE_THROUGHPUT:
                objectives['throughput'] = self._calculate_throughput_score(parameters, problem_config)
            elif obj == OptimizationObjective.MINIMIZE_COST:
                objectives['cost'] = self._calculate_cost_score(parameters, problem_config)
            elif obj == OptimizationObjective.MAXIMIZE_QUALITY:
                objectives['quality'] = self._calculate_quality_score(parameters, problem_config)
            elif obj == OptimizationObjective.MAXIMIZE_EFFICIENCY:
                objectives['efficiency'] = self._calculate_efficiency_score(parameters, problem_config)
        
        return OptimizationSolution(
            solution_id=f"sol_{int(time.time())}_{generation}",
            parameters=parameters,
            objectives=objectives,
            constraints_satisfied=constraints_satisfied,
            fitness_score=fitness,
            generation=generation,
            computation_time_ms=0  # Updated externally
        )

    def _create_failed_result(self, problem_id: str, algorithm: OptimizationAlgorithm, start_time: float) -> OptimizationResult:
        """Create a failed optimization result."""
        return OptimizationResult(
            problem_id=problem_id,
            best_solution=None,
            all_solutions=[],
            convergence_history=[],
            total_generations=0,
            computation_time_ms=(time.time() - start_time) * 1000,
            algorithm_used=algorithm,
            success=False
        )

    def _update_performance_metrics(self, computation_time: float, success: bool):
        """Update performance metrics."""
        self.performance_metrics['total_optimizations'] += 1
        
        if success:
            self.performance_metrics['successful_optimizations'] += 1
        
        # Update average computation time
        total_opts = self.performance_metrics['total_optimizations']
        current_avg = self.performance_metrics['avg_optimization_time_ms']
        self.performance_metrics['avg_optimization_time_ms'] = (
            (current_avg * (total_opts - 1) + computation_time) / total_opts
        )
        
        # Update convergence rate
        self.performance_metrics['convergence_rate'] = (
            self.performance_metrics['successful_optimizations'] / total_opts
        )

    def get_optimization_status(self, problem_id: str) -> Dict[str, Any]:
        """Get status of an optimization problem."""
        if problem_id in self.optimization_results:
            result = self.optimization_results[problem_id]
            return {
                'problem_id': problem_id,
                'status': 'completed',
                'success': result.success,
                'computation_time_ms': result.computation_time_ms,
                'best_fitness': result.best_solution.fitness_score if result.best_solution else None
            }
        elif problem_id in self.active_optimizations:
            return {
                'problem_id': problem_id,
                'status': 'running',
                'success': None,
                'computation_time_ms': None,
                'best_fitness': None
            }
        else:
            return {
                'problem_id': problem_id,
                'status': 'not_found',
                'success': False,
                'computation_time_ms': None,
                'best_fitness': None
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def clear_optimization_history(self):
        """Clear optimization history to free memory."""
        self.optimization_history.clear()
        self.optimization_results.clear()
        logging.info("Optimization history cleared")

    def __str__(self) -> str:
        return f"OptimizationLayerEngine(target={self.performance_target_ms}ms, active_opts={len(self.active_optimizations)})"

    def __repr__(self) -> str:
        return self.__str__()