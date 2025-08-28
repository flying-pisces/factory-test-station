"""AI-enabled Manufacturing Plan Optimizer using genetic algorithms and multi-objective optimization."""

import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import json

from manufacturing_plan import (
    ManufacturingPlan, StationConfig, StationType, LineSimulation,
    create_sample_manufacturing_plans
)


@dataclass
class OptimizationObjective:
    """Optimization objective with weight and target."""
    name: str
    weight: float  # 0.0 to 1.0
    target_value: Optional[float] = None  # Target value (None = maximize/minimize)
    maximize: bool = True  # True to maximize, False to minimize


@dataclass
class OptimizationConstraint:
    """Constraint for manufacturing plan optimization."""
    parameter: str  # e.g., 'max_stations', 'min_yield', 'max_cost'
    value: float
    operator: str  # '<=', '>=', '==', '!=', '<', '>'


class ManufacturingPlanChromosome:
    """Genetic algorithm chromosome representing a manufacturing plan."""
    
    def __init__(self, stations: List[StationConfig], plan_id: str = None):
        self.stations = stations
        self.plan_id = plan_id or f"PLAN_{random.randint(1000, 9999)}"
        self.fitness_scores: Dict[str, float] = {}
        self.total_fitness: float = 0.0
        self.simulation_results: Optional[Dict[str, Any]] = None
    
    def to_manufacturing_plan(self, target_volume: int = 10000) -> ManufacturingPlan:
        """Convert chromosome to ManufacturingPlan."""
        return ManufacturingPlan(
            plan_id=self.plan_id,
            plan_name=f"AI Generated Plan {self.plan_id}",
            stations=self.stations.copy(),
            target_volume=target_volume,
            target_yield=0.8,  # Will be calculated
            target_mva=50.0   # Will be calculated
        )
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate chromosome by randomly adjusting station parameters."""
        for station in self.stations:
            if random.random() < mutation_rate:
                # Randomly mutate one parameter
                mutation_type = random.choice([
                    'yield_rate', 'process_time', 'process_cost', 'capacity'
                ])
                
                if mutation_type == 'yield_rate':
                    # Adjust yield rate by ±5%
                    delta = random.uniform(-0.05, 0.05)
                    station.yield_rate = max(0.1, min(0.99, station.yield_rate + delta))
                
                elif mutation_type == 'process_time':
                    # Adjust process time by ±20%
                    delta = random.uniform(-0.2, 0.2)
                    station.process_time = max(5.0, station.process_time * (1 + delta))
                
                elif mutation_type == 'process_cost':
                    # Adjust process cost by ±30%
                    delta = random.uniform(-0.3, 0.3)
                    station.process_cost = max(0.5, station.process_cost * (1 + delta))
                
                elif mutation_type == 'capacity':
                    # Change capacity (1-3)
                    station.capacity = random.randint(1, 3)
    
    def crossover(self, other: 'ManufacturingPlanChromosome') -> Tuple['ManufacturingPlanChromosome', 'ManufacturingPlanChromosome']:
        """Create offspring through crossover."""
        # Ensure same number of stations
        min_stations = min(len(self.stations), len(other.stations))
        crossover_point = random.randint(1, min_stations - 1)
        
        # Create offspring
        offspring1_stations = (
            self.stations[:crossover_point] + 
            other.stations[crossover_point:min_stations]
        )
        offspring2_stations = (
            other.stations[:crossover_point] + 
            self.stations[crossover_point:min_stations]
        )
        
        offspring1 = ManufacturingPlanChromosome(offspring1_stations)
        offspring2 = ManufacturingPlanChromosome(offspring2_stations)
        
        return offspring1, offspring2


class AIManufacturingOptimizer:
    """AI-enabled optimizer for manufacturing plans using genetic algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.population: List[ManufacturingPlanChromosome] = []
        self.objectives: List[OptimizationObjective] = []
        self.constraints: List[OptimizationConstraint] = []
        self.generation = 0
        self.best_solutions: List[ManufacturingPlanChromosome] = []
        
        # GA parameters
        self.population_size = 50
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.elite_size = 5
        self.max_generations = 100
        
        # Multi-threading for simulation
        self.max_workers = 4
    
    def add_objective(self, name: str, weight: float, maximize: bool = True, 
                     target_value: Optional[float] = None):
        """Add optimization objective."""
        objective = OptimizationObjective(
            name=name, weight=weight, maximize=maximize, target_value=target_value
        )
        self.objectives.append(objective)
        self.logger.info(f"Added objective: {name} (weight: {weight})")
    
    def add_constraint(self, parameter: str, value: float, operator: str = '>='):
        """Add optimization constraint."""
        constraint = OptimizationConstraint(parameter, value, operator)
        self.constraints.append(constraint)
        self.logger.info(f"Added constraint: {parameter} {operator} {value}")
    
    def create_initial_population(self, base_plans: List[ManufacturingPlan] = None) -> List[ManufacturingPlanChromosome]:
        """Create initial population for genetic algorithm."""
        population = []
        
        # Use base plans if provided
        if base_plans:
            for plan in base_plans:
                chromosome = ManufacturingPlanChromosome(plan.stations, plan.plan_id)
                population.append(chromosome)
        
        # Generate random variations to fill population
        while len(population) < self.population_size:
            # Create random station configuration
            num_stations = random.randint(3, 8)
            stations = []
            
            station_types = list(StationType)
            
            for i in range(num_stations):
                station = StationConfig(
                    station_id=f"S{i}",
                    station_type=random.choice(station_types),
                    position=i,
                    yield_rate=random.uniform(0.7, 0.98),
                    process_time=random.uniform(10.0, 60.0),
                    process_cost=random.uniform(1.0, 20.0),
                    capacity=random.randint(1, 3)
                )
                stations.append(station)
            
            chromosome = ManufacturingPlanChromosome(stations)
            population.append(chromosome)
        
        return population
    
    def evaluate_fitness(self, chromosomes: List[ManufacturingPlanChromosome]):
        """Evaluate fitness of chromosomes using parallel simulation."""
        
        def simulate_chromosome(chromosome: ManufacturingPlanChromosome) -> ManufacturingPlanChromosome:
            try:
                # Convert to manufacturing plan and simulate
                plan = chromosome.to_manufacturing_plan(target_volume=1000)  # Smaller volume for speed
                simulation = LineSimulation(plan)
                results = simulation.run_full_simulation()
                
                chromosome.simulation_results = results
                
                # Calculate fitness for each objective
                plan_metrics = results['plan_metrics']
                
                for objective in self.objectives:
                    if objective.name == 'yield':
                        value = plan_metrics['actual_yield'] or 0
                    elif objective.name == 'mva':
                        value = plan_metrics['actual_mva'] or 0
                    elif objective.name == 'throughput':
                        value = plan_metrics['throughput'] or 0
                    elif objective.name == 'cost':
                        value = plan_metrics['total_cost'] or float('inf')
                    elif objective.name == 'cycle_time':
                        value = plan_metrics['cycle_time'] or float('inf')
                    else:
                        value = 0
                    
                    # Normalize and apply objective direction
                    if objective.maximize:
                        fitness = value * objective.weight
                    else:
                        fitness = (1.0 / (value + 0.001)) * objective.weight  # Avoid division by zero
                    
                    chromosome.fitness_scores[objective.name] = fitness
                
                # Calculate total fitness
                chromosome.total_fitness = sum(chromosome.fitness_scores.values())
                
                # Apply constraint penalties
                penalty = self._calculate_constraint_penalty(chromosome)
                chromosome.total_fitness -= penalty
                
                return chromosome
                
            except Exception as e:
                self.logger.error(f"Simulation error for {chromosome.plan_id}: {e}")
                chromosome.total_fitness = 0.0
                return chromosome
        
        # Parallel simulation execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chromosomes[:] = list(executor.map(simulate_chromosome, chromosomes))
    
    def _calculate_constraint_penalty(self, chromosome: ManufacturingPlanChromosome) -> float:
        """Calculate penalty for constraint violations."""
        penalty = 0.0
        
        if not chromosome.simulation_results:
            return 1000.0  # High penalty for failed simulation
        
        metrics = chromosome.simulation_results['plan_metrics']
        
        for constraint in self.constraints:
            if constraint.parameter == 'min_yield':
                actual = metrics.get('actual_yield', 0)
                if actual < constraint.value:
                    penalty += (constraint.value - actual) * 100
            
            elif constraint.parameter == 'max_cost':
                actual = metrics.get('total_cost', 0)
                if actual > constraint.value:
                    penalty += (actual - constraint.value) * 0.1
            
            elif constraint.parameter == 'min_throughput':
                actual = metrics.get('throughput', 0)
                if actual < constraint.value:
                    penalty += (constraint.value - actual) * 0.5
        
        return penalty
    
    def selection(self) -> List[ManufacturingPlanChromosome]:
        """Tournament selection for next generation."""
        selected = []
        tournament_size = 3
        
        # Keep elite solutions
        elite = sorted(self.population, key=lambda x: x.total_fitness, reverse=True)[:self.elite_size]
        selected.extend(elite)
        
        # Tournament selection for remaining slots
        while len(selected) < self.population_size:
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.total_fitness)
            selected.append(winner)
        
        return selected
    
    def run_optimization(self, base_plans: List[ManufacturingPlan] = None, 
                        generations: Optional[int] = None) -> List[ManufacturingPlan]:
        """Run genetic algorithm optimization."""
        max_gens = generations or self.max_generations
        
        self.logger.info(f"Starting AI optimization with {len(self.objectives)} objectives")
        
        # Initialize population
        self.population = self.create_initial_population(base_plans)
        
        # Evolution loop
        for gen in range(max_gens):
            self.generation = gen
            
            # Evaluate fitness
            self.evaluate_fitness(self.population)
            
            # Track best solutions
            best_in_generation = max(self.population, key=lambda x: x.total_fitness)
            self.best_solutions.append(best_in_generation)
            
            self.logger.info(
                f"Generation {gen}: Best fitness = {best_in_generation.total_fitness:.3f}"
            )
            
            # Selection
            selected = self.selection()
            
            # Crossover and mutation
            new_population = selected[:self.elite_size]  # Keep elite
            
            while len(new_population) < self.population_size:
                # Crossover
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(selected, 2)
                    offspring1, offspring2 = parent1.crossover(parent2)
                    new_population.extend([offspring1, offspring2])
                else:
                    new_population.append(random.choice(selected))
            
            # Mutation
            for chromosome in new_population[self.elite_size:]:
                chromosome.mutate(self.mutation_rate)
            
            # Trim to population size
            self.population = new_population[:self.population_size]
        
        # Return best solutions as manufacturing plans
        best_chromosomes = sorted(self.population, key=lambda x: x.total_fitness, reverse=True)[:5]
        
        optimized_plans = []
        for chromosome in best_chromosomes:
            plan = chromosome.to_manufacturing_plan()
            if chromosome.simulation_results:
                metrics = chromosome.simulation_results['plan_metrics']
                plan.actual_yield = metrics.get('actual_yield')
                plan.actual_mva = metrics.get('actual_mva')
                plan.total_cost = metrics.get('total_cost')
                plan.throughput = metrics.get('throughput')
            
            optimized_plans.append(plan)
        
        self.logger.info(f"Optimization complete. Generated {len(optimized_plans)} optimized plans.")
        
        return optimized_plans


class ManufacturingPlanComparator:
    """Compare and analyze different manufacturing plans."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compare_plans(self, plans: List[ManufacturingPlan], target_volume: int = 10000) -> Dict[str, Any]:
        """Compare multiple manufacturing plans with detailed analysis."""
        
        comparison_results = {
            'plans': [],
            'summary': {},
            'recommendations': [],
            'pareto_optimal': []
        }
        
        # Simulate all plans
        for plan in plans:
            self.logger.info(f"Simulating plan {plan.plan_id}")
            
            simulation = LineSimulation(plan)
            results = simulation.run_full_simulation()
            
            plan_analysis = {
                'plan_id': plan.plan_id,
                'plan_name': plan.plan_name,
                'metrics': results['plan_metrics'],
                'station_performance': results['station_metrics'],
                'bottlenecks': results['line_metrics']['bottleneck_station'],
                'cost_breakdown': {
                    'material_cost_per_unit': 10.0,  # Simplified
                    'process_cost_per_unit': results['plan_metrics']['total_cost'] / target_volume,
                    'total_cost_per_unit': (results['plan_metrics']['total_cost'] + 10.0 * target_volume) / target_volume
                }
            }
            
            comparison_results['plans'].append(plan_analysis)
        
        # Identify Pareto optimal solutions (yield vs MVA trade-off)
        pareto_plans = self._find_pareto_optimal(plans)
        comparison_results['pareto_optimal'] = [p.plan_id for p in pareto_plans]
        
        # Generate summary and recommendations
        comparison_results['summary'] = self._generate_summary(comparison_results['plans'])
        comparison_results['recommendations'] = self._generate_recommendations(comparison_results['plans'])
        
        return comparison_results
    
    def _find_pareto_optimal(self, plans: List[ManufacturingPlan]) -> List[ManufacturingPlan]:
        """Find Pareto optimal solutions for yield vs MVA trade-off."""
        pareto_optimal = []
        
        for i, plan1 in enumerate(plans):
            is_dominated = False
            
            for j, plan2 in enumerate(plans):
                if i != j:
                    # Check if plan2 dominates plan1 (better in both yield and MVA)
                    if (plan2.actual_yield >= plan1.actual_yield and 
                        plan2.actual_mva >= plan1.actual_mva and
                        (plan2.actual_yield > plan1.actual_yield or plan2.actual_mva > plan1.actual_mva)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(plan1)
        
        return pareto_optimal
    
    def _generate_summary(self, plan_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison summary."""
        if not plan_analyses:
            return {}
        
        yields = [p['metrics']['actual_yield'] for p in plan_analyses if p['metrics']['actual_yield']]
        mvas = [p['metrics']['actual_mva'] for p in plan_analyses if p['metrics']['actual_mva']]
        throughputs = [p['metrics']['throughput'] for p in plan_analyses if p['metrics']['throughput']]
        
        return {
            'plan_count': len(plan_analyses),
            'yield_range': {'min': min(yields), 'max': max(yields), 'avg': np.mean(yields)} if yields else {},
            'mva_range': {'min': min(mvas), 'max': max(mvas), 'avg': np.mean(mvas)} if mvas else {},
            'throughput_range': {'min': min(throughputs), 'max': max(throughputs), 'avg': np.mean(throughputs)} if throughputs else {},
            'best_yield_plan': max(plan_analyses, key=lambda x: x['metrics']['actual_yield'] or 0)['plan_id'],
            'best_mva_plan': max(plan_analyses, key=lambda x: x['metrics']['actual_mva'] or 0)['plan_id'],
            'best_throughput_plan': max(plan_analyses, key=lambda x: x['metrics']['throughput'] or 0)['plan_id']
        }
    
    def _generate_recommendations(self, plan_analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Identify common bottlenecks
        bottlenecks = [p['bottlenecks'] for p in plan_analyses if p['bottlenecks']]
        if bottlenecks:
            common_bottleneck = max(set(bottlenecks), key=bottlenecks.count)
            recommendations.append(f"Consider optimizing station {common_bottleneck} - identified as bottleneck in multiple plans")
        
        # Yield vs MVA trade-off analysis
        high_yield_plans = [p for p in plan_analyses if p['metrics']['actual_yield'] and p['metrics']['actual_yield'] > 0.8]
        high_mva_plans = [p for p in plan_analyses if p['metrics']['actual_mva'] and p['metrics']['actual_mva'] > 50]
        
        if high_yield_plans and not high_mva_plans:
            recommendations.append("Consider reducing process complexity to improve MVA while maintaining acceptable yield")
        elif high_mva_plans and not high_yield_plans:
            recommendations.append("Consider adding quality control steps to improve yield")
        
        # Cost optimization
        avg_cost = np.mean([p['cost_breakdown']['total_cost_per_unit'] for p in plan_analyses])
        low_cost_plans = [p for p in plan_analyses if p['cost_breakdown']['total_cost_per_unit'] < avg_cost * 0.9]
        
        if low_cost_plans:
            recommendations.append(f"Plans {[p['plan_id'] for p in low_cost_plans]} show superior cost performance")
        
        return recommendations


def run_ai_optimization_example():
    """Example of running AI optimization for manufacturing plans."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = AIManufacturingOptimizer()
    
    # Define objectives
    optimizer.add_objective('yield', weight=0.4, maximize=True)
    optimizer.add_objective('mva', weight=0.3, maximize=True)
    optimizer.add_objective('throughput', weight=0.2, maximize=True)
    optimizer.add_objective('cost', weight=0.1, maximize=False)
    
    # Add constraints
    optimizer.add_constraint('min_yield', 0.6, '>=')
    optimizer.add_constraint('min_throughput', 50, '>=')
    
    # Get base plans
    base_plans = create_sample_manufacturing_plans()
    
    # Run optimization
    optimized_plans = optimizer.run_optimization(base_plans, generations=20)
    
    # Compare plans
    comparator = ManufacturingPlanComparator()
    all_plans = base_plans + optimized_plans
    comparison = comparator.compare_plans(all_plans)
    
    # Print results
    print("\n=== Manufacturing Plan Optimization Results ===")
    for plan_data in comparison['plans']:
        metrics = plan_data['metrics']
        print(f"\nPlan {plan_data['plan_id']}:")
        print(f"  Yield: {metrics['actual_yield']:.1%}")
        print(f"  MVA: {metrics['actual_mva']:.1f}¥")
        print(f"  Throughput: {metrics['throughput']:.0f} UPH")
    
    print(f"\nPareto Optimal Plans: {comparison['pareto_optimal']}")
    print("\nRecommendations:")
    for rec in comparison['recommendations']:
        print(f"  • {rec}")
    
    return optimized_plans, comparison


if __name__ == "__main__":
    run_ai_optimization_example()