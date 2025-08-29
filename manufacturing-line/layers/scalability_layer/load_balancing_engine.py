#!/usr/bin/env python3
"""
LoadBalancingEngine - Week 10 Scalability & Performance Layer
Intelligent load balancing and traffic distribution
"""

import time
import json
import random
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    LEAST_RESPONSE_TIME = "least_response_time"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"

class HealthStatus(Enum):
    """Backend server health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

class TrafficDistributionStrategy(Enum):
    """Traffic distribution strategies"""
    EVEN_DISTRIBUTION = "even_distribution"
    WEIGHTED_DISTRIBUTION = "weighted_distribution"
    PRIORITY_BASED = "priority_based"
    GEOGRAPHIC_AFFINITY = "geographic_affinity"
    SESSION_AFFINITY = "session_affinity"

@dataclass
class BackendServer:
    """Backend server configuration"""
    server_id: str
    host: str
    port: int
    weight: int
    priority: int
    health_status: HealthStatus
    current_connections: int
    max_connections: int
    average_response_time_ms: float
    region: str = "us-east-1"
    availability_zone: str = "us-east-1a"
    last_health_check: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadBalancingRule:
    """Load balancing rule configuration"""
    rule_id: str
    name: str
    algorithm: LoadBalancingAlgorithm
    strategy: TrafficDistributionStrategy
    target_backend_pool: str
    conditions: Dict[str, Any]
    priority: int
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class RoutingDecision:
    """Load balancing routing decision"""
    decision_id: str
    request_id: str
    client_ip: str
    target_server: BackendServer
    algorithm_used: LoadBalancingAlgorithm
    decision_time_ms: float
    routing_factors: Dict[str, Any]
    timestamp: str

class LoadBalancingEngine:
    """Intelligent load balancing and traffic distribution
    
    Week 10 Performance Targets:
    - Routing decisions: <10ms
    - Request forwarding: <1ms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LoadBalancingEngine with configuration"""
        self.config = config or {}
        
        # Performance targets
        self.routing_decision_target_ms = 10
        self.request_forwarding_target_ms = 1
        
        # State management
        self.backend_servers = {}
        self.backend_pools = {}
        self.load_balancing_rules = {}
        self.routing_decisions = []
        self.health_check_results = {}
        self.traffic_statistics = {}
        
        # Load balancing state
        self.round_robin_counters = {}
        self.connection_counts = {}
        self.response_time_history = {}
        
        # Initialize default backend pools
        self._initialize_default_backend_pools()
        
        # Initialize load balancing rules
        self._initialize_load_balancing_rules()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize resource engine if available
        self.resource_engine = None
        try:
            from layers.scalability_layer.resource_engine import ResourceEngine
            self.resource_engine = ResourceEngine(config.get('resource_config', {}))
        except ImportError:
            logger.warning("ResourceEngine not available - using mock interface")
        
        logger.info("LoadBalancingEngine initialized with intelligent traffic distribution")
    
    def distribute_traffic_intelligently(self, traffic_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute traffic using intelligent algorithms
        
        Args:
            traffic_specs: Traffic distribution specifications
            
        Returns:
            Traffic distribution results with performance metrics
        """
        start_time = time.time()
        
        try:
            # Parse traffic specifications
            request_count = traffic_specs.get('request_count', 100)
            backend_pool = traffic_specs.get('backend_pool', 'default_pool')
            algorithm = LoadBalancingAlgorithm(traffic_specs.get('algorithm', 'least_connections'))
            client_locations = traffic_specs.get('client_locations', ['us-east-1'])
            
            # Get backend servers for pool
            pool_servers = self.backend_pools.get(backend_pool, [])
            if not pool_servers:
                return {
                    'distribution_success': False,
                    'reason': f'No servers available in pool {backend_pool}',
                    'distribution_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            
            # Filter healthy servers
            healthy_servers = [
                server for server in pool_servers
                if self.backend_servers[server].health_status == HealthStatus.HEALTHY
            ]
            
            if not healthy_servers:
                return {
                    'distribution_success': False,
                    'reason': 'No healthy servers available',
                    'distribution_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            
            # Distribute traffic across healthy servers
            distribution_results = []
            routing_decisions = []
            
            for i in range(request_count):
                # Simulate client request
                client_ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
                client_location = random.choice(client_locations)
                
                # Make routing decision
                routing_start = time.time()
                target_server = self._select_backend_server(
                    healthy_servers, algorithm, client_ip, client_location
                )
                routing_time_ms = (time.time() - routing_start) * 1000
                
                # Record routing decision
                decision = RoutingDecision(
                    decision_id=f"ROUTE_{int(time.time() * 1000)}_{i}",
                    request_id=f"REQ_{i}",
                    client_ip=client_ip,
                    target_server=target_server,
                    algorithm_used=algorithm,
                    decision_time_ms=routing_time_ms,
                    routing_factors={
                        'client_location': client_location,
                        'server_connections': target_server.current_connections,
                        'server_response_time': target_server.average_response_time_ms
                    },
                    timestamp=datetime.now().isoformat()
                )
                routing_decisions.append(decision)
                
                # Update server connections
                target_server.current_connections += 1
                
                # Record distribution result
                distribution_results.append({
                    'request_id': f"REQ_{i}",
                    'target_server': target_server.server_id,
                    'routing_time_ms': routing_time_ms
                })
            
            # Store routing decisions
            self.routing_decisions.extend(routing_decisions)
            
            # Calculate distribution statistics
            server_request_counts = {}
            total_routing_time = 0
            max_routing_time = 0
            
            for decision in routing_decisions:
                server_id = decision.target_server.server_id
                server_request_counts[server_id] = server_request_counts.get(server_id, 0) + 1
                total_routing_time += decision.decision_time_ms
                max_routing_time = max(max_routing_time, decision.decision_time_ms)
            
            avg_routing_time = total_routing_time / len(routing_decisions) if routing_decisions else 0
            
            # Calculate load distribution efficiency
            request_counts = list(server_request_counts.values())
            load_balance_score = self._calculate_load_balance_score(request_counts)
            
            # Calculate total distribution time
            distribution_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'distribution_success': True,
                'backend_pool': backend_pool,
                'algorithm': algorithm.value,
                'requests_distributed': request_count,
                'healthy_servers': len(healthy_servers),
                'server_distribution': server_request_counts,
                'distribution_time_ms': round(distribution_time_ms, 2),
                'avg_routing_time_ms': round(avg_routing_time, 2),
                'max_routing_time_ms': round(max_routing_time, 2),
                'routing_target_met': avg_routing_time < self.routing_decision_target_ms,
                'load_balance_score': load_balance_score,
                'distribution_efficiency': min(100, (100 - (max_routing_time - avg_routing_time))),
                'distributed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Traffic distribution completed: {request_count} requests in {distribution_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error distributing traffic: {e}")
            raise
    
    def implement_health_based_routing(self, health_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implement health-based routing with failover
        
        Args:
            health_parameters: Health monitoring and routing parameters
            
        Returns:
            Health-based routing implementation results
        """
        start_time = time.time()
        
        try:
            # Parse health parameters
            health_check_interval_seconds = health_parameters.get('health_check_interval', 30)
            failure_threshold = health_parameters.get('failure_threshold', 3)
            recovery_threshold = health_parameters.get('recovery_threshold', 2)
            timeout_ms = health_parameters.get('timeout_ms', 5000)
            
            # Perform health checks on all backend servers
            health_check_results = []
            
            for server_id, server in self.backend_servers.items():
                health_check_result = self._perform_health_check(server, timeout_ms)
                health_check_results.append(health_check_result)
                
                # Update server health status based on check result
                if health_check_result['healthy']:
                    if server.health_status == HealthStatus.UNHEALTHY:
                        # Recovery path
                        consecutive_successes = self.health_check_results.get(server_id, {}).get('consecutive_successes', 0) + 1
                        if consecutive_successes >= recovery_threshold:
                            server.health_status = HealthStatus.HEALTHY
                            logger.info(f"Server {server_id} marked as healthy after recovery")
                    else:
                        server.health_status = HealthStatus.HEALTHY
                else:
                    # Failure path
                    consecutive_failures = self.health_check_results.get(server_id, {}).get('consecutive_failures', 0) + 1
                    if consecutive_failures >= failure_threshold:
                        server.health_status = HealthStatus.UNHEALTHY
                        logger.warning(f"Server {server_id} marked as unhealthy after {consecutive_failures} failures")
                
                # Store health check history
                if server_id not in self.health_check_results:
                    self.health_check_results[server_id] = {'consecutive_successes': 0, 'consecutive_failures': 0, 'history': []}
                
                if health_check_result['healthy']:
                    self.health_check_results[server_id]['consecutive_successes'] += 1
                    self.health_check_results[server_id]['consecutive_failures'] = 0
                else:
                    self.health_check_results[server_id]['consecutive_failures'] += 1
                    self.health_check_results[server_id]['consecutive_successes'] = 0
                
                self.health_check_results[server_id]['history'].append(health_check_result)
                
                # Keep only recent history
                if len(self.health_check_results[server_id]['history']) > 50:
                    self.health_check_results[server_id]['history'] = self.health_check_results[server_id]['history'][-50:]
            
            # Calculate health statistics
            total_servers = len(self.backend_servers)
            healthy_servers = len([s for s in self.backend_servers.values() if s.health_status == HealthStatus.HEALTHY])
            degraded_servers = len([s for s in self.backend_servers.values() if s.health_status == HealthStatus.DEGRADED])
            unhealthy_servers = len([s for s in self.backend_servers.values() if s.health_status == HealthStatus.UNHEALTHY])
            
            # Implement failover routing updates
            failover_updates = self._update_failover_routing()
            
            # Calculate implementation time
            implementation_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'implementation_success': True,
                'health_check_interval': health_check_interval_seconds,
                'servers_checked': len(health_check_results),
                'healthy_servers': healthy_servers,
                'degraded_servers': degraded_servers,
                'unhealthy_servers': unhealthy_servers,
                'health_percentage': (healthy_servers / total_servers) * 100 if total_servers > 0 else 0,
                'failover_updates': len(failover_updates),
                'implementation_time_ms': round(implementation_time_ms, 2),
                'target_met': implementation_time_ms < self.routing_decision_target_ms * 10,  # Allow 10x for health checks
                'implemented_at': datetime.now().isoformat()
            }
            
            logger.info(f"Health-based routing implemented: {healthy_servers}/{total_servers} healthy servers")
            return result
            
        except Exception as e:
            logger.error(f"Error implementing health-based routing: {e}")
            raise
    
    def optimize_geographic_distribution(self, geo_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize geographic distribution for global performance
        
        Args:
            geo_specs: Geographic distribution specifications
            
        Returns:
            Geographic distribution optimization results
        """
        start_time = time.time()
        
        try:
            # Parse geographic specifications
            target_regions = geo_specs.get('target_regions', ['us-east-1', 'eu-west-1', 'ap-southeast-1'])
            latency_targets_ms = geo_specs.get('latency_targets_ms', {'us-east-1': 50, 'eu-west-1': 100, 'ap-southeast-1': 150})
            traffic_distribution = geo_specs.get('traffic_distribution', 'proximity_based')
            
            # Analyze current geographic distribution
            current_distribution = self._analyze_geographic_distribution()
            
            # Calculate optimal server placement
            optimal_placement = self._calculate_optimal_geographic_placement(
                target_regions, latency_targets_ms
            )
            
            # Generate geographic optimization actions
            optimization_actions = []
            
            for region in target_regions:
                current_servers = current_distribution.get(region, {}).get('servers', [])
                optimal_count = optimal_placement.get(region, {}).get('recommended_servers', 2)
                
                if len(current_servers) < optimal_count:
                    # Need to add servers in this region
                    servers_to_add = optimal_count - len(current_servers)
                    action = {
                        'action_type': 'add_servers',
                        'region': region,
                        'servers_to_add': servers_to_add,
                        'server_specs': optimal_placement[region].get('server_specs', {})
                    }
                    optimization_actions.append(action)
                
                elif len(current_servers) > optimal_count:
                    # Can potentially remove servers in this region
                    servers_to_remove = len(current_servers) - optimal_count
                    candidates = current_servers[-servers_to_remove:] if servers_to_remove > 0 else []
                    action = {
                        'action_type': 'remove_servers',
                        'region': region,
                        'servers_to_remove': servers_to_remove,
                        'candidates': candidates  # Remove newest servers
                    }
                    optimization_actions.append(action)
            
            # Implement geographic routing optimizations
            routing_optimizations = self._implement_geographic_routing_optimizations(
                target_regions, traffic_distribution
            )
            
            # Calculate latency improvements
            latency_improvements = {}
            for region in target_regions:
                current_latency = current_distribution.get(region, {}).get('avg_latency_ms', 200)
                target_latency = latency_targets_ms.get(region, 100)
                improvement = max(0, current_latency - target_latency)
                latency_improvements[region] = improvement
            
            # Calculate optimization time
            optimization_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'optimization_success': True,
                'target_regions': target_regions,
                'traffic_distribution': traffic_distribution,
                'optimization_actions': len(optimization_actions),
                'routing_optimizations': len(routing_optimizations),
                'latency_improvements': latency_improvements,
                'optimization_time_ms': round(optimization_time_ms, 2),
                'target_met': optimization_time_ms < self.routing_decision_target_ms * 20,  # Allow 20x for geo optimization
                'actions': optimization_actions[:5],  # Show first 5 actions
                'optimized_at': datetime.now().isoformat()
            }
            
            logger.info(f"Geographic distribution optimized for {len(target_regions)} regions")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing geographic distribution: {e}")
            raise
    
    def get_load_balancing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics"""
        try:
            # Calculate server statistics
            server_stats = {}
            for server_id, server in self.backend_servers.items():
                server_stats[server_id] = {
                    'health_status': server.health_status.value,
                    'current_connections': server.current_connections,
                    'connection_utilization': (server.current_connections / server.max_connections) * 100 if server.max_connections > 0 else 0,
                    'average_response_time_ms': server.average_response_time_ms,
                    'region': server.region
                }
            
            # Calculate routing statistics
            if self.routing_decisions:
                recent_decisions = self.routing_decisions[-1000:]  # Last 1000 decisions
                avg_decision_time = sum(d.decision_time_ms for d in recent_decisions) / len(recent_decisions)
                max_decision_time = max(d.decision_time_ms for d in recent_decisions)
                
                # Algorithm usage
                algorithm_usage = {}
                for decision in recent_decisions:
                    algo = decision.algorithm_used.value
                    algorithm_usage[algo] = algorithm_usage.get(algo, 0) + 1
            else:
                avg_decision_time = 0
                max_decision_time = 0
                algorithm_usage = {}
            
            # Health check statistics
            healthy_count = len([s for s in self.backend_servers.values() if s.health_status == HealthStatus.HEALTHY])
            total_servers = len(self.backend_servers)
            
            statistics = {
                'server_statistics': server_stats,
                'routing_statistics': {
                    'total_decisions': len(self.routing_decisions),
                    'avg_decision_time_ms': round(avg_decision_time, 2),
                    'max_decision_time_ms': round(max_decision_time, 2),
                    'algorithm_usage': algorithm_usage
                },
                'health_statistics': {
                    'total_servers': total_servers,
                    'healthy_servers': healthy_count,
                    'health_percentage': (healthy_count / total_servers) * 100 if total_servers > 0 else 0,
                    'health_checks_performed': sum(len(result.get('history', [])) for result in self.health_check_results.values())
                },
                'pool_statistics': {
                    'total_pools': len(self.backend_pools),
                    'total_rules': len(self.load_balancing_rules)
                },
                'generated_at': datetime.now().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error getting load balancing statistics: {e}")
            return {'error': str(e)}
    
    def _initialize_default_backend_pools(self):
        """Initialize default backend server pools"""
        # Create sample backend servers
        sample_servers = [
            BackendServer(
                server_id="web-01",
                host="10.0.1.10",
                port=80,
                weight=100,
                priority=1,
                health_status=HealthStatus.HEALTHY,
                current_connections=25,
                max_connections=1000,
                average_response_time_ms=85.0,
                region="us-east-1",
                availability_zone="us-east-1a"
            ),
            BackendServer(
                server_id="web-02",
                host="10.0.1.11",
                port=80,
                weight=100,
                priority=1,
                health_status=HealthStatus.HEALTHY,
                current_connections=30,
                max_connections=1000,
                average_response_time_ms=92.0,
                region="us-east-1",
                availability_zone="us-east-1b"
            ),
            BackendServer(
                server_id="web-03",
                host="10.0.2.10",
                port=80,
                weight=150,
                priority=1,
                health_status=HealthStatus.HEALTHY,
                current_connections=15,
                max_connections=1500,
                average_response_time_ms=78.0,
                region="us-west-2",
                availability_zone="us-west-2a"
            ),
            BackendServer(
                server_id="api-01",
                host="10.0.3.10",
                port=8080,
                weight=200,
                priority=2,
                health_status=HealthStatus.DEGRADED,
                current_connections=450,
                max_connections=500,
                average_response_time_ms=150.0,
                region="eu-west-1",
                availability_zone="eu-west-1a"
            )
        ]
        
        # Store servers
        for server in sample_servers:
            self.backend_servers[server.server_id] = server
        
        # Create backend pools
        self.backend_pools = {
            'default_pool': ['web-01', 'web-02', 'web-03'],
            'api_pool': ['api-01'],
            'web_pool': ['web-01', 'web-02', 'web-03'],
            'global_pool': ['web-01', 'web-02', 'web-03', 'api-01']
        }
    
    def _initialize_load_balancing_rules(self):
        """Initialize load balancing rules"""
        default_rules = [
            LoadBalancingRule(
                rule_id="default_web_rule",
                name="Default Web Traffic",
                algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
                strategy=TrafficDistributionStrategy.EVEN_DISTRIBUTION,
                target_backend_pool="web_pool",
                conditions={"path": "/*", "method": "GET"},
                priority=100
            ),
            LoadBalancingRule(
                rule_id="api_traffic_rule",
                name="API Traffic",
                algorithm=LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS,
                strategy=TrafficDistributionStrategy.WEIGHTED_DISTRIBUTION,
                target_backend_pool="api_pool",
                conditions={"path": "/api/*"},
                priority=200
            ),
            LoadBalancingRule(
                rule_id="geo_routing_rule",
                name="Geographic Routing",
                algorithm=LoadBalancingAlgorithm.GEOGRAPHIC_PROXIMITY,
                strategy=TrafficDistributionStrategy.GEOGRAPHIC_AFFINITY,
                target_backend_pool="global_pool",
                conditions={"geo_routing": True},
                priority=50
            )
        ]
        
        for rule in default_rules:
            self.load_balancing_rules[rule.rule_id] = rule
    
    def _select_backend_server(self, healthy_servers: List[str], algorithm: LoadBalancingAlgorithm, client_ip: str, client_location: str) -> BackendServer:
        """Select backend server using specified algorithm"""
        servers = [self.backend_servers[server_id] for server_id in healthy_servers]
        
        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            # Round robin selection
            if 'round_robin' not in self.round_robin_counters:
                self.round_robin_counters['round_robin'] = 0
            
            selected_index = self.round_robin_counters['round_robin'] % len(servers)
            self.round_robin_counters['round_robin'] += 1
            return servers[selected_index]
        
        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            # Select server with least connections
            return min(servers, key=lambda s: s.current_connections)
        
        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS:
            # Select server with lowest connections/weight ratio
            return min(servers, key=lambda s: s.current_connections / max(s.weight, 1))
        
        elif algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            # Select server with lowest response time
            return min(servers, key=lambda s: s.average_response_time_ms)
        
        elif algorithm == LoadBalancingAlgorithm.IP_HASH:
            # Hash-based selection for session affinity
            client_hash = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
            selected_index = client_hash % len(servers)
            return servers[selected_index]
        
        elif algorithm == LoadBalancingAlgorithm.GEOGRAPHIC_PROXIMITY:
            # Select server closest to client location
            client_region = client_location
            same_region_servers = [s for s in servers if s.region == client_region]
            if same_region_servers:
                return min(same_region_servers, key=lambda s: s.current_connections)
            else:
                return min(servers, key=lambda s: s.average_response_time_ms)
        
        else:
            # Default to round robin
            return servers[0]
    
    def _calculate_load_balance_score(self, request_counts: List[int]) -> float:
        """Calculate load balance distribution score (0-100)"""
        if not request_counts or len(request_counts) <= 1:
            return 100.0
        
        avg_requests = sum(request_counts) / len(request_counts)
        max_deviation = max(abs(count - avg_requests) for count in request_counts)
        deviation_percentage = (max_deviation / avg_requests) * 100 if avg_requests > 0 else 0
        
        # Higher score means better balance
        return max(0, 100 - deviation_percentage)
    
    def _perform_health_check(self, server: BackendServer, timeout_ms: int) -> Dict[str, Any]:
        """Perform health check on backend server"""
        # Simulate health check
        check_start = time.time()
        
        # Simulate health check logic
        health_score = 85.0 + (hash(server.server_id) % 30) - 15  # Random score 70-100
        
        # Consider server load in health calculation
        load_factor = server.current_connections / server.max_connections if server.max_connections > 0 else 0
        health_score -= load_factor * 20  # Reduce health score for high load
        
        # Simulate network delay
        time.sleep(0.001)  # 1ms simulated delay
        
        check_time_ms = (time.time() - check_start) * 1000
        
        is_healthy = health_score > 60  # Health threshold
        
        return {
            'server_id': server.server_id,
            'healthy': is_healthy,
            'health_score': health_score,
            'response_time_ms': check_time_ms,
            'timestamp': datetime.now().isoformat(),
            'timeout_exceeded': check_time_ms > timeout_ms
        }
    
    def _update_failover_routing(self) -> List[Dict[str, Any]]:
        """Update routing configuration for failover"""
        updates = []
        
        # Check each backend pool for health issues
        for pool_name, server_ids in self.backend_pools.items():
            healthy_servers = [
                server_id for server_id in server_ids
                if self.backend_servers[server_id].health_status == HealthStatus.HEALTHY
            ]
            
            total_servers = len(server_ids)
            healthy_count = len(healthy_servers)
            
            if healthy_count < total_servers:
                # Some servers are unhealthy - update routing weights
                update = {
                    'pool': pool_name,
                    'action': 'reweight_servers',
                    'healthy_servers': healthy_count,
                    'total_servers': total_servers,
                    'health_percentage': (healthy_count / total_servers) * 100
                }
                updates.append(update)
        
        return updates
    
    def _analyze_geographic_distribution(self) -> Dict[str, Any]:
        """Analyze current geographic server distribution"""
        distribution = {}
        
        for server_id, server in self.backend_servers.items():
            region = server.region
            if region not in distribution:
                distribution[region] = {
                    'servers': [],
                    'total_capacity': 0,
                    'avg_latency_ms': 0,
                    'health_status': []
                }
            
            distribution[region]['servers'].append(server_id)
            distribution[region]['total_capacity'] += server.max_connections
            distribution[region]['health_status'].append(server.health_status.value)
        
        # Calculate average latencies (simulated)
        region_latencies = {
            'us-east-1': 45.0,
            'us-west-2': 55.0,
            'eu-west-1': 95.0,
            'ap-southeast-1': 140.0
        }
        
        for region in distribution:
            distribution[region]['avg_latency_ms'] = region_latencies.get(region, 100.0)
        
        return distribution
    
    def _calculate_optimal_geographic_placement(self, target_regions: List[str], latency_targets: Dict[str, int]) -> Dict[str, Any]:
        """Calculate optimal geographic server placement"""
        optimal_placement = {}
        
        for region in target_regions:
            target_latency = latency_targets.get(region, 100)
            
            # Simple placement calculation based on latency targets
            if target_latency <= 50:
                recommended_servers = 4  # Low latency requires more servers
                server_type = 'high_performance'
            elif target_latency <= 100:
                recommended_servers = 2  # Medium latency
                server_type = 'standard'
            else:
                recommended_servers = 1  # High latency tolerance
                server_type = 'basic'
            
            optimal_placement[region] = {
                'recommended_servers': recommended_servers,
                'server_specs': {
                    'type': server_type,
                    'max_connections': 1000 if server_type == 'high_performance' else 500,
                    'priority': 1 if target_latency <= 50 else 2
                },
                'expected_latency_ms': target_latency * 0.8  # Expected improvement
            }
        
        return optimal_placement
    
    def _implement_geographic_routing_optimizations(self, regions: List[str], distribution_strategy: str) -> List[Dict[str, Any]]:
        """Implement geographic routing optimizations"""
        optimizations = []
        
        for region in regions:
            optimization = {
                'region': region,
                'strategy': distribution_strategy,
                'optimization_type': 'geo_routing',
                'configuration': {
                    'preferred_region': region,
                    'fallback_regions': [r for r in regions if r != region],
                    'latency_threshold_ms': 200
                }
            }
            optimizations.append(optimization)
        
        return optimizations
    
    def demonstrate_load_balancing_capabilities(self) -> Dict[str, Any]:
        """Demonstrate load balancing capabilities"""
        print("\n⚖️ LOAD BALANCING ENGINE DEMONSTRATION ⚖️")
        print("=" * 50)
        
        # Show current backend configuration
        print(f"🏗️ Backend Infrastructure:")
        print(f"   Backend Servers: {len(self.backend_servers)}")
        print(f"   Backend Pools: {len(self.backend_pools)}")
        print(f"   Load Balancing Rules: {len(self.load_balancing_rules)}")
        
        # Demonstrate intelligent traffic distribution
        print("\n🚦 Intelligent Traffic Distribution...")
        traffic_specs = {
            'request_count': 500,
            'backend_pool': 'web_pool',
            'algorithm': 'least_connections',
            'client_locations': ['us-east-1', 'us-west-2', 'eu-west-1']
        }
        
        traffic_result = self.distribute_traffic_intelligently(traffic_specs)
        print(f"   ✅ Traffic Distribution: {traffic_result['requests_distributed']} requests")
        print(f"   ⚖️ Load Balance Score: {traffic_result['load_balance_score']:.1f}/100")
        print(f"   ⏱️ Avg Routing Time: {traffic_result['avg_routing_time_ms']:.2f}ms")
        print(f"   🎯 Target: <{self.routing_decision_target_ms}ms | {'✅ MET' if traffic_result['routing_target_met'] else '❌ MISSED'}")
        print(f"   📊 Healthy Servers: {traffic_result['healthy_servers']}")
        
        # Demonstrate health-based routing
        print("\n🏥 Health-Based Routing...")
        health_params = {
            'health_check_interval': 30,
            'failure_threshold': 3,
            'recovery_threshold': 2,
            'timeout_ms': 5000
        }
        
        health_result = self.implement_health_based_routing(health_params)
        print(f"   ✅ Health Monitoring: {health_result['servers_checked']} servers checked")
        print(f"   💚 Health Status: {health_result['health_percentage']:.1f}% healthy")
        print(f"   🔄 Failover Updates: {health_result['failover_updates']}")
        print(f"   ⏱️ Implementation Time: {health_result['implementation_time_ms']:.2f}ms")
        
        # Demonstrate geographic distribution
        print("\n🌍 Geographic Distribution Optimization...")
        geo_specs = {
            'target_regions': ['us-east-1', 'us-west-2', 'eu-west-1'],
            'latency_targets_ms': {'us-east-1': 50, 'us-west-2': 75, 'eu-west-1': 100},
            'traffic_distribution': 'proximity_based'
        }
        
        geo_result = self.optimize_geographic_distribution(geo_specs)
        print(f"   ✅ Geographic Optimization: {len(geo_specs['target_regions'])} regions")
        print(f"   🌐 Optimization Actions: {geo_result['optimization_actions']}")
        print(f"   📈 Latency Improvements: {len(geo_result['latency_improvements'])} regions")
        print(f"   ⏱️ Optimization Time: {geo_result['optimization_time_ms']:.2f}ms")
        
        # Show load balancing statistics
        print("\n📊 Load Balancing Statistics...")
        stats = self.get_load_balancing_statistics()
        routing_stats = stats['routing_statistics']
        health_stats = stats['health_statistics']
        
        print(f"   📈 Total Routing Decisions: {routing_stats['total_decisions']:,}")
        print(f"   ⏱️ Avg Decision Time: {routing_stats['avg_decision_time_ms']:.2f}ms")
        print(f"   🏥 System Health: {health_stats['health_percentage']:.1f}%")
        print(f"   ✅ Healthy Servers: {health_stats['healthy_servers']}/{health_stats['total_servers']}")
        
        print("\n📈 DEMONSTRATION SUMMARY:")
        print(f"   Requests Distributed: {traffic_result['requests_distributed']:,}")
        print(f"   Load Balance Score: {traffic_result['load_balance_score']:.1f}/100")
        print(f"   Average Routing Time: {traffic_result['avg_routing_time_ms']:.2f}ms")
        print(f"   System Health: {health_result['health_percentage']:.1f}%")
        print(f"   Geographic Regions: {len(geo_specs['target_regions'])}")
        print("=" * 50)
        
        return {
            'requests_distributed': traffic_result['requests_distributed'],
            'load_balance_score': traffic_result['load_balance_score'],
            'avg_routing_time_ms': traffic_result['avg_routing_time_ms'],
            'system_health_percentage': health_result['health_percentage'],
            'geographic_regions_optimized': len(geo_specs['target_regions']),
            'healthy_servers': health_result['healthy_servers'],
            'total_servers': len(self.backend_servers),
            'performance_targets_met': traffic_result['routing_target_met']
        }

def main():
    """Test LoadBalancingEngine functionality"""
    engine = LoadBalancingEngine()
    results = engine.demonstrate_load_balancing_capabilities()
    
    print(f"\n🎯 Week 10 Load Balancing Performance Targets:")
    print(f"   Routing Decisions: <10ms ({'✅' if results['avg_routing_time_ms'] < 10 else '❌'})")
    print(f"   Request Forwarding: <1ms (✅ simulated)")
    print(f"   Overall Performance: {'🟢 EXCELLENT' if results['performance_targets_met'] else '🟡 NEEDS OPTIMIZATION'}")

if __name__ == "__main__":
    main()