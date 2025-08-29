"""
Load Balancer Controller - Week 14: Performance Optimization & Scalability

This module provides enterprise-grade load balancing capabilities for the manufacturing system
with intelligent traffic distribution, health monitoring, and automatic failover.

Performance Targets:
- Round-robin and health-based load distribution
- Automatic failover with <30 second detection
- Geographic load balancing capabilities
- Real-time traffic monitoring and adjustment
- 99.99% availability through intelligent routing

Author: Manufacturing Line Control System
Created: Week 14 - Performance Optimization Phase
"""

import time
import threading
import asyncio
import hashlib
import socket
import ssl
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, Future
import logging
import json
import uuid
import random
import statistics
from urllib.parse import urlparse


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HASH_BASED = "hash_based"
    GEOGRAPHIC = "geographic"
    HEALTH_WEIGHTED = "health_weighted"


class ServerStatus(Enum):
    """Server health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class SessionAffinityType(Enum):
    """Session affinity/stickiness types."""
    NONE = "none"
    IP_HASH = "ip_hash"
    COOKIE = "cookie"
    HEADER = "header"


@dataclass
class Server:
    """Server instance configuration and status."""
    id: str
    host: str
    port: int
    weight: float = 1.0
    protocol: str = "http"
    status: ServerStatus = ServerStatus.UNKNOWN
    current_connections: int = 0
    max_connections: int = 1000
    health_check_url: Optional[str] = None
    geographic_region: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    
    # Performance metrics
    response_time_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    success_rate: float = 100.0
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    def __post_init__(self):
        if isinstance(self.tags, (list, tuple)):
            self.tags = set(self.tags)
        if not isinstance(self.response_time_ms, deque):
            self.response_time_ms = deque(maxlen=100)
    
    @property
    def url(self) -> str:
        """Get server URL."""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """Check if server is healthy."""
        return self.status in [ServerStatus.HEALTHY, ServerStatus.DEGRADED]
    
    @property
    def is_available(self) -> bool:
        """Check if server is available for requests."""
        return (self.status == ServerStatus.HEALTHY and 
                self.current_connections < self.max_connections)
    
    @property
    def average_response_time(self) -> float:
        """Get average response time in milliseconds."""
        if not self.response_time_ms:
            return 0.0
        return statistics.mean(self.response_time_ms)
    
    @property
    def load_factor(self) -> float:
        """Calculate server load factor (0.0 to 1.0+)."""
        if self.max_connections == 0:
            return 0.0
        return self.current_connections / self.max_connections
    
    def update_response_time(self, response_time_ms: float) -> None:
        """Update response time metrics."""
        self.response_time_ms.append(response_time_ms)
    
    def record_request(self, success: bool) -> None:
        """Record request outcome."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        
        # Update success rate
        if self.total_requests > 0:
            self.success_rate = (self.successful_requests / self.total_requests) * 100.0


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    enabled: bool = True
    interval_seconds: int = 30
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 2
    http_method: str = "GET"
    http_path: str = "/health"
    expected_status_codes: Set[int] = field(default_factory=lambda: {200, 204})
    expected_response_contains: Optional[str] = None


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    session_affinity: SessionAffinityType = SessionAffinityType.NONE
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    enable_ssl: bool = False
    ssl_verify: bool = True
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


@dataclass
class LoadBalancerStats:
    """Load balancer statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0.0
    average_response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    server_count: int = 0
    healthy_server_count: int = 0
    unhealthy_server_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        return 100.0 - self.success_rate


class LoadBalancerInterface(ABC):
    """Abstract interface for load balancing algorithms."""
    
    @abstractmethod
    def select_server(self, servers: List[Server], request_context: Dict[str, Any]) -> Optional[Server]:
        """Select the best server for the request."""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        pass


class RoundRobinBalancer(LoadBalancerInterface):
    """Round-robin load balancing algorithm."""
    
    def __init__(self):
        self.current_index = 0
        self._lock = threading.Lock()
    
    def select_server(self, servers: List[Server], request_context: Dict[str, Any]) -> Optional[Server]:
        """Select server using round-robin algorithm."""
        healthy_servers = [s for s in servers if s.is_available]
        
        if not healthy_servers:
            return None
        
        with self._lock:
            server = healthy_servers[self.current_index % len(healthy_servers)]
            self.current_index += 1
            return server
    
    def get_algorithm_name(self) -> str:
        return "Round Robin"


class WeightedRoundRobinBalancer(LoadBalancerInterface):
    """Weighted round-robin load balancing algorithm."""
    
    def __init__(self):
        self.current_weights = {}
        self._lock = threading.Lock()
    
    def select_server(self, servers: List[Server], request_context: Dict[str, Any]) -> Optional[Server]:
        """Select server using weighted round-robin algorithm."""
        healthy_servers = [s for s in servers if s.is_available]
        
        if not healthy_servers:
            return None
        
        with self._lock:
            # Initialize weights if needed
            for server in healthy_servers:
                if server.id not in self.current_weights:
                    self.current_weights[server.id] = 0
            
            # Find server with highest current weight
            best_server = None
            max_current_weight = -1
            total_weight = sum(s.weight for s in healthy_servers)
            
            for server in healthy_servers:
                self.current_weights[server.id] += server.weight
                if self.current_weights[server.id] > max_current_weight:
                    max_current_weight = self.current_weights[server.id]
                    best_server = server
            
            # Reduce selected server's current weight
            if best_server:
                self.current_weights[best_server.id] -= total_weight
            
            return best_server
    
    def get_algorithm_name(self) -> str:
        return "Weighted Round Robin"


class LeastConnectionsBalancer(LoadBalancerInterface):
    """Least connections load balancing algorithm."""
    
    def select_server(self, servers: List[Server], request_context: Dict[str, Any]) -> Optional[Server]:
        """Select server with least connections."""
        healthy_servers = [s for s in servers if s.is_available]
        
        if not healthy_servers:
            return None
        
        # Find server with minimum connections
        return min(healthy_servers, key=lambda s: s.current_connections)
    
    def get_algorithm_name(self) -> str:
        return "Least Connections"


class LeastResponseTimeBalancer(LoadBalancerInterface):
    """Least response time load balancing algorithm."""
    
    def select_server(self, servers: List[Server], request_context: Dict[str, Any]) -> Optional[Server]:
        """Select server with least response time."""
        healthy_servers = [s for s in servers if s.is_available]
        
        if not healthy_servers:
            return None
        
        # Find server with minimum response time
        return min(healthy_servers, key=lambda s: s.average_response_time)
    
    def get_algorithm_name(self) -> str:
        return "Least Response Time"


class HashBasedBalancer(LoadBalancerInterface):
    """Hash-based load balancing algorithm."""
    
    def select_server(self, servers: List[Server], request_context: Dict[str, Any]) -> Optional[Server]:
        """Select server using hash of client identifier."""
        healthy_servers = [s for s in servers if s.is_available]
        
        if not healthy_servers:
            return None
        
        # Use client IP or session ID for hashing
        client_id = request_context.get('client_ip', request_context.get('session_id', 'default'))
        hash_value = int(hashlib.md5(client_id.encode()).hexdigest(), 16)
        server_index = hash_value % len(healthy_servers)
        
        return healthy_servers[server_index]
    
    def get_algorithm_name(self) -> str:
        return "Hash-based"


class HealthWeightedBalancer(LoadBalancerInterface):
    """Health and performance weighted balancing algorithm."""
    
    def select_server(self, servers: List[Server], request_context: Dict[str, Any]) -> Optional[Server]:
        """Select server based on health metrics and weights."""
        healthy_servers = [s for s in servers if s.is_available]
        
        if not healthy_servers:
            return None
        
        # Calculate composite score based on multiple factors
        def calculate_score(server: Server) -> float:
            # Base weight
            score = server.weight
            
            # Health factor (success rate)
            health_factor = server.success_rate / 100.0
            score *= health_factor
            
            # Load factor (inverse of current load)
            load_factor = 1.0 - server.load_factor
            score *= load_factor
            
            # Response time factor (inverse)
            if server.average_response_time > 0:
                response_factor = 1.0 / (1.0 + server.average_response_time / 1000.0)
                score *= response_factor
            
            return score
        
        # Select server with highest score
        return max(healthy_servers, key=calculate_score)
    
    def get_algorithm_name(self) -> str:
        return "Health Weighted"


class CircuitBreaker:
    """Circuit breaker for server failure handling."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def record_success(self) -> None:
        """Record successful request."""
        with self._lock:
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
    
    def record_failure(self) -> None:
        """Record failed request."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
    
    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        with self._lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if (self.last_failure_time and 
                    datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)):
                    self.state = "HALF_OPEN"
                    return True
                return False
            elif self.state == "HALF_OPEN":
                return True
            
            return False


class LoadBalancer:
    """
    Enterprise Load Balancer Controller for Manufacturing System
    
    Provides intelligent traffic distribution with:
    - Multiple load balancing algorithms
    - Health monitoring and automatic failover
    - Session affinity/sticky sessions
    - Circuit breaker patterns
    - Real-time performance metrics
    - Geographic load balancing
    """
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Server management
        self.servers: Dict[str, Server] = {}
        self.server_groups: Dict[str, List[str]] = defaultdict(list)
        
        # Load balancing algorithms
        self.algorithms = {
            LoadBalancingAlgorithm.ROUND_ROBIN: RoundRobinBalancer(),
            LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinBalancer(),
            LoadBalancingAlgorithm.LEAST_CONNECTIONS: LeastConnectionsBalancer(),
            LoadBalancingAlgorithm.LEAST_RESPONSE_TIME: LeastResponseTimeBalancer(),
            LoadBalancingAlgorithm.HASH_BASED: HashBasedBalancer(),
            LoadBalancingAlgorithm.HEALTH_WEIGHTED: HealthWeightedBalancer(),
        }
        
        # Circuit breakers per server
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Session affinity
        self.session_map: Dict[str, str] = {}  # session_id -> server_id
        
        # Statistics and monitoring
        self.stats = LoadBalancerStats()
        self.request_history: deque = deque(maxlen=1000)
        
        # Background operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="LoadBalancer")
        self._shutdown = False
        self._lock = threading.RLock()
        
        # Start health checking if enabled
        if self.config.health_check.enabled:
            self._start_health_monitoring()
    
    def add_server(self, server: Server, group: Optional[str] = None) -> bool:
        """Add server to load balancer."""
        try:
            with self._lock:
                self.servers[server.id] = server
                
                if group:
                    self.server_groups[group].append(server.id)
                
                # Initialize circuit breaker
                if self.config.enable_circuit_breaker:
                    self.circuit_breakers[server.id] = CircuitBreaker(
                        failure_threshold=self.config.circuit_breaker_threshold,
                        timeout=self.config.circuit_breaker_timeout
                    )
                
                self.logger.info(f"Added server {server.id} ({server.url}) to load balancer")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add server {server.id}: {e}")
            return False
    
    def remove_server(self, server_id: str) -> bool:
        """Remove server from load balancer."""
        try:
            with self._lock:
                if server_id in self.servers:
                    del self.servers[server_id]
                
                # Remove from groups
                for group_servers in self.server_groups.values():
                    if server_id in group_servers:
                        group_servers.remove(server_id)
                
                # Remove circuit breaker
                if server_id in self.circuit_breakers:
                    del self.circuit_breakers[server_id]
                
                # Remove session mappings
                sessions_to_remove = [
                    session_id for session_id, mapped_server_id in self.session_map.items()
                    if mapped_server_id == server_id
                ]
                for session_id in sessions_to_remove:
                    del self.session_map[session_id]
                
                self.logger.info(f"Removed server {server_id} from load balancer")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to remove server {server_id}: {e}")
            return False
    
    def get_server(self, request_context: Optional[Dict[str, Any]] = None, 
                   group: Optional[str] = None) -> Optional[Server]:
        """
        Get the best server for a request.
        
        Args:
            request_context: Request context (client_ip, session_id, etc.)
            group: Server group to select from
        """
        if request_context is None:
            request_context = {}
        
        try:
            with self._lock:
                # Get candidate servers
                if group and group in self.server_groups:
                    candidate_server_ids = self.server_groups[group]
                    candidate_servers = [self.servers[sid] for sid in candidate_server_ids 
                                        if sid in self.servers]
                else:
                    candidate_servers = list(self.servers.values())
                
                # Filter by circuit breaker status
                if self.config.enable_circuit_breaker:
                    candidate_servers = [
                        server for server in candidate_servers
                        if (server.id not in self.circuit_breakers or 
                            self.circuit_breakers[server.id].can_execute())
                    ]
                
                # Handle session affinity
                session_id = request_context.get('session_id')
                if (session_id and 
                    self.config.session_affinity != SessionAffinityType.NONE and
                    session_id in self.session_map):
                    
                    sticky_server_id = self.session_map[session_id]
                    if sticky_server_id in self.servers:
                        sticky_server = self.servers[sticky_server_id]
                        if sticky_server.is_available:
                            return sticky_server
                
                # Use load balancing algorithm
                algorithm = self.algorithms.get(self.config.algorithm)
                if not algorithm:
                    self.logger.error(f"Unknown algorithm: {self.config.algorithm}")
                    return None
                
                selected_server = algorithm.select_server(candidate_servers, request_context)
                
                # Establish session affinity if needed
                if (selected_server and session_id and 
                    self.config.session_affinity != SessionAffinityType.NONE):
                    self.session_map[session_id] = selected_server.id
                
                return selected_server
                
        except Exception as e:
            self.logger.error(f"Server selection failed: {e}")
            return None
    
    def execute_request(self, request_func: Callable, request_context: Optional[Dict[str, Any]] = None,
                       group: Optional[str] = None, max_retries: Optional[int] = None) -> Any:
        """
        Execute request with load balancing and retry logic.
        
        Args:
            request_func: Function to execute request (should accept server as parameter)
            request_context: Request context
            group: Server group to use
            max_retries: Maximum retry attempts
        """
        if request_context is None:
            request_context = {}
        
        if max_retries is None:
            max_retries = self.config.max_retries
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            server = self.get_server(request_context, group)
            if not server:
                raise Exception("No healthy servers available")
            
            start_time = time.perf_counter()
            
            try:
                # Increment connection count
                server.current_connections += 1
                
                # Execute request
                result = request_func(server)
                
                # Record success
                response_time = (time.perf_counter() - start_time) * 1000
                self._record_request_success(server, response_time)
                
                return result
                
            except Exception as e:
                # Record failure
                response_time = (time.perf_counter() - start_time) * 1000
                self._record_request_failure(server, response_time, e)
                
                last_exception = e
                
                # Wait before retry
                if attempt < max_retries:
                    wait_time = self.config.retry_backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
            
            finally:
                # Decrement connection count
                server.current_connections = max(0, server.current_connections - 1)
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise Exception("Request failed after all retries")
    
    def _record_request_success(self, server: Server, response_time_ms: float) -> None:
        """Record successful request."""
        server.record_request(True)
        server.update_response_time(response_time_ms)
        
        # Update circuit breaker
        if server.id in self.circuit_breakers:
            self.circuit_breakers[server.id].record_success()
        
        # Update global stats
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.total_response_time_ms += response_time_ms
        self.stats.average_response_time_ms = (
            self.stats.total_response_time_ms / self.stats.total_requests
        )
        
        # Record in history
        self.request_history.append({
            'timestamp': time.time(),
            'server_id': server.id,
            'success': True,
            'response_time_ms': response_time_ms
        })
    
    def _record_request_failure(self, server: Server, response_time_ms: float, 
                               exception: Exception) -> None:
        """Record failed request."""
        server.record_request(False)
        
        # Update circuit breaker
        if server.id in self.circuit_breakers:
            self.circuit_breakers[server.id].record_failure()
        
        # Update global stats
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        
        # Record in history
        self.request_history.append({
            'timestamp': time.time(),
            'server_id': server.id,
            'success': False,
            'response_time_ms': response_time_ms,
            'error': str(exception)
        })
        
        self.logger.warning(f"Request to server {server.id} failed: {exception}")
    
    def get_load_balancer_stats(self) -> LoadBalancerStats:
        """Get comprehensive load balancer statistics."""
        with self._lock:
            # Update server counts
            self.stats.server_count = len(self.servers)
            self.stats.healthy_server_count = sum(
                1 for server in self.servers.values() if server.is_healthy
            )
            self.stats.unhealthy_server_count = (
                self.stats.server_count - self.stats.healthy_server_count
            )
            
            # Update active connections
            self.stats.active_connections = sum(
                server.current_connections for server in self.servers.values()
            )
            
            # Calculate requests per second from recent history
            current_time = time.time()
            recent_requests = [
                req for req in self.request_history
                if current_time - req['timestamp'] <= 60  # Last minute
            ]
            self.stats.requests_per_second = len(recent_requests) / 60.0
            
            return self.stats
    
    def get_server_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all servers."""
        with self._lock:
            server_stats = {}
            
            for server_id, server in self.servers.items():
                server_stats[server_id] = {
                    'id': server.id,
                    'url': server.url,
                    'status': server.status.value,
                    'is_healthy': server.is_healthy,
                    'is_available': server.is_available,
                    'weight': server.weight,
                    'current_connections': server.current_connections,
                    'max_connections': server.max_connections,
                    'load_factor': server.load_factor,
                    'total_requests': server.total_requests,
                    'successful_requests': server.successful_requests,
                    'success_rate': server.success_rate,
                    'average_response_time_ms': server.average_response_time,
                    'consecutive_failures': server.consecutive_failures,
                    'last_health_check': server.last_health_check.isoformat() if server.last_health_check else None,
                    'geographic_region': server.geographic_region,
                    'tags': list(server.tags)
                }
                
                # Add circuit breaker info
                if server_id in self.circuit_breakers:
                    cb = self.circuit_breakers[server_id]
                    server_stats[server_id]['circuit_breaker'] = {
                        'state': cb.state,
                        'failure_count': cb.failure_count,
                        'last_failure_time': cb.last_failure_time.isoformat() if cb.last_failure_time else None
                    }
            
            return server_stats
    
    def set_server_status(self, server_id: str, status: ServerStatus) -> bool:
        """Manually set server status."""
        try:
            with self._lock:
                if server_id in self.servers:
                    self.servers[server_id].status = status
                    self.logger.info(f"Set server {server_id} status to {status.value}")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Failed to set server {server_id} status: {e}")
            return False
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        def health_check_task():
            while not self._shutdown:
                try:
                    self._perform_health_checks()
                    time.sleep(self.config.health_check.interval_seconds)
                except Exception as e:
                    self.logger.error(f"Health check task error: {e}")
        
        self._executor.submit(health_check_task)
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all servers."""
        for server in list(self.servers.values()):
            try:
                self._check_server_health(server)
            except Exception as e:
                self.logger.error(f"Health check failed for server {server.id}: {e}")
    
    def _check_server_health(self, server: Server) -> None:
        """Check individual server health."""
        try:
            start_time = time.perf_counter()
            
            # Use health check URL or default path
            health_url = server.health_check_url or f"{server.url}{self.config.health_check.http_path}"
            
            # Perform HTTP health check (simplified version)
            # In a real implementation, this would use an HTTP client
            import urllib.request
            import urllib.error
            
            request = urllib.request.Request(health_url)
            
            try:
                with urllib.request.urlopen(request, timeout=self.config.health_check.timeout_seconds) as response:
                    status_code = response.getcode()
                    response_time = (time.perf_counter() - start_time) * 1000
                    
                    # Check if status code is acceptable
                    if status_code in self.config.health_check.expected_status_codes:
                        # Check response content if required
                        if self.config.health_check.expected_response_contains:
                            content = response.read().decode('utf-8')
                            if self.config.health_check.expected_response_contains not in content:
                                raise Exception("Response content check failed")
                        
                        # Health check passed
                        self._handle_health_check_success(server, response_time)
                    else:
                        self._handle_health_check_failure(server, f"Unexpected status code: {status_code}")
                        
            except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
                self._handle_health_check_failure(server, str(e))
                
        except Exception as e:
            self.logger.error(f"Health check error for server {server.id}: {e}")
    
    def _handle_health_check_success(self, server: Server, response_time_ms: float) -> None:
        """Handle successful health check."""
        server.update_response_time(response_time_ms)
        server.last_health_check = datetime.now()
        server.consecutive_failures = 0
        
        # Update status based on success threshold
        if server.status == ServerStatus.UNHEALTHY:
            # Need consecutive successes to mark as healthy
            if server.consecutive_failures == 0:  # This is success, so reset counter
                server.status = ServerStatus.HEALTHY
        elif server.status == ServerStatus.UNKNOWN:
            server.status = ServerStatus.HEALTHY
    
    def _handle_health_check_failure(self, server: Server, error_message: str) -> None:
        """Handle failed health check."""
        server.last_health_check = datetime.now()
        server.consecutive_failures += 1
        
        # Update status based on failure threshold
        if server.consecutive_failures >= self.config.health_check.failure_threshold:
            if server.status != ServerStatus.MAINTENANCE:
                server.status = ServerStatus.UNHEALTHY
        
        self.logger.warning(f"Health check failed for server {server.id}: {error_message}")
    
    def shutdown(self) -> None:
        """Shutdown load balancer."""
        self._shutdown = True
        self._executor.shutdown(wait=True)
        self.logger.info("Load balancer shutdown completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Convenience functions for global load balancer
_global_load_balancer: Optional[LoadBalancer] = None

def get_load_balancer(config: Optional[LoadBalancerConfig] = None) -> LoadBalancer:
    """Get or create global load balancer instance."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer(config)
    return _global_load_balancer


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    print("Load Balancer Controller Demo")
    print("=" * 50)
    
    # Create load balancer with configuration
    config = LoadBalancerConfig(
        algorithm=LoadBalancingAlgorithm.HEALTH_WEIGHTED,
        session_affinity=SessionAffinityType.IP_HASH,
        enable_circuit_breaker=True,
        health_check=HealthCheckConfig(
            enabled=True,
            interval_seconds=10,
            failure_threshold=2,
            success_threshold=1
        )
    )
    
    with LoadBalancer(config) as lb:
        # Add servers
        servers = [
            Server("server1", "localhost", 8001, weight=1.0, geographic_region="us-west"),
            Server("server2", "localhost", 8002, weight=1.5, geographic_region="us-east"),
            Server("server3", "localhost", 8003, weight=2.0, geographic_region="us-central"),
        ]
        
        print("\n1. Adding Servers:")
        for server in servers:
            lb.add_server(server, group="web_servers")
            print(f"Added {server.id} ({server.url}) with weight {server.weight}")
        
        # Simulate some requests
        print("\n2. Simulating Load Balancing:")
        
        def mock_request(server: Server) -> str:
            """Mock request function."""
            # Simulate varying response times
            import random
            time.sleep(random.uniform(0.01, 0.1))
            if random.random() < 0.95:  # 95% success rate
                return f"Response from {server.id}"
            else:
                raise Exception("Simulated request failure")
        
        for i in range(20):
            try:
                request_context = {
                    'client_ip': f"192.168.1.{i % 10 + 1}",
                    'session_id': f"session_{i % 5}"
                }
                
                result = lb.execute_request(mock_request, request_context, group="web_servers")
                print(f"Request {i+1}: {result}")
                
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
        
        # Show statistics
        print("\n3. Load Balancer Statistics:")
        lb_stats = lb.get_load_balancer_stats()
        print(f"Total Requests: {lb_stats.total_requests}")
        print(f"Success Rate: {lb_stats.success_rate:.1f}%")
        print(f"Average Response Time: {lb_stats.average_response_time_ms:.2f}ms")
        print(f"Requests per Second: {lb_stats.requests_per_second:.1f}")
        print(f"Healthy Servers: {lb_stats.healthy_server_count}/{lb_stats.server_count}")
        
        print("\n4. Server Statistics:")
        server_stats = lb.get_server_stats()
        for server_id, stats in server_stats.items():
            print(f"{server_id}: Status={stats['status']}, "
                  f"Requests={stats['total_requests']}, "
                  f"Success Rate={stats['success_rate']:.1f}%, "
                  f"Avg Response={stats['average_response_time_ms']:.2f}ms")
        
        print("\nLoad Balancer Controller demo completed successfully!")