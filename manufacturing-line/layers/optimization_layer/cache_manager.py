"""
Multi-Level Cache Manager - Week 14: Performance Optimization & Scalability

This module provides enterprise-grade caching capabilities for the manufacturing system
with multiple cache levels, intelligent policies, and automatic optimization features.

Performance Targets:
- Cache hit rate >90% for frequently accessed data
- Cache lookup time <1ms
- 40%+ performance improvement through caching
- Automatic cache optimization

Author: Manufacturing Line Control System
Created: Week 14 - Performance Optimization Phase
"""

import time
import json
import hashlib
import threading
import tracemalloc
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
import weakref
import pickle
import os
import tempfile


class CacheLevel(Enum):
    """Cache levels for multi-level caching architecture."""
    L1_MEMORY = "L1_Memory"
    L2_REDIS = "L2_Redis"
    L3_DATABASE = "L3_Database"


class CachePolicy(Enum):
    """Cache eviction and management policies."""
    LRU = "LeastRecentlyUsed"
    LFU = "LeastFrequentlyUsed" 
    FIFO = "FirstInFirstOut"
    TTL = "TimeToLive"
    ADAPTIVE = "AdaptivePolicy"


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    access_count: int
    last_access: datetime
    size_bytes: int
    ttl_seconds: Optional[int] = None
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def touch(self) -> None:
        """Update access metadata."""
        self.last_access = datetime.now()
        self.access_count += 1
        self.hit_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    level: str
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    total_requests: int = 0
    cache_size: int = 0
    max_size: int = 0
    memory_usage_bytes: int = 0
    hit_rate: float = 0.0
    average_lookup_time_ms: float = 0.0
    total_lookup_time_ms: float = 0.0
    
    def calculate_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self.total_requests == 0:
            return 0.0
        self.hit_rate = (self.hit_count / self.total_requests) * 100.0
        return self.hit_rate
    
    def update_lookup_time(self, lookup_time_ms: float) -> None:
        """Update average lookup time."""
        self.total_lookup_time_ms += lookup_time_ms
        if self.total_requests > 0:
            self.average_lookup_time_ms = self.total_lookup_time_ms / self.total_requests


class CacheInterface(ABC):
    """Abstract interface for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class MemoryCache(CacheInterface):
    """L1 in-memory cache implementation with LRU policy."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats(level="L1_Memory", max_size=max_size)
        self._lock = threading.RLock()
        
    def _make_key_hash(self, key: str) -> str:
        """Create hash for key if needed."""
        if len(key) > 250:  # Avoid very long keys
            return hashlib.md5(key.encode()).hexdigest()
        return key
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
            self.stats.eviction_count += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries if over capacity."""
        while len(self.cache) >= self.max_size:
            oldest_key, _ = self.cache.popitem(last=False)
            self.stats.eviction_count += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        start_time = time.perf_counter()
        key_hash = self._make_key_hash(key)
        
        with self._lock:
            self.stats.total_requests += 1
            
            # Clean expired entries periodically
            if self.stats.total_requests % 100 == 0:
                self._evict_expired()
            
            if key_hash in self.cache:
                entry = self.cache[key_hash]
                if not entry.is_expired():
                    entry.touch()
                    # Move to end (most recent)
                    self.cache.move_to_end(key_hash)
                    self.stats.hit_count += 1
                    
                    lookup_time = (time.perf_counter() - start_time) * 1000
                    self.stats.update_lookup_time(lookup_time)
                    return entry.value
                else:
                    # Expired entry
                    del self.cache[key_hash]
                    self.stats.eviction_count += 1
            
            self.stats.miss_count += 1
            lookup_time = (time.perf_counter() - start_time) * 1000
            self.stats.update_lookup_time(lookup_time)
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        key_hash = self._make_key_hash(key)
        ttl = ttl_seconds or self.default_ttl
        
        try:
            # Calculate approximate size
            size_bytes = len(pickle.dumps(value))
            
            with self._lock:
                expires_at = None
                if ttl:
                    expires_at = datetime.now() + timedelta(seconds=ttl)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=datetime.now(),
                    access_count=0,
                    last_access=datetime.now(),
                    size_bytes=size_bytes,
                    ttl_seconds=ttl,
                    expires_at=expires_at
                )
                
                # Evict if necessary
                self._evict_lru()
                
                self.cache[key_hash] = entry
                self.cache.move_to_end(key_hash)  # Mark as most recent
                self.stats.cache_size = len(self.cache)
                
            return True
        except Exception as e:
            logging.error(f"MemoryCache.set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        key_hash = self._make_key_hash(key)
        with self._lock:
            if key_hash in self.cache:
                del self.cache[key_hash]
                self.stats.cache_size = len(self.cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from memory cache."""
        with self._lock:
            self.cache.clear()
            self.stats.cache_size = 0
            self.stats.eviction_count += len(self.cache)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        key_hash = self._make_key_hash(key)
        with self._lock:
            return key_hash in self.cache and not self.cache[key_hash].is_expired()
    
    def get_stats(self) -> CacheStats:
        """Get memory cache statistics."""
        with self._lock:
            self.stats.cache_size = len(self.cache)
            self.stats.calculate_hit_rate()
            # Estimate memory usage
            self.stats.memory_usage_bytes = sum(
                entry.size_bytes for entry in self.cache.values()
            )
            return self.stats


class FileCache(CacheInterface):
    """L2 file-based cache implementation for persistence."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size: int = 10000):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "manufacturing_cache")
        self.max_size = max_size
        self.stats = CacheStats(level="L2_File", max_size=max_size)
        self._lock = threading.RLock()
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _get_meta_path(self, key: str) -> str:
        """Get metadata file path for cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.meta")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        start_time = time.perf_counter()
        
        with self._lock:
            self.stats.total_requests += 1
            
            try:
                file_path = self._get_file_path(key)
                meta_path = self._get_meta_path(key)
                
                if not (os.path.exists(file_path) and os.path.exists(meta_path)):
                    self.stats.miss_count += 1
                    return None
                
                # Load metadata
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                # Check expiration
                if meta.get('expires_at'):
                    expires_at = datetime.fromisoformat(meta['expires_at'])
                    if datetime.now() > expires_at:
                        # Clean up expired files
                        os.remove(file_path)
                        os.remove(meta_path)
                        self.stats.miss_count += 1
                        self.stats.eviction_count += 1
                        return None
                
                # Load value
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update metadata
                meta['access_count'] = meta.get('access_count', 0) + 1
                meta['last_access'] = datetime.now().isoformat()
                meta['hit_count'] = meta.get('hit_count', 0) + 1
                
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)
                
                self.stats.hit_count += 1
                lookup_time = (time.perf_counter() - start_time) * 1000
                self.stats.update_lookup_time(lookup_time)
                return value
                
            except Exception as e:
                logging.error(f"FileCache.get error: {e}")
                self.stats.miss_count += 1
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in file cache."""
        with self._lock:
            try:
                file_path = self._get_file_path(key)
                meta_path = self._get_meta_path(key)
                
                # Save value
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Save metadata
                meta = {
                    'key': key,
                    'timestamp': datetime.now().isoformat(),
                    'access_count': 0,
                    'last_access': datetime.now().isoformat(),
                    'size_bytes': os.path.getsize(file_path),
                    'ttl_seconds': ttl_seconds,
                    'hit_count': 0
                }
                
                if ttl_seconds:
                    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
                    meta['expires_at'] = expires_at.isoformat()
                
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)
                
                return True
                
            except Exception as e:
                logging.error(f"FileCache.set error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key from file cache."""
        with self._lock:
            try:
                file_path = self._get_file_path(key)
                meta_path = self._get_meta_path(key)
                
                deleted = False
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted = True
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                    deleted = True
                
                return deleted
                
            except Exception as e:
                logging.error(f"FileCache.delete error: {e}")
                return False
    
    def clear(self) -> None:
        """Clear all files from cache."""
        with self._lock:
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith(('.cache', '.meta')):
                        file_path = os.path.join(self.cache_dir, filename)
                        os.remove(file_path)
            except Exception as e:
                logging.error(f"FileCache.clear error: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in file cache."""
        file_path = self._get_file_path(key)
        meta_path = self._get_meta_path(key)
        return os.path.exists(file_path) and os.path.exists(meta_path)
    
    def get_stats(self) -> CacheStats:
        """Get file cache statistics."""
        with self._lock:
            try:
                cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
                self.stats.cache_size = len(cache_files)
                
                # Calculate total size
                total_size = 0
                for filename in cache_files:
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
                
                self.stats.memory_usage_bytes = total_size
                self.stats.calculate_hit_rate()
                
            except Exception as e:
                logging.error(f"FileCache.get_stats error: {e}")
            
            return self.stats


@dataclass
class CacheConfiguration:
    """Configuration for multi-level cache manager."""
    l1_max_size: int = 1000
    l1_default_ttl: Optional[int] = 3600  # 1 hour
    l2_max_size: int = 10000
    l2_cache_dir: Optional[str] = None
    enable_l3_database: bool = False
    cache_warming_enabled: bool = True
    auto_optimization_enabled: bool = True
    stats_collection_interval: int = 60  # seconds
    background_cleanup_enabled: bool = True
    cleanup_interval: int = 300  # 5 minutes


class CacheManager:
    """
    Multi-Level Cache Manager for Manufacturing System
    
    Provides enterprise-grade caching with:
    - L1: In-memory application cache (fastest)
    - L2: File-based persistent cache (medium speed)
    - L3: Database query result cache (extensible)
    
    Features:
    - Smart cache invalidation policies
    - Cache hit rate optimization
    - Automatic cache warming
    - Cache analytics and monitoring
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[CacheConfiguration] = None):
        self.config = config or CacheConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache levels
        self.l1_cache = MemoryCache(
            max_size=self.config.l1_max_size,
            default_ttl=self.config.l1_default_ttl
        )
        
        self.l2_cache = FileCache(
            cache_dir=self.config.l2_cache_dir,
            max_size=self.config.l2_max_size
        )
        
        # L3 cache would be database-specific (Redis, etc.)
        self.l3_cache: Optional[CacheInterface] = None
        
        # Cache management
        self.cache_levels = [self.l1_cache, self.l2_cache]
        if self.l3_cache:
            self.cache_levels.append(self.l3_cache)
        
        # Statistics and monitoring
        self.global_stats = CacheStats(level="Global")
        self.cache_hit_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Background operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CacheManager")
        self._shutdown = False
        
        # Start background tasks
        if self.config.background_cleanup_enabled:
            self._start_background_cleanup()
        if self.config.stats_collection_interval > 0:
            self._start_stats_collection()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from multi-level cache.
        
        Searches L1 -> L2 -> L3 and promotes successful hits to higher levels.
        """
        start_time = time.perf_counter()
        
        try:
            # Try L1 first
            value = self.l1_cache.get(key)
            if value is not None:
                lookup_time = (time.perf_counter() - start_time) * 1000
                self._record_hit(key, "L1", lookup_time)
                return value
            
            # Try L2
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                self.l1_cache.set(key, value)
                lookup_time = (time.perf_counter() - start_time) * 1000
                self._record_hit(key, "L2", lookup_time)
                return value
            
            # Try L3 if available
            if self.l3_cache:
                value = self.l3_cache.get(key)
                if value is not None:
                    # Promote to L1 and L2
                    self.l1_cache.set(key, value)
                    self.l2_cache.set(key, value)
                    lookup_time = (time.perf_counter() - start_time) * 1000
                    self._record_hit(key, "L3", lookup_time)
                    return value
            
            # Cache miss
            lookup_time = (time.perf_counter() - start_time) * 1000
            self._record_miss(key, lookup_time)
            return default
            
        except Exception as e:
            self.logger.error(f"CacheManager.get error for key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            levels: Optional[List[str]] = None) -> bool:
        """
        Set value in multi-level cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            levels: Specific cache levels to update (default: all)
        """
        if levels is None:
            levels = ["L1", "L2", "L3"]
        
        success = True
        
        try:
            if "L1" in levels:
                success &= self.l1_cache.set(key, value, ttl_seconds)
            
            if "L2" in levels:
                success &= self.l2_cache.set(key, value, ttl_seconds)
            
            if "L3" in levels and self.l3_cache:
                success &= self.l3_cache.set(key, value, ttl_seconds)
            
            return success
            
        except Exception as e:
            self.logger.error(f"CacheManager.set error for key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache levels."""
        success = True
        
        try:
            success &= self.l1_cache.delete(key)
            success &= self.l2_cache.delete(key)
            
            if self.l3_cache:
                success &= self.l3_cache.delete(key)
            
            return success
            
        except Exception as e:
            self.logger.error(f"CacheManager.delete error for key '{key}': {e}")
            return False
    
    def clear(self, levels: Optional[List[str]] = None) -> None:
        """Clear cache entries from specified levels."""
        if levels is None:
            levels = ["L1", "L2", "L3"]
        
        try:
            if "L1" in levels:
                self.l1_cache.clear()
            
            if "L2" in levels:
                self.l2_cache.clear()
            
            if "L3" in levels and self.l3_cache:
                self.l3_cache.clear()
                
        except Exception as e:
            self.logger.error(f"CacheManager.clear error: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any cache level."""
        return (self.l1_cache.exists(key) or 
                self.l2_cache.exists(key) or 
                (self.l3_cache and self.l3_cache.exists(key)))
    
    def get_cache_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels."""
        stats = {
            "L1": self.l1_cache.get_stats(),
            "L2": self.l2_cache.get_stats(),
        }
        
        if self.l3_cache:
            stats["L3"] = self.l3_cache.get_stats()
        
        # Calculate global statistics
        total_hits = sum(s.hit_count for s in stats.values())
        total_misses = sum(s.miss_count for s in stats.values())
        total_requests = total_hits + total_misses
        
        self.global_stats.hit_count = total_hits
        self.global_stats.miss_count = total_misses
        self.global_stats.total_requests = total_requests
        self.global_stats.calculate_hit_rate()
        
        stats["Global"] = self.global_stats
        return stats
    
    def optimize_cache(self) -> Dict[str, Any]:
        """
        Perform automatic cache optimization.
        
        Returns optimization recommendations and actions taken.
        """
        optimization_results = {
            'actions_taken': [],
            'recommendations': [],
            'performance_impact': {}
        }
        
        try:
            stats = self.get_cache_stats()
            
            # Analyze hit rates and suggest optimizations
            for level, stat in stats.items():
                if level == "Global":
                    continue
                
                if stat.hit_rate < 70:  # Low hit rate
                    optimization_results['recommendations'].append(
                        f"{level} cache hit rate is low ({stat.hit_rate:.1f}%). "
                        f"Consider increasing cache size or adjusting TTL."
                    )
                
                if stat.average_lookup_time_ms > 10:  # Slow lookups
                    optimization_results['recommendations'].append(
                        f"{level} cache lookup time is high ({stat.average_lookup_time_ms:.1f}ms). "
                        f"Consider optimizing cache implementation."
                    )
            
            # Auto-tune cache sizes based on usage patterns
            if self.config.auto_optimization_enabled:
                l1_stats = stats["L1"]
                if l1_stats.eviction_count > l1_stats.hit_count * 0.1:
                    # High eviction rate, increase L1 size
                    new_size = min(self.l1_cache.max_size * 2, 5000)
                    self.l1_cache.max_size = new_size
                    optimization_results['actions_taken'].append(
                        f"Increased L1 cache size to {new_size} due to high eviction rate"
                    )
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Cache optimization error: {e}")
            return optimization_results
    
    def warm_cache(self, key_value_pairs: List[Tuple[str, Any]], 
                   ttl_seconds: Optional[int] = None) -> int:
        """
        Warm cache with predefined key-value pairs.
        
        Returns number of successfully cached items.
        """
        if not self.config.cache_warming_enabled:
            return 0
        
        success_count = 0
        
        for key, value in key_value_pairs:
            try:
                if self.set(key, value, ttl_seconds):
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Cache warming failed for key '{key}': {e}")
        
        self.logger.info(f"Cache warming completed: {success_count}/{len(key_value_pairs)} items cached")
        return success_count
    
    def _record_hit(self, key: str, level: str, lookup_time_ms: float) -> None:
        """Record cache hit for analytics."""
        self.cache_hit_patterns[key].append(time.time())
        # Keep only recent history (last 1000 accesses)
        if len(self.cache_hit_patterns[key]) > 1000:
            self.cache_hit_patterns[key] = self.cache_hit_patterns[key][-1000:]
    
    def _record_miss(self, key: str, lookup_time_ms: float) -> None:
        """Record cache miss for analytics."""
        pass  # Could implement miss pattern analysis
    
    def _start_background_cleanup(self) -> None:
        """Start background cleanup thread."""
        def cleanup_task():
            while not self._shutdown:
                try:
                    # Cleanup expired entries every cleanup_interval
                    time.sleep(self.config.cleanup_interval)
                    if not self._shutdown:
                        self._cleanup_expired_entries()
                except Exception as e:
                    self.logger.error(f"Background cleanup error: {e}")
        
        self._executor.submit(cleanup_task)
    
    def _start_stats_collection(self) -> None:
        """Start background statistics collection."""
        def stats_task():
            while not self._shutdown:
                try:
                    time.sleep(self.config.stats_collection_interval)
                    if not self._shutdown:
                        self._collect_stats()
                except Exception as e:
                    self.logger.error(f"Stats collection error: {e}")
        
        self._executor.submit(stats_task)
    
    def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries from all cache levels."""
        # L1 cache cleans itself up automatically
        # L2 and L3 might need explicit cleanup
        pass
    
    def _collect_stats(self) -> None:
        """Collect and log cache statistics."""
        try:
            stats = self.get_cache_stats()
            self.logger.info(f"Cache Stats - Global Hit Rate: {stats['Global'].hit_rate:.1f}%")
        except Exception as e:
            self.logger.error(f"Stats collection error: {e}")
    
    def shutdown(self) -> None:
        """Shutdown cache manager and cleanup resources."""
        self._shutdown = True
        self._executor.shutdown(wait=True)
        self.logger.info("CacheManager shutdown completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Convenience functions for global cache instance
_global_cache_manager: Optional[CacheManager] = None

def get_cache_manager(config: Optional[CacheConfiguration] = None) -> CacheManager:
    """Get or create global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(config)
    return _global_cache_manager

def cache_get(key: str, default: Any = None) -> Any:
    """Convenience function for cache get."""
    return get_cache_manager().get(key, default)

def cache_set(key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
    """Convenience function for cache set."""
    return get_cache_manager().set(key, value, ttl_seconds)

def cache_delete(key: str) -> bool:
    """Convenience function for cache delete."""
    return get_cache_manager().delete(key)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    print("Multi-Level Cache Manager Demo")
    print("=" * 50)
    
    # Create cache manager with custom configuration
    config = CacheConfiguration(
        l1_max_size=100,
        l1_default_ttl=300,
        l2_max_size=1000,
        cache_warming_enabled=True,
        auto_optimization_enabled=True
    )
    
    with CacheManager(config) as cache_manager:
        # Demonstrate basic operations
        print("\n1. Basic Cache Operations:")
        cache_manager.set("test_key", "test_value", ttl_seconds=60)
        print(f"Set: test_key = test_value")
        
        value = cache_manager.get("test_key")
        print(f"Get: test_key = {value}")
        
        exists = cache_manager.exists("test_key")
        print(f"Exists: test_key = {exists}")
        
        # Demonstrate cache warming
        print("\n2. Cache Warming:")
        warm_data = [
            ("sensor_1", {"temperature": 25.5, "timestamp": time.time()}),
            ("sensor_2", {"temperature": 26.1, "timestamp": time.time()}),
            ("production_rate", 98.5),
            ("quality_score", 94.2)
        ]
        
        warmed_count = cache_manager.warm_cache(warm_data, ttl_seconds=120)
        print(f"Warmed {warmed_count} cache entries")
        
        # Demonstrate statistics
        print("\n3. Cache Statistics:")
        stats = cache_manager.get_cache_stats()
        for level, stat in stats.items():
            print(f"{level}: Hit Rate: {stat.hit_rate:.1f}%, "
                  f"Entries: {stat.cache_size}, "
                  f"Avg Lookup: {stat.average_lookup_time_ms:.2f}ms")
        
        # Demonstrate optimization
        print("\n4. Cache Optimization:")
        optimization = cache_manager.optimize_cache()
        print(f"Actions taken: {optimization['actions_taken']}")
        print(f"Recommendations: {optimization['recommendations']}")
        
        print("\nMulti-Level Cache Manager demo completed successfully!")