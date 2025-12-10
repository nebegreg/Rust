"""
Intelligent Caching System for Ultimate Rotoscopy
=================================================

Provides multi-level caching for:
- Model inference results
- Intermediate computations
- Temporal video data
- Memory-efficient processing

Supports both disk and memory caching with LRU eviction.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import hashlib
import pickle
import threading
import time
from collections import OrderedDict
import weakref


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"        # Fast, limited capacity
    DISK = "disk"           # Slower, larger capacity
    HYBRID = "hybrid"        # Both levels


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    FIFO = "fifo"            # First In First Out
    TTL = "ttl"              # Time To Live


@dataclass
class CacheConfig:
    """Caching configuration."""
    level: CacheLevel = CacheLevel.HYBRID
    policy: CachePolicy = CachePolicy.LRU

    # Memory cache
    max_memory_mb: int = 2048
    memory_threshold: float = 0.9  # Evict when this full

    # Disk cache
    disk_cache_path: Optional[Path] = None
    max_disk_gb: float = 50.0
    compress_disk: bool = True

    # TTL settings
    default_ttl_seconds: int = 3600  # 1 hour
    video_ttl_seconds: int = 86400   # 24 hours

    # Features
    enable_temporal_cache: bool = True
    prefetch_enabled: bool = True
    prefetch_window: int = 5


@dataclass
class CacheEntry:
    """A cached item with metadata."""
    key: str
    data: Any
    size_bytes: int
    created_at: float
    accessed_at: float
    access_count: int = 1
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds


class CacheStats:
    """Cache statistics tracking."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bytes_cached = 0
        self.bytes_evicted = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "evictions": self.evictions,
            "bytes_cached": self.bytes_cached,
            "bytes_evicted": self.bytes_evicted,
        }


class MemoryCache:
    """
    In-memory LRU cache with size limits.

    Uses OrderedDict for O(1) LRU operations.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size = 0
        self._max_size = config.max_memory_mb * 1024 * 1024
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self.stats.misses += 1
                return None

            if entry.is_expired:
                self._remove(key)
                self.stats.misses += 1
                return None

            # Update access info
            entry.accessed_at = time.time()
            entry.access_count += 1

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            self.stats.hits += 1
            return entry.data

    def put(
        self,
        key: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ):
        """Put item in cache."""
        with self._lock:
            # Calculate size
            size = self._estimate_size(data)

            # Evict if necessary
            while self._current_size + size > self._max_size * self.config.memory_threshold:
                if not self._evict_one():
                    break

            # Create entry
            entry = CacheEntry(
                key=key,
                data=data,
                size_bytes=size,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
                metadata=metadata or {},
            )

            # Remove old entry if exists
            if key in self._cache:
                self._remove(key)

            # Add new entry
            self._cache[key] = entry
            self._current_size += size
            self.stats.bytes_cached += size

    def _remove(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_size -= entry.size_bytes

    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self._cache:
            return False

        if self.config.policy == CachePolicy.LRU:
            # First item is least recently used
            key = next(iter(self._cache))
        elif self.config.policy == CachePolicy.LFU:
            # Find least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.config.policy == CachePolicy.FIFO:
            # First item is oldest
            key = next(iter(self._cache))
        else:
            key = next(iter(self._cache))

        entry = self._cache.pop(key)
        self._current_size -= entry.size_bytes
        self.stats.evictions += 1
        self.stats.bytes_evicted += entry.size_bytes

        return True

    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, dict):
            return sum(self._estimate_size(v) for v in data.values())
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_size(v) for v in data)
        else:
            return len(pickle.dumps(data))

    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0


class DiskCache:
    """
    Disk-based cache for larger data.

    Uses file-based storage with optional compression.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache_dir = config.disk_cache_path or Path.home() / ".cache" / "ultimate_rotoscopy"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self.stats = CacheStats()

        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        index_path = self._cache_dir / "index.pkl"
        if index_path.exists():
            try:
                with open(index_path, 'rb') as f:
                    self._index = pickle.load(f)
            except Exception:
                self._index = {}

    def _save_index(self):
        """Save cache index to disk."""
        index_path = self._cache_dir / "index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump(self._index, f)

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{hash_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        with self._lock:
            if key not in self._index:
                self.stats.misses += 1
                return None

            meta = self._index[key]

            # Check TTL
            if meta.get('ttl'):
                if time.time() - meta['created'] > meta['ttl']:
                    self._remove(key)
                    self.stats.misses += 1
                    return None

            # Load from disk
            path = self._key_to_path(key)
            if not path.exists():
                del self._index[key]
                self.stats.misses += 1
                return None

            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                # Decompress if needed
                if meta.get('compressed') and self.config.compress_disk:
                    import zlib
                    data = pickle.loads(zlib.decompress(data))

                self._index[key]['accessed'] = time.time()
                self._index[key]['access_count'] = meta.get('access_count', 0) + 1
                self._save_index()

                self.stats.hits += 1
                return data

            except Exception:
                self._remove(key)
                self.stats.misses += 1
                return None

    def put(
        self,
        key: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
    ):
        """Put item in disk cache."""
        with self._lock:
            path = self._key_to_path(key)

            # Serialize
            serialized = pickle.dumps(data)

            # Compress if enabled
            if self.config.compress_disk:
                import zlib
                serialized = zlib.compress(serialized)

            # Check disk space
            self._ensure_space(len(serialized))

            # Write to disk
            with open(path, 'wb') as f:
                f.write(serialized)

            # Update index
            self._index[key] = {
                'path': str(path),
                'size': len(serialized),
                'created': time.time(),
                'accessed': time.time(),
                'access_count': 1,
                'ttl': ttl_seconds or self.config.default_ttl_seconds,
                'compressed': self.config.compress_disk,
            }
            self._save_index()

            self.stats.bytes_cached += len(serialized)

    def _remove(self, key: str):
        """Remove entry from disk cache."""
        if key in self._index:
            path = self._key_to_path(key)
            if path.exists():
                path.unlink()
            del self._index[key]
            self._save_index()

    def _ensure_space(self, needed_bytes: int):
        """Ensure enough disk space is available."""
        max_bytes = int(self.config.max_disk_gb * 1024 * 1024 * 1024)
        current_size = sum(m['size'] for m in self._index.values())

        while current_size + needed_bytes > max_bytes and self._index:
            # Evict oldest accessed
            oldest_key = min(self._index.keys(), key=lambda k: self._index[k]['accessed'])
            evicted_size = self._index[oldest_key]['size']
            self._remove(oldest_key)
            current_size -= evicted_size
            self.stats.evictions += 1
            self.stats.bytes_evicted += evicted_size

    def clear(self):
        """Clear disk cache."""
        with self._lock:
            for key in list(self._index.keys()):
                self._remove(key)
            self._index = {}
            self._save_index()


class TemporalCache:
    """
    Specialized cache for video frame data.

    Maintains temporal locality for efficient
    video processing with frame access patterns.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._frames: Dict[str, Dict[int, Any]] = {}  # video_id -> {frame_idx -> data}
        self._lock = threading.RLock()
        self._prefetch_executor = None

    def get_frame(
        self,
        video_id: str,
        frame_idx: int,
    ) -> Optional[Any]:
        """Get cached frame data."""
        with self._lock:
            if video_id not in self._frames:
                return None
            return self._frames[video_id].get(frame_idx)

    def put_frame(
        self,
        video_id: str,
        frame_idx: int,
        data: Any,
    ):
        """Cache frame data."""
        with self._lock:
            if video_id not in self._frames:
                self._frames[video_id] = {}
            self._frames[video_id][frame_idx] = data

    def prefetch(
        self,
        video_id: str,
        current_frame: int,
        fetch_fn: Callable[[int], Any],
    ):
        """Prefetch upcoming frames."""
        if not self.config.prefetch_enabled:
            return

        window = self.config.prefetch_window

        for offset in range(1, window + 1):
            frame_idx = current_frame + offset

            if self.get_frame(video_id, frame_idx) is None:
                # Fetch in background
                try:
                    data = fetch_fn(frame_idx)
                    self.put_frame(video_id, frame_idx, data)
                except Exception:
                    pass

    def clear_video(self, video_id: str):
        """Clear cache for specific video."""
        with self._lock:
            if video_id in self._frames:
                del self._frames[video_id]


class IntelligentCache:
    """
    Unified intelligent caching system.

    Combines memory, disk, and temporal caching
    with automatic level selection and prefetching.

    Example:
        >>> cache = IntelligentCache(CacheConfig(
        ...     level=CacheLevel.HYBRID,
        ...     max_memory_mb=4096,
        ... ))
        >>>
        >>> # Cache inference result
        >>> cache.put("depth_frame_0", depth_result)
        >>>
        >>> # Retrieve with fallback
        >>> result = cache.get("depth_frame_0")
        >>>
        >>> # Cache frame sequence
        >>> cache.put_frame("video1", 0, matte_result)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()

        self.memory_cache = MemoryCache(self.config)

        if self.config.level in (CacheLevel.DISK, CacheLevel.HYBRID):
            self.disk_cache = DiskCache(self.config)
        else:
            self.disk_cache = None

        if self.config.enable_temporal_cache:
            self.temporal_cache = TemporalCache(self.config)
        else:
            self.temporal_cache = None

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Checks memory first, then disk.
        """
        # Try memory cache
        result = self.memory_cache.get(key)
        if result is not None:
            return result

        # Try disk cache
        if self.disk_cache is not None:
            result = self.disk_cache.get(key)
            if result is not None:
                # Promote to memory
                self.memory_cache.put(key, result)
                return result

        return None

    def put(
        self,
        key: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
        prefer_disk: bool = False,
    ):
        """
        Put item in cache.

        Automatically selects appropriate cache level.
        """
        # Estimate size
        if isinstance(data, np.ndarray):
            size = data.nbytes
        else:
            size = len(pickle.dumps(data))

        # Large items go to disk
        large_threshold = 10 * 1024 * 1024  # 10MB

        if prefer_disk or size > large_threshold:
            if self.disk_cache is not None:
                self.disk_cache.put(key, data, ttl_seconds)
            else:
                self.memory_cache.put(key, data, ttl_seconds)
        else:
            self.memory_cache.put(key, data, ttl_seconds)

            # Also write to disk for persistence
            if self.config.level == CacheLevel.HYBRID and self.disk_cache is not None:
                self.disk_cache.put(key, data, ttl_seconds)

    def get_frame(
        self,
        video_id: str,
        frame_idx: int,
    ) -> Optional[Any]:
        """Get cached video frame."""
        if self.temporal_cache:
            return self.temporal_cache.get_frame(video_id, frame_idx)
        return self.get(f"{video_id}_frame_{frame_idx}")

    def put_frame(
        self,
        video_id: str,
        frame_idx: int,
        data: Any,
    ):
        """Cache video frame."""
        if self.temporal_cache:
            self.temporal_cache.put_frame(video_id, frame_idx, data)
        else:
            self.put(f"{video_id}_frame_{frame_idx}", data, self.config.video_ttl_seconds)

    def invalidate(self, key: str):
        """Invalidate cache entry."""
        self.memory_cache._remove(key)
        if self.disk_cache:
            self.disk_cache._remove(key)

    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory": self.memory_cache.stats.to_dict(),
        }
        if self.disk_cache:
            stats["disk"] = self.disk_cache.stats.to_dict()
        return stats


def cached(
    cache: IntelligentCache,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = None,
):
    """
    Decorator for caching function results.

    Example:
        >>> @cached(cache, key_fn=lambda img: hash(img.tobytes()))
        ... def process_image(img):
        ...     return expensive_operation(img)
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}_{hash((args, tuple(kwargs.items())))}"

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(key, result, ttl)
            return result

        return wrapper
    return decorator
