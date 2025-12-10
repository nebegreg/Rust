"""
Multi-GPU Support for Ultimate Rotoscopy
========================================

Enables distributed processing across multiple GPUs for:
- Parallel model inference
- Batch processing acceleration
- Large resolution handling
- Real-time video processing
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, Future


class DistributionStrategy(Enum):
    """GPU workload distribution strategies."""
    ROUND_ROBIN = "round_robin"       # Alternate between GPUs
    LOAD_BALANCE = "load_balance"     # Send to least loaded GPU
    BATCH_SPLIT = "batch_split"       # Split batches across GPUs
    PIPELINE = "pipeline"             # Pipeline stages across GPUs
    TILE = "tile"                     # Tile image across GPUs


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    compute_capability: Tuple[int, int]
    is_available: bool = True


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU processing."""
    device_ids: Optional[List[int]] = None  # None = use all available
    strategy: DistributionStrategy = DistributionStrategy.LOAD_BALANCE

    # Load balancing
    max_queue_per_gpu: int = 4
    memory_threshold_mb: int = 1000   # Min free memory to use GPU

    # Batch processing
    batch_size_per_gpu: int = 1

    # Tiling for large images
    enable_tiling: bool = True
    tile_size: Tuple[int, int] = (1024, 1024)
    tile_overlap: int = 64

    # Pipeline
    pipeline_stages: List[str] = field(default_factory=list)


class GPUManager:
    """
    Manages GPU resources and allocation.

    Tracks GPU memory, utilization, and availability
    for optimal workload distribution.
    """

    def __init__(self, device_ids: Optional[List[int]] = None):
        self._gpus: Dict[int, GPUInfo] = {}
        self._locks: Dict[int, threading.Lock] = {}
        self._queues: Dict[int, int] = {}

        self._discover_gpus(device_ids)

    def _discover_gpus(self, device_ids: Optional[List[int]] = None):
        """Discover available GPUs."""
        try:
            import torch

            num_gpus = torch.cuda.device_count()

            for i in range(num_gpus):
                if device_ids is not None and i not in device_ids:
                    continue

                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)

                self._gpus[i] = GPUInfo(
                    device_id=i,
                    name=props.name,
                    total_memory_mb=total_mem // (1024 * 1024),
                    free_memory_mb=free_mem // (1024 * 1024),
                    compute_capability=(props.major, props.minor),
                )

                self._locks[i] = threading.Lock()
                self._queues[i] = 0

        except Exception as e:
            print(f"GPU discovery failed: {e}")

    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of available GPUs."""
        return [g for g in self._gpus.values() if g.is_available]

    def select_gpu(
        self,
        strategy: DistributionStrategy = DistributionStrategy.LOAD_BALANCE,
        required_memory_mb: int = 0,
    ) -> Optional[int]:
        """Select best GPU based on strategy."""
        available = [
            g for g in self._gpus.values()
            if g.is_available and g.free_memory_mb >= required_memory_mb
        ]

        if not available:
            return None

        if strategy == DistributionStrategy.ROUND_ROBIN:
            # Simple round-robin
            min_queue = min(self._queues[g.device_id] for g in available)
            for g in available:
                if self._queues[g.device_id] == min_queue:
                    return g.device_id

        elif strategy == DistributionStrategy.LOAD_BALANCE:
            # Select GPU with most free memory and least queue
            def score(g):
                queue_penalty = self._queues[g.device_id] * 100
                return g.free_memory_mb - queue_penalty

            best = max(available, key=score)
            return best.device_id

        # Default: first available
        return available[0].device_id

    def acquire_gpu(self, device_id: int) -> bool:
        """Acquire GPU for processing."""
        if device_id not in self._locks:
            return False

        with self._locks[device_id]:
            self._queues[device_id] += 1
            return True

    def release_gpu(self, device_id: int):
        """Release GPU after processing."""
        if device_id in self._locks:
            with self._locks[device_id]:
                self._queues[device_id] = max(0, self._queues[device_id] - 1)

    def update_memory_info(self):
        """Update GPU memory information."""
        try:
            import torch

            for device_id in self._gpus:
                free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                self._gpus[device_id].free_memory_mb = free_mem // (1024 * 1024)

        except Exception:
            pass


class TileProcessor:
    """
    Process large images by tiling across GPUs.

    Splits image into tiles, processes on multiple GPUs,
    and reassembles the result.
    """

    def __init__(self, config: MultiGPUConfig, gpu_manager: GPUManager):
        self.config = config
        self.gpu_manager = gpu_manager

    def process(
        self,
        image: np.ndarray,
        process_fn: Callable[[np.ndarray, int], np.ndarray],
    ) -> np.ndarray:
        """
        Process image with tiling.

        Args:
            image: Input image
            process_fn: Function to process each tile (tile, device_id) -> result

        Returns:
            Processed full image
        """
        h, w = image.shape[:2]
        tile_h, tile_w = self.config.tile_size
        overlap = self.config.tile_overlap

        # Calculate tiles
        tiles = []
        for y in range(0, h, tile_h - overlap):
            for x in range(0, w, tile_w - overlap):
                y1, y2 = y, min(y + tile_h, h)
                x1, x2 = x, min(x + tile_w, w)
                tiles.append((y1, y2, x1, x2))

        # Process tiles in parallel
        results = [None] * len(tiles)
        gpus = self.gpu_manager.get_available_gpus()

        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = []

            for i, (y1, y2, x1, x2) in enumerate(tiles):
                tile = image[y1:y2, x1:x2].copy()
                device_id = gpus[i % len(gpus)].device_id

                future = executor.submit(
                    self._process_tile, tile, device_id, process_fn, i
                )
                futures.append((i, future))

            for i, future in futures:
                results[i] = future.result()

        # Reassemble
        return self._reassemble(results, tiles, image.shape)

    def _process_tile(
        self,
        tile: np.ndarray,
        device_id: int,
        process_fn: Callable,
        tile_idx: int,
    ) -> np.ndarray:
        """Process a single tile."""
        self.gpu_manager.acquire_gpu(device_id)
        try:
            return process_fn(tile, device_id)
        finally:
            self.gpu_manager.release_gpu(device_id)

    def _reassemble(
        self,
        tiles: List[np.ndarray],
        positions: List[Tuple[int, int, int, int]],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reassemble tiles into full image."""
        result = np.zeros(shape, dtype=tiles[0].dtype)
        weights = np.zeros(shape[:2], dtype=np.float32)

        overlap = self.config.tile_overlap

        for tile, (y1, y2, x1, x2) in zip(tiles, positions):
            # Create blending weights
            th, tw = tile.shape[:2]
            tile_weight = np.ones((th, tw), dtype=np.float32)

            # Feather edges for blending
            if overlap > 0:
                for i in range(overlap):
                    alpha = i / overlap
                    tile_weight[i, :] *= alpha
                    tile_weight[-(i+1), :] *= alpha
                    tile_weight[:, i] *= alpha
                    tile_weight[:, -(i+1)] *= alpha

            # Add to result
            if tile.ndim == 3:
                result[y1:y2, x1:x2] += tile * tile_weight[..., np.newaxis]
            else:
                result[y1:y2, x1:x2] += tile * tile_weight

            weights[y1:y2, x1:x2] += tile_weight

        # Normalize
        weights = np.maximum(weights, 1e-6)
        if result.ndim == 3:
            result = result / weights[..., np.newaxis]
        else:
            result = result / weights

        return result


class PipelineProcessor:
    """
    Pipeline processing across multiple GPUs.

    Different processing stages run on different GPUs
    for improved throughput.
    """

    def __init__(self, config: MultiGPUConfig, gpu_manager: GPUManager):
        self.config = config
        self.gpu_manager = gpu_manager
        self._stages: Dict[str, Tuple[Callable, int]] = {}  # stage -> (fn, device_id)

    def add_stage(
        self,
        name: str,
        process_fn: Callable[[np.ndarray], np.ndarray],
        device_id: Optional[int] = None,
    ):
        """Add a pipeline stage."""
        if device_id is None:
            device_id = self.gpu_manager.select_gpu()

        self._stages[name] = (process_fn, device_id)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image through pipeline."""
        result = image

        for stage_name in self.config.pipeline_stages:
            if stage_name in self._stages:
                process_fn, device_id = self._stages[stage_name]

                self.gpu_manager.acquire_gpu(device_id)
                try:
                    result = process_fn(result)
                finally:
                    self.gpu_manager.release_gpu(device_id)

        return result


class BatchProcessor:
    """
    Batch processing across multiple GPUs.

    Distributes batch items across GPUs for parallel processing.
    """

    def __init__(self, config: MultiGPUConfig, gpu_manager: GPUManager):
        self.config = config
        self.gpu_manager = gpu_manager

    def process_batch(
        self,
        batch: List[np.ndarray],
        process_fn: Callable[[np.ndarray, int], np.ndarray],
    ) -> List[np.ndarray]:
        """
        Process batch across GPUs.

        Args:
            batch: List of images
            process_fn: Function (image, device_id) -> result

        Returns:
            List of results
        """
        gpus = self.gpu_manager.get_available_gpus()
        num_gpus = len(gpus)

        if num_gpus == 0:
            # CPU fallback
            return [process_fn(img, -1) for img in batch]

        results = [None] * len(batch)

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []

            for i, image in enumerate(batch):
                device_id = gpus[i % num_gpus].device_id

                future = executor.submit(
                    self._process_item, image, device_id, process_fn, i
                )
                futures.append((i, future))

            for i, future in futures:
                results[i] = future.result()

        return results

    def _process_item(
        self,
        image: np.ndarray,
        device_id: int,
        process_fn: Callable,
        idx: int,
    ) -> np.ndarray:
        """Process single item."""
        self.gpu_manager.acquire_gpu(device_id)
        try:
            return process_fn(image, device_id)
        finally:
            self.gpu_manager.release_gpu(device_id)


class MultiGPUProcessor:
    """
    Main multi-GPU processing interface.

    Automatically selects the best strategy for
    the workload and available hardware.

    Example:
        >>> multi_gpu = MultiGPUProcessor(MultiGPUConfig(
        ...     strategy=DistributionStrategy.LOAD_BALANCE,
        ...     enable_tiling=True,
        ... ))
        >>>
        >>> # Process large image with tiling
        >>> result = multi_gpu.process_image(large_image, matting_fn)
        >>>
        >>> # Process batch
        >>> results = multi_gpu.process_batch(images, depth_fn)
    """

    def __init__(self, config: Optional[MultiGPUConfig] = None):
        self.config = config or MultiGPUConfig()
        self.gpu_manager = GPUManager(self.config.device_ids)

        self.tile_processor = TileProcessor(self.config, self.gpu_manager)
        self.batch_processor = BatchProcessor(self.config, self.gpu_manager)
        self.pipeline_processor = PipelineProcessor(self.config, self.gpu_manager)

    def process_image(
        self,
        image: np.ndarray,
        process_fn: Callable[[np.ndarray, int], np.ndarray],
        force_tiling: bool = False,
    ) -> np.ndarray:
        """
        Process single image with automatic strategy selection.

        Args:
            image: Input image
            process_fn: Processing function
            force_tiling: Force tiling even for small images

        Returns:
            Processed image
        """
        h, w = image.shape[:2]
        tile_h, tile_w = self.config.tile_size

        # Decide whether to tile
        should_tile = force_tiling or (
            self.config.enable_tiling and
            (h > tile_h * 1.5 or w > tile_w * 1.5)
        )

        if should_tile and len(self.gpu_manager.get_available_gpus()) > 1:
            return self.tile_processor.process(image, process_fn)
        else:
            # Single GPU processing
            device_id = self.gpu_manager.select_gpu(self.config.strategy)
            if device_id is None:
                device_id = 0

            self.gpu_manager.acquire_gpu(device_id)
            try:
                return process_fn(image, device_id)
            finally:
                self.gpu_manager.release_gpu(device_id)

    def process_batch(
        self,
        batch: List[np.ndarray],
        process_fn: Callable[[np.ndarray, int], np.ndarray],
    ) -> List[np.ndarray]:
        """Process batch of images."""
        return self.batch_processor.process_batch(batch, process_fn)

    def process_video(
        self,
        frames: List[np.ndarray],
        process_fn: Callable[[np.ndarray, int], np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """
        Process video frames with progress tracking.

        Args:
            frames: Video frames
            process_fn: Processing function
            progress_callback: Called with (current, total)

        Returns:
            Processed frames
        """
        results = []

        # Process in batches based on number of GPUs
        num_gpus = len(self.gpu_manager.get_available_gpus())
        batch_size = max(1, num_gpus * self.config.batch_size_per_gpu)

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = self.process_batch(batch, process_fn)
            results.extend(batch_results)

            if progress_callback:
                progress_callback(min(i + batch_size, len(frames)), len(frames))

        return results

    def get_gpu_info(self) -> List[GPUInfo]:
        """Get information about available GPUs."""
        self.gpu_manager.update_memory_info()
        return self.gpu_manager.get_available_gpus()


def process_on_best_gpu(
    process_fn: Callable[[int], Any],
) -> Any:
    """
    Execute function on best available GPU.

    Args:
        process_fn: Function that takes device_id as argument

    Returns:
        Result from process_fn
    """
    manager = GPUManager()
    device_id = manager.select_gpu()

    if device_id is None:
        device_id = 0

    manager.acquire_gpu(device_id)
    try:
        return process_fn(device_id)
    finally:
        manager.release_gpu(device_id)
