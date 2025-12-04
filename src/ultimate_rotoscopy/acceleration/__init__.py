"""
Acceleration Module for Ultimate Rotoscopy
==========================================

Hardware acceleration and performance optimization:
- ONNX/TensorRT inference acceleration
- Multi-GPU distributed processing
- Intelligent caching system
"""

from ultimate_rotoscopy.acceleration.onnx_tensorrt import (
    AcceleratedInference,
    AccelerationConfig,
    AccelerationBackend,
    Precision,
    InferenceResult,
    ONNXExporter,
    ONNXInference,
    TensorRTEngine,
)

from ultimate_rotoscopy.acceleration.multi_gpu import (
    MultiGPUProcessor,
    MultiGPUConfig,
    DistributionStrategy,
    GPUManager,
    GPUInfo,
    TileProcessor,
    BatchProcessor,
    PipelineProcessor,
    process_on_best_gpu,
)

from ultimate_rotoscopy.acceleration.caching import (
    IntelligentCache,
    CacheConfig,
    CacheLevel,
    CachePolicy,
    MemoryCache,
    DiskCache,
    TemporalCache,
    cached,
)

__all__ = [
    # ONNX/TensorRT
    "AcceleratedInference",
    "AccelerationConfig",
    "AccelerationBackend",
    "Precision",
    "InferenceResult",
    "ONNXExporter",
    "ONNXInference",
    "TensorRTEngine",
    # Multi-GPU
    "MultiGPUProcessor",
    "MultiGPUConfig",
    "DistributionStrategy",
    "GPUManager",
    "GPUInfo",
    "TileProcessor",
    "BatchProcessor",
    "PipelineProcessor",
    "process_on_best_gpu",
    # Caching
    "IntelligentCache",
    "CacheConfig",
    "CacheLevel",
    "CachePolicy",
    "MemoryCache",
    "DiskCache",
    "TemporalCache",
    "cached",
]
