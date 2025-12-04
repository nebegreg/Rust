"""
Base Model Interface for Ultimate Rotoscopy
============================================

Provides abstract base class for all AI model integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class DeviceType(Enum):
    """Supported compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class PrecisionType(Enum):
    """Model precision types."""
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    model_path: Optional[Path] = None
    model_name: str = ""
    device: DeviceType = DeviceType.CUDA
    precision: PrecisionType = PrecisionType.FP16
    compile_model: bool = True
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "rotoscopy")
    max_batch_size: int = 4
    use_flash_attention: bool = True
    use_sdpa: bool = True  # Scaled Dot Product Attention
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class InferenceResult:
    """Container for model inference results."""
    output: np.ndarray
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    device_used: str = "unknown"


class BaseModel(ABC):
    """
    Abstract base class for all AI models in the rotoscopy pipeline.

    Provides common functionality for:
    - Model loading and caching
    - Device management (CPU/CUDA/MPS)
    - Precision handling (FP32/FP16/BF16)
    - Batch processing
    - Memory optimization
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype()
        self._is_loaded = False

    def _resolve_device(self) -> torch.device:
        """Resolve the best available device."""
        if self.config.device == DeviceType.CUDA:
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        elif self.config.device == DeviceType.MPS:
            if torch.backends.mps.is_available():
                return torch.device("mps")
            print("MPS not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cpu")

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve tensor dtype from precision config."""
        dtype_map = {
            PrecisionType.FP32: torch.float32,
            PrecisionType.FP16: torch.float16,
            PrecisionType.BF16: torch.bfloat16,
            PrecisionType.INT8: torch.int8,
        }
        dtype = dtype_map.get(self.config.precision, torch.float32)

        # Check BF16 support
        if dtype == torch.bfloat16:
            if not (self.device.type == "cuda" and torch.cuda.is_bf16_supported()):
                print("BF16 not supported, falling back to FP16")
                dtype = torch.float16

        return dtype

    @abstractmethod
    def load(self) -> None:
        """Load the model weights and prepare for inference."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass

    @abstractmethod
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        **kwargs
    ) -> InferenceResult:
        """Run inference on a single image."""
        pass

    @abstractmethod
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
        **kwargs
    ) -> List[InferenceResult]:
        """Run inference on a batch of images."""
        pass

    def preprocess(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        """Convert input to tensor and normalize."""
        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(image, np.ndarray):
            # Handle different channel orders
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]  # Remove alpha

            # Normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0

            # HWC to CHW
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        # Add batch dimension if needed
        if image.ndim == 3:
            image = image.unsqueeze(0)

        return image.to(device=self.device, dtype=self.dtype)

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Convert model output to numpy array."""
        if output.device.type != "cpu":
            output = output.cpu()

        if output.dtype in (torch.float16, torch.bfloat16):
            output = output.float()

        return output.numpy()

    @torch.inference_mode()
    def optimize_for_inference(self) -> None:
        """Apply inference optimizations."""
        if self.model is None:
            return

        self.model.eval()

        # Compile model with torch.compile for PyTorch 2.0+
        if self.config.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                print(f"Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}")

        # Enable cuDNN autotuning for consistent input sizes
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        result = {"cpu_mb": 0.0}

        if self.device.type == "cuda":
            result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            result["gpu_cached_mb"] = torch.cuda.memory_reserved() / 1024**2

        return result

    def clear_cache(self) -> None:
        """Clear GPU cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
