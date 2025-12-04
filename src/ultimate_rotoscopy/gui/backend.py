#!/usr/bin/env python3
"""
Processing Backend for Ultimate Rotoscopy GUI
==============================================

Connects the GUI to all AI models and processing pipelines.
Handles async processing via QThread workers.
"""

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from PySide6.QtCore import QObject, QThread, Signal, Slot


class ProcessingStage(Enum):
    """Current processing stage."""
    IDLE = "idle"
    LOADING_MODELS = "loading_models"
    SEGMENTATION = "segmentation"
    MATTING = "matting"
    DEPTH = "depth"
    COMPOSITING = "compositing"


@dataclass
class ProcessingRequest:
    """Request for processing."""
    stage: ProcessingStage
    image: np.ndarray
    background: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    points: Optional[List[Tuple[int, int]]] = None
    point_labels: Optional[List[int]] = None
    box: Optional[Tuple[int, int, int, int]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result from processing."""
    stage: ProcessingStage
    success: bool
    mask: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    composite: Optional[np.ndarray] = None
    foreground: Optional[np.ndarray] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessingWorker(QObject):
    """
    Worker for async processing.

    Runs in a separate thread to keep GUI responsive.
    """

    # Signals
    started = Signal(str)           # Processing stage name
    progress = Signal(int, str)     # Progress percentage, message
    finished = Signal(object)       # ProcessingResult
    error = Signal(str)             # Error message
    model_loaded = Signal(str)      # Model name

    def __init__(self, parent=None):
        super().__init__(parent)

        # Models (loaded on demand)
        self._sam = None
        self._depth_model = None
        self._matanyone = None
        self._gvm = None
        self._compositor = None
        self._vitmatte = None

        # State
        self._device = "cuda"
        self._models_loaded = False
        self._current_image = None
        self._current_mask = None

    def set_device(self, device: str):
        """Set compute device."""
        self._device = device

    @Slot(object)
    def process(self, request: ProcessingRequest):
        """Process a request."""
        try:
            self.started.emit(request.stage.value)

            if request.stage == ProcessingStage.LOADING_MODELS:
                result = self._load_models(request.parameters)
            elif request.stage == ProcessingStage.SEGMENTATION:
                result = self._run_segmentation(request)
            elif request.stage == ProcessingStage.MATTING:
                result = self._run_matting(request)
            elif request.stage == ProcessingStage.DEPTH:
                result = self._run_depth(request)
            elif request.stage == ProcessingStage.COMPOSITING:
                result = self._run_compositing(request)
            else:
                result = ProcessingResult(
                    stage=request.stage,
                    success=False,
                    error=f"Unknown stage: {request.stage}"
                )

            self.finished.emit(result)

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            self.finished.emit(ProcessingResult(
                stage=request.stage,
                success=False,
                error=error_msg
            ))

    def _load_models(self, params: Dict[str, Any]) -> ProcessingResult:
        """Load AI models."""
        models_to_load = params.get("models", ["sam", "depth", "matting"])

        total = len(models_to_load)

        for i, model_name in enumerate(models_to_load):
            self.progress.emit(int((i / total) * 100), f"Loading {model_name}...")

            try:
                if model_name == "sam" and self._sam is None:
                    self._load_sam(params)

                elif model_name == "depth" and self._depth_model is None:
                    self._load_depth(params)

                elif model_name == "matting" and self._matanyone is None:
                    self._load_matting(params)

                elif model_name == "vitmatte" and self._vitmatte is None:
                    self._load_vitmatte(params)

                self.model_loaded.emit(model_name)

            except Exception as e:
                return ProcessingResult(
                    stage=ProcessingStage.LOADING_MODELS,
                    success=False,
                    error=f"Failed to load {model_name}: {str(e)}"
                )

        self.progress.emit(100, "Models loaded")
        self._models_loaded = True

        return ProcessingResult(
            stage=ProcessingStage.LOADING_MODELS,
            success=True,
            metadata={"models_loaded": models_to_load}
        )

    def _load_sam(self, params: Dict[str, Any]):
        """Load SAM model."""
        try:
            from ultimate_rotoscopy.models.sam3 import SAM3Segmentor, SAM3Config, SAM3ModelSize

            model_size_str = params.get("sam_model", "SAM2.1 Large")

            size_map = {
                "SAM2.1 Large": SAM3ModelSize.LARGE,
                "SAM2.1 Base+": SAM3ModelSize.BASE,
                "SAM2.1 Small": SAM3ModelSize.SMALL,
            }

            model_size = size_map.get(model_size_str, SAM3ModelSize.LARGE)

            config = SAM3Config(
                model_size=model_size,
                device=self._device,
            )

            self._sam = SAM3Segmentor(config)
            self._sam.load()

        except ImportError:
            # Fallback to HuggingFace transformers
            self._load_sam_transformers(params)

    def _load_sam_transformers(self, params: Dict[str, Any]):
        """Load SAM via transformers."""
        from transformers import SamModel, SamProcessor
        import torch

        model_id = "facebook/sam-vit-large"

        self._sam_processor = SamProcessor.from_pretrained(model_id)
        self._sam_model = SamModel.from_pretrained(model_id).to(self._device)
        self._sam_model.eval()

        # Create a wrapper
        class SAMWrapper:
            def __init__(self, model, processor, device):
                self.model = model
                self.processor = processor
                self.device = device
                self._image_embedding = None

            def load(self):
                pass

            def set_image(self, image):
                from PIL import Image as PILImage
                if isinstance(image, np.ndarray):
                    pil_image = PILImage.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
                else:
                    pil_image = image
                self._pil_image = pil_image

            def segment(self, image, prompt, refine_edges=True):
                import torch
                from PIL import Image as PILImage

                if isinstance(image, np.ndarray):
                    if image.dtype == np.float32 or image.dtype == np.float64:
                        image = (image * 255).astype(np.uint8)
                    pil_image = PILImage.fromarray(image)
                else:
                    pil_image = image

                # Prepare inputs
                if prompt.points is not None:
                    input_points = [prompt.points.tolist()]
                    input_labels = [prompt.point_labels.tolist()] if prompt.point_labels is not None else None
                    inputs = self.processor(
                        pil_image,
                        input_points=input_points,
                        input_labels=input_labels,
                        return_tensors="pt"
                    ).to(self.device)
                elif prompt.boxes is not None:
                    input_boxes = [prompt.boxes.tolist()]
                    inputs = self.processor(
                        pil_image,
                        input_boxes=input_boxes,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    inputs = self.processor(pil_image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )

                masks = masks[0].numpy()
                scores = outputs.iou_scores[0].cpu().numpy()

                # Return result object
                class Result:
                    pass
                result = Result()
                result.masks = masks[0] if masks.ndim == 4 else masks
                result.scores = scores
                result.logits = outputs.pred_masks[0].cpu().numpy()
                result.refined_mask = result.masks

                return result

        self._sam = SAMWrapper(self._sam_model, self._sam_processor, self._device)

    def _load_depth(self, params: Dict[str, Any]):
        """Load depth estimation model."""
        try:
            from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3, DepthConfig, DepthModelSize

            model_size_str = params.get("depth_model", "Depth Anything V2 Large")

            size_map = {
                "Depth Anything V2 Large": DepthModelSize.LARGE,
                "Depth Anything V2 Base": DepthModelSize.BASE,
                "Depth Anything V2 Small": DepthModelSize.SMALL,
            }

            model_size = size_map.get(model_size_str, DepthModelSize.LARGE)

            config = DepthConfig(
                model_size=model_size,
                device=self._device,
            )

            self._depth_model = DepthAnythingV3(config)
            self._depth_model.load()

        except ImportError:
            self._load_depth_transformers(params)

    def _load_depth_transformers(self, params: Dict[str, Any]):
        """Load depth via transformers."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        model_id = "depth-anything/Depth-Anything-V2-Large-hf"

        self._depth_processor = AutoImageProcessor.from_pretrained(model_id)
        self._depth_model_raw = AutoModelForDepthEstimation.from_pretrained(model_id).to(self._device)
        self._depth_model_raw.eval()

        class DepthWrapper:
            def __init__(self, model, processor, device):
                self.model = model
                self.processor = processor
                self.device = device

            def load(self):
                pass

            def estimate_depth(self, image, generate_normals=True, generate_point_cloud=False):
                import torch
                from PIL import Image as PILImage
                import cv2

                if isinstance(image, np.ndarray):
                    if image.dtype == np.float32 or image.dtype == np.float64:
                        image = (image * 255).astype(np.uint8)
                    pil_image = PILImage.fromarray(image)
                else:
                    pil_image = image

                original_size = pil_image.size[::-1]  # (H, W)

                inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predicted_depth = outputs.predicted_depth

                depth = predicted_depth.squeeze().cpu().numpy()

                # Resize to original
                depth = cv2.resize(depth, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

                # Normalize
                depth_min, depth_max = depth.min(), depth.max()
                depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)

                # Compute normals if requested
                normals = None
                if generate_normals:
                    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

                    normals = np.zeros((*depth.shape, 3), dtype=np.float32)
                    normals[..., 0] = -grad_x
                    normals[..., 1] = -grad_y
                    normals[..., 2] = 1.0

                    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
                    normals = normals / (norm + 1e-8)

                class Result:
                    pass
                result = Result()
                result.depth_map = depth
                result.depth_normalized = depth_normalized
                result.normals = normals
                result.confidence = np.ones_like(depth)
                result.metadata = {}

                return result

        self._depth_model = DepthWrapper(self._depth_model_raw, self._depth_processor, self._device)

    def _load_matting(self, params: Dict[str, Any]):
        """Load matting model."""
        try:
            from ultimate_rotoscopy.models.matanyone import MatAnyone, MatAnyoneConfig

            config = MatAnyoneConfig(device=self._device)
            self._matanyone = MatAnyone(config)
            self._matanyone.load()

        except Exception as e:
            print(f"MatAnyone loading failed: {e}, using fallback")
            self._matanyone = self._create_matting_fallback()

    def _load_vitmatte(self, params: Dict[str, Any]):
        """Load ViTMatte model."""
        try:
            from ultimate_rotoscopy.models.vitmatte import ViTMatteModel, ViTMatteConfig

            config = ViTMatteConfig(device=self._device)
            self._vitmatte = ViTMatteModel(config)
            self._vitmatte.load()

        except Exception as e:
            print(f"ViTMatte loading failed: {e}")
            self._vitmatte = None

    def _create_matting_fallback(self):
        """Create fallback matting using morphological operations."""
        class FallbackMatting:
            def __init__(self):
                self.initialized = False

            def load(self):
                pass

            def initialize(self, image, mask):
                import cv2

                self.initialized = True

                # Refine mask using morphological operations
                mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

                # Create trimap
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                dilated = cv2.dilate(mask_uint8, kernel)
                eroded = cv2.erode(mask_uint8, kernel)

                # Unknown region
                unknown = dilated - eroded

                # Simple alpha estimation
                alpha = mask_uint8.astype(np.float32) / 255.0

                # Smooth alpha in unknown region
                alpha_smooth = cv2.GaussianBlur(alpha, (15, 15), 0)
                unknown_mask = (unknown > 0).astype(np.float32)
                alpha = alpha * (1 - unknown_mask) + alpha_smooth * unknown_mask

                class Result:
                    pass
                result = Result()
                result.alpha = alpha
                result.confidence = 1 - unknown_mask
                result.foreground = image * np.stack([alpha]*3, axis=-1) if image.ndim == 3 else image * alpha
                result.core_mask = (eroded > 0).astype(np.float32)
                result.boundary_mask = unknown_mask
                result.refined_details = np.zeros_like(alpha)
                result.metadata = {"fallback": True}

                return result

            def process_frame(self, frame, frame_idx=None, refine_mask=None):
                return self.initialize(frame, refine_mask if refine_mask is not None else np.ones(frame.shape[:2]))

            def reset(self):
                self.initialized = False

        return FallbackMatting()

    def _run_segmentation(self, request: ProcessingRequest) -> ProcessingResult:
        """Run SAM segmentation."""
        self.progress.emit(10, "Preparing image...")

        # Ensure SAM is loaded
        if self._sam is None:
            self._load_sam(request.parameters)

        self.progress.emit(30, "Running segmentation...")

        # Prepare prompts
        from dataclasses import dataclass as dc

        @dc
        class Prompt:
            prompt_type: str = "point"
            points: Optional[np.ndarray] = None
            point_labels: Optional[np.ndarray] = None
            boxes: Optional[np.ndarray] = None
            mask_input: Optional[np.ndarray] = None
            multimask_output: bool = True

        prompt = Prompt()

        if request.points:
            prompt.points = np.array(request.points)
            prompt.point_labels = np.array(request.point_labels) if request.point_labels else np.ones(len(request.points))
            prompt.prompt_type = "point"

        if request.box:
            prompt.boxes = np.array([request.box])
            prompt.prompt_type = "box"

        prompt.multimask_output = request.parameters.get("multi_mask", True)

        self.progress.emit(50, "Computing masks...")

        # Run segmentation
        result = self._sam.segment(request.image, prompt, refine_edges=True)

        self.progress.emit(90, "Finalizing...")

        # Get best mask
        if hasattr(result, 'masks') and result.masks is not None:
            if result.masks.ndim == 3:
                # Multiple masks - select best one
                best_idx = np.argmax(result.scores) if hasattr(result, 'scores') else 0
                mask = result.masks[best_idx]
            else:
                mask = result.masks
        else:
            mask = result.refined_mask if hasattr(result, 'refined_mask') else None

        self.progress.emit(100, "Segmentation complete")

        # Store for later use
        self._current_mask = mask
        self._current_image = request.image

        return ProcessingResult(
            stage=ProcessingStage.SEGMENTATION,
            success=True,
            mask=mask,
            metadata={
                "scores": result.scores.tolist() if hasattr(result, 'scores') else [],
                "num_masks": len(result.masks) if hasattr(result, 'masks') and result.masks.ndim == 3 else 1,
            }
        )

    def _run_matting(self, request: ProcessingRequest) -> ProcessingResult:
        """Run matting refinement."""
        self.progress.emit(10, "Preparing matting...")

        # Use existing mask if available
        mask = request.mask if request.mask is not None else self._current_mask

        if mask is None:
            return ProcessingResult(
                stage=ProcessingStage.MATTING,
                success=False,
                error="No mask available. Run segmentation first."
            )

        # Ensure matting model is loaded
        if self._matanyone is None:
            self._load_matting(request.parameters)

        self.progress.emit(30, "Initializing matting model...")

        # Get matting model based on selection
        matte_model_name = request.parameters.get("matte_model", "MatAnyone")

        self.progress.emit(50, f"Running {matte_model_name}...")

        # Run matting
        if matte_model_name == "ViTMatte" and self._vitmatte is not None:
            result = self._vitmatte.process(request.image, mask)
            alpha = result.alpha
            confidence = getattr(result, 'confidence', np.ones_like(alpha))
        else:
            # Use MatAnyone
            result = self._matanyone.initialize(request.image, mask)
            alpha = result.alpha
            confidence = result.confidence

        self.progress.emit(90, "Extracting foreground...")

        # Extract foreground
        if request.image.ndim == 3:
            alpha_3ch = np.stack([alpha] * 3, axis=-1)
            foreground = request.image * alpha_3ch
        else:
            foreground = request.image * alpha

        self.progress.emit(100, "Matting complete")

        return ProcessingResult(
            stage=ProcessingStage.MATTING,
            success=True,
            alpha=alpha,
            foreground=foreground,
            metadata={
                "model": matte_model_name,
                "mean_confidence": float(confidence.mean()) if confidence is not None else 1.0,
            }
        )

    def _run_depth(self, request: ProcessingRequest) -> ProcessingResult:
        """Run depth estimation."""
        self.progress.emit(10, "Preparing depth estimation...")

        # Ensure depth model is loaded
        if self._depth_model is None:
            self._load_depth(request.parameters)

        self.progress.emit(30, "Running depth model...")

        # Get output type
        output_type = request.parameters.get("depth_output", "Depth Map")
        generate_normals = output_type in ["Normal Map", "Depth Map"]

        self.progress.emit(50, "Estimating depth...")

        # Run depth estimation
        result = self._depth_model.estimate_depth(
            request.image,
            generate_normals=generate_normals,
            generate_point_cloud=False,
        )

        self.progress.emit(90, "Processing output...")

        # Get requested output
        depth = result.depth_normalized
        normals = result.normals if hasattr(result, 'normals') else None

        # Generate AO if requested
        if output_type == "Ambient Occlusion":
            depth = self._compute_ssao(depth, normals)

        self.progress.emit(100, "Depth estimation complete")

        return ProcessingResult(
            stage=ProcessingStage.DEPTH,
            success=True,
            depth=depth,
            normals=normals,
            metadata={
                "output_type": output_type,
                "depth_range": [float(result.depth_map.min()), float(result.depth_map.max())],
            }
        )

    def _compute_ssao(self, depth: np.ndarray, normals: Optional[np.ndarray]) -> np.ndarray:
        """Compute screen-space ambient occlusion."""
        import cv2

        # Simple SSAO approximation
        kernel_size = 15

        # Blur depth
        depth_blur = cv2.GaussianBlur(depth, (kernel_size, kernel_size), 0)

        # Compute local variance
        depth_sq = depth ** 2
        depth_sq_blur = cv2.GaussianBlur(depth_sq, (kernel_size, kernel_size), 0)

        variance = depth_sq_blur - depth_blur ** 2

        # AO is higher where variance is low (flat areas) and depth difference is high
        ao = 1.0 - np.clip(variance * 10, 0, 1)

        # Also darken based on depth gradient
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        ao = ao * (1.0 - np.clip(grad_mag * 2, 0, 0.5))

        return ao.astype(np.float32)

    def _run_compositing(self, request: ProcessingRequest) -> ProcessingResult:
        """Run compositing pipeline."""
        self.progress.emit(10, "Preparing compositing...")

        if request.background is None:
            return ProcessingResult(
                stage=ProcessingStage.COMPOSITING,
                success=False,
                error="No background provided for compositing."
            )

        alpha = request.mask
        if alpha is None:
            return ProcessingResult(
                stage=ProcessingStage.COMPOSITING,
                success=False,
                error="No alpha/mask available. Run matting first."
            )

        self.progress.emit(20, "Setting up compositor...")

        # Create compositor with parameters
        try:
            from ultimate_rotoscopy.compositing.compositor import UltimateCompositor, CompositorConfig
            from ultimate_rotoscopy.compositing.despill import DespillAlgorithm, SpillChannel
            from ultimate_rotoscopy.compositing.light_wrap import WrapMode
            from ultimate_rotoscopy.compositing.harmonization import HarmonizationMethod

            params = request.parameters

            # Map string parameters to enums
            despill_algo_map = {
                "Adaptive": DespillAlgorithm.ADAPTIVE,
                "Average": DespillAlgorithm.AVERAGE,
                "Maximum": DespillAlgorithm.MAXIMUM,
                "Double Average": DespillAlgorithm.DOUBLE_AVERAGE,
            }

            spill_channel_map = {
                "Green": SpillChannel.GREEN,
                "Blue": SpillChannel.BLUE,
            }

            harmonize_map = {
                "Adaptive": HarmonizationMethod.ADAPTIVE,
                "LAB Transfer": HarmonizationMethod.LAB_TRANSFER,
                "Reinhard": HarmonizationMethod.REINHARD,
                "Histogram": HarmonizationMethod.HISTOGRAM,
            }

            config = CompositorConfig(
                enable_despill=params.get("enable_despill", True),
                despill_algorithm=despill_algo_map.get(params.get("despill_algo", "Adaptive"), DespillAlgorithm.ADAPTIVE),
                despill_channel=spill_channel_map.get(params.get("despill_channel", "Green"), SpillChannel.GREEN),
                despill_strength=params.get("despill_strength", 0.8),
                enable_edge_operations=True,
                edge_erode_amount=max(0, params.get("edge_erode", 0)),
                edge_dilate_amount=max(0, -params.get("edge_erode", 0)),
                pixel_spread_amount=5,
                enable_light_wrap=params.get("enable_light_wrap", False),
                light_wrap_intensity=params.get("light_wrap_intensity", 0.5),
                light_wrap_width=params.get("light_wrap_width", 20),
                enable_harmonization=params.get("enable_harmonize", False),
                harmonization_method=harmonize_map.get(params.get("harmonize_method", "Adaptive"), HarmonizationMethod.ADAPTIVE),
            )

            compositor = UltimateCompositor(config)

        except ImportError:
            # Fallback to simple compositing
            compositor = None

        self.progress.emit(40, "Processing foreground...")

        foreground = request.image
        background = request.background

        # Normalize
        if foreground.dtype == np.uint8:
            foreground = foreground.astype(np.float32) / 255.0
        if background.dtype == np.uint8:
            background = background.astype(np.float32) / 255.0
        if alpha.max() > 1:
            alpha = alpha.astype(np.float32) / 255.0

        self.progress.emit(60, "Running compositor...")

        if compositor is not None:
            result = compositor.composite(foreground, background, alpha)
            composite = result.composite
            processed_fg = result.foreground
        else:
            # Simple alpha over
            alpha_3ch = np.stack([alpha] * 3, axis=-1)
            composite = foreground * alpha_3ch + background * (1 - alpha_3ch)
            processed_fg = foreground

        self.progress.emit(100, "Compositing complete")

        return ProcessingResult(
            stage=ProcessingStage.COMPOSITING,
            success=True,
            composite=np.clip(composite, 0, 1),
            foreground=processed_fg,
            alpha=alpha,
            metadata={
                "despill_enabled": request.parameters.get("enable_despill", True),
                "light_wrap_enabled": request.parameters.get("enable_light_wrap", False),
            }
        )

    def unload_models(self):
        """Unload all models to free memory."""
        if self._sam is not None:
            if hasattr(self._sam, 'unload'):
                self._sam.unload()
            self._sam = None

        if self._depth_model is not None:
            if hasattr(self._depth_model, 'unload'):
                self._depth_model.unload()
            self._depth_model = None

        if self._matanyone is not None:
            if hasattr(self._matanyone, 'unload'):
                self._matanyone.unload()
            self._matanyone = None

        if self._vitmatte is not None:
            if hasattr(self._vitmatte, 'unload'):
                self._vitmatte.unload()
            self._vitmatte = None

        self._models_loaded = False

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class ProcessingBackend(QObject):
    """
    Main processing backend that manages workers and threads.
    """

    # Signals for GUI updates
    processing_started = Signal(str)
    processing_progress = Signal(int, str)
    processing_finished = Signal(object)
    processing_error = Signal(str)
    model_loaded = Signal(str)
    gpu_memory_updated = Signal(float, float)  # used, total

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create worker thread
        self._thread = QThread()
        self._worker = ProcessingWorker()
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._worker.started.connect(self.processing_started.emit)
        self._worker.progress.connect(self.processing_progress.emit)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self.processing_error.emit)
        self._worker.model_loaded.connect(self.model_loaded.emit)

        # Start thread
        self._thread.start()

        # State
        self._current_request = None
        self._results_cache = {}

    def _on_finished(self, result: ProcessingResult):
        """Handle processing finished."""
        # Cache result
        self._results_cache[result.stage] = result

        # Update GPU memory
        self._update_gpu_memory()

        # Emit signal
        self.processing_finished.emit(result)

    def _update_gpu_memory(self):
        """Update GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.gpu_memory_updated.emit(used, total)
        except ImportError:
            pass

    def load_models(self, params: Dict[str, Any] = None):
        """Load AI models."""
        if params is None:
            params = {"models": ["sam", "depth", "matting"]}

        request = ProcessingRequest(
            stage=ProcessingStage.LOADING_MODELS,
            image=np.zeros((1, 1, 3)),  # Dummy
            parameters=params,
        )

        self._process(request)

    def run_segmentation(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]] = None,
        point_labels: List[int] = None,
        box: Tuple[int, int, int, int] = None,
        params: Dict[str, Any] = None,
    ):
        """Run segmentation on image."""
        request = ProcessingRequest(
            stage=ProcessingStage.SEGMENTATION,
            image=image,
            points=points,
            point_labels=point_labels,
            box=box,
            parameters=params or {},
        )

        self._process(request)

    def run_matting(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        params: Dict[str, Any] = None,
    ):
        """Run matting refinement."""
        request = ProcessingRequest(
            stage=ProcessingStage.MATTING,
            image=image,
            mask=mask,
            parameters=params or {},
        )

        self._process(request)

    def run_depth(
        self,
        image: np.ndarray,
        params: Dict[str, Any] = None,
    ):
        """Run depth estimation."""
        request = ProcessingRequest(
            stage=ProcessingStage.DEPTH,
            image=image,
            parameters=params or {},
        )

        self._process(request)

    def run_compositing(
        self,
        image: np.ndarray,
        background: np.ndarray,
        alpha: np.ndarray,
        params: Dict[str, Any] = None,
    ):
        """Run compositing pipeline."""
        request = ProcessingRequest(
            stage=ProcessingStage.COMPOSITING,
            image=image,
            background=background,
            mask=alpha,
            parameters=params or {},
        )

        self._process(request)

    def _process(self, request: ProcessingRequest):
        """Submit request for processing."""
        self._current_request = request

        # Use QMetaObject to invoke in worker thread
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self._worker,
            "process",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(object, request),
        )

    def get_cached_result(self, stage: ProcessingStage) -> Optional[ProcessingResult]:
        """Get cached result for a stage."""
        return self._results_cache.get(stage)

    def clear_cache(self):
        """Clear results cache."""
        self._results_cache.clear()

    def unload_models(self):
        """Unload all models."""
        self._worker.unload_models()
        self.clear_cache()

    def shutdown(self):
        """Shutdown backend."""
        self.unload_models()
        self._thread.quit()
        self._thread.wait()
