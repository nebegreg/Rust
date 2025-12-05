"""
SEA-RAFT / RAFT Optical Flow
============================

State-of-the-art optical flow estimation for mask propagation and tracking.

Models:
- RAFT (Original): Recurrent All-Pairs Field Transforms for Optical Flow
- SEA-RAFT (ECCV 2024 Best Paper Candidate): Simple, Efficient, Accurate RAFT

Key Features:
- Dense per-pixel motion estimation
- Iterative refinement with recurrent units
- Multi-scale correlation volumes
- Uncertainty estimation (SEA-RAFT)

References:
- RAFT: https://github.com/princeton-vl/RAFT
- SEA-RAFT: https://github.com/princeton-vl/SEA-RAFT
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class FlowModelType(Enum):
    """Available optical flow models."""
    RAFT = "raft"                    # Original RAFT
    RAFT_SMALL = "raft_small"        # Smaller/faster RAFT
    SEA_RAFT = "sea_raft"            # SEA-RAFT (ECCV 2024)
    SEA_RAFT_STEREO = "sea_raft_stereo"  # Stereo version


class FlowDirection(Enum):
    """Flow computation direction."""
    FORWARD = "forward"              # Frame t -> t+1
    BACKWARD = "backward"            # Frame t+1 -> t
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class FlowConfig:
    """Optical flow configuration."""
    model_type: FlowModelType = FlowModelType.SEA_RAFT
    direction: FlowDirection = FlowDirection.FORWARD

    # RAFT parameters
    num_iterations: int = 12         # Refinement iterations (more = better)
    corr_levels: int = 4             # Correlation pyramid levels
    corr_radius: int = 4             # Correlation lookup radius

    # SEA-RAFT specific
    use_mixture_of_laplace: bool = True  # New loss from SEA-RAFT
    rigid_motion_pretrain: bool = True   # Improved generalization
    initial_flow: bool = True            # Direct initial flow regression

    # Processing
    max_resolution: int = 1024       # Max dimension for processing
    use_fp16: bool = True
    device: str = "cuda"

    # Post-processing
    filter_outliers: bool = True
    outlier_threshold: float = 50.0  # Max flow magnitude
    temporal_consistency: bool = True


@dataclass
class FlowResult:
    """Result from optical flow computation."""
    flow: np.ndarray                 # HxWx2 flow field (u, v)
    magnitude: np.ndarray            # HxW flow magnitude
    angle: np.ndarray                # HxW flow direction
    uncertainty: Optional[np.ndarray] = None  # HxW uncertainty (SEA-RAFT)
    occlusion_mask: Optional[np.ndarray] = None  # HxW occlusion detection
    backward_flow: Optional[np.ndarray] = None  # For bidirectional
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorrelationVolume:
    """
    Multi-scale correlation volume for RAFT.

    Computes all-pairs correlation between feature maps at multiple scales.
    """

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4):
        self.num_levels = num_levels
        self.corr_pyramid = []

        # Compute correlation at original scale
        corr = self._compute_correlation(fmap1, fmap2)

        # Build correlation pyramid
        batch, h1, w1, _, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, 1, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def _compute_correlation(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dot-product correlation."""
        batch, dim, h, w = fmap1.shape
        fmap1 = fmap1.view(batch, dim, h * w)
        fmap2 = fmap2.view(batch, dim, h * w)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, h, w, 1, h, w)
        return corr / torch.sqrt(torch.tensor(dim).float())

    def __call__(
        self,
        coords: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        """Lookup correlation values around coordinates."""
        out_pyramid = []

        for i, corr in enumerate(self.corr_pyramid):
            dx = torch.linspace(-radius, radius, 2 * radius + 1)
            dy = torch.linspace(-radius, radius, 2 * radius + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(-1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*radius+1, 2*radius+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr_sampled = F.grid_sample(
                corr,
                coords_lvl,
                align_corners=True,
                mode='bilinear',
                padding_mode='zeros'
            )
            out_pyramid.append(corr_sampled.view(-1, (2*radius+1)**2))

        return torch.cat(out_pyramid, dim=-1)


class BasicMotionEncoder(nn.Module):
    """Encode flow and correlation features."""

    def __init__(self, corr_levels: int = 4, corr_radius: int = 4):
        super().__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1)**2

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    """Separable Convolutional GRU for iterative updates."""

    def __init__(self, hidden_dim: int = 128, input_dim: int = 256 + 128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # Vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class FlowHead(nn.Module):
    """Predict flow delta."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class BasicEncoder(nn.Module):
    """Feature encoder for RAFT."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(64)

        self.conv2 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(96)

        self.conv3 = nn.Conv2d(96, 128, 3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(128)

        self.conv4 = nn.Conv2d(128, output_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        x = self.relu(self.conv4(x))
        return x


class ContextEncoder(nn.Module):
    """Context encoder for hidden state initialization."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256)

        self.conv4 = nn.Conv2d(256, output_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        x = self.relu(self.conv4(x))
        return x


class RAFTModel(nn.Module):
    """
    RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.

    Architecture:
    1. Feature encoder extracts features from both frames
    2. Correlation volume computed for all pairs
    3. Iterative updates refine flow estimate
    """

    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config

        # Feature extraction
        self.fnet = BasicEncoder(output_dim=256)
        self.cnet = ContextEncoder(output_dim=256)

        # Update block
        self.update_block = nn.ModuleDict({
            'encoder': BasicMotionEncoder(
                config.corr_levels,
                config.corr_radius
            ),
            'gru': SepConvGRU(
                hidden_dim=128,
                input_dim=256 + 128
            ),
            'flow_head': FlowHead(input_dim=128),
        })

        # SEA-RAFT: Initial flow predictor
        if config.initial_flow:
            self.initial_flow_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 2, 3, padding=1),
            )

        # SEA-RAFT: Uncertainty head
        if config.use_mixture_of_laplace:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Softplus(),
            )

    def initialize_flow(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize flow coordinates."""
        N, C, H, W = img.shape
        coords0 = self._coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = self._coords_grid(N, H // 8, W // 8).to(img.device)
        return coords0, coords1

    def _coords_grid(self, batch: int, h: int, w: int) -> torch.Tensor:
        """Create coordinate grid."""
        coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int = 12,
        test_mode: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute optical flow from image1 to image2.

        Args:
            image1: First frame [B, 3, H, W]
            image2: Second frame [B, 3, H, W]
            iters: Number of refinement iterations
            test_mode: If True, only return final flow

        Returns:
            flow: Optical flow [B, 2, H, W]
            uncertainty: Optional uncertainty map [B, 1, H, W]
        """
        # Feature extraction
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)

        # Context for hidden state
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # Correlation volume
        corr_fn = CorrelationVolume(fmap1, fmap2, num_levels=self.config.corr_levels)

        # Initialize flow
        coords0, coords1 = self.initialize_flow(image1)

        # SEA-RAFT: Predict initial flow
        if self.config.initial_flow and hasattr(self, 'initial_flow_head'):
            init_flow = self.initial_flow_head(fmap1)
            coords1 = coords1 + init_flow

        # Iterative updates
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1, self.config.corr_radius)

            flow = coords1 - coords0
            motion_features = self.update_block['encoder'](flow, corr)
            inp_cat = torch.cat([inp, motion_features], dim=1)

            net = self.update_block['gru'](net, inp_cat)
            delta_flow = self.update_block['flow_head'](net)

            coords1 = coords1 + delta_flow

            if not test_mode:
                flow_predictions.append(coords1 - coords0)

        flow = coords1 - coords0

        # Upsample to original resolution
        flow_up = self._upsample_flow(flow, image1.shape[-2:])

        # SEA-RAFT: Uncertainty
        uncertainty = None
        if self.config.use_mixture_of_laplace and hasattr(self, 'uncertainty_head'):
            uncertainty = self.uncertainty_head(net)
            uncertainty = F.interpolate(
                uncertainty,
                size=image1.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        if test_mode:
            return flow_up, uncertainty
        else:
            return flow_predictions, uncertainty

    def _upsample_flow(
        self,
        flow: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Upsample flow to target resolution."""
        _, _, H, W = flow.shape
        new_H, new_W = target_size

        flow = F.interpolate(
            flow,
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=True
        )

        # Scale flow values
        flow[:, 0] *= new_W / W
        flow[:, 1] *= new_H / H

        return flow


class RAFTOpticalFlow:
    """
    RAFT/SEA-RAFT optical flow estimator.

    Provides dense per-pixel motion estimation between consecutive frames,
    essential for mask propagation in rotoscopy.

    Example:
        >>> flow_model = RAFTOpticalFlow(FlowConfig(
        ...     model_type=FlowModelType.SEA_RAFT,
        ...     num_iterations=12,
        ... ))
        >>> flow_model.load()
        >>>
        >>> # Compute flow
        >>> result = flow_model.compute(frame1, frame2)
        >>>
        >>> # Warp mask using flow
        >>> warped_mask = flow_model.warp(mask, result.flow)
    """

    def __init__(self, config: Optional[FlowConfig] = None):
        self.config = config or FlowConfig()
        self.model = None
        self._loaded = False
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        """Load RAFT model."""
        if self._loaded:
            return

        print(f"Loading {self.config.model_type.value} optical flow model...")

        try:
            # Try to load pretrained model
            self._load_pretrained()
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            print("Using fallback OpenCV optical flow")

        self._loaded = True

    def _load_pretrained(self) -> None:
        """Load pretrained RAFT weights."""
        # Try official RAFT
        try:
            if self.config.model_type == FlowModelType.SEA_RAFT:
                from sea_raft import SEA_RAFT
                self.model = SEA_RAFT().to(self.device)
                print("Loaded SEA-RAFT from official package")
                return
        except ImportError:
            pass

        # Try torchvision RAFT
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device)
            self.model.eval()
            print("Loaded RAFT from torchvision")
            return
        except ImportError:
            pass

        # Build custom model
        self.model = RAFTModel(self.config).to(self.device)
        self.model.eval()
        print("Using custom RAFT implementation")

    @torch.inference_mode()
    def compute(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> FlowResult:
        """
        Compute optical flow between two frames.

        Args:
            frame1: First frame (RGB, HxWx3)
            frame2: Second frame (RGB, HxWx3)

        Returns:
            FlowResult with flow field and metadata
        """
        start_time = time.time()

        # Resize if needed
        h, w = frame1.shape[:2]
        scale = 1.0
        if max(h, w) > self.config.max_resolution:
            scale = self.config.max_resolution / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Ensure divisible by 8
            new_h = (new_h // 8) * 8
            new_w = (new_w // 8) * 8
            import cv2
            frame1_resized = cv2.resize(frame1, (new_w, new_h))
            frame2_resized = cv2.resize(frame2, (new_w, new_h))
        else:
            frame1_resized = frame1
            frame2_resized = frame2

        if self.model is not None:
            flow, uncertainty = self._compute_with_model(frame1_resized, frame2_resized)
        else:
            flow, uncertainty = self._compute_fallback(frame1_resized, frame2_resized)

        # Scale flow back to original resolution
        if scale != 1.0:
            import cv2
            flow = cv2.resize(flow, (w, h))
            flow = flow / scale
            if uncertainty is not None:
                uncertainty = cv2.resize(uncertainty, (w, h))

        # Compute magnitude and angle
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])

        # Filter outliers
        if self.config.filter_outliers:
            outlier_mask = magnitude > self.config.outlier_threshold
            flow[outlier_mask] = 0
            magnitude[outlier_mask] = 0

        # Detect occlusions using forward-backward consistency
        occlusion_mask = None
        backward_flow = None

        if self.config.direction == FlowDirection.BIDIRECTIONAL:
            if self.model is not None:
                backward_flow, _ = self._compute_with_model(frame2_resized, frame1_resized)
            else:
                backward_flow, _ = self._compute_fallback(frame2_resized, frame1_resized)

            if scale != 1.0:
                import cv2
                backward_flow = cv2.resize(backward_flow, (w, h))
                backward_flow = backward_flow / scale

            occlusion_mask = self._detect_occlusions(flow, backward_flow)

        processing_time = (time.time() - start_time) * 1000

        return FlowResult(
            flow=flow,
            magnitude=magnitude,
            angle=angle,
            uncertainty=uncertainty,
            occlusion_mask=occlusion_mask,
            backward_flow=backward_flow,
            metadata={
                "processing_time_ms": processing_time,
                "model_type": self.config.model_type.value,
                "num_iterations": self.config.num_iterations,
            }
        )

    def _compute_with_model(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute flow using neural network."""
        # Prepare tensors
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0

        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)

        if self.config.use_fp16 and self.device.type == "cuda":
            img1 = img1.half()
            img2 = img2.half()

        # Run model
        if hasattr(self.model, 'forward'):
            output = self.model(img1, img2, iters=self.config.num_iterations)
            if isinstance(output, tuple):
                flow, uncertainty = output
            else:
                flow = output
                uncertainty = None
        else:
            # Torchvision RAFT
            flow = self.model(img1, img2)[-1]
            uncertainty = None

        # Convert to numpy
        flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        if uncertainty is not None:
            uncertainty = uncertainty.squeeze().cpu().numpy()

        return flow, uncertainty

    def _compute_fallback(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> Tuple[np.ndarray, None]:
        """Fallback to OpenCV optical flow."""
        import cv2

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )

        return flow, None

    def _detect_occlusions(
        self,
        forward_flow: np.ndarray,
        backward_flow: np.ndarray,
        threshold: float = 1.0,
    ) -> np.ndarray:
        """Detect occlusions using forward-backward consistency."""
        h, w = forward_flow.shape[:2]

        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Warp coordinates with forward flow
        x_warped = x + forward_flow[..., 0]
        y_warped = y + forward_flow[..., 1]

        # Clip to valid range
        x_warped = np.clip(x_warped, 0, w - 1).astype(np.int32)
        y_warped = np.clip(y_warped, 0, h - 1).astype(np.int32)

        # Get backward flow at warped positions
        bw_flow_at_fw = backward_flow[y_warped, x_warped]

        # Forward-backward consistency error
        fb_error = forward_flow + bw_flow_at_fw
        fb_error_mag = np.sqrt(fb_error[..., 0]**2 + fb_error[..., 1]**2)

        # Threshold for occlusion
        occlusion_mask = fb_error_mag > threshold

        return occlusion_mask

    def warp(
        self,
        image: np.ndarray,
        flow: np.ndarray,
        mode: str = "bilinear",
    ) -> np.ndarray:
        """
        Warp image using optical flow.

        Args:
            image: Image to warp (HxW or HxWxC)
            flow: Optical flow field (HxWx2)
            mode: Interpolation mode

        Returns:
            Warped image
        """
        import cv2

        h, w = flow.shape[:2]

        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply flow
        x_new = (x + flow[..., 0]).astype(np.float32)
        y_new = (y + flow[..., 1]).astype(np.float32)

        # Warp using remap
        if image.dtype == np.float32 or image.dtype == np.float64:
            interpolation = cv2.INTER_LINEAR if mode == "bilinear" else cv2.INTER_NEAREST
            warped = cv2.remap(
                image.astype(np.float32),
                x_new, y_new,
                interpolation=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            warped = cv2.remap(
                image,
                x_new, y_new,
                interpolation=cv2.INTER_LINEAR if mode == "bilinear" else cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        return warped

    def warp_mask(
        self,
        mask: np.ndarray,
        flow: np.ndarray,
    ) -> np.ndarray:
        """Warp a mask using optical flow."""
        # Use bilinear for soft masks, nearest for binary
        is_binary = np.allclose(mask, mask.astype(bool))
        mode = "nearest" if is_binary else "bilinear"

        warped = self.warp(mask.astype(np.float32), flow, mode)

        return np.clip(warped, 0, 1)

    def propagate_mask(
        self,
        frames: List[np.ndarray],
        initial_mask: np.ndarray,
        start_frame: int = 0,
        direction: str = "forward",
    ) -> List[np.ndarray]:
        """
        Propagate a mask through a sequence of frames.

        Args:
            frames: List of frames
            initial_mask: Mask on start_frame
            start_frame: Frame index with initial mask
            direction: "forward", "backward", or "both"

        Returns:
            List of masks for each frame
        """
        n_frames = len(frames)
        masks = [None] * n_frames
        masks[start_frame] = initial_mask

        # Forward propagation
        if direction in ["forward", "both"]:
            for i in range(start_frame, n_frames - 1):
                result = self.compute(frames[i], frames[i + 1])
                masks[i + 1] = self.warp_mask(masks[i], result.flow)

        # Backward propagation
        if direction in ["backward", "both"]:
            for i in range(start_frame, 0, -1):
                result = self.compute(frames[i], frames[i - 1])
                masks[i - 1] = self.warp_mask(masks[i], result.flow)

        return masks


# Convenience function
def compute_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    model_type: str = "sea_raft",
) -> FlowResult:
    """Quick function to compute optical flow."""
    config = FlowConfig(
        model_type=FlowModelType(model_type),
        num_iterations=12,
    )

    flow_model = RAFTOpticalFlow(config)
    flow_model.load()

    return flow_model.compute(frame1, frame2)
