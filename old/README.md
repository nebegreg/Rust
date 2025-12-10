# Ultimate Rotoscopy

**AI-Powered Professional VFX Tool for Rotoscopy, Matting, and Depth Estimation**

Ultimate Rotoscopy is a comprehensive application leveraging cutting-edge AI models for professional VFX workflows. Designed for artists working with tools like Autodesk Flame, Nuke, and Fusion.

## Features

### AI Models

- **SAM3 (Segment Anything Model 3)** - State-of-the-art object segmentation with interactive prompting
- **Depth Anything V3** - Monocular depth estimation with metric depth support
- **Matte Anything** - Professional alpha matting for hair, edges, and fine details

### Capabilities

- **Segmentation**
  - Interactive point/box prompting
  - Automatic multi-object detection
  - High-resolution mask refinement
  - Edge-aware segmentation

- **Depth Estimation**
  - Z-depth for compositing
  - Normal map generation
  - 3D point cloud export
  - Camera-relative depth

- **Alpha Matting**
  - Hair and fine detail preservation
  - Motion blur handling
  - Spill suppression
  - Color decontamination
  - Temporal consistency

- **Export**
  - Multi-layer OpenEXR with AOVs
  - Flame-compatible output
  - Nuke/Fusion support
  - PLY/OBJ point clouds

## Installation

```bash
# Clone the repository
git clone https://github.com/ultimate-rotoscopy/ultimate-rotoscopy.git
cd ultimate-rotoscopy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install Python package
pip install -e .

# Build Rust core (optional, for maximum performance)
pip install maturin
maturin develop --release
```

### Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (recommended)
- 8GB+ VRAM (16GB recommended for maximum quality)

## Quick Start

### Command Line

```bash
# Process a single image with point prompts
rotoscopy process image.jpg -p "100,200;150,250" -o output/

# Process with bounding box
rotoscopy process image.jpg -b "50,50,300,400"

# Process video sequence
rotoscopy sequence frames/ --pattern "*.png" --flame

# Generate 3D point cloud
rotoscopy pointcloud image.jpg -o output.ply --with-colors

# Run tests
rotoscopy test
```

### Python API

```python
from ultimate_rotoscopy import RotoscopyEngine, UnifiedPipeline
from ultimate_rotoscopy.models.sam3 import SegmentationPrompt, PromptType
import numpy as np

# Initialize engine
engine = RotoscopyEngine()

# Process with point prompt
prompt = SegmentationPrompt(
    prompt_type=PromptType.POINT,
    points=np.array([[100, 200], [150, 250]]),
    point_labels=np.array([1, 1])  # Foreground points
)

result = engine.process(
    "input.jpg",
    prompt=prompt,
    generate_depth=True,
    generate_normals=True,
    generate_matte=True,
)

# Access results
alpha = result.alpha           # Alpha matte
depth = result.depth_map       # Depth map
normals = result.normals       # Normal map
foreground = result.foreground # Clean foreground

# Get AOV package for compositing
aovs = result.aov_package
```

### GUI Interface

```bash
# Launch web-based GUI
rotoscopy-gui
# Opens at http://localhost:7860
```

## Pipeline

```
Input Image/Video
       |
       v
+------+------+
|   SAM3      |  -> Segmentation masks
+------+------+
       |
       v
+------+------+
| Depth       |  -> Depth map, normals, 3D points
| Anything V3 |
+------+------+
       |
       v
+------+------+
| Matte       |  -> Alpha matte, clean foreground
| Anything    |
+------+------+
       |
       v
+------+------+
| AOV Manager |  -> Multi-layer EXR
+------+------+
       |
       v
Output (Flame/Nuke/Fusion compatible)
```

## AOV Outputs

| AOV Name | Description | Channels |
|----------|-------------|----------|
| `alpha` / `A` | Alpha matte | 1 |
| `depth.Z` | Z-depth | 1 |
| `normal.X/Y/Z` | Camera-space normals | 3 |
| `foreground.R/G/B` | Clean foreground | 3 |
| `edge_mask` | Edge regions | 1 |
| `hair_mask` | Hair/fine detail | 1 |
| `motion_mask` | Motion blur regions | 1 |
| `disparity` | Inverse depth | 1 |

## Configuration

Create a `config.yaml` file:

```yaml
engine:
  processing_mode: quality
  device: cuda

sam3:
  model_size: large
  edge_refinement: true

depth:
  model_size: large
  generate_normals: true

matte:
  quality: high
  edge_mode: hair
  handle_motion_blur: true

output:
  format: exr
  compression: zip
```

## Flame Integration

Ultimate Rotoscopy generates Flame-compatible output:

1. **Multi-layer EXR** with standard AOV naming
2. **Clip XML** for direct import
3. **Batch setup** templates
4. **Action node** presets

```bash
# Export for Flame
rotoscopy sequence frames/ --flame --clip-name "my_shot"
```

## Performance

| Quality Mode | Processing Time (1080p) | VRAM Usage |
|--------------|------------------------|------------|
| Fast | ~200ms | ~4GB |
| Balanced | ~500ms | ~6GB |
| Quality | ~1000ms | ~8GB |
| Maximum | ~2000ms | ~12GB |

## Architecture

```
ultimate-rotoscopy/
├── src/
│   ├── ultimate_rotoscopy/     # Python package
│   │   ├── models/             # AI model integrations
│   │   │   ├── sam3.py         # SAM3 segmentation
│   │   │   ├── depth_anything.py
│   │   │   └── matte_anything.py
│   │   ├── core/               # Engine and session
│   │   ├── pipeline/           # Processing pipelines
│   │   ├── export/             # EXR, AOV, Flame export
│   │   ├── cli.py              # Command-line interface
│   │   └── gui.py              # Gradio web interface
│   └── lib.rs                  # Rust performance core
├── configs/                    # Configuration files
├── Cargo.toml                  # Rust dependencies
└── pyproject.toml              # Python dependencies
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Meta AI - Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
- [ViTMatte](https://github.com/hustvl/ViTMatte)
- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)

---

**Ultimate Rotoscopy** - Professional AI-powered VFX tools for modern workflows.
