
### Features Implementation

#### SAM3 Segmentation
- [x] Text-based prompts (open vocabulary)
- [x] Interactive GUI with PySide6
- [x] Multi-mask selection
- [x] Video tracking with memory banks
- [x] Export to PNG/JSON

#### Depth Anything V3
- [x] Metric depth estimation
- [x] Normal map generation
- [x] Camera intrinsics estimation
- [x] 3D point cloud export (PLY, OBJ, XYZ)
- [x] Multiple colormaps (turbo, viridis, plasma, etc.)
- [x] Batch processing support
- [x] Professional GUI

#### Alpha Matting
- [x] ViTMatte integration (transformer-based)
- [x] MatAnyone integration (video matting)
- [x] Automatic trimap from SAM3 masks
- [x] Layer decomposition (core/edge/hair)
- [x] Edge refinement with guided filtering
- [x] Temporal consistency for video
- [x] Foreground extraction

#### Professional VFX Integration
- [x] Multi-layer OpenEXR export
- [x] AOV management (alpha, depth, normals)
- [x] Flame-compatible clip XML
- [x] Batch setup templates
- [x] ACEScg color space support

#### GUI Applications
- [x] SAM3 GUI (`sam3_gui.py`) - Segmentation interface
- [x] Ultimate Roto GUI (`ultimate_roto_gui.py`) - Complete rotoscopy interface

---

## Ultimate Rotoscopy Pipeline

### Components

| Component | Purpose | Status |
|-----------|---------|--------|
| SAM3 | Initial segmentation | Integrated |
| ViTMatte | Image alpha matting | Integrated |
| MatAnyone | Video alpha matting | Integrated |
| DepthAnything3 | Depth estimation | Integrated |

### Pipeline Flow

```
Input Image/Video
       |
       v
  [SAM3 Segmentation]
  - Text prompt / Point prompt / Box prompt
       |
       v
  [Trimap Generation]
  - Adaptive erosion/dilation
  - Hair region detection
       |
       v
  [Alpha Matting]
  - ViTMatte (images)
  - MatAnyone (video)
       |
       v
  [Edge Refinement]
  - Guided filtering
  - Blur handling
       |
       v
  [Layer Decomposition]
  - Core mask
  - Edge mask
  - Hair mask
       |
       v
  [Optional: Depth Estimation]
  - DepthAnything3
       |
       v
  [Export]
  - Alpha PNG/EXR
  - Foreground RGBA
  - Depth map
  - Layer masks
```

### Usage

```bash
# Image with text prompt
python ultimate_roto.py image photo.jpg --text "person" -o results/

# Image with depth estimation
python ultimate_roto.py image photo.jpg --text "person" --depth -o results/

# Video processing
python ultimate_roto.py video clip.mp4 --text "person" -o results/

# GUI mode
python ultimate_roto_gui.py
```

### Trimap Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--erosion` | 15 | Erosion kernel for foreground |
| `--dilation` | 30 | Dilation kernel for unknown region |
| `--hair-refinement` | True | Enable hair/fine detail detection |

### Output Files

- `*_alpha.png` - Final alpha matte
- `*_foreground.png` - RGBA foreground with alpha
- `*_trimap.png` - Generated trimap
- `*_depth.png` - Depth map (colored)
- `*_depth_raw.png` - Depth map (16-bit raw)
- `*_layer_core.png` - Core/solid foreground mask
- `*_layer_edge.png` - Edge/transition mask
- `*_layer_hair.png` - Hair/fine detail mask
- `*_metadata.json` - Processing metadata


