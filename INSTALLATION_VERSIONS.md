# Installation Version Requirements - Rocky Linux + RTX 4090

## Critical Version Requirements

### SAM3 (Segment Anything Model 3)
- **Python**: >= 3.12
- **PyTorch**: >= 2.7.0
- **CUDA**: >= 12.6
- **transformers**: >= 4.50.0 (recommended)
- **NVIDIA Driver**: >= 550.x (for CUDA 12.6 support)

### Depth Anything 3
- **Python**: >= 3.10
- **PyTorch**: >= 2.0.0
- **CUDA**: >= 11.8
- **transformers**: >= 4.45.0

### RTX 4090 Compatibility
- **Architecture**: Ada Lovelace (Compute Capability 8.9)
- **Optimal CUDA**: 12.6+ (latest features and performance)
- **Minimum Driver**: 550.x for CUDA 12.6
- **Recommended Driver**: 560.x+ (latest stable)

## Installation Strategy

The script installs:
- **Python 3.12** (satisfies both SAM3 and DA3)
- **PyTorch 2.7+ with CUDA 12.6** (satisfies both, optimal for RTX 4090)
- **transformers >= 4.50.0** (SAM3 compatible)

This configuration is **backward compatible**:
- ✅ Supports SAM3 (requires highest versions)
- ✅ Supports Depth Anything 3 (works with PyTorch 2.7+)
- ✅ Optimal performance on RTX 4090

## Fallback Strategy

If SAM3 or DA3 are not yet released:
- **SAM3 → SAM2.1**: `facebook/sam2.1-hiera-large` (HuggingFace)
- **SAM3 → SAM1**: `facebook/sam-vit-large` (HuggingFace)
- **DA3 → DA2**: `depth-anything/Depth-Anything-V2-Large-hf` (HuggingFace)

The code has **automatic fallback** detection in:
- `src/ultimate_rotoscopy/models/sam3.py:102-140`
- `src/ultimate_rotoscopy/models/depth_anything.py:95-133`

## Package Installation Order

1. **System packages** (CUDA, cuDNN, dev tools)
2. **PyTorch** (MUST install before transformers)
3. **transformers** (will auto-detect PyTorch CUDA)
4. **Model packages** (SAM3, DA3 from GitHub)
5. **Other dependencies** (can be in any order)

**Critical**: PyTorch MUST be installed with explicit CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Do NOT use `pip install torch` without index-url, as it may install CPU-only version.

## Version Verification

After installation, verify with:

```bash
python3 << 'EOF'
import torch
import transformers

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Transformers: {transformers.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check versions
assert sys.version_info >= (3, 12), "Python 3.12+ required"
assert torch.__version__ >= "2.7", "PyTorch 2.7+ required"
assert torch.cuda.is_available(), "CUDA not available"
assert transformers.__version__ >= "4.50", "transformers 4.50+ recommended"

print("\n✓ All version requirements satisfied for SAM3")
EOF
```

## Troubleshooting

### Issue: "CUDA not available"
- Check: `nvidia-smi` shows GPU
- Check: Driver version >= 550.x
- Reinstall PyTorch with correct CUDA index

### Issue: "transformers version too old"
```bash
pip install --upgrade "transformers>=4.50.0"
```

### Issue: "SAM3 not found"
- SAM3 may not be released yet
- Application will auto-fallback to SAM2.1
- Check `src/ultimate_rotoscopy/models/sam3.py:102-140`

### Issue: "RuntimeError: CUDA out of memory"
RTX 4090 has 24GB VRAM, but models are large:
- SAM3 Large: ~3GB VRAM
- Depth Anything 3 Giant: ~5GB VRAM
- ViTMatte: ~2GB VRAM

**Solutions:**
- Use smaller model sizes (Base instead of Large/Giant)
- Process smaller images (resize to 1024px max)
- Enable gradient checkpointing (if training)
- Close other GPU processes

### Issue: "Rust build failed"
```bash
# Ensure Rust is in PATH
source "$HOME/.cargo/env"

# Install build dependencies
sudo dnf install -y gcc-c++ cmake

# Build manually
cd /path/to/ultimate-rotoscopy
maturin develop --release
```

## Performance Expectations (RTX 4090)

With CUDA 12.6 + PyTorch 2.7:
- **SAM3 inference**: ~50-100ms per frame (1920x1080)
- **Depth Anything 3**: ~30-60ms per frame
- **Full pipeline**: ~200-500ms per frame (depends on settings)

Batch processing recommended for video sequences.

## Rocky Linux Specific Notes

### Python 3.12 Installation
Rocky Linux 9 default is Python 3.9. Script installs Python 3.12 from EPEL.

### CUDA Repository
Uses official NVIDIA CUDA repository for RHEL 9:
```bash
https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/
```

### Driver Installation
If driver not installed:
```bash
sudo dnf install -y nvidia-driver nvidia-driver-cuda nvidia-driver-libs
```

Or download from NVIDIA: https://www.nvidia.com/Download/index.aspx

## Summary

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Python | 3.10 | 3.12 | SAM3 needs 3.12+ |
| PyTorch | 2.0 | 2.7+ | SAM3 needs 2.7+ |
| CUDA | 11.8 | 12.6 | RTX 4090 optimal with 12.6 |
| transformers | 4.45 | 4.50+ | SAM3 needs 4.50+ |
| NVIDIA Driver | 535 | 550+ | CUDA 12.6 needs 550+ |
| VRAM | 12GB | 24GB | RTX 4090 has 24GB ✓ |

**Conclusion**: RTX 4090 is perfectly suited for this application. All requirements can be satisfied.
