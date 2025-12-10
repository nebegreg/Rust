# Installation Guide - Ultimate Rotoscopy

Complete installation guide for the Ultimate Rotoscopy application.

## 📋 Prerequisites

### System Requirements

**Minimum:**
- OS: Linux (Rocky Linux 9 recommended), macOS, Windows 10/11
- CPU: 4+ cores
- RAM: 8GB
- Storage: 20GB free space
- Python: 3.10+

**Recommended:**
- OS: Rocky Linux 9 / Ubuntu 22.04+ / Windows 11
- CPU: 8+ cores (Intel i7/AMD Ryzen 7+)
- RAM: 16GB+
- GPU: NVIDIA RTX 3060+ (8GB+ VRAM)
- CUDA: 12.1+ (12.6 recommended)
- Storage: 50GB free space
- Python: 3.12

### Software Dependencies

1. **Python 3.10-3.12**
   ```bash
   python3 --version  # Should be >= 3.10
   ```

2. **NVIDIA CUDA Toolkit** (for GPU acceleration)
   ```bash
   nvcc --version  # Should be >= 12.1
   ```
   Download: https://developer.nvidia.com/cuda-downloads

3. **Rust** (optional, for maximum performance)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

4. **Git**
   ```bash
   git --version
   ```

## 🚀 Quick Installation (Linux/macOS)

### Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/ultimate-rotoscopy/ultimate-rotoscopy.git
cd ultimate-rotoscopy

# Make install script executable
chmod +x install.sh

# Run installation (auto-detects CUDA)
./install.sh

# Or specify CUDA version
./install.sh --cuda-version 12.6

# With development tools
./install.sh --cuda-version 12.6 --dev
```

The script will:
1. ✅ Verify system requirements
2. ✅ Create virtual environment
3. ✅ Install PyTorch with correct CUDA version
4. ✅ Install all dependencies in correct order
5. ✅ Install SAM3 and Depth Anything V3 from source
6. ✅ Build Rust core (if available)
7. ✅ Verify installation

### Manual Installation

If you prefer manual installation:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA 12.6
pip install torch>=2.7.0 torchvision>=0.18.0 torchaudio>=2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 4. Install the package
pip install -e .

# 5. Install model packages (from source)
pip install git+https://github.com/facebookresearch/sam3.git
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

# 6. Build Rust core (optional)
pip install maturin
maturin develop --release
```

## 🪟 Windows Installation

### Option 1: WSL2 (Recommended)

1. **Install WSL2** with Ubuntu:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

2. **Follow Linux instructions** above in WSL2

### Option 2: Native Windows

```powershell
# 1. Install Python 3.12 from python.org
# 2. Install CUDA Toolkit 12.6 from NVIDIA
# 3. Install Git for Windows
# 4. Install Visual Studio Build Tools

# Open PowerShell as Administrator

# Clone repository
git clone https://github.com/ultimate-rotoscopy/ultimate-rotoscopy.git
cd ultimate-rotoscopy

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
pip install torch>=2.7.0 torchvision>=0.18.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install package
pip install -e .

# Install models
pip install git+https://github.com/facebookresearch/sam3.git
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

## 🔧 CUDA Version Selection

Choose the appropriate CUDA version based on your system:

| CUDA Version | PyTorch Index URL | Recommended GPU |
|--------------|-------------------|-----------------|
| **12.6** | `cu126` | RTX 4090, RTX 4080, H100 |
| **12.4** | `cu124` | RTX 4070, RTX 4060 |
| **12.1** | `cu121` | RTX 3090, RTX 3080, A100 |
| **11.8** | `cu118` | RTX 3070, RTX 3060 |
| **CPU Only** | `cpu` | No GPU |

Check your CUDA version:
```bash
nvcc --version
nvidia-smi
```

## 📦 Dependency Installation Order

**Critical:** Dependencies must be installed in this order:

1. **PyTorch** (first, before all other ML packages)
2. **NumPy, SciPy** (scientific computing base)
3. **Transformers, Timm** (depends on PyTorch)
4. **OpenCV, Pillow** (computer vision)
5. **OpenEXR, Imath** (professional formats)
6. **SAM3, Depth Anything V3** (models from source)
7. **Ultimate Rotoscopy** (main package)

The install script handles this automatically.

## 🧪 Verify Installation

```bash
# Activate environment
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

# Run tests
rotoscopy test

# Check imports
python3 <<EOF
import torch
import transformers
import ultimate_rotoscopy
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print("✅ Installation verified!")
EOF
```

Expected output:
```
PyTorch: 2.7.0+cu126
CUDA: True
✅ Installation verified!
```

## 📥 Download Model Checkpoints

After installation, download pre-trained model checkpoints:

```bash
python scripts/download_models.py
```

This will download:
- SAM3 checkpoints (~2GB)
- Depth Anything V3 checkpoints (~1.3GB for Giant model)
- Matte Anything checkpoints (~300MB)

Models are cached in `~/.cache/huggingface/` and `~/.cache/torch/`.

## 🎯 Quick Start

### Test Basic Functionality

```bash
# Process a single image
rotoscopy process test_image.jpg -p "100,200" -o output/

# Launch GUI
rotoscopy-gui

# Get info
rotoscopy info
```

### Python API Test

```python
from ultimate_rotoscopy import RotoscopyEngine
import numpy as np

engine = RotoscopyEngine()
result = engine.process(
    "test.jpg",
    points=np.array([[100, 200]]),
    generate_depth=True,
)
print(f"Alpha shape: {result.alpha.shape}")
print(f"Depth shape: {result.depth_map.shape}")
```

## 🐛 Troubleshooting

### PyTorch CUDA Mismatch

**Problem:** `CUDA not available` even with GPU

**Solution:**
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### OpenEXR Installation Failed

**Problem:** `OpenEXR installation failed`

**Solution (Ubuntu/Debian):**
```bash
sudo apt-get install libopenexr-dev
pip install OpenEXR
```

**Solution (Rocky Linux):**
```bash
sudo dnf install OpenEXR-devel
pip install OpenEXR
```

### SAM3 Import Error

**Problem:** `ModuleNotFoundError: No module named 'sam3'`

**Solution:**
```bash
# Clone and install manually
git clone https://github.com/facebookresearch/sam3.git external/sam3
cd external/sam3
pip install -e .
cd ../..
```

Or use the fallback:
```bash
pip install segment-anything
```

### Rust Build Failed

**Problem:** `maturin develop` fails

**Solution:**
- This is optional, the Python version will work
- Ensure Rust is installed: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Or skip with: `./install.sh` (Rust is auto-detected)

### Out of Memory (OOM)

**Problem:** CUDA out of memory errors

**Solution:**
```bash
# Use lower quality mode
rotoscopy process image.jpg --quality fast

# Or in Python
config.processing_mode = ProcessingMode.FAST
```

### Import cv2 Failed

**Problem:** `ImportError: libGL.so.1: cannot open shared object file`

**Solution (Ubuntu):**
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

## 🔄 Updating

```bash
cd ultimate-rotoscopy
git pull origin main
pip install -e . --upgrade
```

## 🗑️ Uninstallation

```bash
# Remove virtual environment
rm -rf venv/

# Remove cached models (optional)
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/

# Remove repository
cd ..
rm -rf ultimate-rotoscopy/
```

## 📚 Additional Resources

- **README.md** - Project overview and features
- **ROADMAP_STATUS.md** - Implementation status
- **requirements.txt** - Python dependencies
- **pyproject.toml** - Package configuration
- **Cargo.toml** - Rust dependencies

## 🆘 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review GitHub Issues: https://github.com/ultimate-rotoscopy/ultimate-rotoscopy/issues
3. Check CUDA compatibility: https://pytorch.org/get-started/locally/
4. Verify Python version: `python3 --version`

## ✅ Installation Checklist

- [ ] Python 3.10+ installed
- [ ] CUDA 12.1+ installed (for GPU)
- [ ] Git installed
- [ ] Virtual environment created
- [ ] PyTorch installed with CUDA
- [ ] All dependencies installed
- [ ] SAM3 installed
- [ ] Depth Anything V3 installed
- [ ] Ultimate Rotoscopy installed
- [ ] Installation verified with `rotoscopy test`
- [ ] Model checkpoints downloaded

**Installation Time:** 15-30 minutes (depending on internet speed)

**Required Download:** ~5-10GB (PyTorch + models + dependencies)

---

**Need help?** Open an issue or check the documentation.
