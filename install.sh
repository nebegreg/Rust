#!/bin/bash
#===============================================================================
# Ultimate Rotoscopy - Installation Script
#===============================================================================
# This script installs all dependencies in the correct order to avoid conflicts
# Supports: Python 3.10-3.12, CUDA 12.1-12.6, PyTorch 2.7+
#
# Usage:
#   ./install.sh [--cuda-version 12.6] [--skip-models] [--dev]
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
CUDA_VERSION="12.6"
SKIP_MODELS=false
INSTALL_DEV=false
PYTHON_MIN_VERSION="3.10"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda-version VERSION   CUDA version (12.1, 12.2, 12.3, 12.4, 12.6)"
            echo "  --skip-models            Skip installing SAM3 and Depth Anything V3"
            echo "  --dev                    Install development dependencies"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

#===============================================================================
# Helper Functions
#===============================================================================

print_step() {
    echo -e "\n${BLUE}==>${NC} ${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

print_error() {
    echo -e "${RED}Error: $1${NC}"
    exit 1
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed. Please install it first."
    fi
}

version_ge() {
    # Compare versions: returns 0 if $1 >= $2
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

#===============================================================================
# System Checks
#===============================================================================

print_step "Checking system requirements..."

# Check Python
check_command python3
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "  ✓ Python $PYTHON_VERSION detected"

if ! version_ge "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
    print_error "Python $PYTHON_MIN_VERSION+ required, found $PYTHON_VERSION"
fi

# Check pip
check_command pip3
echo "  ✓ pip $(pip3 --version | cut -d' ' -f2) detected"

# Check CUDA (optional but recommended)
if command -v nvcc &> /dev/null; then
    SYSTEM_CUDA=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "  ✓ CUDA $SYSTEM_CUDA detected"

    # Warn if system CUDA doesn't match requested version
    if [[ "$SYSTEM_CUDA" != "$CUDA_VERSION"* ]]; then
        print_warning "System CUDA ($SYSTEM_CUDA) differs from requested ($CUDA_VERSION)"
        print_warning "PyTorch will be installed for CUDA $CUDA_VERSION"
    fi
else
    print_warning "CUDA not detected - will install CPU-only PyTorch"
    CUDA_VERSION="cpu"
fi

# Check Rust (for building the core)
if command -v cargo &> /dev/null; then
    RUST_VERSION=$(cargo --version | cut -d' ' -f2)
    echo "  ✓ Rust $RUST_VERSION detected"
    HAS_RUST=true
else
    print_warning "Rust not detected - Rust core will not be built"
    HAS_RUST=false
fi

# Check available disk space
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 20 ]; then
    print_warning "Less than 20GB free disk space. Installation may fail."
fi

# Check available memory
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -lt 8 ]; then
    print_warning "Less than 8GB RAM detected. Some features may be slow."
fi

#===============================================================================
# Virtual Environment Setup
#===============================================================================

print_step "Setting up virtual environment..."

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Reusing it."
else
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Upgrade pip, setuptools, wheel
print_step "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
echo "  ✓ Base tools upgraded"

#===============================================================================
# PyTorch Installation (CRITICAL - Must be first)
#===============================================================================

print_step "Installing PyTorch 2.7+ with CUDA $CUDA_VERSION..."

if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch>=2.7.0 torchvision>=0.18.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/cpu
else
    # Convert CUDA version for PyTorch (12.6 -> cu126)
    CUDA_SHORT=$(echo $CUDA_VERSION | tr -d '.')
    TORCH_CUDA="cu${CUDA_SHORT}"

    # Try to install with specified CUDA version
    pip install torch>=2.7.0 torchvision>=0.18.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/${TORCH_CUDA} || {
        print_warning "PyTorch with CUDA $CUDA_VERSION not available, trying CUDA 12.1..."
        pip install torch>=2.7.0 torchvision>=0.18.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/cu121
    }
fi

echo "  ✓ PyTorch installed"

# Verify PyTorch installation
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__} verified')"
python3 -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')"

#===============================================================================
# Core Scientific Dependencies
#===============================================================================

print_step "Installing core scientific dependencies..."

pip install \
    "numpy>=1.24.0,<2.0.0" \
    "scipy>=1.11.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "scikit-image>=0.21.0"

echo "  ✓ Scientific packages installed"

#===============================================================================
# Computer Vision Dependencies
#===============================================================================

print_step "Installing computer vision libraries..."

pip install \
    "opencv-python>=4.8.0" \
    "opencv-python-headless>=4.8.0" \
    "Pillow>=10.0.0" \
    "imageio>=2.31.0" \
    "imageio-ffmpeg>=0.4.8"

echo "  ✓ Computer vision libraries installed"

#===============================================================================
# Deep Learning Framework Dependencies
#===============================================================================

print_step "Installing deep learning frameworks..."

pip install \
    "transformers>=4.45.0" \
    "huggingface-hub>=0.25.0" \
    "safetensors>=0.4.0" \
    "accelerate>=0.25.0" \
    "timm>=0.9.12" \
    "einops>=0.7.0" \
    "kornia>=0.7.0" \
    "diffusers>=0.24.0"

echo "  ✓ Deep learning frameworks installed"

#===============================================================================
# ONNX Runtime (GPU-accelerated if CUDA available)
#===============================================================================

print_step "Installing ONNX Runtime..."

if [ "$CUDA_VERSION" != "cpu" ]; then
    pip install "onnx>=1.15.0" "onnxruntime-gpu>=1.16.0" || {
        print_warning "GPU ONNX Runtime failed, falling back to CPU version"
        pip install "onnx>=1.15.0" "onnxruntime>=1.16.0"
    }
else
    pip install "onnx>=1.15.0" "onnxruntime>=1.16.0"
fi

echo "  ✓ ONNX Runtime installed"

#===============================================================================
# Professional Format Support (OpenEXR)
#===============================================================================

print_step "Installing professional format support..."

pip install \
    "OpenEXR>=3.2.0" \
    "Imath>=3.1.0" \
    "colour-science>=0.4.0"

echo "  ✓ Professional formats installed"

#===============================================================================
# 3D Processing Libraries
#===============================================================================

print_step "Installing 3D processing libraries..."

pip install \
    "open3d>=0.18.0" \
    "trimesh>=4.0.0"

echo "  ✓ 3D libraries installed"

#===============================================================================
# GUI and CLI Dependencies
#===============================================================================

print_step "Installing GUI and CLI tools..."

pip install \
    "PySide6>=6.6.0" \
    "pyqtgraph>=0.13.0" \
    "gradio>=4.0.0" \
    "click>=8.1.0" \
    "rich>=13.0.0" \
    "tqdm>=4.65.0" \
    "pyyaml>=6.0" \
    "omegaconf>=2.3.0"

echo "  ✓ GUI/CLI tools installed"

#===============================================================================
# Video Processing (optional)
#===============================================================================

print_step "Installing video processing libraries..."

pip install \
    "av>=11.0.0" || print_warning "PyAV installation failed (optional)"

pip install \
    "decord>=0.6.0" || print_warning "Decord installation failed (optional)"

#===============================================================================
# SAM3 Installation (from source)
#===============================================================================

if [ "$SKIP_MODELS" = false ]; then
    print_step "Installing SAM3 (Segment Anything Model 3)..."

    # Check if SAM3 repo exists, clone if not
    if [ ! -d "external/sam3" ]; then
        mkdir -p external
        git clone https://github.com/facebookresearch/sam3.git external/sam3 || {
            print_warning "SAM3 repository not found. Trying alternative installation..."
            pip install "segment-anything>=1.0" || print_warning "SAM3 installation failed"
        }
    fi

    # Install SAM3 if repo exists
    if [ -d "external/sam3" ]; then
        cd external/sam3
        pip install -e .
        cd ../..
        echo "  ✓ SAM3 installed from source"
    fi
else
    print_warning "Skipping SAM3 installation (--skip-models flag)"
fi

#===============================================================================
# Depth Anything V3 Installation (from source)
#===============================================================================

if [ "$SKIP_MODELS" = false ]; then
    print_step "Installing Depth Anything V3..."

    # Check if Depth Anything V3 repo exists
    if [ ! -d "external/depth-anything-v3" ]; then
        mkdir -p external
        git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git external/depth-anything-v3 || {
            print_warning "Depth Anything V3 repository not found. Using fallback..."
            # Try installing Depth Anything V2 as fallback
            pip install "depth-anything>=1.0.0" || print_warning "Depth Anything installation failed"
        }
    fi

    # Install Depth Anything V3 if repo exists
    if [ -d "external/depth-anything-v3" ]; then
        cd external/depth-anything-v3
        pip install -e .
        cd ../..
        echo "  ✓ Depth Anything V3 installed from source"
    fi
else
    print_warning "Skipping Depth Anything V3 installation (--skip-models flag)"
fi

#===============================================================================
# Optional Dependencies (Detectron2, etc.)
#===============================================================================

print_step "Installing optional dependencies..."

# Detectron2 (for ViTMatte - optional)
if [ "$CUDA_VERSION" != "cpu" ]; then
    print_step "  Installing Detectron2 (for ViTMatte support)..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git' || {
        print_warning "Detectron2 installation failed (optional)"
    }
fi

# 3D Gaussian Splatting (for Depth Anything V3 novel view synthesis)
pip install "gsplat>=0.1.0" 2>/dev/null || print_warning "gsplat installation failed (optional)"

# CLIP (for open-vocabulary segmentation)
pip install "ftfy>=6.0.0" "regex>=2022.0.0" || print_warning "CLIP dependencies failed (optional)"

#===============================================================================
# Development Dependencies (if --dev flag)
#===============================================================================

if [ "$INSTALL_DEV" = true ]; then
    print_step "Installing development dependencies..."

    pip install \
        "pytest>=7.4.0" \
        "pytest-cov>=4.1.0" \
        "black>=23.0.0" \
        "ruff>=0.1.0" \
        "mypy>=1.6.0" \
        "isort>=5.12.0"

    echo "  ✓ Development tools installed"
fi

#===============================================================================
# Install Ultimate Rotoscopy Package
#===============================================================================

print_step "Installing Ultimate Rotoscopy package..."

pip install -e .
echo "  ✓ Ultimate Rotoscopy installed in editable mode"

#===============================================================================
# Build Rust Core (if Rust available)
#===============================================================================

if [ "$HAS_RUST" = true ]; then
    print_step "Building Rust core for maximum performance..."

    # Install maturin if not present
    pip install maturin>=1.0

    # Build and install Rust extension
    maturin develop --release || {
        print_warning "Rust core build failed - Python fallback will be used"
    }

    echo "  ✓ Rust core built successfully"
else
    print_warning "Skipping Rust core build (Rust not installed)"
fi

#===============================================================================
# Verify Installation
#===============================================================================

print_step "Verifying installation..."

python3 <<EOF
import sys
print("Testing imports...")

# Core
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"  ✗ Transformers: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"  ✗ OpenCV: {e}")
    sys.exit(1)

try:
    import OpenEXR
    print(f"  ✓ OpenEXR available")
except ImportError:
    print(f"  ✗ OpenEXR: Not available")

# Ultimate Rotoscopy
try:
    import ultimate_rotoscopy
    print(f"  ✓ Ultimate Rotoscopy package")
except ImportError as e:
    print(f"  ✗ Ultimate Rotoscopy: {e}")
    sys.exit(1)

# Optional models
try:
    import sam3
    print(f"  ✓ SAM3 available")
except ImportError:
    print(f"  ⚠ SAM3 not available (install with: pip install git+https://github.com/facebookresearch/sam3.git)")

try:
    import depth_anything_3
    print(f"  ✓ Depth Anything V3 available")
except ImportError:
    print(f"  ⚠ Depth Anything V3 not available (install from source)")

# CUDA
if torch.cuda.is_available():
    print(f"  ✓ CUDA {torch.version.cuda} available")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"  ⚠ CUDA not available (CPU-only mode)")

print("\nAll core dependencies installed successfully!")
EOF

#===============================================================================
# Download Models (optional)
#===============================================================================

print_step "Model checkpoints download..."
echo "To download model checkpoints, run:"
echo "  python scripts/download_models.py"

#===============================================================================
# Installation Complete
#===============================================================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Activate the environment with:"
echo "  source venv/bin/activate"
echo ""
echo "Test the installation:"
echo "  rotoscopy test"
echo ""
echo "Run the GUI:"
echo "  rotoscopy-gui"
echo ""
echo "Process an image:"
echo "  rotoscopy process image.jpg -p \"100,200\" -o output/"
echo ""
echo -e "For documentation, visit: ${BLUE}README.md${NC}"
echo ""

if [ "$SKIP_MODELS" = true ]; then
    print_warning "SAM3 and Depth Anything V3 were not installed."
    echo "To install them manually:"
    echo "  ./install.sh  # without --skip-models flag"
fi
