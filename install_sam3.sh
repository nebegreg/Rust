#!/bin/bash
#===============================================================================
# SAM3 Complete Tool - Installation Script
#===============================================================================
# Installs SAM3 with all dependencies for the complete tool
#
# Usage:
#   ./install_sam3.sh [--cuda-version 12.6] [--cpu-only]
#===============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
CUDA_VERSION="12.6"
CPU_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda-version VERSION   CUDA version (12.1, 12.4, 12.6)"
            echo "  --cpu-only               Install CPU-only version"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

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

echo "==============================================================================="
echo "  SAM3 Complete Tool - Installation"
echo "==============================================================================="

#===============================================================================
# System Checks
#===============================================================================

print_step "Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.12 or higher."
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "  ✓ Python $PYTHON_VERSION detected"

# Check if Python 3.12+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    print_error "Python 3.12+ required, found $PYTHON_VERSION"
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 not found. Please install pip."
fi

echo "  ✓ pip $(pip3 --version | cut -d' ' -f2) detected"

# Check CUDA (if not CPU-only)
if [ "$CPU_ONLY" = false ]; then
    if command -v nvcc &> /dev/null; then
        SYSTEM_CUDA=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        echo "  ✓ CUDA $SYSTEM_CUDA detected"

        if [[ "$SYSTEM_CUDA" != "$CUDA_VERSION"* ]]; then
            print_warning "System CUDA ($SYSTEM_CUDA) differs from requested ($CUDA_VERSION)"
            print_warning "PyTorch will be installed for CUDA $CUDA_VERSION"
        fi
    else
        print_warning "CUDA not detected"
        read -p "Continue with CPU-only installation? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        CPU_ONLY=true
    fi
fi

# Check disk space
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    print_warning "Less than 10GB free disk space. Installation may fail."
fi

#===============================================================================
# Virtual Environment
#===============================================================================

print_step "Setting up virtual environment..."

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Reusing it."
else
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
fi

source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "  ✓ pip upgraded"

#===============================================================================
# PyTorch Installation
#===============================================================================

print_step "Installing PyTorch 2.7+..."

if [ "$CPU_ONLY" = true ]; then
    pip install torch>=2.7.0 torchvision>=0.18.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/cpu
    echo "  ✓ PyTorch (CPU) installed"
else
    # Convert CUDA version (12.6 -> cu126)
    CUDA_SHORT=$(echo $CUDA_VERSION | tr -d '.')
    TORCH_CUDA="cu${CUDA_SHORT}"

    pip install torch>=2.7.0 torchvision>=0.18.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/${TORCH_CUDA} || {
        print_warning "PyTorch with CUDA $CUDA_VERSION not available, trying CUDA 12.1..."
        pip install torch>=2.7.0 torchvision>=0.18.0 torchaudio>=2.7.0 --index-url https://download.pytorch.org/whl/cu121
    }
    echo "  ✓ PyTorch (CUDA) installed"
fi

# Verify PyTorch
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__} verified')"
python3 -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')"

#===============================================================================
# SAM3 Installation
#===============================================================================

print_step "Installing SAM3 from GitHub..."

pip install git+https://github.com/facebookresearch/sam3.git || {
    print_error "SAM3 installation failed. Check HuggingFace authentication."
}

echo "  ✓ SAM3 installed"

#===============================================================================
# GUI Dependencies
#===============================================================================

print_step "Installing GUI dependencies..."

pip install \
    "PySide6>=6.6.0" \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "Pillow>=10.0.0"

echo "  ✓ GUI dependencies installed"

#===============================================================================
# Optional Dependencies
#===============================================================================

print_step "Installing optional dependencies..."

pip install \
    "matplotlib>=3.7.0" \
    "tqdm>=4.65.0"

echo "  ✓ Optional dependencies installed"

#===============================================================================
# Verify Installation
#===============================================================================

print_step "Verifying installation..."

python3 <<EOF
import sys

print("Testing imports...")

# PyTorch
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  ✗ PyTorch: {e}")
    sys.exit(1)

# SAM3
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    print(f"  ✓ SAM3 model builders available")
except ImportError as e:
    print(f"  ✗ SAM3: {e}")
    print(f"  ⚠ Run: hf auth login")
    sys.exit(1)

# GUI
try:
    from PySide6.QtWidgets import QApplication
    print(f"  ✓ PySide6 available")
except ImportError as e:
    print(f"  ✗ PySide6: {e}")
    sys.exit(1)

# OpenCV
try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"  ✗ OpenCV: {e}")
    sys.exit(1)

# NumPy
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy: {e}")
    sys.exit(1)

print("\nAll dependencies installed successfully!")
EOF

#===============================================================================
# HuggingFace Authentication
#===============================================================================

print_step "HuggingFace Authentication Setup..."

echo ""
echo "SAM3 requires HuggingFace authentication:"
echo "  1. Request access: https://huggingface.co/facebook/sam3"
echo "  2. Generate token: https://huggingface.co/settings/tokens"
echo "  3. Run: hf auth login"
echo ""

read -p "Authenticate with HuggingFace now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if huggingface-cli is installed
    if ! command -v hf &> /dev/null; then
        print_step "Installing HuggingFace CLI..."
        pip install huggingface-hub[cli]
    fi

    hf auth login
    echo "  ✓ HuggingFace authentication complete"
else
    print_warning "Skipping HuggingFace authentication"
    echo "  Run 'hf auth login' before using SAM3"
fi

#===============================================================================
# Installation Complete
#===============================================================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Files installed:"
echo "  - sam3_complete.py     (CLI + API)"
echo "  - sam3_gui.py          (GUI interface)"
echo "  - SAM3_README.md       (Documentation)"
echo ""
echo "Activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Test SAM3 CLI:"
echo "  python sam3_complete.py image test.jpg --text \"red car\" --output mask.png"
echo ""
echo "Launch GUI:"
echo "  python sam3_gui.py"
echo ""
echo "Check installation:"
echo "  python -c 'from sam3.model_builder import build_sam3_image_model; print(\"SAM3 OK\")'"
echo ""
echo -e "For documentation, see: ${BLUE}SAM3_README.md${NC}"
echo ""

if [ "$CPU_ONLY" = true ]; then
    print_warning "CPU-only installation - processing will be slower"
    echo "  For GPU support, install CUDA and re-run with --cuda-version"
fi
