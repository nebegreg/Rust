#!/bin/bash
#===============================================================================
# Ultimate Rotoscopy - Installation Script for Rocky Linux 9 + RTX 4090
#===============================================================================
# This script installs all dependencies with specific versions required by:
# - SAM3: Python 3.12+, PyTorch 2.7+, CUDA 12.6+, transformers >= 4.50.0
# - Depth Anything 3: Python 3.10+, PyTorch 2.0+, CUDA 11.8+
#
# RTX 4090 requires CUDA 12.x for optimal performance
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Ultimate Rotoscopy - Installation for Rocky Linux 9 + RTX 4090 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}[ERROR] Do not run this script as root. Run as regular user.${NC}"
    exit 1
fi

# Check Rocky Linux version
if [ -f /etc/rocky-release ]; then
    ROCKY_VERSION=$(cat /etc/rocky-release | grep -oP '\d+' | head -1)
    echo -e "${GREEN}[✓] Detected Rocky Linux ${ROCKY_VERSION}${NC}"
else
    echo -e "${YELLOW}[!] Warning: Not Rocky Linux. Proceed with caution.${NC}"
fi

# Check NVIDIA GPU
echo ""
echo -e "${BLUE}[1/10] Checking NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo -e "${GREEN}[✓] GPU: ${GPU_NAME}${NC}"
    echo -e "${GREEN}[✓] NVIDIA Driver: ${DRIVER_VERSION}${NC}"

    # Check if driver supports CUDA 12.6
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -lt 550 ]; then
        echo -e "${YELLOW}[!] Warning: Driver ${DRIVER_VERSION} may not support CUDA 12.6${NC}"
        echo -e "${YELLOW}    SAM3 requires CUDA 12.6+ which needs driver >= 550.x${NC}"
        echo -e "${YELLOW}    Recommend updating NVIDIA driver to 550+ for best performance${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo -e "${RED}[✗] NVIDIA GPU not detected or drivers not installed${NC}"
    echo -e "${YELLOW}Please install NVIDIA drivers first:${NC}"
    echo "  sudo dnf install -y kernel-devel kernel-headers gcc make dkms"
    echo "  sudo dnf install -y nvidia-driver nvidia-driver-cuda nvidia-driver-libs"
    exit 1
fi

#===============================================================================
# System Dependencies
#===============================================================================
echo ""
echo -e "${BLUE}[2/10] Installing system dependencies...${NC}"

sudo dnf update -y

# Development tools
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    git wget curl vim \
    gcc gcc-c++ make cmake ninja-build \
    kernel-devel kernel-headers \
    epel-release

# Python 3.12 (required for SAM3)
echo ""
echo -e "${BLUE}[3/10] Installing Python 3.12...${NC}"
sudo dnf install -y python3.12 python3.12-devel python3.12-pip

# Make Python 3.12 default (optional)
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo alternatives --set python3 /usr/bin/python3.12

PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+')
echo -e "${GREEN}[✓] Python ${PYTHON_VERSION} installed${NC}"

if [[ "${PYTHON_VERSION}" < "3.12" ]]; then
    echo -e "${RED}[✗] Python 3.12+ required for SAM3${NC}"
    exit 1
fi

# Image processing libraries
echo ""
echo -e "${BLUE}[4/10] Installing image/video libraries...${NC}"
sudo dnf install -y \
    opencv opencv-devel \
    ffmpeg ffmpeg-devel \
    libjpeg-turbo-devel \
    libpng-devel \
    libtiff-devel \
    libwebp-devel \
    OpenEXR-devel \
    ilmbase-devel

# OpenGL/Qt dependencies for GUI
sudo dnf install -y \
    mesa-libGL-devel \
    mesa-libGLU-devel \
    libX11-devel \
    libXrender-devel \
    libxcb-devel \
    xcb-util-wm-devel

#===============================================================================
# CUDA 12.6 Installation
#===============================================================================
echo ""
echo -e "${BLUE}[5/10] Installing CUDA 12.6...${NC}"

CUDA_VERSION="12-6"
CUDA_VERSION_DOT="12.6"

# Check if CUDA already installed
if [ -d "/usr/local/cuda-${CUDA_VERSION_DOT}" ]; then
    echo -e "${GREEN}[✓] CUDA ${CUDA_VERSION_DOT} already installed${NC}"
else
    echo "Installing CUDA ${CUDA_VERSION_DOT}..."

    # Add CUDA repository
    sudo dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

    # Install CUDA toolkit
    sudo dnf install -y cuda-toolkit-${CUDA_VERSION}

    # Set CUDA environment variables
    echo 'export CUDA_HOME=/usr/local/cuda' | sudo tee -a /etc/profile.d/cuda.sh
    echo 'export PATH=$CUDA_HOME/bin:$PATH' | sudo tee -a /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh

    source /etc/profile.d/cuda.sh

    echo -e "${GREEN}[✓] CUDA ${CUDA_VERSION_DOT} installed${NC}"
fi

# Verify CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    echo -e "${GREEN}[✓] CUDA ${CUDA_VER} verified${NC}"
else
    echo -e "${RED}[✗] CUDA not found in PATH${NC}"
    echo "Run: source /etc/profile.d/cuda.sh"
fi

#===============================================================================
# cuDNN Installation (for optimal performance)
#===============================================================================
echo ""
echo -e "${BLUE}[6/10] Installing cuDNN 9.x for CUDA 12.6...${NC}"

if [ ! -f "/usr/local/cuda/include/cudnn.h" ]; then
    echo "Installing cuDNN..."
    sudo dnf install -y libcudnn9-cuda-12 libcudnn9-devel-cuda-12
    echo -e "${GREEN}[✓] cuDNN installed${NC}"
else
    echo -e "${GREEN}[✓] cuDNN already installed${NC}"
fi

#===============================================================================
# Rust Installation (for Rust core)
#===============================================================================
echo ""
echo -e "${BLUE}[7/10] Installing Rust...${NC}"

if command -v cargo &> /dev/null; then
    echo -e "${GREEN}[✓] Rust already installed${NC}"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}[✓] Rust installed${NC}"
fi

# Install Maturin for Python-Rust bindings
pip3 install maturin

#===============================================================================
# Python Virtual Environment
#===============================================================================
echo ""
echo -e "${BLUE}[8/10] Creating Python virtual environment...${NC}"

# Create venv
if [ -d "venv" ]; then
    echo -e "${YELLOW}[!] Virtual environment already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}[✓] Virtual environment activated${NC}"

# Upgrade pip
pip install --upgrade pip setuptools wheel

#===============================================================================
# PyTorch 2.7+ with CUDA 12.6
#===============================================================================
echo ""
echo -e "${BLUE}[9/10] Installing PyTorch 2.7+ with CUDA 12.6...${NC}"
echo -e "${YELLOW}Note: SAM3 requires PyTorch 2.7+, DA3 requires 2.0+${NC}"
echo -e "${YELLOW}Installing PyTorch 2.7+ will satisfy both requirements${NC}"

# Check current PyTorch version
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo -e "${YELLOW}[!] PyTorch ${TORCH_VERSION} already installed${NC}"

    TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d. -f1)
    TORCH_MINOR=$(echo $TORCH_VERSION | cut -d. -f2)

    if [ "$TORCH_MAJOR" -ge 2 ] && [ "$TORCH_MINOR" -ge 7 ]; then
        echo -e "${GREEN}[✓] PyTorch version sufficient for SAM3${NC}"
    else
        echo -e "${YELLOW}[!] PyTorch < 2.7, upgrading for SAM3 compatibility...${NC}"
        pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    fi
else
    echo "Installing PyTorch 2.7+ with CUDA 12.6..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

# Verify PyTorch CUDA
python3 << END
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
END

if [ $? -ne 0 ]; then
    echo -e "${RED}[✗] PyTorch CUDA verification failed${NC}"
    exit 1
fi

echo -e "${GREEN}[✓] PyTorch with CUDA 12.6 installed${NC}"

#===============================================================================
# Python Dependencies
#===============================================================================
echo ""
echo -e "${BLUE}[10/10] Installing Python dependencies...${NC}"

# Install base requirements
pip install -r requirements.txt

# Install transformers >= 4.50.0 for SAM3
echo ""
echo -e "${BLUE}Installing transformers >= 4.50.0 (for SAM3)...${NC}"
pip install --upgrade "transformers>=4.50.0"

# Install SAM3 from source (if available)
echo ""
echo -e "${BLUE}Attempting to install SAM3...${NC}"
if git ls-remote https://github.com/facebookresearch/sam3.git &>/dev/null; then
    echo "Installing SAM3 from GitHub..."
    pip install git+https://github.com/facebookresearch/sam3.git
    echo -e "${GREEN}[✓] SAM3 installed${NC}"
else
    echo -e "${YELLOW}[!] SAM3 repository not yet available${NC}"
    echo -e "${YELLOW}    Fallback: Will use SAM2.1 from transformers (facebook/sam2.1-hiera-large)${NC}"
fi

# Install Depth Anything 3 from source (if available)
echo ""
echo -e "${BLUE}Attempting to install Depth Anything 3...${NC}"
if git ls-remote https://github.com/ByteDance-Seed/Depth-Anything-3.git &>/dev/null; then
    echo "Installing Depth Anything 3 from GitHub..."
    pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
    echo -e "${GREEN}[✓] Depth Anything 3 installed${NC}"
else
    echo -e "${YELLOW}[!] Depth Anything 3 repository not yet available${NC}"
    echo -e "${YELLOW}    Fallback: Will use Depth Anything V2 from transformers${NC}"
fi

# Install optional packages
echo ""
echo -e "${BLUE}Installing optional packages...${NC}"

# 3D Gaussian Splatting (for DA3 novel view synthesis)
pip install gsplat || echo -e "${YELLOW}[!] gsplat not available, skipping${NC}"

# CuPy for GPU-accelerated NumPy operations
pip install cupy-cuda12x || echo -e "${YELLOW}[!] cupy not available, skipping${NC}"

# Detectron2 for ViTMatte
pip install 'git+https://github.com/facebookresearch/detectron2.git' || echo -e "${YELLOW}[!] detectron2 failed, skipping${NC}"

# Video processing
pip install av decord

#===============================================================================
# Build Rust Extension
#===============================================================================
echo ""
echo -e "${BLUE}Building Rust extension module...${NC}"

source "$HOME/.cargo/env"
maturin develop --release

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[✓] Rust extension built successfully${NC}"
else
    echo -e "${RED}[✗] Rust extension build failed${NC}"
    echo "You can build it manually later with: maturin develop --release"
fi

#===============================================================================
# Verification
#===============================================================================
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Running verification tests...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

python3 << 'VERIFY'
import sys
print("\n" + "="*60)
print("VERIFICATION REPORT")
print("="*60)

# Check Python version
print(f"\n✓ Python: {sys.version.split()[0]}")

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

# Check transformers version for SAM3
try:
    import transformers
    version = transformers.__version__
    major, minor = map(int, version.split('.')[:2])
    print(f"✓ Transformers: {version}")
    if major > 4 or (major == 4 and minor >= 50):
        print(f"  - SAM3 compatible ✓")
    elif major == 4 and minor >= 45:
        print(f"  - SAM2.1 compatible ✓ (SAM3 needs >= 4.50)")
    else:
        print(f"  - WARNING: May need upgrade for SAM3")
except Exception as e:
    print(f"✗ Transformers: {e}")

# Check if SAM3 available
try:
    import sam3
    print(f"✓ SAM3: Installed")
except ImportError:
    print(f"! SAM3: Not installed (will use SAM2.1 fallback)")

# Check if Depth Anything 3 available
try:
    import depth_anything_v3
    print(f"✓ Depth Anything 3: Installed")
except ImportError:
    print(f"! Depth Anything 3: Not installed (will use DA2 fallback)")

# Check core dependencies
packages = [
    'numpy', 'scipy', 'opencv-python', 'PIL', 'timm',
    'einops', 'accelerate', 'diffusers', 'PySide6'
]
for pkg in packages:
    module = pkg.replace('-', '_').replace('opencv_python', 'cv2')
    try:
        __import__(module)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg}")

# Check Rust extension
try:
    import rotoscopy_core
    print(f"✓ Rust extension: rotoscopy_core")
except ImportError:
    print(f"! Rust extension: Not built (run: maturin develop --release)")

# Check Ultimate Rotoscopy package
try:
    import ultimate_rotoscopy
    print(f"✓ Ultimate Rotoscopy: {ultimate_rotoscopy.__version__}")
except Exception as e:
    print(f"✗ Ultimate Rotoscopy: {e}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60 + "\n")
VERIFY

#===============================================================================
# Installation Complete
#===============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Installation Complete!                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo -e "1. Activate the virtual environment:"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "2. Test the installation:"
echo -e "   ${YELLOW}python3 -c 'import ultimate_rotoscopy; print(ultimate_rotoscopy.__version__)'${NC}"
echo ""
echo -e "3. Launch the GUI:"
echo -e "   ${YELLOW}rotoscopy-gui${NC}"
echo ""
echo -e "4. Or use the CLI:"
echo -e "   ${YELLOW}rotoscopy --help${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo -e "   README.md - Getting started"
echo -e "   docs/     - Full documentation"
echo ""
echo -e "${BLUE}GPU Info:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
echo ""
echo -e "${GREEN}Happy Rotoscoping! 🎬${NC}"
echo ""
