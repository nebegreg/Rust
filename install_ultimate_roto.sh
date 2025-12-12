#!/bin/bash
# Ultimate Rotoscopy Installation Script
# ======================================
# Installs all dependencies for the Ultimate Roto pipeline

set -e

echo "=============================================="
echo "  Ultimate Rotoscopy - Installation"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/8] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}  Python $python_version detected${NC}"
else
    echo -e "${RED}  Error: Python $required_version+ required (found $python_version)${NC}"
    exit 1
fi

# Create virtual environment (optional)
if [ "$1" = "--venv" ]; then
    echo -e "\n${YELLOW}[2/8] Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}  Virtual environment activated${NC}"
else
    echo -e "\n${YELLOW}[2/8] Skipping virtual environment (use --venv to create)${NC}"
fi

# Upgrade pip
echo -e "\n${YELLOW}[3/8] Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch with CUDA first (required for detectron2 build)
echo -e "\n${YELLOW}[4/8] Installing PyTorch with CUDA...${NC}"

# Detect CUDA version and install appropriate PyTorch
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "  Detected CUDA version: $CUDA_VERSION"

    # Choose appropriate PyTorch CUDA version
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "  Installing PyTorch for CUDA 12.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        echo "  Installing PyTorch for CUDA 11.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  Installing PyTorch for CUDA 12.1 (default)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo "  No NVIDIA GPU detected, installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}  PyTorch installed${NC}"

# Verify PyTorch installation
python3 -c "import torch; print(f'  PyTorch {torch.__version__} installed, CUDA: {torch.cuda.is_available()}')"

# Install numpy with compatible version for sam3
echo -e "\n${YELLOW}[5/8] Installing numpy (compatible version)...${NC}"
pip install 'numpy>=1.24.0,<2.0.0'
echo -e "${GREEN}  numpy installed${NC}"

# Install core dependencies
echo -e "\n${YELLOW}[6/8] Installing core dependencies...${NC}"
pip install \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.8 \
    tqdm>=4.65.0 \
    PySide6>=6.5.0 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    huggingface_hub>=0.16.0 \
    requests \
    scipy

echo -e "${GREEN}  Core dependencies installed${NC}"

# Install detectron2 for ViTMatte
echo -e "\n${YELLOW}[7/8] Installing detectron2 for ViTMatte...${NC}"

# Get PyTorch version info
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'cpu')")

echo "  PyTorch version: $TORCH_VERSION"
echo "  PyTorch CUDA: $TORCH_CUDA"

# Check for CUDA version mismatch
SYSTEM_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "none")
echo "  System CUDA: $SYSTEM_CUDA"

# Try pre-built wheels first
CUDA_TAG="cu$(echo $TORCH_CUDA | tr -d '.')"
echo "  Attempting to install detectron2..."

if pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_TAG/torch${TORCH_VERSION%.*}/index.html" 2>/dev/null; then
    echo -e "${GREEN}  detectron2 installed from pre-built wheel${NC}"
else
    echo "  Pre-built wheel not available..."

    # Check for CUDA mismatch
    if [ "$SYSTEM_CUDA" != "none" ] && [ "$SYSTEM_CUDA" != "$TORCH_CUDA" ]; then
        echo -e "${YELLOW}  CUDA version mismatch detected (system: $SYSTEM_CUDA, PyTorch: $TORCH_CUDA)${NC}"
        echo "  Installing detectron2 WITHOUT CUDA extensions (CPU mode for custom ops)..."

        # Install without compiling CUDA extensions
        FORCE_CUDA=0 pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation 2>/dev/null || \
        pip install 'git+https://github.com/facebookresearch/detectron2.git' --global-option="build_ext" --global-option="--no-cuda" --no-build-isolation 2>/dev/null || \
        {
            echo "  Trying alternative installation method..."
            # Clone and install in develop mode without building extensions
            TEMP_DIR=$(mktemp -d)
            git clone --depth 1 https://github.com/facebookresearch/detectron2.git "$TEMP_DIR/detectron2"
            cd "$TEMP_DIR/detectron2"
            # Disable CUDA extension building
            export FORCE_CUDA=0
            export TORCH_CUDA_ARCH_LIST=""
            pip install -e . --no-build-isolation 2>/dev/null || pip install . --no-deps
            cd -
            rm -rf "$TEMP_DIR"
        }
        echo -e "${GREEN}  detectron2 installed (CPU mode for custom ops)${NC}"
        echo -e "${YELLOW}  Note: Some operations may be slower without CUDA extensions${NC}"
    else
        echo "  Building from source..."
        pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
        echo -e "${GREEN}  detectron2 installed from source${NC}"
    fi
fi

# Install SAM3 (optional - comment out if not needed)
echo -e "\n${YELLOW}[8/8] Installing SAM3 (optional)...${NC}"
if pip install 'git+https://github.com/facebookresearch/sam3.git' 2>/dev/null; then
    echo -e "${GREEN}  SAM3 installed${NC}"
else
    echo -e "${YELLOW}  SAM3 installation failed - you may need to install manually${NC}"
    echo "  Try: pip install git+https://github.com/facebookresearch/sam3.git"
fi

# Extract model files if not already done
echo -e "\n${YELLOW}Checking model repositories...${NC}"

if [ -f "ViTMatte-main.zip" ] && [ ! -d "ViTMatte-main" ]; then
    unzip -q ViTMatte-main.zip
    echo -e "${GREEN}  ViTMatte extracted${NC}"
elif [ -d "ViTMatte-main" ]; then
    echo "  ViTMatte already extracted"
fi

if [ -f "MatAnyone.zip" ] && [ ! -d "matanyone" ]; then
    unzip -q MatAnyone.zip
    echo -e "${GREEN}  MatAnyone extracted${NC}"
elif [ -d "matanyone" ]; then
    echo "  MatAnyone already extracted"
fi

if [ -f "Depth-Anything-3-main.zip" ] && [ ! -d "Depth-Anything-3-main" ]; then
    unzip -q Depth-Anything-3-main.zip
    echo -e "${GREEN}  Depth-Anything-3 extracted${NC}"
elif [ -d "Depth-Anything-3-main" ]; then
    echo "  Depth-Anything-3 already extracted"
fi

# Download model weights
echo -e "\n${YELLOW}Downloading model weights...${NC}"
mkdir -p pretrained_models

# MatAnyone weights
if [ ! -f "pretrained_models/matanyone.pth" ]; then
    echo "  Downloading MatAnyone weights..."
    wget -q --show-progress -O pretrained_models/matanyone.pth \
        https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth || \
    curl -L -o pretrained_models/matanyone.pth \
        https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth
    echo -e "${GREEN}  MatAnyone weights downloaded${NC}"
else
    echo "  MatAnyone weights already present"
fi

# HuggingFace authentication reminder
echo -e "\n${YELLOW}HuggingFace authentication:${NC}"
echo "  SAM3 requires HuggingFace authentication."
echo "  Run: huggingface-cli login"
echo "  Or set: export HF_TOKEN=your_token"

# ViTMatte weights (manual download required)
echo -e "\n${YELLOW}ViTMatte weights:${NC}"
echo "  Download from: https://github.com/hustvl/ViTMatte"
echo "  Place in: ViTMatte-main/pretrained/"

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
python3 << 'PYEOF'
import sys
print("Python:", sys.version.split()[0])

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"PyTorch: ERROR - {e}")

try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except Exception as e:
    print(f"NumPy: ERROR - {e}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV: ERROR - {e}")

try:
    from PySide6.QtWidgets import QApplication
    print("PySide6: OK")
except Exception as e:
    print(f"PySide6: ERROR - {e}")

try:
    import detectron2
    print(f"Detectron2: {detectron2.__version__}")
except Exception as e:
    print(f"Detectron2: ERROR - {e}")

try:
    from omegaconf import DictConfig
    print("OmegaConf: OK")
except Exception as e:
    print(f"OmegaConf: ERROR - {e}")
PYEOF

echo -e "\n=============================================="
echo -e "${GREEN}  Installation Complete!${NC}"
echo "=============================================="
echo ""
echo "Usage:"
echo "  # Image rotoscopy"
echo "  python ultimate_roto.py image input.jpg --text 'person' -o results/"
echo ""
echo "  # Video rotoscopy"
echo "  python ultimate_roto.py video input.mp4 --text 'person' -o results/"
echo ""
echo "  # GUI mode"
echo "  python ultimate_roto_gui.py"
echo ""
