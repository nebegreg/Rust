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
echo -e "\n${YELLOW}[1/7] Checking Python version...${NC}"
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
    echo -e "\n${YELLOW}[2/7] Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}  Virtual environment activated${NC}"
else
    echo -e "\n${YELLOW}[2/7] Skipping virtual environment (use --venv to create)${NC}"
fi

# Install PyTorch with CUDA
echo -e "\n${YELLOW}[3/7] Installing PyTorch with CUDA...${NC}"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo -e "${GREEN}  PyTorch installed${NC}"

# Install core dependencies
echo -e "\n${YELLOW}[4/7] Installing core dependencies...${NC}"
pip install \
    numpy>=1.24.0 \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.8 \
    tqdm>=4.65.0 \
    PySide6>=6.5.0 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    huggingface_hub>=0.16.0

echo -e "${GREEN}  Core dependencies installed${NC}"

# Install detectron2 for ViTMatte
echo -e "\n${YELLOW}[5/7] Installing detectron2 for ViTMatte...${NC}"
pip install 'git+https://github.com/facebookresearch/detectron2.git'
echo -e "${GREEN}  detectron2 installed${NC}"

# Install SAM3
echo -e "\n${YELLOW}[6/7] Installing SAM3...${NC}"
pip install 'git+https://github.com/facebookresearch/sam3.git'
echo -e "${GREEN}  SAM3 installed${NC}"

# Authenticate with HuggingFace
echo -e "\n${YELLOW}[7/7] HuggingFace authentication...${NC}"
echo "  SAM3 requires HuggingFace authentication."
echo "  Run: huggingface-cli login"
echo "  Or set: export HF_TOKEN=your_token"

# Extract model files
echo -e "\n${YELLOW}Extracting model repositories...${NC}"

if [ -f "ViTMatte-main.zip" ] && [ ! -d "ViTMatte-main" ]; then
    unzip -q ViTMatte-main.zip
    echo -e "${GREEN}  ViTMatte extracted${NC}"
fi

if [ -f "MatAnyone.zip" ] && [ ! -d "matanyone" ]; then
    unzip -q MatAnyone.zip
    echo -e "${GREEN}  MatAnyone extracted${NC}"
fi

if [ -f "Depth-Anything-3-main.zip" ] && [ ! -d "Depth-Anything-3-main" ]; then
    unzip -q Depth-Anything-3-main.zip
    echo -e "${GREEN}  Depth-Anything-3 extracted${NC}"
fi

# Download model weights
echo -e "\n${YELLOW}Downloading model weights...${NC}"
mkdir -p pretrained_models

# MatAnyone weights
if [ ! -f "pretrained_models/matanyone.pth" ]; then
    echo "  Downloading MatAnyone weights..."
    wget -q -O pretrained_models/matanyone.pth \
        https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth
    echo -e "${GREEN}  MatAnyone weights downloaded${NC}"
fi

# ViTMatte weights (manual download required)
echo -e "\n${YELLOW}ViTMatte weights:${NC}"
echo "  Download from: https://github.com/hustvl/ViTMatte"
echo "  Place in: ViTMatte-main/pretrained/"

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

python3 -c "
try:
    import cv2
    print(f'  OpenCV: {cv2.__version__}')
except: print('  OpenCV: NOT INSTALLED')
"

python3 -c "
try:
    from PySide6.QtWidgets import QApplication
    print('  PySide6: OK')
except: print('  PySide6: NOT INSTALLED')
"

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
