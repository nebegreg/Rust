#!/bin/bash
#===============================================================================
# Ultimate Rotoscopy - Installation Script for Rocky Linux 9 + RTX 4090
#===============================================================================
# This script installs all dependencies for the Ultimate Rotoscopy application
# Tested on: Rocky Linux 9.x with NVIDIA RTX 4090
#
# Sources:
# - SAM2: https://github.com/facebookresearch/sam2
# - Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
# - MatAnyone: https://github.com/pq-yang/MatAnyone
# - ViTMatte: https://github.com/hustvl/ViTMatte
# - Matte Anything: https://github.com/hustvl/Matte-Anything
# - Background Matting V2: https://github.com/PeterL1n/BackgroundMattingV2
#===============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CUDA_VERSION="12.4"
PYTHON_VERSION="3.10"
INSTALL_DIR="$HOME/ultimate_rotoscopy"
VENV_NAME="ultimate_roto_env"
MODELS_DIR="$INSTALL_DIR/pretrained_models"

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}     Ultimate Rotoscopy - Installation for Rocky Linux 9 + RTX 4090${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""

#-------------------------------------------------------------------------------
# Function: Check if running as root
#-------------------------------------------------------------------------------
check_root() {
    if [[ $EUID -eq 0 ]]; then
        echo -e "${RED}Error: Do not run this script as root${NC}"
        echo "Run as a regular user. Script will ask for sudo when needed."
        exit 1
    fi
}

#-------------------------------------------------------------------------------
# Function: Check Rocky Linux version
#-------------------------------------------------------------------------------
check_os() {
    echo -e "${YELLOW}[1/10] Checking operating system...${NC}"

    if [ ! -f /etc/rocky-release ]; then
        echo -e "${RED}Error: This script is designed for Rocky Linux${NC}"
        exit 1
    fi

    VERSION=$(cat /etc/rocky-release | grep -oP '\d+' | head -1)
    if [ "$VERSION" -lt 9 ]; then
        echo -e "${RED}Error: Rocky Linux 9 or higher required${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Rocky Linux $VERSION detected${NC}"
}

#-------------------------------------------------------------------------------
# Function: Check NVIDIA GPU
#-------------------------------------------------------------------------------
check_gpu() {
    echo -e "${YELLOW}[2/10] Checking NVIDIA GPU...${NC}"

    if ! lspci | grep -i nvidia > /dev/null; then
        echo -e "${RED}Error: No NVIDIA GPU detected${NC}"
        exit 1
    fi

    GPU_NAME=$(lspci | grep -i nvidia | head -1 | cut -d: -f3)
    echo -e "${GREEN}✓ NVIDIA GPU found: $GPU_NAME${NC}"
}

#-------------------------------------------------------------------------------
# Function: Install system dependencies
#-------------------------------------------------------------------------------
install_system_deps() {
    echo -e "${YELLOW}[3/10] Installing system dependencies...${NC}"

    # Enable EPEL and PowerTools
    sudo dnf install -y epel-release
    sudo dnf config-manager --set-enabled crb || sudo dnf config-manager --set-enabled powertools

    # Install development tools
    sudo dnf groupinstall -y "Development Tools"

    # Install required packages
    sudo dnf install -y \
        kernel-headers-$(uname -r) \
        kernel-devel-$(uname -r) \
        gcc gcc-c++ \
        cmake \
        git \
        wget curl \
        tar bzip2 \
        make automake \
        pciutils \
        elfutils-libelf-devel \
        libglvnd-opengl libglvnd-glx libglvnd-devel \
        acpid \
        pkgconfig \
        dkms \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-devel \
        python${PYTHON_VERSION}-pip \
        ffmpeg \
        opencv opencv-devel \
        openexr openexr-devel \
        qt5-qtbase qt5-qtbase-devel \
        xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil

    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

#-------------------------------------------------------------------------------
# Function: Install NVIDIA drivers and CUDA
#-------------------------------------------------------------------------------
install_nvidia_cuda() {
    echo -e "${YELLOW}[4/10] Installing NVIDIA drivers and CUDA ${CUDA_VERSION}...${NC}"

    # Check if NVIDIA driver is already installed
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        echo -e "${GREEN}✓ NVIDIA driver already installed: $DRIVER_VERSION${NC}"
    else
        echo "Installing NVIDIA drivers..."

        # Disable Nouveau
        if lsmod | grep -q nouveau; then
            echo "Disabling Nouveau driver..."
            sudo bash -c 'cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF'
            sudo dracut --force
            echo -e "${YELLOW}Nouveau disabled. A reboot may be required.${NC}"
        fi

        # Add NVIDIA CUDA repository
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
        sudo dnf clean all

        # Install CUDA toolkit (includes drivers)
        sudo dnf module install -y nvidia-driver:latest-dkms
        sudo dnf install -y cuda-toolkit-12-4
    fi

    # Install cuDNN
    echo "Installing cuDNN..."
    sudo dnf install -y cudnn9-cuda-12

    # Set environment variables
    echo "Setting up CUDA environment variables..."

    CUDA_ENV_FILE="$HOME/.cuda_env"
    cat > "$CUDA_ENV_FILE" << 'EOF'
# CUDA Environment Variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
EOF

    # Add to bashrc if not already there
    if ! grep -q "source $CUDA_ENV_FILE" ~/.bashrc; then
        echo "source $CUDA_ENV_FILE" >> ~/.bashrc
    fi
    source "$CUDA_ENV_FILE"

    echo -e "${GREEN}✓ NVIDIA CUDA ${CUDA_VERSION} installed${NC}"
}

#-------------------------------------------------------------------------------
# Function: Create Python virtual environment
#-------------------------------------------------------------------------------
create_venv() {
    echo -e "${YELLOW}[5/10] Creating Python virtual environment...${NC}"

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    # Create venv
    python${PYTHON_VERSION} -m venv $VENV_NAME
    source $VENV_NAME/bin/activate

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    echo -e "${GREEN}✓ Virtual environment created: $INSTALL_DIR/$VENV_NAME${NC}"
}

#-------------------------------------------------------------------------------
# Function: Install PyTorch with CUDA support
#-------------------------------------------------------------------------------
install_pytorch() {
    echo -e "${YELLOW}[6/10] Installing PyTorch with CUDA ${CUDA_VERSION} support...${NC}"

    source "$INSTALL_DIR/$VENV_NAME/bin/activate"

    # Install PyTorch 2.5+ with CUDA 12.4
    # Source: https://pytorch.org/get-started/locally/
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # Verify installation
    python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

    echo -e "${GREEN}✓ PyTorch installed with CUDA support${NC}"
}

#-------------------------------------------------------------------------------
# Function: Install AI model dependencies
#-------------------------------------------------------------------------------
install_ai_deps() {
    echo -e "${YELLOW}[7/10] Installing AI model dependencies...${NC}"

    source "$INSTALL_DIR/$VENV_NAME/bin/activate"

    # Core dependencies
    pip install \
        numpy>=1.24.0 \
        scipy>=1.10.0 \
        opencv-python>=4.8.0 \
        opencv-python-headless>=4.8.0 \
        Pillow>=10.0.0 \
        imageio>=2.31.0 \
        imageio-ffmpeg>=0.4.8 \
        scikit-image>=0.21.0 \
        scikit-learn>=1.3.0

    # Deep learning frameworks
    pip install \
        transformers>=4.36.0 \
        huggingface-hub>=0.20.0 \
        safetensors>=0.4.0 \
        accelerate>=0.25.0 \
        timm>=0.9.12 \
        einops>=0.7.0

    # SAM2 - Segment Anything Model 2
    # Source: https://github.com/facebookresearch/sam2
    echo "Installing SAM2..."
    pip install git+https://github.com/facebookresearch/sam2.git

    # Depth Anything V2
    # Source: https://github.com/DepthAnything/Depth-Anything-V2
    echo "Installing Depth Anything V2 dependencies..."
    pip install \
        gradio>=4.0.0 \
        tqdm>=4.65.0

    # ViTMatte dependencies (detectron2)
    # Source: https://github.com/hustvl/ViTMatte
    echo "Installing detectron2 for ViTMatte..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git'

    # GroundingDINO for Matte Anything
    # Source: https://github.com/IDEA-Research/GroundingDINO
    echo "Installing GroundingDINO..."
    pip install groundingdino-py

    # Diffusers for GVM
    pip install diffusers>=0.24.0

    # ONNX Runtime with CUDA
    pip install \
        onnx>=1.15.0 \
        onnxruntime-gpu>=1.16.0

    # GUI dependencies
    pip install \
        PySide6>=6.6.0 \
        pyqtgraph>=0.13.0

    # OpenEXR support
    pip install OpenEXR>=3.2.0 Imath>=3.1.0

    # Color science
    pip install colour-science>=0.4.0

    # Video processing
    pip install \
        av>=11.0.0 \
        decord>=0.6.0

    # Additional utilities
    pip install \
        pyyaml>=6.0 \
        omegaconf>=2.3.0 \
        rich>=13.0.0 \
        click>=8.1.0 \
        typer>=0.9.0

    echo -e "${GREEN}✓ AI model dependencies installed${NC}"
}

#-------------------------------------------------------------------------------
# Function: Clone and install third-party repositories
#-------------------------------------------------------------------------------
install_third_party() {
    echo -e "${YELLOW}[8/10] Installing third-party repositories...${NC}"

    source "$INSTALL_DIR/$VENV_NAME/bin/activate"

    THIRD_PARTY_DIR="$INSTALL_DIR/third_party"
    mkdir -p "$THIRD_PARTY_DIR"
    cd "$THIRD_PARTY_DIR"

    # Clone Depth Anything V2
    if [ ! -d "Depth-Anything-V2" ]; then
        echo "Cloning Depth Anything V2..."
        git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    fi

    # Clone MatAnyone
    if [ ! -d "MatAnyone" ]; then
        echo "Cloning MatAnyone..."
        git clone https://github.com/pq-yang/MatAnyone.git
        cd MatAnyone
        pip install -e .
        cd ..
    fi

    # Clone ViTMatte
    if [ ! -d "ViTMatte" ]; then
        echo "Cloning ViTMatte..."
        git clone https://github.com/hustvl/ViTMatte.git
    fi

    # Clone Matte-Anything
    if [ ! -d "Matte-Anything" ]; then
        echo "Cloning Matte-Anything..."
        git clone https://github.com/hustvl/Matte-Anything.git
    fi

    # Clone Background Matting V2
    if [ ! -d "BackgroundMattingV2" ]; then
        echo "Cloning Background Matting V2..."
        git clone https://github.com/PeterL1n/BackgroundMattingV2.git
    fi

    echo -e "${GREEN}✓ Third-party repositories installed${NC}"
}

#-------------------------------------------------------------------------------
# Function: Download pretrained models
#-------------------------------------------------------------------------------
download_models() {
    echo -e "${YELLOW}[9/10] Downloading pretrained models...${NC}"

    source "$INSTALL_DIR/$VENV_NAME/bin/activate"

    mkdir -p "$MODELS_DIR"
    cd "$MODELS_DIR"

    # SAM2 checkpoints
    echo "Downloading SAM2 checkpoints..."
    SAM2_DIR="$MODELS_DIR/sam2"
    mkdir -p "$SAM2_DIR"

    if [ ! -f "$SAM2_DIR/sam2.1_hiera_large.pt" ]; then
        wget -O "$SAM2_DIR/sam2.1_hiera_large.pt" \
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    fi

    # Depth Anything V2 checkpoints
    echo "Downloading Depth Anything V2 checkpoints..."
    DAV2_DIR="$MODELS_DIR/depth_anything_v2"
    mkdir -p "$DAV2_DIR"

    if [ ! -f "$DAV2_DIR/depth_anything_v2_vitl.pth" ]; then
        wget -O "$DAV2_DIR/depth_anything_v2_vitl.pth" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    fi

    # ViTMatte checkpoint
    echo "Downloading ViTMatte checkpoint..."
    VITMATTE_DIR="$MODELS_DIR/vitmatte"
    mkdir -p "$VITMATTE_DIR"

    if [ ! -f "$VITMATTE_DIR/ViTMatte_B_DIS.pth" ]; then
        wget -O "$VITMATTE_DIR/ViTMatte_B_DIS.pth" \
            "https://huggingface.co/hustvl/vitmatte-base-distinctions-646/resolve/main/pytorch_model.bin"
    fi

    # GroundingDINO checkpoint
    echo "Downloading GroundingDINO checkpoint..."
    GDINO_DIR="$MODELS_DIR/groundingdino"
    mkdir -p "$GDINO_DIR"

    if [ ! -f "$GDINO_DIR/groundingdino_swint_ogc.pth" ]; then
        wget -O "$GDINO_DIR/groundingdino_swint_ogc.pth" \
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    fi

    # MatAnyone will download automatically on first use
    echo "MatAnyone checkpoint will be downloaded on first use..."

    echo -e "${GREEN}✓ Pretrained models downloaded${NC}"
}

#-------------------------------------------------------------------------------
# Function: Install Ultimate Rotoscopy
#-------------------------------------------------------------------------------
install_ultimate_rotoscopy() {
    echo -e "${YELLOW}[10/10] Installing Ultimate Rotoscopy...${NC}"

    source "$INSTALL_DIR/$VENV_NAME/bin/activate"

    cd "$INSTALL_DIR"

    # Copy source code (assuming it's in current directory)
    if [ -d "src/ultimate_rotoscopy" ]; then
        echo "Ultimate Rotoscopy source found, installing..."
        pip install -e .
    else
        echo "Source not found in current directory."
        echo "Please copy the source code to $INSTALL_DIR/src/"
    fi

    # Create launcher script
    cat > "$INSTALL_DIR/launch_ultimate_rotoscopy.sh" << EOF
#!/bin/bash
source "$HOME/.cuda_env"
source "$INSTALL_DIR/$VENV_NAME/bin/activate"
cd "$INSTALL_DIR"
python -m ultimate_rotoscopy.gui.main_window "\$@"
EOF
    chmod +x "$INSTALL_DIR/launch_ultimate_rotoscopy.sh"

    # Create desktop entry
    cat > "$HOME/.local/share/applications/ultimate_rotoscopy.desktop" << EOF
[Desktop Entry]
Name=Ultimate Rotoscopy
Comment=AI-Powered Professional VFX Tool
Exec=$INSTALL_DIR/launch_ultimate_rotoscopy.sh
Icon=$INSTALL_DIR/icons/app_icon.png
Terminal=false
Type=Application
Categories=Graphics;Video;
EOF

    echo -e "${GREEN}✓ Ultimate Rotoscopy installed${NC}"
}

#-------------------------------------------------------------------------------
# Function: Print summary
#-------------------------------------------------------------------------------
print_summary() {
    echo ""
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${GREEN}     Installation Complete!${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
    echo ""
    echo "Installation directory: $INSTALL_DIR"
    echo "Virtual environment: $INSTALL_DIR/$VENV_NAME"
    echo "Pretrained models: $MODELS_DIR"
    echo ""
    echo "To activate the environment:"
    echo "  source $INSTALL_DIR/$VENV_NAME/bin/activate"
    echo ""
    echo "To launch the application:"
    echo "  $INSTALL_DIR/launch_ultimate_rotoscopy.sh"
    echo ""
    echo "Or use the desktop application 'Ultimate Rotoscopy'"
    echo ""
    echo -e "${YELLOW}NOTE: If NVIDIA drivers were just installed, please reboot your system.${NC}"
    echo ""
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
main() {
    check_root
    check_os
    check_gpu
    install_system_deps
    install_nvidia_cuda
    create_venv
    install_pytorch
    install_ai_deps
    install_third_party
    download_models
    install_ultimate_rotoscopy
    print_summary
}

# Run main function
main "$@"
