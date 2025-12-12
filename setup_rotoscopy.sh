#!/bin/bash
# =============================================================================
# Ultimate Rotoscopy - Complete Installation Script
# =============================================================================
#
# Installs:
#   - SAM2 (Segment Anything Model 2) - Interactive segmentation
#   - MatAnyone (CVPR 2025) - Video matting with memory propagation
#   - Depth Anything V2 - Metric depth estimation
#   - PySide6 - Professional GUI
#
# Usage:
#   chmod +x setup_rotoscopy.sh
#   ./setup_rotoscopy.sh
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
REPOS_DIR="$SCRIPT_DIR/repos"

echo "=============================================="
echo "   Ultimate Rotoscopy - Installation"
echo "=============================================="
echo ""

# Create directories
mkdir -p "$MODELS_DIR"
mkdir -p "$REPOS_DIR"

# -----------------------------------------------------------------------------
# 1. Python Dependencies
# -----------------------------------------------------------------------------
echo "[1/5] Installing Python dependencies..."

pip install --upgrade pip

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# GUI
pip install PySide6

# Image/Video processing
pip install opencv-python numpy pillow scipy

# Deep learning utilities
pip install transformers einops timm safetensors

# Export formats
pip install OpenEXR

echo "  Python dependencies installed"

# -----------------------------------------------------------------------------
# 2. SAM3 (Segment Anything Model 3)
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Installing SAM3..."

pip install git+https://github.com/facebookresearch/sam3.git

# HuggingFace auth reminder
echo "  SAM3 installed"
echo "  NOTE: Run 'huggingface-cli login' for model access"

# -----------------------------------------------------------------------------
# 3. MatAnyone (CVPR 2025 - Video Matting)
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Installing MatAnyone..."

if [ ! -d "$REPOS_DIR/MatAnyone" ]; then
    cd "$REPOS_DIR"
    git clone https://github.com/pq-yang/MatAnyone.git
    cd MatAnyone
    pip install -e .
else
    echo "  MatAnyone already cloned"
    cd "$REPOS_DIR/MatAnyone"
    pip install -e . --quiet
fi

# Download MatAnyone checkpoint
MATANYONE_CKPT="$MODELS_DIR/matanyone.pth"
if [ ! -f "$MATANYONE_CKPT" ]; then
    echo "  Downloading MatAnyone checkpoint..."
    wget -q --show-progress -O "$MATANYONE_CKPT" \
        "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
else
    echo "  MatAnyone checkpoint exists"
fi

echo "  MatAnyone installed"

# -----------------------------------------------------------------------------
# 4. Depth Anything V2
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Installing Depth Anything V2..."

if [ ! -d "$REPOS_DIR/Depth-Anything-V2" ]; then
    cd "$REPOS_DIR"
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    cd Depth-Anything-V2
    pip install -r requirements.txt
else
    echo "  Depth Anything V2 already cloned"
fi

# Download Depth Anything V2 checkpoint (Large model)
DA2_CKPT="$MODELS_DIR/depth_anything_v2_vitl.pth"
if [ ! -f "$DA2_CKPT" ]; then
    echo "  Downloading Depth Anything V2 checkpoint..."
    # Note: Model is on HuggingFace
    pip install huggingface_hub
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='depth-anything/Depth-Anything-V2-Large',
    filename='depth_anything_v2_vitl.pth',
    local_dir='$MODELS_DIR'
)
"
else
    echo "  Depth Anything V2 checkpoint exists"
fi

echo "  Depth Anything V2 installed"

# -----------------------------------------------------------------------------
# 5. Create symlinks and config
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Creating configuration..."

cd "$SCRIPT_DIR"

# Create config file
cat > "$SCRIPT_DIR/config_paths.py" << 'PYEOF'
"""
Auto-generated paths configuration.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
REPOS_DIR = BASE_DIR / "repos"

# SAM3 (installed via pip, models downloaded automatically)
SAM3_AVAILABLE = True  # Installed via pip install git+...sam3.git

# MatAnyone
MATANYONE_REPO = REPOS_DIR / "MatAnyone"
MATANYONE_CHECKPOINT = MODELS_DIR / "matanyone.pth"

# Depth Anything V2
DA2_REPO = REPOS_DIR / "Depth-Anything-V2"
DA2_CHECKPOINT = MODELS_DIR / "depth_anything_v2_vitl.pth"

def check_installation():
    """Verify all components are installed."""
    missing = []

    # Check SAM3
    try:
        import sam3
        print("  SAM3: OK")
    except ImportError:
        missing.append("SAM3 (pip install git+https://github.com/facebookresearch/sam3.git)")

    if not MATANYONE_CHECKPOINT.exists():
        missing.append("MatAnyone checkpoint")
    if not DA2_CHECKPOINT.exists():
        missing.append("Depth Anything V2 checkpoint")
    if not MATANYONE_REPO.exists():
        missing.append("MatAnyone repository")
    if not DA2_REPO.exists():
        missing.append("Depth Anything V2 repository")

    if missing:
        print("Missing components:")
        for m in missing:
            print(f"  - {m}")
        return False

    print("All components installed!")
    return True

if __name__ == "__main__":
    check_installation()
PYEOF

echo "  Configuration created"

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "   Installation Complete!"
echo "=============================================="
echo ""
echo "Models directory: $MODELS_DIR"
echo "Repos directory:  $REPOS_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify: python config_paths.py"
echo "  2. Run:    python ultimate_gui.py"
echo ""
