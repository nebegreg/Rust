#!/bin/bash
# Installation Script - SAM3 + Depth Anything V3
# Compatible: Rocky Linux 9 + RTX 4090

set -e  # Exit on error

echo "======================================================================"
echo "  SAM3 + Depth Anything V3 Installation"
echo "======================================================================"
echo ""

# Vérifications préalables
echo "[1/8] Checking system..."

if ! command -v python3.12 &> /dev/null; then
    echo "✗ Python 3.12 not found!"
    echo "Install with: sudo dnf install python3.12"
    exit 1
fi

echo "✓ Python 3.12 found: $(python3.12 --version)"

if ! nvidia-smi &> /dev/null; then
    echo "✗ NVIDIA GPU not detected!"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "✓ GPU detected: $GPU_NAME"

# Créer venv
echo ""
echo "[2/8] Creating virtual environment..."
python3.12 -m venv venv_ultimate --clear
source venv_ultimate/bin/activate

echo "✓ Virtual environment created"

# Upgrade pip
echo ""
echo "[3/8] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# PyTorch 2.7+ avec CUDA 12.6
echo ""
echo "[4/8] Installing PyTorch 2.7+ with CUDA 12.6..."
echo "  This may take a few minutes..."

# Vérifier si PyTorch 2.7 existe, sinon utiliser 2.5+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "✓ PyTorch installed"

# numpy <2
echo ""
echo "[5/8] Installing numpy <2..."
pip install "numpy<2"

# Depth Anything V3 (en premier car plus de dépendances)
echo ""
echo "[6/8] Installing Depth Anything V3..."
echo "  This will install ~20 dependencies..."

pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

echo "✓ Depth Anything V3 installed"

# HuggingFace CLI
echo ""
echo "[7/8] Installing HuggingFace CLI..."
pip install huggingface_hub[cli]

echo ""
echo "⚠️  IMPORTANT: SAM3 Authentication Required"
echo ""
echo "SAM3 requires HuggingFace token access:"
echo "  1. Create account: https://huggingface.co/join"
echo "  2. Request access: https://huggingface.co/facebook/sam3"
echo "  3. Create token: https://huggingface.co/settings/tokens"
echo "  4. Run: huggingface-cli login"
echo ""
read -p "Have you completed steps 1-3 and ready to authenticate? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Authenticating with HuggingFace..."
    huggingface-cli login
else
    echo ""
    echo "⚠️  Skipping authentication - you'll need to run manually:"
    echo "    huggingface-cli login"
    echo ""
fi

# SAM3
echo ""
echo "[8/8] Installing SAM3..."
pip install git+https://github.com/facebookresearch/sam3.git

echo "✓ SAM3 installed"

# Vérification finale
echo ""
echo "======================================================================"
echo "  Installation Verification"
echo "======================================================================"

echo ""
echo "Running pip check..."
pip check && echo "✓ No conflicts detected" || echo "⚠️  Some conflicts detected (may be OK)"

echo ""
echo "Testing imports..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
python -c "
try:
    from sam3 import sam3_model_registry
    print('✓ SAM3 importable')
except ImportError as e:
    print(f'✗ SAM3 import failed: {e}')
" || echo "⚠️  SAM3 not accessible - check HuggingFace authentication"

echo ""
python -c "
try:
    from depth_anything_3 import DepthAnythingV3
    print('✓ Depth Anything V3 importable')
except ImportError as e:
    print(f'✗ DA3 import failed: {e}')
"

echo ""
echo "======================================================================"
echo "  ✓ Installation Complete!"
echo "======================================================================"
echo ""
echo "Activate environment with:"
echo "  source venv_ultimate/bin/activate"
echo ""
echo "Test with:"
echo "  python test_models.py"
echo ""
