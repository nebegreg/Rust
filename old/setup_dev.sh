#!/bin/bash
#===============================================================================
# Development Setup Script - Fix Installation Issues
#===============================================================================
# This script properly installs both the Rust extension and Python package
#===============================================================================

set -e

echo "🔧 Ultimate Rotoscopy - Development Setup"
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in virtual environment"
    echo "   Run: source venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo ""

# Step 1: Build Rust extension with maturin
echo "[1/3] Building Rust extension..."
maturin develop --release

# Step 2: Install Python package with dependencies
echo ""
echo "[2/3] Installing Python package and dependencies..."
pip install -e .

# Step 3: Verify installation
echo ""
echo "[3/3] Verifying installation..."

python3 << 'EOF'
import sys

try:
    import ultimate_rotoscopy
    print(f"✓ Python package: {ultimate_rotoscopy.__version__}")
except Exception as e:
    print(f"✗ Python package: {e}")
    sys.exit(1)

try:
    import rotoscopy_core
    print(f"✓ Rust extension: rotoscopy_core")
except ImportError:
    print(f"⚠ Rust extension not found (optional)")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print(f"✗ PyTorch not installed")

print("\n✓ Installation complete!")
EOF

echo ""
echo "🎉 Setup complete! You can now use:"
echo "   rotoscopy --help"
echo "   rotoscopy-gui"
echo ""
