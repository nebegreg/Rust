#!/bin/bash
# Installation Minimale - SAM3 uniquement

echo "Installing SAM3..."

# Installer SAM3
pip install git+https://github.com/facebookresearch/sam3.git

# Installer dépendances basiques
pip install opencv-python numpy pillow

echo ""
echo "✓ Installation done"
echo ""
echo "Next: huggingface-cli login"
echo "Then: python roto.py image.jpg 100,100"
