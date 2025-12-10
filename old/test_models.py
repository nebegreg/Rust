#!/usr/bin/env python3
"""
Test Script - SAM3 + Depth Anything V3
======================================

Vérifie que les deux modèles peuvent charger et fonctionner ensemble.
"""

import sys
import time
import numpy as np
import torch

print("=" * 70)
print("  SAM3 + Depth Anything V3 - Test Script")
print("=" * 70)
print()

# 1. Test PyTorch et GPU
print("[1/5] PyTorch & GPU Status")
print("-" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {total_mem:.1f} GB")
    print("✓ GPU ready")
else:
    print("✗ No GPU detected - models will run on CPU (slow!)")

print()

# 2. Test SAM3 Import
print("[2/5] Testing SAM3 Import")
print("-" * 70)

try:
    from sam3 import sam3_model_registry, Sam3ImagePredictor
    print("✓ SAM3 modules imported successfully")

    # List available models
    print(f"Available SAM3 models: {list(sam3_model_registry.keys())}")

except ImportError as e:
    print(f"✗ SAM3 import failed: {e}")
    print()
    print("Possible causes:")
    print("  1. SAM3 not installed: pip install git+https://github.com/facebookresearch/sam3.git")
    print("  2. HuggingFace not authenticated: huggingface-cli login")
    print("  3. No access to SAM3 repo: https://huggingface.co/facebook/sam3")
    sys.exit(1)

print()

# 3. Test Depth Anything V3 Import
print("[3/5] Testing Depth Anything V3 Import")
print("-" * 70)

try:
    from depth_anything_3 import DepthAnythingV3
    print("✓ Depth Anything V3 imported successfully")

except ImportError as e:
    print(f"✗ DA3 import failed: {e}")
    print()
    print("Install with:")
    print("  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git")
    sys.exit(1)

print()

# 4. Test SAM3 Model Loading
print("[4/5] Testing SAM3 Model Loading")
print("-" * 70)

try:
    print("Loading SAM3 (smallest model for quick test)...")
    start = time.time()

    # Use smallest model for testing
    model_type = "sam3_hiera_tiny"

    # Check if model exists in registry
    if model_type not in sam3_model_registry:
        print(f"Model {model_type} not in registry, trying base...")
        model_type = list(sam3_model_registry.keys())[0]

    sam_checkpoint = sam3_model_registry[model_type]()
    predictor = Sam3ImagePredictor(sam_checkpoint)

    elapsed = time.time() - start
    print(f"✓ SAM3 loaded in {elapsed:.2f}s")

    # Test on dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    predictor.set_image(dummy_image)
    print("✓ SAM3 can process images")

    del predictor
    torch.cuda.empty_cache()

except Exception as e:
    print(f"✗ SAM3 loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 5. Test Depth Anything V3 Model Loading
print("[5/5] Testing Depth Anything V3 Model Loading")
print("-" * 70)

try:
    print("Loading Depth Anything V3 (small model)...")
    start = time.time()

    depth_model = DepthAnythingV3.from_pretrained(
        "depth-anything/Depth-Anything-V3-Small",  # Smallest for testing
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    elapsed = time.time() - start
    print(f"✓ Depth Anything V3 loaded in {elapsed:.2f}s")

    # Test on dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    with torch.no_grad():
        depth = depth_model.infer_image(dummy_image)

    print(f"✓ DA3 can estimate depth (output shape: {depth.shape})")

    del depth_model
    torch.cuda.empty_cache()

except Exception as e:
    print(f"✗ DA3 loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Summary
print("=" * 70)
print("  ✓ SUCCESS - Both models working!")
print("=" * 70)
print()
print("Both SAM3 and Depth Anything V3 loaded and processed test images.")
print()
print("Memory usage during tests:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
else:
    print("  CPU mode (no GPU memory used)")

print()
print("Next steps:")
print("  1. See examples/demo_sam3.py for SAM3 usage")
print("  2. See examples/demo_depth.py for DA3 usage")
print("  3. See examples/demo_combined.py for both together")
print()
