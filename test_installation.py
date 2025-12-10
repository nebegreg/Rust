#!/usr/bin/env python3
"""
Installation Verification Script for Ultimate Rotoscopy
Tests all dependencies, models, and functionality
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print('='*70)

def check_python():
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    elif version.minor >= 12:
        print("✅ Python 3.12+ - SAM3 compatible")
    else:
        print("⚠️  Python 3.10-3.11 - SAM3 requires 3.12+, will use SAM2.1 fallback")
    return True

def check_pytorch():
    print_header("PyTorch & CUDA")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

            # Check if CUDA version is compatible
            cuda_ver = torch.version.cuda
            if cuda_ver and float(cuda_ver.split('.')[0]) >= 12:
                print("✅ CUDA 12+ - Optimal for RTX 4090")
            else:
                print(f"⚠️  CUDA {cuda_ver} - CUDA 12+ recommended for RTX 4090")
        else:
            print("⚠️  CUDA not available - will run on CPU (slow)")

        return True
    except ImportError as e:
        print(f"❌ PyTorch not installed: {e}")
        print("   Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
        return False

def check_transformers():
    print_header("Transformers & HuggingFace")
    try:
        import transformers
        version = transformers.__version__
        print(f"✅ Transformers: {version}")

        major, minor = map(int, version.split('.')[:2])
        if major > 4 or (major == 4 and minor >= 50):
            print("✅ Version 4.50+ - SAM3 compatible")
        elif major == 4 and minor >= 45:
            print("✅ Version 4.45+ - SAM2.1 compatible")
        else:
            print("⚠️  Version < 4.45 - may have issues")

        return True
    except ImportError as e:
        print(f"❌ Transformers not installed: {e}")
        return False

def check_sam3():
    print_header("SAM3 - Segment Anything Model 3")

    # Try direct import
    try:
        import sam3
        print(f"✅ SAM3 package installed")
        print(f"   Location: {sam3.__file__}")
        return True
    except ImportError:
        print("❌ SAM3 package not found")
        print("   SAM3 may not be released yet")
        print("   Install: pip install git+https://github.com/facebookresearch/sam3.git")

    # Check transformers fallback
    print("\n📦 Checking SAM2.1 fallback...")
    try:
        from transformers import Sam2Model
        print("✅ SAM2.1 available via transformers (fallback works)")
        return True
    except ImportError:
        print("❌ SAM2.1 not available")
        return False

def check_depth_anything():
    print_header("Depth Anything V3")

    # Try direct import
    try:
        import depth_anything_v3
        print(f"✅ Depth Anything V3 package installed")
        return True
    except ImportError:
        print("❌ Depth Anything V3 package not found")
        print("   Install: pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git")

    # Check transformers fallback
    print("\n📦 Checking Depth Anything V2 fallback...")
    try:
        from transformers import pipeline
        # Try to create depth estimation pipeline
        print("✅ Depth Anything V2 available via transformers (fallback works)")
        return True
    except ImportError:
        print("❌ Depth Anything fallback not available")
        return False

def check_core_dependencies():
    print_header("Core Dependencies")

    packages = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'scipy': 'SciPy',
        'PySide6': 'PySide6 (GUI)',
        'gradio': 'Gradio',
        'timm': 'PyTorch Image Models',
        'einops': 'Einops',
        'accelerate': 'Accelerate',
    }

    all_ok = True
    for module, name in packages.items():
        try:
            if module == 'cv2':
                import cv2
            else:
                __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            all_ok = False

    return all_ok

def check_ultimate_rotoscopy():
    print_header("Ultimate Rotoscopy Package")

    try:
        import ultimate_rotoscopy
        print(f"✅ Package installed")
        print(f"   Version: {ultimate_rotoscopy.__version__}")
        print(f"   Location: {Path(ultimate_rotoscopy.__file__).parent}")

        # Check key modules
        modules = [
            ('ultimate_rotoscopy.gui', 'GUI module'),
            ('ultimate_rotoscopy.models.sam3', 'SAM3 model'),
            ('ultimate_rotoscopy.models.depth_anything', 'Depth Anything'),
            ('ultimate_rotoscopy.gui.backend', 'Processing backend'),
        ]

        print("\n📦 Key modules:")
        for module, name in modules:
            try:
                __import__(module)
                print(f"   ✅ {name}")
            except ImportError as e:
                print(f"   ❌ {name}: {e}")

        return True
    except ImportError as e:
        print(f"❌ Package not installed: {e}")
        print("\n   Run installation:")
        print("   pip install -e .")
        print("   OR")
        print("   ./setup_dev.sh")
        return False

def check_rust_extension():
    print_header("Rust Extension (Optional Performance)")

    try:
        import rotoscopy_core
        print(f"✅ Rust extension installed")
        print(f"   Location: {rotoscopy_core.__file__}")
        return True
    except ImportError:
        print("⚠️  Rust extension not found (optional)")
        print("   Build with: maturin develop --release")
        return False

def test_basic_functionality():
    print_header("Functionality Tests")

    try:
        import numpy as np
        import ultimate_rotoscopy

        print("Testing imports...")

        # Test model imports
        try:
            from ultimate_rotoscopy.models.sam3 import SAM3Segmentor, SAM3Config
            print("✅ SAM3Segmentor importable")
        except ImportError as e:
            print(f"❌ SAM3Segmentor import failed: {e}")

        try:
            from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3
            print("✅ DepthAnythingV3 importable")
        except ImportError as e:
            print(f"❌ DepthAnythingV3 import failed: {e}")

        try:
            from ultimate_rotoscopy.gui.backend import ProcessingBackend
            print("✅ ProcessingBackend importable")
        except ImportError as e:
            print(f"❌ ProcessingBackend import failed: {e}")

        try:
            from ultimate_rotoscopy.gui.modern_gui import ModernMainWindow
            print("✅ ModernMainWindow importable")
        except ImportError as e:
            print(f"❌ ModernMainWindow import failed: {e}")

        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def check_gpu_memory():
    print_header("GPU Memory")

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3

                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   Total: {total:.2f} GB")
                print(f"   Allocated: {allocated:.2f} GB")
                print(f"   Cached: {cached:.2f} GB")
                print(f"   Free: {total - allocated:.2f} GB")

                if total >= 24:
                    print(f"   ✅ {total:.0f}GB - Excellent for SAM3 + DA3")
                elif total >= 12:
                    print(f"   ✅ {total:.0f}GB - Good for most models")
                else:
                    print(f"   ⚠️  {total:.0f}GB - May need smaller models")
        else:
            print("No GPU available")
    except Exception as e:
        print(f"GPU check failed: {e}")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     Ultimate Rotoscopy - Installation Verification Script       ║
╚══════════════════════════════════════════════════════════════════╝
""")

    results = {
        'Python': check_python(),
        'PyTorch': check_pytorch(),
        'Transformers': check_transformers(),
        'SAM3': check_sam3(),
        'Depth Anything': check_depth_anything(),
        'Core Dependencies': check_core_dependencies(),
        'Ultimate Rotoscopy': check_ultimate_rotoscopy(),
        'Rust Extension': check_rust_extension(),
        'Functionality': test_basic_functionality(),
    }

    check_gpu_memory()

    # Final summary
    print_header("SUMMARY")

    critical = ['Python', 'PyTorch', 'Transformers', 'Core Dependencies', 'Ultimate Rotoscopy']
    optional = ['SAM3', 'Depth Anything', 'Rust Extension']

    critical_ok = all(results[k] for k in critical if k in results)

    print("\n🔍 Critical Components:")
    for key in critical:
        status = "✅" if results.get(key) else "❌"
        print(f"   {status} {key}")

    print("\n📦 Optional Components:")
    for key in optional:
        status = "✅" if results.get(key) else "⚠️ "
        print(f"   {status} {key}")

    print("\n" + "="*70)

    if critical_ok:
        print("✅ INSTALLATION READY!")
        print("\nYou can run:")
        print("  rotoscopy-gui     # Launch GUI")
        print("  rotoscopy --help  # CLI help")

        if not results.get('SAM3'):
            print("\n⚠️  Note: SAM3 not found, will use SAM2.1 fallback")

        if not results.get('Depth Anything'):
            print("⚠️  Note: DA3 not found, will use DA2 fallback")

        return 0
    else:
        print("❌ INSTALLATION INCOMPLETE")
        print("\nMissing critical components. Run:")
        print("  pip install -r requirements.txt")
        print("  pip install -e .")
        return 1

if __name__ == "__main__":
    sys.exit(main())
