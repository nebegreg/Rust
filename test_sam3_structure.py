#!/usr/bin/env python3
"""
SAM3 Structure Test - Verify code structure without running models
=================================================================

Tests the SAM3 complete wrapper structure, classes, and methods
without requiring SAM3 to be installed.

Usage:
    python test_sam3_structure.py
"""

import sys
import ast
import inspect
from pathlib import Path


def test_file_exists():
    """Test that required files exist."""
    print("Testing file structure...")

    required_files = [
        "sam3_complete.py",
        "sam3_gui.py",
        "SAM3_README.md",
        "install_sam3.sh"
    ]

    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file} exists ({path.stat().st_size} bytes)")
        else:
            print(f"  ✗ {file} NOT FOUND")
            return False

    return True


def test_sam3_complete_structure():
    """Test sam3_complete.py structure."""
    print("\nTesting sam3_complete.py structure...")

    try:
        # Parse the file without importing (to avoid SAM3 dependency)
        with open("sam3_complete.py", "r") as f:
            tree = ast.parse(f.read())

        # Expected classes
        expected_classes = [
            "PromptType",
            "SegmentationResult",
            "VideoTrackingSession",
            "SAM3ImageProcessor",
            "SAM3VideoTracker"
        ]

        # Find all class definitions
        classes_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)

        # Check each expected class
        for cls_name in expected_classes:
            if cls_name in classes_found:
                print(f"  ✓ Class {cls_name} defined")
            else:
                print(f"  ✗ Class {cls_name} NOT FOUND")
                return False

        # Find all function definitions
        functions_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions_found.append(node.name)

        expected_functions = [
            "save_mask",
            "save_visualization",
            "main"
        ]

        for func_name in expected_functions:
            if func_name in functions_found:
                print(f"  ✓ Function {func_name} defined")
            else:
                print(f"  ✗ Function {func_name} NOT FOUND")
                return False

        # Check for key methods in SAM3ImageProcessor
        processor_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SAM3ImageProcessor":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        processor_methods.append(item.name)

        expected_processor_methods = [
            "__init__",
            "_load_model",
            "segment_with_text",
            "segment_with_points",
            "segment_with_box"
        ]

        for method in expected_processor_methods:
            if method in processor_methods:
                print(f"  ✓ SAM3ImageProcessor.{method} defined")
            else:
                print(f"  ✗ SAM3ImageProcessor.{method} NOT FOUND")
                return False

        # Check for key methods in SAM3VideoTracker
        tracker_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SAM3VideoTracker":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        tracker_methods.append(item.name)

        expected_tracker_methods = [
            "__init__",
            "_load_model",
            "start_session",
            "add_text_prompt",
            "add_point_prompt",
            "propagate_tracking"
        ]

        for method in expected_tracker_methods:
            if method in tracker_methods:
                print(f"  ✓ SAM3VideoTracker.{method} defined")
            else:
                print(f"  ✗ SAM3VideoTracker.{method} NOT FOUND")
                return False

        return True

    except Exception as e:
        print(f"  ✗ Error parsing sam3_complete.py: {e}")
        return False


def test_sam3_gui_structure():
    """Test sam3_gui.py structure."""
    print("\nTesting sam3_gui.py structure...")

    try:
        with open("sam3_gui.py", "r") as f:
            tree = ast.parse(f.read())

        # Expected classes
        expected_classes = [
            "ImageViewport",
            "SAM3Worker",
            "SAM3MainWindow"
        ]

        classes_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)

        for cls_name in expected_classes:
            if cls_name in classes_found:
                print(f"  ✓ Class {cls_name} defined")
            else:
                print(f"  ✗ Class {cls_name} NOT FOUND")
                return False

        # Check ImageViewport methods
        viewport_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ImageViewport":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        viewport_methods.append(item.name)

        expected_viewport_methods = [
            "set_image",
            "set_mask",
            "set_annotation_mode",
            "clear_annotations",
            "mousePressEvent",
            "mouseMoveEvent",
            "mouseReleaseEvent"
        ]

        for method in expected_viewport_methods:
            if method in viewport_methods:
                print(f"  ✓ ImageViewport.{method} defined")
            else:
                print(f"  ✗ ImageViewport.{method} NOT FOUND")
                return False

        return True

    except Exception as e:
        print(f"  ✗ Error parsing sam3_gui.py: {e}")
        return False


def test_documentation():
    """Test documentation completeness."""
    print("\nTesting documentation...")

    try:
        with open("SAM3_README.md", "r") as f:
            readme = f.read()

        required_sections = [
            "## Requirements",
            "## Installation",
            "## Usage",
            "### Command Line Interface",
            "### Graphical User Interface",
            "### Python API",
            "## Architecture",
            "## Troubleshooting"
        ]

        for section in required_sections:
            if section in readme:
                print(f"  ✓ Documentation has '{section}'")
            else:
                print(f"  ✗ Documentation missing '{section}'")
                return False

        # Check for code examples
        if "```bash" in readme and "```python" in readme:
            print("  ✓ Documentation has code examples")
        else:
            print("  ✗ Documentation missing code examples")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error reading SAM3_README.md: {e}")
        return False


def test_install_script():
    """Test installation script structure."""
    print("\nTesting install_sam3.sh structure...")

    try:
        with open("install_sam3.sh", "r") as f:
            script = f.read()

        required_elements = [
            "#!/bin/bash",
            "set -e",
            "CUDA_VERSION=",
            "PyTorch Installation",
            "SAM3 Installation",
            "GUI Dependencies",
            "HuggingFace Authentication"
        ]

        for element in required_elements:
            if element in script:
                print(f"  ✓ Script has '{element}'")
            else:
                print(f"  ✗ Script missing '{element}'")
                return False

        # Check for error handling
        if "print_error" in script and "print_warning" in script:
            print("  ✓ Script has error handling")
        else:
            print("  ✗ Script missing error handling")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error reading install_sam3.sh: {e}")
        return False


def test_code_quality():
    """Test code quality metrics."""
    print("\nTesting code quality...")

    try:
        # Count lines of code
        with open("sam3_complete.py", "r") as f:
            complete_lines = len(f.readlines())

        with open("sam3_gui.py", "r") as f:
            gui_lines = len(f.readlines())

        print(f"  ✓ sam3_complete.py: {complete_lines} lines")
        print(f"  ✓ sam3_gui.py: {gui_lines} lines")

        # Check for docstrings
        with open("sam3_complete.py", "r") as f:
            complete_content = f.read()

        docstring_count = complete_content.count('"""')
        if docstring_count >= 10:
            print(f"  ✓ sam3_complete.py has {docstring_count // 2} docstrings")
        else:
            print(f"  ⚠ sam3_complete.py has few docstrings ({docstring_count // 2})")

        return True

    except Exception as e:
        print(f"  ✗ Error in code quality check: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("  SAM3 Complete Tool - Structure Verification")
    print("=" * 70)

    tests = [
        ("File Structure", test_file_exists),
        ("sam3_complete.py", test_sam3_complete_structure),
        ("sam3_gui.py", test_sam3_gui_structure),
        ("Documentation", test_documentation),
        ("Install Script", test_install_script),
        ("Code Quality", test_code_quality)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_name, result in results:
        if result:
            print(f"  ✓ {test_name}: PASSED")
            passed += 1
        else:
            print(f"  ✗ {test_name}: FAILED")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 All structure tests passed!")
        print("\nNext steps:")
        print("  1. Install SAM3: ./install_sam3.sh")
        print("  2. Authenticate: hf auth login")
        print("  3. Test with image: python sam3_complete.py image test.jpg --text 'object'")
        print("  4. Launch GUI: python sam3_gui.py")
        return 0
    else:
        print("\n⚠ Some tests failed. Review the code.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
