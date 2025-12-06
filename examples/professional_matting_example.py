"""
Professional Matting Example
=============================

Demonstrates the advanced alpha split and motion blur-aware matting system.

This example shows:
1. Loading an alpha matte
2. Splitting into core/edge/hair components
3. Detecting and handling motion blur
4. Exporting all layers for compositing
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultimate_rotoscopy.matting import (
    ProfessionalMatting,
    ProfessionalMattingConfig,
    AlphaSplitConfig,
    MotionBlurConfig,
    visualize_alpha_split,
    visualize_motion_blur_result,
)


def create_sample_alpha(size=(512, 512)):
    """Create a sample alpha matte for testing."""
    h, w = size
    alpha = np.zeros((h, w), dtype=np.float32)

    # Create center circle with gradient edge
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 3

    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < radius * 0.7:
                # Core: solid
                alpha[y, x] = 1.0
            elif dist < radius:
                # Edge: gradient
                alpha[y, x] = 1.0 - (dist - radius * 0.7) / (radius * 0.3)
            else:
                # Outside
                alpha[y, x] = 0.0

    # Add hair-like details (random thin strands)
    for i in range(50):
        angle = np.random.rand() * 2 * np.pi
        strand_length = radius * 0.3
        start_x = center_x + int(radius * 0.9 * np.cos(angle))
        start_y = center_y + int(radius * 0.9 * np.sin(angle))
        end_x = start_x + int(strand_length * np.cos(angle))
        end_y = start_y + int(strand_length * np.sin(angle))

        cv2.line(alpha, (start_x, start_y), (end_x, end_y), 0.8, 1)

    return alpha


def create_sample_image(size=(512, 512)):
    """Create a sample RGB image."""
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.float32)

    # Gradient background
    for y in range(h):
        image[y, :, 0] = y / h  # Red gradient
        image[y, :, 1] = 0.5    # Constant green
        image[y, :, 2] = 1 - y / h  # Blue gradient

    # Add some texture
    noise = np.random.randn(h, w, 3) * 0.1
    image = np.clip(image + noise, 0, 1)

    return image


def main():
    """Run professional matting example."""
    print("=" * 70)
    print("Professional Matting Example")
    print("=" * 70)

    # Create output directory
    output_dir = Path("output/professional_matting")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample data
    print("\n1. Generating sample alpha and image...")
    alpha = create_sample_alpha((512, 512))
    image = create_sample_image((512, 512))

    # Add some motion blur to simulate fast motion
    alpha_blurred = cv2.GaussianBlur(alpha, (11, 11), 0)
    alpha_with_blur = 0.7 * alpha + 0.3 * alpha_blurred

    # Save input
    cv2.imwrite(str(output_dir / "input_alpha.png"), (alpha_with_blur * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "input_image.png"), (image * 255).astype(np.uint8))

    print(f"   ✓ Sample data created: {alpha.shape}")

    # Configure professional matting
    print("\n2. Configuring professional matting pipeline...")
    config = ProfessionalMattingConfig(
        alpha_split=AlphaSplitConfig(
            core_threshold_low=0.15,
            core_erosion_size=2,
            edge_band_width=10,
            hair_threshold=0.05,
        ),
        motion_blur=MotionBlurConfig(
            laplacian_threshold=100.0,
            use_optical_flow=False,  # No previous frame for single image
        ),
        enable_motion_blur=True,
        export_all_layers=True,
        export_format="png",
        output_bit_depth=16,
    )

    # Initialize matting system
    matting = ProfessionalMatting(config)
    print("   ✓ Pipeline configured")

    # Process frame
    print("\n3. Processing alpha through professional matting...")
    result = matting.process_frame(alpha_with_blur, image)

    print(f"   ✓ Alpha split completed:")
    print(f"     - Core coverage: {result.alpha_core.sum() / alpha.size * 100:.1f}%")
    print(f"     - Edge coverage: {result.alpha_edge.sum() / alpha.size * 100:.1f}%")
    print(f"     - Hair coverage: {result.alpha_hair.sum() / alpha.size * 100:.1f}%")

    if result.has_motion_blur:
        print(f"   ✓ Motion blur detected:")
        print(f"     - Level: {result.blur_level.value}")
        print(f"     - Percentage: {result.blur_percentage:.1f}%")

    # Export all layers
    print("\n4. Exporting all layers...")
    exported = matting.export_layers(result, str(output_dir / "shot_001"))

    for layer_name, filepath in exported.items():
        print(f"   ✓ {layer_name}: {Path(filepath).name}")

    # Create visualizations
    print("\n5. Creating visualizations...")

    # Visualization 1: Alpha split components side-by-side
    vis_split = np.hstack([
        result.alpha_core,
        result.alpha_edge,
        result.alpha_hair,
        result.alpha_final,
    ])
    cv2.imwrite(
        str(output_dir / "visualization_split.png"),
        (vis_split * 255).astype(np.uint8)
    )
    print("   ✓ Alpha split visualization saved")

    # Visualization 2: Motion blur processing (if available)
    if result.has_motion_blur:
        vis_motion = np.hstack([
            result.alpha_sharp,
            result.alpha_motion_blur,
            result.alpha_final,
            result.blur_mask,
        ])
        cv2.imwrite(
            str(output_dir / "visualization_motion_blur.png"),
            (vis_motion * 255).astype(np.uint8)
        )
        print("   ✓ Motion blur visualization saved")

    # Visualization 3: Color-coded components
    vis_color = np.zeros((512, 512, 3), dtype=np.float32)
    vis_color[:, :, 0] = result.alpha_core   # Red = Core
    vis_color[:, :, 1] = result.alpha_edge   # Green = Edge
    vis_color[:, :, 2] = result.alpha_hair   # Blue = Hair
    cv2.imwrite(
        str(output_dir / "visualization_color_coded.png"),
        (vis_color * 255).astype(np.uint8)
    )
    print("   ✓ Color-coded visualization saved")

    # Try multi-layer EXR export (if OpenEXR available)
    print("\n6. Attempting multi-layer EXR export...")
    try:
        exr_path = matting.export_multi_layer_exr(
            result,
            str(output_dir / "shot_001_multilayer.exr")
        )
        print(f"   ✓ Multi-layer EXR exported: {Path(exr_path).name}")
    except Exception as e:
        print(f"   ⚠ Multi-layer EXR export failed: {e}")
        print("   ℹ Individual layers exported as PNG instead")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Layers exported: {len(exported)}")
    print("\nFiles created:")
    for layer_name, filepath in sorted(exported.items()):
        print(f"  - {layer_name}: {Path(filepath).name}")

    print("\nVisualization files:")
    print("  - visualization_split.png (core/edge/hair/final)")
    if result.has_motion_blur:
        print("  - visualization_motion_blur.png (sharp/blur/final/mask)")
    print("  - visualization_color_coded.png (RGB color-coded components)")

    print("\n✓ Professional matting example completed successfully!")
    print(f"\nView results: {output_dir}")


if __name__ == "__main__":
    main()
