"""
GUI for Ultimate Rotoscopy
===========================

Gradio-based web interface for the Ultimate Rotoscopy application.
"""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Gradio import with fallback
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


def launch():
    """Launch the Gradio web interface."""
    if not GRADIO_AVAILABLE:
        print("Gradio is not installed. Install with: pip install gradio")
        return

    from ultimate_rotoscopy.core.engine import RotoscopyEngine, EngineConfig, ProcessingMode
    from ultimate_rotoscopy.models.sam3 import SegmentationPrompt, PromptType
    from ultimate_rotoscopy.export.exr_writer import EXRWriter

    # Global engine instance
    engine: Optional[RotoscopyEngine] = None
    current_image: Optional[np.ndarray] = None
    points: List[Tuple[int, int]] = []
    labels: List[int] = []

    def initialize_engine(quality: str, device: str):
        """Initialize the rotoscopy engine."""
        nonlocal engine

        mode_map = {
            "Fast": ProcessingMode.FAST,
            "Balanced": ProcessingMode.BALANCED,
            "Quality": ProcessingMode.QUALITY,
            "Maximum": ProcessingMode.MAXIMUM,
        }

        config = EngineConfig(
            processing_mode=mode_map.get(quality, ProcessingMode.BALANCED),
            device=device,
        )

        if engine is not None:
            engine.unload_models()

        engine = RotoscopyEngine(config)

        return "Engine initialized successfully!"

    def process_image(
        image: np.ndarray,
        generate_depth: bool,
        generate_normals: bool,
        generate_matte: bool,
    ):
        """Process the uploaded image."""
        nonlocal current_image, points, labels

        if engine is None:
            return None, None, None, None, "Please initialize the engine first!"

        current_image = image
        points.clear()
        labels.clear()

        # Create prompt if points exist
        prompt = None
        if len(points) > 0:
            prompt = SegmentationPrompt(
                prompt_type=PromptType.POINT,
                points=np.array(points),
                point_labels=np.array(labels),
            )

        # Process
        result = engine.process(
            image,
            prompt=prompt,
            generate_depth=generate_depth,
            generate_normals=generate_normals,
            generate_matte=generate_matte,
        )

        # Prepare outputs
        alpha_vis = None
        depth_vis = None
        normal_vis = None
        foreground = None

        if result.alpha is not None:
            alpha_vis = (result.alpha * 255).astype(np.uint8)

        if result.depth_normalized is not None:
            # Apply colormap
            import cv2
            depth_colored = cv2.applyColorMap(
                (result.depth_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_VIRIDIS
            )
            depth_vis = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        if result.normals is not None:
            normal_vis = ((result.normals + 1) / 2 * 255).astype(np.uint8)

        if result.foreground is not None:
            if result.foreground.max() <= 1:
                foreground = (result.foreground * 255).astype(np.uint8)
            else:
                foreground = result.foreground.astype(np.uint8)

        status = f"Processing complete! Time: {result.processing_time_ms:.1f}ms"

        return alpha_vis, depth_vis, normal_vis, foreground, status

    def add_point(image: np.ndarray, evt: gr.SelectData, is_foreground: bool):
        """Add a point to the image."""
        nonlocal points, labels

        x, y = evt.index
        points.append((x, y))
        labels.append(1 if is_foreground else 0)

        # Draw point on image
        result_image = image.copy()

        import cv2
        for (px, py), label in zip(points, labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(result_image, (px, py), 5, color, -1)

        return result_image, f"Points: {len(points)} (FG: {sum(labels)}, BG: {len(labels) - sum(labels)})"

    def clear_points(image: np.ndarray):
        """Clear all points."""
        nonlocal points, labels

        points.clear()
        labels.clear()

        return image, "Points cleared"

    def segment_with_points():
        """Run segmentation with current points."""
        if engine is None or current_image is None:
            return None, "Please load an image and initialize the engine first!"

        if len(points) == 0:
            return None, "Please add at least one point!"

        prompt = SegmentationPrompt(
            prompt_type=PromptType.POINT,
            points=np.array(points),
            point_labels=np.array(labels),
        )

        result = engine.process(
            current_image,
            prompt=prompt,
            generate_depth=False,
            generate_normals=False,
            generate_matte=True,
        )

        # Create visualization
        if result.alpha is not None:
            # Overlay alpha on image
            alpha_colored = np.zeros_like(current_image)
            alpha_colored[..., 1] = (result.alpha * 255).astype(np.uint8)

            overlay = (current_image * 0.7 + alpha_colored * 0.3).astype(np.uint8)
            return overlay, f"Segmentation complete! Time: {result.processing_time_ms:.1f}ms"

        return None, "Segmentation failed"

    def export_exr(
        alpha: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        normals: Optional[np.ndarray],
    ):
        """Export results to EXR."""
        if alpha is None and depth is None and normals is None:
            return None, "No results to export!"

        writer = EXRWriter()

        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            output_path = f.name

        aovs = {}
        if alpha is not None:
            aovs["alpha"] = alpha.astype(np.float32) / 255.0
        if depth is not None:
            aovs["depth"] = depth.astype(np.float32) / 255.0
        if normals is not None:
            aovs["normals"] = normals.astype(np.float32) / 255.0

        writer.write_multilayer(output_path, aovs)

        return output_path, f"Exported to: {output_path}"

    # Build the interface
    with gr.Blocks(title="Ultimate Rotoscopy", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Ultimate Rotoscopy
        AI-Powered Professional VFX Tool

        Leveraging SAM3, Depth Anything V3, and Matte Anything for professional rotoscopy workflows.
        """)

        with gr.Tab("Setup"):
            with gr.Row():
                quality_dropdown = gr.Dropdown(
                    choices=["Fast", "Balanced", "Quality", "Maximum"],
                    value="Balanced",
                    label="Quality Mode"
                )
                device_dropdown = gr.Dropdown(
                    choices=["cuda", "cpu", "mps"],
                    value="cuda",
                    label="Device"
                )
                init_btn = gr.Button("Initialize Engine", variant="primary")

            init_status = gr.Textbox(label="Status", interactive=False)
            init_btn.click(initialize_engine, [quality_dropdown, device_dropdown], [init_status])

        with gr.Tab("Process Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="numpy")

                    with gr.Row():
                        depth_check = gr.Checkbox(value=True, label="Depth")
                        normals_check = gr.Checkbox(value=True, label="Normals")
                        matte_check = gr.Checkbox(value=True, label="Matte")

                    process_btn = gr.Button("Process", variant="primary")

                with gr.Column():
                    with gr.Row():
                        alpha_output = gr.Image(label="Alpha Matte")
                        depth_output = gr.Image(label="Depth Map")

                    with gr.Row():
                        normals_output = gr.Image(label="Normal Map")
                        fg_output = gr.Image(label="Foreground")

            process_status = gr.Textbox(label="Status", interactive=False)

            process_btn.click(
                process_image,
                [input_image, depth_check, normals_check, matte_check],
                [alpha_output, depth_output, normals_output, fg_output, process_status]
            )

        with gr.Tab("Interactive Segmentation"):
            gr.Markdown("""
            Click on the image to add foreground points (green) or background points (red).
            Use the radio button to switch between foreground and background mode.
            """)

            with gr.Row():
                with gr.Column():
                    seg_input = gr.Image(label="Click to add points", type="numpy")
                    with gr.Row():
                        point_mode = gr.Radio(
                            choices=["Foreground", "Background"],
                            value="Foreground",
                            label="Point Mode"
                        )
                        clear_btn = gr.Button("Clear Points")
                        segment_btn = gr.Button("Segment", variant="primary")

                with gr.Column():
                    seg_output = gr.Image(label="Segmentation Result")

            point_status = gr.Textbox(label="Status", interactive=False)

            seg_input.select(
                lambda img, evt, mode: add_point(img, evt, mode == "Foreground"),
                [seg_input, point_mode],
                [seg_input, point_status]
            )

            clear_btn.click(clear_points, [seg_input], [seg_input, point_status])
            segment_btn.click(segment_with_points, [], [seg_output, point_status])

        with gr.Tab("Export"):
            gr.Markdown("Export results to EXR format for use in Flame, Nuke, or other compositing software.")

            with gr.Row():
                export_btn = gr.Button("Export to EXR", variant="primary")
                export_file = gr.File(label="Download EXR")

            export_status = gr.Textbox(label="Status", interactive=False)

            export_btn.click(
                export_exr,
                [alpha_output, depth_output, normals_output],
                [export_file, export_status]
            )

        with gr.Tab("About"):
            gr.Markdown("""
            ## Ultimate Rotoscopy

            ### AI Models
            - **SAM3** (Segment Anything Model 3) - State-of-the-art object segmentation
            - **Depth Anything V3** - Monocular depth estimation
            - **Matte Anything** - Professional alpha matting

            ### Features
            - Hair and fine detail matting
            - Edge-aware segmentation
            - Motion blur handling
            - Depth and normal map generation
            - 3D point cloud export
            - Flame/Nuke compatible EXR export

            ### Keyboard Shortcuts
            - Click to add foreground points
            - Shift+Click to add background points

            ### Output Formats
            - OpenEXR (multi-layer, 16/32-bit float)
            - PNG (8/16-bit)
            - TIFF (16-bit)

            ---
            Version 1.0.0
            """)

    return demo


def main():
    """Main entry point for the GUI."""
    if not GRADIO_AVAILABLE:
        print("Error: Gradio is not installed.")
        print("Install it with: pip install gradio>=4.0.0")
        return

    demo = launch()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
