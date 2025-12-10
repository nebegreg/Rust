"""
CLI for Ultimate Rotoscopy
===========================

Command-line interface for the Ultimate Rotoscopy application.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


def parse_points(points_str: str) -> np.ndarray:
    """Parse points from string format 'x1,y1;x2,y2;...'"""
    if not points_str:
        return None

    points = []
    for point in points_str.split(";"):
        x, y = map(float, point.split(","))
        points.append([x, y])

    return np.array(points)


def parse_box(box_str: str) -> np.ndarray:
    """Parse box from string format 'x1,y1,x2,y2'"""
    if not box_str:
        return None

    coords = list(map(float, box_str.split(",")))
    if len(coords) != 4:
        raise ValueError("Box must have 4 coordinates: x1,y1,x2,y2")

    return np.array([coords])


@click.group()
@click.version_option(version="1.0.0", prog_name="Ultimate Rotoscopy")
def main():
    """
    Ultimate Rotoscopy - AI-Powered Professional VFX Tool

    A comprehensive rotoscopy application leveraging SAM3, Depth Anything V3,
    and Matte Anything for professional VFX workflows.
    """
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.option("-p", "--points", type=str, help="Foreground points: 'x1,y1;x2,y2;...'")
@click.option("-b", "--box", type=str, help="Bounding box: 'x1,y1,x2,y2'")
@click.option("--format", type=click.Choice(["exr", "png", "tiff"]), default="exr", help="Output format")
@click.option("--quality", type=click.Choice(["fast", "balanced", "quality", "maximum"]), default="balanced")
@click.option("--depth/--no-depth", default=True, help="Generate depth map")
@click.option("--normals/--no-normals", default=True, help="Generate normal map")
@click.option("--matte/--no-matte", default=True, help="Generate alpha matte")
@click.option("--aovs/--no-aovs", default=True, help="Generate AOV package")
@click.option("--device", type=str, default="cuda", help="Compute device (cuda, cpu, mps)")
def process(
    input_path: str,
    output: Optional[str],
    points: Optional[str],
    box: Optional[str],
    format: str,
    quality: str,
    depth: bool,
    normals: bool,
    matte: bool,
    aovs: bool,
    device: str,
):
    """
    Process a single image through the rotoscopy pipeline.

    Examples:

        # Process with point prompt
        rotoscopy process image.jpg -p "100,200;150,250"

        # Process with bounding box
        rotoscopy process image.jpg -b "50,50,200,200"

        # High quality processing to EXR
        rotoscopy process image.jpg --quality quality --format exr
    """
    from ultimate_rotoscopy.pipeline.unified import UnifiedPipeline, PipelineConfig
    from ultimate_rotoscopy.core.engine import ProcessingMode

    input_path = Path(input_path)

    # Set output directory
    if output:
        output_dir = Path(output)
    else:
        output_dir = input_path.parent / "output"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse quality mode
    mode_map = {
        "fast": ProcessingMode.FAST,
        "balanced": ProcessingMode.BALANCED,
        "quality": ProcessingMode.QUALITY,
        "maximum": ProcessingMode.MAXIMUM,
    }

    # Parse prompts
    point_array = parse_points(points) if points else None
    box_array = parse_box(box) if box else None

    # Create pipeline config
    config = PipelineConfig(
        processing_mode=mode_map[quality],
        device=device,
        output_directory=output_dir,
        output_format=format,
        generate_depth=depth,
        generate_normals=normals,
        generate_matte=matte,
        generate_aovs=aovs,
    )

    console.print(Panel.fit(
        f"[bold blue]Ultimate Rotoscopy[/bold blue]\n"
        f"Processing: {input_path.name}\n"
        f"Quality: {quality}\n"
        f"Device: {device}",
        title="Configuration"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Load models
        task = progress.add_task("Loading models...", total=1)
        pipeline = UnifiedPipeline(config)
        progress.update(task, completed=1)

        # Process image
        task = progress.add_task("Processing image...", total=1)
        result = pipeline.process_image(
            input_path,
            points=point_array,
            boxes=box_array,
            save_output=True,
            output_name=input_path.stem,
        )
        progress.update(task, completed=1)

    # Display results
    table = Table(title="Processing Results")
    table.add_column("Output", style="cyan")
    table.add_column("Status", style="green")

    if result.alpha is not None:
        table.add_row("Alpha Matte", "Generated")
    if result.depth_map is not None:
        table.add_row("Depth Map", "Generated")
    if result.normals is not None:
        table.add_row("Normal Map", "Generated")
    if result.foreground is not None:
        table.add_row("Foreground", "Generated")

    table.add_row("Processing Time", f"{result.processing_time_ms:.1f}ms")
    table.add_row("Output Directory", str(output_dir))

    console.print(table)


@main.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.option("-p", "--points", type=str, help="Foreground points: 'x1,y1;x2,y2;...'")
@click.option("-b", "--box", type=str, help="Bounding box: 'x1,y1,x2,y2'")
@click.option("--pattern", type=str, default="*.png", help="File pattern for frames")
@click.option("--start", type=int, default=0, help="Start frame")
@click.option("--end", type=int, default=None, help="End frame (inclusive)")
@click.option("--format", type=click.Choice(["exr", "png", "tiff"]), default="exr")
@click.option("--quality", type=click.Choice(["fast", "balanced", "quality", "maximum"]), default="balanced")
@click.option("--clip-name", type=str, default="rotoscopy", help="Clip name for output")
@click.option("--flame", is_flag=True, help="Generate Flame-compatible output")
def sequence(
    input_dir: str,
    output: Optional[str],
    points: Optional[str],
    box: Optional[str],
    pattern: str,
    start: int,
    end: Optional[int],
    format: str,
    quality: str,
    clip_name: str,
    flame: bool,
):
    """
    Process a sequence of frames.

    Examples:

        # Process all PNG frames
        rotoscopy sequence frames/ -p "100,200"

        # Process specific range with Flame export
        rotoscopy sequence frames/ --start 1 --end 100 --flame

        # Use custom pattern
        rotoscopy sequence frames/ --pattern "frame_*.exr"
    """
    from ultimate_rotoscopy.pipeline.unified import UnifiedPipeline, PipelineConfig
    from ultimate_rotoscopy.export.flame_export import FlameExporter
    from ultimate_rotoscopy.core.engine import ProcessingMode

    input_dir = Path(input_dir)

    if output:
        output_dir = Path(output)
    else:
        output_dir = input_dir.parent / "output"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse quality mode
    mode_map = {
        "fast": ProcessingMode.FAST,
        "balanced": ProcessingMode.BALANCED,
        "quality": ProcessingMode.QUALITY,
        "maximum": ProcessingMode.MAXIMUM,
    }

    # Parse prompts
    point_array = parse_points(points) if points else None
    box_array = parse_box(box) if box else None

    # Get frame list
    frames = sorted(input_dir.glob(pattern))
    if end is None:
        end = len(frames) - 1

    frames = frames[start:end + 1]
    total_frames = len(frames)

    console.print(Panel.fit(
        f"[bold blue]Ultimate Rotoscopy - Sequence Processing[/bold blue]\n"
        f"Input: {input_dir}\n"
        f"Frames: {start} - {end} ({total_frames} frames)\n"
        f"Quality: {quality}",
        title="Configuration"
    ))

    # Create pipeline config
    config = PipelineConfig(
        processing_mode=mode_map[quality],
        output_directory=output_dir,
        output_format=format,
        output_prefix=clip_name,
        temporal_consistency=True,
    )

    # Initialize Flame exporter if needed
    flame_exporter = None
    if flame:
        flame_exporter = FlameExporter(
            output_dir=output_dir,
            clip_name=clip_name,
        )

    pipeline = UnifiedPipeline(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing sequence...", total=total_frames)

        for result in pipeline.process_sequence(
            frames,
            points=point_array,
            boxes=box_array,
            start_frame=start,
            end_frame=end,
        ):
            # Export to Flame format
            if flame_exporter:
                frame_idx = result.metadata.get("frame_index", 0)
                flame_exporter.export_frame(result, frame_idx)

            progress.update(task, advance=1)

    # Finalize Flame export
    if flame_exporter:
        files = flame_exporter.finalize()
        console.print("\n[bold green]Flame Export Complete[/bold green]")
        for name, path in files.items():
            console.print(f"  {name}: {path}")

    stats = pipeline.get_statistics()
    console.print(f"\n[bold]Processing complete![/bold]")
    console.print(f"  Frames processed: {stats['session']['processed_frames']}")
    console.print(f"  Output: {output_dir}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="Output file path")
@click.option("--format", type=click.Choice(["ply", "obj", "xyz"]), default="ply")
@click.option("--with-colors", is_flag=True, help="Include RGB colors in point cloud")
def pointcloud(
    input_path: str,
    output: str,
    format: str,
    with_colors: bool,
):
    """
    Generate 3D point cloud from an image.

    Examples:

        # Export to PLY with colors
        rotoscopy pointcloud image.jpg -o output.ply --with-colors

        # Export to OBJ
        rotoscopy pointcloud image.jpg -o output.obj --format obj
    """
    from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3, DepthConfig
    from PIL import Image

    console.print("[bold]Generating point cloud...[/bold]")

    # Load image
    image = np.array(Image.open(input_path))

    # Initialize depth model
    config = DepthConfig(generate_normals=False)
    depth_model = DepthAnythingV3(config)

    with console.status("Loading model..."):
        depth_model.load()

    # Set camera intrinsics (estimate from image size)
    h, w = image.shape[:2]
    fx = fy = max(h, w)
    cx, cy = w / 2, h / 2
    depth_model.set_camera_intrinsics(fx, fy, cx, cy)

    with console.status("Estimating depth..."):
        result = depth_model.estimate_depth(image, generate_point_cloud=True)

    # Get colors if requested
    colors = None
    if with_colors:
        colors = image.reshape(-1, 3)
        # Filter to match point cloud
        valid_mask = result.depth_map.reshape(-1) > 0
        colors = colors[valid_mask]

    # Export
    with console.status("Exporting point cloud..."):
        depth_model.export_point_cloud(
            result.point_cloud,
            colors,
            Path(output),
            format=format,
        )

    console.print(f"[bold green]Point cloud exported to:[/bold green] {output}")
    console.print(f"  Points: {len(result.point_cloud):,}")


@main.command()
@click.option("--device", type=str, default="cuda", help="Device to test")
def test(device: str):
    """
    Test the installation and model loading.
    """
    console.print("[bold]Testing Ultimate Rotoscopy installation...[/bold]\n")

    tests = []

    # Test PyTorch
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        tests.append(("PyTorch", True, f"v{torch.__version__}"))
        tests.append(("CUDA", gpu_available, f"{'Available' if gpu_available else 'Not available'}"))
    except ImportError:
        tests.append(("PyTorch", False, "Not installed"))

    # Test transformers
    try:
        import transformers
        tests.append(("Transformers", True, f"v{transformers.__version__}"))
    except ImportError:
        tests.append(("Transformers", False, "Not installed"))

    # Test OpenEXR
    try:
        import OpenEXR
        tests.append(("OpenEXR", True, "Available"))
    except ImportError:
        tests.append(("OpenEXR", False, "Not installed"))

    # Test models
    try:
        from ultimate_rotoscopy.models.sam3 import SAM3Segmentor, SAM3Config
        tests.append(("SAM3 Module", True, "OK"))
    except Exception as e:
        tests.append(("SAM3 Module", False, str(e)))

    try:
        from ultimate_rotoscopy.models.depth_anything import DepthAnythingV3, DepthConfig
        tests.append(("Depth Anything", True, "OK"))
    except Exception as e:
        tests.append(("Depth Anything", False, str(e)))

    try:
        from ultimate_rotoscopy.models.matte_anything import MatteAnything, MatteConfig
        tests.append(("Matte Anything", True, "OK"))
    except Exception as e:
        tests.append(("Matte Anything", False, str(e)))

    # Display results
    table = Table(title="Installation Test Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Info")

    for name, success, info in tests:
        status = "[green]OK" if success else "[red]Failed"
        table.add_row(name, status, info)

    console.print(table)

    all_ok = all(t[1] for t in tests)
    if all_ok:
        console.print("\n[bold green]All tests passed![/bold green]")
    else:
        console.print("\n[bold yellow]Some tests failed. Check the dependencies.[/bold yellow]")


@main.command()
def info():
    """
    Display information about Ultimate Rotoscopy.
    """
    console.print(Panel.fit(
        """[bold blue]Ultimate Rotoscopy[/bold blue]
Version: 1.0.0

[bold]AI Models:[/bold]
  - SAM3 (Segment Anything Model 3) - Object segmentation
  - Depth Anything V3 - Monocular depth estimation
  - Matte Anything - Professional alpha matting

[bold]Features:[/bold]
  - Hair and fine detail matting
  - Edge-aware segmentation
  - Motion blur handling
  - Depth and normal map generation
  - 3D point cloud export
  - Flame/Nuke compatible EXR export
  - Multi-layer AOV support

[bold]Supported Output Formats:[/bold]
  - OpenEXR (multi-layer, 16/32-bit float)
  - PNG (8/16-bit)
  - TIFF (16-bit)
  - PLY/OBJ/XYZ (point clouds)

[bold]Documentation:[/bold]
  https://github.com/ultimate-rotoscopy/ultimate-rotoscopy
""",
        title="About"
    ))


if __name__ == "__main__":
    main()
