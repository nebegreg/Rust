#!/usr/bin/env python3
"""
Ultimate Rotoscopy CLI
======================

Professional command-line interface for cinema-quality rotoscopy.

Usage:
    # Automatic segmentation with text prompts
    ultimate-roto process video.mp4 -o output/ --prompt "person in foreground"

    # Process image sequence
    ultimate-roto process input/*.exr -o output/ --prompt "main character"

    # Keyframe-based workflow
    ultimate-roto keyframe video.mp4 --keyframe 0:mask_0.png --keyframe 100:mask_100.png

    # Batch processing
    ultimate-roto batch shots.json -o renders/

    # Interactive mode
    ultimate-roto interactive video.mp4
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import json

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_pipeline():
    """Load the rotoscopy pipeline."""
    from ultimate_rotoscopy.pipeline import (
        UltimateRotoPipeline,
        RotoConfig,
        RotoMode,
        EdgeMode,
        OutputFormat,
    )
    return UltimateRotoPipeline, RotoConfig, RotoMode, EdgeMode, OutputFormat


def load_frames(input_path: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load frames from video or image sequence."""
    import cv2

    input_path = Path(input_path)
    frames = []
    frame_paths = []

    if input_path.is_file():
        # Video file
        if input_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            cap = cv2.VideoCapture(str(input_path))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_paths.append(f"{input_path.stem}_{frame_idx:04d}")
                frame_idx += 1

            cap.release()
        else:
            # Single image
            frame = cv2.imread(str(input_path))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_paths.append(input_path.stem)
    else:
        # Glob pattern - image sequence
        import glob
        paths = sorted(glob.glob(str(input_path)))

        for path in paths:
            frame = cv2.imread(path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_paths.append(Path(path).stem)

    return frames, frame_paths


@click.group()
@click.version_option(version="3.0.0", prog_name="Ultimate Rotoscopy")
def cli():
    """
    Ultimate Rotoscopy - Cinema-Quality AI Rotoscopy Tool

    Professional rotoscopy pipeline integrating SAM3 and Depth Anything 3
    for feature film quality matte extraction.
    """
    pass


@cli.command()
@click.argument('input_path')
@click.option('-o', '--output', 'output_dir', required=True, help='Output directory')
@click.option('-p', '--prompt', 'prompts', multiple=True, help='Text prompts for segmentation')
@click.option('--mode', type=click.Choice(['automatic', 'semi_auto', 'tracking']), default='automatic')
@click.option('--edge', type=click.Choice(['hard', 'soft', 'motion_aware', 'hair_detail', 'adaptive']), default='adaptive')
@click.option('--format', 'output_format', type=click.Choice(['exr_16', 'exr_32', 'png_16', 'png_8', 'dpx']), default='exr_16')
@click.option('--sam-size', type=click.Choice(['tiny', 'small', 'base', 'large']), default='large')
@click.option('--depth-size', type=click.Choice(['small', 'base', 'large', 'giant']), default='large')
@click.option('--feather', type=float, default=0.0, help='Edge feather radius')
@click.option('--temporal-smooth', type=float, default=0.85, help='Temporal smoothing (0-1)')
@click.option('--start-frame', type=int, default=0, help='Start frame number')
@click.option('--end-frame', type=int, default=-1, help='End frame number (-1 = all)')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration')
@click.option('--batch-size', type=int, default=4, help='Processing batch size')
def process(
    input_path: str,
    output_dir: str,
    prompts: Tuple[str],
    mode: str,
    edge: str,
    output_format: str,
    sam_size: str,
    depth_size: str,
    feather: float,
    temporal_smooth: float,
    start_frame: int,
    end_frame: int,
    gpu: bool,
    batch_size: int,
):
    """
    Process video or image sequence for rotoscopy.

    Examples:

        ultimate-roto process video.mp4 -o output/ -p "person"

        ultimate-roto process "frames/*.exr" -o mattes/ -p "character" --edge hair_detail
    """
    UltimateRotoPipeline, RotoConfig, RotoMode, EdgeMode, OutputFormat = load_pipeline()

    console.print(Panel.fit(
        "[bold blue]Ultimate Rotoscopy[/bold blue]\n"
        "Cinema-Quality AI Rotoscopy",
        border_style="blue"
    ))

    # Load frames
    console.print("\n[yellow]Loading frames...[/yellow]")
    frames, frame_paths = load_frames(input_path)

    if not frames:
        console.print("[red]Error: No frames found![/red]")
        sys.exit(1)

    console.print(f"[green]Loaded {len(frames)} frames[/green]")

    # Apply frame range
    if end_frame > 0:
        frames = frames[start_frame:end_frame]
        frame_paths = frame_paths[start_frame:end_frame]
    elif start_frame > 0:
        frames = frames[start_frame:]
        frame_paths = frame_paths[start_frame:]

    # Configure pipeline
    mode_map = {
        'automatic': RotoMode.AUTOMATIC,
        'semi_auto': RotoMode.SEMI_AUTO,
        'tracking': RotoMode.TRACKING,
    }

    edge_map = {
        'hard': EdgeMode.HARD,
        'soft': EdgeMode.SOFT,
        'motion_aware': EdgeMode.MOTION_AWARE,
        'hair_detail': EdgeMode.HAIR_DETAIL,
        'adaptive': EdgeMode.ADAPTIVE,
    }

    format_map = {
        'exr_16': OutputFormat.EXR_16,
        'exr_32': OutputFormat.EXR_32,
        'png_16': OutputFormat.PNG_16,
        'png_8': OutputFormat.PNG_8,
        'dpx': OutputFormat.DPX,
    }

    config = RotoConfig(
        mode=mode_map[mode],
        edge_mode=edge_map[edge],
        sam_model_size=sam_size,
        depth_model_size=depth_size,
        output_format=format_map[output_format],
        feather_radius=feather,
        temporal_smoothing=temporal_smooth,
        use_gpu=gpu,
        batch_size=batch_size,
    )

    # Create pipeline
    pipeline = UltimateRotoPipeline(config)

    # Load models
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading AI models...", total=None)

        def model_progress(pct, msg):
            progress.update(task, description=f"[cyan]{msg}")

        pipeline.load_models(progress_callback=model_progress)

    # Process frames
    console.print("\n[yellow]Processing frames...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(frames))

        def process_progress(pct, msg):
            progress.update(task, completed=int(pct * len(frames)), description=f"[cyan]{msg}")

        results = pipeline.process_sequence(
            frames,
            text_prompts=list(prompts) if prompts else None,
            start_frame=start_frame,
            progress_callback=process_progress,
        )

    # Export results
    console.print("\n[yellow]Exporting results...[/yellow]")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Exporting...", total=len(results))

        def export_progress(pct, msg):
            progress.update(task, completed=int(pct * len(results)))

        output_files = pipeline.export_sequence(
            results,
            output_path,
            base_name="roto",
            progress_callback=export_progress,
        )

    # Summary
    console.print("\n[bold green]Processing complete![/bold green]")

    table = Table(title="Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Frames processed", str(len(results)))
    table.add_row("Objects detected", str(len(pipeline.objects)))
    table.add_row("Output files", str(len(output_files)))
    table.add_row("Output directory", str(output_path))

    console.print(table)

    # Cleanup
    pipeline.unload_models()


@cli.command()
@click.argument('input_path')
@click.option('-o', '--output', 'output_dir', required=True, help='Output directory')
@click.option('-k', '--keyframe', 'keyframes', multiple=True, help='Keyframe in format "frame:mask_path"')
@click.option('--interpolation', type=click.Choice(['linear', 'smooth', 'hold']), default='smooth')
@click.option('--format', 'output_format', type=click.Choice(['exr_16', 'exr_32', 'png_16']), default='exr_16')
def keyframe(
    input_path: str,
    output_dir: str,
    keyframes: Tuple[str],
    interpolation: str,
    output_format: str,
):
    """
    Keyframe-based rotoscopy workflow.

    Define keyframes at important frames and let AI propagate between them.

    Examples:

        ultimate-roto keyframe video.mp4 -o output/ -k "0:mask_0.png" -k "100:mask_100.png"
    """
    UltimateRotoPipeline, RotoConfig, RotoMode, EdgeMode, OutputFormat = load_pipeline()
    import cv2

    console.print(Panel.fit(
        "[bold blue]Keyframe Rotoscopy[/bold blue]",
        border_style="blue"
    ))

    # Parse keyframes
    keyframe_data = []
    for kf in keyframes:
        parts = kf.split(':')
        if len(parts) != 2:
            console.print(f"[red]Invalid keyframe format: {kf}[/red]")
            continue

        frame_num = int(parts[0])
        mask_path = Path(parts[1])

        if not mask_path.exists():
            console.print(f"[red]Mask file not found: {mask_path}[/red]")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            console.print(f"[red]Failed to load mask: {mask_path}[/red]")
            continue

        mask = mask.astype(np.float32) / 255.0
        keyframe_data.append((frame_num, mask))

    if not keyframe_data:
        console.print("[red]No valid keyframes provided![/red]")
        sys.exit(1)

    console.print(f"[green]Loaded {len(keyframe_data)} keyframes[/green]")

    # Load frames
    console.print("\n[yellow]Loading frames...[/yellow]")
    frames, frame_paths = load_frames(input_path)

    if not frames:
        console.print("[red]Error: No frames found![/red]")
        sys.exit(1)

    console.print(f"[green]Loaded {len(frames)} frames[/green]")

    # Configure pipeline
    format_map = {
        'exr_16': OutputFormat.EXR_16,
        'exr_32': OutputFormat.EXR_32,
        'png_16': OutputFormat.PNG_16,
    }

    config = RotoConfig(
        mode=RotoMode.KEYFRAME,
        output_format=format_map[output_format],
    )

    pipeline = UltimateRotoPipeline(config)

    # Load models
    console.print("\n[yellow]Loading AI models...[/yellow]")
    pipeline.load_models()

    # Create object and add keyframes
    obj_id = pipeline.create_object("Main Object")

    for frame_num, mask in keyframe_data:
        pipeline.add_keyframe(
            frame_num,
            mask,
            obj_id,
            interpolation=interpolation,
        )

    # Propagate keyframes
    console.print("\n[yellow]Propagating keyframes...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Propagating...", total=len(frames))

        def prop_progress(pct, msg):
            progress.update(task, completed=int(pct * len(frames)))

        results = pipeline.propagate_keyframes(
            frames,
            progress_callback=prop_progress,
        )

    # Export
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline.export_sequence(results, output_path)

    console.print(f"\n[bold green]Exported {len(results)} frames to {output_path}[/bold green]")

    pipeline.unload_models()


@cli.command()
@click.argument('config_file')
@click.option('-o', '--output', 'output_dir', required=True, help='Output directory')
@click.option('--parallel', type=int, default=1, help='Number of parallel jobs')
def batch(config_file: str, output_dir: str, parallel: int):
    """
    Batch process multiple shots from a configuration file.

    Config file format (JSON):
    {
        "shots": [
            {
                "name": "shot_001",
                "input": "footage/shot_001.mov",
                "prompts": ["character"],
                "start_frame": 0,
                "end_frame": 100
            }
        ],
        "global_config": {
            "edge_mode": "adaptive",
            "output_format": "exr_16"
        }
    }
    """
    UltimateRotoPipeline, RotoConfig, RotoMode, EdgeMode, OutputFormat = load_pipeline()

    console.print(Panel.fit(
        "[bold blue]Batch Processing[/bold blue]",
        border_style="blue"
    ))

    # Load config
    with open(config_file, 'r') as f:
        batch_config = json.load(f)

    shots = batch_config.get('shots', [])
    global_config = batch_config.get('global_config', {})

    if not shots:
        console.print("[red]No shots defined in config![/red]")
        sys.exit(1)

    console.print(f"[green]Processing {len(shots)} shots[/green]")

    # Process each shot
    output_path = Path(output_dir)

    for i, shot in enumerate(shots):
        shot_name = shot.get('name', f'shot_{i:03d}')
        input_path = shot.get('input')
        prompts = shot.get('prompts', [])

        console.print(f"\n[yellow]Processing {shot_name} ({i+1}/{len(shots)})[/yellow]")

        # Load frames
        frames, _ = load_frames(input_path)

        if not frames:
            console.print(f"[red]No frames found for {shot_name}[/red]")
            continue

        # Apply frame range
        start = shot.get('start_frame', 0)
        end = shot.get('end_frame', -1)
        if end > 0:
            frames = frames[start:end]
        elif start > 0:
            frames = frames[start:]

        # Configure and process
        config = RotoConfig()

        pipeline = UltimateRotoPipeline(config)
        pipeline.load_models()

        results = pipeline.process_sequence(frames, text_prompts=prompts, start_frame=start)

        # Export
        shot_output = output_path / shot_name
        shot_output.mkdir(parents=True, exist_ok=True)

        pipeline.export_sequence(results, shot_output, base_name=shot_name)

        pipeline.unload_models()

        console.print(f"[green]Completed {shot_name}[/green]")

    console.print(f"\n[bold green]Batch processing complete![/bold green]")


@cli.command()
@click.argument('input_path')
def info(input_path: str):
    """
    Display information about input video/sequence.
    """
    import cv2

    console.print(Panel.fit(
        "[bold blue]Media Information[/bold blue]",
        border_style="blue"
    ))

    input_path = Path(input_path)

    if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
        # Video file
        cap = cv2.VideoCapture(str(input_path))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        table = Table(title=str(input_path))
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Type", "Video")
        table.add_row("Resolution", f"{width}x{height}")
        table.add_row("Frame Rate", f"{fps:.2f} fps")
        table.add_row("Frame Count", str(frame_count))
        table.add_row("Duration", f"{duration:.2f} seconds")

        console.print(table)

    else:
        # Image sequence
        import glob
        paths = sorted(glob.glob(str(input_path)))

        if paths:
            first_frame = cv2.imread(paths[0])
            if first_frame is not None:
                height, width = first_frame.shape[:2]

                table = Table(title="Image Sequence")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Type", "Image Sequence")
                table.add_row("Frame Count", str(len(paths)))
                table.add_row("Resolution", f"{width}x{height}")
                table.add_row("First Frame", paths[0])
                table.add_row("Last Frame", paths[-1])

                console.print(table)
        else:
            console.print(f"[red]No files found matching: {input_path}[/red]")


@cli.command()
def models():
    """
    List available AI models and their requirements.
    """
    console.print(Panel.fit(
        "[bold blue]Available Models[/bold blue]",
        border_style="blue"
    ))

    # SAM3 models
    table = Table(title="SAM3 Segmentation Models")
    table.add_column("Size", style="cyan")
    table.add_column("Parameters", style="green")
    table.add_column("Speed", style="yellow")
    table.add_column("Quality", style="magenta")

    table.add_row("tiny", "~30M", "Fastest", "Basic")
    table.add_row("small", "~90M", "Fast", "Good")
    table.add_row("base", "~300M", "Medium", "Very Good")
    table.add_row("large", "~848M", "Slower", "Best")

    console.print(table)
    console.print()

    # Depth models
    table = Table(title="Depth Anything 3 Models")
    table.add_column("Size", style="cyan")
    table.add_column("Parameters", style="green")
    table.add_column("Speed", style="yellow")
    table.add_column("Quality", style="magenta")

    table.add_row("small", "25M", "Fastest", "Basic")
    table.add_row("base", "99M", "Fast", "Good")
    table.add_row("large", "335M", "Medium", "Very Good")
    table.add_row("giant", "1.3B", "Slower", "Best")

    console.print(table)
    console.print()

    # Features
    console.print("[bold]SAM3 Features:[/bold]")
    console.print("  - Open-vocabulary text prompts")
    console.print("  - Visual exemplar prompts")
    console.print("  - Presence token disambiguation")
    console.print("  - Video tracking with memory")
    console.print()

    console.print("[bold]Depth Anything 3 Features:[/bold]")
    console.print("  - Multi-view depth estimation")
    console.print("  - Camera pose estimation")
    console.print("  - 3D Gaussian splatting")
    console.print("  - Sky segmentation")
    console.print("  - Camera intrinsics estimation")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
