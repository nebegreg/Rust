#!/usr/bin/env python3
"""
Ultimate Rotoscopy - Model Downloader
=====================================

Downloads all required pretrained models for the Ultimate Rotoscopy application.

Models included:
- SAM2 (Segment Anything Model 2) - Meta
- Depth Anything V2 - NeurIPS 2024
- ViTMatte - Information Fusion 2024
- GroundingDINO - IDEA Research
- MatAnyone - CVPR 2025
- Background Matting V2 - CVPR 2021

Sources:
- https://github.com/facebookresearch/sam2
- https://github.com/DepthAnything/Depth-Anything-V2
- https://github.com/hustvl/ViTMatte
- https://github.com/IDEA-Research/GroundingDINO
- https://github.com/pq-yang/MatAnyone
- https://github.com/PeterL1n/BackgroundMattingV2
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install 'rich' for better progress display: pip install rich")


@dataclass
class ModelInfo:
    """Information about a model checkpoint."""
    name: str
    url: str
    filename: str
    size_mb: float
    sha256: Optional[str] = None
    description: str = ""


# Model definitions
MODELS: Dict[str, List[ModelInfo]] = {
    "sam2": [
        ModelInfo(
            name="SAM2.1 Hiera Large",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            filename="sam2.1_hiera_large.pt",
            size_mb=897.0,
            description="Best quality SAM2 model for segmentation"
        ),
        ModelInfo(
            name="SAM2.1 Hiera Base+",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            filename="sam2.1_hiera_base_plus.pt",
            size_mb=323.0,
            description="Balanced SAM2 model"
        ),
        ModelInfo(
            name="SAM2.1 Hiera Small",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            filename="sam2.1_hiera_small.pt",
            size_mb=184.0,
            description="Fast SAM2 model"
        ),
    ],
    "depth_anything_v2": [
        ModelInfo(
            name="Depth Anything V2 ViT-L",
            url="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
            filename="depth_anything_v2_vitl.pth",
            size_mb=1340.0,
            description="Large model - best quality depth estimation"
        ),
        ModelInfo(
            name="Depth Anything V2 ViT-B",
            url="https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
            filename="depth_anything_v2_vitb.pth",
            size_mb=389.0,
            description="Base model - balanced quality/speed"
        ),
        ModelInfo(
            name="Depth Anything V2 ViT-S",
            url="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
            filename="depth_anything_v2_vits.pth",
            size_mb=99.0,
            description="Small model - fast inference"
        ),
    ],
    "vitmatte": [
        ModelInfo(
            name="ViTMatte Base DIS",
            url="https://huggingface.co/hustvl/vitmatte-base-distinctions-646/resolve/main/pytorch_model.bin",
            filename="vitmatte_base_dis.pth",
            size_mb=430.0,
            description="ViTMatte trained on Distinctions-646 dataset"
        ),
        ModelInfo(
            name="ViTMatte Small DIS",
            url="https://huggingface.co/hustvl/vitmatte-small-distinctions-646/resolve/main/pytorch_model.bin",
            filename="vitmatte_small_dis.pth",
            size_mb=110.0,
            description="Smaller ViTMatte model"
        ),
    ],
    "groundingdino": [
        ModelInfo(
            name="GroundingDINO SwinT OGC",
            url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            filename="groundingdino_swint_ogc.pth",
            size_mb=694.0,
            description="GroundingDINO for text-guided detection"
        ),
    ],
    "matanyone": [
        ModelInfo(
            name="MatAnyone Pretrained",
            url="https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone_pretrained.pth",
            filename="matanyone_pretrained.pth",
            size_mb=450.0,
            description="MatAnyone video matting model"
        ),
    ],
    "background_matting_v2": [
        ModelInfo(
            name="BGMv2 ResNet50 4K",
            url="https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/pytorch_resnet50.pth",
            filename="bgmv2_resnet50.pth",
            size_mb=135.0,
            description="Background Matting V2 ResNet50 for 4K"
        ),
        ModelInfo(
            name="BGMv2 MobileNetV2",
            url="https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/pytorch_mobilenetv2.pth",
            filename="bgmv2_mobilenetv2.pth",
            size_mb=25.0,
            description="Background Matting V2 MobileNet - fast"
        ),
    ],
}


class ModelDownloader:
    """Downloads and manages model checkpoints."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.home() / "ultimate_rotoscopy" / "pretrained_models"
        self.console = Console() if RICH_AVAILABLE else None

    def _print(self, message: str, style: str = ""):
        """Print message with optional styling."""
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)

    def download_file(
        self,
        url: str,
        dest_path: Path,
        description: str = "",
        expected_size_mb: float = 0,
    ) -> bool:
        """Download a file with progress tracking."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            existing_size = dest_path.stat().st_size / (1024 * 1024)
            if expected_size_mb > 0 and abs(existing_size - expected_size_mb) < 10:
                self._print(f"  ✓ Already exists: {dest_path.name}", "green")
                return True

        self._print(f"  Downloading: {description or dest_path.name}")

        try:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    console=self.console,
                ) as progress:
                    task = progress.add_task(dest_path.name, total=expected_size_mb * 1024 * 1024)

                    def reporthook(block_num, block_size, total_size):
                        progress.update(task, completed=block_num * block_size)

                    urllib.request.urlretrieve(url, dest_path, reporthook)
            else:
                def reporthook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, downloaded * 100 // total_size)
                        print(f"\r  Progress: {percent}%", end="", flush=True)

                urllib.request.urlretrieve(url, dest_path, reporthook)
                print()

            self._print(f"  ✓ Downloaded: {dest_path.name}", "green")
            return True

        except Exception as e:
            self._print(f"  ✗ Failed: {e}", "red")
            return False

    def download_category(self, category: str, models: List[ModelInfo]) -> int:
        """Download all models in a category."""
        category_dir = self.base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        self._print(f"\n[{category.upper()}]", "bold blue")

        success_count = 0
        for model in models:
            dest_path = category_dir / model.filename
            if self.download_file(
                model.url,
                dest_path,
                model.description,
                model.size_mb,
            ):
                success_count += 1

        return success_count

    def download_all(self, categories: Optional[List[str]] = None) -> Dict[str, int]:
        """Download all models or specified categories."""
        results = {}

        if categories is None:
            categories = list(MODELS.keys())

        self._print("=" * 70, "blue")
        self._print("Ultimate Rotoscopy - Model Downloader", "bold")
        self._print("=" * 70, "blue")
        self._print(f"Download directory: {self.base_dir}")

        total_size = sum(
            model.size_mb
            for cat in categories
            for model in MODELS.get(cat, [])
        )
        self._print(f"Total download size: ~{total_size / 1024:.1f} GB")

        for category in categories:
            if category in MODELS:
                results[category] = self.download_category(category, MODELS[category])
            else:
                self._print(f"Unknown category: {category}", "yellow")

        # Print summary
        self._print("\n" + "=" * 70, "blue")
        self._print("Download Summary", "bold")
        self._print("=" * 70, "blue")

        if RICH_AVAILABLE:
            table = Table()
            table.add_column("Category", style="cyan")
            table.add_column("Downloaded", style="green")
            table.add_column("Total", style="white")

            for cat, count in results.items():
                total = len(MODELS.get(cat, []))
                table.add_row(cat, str(count), str(total))

            self.console.print(table)
        else:
            for cat, count in results.items():
                total = len(MODELS.get(cat, []))
                print(f"  {cat}: {count}/{total}")

        return results

    def verify_models(self) -> Dict[str, List[Tuple[str, bool]]]:
        """Verify all downloaded models exist."""
        results = {}

        for category, models in MODELS.items():
            category_dir = self.base_dir / category
            category_results = []

            for model in models:
                path = category_dir / model.filename
                exists = path.exists()
                if exists:
                    size = path.stat().st_size / (1024 * 1024)
                    valid = abs(size - model.size_mb) < model.size_mb * 0.1  # 10% tolerance
                else:
                    valid = False

                category_results.append((model.name, valid))

            results[category] = category_results

        return results

    def print_paths(self):
        """Print paths to all model files."""
        self._print("\nModel Paths:", "bold")

        for category, models in MODELS.items():
            category_dir = self.base_dir / category
            self._print(f"\n[{category}]", "cyan")

            for model in models:
                path = category_dir / model.filename
                exists = "✓" if path.exists() else "✗"
                self._print(f"  {exists} {model.name}: {path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Ultimate Rotoscopy models")
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=None,
        help="Download directory (default: ~/ultimate_rotoscopy/pretrained_models)"
    )
    parser.add_argument(
        "--category", "-c",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Categories to download"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing downloads"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List model paths"
    )

    args = parser.parse_args()

    downloader = ModelDownloader(args.dir)

    if args.list:
        downloader.print_paths()
        return

    if args.verify:
        results = downloader.verify_models()
        for category, models in results.items():
            print(f"\n[{category}]")
            for name, valid in models:
                status = "✓" if valid else "✗"
                print(f"  {status} {name}")
        return

    categories = None if "all" in args.category else args.category
    downloader.download_all(categories)


if __name__ == "__main__":
    main()
