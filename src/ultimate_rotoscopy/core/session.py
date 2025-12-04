"""
Session Management for Ultimate Rotoscopy
==========================================

Provides persistent session management for rotoscopy projects,
including state tracking, undo/redo, and project persistence.
"""

import json
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image


@dataclass
class SessionConfig:
    """Session configuration."""
    project_name: str = "Untitled Project"
    output_directory: Path = field(default_factory=lambda: Path.cwd() / "output")
    cache_directory: Path = field(default_factory=lambda: Path.cwd() / ".cache")
    auto_save: bool = True
    auto_save_interval: int = 300  # seconds
    max_undo_history: int = 50
    save_intermediate_results: bool = True
    frame_rate: float = 24.0
    resolution: Optional[tuple] = None  # (width, height)


@dataclass
class FrameData:
    """Data for a single frame."""
    frame_index: int
    image_path: Optional[Path] = None
    alpha: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    masks: Optional[np.ndarray] = None
    foreground: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    processing_time_ms: float = 0.0


@dataclass
class Shot:
    """A shot (sequence of frames)."""
    shot_id: str
    name: str
    start_frame: int
    end_frame: int
    frames: Dict[int, FrameData] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Session:
    """
    Session manager for rotoscopy projects.

    Handles:
    - Project state management
    - Frame data storage
    - Undo/redo history
    - Auto-save
    - Project serialization

    Example:
        >>> config = SessionConfig(project_name="My VFX Project")
        >>> session = Session(config)
        >>>
        >>> # Add frames
        >>> session.add_frame(0, image_path=Path("frame_0001.exr"))
        >>>
        >>> # Process with engine
        >>> result = engine.process(session.get_frame_image(0))
        >>> session.update_frame(0, result)
        >>>
        >>> # Save project
        >>> session.save("project.roto")
    """

    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.modified_at = datetime.now()

        # Shots and frames
        self.shots: Dict[str, Shot] = {}
        self.current_shot_id: Optional[str] = None
        self.current_frame: int = 0

        # Undo/redo
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []

        # Create directories
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        self.config.cache_directory.mkdir(parents=True, exist_ok=True)

        # Create default shot
        self._create_default_shot()

    def _create_default_shot(self) -> None:
        """Create a default shot."""
        shot_id = str(uuid.uuid4())
        shot = Shot(
            shot_id=shot_id,
            name="Shot_001",
            start_frame=0,
            end_frame=100,
        )
        self.shots[shot_id] = shot
        self.current_shot_id = shot_id

    def create_shot(
        self,
        name: str,
        start_frame: int,
        end_frame: int,
    ) -> Shot:
        """Create a new shot."""
        shot_id = str(uuid.uuid4())
        shot = Shot(
            shot_id=shot_id,
            name=name,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self.shots[shot_id] = shot
        self._mark_modified()
        return shot

    def get_current_shot(self) -> Optional[Shot]:
        """Get the current shot."""
        if self.current_shot_id:
            return self.shots.get(self.current_shot_id)
        return None

    def set_current_shot(self, shot_id: str) -> None:
        """Set the current shot."""
        if shot_id in self.shots:
            self.current_shot_id = shot_id

    def add_frame(
        self,
        frame_index: int,
        image_path: Optional[Path] = None,
        image_data: Optional[np.ndarray] = None,
        shot_id: Optional[str] = None,
    ) -> FrameData:
        """Add a frame to the session."""
        shot_id = shot_id or self.current_shot_id
        if not shot_id or shot_id not in self.shots:
            raise ValueError("No valid shot selected")

        shot = self.shots[shot_id]

        frame = FrameData(
            frame_index=frame_index,
            image_path=image_path,
        )

        # Cache image data if provided
        if image_data is not None:
            cache_path = self.config.cache_directory / f"frame_{frame_index:06d}.npy"
            np.save(cache_path, image_data)
            frame.metadata["cached_image"] = str(cache_path)

        shot.frames[frame_index] = frame
        self._mark_modified()

        return frame

    def update_frame(
        self,
        frame_index: int,
        result: Any,  # ProcessingResult
        shot_id: Optional[str] = None,
    ) -> None:
        """Update frame with processing results."""
        shot_id = shot_id or self.current_shot_id
        if not shot_id or shot_id not in self.shots:
            return

        shot = self.shots[shot_id]
        if frame_index not in shot.frames:
            self.add_frame(frame_index, shot_id=shot_id)

        frame = shot.frames[frame_index]

        # Store previous state for undo
        self._push_undo(frame_index, shot_id, frame)

        # Update frame data
        if hasattr(result, "alpha"):
            frame.alpha = result.alpha
        if hasattr(result, "depth_map"):
            frame.depth = result.depth_map
        if hasattr(result, "normals"):
            frame.normals = result.normals
        if hasattr(result, "masks"):
            frame.masks = result.masks
        if hasattr(result, "foreground"):
            frame.foreground = result.foreground
        if hasattr(result, "processing_time_ms"):
            frame.processing_time_ms = result.processing_time_ms
        if hasattr(result, "metadata"):
            frame.metadata.update(result.metadata)

        frame.processed = True
        self._mark_modified()

        # Save intermediate if enabled
        if self.config.save_intermediate_results:
            self._save_frame_data(frame_index, shot_id)

    def get_frame(
        self,
        frame_index: int,
        shot_id: Optional[str] = None
    ) -> Optional[FrameData]:
        """Get frame data."""
        shot_id = shot_id or self.current_shot_id
        if not shot_id or shot_id not in self.shots:
            return None

        return self.shots[shot_id].frames.get(frame_index)

    def get_frame_image(
        self,
        frame_index: int,
        shot_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Load and return frame image."""
        frame = self.get_frame(frame_index, shot_id)
        if not frame:
            return None

        # Try cached image first
        if "cached_image" in frame.metadata:
            cache_path = Path(frame.metadata["cached_image"])
            if cache_path.exists():
                return np.load(cache_path)

        # Load from image path
        if frame.image_path and frame.image_path.exists():
            return np.array(Image.open(frame.image_path))

        return None

    def get_frame_range(
        self,
        start: int,
        end: int,
        shot_id: Optional[str] = None
    ) -> List[FrameData]:
        """Get frames in range."""
        shot_id = shot_id or self.current_shot_id
        if not shot_id or shot_id not in self.shots:
            return []

        shot = self.shots[shot_id]
        return [
            shot.frames[i]
            for i in range(start, end + 1)
            if i in shot.frames
        ]

    def _push_undo(
        self,
        frame_index: int,
        shot_id: str,
        frame: FrameData
    ) -> None:
        """Push state to undo stack."""
        state = {
            "frame_index": frame_index,
            "shot_id": shot_id,
            "frame_data": {
                "alpha": frame.alpha.copy() if frame.alpha is not None else None,
                "depth": frame.depth.copy() if frame.depth is not None else None,
                "normals": frame.normals.copy() if frame.normals is not None else None,
                "masks": frame.masks.copy() if frame.masks is not None else None,
                "foreground": frame.foreground.copy() if frame.foreground is not None else None,
                "metadata": frame.metadata.copy(),
            }
        }

        self._undo_stack.append(state)
        self._redo_stack.clear()

        # Limit undo history
        if len(self._undo_stack) > self.config.max_undo_history:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Undo last operation."""
        if not self._undo_stack:
            return False

        state = self._undo_stack.pop()
        frame_index = state["frame_index"]
        shot_id = state["shot_id"]

        # Get current state for redo
        if shot_id in self.shots and frame_index in self.shots[shot_id].frames:
            current = self.shots[shot_id].frames[frame_index]
            self._redo_stack.append({
                "frame_index": frame_index,
                "shot_id": shot_id,
                "frame_data": {
                    "alpha": current.alpha.copy() if current.alpha is not None else None,
                    "depth": current.depth.copy() if current.depth is not None else None,
                    "normals": current.normals.copy() if current.normals is not None else None,
                    "masks": current.masks.copy() if current.masks is not None else None,
                    "foreground": current.foreground.copy() if current.foreground is not None else None,
                    "metadata": current.metadata.copy(),
                }
            })

            # Restore previous state
            frame_data = state["frame_data"]
            current.alpha = frame_data["alpha"]
            current.depth = frame_data["depth"]
            current.normals = frame_data["normals"]
            current.masks = frame_data["masks"]
            current.foreground = frame_data["foreground"]
            current.metadata = frame_data["metadata"]

        self._mark_modified()
        return True

    def redo(self) -> bool:
        """Redo last undone operation."""
        if not self._redo_stack:
            return False

        state = self._redo_stack.pop()
        frame_index = state["frame_index"]
        shot_id = state["shot_id"]

        if shot_id in self.shots and frame_index in self.shots[shot_id].frames:
            current = self.shots[shot_id].frames[frame_index]

            # Push to undo
            self._undo_stack.append({
                "frame_index": frame_index,
                "shot_id": shot_id,
                "frame_data": {
                    "alpha": current.alpha.copy() if current.alpha is not None else None,
                    "depth": current.depth.copy() if current.depth is not None else None,
                    "normals": current.normals.copy() if current.normals is not None else None,
                    "masks": current.masks.copy() if current.masks is not None else None,
                    "foreground": current.foreground.copy() if current.foreground is not None else None,
                    "metadata": current.metadata.copy(),
                }
            })

            # Apply redo state
            frame_data = state["frame_data"]
            current.alpha = frame_data["alpha"]
            current.depth = frame_data["depth"]
            current.normals = frame_data["normals"]
            current.masks = frame_data["masks"]
            current.foreground = frame_data["foreground"]
            current.metadata = frame_data["metadata"]

        self._mark_modified()
        return True

    def _mark_modified(self) -> None:
        """Mark session as modified."""
        self.modified_at = datetime.now()

    def _save_frame_data(self, frame_index: int, shot_id: str) -> None:
        """Save frame data to cache."""
        shot = self.shots.get(shot_id)
        if not shot or frame_index not in shot.frames:
            return

        frame = shot.frames[frame_index]
        base_path = self.config.cache_directory / f"frame_{frame_index:06d}"

        if frame.alpha is not None:
            np.save(f"{base_path}_alpha.npy", frame.alpha)
        if frame.depth is not None:
            np.save(f"{base_path}_depth.npy", frame.depth)
        if frame.normals is not None:
            np.save(f"{base_path}_normals.npy", frame.normals)

    def save(self, path: Union[str, Path]) -> None:
        """Save session to file."""
        path = Path(path)

        data = {
            "version": "1.0",
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "config": {
                "project_name": self.config.project_name,
                "output_directory": str(self.config.output_directory),
                "cache_directory": str(self.config.cache_directory),
                "frame_rate": self.config.frame_rate,
                "resolution": self.config.resolution,
            },
            "shots": {},
        }

        for shot_id, shot in self.shots.items():
            shot_data = {
                "shot_id": shot.shot_id,
                "name": shot.name,
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "metadata": shot.metadata,
                "frames": {},
            }

            for frame_idx, frame in shot.frames.items():
                frame_data = {
                    "frame_index": frame.frame_index,
                    "image_path": str(frame.image_path) if frame.image_path else None,
                    "processed": frame.processed,
                    "processing_time_ms": frame.processing_time_ms,
                    "metadata": frame.metadata,
                }
                shot_data["frames"][str(frame_idx)] = frame_data

            data["shots"][shot_id] = shot_data

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Session":
        """Load session from file."""
        path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        config = SessionConfig(
            project_name=data["config"]["project_name"],
            output_directory=Path(data["config"]["output_directory"]),
            cache_directory=Path(data["config"]["cache_directory"]),
            frame_rate=data["config"]["frame_rate"],
            resolution=data["config"].get("resolution"),
        )

        session = cls(config)
        session.session_id = data["session_id"]
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.modified_at = datetime.fromisoformat(data["modified_at"])
        session.shots = {}

        for shot_id, shot_data in data["shots"].items():
            shot = Shot(
                shot_id=shot_data["shot_id"],
                name=shot_data["name"],
                start_frame=shot_data["start_frame"],
                end_frame=shot_data["end_frame"],
                metadata=shot_data.get("metadata", {}),
            )

            for frame_idx_str, frame_data in shot_data["frames"].items():
                frame = FrameData(
                    frame_index=frame_data["frame_index"],
                    image_path=Path(frame_data["image_path"]) if frame_data["image_path"] else None,
                    processed=frame_data["processed"],
                    processing_time_ms=frame_data["processing_time_ms"],
                    metadata=frame_data.get("metadata", {}),
                )

                # Load cached arrays
                base_path = config.cache_directory / f"frame_{frame.frame_index:06d}"
                if (base_path.parent / f"{base_path.name}_alpha.npy").exists():
                    frame.alpha = np.load(f"{base_path}_alpha.npy")
                if (base_path.parent / f"{base_path.name}_depth.npy").exists():
                    frame.depth = np.load(f"{base_path}_depth.npy")
                if (base_path.parent / f"{base_path.name}_normals.npy").exists():
                    frame.normals = np.load(f"{base_path}_normals.npy")

                shot.frames[int(frame_idx_str)] = frame

            session.shots[shot_id] = shot

        if session.shots:
            session.current_shot_id = list(session.shots.keys())[0]

        return session

    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        total_frames = 0
        processed_frames = 0
        total_time = 0.0

        for shot in self.shots.values():
            total_frames += len(shot.frames)
            for frame in shot.frames.values():
                if frame.processed:
                    processed_frames += 1
                    total_time += frame.processing_time_ms

        return {
            "session_id": self.session_id,
            "project_name": self.config.project_name,
            "num_shots": len(self.shots),
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "total_processing_time_ms": total_time,
            "avg_processing_time_ms": total_time / processed_frames if processed_frames > 0 else 0,
        }
