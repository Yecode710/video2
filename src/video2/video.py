"""Bounded, uniform video sampling for visual-only Qwen evaluation.

OpenCV's frame count and FPS metadata define the sampling timeline. Variable
frame-rate inputs should be normalized to constant FPS before evaluation when
precise timestamps matter. Audio is deliberately not decoded.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


@dataclass
class SampledVideo:
    """RGB frames and their source-timeline metadata, in temporal order."""

    frames: list[Any]
    timestamps: list[float]
    duration_seconds: float
    source_fps: float
    effective_fps: float


def sample_video(
    path: Path, num_frames: int = 32, max_frame_size: int = 448
) -> SampledVideo:
    """Read an even number of uniformly spaced frames across a complete clip.

    The first and last source frames are included. Short clips repeat frames to
    preserve the requested even length (Qwen's temporal patch size is two), and
    repeated frames retain the same source timestamp. Each frame's longest edge
    is at most ``max_frame_size``; smaller source images are not enlarged.

    Raises ValueError for invalid options or unusable metadata, FileNotFoundError
    for a missing file, and RuntimeError for missing dependencies or failed
    decoding. Requested frames are never silently dropped.
    """
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise ValueError("num_frames must be an even integer >= 2")
    if num_frames < 2 or num_frames % 2:
        raise ValueError("num_frames must be an even integer >= 2")
    if (
        isinstance(max_frame_size, bool)
        or not isinstance(max_frame_size, int)
        or max_frame_size < 28
    ):
        raise ValueError("max_frame_size must be an integer >= 28")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    try:
        import cv2
        import numpy  # noqa: F401 - explicitly check the video dependency set.
    except ImportError as exc:
        raise RuntimeError(
            "Video decoding requires OpenCV and NumPy; install with "
            "pip install -e '.[inference]'"
        ) from exc

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has invalid FPS metadata: {path}")
        if not math.isfinite(count) or count < 1:
            raise ValueError(f"Video has invalid frame-count metadata: {path}")
        frame_count = int(count)
        duration = frame_count / fps
        if not math.isfinite(duration):
            raise ValueError(f"Video has invalid duration metadata: {path}")
        indices = [
            round((frame_count - 1) * i / (num_frames - 1))
            for i in range(num_frames)
        ]
        frames: list[Any] = []
        timestamps: list[float] = []
        previous_index = None
        previous_frame = None
        expected_shape = None
        for index in indices:
            if index == previous_index:
                # Avoid asking a decoder to seek/read the same frame repeatedly.
                rgb = previous_frame.copy()
            else:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise RuntimeError(f"Cannot seek video {path} to frame {index}")
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot decode video {path} at frame {index}")
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    raise RuntimeError(f"Unsupported frame shape in {path} at frame {index}")
                height, width = frame.shape[:2]
                if height < 1 or width < 1:
                    raise RuntimeError(f"Empty image in {path} at frame {index}")
                scale = min(1.0, max_frame_size / max(height, width))
                if scale < 1:
                    size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if expected_shape is not None and rgb.shape != expected_shape:
                raise RuntimeError(f"Video changes frame dimensions: {path}")
            expected_shape = rgb.shape
            frames.append(rgb)
            timestamps.append(index / fps)
            previous_index = index
            previous_frame = rgb
        span = timestamps[-1] - timestamps[0]
        effective_fps = (num_frames - 1) / span if span > 0 else fps
        return SampledVideo(frames, timestamps, duration, fps, effective_fps)
    finally:
        capture.release()
