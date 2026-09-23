"""Video metadata and strided frame reading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        # Same definition as run_submission.py, so our end times never exceed theirs.
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = VideoInfo(
        path=path,
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def stride_for(fps: float, target_fps: float) -> int:
    return max(1, round(fps / target_fps))


def iter_frames(path: str, stride: int) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, t_sec, BGR frame) for every `stride`-th frame.

    Skipped frames use grab(), which still decodes (H.264 needs it) but skips the
    colour conversion and copy that retrieve() does. That's the cheap part we can save.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / fps, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
