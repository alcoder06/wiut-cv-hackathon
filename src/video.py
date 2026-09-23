"""Video metadata and strided frame reading."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator

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


def downscale(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Shrink to at most max_width wide. Returns (frame, scale) with scale = new / original."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    s = max_width / w
    return cv2.resize(frame, (max_width, round(h * s)), interpolation=cv2.INTER_AREA), s


_END = object()


def iter_frames(path: str, stride: int, max_width: int | None = None,
                hook: Callable[[np.ndarray], dict] | None = None,
                prefetch: int = 32) -> Iterator[tuple[int, float, np.ndarray, dict]]:
    """Yield (frame_idx, t_sec, frame, hook_result) for every `stride`-th frame.

    Decoding runs in a background thread so it overlaps with detection and tracking on
    the main thread (OpenCV releases the GIL while decoding and resizing). The queue keeps
    frame order, so results are identical to a sequential read.
    Skipped frames use grab(): it still decodes (H.264 needs that) but skips the costly
    colour conversion and copy of retrieve(), which nearly doubles speed at 4K.
    `hook` runs on the full-resolution frame (e.g. reading a small signal-lamp ROI) before
    it is downscaled to `max_width`.
    """
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def reader():
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idx = 0
        try:
            while not stop.is_set():
                if idx % stride == 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    extra = hook(frame) if hook else {}
                    if max_width:
                        frame, _ = downscale(frame, max_width)
                    q.put((idx, idx / fps, frame, extra))
                elif not cap.grab():
                    break
                idx += 1
        finally:
            cap.release()
            q.put(_END)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while (item := q.get()) is not _END:
            yield item
    finally:
        stop.set()
        while thread.is_alive():        # drain so the reader can see `stop` and exit
            try:
                q.get_nowait()
            except queue.Empty:
                thread.join(timeout=0.05)
