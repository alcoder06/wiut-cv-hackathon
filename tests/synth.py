"""Synthetic trajectories with known answers, so rules can be tested without video.

Scene: a 1280x720 frame with a road of 7 lanes between y=300 and y=524. The top lanes
flow right (+x), the bottom lanes flow left (-x). Background traffic teaches the flow
field; each test adds its own actors on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.events import Context
from src.kinematics import add_kinematics
from src.pipeline import build_scene
from src.tracking import TRACK_COLS
from src.video import VideoInfo

FPS, SAMPLE_FPS, W, H = 25.0, 8.0, 1280, 720
RIGHT_LANES = [300, 332, 364, 396]
LEFT_LANES = [428, 460, 492, 524]


class Synth:
    def __init__(self, duration: float = 60.0):
        self.duration = duration
        self.t = np.arange(0, duration, 1 / SAMPLE_FPS)
        self.rows: list[tuple] = []
        self.next_id = 1

    def actor(self, cls: str, xy: np.ndarray, t: np.ndarray, w: float = 40, h: float = 30) -> int:
        """xy: (n, 2) ground points (bottom-centre) at times t."""
        tid, self.next_id = self.next_id, self.next_id + 1
        for (x, y), ti in zip(xy, t):
            self.rows.append((int(round(ti * FPS)), float(ti), tid, cls, 0.9,
                              x - w / 2, y - h, x + w / 2, y))
        return tid

    def background(self, every: float = 1.0, speed: float = 120.0, until: float | None = None):
        """Steady traffic in every lane, so the flow field learns the lane directions."""
        until = self.duration if until is None else until
        for lanes, sign, x0 in ((RIGHT_LANES, 1, 0), (LEFT_LANES, -1, W)):
            for y in lanes:
                for start in np.arange(0, until, every):
                    t = self.t[(self.t >= start) & (self.t <= start + W / speed)]
                    x = x0 + sign * speed * (t - start)
                    self.actor("car", np.stack([x, np.full_like(x, y)], 1), t)
        return self

    def context(self) -> Context:
        cfg = load_config()
        info = VideoInfo("synthetic.mp4", FPS, int(self.duration * FPS), W, H)
        tracks = add_kinematics(pd.DataFrame(self.rows, columns=TRACK_COLS),
                                cfg.kinematics.smooth_window_sec, centered=True)
        scene = build_scene(tracks, info, cfg, manual={})   # no real-camera zones in tests
        tracks["on_road"] = scene.on_road(tracks["gx"], tracks["gy"])
        return Context(tracks=tracks, frames=pd.DataFrame({"frame": [], "t": []}),
                       scene=scene, info=info, cfg=cfg)


def path(t: np.ndarray, t0: float, t1: float, p0, p1) -> np.ndarray:
    """Linear motion from p0 at t0 to p1 at t1 (clamped outside)."""
    a = np.clip((t - t0) / (t1 - t0), 0, 1)[:, None]
    return np.asarray(p0, float) * (1 - a) + np.asarray(p1, float) * a
