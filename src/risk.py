"""Part B: causal accident risk from time-to-collision and sudden braking.

Strictly causal: we only see frames the harness hands us, smooth with past samples only
(exponential moving averages), and never touch Part A's output (the FAQ forbids B -> A).
Scene knowledge built from the *sample* videos (the prebuilt flow field) is allowed:
it's layout, like camera.md, not information from the test video's future.

Score design follows the metric: near 0 in normal traffic (AP is pooled over every
frame, so calm-traffic bumps hurt), rising 2-5 s before a predicted contact.
"""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import load_config, resolve
from .kinematics import ttc_matrix
from .scene import FlowField
from .tracking import make_tracker, to_sv
from .video import downscale, stride_for

MOVERS = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}


class _Track:
    __slots__ = ("pos", "vel", "size", "last_t", "speeds")

    def __init__(self, pos, size, t):
        self.pos, self.vel, self.size, self.last_t = pos, np.zeros(2), size, t
        self.speeds: deque = deque()


class CausalRisk:
    def __init__(self):
        self.cfg = load_config()
        self.road = None

    def reset(self, meta: dict) -> None:
        cfg = self.cfg
        self.fps = float(meta["fps"]) or 25.0
        self.stride = stride_for(self.fps, cfg.sampling.part_b_fps)
        self.tracker = make_tracker(self.fps / self.stride, cfg.tracker)
        self.names = {int(k): v for k, v in cfg.detector.classes.items()}
        self.tracks: dict[int, _Track] = {}
        self.score, self.last_t = 0.0, None
        self.road = self._load_road(int(meta["width"]), int(meta["height"]))
        self.wall_start = time.perf_counter()
        self.pool = ThreadPoolExecutor(max_workers=1)   # one worker keeps frame order
        self.pending = None

    def _behind_schedule(self, t: float) -> bool:
        """True when this part has used more than part_b_max_x times the video time so far."""
        return time.perf_counter() - self.wall_start > self.cfg.sampling.part_b_max_x * t + 5.0

    def _load_road(self, w: int, h: int):
        path = resolve(self.cfg.scene.learned)
        if not path.exists():
            return None
        flow = FlowField.load(path)
        return flow.road_mask(self.cfg.scene.min_cell_obs) if (flow.width, flow.height) == (w, h) else None

    def step(self, frame: np.ndarray, t: float) -> float:
        """Detection runs in a worker thread so it overlaps the harness decoding the next
        frames. Fixed one-sample lag: the result for sampled frame k is folded in at sampled
        frame k+1 (0.2 s later), always, so the curve is identical run to run."""
        if int(round(t * self.fps)) % self.stride or self._behind_schedule(t):
            return self.score           # skipped frame (or catching up): O(1)
        if self.pending is not None:
            tracked, t_prev = self.pending.result()
            self._fold_in(tracked, t_prev)
        self.pending = self.pool.submit(self._detect, frame, t)
        return self.score

    def _detect(self, frame: np.ndarray, t: float):
        from .detector import get_detector

        small, s = downscale(frame, int(self.cfg.detector.max_input_width))
        det = get_detector().predict([small])[0]
        det.xyxy /= s
        return self.tracker.update_with_detections(to_sv(det)), t

    def _fold_in(self, tracked, t: float) -> None:
        self._update_tracks(tracked, t)
        raw = self._instant_risk(t)
        dt = t - self.last_t if self.last_t is not None else 0.0
        alpha = 1 - np.exp(-dt / self.cfg.risk.ema_sec) if dt > 0 else 1.0
        self.score = float(np.clip(self.score + alpha * (raw - self.score), 0, 1))
        self.last_t = t

    # -- internals -------------------------------------------------------------------
    def _update_tracks(self, tracked, t: float) -> None:
        for (x1, y1, x2, y2), k, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
            if self.names.get(int(k)) not in MOVERS:
                continue
            pos = np.array([(x1 + x2) / 2, y2])
            size = float(np.sqrt(max(x2 - x1, 1) * max(y2 - y1, 1)))
            tr = self.tracks.get(tid)
            if tr is None:
                self.tracks[tid] = _Track(pos, size, t)
                continue
            dt = t - tr.last_t
            if dt <= 0:
                continue
            tr.vel = 0.5 * tr.vel + 0.5 * (pos - tr.pos) / dt       # causal EMA
            tr.pos, tr.size, tr.last_t = pos, 0.8 * tr.size + 0.2 * size, t
            tr.speeds.append((t, np.linalg.norm(tr.vel) / tr.size))
            while tr.speeds and tr.speeds[0][0] < t - 2.0:
                tr.speeds.popleft()
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > 2.0]:
            del self.tracks[tid]

    def _instant_risk(self, t: float) -> float:
        live = [tr for tr in self.tracks.values() if tr.last_t == t and self._on_road(tr.pos)]
        if len(live) < 2:
            return 0.0
        cfg = self.cfg
        pos = np.array([tr.pos for tr in live])
        vel = np.array([tr.vel for tr in live])
        size = np.array([tr.size for tr in live])
        ttc = ttc_matrix(pos, vel, size, cfg.rules.collision.radius_factor)
        # ignore pairs already touching while both crawl: that's a queue, not a threat
        speed = np.linalg.norm(vel, axis=1) / size
        both_slow = (speed[:, None] < cfg.kinematics.moving_speed) & (speed[None, :] < cfg.kinematics.moving_speed)
        ttc = np.where(both_slow, np.inf, ttc)
        min_ttc = float(ttc.min())
        r_ttc = 1 / (1 + np.exp((min_ttc - cfg.risk.ttc_mid_sec) / cfg.risk.ttc_scale_sec))
        r_brake = max((self._brake(tr) for tr in live), default=0.0)
        return float(1 - (1 - r_ttc) * (1 - 0.5 * r_brake * r_ttc ** 0.5))

    def _brake(self, tr: _Track) -> float:
        """Relative speed drop over the last second, 0..1."""
        if len(tr.speeds) < 3:
            return 0.0
        recent = [s for ts, s in tr.speeds if ts >= tr.last_t - 1.0]
        peak = max(recent)
        if peak < self.cfg.kinematics.moving_speed:
            return 0.0
        return float(np.clip((peak - recent[-1]) / peak, 0, 1))

    def _on_road(self, p: np.ndarray) -> bool:
        if self.road is None:
            return True
        h, w = self.road.shape
        return bool(self.road[min(int(p[1]), h - 1), min(int(p[0]), w - 1)])
