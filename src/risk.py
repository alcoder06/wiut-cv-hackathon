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
from .kinematics import VEHICLES, ttc_matrix
from .scene import FlowField
from .tracking import make_tracker, to_sv
from .video import downscale, stride_for

MOVERS = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}


class _Track:
    __slots__ = ("pos", "vel", "size", "last_t", "speeds", "name")

    def __init__(self, pos, size, t, name):
        self.pos, self.vel, self.size, self.last_t, self.name = pos, np.zeros(2), size, t, name
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
        self.own_sec = 0.0       # time spent inside step() on analysed frames
        self.pool = ThreadPoolExecutor(max_workers=1)   # one worker keeps frame order
        self.pending = None
        self.streak: dict = {}

    def _behind_schedule(self, t: float) -> bool:
        """True when OUR OWN processing has used more than part_b_own_max_x of the video time.

        Only our own time counts: the harness's 4K decoding is most of Part B's wall clock and
        skipping can't shorten it. Measuring total wall time (the first version) meant that on a
        loaded machine Part B fell behind once and never caught up, and the risk curve went flat
        for the rest of the video. Our own time stops growing while we skip, so we resume."""
        return self.own_sec > self.cfg.sampling.part_b_own_max_x * t + 2.0

    def _load_road(self, w: int, h: int):
        path = resolve(self.cfg.scene.learned)
        if not path.exists():
            return None
        flow = FlowField.load(path).rescaled(w, h)
        return flow.road_mask(self.cfg.scene.min_cell_obs) if flow is not None else None

    def step(self, frame: np.ndarray, t: float) -> float:
        """Detection runs in a worker thread so it overlaps the harness decoding the next
        frames. Fixed one-sample lag: the result for sampled frame k is folded in at sampled
        frame k+1 (0.2 s later), always, so the curve is identical run to run."""
        if int(round(t * self.fps)) % self.stride or self._behind_schedule(t):
            return self.score           # skipped frame (or catching up): O(1)
        started = time.perf_counter()
        if self.pending is not None:
            tracked, t_prev = self.pending.result()
            self._fold_in(tracked, t_prev)
        self.pending = self.pool.submit(self._detect, frame, t)
        self.own_sec += time.perf_counter() - started
        return self.score

    def _detect(self, frame: np.ndarray, t: float):
        from .detector import get_detector

        small, s = downscale(frame, int(self.cfg.detector.max_input_width))
        det = get_detector().predict([small])[0]
        det.xyxy /= s
        return self.tracker.update_with_detections(to_sv(det)), t

    def _fold_in(self, tracked, t: float) -> None:
        names = [self.names.get(int(k)) for k in tracked.class_id]
        self.observe(tracked.xyxy, names, tracked.tracker_id, t)

    def observe(self, xyxy, names, tids, t: float) -> float:
        """Scoring half of Part B, detector-free: tracked boxes at time t -> updated score.
        scripts/replay_risk.py feeds cached tracks through this to tune in seconds."""
        self._update_tracks(xyxy, names, tids, t)
        raw = self._instant_risk(t)
        dt = t - self.last_t if self.last_t is not None else 0.0
        alpha = 1 - np.exp(-dt / self.cfg.risk.ema_sec) if dt > 0 else 1.0
        self.score = float(np.clip(self.score + alpha * (raw - self.score), 0, 1))
        self.last_t = t
        return self.score

    # -- internals -------------------------------------------------------------------
    def _update_tracks(self, xyxy, names, tids, t: float) -> None:
        for (x1, y1, x2, y2), name, tid in zip(xyxy, names, tids):
            if name not in MOVERS:
                continue
            pos = np.array([(x1 + x2) / 2, y2])
            size = float(np.sqrt(max(x2 - x1, 1) * max(y2 - y1, 1)))
            tr = self.tracks.get(tid)
            if tr is None:
                self.tracks[tid] = _Track(pos, size, t, name)
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
        cfg, r = self.cfg, self.cfg.risk
        live = [(tid, tr) for tid, tr in self.tracks.items()
                if tr.last_t == t and len(tr.speeds) >= r.min_track_samples and self._on_road(tr.pos)]
        if len(live) < 2:
            self.streak.clear()
            return 0.0
        tids = [tid for tid, _ in live]
        trs = [tr for _, tr in live]
        pos = np.array([tr.pos for tr in trs])
        vel = np.array([tr.vel for tr in trs])
        size = np.array([tr.size for tr in trs])
        vehicle = np.array([tr.name in VEHICLES for tr in trs])
        speed = np.linalg.norm(vel, axis=1) / size
        moving = speed > cfg.kinematics.moving_speed
        unit = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9)
        cos = unit @ unit.T
        p = pos[None, :, :] - pos[:, None, :]
        v = vel[None, :, :] - vel[:, None, :]
        mean_size = 0.5 * (size[:, None] + size[None, :])

        # Time until two discs at the ground points touch (kinematics.ttc_matrix). Tried and
        # rejected on the sample video: moving boxes / footprints (22-26% of the time in
        # alarm; crossing streams overlap constantly in the image). Radius per pair type:
        #   two movers    -> rules.collision.radius_factor (side impacts, tests/test_risk.py)
        #   one is stopped -> risk.radius_factor, tighter: crossing traffic brushing past cars
        #                     waiting at a stop line was the main false alarm; a rear-end
        #                     into the stopped car still closes to zero distance
        both_moving = moving[:, None] & moving[None, :]
        rf = np.where(both_moving, cfg.rules.collision.radius_factor, r.radius_factor)
        ttc = ttc_matrix(pos, vel, size, rf)

        # parallel movers (same or opposite direction) only conflict in the same lane:
        # overtaking / passing in the next lane has a sideways offset of ~a lane width
        parallel = both_moving & (np.abs(cos) > r.parallel_cos)
        axis = unit[:, None, :] + np.sign(cos)[..., None] * unit[None, :, :]
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9
        lateral = np.abs(p[..., 0] * axis[..., 1] - p[..., 1] * axis[..., 0]) / mean_size
        ttc = np.where(parallel & (lateral > r.max_lane_offset), np.inf, ttc)

        # closing speed in box sizes per second: how fast the gap shrinks
        dist = np.linalg.norm(p, axis=-1) + 1e-9
        closing = -(p * v).sum(-1) / dist / mean_size
        # Traffic bunching up in a lane, or arriving at a stopped queue, closes slowly and
        # constantly; two movers crossing each other's path at a wide angle rarely close at
        # all unless one is about to hit the other, so they get a lower floor (side impacts)
        crossing = both_moving & (np.abs(cos) < r.crossing_cos)
        floor = np.where(crossing, r.min_closing_crossing, r.min_closing)
        valid = (closing >= floor) & (vehicle[:, None] | vehicle[None, :])
        threat = np.triu(valid & (ttc < r.ttc_mid_sec + 2 * r.ttc_scale_sec), 1)

        # persistence: a pair must stay threatening for consecutive samples
        streak = {}
        for i, j in zip(*np.nonzero(threat)):
            key = (tids[i], tids[j])
            streak[key] = self.streak.get(key, 0) + 1
        self.streak = streak
        best = min((ttc[tids.index(a), tids.index(b)] for (a, b), n in streak.items()
                    if n >= r.persist_samples), default=np.inf)
        r_ttc = 1 / (1 + np.exp((best - r.ttc_mid_sec) / r.ttc_scale_sec))
        r_brake = max((self._brake(tr) for tr in trs), default=0.0)
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


def replay_tracks(tracks, info, video_id: str) -> list[list[float]]:
    """Risk curve [[t, score], ...] from already-tracked boxes, detector-free.

    Feeds each sampled frame's boxes, in time order, through the same causal scorer
    the harness runs (CausalRisk.observe). Used to tune Part B in seconds
    (scripts/replay_risk.py) and by the web demo, which can't afford a second detector
    pass on a CPU. The submission itself runs Part B's own detector and tracker."""
    est = CausalRisk()
    est.reset({"video_id": video_id, "fps": info.fps, "width": info.width,
               "height": info.height, "n_frames": info.n_frames})
    curve = []
    for (_, t), g in tracks.groupby(["frame", "t"], sort=True):
        s = est.observe(g[["x1", "y1", "x2", "y2"]].to_numpy(), g["cls"].tolist(), g["tid"].to_numpy(), t)
        curve.append([round(float(t), 4), round(s, 4)])
    return curve
