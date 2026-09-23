"""accident and near_miss, from trajectories only (no trained model).

accident: two road users' boxes touch AND at least one of them brakes abruptly at that
moment AND they stay (nearly) stopped afterwards. Box overlap alone is useless from a
CCTV angle (vehicles overlap in perspective all the time); the abrupt stop is what
separates a crash from a car passing behind another.
  start = first frame of contact; end = every involved user has stopped or left.

near_miss: time-to-collision drops below `ttc_sec`, one of the pair brakes hard or
swerves, and no contact follows.
  start = onset of the evasive action; end = the pair is clear (TTC back above
  `clear_ttc_sec`).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..kinematics import ttc_matrix
from . import Context
from .pedestrian import walkers


def _road_users(ctx: Context) -> pd.DataFrame:
    users = pd.concat([ctx.vehicles(), walkers(ctx)])
    return users[users["on_road"]].sort_values(["frame", "tid"])


def _pairwise_iou(b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
    np.fill_diagonal(iou, 0)
    return iou


def pair_timelines(users: pd.DataFrame, cfg) -> tuple[dict, dict]:
    """Per pair (tid_a, tid_b): times of box contact and times of low TTC."""
    contact, close = defaultdict(list), defaultdict(list)
    c, n = cfg.rules.collision, cfg.rules.near_miss
    for _, f in users.groupby("frame"):
        if len(f) < 2:
            continue
        t = f["t"].iat[0]
        tids = f["tid"].to_numpy()
        iou = _pairwise_iou(f[["x1", "y1", "x2", "y2"]].to_numpy())
        ttc = ttc_matrix(f[["gx", "gy"]].to_numpy(), f[["vx", "vy"]].to_numpy(),
                         f["size"].to_numpy(), c.radius_factor)
        for i, j in zip(*np.where(np.triu(iou >= c.contact_iou))):
            contact[(tids[i], tids[j])].append(t)
        for i, j in zip(*np.where(np.triu(ttc < n.clear_ttc_sec))):
            close[(tids[i], tids[j])].append((t, ttc[i, j]))
    return contact, close


def _speed_between(track: pd.DataFrame, t0: float, t1: float, how: str) -> float:
    s = track.loc[(track["t"] >= t0) & (track["t"] <= t1), "speed"]
    return float(getattr(s, how)()) if len(s) else np.nan


def _abrupt_drop(track: pd.DataFrame, t: float, window: float, drop: float, moving: float) -> float | None:
    """Time the abrupt deceleration started, or None."""
    before = track[(track["t"] >= t - 1.5) & (track["t"] <= t + 0.5)]
    if before.empty or before["speed"].max() < moving:
        return None
    peak = before["speed"].max()
    # the LAST moment at (near) top speed is where braking begins; at constant speed the
    # first maximum can be seconds earlier and the window would miss the stop
    t_peak = float(before.loc[before["speed"] >= 0.95 * peak, "t"].iloc[-1])
    after_min = _speed_between(track, t_peak, t_peak + window, "min")
    if np.isnan(after_min) or after_min > (1 - drop) * peak:
        return None
    return t_peak


def _stop_or_leave_time(track: pd.DataFrame, t: float, stopped: float) -> float:
    rest = track[track["t"] >= t]
    idx = np.flatnonzero(rest["speed"].to_numpy() < stopped)
    return float(rest["t"].iat[idx[0]] if len(idx) else rest["t"].iat[-1])


def detect(ctx: Context) -> list[list]:
    users = _road_users(ctx)
    if users.empty:
        return []
    cfg, k = ctx.cfg, ctx.cfg.kinematics
    c, n = cfg.rules.collision, cfg.rules.near_miss
    by_tid = {tid: g for tid, g in users.groupby("tid")}
    contact, close = pair_timelines(users, cfg)
    events, crash_pairs = [], []

    for (a, b), times in contact.items():
        t0 = times[0]
        ta, tb = by_tid[a], by_tid[b]
        if not any(_abrupt_drop(tr, t0, c.decel_window_sec, c.decel_drop, k.moving_speed) for tr in (ta, tb)):
            continue
        after = [_speed_between(tr, t0 + 0.5, t0 + 0.5 + c.after_stop_sec, "mean") for tr in (ta, tb)]
        if any(s > k.moving_speed for s in after if not np.isnan(s)):
            continue
        end = max(_stop_or_leave_time(tr, t0, k.stopped_speed) for tr in (ta, tb))
        events.append([t0, max(end, t0 + 1.0), "accident"])
        crash_pairs.append(((a, b), t0))

    for (a, b), samples in close.items():
        t = np.array([s[0] for s in samples])
        ttc = np.array([s[1] for s in samples])
        if ttc.min() >= n.ttc_sec:
            continue
        if any(p == (a, b) and abs(t0 - t.min()) < 5 for p, t0 in crash_pairs):
            continue
        t_crit = float(t[ttc.argmin()])
        onsets = [_abrupt_drop(tr, t_crit, 1.5, n.decel_drop, k.moving_speed) for tr in (by_tid[a], by_tid[b])]
        onsets = [o for o in onsets if o is not None]
        if not onsets:
            continue
        # end of the continuous close-approach stretch (gaps > 1 s split it) that holds t_crit
        breaks = np.flatnonzero(np.diff(t) > 1.0)
        starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(t) - 1]
        end = float(t[ends[(t[starts] <= t_crit) & (t[ends] >= t_crit)][0]])
        events.append([min(onsets), max(end, t_crit + 0.5), "near_miss"])
    return events
