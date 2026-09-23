"""Per-track motion features from raw boxes.

Positions use the bottom-centre of the box (where the vehicle touches the road), so
"is it on the road / past the stop line" tests use the right point.
Speed is divided by the box size sqrt(w*h): a car moving 1 body-length per second has
speed 1 whether it's near or far from the camera. No homography needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VEHICLES = ("car", "truck", "bus", "motorcycle", "bicycle")


def add_kinematics(tracks: pd.DataFrame, smooth_sec: float, centered: bool = True) -> pd.DataFrame:
    """centered=True uses past and future samples (Part A only, offline)."""
    if tracks.empty:
        for c in ("gx", "gy", "size", "vx", "vy", "speed", "heading"):
            tracks[c] = pd.Series(dtype=float)
        return tracks
    df = tracks.sort_values(["tid", "t"]).copy()
    df["gx"] = (df["x1"] + df["x2"]) / 2
    df["gy"] = df["y2"]
    df["size"] = np.sqrt((df["x2"] - df["x1"]).clip(lower=1) * (df["y2"] - df["y1"]).clip(lower=1))

    dt = df.groupby("tid")["t"].diff().median()
    win = max(1, int(round(smooth_sec / dt))) if dt and dt > 0 else 1

    # Velocity from RAW positions, then smoothed. Smoothing positions first and
    # differentiating would bias speed low at both ends of every track (the window is
    # one-sided there), which looks like braking every time a car leaves the frame.
    vx, vy = [], []
    for _, grp in df.groupby("tid", sort=False):
        t = grp["t"].to_numpy()
        if len(t) < 2:
            vx.append(np.zeros(len(t)))
            vy.append(np.zeros(len(t)))
            continue
        vx.append(np.gradient(grp["gx"].to_numpy(), t))
        vy.append(np.gradient(grp["gy"].to_numpy(), t))
    df["vx"] = np.concatenate(vx)
    df["vy"] = np.concatenate(vy)

    g = df.groupby("tid", sort=False)
    for col in ("gx", "gy", "size", "vx", "vy"):
        df[col] = g[col].transform(lambda s: s.rolling(win, center=centered, min_periods=1).mean())
    df["speed"] = np.hypot(df["vx"], df["vy"]) / df["size"]
    df["heading"] = np.degrees(np.arctan2(df["vy"], df["vx"]))
    return df.reset_index(drop=True)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [i, j] (inclusive) where mask is True."""
    if len(mask) == 0:
        return []
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def fill_gaps(mask: np.ndarray, t: np.ndarray, max_gap: float) -> np.ndarray:
    """Set short False stretches between two True stretches to True (flicker removal)."""
    out = mask.astype(bool).copy()
    true_runs = runs(out)
    for (_, j), (i2, _) in zip(true_runs, true_runs[1:]):
        if t[i2] - t[j] <= max_gap:
            out[j:i2] = True
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ttc_matrix(pos: np.ndarray, vel: np.ndarray, size: np.ndarray,
               radius_factor: float | np.ndarray) -> np.ndarray:
    """Pairwise time-to-collision (s) for n road users; inf on the diagonal and for misses.

    Each user is a disc of radius radius_factor * size at its ground point (radius_factor
    may be an (n, n) array to set it per pair). For every pair
    we solve |p + v t| = r for the smallest t >= 0, with p, v the relative position and
    velocity and r the sum of radii. Already touching -> 0. Separating or parallel -> inf.
    """
    p = pos[None, :, :] - pos[:, None, :]
    v = vel[None, :, :] - vel[:, None, :]
    r = radius_factor * (size[None, :] + size[:, None])
    a = (v * v).sum(-1)
    b = 2 * (p * v).sum(-1)
    c = (p * p).sum(-1) - r * r
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    ttc = np.where((a > 1e-9) & (b < 0) & (disc >= 0), t, np.inf)
    ttc = np.where(c <= 0, 0.0, ttc)
    np.fill_diagonal(ttc, np.inf)
    return ttc
