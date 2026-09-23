"""Traffic-signal state, red_light and stop_line.

Signal state: share of red vs green pixels (HSV) in the lamp ROI of each approach,
measured on every analysed frame during the tracking pass, then smoothed with a
majority vote over `vote_sec` so a single glare frame can't flip it.

red_light: the vehicle's ground point crosses the stop line (upstream -> downstream,
moving in the approach direction) while the vote says red. End = leaves the
intersection polygon, or its track ends.

stop_line: stationary downstream of the stop line on red but not inside the
intersection. End = signal turns green.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..kinematics import runs
from ..scene import Approach, in_polygon, side_of_line
from . import Context

# HSV ranges (OpenCV hue is 0..179). Red wraps around 0.
RED = [((0, 90, 120), (10, 255, 255)), ((165, 90, 120), (179, 255, 255))]
GREEN = [((45, 80, 110), (95, 255, 255))]


def _frac(hsv: np.ndarray, ranges) -> float:
    mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in ranges)
    return float((mask > 0).mean())


def signal_colours(frame: np.ndarray, approaches: list[Approach]) -> dict:
    """Frame hook for the tracking pass: red/green pixel fractions per approach."""
    out = {}
    for a in approaches:
        if a.signal_roi is None:
            continue
        x1, y1, x2, y2 = a.signal_roi
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        out[f"{a.name}_red"] = _frac(hsv, RED)
        out[f"{a.name}_green"] = _frac(hsv, GREEN)
    return out


def signal_state(ctx: Context, name: str):
    """(t, state[]) with state in {"red", "green", "unknown"}, majority-voted. None if unmeasured."""
    f = ctx.frames
    if f.empty or f"{name}_red" not in f:
        return None
    c = ctx.cfg.rules.red_light
    red, green = f[f"{name}_red"].to_numpy(), f[f"{name}_green"].to_numpy()
    raw = np.where((red >= c.red_min_frac) & (red > green), 1,
                   np.where((green >= c.red_min_frac) & (green > red), 2, 0))
    dt = np.median(np.diff(f["t"])) if len(f) > 1 else 1.0
    win = max(1, int(round(c.vote_sec / dt)))
    voted = raw.copy()
    for i in range(len(raw)):
        lo, hi = max(0, i - win // 2), min(len(raw), i + win // 2 + 1)
        counts = np.bincount(raw[lo:hi], minlength=3)
        voted[i] = counts.argmax()
    labels = np.array(["unknown", "red", "green"])[voted]
    return f["t"].to_numpy(), labels


def upstream_sign(a: Approach) -> float:
    """Side of the stop line that traffic comes from."""
    p = a.stop_line.mean(axis=0) - a.direction * 50
    return float(side_of_line(a.stop_line, p[0], p[1]))


def _state_at(state, t: float) -> str:
    ts, labels = state
    return labels[min(np.searchsorted(ts, t), len(ts) - 1)]


def _exit_time(in_box: np.ndarray, t: np.ndarray) -> float:
    """First time the vehicle is outside the intersection after having entered it;
    the end of its track if it never leaves (left the frame) or there's no polygon."""
    entered = np.flatnonzero(in_box)
    if len(entered):
        out = np.flatnonzero(~in_box[entered[0]:])
        if len(out):
            return float(t[entered[0] + out[0]])
    return float(t[-1])


def detect(ctx: Context) -> list[list]:
    k = ctx.cfg.kinematics
    events = []
    for a in ctx.scene.approaches:
        state = signal_state(ctx, a.name)
        if state is None:
            continue
        up = upstream_sign(a)
        for _, g in ctx.vehicles().groupby("tid"):
            x, y, t = g["gx"].to_numpy(), g["gy"].to_numpy(), g["t"].to_numpy()
            side = side_of_line(a.stop_line, x, y)
            along = g[["vx", "vy"]].to_numpy() @ a.direction
            in_box = (np.array([in_polygon(a.intersection, p, q) for p, q in zip(x, y)])
                      if a.intersection is not None else np.zeros(len(t), bool))

            # red_light: upstream -> downstream crossing while moving forward on red
            for i in range(1, len(t)):
                if side[i - 1] == up and side[i] == -up and along[i] > 0 and _state_at(state, t[i]) == "red":
                    events.append([t[i - 1], _exit_time(in_box[i:], t[i:]), "red_light"])
                    break

            # stop_line: stopped past the line (not in the intersection) on red, until green
            past = (side == -up) & ~in_box & (g["speed"].to_numpy() < k.stopped_speed)
            for i, j in runs(past):
                if _state_at(state, t[i]) != "red":
                    continue
                ts, labels = state
                green = np.where((ts > t[i]) & (labels == "green"))[0]
                end = ts[green[0]] if len(green) else t[j]
                events.append([t[i], end, "stop_line"])
    return events
