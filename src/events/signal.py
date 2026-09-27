"""Traffic-signal state, red_light and stop_line.

Signal state: share of lit red vs lit green pixels (HSV) in the lamp ROI of each approach,
measured on every analysed frame during the tracking pass, then smoothed with a
majority vote over `vote_sec` so a single glare frame can't flip it.

The camera often can't read the approach's own lamp (the gantry heads here are side-on),
but can read another lamp locked to the same cycle with an offset. On C3897 the readable
left-pole lamp turns red ~4.5 s before boulevard traffic stops and green ~0.6 s after it
starts, every cycle; read naively that is ~3 false red_light events per 75 s cycle. So the
approach's effective red is the lamp's red shifted by red_delay_sec / red_early_end_sec.

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
from ..scene import Approach, side_of_line
from . import Context

# Lit-lamp HSV ranges (OpenCV hue is 0..179; red wraps around 0). Measured on the real
# lamps of C3897: saturation >= 90 and value >= 150 separates a lit lamp from an unlit one.
RED = [((0, 90, 150), (10, 255, 255)), ((165, 90, 150), (179, 255, 255))]
GREEN = [((40, 90, 150), (100, 255, 255))]


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


def effective_red(state, a: Approach) -> np.ndarray:
    """Per sample: is THIS approach on red? The lamp's red runs, each moved by the
    approach's offsets (start later by red_delay_sec, end red_early_end_sec before green)."""
    ts, labels = state
    red = np.zeros(len(ts), bool)
    for i, j in runs(labels == "red"):
        # Red is always followed by green, so the lamp's red ending IS the green onset. Don't
        # wait for a green reading: at dusk (C3905) the green lamp never reads, and waiting for
        # it let two cars pulling away on green count as red_light.
        green_at = ts[j + 1] if j + 1 < len(ts) else np.inf
        red |= (ts >= ts[i] + a.red_delay_sec) & (ts < green_at - a.red_early_end_sec) & (ts <= ts[j])
    return red


def along_line(line: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Position along the stop line: 0 at one end, 1 at the other. Used to keep "past the
    line" to the lanes the line actually spans; as an infinite line it also caught cars on
    the neighbouring arm (5 false stop_line events on C3897)."""
    a, b = line[0], line[-1]
    d = b - a
    return ((np.asarray(x) - a[0]) * d[0] + (np.asarray(y) - a[1]) * d[1]) / float(d @ d)


def _at(ts: np.ndarray, values: np.ndarray, t: float):
    return values[min(np.searchsorted(ts, t), len(ts) - 1)]


def _mask_of(poly: np.ndarray | None, w: int, h: int) -> np.ndarray | None:
    if poly is None:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


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
    k, c = ctx.cfg.kinematics, ctx.cfg.rules.red_light
    w, h = ctx.info.width, ctx.info.height
    events = []
    for a in ctx.scene.approaches:
        state = signal_state(ctx, a.name)
        if state is None:
            continue
        ts, labels = state
        red = effective_red(state, a)
        box = _mask_of(a.intersection, w, h)
        up = upstream_sign(a)
        for _, g in ctx.vehicles().groupby("tid"):
            x, y, t = g["gx"].to_numpy(), g["gy"].to_numpy(), g["t"].to_numpy()
            within = (along_line(a.stop_line, x, y) >= -0.05) & (along_line(a.stop_line, x, y) <= 1.05)
            side = np.where(within, side_of_line(a.stop_line, x, y), 0)   # 0 = not level with the line
            vel = g[["vx", "vy"]].to_numpy()
            heading_cos = (vel @ a.direction) / (np.linalg.norm(vel, axis=1) + 1e-9)
            speed = g["speed"].to_numpy()
            # signed distance past the line in box sizes (>0 = downstream)
            (x1, y1), (x2, y2) = a.stop_line[0], a.stop_line[-1]
            normal = np.array([y2 - y1, x1 - x2], float); normal /= np.linalg.norm(normal) + 1e-9
            if normal @ a.direction < 0:
                normal = -normal
            past_by = ((x - x1) * normal[0] + (y - y1) * normal[1]) / g["size"].to_numpy()
            in_box = (box[np.clip(y.astype(int), 0, h - 1), np.clip(x.astype(int), 0, w - 1)] > 0
                      if box is not None else np.zeros(len(t), bool))

            # red_light: crossing upstream -> downstream on red, driving ALONG the approach, having
            # come up to the line from behind it (cross traffic passing over the line's end was
            # the false alarm on C3896)
            for i in range(1, len(t)):
                if (side[i - 1] == up and side[i] == -up and _at(ts, red, t[i])
                        and heading_cos[i] >= c.min_heading_cos
                        and t[i] - t[max(0, np.flatnonzero(side[:i] == up)[0])] >= c.min_approach_sec):
                    # ...and drives on into the junction. Creeping over the line and waiting
                    # there is stop_line, not red_light (C3896, 79 s).
                    after = (t >= t[i]) & (t <= t[i] + c.enter_within_sec)
                    keeps_going = speed[after].min() > k.stopped_speed
                    entered = box is None or bool((in_box & after).any())
                    if keeps_going and entered:
                        events.append([t[i - 1], _exit_time(in_box[i:], t[i:]), "red_light"])
                    break

            # a vehicle never seen moving is parked or a phantom box (C3896: the median kerb)
            if speed.max() <= k.moving_speed:
                continue

            # stop_line: stopped past the line (not in the intersection) on red, until green
            # clearly past the line, not just sitting on it with its nose over
            past = (side == -up) & (past_by >= c.past_margin) & ~in_box & (speed < k.stopped_speed)
            for i, j in runs(past):
                if not _at(ts, red, t[i]):
                    continue
                # ends when this approach turns green: the lamp's red ending, shifted by the offset
                red_over = np.flatnonzero((ts > t[i]) & (labels != "red"))
                end = ts[red_over[0]] - a.red_early_end_sec if len(red_over) else t[j]
                end = min(end, t[min(j + 1, len(t) - 1)])      # or earlier, when it moves on / leaves
                events.append([t[i], max(end, t[i] + 0.5), "stop_line"])
    return events
