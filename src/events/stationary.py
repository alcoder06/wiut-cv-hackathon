"""stopped_vehicle: stationary on the carriageway >= 10 s, not queued at a signal.

Start = the moment the vehicle stopped (the full track is known offline, so the start is
naturally "backdated"; no waiting for the 10 s to pass). End = it moves again or its
track ends (removed / left the frame).

ByteTrack often loses a parked vehicle behind passing traffic and gives it a new id.
We stitch stationary runs that sit on the same spot with only a short gap, the same
trick the AI City Challenge winners used to get accurate start times.
"""
from __future__ import annotations

import numpy as np

from ..kinematics import box_iou, runs
from . import Context


def stationary_runs(ctx: Context) -> list[dict]:
    k = ctx.cfg.kinematics
    out = []
    for tid, g in ctx.vehicles().groupby("tid"):
        # Never seen moving = parked. The learned road area spills onto kerbside parking,
        # and nothing around a parked car stops with it, so the queue test can't catch it.
        if (g["speed"].to_numpy() <= k.moving_speed).all():
            continue
        stopped = (g["speed"].to_numpy() < k.stopped_speed) & g["on_road"].to_numpy()
        t = g["t"].to_numpy()
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
        for i, j in runs(stopped):
            out.append({"tid": tid, "start": t[i], "end": t[j],
                        "box": np.median(boxes[i: j + 1], axis=0)})
    return out


def stitch(items: list[dict], max_gap: float, min_iou: float) -> list[dict]:
    items = sorted(items, key=lambda r: r["start"])
    merged: list[dict] = []
    for r in items:
        for m in merged:
            if 0 <= r["start"] - m["end"] <= max_gap and box_iou(m["box"], r["box"]) >= min_iou:
                m["end"] = max(m["end"], r["end"])
                break
        else:
            merged.append(dict(r))
    return merged


def in_signal_queue(ctx: Context, run: dict) -> bool:
    """Queued at a signal = upstream of a stop line while that approach shows red."""
    from ..scene import side_of_line
    from .signal import signal_state, upstream_sign

    x, y = (run["box"][0] + run["box"][2]) / 2, run["box"][3]
    for a in ctx.scene.approaches:
        if side_of_line(a.stop_line, x, y) != upstream_sign(a):
            continue
        state = signal_state(ctx, a.name)
        if state is None:
            continue
        t, s = state
        during = (t >= run["start"]) & (t <= run["end"])
        if during.any() and (s[during] == "red").mean() > 0.5:
            return True
    return False


def in_queue(ctx: Context, run: dict) -> bool:
    """Queued = its neighbours were mostly stopped too while it waited.

    A broken-down car sits still while traffic flows around it; a car at a red light sits
    still together with the cars around it. This needs no signal or stop-line config, which
    matters because on the sample video nearly every 10 s+ stop was a signal queue.
    """
    c, k = ctx.cfg.rules.stopped_vehicle, ctx.cfg.kinematics
    x1, y1, x2, y2 = run["box"]
    cx, cy, size = (x1 + x2) / 2, y2, np.sqrt((x2 - x1) * (y2 - y1))
    v = ctx.vehicles()
    near = v[(v["tid"] != run["tid"]) & v["t"].between(run["start"], run["end"])
             & (np.hypot(v["gx"] - cx, v["gy"] - cy) < c.neighbour_radius * size)]
    if near.empty:
        return False                     # isolated stop: nothing to compare with
    per_second = near.groupby((near["t"] // 1).astype(int))["speed"].median()
    return float((per_second < k.moving_speed).mean()) >= c.queue_share


def detect(ctx: Context) -> list[list]:
    c = ctx.cfg.rules.stopped_vehicle
    stitched = stitch(stationary_runs(ctx), c.stitch_gap_sec, c.stitch_iou)
    return [[r["start"], r["end"], "stopped_vehicle"] for r in stitched
            if r["end"] - r["start"] >= c.min_stop_sec
            and not in_queue(ctx, r) and not in_signal_queue(ctx, r)]
