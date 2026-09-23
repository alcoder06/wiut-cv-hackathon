"""jaywalking and failure_to_yield.

jaywalking: a pedestrian's feet on the carriageway, outside every crosswalk. The road
mask is eroded a little so people waiting on the kerb edge don't count. People riding a
bicycle/motorcycle are detected as `person` too; they're removed by overlap with a
two-wheeler box in the same frame.

failure_to_yield: a moving vehicle inside a crosswalk while a pedestrian is on it.
Start/end = the vehicle entering/leaving the crosswalk.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from ..kinematics import runs
from ..scene import in_any, in_polygon
from . import Context


def walkers(ctx: Context) -> pd.DataFrame:
    """Person rows that are not riders of a two-wheeler."""
    people = ctx.people()
    riders = ctx.tracks[ctx.tracks["cls"].isin(("bicycle", "motorcycle"))]
    if people.empty or riders.empty:
        return people
    m = people.reset_index().merge(riders[["frame", "x1", "y1", "x2", "y2"]], on="frame",
                                   suffixes=("", "_r"))
    ix = (np.minimum(m["x2"], m["x2_r"]) - np.maximum(m["x1"], m["x1_r"])).clip(lower=0)
    iy = (np.minimum(m["y2"], m["y2_r"]) - np.maximum(m["y1"], m["y1_r"])).clip(lower=0)
    person_area = (m["x2"] - m["x1"]) * (m["y2"] - m["y1"])
    rider_idx = m.loc[(ix * iy) / person_area > 0.3, "index"].unique()
    return people.drop(index=rider_idx)


def jaywalking(ctx: Context) -> list[list]:
    c = ctx.cfg.rules.jaywalking
    road = cv2.erode(ctx.scene.road, np.ones((c.road_erode_px, c.road_erode_px), np.uint8))
    h, w = road.shape
    events = []
    for _, g in walkers(ctx).groupby("tid"):
        x = np.clip(g["gx"].to_numpy().astype(int), 0, w - 1)
        y = np.clip(g["gy"].to_numpy().astype(int), 0, h - 1)
        on_road = road[y, x] > 0
        if ctx.scene.crosswalks:
            on_road &= ~np.array([in_any(ctx.scene.crosswalks, a, b) for a, b in zip(x, y)])
        t = g["t"].to_numpy()
        for i, j in runs(on_road):
            if t[j] - t[i] >= c.min_len_sec:
                events.append([t[i], t[j], "jaywalking"])
    return events


def failure_to_yield(ctx: Context) -> list[list]:
    if not ctx.scene.crosswalks:
        return []
    c, k = ctx.cfg.rules.failure_to_yield, ctx.cfg.kinematics
    people = walkers(ctx)
    events = []
    for cw in ctx.scene.crosswalks:
        grown = _grow(cw, c.ped_margin_px)
        ped_frames = set(people.loc[[in_polygon(grown, x, y) for x, y in zip(people["gx"], people["gy"])], "frame"]) \
            if not people.empty else set()
        if not ped_frames:
            continue
        for _, g in ctx.vehicles().groupby("tid"):
            inside = np.array([in_polygon(cw, x, y) for x, y in zip(g["gx"], g["gy"])])
            t, frames, speed = g["t"].to_numpy(), g["frame"].to_numpy(), g["speed"].to_numpy()
            for i, j in runs(inside):
                if speed[i: j + 1].max() > k.moving_speed and ped_frames & set(frames[i: j + 1]):
                    events.append([t[i], t[j], "failure_to_yield"])
    return events


def _grow(poly: np.ndarray, px: float) -> np.ndarray:
    centre = poly.mean(axis=0)
    d = poly - centre
    scale = 1 + px / (np.linalg.norm(d, axis=1).mean() + 1e-9)
    return (centre + d * scale).astype(np.float32)


def detect(ctx: Context) -> list[list]:
    return jaywalking(ctx) + failure_to_yield(ctx)
