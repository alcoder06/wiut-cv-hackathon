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
        # the team's zebra outlines are freehand and a little tight: people at the painted
        # edge are crossing, not jaywalking. Margin is a fraction of frame width (4K = 1080p)
        grow = int(round(c.crosswalk_margin_frac * ctx.info.width))
        on_road = ((road[y, x] > 0) & (ctx.scene.crosswalk_at(x, y, grow_px=grow) == 0)
                   & ~ctx.scene.in_refuge(x, y))
        t = g["t"].to_numpy()
        for i, j in runs(on_road):
            if t[j] - t[i] >= c.min_len_sec:
                events.append([t[i], t[j], "jaywalking"])
    return events


def failure_to_yield(ctx: Context) -> list[list]:
    """A moving vehicle inside a crosswalk while a pedestrian on (or stepping onto) the SAME
    crosswalk is near its path. "Near" matters: the boulevard crossing here spans 7+ lanes,
    and a person at the far end is not being cut off by a car at the near end (without this,
    the sample video produced 25 detections in 5 minutes)."""
    if not ctx.scene.crosswalks:
        return []
    c, k = ctx.cfg.rules.failure_to_yield, ctx.cfg.kinematics
    people = walkers(ctx)
    veh = ctx.vehicles()
    if people.empty or veh.empty:
        return []
    ped = people.assign(cw=ctx.scene.crossing_group_at(people["gx"], people["gy"], grow_px=c.ped_margin_px))
    # on (or stepping onto) the crossing, not standing on an island, and actually walking:
    # "a person waiting on the zebra, not crossing" was the second most common reject
    keep = (ped["cw"] > 0) & ~ctx.scene.in_refuge(ped["gx"], ped["gy"]) & (ped["speed"] >= c.ped_min_speed)
    ped = ped[keep][["frame", "cw", "gx", "gy"]]
    veh = veh.assign(cw=ctx.scene.crossing_group_at(veh["gx"], veh["gy"]))

    # vehicle rows with a pedestrian on the same crosswalk, in the same frame, close to it
    m = veh[veh["cw"] > 0].reset_index().merge(ped, on=["frame", "cw"], suffixes=("", "_p"))
    # "near" in car lengths, capped at a normal car: a bus's box is so big that 2.5 bus
    # sizes covered half the intersection
    car_size = ctx.vehicles().loc[lambda d: d["cls"] == "car", "size"].median()
    reach = c.near_sizes * np.minimum(m["size"], car_size if np.isfinite(car_size) else m["size"])
    close = np.hypot(m["gx"] - m["gx_p"], m["gy"] - m["gy_p"]) < reach
    conflict_rows = set(m.loc[close, "index"])

    events = []
    for _, g in veh.groupby("tid"):
        cw, t, speed = g["cw"].to_numpy(), g["t"].to_numpy(), g["speed"].to_numpy()
        conflict = g.index.isin(conflict_rows)
        for i in np.unique(cw[cw > 0]):
            for a_, b_ in runs(cw == i):
                if speed[a_: b_ + 1].max() > k.moving_speed and conflict[a_: b_ + 1].any():
                    events.append([t[a_], t[b_], "failure_to_yield"])
    return events


def detect(ctx: Context) -> list[list]:
    return jaywalking(ctx) + failure_to_yield(ctx)
