"""congestion: standstill or crawling traffic across a whole direction of travel.

Each vehicle observation is assigned to one of two direction groups by projecting the
learned lane direction at its position onto the scene's main traffic axis. For every
1 s bin and group: enough vehicles present and their median speed is a crawl.
"""
from __future__ import annotations

import numpy as np

from ..kinematics import runs
from . import Context


def detect(ctx: Context) -> list[list]:
    c = ctx.cfg.rules.congestion
    v = ctx.vehicles()
    v = v[v["on_road"]]
    if v.empty:
        return []
    lane_dir, _ = ctx.scene.lane_direction(v["gx"].to_numpy(), v["gy"].to_numpy())
    group = np.sign(lane_dir @ ctx.scene.flow.main_axis())
    df = v.assign(group=group, bin=(v["t"] // c.bin_sec).astype(int))
    stats = df.groupby(["group", "bin"]).agg(n=("tid", "nunique"), speed=("speed", "median"))

    events = []
    n_bins = int(ctx.info.duration // c.bin_sec) + 1
    for g in (-1.0, 1.0):
        if g not in stats.index.get_level_values(0):
            continue
        s = stats.loc[g].reindex(range(n_bins))
        jammed = ((s["n"] >= c.min_vehicles) & (s["speed"] < c.crawl_speed)).fillna(False).to_numpy()
        for i, j in runs(jammed):
            start, end = i * c.bin_sec, (j + 1) * c.bin_sec
            if end - start >= c.min_len_sec:
                events.append([start, end, "congestion"])
    return events
