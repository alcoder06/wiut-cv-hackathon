"""wrong_way: a moving vehicle travelling against the learned lane direction.

Only cells where traffic direction is consistent (high coherence) are trusted, which
automatically ignores intersections where vehicles legitimately move every way.
"""
from __future__ import annotations

import numpy as np

from ..kinematics import fill_gaps, runs
from . import Context


def detect(ctx: Context) -> list[list]:
    c, k = ctx.cfg.rules.wrong_way, ctx.cfg.kinematics
    events = []
    for _, g in ctx.vehicles().groupby("tid"):
        x, y = g["gx"].to_numpy(), g["gy"].to_numpy()
        vel = g[["vx", "vy"]].to_numpy()
        unit = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9)
        lane, coh = ctx.scene.lane_direction(x, y)
        against = ((g["speed"].to_numpy() > k.moving_speed)
                   & g["on_road"].to_numpy()
                   & (coh >= ctx.cfg.scene.min_coherence)
                   & ((unit * lane).sum(axis=1) < c.against_cos))
        if c.ignore_junction:          # turning inside the junction is not wrong-way driving
            against &= ~ctx.scene.in_junction(x, y)
        t = g["t"].to_numpy()
        for i, j in runs(fill_gaps(against, t, 0.75)):
            if t[j] - t[i] >= c.min_len_sec:
                events.append([t[i], t[j], "wrong_way"])
    return events
