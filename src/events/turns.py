"""illegal_u_turn and illegal_turn.

A turn is a stretch where the heading changes faster than `rate_deg_s`. Its start and
end are exactly the annotation convention ("starts turning" / "completes the turn").
  illegal_u_turn: total heading change >= u_turn_deg inside a no-U-turn polygon.
  illegal_turn:   total change >= turn_deg on a track that goes from a forbidden
                  entry zone to its exit zone.
Both need zones in configs/scene.yaml; without them they never fire (on purpose).
"""
from __future__ import annotations

import numpy as np

from ..kinematics import fill_gaps, runs
from ..scene import in_any, in_polygon
from . import Context


def turn_segments(g, rate_deg_s: float, moving_speed: float) -> list[tuple[int, int, float]]:
    """(i, j, total_degrees) for each continuous turning stretch of one track."""
    t = g["t"].to_numpy()
    moving = g["speed"].to_numpy() > moving_speed
    if moving.sum() < 3:
        return []
    heading = np.degrees(np.unwrap(np.radians(g["heading"].to_numpy())))
    rate = np.abs(np.gradient(heading, t))
    out = []
    for i, j in runs(fill_gaps((rate > rate_deg_s) & moving, t, 0.75)):
        if j > i:
            out.append((i, j, abs(heading[j] - heading[i])))
    return out


def detect(ctx: Context) -> list[list]:
    c, k, s = ctx.cfg.rules.turns, ctx.cfg.kinematics, ctx.scene
    if not s.u_turn_prohibited and not s.forbidden_moves:
        return []
    events = []
    for _, g in ctx.vehicles().groupby("tid"):
        t, x, y = g["t"].to_numpy(), g["gx"].to_numpy(), g["gy"].to_numpy()
        for i, j, deg in turn_segments(g, c.rate_deg_s, k.moving_speed):
            mid = (i + j) // 2
            if deg >= c.u_turn_deg and in_any(s.u_turn_prohibited, x[mid], y[mid]):
                events.append([t[i], t[j], "illegal_u_turn"])
            elif deg >= c.turn_deg:
                for src, dst in s.forbidden_moves:
                    if in_polygon(src, x[0], y[0]) and in_polygon(dst, x[-1], y[-1]):
                        events.append([t[i], t[j], "illegal_turn"])
                        break
    return events
