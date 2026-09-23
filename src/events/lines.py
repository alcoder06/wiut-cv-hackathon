"""solid_line_crossing: a vehicle's wheels cross a solid lane marking.

We follow the two bottom corners of the box (the wheels closest to the road). Start =
the first corner changes side of the line; end = both corners are on the new side
("fully in the new lane").
"""
from __future__ import annotations

import numpy as np

from . import Context


def polyline_side(line: np.ndarray, x: np.ndarray, y: np.ndarray, max_dist: np.ndarray) -> np.ndarray:
    """+1/-1 side of the nearest polyline segment; 0 where the point is not near the line."""
    pts = np.stack([x, y], axis=1)
    best_d = np.full(len(pts), np.inf)
    side = np.zeros(len(pts))
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        u = np.clip(((pts - a) @ ab) / (ab @ ab + 1e-9), 0, 1)
        d = np.linalg.norm(pts - (a + u[:, None] * ab), axis=1)
        s = np.sign(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0]))
        closer = d < best_d
        best_d[closer], side[closer] = d[closer], s[closer]
    side[best_d > max_dist] = 0
    return side


def detect(ctx: Context) -> list[list]:
    events = []
    for line in ctx.scene.solid_lines:
        for _, g in ctx.vehicles().groupby("tid"):
            t = g["t"].to_numpy()
            reach = 1.5 * g["size"].to_numpy()
            left = polyline_side(line, g["x1"].to_numpy(), g["y2"].to_numpy(), reach)
            right = polyline_side(line, g["x2"].to_numpy(), g["y2"].to_numpy(), reach)
            both = np.where(left == right, left, 0)
            known = np.flatnonzero(both != 0)
            if len(known) < 2:
                continue
            initial = both[known[0]]
            crossed = np.flatnonzero((left == -initial) | (right == -initial))
            done = np.flatnonzero(both == -initial)
            if len(crossed) and len(done) and done[0] >= crossed[0]:
                events.append([t[crossed[0]], t[done[0]], "solid_line_crossing"])
    return events
