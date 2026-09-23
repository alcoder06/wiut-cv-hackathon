"""Scene layout: hand-drawn zones (configs/scene.yaml) + a flow field learned from traffic.

The flow field is a grid over the image. For every cell we store how often moving
vehicles pass through it and their dominant direction. From that we get:
  * the carriageway (cells vehicles actually drive through),
  * the legal direction of travel per cell (for wrong-way),
  * two direction groups (for per-direction congestion).
Learning it from trajectories means a new camera works without hand-drawing lanes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


@dataclass
class FlowField:
    grid: int
    width: int
    height: int
    count: np.ndarray        # (ny, nx) moving observations per cell
    dir_sum: np.ndarray      # (ny, nx, 2) sum of unit direction vectors

    @classmethod
    def learn(cls, tracks: pd.DataFrame, width: int, height: int, grid: int,
              moving_speed: float) -> "FlowField":
        ny, nx = -(-height // grid), -(-width // grid)
        count = np.zeros((ny, nx), np.float32)
        dir_sum = np.zeros((ny, nx, 2), np.float32)
        if not tracks.empty:
            mv = tracks[(tracks["speed"] > moving_speed) & (tracks["cls"] != "person")]
            cx = np.clip((mv["gx"].to_numpy() // grid).astype(int), 0, nx - 1)
            cy = np.clip((mv["gy"].to_numpy() // grid).astype(int), 0, ny - 1)
            norm = np.hypot(mv["vx"], mv["vy"]).to_numpy() + 1e-9
            np.add.at(count, (cy, cx), 1)
            np.add.at(dir_sum, (cy, cx, 0), mv["vx"].to_numpy() / norm)
            np.add.at(dir_sum, (cy, cx, 1), mv["vy"].to_numpy() / norm)
        return cls(grid, width, height, count, dir_sum)

    def merged(self, other: "FlowField") -> "FlowField":
        return FlowField(self.grid, self.width, self.height,
                         self.count + other.count, self.dir_sum + other.dir_sum)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, grid=self.grid, width=self.width, height=self.height,
                            count=self.count, dir_sum=self.dir_sum)

    @classmethod
    def load(cls, path: Path) -> "FlowField":
        z = np.load(path)
        return cls(int(z["grid"]), int(z["width"]), int(z["height"]), z["count"], z["dir_sum"])

    # -- derived views -------------------------------------------------------------
    def direction(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit dominant direction (ny, nx, 2) and coherence in [0, 1] per cell."""
        n = np.linalg.norm(self.dir_sum, axis=2)
        unit = self.dir_sum / (n[..., None] + 1e-9)
        coherence = n / (self.count + 1e-9)
        return unit, coherence

    def road_mask(self, min_obs: int) -> np.ndarray:
        """Full-resolution uint8 mask of the learned carriageway.

        Built from MOVING vehicles only: queue lanes still qualify (cars roll through them
        on every green) while parked cars, which never move, stay off the road."""
        cells = (self.count >= min_obs).astype(np.uint8)
        cells = cv2.morphologyEx(cells, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        full = cv2.resize(cells, (cells.shape[1] * self.grid, cells.shape[0] * self.grid),
                          interpolation=cv2.INTER_NEAREST)
        return full[: self.height, : self.width]

    def main_axis(self) -> np.ndarray:
        """Principal traffic axis as a unit vector (axial average: opposite lanes agree)."""
        unit, _ = self.direction()
        w = self.count
        ang = np.arctan2(unit[..., 1], unit[..., 0])
        a = 0.5 * np.arctan2((w * np.sin(2 * ang)).sum(), (w * np.cos(2 * ang)).sum())
        return np.array([np.cos(a), np.sin(a)])


def grid_for(width: int, cells_across: int) -> int:
    """Cell size in pixels, proportional to resolution so density per cell stays similar."""
    return max(8, round(width / cells_across))


def _denorm(points, w: int, h: int) -> np.ndarray:
    return (np.asarray(points, np.float32) * np.array([w, h], np.float32)).astype(np.float32)


@dataclass
class Approach:
    name: str
    stop_line: np.ndarray          # (2, 2) pixels
    direction: np.ndarray          # unit vector of travel
    signal_roi: tuple[int, int, int, int] | None
    intersection: np.ndarray | None


def load_manual(path: Path) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}


def load_approaches(manual: dict, w: int, h: int) -> list[Approach]:
    """Signal approaches alone: the tracking pass needs the lamp ROIs before any flow exists."""
    out = []
    for a in manual.get("approaches") or []:
        roi = a.get("signal_roi")
        out.append(Approach(
            name=a["name"],
            stop_line=_denorm(a["stop_line"], w, h),
            direction=np.asarray(a["direction"], np.float32) / np.linalg.norm(a["direction"]),
            signal_roi=tuple(int(v) for v in _denorm([roi[:2], roi[2:]], w, h).ravel()) if roi else None,
            intersection=_denorm(a["intersection"], w, h) if a.get("intersection") else None,
        ))
    return out


@dataclass
class Scene:
    width: int
    height: int
    flow: FlowField
    road: np.ndarray                               # uint8 (H, W)
    crosswalks: list[np.ndarray] = field(default_factory=list)
    solid_lines: list[np.ndarray] = field(default_factory=list)
    approaches: list[Approach] = field(default_factory=list)
    u_turn_prohibited: list[np.ndarray] = field(default_factory=list)
    forbidden_moves: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, manual: dict, flow: FlowField, min_obs: int) -> "Scene":
        w, h = flow.width, flow.height
        m = manual
        if m.get("carriageway"):
            road = np.zeros((h, w), np.uint8)
            cv2.fillPoly(road, [_denorm(p, w, h).astype(np.int32) for p in m["carriageway"]], 1)
        else:
            road = flow.road_mask(min_obs)
        moves = m.get("movements") or {}
        return cls(
            width=w, height=h, flow=flow, road=road,
            crosswalks=[_denorm(p, w, h) for p in m.get("crosswalks") or []],
            solid_lines=[_denorm(p, w, h) for p in m.get("solid_lines") or []],
            approaches=load_approaches(m, w, h),
            u_turn_prohibited=[_denorm(p, w, h) for p in moves.get("u_turn_prohibited") or []],
            forbidden_moves=[(_denorm(f["from"], w, h), _denorm(f["to"], w, h))
                             for f in moves.get("forbidden") or []],
        )

    # -- point queries (vectorised over arrays of x, y) ------------------------------
    def on_road(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xi = np.clip(np.asarray(x).astype(int), 0, self.width - 1)
        yi = np.clip(np.asarray(y).astype(int), 0, self.height - 1)
        return self.road[yi, xi] > 0

    def crosswalk_at(self, x, y, grow_px: int = 0) -> np.ndarray:
        """Index+1 of the crosswalk under each point (0 = none), optionally grown by grow_px.

        Polygons are painted into a mask once and cached, so a lookup is one array index
        instead of a Python-level polygon test per point (that cost minutes per video).
        """
        key = ("crosswalk_mask", grow_px)
        if key not in self._cache:
            mask = np.zeros((self.height, self.width), np.uint8)
            for i, poly in enumerate(self.crosswalks):
                layer = np.zeros_like(mask)
                cv2.fillPoly(layer, [poly.astype(np.int32)], 1)
                if grow_px:
                    layer = cv2.dilate(layer, np.ones((2 * grow_px + 1,) * 2, np.uint8))
                mask[(layer > 0) & (mask == 0)] = i + 1
            self._cache[key] = mask
        mask = self._cache[key]
        xi = np.clip(np.asarray(x).astype(int), 0, self.width - 1)
        yi = np.clip(np.asarray(y).astype(int), 0, self.height - 1)
        return mask[yi, xi]

    def lane_direction(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unit, coherence = self.flow.direction()
        g = self.flow.grid
        cx = np.clip((np.asarray(x) // g).astype(int), 0, unit.shape[1] - 1)
        cy = np.clip((np.asarray(y) // g).astype(int), 0, unit.shape[0] - 1)
        return unit[cy, cx], coherence[cy, cx]


def in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def in_any(polys: list[np.ndarray], x: float, y: float) -> bool:
    return any(in_polygon(p, x, y) for p in polys)


def side_of_line(line: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed side of points relative to a 2-point line (cross product sign)."""
    (x1, y1), (x2, y2) = line[0], line[-1]
    return np.sign((x2 - x1) * (np.asarray(y) - y1) - (y2 - y1) * (np.asarray(x) - x1))
