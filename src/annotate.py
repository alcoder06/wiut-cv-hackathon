"""Annotated video: boxes with track ids, 2 s trails, the learned road, drawn zones, and a
banner with the time, the events active at that moment and the risk score.

Used by scripts/render.py, scripts/build_site.py and the web demo. Frames are shrunk
to the output width before drawing (drawing on 4K and then shrinking costs ~5x more),
and written as H.264 with PyAV so every browser can play the result.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable

import cv2
import numpy as np

from .stats import GROUPS
from .video import downscale

# BGR, the same hue per group as the website's charts (web/static/style.css).
COLOURS = {"car": (214, 120, 42), "person": (52, 104, 235), "heavy": (122, 175, 27),
           "two_wheeler": (0, 161, 237), "other": (200, 200, 200)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class _Tracks:
    """Fast per-frame lookups over the tracks table (a DataFrame filter per box is too slow)."""

    def __init__(self, tracks, scale: float):
        self.scale = scale
        self.by_frame = {f: g for f, g in tracks.groupby("frame")} if len(tracks) else {}
        self.frames = np.array(sorted(self.by_frame))
        self.paths = {tid: (g["t"].to_numpy(), g["gx"].to_numpy() * scale, g["gy"].to_numpy() * scale)
                      for tid, g in tracks.groupby("tid")} if len(tracks) else {}

    def at(self, idx: int):
        """Boxes of the last analysed frame at or before `idx` (Part A samples every few frames)."""
        k = np.searchsorted(self.frames, idx, side="right") - 1
        return self.by_frame[self.frames[k]] if k >= 0 else None

    def trail(self, tid: int, t: float, seconds: float = 2.0, max_step: float = 40.0) -> np.ndarray:
        """Last `seconds` of the track's ground point, cut at any jump over `max_step` px
        (a tracker id switch would otherwise draw a line across the frame)."""
        ts, xs, ys = self.paths[tid]
        i, j = np.searchsorted(ts, t - seconds), np.searchsorted(ts, t, side="right")
        pts = np.stack([xs[i:j], ys[i:j]], axis=1)
        jumps = np.nonzero(np.linalg.norm(np.diff(pts, axis=0), axis=1) > max_step)[0]
        if len(jumps):
            pts = pts[jumps[-1] + 1:]
        return pts.astype(np.int32)


def _static_layer(scene, width: int, height: int, scale: float):
    """Road tint and zone outlines, drawn once at output size: (tint, outlines, outline_mask)."""
    tint = np.zeros((height, width, 3), np.uint8)
    tint[cv2.resize(scene.road, (width, height), interpolation=cv2.INTER_NEAREST) > 0] = (0, 160, 0)
    lines = np.zeros_like(tint)
    for poly in scene.crosswalks:
        cv2.polylines(lines, [(poly * scale).astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
    for line in scene.solid_lines:
        cv2.polylines(lines, [(line * scale).astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
    for a in scene.approaches:
        p, q = (a.stop_line[:2] * scale).astype(int)
        cv2.line(lines, tuple(p), tuple(q), (0, 0, 255), 3, cv2.LINE_AA)
    return tint, lines, lines.any(axis=2)


def _banner(img: np.ndarray, t: float, active: list, risk: float | None) -> None:
    w = img.shape[1]
    ui = w / 960
    h = int((34 + 26 * len(active)) * ui)
    img[:h] = (img[:h] * 0.35).astype(np.uint8)
    cv2.putText(img, f"t = {t:6.1f} s", (int(10 * ui), int(24 * ui)), FONT, 0.65 * ui, (255, 255, 255), 2, cv2.LINE_AA)
    if risk is not None:
        x0, bw = int(w - 250 * ui), int(150 * ui)
        colour = (59, 59, 208) if risk >= 0.5 else (255, 255, 255)
        cv2.putText(img, "risk", (x0 - int(50 * ui), int(24 * ui)), FONT, 0.6 * ui, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, int(10 * ui)), (x0 + bw, int(26 * ui)), (255, 255, 255), 1)
        cv2.rectangle(img, (x0, int(10 * ui)), (x0 + int(bw * risk), int(26 * ui)), colour, -1)
        cv2.putText(img, f"{risk:.2f}", (x0 + bw + int(8 * ui), int(24 * ui)), FONT, 0.6 * ui, colour, 2, cv2.LINE_AA)
    for i, (s, e, label) in enumerate(active):
        cv2.putText(img, f"{label}  {s:.1f}-{e:.1f} s", (int(10 * ui), int((54 + 26 * i) * ui)),
                    FONT, 0.6 * ui, (80, 80, 255), 2, cv2.LINE_AA)


def _risk_at(risk: list[list[float]] | None):
    if not risk:
        return lambda t: None
    ts = np.array([r[0] for r in risk])
    vs = np.array([r[1] for r in risk])
    return lambda t: float(vs[max(0, np.searchsorted(ts, t, side="right") - 1)])


def render(video: str, ctx, events: list[list], out_path: str, width: int = 960,
           risk: list[list[float]] | None = None, start: float = 0.0, end: float | None = None,
           progress: Callable[[float], None] | None = None) -> None:
    """Write the annotated H.264 mp4 for [start, end] of `video` to `out_path`."""
    import av  # only the rendering tools need PyAV; the submission never imports this module

    info = ctx.info
    width = min(width, info.width) // 2 * 2
    scale = width / info.width
    height = round(info.height * scale) // 2 * 2
    tracks = _Tracks(ctx.tracks, scale)
    tint, lines, lines_mask = _static_layer(ctx.scene, width, height, scale)
    risk_at = _risk_at(risk)
    end = info.duration if end is None else min(end, info.duration)

    out = av.open(out_path, mode="w", options={"movflags": "+faststart"})
    stream = out.add_stream("libx264", rate=Fraction(info.fps).limit_denominator(1001))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "28", "preset": "veryfast"}

    cap = cv2.VideoCapture(info.path)
    idx = int(round(start * info.fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    try:
        while idx / info.fps < end:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / info.fps
            img, _ = downscale(frame, width)
            img = cv2.resize(img, (width, height)) if img.shape[:2] != (height, width) else img
            img = cv2.addWeighted(img, 1.0, tint, 0.18, 0)
            img[lines_mask] = lines[lines_mask]
            boxes = tracks.at(idx)
            if boxes is not None:
                for tid, cls, x1, y1, x2, y2 in boxes[["tid", "cls", "x1", "y1", "x2", "y2"]].itertuples(index=False):
                    col = COLOURS[GROUPS.get(cls, "other")]
                    p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
                    cv2.rectangle(img, p1, p2, col, 2)
                    cv2.putText(img, f"{cls} {tid}", (p1[0], p1[1] - 4), FONT, 0.4 * width / 960, col, 1, cv2.LINE_AA)
                    trail = tracks.trail(tid, t)
                    if len(trail) > 1:
                        cv2.polylines(img, [trail], False, col, 2, cv2.LINE_AA)
            _banner(img, t, [e for e in events if e[0] <= t <= e[1]], risk_at(t))
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                out.mux(packet)
            idx += 1
            if progress and idx % 30 == 0:
                progress(min(1.0, (t - start) / max(end - start, 1e-6)))
        for packet in stream.encode():
            out.mux(packet)
    finally:
        cap.release()
        out.close()
