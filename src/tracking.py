"""Offline detection + tracking pass for Part A, with an on-disk cache for development.

Output is a tidy DataFrame, one row per (frame, track):
    frame, t, tid, cls, conf, x1, y1, x2, y2
plus a per-frame DataFrame of scene measurements (signal colours) taken in the same pass,
so the video is decoded exactly once.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .video import VideoInfo, iter_frames, stride_for

TRACK_COLS = ["frame", "t", "tid", "cls", "conf", "x1", "y1", "x2", "y2"]
FrameHook = Callable[[np.ndarray], dict]


def make_tracker(fps: float, cfg):
    import supervision as sv

    return sv.ByteTrack(
        track_activation_threshold=cfg.track_activation_threshold,
        # supervision measures the buffer in frames at 30 fps and rescales by frame_rate
        lost_track_buffer=int(round(cfg.lost_track_buffer_sec * 30)),
        minimum_matching_threshold=cfg.minimum_matching_threshold,
        frame_rate=max(1, int(round(fps))),
    )


def to_sv(det):
    import supervision as sv

    return sv.Detections(xyxy=det.xyxy, confidence=det.conf, class_id=det.cls)


def _cache_path(info: VideoInfo, cfg) -> Path | None:
    root = os.environ.get("TRAFFIC_CACHE")
    if not root:
        return None
    st = os.stat(info.path)
    key = json.dumps([Path(info.path).name, st.st_size, cfg.detector, cfg.sampling.part_a_fps,
                      cfg.tracker], sort_keys=True, default=str)
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(root) / f"{Path(info.path).stem}_{h}"


def track_video(info: VideoInfo, frame_hook: FrameHook | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    cache = _cache_path(info, cfg)
    if cache and cache.with_suffix(".tracks.parquet").exists():
        return (pd.read_parquet(cache.with_suffix(".tracks.parquet")),
                pd.read_parquet(cache.with_suffix(".frames.parquet")))

    from .detector import get_detector

    detector = get_detector()
    names = {int(k): v for k, v in cfg.detector.classes.items()}
    stride = stride_for(info.fps, cfg.sampling.part_a_fps)
    tracker = make_tracker(info.fps / stride, cfg.tracker)
    batch_size = int(cfg.detector.batch)

    max_w = int(cfg.detector.max_input_width)
    to_full = info.width / min(info.width, max_w)   # detector boxes -> full-resolution pixels

    rows, frame_rows = [], []
    buf: list[tuple[int, float, np.ndarray]] = []

    def flush():
        for (idx, t, _), det in zip(buf, detector.predict([f for _, _, f in buf])):
            det.xyxy *= to_full
            tracked = tracker.update_with_detections(to_sv(det))
            for (x1, y1, x2, y2), c, k, tid in zip(tracked.xyxy, tracked.confidence,
                                                   tracked.class_id, tracked.tracker_id):
                rows.append((idx, t, int(tid), names[int(k)], float(c), x1, y1, x2, y2))
        buf.clear()

    # The time cap protects the official run; dev cache builds (TRAFFIC_CACHE set) must be complete.
    deadline = float("inf") if cache else time.perf_counter() + cfg.sampling.part_a_max_x * info.duration
    capped = False
    for idx, t, frame, extra in iter_frames(info.path, stride, max_w, frame_hook):
        if time.perf_counter() > deadline:
            print(f"[part A] time cap hit at t={t:.1f}s of {info.duration:.1f}s; "
                  "returning events found so far", file=sys.stderr)
            capped = True
            break
        if frame_hook is not None:
            frame_rows.append({"frame": idx, "t": t, **extra})
        buf.append((idx, t, frame))
        if len(buf) == batch_size:
            flush()
    if buf:
        flush()

    tracks = pd.DataFrame(rows, columns=TRACK_COLS)
    tracks = drop_short_tracks(tracks, cfg.tracker.min_track_len_sec)
    frames = pd.DataFrame(frame_rows) if frame_rows else pd.DataFrame({"frame": [], "t": []})

    if cache and not capped:        # never cache a truncated pass
        cache.parent.mkdir(parents=True, exist_ok=True)
        tracks.to_parquet(cache.with_suffix(".tracks.parquet"))
        frames.to_parquet(cache.with_suffix(".frames.parquet"))
    return tracks, frames


def drop_short_tracks(tracks: pd.DataFrame, min_len_sec: float) -> pd.DataFrame:
    if tracks.empty:
        return tracks
    span = tracks.groupby("tid")["t"].agg(lambda s: s.max() - s.min())
    keep = span[span >= min_len_sec].index
    tracks = tracks[tracks["tid"].isin(keep)].copy()
    # A track's class is its most frequent detection class (car/truck flicker is common).
    tracks["cls"] = tracks.groupby("tid")["cls"].transform(lambda s: s.mode().iat[0])
    return tracks.reset_index(drop=True)
