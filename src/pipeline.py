"""Part A end to end: video -> tracks -> kinematics -> scene -> rules -> clean segments."""
from __future__ import annotations

import sys
import time
import traceback

from .config import load_config, resolve
from .events import Context, all_detectors
from .events.signal import signal_colours
from .kinematics import add_kinematics
from .scene import FlowField, Scene, grid_for, load_approaches, load_manual
from .segments import finalize
from .tracking import Progress, track_video
from .video import VideoInfo, probe
from .view import view_transform, warp_grid, warp_manual


def build_scene(tracks, info: VideoInfo, cfg, manual: dict | None = None, view=None) -> Scene:
    """Flow learned from this video, plus the one prebuilt from the sample videos
    (same camera) when it exists at the same resolution: more data, steadier lanes.
    `manual` overrides configs/scene.yaml (tests pass {} to stay camera-independent).
    `view` (src/view.py) moves the prebuilt field into this recording's framing."""
    flow = FlowField.learn(tracks, info.width, info.height,
                           grid_for(info.width, cfg.scene.cells_across), cfg.kinematics.moving_speed)
    prebuilt = resolve(cfg.scene.learned)
    if prebuilt.exists():
        base = FlowField.load(prebuilt).rescaled(info.width, info.height)
        if base is not None and base.grid == flow.grid:
            if view is not None:
                base = FlowField(base.grid, base.width, base.height,
                                 warp_grid(base.count, view, base.width, base.height, base.grid),
                                 warp_grid(base.dir_sum, view, base.width, base.height, base.grid))
            flow = flow.merged(base)
    if manual is None:
        manual = load_manual(resolve(cfg.scene.manual))
    return Scene.build(manual, flow, cfg.scene.min_cell_obs)


def shift_boundaries(events: list[list], cfg) -> list[list]:
    """Per-class start/end corrections learned from labels (scripts/tune.py). A rule that
    fires consistently 0.8 s late loses IoU 0.7 matches a constant shift wins back."""
    shifts = cfg.segments.get("boundary_shift") or {}
    return [[s + shifts.get(c, (0, 0))[0], e + shifts.get(c, (0, 0))[1], c] for s, e, c in events]


def analyse(path: str, progress: Progress | None = None) -> tuple[Context, list[list]]:
    """Run Part A and return the context too (scripts and the web demo reuse it)."""
    cfg = load_config()
    info = probe(path)
    t0 = time.perf_counter()

    view = view_transform(path, info.width, info.height, cfg)
    manual = warp_manual(load_manual(resolve(cfg.scene.manual)), view)
    approaches = load_approaches(manual, info.width, info.height)
    hook = (lambda frame: signal_colours(frame, approaches)) if approaches else None
    view_key = None if view is None else view.round(5).tolist()
    tracks, frames = track_video(info, hook, progress, view_key)
    t_track = time.perf_counter() - t0

    tracks = add_kinematics(tracks, cfg.kinematics.smooth_window_sec, centered=True)
    scene = build_scene(tracks, info, cfg, manual, view)
    tracks["on_road"] = scene.on_road(tracks["gx"], tracks["gy"]) if len(tracks) else []
    ctx = Context(tracks=tracks, frames=frames, scene=scene, info=info, cfg=cfg)

    raw = []
    enabled = set(cfg.enabled_classes)
    for detector in all_detectors() if len(tracks) else []:
        try:
            raw += [e for e in detector(ctx) if e[2] in enabled]
        except Exception:  # one broken rule must not empty the whole video
            print(f"[{detector.__module__}] failed:\n{traceback.format_exc()}", file=sys.stderr)

    events = finalize(shift_boundaries(raw, cfg), info.duration, cfg.segments.max_gap_sec, cfg.segments.min_len_sec)
    print(f"[part A] {info.path}: {len(tracks)} track rows, {len(events)} events, "
          f"tracking {t_track:.1f}s, total {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return ctx, events


def detect_events(path: str) -> list[list]:
    return analyse(path)[1]
