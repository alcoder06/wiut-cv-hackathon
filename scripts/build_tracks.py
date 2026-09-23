"""Detect + track every sample video once, cache the tracks, and learn the scene flow field.

    python scripts/build_tracks.py samples
Tracks are cached under cache/ (keyed by video + detector/tracker config), so every rule
experiment after this reads a small parquet file instead of re-running YOLO.
The flow field learned from all samples is saved to configs/scene_flow.npz and shipped
with the solution (same camera as the test set).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TRAFFIC_CACHE", str(Path(__file__).resolve().parent.parent / "cache"))

from src.config import load_config, resolve, seed_everything  # noqa: E402
from src.events.signal import signal_colours  # noqa: E402
from src.kinematics import add_kinematics  # noqa: E402
from src.scene import FlowField, grid_for, load_approaches, load_manual  # noqa: E402
from src.tracking import track_video  # noqa: E402
from src.video import probe  # noqa: E402


def main(folder: str) -> None:
    cfg = load_config()
    seed_everything(cfg.seed)
    flow = None
    for p in sorted(p for p in Path(folder).iterdir() if p.suffix.lower() == ".mp4"):
        info = probe(str(p))
        approaches = load_approaches(load_manual(resolve(cfg.scene.manual)), info.width, info.height)
        hook = (lambda f, a=approaches: signal_colours(f, a)) if approaches else None
        t0 = time.perf_counter()
        tracks, _ = track_video(info, hook)
        took = time.perf_counter() - t0
        print(f"{p.name}: {tracks['tid'].nunique()} tracks, {len(tracks)} rows, "
              f"{took:.0f}s for {info.duration:.0f}s of video ({took / info.duration:.2f}x real time)")
        k = add_kinematics(tracks, cfg.kinematics.smooth_window_sec)
        f = FlowField.learn(k, info.width, info.height, grid_for(info.width, cfg.scene.cells_across),
                              cfg.kinematics.moving_speed)
        flow = f if flow is None else flow.merged(f)
    if flow is None or flow.count.sum() == 0:
        print("no moving vehicles seen: flow field NOT saved (the old one, if any, is kept)")
        return
    flow.save(resolve(cfg.scene.learned))
    print(f"flow field saved to {cfg.scene.learned} ({int(flow.count.sum())} observations)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples")
