"""Tune Part B in seconds: replay cached tracks through the causal risk scorer.

    python scripts/replay_risk.py samples/C3897.MP4 [--gt dev/labels.json]
Needs the track cache (scripts/build_tracks.py). The detector+tracker half is skipped: the
cached raw boxes (Part A's tracker, ~7.5 fps) stand in for Part B's own (5 fps), and the
scoring code is exactly the one the harness runs (CausalRisk.observe). Final numbers
should still be confirmed with run_submission.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TRAFFIC_CACHE", str(Path(__file__).resolve().parent.parent / "cache"))

from evaluate import THETA, alarm_starts, evaluate_part_b  # noqa: E402
from src.risk import CausalRisk  # noqa: E402
from src.tracking import track_video  # noqa: E402
from src.video import probe  # noqa: E402


def replay(video: str) -> list[list[float]]:
    info = probe(video)
    tracks, _ = track_video(info)
    est = CausalRisk()
    est.reset({"video_id": Path(video).name, "fps": info.fps, "width": info.width,
               "height": info.height, "n_frames": info.n_frames})
    curve = []
    for (_, t), g in tracks.groupby(["frame", "t"], sort=True):
        s = est.observe(g[["x1", "y1", "x2", "y2"]].to_numpy(), g["cls"].tolist(), g["tid"].to_numpy(), t)
        curve.append([round(float(t), 4), round(s, 4)])
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gt", help="labels in ground-truth format; prints the official Score B")
    args = ap.parse_args()
    curve = replay(args.video)
    scores = np.array([s for _, s in curve])
    starts = alarm_starts(curve)
    dur = curve[-1][0] if curve else 0
    print(f"samples {len(curve)}, mean risk {scores.mean():.3f}, >= {THETA}: {(scores >= THETA).mean() * 100:.1f}% "
          f"of samples, {len(starts)} alarms ({len(starts) / max(dur, 1) * 60:.1f} per minute)")
    print("alarm starts (s):", [round(a, 1) for a in starts])
    if args.gt:
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        name = Path(args.video).name
        if name in gt:
            rep = evaluate_part_b({name: gt[name]}, {name: {"risk": curve}})
            if rep is None:
                print("no accidents in the labels for this video: Score B is not defined here "
                      "(every alarm above is a false alarm unless it starts near a labelled near_miss)")
            else:
                print(f"Score B {rep['score_b']:.3f}  AP {rep['ap']:.3f}  alarm P/R/F1 "
                      f"{rep['alarm_precision']:.2f}/{rep['alarm_recall']:.2f}/{rep['f1_alarm']:.2f}  "
                      f"mTTA {rep['mtta_sec']:.2f}s")


if __name__ == "__main__":
    main()
