"""Render an annotated video + events JSON for one input video (debugging and the website).

    python scripts/render.py samples/video1.mp4 [--start 0 --end 120] [--scale 0.5]
Writes out/<name>_annotated.mp4 and out/<name>_events.json. Draws boxes with track ids,
short motion trails, the learned road mask, zones from scene.yaml, and a banner listing
the events active at each moment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TRAFFIC_CACHE", str(Path(__file__).resolve().parent.parent / "cache"))

from src.pipeline import analyse  # noqa: E402

COLOURS = {"person": (60, 200, 255), "car": (255, 170, 60), "truck": (200, 90, 255),
           "bus": (90, 90, 255), "motorcycle": (80, 255, 120), "bicycle": (80, 255, 200),
           "animal": (0, 255, 255)}


def draw_static(frame, scene, road_tint):
    frame[:] = cv2.addWeighted(frame, 1.0, road_tint, 0.18, 0)
    for poly in scene.crosswalks:
        cv2.polylines(frame, [poly.astype(np.int32)], True, (255, 255, 255), 2)
    for line in scene.solid_lines:
        cv2.polylines(frame, [line.astype(np.int32)], False, (0, 255, 255), 2)
    for a in scene.approaches:
        cv2.line(frame, *[tuple(map(int, p)) for p in a.stop_line[:2]], (0, 0, 255), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--scale", type=float, default=0.5, help="output size relative to input")
    args = ap.parse_args()

    ctx, events = analyse(args.video)
    name = Path(args.video).stem
    Path("out").mkdir(exist_ok=True)
    Path(f"out/{name}_events.json").write_text(json.dumps(events, indent=1))

    tracks = ctx.tracks
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    analysed = np.array(sorted(by_frame))
    road_tint = np.zeros((ctx.info.height, ctx.info.width, 3), np.uint8)
    road_tint[ctx.scene.road > 0] = (0, 180, 0)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps = ctx.info.fps
    size = (int(ctx.info.width * args.scale), int(ctx.info.height * args.scale))
    out = cv2.VideoWriter(f"out/{name}_annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    idx = int(round(args.start * fps))
    end = args.end if args.end is not None else ctx.info.duration
    while idx / fps < end:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        draw_static(frame, ctx.scene, road_tint)
        # nearest analysed frame at or before this one (Part A samples every few frames)
        k = np.searchsorted(analysed, idx, side="right") - 1
        if k >= 0:
            for _, r in by_frame[analysed[k]].iterrows():
                col = COLOURS.get(r["cls"], (200, 200, 200))
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 2)
                cv2.putText(frame, f"{r['cls']} {r['tid']}", (int(r.x1), int(r.y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                trail = tracks[(tracks["tid"] == r["tid"]) & (tracks["t"] <= t) & (tracks["t"] >= t - 2)]
                if len(trail) > 1:
                    cv2.polylines(frame, [trail[["gx", "gy"]].to_numpy(np.int32)], False, col, 2)
        active = [e for e in events if e[0] <= t <= e[1]]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36 + 30 * len(active)), (0, 0, 0), -1)
        cv2.putText(frame, f"t = {t:7.2f}s", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for i, (s, e, label) in enumerate(active):
            cv2.putText(frame, f"{label}  [{s:.1f} - {e:.1f}]", (10, 58 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out.write(cv2.resize(frame, size))
        idx += 1
    out.release()
    print(f"out/{name}_annotated.mp4, {len(events)} events")


if __name__ == "__main__":
    main()
