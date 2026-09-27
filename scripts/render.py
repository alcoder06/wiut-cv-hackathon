"""Render an annotated video + events JSON for one input video (debugging and the website).

    python scripts/render.py samples/video1.mp4 [--start 0 --end 120] [--width 960]
Writes out/<name>_annotated.mp4 (H.264, plays in a browser) and out/<name>_events.json.
Drawing is in src/annotate.py: boxes with track ids, 2 s trails, the learned road, zones
from scene.yaml, and a banner listing the events active at each moment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TRAFFIC_CACHE", str(Path(__file__).resolve().parent.parent / "cache"))

from src.annotate import render  # noqa: E402
from src.pipeline import analyse  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--width", type=int, default=960, help="output width in pixels")
    args = ap.parse_args()

    ctx, events = analyse(args.video)
    name = Path(args.video).stem
    Path("out").mkdir(exist_ok=True)
    Path(f"out/{name}_events.json").write_text(json.dumps(events, indent=1))
    render(args.video, ctx, events, f"out/{name}_annotated.mp4", args.width, start=args.start, end=args.end)
    print(f"out/{name}_annotated.mp4, {len(events)} events")


if __name__ == "__main__":
    main()
