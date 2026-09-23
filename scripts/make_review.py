"""Turn the system's own detections into a starting point for human review.

    python scripts/make_review.py samples/C3897.MP4
Writes dev/review/<video>_to_review.json. Open scripts/label_tool.html, load the video,
click "Import JSON" and pick that file: every detection appears as a row to check
(keep / delete / fix times). Guide: dev/REVIEW.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TRAFFIC_CACHE", str(Path(__file__).resolve().parent.parent / "cache"))

from src.pipeline import analyse  # noqa: E402


def main(video: str) -> None:
    ctx, events = analyse(video)
    name = Path(video).name
    out = Path("dev/review") / f"{Path(video).stem}_to_review.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({name: {"duration": round(ctx.info.duration, 3), "fps": ctx.info.fps,
                                      "events": events}}, indent=1))
    by_class: dict[str, int] = {}
    for _, _, c in events:
        by_class[c] = by_class.get(c, 0) + 1
    print(f"{out}: {len(events)} detections to review {by_class}")


if __name__ == "__main__":
    main(sys.argv[1])
