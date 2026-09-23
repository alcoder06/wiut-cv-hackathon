"""Merge per-person label exports into dev/labels.json (the file evaluate.py --gt reads).

    python scripts/merge_labels.py dev/labels/*.json
Each export from label_tool.html is {video: {duration, fps, events}}. Files for the same
video are combined (e.g. two people each did half of a long video); same-class overlaps
across files are reported so the team can decide, not silently merged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(paths: list[str]) -> None:
    merged: dict = {}
    for p in paths:
        for video, entry in json.loads(Path(p).read_text(encoding="utf-8")).items():
            m = merged.setdefault(video, {"duration": entry["duration"], "fps": entry["fps"], "events": []})
            m["events"] += entry["events"]
    problems = 0
    for video, m in merged.items():
        m["events"].sort()
        by_cls: dict = {}
        for s, e, c in m["events"]:
            for s2, e2 in by_cls.get(c, []):
                if s < e2 and s2 < e:
                    print(f"OVERLAP {video} {c}: [{s2}, {e2}] and [{s}, {e}]")
                    problems += 1
            by_cls.setdefault(c, []).append((s, e))
        counts = {}
        for _, _, c in m["events"]:
            counts[c] = counts.get(c, 0) + 1
        print(f"{video}: {len(m['events'])} events {counts}")
    out = Path("dev/labels.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(merged, indent=1))
    print(f"wrote {out}" + (f" with {problems} overlap(s) to fix" if problems else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
