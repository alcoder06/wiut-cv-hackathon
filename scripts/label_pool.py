"""Everything the tuner could output, as one list of candidates for people to check.

    python scripts/label_pool.py --videos D:/Projects/extra --out dev/labels/pool.json

scripts/tune.py scores every threshold combination in its GRID against the answer key.
If people only checked the detections of the current settings, a looser setting that
finds a real but unchecked event would be scored as a false alarm, and tuning would be
biased towards whatever we ship today. So this runs each rule over its whole grid (on
cached tracks, seconds per video), pools the segments into candidates: one per current
detection, plus the segments only a looser or stricter setting finds. Each candidate
keeps one setting that produces it, so scripts/label_clips.py can find who is involved.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / "cache"))

from src.config import Cfg, load_config  # noqa: E402
from src.pipeline import analyse  # noqa: E402
from src.segments import finalize  # noqa: E402
from tune import GRID, set_key  # noqa: E402


def grid_segments(ctx, base: dict, label: str, module, grid: dict) -> list[tuple]:
    """(start, end, overrides) for every segment any grid setting produces."""
    out = []
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        trial = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            set_key(trial, k, v)
        cfg = Cfg(trial)
        c = dataclasses.replace(ctx, cfg=cfg)
        raw = [e for e in module.detect(c) if e[2] == label]
        out += [(s, e, dict(zip(keys, combo)))
                for s, e, _ in finalize(raw, ctx.info.duration, cfg.segments.max_gap_sec, cfg.segments.min_len_sec)]
    return out


def iou(a: tuple, b: tuple) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0.0, inter) / (max(a[1], b[1]) - min(a[0], b[0]))


def candidates(segments: list[tuple[float, float, dict | None]]) -> list[dict]:
    """One candidate per detection of the current settings (`overrides` None); grid
    segments overlapping one are variants of it. The rest are grouped with segments they
    overlap by IoU >= 0.2 (a looser setting often fires a few times on one event).
    Plain overlap chaining is not enough: one long loose segment can bridge two separate
    stops into a single minute-long candidate that no one can judge."""
    current = sorted((s, e) for s, e, ov in segments if ov is None)
    out = [{"start": s, "end": e, "n": 1, "rep": [s, e], "rep_overrides": {}, "current": True}
           for s, e in current]
    for s, e, ov in sorted((x for x in segments if x[2] is not None), key=lambda x: (x[0], x[1])):
        hits = [c for c in out if c["current"] and s < c["end"] and c["start"] < e]
        if hits:
            for c in hits:
                c["n"] += 1
            continue
        group = next((c for c in out if not c["current"] and iou((s, e), c["rep"]) >= 0.2), None)
        if group:
            group["start"], group["end"] = min(group["start"], s), max(group["end"], e)
            group["n"] += 1
        else:
            out.append({"start": s, "end": e, "n": 1, "rep": [s, e], "rep_overrides": ov, "current": False})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="samples")
    ap.add_argument("--out", default="dev/labels/pool.json")
    ap.add_argument("--classes", nargs="*", help="only these classes (e.g. after redrawing crossings)")
    ap.add_argument("--names", nargs="*", help="only these video files in --videos")
    ap.add_argument("--grid-classes", nargs="*",
                    help="run the whole grid only for these; the rest contribute today's detections only "
                         "(the near_miss grid alone takes minutes per video on a busy machine)")
    args = ap.parse_args()
    wanted = set(args.classes or GRID) | ({"solid_line_crossing"} if not args.classes else set())

    base = copy.deepcopy(dict(load_config()))
    pool = {}
    for path in sorted(Path(args.videos).iterdir()):
        if path.suffix.lower() != ".mp4" or (args.names and path.name not in args.names):
            continue
        ctx, current = analyse(str(path))
        segs: dict[str, list] = {}
        for s, e, label in current:
            if label in wanted:
                segs.setdefault(label, []).append((s, e, None))
        for label, (module, grid) in GRID.items():
            if label not in wanted or (args.grid_classes is not None and label not in args.grid_classes):
                continue
            segs.setdefault(label, []).extend(grid_segments(ctx, base, label, module, grid))
        cands = []
        for label, items in sorted(segs.items()):
            for c in candidates(items):
                cands.append({"label": label, "start": round(c["start"], 2), "end": round(c["end"], 2),
                              "current": c["current"], "grid_hits": c["n"],
                              "rep": [round(x, 3) for x in c["rep"]], "rep_overrides": c["rep_overrides"]})
        cands.sort(key=lambda c: (c["start"], c["label"]))
        pool[path.name] = {"duration": round(ctx.info.duration, 3), "fps": ctx.info.fps,
                           "width": ctx.info.width, "height": ctx.info.height, "candidates": cands}
        by = {}
        for c in cands:
            by[c["label"]] = by.get(c["label"], 0) + 1
        print(f"{path.name}: {len(current)} current detections, {len(cands)} candidates {by}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
