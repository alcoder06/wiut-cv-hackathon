"""Fit rule thresholds and boundary shifts to the team's answer key ("training" for rules).

    python scripts/tune.py                      # report only
    python scripts/tune.py --apply              # also write configs/tuned.yaml
    python scripts/tune.py --gt dev/labels.json --videos samples

Score A is a per-class F1 averaged over classes, so each class is tuned on its own:
a small grid over that class's thresholds, then a start/end shift. Only the class's own
rule runs, on cached tracks, so a full pass takes seconds. With one short video it is easy
to over-fit, so a change is kept only if it beats the current setting by MIN_GAIN.
configs/tuned.yaml is loaded on top of pipeline.yaml; delete it to undo.
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

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / "cache"))

from evaluate import evaluate_part_a  # noqa: E402
from src.config import Cfg, load_config  # noqa: E402
from src.events import collision, congestion, pedestrian, stationary, wrong_way  # noqa: E402
from src.pipeline import analyse  # noqa: E402
from src.segments import finalize  # noqa: E402

MIN_GAIN = 0.02
SHIFTS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

# class -> (rule module, {dotted config key: candidate values})
GRID = {
    "stopped_vehicle": (stationary, {"rules.stopped_vehicle.queue_share": [0.3, 0.5, 0.7],
                                     "rules.stopped_vehicle.neighbour_radius": [3.0, 4.0, 6.0],
                                     "kinematics.stopped_speed": [0.08, 0.12, 0.2]}),
    "congestion": (congestion, {"rules.congestion.min_vehicles": [3, 4, 6],
                                "rules.congestion.crawl_speed": [0.15, 0.25, 0.4],
                                "rules.congestion.min_len_sec": [10.0, 20.0, 30.0]}),
    "jaywalking": (pedestrian, {"rules.jaywalking.min_len_sec": [0.5, 1.0, 2.0],
                                "rules.jaywalking.road_erode_px": [1, 12, 30]}),
    "failure_to_yield": (pedestrian, {"rules.failure_to_yield.near_sizes": [1.5, 2.5, 4.0],
                                      "rules.failure_to_yield.ped_margin_px": [0, 20, 50]}),
    "near_miss": (collision, {"rules.near_miss.ttc_sec": [1.0, 1.5, 2.0],
                              "rules.near_miss.decel_drop": [0.4, 0.5, 0.6]}),
    "accident": (collision, {"rules.collision.impact_speed": [0.8, 1.0, 1.4],
                             "rules.collision.decel_drop": [0.5, 0.6, 0.7]}),
    "wrong_way": (wrong_way, {"rules.wrong_way.against_cos": [-0.7, -0.5, -0.3],
                              "rules.wrong_way.min_len_sec": [1.0, 1.5, 3.0]}),
}


def get_key(cfg: dict, key: str):
    for part in key.split("."):
        cfg = cfg[part]
    return cfg


def set_key(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        cfg = cfg.setdefault(part, {})
    cfg[parts[-1]] = value


def class_score(gt: dict, preds: dict, label: str) -> float:
    """Mean F1 over IoU 0.3/0.5/0.7 for one class, pooled over videos (the official code)."""
    g = {v: {**e, "events": [x for x in e["events"] if x[2] == label]} for v, e in gt.items()}
    p = {v: {"events": [x for x in ev if x[2] == label]} for v, ev in preds.items()}
    rep = evaluate_part_a(g, p)
    return rep["per_class"].get(label, {}).get("f1_mean", 0.0)


def run_class(ctxs: dict, raw_cfg: dict, label: str, module, shift=(0.0, 0.0)) -> dict:
    cfg = Cfg(raw_cfg)
    out = {}
    for vid, ctx in ctxs.items():
        c = dataclasses.replace(ctx, cfg=cfg)
        raw = [[s + shift[0], e + shift[1], lab] for s, e, lab in module.detect(c) if lab == label]
        out[vid] = finalize(raw, ctx.info.duration, cfg.segments.max_gap_sec, cfg.segments.min_len_sec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="dev/labels.json")
    ap.add_argument("--videos", default="samples")
    ap.add_argument("--apply", action="store_true", help="write configs/tuned.yaml")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    ctxs = {}
    for vid in gt:
        path = Path(args.videos) / vid
        if path.exists():
            ctxs[vid] = analyse(str(path))[0]
        else:
            print(f"skipping {vid}: not found in {args.videos}")
    gt = {v: e for v, e in gt.items() if v in ctxs}
    if not gt:
        sys.exit("no labelled video found")

    base = copy.deepcopy(dict(load_config()))
    base_shift = base["segments"].get("boundary_shift") or {}
    labelled = {x[2] for e in gt.values() for x in e["events"]}
    tuned: dict = {}
    print(f"\n{'class':<18}{'labels':>7}{'before':>9}{'after':>8}  changes")
    for label, (module, grid) in GRID.items():
        n_gt = sum(1 for e in gt.values() for x in e["events"] if x[2] == label)
        shift0 = tuple(base_shift.get(label, (0.0, 0.0)))
        before = class_score(gt, run_class(ctxs, base, label, module, shift0), label)
        best, best_cfg, changes = before, base, {}
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            trial = copy.deepcopy(base)
            for k, v in zip(keys, combo):
                set_key(trial, k, v)
            score = class_score(gt, run_class(ctxs, trial, label, module, shift0), label)
            if score > best + MIN_GAIN:
                best, best_cfg = score, trial
                changes = {k: v for k, v in zip(keys, combo) if get_key(base, k) != v}
        best_shift = shift0
        for ds, de in itertools.product(SHIFTS, SHIFTS):
            score = class_score(gt, run_class(ctxs, best_cfg, label, module, (ds, de)), label)
            if score > best + MIN_GAIN:
                best, best_shift = score, (ds, de)
        for k, v in changes.items():
            set_key(tuned, k, v)
        if best_shift != shift0:
            set_key(tuned, f"segments.boundary_shift.{label}", list(best_shift))
        note = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in changes.items())
        if best_shift != shift0:
            note += (", " if note else "") + f"shift {best_shift[0]:+.1f}/{best_shift[1]:+.1f} s"
        if n_gt == 0 and label not in labelled:
            note = "no labels of this class: every detection is a false positive here"
        print(f"{label:<18}{n_gt:>7}{before:>9.3f}{best:>8.3f}  {note or 'keep'}")

    missing = sorted(labelled - set(GRID))
    if missing:
        print(f"\nlabelled but not tunable here (no rule, or it needs zones): {', '.join(missing)}")
    if args.apply and tuned:
        out = ROOT / "configs" / "tuned.yaml"
        out.write_text("# written by scripts/tune.py against " + args.gt + "; delete to undo\n"
                       + yaml.safe_dump(tuned, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {out.relative_to(ROOT)}")
    elif tuned:
        print("\nnot applied; re-run with --apply to write configs/tuned.yaml:\n" + yaml.safe_dump(tuned, sort_keys=True))
    else:
        print("\nnothing beat the current settings by more than", MIN_GAIN)


if __name__ == "__main__":
    main()
