"""Short, zoomed clips of every candidate in dev/labels/pool.json, for the labelling page.

    python scripts/label_clips.py --videos D:/Projects/extra --out out/label_site

From this 4K camera a pedestrian is ~10 px tall once the frame is shrunk to 720p, and a
question like "is someone jaywalking?" has no answer when the whole intersection is on
screen. So each clip is cropped around the road users the rule actually used, with
their boxes drawn. Rules return only [start, end, label], so the users are found by
leave-one-out: re-run the rule on the event's time window with groups of tracks
removed; a group whose removal makes the event disappear holds someone involved
(binary splitting keeps this to a few dozen rule runs per candidate). Events that no
single user carries (congestion) keep the full frame.

Also writes 60 s overview parts of each whole video, for spotting events the system
missed, and candidates.json, which the page reads. One decode pass serves every clip.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / "cache"))

from src.config import Cfg, load_config  # noqa: E402
from src.events import collision, congestion, lines, pedestrian, signal, stationary, turns, wrong_way  # noqa: E402
from src.pipeline import analyse  # noqa: E402
from src.video import iter_frames  # noqa: E402
from tune import set_key  # noqa: E402

MODULE = {"accident": collision, "near_miss": collision, "congestion": congestion,
          "solid_line_crossing": lines, "jaywalking": pedestrian, "failure_to_yield": pedestrian,
          "red_light": signal, "stop_line": signal, "stopped_vehicle": stationary,
          "illegal_turn": turns, "illegal_u_turn": turns, "wrong_way": wrong_way}
WHOLE_SCENE = {"congestion"}
SIGNAL = {"red_light", "stop_line"}     # the lamp must stay in view: box the car, keep the full frame
WHOLE_TRACK = {"stopped_vehicle"}
PER_USER = {"jaywalking", "wrong_way", "solid_line_crossing"}   # one road user fires it alone
OUT_W, OUT_H = 960, 540
PAD_SEC = 2.0          # clip starts this long before the candidate and ends this long after
STRIDE = 2             # 30 fps -> 15 fps clips: half the size, still smooth enough to judge
OVERVIEW_SEC = 60.0
HIGHLIGHT = (0, 200, 255)   # BGR amber, the page's accent
ZONE_COLOURS = {"crosswalk": (255, 255, 255), "refuge": (132, 220, 61), "solid": (0, 255, 255), "stop": (60, 60, 255)}


def involved(ctx, label: str, rep: tuple[float, float], cfg) -> list[int]:
    """Road users the rule needs for this event: a smallest set of tracks that still
    produces it (both parties of a near miss; the car and the pedestrian of a failure to
    yield), plus anyone else who produces it on their own (several people jaywalking at
    once, which one smallest set would miss)."""
    s, e = rep
    tr = ctx.tracks
    # everyone seen near the event; stopped_vehicle judges a track as a whole ("never seen
    # moving" = parked), so it gets whole tracks, the rest a few seconds around the event
    near = tr.loc[(tr["t"] >= s - 3) & (tr["t"] <= e + 3), "tid"].unique()
    win = tr[tr["tid"].isin(near)]
    if label not in WHOLE_TRACK:        # a parked car's track spans the whole video: slow
        win = win[(win["t"] >= s - 5) & (win["t"] <= e + 5)]
    module = MODULE[label]

    def fires(tids) -> bool:
        c = dataclasses.replace(ctx, tracks=win[win["tid"].isin(tids)], cfg=cfg)
        return any(lab == label and a < e and s < b for a, b, lab in module.detect(c))

    keep = sorted(int(t) for t in near)
    if not keep or not fires(keep):
        return []
    for size in (64, 16, 4, 1):             # drop chunks that the event doesn't need
        i = 0
        while i < len(keep):
            rest = keep[:i] + keep[i + size:]
            if rest and fires(rest):
                keep = rest
            else:
                i += size
    alone = ([t for t in near if int(t) not in keep and fires([t])]
             if len(keep) == 1 and label in PER_USER else [])
    return (keep + [int(t) for t in alone])[:6]


def crop_box(ctx, tids: list[int], s: float, e: float) -> list[int]:
    """16:9 crop around the involved boxes (full-res px), at least 30% of the frame wide."""
    W, H = ctx.info.width, ctx.info.height
    rows = ctx.tracks[ctx.tracks["tid"].isin(tids) & (ctx.tracks["t"] >= s - 0.5) & (ctx.tracks["t"] <= e + 0.5)]
    if rows.empty:
        return [0, 0, W, H]
    x1, y1, x2, y2 = rows["x1"].min(), rows["y1"].min(), rows["x2"].max(), rows["y2"].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max((x2 - x1) * 1.6, (y2 - y1) * 1.6 * 16 / 9, 0.3 * W)
    w = min(w, W, H * 16 / 9)
    h = w * 9 / 16
    x0 = int(np.clip(cx - w / 2, 0, W - w))
    y0 = int(np.clip(cy - h / 2, 0, H - h))
    return [x0, y0, int(x0 + w), int(y0 + h)]


class Clip:
    """One output video: a crop of the frame between t0 and t1, involved boxes drawn."""

    def __init__(self, path: Path, fps: float, t0: float, t1: float, crop: list[int], boxes, zones):
        self.path, self.t0, self.t1, self.crop = path, t0, t1, crop
        self.boxes, self.zones = boxes, zones      # boxes: {frame_idx: [(tid, cls, x1, y1, x2, y2)]}
        self.box_frames = np.array(sorted(boxes)) if boxes else np.array([], int)
        self.fps, self.out, self.stream = fps, None, None

    def write(self, idx: int, frame: np.ndarray) -> None:
        if self.out is None:
            self.out = av.open(str(self.path), mode="w", options={"movflags": "+faststart"})
            self.stream = self.out.add_stream("libx264", rate=Fraction(self.fps / STRIDE).limit_denominator(1001))
            self.stream.width, self.stream.height, self.stream.pix_fmt = OUT_W, OUT_H, "yuv420p"
            self.stream.options = {"crf": "30", "preset": "veryfast"}
        x0, y0, x1, y1 = self.crop
        k = OUT_W / (x1 - x0)
        img = frame[y0:y1, x0:x1]
        img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        for kind, pts in self.zones:
            p = ((pts - [x0, y0]) * k).astype(np.int32)
            colour = ZONE_COLOURS[kind]
            cv2.polylines(img, [p], kind in ("crosswalk", "refuge"), colour, 1, cv2.LINE_AA)
        j = np.searchsorted(self.box_frames, idx, side="right") - 1
        if j >= 0 and idx - self.box_frames[j] <= 8:
            for tid, cls, bx1, by1, bx2, by2 in self.boxes[self.box_frames[j]]:
                p1 = (int((bx1 - x0) * k) - 3, int((by1 - y0) * k) - 3)
                p2 = (int((bx2 - x0) * k) + 3, int((by2 - y0) * k) + 3)
                cv2.rectangle(img, p1, p2, HIGHLIGHT, 2, cv2.LINE_AA)
                cv2.putText(img, f"{cls} {tid}", (p1[0], max(12, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, HIGHLIGHT, 1, cv2.LINE_AA)
        vf = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")
        for packet in self.stream.encode(vf):
            self.out.mux(packet)

    def close(self) -> None:
        if self.out is not None:
            for packet in self.stream.encode():
                self.out.mux(packet)
            self.out.close()


def zones_of(scene) -> list[tuple[str, np.ndarray]]:
    z = [("crosswalk", p) for p in scene.crosswalks] + [("solid", p) for p in scene.solid_lines]
    z += [("refuge", p) for p in getattr(scene, "refuges", [])]
    return z + [("stop", a.stop_line[:2]) for a in scene.approaches]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="samples")
    ap.add_argument("--pool", default="dev/labels/pool.json")
    ap.add_argument("--out", default="out/label_site")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing candidates.json: new ids continue after the last one per video")
    ap.add_argument("--round", type=int, default=1, help="labelling round, shown on the page")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "overview").mkdir(parents=True, exist_ok=True)
    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    base = copy.deepcopy(dict(load_config()))
    site = {"videos": {}, "candidates": []}
    if args.append and (out / "candidates.json").exists():
        site = json.loads((out / "candidates.json").read_text(encoding="utf-8"))

    for name, entry in pool.items():
        path = Path(args.videos) / name
        ctx, _ = analyse(str(path))
        info, stem = ctx.info, Path(name).stem
        zones = zones_of(ctx.scene)
        clips = []
        first = 1 + max((int(c["id"].rsplit("-", 1)[1]) for c in site["candidates"] if c["video"] == name), default=0)
        for n, c in enumerate(entry["candidates"], start=first - 1):
            cfg_raw = copy.deepcopy(base)
            for k, v in c["rep_overrides"].items():
                set_key(cfg_raw, k, v)
            # congestion is about a whole direction of traffic: a zoom on 3 cars would mislead
            tids = [] if c["label"] in WHOLE_SCENE else involved(ctx, c["label"], tuple(c["rep"]), Cfg(cfg_raw))
            crop = [0, 0, ctx.info.width, ctx.info.height] if c["label"] in SIGNAL else crop_box(ctx, tids, c["start"], c["end"])
            t0, t1 = max(0.0, c["start"] - PAD_SEC), min(info.duration, c["end"] + PAD_SEC)
            rows = ctx.tracks[ctx.tracks["tid"].isin(tids) & (ctx.tracks["t"] >= t0) & (ctx.tracks["t"] <= t1)]
            boxes = {f: list(g[["tid", "cls", "x1", "y1", "x2", "y2"]].itertuples(index=False, name=None))
                     for f, g in rows.groupby("frame")}
            cid = f"{stem}-{n + 1:02d}"
            clips.append(Clip(out / "clips" / f"{cid}.mp4", info.fps, t0, t1, crop, boxes, zones))
            site["candidates"].append({
                "id": cid, "video": name, "label": c["label"], "start": c["start"], "end": c["end"],
                "current": c["current"], "clip": f"clips/{cid}.mp4", "clip_start": round(t0, 3),
                "clip_end": round(t1, 3), "zoomed": crop != [0, 0, info.width, info.height],
                "involved": sorted({str(cls) for cls in rows["cls"]}) if len(rows) else [], "round": args.round,
                **({"retime_of": c["retime_of"]} if c.get("retime_of") else {})})
            print(f"{cid} {c['label']:<20} {c['start']:7.1f}-{c['end']:7.1f}  involved {tids}")
        n_parts = 0 if name in site["videos"] else int(np.ceil(info.duration / OVERVIEW_SEC))
        parts = [Clip(out / "overview" / f"{stem}_p{i + 1}.mp4", info.fps, i * OVERVIEW_SEC,
                      min(info.duration, (i + 1) * OVERVIEW_SEC), [0, 0, info.width, info.height], {}, zones)
                 for i in range(n_parts)]
        if parts:
            site["videos"][name] = {"duration": round(info.duration, 3), "fps": info.fps,
                                    "parts": [{"src": f"overview/{p.path.name}", "start": p.t0, "end": p.t1} for p in parts]}

        pending = sorted(clips + parts, key=lambda c: c.t0)
        active: list[Clip] = []
        for idx, t, frame, _ in iter_frames(str(path), STRIDE, prefetch=8):
            while pending and pending[0].t0 <= t:
                active.append(pending.pop(0))
            for c in [c for c in active if t > c.t1]:
                c.close()
                active.remove(c)
            for c in active:
                c.write(idx, frame)
            if not active and not pending:
                break
        for c in active:
            c.close()
        print(f"{name}: {len(clips)} clips, {n_parts} overview parts")

    (out / "candidates.json").write_text(json.dumps(site, indent=1), encoding="utf-8")
    print(f"wrote {out / 'candidates.json'}")


if __name__ == "__main__":
    main()
