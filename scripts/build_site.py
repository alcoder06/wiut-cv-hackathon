"""Build the website's data for every sample video: EDA, results, annotated videos.

    python scripts/build_site.py --videos samples --pred predictions_samples.json [--no-video | --keep-video]

Events and risk curves come from the predictions file (the harness output we submit), so
the site shows exactly what run_submission.py produced. Tracks come from the Part A cache
(scripts/build_tracks.py), so re-running this takes minutes, not a full detector pass.
Writes web/static/data/<video>.json + images, web/static/data/index.json and
web/static/media/<video>.mp4. Both folders stay out of git (footage frames and videos).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / "cache"))

from evaluate import alarm_starts  # noqa: E402
from src.annotate import render  # noqa: E402
from src.config import load_config, resolve  # noqa: E402
from src.pipeline import analyse  # noqa: E402
from src.scene import FlowField  # noqa: E402
from src.stats import GROUP_ORDER, GROUPS, object_counts  # noqa: E402
from src.video import downscale  # noqa: E402

DATA = ROOT / "web" / "static" / "data"
MEDIA = ROOT / "web" / "static" / "media"
IMG_W = 960
# Reference-palette hues (BGR) for the EDA images, matching the chart colours on the site.
BLUE, ORANGE = (214, 120, 42), (52, 104, 235)


def frame_at(path: str, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def small(frame: np.ndarray) -> tuple[np.ndarray, float]:
    img, s = downscale(frame, IMG_W)
    return img, s


def lighting(path: str, duration: float, every: float = 5.0) -> dict:
    """Mean luminance (0-255) and share of near-saturated pixels, every `every` seconds."""
    cap = cv2.VideoCapture(path)
    t_s, lum, glare = [], [], []
    for t in np.arange(0, duration, every):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(small(f)[0], cv2.COLOR_BGR2GRAY)
        t_s.append(round(float(t), 1))
        lum.append(round(float(g.mean()), 1))
        glare.append(round(float((g > 245).mean() * 100), 2))
    cap.release()
    return {"t": t_s, "luminance": lum, "saturated_pct": glare}


def heatmap(bg: np.ndarray, xs: np.ndarray, ys: np.ndarray, colour: tuple) -> np.ndarray:
    """Where objects moved, as a single-hue glow over the darkened frame. Log density scaled
    to its 99th percentile, so a few very busy cells don't wash out everything else."""
    h, w = bg.shape[:2]
    hist = np.zeros((h // 4, w // 4), np.float32)
    ix = np.clip((xs / 4).astype(int), 0, w // 4 - 1)
    iy = np.clip((ys / 4).astype(int), 0, h // 4 - 1)
    np.add.at(hist, (iy, ix), 1)
    hist = np.log1p(cv2.GaussianBlur(hist, (0, 0), 3.0) * 50)
    top = np.percentile(hist[hist > 0], 99) if (hist > 0).any() else 1.0
    hist = cv2.resize(np.clip(hist / top, 0, 1), (w, h))[..., None]
    out = (bg * 0.35).astype(np.float32)
    glow = np.array(colour, np.float32) * (1 - 0.5 * hist) + 255 * 0.5 * hist   # lighter where denser
    return np.clip(out * (1 - 0.85 * hist) + glow * 0.85 * hist, 0, 255).astype(np.uint8)


def trajectories(bg: np.ndarray, tracks, scale: float) -> np.ndarray:
    out = (bg * 0.35).astype(np.uint8)
    layer = np.zeros_like(out)
    for tid, g in tracks.groupby("tid"):
        pts = (g[["gx", "gy"]].to_numpy() * scale).astype(np.int32)
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else []
        if len(pts) < 3 or np.any(steps > 40):
            continue
        colour = ORANGE if g["cls"].iat[0] == "person" else BLUE
        cv2.polylines(layer, [pts], False, colour, 1, cv2.LINE_AA)
    return cv2.addWeighted(out, 1.0, layer, 0.9, 0)


def lane_arrows(bg: np.ndarray, flow: FlowField, scale: float, min_obs: int, min_coh: float) -> np.ndarray:
    """The learned flow field: one arrow per road cell, coloured by which way along the
    main traffic axis it points (the two carriageways)."""
    out = (bg * 0.4).astype(np.uint8)
    unit, coh = flow.direction()
    axis = flow.main_axis()
    g = flow.grid * scale
    for cy, cx in zip(*np.nonzero((flow.count >= min_obs) & (coh >= min_coh))):
        if (cx + cy) % 2:          # every other cell keeps the arrows readable
            continue
        c = np.array([(cx + 0.5) * g, (cy + 0.5) * g])
        d = unit[cy, cx] * g * 0.9
        colour = BLUE if unit[cy, cx] @ axis >= 0 else ORANGE
        cv2.arrowedLine(out, tuple((c - d / 2).astype(int)), tuple((c + d / 2).astype(int)),
                        colour, 2, cv2.LINE_AA, tipLength=0.4)
    return out


def speed_hist(tracks, cfg) -> dict:
    """Distribution of speed (box sizes per second) per group, moving samples only."""
    bins = np.round(np.arange(0, 4.01, 0.1), 2)
    out = {"bins": bins[:-1].tolist()}
    for grp in GROUP_ORDER:
        s = tracks.loc[tracks["cls"].map(GROUPS) == grp, "speed"].to_numpy()
        s = s[s > cfg.kinematics.stopped_speed]
        h, _ = np.histogram(np.clip(s, 0, 3.99), bins)
        out[grp] = np.round(h / max(1, h.sum()) * 100, 2).tolist()
    return out


def stopped_share(tracks, cfg, duration: float, bin_sec: float = 5.0) -> dict:
    """Share of on-road vehicles that are stopped, per bin: the signal cycle shows up here."""
    v = tracks[tracks["cls"].map(GROUPS).isin(["car", "heavy"]) & tracks["on_road"]]
    n = max(1, int(np.ceil(duration / bin_sec)))
    b = (v["t"] // bin_sec).astype(int).clip(0, n - 1)
    stopped = (v["speed"] < cfg.kinematics.stopped_speed).groupby(b).mean().reindex(range(n))
    return {"t": [i * bin_sec for i in range(n)],
            "share": [None if np.isnan(x) else round(float(x) * 100, 1) for x in stopped]}


def thumbs(path: str, name: str, events: list, per_class: int = 3) -> list[dict]:
    """A still at the middle of the first few events of each class (the 'examples' strip)."""
    seen: Counter = Counter()
    out = []
    for i, (s, e, label) in enumerate(events):
        if seen[label] >= per_class:
            continue
        f = frame_at(path, (s + e) / 2)
        if f is None:
            continue
        seen[label] += 1
        file = f"{name}_ev{i}.jpg"
        cv2.imwrite(str(DATA / file), downscale(f, 640)[0], [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append({"i": i, "label": label, "start": s, "end": e, "img": f"data/{file}"})
    return out


def step_points(curve: list) -> list:
    """The risk curve holds its value between Part B samples (5 fps of a 30 fps video), so
    keeping only the points where it changes (plus the last) is lossless and ~6x smaller.
    Thinning by stride instead would drop the short spikes that are the alarms."""
    return [p for i, p in enumerate(curve) if i == 0 or i == len(curve) - 1 or p[1] != curve[i - 1][1]]


def build(path: Path, pred: dict | None, cfg, flow: FlowField | None, with_video: bool, keep_video: bool = False) -> dict:
    name = path.stem
    ctx, own_events = analyse(str(path))
    info, tracks = ctx.info, ctx.tracks
    events = pred["events"] if pred else own_events
    risk = pred.get("risk") if pred else None

    bg, scale = small(frame_at(str(path), min(10.0, info.duration / 2)))
    cv2.imwrite(str(DATA / f"{name}_frame.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 85])
    moving = tracks[tracks["speed"] > cfg.kinematics.stopped_speed]
    veh, ppl = moving[moving["cls"] != "person"], moving[moving["cls"] == "person"]
    cv2.imwrite(str(DATA / f"{name}_heat_vehicles.jpg"), heatmap(bg, veh["gx"].to_numpy() * scale, veh["gy"].to_numpy() * scale, BLUE))
    cv2.imwrite(str(DATA / f"{name}_heat_people.jpg"), heatmap(bg, ppl["gx"].to_numpy() * scale, ppl["gy"].to_numpy() * scale, ORANGE))
    cv2.imwrite(str(DATA / f"{name}_tracks.jpg"), trajectories(bg, tracks, scale))
    if flow is not None and (f := flow.rescaled(info.width, info.height)) is not None:
        cv2.imwrite(str(DATA / f"{name}_lanes.jpg"),
                    lane_arrows(bg, f, scale, cfg.scene.min_cell_obs, cfg.scene.min_coherence))

    if with_video and not (keep_video and (MEDIA / f"{name}.mp4").exists()):
        MEDIA.mkdir(parents=True, exist_ok=True)
        render(str(path), ctx, events, str(MEDIA / f"{name}.mp4"), width=960, risk=risk)

    per_class = Counter(e[2] for e in events)
    out = {
        "name": path.name,
        "video": {"width": info.width, "height": info.height, "fps": round(info.fps, 3),
                  "frames": info.n_frames, "duration": round(info.duration, 2),
                  "size_mb": round(path.stat().st_size / 2**20, 1),
                  "codec": int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")},
        "media": f"media/{name}.mp4",
        "images": {k: f"data/{name}_{k}.jpg" for k in ("frame", "heat_vehicles", "heat_people", "tracks", "lanes")},
        "events": events,
        "event_counts": dict(per_class),
        "examples": thumbs(str(path), name, events),
        "risk": step_points(risk) if risk else [],
        "alarms": alarm_starts(risk) if risk else [],
        "objects": object_counts(tracks, info.duration),
        "stopped_share": stopped_share(tracks, cfg, info.duration),
        "speed": speed_hist(tracks, cfg),
        "lighting": lighting(str(path), info.duration),
        "tracks": {g: int(tracks.loc[tracks["cls"].map(GROUPS) == g, "tid"].nunique()) for g in GROUP_ORDER},
    }
    (DATA / f"{name}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"{path.name}: {len(events)} events, {sum(out['tracks'].values())} tracks")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="samples")
    ap.add_argument("--pred", default="predictions_samples.json")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated videos (slow at 4K)")
    ap.add_argument("--keep-video", action="store_true", help="render only videos that have no annotated mp4 yet")
    args = ap.parse_args()

    cfg = load_config()
    DATA.mkdir(parents=True, exist_ok=True)
    preds = json.loads(Path(args.pred).read_text(encoding="utf-8"))["videos"] if Path(args.pred).exists() else {}
    flow_path = resolve(cfg.scene.learned)
    flow = FlowField.load(flow_path) if flow_path.exists() else None
    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() == ".mp4")
    if preds:   # the site shows exactly the videos in the submitted predictions, no stray clips
        missing = set(preds) - {p.name for p in videos}
        if missing:
            raise SystemExit(f"videos in {args.pred} but not in {args.videos}: {sorted(missing)}")
        videos = [p for p in videos if p.name in preds]
    summary = []
    for p in videos:
        v = build(p, preds.get(p.name), cfg, flow, not args.no_video, args.keep_video)
        summary.append({"name": v["name"], "stem": p.stem, "duration": v["video"]["duration"],
                        "resolution": f'{v["video"]["width"]}x{v["video"]["height"]}',
                        "fps": v["video"]["fps"], "events": len(v["events"]),
                        "event_counts": v["event_counts"], "alarms": len(v["alarms"]),
                        "from_predictions": p.name in preds})
    if Path(args.pred).exists():     # the site links to exactly the file its charts were built from
        shutil.copy(args.pred, DATA.parent / "predictions_samples.json")
    summary.sort(key=lambda v: -v["duration"])          # the longest sample leads the tabs
    (DATA / "index.json").write_text(json.dumps({"videos": summary}, indent=1), encoding="utf-8")
    print(f"wrote {DATA / 'index.json'} ({len(summary)} videos)")


if __name__ == "__main__":
    main()
