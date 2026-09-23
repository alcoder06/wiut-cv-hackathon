"""Metadata + raw decode speed for every video in a folder (EDA table and time budget).

    python scripts/probe_videos.py samples
Decode speed matters because the harness decodes every frame for Part B no matter what:
if decoding alone is slower than ~3x real time we must cut resolution work elsewhere.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.video import probe  # noqa: E402


def decode_fps(path: str, seconds: float = 20.0) -> float:
    cap = cv2.VideoCapture(path)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds and cap.read()[0]:
        n += 1
    cap.release()
    return n / (time.perf_counter() - t0)


def main(folder: str) -> None:
    rows = []
    for p in sorted(p for p in Path(folder).iterdir() if p.suffix.lower() == ".mp4"):
        info = probe(str(p))
        fourcc = int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FOURCC))
        rows.append({
            "video": p.name, "width": info.width, "height": info.height, "fps": info.fps,
            "frames": info.n_frames, "duration_s": round(info.duration, 1),
            "size_gb": round(p.stat().st_size / 1e9, 2),
            "codec": fourcc.to_bytes(4, "little").decode(errors="replace"),
            "decode_fps": round(decode_fps(str(p)), 1),
        })
        r = rows[-1]
        print(f"{r['video']}: {r['width']}x{r['height']} @ {r['fps']:.2f} fps, {r['duration_s']} s, "
              f"{r['size_gb']} GB, {r['codec']}, decodes at {r['decode_fps']} fps "
              f"({r['decode_fps'] / r['fps']:.1f}x real time)")
    Path("dev").mkdir(exist_ok=True)
    Path("dev/video_meta.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples")
