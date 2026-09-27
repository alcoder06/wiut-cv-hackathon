"""The team website and its live demo: static pages from web/static plus an upload API
that runs the same pipeline as the submission on the visitor's video.

    python -m uvicorn web.server:app --port 7860        # from the repo root

The pages can also be hosted elsewhere and call this API across origins (CORS is open: the
API is public and takes no credentials); web/modal_app.py deploys it API-only.

The demo uses configs/demo.yaml (YOLO11n at 640 px, 4 fps) so a CPU-only host keeps up,
and builds the risk curve by replaying the tracks through Part B's causal scorer
(src/risk.replay_tracks) instead of running a second detector. Jobs run one at a time;
uploads and results are deleted after KEEP_SEC.
"""
from __future__ import annotations

import os

os.environ.setdefault("TRAFFIC_PROFILE", "demo")
os.environ.setdefault("YOLO_OFFLINE", "1")

import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, seed_everything  # noqa: E402
from src.video import probe  # noqa: E402

MAX_MB = 200
MAX_SEC = 120.0
MAX_QUEUE = 6
KEEP_SEC = 2 * 3600
JOBS_DIR = Path(os.environ.get("DEMO_JOBS_DIR", Path(tempfile.gettempdir()) / "traffic_demo_jobs"))
STATIC = Path(__file__).resolve().parent / "static"


@dataclass
class Job:
    id: str
    dir: Path
    name: str
    created: float = field(default_factory=time.time)
    state: str = "queued"            # queued | running | done | error
    stage: str = "Waiting in the queue"
    progress: float = 0.0
    started: float | None = None
    error: str | None = None
    result: dict | None = None

    def public(self) -> dict:
        ahead = sum(1 for j in list(JOBS.values()) if j.state == "queued" and j.created < self.created)
        busy = any(j.state == "running" for j in list(JOBS.values()))
        elapsed = time.time() - self.started if self.started else 0.0
        eta = elapsed * (1 - self.progress) / self.progress if self.progress > 0.05 else None
        return {"id": self.id, "state": self.state, "stage": self.stage, "progress": round(self.progress, 3),
                "queue_position": ahead + int(busy) if self.state == "queued" else 0,
                "elapsed_sec": round(elapsed, 1), "eta_sec": round(eta) if eta is not None else None,
                "error": self.error, "result": self.result}


JOBS: dict[str, Job] = {}
QUEUE: Queue[str] = Queue()


def _stage(job: Job, text: str, lo: float, hi: float):
    """Set the stage label and return a callback mapping 0..1 inside it to lo..hi overall."""
    job.stage, job.progress = text, lo
    return lambda f: setattr(job, "progress", lo + (hi - lo) * max(0.0, min(1.0, f)))


def process(job: Job) -> None:
    from src.annotate import render
    from src.pipeline import analyse
    from src.risk import replay_tracks
    from src.stats import object_counts

    video = str(job.dir / "input.mp4")
    t0 = time.perf_counter()
    ctx, events = analyse(video, progress=_stage(job, "Detecting and tracking road users", 0.02, 0.72))
    t_a = time.perf_counter() - t0
    _stage(job, "Scoring accident risk", 0.72, 0.75)
    risk = replay_tracks(ctx.tracks, ctx.info, job.name)
    render(video, ctx, events, str(job.dir / "annotated.mp4"), width=960, risk=risk,
           progress=_stage(job, "Rendering the annotated video", 0.75, 1.0))
    info = ctx.info
    job.result = {
        "video": {"name": job.name, "width": info.width, "height": info.height, "fps": round(info.fps, 3),
                  "duration": round(info.duration, 2), "frames": info.n_frames},
        "events": events,
        "risk": risk,
        "objects": object_counts(ctx.tracks, info.duration),
        "tracks": int(ctx.tracks["tid"].nunique()) if len(ctx.tracks) else 0,
        "timing": {"part_a_sec": round(t_a, 1), "total_sec": round(time.perf_counter() - t0, 1)},
        "annotated_url": f"/api/jobs/{job.id}/video",
    }


def worker() -> None:
    seed_everything(load_config().seed)
    try:                             # load + warm up once, before the first visitor waits on it
        from src.detector import get_detector

        get_detector()
    except Exception:                # keep serving: each job retries and reports the error itself
        traceback.print_exc()
    while True:
        job = JOBS.get(QUEUE.get())
        if job is None:
            continue
        job.state, job.started = "running", time.time()
        try:
            process(job)
            job.state, job.stage, job.progress = "done", "Done", 1.0
        except Exception as exc:     # a bad upload must never take the server down
            traceback.print_exc()
            job.state, job.error = "error", f"Processing failed: {exc}"
        finally:
            (job.dir / "input.mp4").unlink(missing_ok=True)


def cleanup() -> None:
    for jid, job in list(JOBS.items()):
        if time.time() - job.created > KEEP_SEC and job.state in ("done", "error"):
            shutil.rmtree(job.dir, ignore_errors=True)
            JOBS.pop(jid, None)


_worker_lock = threading.Lock()
_worker_started = False


def ensure_worker() -> None:
    """Start the job worker once. Called at startup and again on the first upload, for hosts
    that don't run ASGI lifespan events."""
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            threading.Thread(target=worker, daemon=True).start()
            _worker_started = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_worker()
    yield


app = FastAPI(title="Traffic event detection demo", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/limits")
def limits() -> dict:
    return {"max_mb": MAX_MB, "max_sec": MAX_SEC}


@app.post("/api/jobs")
async def create_job(file: UploadFile) -> dict:
    ensure_worker()
    cleanup()
    if sum(1 for j in JOBS.values() if j.state in ("queued", "running")) >= MAX_QUEUE:
        raise HTTPException(503, "The demo is busy. Please try again in a few minutes.")
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    path = job_dir / "input.mp4"
    size = 0
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB << 20:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"The file is larger than {MAX_MB} MB.")
            f.write(chunk)
    try:
        info = probe(str(path))
        if info.n_frames <= 0 or info.fps <= 0:
            raise RuntimeError("no frames")
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(415, "This file could not be read as a video. Please upload an .mp4 (H.264).")
    if info.duration > MAX_SEC + 0.5:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(413, f"The video is {info.duration:.0f} s long; the demo accepts up to {MAX_SEC:.0f} s.")
    name = Path(file.filename or "upload.mp4").name
    JOBS[job_id] = Job(job_id, job_dir, name)
    QUEUE.put(job_id)
    return JOBS[job_id].public()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    return job.public()


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    path = job.dir / "annotated.mp4" if job else None
    if path is None or job.state != "done" or not path.exists():
        raise HTTPException(404, "No annotated video for this job.")
    return FileResponse(path, media_type="video/mp4")


if (STATIC / "index.html").exists():          # absent in the API-only deployment
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="site")
