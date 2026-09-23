"""
solution.py: the interface the organizers' harness (run_submission.py) imports.

    detect_events(video_path) -> [[start_sec, end_sec, label], ...]   Part A (src/pipeline.py)
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float          Part B (src/risk.py)

The detector is loaded at import time: the harness imports this module before it starts
any per-video timer, so model loading and CUDA warm-up cost nothing against the budget.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# The evaluation machine has no internet; stop Ultralytics from trying to reach it.
os.environ.setdefault("YOLO_OFFLINE", "1")

from src.config import load_config, seed_everything

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0

seed_everything(load_config().seed)
try:
    from src.detector import get_detector

    get_detector()
except Exception as exc:  # still importable (e.g. for format checks); calls will raise later
    print(f"solution.py: detector not loaded at import: {exc}", file=sys.stderr)


def detect_events(video_path: str) -> list[list]:
    from src.pipeline import detect_events as run

    return run(video_path)


class RiskEstimator:
    """Causal: step() sees frames in order and nothing else."""

    def __init__(self):
        from src.risk import CausalRisk

        self._impl = CausalRisk()

    def reset(self, meta: dict) -> None:
        self._impl.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._impl.step(frame, t_sec)
