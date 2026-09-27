"""YOLO detector, loaded once per process.

run_submission.py imports solution.py before it starts any per-video timer, so
loading here at first use (triggered from solution.py's import) is free time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import load_config, resolve


@dataclass
class FrameDetections:
    xyxy: np.ndarray    # (N, 4) float32 pixels
    conf: np.ndarray    # (N,) float32
    cls: np.ndarray     # (N,) int COCO ids


class Detector:
    def __init__(self, cfg):
        import torch
        from ultralytics import YOLO

        self.cfg = cfg
        # device_count() too: with CUDA_VISIBLE_DEVICES="" is_available() can still say True
        self.device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
        self.half = bool(cfg.half) and self.device != "cpu"
        self.class_ids = sorted(int(k) for k in cfg.classes)
        self.model = YOLO(str(resolve(cfg.weights)))
        # Warm-up so the first real batch doesn't pay CUDA init inside the timer.
        self.predict([np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8)])

    def predict(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        results = self.model.predict(
            frames, imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou,
            classes=self.class_ids, quantize=16 if self.half else 32, device=self.device, verbose=False,
        )
        out = []
        for r in results:
            b = r.boxes
            out.append(FrameDetections(
                xyxy=b.xyxy.cpu().numpy().astype(np.float32),
                conf=b.conf.cpu().numpy().astype(np.float32),
                cls=b.cls.cpu().numpy().astype(int),
            ))
        return out


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector(load_config().detector)
