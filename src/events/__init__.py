"""Rule-based event detectors.

Every detector has the same signature `detect(ctx) -> list[[start, end, label]]` and
reads only the shared Context. The pipeline runs them all and keeps the classes listed
in `enabled_classes`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..scene import Scene
from ..video import VideoInfo


@dataclass
class Context:
    tracks: pd.DataFrame     # tracks + kinematics + on_road
    frames: pd.DataFrame     # per analysed frame: t, signal colour fractions
    scene: Scene
    info: VideoInfo
    cfg: object              # Cfg from configs/pipeline.yaml

    def vehicles(self) -> pd.DataFrame:
        from ..kinematics import VEHICLES

        return self.tracks[self.tracks["cls"].isin(VEHICLES)]

    def people(self) -> pd.DataFrame:
        return self.tracks[self.tracks["cls"] == "person"]


def all_detectors():
    from . import collision, congestion, lines, pedestrian, signal, stationary, turns, wrong_way

    return [
        stationary.detect,
        congestion.detect,
        wrong_way.detect,
        pedestrian.detect,
        signal.detect,
        lines.detect,
        turns.detect,
        collision.detect,
    ]
