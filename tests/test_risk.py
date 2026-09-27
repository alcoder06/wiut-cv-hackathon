"""Part B scorer on synthetic scenes: crashes must alarm in time, calm traffic must not.

Tightening the scorer to cut false alarms on the sample video is only safe if these still
pass: the metric matches an alarm that starts within 10 s BEFORE the crash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluate import alarm_starts
from src.risk import CausalRisk
from src.tracking import TRACK_COLS
from tests.synth import FPS, H, LEFT_LANES, RIGHT_LANES, W, Synth, path


def risk_curve(s: Synth) -> list[list[float]]:
    tracks = pd.DataFrame(s.rows, columns=TRACK_COLS)
    est = CausalRisk()
    est.reset({"video_id": "synthetic.mp4", "fps": FPS, "width": W, "height": H, "n_frames": int(s.duration * FPS)})
    return [[t, est.observe(g[["x1", "y1", "x2", "y2"]].to_numpy(), g["cls"].tolist(), g["tid"].to_numpy(), t)]
            for (_, t), g in tracks.groupby(["frame", "t"], sort=True)]


def alarm_before(curve, crash_t: float) -> float | None:
    """Earliest alarm start in [crash_t - 10, crash_t), i.e. one the metric would match."""
    hits = [a for a in alarm_starts(curve) if crash_t - 10 <= a < crash_t]
    return min(hits) if hits else None


def test_calm_traffic_raises_no_alarm():
    assert alarm_starts(risk_curve(Synth().background())) == []


def test_head_on_crash_alarms_in_time():
    s = Synth()
    t = s.t[(s.t >= 20) & (s.t <= 35)]
    y = RIGHT_LANES[1]
    s.actor("car", path(t, 20, 30, (40, y), (620, y)), t)      # both stop dead at contact, t = 30
    s.actor("car", path(t, 20, 30, (1240, y), (660, y)), t)
    first = alarm_before(risk_curve(s), 30.0)
    assert first is not None and first <= 29.5, first


def test_side_impact_alarms_in_time():
    s = Synth()
    t = s.t[(s.t >= 20) & (s.t <= 35)]
    s.actor("car", path(t, 20, 30, (40, LEFT_LANES[0]), (640, LEFT_LANES[0])), t)   # driving right
    # coming down into its side. From an elevated camera the striking car's front ends up
    # inside the other car's box at contact (that box includes roof and far side).
    s.actor("car", path(t, 22, 30, (640, 120), (640, LEFT_LANES[0] - 12)), t)
    first = alarm_before(risk_curve(s), 30.0)
    assert first is not None and first <= 29.5, first


def test_passing_a_stopped_queue_is_not_a_threat():
    s = Synth()
    t = s.t[(s.t >= 5) & (s.t <= 40)]
    for i in range(5):                          # a queue standing still in one lane
        s.actor("car", np.tile([[300 + 60 * i, RIGHT_LANES[2]]], (len(t), 1)), t)
    for start in range(5, 30, 3):               # traffic flowing past in the next lane
        tt = t[(t >= start) & (t <= start + 8)]
        s.actor("car", path(tt, start, start + 8, (0, RIGHT_LANES[3]), (1280, RIGHT_LANES[3])), tt)
    assert alarm_starts(risk_curve(s)) == []


def test_part_b_catches_up_after_falling_behind():
    """Only Part B's own processing counts against its cap, so skipping lets it resume.
    (Capping total wall time froze the risk curve for good on a loaded machine.)"""
    est = CausalRisk()
    est.reset({"video_id": "x.mp4", "fps": FPS, "width": W, "height": H, "n_frames": 1000})
    est.own_sec = 10.0
    assert est._behind_schedule(5.0)          # 10 s of our own work in 5 s of video: skip
    assert not est._behind_schedule(40.0)     # the video moved on while we skipped: resume
