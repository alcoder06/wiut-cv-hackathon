"""Each rule on a synthetic scene where the correct segment is known."""
from __future__ import annotations

import numpy as np
import pytest

from src.events import collision, congestion, pedestrian, stationary, wrong_way
from src.kinematics import ttc_matrix
from src.segments import finalize
from tests.synth import LEFT_LANES, RIGHT_LANES, Synth, path


def overlap(event, start, end):
    s, e = event[0], event[1]
    inter = max(0.0, min(e, end) - max(s, start))
    return inter / (max(e, end) - min(s, start))


@pytest.fixture(scope="module")
def plain():
    return Synth().background().context()


def test_quiet_traffic_raises_nothing(plain):
    for rule in (stationary, congestion, wrong_way, pedestrian, collision):
        assert rule.detect(plain) == [], rule.__name__


def test_stopped_vehicle():
    s = Synth().background()
    t = s.t
    # drives in, stops at x=640 from t=20 to t=38, drives off
    xy = np.where((t < 20)[:, None], path(t, 10, 20, (0, 364), (640, 364)),
                  np.where((t < 38)[:, None], [[640, 364]], path(t, 38, 45, (640, 364), (1280, 364))))
    keep = (t >= 10) & (t <= 45)
    s.actor("car", xy[keep], t[keep])
    ev = stationary.detect(s.context())
    assert len(ev) == 1 and overlap(ev[0], 20, 38) > 0.8, ev


def test_short_stop_is_not_an_event():
    s = Synth().background()
    t = s.t[(s.t >= 20) & (s.t <= 26)]
    s.actor("car", np.tile([[640, 364]], (len(t), 1)), t)
    assert stationary.detect(s.context()) == []


def test_wrong_way():
    s = Synth().background()
    t = s.t[(s.t >= 10) & (s.t <= 20)]
    s.actor("car", path(t, 10, 20, (1200, RIGHT_LANES[1]), (100, RIGHT_LANES[1])), t)
    ev = wrong_way.detect(s.context())
    assert len(ev) == 1 and overlap(ev[0], 10, 20) > 0.7, ev


def test_jaywalking():
    s = Synth().background()
    t = s.t[(s.t >= 5) & (s.t <= 15)]
    s.actor("person", path(t, 5, 15, (600, 260), (600, 570)), t, w=16, h=40)
    ev = finalize(pedestrian.detect(s.context()), 60, 1.0, 0.5)
    assert len(ev) == 1 and ev[0][2] == "jaywalking", ev
    assert 5 < ev[0][0] < 8 and 11 < ev[0][1] < 15, ev


def test_accident():
    s = Synth().background(until=10)          # lanes are empty again by t ~ 21 s
    t = s.t[(s.t >= 25) & (s.t <= 40)]
    # two cars converge on (640, 460), boxes overlap by 20 px, and stop dead at t=30
    a = path(t, 25, 30, (40, LEFT_LANES[1]), (630, LEFT_LANES[1]))
    b = path(t, 25, 30, (1240, LEFT_LANES[1]), (650, LEFT_LANES[1]))
    s.actor("car", a, t)
    s.actor("car", b, t)
    ev = [e for e in collision.detect(s.context()) if e[2] == "accident"]
    assert len(ev) == 1 and abs(ev[0][0] - 30) < 1.0, ev


def test_ttc_head_on():
    pos = np.array([[0.0, 0.0], [100.0, 0.0]])
    vel = np.array([[10.0, 0.0], [-10.0, 0.0]])
    size = np.array([20.0, 20.0])
    ttc = ttc_matrix(pos, vel, size, radius_factor=0.25)
    # gap 100 px closes at 20 px/s until the discs (radius 5 each) touch: (100 - 10) / 20
    assert ttc[0, 1] == pytest.approx(4.5)
    assert np.isinf(ttc[0, 0])


def test_finalize_unions_same_class_and_clips():
    ev = finalize([[1, 5, "jaywalking"], [4, 9, "jaywalking"], [9.5, 10, "jaywalking"],
                   [58, 70, "congestion"], [20, 20.2, "wrong_way"]], duration=60, max_gap=1.0, min_len=0.5)
    assert ev == [[1.0, 10.0, "jaywalking"], [58.0, 60.0, "congestion"]]
