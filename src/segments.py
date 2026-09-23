"""Turn raw detections into clean [start, end, label] segments.

The metric matches by temporal IoU up to 0.7 and forbids same-class overlaps, so every
rule's output passes through the same clean-up: merge fragments, drop blips, union
overlapping same-class segments (the FAQ says two simultaneous events of one class are
one segment), and clip to the video duration.
"""
from __future__ import annotations

from collections import defaultdict

Event = list  # [start_sec, end_sec, label]


def merge_intervals(intervals: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def finalize(events: list[Event], duration: float, max_gap: float, min_len: float) -> list[Event]:
    by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s, e, label in events:
        s, e = max(0.0, float(s)), min(float(duration), float(e))
        if e > s:
            by_class[label].append((s, e))
    out = []
    for label, ivs in by_class.items():
        for s, e in merge_intervals(ivs, max_gap):
            if e - s >= min_len:
                out.append([round(s, 3), round(e, 3), label])
    return sorted(out)
