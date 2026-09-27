"""Summaries of the tracks table for the website (EDA charts and the live demo's results)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Detector classes folded into four groups: charts stay readable and colours stay fixed.
GROUPS = {"car": "car", "person": "person", "bus": "heavy", "truck": "heavy",
          "motorcycle": "two_wheeler", "bicycle": "two_wheeler"}
GROUP_ORDER = ["car", "person", "heavy", "two_wheeler"]


def object_counts(tracks: pd.DataFrame, duration: float, bin_sec: float = 1.0) -> dict:
    """Mean number of tracked objects per analysed frame, per group, in `bin_sec` bins.
    Returns {"t": [...], "car": [...], ...} (t = bin start)."""
    n = max(1, int(np.ceil(duration / bin_sec)))
    out = {"t": [round(i * bin_sec, 2) for i in range(n)]}
    if tracks.empty:
        return out | {g: [0.0] * n for g in GROUP_ORDER}
    df = tracks.assign(group=tracks["cls"].map(GROUPS), bin=(tracks["t"] // bin_sec).astype(int).clip(0, n - 1))
    frames_per_bin = df.groupby("bin")["frame"].nunique().reindex(range(n)).fillna(0).to_numpy()
    for g in GROUP_ORDER:
        rows = df[df["group"] == g].groupby("bin").size().reindex(range(n)).fillna(0).to_numpy()
        out[g] = np.round(rows / np.maximum(frames_per_bin, 1), 2).tolist()
    return out
