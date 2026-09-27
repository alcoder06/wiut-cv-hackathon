"""Build configs/reference_view.npz: the view that configs/scene.yaml was drawn on, as features.

    python scripts/make_reference.py D:/videos/C3897.MP4

src/view.py aligns every recording to this view before placing the zones. It stores ORB
keypoints and descriptors of a median background of the video (moving traffic drops out),
not the image, so no frame of the footage ships with the code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.view import _background, _features  # noqa: E402


def main() -> None:
    video = sys.argv[1]
    background = _background(video, 9)
    points, descriptors = _features(background)
    out = ROOT / "configs" / "reference_view.npz"
    np.savez_compressed(out, points=points, descriptors=descriptors)
    print(f"{out}: {len(points)} features from {video}")


if __name__ == "__main__":
    main()
