"""Align each video to the view the zones were drawn on.

The camera is fixed, but not between recordings: C3902 and C3905 sit up to ~130 px (4K)
and ~1 degree off the C3897 framing that configs/scene.yaml was drawn on, which put the
crossings, stop line and traffic-lamp window in the wrong place. Before tracking, a few
frames are median-stacked into a background (moving traffic drops out), matched to
configs/reference_view.png (a C3897 background) with ORB features on contrast-equalised
images (dusk vs daylight), and a similarity transform (shift, rotation, scale) is fitted
with RANSAC. The hand-drawn zones are moved with it. A view within a few pixels of the
reference is left exactly as drawn, so aligned videos give the same output as before.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np

from .config import resolve

REF_W, REF_H = 960, 540
S = np.diag([REF_W, REF_H, 1.0])       # normalised -> reference pixels


def _background(path: str, n: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    frames = []
    for t in np.linspace(min(1.0, duration / 2), max(duration - 1.0, 0.0), n):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, frame = cap.read()
        if ok:
            small = cv2.resize(frame, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.median(np.stack(frames), 0).astype(np.uint8) if frames else None


def view_transform(path: str, width: int, height: int, cfg) -> np.ndarray | None:
    """3x3 map from reference-normalised to this video's normalised coordinates, or None
    when the view matches the reference or can't be matched reliably (zones stay as drawn)."""
    c = cfg.scene.registration
    ref_path = resolve(c.reference)
    if not c.enabled or not ref_path.exists() or abs(width / height - REF_W / REF_H) > 0.01:
        return None
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    img = _background(path, c.frames)
    if img is None:
        return None
    clahe = cv2.createCLAHE(3.0, (8, 8))
    orb = cv2.ORB_create(6000, fastThreshold=10)
    kr, dr = orb.detectAndCompute(clahe.apply(ref), None)
    k, d = orb.detectAndCompute(clahe.apply(img), None)
    if dr is None or d is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dr, d, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < c.min_inliers:
        return None
    src = np.float32([kr[m.queryIdx].pt for m in good])
    dst = np.float32([k[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(cfg.seed)                         # RANSAC samples: same answer every run
    A, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                             maxIters=5000, confidence=0.999)
    if A is None or int(inliers.sum()) < c.min_inliers:
        return None
    scale = float(np.hypot(A[0, 0], A[1, 0]))
    degrees = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    shift = float(np.hypot(A[0, 2], A[1, 2])) / REF_W
    if abs(scale - 1) < c.same_scale and abs(degrees) < c.same_deg and shift < c.same_shift_frac:
        return None
    if abs(scale - 1) > 0.15 or abs(degrees) > 10:   # not a drifted fixed camera: don't trust it
        return None
    return np.linalg.inv(S) @ np.vstack([A, [0, 0, 1]]) @ S


def warp_manual(manual: dict, T: np.ndarray | None) -> dict:
    """configs/scene.yaml geometry (normalised coordinates) moved into this video's view."""
    if T is None:
        return manual

    def pts(p):
        a = np.asarray(p, float).reshape(-1, 2)
        return (a @ T[:2, :2].T + T[:2, 2]).round(5).tolist()

    linear = S[:2, :2] @ T[:2, :2] @ np.linalg.inv(S[:2, :2])     # pixel-space rotation/scale
    m = copy.deepcopy(manual)
    for key in ("carriageway", "crosswalks", "refuges", "solid_lines"):
        m[key] = [pts(p) for p in m.get(key) or []]
    for a in m.get("approaches") or []:
        a["stop_line"] = pts(a["stop_line"])
        if a.get("intersection"):
            a["intersection"] = pts(a["intersection"])
        if a.get("signal_roi"):
            x1, y1, x2, y2 = a["signal_roi"]
            corners = np.array(pts([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]))
            a["signal_roi"] = [*corners.min(0).tolist(), *corners.max(0).tolist()]
        a["direction"] = (linear @ np.asarray(a["direction"], float)).tolist()
    moves = m.get("movements") or {}
    if moves:
        moves["u_turn_prohibited"] = [pts(p) for p in moves.get("u_turn_prohibited") or []]
        moves["forbidden"] = [{**f, "from": pts(f["from"]), "to": pts(f["to"])}
                              for f in moves.get("forbidden") or []]
    return m


def warp_grid(values: np.ndarray, T: np.ndarray, width: int, height: int, grid: int) -> np.ndarray:
    """A per-cell array learned in the reference view (the prebuilt flow field), moved into
    this video's view."""
    ny, nx = values.shape[:2]
    C = np.diag([width / grid, height / grid, 1.0])      # normalised -> cell coordinates
    M = (C @ T @ np.linalg.inv(C))[:2]
    return cv2.warpAffine(values, M, (nx, ny), flags=cv2.INTER_NEAREST, borderValue=0)
