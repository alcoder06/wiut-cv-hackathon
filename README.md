# Traffic event detection & accident anticipation — WIUT Hackathon 2026, CV track

A fixed CCTV camera → every traffic event as `[start_sec, end_sec, label]` (Part A) and a
causal per-frame accident risk (Part B). Pretrained detector + tracker + rules on
trajectories. No model is trained on our side.

## Install and run

```bash
pip install -r requirements.txt          # or: docker build -t team .
sh weights/download.sh                   # weights are committed; this only verifies / re-fetches them
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

`run_submission.py` and `evaluate.py` are the organizers' files, unchanged.

Development:

```bash
python scripts/probe_videos.py samples      # resolution, fps, decode speed -> dev/video_meta.json
python scripts/build_tracks.py samples      # detect+track once, cache, learn configs/scene_flow.npz
python run_submission.py --videos samples --out predictions_samples.json --team <team>
python evaluate.py --pred predictions_samples.json --gt dev/labels.json --per-video
python scripts/render.py samples/<video>.mp4 --start 0 --end 120   # annotated video for review
pytest -q                                   # rules on synthetic trajectories
```

## Approach

```
video ──► YOLO11s (FP16, 8 fps) ──► ByteTrack ──► tracks ──► kinematics (ground point, speed in box-sizes/s)
                                                               │
             scene: hand-drawn zones (configs/scene.yaml) + flow field learned from trajectories
                                                               │
                         one rule module per event family (src/events/)
                                                               │
               segment clean-up: merge gaps, drop blips, union same-class, clip ──► events
Part B (causal): own YOLO + ByteTrack at 5 fps ──► pairwise time-to-collision + braking ──► EMA ──► risk
```

| Component | Learned or rule-based | Where |
|---|---|---|
| Road-user detection | learned (COCO-pretrained YOLO11s, not fine-tuned) | `src/detector.py` |
| Tracking | algorithmic (ByteTrack) | `src/tracking.py` |
| Carriageway + lane directions | learned from trajectories (statistics, no training) | `src/scene.py` |
| All 12 enabled event classes | rules on trajectories + scene | `src/events/*.py` |
| Signal colour | rule (HSV share in the lamp ROI, majority vote) | `src/events/signal.py` |
| Accident risk | rule (time-to-collision + braking, logistic) | `src/risk.py` |

Speeds are measured in *box sizes per second* (speed / sqrt(w·h)), which makes thresholds
hold near and far from the camera without calibrating a homography.

`road_obstacle` and `fire_smoke` are disabled: predicting a class that never occurs adds a
zero to the macro-F1, and we have no reliable detector for them.

## Models and datasets

| Item | Licence | Use |
|---|---|---|
| YOLO11s weights (Ultralytics, COCO) | AGPL-3.0 | detection, used as-is |
| `ultralytics` | AGPL-3.0 | inference |
| `supervision` ByteTrack (Roboflow) | MIT | tracking |
| Our own labels of the sample videos (`dev/labels.json`) | ours | tuning and evaluation |

No external training data is used.

## Determinism

Seeds are fixed in `configs/pipeline.yaml` (`seed`) and applied by `src/config.seed_everything`
(Python, NumPy, PyTorch; cuDNN deterministic, benchmark off). Frame sampling is by fixed
stride. Remaining non-determinism: GPU floating-point differences across hardware can move a
box by a fraction of a pixel.

## Runtime

Models load when `solution.py` is imported (before the harness starts per-video timers).
TODO: timing table on a T4 (Part A, Part B, total, × real time).

## Prior work we learned from

- AI City Challenge Track 4 winners: background-modelling for stalled vehicles, road masks
  from trajectories, backtracking to accurate start times ("Good Practices and A Strong
  Baseline for Traffic Anomaly Detection", arXiv:2105.03827).
- Open-source red-light violation repos: HSV lamp state + majority vote + stop-line crossing.
- smart-traffic-vision (GitHub): sudden stop + box overlap for accidents without training data.

## Team

TODO: members, roles, who did what.
