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
| Signal colour | rule (lit red/green share in the lamp ROI, majority vote, per-approach timing offset) | `src/events/signal.py` |
| Accident risk | rule (time-to-collision + braking, logistic) | `src/risk.py` |

Speeds are measured in *box sizes per second* (speed / sqrt(w·h)), which makes thresholds
hold near and far from the camera without calibrating a homography.

**Traffic signal.** The boulevard's own lamps hang side-on to the camera and can't be read. The
left-pole lamp can, and it runs the same 75 s cycle (30 s green, ~3 s flashing, 42 s red), but it
turns red ~4.5 s before boulevard traffic stops and green ~0.6 s after it starts. We measured
that on 126 stop-line crossings in C3897. Read naively, that is ~3 false `red_light` events per
cycle, so each approach in `configs/scene.yaml` carries `red_delay_sec` / `red_early_end_sec`.
The stop line, solid lane lines and crossings were drawn by the team on a frame of the sample.

**Safety valves.** Over 3x the video length the harness scores the whole video as empty, so Part A
stops analysing new frames at 1.3x (returning what it found) and Part B repeats its last score
when it falls behind 1.4x. Each rule runs in its own `try` so one failure can't empty a video.

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
stride. Part B runs detection in a worker thread with a fixed one-sample lag, so its curve
does not depend on timing. Checked: two full harness runs on `0924.mp4` gave identical events
and an identical risk curve (1,153 samples, max difference 0). Remaining non-determinism: GPU
floating-point differences across hardware can move a box by a fraction of a pixel.

## Runtime

Models load when `solution.py` is imported (before the harness starts per-video timers).
Frames are decoded in a background thread and shrunk to the detector's 960 px input with exact
2x halvings; skipped frames use `grab()` (no colour conversion), which nearly doubles 4K decoding.

Measured with the official harness on an RTX 3050 laptop GPU, plugged in (budget = 3x duration):

| Video | CPU threads | Part A | Part B | Total | x video length |
|---|---|---|---|---|---|
| C3897, 4K, 317.8 s | 8 (like the evaluation machine) | 197 s | 263 s | 460 s | **1.45x** |
| C3897, 4K, 317.8 s | 16 | 175 s | 223 s | 398 s | 1.25x |
| 0924, 1080p, 38.4 s | 16 | 6 s | 5 s | 11 s | 0.30x |

About half of Part B is the harness's own 4K decoding (0.8x on 8 threads), which no solution
avoids. Not yet measured on a T4.

## Dev labels and tuning

The team's answer key (`dev/labels.json`, the organizers' ground-truth format) is the only
training signal: rule thresholds and per-class start/end shifts are fitted to it, one class at a
time because Score A averages per-class F1.

```bash
python scripts/merge_labels.py dev/labels/*.json            # -> dev/labels.json
python scripts/tune.py --gt dev/labels.json --videos <dir>  # report
python scripts/tune.py --gt dev/labels.json --videos <dir> --apply   # writes configs/tuned.yaml
```

A change is kept only if it beats the current setting by 0.02 F1 (one short video over-fits
easily). `configs/tuned.yaml` is loaded on top of `configs/pipeline.yaml`; delete it to undo.

## Known limitations

- **Real crashes are untested on this camera.** None of the sample videos contains one. On a
  public roadside-camera crash clip (TU-DAT, used only for this check), the accident rule and
  Part B both missed a side-swipe. The detector saw both cars, but the tracker lost the car
  that spun and gave it a new id, and the rule needs both tracks to continue after contact.
  Part B treats side-by-side cars as normal lane-keeping, which removed most false alarms on
  our camera and also hides side-swipes. Next steps: re-link a lost track near the contact
  point, and a learned clip classifier to confirm accident candidates.
- **The traffic light is read indirectly.** The boulevard's own lamps are side-on to the
  camera; we read a lamp on the same 75 s cycle and correct for its measured 4.5 s / 0.6 s
  offset. A retimed signal plan would need re-measuring.
- **Labels are small.** Rules were tuned on ~40 real events judged by the team on the
  samples, then checked by eye on two more videos (one at dusk). Expect lower scores on
  the hidden set than on our dev key.
- `road_obstacle`, `fire_smoke`, `wrong_way`, `illegal_turn` and `illegal_u_turn` are not
  emitted (no reliable rule, or no zones for them).

## Prior work we learned from

- AI City Challenge Track 4 winners: background-modelling for stalled vehicles, road masks
  from trajectories, backtracking to accurate start times ("Good Practices and A Strong
  Baseline for Traffic Anomaly Detection", arXiv:2105.03827).
- Open-source red-light violation repos: HSV lamp state + majority vote + stop-line crossing.
- smart-traffic-vision (GitHub): sudden stop + box overlap for accidents without training data.

## Team

TODO: members, roles, who did what.
