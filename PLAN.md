# Hackathon plan — traffic event detection + accident anticipation

> **Deadline: Sun 27 Sep 2026 — 4-day sprint.** Anything not on the Day 1–4 list is out of scope until after elimination.
> Never cut: a valid tagged submission every night, and the clean-machine test.

---

## 1. What we're actually building

| Deliverable | What it is | Hard requirement |
|---|---|---|
| **Repo** | `solution.py` with `detect_events(path)` + `RiskEstimator`, weights, `requirements.txt`/Dockerfile | Runs on their machine with `run_submission.py` unchanged, offline, T4 16 GB, ≤ 3× video duration, ≤ 5 GB weights, deterministic |
| **Part A** | `.mp4` → `[[start, end, label], ...]` over 14 classes | Same-class segments never overlap; `0 ≤ start < end ≤ duration` |
| **Part B** (bonus, but 18% of total) | Causal per-frame `P(accident starts in next 5 s)` | Only frames already seen; can't reuse Part A output |
| **Website** | Team, approach, EDA, annotated sample videos, **live upload demo**, report, links | Online through the whole judging period |
| **Report** | One page: what worked, what didn't, next steps | Can be a page on the site |

### What we submit (from the PDF's "Submission package" section)
One link to a **public Git repo (tag or commit hash)** + one link to the **website**. Required layout:

```
your-repo/
├── solution.py               # the interface
├── run_submission.py         # starter kit, UNCHANGED
├── evaluate.py               # starter kit, UNCHANGED
├── requirements.txt          # or Dockerfile in repo root
├── weights/                  # weights, or download.sh (run once WITH internet before the offline run), ≤ 5 GB
├── src/                      # models, tracking, rules, training scripts
├── notebooks/                # optional
├── predictions_samples.json  # our output on the sample videos
└── README.md
```
They run: `pip install -r requirements.txt` (or `docker build -t team .`) then
`python run_submission.py --videos /data/test --out predictions.json`.
README **must** contain: install/run + how weights are obtained; architecture, models, training datasets **with licences**, what's learned vs rule-based; seeds and anything non-deterministic; team members and who did what.

### Facts from reading the starter kit code (these change the design)
- **Model loading is free if done at import time.** `load_solution()` imports `solution.py` *before* the per-video timer starts. `RiskEstimator()` is re-created **per video**, and `detect_events` is called per video. → Load all weights at module level (singletons), never in `__init__` or inside `detect_events`.
- **Part A + Part B share one clock.** If Part A alone exceeds 3× duration, Part B is skipped and the whole video is scored empty. The harness decodes **every** frame on CPU for Part B and calls `step` each time → on frames we skip, `step` must return the cached score in microseconds.
- **Two events of the same class at the same moment → one segment covering both** (FAQ; "our annotations do the same"). Otherwise the harness keeps only the earlier-starting one.
- **Part A may use Part B's risk curve; Part B may NOT use Part A** (FAQ). So Part B needs its own causal detector+tracker; Part A can reuse the Part B code (e.g. run the estimator inside `detect_events` to propose accident/near_miss candidates).
- **If the test set has no accidents, M = Score_A** — Part B then counts for nothing. With accidents present, it's 30% of M.
- Harness clips `end` to the video duration and rounds to 3 decimals; events don't need sorting. Risk scores are clamped to [0, 1].
- Part B AP runs over **every frame of every video** (tens of thousands of negatives). Any bump in calm traffic ranks above real pre-crash frames → the baseline must sit near 0 in normal traffic. The example `predictions.json` shows the target shape: flat 0 → ramps ~4 s before the crash → peaks → falls off.
- Alarms starting inside an accident or a near_miss window are **dropped, not counted as false** — confirmed in `evaluate.py` (`frame_label(a) is not None`).
- `CLASSES` may only shrink. Removing a class from it doesn't change scoring (C comes from GT ∪ predictions); it just makes the harness drop that label.

### Resources status
- [x] Task PDF, starter kit (`wiut_cv_scripts/`) — read in full
- [ ] **Sample videos (4, Google Drive links in `Videos.pdf`)** — automated download blocked by Drive's "quota exceeded". Download manually in a browser into `samples/`. If Drive still blocks it: in Drive, right-click → *Make a copy* into your own Drive, then download the copy.
- [ ] **`camera.md` is missing** from what we received — the task says it ships with the samples. Ask in the hackathon channel. If it never comes, we derive the scene layout from the videos ourselves (see Innovation, §2).
- Local machine: RTX 3050 **6 GB** laptop GPU, no Python CV stack installed yet. Target is a **T4 16 GB** → benchmark final timings on a Kaggle/Colab T4, not the laptop. T4 has no bf16: use FP16.

---

## 2. Where the points really are

```
Elimination = 0.6·M + 0.25·Website + 0.15·Code      M = 0.7·A + 0.3·B
```

| Component | Share of final score | Notes |
|---|---|---|
| Score A (event detection) | **42%** | Hardest, most uncertain (hidden test, unseen events) |
| Website | **25%** | Mostly *effort-deterministic* — bank it |
| Score B (anticipation) | **18%** | A good TTC heuristic gets real points cheaply |
| Code | **15%** | 40% of it is just "runs on a clean machine" |

**Takeaway:** 40% of the score (website + code) is under our full control. Nail those, then push A and B as far as possible.
Also: *a package that doesn't run = model score 0*. That's 60% gone. The clean-machine test is the single most important task.

### Second stage: the judging criteria (from the hackathon page)

The task file only covers the **elimination** score. The judging criteria from the hackathon page are for the round after it:

| Criterion | Weight | What we show for it |
|---|---|---|
| Innovation & Originality | 25% | Scene layout **learned from traffic flow** (lane directions and stop zones found from trajectories, not only hand-drawn) → a new camera needs no setup. Every alert comes with its evidence (the track, the rule that fired, a short clip). |
| Technical Complexity | 25% | Offline detection for Part A plus a separate causal warning system for Part B, learned + rule-based hybrid, ablations, speed tuned for a T4 |
| Impact & Practicality | 25% | Numbers an operator cares about: seconds of warning before a crash, false alarms per hour, cameras per GPU, runs offline on one edge GPU |
| Presentation | 15% | Pitch deck, a 2-min demo video (backup if live demo fails), a well-rehearsed live demo |
| Business Potential | 10% | Buyers (city traffic centres, highway operators, insurers, fleet depots), cost per camera per month from measured speed, path from pilot to rollout |

**Priority rule:** the second-stage criteria only count if we survive elimination. Until elimination, **the model score and the elimination rubric come first.** Most second-stage material is free if we build it right: the dashboard, ablations and learned scene layout double as website extra credit.

---

## 3. Metric traps (read these twice)

### Part A — macro F1 over classes, averaged over IoU 0.3/0.5/0.7
- **Predicting a class that isn't in the test set adds a 0 to the macro average.** One false `fire_smoke` can cost ~1/|C| of Score A. → Rare/exotic classes (`fire_smoke`, `road_obstacle`, `illegal_u_turn`) fire **only on very high confidence**. Precision > recall for anything uncertain.
- **A class in the test set that we never predict scores 0 anyway.** → Common classes for this camera (probably `congestion`, `stopped_vehicle`, `jaywalking`, `red_light` if the signal is visible, `solid_line_crossing`) must be predicted, even imperfectly.
- **IoU 0.7 is strict.** A 6 s event needs boundaries within ~1 s. Follow the annotator conventions exactly:
  - `stopped_vehicle` starts when the vehicle **stops**, not when the 10 s rule is confirmed → backdate the start.
  - `accident` ends when objects **stop moving or leave frame**, not at impact.
  - Events running past the video end → `end = duration`.
- **Part A is offline — use future frames.** Smooth tracks both directions, backdate starts, fill gaps. That's a big advantage over Part B.
- **Same-class overlap is forbidden.** Two jaywalkers at once, or congestion in both directions → merge into one union segment ourselves (the harness would silently *drop* the later one).
- F1 is pooled per class over all videos → with few events per class, every single TP matters.

### Part B — `0.4·AP + 0.4·F1_alarm + 0.2·mTTA/10`
- **Alarm starting inside a near_miss window (`[s−5, e]`) or inside an accident is discarded, not counted as a false alarm.** Firing on dangerous-but-no-crash situations is free. Only alarms in calm traffic hurt precision.
- Alarm matches if it starts in `[s−10, s)`. Earlier (up to 10 s) = better mTTA.
- Alarms separated by < 2 s get merged; long false alarm runs = one FP each. Use **hysteresis** (on at ≥ 0.5, off at < 0.3) so noise doesn't create many alarms.
- AP is pooled across videos → scores must be **calibrated the same way on every video** (no per-video normalisation).
- Constant score = 0. A mostly-low baseline with sharp spikes is what we want.

---

## 4. Proposed pipeline

```
video ─► decode (stride 2–5) ─► detector (YOLO11/RT-DETR, FP16)
                                  │
                                  ▼
                         tracker (ByteTrack/BoT-SORT)
                                  │  tracks in image coords (+ optional homography → metres)
                                  ▼
   scene config (from camera.md: lane polygons + direction vectors, stop lines,
   crosswalks, carriageway mask, signal ROI, no-U-turn zones)
                                  │
             ┌────────────────────┼─────────────────────────┐
             ▼                    ▼                         ▼
     rule engines per class   learned models          VLM verifier (optional)
     (wrong_way, red_light,   accident/near_miss      Qwen2.5-VL-3B on short
      stopped, congestion,    (VideoMAE on CCD/DoTA)  candidate clips only
      jaywalking, yield, ...) fire/smoke detector
                                  │
                                  ▼
      segment post-processing: merge fragments, drop blips,
      per-class boundary offsets (tuned on dev set), union same-class overlaps
                                  │
                                  ▼
                           [[start, end, label], ...]

Part B (causal, separate): own light detector+tracker at ~5 fps
 → min time-to-collision between pairs, sudden decel, wrong-way/red-light-in-progress,
   pedestrian entering road → logistic combination → EMA → calibrated score
```

**Class → method map**

| Class | Method | Key signal |
|---|---|---|
| wrong_way | rule | track velocity · lane direction < 0 for N frames |
| red_light | rule + signal colour (HSV in signal ROI) | front crosses stop line while red |
| stop_line | rule | stopped past stop line on red; end when green |
| stopped_vehicle | rule | speed ≈ 0 ≥ 10 s on carriageway, not in signal queue |
| congestion | rule | median speed per direction ≈ 0 across all lanes |
| jaywalking | rule | person foot-point on carriageway, outside crosswalk |
| failure_to_yield | rule | vehicle in crosswalk polygon while person in/entering it |
| solid_line_crossing | rule | bottom-centre of box crosses solid-line polyline |
| illegal_turn / illegal_u_turn | rule | heading change + entry/exit zone pair in forbidden list |
| accident | learned + rule | box overlap + abrupt stop/deflection; video classifier on candidate windows |
| near_miss | learned + rule | hard decel / swerve with low TTC, no contact |
| road_obstacle | background model + open-vocab detector | static non-vehicle blob on carriageway |
| fire_smoke | open-weights fire/smoke detector, high threshold | |

### Methods borrowed from prior work (researched 23 Sep)
| Borrowed | From | Used for |
|---|---|---|
| `supervision` (MIT): `PolygonZone`, `LineZone` (in/out direction), `ByteTrack`, heatmap/trace annotators | Roboflow | zone rules, line crossings, all website visuals |
| MOG2 background subtraction to find things that sit still | AI City Challenge Track 4 winners | `stopped_vehicle`, `road_obstacle` |
| Backtracking from a confirmed stop to the moment motion ended | AI City 2021 winner ("Good Practices and a Strong Baseline") | accurate start times → IoU 0.7 |
| Road mask + lane directions clustered from trajectories | same paper | learned scene layout (innovation story) |
| HSV on the signal ROI + temporal majority vote + stop-line crossing | open-source red-light repos | `red_light`, `stop_line` |
| Sudden stop + box overlap, with min-duration and cooldown filters | smart-traffic-vision | `accident` / `near_miss` without training |

Sources: arxiv.org/abs/2105.03827 · supervision.roboflow.com · github.com/luthfirahmn09/smart-traffic-vision · github.com/HassanRasheed91/Red-Light-Violation-Detection

Detector note: Ultralytics YOLO is **AGPL-3.0** (fine for a public repo, list it). RT-DETR via HF transformers is Apache-2.0 if we want to avoid AGPL.

---

## 5. Scope for a 4-day sprint — what we cut

Deadline: **Sun 27 Sep 2026** (4 days from Wed 23 Sep). There's no time to train anything, so **no training**: pretrained detector + tracker + rules only.

| Keep | Cut (say so honestly in the report as "next steps") |
|---|---|
| YOLO (pretrained COCO) + ByteTrack, FP16, stride 3–5 | Fine-tuning VideoMAE on CCD/DoTA |
| Rules for classes we can actually see in the samples | VLM verifier |
| Accident / near_miss from trajectory heuristics (box overlap + sudden stop, hard decel + low TTC) | Learned scene layout (only if a day is spare — otherwise hand-draw zones) |
| Part B: TTC + decel → logistic → EMA + hysteresis | Webcam / live-stream demo |
| Static site + HF Space Gradio demo | Big ablation grid (do 2 cheap ones from cached runs) |
| | Pitch deck — **after** elimination results, not now |

**Class policy:** predict only classes we've seen or can verify on the samples. `fire_smoke`, `road_obstacle`, `illegal_u_turn`, `illegal_turn` stay **off** unless we see one in the samples and the rule nails it — a wrong guess adds a 0 to the macro average.

**Team split**
- **P1 — CV/pipeline:** env, detector+tracker, track cache, Part B, timing, Docker.
- **P2 — Rules & eval:** annotation, scene zones, rule engines, segment post-processing, boundary tuning.
- **P3 — Web:** site, HF Space demo, EDA charts, annotated videos, report/team pages.

---

## 6. Timeline & checklist

### Day 1 — Wed 23 Sep: foundations (goal: a valid submission exists tonight)
- [x] Read task PDF, starter kit, `evaluate.py`, `run_submission.py` (findings in §1)
- [ ] **P1:** install ffmpeg (`winget install Gyan.FFmpeg`); `ffprobe` each sample (resolution/bitrate decide the decode budget); cut 2-min clips with `-c copy` into `samples/clips/` for labeling + demo tests (dev only — the solution always reads full files)
- [ ] **All:** watch all 4 samples; list what events actually occur; ask channel for `camera.md`
- [ ] **All (split videos):** annotate samples → `dev/labels.json` in GT format, following start/end conventions literally
- [ ] **P1:** Python 3.10/3.11 venv; torch+CUDA, ultralytics, opencv. Public repo in the required layout, starter files **unchanged**
- [ ] **P1:** detector+tracker over each sample → cache tracks to `cache/<video>.parquet` (rules then iterate in seconds)
- [ ] **P1:** measure decode time + detector fps; pick stride so A+B ≤ 1× duration on the laptop (T4 will be faster)
- [ ] **P2:** draw scene zones on a reference frame → `src/scene.yaml` (lanes + directions, stop line, crosswalk, carriageway, signal ROI)
- [ ] **P3:** site skeleton live (Vercel/GitHub Pages) + HF Space "hello world" Gradio app; EDA metadata table
- [ ] **Tag `v0`**: stub `solution.py` passes `run_submission.py` + `evaluate.py --validate-only`

### Day 2 — Thu 24 Sep: core detection working
- [ ] **P2:** rules for the classes present in samples (likely `stopped_vehicle`, `congestion`, `wrong_way`, `jaywalking`, `red_light`/`stop_line` if the signal is visible, `failure_to_yield`)
- [ ] **P2:** segment post-processing: merge gaps < 1 s, drop blips < 0.5 s, union same-class overlaps, backdate `stopped_vehicle` start
- [ ] **P2:** `scripts/eval_dev.sh` → Score A + per-class table on our labels
- [ ] **P1:** Part B v1 in `RiskEstimator` (own causal tracker, stride 3–5, cached score on skipped frames)
- [ ] **P1:** accident / near_miss heuristics (runs offline in Part A, can reuse Part B's curve)
- [ ] **P3:** demo wired to the real pipeline: upload ≤ 2 min mp4 → progress → timeline + annotated video + risk curve
- [ ] **P3:** EDA: object counts over time, motion heatmap, trajectories + lane directions, density
- [ ] **Tag `v1`** — first real submission

### Day 3 — Fri 25 Sep: accuracy + hardening
- [ ] **P2:** per-class boundary tuning at IoU 0.7; review every FP/FN on the samples
- [ ] **P1:** Part B calibration: baseline ~0 in calm traffic, crosses 0.5 only on real danger; hysteresis
- [ ] **P1:** time on a Kaggle/Colab T4; Dockerfile; **clean-machine test** (`--network none`); `weights/download.sh` from HF Hub/GitHub Release; fixed seeds, run twice and diff the outputs
- [ ] **P1:** 2 cheap ablations from cache: detector n vs s, stride 2 vs 5 → numbers for the site
- [ ] **P3:** results page: every sample annotated, clickable timeline that seeks the video, risk curve, per-class examples, failure cases
- [ ] **P3:** approach page with pipeline diagram; team page with roles + links
- [ ] **Tag `v2`**

### Day 4 — Sat 26 Sep: freeze and ship (feature freeze at 14:00)
- [ ] Regenerate `predictions_samples.json` from the tagged code; matches the repo output
- [ ] README: install/run, weights, approach, datasets + licences, learned vs rules, seeds, team roles, timing table, attribution (Ultralytics AGPL, ByteTrack)
- [ ] Final clean-machine run from a fresh clone; `evaluate.py --validate-only` passes
- [ ] Report page: what worked / didn't / next; phone check; test the demo with a weird upload (10 s clip, no events)
- [ ] Set up a GitHub Action that pings the HF Space so it doesn't sleep during judging
- [ ] Tag final commit → submit repo link + commit hash + website link; fill `submission.txt`

### Sun 27 Sep — buffer only
Nothing new. Fix only what's broken. Submit early.

### After elimination — pitch for the second stage
- [ ] Deck (~8 slides), 2-min backup demo video, impact numbers from our runs, business slide, prepared judge Q&A, 2 timed rehearsals

---

## 7. Rules we must not break
- No closed/paid API at inference. Open weights only, shipped in package.
- `run_submission.py` / `evaluate.py` unchanged.
- `RiskEstimator` never opens the video; never uses future frames or Part A results (FAQ confirms: A may use B, never B→A).
- Fixed seeds, deterministic output.
- No extra footage from the same camera, even if found.
- Attribute every reused repo and dataset in README.
- No personal data from footage beyond what the video shows (blur faces/plates on the site to be safe).
