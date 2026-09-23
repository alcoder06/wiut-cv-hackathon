# Progress — WIUT Hackathon 2026, CV track (elimination)

**Deadline:** Sun 27 Sep 2026 · **Last updated:** Thu 24 Sep 2026
**Repo:** local git in `D:\Projects\hakathon` (not on GitHub yet; required before submission)

---

## Where we are

| Area | Status |
|---|---|
| Submission package runs end to end | ✅ The organizers' `run_submission.py` works with our `solution.py`; `evaluate.py --validate-only` says VALID |
| Part A: event detection | ✅ Working on the real video; ⏳ being tuned (needs reviewed labels) |
| Part B: accident warning | ✅ Tuned for few false alarms; ⚠️ not yet tested on a real crash |
| Time budget | ✅ 1.45× video length on an 8-core machine (limit 3×) |
| Scene zones | ✅ 4 zebra crossings drawn · ❌ stop lines / traffic lights not yet |
| Dev labels | ⏳ **Waiting on review** of 47 detections (see "Your next step") |
| Website | ❌ Not started |
| GitHub | ❌ Not created (needed on the last day) |

## Your next step
Review the system's detections (30–40 min, no rulebook needed). Full guide: `dev/REVIEW.md`.
1. Open `scripts/label_tool.html` in Chrome → **Choose File** → `samples/C3897.MP4`
2. **Import JSON** → `dev/review/C3897_to_review.json`
3. Each row: **▶**, watch. Real → keep. Not real or unsure → **✕**.
4. **Export labels JSON** → move the file into `dev/labels/` → tell Claude "labels are ready".

---

## The sample video
`samples/C3897.MP4`: 3840×2160 (4K), 29.97 fps, 5 min 18 s, H.264, 5.8 GB.
A big signalised intersection filmed from above: a divided boulevard, 4 zebra crossings, buses,
heavy pedestrian traffic. Two traffic lights are readable at 4K (centre pole, left pole).
No crash in this video.

## How the system works (short)
1. **YOLO11s** (pretrained, not trained by us) finds cars, buses, trucks, bikes and people at ~8 fps.
2. **ByteTrack** links detections over time into tracks ("car #17 moved from here to here").
3. **Speed** is measured in box sizes per second, so the same thresholds work near and far from the camera.
4. **Road layout** is learned from where vehicles drive, and which way, plus hand-drawn crossings.
5. **One rule per event type** turns tracks into `[start, end, class]`.
6. **Part B** uses its own tracker on past frames only and scores how soon two road users would collide.

## Results on the sample video (before labels, so unverified)
Part A found 47 events: failure_to_yield 18, jaywalking 10, near_miss 7, stopped_vehicle 6,
congestion 5, wrong_way 1, accident 0.

Part B: 7 alarms in 5 min, risk ≥ 0.5 for 2.1% of the time (started at 35 alarms and 7.9%).

Speed on 8 CPU cores (their machine has 8): Part A 197 s + Part B 263 s = **460 s for a 318 s video (1.45×)**.

## Problems found and fixed
- **Fake accidents** (pedestrians walking together, cars rolling into queues): a crash now needs a vehicle still moving fast at the moment of contact.
- **Every red-light queue counted as "stopped vehicle":** a stop now only counts if the traffic around it keeps moving.
- **The road area covered only a thin strip:** the grid is now sized to the video resolution.
- **Part B false alarms from cars passing in the next lane or brushing past queues:** filtered by lane offset and closing speed.
- **Failure to yield fired whenever anyone was anywhere on a 7-lane crossing:** the pedestrian must now be near the car.
- **Too slow at 4K:** decode moved to a background thread, detector input set to 960 px (787 s → 460 s on 8 cores).
- **Rules were slow:** the crossing check now uses a mask (50 s → 6 s).
- **Safety valves:** each part caps its own time, so a slow machine loses some events instead of scoring zero.
- **Laptop on battery made everything 2–3× slower.** Keep it plugged in.

## Still open (in priority order)
1. **Labels:** the review above, then tune every rule against them with `evaluate.py`.
2. **Traffic lights:** work out which light controls which lanes and draw the stop lines. This switches on `red_light` / `stop_line`.
3. **A real crash clip** (public CCTV dataset) to check that Part B and the accident rule fire on a real crash.
4. **Website** (25% of the elimination score): team, approach, EDA, annotated videos, live upload demo, report.
5. **GitHub repo, final clean-machine test, README team section, tag and submit.**

## Plan for the remaining days
- **Thu 24:** labels reviewed → tune Part A; traffic lights; crash clip check; start the website.
- **Fri 25:** website content (EDA charts, annotated videos, timeline) + live demo; GitHub repo.
- **Sat 26:** freeze at 14:00; clean-machine test; README; final `predictions_samples.json`; tag; submit.
- **Sun 27:** buffer only.

---

## Useful commands (PowerShell, from `D:\Projects\hakathon`)
```powershell
$env:TRAFFIC_CACHE="cache"                                     # reuse saved tracks (fast)
.venv\Scripts\python scripts\make_review.py samples\C3897.MP4    # detections -> review file
.venv\Scripts\python scripts\render.py samples\C3897.MP4 --start 0 --end 60   # annotated video in out\
.venv\Scripts\python scripts\replay_risk.py samples\C3897.MP4    # Part B stats in seconds
.venv\Scripts\python -m pytest -q tests                          # 12 rule/risk tests
.venv\Scripts\python run_submission.py --videos samples --out predictions_samples.json   # official run (~8 min)
.venv\Scripts\python evaluate.py --pred predictions_samples.json --gt dev\labels.json --per-video   # once labels exist
```

## Key files
| File | What |
|---|---|
| `solution.py` | The interface the organizers call |
| `src/` | Pipeline code (detector, tracking, scene, rules in `src/events/`, Part B in `risk.py`) |
| `configs/pipeline.yaml` | Every threshold |
| `configs/scene.yaml` | Hand-drawn zones (crossings) |
| `PLAN.md` | Overall plan and scoring analysis |
| `dev/REVIEW.md` | How to review detections |
| `dev/LABELING.md` | Full labeling conventions |
