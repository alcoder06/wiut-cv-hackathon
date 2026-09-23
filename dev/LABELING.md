# How we label the sample videos

Goal: our own `dev/labels.json` in the organizers' ground-truth format, so
`python evaluate.py --pred predictions_samples.json --gt dev/labels.json --per-video`
gives a real score after every change. Label what **happens**, not what our model can see.

## Tool
Open `scripts/label_tool.html` in Chrome → pick the video (plays from disk, no upload) →
choose a class (`1`–`9`, `0`, `Q`–`R`) → `S` at the start frame → `E` at the end frame.
Labels autosave in the browser per video; **Export** often and put the file in `dev/labels/`.
Merge everyone's exports: `python scripts/merge_labels.py dev/labels/*.json`.

If the video won't play (HEVC codec), make a small H.264 copy; timestamps stay identical:
`ffmpeg -i samples/X.mp4 -vf scale=1280:-2 -c:v libx264 -preset veryfast -crf 26 -an dev/X_proxy.mp4`
The export is named after the file you opened; rename the key to the original name (`X.mp4`).

## Two passes (boundaries decide IoU 0.7)
1. **Find** — watch at 2–4×. Rough `S`/`E` for anything that might be an event.
2. **Refine** — for each event, jump to it (▶), go to 0.25×, step with `,` `.` frame by frame,
   and fix the start/end in the table to the exact convention below.

Unlabelled time means "no event". Watch everything, including boring stretches: a missed
congestion or stopped vehicle turns our correct prediction into a false positive.

## Conventions (from the task — follow literally)
| Class | Start | End |
|---|---|---|
| accident | first frame contact is visible | all involved stop moving or leave frame |
| near_miss | onset of braking/swerving | road users clear of each other |
| red_light | front of vehicle crosses stop line on red | vehicle leaves intersection or frame |
| wrong_way | vehicle enters opposing lane | back in a correct lane or leaves frame |
| illegal_u_turn | vehicle starts turning | turn completed |
| stopped_vehicle (≥10 s, not queued at signal) | vehicle **stops** (not +10 s) | moves again or removed |
| jaywalking (outside crossing) | pedestrian steps onto road | pedestrian leaves road |
| failure_to_yield | vehicle enters crossing (ped on it / stepping on) | vehicle leaves crossing |
| illegal_turn | vehicle starts turning | turn completed |
| solid_line_crossing | wheel crosses the line | vehicle fully in new lane |
| stop_line (stops past line on red) | vehicle stops | signal turns green |
| congestion (all lanes of a direction) | queue stops moving | queue clears |
| road_obstacle | obstacle appears | obstacle removed |
| fire_smoke | first visible smoke | smoke clears or video ends |

- Two events of the same class at once → **one** segment covering both (the tool flags overlaps).
- Different classes may overlap (wrong_way that causes an accident = two events).
- Event still running at the end of the video → end = video end.

## Team decisions (write every judgement call here so all three label the same way)
- …

## Consistency check (do once, 10 minutes)
Two people label the same 5-minute stretch independently. Treat one as ground truth and the
other as prediction and run evaluate.py on them. That score is roughly the best any model can get
against our labels; big disagreements go into "Team decisions" above.
