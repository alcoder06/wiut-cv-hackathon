# Checking the system's detections (no rulebook needed)

The system has already found possible events. Your job is only to answer
**"is this really that event?"** for each one. About 30–40 minutes for one video.

## Setup (once)
1. Double-click `scripts/label_tool.html` (opens in Chrome).
2. Click **Choose File** and pick `samples/C3897.MP4`.
3. Click **Import JSON** and pick `dev/review/C3897_to_review.json`.
   A list of detections appears on the right, and coloured bars appear under the video.

## For each row in the list
1. Click **▶** on the row. The video jumps to where the event starts.
2. Press **Space** to play. Watch until the end time (shown in the row).
3. Answer the question for that event type (table below):
   - **Yes** → leave it.
   - **No** → click **✕** to delete the row.
   - **Yes, but the times are wrong** → type the right start/end seconds into the row.
     (The clock under the video shows the current second. Use `,` and `.` to step one frame.)

## The question for each event type

| Event | Keep it only if the answer is YES |
|---|---|
| **jaywalking** | Is a person walking **on the road**, **not** on the zebra stripes? (Start = foot on the road, end = back on the pavement.) |
| **failure_to_yield** | Does a car drive **over the zebra stripes** while a person is **on them or stepping onto them**? |
| **stopped_vehicle** | Is a vehicle standing still **10 seconds or more**, and **not** because it's waiting at a red light or in a line of cars? (A broken-down car, or someone parked in a traffic lane = yes. A bus at its bus stop = no.) |
| **congestion** | Are **all lanes** of one direction stuck or crawling, and does the jam **fail to clear when the light turns green**? A normal queue at a red light = no. |
| **near_miss** | Does someone **brake hard or swerve** to avoid hitting a car or person, without touching? Normal slowing down = no. |
| **wrong_way** | Is a car driving **against the direction** of its lane, or in the lanes of oncoming traffic? A normal turn at the intersection = no. |
| **accident** | Do two road users (or a car and an object) **actually touch**? |

Not sure? **Delete it.** An unsure event in the labels does more harm than a missing one.

## If you notice something the system missed (optional, but valuable)
Pick the event type with the number/letter keys (shown on the buttons at the top right), press **S** when
it starts, and **E** when it ends. It's added to the list.

## When done
Click **Export labels JSON**. A file `C3897_labels.json` downloads.
Move it to `dev/labels/` in the project and tell Claude: "labels are ready".
