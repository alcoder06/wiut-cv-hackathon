"""Save one frame as a PNG for drawing scene zones in scripts/zone_tool.html.

    python scripts/export_frame.py samples/video1.mp4 120      # frame at t = 120 s
"""
import sys
from pathlib import Path

import cv2

video, t = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
ok, frame = cap.read()
if not ok:
    sys.exit(f"could not read {video} at {t}s")
out = Path("dev") / f"{Path(video).stem}_{int(t)}s.png"
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), frame)
print(out)
