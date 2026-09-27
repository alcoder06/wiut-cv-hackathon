#!/usr/bin/env sh
# Fetch model weights once, with internet, before the offline evaluation run.
# The weights are also committed to this folder; this script is the fallback.
#   yolo11s.pt  the submission's detector (configs/pipeline.yaml)
#   yolo11n.pt  the website's CPU live demo only (configs/demo.yaml)
set -eu
cd "$(dirname "$0")"
if [ ! -f yolo11s.pt ]; then
  curl -fL -o yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
fi
if [ ! -f yolo11n.pt ]; then
  curl -fL -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
fi
sha256sum -c - <<'EOF'
85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5  yolo11s.pt
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1  yolo11n.pt
EOF
