#!/usr/bin/env sh
# Fetch model weights once, with internet, before the offline evaluation run.
# The weights are also committed to this folder; this script is the fallback.
set -eu
cd "$(dirname "$0")"
if [ ! -f yolo11s.pt ]; then
  curl -fL -o yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
fi
echo "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5  yolo11s.pt" | sha256sum -c -
