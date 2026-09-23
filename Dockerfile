# docker build -t team .
# docker run --gpus all --network none -v /data/test:/data/test -v $PWD/out:/out team \
#     python run_submission.py --videos /data/test --out /out/predictions.json
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# libGL/glib are needed by opencv-python (pulled in by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Ultralytics would try to reach the internet for settings/font checks; keep it offline.
ENV YOLO_OFFLINE=1
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/out/predictions.json"]
