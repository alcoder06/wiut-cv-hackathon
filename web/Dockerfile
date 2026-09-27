# The website and live demo, as a Hugging Face Docker Space (CPU). scripts/deploy_space.py
# copies this file to the Space's root next to src/, configs/, weights/ and web/.
#   local test:  docker build -f web/Dockerfile -t site . && docker run -p 7860:7860 site
FROM python:3.11-slim

# libGL/glib for opencv-python (pulled in by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app
COPY web/requirements.txt web/requirements.txt
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r web/requirements.txt
COPY --chown=user . .
USER user

# 2 vCPUs on the free tier: more threads than cores only adds contention
ENV YOLO_OFFLINE=1 OMP_NUM_THREADS=2 TRAFFIC_PROFILE=demo DEMO_JOBS_DIR=/tmp/traffic_jobs \
    YOLO_CONFIG_DIR=/tmp/ultralytics PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "7860"]
