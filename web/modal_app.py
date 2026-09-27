"""The live demo's API on Modal. The website's pages are hosted separately as a static site
(scripts/deploy_space.py --api <url>) and call this API across origins.

    modal deploy web/modal_app.py        # from the repo root; prints the API's URL

At most one container, so every progress poll reaches the container that holds the job
(jobs live in its memory and /tmp). It scales to zero when idle: the first request after a
quiet spell waits ~30-60 s for the container to start and load the model.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")          # for opencv-python (pulled in by ultralytics)
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "web" / "requirements.txt"))
    .env({"YOLO_OFFLINE": "1", "OMP_NUM_THREADS": "2", "TRAFFIC_PROFILE": "demo",
          "DEMO_JOBS_DIR": "/tmp/traffic_jobs", "YOLO_CONFIG_DIR": "/tmp/ultralytics"})
    .add_local_dir(ROOT / "src", "/app/src", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "configs", "/app/configs")
    .add_local_file(ROOT / "weights" / "yolo11n.pt", "/app/weights/yolo11n.pt")
    .add_local_file(ROOT / "web" / "__init__.py", "/app/web/__init__.py")
    .add_local_file(ROOT / "web" / "server.py", "/app/web/server.py")
)

app = modal.App("nexvision-demo", image=image)


@app.function(cpu=2.0, memory=3072, max_containers=1, scaledown_window=300, timeout=900)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    import sys

    sys.path.insert(0, "/app")
    from web.server import app as fastapi_app

    return fastapi_app
