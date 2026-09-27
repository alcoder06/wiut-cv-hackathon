"""Publish the website as a Hugging Face Space.

    python scripts/build_site.py --videos samples                           # first: data, images, videos
    python scripts/deploy_space.py <hf-user>/<space> --api <demo-api-url>   # static pages (free tier)
    python scripts/deploy_space.py <hf-user>/<space>                        # pages + demo in one Docker Space

With --api the Space is static (free): only web/static and predictions_samples.json, with the
page pointed at the demo API hosted elsewhere (web/modal_app.py). Without it, the Space runs
web/Dockerfile and serves pages and API together (Docker Spaces need HF PRO). Needs
`hf auth login` once (or HF_TOKEN). Re-running updates the same Space.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CARD = """---
title: NexVision Traffic Event Watch
emoji: 🚦
colorFrom: yellow
colorTo: gray
{sdk}
pinned: false
short_description: Traffic events and accident warning from a fixed CCTV camera
---

Website and live demo for our WIUT Hackathon 2026 CV-track solution.
Source: {repo}
"""


def stage(dst: Path, api: str | None) -> None:
    def copy(rel: str, to: str | None = None) -> None:
        src, out = ROOT / rel, dst / (to or rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
        else:
            shutil.copy2(src, out)

    if api:     # static Space: the pages at the root, pointed at the external demo API
        copy("web/static", ".")         # includes the predictions file build_site.py built from
        page = dst / "index.html"
        html = page.read_text(encoding="utf-8")
        assert '<meta name="demo-api" content="">' in html
        page.write_text(html.replace('<meta name="demo-api" content="">', f'<meta name="demo-api" content="{api}">'),
                        encoding="utf-8")
        return
    copy("src")
    for p in (ROOT / "configs").iterdir():
        if p.suffix in (".yaml", ".npz"):
            copy(f"configs/{p.name}")
    copy("weights/yolo11n.pt")
    copy("web/__init__.py")
    copy("web/server.py")
    copy("web/requirements.txt")
    copy("web/static")
    copy("web/Dockerfile", "Dockerfile")


def check() -> None:
    index = ROOT / "web/static/data/index.json"
    if not (ROOT / "web/static/predictions_samples.json").exists():
        raise SystemExit("web/static/predictions_samples.json missing: run scripts/build_site.py first")
    if not index.exists():
        raise SystemExit("web/static/data/index.json missing: run scripts/build_site.py first")
    for v in json.loads(index.read_text(encoding="utf-8"))["videos"]:
        if not (ROOT / f"web/static/media/{v['stem']}.mp4").exists():
            raise SystemExit(f"annotated video for {v['name']} missing: run scripts/build_site.py without --no-video")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("space", help="<hf-user>/<space-name>")
    ap.add_argument("--repo", default="https://github.com/alcoder06/wiut-cv-hackathon")
    ap.add_argument("--api", help="demo API URL (web/modal_app.py); makes a free static Space")
    args = ap.parse_args()
    check()

    from huggingface_hub import HfApi

    api = HfApi()
    sdk = "static" if args.api else "docker"
    api.create_repo(args.space, repo_type="space", space_sdk=sdk, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp)
        stage(dst, args.api.rstrip("/") if args.api else None)
        card_sdk = "sdk: static" if args.api else "sdk: docker\napp_port: 7860"
        (dst / "README.md").write_text(CARD.format(sdk=card_sdk, repo=args.repo), encoding="utf-8")
        size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 2**20
        print(f"uploading {size:.0f} MB to spaces/{args.space} ...")
        api.upload_folder(folder_path=dst, repo_id=args.space, repo_type="space",
                          commit_message="Deploy website and live demo",
                          delete_patterns=["data/**", "media/**", "src/**", "configs/**", "web/**"])   # drop files removed locally
    host = args.space.replace("/", "-").lower() + (".static.hf.space" if args.api else ".hf.space")
    print(f"done: https://huggingface.co/spaces/{args.space}  (site: https://{host})")


if __name__ == "__main__":
    main()
