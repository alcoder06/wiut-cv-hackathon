"""Load configs/pipeline.yaml once and fix every random seed."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


class Cfg(dict):
    """dict with attribute access, recursively (cfg.rules.congestion.min_len_sec)."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache(maxsize=None)
def load_config(path: str | None = None) -> Cfg:
    """configs/pipeline.yaml, with configs/tuned.yaml (written by scripts/tune.py) on top.
    Delete tuned.yaml to go back to the hand-set values."""
    path = Path(path) if path else ROOT / "configs" / "pipeline.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tuned = path.parent / "tuned.yaml"
    if tuned.exists():
        cfg = deep_merge(cfg, yaml.safe_load(tuned.read_text(encoding="utf-8")) or {})
    return Cfg(cfg)


def resolve(rel: str) -> Path:
    """Paths in the config are relative to the repo root, not the working directory."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
