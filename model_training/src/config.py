from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config file: {path}")
    cfg["config_path"] = str(path)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if k != "config_path"}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)


def get_model_cfg(cfg: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    out = dict(cfg)
    if model_name is not None:
        out["model_name"] = model_name
    if isinstance(out.get("model_name"), list):
        raise ValueError("This script expects a single model_name. Pass --model-name.")
    return out
