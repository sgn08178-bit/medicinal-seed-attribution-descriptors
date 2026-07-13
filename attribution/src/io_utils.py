from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(output_dir: str | Path, run_name: str | None, overwrite: bool = False) -> Path:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    name = run_name or f"stage2_attribution_{timestamp()}"
    run_dir = base / name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory exists: {run_dir}. Use --overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def require_file(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p

