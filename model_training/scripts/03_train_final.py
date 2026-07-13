#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_model_cfg, load_config, save_config
from src.models import build_model, canonical_model_name
from src.train_utils import make_loader, seed_everything, train_with_early_stopping


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing checkpoints, histories, and logs.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_model_cfg(load_config(args.config), args.model_name)
    cfg["model_name"] = canonical_model_name(cfg["model_name"])
    model_dir = Path(args.output_dir) / cfg["model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "checkpoints").mkdir(exist_ok=True)
    (model_dir / "logs").mkdir(exist_ok=True)
    save_config(cfg, model_dir / "config.yaml")
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_root = Path(args.output_dir)
    final_train = pd.read_csv(split_root / "final_train.csv")
    final_val = pd.read_csv(split_root / "final_val.csv")
    final_train.to_csv(model_dir / "final_train.csv", index=False)
    final_val.to_csv(model_dir / "final_val.csv", index=False)
    if (split_root / "train.csv").exists():
        pd.read_csv(split_root / "train.csv").to_csv(model_dir / "train.csv", index=False)
    if (split_root / "test.csv").exists():
        pd.read_csv(split_root / "test.csv").to_csv(model_dir / "test.csv", index=False)

    model = build_model(cfg["model_name"], int(cfg["num_classes"]), bool(cfg["pretrained"])).to(device)
    meta = train_with_early_stopping(
        model,
        make_loader(final_train, cfg, train=True),
        make_loader(final_val, cfg, train=False),
        cfg,
        model_dir / "checkpoints" / "best_model.pth",
        model_dir / "final_training_history.csv",
        model_dir / "logs" / "final_train.jsonl",
        device,
        overwrite=args.overwrite,
    )
    summary_path = model_dir / "final_training_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Existing final training summary found. Refusing to overwrite without --overwrite: {summary_path}")
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"model_name": cfg["model_name"], **meta}, indent=2))


if __name__ == "__main__":
    main()
