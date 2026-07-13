#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_model_cfg, load_config, save_config
from src.models import build_model, canonical_model_name
from src.train_utils import criterion_from_cfg, make_loader, run_epoch, seed_everything, train_with_early_stopping


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
    train_csv = split_root / "train.csv"
    test_csv = split_root / "test.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Run 01_make_splits.py first. Missing: {train_csv}")
    pd.read_csv(train_csv).to_csv(model_dir / "train.csv", index=False)
    pd.read_csv(test_csv).to_csv(model_dir / "test.csv", index=False)

    rows = []
    for fold in range(1, int(cfg["n_folds"]) + 1):
        fold_train_csv = split_root / "cv_splits" / f"fold{fold}_train.csv"
        fold_val_csv = split_root / "cv_splits" / f"fold{fold}_val.csv"
        fold_out = model_dir / "cv_splits"
        fold_out.mkdir(exist_ok=True)
        fold_train = pd.read_csv(fold_train_csv)
        fold_val = pd.read_csv(fold_val_csv)
        fold_train.to_csv(fold_out / f"fold{fold}_train.csv", index=False)
        fold_val.to_csv(fold_out / f"fold{fold}_val.csv", index=False)

        seed_everything(int(cfg["seed"]) + fold)
        model = build_model(cfg["model_name"], int(cfg["num_classes"]), bool(cfg["pretrained"])).to(device)
        tr_loader = make_loader(fold_train, cfg, train=True)
        va_loader = make_loader(fold_val, cfg, train=False)
        checkpoint = model_dir / "checkpoints" / f"fold{fold}_best.pth"
        history = model_dir / f"fold{fold}_training_history.csv"
        log = model_dir / "logs" / f"fold{fold}.jsonl"
        meta = train_with_early_stopping(model, tr_loader, va_loader, cfg, checkpoint, history, log, device, overwrite=args.overwrite)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        val_loss, val_acc, preds, trues = run_epoch(model, va_loader, criterion_from_cfg(cfg), device)
        rows.append(
            {
                "fold": fold,
                "best_epoch": meta["best_epoch"],
                "best_val_loss": meta["best_val_loss"],
                "eval_val_loss": float(val_loss),
                "accuracy": float(val_acc),
                "precision": float(precision_score(trues, preds, average="macro", zero_division=0)),
                "recall": float(recall_score(trues, preds, average="macro", zero_division=0)),
                "f1": float(f1_score(trues, preds, average="macro", zero_division=0)),
                "checkpoint": str(checkpoint),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv = pd.DataFrame(rows)
    cv_path = model_dir / "cv_results.csv"
    if cv_path.exists() and not args.overwrite:
        raise FileExistsError(f"Existing CV results found. Refusing to overwrite without --overwrite: {cv_path}")
    cv.to_csv(cv_path, index=False)
    print(json.dumps({"model_name": cfg["model_name"], "cv_results": str(model_dir / "cv_results.csv")}, indent=2))


if __name__ == "__main__":
    main()
