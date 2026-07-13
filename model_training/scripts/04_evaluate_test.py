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
from src.metrics import predict, save_classification_outputs
from src.models import build_model, canonical_model_name
from src.train_utils import criterion_from_cfg, make_loader, run_epoch, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing test metrics and prediction files.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_model_cfg(load_config(args.config), args.model_name)
    cfg["model_name"] = canonical_model_name(cfg["model_name"])
    model_dir = Path(args.output_dir) / cfg["model_name"]
    save_config(cfg, model_dir / "config.yaml")
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(model_dir / "test.csv" if (model_dir / "test.csv").exists() else Path(args.output_dir) / "test.csv")
    test_loader = make_loader(test_df, cfg, train=False)
    model = build_model(cfg["model_name"], int(cfg["num_classes"]), pretrained=False).to(device)
    checkpoint = model_dir / "checkpoints" / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Final checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    protected = [
        model_dir / "test_predictions.csv",
        model_dir / "classification_report.csv",
        model_dir / "per_class_metrics.csv",
        model_dir / "confusion_matrix_counts.csv",
        model_dir / "confusion_matrix_normalized.csv",
        model_dir / "results.json",
    ]
    existing = [p for p in protected if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Existing evaluation outputs found. Refusing to overwrite without --overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )
    test_loss, _, _, _ = run_epoch(model, test_loader, criterion_from_cfg(cfg), device)
    probs, preds, trues = predict(model, test_loader, device)
    metrics = save_classification_outputs(test_df, probs, preds, trues, model_dir)

    cv_path = model_dir / "cv_results.csv"
    final_summary_path = model_dir / "final_training_summary.json"
    results = {
        "model_name": cfg["model_name"],
        "checkpoint": str(checkpoint),
        "test_loss": float(test_loss),
        **metrics,
    }
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        results.update(
            {
                "cv_accuracy_mean": float(cv["accuracy"].mean()),
                "cv_accuracy_std": float(cv["accuracy"].std(ddof=1)),
                "cv_macro_f1_mean": float(cv["f1"].mean()),
                "cv_macro_f1_std": float(cv["f1"].std(ddof=1)),
            }
        )
    if final_summary_path.exists():
        results["final_training"] = json.loads(final_summary_path.read_text(encoding="utf-8"))
    (model_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
