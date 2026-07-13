#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=False, help="Accepted for interface consistency; not required.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models", nargs="*", help="Explicit model directories to collect. Overrides config model_name.")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting model_comparison_summary.csv.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if args.models:
        models = args.models
    elif args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        names = cfg.get("model_name", [])
        models = names if isinstance(names, list) else [names]
    else:
        models = [p.name for p in out_dir.iterdir() if p.is_dir()]
    rows = []
    for model_name in models:
        result_path = out_dir / model_name / "results.json"
        if not result_path.exists():
            print(f"Skipping missing results: {result_path}")
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model_name": data.get("model_name", result_path.parent.name),
                "cv_accuracy_mean": data.get("cv_accuracy_mean"),
                "cv_accuracy_std": data.get("cv_accuracy_std"),
                "cv_macro_f1_mean": data.get("cv_macro_f1_mean"),
                "cv_macro_f1_std": data.get("cv_macro_f1_std"),
                "test_loss": data.get("test_loss"),
                "test_accuracy": data.get("test_accuracy"),
                "test_precision_macro": data.get("test_precision_macro"),
                "test_recall_macro": data.get("test_recall_macro"),
                "test_macro_f1": data.get("test_macro_f1"),
                "checkpoint": data.get("checkpoint"),
                "run_dir": str(result_path.parent),
            }
        )
    if not rows:
        raise RuntimeError(f"No */results.json files found under {out_dir}")
    df = pd.DataFrame(rows).sort_values(["test_macro_f1", "test_accuracy"], ascending=False)
    path = out_dir / "model_comparison_summary.csv"
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Existing model comparison summary found. Refusing to overwrite without --overwrite: {path}")
    df.to_csv(path, index=False)
    print(path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
