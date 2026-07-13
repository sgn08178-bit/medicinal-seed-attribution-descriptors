#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config
from src.dataset import CLASS_ORDER, CLASS_TO_IDX, audit_dataset, build_manifest
from src.splits import make_final_train_val_split, make_train_test_split, save_cv_splits
from src.train_utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite-splits", action="store_true", help="Allow overwriting existing split CSV files.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        out_dir / "manifest.csv",
        out_dir / "train.csv",
        out_dir / "test.csv",
        out_dir / "final_train.csv",
        out_dir / "final_val.csv",
        out_dir / "split_summary.json",
    ]
    existing = [p for p in protected if p.exists()]
    if existing and not args.overwrite_splits:
        raise FileExistsError(
            "Existing split outputs found. Refusing to overwrite without --overwrite-splits:\n"
            + "\n".join(str(p) for p in existing)
        )
    seed_everything(int(cfg["seed"]))
    save_config(cfg, out_dir / "config.yaml")

    manifest = build_manifest(cfg["image_root"], cfg["mask_root"])
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    dataset_audit = audit_dataset(cfg["image_root"], cfg["mask_root"], manifest)
    (out_dir / "dataset_audit.json").write_text(json.dumps(dataset_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    train_df, test_df = make_train_test_split(manifest, float(cfg["test_size"]), int(cfg["seed"]))
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    fold_dir = out_dir / "cv_splits"
    fold_summary = save_cv_splits(train_df, fold_dir, int(cfg["n_folds"]), int(cfg["seed"]))
    fold_summary.to_csv(fold_dir / "cv_split_summary.csv", index=False)

    final_train, final_val = make_final_train_val_split(train_df, float(cfg["final_val_size"]), int(cfg["seed"]))
    final_train.to_csv(out_dir / "final_train.csv", index=False)
    final_val.to_csv(out_dir / "final_val.csv", index=False)

    split_summary = {
        "seed": cfg["seed"],
        "test_size": cfg["test_size"],
        "final_val_size": cfg["final_val_size"],
        "n_folds": cfg["n_folds"],
        "total_n": int(len(manifest)),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "final_train_n": int(len(final_train)),
        "final_val_n": int(len(final_val)),
        "class_order": CLASS_ORDER,
        "class_to_idx": CLASS_TO_IDX,
        "class_counts_total": manifest["class"].value_counts().reindex(CLASS_ORDER).astype(int).to_dict(),
        "class_counts_train": train_df["class"].value_counts().reindex(CLASS_ORDER).astype(int).to_dict(),
        "class_counts_test": test_df["class"].value_counts().reindex(CLASS_ORDER).astype(int).to_dict(),
        "dataset_audit": dataset_audit,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
