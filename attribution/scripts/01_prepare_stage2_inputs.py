#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import prepare_test_dataframe
from src.io_utils import load_yaml, make_run_dir, require_file, save_json, save_yaml
from src.seed_utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    run_dir = make_run_dir(args.output_dir, args.run_name, overwrite=args.overwrite)

    for key in ["image_root", "mask_root", "test_csv"]:
        require_file(cfg[key] if key == "test_csv" else Path(cfg[key]), key)
    test_df = prepare_test_dataframe(cfg["test_csv"], cfg["image_root"], cfg["mask_root"])

    paths = {}
    for model_name, mcfg in cfg["models"].items():
        checkpoint = require_file(mcfg["checkpoint"], f"{model_name} checkpoint")
        stage1_config = require_file(mcfg["stage1_config"], f"{model_name} Stage 1 config")
        stage1_test = Path(cfg["stage1_runs_root"]) / model_name / "test.csv"
        require_file(stage1_test, f"{model_name} Stage 1 test.csv")
        paths[model_name] = {
            "checkpoint": str(checkpoint),
            "stage1_config": str(stage1_config),
            "stage1_test_csv": str(stage1_test),
            "compute_ig": bool(mcfg.get("compute_ig", False)),
            "compute_gradcam": bool(mcfg.get("compute_gradcam", False)),
        }

    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "summaries").mkdir(parents=True, exist_ok=True)
    test_df.to_csv(run_dir / "inputs" / "test.csv", index=False)
    save_yaml(cfg, run_dir / "config.yaml")
    save_json(paths, run_dir / "inputs" / "stage1_model_paths.json")
    save_json(
        {
            "run_dir": str(run_dir),
            "n_test": int(len(test_df)),
            "class_counts": test_df["class"].value_counts().reindex(cfg["class_order"]).fillna(0).astype(int).to_dict(),
            "stage": "Stage 2A attribution map generation",
            "excluded": ["descriptor maps", "attribution-descriptor correlation", "occlusion sensitivity", "mask overlap analysis"],
        },
        run_dir / "metadata" / "attribution_run_metadata.json",
    )
    print(f"Prepared Stage 2 run directory: {run_dir}")
    print(json.dumps({"n_test": len(test_df), "class_counts": test_df["class"].value_counts().to_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

