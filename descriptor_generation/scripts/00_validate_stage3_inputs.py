#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import prepare_test_dataframe, stem_to_gradcam_path, stem_to_ig_abs_path, stem_to_ig_pos_path, load_mask_224
from src.io_utils import load_yaml, save_json, save_yaml
from src.manifest_utils import build_input_manifest, validate_allowed_stage2_paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def map_ok(path: Path, expected_shape: tuple[int, int]) -> tuple[bool, str, dict]:
    if not path.exists():
        return False, "missing", {}
    try:
        arr = np.load(path)
        meta = {
            "shape": "x".join(map(str, arr.shape)),
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
        }
        ok = tuple(arr.shape) == expected_shape and meta["nan_count"] == 0 and meta["inf_count"] == 0
        return ok, "ok" if ok else "bad_shape_or_nonfinite", meta
    except Exception as exc:
        return False, f"load_error:{exc}", {}


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    validate_allowed_stage2_paths(cfg)
    save_yaml(cfg, run_dir / "config.yaml")
    save_json(build_input_manifest(cfg, run_dir), run_dir / "input_manifest.json")

    df = prepare_test_dataframe(cfg["test_csv"], cfg["image_root"], cfg["mask_root"])
    records = []
    img_size = int(cfg["img_size"])
    for _, row in df.iterrows():
        stem = str(row["stem"])
        rec = {"stem": stem, "class": row["class"]}
        image_exists = Path(row["filepath"]).exists()
        mask_exists = Path(row["maskpath"]).exists()
        rec["image_exists"] = image_exists
        rec["mask_exists"] = mask_exists
        rec["foreground_pixels"] = -1
        rec["mask_ok"] = False
        if mask_exists:
            fg = load_mask_224(row["maskpath"], img_size)
            rec["foreground_pixels"] = int(fg.sum())
            rec["mask_ok"] = bool(fg.sum() > 0)
        for label, path in [
            ("ig_absolute", stem_to_ig_abs_path(stem, cfg["canonical_ig_zero_absolute_dir"])),
            ("ig_positive", stem_to_ig_pos_path(stem, cfg["canonical_ig_zero_positive_dir"])),
            ("gradcam", stem_to_gradcam_path(stem, cfg["convnext_gradcam_final_dir"])),
        ]:
            ok, reason, meta = map_ok(path, (img_size, img_size))
            rec[f"{label}_path"] = str(path)
            rec[f"{label}_ok"] = ok
            rec[f"{label}_status"] = reason
            for k, v in meta.items():
                rec[f"{label}_{k}"] = v
        rec["valid"] = bool(image_exists and mask_exists and rec["mask_ok"] and rec["ig_absolute_ok"] and rec["ig_positive_ok"] and rec["gradcam_ok"])
        records.append(rec)
    report = pd.DataFrame(records)
    out_dir = run_dir / "input_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "input_validation_report.csv", index=False)
    summary = {
        "test_csv": cfg["test_csv"],
        "n_samples": int(len(df)),
        "expected_samples": 225,
        "valid_samples": int(report["valid"].sum()),
        "failed_samples": int((~report["valid"]).sum()),
        "all_valid": bool(len(df) == 225 and report["valid"].all()),
        "blocked_deprecated_paths": False,
        "blocked_candidate_gradcam_paths": False,
    }
    save_json(summary, out_dir / "input_validation_report.json")
    if not summary["all_valid"]:
        print(report.loc[~report["valid"]].head(20).to_string(index=False))
        raise SystemExit("Stage 3 input validation failed. Analysis stopped.")
    print(f"Input validation passed: {summary['valid_samples']}/{summary['n_samples']} samples.")


if __name__ == "__main__":
    main()
