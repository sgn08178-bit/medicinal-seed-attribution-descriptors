#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_mask_224, prepare_test_dataframe, stem_to_gradcam_path
from src.descriptor_maps import descriptor_category
from src.io_utils import load_yaml
from src.spatial_association import spearman_foreground
from src.stats_utils import fdr_bh, one_sample_ttest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - mn) / (mx - mn), 0, 1).astype(np.float32)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "association_gradcam_convnext"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_test_dataframe(cfg["test_csv"], cfg["image_root"], cfg["mask_root"])
    desc_meta = pd.read_csv(run_dir / "descriptor_maps/descriptor_generation_metadata.csv")
    descriptors = sorted(desc_meta["descriptor"].unique())
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grad-CAM association"):
        stem = str(row["stem"])
        fg = load_mask_224(row["maskpath"], int(cfg["img_size"]))
        cam = minmax(np.load(stem_to_gradcam_path(stem, cfg["convnext_gradcam_final_dir"])))
        for d in descriptors:
            desc = np.load(run_dir / "descriptor_maps/raw_npy" / stem / f"{d}.npy").astype(np.float32)
            rows.append({"stem": stem, "class": row["class"], "descriptor": d, "category": descriptor_category(d), "spearman_r": spearman_foreground(cam, desc, fg)})
    rec = pd.DataFrame(rows)
    rec.to_csv(out_dir / "image_level_correlation.csv", index=False)
    desc = rec.groupby(["descriptor", "category"])["spearman_r"].agg(mean_spearman_r="mean", sd="std", median="median", n="count").reset_index()
    tests = []
    for (d, c), sub in rec.groupby(["descriptor", "category"]):
        t, p = one_sample_ttest(sub["spearman_r"])
        tests.append({"descriptor": d, "category": c, "t_statistic": t, "p_value": p})
    tests = pd.DataFrame(tests)
    tests["fdr_adjusted_p_value"] = fdr_bh(tests["p_value"].tolist())
    tests["significant_fdr_0.05"] = tests["fdr_adjusted_p_value"] < 0.05
    desc.merge(tests, on=["descriptor", "category"], how="left").sort_values("mean_spearman_r", ascending=False).to_csv(out_dir / "descriptor_summary.csv", index=False)
    rec.groupby(["class", "descriptor", "category"])["spearman_r"].agg(mean_spearman_r="mean", sd="std", n="count").reset_index().to_csv(out_dir / "classwise_summary.csv", index=False)

    avg_rows = []
    for cls in cfg["class_order"]:
        sub = df[df["class"] == cls]
        if sub.empty:
            continue
        cam_sum = None
        fg_sum = None
        desc_sum = {d: None for d in descriptors}
        n = 0
        for _, row in sub.iterrows():
            stem = str(row["stem"])
            fg = load_mask_224(row["maskpath"], int(cfg["img_size"]))
            cam = minmax(np.load(stem_to_gradcam_path(stem, cfg["convnext_gradcam_final_dir"])))
            cam_sum = cam.astype(np.float64) if cam_sum is None else cam_sum + cam
            fg_sum = fg.astype(np.float64) if fg_sum is None else fg_sum + fg
            for d in descriptors:
                arr = np.load(run_dir / "descriptor_maps/raw_npy" / stem / f"{d}.npy").astype(np.float32)
                desc_sum[d] = arr.astype(np.float64) if desc_sum[d] is None else desc_sum[d] + arr
            n += 1
        avg_cam = (cam_sum / n).astype(np.float32)
        avg_fg = (fg_sum > 0).astype(bool)
        for d in descriptors:
            avg_desc = (desc_sum[d] / n).astype(np.float32)
            avg_rows.append({
                "class": cls,
                "descriptor": d,
                "category": descriptor_category(d),
                "spearman_r": spearman_foreground(avg_cam, avg_desc, avg_fg),
                "n_images": n,
            })
    pd.DataFrame(avg_rows).to_csv(out_dir / "class_average_correlation.csv", index=False)
    print(f"Grad-CAM association complete: {len(rec)} rows.")


if __name__ == "__main__":
    main()
