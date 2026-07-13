#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_mask_224, prepare_test_dataframe, stem_to_ig_abs_path, stem_to_ig_pos_path
from src.descriptor_maps import descriptor_category
from src.io_utils import load_yaml
from src.spatial_association import spearman_foreground
from src.stats_utils import fdr_bh, one_sample_ttest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mode", choices=["absolute", "positive"], default="absolute")
    return p.parse_args()


def summarize(records: pd.DataFrame, out_dir: Path, class_order: list[str]) -> None:
    desc = (
        records.groupby(["descriptor", "category"])["spearman_r"]
        .agg(mean_spearman_r="mean", sd="std", median="median", n="count")
        .reset_index()
    )
    q1 = records.groupby(["descriptor", "category"])["spearman_r"].quantile(0.25).reset_index(name="q1")
    q3 = records.groupby(["descriptor", "category"])["spearman_r"].quantile(0.75).reset_index(name="q3")
    desc = desc.merge(q1, on=["descriptor", "category"]).merge(q3, on=["descriptor", "category"])
    desc["iqr"] = desc["q3"] - desc["q1"]
    tests = []
    for (descriptor, category), sub in records.groupby(["descriptor", "category"]):
        t, p = one_sample_ttest(sub["spearman_r"])
        tests.append({"descriptor": descriptor, "category": category, "t_statistic": t, "p_value": p})
    tests = pd.DataFrame(tests)
    tests["fdr_adjusted_p_value"] = fdr_bh(tests["p_value"].tolist())
    tests["significant_fdr_0.05"] = tests["fdr_adjusted_p_value"] < 0.05
    desc = desc.merge(tests, on=["descriptor", "category"], how="left")
    desc.sort_values("mean_spearman_r", ascending=False).to_csv(out_dir / "descriptor_summary.csv", index=False)
    tests.sort_values("fdr_adjusted_p_value").to_csv(out_dir / "statistical_tests.csv", index=False)
    desc.sort_values("mean_spearman_r", ascending=False).to_csv(out_dir / "fdr_corrected_results.csv", index=False)

    classwise = (
        records.groupby(["class", "descriptor", "category"])["spearman_r"]
        .agg(mean_spearman_r="mean", sd="std", n="count")
        .reset_index()
    )
    classwise["class"] = pd.Categorical(classwise["class"], categories=class_order, ordered=True)
    classwise.sort_values(["class", "mean_spearman_r"], ascending=[True, False]).to_csv(out_dir / "classwise_summary.csv", index=False)


def class_average(records: pd.DataFrame, cfg: dict, run_dir: Path, out_dir: Path, mode: str) -> None:
    df = prepare_test_dataframe(cfg["test_csv"], cfg["image_root"], cfg["mask_root"])
    class_order = cfg["class_order"]
    descriptors = sorted(records["descriptor"].unique())
    rows = []
    for cls in class_order:
        sub = df[df["class"] == cls]
        if sub.empty:
            continue
        ig_sum = None
        fg_sum = None
        desc_sum = {d: None for d in descriptors}
        n = 0
        for _, row in sub.iterrows():
            stem = str(row["stem"])
            ig_path = stem_to_ig_abs_path(stem, cfg["canonical_ig_zero_absolute_dir"]) if mode == "absolute" else stem_to_ig_pos_path(stem, cfg["canonical_ig_zero_positive_dir"])
            ig = np.load(ig_path).astype(np.float32)
            fg = load_mask_224(row["maskpath"], int(cfg["img_size"]))
            ig_sum = ig.astype(np.float64) if ig_sum is None else ig_sum + ig
            fg_sum = fg.astype(np.float64) if fg_sum is None else fg_sum + fg
            for d in descriptors:
                arr = np.load(run_dir / "descriptor_maps/raw_npy" / stem / f"{d}.npy").astype(np.float32)
                desc_sum[d] = arr.astype(np.float64) if desc_sum[d] is None else desc_sum[d] + arr
            n += 1
        avg_ig = (ig_sum / n).astype(np.float32)
        avg_fg = (fg_sum > 0).astype(bool)
        for d in descriptors:
            avg_desc = (desc_sum[d] / n).astype(np.float32)
            rows.append({
                "class": cls,
                "descriptor": d,
                "category": descriptor_category(d),
                "spearman_r": spearman_foreground(avg_ig, avg_desc, avg_fg),
                "n_images": n,
            })
    pd.DataFrame(rows).to_csv(out_dir / "class_average_correlation.csv", index=False)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = Path(args.run_dir)
    out_dir = run_dir / ("association_ig_zero_absolute" if args.mode == "absolute" else "association_ig_zero_positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_test_dataframe(cfg["test_csv"], cfg["image_root"], cfg["mask_root"])
    desc_meta = pd.read_csv(run_dir / "descriptor_maps/descriptor_generation_metadata.csv")
    descriptors = sorted(desc_meta["descriptor"].unique())
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"IG {args.mode} association"):
        stem = str(row["stem"])
        fg = load_mask_224(row["maskpath"], int(cfg["img_size"]))
        ig_path = stem_to_ig_abs_path(stem, cfg["canonical_ig_zero_absolute_dir"]) if args.mode == "absolute" else stem_to_ig_pos_path(stem, cfg["canonical_ig_zero_positive_dir"])
        ig = np.load(ig_path).astype(np.float32)
        for d in descriptors:
            desc = np.load(run_dir / "descriptor_maps/raw_npy" / stem / f"{d}.npy").astype(np.float32)
            rows.append({
                "stem": stem,
                "class": row["class"],
                "descriptor": d,
                "category": descriptor_category(d),
                "spearman_r": spearman_foreground(ig, desc, fg),
            })
    rec = pd.DataFrame(rows)
    rec.to_csv(out_dir / "image_level_correlation.csv", index=False)
    summarize(rec, out_dir, cfg["class_order"])
    class_average(rec, cfg, run_dir, out_dir, args.mode)
    print(f"IG {args.mode} association complete: {len(rec)} rows.")


if __name__ == "__main__":
    main()
