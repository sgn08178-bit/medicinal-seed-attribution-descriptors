#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def latest_run_dir() -> Path:
    runs = sorted([p for p in (ROOT / "runs").glob("stage2_attribution_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No Stage 2 run directory under {ROOT / 'runs'}")
    return runs[-1]


def minmax_foreground(arr: np.ndarray, mask: np.ndarray, low: float = 1, high: float = 99, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    fg = mask.astype(bool)
    out = np.zeros_like(arr, dtype=np.float32)
    if fg.sum() == 0:
        return out
    vals = arr[fg]
    lo, hi = np.percentile(vals, [low, high])
    clipped = np.clip(arr, lo, hi)
    fg_vals = clipped[fg]
    mn, mx = float(np.nanmin(fg_vals)), float(np.nanmax(fg_vals))
    if mx - mn < eps:
        return out
    out[fg] = np.clip((clipped[fg] - mn) / (mx - mn), 0, 1)
    return out


def save_overlay(rgb: np.ndarray, heat01: np.ndarray, mask: np.ndarray, out_path: Path, alpha: float = 0.40, cmap_name: str = "turbo") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgb, 0, 1)
    heat = plt.get_cmap(cmap_name)(np.clip(heat01, 0, 1))[..., :3]
    m = mask.astype(bool)[..., None]
    overlay = np.where(m, (1 - alpha) * rgb + alpha * heat, rgb)
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(out_path)


def save_heatmap(heat01: np.ndarray, mask: np.ndarray, out_path: Path, cmap_name: str = "turbo") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heat = plt.get_cmap(cmap_name)(np.clip(heat01, 0, 1))[..., :3]
    heat = np.where(mask.astype(bool)[..., None], heat, 0.0)
    Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).save(out_path)


def load_rgb_mask(row: pd.Series, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = Image.open(row["filepath"]).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
    mask = Image.open(row["maskpath"]).convert("L").resize((img_size, img_size), Image.Resampling.NEAREST)
    return np.asarray(rgb).astype(np.float32) / 255.0, (np.asarray(mask) > 0)


def pick_representatives(meta: pd.DataFrame, class_order: list[str]) -> pd.DataFrame:
    rows = []
    for cls in class_order:
        cdf = meta[meta["true_class"] == cls].copy()
        if "confidence" in cdf.columns:
            cdf = cdf.sort_values("confidence", ascending=False)
        rows.append(cdf.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def make_grid(rep: pd.DataFrame, out_dir: Path, mode: str, class_order: list[str]) -> None:
    fig, axes = plt.subplots(len(rep), 3, figsize=(7.2, 2.3 * len(rep)), dpi=300)
    if len(rep) == 1:
        axes = axes[None, :]
    for i, row in rep.iterrows():
        stem = row["stem"]
        cls = row["true_class"]
        img = Image.open(row["filepath"]).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        heat = Image.open(out_dir / f"{mode}_heatmap_png" / f"{stem}_{mode}_ig_heatmap.png").convert("RGB")
        overlay = Image.open(out_dir / f"{mode}_overlay_png" / f"{stem}_{mode}_ig_overlay.png").convert("RGB")
        for ax, im, title in zip(axes[i], [img, heat, overlay], ["Input", f"{mode} heatmap", f"{mode} overlay"]):
            ax.imshow(im)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(cls, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"representative_grid_{mode}.png", dpi=300)
    fig.savefig(out_dir / f"representative_grid_{mode}.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    cfg = load_yaml(run_dir / "config.yaml")
    ig_dir = run_dir / "01_ig_convnext" / "convnext_small"
    out_dir = ig_dir / "visualization_check"
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_dir}. Use --overwrite to replace visualization-check outputs.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["positive_overlay_png", "absolute_overlay_png", "positive_heatmap_png", "absolute_heatmap_png"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(ig_dir / "attribution_metadata.csv")
    input_df = pd.read_csv(run_dir / "inputs" / "test.csv")[["stem", "filepath", "maskpath", "class", "label"]]
    meta = meta.drop(columns=[c for c in ["filepath", "maskpath"] if c in meta.columns]).merge(input_df, on="stem", how="left")
    if meta["filepath"].isna().any() or meta["maskpath"].isna().any():
        missing = meta[meta["filepath"].isna() | meta["maskpath"].isna()]["stem"].tolist()[:10]
        raise RuntimeError(f"Could not resolve filepath/maskpath for stems: {missing}")
    settings = json.loads((ig_dir / "ig_settings.json").read_text())
    raw_dir = ig_dir / "raw_npy"
    stats = []
    mode_diff_max = []
    for _, row in meta.iterrows():
        stem = row["stem"]
        raw = np.load(raw_dir / f"{stem}.npy")
        rgb, mask = load_rgb_mask(row, int(cfg["img_size"]))
        positive_raw = np.maximum(raw, 0)
        absolute_raw = np.abs(raw)
        positive = minmax_foreground(positive_raw, mask, 1, 99)
        absolute = minmax_foreground(absolute_raw, mask, 1, 99)
        mode_diff_max.append(float(np.max(np.abs(positive - absolute))))
        save_overlay(rgb, positive, mask, out_dir / "positive_overlay_png" / f"{stem}_positive_ig_overlay.png")
        save_overlay(rgb, absolute, mask, out_dir / "absolute_overlay_png" / f"{stem}_absolute_ig_overlay.png")
        save_heatmap(positive, mask, out_dir / "positive_heatmap_png" / f"{stem}_positive_ig_heatmap.png")
        save_heatmap(absolute, mask, out_dir / "absolute_heatmap_png" / f"{stem}_absolute_ig_heatmap.png")
        finite = np.isfinite(raw)
        stats.append(
            {
                "stem": stem,
                "true_class": row["true_class"],
                "raw_shape": "x".join(map(str, raw.shape)),
                "raw_ndim": int(raw.ndim),
                "raw_min": float(np.nanmin(raw)),
                "raw_max": float(np.nanmax(raw)),
                "raw_mean": float(np.nanmean(raw)),
                "raw_std": float(np.nanstd(raw)),
                "zero_ratio": float(np.mean(raw == 0)),
                "nan_count": int(np.isnan(raw).sum()),
                "finite_count": int(finite.sum()),
                "fg_raw_min": float(np.nanmin(raw[mask])),
                "fg_raw_max": float(np.nanmax(raw[mask])),
                "fg_raw_mean": float(np.nanmean(raw[mask])),
                "fg_raw_std": float(np.nanstd(raw[mask])),
                "positive_abs_max_diff_after_norm": mode_diff_max[-1],
                "near_constant_fg": bool(np.nanstd(raw[mask]) < 1e-8),
                "all_zero": bool(np.all(raw == 0)),
            }
        )
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / "ig_value_stats.csv", index=False)
    reps = pick_representatives(meta, cfg["class_order"])
    make_grid(reps, out_dir, "positive", cfg["class_order"])
    make_grid(reps, out_dir, "absolute", cfg["class_order"])

    all_zero = int(stats_df["all_zero"].sum())
    near_constant = int(stats_df["near_constant_fg"].sum())
    shape_counts = stats_df["raw_shape"].value_counts().to_dict()
    raw_min = float(stats_df["raw_min"].min())
    raw_max = float(stats_df["raw_max"].max())
    diff_max = float(max(mode_diff_max))
    if settings.get("attribution_2d") == "abs_sum_channels":
        aggregation_note = "Stored raw IG maps are already 2D absolute channel-sum maps; original signed RGB-channel attribution was not retained."
        recommendation = "absolute"
        stability = "positive and absolute visualizations are effectively identical because the stored raw maps are non-negative abs-sum maps."
    else:
        aggregation_note = f"Stored attribution_2d setting: {settings.get('attribution_2d')}"
        recommendation = "absolute"
        stability = "absolute mode is generally more stable for magnitude visualization; positive mode can be useful for signed maps."
    report = f"""# IG Visualization Check Report

## Input

- Run directory: `{run_dir}`
- IG directory: `{ig_dir}`
- Raw IG directory: `{raw_dir}`
- Output directory: `{out_dir}`

## Raw IG Map Summary

- Number of raw maps: {len(stats_df)}
- Raw map shape counts: `{shape_counts}`
- Raw IG global min: {raw_min}
- Raw IG global max: {raw_max}
- All-zero maps: {all_zero}
- Near-constant foreground maps: {near_constant}
- Maximum positive-vs-absolute normalized difference: {diff_max}

## Channel Aggregation

- Recorded attribution_2d setting: `{settings.get('attribution_2d')}`
- {aggregation_note}
- Because signed RGB-channel attribution was not retained, true channel-sum ReLU visualization cannot be reconstructed from the existing raw npy files without recomputing IG.

## Visualization Settings

- Foreground mask resized to 224 x 224 with nearest-neighbor interpolation.
- Percentile clipping: 1 and 99, computed from foreground pixels only.
- Min-max normalization: computed from foreground pixels only after clipping.
- Foreground outside heatmap: removed.
- Overlay: original RGB image, turbo colormap, alpha 0.40.
- Raw IG npy files were not modified.

## Positive vs Absolute Mode

- Positive mode: ReLU applied to the stored raw 2D map.
- Absolute mode: absolute value applied to the stored raw 2D map.
- Visual stability: {stability}
- Recommended manuscript visualization mode: `{recommendation}`.

## Interpretation Notes

- No all-zero or near-constant foreground maps were detected if the counts above are zero.
- If signed channel attribution is needed to assess positive/negative cancellation, IG must be recomputed or cached before 2D aggregation.
- The current stored raw maps are appropriate for magnitude-based visualization, but not for signed attribution interpretation.
"""
    (out_dir / "visualization_check_report.md").write_text(report, encoding="utf-8")
    print(out_dir)
    print(stats_df[["raw_shape", "raw_min", "raw_max", "raw_mean", "raw_std", "zero_ratio", "nan_count"]].describe(include="all").to_string())
    print(report)


if __name__ == "__main__":
    main()
