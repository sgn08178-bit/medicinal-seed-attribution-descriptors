#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_mask_224, load_rgb_224, prepare_test_dataframe, stem_to_ig_abs_path
from src.io_utils import load_yaml
from src.visualization import display_name, save_overlay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def minmax_fg(arr, fg):
    valid = fg.astype(bool)
    vals = arr[valid]
    if vals.size == 0 or np.nanmax(vals) - np.nanmin(vals) < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals)), 0, 1).astype(np.float32)


def make_rep_descriptor_figure(cfg, run_dir: Path, fig_dir: Path):
    df = prepare_test_dataframe(cfg["test_csv"], cfg["image_root"], cfg["mask_root"])
    pred_path = Path(cfg.get("test_predictions_csv", "source_data/stage1/convnext_small_test_predictions.csv"))
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        correct_stems = set(pred.loc[pred["true_label"] == pred["pred_label"], "stem"].astype(str))
    else:
        correct_stems = set(df["stem"].astype(str))
    descs = cfg["representative_descriptors"]
    rows = []
    for cls in cfg["class_order"]:
        sub = df[(df["class"] == cls) & (df["stem"].astype(str).isin(correct_stems))]
        if sub.empty:
            sub = df[df["class"] == cls]
        rows.append(sub.iloc[0])
    fig, axes = plt.subplots(len(rows), 2 + len(descs), figsize=(2.0 * (2 + len(descs)), 2.0 * len(rows)))
    for i, row in enumerate(rows):
        stem = str(row["stem"])
        rgb = load_rgb_224(row["filepath"], int(cfg["img_size"]))
        fg = load_mask_224(row["maskpath"], int(cfg["img_size"]))
        ig = minmax_fg(np.load(stem_to_ig_abs_path(stem, cfg["canonical_ig_zero_absolute_dir"])), fg)
        tmp = fig_dir / "representative_descriptor_maps" / f"{stem}_ig_overlay.png"
        save_overlay(rgb, ig, fg, tmp, alpha=0.4, cmap="turbo")
        overlay = plt.imread(tmp)[..., :3]
        panels = [rgb, overlay]
        for d in descs:
            panels.append(np.load(run_dir / "descriptor_maps/raw_npy" / stem / f"{d}.npy"))
        titles = ["Input", "IG overlay"] + [display_name(d) for d in descs]
        for j, panel in enumerate(panels):
            ax = axes[i, j]
            if j < 2:
                ax.imshow(panel)
            else:
                ax.imshow(np.where(fg, panel, 0.0), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(row["class"], fontsize=10, rotation=0, labelpad=24, va="center")
    fig.tight_layout(w_pad=0.15, h_pad=0.15)
    out = fig_dir / "representative_descriptor_maps" / "representative_descriptor_maps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [str(out), str(out.with_suffix(".pdf"))]


def make_heatmaps(cfg, run_dir: Path, fig_dir: Path):
    paths = []
    summary = pd.read_csv(run_dir / "association_ig_zero_absolute/descriptor_summary.csv")
    classwise = pd.read_csv(run_dir / "association_ig_zero_absolute/classwise_summary.csv")
    selected = cfg["representative_descriptors"]
    heat = classwise[classwise["descriptor"].isin(selected)].pivot(index="descriptor", columns="class", values="mean_spearman_r")
    heat = heat.reindex(index=selected, columns=cfg["class_order"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(heat.values, cmap="YlOrRd", vmin=0, vmax=max(0.5, np.nanmax(heat.values)))
    ax.set_xticks(range(len(heat.columns)), heat.columns, fontsize=10)
    ax.set_yticks(range(len(heat.index)), [display_name(x) for x in heat.index], fontsize=9)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Descriptor", fontsize=11)
    ax.set_title("IG–descriptor spatial association", fontsize=12)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Mean Spearman r", fontsize=10)
    out = fig_dir / "correlation_heatmaps" / "ig_descriptor_classwise_selected_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    paths += [str(out), str(out.with_suffix(".pdf"))]

    top = summary.sort_values("mean_spearman_r", ascending=False).head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.barh([display_name(x) for x in top["descriptor"]], top["mean_spearman_r"], color="#8fb9a8")
    ax.set_xlabel("Mean Spearman r", fontsize=11)
    ax.set_title("Image-level IG–descriptor spatial association", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    out = fig_dir / "correlation_heatmaps" / "ig_descriptor_top_barplot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    paths += [str(out), str(out.with_suffix(".pdf"))]
    return paths


def make_class_average_figure(cfg, run_dir: Path, fig_dir: Path):
    descs = cfg["representative_descriptors"]
    avg_corr = pd.read_csv(run_dir / "association_ig_zero_absolute/class_average_correlation.csv")
    heat = avg_corr[avg_corr["descriptor"].isin(descs)].pivot(index="descriptor", columns="class", values="spearman_r")
    heat = heat.reindex(index=descs, columns=cfg["class_order"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(heat.values, cmap="YlOrRd", vmin=0, vmax=max(0.5, np.nanmax(heat.values)))
    ax.set_xticks(range(len(heat.columns)), heat.columns, fontsize=10)
    ax.set_yticks(range(len(heat.index)), [display_name(x) for x in heat.index], fontsize=9)
    ax.set_title("Class-average spatial association", fontsize=12)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Spearman r", fontsize=10)
    out = fig_dir / "class_average_maps" / "class_average_correlation_selected_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [str(out), str(out.with_suffix(".pdf"))]


def make_gradcam_supplement(cfg, run_dir: Path, fig_dir: Path):
    summary = pd.read_csv(run_dir / "association_gradcam_convnext/descriptor_summary.csv").sort_values("mean_spearman_r", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.3, 5.3))
    ax.barh([display_name(x) for x in summary["descriptor"]], summary["mean_spearman_r"], color="#b7a6d8")
    ax.set_xlabel("Mean Spearman r", fontsize=11)
    ax.set_title("Supplementary Grad-CAM–descriptor spatial association", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    out = fig_dir / "correlation_heatmaps" / "supp_gradcam_descriptor_top_barplot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [str(out), str(out.with_suffix(".pdf"))]


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = Path(args.run_dir)
    fig_dir = run_dir / "figures"
    paths = []
    paths += make_rep_descriptor_figure(cfg, run_dir, fig_dir)
    paths += make_heatmaps(cfg, run_dir, fig_dir)
    paths += make_class_average_figure(cfg, run_dir, fig_dir)
    if cfg.get("run_gradcam_descriptor_association", True):
        paths += make_gradcam_supplement(cfg, run_dir, fig_dir)
    pd.DataFrame({"figure_path": paths}).to_csv(fig_dir / "stage3_figure_manifest.csv", index=False)
    print("Figures generated:")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
