#!/usr/bin/env python3
"""Generate supported supplementary figure drafts from existing project data."""

from __future__ import annotations

import os

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


ROOT = Path(os.environ.get("MEDICINAL_SEED_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
STAGE_PACKAGE = ROOT / "manuscript_v3_supplement_code_submission"
STAGE1_CONVNEXT = ROOT / "stage1_model_performance_comparison_runs/convnext_small"
IG_BASE = ROOT / "stage2_attribution_maps/runs/stage2_attribution_20260605/01_ig_convnext_canonical_rawrgb_baseline"
GRADCAM_SELECTED = ROOT / "stage2_attribution_maps/runs/stage2_attribution_20260605/03_gradcam_final_selected_layers/convnext_small/selected_layer_stages.2.blocks.26"
DESCRIPTOR_RUN = ROOT / "stage3_descriptor_association/runs/stage3_descriptor_association_20260606_020607/descriptor_maps"
SOURCE_DIR = ROOT / "supplementary_figure_source_data"


CLASS_ORDER = ["ARSE", "ARSS", "PJNA", "PRDA", "PRPE"]
CLASS_NAMES = {
    "ARSE": "ARSE\nArmeniaca vulgaris",
    "ARSS": "ARSS\nArmeniaca sibirica",
    "PJNA": "PJNA\nPrunus japonica",
    "PRDA": "PRDA\nPrunus davidiana",
    "PRPE": "PRPE\nPrunus persica",
}
SHORT_CLASS_NAMES = {
    "ARSE": "ARSE",
    "ARSS": "ARSS",
    "PJNA": "PJNA",
    "PRDA": "PRDA",
    "PRPE": "PRPE",
}
DESCRIPTOR_DISPLAY = {
    "LAB_L": "LAB L",
    "FFT_LowPass": "FFT low-pass",
    "DistanceTransform": "Distance transform",
}


def ensure_dirs() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def normalize(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(x)
    if mask is not None:
        valid &= mask.astype(bool)
    if valid.sum() == 0:
        return np.zeros_like(x)
    vals = x[valid]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def load_rgb(path: str | Path, size: int = 224) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def load_mask(path: str | Path, size: int = 224) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0


def load_ig(stem: str, baseline: str) -> np.ndarray:
    if baseline == "zero":
        path = IG_BASE / f"zero_baseline/map_absolute_npy/{stem}_zero_baseline_absolute.npy"
    elif baseline == "blur":
        path = IG_BASE / f"blur_baseline_sigma20/map_absolute_npy/{stem}_blur_baseline_sigma20_absolute.npy"
    else:
        raise ValueError(f"Unsupported IG baseline: {baseline}")
    return np.load(path).astype(np.float32)


def overlay(rgb: np.ndarray, heat: np.ndarray, mask: np.ndarray, alpha: float = 0.46, cmap: str = "inferno") -> np.ndarray:
    h = normalize(np.abs(heat), mask)
    cm = plt.get_cmap(cmap)(h)[..., :3]
    m = mask[..., None].astype(bool)
    return np.where(m, (1 - alpha) * rgb + alpha * cm, rgb)


def imshow_clean(ax, img, title: str | None = None, cmap: str | None = None) -> None:
    ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.75")


def make_s1() -> None:
    reps = pd.read_csv(GRADCAM_SELECTED / "representative_samples.csv")
    descriptors = ["LAB_L", "FFT_LowPass", "DistanceTransform"]
    rows = []

    fig, axes = plt.subplots(len(reps), 6, figsize=(9.2, 7.7), dpi=320)
    col_titles = ["Input", "Zero-baseline\nabsolute IG", "Grad-CAM", "LAB L", "FFT low-pass", "Distance\ntransform"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9, pad=6)

    for r, row in reps.iterrows():
        stem, cls = row["stem"], row["true_class"]
        rgb = load_rgb(row["filepath"])
        mask = load_mask(row["maskpath"])
        zero_overlay_path = IG_BASE / f"zero_baseline/overlay_absolute_png/{stem}_zero_baseline_absolute_overlay.png"
        gradcam_overlay = load_rgb(row["overlay_png"])

        imshow_clean(axes[r, 0], rgb)
        imshow_clean(axes[r, 1], load_rgb(zero_overlay_path))
        imshow_clean(axes[r, 2], gradcam_overlay)
        for j, desc in enumerate(descriptors, start=3):
            desc_path = DESCRIPTOR_RUN / f"visualization_png/{stem}/{desc}.png"
            desc_img = np.asarray(Image.open(desc_path).convert("L"), dtype=np.float32) / 255.0
            imshow_clean(axes[r, j], desc_img, cmap="gray")

        axes[r, 0].set_ylabel(CLASS_NAMES[cls], fontsize=8, rotation=0, ha="right", va="center", labelpad=40)
        rows.append(
            {
                "class": cls,
                "stem": stem,
                "input_image": row["filepath"],
                "mask_image": row["maskpath"],
                "zero_ig_absolute_npy": str(IG_BASE / f"zero_baseline/map_absolute_npy/{stem}_zero_baseline_absolute.npy"),
                "zero_ig_overlay_png": str(zero_overlay_path),
                "gradcam_selected_layer": row["selected_layer"],
                "gradcam_overlay_png": row["overlay_png"],
                "descriptor_visualization_pngs": ";".join(str(DESCRIPTOR_RUN / f"visualization_png/{stem}/{d}.png") for d in descriptors),
                "descriptor_raw_npy": ";".join(str(DESCRIPTOR_RUN / f"raw_npy/{stem}/{d}.npy") for d in descriptors),
            }
        )

    fig.text(0.013, 0.982, "a", fontsize=12, fontweight="bold", va="top")
    fig.tight_layout(rect=(0.04, 0.0, 1, 0.98), w_pad=0.8, h_pad=0.7)
    fig.savefig(ROOT / "Supplementary_Fig_S1.png", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "Supplementary_Fig_S1_source_data.csv", index=False)


def make_s2() -> None:
    consistency = pd.read_csv(IG_BASE / "visualization_check/zero_vs_blur_correlation_summary.csv")
    consistency_samples = consistency.loc[consistency["stem"].notna()].copy()
    class_arrays: dict[tuple[str, str], np.ndarray] = {}
    source_rows = []

    for cls in CLASS_ORDER:
        for baseline in ["zero", "blur"]:
            if baseline == "zero":
                source_path = IG_BASE / f"zero_baseline/class_average_absolute_npy/{cls}_zero_baseline_absolute_average.npy"
                png_path = IG_BASE / f"zero_baseline/class_average_absolute_png/{cls}_zero_baseline_absolute_average.png"
            else:
                source_path = IG_BASE / f"blur_baseline_sigma20/class_average_absolute_npy/{cls}_blur_baseline_sigma20_absolute_average.npy"
                png_path = IG_BASE / f"blur_baseline_sigma20/class_average_absolute_png/{cls}_blur_baseline_sigma20_absolute_average.png"
            arr = np.load(source_path).astype(np.float32)
            class_arrays[(cls, baseline)] = arr
            n = int(consistency_samples.loc[consistency_samples["class"] == cls].shape[0])
            source_rows.append({"class": cls, "baseline": baseline, "n": n, "mean_map_npy_key": f"{cls}_{baseline}", "source_npy": str(source_path), "source_png": str(png_path)})

    np.savez_compressed(SOURCE_DIR / "Supplementary_Fig_S2_class_average_maps.npz", **{f"{k[0]}_{k[1]}": v for k, v in class_arrays.items()})
    pd.DataFrame(source_rows).to_csv(SOURCE_DIR / "Supplementary_Fig_S2_class_average_source.csv", index=False)
    consistency.to_csv(SOURCE_DIR / "Supplementary_Fig_S2_zero_vs_blur_spearman_source.csv", index=False)

    fig = plt.figure(figsize=(8.7, 8.0), dpi=320)
    gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 0.85], width_ratios=[1, 1, 0.16], hspace=0.24, wspace=0.05)
    last_im = None
    for r, cls in enumerate(CLASS_ORDER):
        for c, baseline in enumerate(["zero", "blur"]):
            ax = fig.add_subplot(gs[r, c])
            if baseline == "zero":
                png_path = IG_BASE / f"zero_baseline/class_average_absolute_png/{cls}_zero_baseline_absolute_average.png"
            else:
                png_path = IG_BASE / f"blur_baseline_sigma20/class_average_absolute_png/{cls}_blur_baseline_sigma20_absolute_average.png"
            ax.imshow(load_rgb(png_path))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("Zero baseline" if c == 0 else "Gaussian-blurred baseline", fontsize=9)
            if c == 0:
                ax.set_ylabel(CLASS_NAMES[cls], fontsize=8, rotation=0, ha="right", va="center", labelpad=42)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("0.75")
    cax = fig.add_subplot(gs[:5, 2])
    gradient = np.linspace(0, 1, 256)[:, None]
    cax.imshow(gradient, cmap="jet", origin="lower", aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([0, 64, 128, 192, 255], ["0.0", "0.25", "0.5", "0.75", "1.0"])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("Normalized class-average IG")

    ax = fig.add_subplot(gs[5, :])
    data = [consistency_samples.loc[consistency_samples["class"] == cls, "absolute_spearman_r"].astype(float).to_numpy() for cls in CLASS_ORDER]
    ax.boxplot(data, tick_labels=CLASS_ORDER, widths=0.55, showfliers=False, medianprops={"color": "black", "linewidth": 1.2})
    ax.set_ylabel("Spearman r", fontsize=9)
    ax.set_xlabel("Class", fontsize=9)
    ax.set_title("Per-sample spatial agreement between zero- and blurred-baseline IG maps", fontsize=9, pad=6)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.tick_params(labelsize=8)
    fig.text(0.018, 0.982, "a", fontsize=12, fontweight="bold", va="top")
    fig.text(0.018, 0.165, "b", fontsize=12, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.12, right=0.94, top=0.94, bottom=0.08)
    fig.savefig(ROOT / "Supplementary_Fig_S2.png", bbox_inches="tight")
    plt.close(fig)


def make_s3() -> None:
    cm = pd.read_csv(STAGE1_CONVNEXT / "confusion_matrix_counts.csv", index_col=0).loc[CLASS_ORDER, CLASS_ORDER]
    norm = pd.read_csv(STAGE1_CONVNEXT / "confusion_matrix_normalized.csv", index_col=0).loc[CLASS_ORDER, CLASS_ORDER]
    pred = pd.read_csv(STAGE1_CONVNEXT / "test_predictions.csv")
    mis = pred.loc[pred["true_label"] != pred["pred_label"]].copy()
    mis["filename"] = mis["filepath"].map(lambda p: Path(p).name)
    mis["pred_class"] = mis["pred_label"]
    cm.to_csv(SOURCE_DIR / "Supplementary_Fig_S3_confusion_matrix_counts.csv")
    norm.to_csv(SOURCE_DIR / "Supplementary_Fig_S3_confusion_matrix_normalized.csv")
    mis.to_csv(SOURCE_DIR / "Supplementary_Fig_S3_misclassified_samples.csv", index=False)

    fig = plt.figure(figsize=(6.8, 4.55), dpi=320)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.85], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER, fontsize=8)
    ax.set_xlabel("Predicted class", fontsize=9)
    ax.set_ylabel("True class", fontsize=9)
    ax.set_title("Confusion matrix", fontsize=10)
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            value = cm.iloc[i, j]
            color = "white" if norm.iloc[i, j] > 0.55 else "black"
            ax.text(j, i, str(int(value)), ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Row-normalized value", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    for idx, (_, row) in enumerate(mis.iterrows()):
        ax_img = fig.add_subplot(gs[0, idx + 1])
        img = load_rgb(row["filepath"])
        imshow_clean(ax_img, img)
        ax_img.set_title(
            f"{row['filename']}\ntrue {row['true_label']} / predicted {row['pred_label']}\nconfidence {float(row['confidence']):.3f}",
            fontsize=8,
            pad=5,
        )
    fig.text(0.014, 0.98, "a", fontsize=12, fontweight="bold", va="top")
    fig.text(0.58, 0.98, "b", fontsize=12, fontweight="bold", va="top")
    fig.savefig(ROOT / "Supplementary_Fig_S3.png", bbox_inches="tight")
    plt.close(fig)


def make_s4() -> None:
    df = pd.read_csv(STAGE_PACKAGE / "source_data/stage3_descriptor_association/ig_zero_absolute_classwise_summary.csv")
    overall = pd.read_csv(STAGE_PACKAGE / "source_data/stage3_descriptor_association/ig_zero_absolute_descriptor_summary.csv")
    order = overall.sort_values("mean_spearman_r", ascending=False)["descriptor"].tolist()
    pivot = df.pivot(index="class", columns="descriptor", values="mean_spearman_r").loc[CLASS_ORDER, order]
    pivot.to_csv(SOURCE_DIR / "Supplementary_Fig_S4_classwise_heatmap_source.csv")

    fig, ax = plt.subplots(figsize=(11.5, 3.6), dpi=320)
    vmax = float(np.nanmax(np.abs(pivot.values)))
    im = ax.imshow(pivot.values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER, fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([d.replace("_", " ") for d in order], rotation=55, ha="right", fontsize=6.6)
    ax.set_title("Classwise descriptor association with zero-baseline absolute IG", fontsize=10, pad=8)
    ax.set_ylabel("Class", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Mean Spearman r", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "Supplementary_Fig_S4.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    make_s1()
    make_s2()
    make_s3()
    make_s4()
    with (SOURCE_DIR / "generated_files_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "description"])
        for name in ["Supplementary_Fig_S1.png", "Supplementary_Fig_S2.png", "Supplementary_Fig_S3.png", "Supplementary_Fig_S4.png"]:
            writer.writerow([str(ROOT / name), "Generated supplementary figure draft"])
        for path in sorted(SOURCE_DIR.iterdir()):
            writer.writerow([str(path), "Supplementary figure source data or generated descriptor map array"])


if __name__ == "__main__":
    main()
