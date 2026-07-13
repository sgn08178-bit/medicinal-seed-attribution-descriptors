#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_yaml, save_json
from src.models import build_model, load_checkpoint, predict_logits
from src.seed_utils import set_seed


CLASS_ORDER = ["ARSE", "ARSS", "PJNA", "PRDA", "PRPE"]
BASELINES = [("zero", "zero_baseline", None), ("blur", "blur_baseline_sigma20", 20.0)]


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean: list[float], std: list[float]):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, raw_rgb: torch.Tensor) -> torch.Tensor:
        return self.model((raw_rgb - self.mean) / self.std)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--internal-batch-size", type=int, default=10)
    return p.parse_args()


def latest_run_dir() -> Path:
    runs = sorted([p for p in (ROOT / "runs").glob("stage2_attribution_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No Stage 2 run directory found under {ROOT / 'runs'}")
    return runs[-1]


def load_raw_rgb(path: str | Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def load_fg_mask(path: str | Path, size: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0)


def blur_baseline(raw_rgb: torch.Tensor, sigma: float) -> torch.Tensor:
    arr = raw_rgb.detach().cpu().numpy().transpose(1, 2, 0)
    blurred = gaussian_filter(arr, sigma=[sigma, sigma, 0])
    blurred = np.clip(blurred, 0, 1).astype(np.float32)
    return torch.from_numpy(blurred.transpose(2, 0, 1)).float()


def maps_from_attr(attr_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signed_sum = attr_chw.sum(axis=0)
    positive = np.maximum(signed_sum, 0).astype(np.float32)
    absolute = np.abs(attr_chw).sum(axis=0).astype(np.float32)
    return positive, absolute


def minmax_fg(arr: np.ndarray, mask: np.ndarray, low: float = 1, high: float = 99, smooth_sigma: float = 0.8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    fg = mask.astype(bool)
    out = np.zeros_like(arr, dtype=np.float32)
    if fg.sum() == 0:
        return out
    lo, hi = np.percentile(arr[fg], [low, high])
    clipped = np.clip(arr, lo, hi)
    if smooth_sigma and smooth_sigma > 0:
        smooth = gaussian_filter(clipped, sigma=smooth_sigma)
        clipped = np.where(fg, smooth, clipped)
    vals = clipped[fg]
    mn, mx = float(np.nanmin(vals)), float(np.nanmax(vals))
    if mx - mn < 1e-8:
        return out
    out[fg] = np.clip((clipped[fg] - mn) / (mx - mn), 0, 1)
    return out


def save_overlay(rgb_chw: np.ndarray, heat01: np.ndarray, mask: np.ndarray, path: Path, alpha: float = 0.40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgb_chw.transpose(1, 2, 0), 0, 1)
    heat = plt.get_cmap("turbo")(np.clip(heat01, 0, 1))[..., :3]
    out = np.where(mask[..., None], (1 - alpha) * rgb + alpha * heat, rgb)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(path)


def save_heatmap(heat01: np.ndarray, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heat = plt.get_cmap("turbo")(np.clip(heat01, 0, 1))[..., :3]
    heat = np.where(mask[..., None], heat, 0.0)
    Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).save(path)


def make_dirs(base: Path) -> None:
    for _, folder, _ in BASELINES:
        b = base / folder
        for sub in [
            "raw_rgb_attr_npy",
            "map_absolute_npy",
            "map_positive_npy",
            "overlay_absolute_png",
            "overlay_positive_png",
            "heatmap_absolute_png",
            "heatmap_positive_png",
            "class_average_absolute_npy",
            "class_average_positive_npy",
            "class_average_absolute_png",
            "class_average_positive_png",
        ]:
            (b / sub).mkdir(parents=True, exist_ok=True)
    (base / "metadata").mkdir(parents=True, exist_ok=True)
    (base / "visualization_check").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)


def spearman_fg(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    fg = mask.astype(bool)
    if fg.sum() < 3:
        return float("nan")
    stat = spearmanr(a[fg].ravel(), b[fg].ravel()).statistic
    return float(stat)


def pick_representatives(meta: pd.DataFrame, class_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    correct = meta[meta["true_label"] == meta["pred_label"]].copy()
    reps = []
    for cls in class_order:
        cdf = correct[correct["true_class"] == cls].sort_values("confidence", ascending=False)
        if cdf.empty:
            continue
        high = cdf.iloc[0].copy()
        high["selection_type"] = "high_confidence_correct"
        reps.append(high)
        mid = cdf.iloc[len(cdf) // 2].copy()
        mid["selection_type"] = "middle_confidence_correct"
        if str(mid["stem"]) != str(high["stem"]):
            reps.append(mid)
    rep = pd.DataFrame(reps).drop_duplicates("stem").reset_index(drop=True)
    mis = meta[meta["true_label"] != meta["pred_label"]].copy().reset_index(drop=True)
    return rep, mis


def make_rep_grid(rep: pd.DataFrame, base: Path, baseline_folder: str, mode: str, out_name: str, img_size: int) -> None:
    n = len(rep)
    fig, axes = plt.subplots(n, 3, figsize=(7.4, 2.35 * n), dpi=300)
    if n == 1:
        axes = axes[None, :]
    for i, row in rep.iterrows():
        stem = row["stem"]
        cls = row["true_class"]
        rgb = Image.open(row["filepath"]).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
        heat = Image.open(base / baseline_folder / f"heatmap_{mode}_png" / f"{stem}_{baseline_folder}_{mode}_heatmap.png").convert("RGB")
        overlay = Image.open(base / baseline_folder / f"overlay_{mode}_png" / f"{stem}_{baseline_folder}_{mode}_overlay.png").convert("RGB")
        for ax, im, title in zip(axes[i], [rgb, heat, overlay], ["Input", f"{mode} heatmap", f"{mode} overlay"]):
            ax.imshow(im)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(f"{cls}\n{stem}", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(base / "visualization_check" / f"{out_name}.png", dpi=300)
    fig.savefig(base / "visualization_check" / f"{out_name}.pdf")
    plt.close(fig)


def compare_deprecated(base: Path, run_dir: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    dep_dir = run_dir / "01_ig_convnext" / "convnext_small" / "raw_npy"
    rows = []
    if not dep_dir.exists():
        return pd.DataFrame(rows)
    for _, row in test_df.iterrows():
        stem = row["stem"]
        dep_path = dep_dir / f"{stem}.npy"
        can_path = base / "zero_baseline" / "map_absolute_npy" / f"{stem}_zero_baseline_absolute.npy"
        if not dep_path.exists() or not can_path.exists():
            rows.append({"stem": stem, "class": row["class"], "spearman_r": np.nan, "note": "missing map"})
            continue
        dep = np.load(dep_path)
        can = np.load(can_path)
        fg = load_fg_mask(row["maskpath"], dep.shape[-1])
        rows.append(
            {
                "stem": stem,
                "class": row["class"],
                "spearman_r": spearman_fg(dep, can, fg),
                "deprecated_map": str(dep_path),
                "canonical_zero_absolute_map": str(can_path),
                "note": "sanity check only; deprecated normalized-baseline IG used a different baseline definition",
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    cfg = load_yaml(run_dir / "config.yaml")
    set_seed(int(cfg["seed"]))
    test_csv = run_dir / "inputs" / "test.csv"
    checkpoint = Path(cfg.get("convnext_checkpoint", "data/model_checkpoints/convnext_small_best_model.pth"))
    for path, label in [
        (run_dir, "Stage 2 run directory"),
        (test_csv, "Stage 2 test.csv"),
        (checkpoint, "Stage 1 ConvNeXt checkpoint"),
        (Path(cfg["image_root"]), "image_root"),
        (Path(cfg["mask_root"]), "mask_root"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    out = run_dir / "01_ig_convnext_canonical_rawrgb_baseline"
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out}. Use --overwrite only if you intend to replace this canonical output.")
    make_dirs(out)
    test_df = pd.read_csv(test_csv)
    missing_img = test_df[~test_df["filepath"].map(lambda p: Path(p).exists())]
    missing_mask = test_df[~test_df["maskpath"].map(lambda p: Path(p).exists())]
    if len(missing_img) or len(missing_mask):
        raise FileNotFoundError(f"Missing images={len(missing_img)}, missing masks={len(missing_mask)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(build_model("convnext_small", int(cfg["num_classes"]), pretrained=False), checkpoint, device)
    wrapper = NormalizedModel(model, cfg["imagenet_mean"], cfg["imagenet_std"]).to(device).eval()
    ig = IntegratedGradients(wrapper, multiply_by_inputs=True)

    records = {folder: [] for _, folder, _ in BASELINES}
    failed = []
    maps_by_class = {
        folder: {mode: {cls: [] for cls in CLASS_ORDER} for mode in ["absolute", "positive"]}
        for _, folder, _ in BASELINES
    }
    stats_rows = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Canonical raw-RGB IG"):
        stem = row["stem"]
        try:
            raw = load_raw_rgb(row["filepath"], int(cfg["img_size"]))
            fg = load_fg_mask(row["maskpath"], int(cfg["img_size"]))
            inp = raw.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = wrapper(inp)
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
            target = int(pred.item())
            for base_short, folder, sigma in BASELINES:
                if base_short == "zero":
                    baseline = torch.zeros_like(inp)
                else:
                    baseline = blur_baseline(raw, float(sigma)).unsqueeze(0).to(device)
                attr, delta = ig.attribute(
                    inp,
                    baselines=baseline,
                    target=torch.tensor([target], device=device),
                    n_steps=int(args.n_steps),
                    internal_batch_size=int(args.internal_batch_size),
                    return_convergence_delta=True,
                )
                attr_chw = attr.squeeze(0).detach().cpu().numpy().astype(np.float32)
                positive, absolute = maps_from_attr(attr_chw)
                bdir = out / folder
                np.save(bdir / "raw_rgb_attr_npy" / f"{stem}_{folder}_raw_rgb_attr.npy", attr_chw)
                np.save(bdir / "map_absolute_npy" / f"{stem}_{folder}_absolute.npy", absolute)
                np.save(bdir / "map_positive_npy" / f"{stem}_{folder}_positive.npy", positive)
                for mode, mmap in [("absolute", absolute), ("positive", positive)]:
                    vis = minmax_fg(mmap, fg, 1, 99, smooth_sigma=0.8)
                    save_overlay(raw.numpy(), vis, fg, bdir / f"overlay_{mode}_png" / f"{stem}_{folder}_{mode}_overlay.png")
                    save_heatmap(vis, fg, bdir / f"heatmap_{mode}_png" / f"{stem}_{folder}_{mode}_heatmap.png")
                    maps_by_class[folder][mode][row["class"]].append(mmap)
                records[folder].append(
                    {
                        "stem": stem,
                        "filepath": row["filepath"],
                        "maskpath": row["maskpath"],
                        "true_class": row["class"],
                        "true_label": int(row["label"]),
                        "pred_label": target,
                        "pred_class": CLASS_ORDER[target],
                        "confidence": float(conf.item()),
                        "target_label": target,
                        "target_class": CLASS_ORDER[target],
                        "baseline_type": folder,
                        "convergence_delta": float(delta.detach().cpu().numpy().reshape(-1)[0]),
                        "raw_rgb_attr_shape": "3x224x224",
                        "map_shape": "224x224",
                        "raw_rgb_attr_npy": str(bdir / "raw_rgb_attr_npy" / f"{stem}_{folder}_raw_rgb_attr.npy"),
                        "map_absolute_npy": str(bdir / "map_absolute_npy" / f"{stem}_{folder}_absolute.npy"),
                        "map_positive_npy": str(bdir / "map_positive_npy" / f"{stem}_{folder}_positive.npy"),
                    }
                )
                stats_rows.append(
                    {
                        "stem": stem,
                        "class": row["class"],
                        "baseline_type": folder,
                        "attr_nan_count": int(np.isnan(attr_chw).sum()),
                        "attr_inf_count": int(np.isinf(attr_chw).sum()),
                        "absolute_nan_count": int(np.isnan(absolute).sum()),
                        "positive_nan_count": int(np.isnan(positive).sum()),
                        "absolute_all_zero": bool(np.all(absolute == 0)),
                        "positive_all_zero": bool(np.all(positive == 0)),
                        "absolute_fg_std": float(np.nanstd(absolute[fg])),
                        "positive_fg_std": float(np.nanstd(positive[fg])),
                        "absolute_near_constant_fg": bool(np.nanstd(absolute[fg]) < 1e-8),
                        "positive_near_constant_fg": bool(np.nanstd(positive[fg]) < 1e-8),
                    }
                )
        except Exception as exc:
            failed.append({"stem": stem, "class": row.get("class", ""), "error": repr(exc)})

    for _, folder, _ in BASELINES:
        pd.DataFrame(records[folder]).to_csv(out / folder / "attribution_metadata.csv", index=False)
        for mode in ["absolute", "positive"]:
            for cls in CLASS_ORDER:
                maps = maps_by_class[folder][mode][cls]
                if not maps:
                    continue
                avg = np.mean(np.stack(maps), axis=0).astype(np.float32)
                np.save(out / folder / f"class_average_{mode}_npy" / f"{cls}_{folder}_{mode}_average.npy", avg)
                vis = minmax_fg(avg, np.ones_like(avg, dtype=bool), 1, 99, smooth_sigma=0.8)
                save_heatmap(vis, np.ones_like(avg, dtype=bool), out / folder / f"class_average_{mode}_png" / f"{cls}_{folder}_{mode}_average.png")

    pd.DataFrame(failed, columns=["stem", "class", "error"]).to_csv(out / "metadata" / "ig_canonical_failed.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(out / "metadata" / "ig_canonical_value_checks.csv", index=False)

    # Baseline consistency.
    corr_rows = []
    for _, row in test_df.iterrows():
        stem = row["stem"]
        fg = load_fg_mask(row["maskpath"], int(cfg["img_size"]))
        try:
            z_abs = np.load(out / "zero_baseline" / "map_absolute_npy" / f"{stem}_zero_baseline_absolute.npy")
            b_abs = np.load(out / "blur_baseline_sigma20" / "map_absolute_npy" / f"{stem}_blur_baseline_sigma20_absolute.npy")
            z_pos = np.load(out / "zero_baseline" / "map_positive_npy" / f"{stem}_zero_baseline_positive.npy")
            b_pos = np.load(out / "blur_baseline_sigma20" / "map_positive_npy" / f"{stem}_blur_baseline_sigma20_positive.npy")
            corr_rows.append(
                {
                    "stem": stem,
                    "class": row["class"],
                    "absolute_spearman_r": spearman_fg(z_abs, b_abs, fg),
                    "positive_spearman_r": spearman_fg(z_pos, b_pos, fg),
                }
            )
        except Exception as exc:
            corr_rows.append({"stem": stem, "class": row["class"], "absolute_spearman_r": np.nan, "positive_spearman_r": np.nan, "error": repr(exc)})
    corr_df = pd.DataFrame(corr_rows)
    class_summary = corr_df.groupby("class").agg(
        absolute_mean=("absolute_spearman_r", "mean"),
        absolute_std=("absolute_spearman_r", "std"),
        positive_mean=("positive_spearman_r", "mean"),
        positive_std=("positive_spearman_r", "std"),
        n=("stem", "count"),
    ).reset_index()
    overall = pd.DataFrame(
        [
            {
                "class": "ALL",
                "absolute_mean": corr_df["absolute_spearman_r"].mean(),
                "absolute_std": corr_df["absolute_spearman_r"].std(),
                "positive_mean": corr_df["positive_spearman_r"].mean(),
                "positive_std": corr_df["positive_spearman_r"].std(),
                "n": len(corr_df),
            }
        ]
    )
    corr_out = pd.concat([corr_df, class_summary, overall], ignore_index=True, sort=False)
    corr_out.to_csv(out / "visualization_check" / "zero_vs_blur_correlation_summary.csv", index=False)

    dep_df = compare_deprecated(out, run_dir, test_df)
    dep_df.to_csv(out / "metadata" / "comparison_with_deprecated_ig.csv", index=False)
    dep_df.to_csv(out / "comparison_with_deprecated_ig.csv", index=False)

    zero_meta = pd.DataFrame(records["zero_baseline"])
    reps, mis = pick_representatives(zero_meta, CLASS_ORDER)
    reps.to_csv(out / "visualization_check" / "representative_samples.csv", index=False)
    mis.to_csv(out / "visualization_check" / "misclassified_samples.csv", index=False)
    for folder in ["zero_baseline", "blur_baseline_sigma20"]:
        for mode in ["absolute", "positive"]:
            out_name = f"representative_grid_{'zero' if folder == 'zero_baseline' else 'blur'}_{mode}"
            make_rep_grid(reps, out, folder, mode, out_name, int(cfg["img_size"]))

    baseline_definition = {
        "input_space": "raw RGB tensor in [0, 1]",
        "model_forward": "NormalizedModel applies ImageNet mean/std inside forward before ConvNeXt-Small.",
        "zero_baseline": "raw RGB black image, all channels equal 0.0",
        "gaussian_blurred_baseline": "Gaussian blur applied to raw RGB input, sigma=20, values clipped to [0, 1]",
        "interpolation_space": "raw RGB input space",
        "target": "predicted class logit",
        "n_steps": int(args.n_steps),
        "channel_maps": {
            "raw_rgb_attr_npy": "signed RGB attribution, shape [3, 224, 224]",
            "absolute": "absolute attribution summed across RGB channels",
            "positive": "channel-summed attribution followed by ReLU",
        },
    }
    save_json(baseline_definition, out / "metadata" / "baseline_definition.json")
    save_json(baseline_definition, out / "baseline_definition.json")
    out_cfg = dict(cfg)
    out_cfg["canonical_ig"] = {
        "n_steps": int(args.n_steps),
        "internal_batch_size": int(args.internal_batch_size),
        "blur_sigma": 20,
        "visualization_cmap": "turbo",
        "visualization_alpha": 0.40,
        "visualization_percentile_clip": [1, 99],
        "visualization_smoothing_sigma": 0.8,
        "raw_npy_smoothing": False,
    }
    import yaml

    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, allow_unicode=True)

    zero_df = pd.DataFrame(records["zero_baseline"])
    blur_df = pd.DataFrame(records["blur_baseline_sigma20"])
    value_df = pd.DataFrame(stats_rows)
    metadata = {
        "output_dir": str(out),
        "checkpoint": str(checkpoint),
        "test_csv": str(test_csv),
        "zero_success": int(len(zero_df)),
        "blur_success": int(len(blur_df)),
        "failed": int(len(failed)),
        "raw_rgb_attr_shape": "[3, 224, 224]",
        "map_shape": "[224, 224]",
        "nan_or_inf_rows": int(((value_df.get("attr_nan_count", pd.Series(dtype=int)) > 0) | (value_df.get("attr_inf_count", pd.Series(dtype=int)) > 0)).sum()) if not value_df.empty else 0,
        "zero_convergence_delta_mean": float(zero_df["convergence_delta"].mean()) if not zero_df.empty else None,
        "zero_convergence_delta_std": float(zero_df["convergence_delta"].std()) if not zero_df.empty else None,
        "blur_convergence_delta_mean": float(blur_df["convergence_delta"].mean()) if not blur_df.empty else None,
        "blur_convergence_delta_std": float(blur_df["convergence_delta"].std()) if not blur_df.empty else None,
        "zero_vs_blur_absolute_spearman_mean": float(corr_df["absolute_spearman_r"].mean()),
        "zero_vs_blur_absolute_spearman_std": float(corr_df["absolute_spearman_r"].std()),
        "zero_vs_blur_positive_spearman_mean": float(corr_df["positive_spearman_r"].mean()),
        "zero_vs_blur_positive_spearman_std": float(corr_df["positive_spearman_r"].std()),
        "deprecated_comparison_spearman_mean": float(dep_df["spearman_r"].mean()) if not dep_df.empty else None,
        "deprecated_comparison_spearman_std": float(dep_df["spearman_r"].std()) if not dep_df.empty else None,
        "deprecated_note": "Previous Stage 2 normalized-baseline IG is deprecated because its zero baseline was defined in ImageNet-normalized tensor space rather than raw RGB black-image space.",
        "recommended_main_quantitative_map": str(out / "zero_baseline" / "map_absolute_npy"),
        "recommended_visualization": str(out / "zero_baseline" / "overlay_absolute_png"),
    }
    save_json(metadata, out / "metadata" / "ig_canonical_run_metadata.json")

    report = f"""# IG Canonical Raw-RGB Baseline Report

## Why the Previous Normalized-Baseline IG Is Deprecated

The previous Stage 2 IG output used Captum on already ImageNet-normalized tensors. In that setup, a zero baseline is not a raw black image; it corresponds to a zero tensor after normalization. Therefore, its interpolation path differs from the canonical raw RGB baseline definition.

The previous output was not deleted. It remains at:

`{run_dir / '01_ig_convnext'}`

## Canonical Baseline Definition

- Input space: raw RGB tensor in [0, 1].
- Zero baseline: raw black image, RGB = 0.
- Gaussian-blurred baseline: raw RGB input blurred with Gaussian sigma = 20.
- Interpolation: raw RGB input space.
- Model output: predicted class logit.
- Steps: {int(args.n_steps)}.

## Model Forward Wrapper

`NormalizedModel` receives raw RGB tensors and applies ImageNet mean/std normalization inside `forward()` before passing the tensor to ConvNeXt-Small.

## Generated Counts

- Zero baseline success: {len(zero_df)}
- Blur baseline success: {len(blur_df)}
- Failed samples: {len(failed)}
- Raw RGB attribution shape: [3, 224, 224]
- 2D map shape: [224, 224]

## Convergence Delta

- Zero baseline mean ± SD: {metadata['zero_convergence_delta_mean']} ± {metadata['zero_convergence_delta_std']}
- Blur baseline mean ± SD: {metadata['blur_convergence_delta_mean']} ± {metadata['blur_convergence_delta_std']}

## Zero vs Blur Baseline Consistency

- Absolute map Spearman mean ± SD: {metadata['zero_vs_blur_absolute_spearman_mean']} ± {metadata['zero_vs_blur_absolute_spearman_std']}
- Positive map Spearman mean ± SD: {metadata['zero_vs_blur_positive_spearman_mean']} ± {metadata['zero_vs_blur_positive_spearman_std']}
- Detailed file: `{out / 'visualization_check/zero_vs_blur_correlation_summary.csv'}`

## Deprecated Normalized-Baseline IG Comparison

- Deprecated-vs-canonical zero absolute Spearman mean ± SD: {metadata['deprecated_comparison_spearman_mean']} ± {metadata['deprecated_comparison_spearman_std']}
- Detailed file: `{out / 'comparison_with_deprecated_ig.csv'}`
- This comparison is a sanity check only and is not a manuscript main result.

## Recommended Manuscript Map

- Main quantitative map candidate: zero baseline absolute map.
- Path: `{out / 'zero_baseline/map_absolute_npy'}`
- Visualization candidate: zero baseline absolute overlay.
- Path: `{out / 'zero_baseline/overlay_absolute_png'}`

Positive maps are also saved for comparison:

`{out / 'zero_baseline/map_positive_npy'}`

## Candidate Input for Stage 3 Descriptor Association

Recommended Stage 3 IG input:

`{out / 'zero_baseline/map_absolute_npy'}`

## Representative Visualization Grids

- `{out / 'visualization_check/representative_grid_zero_absolute.png'}`
- `{out / 'visualization_check/representative_grid_zero_positive.png'}`
- `{out / 'visualization_check/representative_grid_blur_absolute.png'}`
- `{out / 'visualization_check/representative_grid_blur_positive.png'}`

## Exclusions

No Grad-CAM modification, descriptor map generation, attribution-descriptor association, occlusion sensitivity, or mask overlap analysis was performed.
"""
    (out / "IG_CANONICAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
