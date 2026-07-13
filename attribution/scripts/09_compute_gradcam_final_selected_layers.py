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
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attribution_gradcam import GradCAM
from src.dataset import AttributionDataset
from src.io_utils import load_yaml, save_json
from src.models import build_model, load_checkpoint, predict_logits
from src.seed_utils import seed_worker, set_seed
from src.visualization import minmax01


SELECTED = {
    "convnext_small": {
        "layer": "stages.2.blocks.26",
        "expected_shape": [1, 384, 14, 14],
        "note": "Selected as the balanced ConvNeXt-Small Grad-CAM layer after candidate-layer comparison.",
        "usage": "supplementary spatial association comparison; IG remains the main attribution input.",
    },
    "resnet50": {
        "layer": "layer4.2",
        "expected_shape": [1, 2048, 7, 7],
        "note": "Existing full-test Grad-CAM used the same layer, but this folder regenerates outputs with final selected naming and turbo visualization.",
        "usage": "supplementary model comparison.",
    },
    "efficientnet_b0": {
        "layer": "blocks.4",
        "expected_shape": [1, 112, 14, 14],
        "note": "Selected as the best available EfficientNet-B0 candidate, but foreground coverage was low in candidate screening.",
        "usage": "supplementary comparison only; interpret with caution.",
    },
}


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


def layer_shape(model: torch.nn.Module, module: torch.nn.Module, sample: torch.Tensor) -> list[int]:
    captured = {}
    handle = module.register_forward_hook(lambda _m, _i, o: captured.setdefault("out", o))
    try:
        with torch.no_grad():
            _ = model(sample)
    finally:
        handle.remove()
    out = captured.get("out")
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out) or out.ndim != 4 or out.shape[-1] <= 1 or out.shape[-2] <= 1:
        raise RuntimeError(f"Selected layer output is not BCHW spatial: {out}")
    return list(out.shape)


def save_overlay(rgb_chw: np.ndarray, cam01: np.ndarray, mask: np.ndarray, path: Path, alpha: float = 0.40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgb_chw.transpose(1, 2, 0), 0, 1)
    heat = plt.get_cmap("turbo")(np.clip(cam01, 0, 1))[..., :3]
    out = np.where(mask[..., None], (1 - alpha) * rgb + alpha * heat, rgb)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(path)


def save_heatmap(cam01: np.ndarray, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heat = plt.get_cmap("turbo")(np.clip(cam01, 0, 1))[..., :3]
    heat = np.where(mask[..., None], heat, 0.0)
    Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).save(path)


def foreground_coverage_top20(cam01: np.ndarray, mask: np.ndarray) -> float:
    if float(np.nanmax(cam01) - np.nanmin(cam01)) < 1e-12:
        return float("nan")
    flat = cam01.reshape(-1)
    k = max(1, int(np.ceil(flat.size * 0.20)))
    top_idx = np.argpartition(flat, -k)[-k:]
    return float(mask.astype(bool).reshape(-1)[top_idx].mean())


def make_grid(meta: pd.DataFrame, out_dir: Path, model_name: str, layer_name: str, img_size: int) -> None:
    correct = meta[meta["true_label"] == meta["pred_label"]].copy()
    reps = []
    for cls in ["ARSE", "ARSS", "PJNA", "PRDA", "PRPE"]:
        cdf = correct[correct["true_class"] == cls].sort_values("confidence", ascending=False)
        if not cdf.empty:
            reps.append(cdf.iloc[0])
    reps = pd.DataFrame(reps).reset_index(drop=True)
    reps.to_csv(out_dir / "representative_samples.csv", index=False)
    n = len(reps)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.35 * n), dpi=300)
    if n == 1:
        axes = axes[None, :]
    for i, row in reps.iterrows():
        stem = row["stem"]
        rgb = Image.open(row["filepath"]).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
        heat = Image.open(row["heatmap_png"]).convert("RGB")
        overlay = Image.open(row["overlay_png"]).convert("RGB")
        for ax, im, title in zip(axes[i], [rgb, heat, overlay], ["Input", "Grad-CAM heatmap", "Grad-CAM overlay"]):
            ax.imshow(im)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(row["true_class"], fontsize=10, fontweight="bold")
    fig.suptitle(f"{model_name} selected-layer Grad-CAM ({layer_name})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_dir / "representative_grid.png", dpi=300)
    fig.savefig(out_dir / "representative_grid.pdf")
    plt.close(fig)


def run_model(model_name: str, run_dir: Path, cfg: dict, final_root: Path, device: torch.device, overwrite: bool = False) -> dict:
    selected = SELECTED[model_name]
    layer_name = selected["layer"]
    out_dir = final_root / model_name / f"selected_layer_{layer_name}"
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Final selected Grad-CAM output already exists: {out_dir}")
    for sub in ["raw_npy", "heatmap_png", "overlay_png", "class_average_npy", "class_average_png"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_dir / "inputs" / "test.csv")
    ds = AttributionDataset(df, int(cfg["img_size"]), cfg["imagenet_mean"], cfg["imagenet_std"])
    dl = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        worker_init_fn=seed_worker,
    )
    model = load_checkpoint(build_model(model_name, int(cfg["num_classes"]), pretrained=False), cfg["models"][model_name]["checkpoint"], device)
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"{model_name} selected layer not found: {layer_name}")
    sample = ds[0]["image"].unsqueeze(0).to(device)
    actual_shape = layer_shape(model, modules[layer_name], sample)
    print(f"{model_name}: selected layer {layer_name}, activation shape {actual_shape}")
    gradcam = GradCAM(model, modules[layer_name], relu=True)

    records = []
    failed = []
    class_maps = {cls: [] for cls in ["ARSE", "ARSS", "PJNA", "PRDA", "PRPE"]}
    for batch in tqdm(dl, desc=f"{model_name} selected Grad-CAM"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            _, pred, conf = predict_logits(model, images)
        try:
            cams = gradcam(images, pred, out_size=(int(cfg["img_size"]), int(cfg["img_size"])))
        except Exception as exc:
            for stem, cls in zip(batch["stem"], batch["class"]):
                failed.append({"stem": str(stem), "class": str(cls), "error": repr(exc)})
            continue
        for i, stem in enumerate(batch["stem"]):
            stem = str(stem)
            cls = str(batch["class"][i])
            raw = cams[i].astype(np.float32)
            cam01 = minmax01(raw)
            mask = batch["mask"][i].numpy().astype(bool)
            masked = np.where(mask, cam01, 0.0).astype(np.float32)
            layer_for_file = layer_name
            raw_path = out_dir / "raw_npy" / f"{model_name}_{layer_for_file}_{stem}_raw.npy"
            heat_path = out_dir / "heatmap_png" / f"{model_name}_{layer_for_file}_{stem}_heatmap.png"
            overlay_path = out_dir / "overlay_png" / f"{model_name}_{layer_for_file}_{stem}_overlay.png"
            np.save(raw_path, raw)
            save_heatmap(masked, mask, heat_path)
            save_overlay(batch["rgb"][i].numpy(), masked, mask, overlay_path)
            pred_label = int(pred[i].detach().cpu())
            true_label = int(labels[i].detach().cpu())
            if pred_label == true_label:
                class_maps[cls].append(cam01)
            records.append(
                {
                    "stem": stem,
                    "filepath": str(batch["filepath"][i]),
                    "maskpath": str(batch["maskpath"][i]),
                    "true_class": cls,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_class": cfg["class_order"][pred_label],
                    "confidence": float(conf[i].detach().cpu()),
                    "target_label": pred_label,
                    "target_class": cfg["class_order"][pred_label],
                    "model": model_name,
                    "selected_layer": layer_name,
                    "activation_shape": json.dumps(actual_shape),
                    "raw_shape": json.dumps(list(raw.shape)),
                    "raw_min": float(np.nanmin(raw)),
                    "raw_max": float(np.nanmax(raw)),
                    "raw_mean": float(np.nanmean(raw)),
                    "raw_std": float(np.nanstd(raw)),
                    "nan_count": int(np.isnan(raw).sum()),
                    "inf_count": int(np.isinf(raw).sum()),
                    "all_zero": bool(np.all(raw == 0)),
                    "near_constant": bool(np.nanstd(raw) < 1e-8),
                    "foreground_coverage_top20": foreground_coverage_top20(cam01, mask),
                    "raw_npy": str(raw_path),
                    "heatmap_png": str(heat_path),
                    "overlay_png": str(overlay_path),
                }
            )
    gradcam.remove()
    meta = pd.DataFrame(records)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    pd.DataFrame(failed, columns=["stem", "class", "error"]).to_csv(out_dir / "failed_samples.csv", index=False)

    class_avg_rows = []
    for cls, maps in class_maps.items():
        if not maps:
            continue
        avg = np.mean(np.stack(maps), axis=0).astype(np.float32)
        avg_path = out_dir / "class_average_npy" / f"{model_name}_{layer_name}_{cls}_class_average.npy"
        heat_path = out_dir / "class_average_png" / f"{model_name}_{layer_name}_{cls}_class_average.png"
        np.save(avg_path, avg)
        save_heatmap(minmax01(avg), np.ones_like(avg, dtype=bool), heat_path)
        class_avg_rows.append({"class": cls, "n_correct": len(maps), "class_average_npy": str(avg_path), "class_average_png": str(heat_path)})
    pd.DataFrame(class_avg_rows).to_csv(out_dir / "class_average_metadata.csv", index=False)
    make_grid(meta, out_dir, model_name, layer_name, int(cfg["img_size"]))

    summary = {
        "model": model_name,
        "selected_layer": layer_name,
        "activation_shape": actual_shape,
        "expected_shape": selected["expected_shape"],
        "n_test": int(len(df)),
        "successful_maps": int(len(meta)),
        "failed_maps": int(len(failed)),
        "nan_maps": int((meta["nan_count"] > 0).sum()) if not meta.empty else 0,
        "inf_maps": int((meta["inf_count"] > 0).sum()) if not meta.empty else 0,
        "all_zero_maps": int(meta["all_zero"].sum()) if not meta.empty else 0,
        "near_constant_maps": int(meta["near_constant"].sum()) if not meta.empty else 0,
        "foreground_coverage_mean": float(meta["foreground_coverage_top20"].mean()) if not meta.empty else None,
        "foreground_coverage_std": float(meta["foreground_coverage_top20"].std()) if not meta.empty else None,
        "raw_map_dir": str(out_dir / "raw_npy"),
        "overlay_dir": str(out_dir / "overlay_png"),
        "heatmap_dir": str(out_dir / "heatmap_png"),
        "class_average_dir": str(out_dir / "class_average_npy"),
        "final_output_path": str(out_dir),
        "note": selected["note"],
        "usage": selected["usage"],
    }
    save_json(summary, out_dir / "generation_summary.json")
    return summary


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    cfg = load_yaml(run_dir / "config.yaml")
    set_seed(int(cfg["seed"]))
    final_root = run_dir / "03_gradcam_final_selected_layers"
    final_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")

    summaries = []
    for model_name in ["convnext_small", "resnet50", "efficientnet_b0"]:
        summaries.append(run_model(model_name, run_dir, cfg, final_root, device, overwrite=args.overwrite))

    summary_df = pd.DataFrame(
        [
            {
                "model": s["model"],
                "selected_layer": s["selected_layer"],
                "number_of_test_images": s["n_test"],
                "successful_maps": s["successful_maps"],
                "failed_maps": s["failed_maps"],
                "nan_maps": s["nan_maps"],
                "all_zero_maps": s["all_zero_maps"],
                "near_constant_maps": s["near_constant_maps"],
                "foreground_coverage_mean": s["foreground_coverage_mean"],
                "foreground_coverage_std": s["foreground_coverage_std"],
                "raw_map_dir": s["raw_map_dir"],
                "overlay_dir": s["overlay_dir"],
                "class_average_dir": s["class_average_dir"],
                "usage": s["usage"],
            }
            for s in summaries
        ]
    )
    summary_df.to_csv(final_root / "gradcam_final_generation_summary.csv", index=False)
    final_json = {
        "stage2_run_dir": str(run_dir),
        "stage3_convnext_gradcam_raw_map_dir": next(s["raw_map_dir"] for s in summaries if s["model"] == "convnext_small"),
        "models": summaries,
        "note": "Final selected-layer Grad-CAM outputs generated without modifying Stage 1 checkpoints, canonical IG, or previous Grad-CAM folders.",
    }
    save_json(final_json, final_root / "selected_layers_final.json")
    readme = [
        "# Final Selected-Layer Grad-CAM Outputs",
        "",
        "This folder contains full-test Grad-CAM outputs generated with the selected target layer for each model. Previous full-test Grad-CAM folders were not deleted or overwritten.",
        "",
        "The previous ConvNeXt-Small full-test Grad-CAM used `stages.3.blocks.2`, while the selected layer is `stages.2.blocks.26`; therefore a new full-test output was required.",
        "The previous EfficientNet-B0 full-test Grad-CAM used `conv_head`, while the selected layer is `blocks.4`; therefore a new full-test output was required.",
        "ResNet50 was regenerated with `layer4.2` for consistent naming, turbo visualization, and final manifest registration.",
        "",
        "## Stage 3 ConvNeXt Grad-CAM Input",
        "",
        f"`{final_json['stage3_convnext_gradcam_raw_map_dir']}`",
        "",
        "## Model Outputs",
        "",
    ]
    for s in summaries:
        readme.append(f"- {s['model']}: layer `{s['selected_layer']}`, raw maps `{s['raw_map_dir']}`, foreground coverage mean ± SD `{s['foreground_coverage_mean']} ± {s['foreground_coverage_std']}`")
    readme += [
        "",
        "## Caution",
        "",
        "EfficientNet-B0 Grad-CAM had low foreground coverage in candidate screening and should be treated as supplementary comparison only.",
        "Grad-CAM outputs are localization diagnostics and should not be described as causal evidence.",
    ]
    (final_root / "README_FINAL_GRADCAM.md").write_text("\n".join(readme), encoding="utf-8")

    root_selected = run_dir / "selected_layers.json"
    if root_selected.exists():
        data = json.loads(root_selected.read_text())
        data["final_selected_gradcam_outputs"] = {
            s["model"]: {
                "selected_layer": s["selected_layer"],
                "activation_shape": s["activation_shape"],
                "raw_map_dir": s["raw_map_dir"],
                "final_output_path": s["final_output_path"],
                "foreground_coverage_mean": s["foreground_coverage_mean"],
                "foreground_coverage_std": s["foreground_coverage_std"],
            }
            for s in summaries
        }
        root_selected.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    stage_summary = run_dir / "stage2_run_summary.md"
    if stage_summary.exists():
        text = stage_summary.read_text(encoding="utf-8")
        section = "\n\n## Final Selected-Layer Grad-CAM Update\n\n"
        section += f"- Final selected Grad-CAM root: `{final_root}`\n"
        section += f"- Stage 3 ConvNeXt Grad-CAM raw map path: `{final_json['stage3_convnext_gradcam_raw_map_dir']}`\n"
        for s in summaries:
            section += f"- {s['model']}: selected layer `{s['selected_layer']}`, successful maps {s['successful_maps']}/{s['n_test']}, foreground coverage mean ± SD {s['foreground_coverage_mean']} ± {s['foreground_coverage_std']}\n"
        section += "- Existing previous Grad-CAM folders were retained and not overwritten.\n"
        if "## Final Selected-Layer Grad-CAM Update" not in text:
            stage_summary.write_text(text.rstrip() + section + "\n", encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(final_root / "selected_layers_final.json")
    print(final_json["stage3_convnext_gradcam_raw_map_dir"])


if __name__ == "__main__":
    main()
