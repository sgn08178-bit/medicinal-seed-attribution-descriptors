#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import AttributionDataset
from src.io_utils import load_yaml, save_json
from src.models import build_model, load_checkpoint
from src.seed_utils import set_seed
from src.visualization import minmax01


MODEL_CANDIDATES = {
    "convnext_small": [
        ("final stage last block", "stages.3.blocks.2", "High-level final stage candidate; expected to be class-discriminative but coarse."),
        ("one stage before final stage last block", "stages.2.blocks.26", "Intermediate candidate; expected to balance spatial detail and class specificity."),
        ("two stages before final stage last block", "stages.1.blocks.2", "Higher-resolution candidate; may be detailed but noisier."),
    ],
    "resnet50": [
        ("layer4 last block", "layer4.2", "Class-discriminative final residual stage candidate; expected 7x7 and potentially blocky."),
        ("layer3 last block", "layer3.5", "Intermediate residual stage candidate; expected 14x14 balance layer."),
        ("layer2 last block", "layer2.3", "Higher-resolution residual stage candidate; expected 28x28 and potentially less class-specific."),
    ],
    "efficientnet_b0": [
        ("classifier-preceding conv head", "conv_head", "Classifier-preceding spatial convolution candidate; may be coarse."),
        ("last EfficientNet block", "blocks.6", "Last MBConv stage before conv head."),
        ("one block group before last", "blocks.5", "Earlier MBConv stage expected to retain more spatial detail."),
        ("two block groups before last", "blocks.4", "Earlier MBConv stage candidate; may be detailed but less class-specific."),
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None)
    p.add_argument("--alpha", type=float, default=0.40)
    p.add_argument("--colormap", default="turbo", choices=["turbo", "jet"])
    return p.parse_args()


def latest_run_dir() -> Path:
    candidates = sorted([p for p in (ROOT / "runs").glob("stage2_attribution_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No Stage 2 run directory found under {ROOT / 'runs'}")
    return candidates[-1]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "-")


def output_shape(model: torch.nn.Module, module: torch.nn.Module, sample: torch.Tensor) -> list[int] | None:
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
    if not torch.is_tensor(out):
        return None
    return list(out.shape)


class LayerGradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            layer.register_forward_hook(self._forward_hook),
            layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output[0] if isinstance(output, (tuple, list)) else output

    def _backward_hook(self, _module, _grad_input, grad_output):
        grad = grad_output[0]
        self.gradients = grad[0] if isinstance(grad, (tuple, list)) else grad

    def remove(self):
        for h in self.handles:
            h.remove()

    def __call__(self, image: torch.Tensor, target: torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        score = logits.gather(1, target.view(-1, 1)).sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook failed.")
        if self.activations.ndim != 4:
            raise RuntimeError(f"Target activation is not BCHW spatial: {tuple(self.activations.shape)}")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        return cam.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def save_overlay(rgb_chw: np.ndarray, cam01: np.ndarray, mask: np.ndarray, path: Path, alpha: float, cmap_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.transpose(rgb_chw, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)
    heat = plt.get_cmap(cmap_name)(np.clip(cam01, 0, 1))[..., :3]
    m = mask.astype(bool)[..., None]
    overlay = np.where(m, (1.0 - alpha) * rgb + alpha * heat, rgb)
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(path)


def save_heatmap(cam01: np.ndarray, mask: np.ndarray, path: Path, cmap_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heat = plt.get_cmap(cmap_name)(np.clip(cam01, 0, 1))[..., :3]
    heat = np.where(mask.astype(bool)[..., None], heat, 0.0)
    Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).save(path)


def foreground_coverage_top20(cam01: np.ndarray, mask: np.ndarray) -> float:
    flat = cam01.reshape(-1)
    k = max(1, int(np.ceil(flat.size * 0.20)))
    top_idx = np.argpartition(flat, -k)[-k:]
    return float(mask.astype(bool).reshape(-1)[top_idx].mean())


def read_stage1_predictions(stage1_root: Path, model_name: str) -> pd.DataFrame:
    path = stage1_root / model_name / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage 1 test_predictions.csv not found: {path}")
    return pd.read_csv(path)


def select_representatives(pred_df: pd.DataFrame, class_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    correct = pred_df[pred_df["true_label"] == pred_df["pred_label"]].copy()
    reps = []
    for cls in class_order:
        class_df = correct[correct["true_label"] == cls].sort_values("confidence", ascending=False)
        if class_df.empty:
            raise RuntimeError(f"No correctly classified sample found for {cls}.")
        high = class_df.iloc[0].copy()
        high["selection_type"] = "high_confidence_correct"
        reps.append(high)
        mid_idx = len(class_df) // 2
        mid = class_df.iloc[mid_idx].copy()
        mid["selection_type"] = "middle_confidence_correct"
        if str(mid["stem"]) != str(high["stem"]):
            reps.append(mid)
    rep_df = pd.DataFrame(reps).drop_duplicates("stem").reset_index(drop=True)
    mis_df = pred_df[pred_df["true_label"] != pred_df["pred_label"]].copy().reset_index(drop=True)
    return rep_df, mis_df


def layer_visual_note(shape: list[int], mean_cov: float) -> str:
    h, w = int(shape[-2]), int(shape[-1])
    notes = []
    if h <= 7 or w <= 7:
        notes.append("coarse spatial grid; possible blocky artifact")
    elif h >= 28 or w >= 28:
        notes.append("fine spatial grid; inspect for point-like noise or weak class specificity")
    else:
        notes.append("intermediate spatial grid; balance candidate")
    if mean_cov < 0.80:
        notes.append("relatively low top-20% foreground coverage")
    return "; ".join(notes)


def choose_recommendation(layer_summary: pd.DataFrame) -> str:
    valid = layer_summary.copy()
    valid["h"] = valid["activation_h"].astype(int)
    valid["score"] = valid["mean_foreground_coverage_top20"].astype(float)
    valid.loc[valid["h"] <= 7, "score"] -= 0.03
    valid.loc[valid["h"] >= 28, "score"] -= 0.05
    valid = valid.sort_values(["score", "mean_foreground_coverage_top20"], ascending=False)
    return str(valid.iloc[0]["candidate_layer_name"])


def make_grid(model_name: str, out_dir: Path, rep_ds: AttributionDataset, candidates: list[dict], overlay_lookup: dict[tuple[str, str], Path], layer_summary: pd.DataFrame) -> None:
    nrows = len(rep_ds)
    ncols = len(candidates) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.65 * nrows), dpi=300)
    if nrows == 1:
        axes = axes[None, :]
    mean_cov = dict(zip(layer_summary["candidate_layer_name"], layer_summary["mean_foreground_coverage_top20"]))
    for i in range(nrows):
        item = rep_ds[i]
        stem = item["stem"]
        cls = item["class"]
        rgb = np.transpose(item["rgb"].numpy(), (1, 2, 0))
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(f"{cls}\n{stem}", fontsize=8, fontweight="bold")
        axes[i, 0].set_title("Input", fontsize=9)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for j, cand in enumerate(candidates, start=1):
            img = Image.open(overlay_lookup[(cand["name"], stem)]).convert("RGB")
            axes[i, j].imshow(img)
            title = f"{cand['label']}\n{cand['name']}\n{cand['shape']} | mean cov={mean_cov[cand['name']]:.3f}"
            axes[i, j].set_title(title, fontsize=7)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle(f"{model_name} Grad-CAM target layer candidates", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    grid_dir = out_dir / "candidate_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(grid_dir / f"{model_name}_gradcam_layer_candidate_grid.png", dpi=300)
    fig.savefig(grid_dir / f"{model_name}_gradcam_layer_candidate_grid.pdf")
    plt.close(fig)


def run_model(model_name: str, run_dir: Path, all_out: Path, cfg: dict, device: torch.device) -> dict:
    model_out = all_out / model_name
    overlay_dir = model_out / "candidate_overlays"
    heat_dir = model_out / "candidate_heatmaps"
    raw_dir = model_out / "candidate_raw_npy"
    mis_dir = model_out / "misclassified_samples"
    for d in [overlay_dir, heat_dir, raw_dir, model_out / "candidate_grids", mis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(build_model(model_name, cfg["num_classes"], pretrained=False), cfg["models"][model_name]["checkpoint"], device)
    modules = dict(model.named_modules())
    stage1_root = Path(cfg["stage1_runs_root"])
    pred_df = read_stage1_predictions(stage1_root, model_name)
    reps, mis_df = select_representatives(pred_df, cfg["class_order"])
    reps.to_csv(model_out / "representative_samples.csv", index=False)
    mis_df.to_csv(mis_dir / "misclassified_samples.csv", index=False)
    if not mis_df.empty:
        for _, row in mis_df.iterrows():
            Image.open(row["filepath"]).convert("RGB").resize((cfg["img_size"], cfg["img_size"]), Image.Resampling.BILINEAR).save(mis_dir / f"{row['stem']}_input.png")

    rep_ds = AttributionDataset(reps, cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"])
    first_df = pd.read_csv(run_dir / "inputs" / "test.csv").iloc[:1].copy()
    first_ds = AttributionDataset(first_df, cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"])
    sample = first_ds[0]["image"].unsqueeze(0).to(device)

    valid_candidates = []
    candidate_rows = []
    for label, name, notes in MODEL_CANDIDATES[model_name]:
        row = {"model": model_name, "candidate_label": label, "candidate_layer_name": name, "notes": notes}
        if name not in modules:
            row.update({"activation_shape": None, "valid": False, "exclusion_reason": "module not found"})
            candidate_rows.append(row)
            continue
        shape = output_shape(model, modules[name], sample)
        valid = bool(shape and len(shape) == 4 and shape[-1] > 1 and shape[-2] > 1)
        row.update({"activation_shape": json.dumps(shape), "valid": valid, "exclusion_reason": "" if valid else "not BCHW spatial activation"})
        candidate_rows.append(row)
        print(f"{model_name} candidate: {name}, shape={shape}, valid={valid}")
        if valid:
            valid_candidates.append({"label": label, "name": name, "notes": notes, "shape": shape})
    if not valid_candidates:
        raise RuntimeError(f"No valid Grad-CAM target candidate for {model_name}")

    report_rows = []
    overlay_lookup = {}
    for cand in valid_candidates:
        gradcam = LayerGradCAM(model, modules[cand["name"]])
        layer_safe = safe_name(cand["name"])
        for i in range(len(rep_ds)):
            item = rep_ds[i]
            stem = str(item["stem"])
            image = item["image"].unsqueeze(0).to(device)
            target_label = int(reps.iloc[i]["pred_label"] if str(reps.iloc[i]["pred_label"]).isdigit() else cfg["class_order"].index(reps.iloc[i]["pred_label"]))
            target = torch.tensor([target_label], device=device)
            raw = gradcam(image, target, (cfg["img_size"], cfg["img_size"]))
            cam01 = minmax01(raw)
            mask = item["mask"].numpy().astype(bool)
            masked = np.where(mask, cam01, 0.0).astype(np.float32)
            cov = foreground_coverage_top20(cam01, mask)
            raw_path = raw_dir / f"{model_name}_{layer_safe}_{stem}_raw.npy"
            overlay_path = overlay_dir / f"{model_name}_{layer_safe}_{stem}_overlay.png"
            heat_path = heat_dir / f"{model_name}_{layer_safe}_{stem}_masked_heatmap.png"
            np.save(raw_path, raw.astype(np.float32))
            save_overlay(item["rgb"].numpy(), masked, mask, overlay_path, 0.40, "turbo")
            save_heatmap(masked, mask, heat_path, "turbo")
            overlay_lookup[(cand["name"], stem)] = overlay_path
            report_rows.append(
                {
                    "model": model_name,
                    "candidate_label": cand["label"],
                    "candidate_layer_name": cand["name"],
                    "activation_shape": json.dumps(cand["shape"]),
                    "activation_h": int(cand["shape"][-2]),
                    "activation_w": int(cand["shape"][-1]),
                    "sample_id": stem,
                    "class": item["class"],
                    "selection_type": reps.iloc[i]["selection_type"],
                    "gradcam_map_shape": json.dumps(list(raw.shape)),
                    "foreground_coverage_top20": cov,
                    "raw_npy": str(raw_path),
                    "overlay_png": str(overlay_path),
                    "heatmap_png": str(heat_path),
                    "selected": False,
                    "notes": cand["notes"],
                }
            )
        gradcam.remove()

    report = pd.DataFrame(report_rows)
    layer_summary = (
        report.groupby(["model", "candidate_label", "candidate_layer_name", "activation_shape", "activation_h", "activation_w"], as_index=False)
        .agg(
            n=("sample_id", "count"),
            mean_foreground_coverage_top20=("foreground_coverage_top20", "mean"),
            std_foreground_coverage_top20=("foreground_coverage_top20", "std"),
        )
    )
    recommended = choose_recommendation(layer_summary)
    report["mean_foreground_coverage_top20_for_layer"] = report["candidate_layer_name"].map(dict(zip(layer_summary["candidate_layer_name"], layer_summary["mean_foreground_coverage_top20"])))
    report["std_foreground_coverage_top20_for_layer"] = report["candidate_layer_name"].map(dict(zip(layer_summary["candidate_layer_name"], layer_summary["std_foreground_coverage_top20"])))
    report["selected"] = report["candidate_layer_name"] == recommended
    layer_summary["selected"] = layer_summary["candidate_layer_name"] == recommended
    layer_summary["visual_notes"] = [layer_visual_note(json.loads(shape), cov) for shape, cov in zip(layer_summary["activation_shape"], layer_summary["mean_foreground_coverage_top20"])]
    report = report.merge(layer_summary[["candidate_layer_name", "visual_notes"]], on="candidate_layer_name", how="left")
    report.to_csv(model_out / "selected_layer_report.csv", index=False)

    make_grid(model_name, model_out, rep_ds, valid_candidates, overlay_lookup, layer_summary)
    json_report = {
        "model": model_name,
        "recommended_layer_for_review": recommended,
        "note": "Recommendation is recorded for review only; existing final Grad-CAM outputs were not overwritten.",
        "candidate_layers": layer_summary.to_dict("records"),
        "invalid_or_excluded_candidates": [r for r in candidate_rows if not r.get("valid")],
        "representative_samples_csv": str(model_out / "representative_samples.csv"),
        "misclassified_samples_csv": str(mis_dir / "misclassified_samples.csv"),
        "grid_png": str(model_out / "candidate_grids" / f"{model_name}_gradcam_layer_candidate_grid.png"),
        "grid_pdf": str(model_out / "candidate_grids" / f"{model_name}_gradcam_layer_candidate_grid.pdf"),
    }
    save_json(json_report, model_out / "selected_layer_report.json")
    return json_report


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    cfg = load_yaml(run_dir / "config.yaml")
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    all_out = run_dir / "gradcam_layer_selection_all_models"
    if all_out.exists():
        raise FileExistsError(f"Output directory already exists and will not be overwritten: {all_out}")
    all_out.mkdir(parents=True)

    reports = []
    for model_name in ["convnext_small", "resnet50", "efficientnet_b0"]:
        reports.append(run_model(model_name, run_dir, all_out, cfg, device))

    summary_rows = []
    for report in reports:
        for row in report["candidate_layers"]:
            summary_rows.append(
                {
                    "model": report["model"],
                    "candidate_layer_name": row["candidate_layer_name"],
                    "candidate_label": row["candidate_label"],
                    "activation_shape": row["activation_shape"],
                    "n": row["n"],
                    "foreground_coverage_mean": row["mean_foreground_coverage_top20"],
                    "foreground_coverage_std": row["std_foreground_coverage_top20"],
                    "selected": row["selected"],
                    "visual_notes": row["visual_notes"],
                    "grid_png": report["grid_png"],
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_dir = all_out / "summary"
    summary_dir.mkdir(parents=True)
    summary_df.to_csv(summary_dir / "all_model_layer_selection_summary.csv", index=False)
    save_json({"run_dir": str(run_dir), "reports": reports}, summary_dir / "all_model_layer_selection_summary.json")
    readme = [
        "# Grad-CAM Layer Selection Across Models",
        "",
        "This folder contains target-layer candidate comparisons only. Stage 1 checkpoints and existing Stage 2 final Grad-CAM outputs were not overwritten.",
        "",
        "Visualization settings: per-image min-max normalization, foreground-masked heatmap, RGB overlay alpha 0.40, turbo colormap.",
        "",
        "Recommended layers are suggestions for user review, not applied to final Grad-CAM outputs.",
        "",
    ]
    for report in reports:
        readme.append(f"- {report['model']}: recommended `{report['recommended_layer_for_review']}`; grid `{report['grid_png']}`")
    (summary_dir / "README_layer_selection.md").write_text("\n".join(readme), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"Saved: {all_out}")


if __name__ == "__main__":
    main()

