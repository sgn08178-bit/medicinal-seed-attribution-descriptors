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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import AttributionDataset
from src.io_utils import load_yaml, save_json
from src.models import build_model, load_checkpoint, predict_logits
from src.seed_utils import set_seed
from src.visualization import minmax01


CANDIDATES = [
    {
        "label": "final stage last block",
        "name": "stages.3.blocks.2",
        "notes": "Final ConvNeXt stage last block before classifier head; high-level but coarse 7x7 spatial grid.",
    },
    {
        "label": "one stage before final stage last block",
        "name": "stages.2.blocks.26",
        "notes": "Last block of the preceding ConvNeXt stage; expected to retain a finer 14x14 spatial grid.",
    },
    {
        "label": "two stages before final stage last block",
        "name": "stages.1.blocks.2",
        "notes": "Last block two stages before the final stage; expected to retain a finer 28x28 spatial grid.",
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None, help="Stage 2 run directory. If omitted, the latest run under ROOT/runs is used.")
    p.add_argument("--alpha", type=float, default=0.40)
    p.add_argument("--colormap", default="turbo", choices=["turbo", "jet"])
    return p.parse_args()


def latest_run_dir() -> Path:
    runs_root = ROOT / "runs"
    candidates = sorted([p for p in runs_root.glob("stage2_attribution_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No Stage 2 run directory found under {runs_root}")
    return candidates[-1]


def safe_layer_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "-")


def module_output_shape(model: torch.nn.Module, module: torch.nn.Module, sample: torch.Tensor) -> list[int] | None:
    captured = {}
    handle = module.register_forward_hook(lambda _m, _i, o: captured.setdefault("out", o))
    try:
        with torch.no_grad():
            _ = model(sample)
    finally:
        handle.remove()
    out = captured.get("out")
    if isinstance(out, (list, tuple)):
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
        self.activations = output[0] if isinstance(output, (list, tuple)) else output

    def _backward_hook(self, _module, _grad_input, grad_output):
        grad = grad_output[0]
        self.gradients = grad[0] if isinstance(grad, (list, tuple)) else grad

    def remove(self):
        for h in self.handles:
            h.remove()

    def __call__(self, image: torch.Tensor, target: torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        score = logits.gather(1, target.view(-1, 1)).sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook failed to capture activation or gradient.")
        if self.activations.ndim != 4:
            raise RuntimeError(f"Target activation is not spatial BCHW: {tuple(self.activations.shape)}")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        return cam.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def save_overlay(rgb_chw: np.ndarray, cam01: np.ndarray, mask: np.ndarray, path: Path, alpha: float, cmap_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.transpose(rgb_chw, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)
    cmap = plt.get_cmap(cmap_name)
    heat = cmap(np.clip(cam01, 0, 1))[..., :3]
    m = mask.astype(bool)[..., None]
    out = np.where(m, (1.0 - alpha) * rgb + alpha * heat, rgb)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(path)


def save_masked_heatmap(cam01: np.ndarray, mask: np.ndarray, path: Path, cmap_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(cmap_name)
    heat = cmap(np.clip(cam01, 0, 1))[..., :3]
    heat = np.where(mask.astype(bool)[..., None], heat, 0.0)
    Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).save(path)


def foreground_coverage_top20(cam01: np.ndarray, mask: np.ndarray) -> float:
    flat = cam01.reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(np.ceil(flat.size * 0.20)))
    top_idx = np.argpartition(flat, -k)[-k:]
    fg_flat = mask.astype(bool).reshape(-1)
    return float(fg_flat[top_idx].mean())


def choose_representatives(df: pd.DataFrame, cfg: dict, model: torch.nn.Module, device: torch.device) -> pd.DataFrame:
    ds = AttributionDataset(df, cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"])
    rows = []
    for idx in range(len(ds)):
        item = ds[idx]
        image = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            _, pred, conf = predict_logits(model, image)
        if int(pred.item()) == int(item["label"]):
            row = df.iloc[idx].copy()
            row["pred_label"] = int(pred.item())
            row["confidence"] = float(conf.item())
            rows.append(row)
    pred_df = pd.DataFrame(rows)
    selected = []
    for cls in cfg["class_order"]:
        class_df = pred_df[pred_df["class"] == cls].sort_values("confidence", ascending=False)
        if class_df.empty:
            raise RuntimeError(f"No correctly classified representative sample found for class {cls}.")
        selected.append(class_df.iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    cfg = load_yaml(run_dir / "config.yaml")
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    out = run_dir / "convnext_small" / "gradcam_layer_selection"
    overlay_dir = out / "candidate_overlays"
    raw_dir = out / "candidate_raw_npy"
    heat_dir = out / "candidate_heatmaps"
    for d in [overlay_dir, raw_dir, heat_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_dir / "inputs" / "test.csv")
    model = load_checkpoint(
        build_model("convnext_small", cfg["num_classes"], pretrained=False),
        cfg["models"]["convnext_small"]["checkpoint"],
        device,
    )
    modules = dict(model.named_modules())
    first_ds = AttributionDataset(df.iloc[:1].copy(), cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"])
    sample = first_ds[0]["image"].unsqueeze(0).to(device)

    valid_candidates = []
    for cand in CANDIDATES:
        name = cand["name"]
        if name not in modules:
            cand["activation_shape"] = None
            cand["valid"] = False
            cand["notes"] += " Module not found."
            continue
        shape = module_output_shape(model, modules[name], sample)
        cand["activation_shape"] = shape
        cand["valid"] = bool(shape and len(shape) == 4 and shape[-1] > 1 and shape[-2] > 1)
        if not cand["valid"]:
            cand["notes"] += " Excluded because output is not BCHW spatial activation."
        else:
            valid_candidates.append(cand)
        print(f"candidate: {name}, shape={shape}, valid={cand['valid']}")
    if not valid_candidates:
        raise RuntimeError("No valid ConvNeXt candidate target layers found.")

    reps = choose_representatives(df, cfg, model, device)
    reps.to_csv(out / "representative_samples.csv", index=False)
    rep_ds = AttributionDataset(reps, cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"])
    report_rows = []
    coverage_by_layer: dict[str, list[float]] = {c["name"]: [] for c in valid_candidates}
    overlay_paths: dict[tuple[str, str], Path] = {}

    for cand in valid_candidates:
        name = cand["name"]
        safe = safe_layer_name(name)
        gradcam = LayerGradCAM(model, modules[name])
        for i in range(len(rep_ds)):
            item = rep_ds[i]
            stem = item["stem"]
            cls = item["class"]
            image = item["image"].unsqueeze(0).to(device)
            target = torch.tensor([int(item["label"])], device=device)
            raw = gradcam(image, target, out_size=(cfg["img_size"], cfg["img_size"]))
            cam01 = minmax01(raw)
            mask = item["mask"].numpy().astype(bool)
            masked_cam = np.where(mask, cam01, 0.0).astype(np.float32)
            coverage = foreground_coverage_top20(cam01, mask)
            coverage_by_layer[name].append(coverage)
            raw_path = raw_dir / f"convnext_small_{safe}_{stem}_raw.npy"
            overlay_path = overlay_dir / f"convnext_small_{safe}_{stem}_overlay.png"
            heat_path = heat_dir / f"convnext_small_{safe}_{stem}_masked_heatmap.png"
            np.save(raw_path, raw.astype(np.float32))
            save_overlay(item["rgb"].numpy(), masked_cam, mask, overlay_path, args.alpha, args.colormap)
            save_masked_heatmap(masked_cam, mask, heat_path, args.colormap)
            overlay_paths[(name, stem)] = overlay_path
            report_rows.append(
                {
                    "candidate_label": cand["label"],
                    "candidate_layer_name": name,
                    "activation_shape": json.dumps(cand["activation_shape"]),
                    "sample_id": stem,
                    "class": cls,
                    "gradcam_map_shape": json.dumps(list(raw.shape)),
                    "foreground_coverage_top20": coverage,
                    "raw_npy": str(raw_path),
                    "overlay_png": str(overlay_path),
                    "notes": cand["notes"],
                    "selected": False,
                }
            )
        gradcam.remove()

    mean_cov = {name: float(np.mean(vals)) for name, vals in coverage_by_layer.items()}
    recommended = max(mean_cov, key=mean_cov.get)
    for row in report_rows:
        row["mean_foreground_coverage_top20_for_layer"] = mean_cov[row["candidate_layer_name"]]
        row["selected"] = row["candidate_layer_name"] == recommended
        if row["selected"]:
            row["notes"] = row["notes"] + " Recommended for user review based on highest mean foreground top-20% coverage among candidates."

    report = pd.DataFrame(report_rows)
    report.to_csv(out / "selected_layer_report.csv", index=False)
    json_report = {
        "model": "convnext_small",
        "purpose": "Grad-CAM target layer candidate comparison without overwriting existing Grad-CAM results.",
        "colormap": args.colormap,
        "alpha": args.alpha,
        "candidates": [
            {
                "candidate_label": c["label"],
                "candidate_layer_name": c["name"],
                "activation_shape": c.get("activation_shape"),
                "valid": c.get("valid", False),
                "mean_foreground_coverage_top20": mean_cov.get(c["name"]),
                "selected": c["name"] == recommended,
                "notes": c["notes"],
            }
            for c in CANDIDATES
        ],
        "recommended_layer_for_review": recommended,
        "representative_samples_csv": str(out / "representative_samples.csv"),
    }
    save_json(json_report, out / "selected_layer_report.json")

    nrows = len(rep_ds)
    ncols = len(valid_candidates) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), dpi=300)
    if nrows == 1:
        axes = axes[None, :]
    for i in range(nrows):
        item = rep_ds[i]
        stem = item["stem"]
        cls = item["class"]
        rgb = np.transpose(item["rgb"].numpy(), (1, 2, 0))
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(cls, fontsize=12, fontweight="bold")
        axes[i, 0].set_title("Input", fontsize=11)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for j, cand in enumerate(valid_candidates, start=1):
            img = Image.open(overlay_paths[(cand["name"], stem)]).convert("RGB")
            axes[i, j].imshow(img)
            title = cand["label"].replace(" before ", "\nbefore ")
            axes[i, j].set_title(f"{title}\n{cand['activation_shape']}", fontsize=9)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle("ConvNeXt-Small Grad-CAM target layer candidates", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "convnext_gradcam_layer_candidate_grid.png", dpi=300)
    fig.savefig(out / "convnext_gradcam_layer_candidate_grid.pdf")
    plt.close(fig)

    print(f"Recommended layer for review: {recommended}")
    print(json.dumps(json_report, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
