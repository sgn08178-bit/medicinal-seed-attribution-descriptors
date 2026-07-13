#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attribution_ig import compute_ig_for_batch
from src.dataset import AttributionDataset
from src.io_utils import load_yaml, save_json
from src.models import build_model, load_checkpoint, predict_logits
from src.seed_utils import seed_worker, set_seed
from src.visualization import save_heatmap, save_overlay, vis_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    mcfg = cfg["models"]["convnext_small"]
    if not mcfg.get("compute_ig", False):
        print("ConvNeXt-Small IG disabled in config.")
        return

    out = run_dir / "convnext_small" / "ig"
    raw_dir = out / "raw_npy"
    heat_dir = out / "heatmap_png"
    overlay_dir = out / "overlay_png"
    avg_npy_dir = out / "class_average_npy"
    avg_png_dir = out / "class_average_png"
    for d in [raw_dir, heat_dir, overlay_dir, avg_npy_dir, avg_png_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_dir / "inputs" / "test.csv")
    ds = AttributionDataset(df, cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"])
    dl = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        worker_init_fn=seed_worker,
    )
    model = load_checkpoint(build_model("convnext_small", cfg["num_classes"], pretrained=False), mcfg["checkpoint"], device)
    ig_cfg = cfg["integrated_gradients"]
    vis_cfg = cfg["visualization"]
    records = []
    class_maps: dict[str, list[np.ndarray]] = {c: [] for c in cfg["class_order"]}

    for batch in tqdm(dl, desc="ConvNeXt IG"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            _, pred, conf = predict_logits(model, images)
        targets = pred if ig_cfg.get("use_predicted_class", True) else labels
        attrs_2d, deltas = compute_ig_for_batch(
            model=model,
            images=images,
            targets=targets,
            n_steps=int(ig_cfg["n_steps"]),
            internal_batch_size=ig_cfg.get("internal_batch_size"),
            multiply_by_inputs=bool(ig_cfg.get("multiply_by_inputs", True)),
        )
        for i, stem in enumerate(batch["stem"]):
            stem = str(stem)
            cls = str(batch["class"][i])
            raw = attrs_2d[i]
            mask = batch["mask"][i].numpy().astype(bool)
            rgb = batch["rgb"][i].numpy()
            np.save(raw_dir / f"{stem}.npy", raw)
            vmap = vis_map(raw, mask, vis_cfg["percentile_clip_low"], vis_cfg["percentile_clip_high"], vis_cfg["gaussian_smoothing_sigma"])
            save_heatmap(vmap, heat_dir / f"{stem}_ig_heatmap.png")
            save_overlay(rgb, vmap, mask, overlay_dir / f"{stem}_ig_overlay.png", alpha=float(vis_cfg["alpha"]))
            class_maps[cls].append(raw)
            records.append(
                {
                    "stem": stem,
                    "filepath": str(batch["filepath"][i]),
                    "true_class": cls,
                    "true_label": int(labels[i].detach().cpu()),
                    "pred_label": int(pred[i].detach().cpu()),
                    "pred_class": cfg["class_order"][int(pred[i].detach().cpu())],
                    "confidence": float(conf[i].detach().cpu()),
                    "attribution_target_label": int(targets[i].detach().cpu()),
                    "attribution_target_class": cfg["class_order"][int(targets[i].detach().cpu())],
                    "convergence_delta": float(deltas[i]),
                    "raw_npy": str(raw_dir / f"{stem}.npy"),
                    "overlay_png": str(overlay_dir / f"{stem}_ig_overlay.png"),
                }
            )

    pd.DataFrame(records).to_csv(out / "attribution_metadata.csv", index=False)
    avg_records = []
    for cls in cfg["class_order"]:
        maps = class_maps[cls]
        if not maps:
            continue
        avg = np.mean(np.stack(maps), axis=0).astype(np.float32)
        np.save(avg_npy_dir / f"{cls}_ig_average.npy", avg)
        vavg = vis_map(avg, None, vis_cfg["percentile_clip_low"], vis_cfg["percentile_clip_high"], vis_cfg["gaussian_smoothing_sigma"])
        save_heatmap(vavg, avg_png_dir / f"{cls}_ig_average.png")
        avg_records.append({"model": "convnext_small", "method": "ig", "class": cls, "n": len(maps), "class_average_npy": str(avg_npy_dir / f"{cls}_ig_average.npy")})
    pd.DataFrame(avg_records).to_csv(out / "class_average_summary.csv", index=False)
    save_json(
        {
            "model": "convnext_small",
            "method": "Integrated Gradients",
            "baseline": ig_cfg["baseline"],
            "n_steps": int(ig_cfg["n_steps"]),
            "target": "predicted_class" if ig_cfg.get("use_predicted_class", True) else "true_class",
            "attribution_2d": ig_cfg.get("attribution_2d", "abs_sum_channels"),
            "raw_maps_are_unsmoothed_unclipped": True,
            "visualization_percentile_clipping": [vis_cfg["percentile_clip_low"], vis_cfg["percentile_clip_high"]],
            "visualization_gaussian_smoothing_sigma": vis_cfg["gaussian_smoothing_sigma"],
        },
        out / "ig_settings.json",
    )
    print(f"IG complete: {out}")


if __name__ == "__main__":
    main()

