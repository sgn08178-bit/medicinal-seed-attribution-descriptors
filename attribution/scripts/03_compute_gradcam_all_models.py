#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attribution_gradcam import GradCAM, select_target_layer
from src.dataset import AttributionDataset
from src.io_utils import load_yaml, save_json
from src.models import build_model, load_checkpoint, predict_logits
from src.seed_utils import seed_worker, set_seed
from src.visualization import minmax01, save_heatmap, save_overlay, vis_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def normalize_cam(cam: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    return minmax01(cam, mask)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
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
    first = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))["image"].to(device)
    selected_layers = {}
    vis_cfg = cfg["visualization"]

    for model_name, mcfg in cfg["models"].items():
        if not mcfg.get("compute_gradcam", False):
            continue
        out = run_dir / model_name / "gradcam"
        raw_dir = out / "raw_npy"
        heat_dir = out / "heatmap_png"
        overlay_dir = out / "overlay_png"
        avg_npy_dir = out / "class_average_npy"
        avg_png_dir = out / "class_average_png"
        for d in [raw_dir, heat_dir, overlay_dir, avg_npy_dir, avg_png_dir]:
            d.mkdir(parents=True, exist_ok=True)

        model = load_checkpoint(build_model(model_name, cfg["num_classes"], pretrained=False), mcfg["checkpoint"], device)
        target_info = select_target_layer(model, model_name, first)
        selected_layers[model_name] = {
            "target_layer": target_info.name,
            "output_shape": target_info.output_shape,
            "reason": target_info.reason,
        }
        print(f"{model_name} target layer: {target_info.name}, output_shape={target_info.output_shape}")
        gradcam = GradCAM(model, target_info.module, relu=bool(cfg["gradcam"].get("relu", True)))
        records = []
        class_maps: dict[str, list[np.ndarray]] = {c: [] for c in cfg["class_order"]}

        for batch in tqdm(dl, desc=f"{model_name} Grad-CAM"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                _, pred, conf = predict_logits(model, images)
            targets = pred if cfg["gradcam"].get("use_predicted_class", True) else labels
            cams = gradcam(images, targets, out_size=(cfg["img_size"], cfg["img_size"]))
            for i, stem in enumerate(batch["stem"]):
                stem = str(stem)
                cls = str(batch["class"][i])
                mask = batch["mask"][i].numpy().astype(bool)
                rgb = batch["rgb"][i].numpy()
                raw = cams[i].astype(np.float32)
                if bool(cfg["gradcam"].get("normalize_per_image", True)):
                    raw = normalize_cam(raw, mask)
                np.save(raw_dir / f"{stem}.npy", raw)
                vmap = vis_map(raw, mask, vis_cfg["percentile_clip_low"], vis_cfg["percentile_clip_high"], vis_cfg["gaussian_smoothing_sigma"])
                save_heatmap(vmap, heat_dir / f"{stem}_gradcam_heatmap.png")
                save_overlay(rgb, vmap, mask, overlay_dir / f"{stem}_gradcam_overlay.png", alpha=float(vis_cfg["alpha"]))
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
                        "target_layer": target_info.name,
                        "raw_npy": str(raw_dir / f"{stem}.npy"),
                        "overlay_png": str(overlay_dir / f"{stem}_gradcam_overlay.png"),
                    }
                )
        gradcam.remove()
        pd.DataFrame(records).to_csv(out / "attribution_metadata.csv", index=False)
        avg_records = []
        for cls in cfg["class_order"]:
            maps = class_maps[cls]
            if not maps:
                continue
            avg = np.mean(np.stack(maps), axis=0).astype(np.float32)
            np.save(avg_npy_dir / f"{cls}_gradcam_average.npy", avg)
            vavg = vis_map(avg, None, vis_cfg["percentile_clip_low"], vis_cfg["percentile_clip_high"], vis_cfg["gaussian_smoothing_sigma"])
            save_heatmap(vavg, avg_png_dir / f"{cls}_gradcam_average.png")
            avg_records.append({"model": model_name, "method": "gradcam", "class": cls, "n": len(maps), "class_average_npy": str(avg_npy_dir / f"{cls}_gradcam_average.npy")})
        pd.DataFrame(avg_records).to_csv(out / "class_average_summary.csv", index=False)

    save_json(selected_layers, run_dir / "metadata" / "selected_target_layers.json")
    print(f"Grad-CAM complete: {run_dir}")


if __name__ == "__main__":
    main()

