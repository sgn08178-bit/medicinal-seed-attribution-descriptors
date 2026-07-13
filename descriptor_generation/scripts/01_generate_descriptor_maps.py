#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_mask_224, load_rgb_224, prepare_test_dataframe
from src.descriptor_maps import compute_descriptor_maps, descriptor_category, descriptor_quality
from src.io_utils import load_yaml
from src.visualization import save_map_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = Path(args.run_dir)
    df = prepare_test_dataframe(cfg["test_csv"], cfg["image_root"], cfg["mask_root"])
    raw_root = run_dir / "descriptor_maps/raw_npy"
    vis_root = run_dir / "descriptor_maps/visualization_png"
    class_sum = {}
    class_count = {}
    meta, qc = [], []
    img_size = int(cfg["img_size"])
    for _, row in tqdm(df.iterrows(), total=len(df), desc="descriptor maps"):
        stem = str(row["stem"])
        cls = str(row["class"])
        rgb = load_rgb_224(row["filepath"], img_size)
        fg = load_mask_224(row["maskpath"], img_size)
        maps = compute_descriptor_maps(rgb, fg, cfg)
        (raw_root / stem).mkdir(parents=True, exist_ok=True)
        (vis_root / stem).mkdir(parents=True, exist_ok=True)
        for name, arr in maps.items():
            np.save(raw_root / stem / f"{name}.npy", arr.astype(np.float32))
            save_map_png(arr, fg, vis_root / stem / f"{name}.png", cmap="gray")
            q = descriptor_quality(arr, fg)
            q.update({"stem": stem, "class": cls, "descriptor": name, "category": descriptor_category(name)})
            qc.append(q)
            meta.append({
                "stem": stem,
                "class": cls,
                "descriptor": name,
                "category": descriptor_category(name),
                "path": str(raw_root / stem / f"{name}.npy"),
            })
            key = (cls, name)
            class_sum[key] = class_sum.get(key, np.zeros_like(arr, dtype=np.float64)) + arr
            class_count[key] = class_count.get(key, 0) + 1
    pd.DataFrame(meta).to_csv(run_dir / "descriptor_maps/descriptor_generation_metadata.csv", index=False)
    pd.DataFrame(qc).to_csv(run_dir / "descriptor_maps/descriptor_quality_check.csv", index=False)
    avg_root = run_dir / "descriptor_maps/class_average_npy"
    avg_png = run_dir / "descriptor_maps/class_average_png"
    avg_records = []
    for (cls, name), total in class_sum.items():
        avg = (total / class_count[(cls, name)]).astype(np.float32)
        d = avg_root / cls
        p = avg_png / cls
        d.mkdir(parents=True, exist_ok=True)
        p.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{name}.npy", avg)
        save_map_png(avg, np.ones_like(avg, dtype=bool), p / f"{name}.png", cmap="gray")
        avg_records.append({"class": cls, "descriptor": name, "category": descriptor_category(name), "n": class_count[(cls, name)], "path": str(d / f"{name}.npy")})
    pd.DataFrame(avg_records).to_csv(run_dir / "descriptor_maps/class_average_metadata.csv", index=False)
    print(f"Descriptor generation complete: {len(meta)} maps for {len(df)} samples.")


if __name__ == "__main__":
    main()
