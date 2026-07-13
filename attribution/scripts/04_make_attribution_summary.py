#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_yaml, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def main():
    run_dir = Path(parse_args().run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    rows = []
    avg_rows = []
    test_df = pd.read_csv(run_dir / "inputs" / "test.csv")
    original_dir = run_dir / "visualization" / "original_png"
    original_dir.mkdir(parents=True, exist_ok=True)
    for _, row in test_df.iterrows():
        out_path = original_dir / f"{row['stem']}_input.png"
        if not out_path.exists():
            Image.open(row["filepath"]).convert("RGB").resize((cfg["img_size"], cfg["img_size"]), Image.Resampling.BILINEAR).save(out_path)

    for model_name, mcfg in cfg["models"].items():
        for method in ["ig", "gradcam"]:
            meta_path = run_dir / model_name / method / "attribution_metadata.csv"
            if not meta_path.exists():
                continue
            df = pd.read_csv(meta_path)
            rep_dir = run_dir / model_name / method / "representative_examples"
            rep_dir.mkdir(parents=True, exist_ok=True)
            for cls in cfg["class_order"]:
                class_df = df[df["true_class"] == cls]
                if class_df.empty:
                    continue
                correct_df = class_df[class_df["true_label"] == class_df["pred_label"]]
                row = (correct_df if not correct_df.empty else class_df).sort_values("confidence", ascending=False).iloc[0]
                stem = row["stem"]
                source_original = original_dir / f"{stem}_input.png"
                heatmap = run_dir / model_name / method / "heatmap_png" / f"{stem}_{method}_heatmap.png"
                overlay = run_dir / model_name / method / "overlay_png" / f"{stem}_{method}_overlay.png"
                for src, suffix in [(source_original, "input"), (heatmap, "heatmap"), (overlay, "overlay")]:
                    if src.exists():
                        Image.open(src).save(rep_dir / f"{cls}_{stem}_{suffix}.png")
            rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "n_samples": len(df),
                    "n_raw_npy": sum(Path(p).exists() for p in df["raw_npy"]),
                    "n_heatmap_png": len(list((run_dir / model_name / method / "heatmap_png").glob("*.png"))),
                    "n_overlay_png": sum(Path(p).exists() for p in df["overlay_png"]),
                    "n_original_png": len(list(original_dir.glob("*.png"))),
                    "representative_examples_dir": str(rep_dir),
                    "accuracy_on_test_csv": float((df["true_label"] == df["pred_label"]).mean()),
                    "mean_confidence": float(df["confidence"].mean()),
                }
            )
            avg_path = run_dir / model_name / method / "class_average_summary.csv"
            if avg_path.exists():
                avg_rows.extend(pd.read_csv(avg_path).to_dict("records"))

    summaries = run_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    avg_df = pd.DataFrame(avg_rows)
    summary_df.to_csv(summaries / "attribution_generation_summary.csv", index=False)
    avg_df.to_csv(summaries / "class_average_summary.csv", index=False)
    save_json(
        {
            "run_dir": str(run_dir),
            "n_summary_rows": int(len(summary_df)),
            "outputs": {
                "attribution_generation_summary": str(summaries / "attribution_generation_summary.csv"),
                "class_average_summary": str(summaries / "class_average_summary.csv"),
            },
        },
        summaries / "summary_manifest.json",
    )
    print(summary_df.to_string(index=False))
    print(f"Summary complete: {summaries}")


if __name__ == "__main__":
    main()
