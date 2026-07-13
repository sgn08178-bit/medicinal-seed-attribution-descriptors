#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_yaml, save_json
from src.visualization import display_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def select_stage4_candidates(summary: pd.DataFrame, classwise: pd.DataFrame) -> pd.DataFrame:
    categories = ["Color and intensity", "Spatial frequency", "Texture", "Edge and shape related"]
    rows = []
    for cat in categories:
        sub = summary[(summary["category"] == cat) & (summary["mean_spearman_r"] > 0)].sort_values("mean_spearman_r", ascending=False)
        if sub.empty:
            continue
        top = sub.iloc[0]
        cw = classwise[classwise["descriptor"] == top["descriptor"]].sort_values("class")
        rows.append({
            "candidate_descriptor": top["descriptor"],
            "display_name": display_name(top["descriptor"]),
            "category": cat,
            "mean_spearman_r": top["mean_spearman_r"],
            "sd": top["sd"],
            "selection_reason": f"Highest positive mean spatial association within {cat}.",
            "classwise_pattern": "; ".join(f"{r['class']}={r['mean_spearman_r']}" for _, r in cw.iterrows()),
            "caution": "Candidate for Stage 4 perturbation only; this is not causal feature importance.",
        })
    # Add one descriptor from configured manuscript-prior set if not already represented.
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = Path(args.run_dir)
    validation = pd.read_csv(run_dir / "input_validation/input_validation_report.csv")
    desc_meta = pd.read_csv(run_dir / "descriptor_maps/descriptor_generation_metadata.csv")
    main = pd.read_csv(run_dir / "association_ig_zero_absolute/descriptor_summary.csv").sort_values("mean_spearman_r", ascending=False)
    pos = pd.read_csv(run_dir / "association_ig_zero_positive/descriptor_summary.csv").sort_values("mean_spearman_r", ascending=False)
    grad = pd.read_csv(run_dir / "association_gradcam_convnext/descriptor_summary.csv").sort_values("mean_spearman_r", ascending=False)
    classwise = pd.read_csv(run_dir / "association_ig_zero_absolute/classwise_summary.csv")
    fig_manifest = pd.read_csv(run_dir / "figures/stage3_figure_manifest.csv") if (run_dir / "figures/stage3_figure_manifest.csv").exists() else pd.DataFrame(columns=["figure_path"])

    cand_dir = run_dir / "stage4_candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_stage4_candidates(main, classwise)
    candidates.to_csv(cand_dir / "representative_descriptor_candidates.csv", index=False)
    save_json(candidates.to_dict(orient="records"), cand_dir / "representative_descriptor_candidates.json")

    manifest = {
        "run_id": run_dir.name,
        "input_test_csv": cfg["test_csv"],
        "input_image_root": cfg["image_root"],
        "input_mask_root": cfg["mask_root"],
        "input_ig_map_directory": cfg["canonical_ig_zero_absolute_dir"],
        "input_gradcam_directory": cfg["convnext_gradcam_final_dir"],
        "descriptor_output_directory": str(run_dir / "descriptor_maps"),
        "main_ig_descriptor_association_result_path": str(run_dir / "association_ig_zero_absolute"),
        "supplementary_positive_ig_association_result_path": str(run_dir / "association_ig_zero_positive"),
        "supplementary_gradcam_association_result_path": str(run_dir / "association_gradcam_convnext"),
        "figure_paths": fig_manifest["figure_path"].tolist(),
        "stage4_candidate_paths": [str(cand_dir / "representative_descriptor_candidates.csv"), str(cand_dir / "representative_descriptor_candidates.json")],
        "failed_samples": validation.loc[~validation["valid"], "stem"].tolist(),
        "warnings": [
            "Descriptor maps are predefined image-derived visual descriptors, not CNN internal feature representations.",
            "Correlation results indicate foreground-restricted spatial association, not causal feature importance.",
            "Grad-CAM association is supplementary because Grad-CAM depends on spatial resolution and target layer selection.",
        ],
    }
    summary_dir = run_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, summary_dir / "STAGE3_OUTPUT_MANIFEST.json")

    lines = []
    lines.append("# Stage 3 Descriptor Association Summary\n")
    lines.append("## Purpose\n")
    lines.append("Stage 3 generated predefined image-derived visual descriptor maps and quantified foreground-restricted spatial association with ConvNeXt-Small canonical IG absolute maps. Positive IG and ConvNeXt selected-layer Grad-CAM were analyzed as supplementary comparisons.\n")
    lines.append("## Final Stage 2 Inputs\n")
    lines.append(f"- IG zero absolute: `{cfg['canonical_ig_zero_absolute_dir']}`\n")
    lines.append(f"- IG zero positive: `{cfg['canonical_ig_zero_positive_dir']}`\n")
    lines.append(f"- ConvNeXt Grad-CAM: `{cfg['convnext_gradcam_final_dir']}`\n")
    lines.append("## Descriptor Categories\n")
    for cat, sub in desc_meta.groupby("category"):
        lines.append(f"- {cat}: {', '.join(sorted(sub['descriptor'].unique()))}\n")
    lines.append("## Generation Counts\n")
    lines.append(f"- Samples: {validation.shape[0]}\n")
    lines.append(f"- Valid samples: {int(validation['valid'].sum())}\n")
    lines.append(f"- Descriptor maps: {desc_meta.shape[0]}\n")
    lines.append("## Main IG Absolute Top Descriptors\n")
    for _, r in main.head(10).iterrows():
        lines.append(f"- {display_name(r['descriptor'])}: mean Spearman r = {r['mean_spearman_r']}, SD = {r['sd']}, FDR p = {r.get('fdr_adjusted_p_value')}\n")
    lines.append("## Low Or Negative Association Descriptors\n")
    for _, r in main.sort_values("mean_spearman_r").head(8).iterrows():
        lines.append(f"- {display_name(r['descriptor'])}: mean Spearman r = {r['mean_spearman_r']}\n")
    lines.append("## Positive IG Supplementary Result\n")
    for _, r in pos.head(5).iterrows():
        lines.append(f"- {display_name(r['descriptor'])}: mean Spearman r = {r['mean_spearman_r']}\n")
    lines.append("## Grad-CAM Supplementary Result\n")
    for _, r in grad.head(5).iterrows():
        lines.append(f"- {display_name(r['descriptor'])}: mean Spearman r = {r['mean_spearman_r']}\n")
    lines.append("## Stage 4 Candidate Descriptors\n")
    for _, r in candidates.iterrows():
        lines.append(f"- {r['display_name']} ({r['category']}): {r['selection_reason']}\n")
    lines.append("## Cautions\n")
    lines.append("- These results should be interpreted as foreground-restricted spatial association, not causal feature importance.\n")
    lines.append("- Grad-CAM results are supplementary because they are sensitive to spatial resolution and target layer selection.\n")
    lines.append("## Next Step\n")
    lines.append("- Stage 4 may use the candidate descriptor list to design occlusion sensitivity analysis, but Stage 4 was not performed here.\n")
    (summary_dir / "STAGE3_RUN_SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    print((summary_dir / "STAGE3_RUN_SUMMARY.md"))
    print((summary_dir / "STAGE3_OUTPUT_MANIFEST.json"))


if __name__ == "__main__":
    main()
