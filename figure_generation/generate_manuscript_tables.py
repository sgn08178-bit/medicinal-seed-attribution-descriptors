#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("MEDICINAL_SEED_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "manuscript_tables"
MAIN = OUT / "main_tables"
SUPP = OUT / "supplementary_tables"
MAIN_MD = OUT / "main_tables_markdown"
SUPP_MD = OUT / "supplementary_tables_markdown"

REQUIRED_DOCS = [
    ROOT / "FINAL_ANALYSIS_MANIFEST.md",
    ROOT / "FINAL_ANALYSIS_MANIFEST.json",
    ROOT / "FINAL_RESULTS_FOR_MANUSCRIPT.md",
    ROOT / "FINAL_FIGURE_SOURCE_MAP.md",
    ROOT / "MANUSCRIPT_RESULTS_OUTLINE.md",
    ROOT / "MANUSCRIPT_CLAIM_BOUNDARY_NOTES.md",
]

SRC = {
    "model_comparison": ROOT / "stage1_model_performance_comparison_runs/model_comparison_summary.csv",
    "test_csv": ROOT / "stage1_model_performance_comparison_runs/test.csv",
    "gradcam_final": ROOT / "stage2_attribution_maps/runs/stage2_attribution_20260605/03_gradcam_final_selected_layers/gradcam_final_generation_summary.csv",
    "gradcam_candidate": ROOT / "stage2_attribution_maps/runs/stage2_attribution_20260605/02_gradcam_candidate_layers/all_models/summary/all_model_layer_selection_summary.csv",
    "ig_abs_assoc": ROOT / "stage3_descriptor_association/runs/stage3_descriptor_association_20260606_020607/association_ig_zero_absolute/descriptor_summary.csv",
    "ig_abs_fdr": ROOT / "stage3_descriptor_association/runs/stage3_descriptor_association_20260606_020607/association_ig_zero_absolute/fdr_corrected_results.csv",
    "ig_pos_assoc": ROOT / "stage3_descriptor_association/runs/stage3_descriptor_association_20260606_020607/association_ig_zero_positive/descriptor_summary.csv",
    "gradcam_assoc": ROOT / "stage3_descriptor_association/runs/stage3_descriptor_association_20260606_020607/association_gradcam_convnext/descriptor_summary.csv",
    "occlusion_metrics": ROOT / "stage4_occlusion_overlap/runs/stage4_occlusion_20260606_065203/occlusion_results/condition_level_metrics.csv",
    "accuracy_drop": ROOT / "stage4_occlusion_overlap/runs/stage4_occlusion_20260606_065203/occlusion_results/accuracy_drop_summary.csv",
    "confidence_drop": ROOT / "stage4_occlusion_overlap/runs/stage4_occlusion_20260606_065203/occlusion_results/confidence_drop_summary.csv",
    "mask_overlap": ROOT / "stage4_occlusion_overlap/runs/stage4_occlusion_20260606_065203/mask_overlap/condition_level_pairwise_iou_summary.csv",
    "ig_baseline": ROOT / "stage2_attribution_maps/runs/stage2_attribution_20260605/01_ig_convnext_canonical_rawrgb_baseline/visualization_check/zero_vs_blur_correlation_summary.csv",
}

DESCRIPTOR_LABELS = {
    "Brightness": "Brightness",
    "LAB_L": "LAB L",
    "LAB L": "LAB L",
    "LAB_Chroma": "LAB Chroma",
    "LAB Chroma": "LAB Chroma",
    "Saturation_HSV": "HSV Saturation",
    "HSV saturation": "HSV Saturation",
    "FFT_LowPass": "FFT low-pass",
    "FFT LowPass": "FFT low-pass",
    "FFT low-pass": "FFT low-pass",
    "FFT_HighPass": "FFT high-pass",
    "Wavelet_L1_V": "Wavelet L1 vertical detail",
    "Wavelet L1 vertical": "Wavelet L1 vertical detail",
    "Wavelet_L2_V": "Wavelet L2 vertical detail",
    "Wavelet L2 vertical": "Wavelet L2 vertical detail",
    "Wavelet_L1_D": "Wavelet L1 diagonal detail",
    "Wavelet L1 diagonal": "Wavelet L1 diagonal detail",
    "Wavelet_L1_H": "Wavelet L1 horizontal detail",
    "Wavelet_L2_D": "Wavelet L2 diagonal detail",
    "Wavelet_L2_H": "Wavelet L2 horizontal detail",
    "Gabor_f0.2_t45°": "Gabor response (f = 0.2, θ = 45°)",
    "Gabor_f0.2_t45": "Gabor response (f = 0.2, θ = 45°)",
    "Gabor f = 0.2, θ = 45°": "Gabor response (f = 0.2, θ = 45°)",
    "Gabor_f0.3_t0°": "Gabor response (f = 0.3, θ = 0°)",
    "Gabor_f0.3_t0": "Gabor response (f = 0.3, θ = 0°)",
    "Gabor f = 0.3, θ = 0°": "Gabor response (f = 0.3, θ = 0°)",
    "LBP": "Local binary pattern (LBP)",
    "Sobel_Edge": "Sobel edge response",
    "Sobel Edge": "Sobel edge response",
    "Edge_Sobel": "Sobel edge response",
    "Curvature_Laplacian": "Laplacian-based local variation",
    "Laplacian-based curvature": "Laplacian-based local variation",
    "DistanceTransform": "Distance transform",
    "FourierDescriptor": "Fourier descriptor",
}

MODEL_LABELS = {
    "convnext_small": "ConvNeXt-Small",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
}

warnings: list[str] = []
missing_values: list[str] = []
generated: list[str] = []
sources_used: dict[str, list[str]] = {}


def ensure_dirs() -> None:
    for p in [MAIN, SUPP, MAIN_MD, SUPP_MD]:
        p.mkdir(parents=True, exist_ok=True)


def descriptor_label(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    if s in DESCRIPTOR_LABELS:
        return DESCRIPTOR_LABELS[s]
    if s.startswith("Gabor_f"):
        # Preserve all non-primary Gabor combinations in readable form.
        t = s.replace("Gabor_f", "Gabor response (f = ").replace("_t", ", θ = ")
        if t.endswith("°"):
            t = t[:-1] + "°)"
        else:
            t += "°)"
        return t
    return s.replace("_", " ")


def condition_label(condition, descriptor=None) -> str:
    if pd.isna(condition):
        return ""
    c = str(condition)
    if c == "full":
        return "Full image baseline"
    if c == "IG":
        return "IG mask"
    if c == "random":
        return "Random mask"
    if c.startswith("descriptor:"):
        return f"{descriptor_label(c.split(':', 1)[1])} mask"
    if descriptor and not pd.isna(descriptor):
        return f"{descriptor_label(descriptor)} mask"
    return c


def mask_label(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    if s == "IG":
        return "IG mask"
    if s == "random":
        return "Random mask"
    if s.startswith("descriptor:"):
        return f"{descriptor_label(s.split(':', 1)[1])} mask"
    return condition_label(s)


def fmt4(x):
    if pd.isna(x):
        return "[MISSING]"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def fmt2(x):
    if pd.isna(x):
        return "[MISSING]"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def fmt_sci(x):
    if pd.isna(x):
        return "[MISSING]"
    try:
        return f"{float(x):.3e}"
    except Exception:
        return str(x)


def ratio_label(x) -> str:
    if pd.isna(x):
        return ""
    return "Full" if float(x) == 0 else f"{int(round(float(x) * 100))}%"


def md_escape(x) -> str:
    s = "" if pd.isna(x) else str(x)
    return s.replace("|", "\\|").replace("\n", "<br>")


def to_markdown(df: pd.DataFrame, path: Path, title: str, source_paths: Iterable[Path]) -> None:
    lines = [f"# {title}", ""]
    lines.append("Source file(s):")
    for sp in source_paths:
        lines.append(f"- `{sp}`")
    lines.append("")
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(md_escape(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, csv_path: Path, md_dir: Path, title: str, source_paths: Iterable[Path]) -> None:
    df.to_csv(csv_path, index=False)
    generated.append(str(csv_path))
    md_path = md_dir / (csv_path.stem + ".md")
    to_markdown(df, md_path, title, source_paths)
    generated.append(str(md_path))
    sources_used[str(csv_path)] = [str(p) for p in source_paths]


def read_csv(path: Path, table_name: str) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing source for {table_name}: {path}")
        missing_values.append(str(path))
        return pd.DataFrame()
    return pd.read_csv(path)


def table1() -> None:
    df = pd.DataFrame([
        ["ARSE", "Armeniaca vulgaris", 205, 164, 41],
        ["ARSS", "Armeniaca sibirica", 223, 179, 44],
        ["PJNA", "Prunus japonica", 228, 182, 46],
        ["PRDA", "Prunus davidiana", 228, 182, 46],
        ["PRPE", "Prunus persica", 240, 192, 48],
        ["Total", "", 1124, 899, 225],
    ], columns=["Class code", "Species", "Total images", "Train images", "Test images"])
    save_table(df, MAIN / "Table1_dataset_composition.csv", MAIN_MD, "Table 1. Dataset composition", [SRC["test_csv"]])


def table2() -> None:
    rows = [
        ["Color and intensity", "Brightness", "Mean RGB intensity", "Overall seed brightness under the fixed black-background input representation."],
        ["Color and intensity", "LAB L", "Lightness in CIELAB color space", "Perceptual lightness variation across the seed surface."],
        ["Color and intensity", "LAB Chroma", "Chroma magnitude in CIELAB color space", "Color saturation-like variation independent of lightness."],
        ["Color and intensity", "HSV Saturation", "Saturation in HSV color space", "Relative color saturation across foreground seed pixels."],
        ["Spatial frequency", "FFT low-pass", "Low spatial-frequency intensity structure", "Broad smooth intensity components and slowly varying surface patterns."],
        ["Spatial frequency", "FFT high-pass", "High spatial-frequency intensity structure", "Fine intensity transitions and high-frequency surface variation."],
        ["Spatial frequency", "Wavelet detail maps", "Wavelet detail coefficients", "Directional multiscale detail patterns, including vertical, horizontal, and diagonal components."],
        ["Texture", "Gabor response", "Orientation- and frequency-selective texture response", "Local oriented texture responses at predefined frequencies and angles."],
        ["Texture", "Local binary pattern (LBP)", "Local binary texture code", "Local intensity relationships among neighboring pixels."],
        ["Edge and shape", "Sobel edge response", "Gradient magnitude", "Local edge strength along seed surface and boundary transitions."],
        ["Edge and shape", "Laplacian-based local variation", "Second-order local intensity variation", "Local curvature-like intensity variation and fine surface transitions."],
        ["Edge and shape", "Distance transform", "Distance from seed foreground boundary", "Spatial position within the seed foreground, increasing toward the interior."],
        ["Edge and shape", "Fourier descriptor", "Contour-derived shape representation", "Descriptor map derived from foreground contour structure."],
    ]
    df = pd.DataFrame(rows, columns=["Category", "Descriptor map", "Visual property represented", "Description in seed images"])
    save_table(df, MAIN / "Table2_descriptor_maps.csv", MAIN_MD, "Table 2. Predefined image-derived descriptor maps", [SRC["ig_abs_assoc"]])


def table3() -> None:
    mc = read_csv(SRC["model_comparison"], "Table 3")
    rows = []
    test_n = len(read_csv(SRC["test_csv"], "test csv"))
    for _, r in mc.iterrows():
        rows.append({
            "Model": MODEL_LABELS.get(r["model_name"], r["model_name"]),
            "CV accuracy, mean ± SD": f"{float(r.cv_accuracy_mean):.4f} ± {float(r.cv_accuracy_std):.4f}",
            "CV macro F1, mean ± SD": f"{float(r.cv_macro_f1_mean):.4f} ± {float(r.cv_macro_f1_std):.4f}",
            "Test accuracy": fmt4(r.test_accuracy),
            "Test macro precision": fmt4(r.test_precision_macro),
            "Test macro recall": fmt4(r.test_recall_macro),
            "Test macro F1": fmt4(r.test_macro_f1),
            "Test samples": test_n,
        })
    df = pd.DataFrame(rows)
    save_table(df, MAIN / "Table3_model_performance.csv", MAIN_MD, "Table 3. Model performance comparison", [SRC["model_comparison"], SRC["test_csv"]])


def table4() -> None:
    occ = read_csv(SRC["occlusion_metrics"], "Table 4")
    desired = [
        ("full", 0.00, "Reference performance without masking"),
        ("IG", 0.05, "Attribution-defined foreground perturbation"),
        ("IG", 0.10, "Attribution-defined foreground perturbation"),
        ("IG", 0.15, "Attribution-defined foreground perturbation"),
        ("descriptor:LAB_Chroma", 0.15, "Largest descriptor-defined accuracy drop among selected descriptors"),
        ("descriptor:Gabor_f0.2_t45°", 0.15, "Texture descriptor-defined mask"),
        ("descriptor:Wavelet_L1_V", 0.15, "Wavelet detail descriptor-defined mask"),
        ("random", 0.05, "Distributed perturbation baseline; 10 trials"),
        ("random", 0.10, "Distributed perturbation baseline; 10 trials"),
        ("random", 0.15, "Distributed perturbation baseline; 10 trials"),
        ("descriptor:LAB_L", 0.15, "Luminance descriptor-defined mask"),
        ("descriptor:FFT_LowPass", 0.15, "Low-frequency descriptor-defined mask"),
        ("descriptor:DistanceTransform", 0.15, "Interior-position descriptor-defined mask"),
    ]
    rows = []
    for cond, ratio, note in desired:
        ss = occ[(occ["condition"] == cond) & np.isclose(occ["masking_ratio"], ratio)]
        if ss.empty:
            warnings.append(f"Missing Table 4 row: {cond} {ratio}")
            missing_values.append(f"Table 4 {cond} {ratio}")
            rows.append({
                "Mask condition": condition_label(cond),
                "Masking ratio": ratio_label(ratio),
                "Accuracy": "[MISSING]",
                "Accuracy drop, pp": "[MISSING]",
                "Macro F1": "[MISSING]",
                "Mean confidence": "[MISSING]",
                "Notes": note,
            })
            continue
        r = ss.iloc[0]
        rows.append({
            "Mask condition": condition_label(r.condition, r.get("descriptor", None)),
            "Masking ratio": ratio_label(r.masking_ratio),
            "Accuracy": fmt4(r.accuracy),
            "Accuracy drop, pp": fmt2(float(r.accuracy_drop) * 100),
            "Macro F1": fmt4(r.macro_f1),
            "Mean confidence": fmt4(r.mean_confidence),
            "Notes": note,
        })
    df = pd.DataFrame(rows)
    save_table(df, MAIN / "Table4_key_occlusion_results.csv", MAIN_MD, "Table 4. Key occlusion sensitivity results", [SRC["occlusion_metrics"]])


def supp_s1() -> None:
    cand = read_csv(SRC["gradcam_candidate"], "Supplementary Table S1 candidate")
    final = read_csv(SRC["gradcam_final"], "Supplementary Table S1 final")
    final_sel = final.set_index(["model", "selected_layer"]) if not final.empty else pd.DataFrame()
    rows = []
    for _, r in cand.iterrows():
        key = (r.model, r.candidate_layer_name)
        matched = key in final_sel.index if not final.empty else False
        f = final_sel.loc[key] if matched else None
        rows.append({
            "Model": MODEL_LABELS.get(r.model, r.model),
            "Candidate layer": r.candidate_layer_name,
            "Candidate label": r.candidate_label,
            "Activation shape": r.activation_shape,
            "Screening samples": int(r.n),
            "Screening foreground coverage mean": r.foreground_coverage_mean,
            "Screening foreground coverage SD": r.foreground_coverage_std,
            "Selected": bool(r.selected),
            "Final test images": int(f.number_of_test_images) if matched else "[CHECK]",
            "Final successful maps": int(f.successful_maps) if matched else "[CHECK]",
            "Final failed maps": int(f.failed_maps) if matched else "[CHECK]",
            "Final foreground coverage mean": f.foreground_coverage_mean if matched else "[CHECK]",
            "Final foreground coverage SD": f.foreground_coverage_std if matched else "[CHECK]",
            "Notes": r.visual_notes,
        })
    df = pd.DataFrame(rows)
    save_table(df, SUPP / "Supplementary_Table_S1_gradcam_layer_selection.csv", SUPP_MD, "Supplementary Table S1. Grad-CAM layer selection", [SRC["gradcam_candidate"], SRC["gradcam_final"]])


def association_table(source_key: str, filename: str, title: str) -> None:
    df = read_csv(SRC[source_key], title)
    out = df.copy()
    out.insert(1, "Descriptor map", out["descriptor"].map(descriptor_label))
    out = out.drop(columns=["descriptor"])
    # Keep source precision in CSV.
    save_table(out, SUPP / filename, SUPP_MD, title, [SRC[source_key]])


def supp_s5() -> None:
    df = read_csv(SRC["occlusion_metrics"], "Supplementary Table S5").copy()
    df.insert(1, "Mask condition", df.apply(lambda r: condition_label(r["condition"], r.get("descriptor", None)), axis=1))
    df.insert(2, "Masking ratio label", df["masking_ratio"].map(ratio_label))
    if "descriptor" in df.columns:
        df["descriptor"] = df["descriptor"].map(descriptor_label)
    save_table(df, SUPP / "Supplementary_Table_S5_full_occlusion_metrics.csv", SUPP_MD, "Supplementary Table S5. Full occlusion metrics", [SRC["occlusion_metrics"], SRC["accuracy_drop"], SRC["confidence_drop"]])


def supp_s6() -> None:
    df = read_csv(SRC["mask_overlap"], "Supplementary Table S6").copy()
    df.insert(1, "Masking ratio label", df["masking_ratio"].map(ratio_label))
    df.insert(3, "Mask A", df["mask_a"].map(mask_label))
    df.insert(5, "Mask B", df["mask_b"].map(mask_label))
    save_table(df, SUPP / "Supplementary_Table_S6_mask_overlap_iou.csv", SUPP_MD, "Supplementary Table S6. Mask overlap IoU", [SRC["mask_overlap"]])


def supp_s7() -> None:
    df = read_csv(SRC["ig_baseline"], "Supplementary Table S7").copy()
    df.insert(0, "Row type", np.where(df["stem"].isna(), "Class/overall summary", "Sample"))
    df = df.rename(columns={
        "stem": "Sample ID",
        "class": "Class",
        "absolute_spearman_r": "Absolute map Spearman r",
        "positive_spearman_r": "Positive map Spearman r",
        "absolute_mean": "Absolute map mean Spearman r",
        "absolute_std": "Absolute map SD",
        "positive_mean": "Positive map mean Spearman r",
        "positive_std": "Positive map SD",
        "n": "n",
    })
    save_table(df, SUPP / "Supplementary_Table_S7_ig_baseline_consistency.csv", SUPP_MD, "Supplementary Table S7. IG baseline consistency", [SRC["ig_baseline"]])


def check_required_docs() -> None:
    for p in REQUIRED_DOCS:
        if not p.exists():
            warnings.append(f"Required manuscript/final manifest file missing: {p}")
            missing_values.append(str(p))
        else:
            # Read to satisfy audit trail and catch file errors.
            p.read_text(encoding="utf-8")


def write_readme() -> None:
    lines = [
        "# Manuscript Table Generation Summary",
        "",
        "No new analysis was performed. Tables were generated from final result files and final manifests only.",
        "",
        "## Generated tables",
    ]
    for p in generated:
        lines.append(f"- `{p}`")
    lines += ["", "## Source files used"]
    for table, srcs in sources_used.items():
        lines.append(f"- `{table}`")
        for src in srcs:
            lines.append(f"  - `{src}`")
    lines += ["", "## Labels standardized", ""]
    lines.append("- Descriptor and mask labels were standardized to manuscript-facing labels, including FFT low-pass, Wavelet L1 vertical detail, Gabor response labels, Local binary pattern (LBP), Laplacian-based local variation, Distance transform, and Fourier descriptor.")
    lines += ["", "## Missing values", ""]
    if missing_values:
        for m in missing_values:
            lines.append(f"- `{m}`")
    else:
        lines.append("- None.")
    lines += ["", "## Warnings", ""]
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None.")
    lines += ["", "## Manuscript readiness", ""]
    if warnings:
        lines.append("- Tables were generated, but warnings should be manually reviewed before insertion.")
    else:
        lines.append("- Tables are ready for manuscript insertion, subject to final journal formatting.")
    lines += ["", "## Manual review recommended", ""]
    lines.append("- Supplementary Table S1 contains `[CHECK]` in final-test columns for non-selected candidate layers by design.")
    lines.append("- Supplementary Table S7 includes sample-level rows and class/overall summary rows; journal formatting may require splitting if too long.")
    (OUT / "TABLE_GENERATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    check_required_docs()
    table1()
    table2()
    table3()
    table4()
    supp_s1()
    association_table("ig_abs_assoc", "Supplementary_Table_S2_full_ig_descriptor_association.csv", "Supplementary Table S2. Full IG descriptor association")
    association_table("ig_pos_assoc", "Supplementary_Table_S3_positive_ig_descriptor_association.csv", "Supplementary Table S3. Positive IG descriptor association")
    association_table("gradcam_assoc", "Supplementary_Table_S4_gradcam_descriptor_association.csv", "Supplementary Table S4. Grad-CAM descriptor association")
    supp_s5()
    supp_s6()
    supp_s7()
    write_readme()

    print(json.dumps({
        "output_root": str(OUT),
        "generated_table_paths": generated,
        "missing_values_count": len(missing_values),
        "warnings_count": len(warnings),
        "warnings": warnings,
        "manual_review": [
            "Supplementary Table S1 non-selected candidate layers contain [CHECK] for final-test columns.",
            "Supplementary Table S7 may be long because it preserves sample-level IG baseline consistency rows.",
        ],
        "summary": str(OUT / "TABLE_GENERATION_SUMMARY.md"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
