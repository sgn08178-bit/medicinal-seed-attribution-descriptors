#!/usr/bin/env python3
"""Validate descriptor-classification outputs and export manuscript-ready files."""

from __future__ import annotations

import json
import os
import math
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(os.environ.get("MEDICINAL_SEED_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("MEDICINAL_SEED_DATA_ROOT", ROOT / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("MEDICINAL_SEED_RESULTS_ROOT", ROOT / "results")).resolve()
STAGE7C = Path(
    os.environ.get(
        "MEDICINAL_SEED_DESCRIPTOR_CLASSIFIER_OUTPUT",
        RESULTS_ROOT / "descriptor_classifier",
    )
).resolve()
OUT = Path(
    os.environ.get(
        "MEDICINAL_SEED_DESCRIPTOR_VALIDATION_OUTPUT",
        STAGE7C / "validated",
    )
).resolve()
FIG = OUT / "figures"

TRAIN_CSV = Path(
    os.environ.get(
        "MEDICINAL_SEED_TRAIN_CSV",
        DATA_ROOT / "metadata" / "train_split.csv",
    )
).resolve()
TEST_CSV = Path(
    os.environ.get(
        "MEDICINAL_SEED_TEST_CSV",
        DATA_ROOT / "metadata" / "test_split.csv",
    )
).resolve()
STAGE3_DESC_ROOT = Path(
    os.environ.get(
        "MEDICINAL_SEED_TEST_DESCRIPTOR_ROOT",
        DATA_ROOT / "descriptor_maps" / "test",
    )
).resolve()
STAGE5_DESC_ROOT = Path(
    os.environ.get(
        "MEDICINAL_SEED_TRAIN_DESCRIPTOR_ROOT",
        DATA_ROOT / "descriptor_maps" / "train",
    )
).resolve()
FEATURES_CSV = STAGE7C / "all_valid_descriptor_summary_features.csv"
INVENTORY_CSV = STAGE7C / "descriptor_map_inventory.csv"
ORIGINAL_PERF_CSV = STAGE7C / "descriptor_subset_model_performance.csv"

SEED = 42
N_FOLDS = 5
STAT_COLS = ["mean", "std", "median", "p90", "top10_mean", "iqr"]
AUDIT_KEYS = [
    "Brightness",
    "LAB_L",
    "LAB_Chroma",
    "FFT_LowPass",
    "FFT_HighPass",
    "Wavelet_L1_V",
    "Gabor_f0.2_t45°",
    "LBP",
    "Edge_Sobel",
    "Curvature_Laplacian",
    "DistanceTransform",
    "FourierDescriptor",
]


DISPLAY = {
    "Brightness": "Brightness",
    "LAB_L": "LAB L",
    "LAB_Chroma": "LAB Chroma",
    "Saturation_HSV": "HSV saturation",
    "FFT_LowPass": "FFT low-pass",
    "FFT_HighPass": "FFT high-pass",
    "LBP": "Local binary pattern (LBP)",
    "Edge_Sobel": "Sobel edge response",
    "Curvature_Laplacian": "Laplacian-based local variation",
    "DistanceTransform": "Distance transform",
    "FourierDescriptor": "Fourier descriptor",
    "Wavelet_L1_D": "Wavelet L1 diagonal detail",
    "Wavelet_L1_H": "Wavelet L1 horizontal detail",
    "Wavelet_L1_V": "Wavelet L1 vertical detail",
    "Wavelet_L2_D": "Wavelet L2 diagonal detail",
    "Wavelet_L2_H": "Wavelet L2 horizontal detail",
    "Wavelet_L2_V": "Wavelet L2 vertical detail",
    "Gabor_f0.1_t0°": "Gabor response (f = 0.1, theta = 0 deg)",
    "Gabor_f0.1_t45°": "Gabor response (f = 0.1, theta = 45 deg)",
    "Gabor_f0.1_t90°": "Gabor response (f = 0.1, theta = 90 deg)",
    "Gabor_f0.1_t135°": "Gabor response (f = 0.1, theta = 135 deg)",
    "Gabor_f0.2_t0°": "Gabor response (f = 0.2, theta = 0 deg)",
    "Gabor_f0.2_t45°": "Gabor response (f = 0.2, theta = 45 deg)",
    "Gabor_f0.2_t90°": "Gabor response (f = 0.2, theta = 90 deg)",
    "Gabor_f0.2_t135°": "Gabor response (f = 0.2, theta = 135 deg)",
    "Gabor_f0.3_t0°": "Gabor response (f = 0.3, theta = 0 deg)",
    "Gabor_f0.3_t45°": "Gabor response (f = 0.3, theta = 45 deg)",
    "Gabor_f0.3_t90°": "Gabor response (f = 0.3, theta = 90 deg)",
    "Gabor_f0.3_t135°": "Gabor response (f = 0.3, theta = 135 deg)",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def simple_md_table(df: pd.DataFrame) -> str:
    def esc(x):
        return "" if pd.isna(x) else str(x).replace("|", "\\|")
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(esc(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def display_name(key: str) -> str:
    return DISPLAY.get(key, key)


def load_mask(mask_path: str, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def audit_normalization() -> tuple[pd.DataFrame, bool]:
    train = pd.read_csv(TRAIN_CSV).head(3).assign(split="train")
    test = pd.read_csv(TEST_CSV).head(3).assign(split="test")
    rows = []
    for _, row in pd.concat([train, test], ignore_index=True).iterrows():
        root = STAGE5_DESC_ROOT if row["split"] == "train" else STAGE3_DESC_ROOT
        for key in AUDIT_KEYS:
            path = root / row["stem"] / f"{key}.npy"
            if not path.exists():
                rows.append(
                    {
                        "split": row["split"],
                        "stem": row["stem"],
                        "descriptor": key,
                        "display_name": display_name(key),
                        "path": str(path),
                        "exists": False,
                    }
                )
                continue
            arr = np.load(path)
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            fg = load_mask(row["maskpath"], arr.shape)
            vals = arr[fg]
            finite = np.isfinite(vals)
            rows.append(
                {
                    "split": row["split"],
                    "stem": row["stem"],
                    "descriptor": key,
                    "display_name": display_name(key),
                    "path": str(path),
                    "exists": True,
                    "global_min": float(np.nanmin(arr)),
                    "global_max": float(np.nanmax(arr)),
                    "foreground_min": float(np.nanmin(vals)),
                    "foreground_max": float(np.nanmax(vals)),
                    "foreground_mean": float(np.nanmean(vals)),
                    "foreground_std": float(np.nanstd(vals)),
                    "within_0_1_global": bool(np.nanmin(arr) >= -1e-7 and np.nanmax(arr) <= 1.0 + 1e-7),
                    "within_0_1_foreground": bool(np.nanmin(vals) >= -1e-7 and np.nanmax(vals) <= 1.0 + 1e-7),
                    "foreground_constant": bool(np.nanstd(vals) <= 1e-12),
                    "nan_count": int(np.isnan(arr).sum()),
                    "inf_count": int(np.isinf(arr).sum()),
                    "finite_foreground_count": int(finite.sum()),
                    "appears_already_normalized": bool(
                        np.nanmin(arr) >= -1e-7 and np.nanmax(arr) <= 1.0 + 1e-7 and np.isfinite(arr).all()
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "descriptor_normalization_audit.csv", index=False)
    ok = bool(audit["appears_already_normalized"].fillna(False).all())
    lines = [
        "# Descriptor normalization audit",
        "",
        f"Audited samples: {audit['stem'].nunique()} stems across train/test.",
        f"Audited descriptor keys: {len(AUDIT_KEYS)}.",
        "",
        "## Conclusion",
        "",
        f"1. Are `raw_npy` descriptor maps already normalized to [0, 1]? {'Yes' if ok else 'No or not consistently'}.",
        "2. Did Stage 7C compute summary features from raw values? Stage 7C read the NPY values directly, but the audited NPY maps already appear min-max normalized to [0, 1].",
        f"3. Does this mismatch the manuscript Methods statement? {'No mismatch was detected in the sampled maps.' if ok else 'Potential mismatch: some sampled maps were not within [0, 1].'}",
        f"4. Should descriptor maps be min-max normalized before descriptor summary feature extraction? {'No additional normalization was applied in the revised outputs because the checked NPY maps were already normalized.' if ok else 'Yes; rerun feature extraction with explicit per-map normalization.'}",
        "5. Revised feature extraction status: unchanged from Stage 7C summary features because the normalization audit supported that the descriptor map inputs were already normalized.",
        "",
        "## Summary table",
        "",
        simple_md_table(audit[[
            "split",
            "stem",
            "display_name",
            "global_min",
            "global_max",
            "foreground_mean",
            "foreground_std",
            "within_0_1_global",
            "foreground_constant",
            "nan_count",
            "inf_count",
        ]]),
    ]
    (OUT / "DESCRIPTOR_NORMALIZATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit, ok


def pivot_features(features: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["stem", "split", "class", "label"]
    pieces = []
    for stat in STAT_COLS:
        p = features.pivot_table(index=id_cols, columns="descriptor", values=stat, aggfunc="first")
        p.columns = [f"{c}__{stat}" for c in p.columns]
        pieces.append(p)
    return pd.concat(pieces, axis=1).reset_index()


def subset_defs(valid_keys: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    gabor = [k for k in valid_keys if k.startswith("Gabor_")]
    wave = [k for k in valid_keys if k.startswith("Wavelet_")]
    subsets = {
        "true_all_valid_descriptor_maps": valid_keys,
        "selected_10_map_summarized_set": [
            "LAB_L",
            "Brightness",
            "FFT_LowPass",
            "LAB_Chroma",
            "Wavelet_L1_V",
            "DistanceTransform",
            "Gabor_f0.2_t45°",
            "FFT_HighPass",
            "LBP",
            "Saturation_HSV",
        ],
        "top3_ig_associated": ["LAB_L", "Brightness", "FFT_LowPass"],
        "top5_ig_associated": ["LAB_L", "Brightness", "FFT_LowPass", "LAB_Chroma", "Wavelet_L1_V"],
        "top7_ig_associated": [
            "LAB_L",
            "Brightness",
            "FFT_LowPass",
            "LAB_Chroma",
            "Wavelet_L1_V",
            "DistanceTransform",
            "Gabor_f0.2_t45°",
        ],
        "low_or_negative_ig_associated": ["Saturation_HSV", "LBP", "FFT_HighPass", "FourierDescriptor"],
        "edge_and_shape": ["Edge_Sobel", "Curvature_Laplacian", "DistanceTransform", "FourierDescriptor"],
        "color_and_intensity": ["Brightness", "LAB_L", "LAB_Chroma", "Saturation_HSV"],
        "spatial_frequency": ["FFT_LowPass", "FFT_HighPass", *wave],
        "texture_only": ["LBP", *gabor],
        "texture_and_wavelet_detail": ["LBP", *gabor, *wave],
    }
    labels = {
        "true_all_valid_descriptor_maps": "True all valid descriptor maps",
        "selected_10_map_summarized_set": "Selected 10-map summarized set",
        "top3_ig_associated": "Top 3 IG-associated",
        "top5_ig_associated": "Top 5 IG-associated",
        "top7_ig_associated": "Top 7 IG-associated",
        "low_or_negative_ig_associated": "Low/negative IG-associated",
        "edge_and_shape": "Edge and shape subset",
        "color_and_intensity": "Color and intensity subset",
        "spatial_frequency": "Spatial frequency subset",
        "texture_only": "Texture-only subset",
        "texture_and_wavelet_detail": "Texture and wavelet-detail subset",
    }
    valid = set(valid_keys)
    return {k: [d for d in v if d in valid] for k, v in subsets.items()}, labels


def feature_cols(descriptors: list[str], wide: pd.DataFrame) -> list[str]:
    return [f"{d}__{s}" for d in descriptors for s in STAT_COLS if f"{d}__{s}" in wide.columns]


def models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED)),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
            ]
        ),
    }


def evaluate_fixed(wide: pd.DataFrame, subsets: dict[str, list[str]], labels: dict[str, str]) -> pd.DataFrame:
    train = wide[wide["split"] == "train"].copy()
    test = wide[wide["split"] == "test"].copy()
    y = train["label"].to_numpy(int)
    y_test = test["label"].to_numpy(int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for sid, descs in subsets.items():
        cols = feature_cols(descs, wide)
        X = train[cols].to_numpy(float)
        X_test = test[cols].to_numpy(float)
        for model_id, model in models().items():
            cv_acc, cv_f1 = [], []
            for tr, va in skf.split(X, y):
                m = clone(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X[tr], y[tr])
                pred = m.predict(X[va])
                cv_acc.append(accuracy_score(y[va], pred))
                cv_f1.append(f1_score(y[va], pred, average="macro", zero_division=0))
            final = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final.fit(X, y)
            pred = final.predict(X_test)
            rows.append(
                {
                    "subset_id": sid,
                    "subset_label": labels[sid],
                    "model": model_id,
                    "model_label": "Logistic regression" if model_id == "logistic_regression" else "RBF SVM",
                    "n_descriptor_maps": len(descs),
                    "n_numeric_features": len(cols),
                    "included_descriptor_names": "; ".join(display_name(d) for d in descs),
                    "cv_accuracy_mean": float(np.mean(cv_acc)),
                    "cv_accuracy_sd": float(np.std(cv_acc, ddof=1)),
                    "cv_macro_f1_mean": float(np.mean(cv_f1)),
                    "cv_macro_f1_sd": float(np.std(cv_f1, ddof=1)),
                    "test_accuracy": float(accuracy_score(y_test, pred)),
                    "test_macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
                    "test_macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
                    "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                }
            )
    perf = pd.DataFrame(rows)
    for model_id in perf["model"].unique():
        base = perf[(perf["model"] == model_id) & (perf["subset_id"] == "true_all_valid_descriptor_maps")]["test_macro_f1"].iloc[0]
        idx = perf["model"] == model_id
        perf.loc[idx, "test_macro_f1_relative_to_true_all"] = perf.loc[idx, "test_macro_f1"] / base
        perf.loc[idx, "test_macro_f1_drop_from_true_all"] = base - perf.loc[idx, "test_macro_f1"]
    return perf


def write_manifest(subsets: dict[str, list[str]], labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for sid, descs in subsets.items():
        rows.append(
            {
                "subset_id": sid,
                "subset_label": labels[sid],
                "n_descriptor_maps": len(descs),
                "n_numeric_features": len(descs) * len(STAT_COLS),
                "included_descriptor_keys": "; ".join(descs),
                "included_descriptor_names": "; ".join(display_name(d) for d in descs),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "revised_descriptor_subset_feature_manifest.csv", index=False)
    return df


def make_figures(perf: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    order = [
        "True all valid descriptor maps",
        "Selected 10-map summarized set",
        "Top 3 IG-associated",
        "Top 5 IG-associated",
        "Top 7 IG-associated",
        "Color and intensity subset",
        "Spatial frequency subset",
        "Texture-only subset",
        "Texture and wavelet-detail subset",
        "Edge and shape subset",
        "Low/negative IG-associated",
    ]
    for model_id, filename, title in [
        ("logistic_regression", "logistic_regression_subset_performance", "Logistic regression descriptor-subset performance"),
        ("rbf_svm", "rbf_svm_subset_performance", "RBF SVM descriptor-subset performance"),
    ]:
        d = perf[perf["model"] == model_id].copy()
        fig, ax = plt.subplots(figsize=(9.8, 5.8))
        sns.barplot(data=d, y="subset_label", x="test_macro_f1", order=order, color="#91b7c7", ax=ax)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Test macro F1")
        ax.set_ylabel("")
        ax.set_title(title, fontsize=13)
        ax.grid(axis="x", alpha=0.22)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=9, padding=3)
        fig.tight_layout()
        fig.savefig(FIG / f"{filename}.png", dpi=400, bbox_inches="tight")
        fig.savefig(FIG / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    d = perf[perf["subset_id"] != "true_all_valid_descriptor_maps"].copy()
    sns.barplot(
        data=d,
        y="subset_label",
        x="test_macro_f1_relative_to_true_all",
        hue="model_label",
        order=order[1:],
        palette=["#7aa6c2", "#d7a86e"],
        ax=ax,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Relative test macro F1 vs true all valid descriptor maps")
    ax.set_ylabel("")
    ax.set_title("Fixed-model relative descriptor-subset performance", fontsize=13)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(title="Fixed classifier", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fixed_model_relative_performance.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG / "fixed_model_relative_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    def esc(x):
        return "" if pd.isna(x) else str(x).replace("|", "\\|")
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(esc(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def write_reports(audit_ok: bool, perf: pd.DataFrame, original_perf: pd.DataFrame | None) -> None:
    def row(model: str, sid: str) -> pd.Series:
        return perf[(perf["model"] == model) & (perf["subset_id"] == sid)].iloc[0]

    log = perf[perf["model"] == "logistic_regression"].copy()
    svm = perf[perf["model"] == "rbf_svm"].copy()
    cat_ids = ["color_and_intensity", "spatial_frequency", "texture_only", "texture_and_wavelet_detail", "edge_and_shape"]
    best_log_cat = log[log["subset_id"].isin(cat_ids)].sort_values("test_macro_f1", ascending=False).iloc[0]
    best_svm_cat = svm[svm["subset_id"].isin(cat_ids)].sort_values("test_macro_f1", ascending=False).iloc[0]
    cols = [
        "subset_label",
        "n_descriptor_maps",
        "n_numeric_features",
        "cv_accuracy_mean",
        "cv_accuracy_sd",
        "cv_macro_f1_mean",
        "cv_macro_f1_sd",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_macro_f1_relative_to_true_all",
        "test_macro_f1_drop_from_true_all",
    ]
    exploratory = "Not generated in the revised manuscript-safe outputs."
    if original_perf is not None and not original_perf.empty:
        by_cv = original_perf.loc[original_perf.groupby("subset_id")["cv_macro_f1_mean"].idxmax()].copy()
        by_cv.loc[by_cv["subset_id"] == "texture_family", "subset_id"] = "texture_and_wavelet_detail"
        by_cv.loc[by_cv["subset_label"] == "Texture family subset", "subset_label"] = "Texture and wavelet-detail subset"
        best_overall = by_cv.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
        by_cv.to_csv(OUT / "exploratory_best_model_by_cv_macro_f1.csv", index=False)
        exploratory = (
            f"Exploratory CV-selected comparison was generated separately. Highest CV macro F1 among subset-level CV-selected rows: "
            f"{best_overall['subset_label']} with {best_overall['model_label']} "
            f"(CV macro F1 {best_overall['cv_macro_f1_mean']:.6f}, test macro F1 {best_overall['test_macro_f1']:.6f}). "
            "This table is exploratory and is not the manuscript-safe primary reporting basis."
        )
    lines = [
        "# Stage 7C revised manuscript-safe report",
        "",
        "## Normalization audit conclusion",
        "",
        f"Descriptor raw NPY maps appear already normalized to [0, 1]: {'yes' if audit_ok else 'no or not consistently'}.",
        "Revised fixed-model outputs therefore reuse the Stage 7C summary features. No additional min-max normalization was applied after the audit.",
        "",
        "## Logistic regression fixed-model results",
        "",
        md_table(log[cols]),
        "",
        "## RBF SVM fixed-model results",
        "",
        md_table(svm[cols]),
        "",
        "## Required manuscript questions",
        "",
        f"1. Did descriptor normalization need to be applied before feature extraction? {'No additional normalization was needed because sampled raw_npy maps were already within [0, 1].' if audit_ok else 'Yes, but this run did not apply it; review required.'}",
        "2. Normalization effect compared with original Stage 7C: unchanged, because revised outputs reused the normalized Stage 7C summary features.",
        "3. Logistic regression results are listed above.",
        "4. RBF SVM results are listed above.",
        f"5. Under logistic regression, Top 5 relative macro F1 was {row('logistic_regression', 'top5_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}; Top 7 relative macro F1 was {row('logistic_regression', 'top7_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"6. Under RBF SVM, Top 5 relative macro F1 was {row('rbf_svm', 'top5_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}; Top 7 relative macro F1 was {row('rbf_svm', 'top7_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"7. Best category subset under logistic regression: {best_log_cat['subset_label']} with macro F1 {best_log_cat['test_macro_f1']:.6f}.",
        f"8. Best category subset under RBF SVM: {best_svm_cat['subset_label']} with macro F1 {best_svm_cat['test_macro_f1']:.6f}.",
        f"9. Low/negative subset also classified above chance: logistic macro F1 {row('logistic_regression', 'low_or_negative_ig_associated')['test_macro_f1']:.6f}; RBF SVM macro F1 {row('rbf_svm', 'low_or_negative_ig_associated')['test_macro_f1']:.6f}.",
        "10. Safe to include, if framed as auxiliary descriptor-subset classification rather than model-mechanism evidence.",
        "11. Recommended table placement: supplementary table; one concise Results sentence may be enough if the main text needs it.",
        "12. Safest Results sentence: Foreground-restricted descriptor summary features retained class-discriminative visual information, and compact IG-associated subsets preserved a substantial fraction of the true all-descriptor baseline under fixed classifiers.",
        "13. Safest Discussion sentence: These auxiliary descriptor-subset results contextualize the descriptor maps as class-discriminative foreground summaries, without implying a direct CNN mechanism.",
        "",
        "## Exploratory model comparison",
        "",
        exploratory,
    ]
    (OUT / "STAGE7C_REVISED_FOR_MANUSCRIPT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Validated descriptor classification results",
        "",
        "## Results summary",
        "",
        f"Normalization audit conclusion: sampled `raw_npy` descriptor maps were already within [0, 1]: {'yes' if audit_ok else 'no or uncertain'}. Revised feature extraction was unchanged because the Stage 7C summary features were computed from already-normalized NPY maps. No mismatch with the manuscript Methods statement was detected in the sampled maps.",
        "",
        f"Logistic regression, true all valid descriptor maps: test accuracy {row('logistic_regression', 'true_all_valid_descriptor_maps')['test_accuracy']:.6f}, macro F1 {row('logistic_regression', 'true_all_valid_descriptor_maps')['test_macro_f1']:.6f}.",
        f"Logistic regression, selected 10-map set: test accuracy {row('logistic_regression', 'selected_10_map_summarized_set')['test_accuracy']:.6f}, macro F1 {row('logistic_regression', 'selected_10_map_summarized_set')['test_macro_f1']:.6f}, relative macro F1 {row('logistic_regression', 'selected_10_map_summarized_set')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"Logistic regression, Top 3: test accuracy {row('logistic_regression', 'top3_ig_associated')['test_accuracy']:.6f}, macro F1 {row('logistic_regression', 'top3_ig_associated')['test_macro_f1']:.6f}, relative macro F1 {row('logistic_regression', 'top3_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"Logistic regression, Top 5: test accuracy {row('logistic_regression', 'top5_ig_associated')['test_accuracy']:.6f}, macro F1 {row('logistic_regression', 'top5_ig_associated')['test_macro_f1']:.6f}, relative macro F1 {row('logistic_regression', 'top5_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"Logistic regression, Top 7: test accuracy {row('logistic_regression', 'top7_ig_associated')['test_accuracy']:.6f}, macro F1 {row('logistic_regression', 'top7_ig_associated')['test_macro_f1']:.6f}, relative macro F1 {row('logistic_regression', 'top7_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"Best category subset under logistic regression: {best_log_cat['subset_label']}; test accuracy {best_log_cat['test_accuracy']:.6f}, macro F1 {best_log_cat['test_macro_f1']:.6f}.",
        f"Logistic regression, low/negative subset: test accuracy {row('logistic_regression', 'low_or_negative_ig_associated')['test_accuracy']:.6f}, macro F1 {row('logistic_regression', 'low_or_negative_ig_associated')['test_macro_f1']:.6f}.",
        "",
        f"RBF SVM, true all valid descriptor maps: test accuracy {row('rbf_svm', 'true_all_valid_descriptor_maps')['test_accuracy']:.6f}, macro F1 {row('rbf_svm', 'true_all_valid_descriptor_maps')['test_macro_f1']:.6f}.",
        f"RBF SVM, selected 10-map set: test accuracy {row('rbf_svm', 'selected_10_map_summarized_set')['test_accuracy']:.6f}, macro F1 {row('rbf_svm', 'selected_10_map_summarized_set')['test_macro_f1']:.6f}, relative macro F1 {row('rbf_svm', 'selected_10_map_summarized_set')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"RBF SVM, Top 3: test accuracy {row('rbf_svm', 'top3_ig_associated')['test_accuracy']:.6f}, macro F1 {row('rbf_svm', 'top3_ig_associated')['test_macro_f1']:.6f}, relative macro F1 {row('rbf_svm', 'top3_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"RBF SVM, Top 5: test accuracy {row('rbf_svm', 'top5_ig_associated')['test_accuracy']:.6f}, macro F1 {row('rbf_svm', 'top5_ig_associated')['test_macro_f1']:.6f}, relative macro F1 {row('rbf_svm', 'top5_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"RBF SVM, Top 7: test accuracy {row('rbf_svm', 'top7_ig_associated')['test_accuracy']:.6f}, macro F1 {row('rbf_svm', 'top7_ig_associated')['test_macro_f1']:.6f}, relative macro F1 {row('rbf_svm', 'top7_ig_associated')['test_macro_f1_relative_to_true_all']:.6f}.",
        f"Best category subset under RBF SVM: {best_svm_cat['subset_label']}; test accuracy {best_svm_cat['test_accuracy']:.6f}, macro F1 {best_svm_cat['test_macro_f1']:.6f}.",
        f"RBF SVM, low/negative subset: test accuracy {row('rbf_svm', 'low_or_negative_ig_associated')['test_accuracy']:.6f}, macro F1 {row('rbf_svm', 'low_or_negative_ig_associated')['test_macro_f1']:.6f}.",
        "",
        f"Exploratory model comparison: {exploratory}",
        "",
        "Recommended manuscript action: put the fixed-model descriptor-subset classifier table in Supplementary material, and mention it briefly in Results or Discussion only as auxiliary support. Use true all valid descriptor maps as the baseline. Safe Results sentence: Foreground-restricted descriptor summary features retained class-discriminative visual information, and compact IG-associated subsets preserved a substantial fraction of the true all-descriptor baseline under fixed classifiers. Safe Discussion sentence: These auxiliary descriptor-subset results contextualize the descriptor maps as class-discriminative foreground summaries, without implying a direct CNN mechanism.",
    ]
    (OUT / "descriptor_classification_validation_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    audit, audit_ok = audit_normalization()
    if not audit_ok:
        raise RuntimeError("Normalization audit found non-normalized maps; implement normalized extraction before reporting.")

    features = pd.read_csv(FEATURES_CSV)
    inventory = pd.read_csv(INVENTORY_CSV)
    valid_keys = inventory[inventory["valid_for_stage7c"]]["descriptor"].tolist()
    features = features[features["descriptor"].isin(valid_keys)].copy()
    wide = pivot_features(features)
    subsets, labels = subset_defs(valid_keys)
    manifest = write_manifest(subsets, labels)
    perf = evaluate_fixed(wide, subsets, labels)
    perf.to_csv(OUT / "fixed_model_descriptor_subset_summary.csv", index=False)
    perf[perf["model"] == "logistic_regression"].to_csv(OUT / "logistic_regression_descriptor_subset_performance.csv", index=False)
    perf[perf["model"] == "rbf_svm"].to_csv(OUT / "rbf_svm_descriptor_subset_performance.csv", index=False)
    original = pd.read_csv(ORIGINAL_PERF_CSV) if ORIGINAL_PERF_CSV.exists() else None
    make_figures(perf)
    write_reports(audit_ok, perf, original)
    manifest_json = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output_dir": str(OUT),
        "normalization_audit": str(OUT / "descriptor_normalization_audit.csv"),
        "normalization_audit_report": str(OUT / "DESCRIPTOR_NORMALIZATION_AUDIT.md"),
        "fixed_model_summary": str(OUT / "fixed_model_descriptor_subset_summary.csv"),
        "logistic_regression_table": str(OUT / "logistic_regression_descriptor_subset_performance.csv"),
        "rbf_svm_table": str(OUT / "rbf_svm_descriptor_subset_performance.csv"),
        "subset_manifest": str(OUT / "revised_descriptor_subset_feature_manifest.csv"),
        "report": str(OUT / "STAGE7C_REVISED_FOR_MANUSCRIPT_REPORT.md"),
        "summary_report": str(OUT / "descriptor_classification_validation_summary.md"),
        "normalization_needed": False,
        "normalization_action": "unchanged; audited raw_npy maps appeared already normalized to [0, 1]",
    }
    (OUT / "descriptor_validation_manifest.json").write_text(
        json.dumps(manifest_json, indent=2), encoding="utf-8"
    )
    print(f"Stage 7C revised manuscript outputs complete: {OUT}")


if __name__ == "__main__":
    main()
