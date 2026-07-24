#!/usr/bin/env python3
"""Stage 7C: descriptor-subset classifiers using all valid descriptor maps.

This script does not reuse the incomplete Stage 5 descriptor summary table.
It recomputes foreground-restricted summary features directly from descriptor
map NPY files, using Stage 5 generated raw maps for train samples and Stage 3
raw maps for test samples.
"""

from __future__ import annotations

import json
import os
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(os.environ.get("MEDICINAL_SEED_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("MEDICINAL_SEED_DATA_ROOT", ROOT / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("MEDICINAL_SEED_RESULTS_ROOT", ROOT / "results")).resolve()
OUT = Path(
    os.environ.get(
        "MEDICINAL_SEED_DESCRIPTOR_CLASSIFIER_OUTPUT",
        RESULTS_ROOT / "descriptor_classifier",
    )
).resolve()
FIG = OUT / "figures"
CM_DIR = OUT / "confusion_matrices"

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
STAGE3_META = Path(
    os.environ.get(
        "MEDICINAL_SEED_DESCRIPTOR_METADATA_CSV",
        DATA_ROOT / "descriptor_generation_metadata.csv",
    )
).resolve()
STAGE3_QC = Path(
    os.environ.get(
        "MEDICINAL_SEED_DESCRIPTOR_QC_CSV",
        DATA_ROOT / "descriptor_quality_check.csv",
    )
).resolve()
STAGE3_TOP = Path(
    os.environ.get(
        "MEDICINAL_SEED_TOP_DESCRIPTORS_CSV",
        DATA_ROOT / "stage3_top_descriptors.csv",
    )
).resolve()
AUDIT_TABLE = Path(
    os.environ.get(
        "MEDICINAL_SEED_DESCRIPTOR_AUDIT_CSV",
        DATA_ROOT / "descriptor_inclusion_table.csv",
    )
).resolve()

SEED = 42
N_FOLDS = 5
CLASS_ORDER = ["ARSE", "ARSS", "PJNA", "PRDA", "PRPE"]
STAT_COLS = ["mean", "std", "median", "p90", "top10_mean", "iqr"]


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
}


def display_name(key: str) -> str:
    if key in DISPLAY:
        return DISPLAY[key]
    m = re.match(r"Gabor_f([0-9.]+)_t([0-9]+)°", key)
    if m:
        return f"Gabor response (f = {m.group(1)}, theta = {m.group(2)} deg)"
    return key


def category_for(key: str) -> str:
    if key in {"Brightness", "LAB_L", "LAB_Chroma", "Saturation_HSV"}:
        return "Color and intensity"
    if key == "LBP" or key.startswith("Gabor_"):
        return "Texture"
    if key.startswith("FFT_") or key.startswith("Wavelet_"):
        return "Spatial frequency"
    if key in {"Edge_Sobel", "Curvature_Laplacian", "DistanceTransform", "FourierDescriptor"}:
        return "Edge and shape related"
    return "Other"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)


def read_split() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train["split"] = "train"
    test["split"] = "test"
    df = pd.concat([train, test], ignore_index=True)
    return df


def available_descriptor_keys() -> list[str]:
    keys = {p.stem for d in [STAGE3_DESC_ROOT, STAGE5_DESC_ROOT] for p in d.glob("*/*.npy")}
    top_order = []
    if STAGE3_TOP.exists():
        top_order = pd.read_csv(STAGE3_TOP)["descriptor"].tolist()
    preferred = [
        "Brightness",
        "LAB_L",
        "LAB_Chroma",
        "Saturation_HSV",
        "FFT_LowPass",
        "FFT_HighPass",
        "LBP",
        "Edge_Sobel",
        "Curvature_Laplacian",
        "DistanceTransform",
        "FourierDescriptor",
    ]
    wave = sorted([k for k in keys if k.startswith("Wavelet_")])
    gabor = sorted([k for k in keys if k.startswith("Gabor_")])
    order = []
    for group in [top_order, preferred, wave, gabor, sorted(keys)]:
        for k in group:
            if k in keys and k not in order:
                order.append(k)
    return order


def descriptor_path(stem: str, split: str, key: str) -> tuple[Path | None, str]:
    roots = [STAGE5_DESC_ROOT, STAGE3_DESC_ROOT] if split == "train" else [STAGE3_DESC_ROOT, STAGE5_DESC_ROOT]
    for root in roots:
        p = root / stem / f"{key}.npy"
        if p.exists():
            return p, str(root)
    return None, ""


def load_fg_mask(mask_path: str, shape: tuple[int, int]) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    if m.size != (shape[1], shape[0]):
        m = m.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    arr = np.asarray(m)
    return arr > 127


def summarize_values(vals: np.ndarray) -> dict[str, float]:
    vals = vals.astype(np.float64)
    if vals.size == 0:
        return {c: np.nan for c in STAT_COLS}
    q90 = np.percentile(vals, 90)
    q75 = np.percentile(vals, 75)
    q25 = np.percentile(vals, 25)
    k = max(1, int(math.ceil(vals.size * 0.10)))
    top10 = np.partition(vals, vals.size - k)[-k:]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "median": float(np.median(vals)),
        "p90": float(q90),
        "top10_mean": float(np.mean(top10)),
        "iqr": float(q75 - q25),
    }


def build_descriptor_features(df: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows = []
    inventory = []
    invalid_rows = []
    per_desc = {
        k: {
            "missing_count": 0,
            "nan_map_count": 0,
            "inf_map_count": 0,
            "empty_foreground_count": 0,
            "constant_foreground_count": 0,
            "valid_sample_count": 0,
            "sources": set(),
        }
        for k in keys
    }

    for _, row in df.iterrows():
        stem = row["stem"]
        split = row["split"]
        for key in keys:
            path, source_root = descriptor_path(stem, split, key)
            if path is None:
                per_desc[key]["missing_count"] += 1
                invalid_rows.append({"stem": stem, "split": split, "descriptor": key, "issue": "missing_npy"})
                continue
            arr = np.load(path)
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                invalid_rows.append({"stem": stem, "split": split, "descriptor": key, "issue": f"invalid_shape_{arr.shape}"})
                continue
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            if nan_count:
                per_desc[key]["nan_map_count"] += 1
            if inf_count:
                per_desc[key]["inf_map_count"] += 1
            fg = load_fg_mask(row["maskpath"], arr.shape)
            if not fg.any():
                per_desc[key]["empty_foreground_count"] += 1
                invalid_rows.append({"stem": stem, "split": split, "descriptor": key, "issue": "empty_foreground"})
                continue
            vals = arr[fg]
            finite = np.isfinite(vals)
            if not finite.any():
                invalid_rows.append({"stem": stem, "split": split, "descriptor": key, "issue": "no_finite_foreground_values"})
                continue
            vals = vals[finite]
            if float(np.nanstd(vals)) == 0.0:
                per_desc[key]["constant_foreground_count"] += 1
            stats = summarize_values(vals)
            feature_rows.append(
                {
                    "stem": stem,
                    "split": split,
                    "class": row["class"],
                    "label": int(row["label"]),
                    "descriptor": key,
                    "display_name": display_name(key),
                    "category": category_for(key),
                    "source_root": source_root,
                    "path": str(path),
                    **stats,
                }
            )
            per_desc[key]["valid_sample_count"] += 1
            per_desc[key]["sources"].add(source_root)

    n_samples = len(df)
    valid_keys = []
    for key in keys:
        item = per_desc[key]
        valid = (
            item["valid_sample_count"] == n_samples
            and item["missing_count"] == 0
            and item["nan_map_count"] == 0
            and item["inf_map_count"] == 0
            and item["empty_foreground_count"] == 0
        )
        if valid:
            valid_keys.append(key)
        inventory.append(
            {
                "descriptor": key,
                "display_name": display_name(key),
                "category": category_for(key),
                "valid_for_stage7c": bool(valid),
                "valid_sample_count": item["valid_sample_count"],
                "expected_sample_count": n_samples,
                "missing_count": item["missing_count"],
                "nan_map_count": item["nan_map_count"],
                "inf_map_count": item["inf_map_count"],
                "empty_foreground_count": item["empty_foreground_count"],
                "constant_foreground_count": item["constant_foreground_count"],
                "source_roots": "; ".join(sorted(item["sources"])),
            }
        )
    feat = pd.DataFrame(feature_rows)
    inv = pd.DataFrame(inventory)
    invalid = pd.DataFrame(invalid_rows)
    if invalid.empty:
        invalid = pd.DataFrame(columns=["stem", "split", "descriptor", "issue"])
    return feat, inv, invalid


def pivot_features(features: pd.DataFrame, valid_keys: list[str]) -> pd.DataFrame:
    id_cols = ["stem", "split", "class", "label"]
    pieces = []
    use = features[features["descriptor"].isin(valid_keys)]
    for stat in STAT_COLS:
        p = use.pivot_table(index=id_cols, columns="descriptor", values=stat, aggfunc="first")
        p.columns = [f"{c}__{stat}" for c in p.columns]
        pieces.append(p)
    return pd.concat(pieces, axis=1).reset_index()


def build_subsets(valid_keys: list[str]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    valid = set(valid_keys)
    gabor = [k for k in valid_keys if k.startswith("Gabor_")]
    wave = [k for k in valid_keys if k.startswith("Wavelet_")]
    subsets_requested = {
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
        "texture_family": ["LBP", *gabor, *wave],
        "color_and_intensity": ["Brightness", "LAB_L", "LAB_Chroma", "Saturation_HSV"],
        "spatial_frequency": ["FFT_LowPass", "FFT_HighPass", *wave],
    }
    labels = {
        "true_all_valid_descriptor_maps": "True all valid descriptor maps",
        "selected_10_map_summarized_set": "Selected 10-map summarized set",
        "top3_ig_associated": "Top 3 IG-associated",
        "top5_ig_associated": "Top 5 IG-associated",
        "top7_ig_associated": "Top 7 IG-associated",
        "low_or_negative_ig_associated": "Low/negative IG-associated",
        "edge_and_shape": "Edge and shape subset",
        "texture_family": "Texture family subset",
        "color_and_intensity": "Color and intensity subset",
        "spatial_frequency": "Spatial frequency subset",
    }
    subsets = {}
    rows = []
    for sid, req in subsets_requested.items():
        included = [d for d in req if d in valid]
        missing = [d for d in req if d not in valid]
        subsets[sid] = included
        rows.append(
            {
                "subset_id": sid,
                "subset_label": labels[sid],
                "requested_descriptors": "; ".join(display_name(d) for d in req),
                "included_descriptors": "; ".join(display_name(d) for d in included),
                "included_descriptor_keys": "; ".join(included),
                "missing_requested_descriptors": "; ".join(display_name(d) for d in missing),
                "n_descriptor_maps": len(included),
                "n_numeric_features": len(included) * len(STAT_COLS),
            }
        )
    return subsets, pd.DataFrame(rows)


def feature_cols(descriptors: list[str], wide: pd.DataFrame) -> list[str]:
    return [f"{d}__{s}" for d in descriptors for s in STAT_COLS if f"{d}__{s}" in wide.columns]


def get_models() -> tuple[dict[str, object], dict[str, str]]:
    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED),
        "knn": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        alpha=1e-4,
                        max_iter=600,
                        early_stopping=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }
    status = {}
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
        status["xgboost"] = "available"
    except Exception as e:
        status["xgboost"] = f"unavailable: {e}"
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=SEED,
            verbose=-1,
        )
        status["lightgbm"] = "available"
    except Exception as e:
        status["lightgbm"] = f"unavailable: {e}"
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            loss_function="MultiClass",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
        status["catboost"] = "available"
    except Exception as e:
        status["catboost"] = f"unavailable: {e}"
    return models, status


def evaluate_one(subset_id: str, subset_label: str, descriptors: list[str], wide: pd.DataFrame, models: dict[str, object]) -> list[dict]:
    cols = feature_cols(descriptors, wide)
    train = wide[wide["split"] == "train"].copy()
    test = wide[wide["split"] == "test"].copy()
    X = train[cols].to_numpy(dtype=np.float64)
    y = train["label"].to_numpy(dtype=int)
    X_test = test[cols].to_numpy(dtype=np.float64)
    y_test = test["label"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = []
    for model_id, model in models.items():
        cv_acc = []
        cv_f1 = []
        for tr_idx, va_idx in skf.split(X, y):
            m = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[va_idx])
            cv_acc.append(accuracy_score(y[va_idx], pred))
            cv_f1.append(f1_score(y[va_idx], pred, average="macro", zero_division=0))
        final = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final.fit(X, y)
        pred_test = final.predict(X_test)
        out.append(
            {
                "subset_id": subset_id,
                "subset_label": subset_label,
                "model": model_id,
                "model_label": model_id.replace("_", " ").title(),
                "cv_accuracy_mean": float(np.mean(cv_acc)),
                "cv_accuracy_sd": float(np.std(cv_acc, ddof=1)),
                "cv_macro_f1_mean": float(np.mean(cv_f1)),
                "cv_macro_f1_sd": float(np.std(cv_f1, ddof=1)),
                "test_accuracy": float(accuracy_score(y_test, pred_test)),
                "test_macro_precision": float(precision_score(y_test, pred_test, average="macro", zero_division=0)),
                "test_macro_recall": float(recall_score(y_test, pred_test, average="macro", zero_division=0)),
                "test_macro_f1": float(f1_score(y_test, pred_test, average="macro", zero_division=0)),
                "n_descriptor_maps": int(len(descriptors)),
                "n_numeric_features": int(len(cols)),
                "included_descriptor_keys": "; ".join(descriptors),
                "included_descriptor_names": "; ".join(display_name(d) for d in descriptors),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            }
        )
    return out


def derive_tables(perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_model_by_subset = perf.loc[perf.groupby("subset_id")["test_macro_f1"].idxmax()].sort_values("subset_id").reset_index(drop=True)
    best_subset_by_model = perf.loc[perf.groupby("model")["test_macro_f1"].idxmax()].sort_values("model").reset_index(drop=True)
    true_base = perf[perf["subset_id"] == "true_all_valid_descriptor_maps"][["model", "test_macro_f1", "test_accuracy"]].rename(
        columns={"test_macro_f1": "true_all_test_macro_f1", "test_accuracy": "true_all_test_accuracy"}
    )
    ten_base = perf[perf["subset_id"] == "selected_10_map_summarized_set"][["model", "test_macro_f1", "test_accuracy"]].rename(
        columns={"test_macro_f1": "selected_10_test_macro_f1", "test_accuracy": "selected_10_test_accuracy"}
    )
    rel_true = perf.merge(true_base, on="model", how="left")
    rel_true["test_macro_f1_drop_from_true_all"] = rel_true["true_all_test_macro_f1"] - rel_true["test_macro_f1"]
    rel_true["test_macro_f1_relative_to_true_all"] = rel_true["test_macro_f1"] / rel_true["true_all_test_macro_f1"]
    rel_true["test_accuracy_drop_from_true_all"] = rel_true["true_all_test_accuracy"] - rel_true["test_accuracy"]
    rel_true["test_accuracy_relative_to_true_all"] = rel_true["test_accuracy"] / rel_true["true_all_test_accuracy"]
    rel_ten = perf.merge(ten_base, on="model", how="left")
    rel_ten["test_macro_f1_drop_from_selected_10"] = rel_ten["selected_10_test_macro_f1"] - rel_ten["test_macro_f1"]
    rel_ten["test_macro_f1_relative_to_selected_10"] = rel_ten["test_macro_f1"] / rel_ten["selected_10_test_macro_f1"]
    rel_ten["test_accuracy_drop_from_selected_10"] = rel_ten["selected_10_test_accuracy"] - rel_ten["test_accuracy"]
    rel_ten["test_accuracy_relative_to_selected_10"] = rel_ten["test_accuracy"] / rel_ten["selected_10_test_accuracy"]
    return best_model_by_subset, best_subset_by_model, rel_true, rel_ten


def save_best_confusions(best: pd.DataFrame, wide: pd.DataFrame, subsets: dict[str, list[str]], models: dict[str, object]) -> None:
    train = wide[wide["split"] == "train"].copy()
    test = wide[wide["split"] == "test"].copy()
    y_train = train["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)
    for _, row in best.iterrows():
        sid = row["subset_id"]
        model_id = row["model"]
        cols = feature_cols(subsets[sid], wide)
        m = clone(models[model_id])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(train[cols].to_numpy(dtype=np.float64), y_train)
        pred = m.predict(test[cols].to_numpy(dtype=np.float64))
        cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASS_ORDER))))
        cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
        cm_df.to_csv(CM_DIR / f"{sid}__{model_id}__confusion_matrix_counts.csv")
        pred_df = test[["stem", "class", "label"]].copy()
        pred_df["predicted_label"] = pred
        pred_df["predicted_class"] = [CLASS_ORDER[i] for i in pred]
        pred_df.to_csv(CM_DIR / f"{sid}__{model_id}__test_predictions.csv", index=False)


def make_figures(perf: pd.DataFrame, best: pd.DataFrame, rel_true: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    order = [
        "True all valid descriptor maps",
        "Selected 10-map summarized set",
        "Top 3 IG-associated",
        "Top 5 IG-associated",
        "Top 7 IG-associated",
        "Low/negative IG-associated",
        "Edge and shape subset",
        "Texture family subset",
        "Color and intensity subset",
        "Spatial frequency subset",
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.barplot(data=best, y="subset_label", x="test_macro_f1", order=order, color="#8fb9d6", ax=ax)
    ax.set_xlabel("Test macro F1")
    ax.set_ylabel("")
    ax.set_title("Best classifier performance by descriptor subset", fontsize=13)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "best_model_per_subset.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG / "best_model_per_subset.pdf", bbox_inches="tight")
    plt.close(fig)

    primary = rel_true[rel_true["subset_id"] != "true_all_valid_descriptor_maps"].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    sns.barplot(
        data=primary,
        y="subset_label",
        x="test_macro_f1_relative_to_true_all",
        hue="model_label",
        order=order[1:],
        ax=ax,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Relative test macro F1 vs true all valid descriptor maps")
    ax.set_ylabel("")
    ax.set_title("Relative performance compared with the true all-descriptor set", fontsize=13)
    ax.legend(title="Classifier", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "relative_performance_vs_true_all_descriptors.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG / "relative_performance_vs_true_all_descriptors.pdf", bbox_inches="tight")
    plt.close(fig)

    heat = perf.pivot_table(index="subset_label", columns="model_label", values="test_macro_f1")
    heat = heat.reindex(order)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sns.heatmap(heat, vmin=0, vmax=1, cmap="YlGnBu", annot=True, fmt=".3f", linewidths=0.4, cbar_kws={"label": "Test macro F1"}, ax=ax)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("")
    ax.set_title("Descriptor-subset classifier performance", fontsize=13)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIG / "subset_performance_heatmap.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG / "subset_performance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if cols is not None:
        df = df[cols]
    def esc(x):
        if pd.isna(x):
            return ""
        return str(x).replace("|", "\\|").replace("\n", "<br>")
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(esc(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def write_reports(
    inventory: pd.DataFrame,
    invalid: pd.DataFrame,
    manifest: pd.DataFrame,
    perf: pd.DataFrame,
    best: pd.DataFrame,
    best_by_model: pd.DataFrame,
    rel_true: pd.DataFrame,
    rel_ten: pd.DataFrame,
    optional_status: dict[str, str],
) -> None:
    n_found = len(inventory)
    n_valid = int(inventory["valid_for_stage7c"].sum())
    invalid_desc = inventory[~inventory["valid_for_stage7c"]]["display_name"].tolist()
    def best_row(sid: str) -> pd.Series:
        return best[best["subset_id"] == sid].iloc[0]
    true = best_row("true_all_valid_descriptor_maps")
    ten = best_row("selected_10_map_summarized_set")
    top3 = best_row("top3_ig_associated")
    top5 = best_row("top5_ig_associated")
    top7 = best_row("top7_ig_associated")
    low = best_row("low_or_negative_ig_associated")
    category_ids = ["edge_and_shape", "texture_family", "color_and_intensity", "spatial_frequency"]
    cat_best = best[best["subset_id"].isin(category_ids)].sort_values("test_macro_f1", ascending=False).iloc[0]

    best_cols = [
        "subset_label",
        "model_label",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "cv_accuracy_mean",
        "cv_accuracy_sd",
        "cv_macro_f1_mean",
        "cv_macro_f1_sd",
        "n_descriptor_maps",
        "n_numeric_features",
    ]
    report = [
        "# Stage 7C all-valid descriptor classifier report",
        "",
        "## Analysis design",
        "",
        "Stage 7C recomputed foreground-restricted descriptor summary features directly from descriptor map NPY files. It did not reuse the incomplete Stage 5 descriptor summary table as the full descriptor source.",
        "",
        "Train samples use Stage 5 generated raw descriptor maps because Stage 3 raw descriptor maps contain the independent test split only. Test samples use Stage 3 raw descriptor maps. This preserves the original train/test split while using raw descriptor maps for feature extraction.",
        "",
        "## Descriptor map inventory",
        "",
        f"- Individual descriptor maps found: {n_found}",
        f"- QC-valid descriptor maps included in true all-descriptor set: {n_valid}",
        f"- Invalid or dropped descriptor maps: {'none' if not invalid_desc else '; '.join(invalid_desc)}",
        "",
        md_table(inventory[["descriptor", "display_name", "category", "valid_for_stage7c", "valid_sample_count", "missing_count", "nan_map_count", "inf_map_count", "constant_foreground_count"]]),
        "",
        "## Feature subset manifest",
        "",
        md_table(manifest),
        "",
        "## Best model by subset",
        "",
        md_table(best, best_cols),
        "",
        "## Best subset by model",
        "",
        md_table(best_by_model, ["model_label", "subset_label", "test_accuracy", "test_macro_f1", "n_descriptor_maps"]),
        "",
        "## Required interpretation questions",
        "",
        f"1. Individual descriptor maps found: {n_found}.",
        f"2. Valid maps included in true all-descriptor set: {n_valid}.",
        f"3. Invalid or dropped maps: {'none' if not invalid_desc else '; '.join(invalid_desc)}.",
        f"4. Best model for true all valid descriptor maps: {true['model_label']} with test accuracy {true['test_accuracy']:.6f} and macro F1 {true['test_macro_f1']:.6f}.",
        f"5. Best model for selected 10-map set: {ten['model_label']} with test accuracy {ten['test_accuracy']:.6f} and macro F1 {ten['test_macro_f1']:.6f}.",
        f"6. Best Top 3/5/7 models: Top 3 {top3['model_label']} macro F1 {top3['test_macro_f1']:.6f}; Top 5 {top5['model_label']} macro F1 {top5['test_macro_f1']:.6f}; Top 7 {top7['model_label']} macro F1 {top7['test_macro_f1']:.6f}.",
        f"7. Top 5 relative to true all descriptor performance for its own model: {rel_true[(rel_true['subset_id']=='top5_ig_associated') & (rel_true['model']==top5['model'])]['test_macro_f1_relative_to_true_all'].iloc[0]:.6f}; Top 7: {rel_true[(rel_true['subset_id']=='top7_ig_associated') & (rel_true['model']==top7['model'])]['test_macro_f1_relative_to_true_all'].iloc[0]:.6f}.",
        f"8. Selected 10-map set relative to true all descriptor performance for its own best model: {rel_true[(rel_true['subset_id']=='selected_10_map_summarized_set') & (rel_true['model']==ten['model'])]['test_macro_f1_relative_to_true_all'].iloc[0]:.6f}.",
        f"9. Best category-based subset: {cat_best['subset_label']} using {cat_best['model_label']}, macro F1 {cat_best['test_macro_f1']:.6f}.",
        f"10. Low/negative IG-associated subset: {low['model_label']}, test accuracy {low['test_accuracy']:.6f}, macro F1 {low['test_macro_f1']:.6f}.",
        "11. Recommended placement: auxiliary descriptor-subset classification analysis, preferably supplementary unless the main Results need an additional validation paragraph.",
        "12. Safest interpretation: compact descriptor subsets contain class-discriminative visual information in foreground-restricted summary form, without implying a direct CNN mechanism.",
        "",
        "## Optional classifier availability",
        "",
        json.dumps(optional_status, indent=2),
        "",
        "## Interpretation boundaries",
        "",
        "- Do not claim that the CNN directly relies on these descriptors.",
        "- Do not claim causal model evidence.",
        "- Do not present these descriptors as definitive botanical diagnostic traits.",
        "- Do not present these descriptor subsets as uniquely privileged explanatory variables.",
        "- Use cautious terms such as class-discriminative visual information, descriptor summary features, compact descriptor subset, and foreground-restricted visual context.",
    ]
    (OUT / "STAGE7C_INTEGRATED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    def rel_value(sid: str, model: str, col: str) -> float:
        return float(rel_true[(rel_true["subset_id"] == sid) & (rel_true["model"] == model)][col].iloc[0])

    summary_lines = [
        "# Descriptor classification results",
        "",
        "## Best model by subset",
        "",
        md_table(best, best_cols + ["included_descriptor_names"]),
        "",
        "## Results summary",
        "",
        f"Descriptor map inventory: {n_found} individual descriptor maps were found; {n_valid} valid descriptor maps were included in the true all valid descriptor map set; invalid/dropped maps: {'none' if not invalid_desc else '; '.join(invalid_desc)}.",
        "",
        f"Best true all valid descriptor map set performance: {int(true['n_descriptor_maps'])} descriptor maps, {int(true['n_numeric_features'])} numeric features; best model {true['model_label']}; test accuracy {true['test_accuracy']:.6f}; test macro F1 {true['test_macro_f1']:.6f}.",
        "",
        f"Best selected 10-map set performance: best model {ten['model_label']}; test accuracy {ten['test_accuracy']:.6f}; test macro F1 {ten['test_macro_f1']:.6f}; relative macro F1 compared with true all valid descriptor set for the same model {rel_value('selected_10_map_summarized_set', ten['model'], 'test_macro_f1_relative_to_true_all'):.6f}.",
        "",
        f"Best Top 3 IG-associated subset performance: descriptors {top3['included_descriptor_names']}; best model {top3['model_label']}; test accuracy {top3['test_accuracy']:.6f}; test macro F1 {top3['test_macro_f1']:.6f}; relative macro F1 compared with true all valid descriptor set for the same model {rel_value('top3_ig_associated', top3['model'], 'test_macro_f1_relative_to_true_all'):.6f}.",
        "",
        f"Best Top 5 IG-associated subset performance: descriptors {top5['included_descriptor_names']}; best model {top5['model_label']}; test accuracy {top5['test_accuracy']:.6f}; test macro F1 {top5['test_macro_f1']:.6f}; relative macro F1 compared with true all valid descriptor set for the same model {rel_value('top5_ig_associated', top5['model'], 'test_macro_f1_relative_to_true_all'):.6f}.",
        "",
        f"Best Top 7 IG-associated subset performance: descriptors {top7['included_descriptor_names']}; best model {top7['model_label']}; test accuracy {top7['test_accuracy']:.6f}; test macro F1 {top7['test_macro_f1']:.6f}; relative macro F1 compared with true all valid descriptor set for the same model {rel_value('top7_ig_associated', top7['model'], 'test_macro_f1_relative_to_true_all'):.6f}.",
        "",
        f"Best category-based subset performance: {cat_best['subset_label']}; descriptors {cat_best['included_descriptor_names']}; best model {cat_best['model_label']}; test accuracy {cat_best['test_accuracy']:.6f}; test macro F1 {cat_best['test_macro_f1']:.6f}; relative macro F1 compared with true all valid descriptor set for the same model {rel_value(cat_best['subset_id'], cat_best['model'], 'test_macro_f1_relative_to_true_all'):.6f}.",
        "",
        f"Low/negative IG-associated subset performance: descriptors {low['included_descriptor_names']}; best model {low['model_label']}; test accuracy {low['test_accuracy']:.6f}; test macro F1 {low['test_macro_f1']:.6f}.",
        "",
        "Recommended manuscript action: use this as a supplementary or auxiliary descriptor-subset classification table. Mention briefly in Results only if the manuscript needs support that descriptor summary features contain class-discriminative visual information. Use the true all valid descriptor set, not the selected 10-map set, as the all-descriptor baseline. Safe Results sentence: Descriptor summary features computed within the seed foreground retained class-discriminative visual information across standard classifiers, with compact IG-associated subsets preserving part of the full descriptor-set performance. Safe Discussion sentence: These descriptor-subset results provide auxiliary foreground-restricted visual context and should not be interpreted as evidence that the CNN directly relies on predefined descriptors.",
    ]
    (OUT / "descriptor_classification_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    np.random.seed(SEED)
    df = read_split()
    keys = available_descriptor_keys()
    print(f"Found {len(keys)} descriptor keys.")
    features, inventory, invalid = build_descriptor_features(df, keys)
    inventory.to_csv(OUT / "descriptor_map_inventory.csv", index=False)
    invalid.to_csv(OUT / "invalid_or_dropped_descriptor_maps.csv", index=False)
    valid_keys = inventory[inventory["valid_for_stage7c"]]["descriptor"].tolist()
    features = features[features["descriptor"].isin(valid_keys)].copy()
    features.to_csv(OUT / "all_valid_descriptor_summary_features.csv", index=False)
    pd.DataFrame(
        [{"descriptor": k, "display_name": display_name(k), "category": category_for(k)} for k in keys]
    ).to_csv(OUT / "descriptor_key_to_display_name_mapping.csv", index=False)
    wide = pivot_features(features, valid_keys)
    subsets, feature_manifest = build_subsets(valid_keys)
    feature_manifest.to_csv(OUT / "descriptor_subset_feature_manifest.csv", index=False)
    models, optional_status = get_models()
    rows = []
    for sid, descs in subsets.items():
        label = feature_manifest.loc[feature_manifest["subset_id"] == sid, "subset_label"].iloc[0]
        print(f"Evaluating {label}: {len(descs)} descriptors")
        rows.extend(evaluate_one(sid, label, descs, wide, models))
    perf = pd.DataFrame(rows)
    perf.to_csv(OUT / "descriptor_subset_model_performance.csv", index=False)
    best, best_by_model, rel_true, rel_ten = derive_tables(perf)
    best.to_csv(OUT / "best_model_by_subset.csv", index=False)
    best_by_model.to_csv(OUT / "best_subset_by_model.csv", index=False)
    rel_true.to_csv(OUT / "relative_performance_vs_true_all_descriptors.csv", index=False)
    rel_ten.to_csv(OUT / "relative_performance_vs_selected_10_maps.csv", index=False)
    save_best_confusions(best, wide, subsets, models)
    make_figures(perf, best, rel_true)
    write_reports(inventory, invalid, feature_manifest, perf, best, best_by_model, rel_true, rel_ten, optional_status)
    manifest = {
        "stage": "Stage 7C all valid descriptor classifier",
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "inputs": {
            "train_csv": str(TRAIN_CSV),
            "test_csv": str(TEST_CSV),
            "stage3_descriptor_root": str(STAGE3_DESC_ROOT),
            "stage5_descriptor_root_for_train_maps": str(STAGE5_DESC_ROOT),
            "stage3_top_descriptors": str(STAGE3_TOP),
            "descriptor_set_audit": str(AUDIT_TABLE),
        },
        "outputs": {
            "summary_features": str(OUT / "all_valid_descriptor_summary_features.csv"),
            "inventory": str(OUT / "descriptor_map_inventory.csv"),
            "performance": str(OUT / "descriptor_subset_model_performance.csv"),
            "best_model_by_subset": str(OUT / "best_model_by_subset.csv"),
            "report": str(OUT / "STAGE7C_INTEGRATED_REPORT.md"),
            "summary_report": str(OUT / "descriptor_classification_summary.md"),
        },
    }
    (OUT / "descriptor_classification_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Stage 7C complete: {OUT}")


if __name__ == "__main__":
    main()
