from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from .dataset import CLASS_ORDER


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_preds, all_trues = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_trues.extend(labels.numpy().tolist())
    return np.vstack(all_probs), np.asarray(all_preds), np.asarray(all_trues)


def save_classification_outputs(df: pd.DataFrame, probs: np.ndarray, preds: np.ndarray, trues: np.ndarray, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = df.copy().reset_index(drop=True)
    pred_df["true_label"] = [CLASS_ORDER[i] for i in trues]
    pred_df["pred_label"] = [CLASS_ORDER[i] for i in preds]
    pred_df["confidence"] = probs.max(axis=1)
    for i, cls in enumerate(CLASS_ORDER):
        pred_df[f"prob_{cls}"] = probs[:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    report_dict = classification_report(trues, preds, target_names=CLASS_ORDER, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(out_dir / "classification_report.csv")
    per_class = report_df.loc[CLASS_ORDER].reset_index().rename(columns={"index": "class"})
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)

    counts = confusion_matrix(trues, preds, labels=list(range(len(CLASS_ORDER))))
    counts_df = pd.DataFrame(counts, index=CLASS_ORDER, columns=CLASS_ORDER)
    counts_df.to_csv(out_dir / "confusion_matrix_counts.csv")
    norm = counts.astype(float) / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
    norm_df = pd.DataFrame(norm, index=CLASS_ORDER, columns=CLASS_ORDER)
    norm_df.to_csv(out_dir / "confusion_matrix_normalized.csv")

    return {
        "test_accuracy": float(accuracy_score(trues, preds)),
        "test_precision_macro": float(precision_score(trues, preds, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(trues, preds, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(trues, preds, average="macro", zero_division=0)),
    }
