from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split


def make_train_test_split(df: pd.DataFrame, test_size: float, seed: int):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
        shuffle=True,
    )
    return train_df.sort_values("stem").reset_index(drop=True), test_df.sort_values("stem").reset_index(drop=True)


def save_cv_splits(train_df: pd.DataFrame, out_dir: str | Path, n_folds: int, seed: int):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"]), 1):
        fold_train = train_df.iloc[tr_idx].sort_values("stem").reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].sort_values("stem").reset_index(drop=True)
        fold_train.to_csv(out_dir / f"fold{fold}_train.csv", index=False)
        fold_val.to_csv(out_dir / f"fold{fold}_val.csv", index=False)
        rows.append({"fold": fold, "train_n": len(fold_train), "val_n": len(fold_val)})
    return pd.DataFrame(rows)


def make_final_train_val_split(train_df: pd.DataFrame, final_val_size: float, seed: int):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=final_val_size, random_state=seed)
    tr_idx, val_idx = next(sss.split(train_df, train_df["label"]))
    final_train = train_df.iloc[tr_idx].sort_values("stem").reset_index(drop=True)
    final_val = train_df.iloc[val_idx].sort_values("stem").reset_index(drop=True)
    return final_train, final_val
