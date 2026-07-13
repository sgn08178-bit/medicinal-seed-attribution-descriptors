from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def resolve_path(value: str, root: str | Path) -> str:
    p = Path(str(value))
    return str(p if p.is_absolute() else Path(root) / p)


def prepare_test_dataframe(test_csv: str | Path, image_root: str | Path, mask_root: str | Path) -> pd.DataFrame:
    df = pd.read_csv(test_csv).copy()
    if "stem" not in df.columns:
        if "sample_id" in df.columns:
            df["stem"] = df["sample_id"]
        elif "filepath" in df.columns:
            df["stem"] = df["filepath"].map(lambda x: Path(str(x)).stem)
        else:
            raise ValueError("test.csv must contain stem, sample_id, or filepath.")
    if "filepath" not in df.columns:
        raise ValueError("test.csv must contain filepath.")
    df["filepath"] = df["filepath"].map(lambda x: resolve_path(x, image_root))
    if "maskpath" in df.columns:
        df["maskpath"] = df["maskpath"].map(lambda x: resolve_path(x, mask_root))
    else:
        def find_mask(stem: str) -> str:
            matches = list(Path(mask_root).glob(f"**/{stem}_mask.png"))
            if not matches:
                raise FileNotFoundError(f"Mask not found for {stem}")
            return str(matches[0])
        df["maskpath"] = df["stem"].map(find_mask)
    if "class" not in df.columns:
        raise ValueError("test.csv must contain class.")
    if "label" not in df.columns:
        classes = sorted(df["class"].unique())
        df["label"] = df["class"].map({c: i for i, c in enumerate(classes)})
    return df.reset_index(drop=True)


def load_rgb_224(path: str | Path, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask_224(path: str | Path, img_size: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((img_size, img_size), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0)


def stem_to_ig_abs_path(stem: str, directory: str | Path) -> Path:
    return Path(directory) / f"{stem}_zero_baseline_absolute.npy"


def stem_to_ig_pos_path(stem: str, directory: str | Path) -> Path:
    return Path(directory) / f"{stem}_zero_baseline_positive.npy"


def stem_to_gradcam_path(stem: str, directory: str | Path) -> Path:
    matches = list(Path(directory).glob(f"*_{stem}_raw.npy"))
    if len(matches) == 1:
        return matches[0]
    return Path(directory) / f"convnext_small_stages.2.blocks.26_{stem}_raw.npy"
