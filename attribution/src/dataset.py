from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resolve_path(value: str, root: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(root) / path)


def prepare_test_dataframe(test_csv: str | Path, image_root: str | Path, mask_root: str | Path) -> pd.DataFrame:
    df = pd.read_csv(test_csv).copy()
    if "stem" not in df.columns:
        if "sample_id" in df.columns:
            df["stem"] = df["sample_id"]
        else:
            df["stem"] = df["filepath"].map(lambda x: Path(str(x)).stem)
    if "filepath" not in df.columns:
        raise ValueError("test.csv must contain filepath or image path column named 'filepath'.")
    df["filepath"] = df["filepath"].map(lambda x: resolve_path(str(x), image_root))
    if "maskpath" in df.columns:
        df["maskpath"] = df["maskpath"].map(lambda x: resolve_path(str(x), mask_root))
    else:
        df["maskpath"] = df["filepath"].map(lambda x: str(next(Path(mask_root).glob(f"**/{Path(x).stem}_mask.png"))))
    required = ["stem", "filepath", "maskpath", "class", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"test.csv missing columns: {missing}")
    for col in ["filepath", "maskpath"]:
        bad = [p for p in df[col].tolist() if not Path(p).exists()]
        if bad:
            raise FileNotFoundError(f"{col} has missing files. First missing: {bad[:5]}")
    return df.reset_index(drop=True)


def build_eval_transform(img_size: int, mean: list[float], std: list[float]):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class AttributionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_size: int, mean: list[float], std: list[float]):
        self.df = dataframe.reset_index(drop=True)
        self.img_size = img_size
        self.transform = build_eval_transform(img_size, mean, std)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        rgb = image.resize((self.img_size, self.img_size), resample=Image.Resampling.BILINEAR)
        mask = Image.open(row["maskpath"]).convert("L").resize((self.img_size, self.img_size), resample=Image.Resampling.NEAREST)
        mask_np = (np.array(mask) > 0).astype(np.uint8)
        return {
            "image": self.transform(image),
            "rgb": torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0).permute(2, 0, 1),
            "mask": torch.from_numpy(mask_np),
            "label": int(row["label"]),
            "stem": str(row["stem"]),
            "class": str(row["class"]),
            "filepath": str(row["filepath"]),
            "maskpath": str(row["maskpath"]),
        }

