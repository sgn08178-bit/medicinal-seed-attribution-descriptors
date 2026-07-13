from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


CLASS_ORDER = ["ARSE", "ARSS", "PJNA", "PRDA", "PRPE"]
CLASS_NAMES = {
    "ARSE": "Armeniaca vulgaris",
    "ARSS": "Armeniaca sibirica",
    "PJNA": "Prunus japonica",
    "PRDA": "Prunus davidiana",
    "PRPE": "Prunus persica",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def infer_class_from_stem(stem: str) -> str:
    prefix = stem.split("_")[0]
    if prefix not in CLASS_TO_IDX:
        raise ValueError(f"Cannot infer class from stem: {stem}")
    return prefix


def find_mask_for_image(image_path: Path, mask_root: Path) -> Path:
    cls = infer_class_from_stem(image_path.stem)
    candidates = sorted(mask_root.glob(f"**/{image_path.stem}_mask.png"))
    if not candidates:
        candidates = sorted(mask_root.glob(f"**/{image_path.stem}.png"))
    if not candidates:
        raise FileNotFoundError(f"Mask not found for image: {image_path}")
    same_class = [p for p in candidates if p.parent.name.endswith(f"_{cls}")]
    return same_class[0] if same_class else candidates[0]


def build_manifest(image_root: str | Path, mask_root: str | Path) -> pd.DataFrame:
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    rows = []
    for image_path in sorted(image_root.glob("**/*.png")):
        stem = image_path.stem
        cls = infer_class_from_stem(stem)
        mask_path = find_mask_for_image(image_path, mask_root)
        rows.append(
            {
                "stem": stem,
                "filepath": str(image_path),
                "maskpath": str(mask_path),
                "class": cls,
                "class_name": CLASS_NAMES[cls],
                "label": CLASS_TO_IDX[cls],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No PNG images found in {image_root}")
    df = df.sort_values(["label", "stem"]).reset_index(drop=True)
    return df


def audit_dataset(image_root: str | Path, mask_root: str | Path, manifest: pd.DataFrame) -> dict:
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    image_stems = set(manifest["stem"])
    mask_paths = sorted(mask_root.glob("**/*.png"))
    mask_stems = set()
    for path in mask_paths:
        stem = path.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        mask_stems.add(stem)
    return {
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "image_png_count": int(len(image_stems)),
        "mask_png_count": int(len(mask_paths)),
        "manifest_rows": int(len(manifest)),
        "class_counts": manifest["class"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict(),
        "missing_mask_stems": sorted(image_stems - mask_stems),
        "extra_mask_stems": sorted(mask_stems - image_stems),
        "image_mask_stem_match": len(image_stems - mask_stems) == 0,
    }


class SeedClassificationDataset(Dataset):
    def __init__(self, csv_or_df: str | Path | pd.DataFrame, transform: Callable | None = None):
        self.df = pd.read_csv(csv_or_df) if not isinstance(csv_or_df, pd.DataFrame) else csv_or_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])
