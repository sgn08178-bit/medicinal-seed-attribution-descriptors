from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def minmax01(arr: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    valid = mask.astype(bool) if mask is not None else np.isfinite(arr)
    if valid.sum() == 0:
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[valid]
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def vis_map(raw: np.ndarray, mask: np.ndarray | None, low: float, high: float, sigma: float) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).copy()
    valid = mask.astype(bool) if mask is not None else np.isfinite(arr)
    if valid.sum() > 0:
        lo, hi = np.percentile(arr[valid], [low, high])
        arr = np.clip(arr, lo, hi)
    arr = minmax01(arr, valid)
    if sigma and sigma > 0:
        arr = gaussian_filter(arr, sigma=sigma)
        arr = minmax01(arr, valid)
    if mask is not None:
        arr = np.where(valid, arr, 0.0)
    return arr


def save_heatmap(arr: np.ndarray, path: str | Path, cmap: str = "magma") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, arr, cmap=cmap, vmin=0, vmax=1)


def save_overlay(rgb_chw: np.ndarray, attr: np.ndarray, mask: np.ndarray | None, path: str | Path, alpha: float = 0.45, cmap: str = "magma") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.transpose(rgb_chw, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)
    cm = plt.get_cmap(cmap)(np.clip(attr, 0, 1))[..., :3]
    if mask is not None:
        m = mask.astype(bool)[..., None]
        overlay = np.where(m, (1 - alpha) * rgb + alpha * cm, rgb * 0.45)
    else:
        overlay = (1 - alpha) * rgb + alpha * cm
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(path)

