from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DISPLAY_NAMES = {
    "LAB_L": "LAB L",
    "LAB_Chroma": "LAB Chroma",
    "Saturation_HSV": "HSV saturation",
    "FFT_LowPass": "FFT low-pass",
    "FFT_HighPass": "FFT high-pass",
    "Wavelet_L1_H": "Wavelet L1 horizontal",
    "Wavelet_L1_V": "Wavelet L1 vertical",
    "Wavelet_L1_D": "Wavelet L1 diagonal",
    "Wavelet_L2_H": "Wavelet L2 horizontal",
    "Wavelet_L2_V": "Wavelet L2 vertical",
    "Wavelet_L2_D": "Wavelet L2 diagonal",
    "Gabor_f0.3_t0°": "Gabor f = 0.3, θ = 0°",
    "Curvature_Laplacian": "Laplacian-based local variation",
    "Edge_Sobel": "Sobel edge",
    "DistanceTransform": "Distance transform",
    "FourierDescriptor": "Fourier descriptor",
    "LBP": "Local binary pattern",
    "Brightness": "Brightness",
}


def display_name(name: str) -> str:
    m = re.match(r"Gabor_f([0-9.]+)_t([0-9]+)°", name)
    if m:
        return f"Gabor f = {m.group(1)}, θ = {m.group(2)}°"
    return DISPLAY_NAMES.get(name, name.replace("_", " "))


def save_map_png(arr: np.ndarray, mask: np.ndarray, path: str | Path, cmap: str = "gray") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = np.where(mask.astype(bool), arr, 0.0)
    plt.imsave(path, shown, cmap=cmap, vmin=0, vmax=1)


def save_overlay(rgb: np.ndarray, heat: np.ndarray, mask: np.ndarray, path: str | Path, alpha: float = 0.4, cmap: str = "turbo") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = plt.get_cmap(cmap)(np.clip(heat, 0, 1))[..., :3]
    m = mask.astype(bool)[..., None]
    out = np.where(m, (1 - alpha) * rgb + alpha * cm, rgb)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(path)
