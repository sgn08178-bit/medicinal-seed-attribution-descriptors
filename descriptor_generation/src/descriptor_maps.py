from __future__ import annotations

import cv2
import numpy as np
import pywt
from scipy.ndimage import distance_transform_edt
from skimage.feature import local_binary_pattern
from skimage.filters import gabor


CATEGORY_BY_DESCRIPTOR = {
    "Brightness": "Color and intensity",
    "LAB_L": "Color and intensity",
    "LAB_Chroma": "Color and intensity",
    "Saturation_HSV": "Color and intensity",
    "FFT_LowPass": "Spatial frequency",
    "FFT_HighPass": "Spatial frequency",
    "LBP": "Texture",
    "Edge_Sobel": "Edge and shape related",
    "Curvature_Laplacian": "Edge and shape related",
    "FourierDescriptor": "Edge and shape related",
    "DistanceTransform": "Edge and shape related",
}


def normalize_map(arr: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(x)
    if mask is not None:
        valid &= mask.astype(bool)
    if valid.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)
    vals = x[valid]
    mn, mx = float(np.nanmin(vals)), float(np.nanmax(vals))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - mn) / (mx - mn)
    return np.clip(out, 0, 1).astype(np.float32)


def descriptor_category(name: str) -> str:
    if name.startswith("Gabor"):
        return "Texture"
    if name.startswith("Wavelet"):
        return "Spatial frequency"
    return CATEGORY_BY_DESCRIPTOR.get(name, "Other")


def compute_descriptor_maps(rgb: np.ndarray, fg: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    h, w = fg.shape
    img_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    fg_bool = fg.astype(bool)
    out: dict[str, np.ndarray] = {}

    out["Brightness"] = normalize_map(gray, fg_bool)

    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    out["LAB_L"] = normalize_map(lab[:, :, 0], fg_bool)
    out["LAB_Chroma"] = normalize_map(np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2), fg_bool)

    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    out["Saturation_HSV"] = normalize_map(hsv[:, :, 1].astype(np.float32), fg_bool)

    lbp = local_binary_pattern(
        gray,
        P=int(cfg["lbp_n_points"]),
        R=int(cfg["lbp_radius"]),
        method=str(cfg.get("lbp_method", "uniform")),
    ).astype(np.float32)
    out["LBP"] = normalize_map(lbp, fg_bool)

    for freq in cfg["gabor_frequencies"]:
        for theta_deg in cfg["gabor_thetas"]:
            fr, fi = gabor(gray, frequency=float(freq), theta=np.radians(float(theta_deg)))
            name = f"Gabor_f{float(freq):.1f}_t{int(theta_deg)}°"
            out[name] = normalize_map(np.sqrt(fr ** 2 + fi ** 2), fg_bool)

    gray_fg = gray * fg_bool.astype(np.float32)
    fft_shift = np.fft.fftshift(np.fft.fft2(gray_fg))
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * float(cfg["fft_low_pass_ratio"]) / 2)
    ys, xs = np.ogrid[:h, :w]
    circ = ((ys - cy) ** 2 + (xs - cx) ** 2) <= radius ** 2
    lp = np.zeros((h, w), dtype=complex)
    lp[circ] = fft_shift[circ]
    out["FFT_LowPass"] = normalize_map(np.abs(np.fft.ifft2(np.fft.ifftshift(lp))), fg_bool)
    hp = fft_shift.copy()
    hp[circ] = 0
    out["FFT_HighPass"] = normalize_map(np.abs(np.fft.ifft2(np.fft.ifftshift(hp))), fg_bool)

    coeffs = pywt.wavedec2(gray_fg, wavelet=str(cfg["wavelet"]), level=int(cfg["wavelet_level"]))
    for level_idx in range(1, int(cfg["wavelet_level"]) + 1):
        cH, cV, cD = coeffs[level_idx]
        for suffix, coeff in zip(["H", "V", "D"], [cH, cV, cD]):
            energy = cv2.resize((coeff ** 2).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            out[f"Wavelet_L{level_idx}_{suffix}"] = normalize_map(energy, fg_bool)

    gray64 = gray.astype(np.float64)
    sx = cv2.Sobel(gray64, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray64, cv2.CV_64F, 0, 1, ksize=3)
    out["Edge_Sobel"] = normalize_map(np.sqrt(sx ** 2 + sy ** 2), fg_bool)
    out["Curvature_Laplacian"] = normalize_map(np.abs(cv2.Laplacian(gray64, cv2.CV_64F)), fg_bool)
    out["DistanceTransform"] = normalize_map(distance_transform_edt(fg_bool), fg_bool)

    fg_u8 = fg_bool.astype(np.uint8)
    contours, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fd_map = np.zeros((h, w), dtype=np.float32)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        pts_complex = cnt[:, 0, 0].astype(float) + 1j * cnt[:, 0, 1].astype(float)
        if len(pts_complex) >= 4:
            fft_cnt = np.fft.fft(pts_complex)
            n_keep = min(int(cfg["fd_n_components"]), len(fft_cnt) // 2)
            filt = np.zeros_like(fft_cnt)
            filt[:n_keep] = fft_cnt[:n_keep]
            filt[-n_keep:] = fft_cnt[-n_keep:]
            recon = np.fft.ifft(filt)
            rx = np.clip(recon.real.astype(int), 0, w - 1)
            ry = np.clip(recon.imag.astype(int), 0, h - 1)
            recon_pts = np.stack([rx, ry], axis=1).reshape(-1, 1, 2).astype(np.int32)
            recon_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(recon_mask, [recon_pts], -1, 1, thickness=1)
            fd_map = 1.0 / (distance_transform_edt(1 - np.clip(recon_mask, 0, 1)) + 1.0)
    out["FourierDescriptor"] = normalize_map(fd_map, fg_bool)

    return {k: np.where(fg_bool, v.astype(np.float32), 0.0).astype(np.float32) for k, v in out.items()}


def descriptor_quality(arr: np.ndarray, fg: np.ndarray) -> dict:
    valid = fg.astype(bool)
    vals = arr[valid] if valid.sum() else arr.ravel()
    return {
        "min": float(np.nanmin(vals)) if vals.size else float("nan"),
        "max": float(np.nanmax(vals)) if vals.size else float("nan"),
        "mean": float(np.nanmean(vals)) if vals.size else float("nan"),
        "std": float(np.nanstd(vals)) if vals.size else float("nan"),
        "foreground_pixel_count": int(valid.sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "all_zero": bool(np.nanmax(np.abs(vals)) < 1e-12) if vals.size else True,
    }
