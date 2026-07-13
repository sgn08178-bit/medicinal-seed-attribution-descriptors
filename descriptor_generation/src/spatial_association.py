from __future__ import annotations

import numpy as np
from scipy import stats


def spearman_foreground(a: np.ndarray, b: np.ndarray, fg: np.ndarray) -> float:
    mask = fg.astype(bool) & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    av = np.asarray(a[mask], dtype=float)
    bv = np.asarray(b[mask], dtype=float)
    if np.nanstd(av) < 1e-12 or np.nanstd(bv) < 1e-12:
        return float("nan")
    r, _ = stats.spearmanr(av, bv)
    return float(r)
