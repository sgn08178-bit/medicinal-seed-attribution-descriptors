from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def fdr_bh(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return out.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[valid] = tmp
    return out.tolist()


def one_sample_ttest(values: pd.Series | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan"), float("nan")
    stat, p = stats.ttest_1samp(arr, popmean=0.0)
    return float(stat), float(p)
