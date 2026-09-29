"""Forecast accuracy metrics. Arrays: y (N, ...), point (N, ...), quantiles (N, ..., Q), mask like y."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _m(y: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    return np.ones_like(y, dtype=bool) if mask is None else mask.astype(bool)


def wape(y: np.ndarray, point: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    m = _m(y, mask)
    return float(np.abs(y - point)[m].sum() / max(np.abs(y)[m].sum(), 1e-9))


def seasonal_naive_scale(history: np.ndarray, valid: np.ndarray, season: int = 7) -> np.ndarray:
    """Per-series in-sample MAE of the seasonal naive forecast. history/valid: (N, T)."""
    diff = np.abs(history[:, season:] - history[:, :-season])
    ok = valid[:, season:] & valid[:, :-season]
    cnt = ok.sum(1)
    return np.where(cnt > 0, (diff * ok).sum(1) / np.maximum(cnt, 1), np.nan)


def mase(y: np.ndarray, point: np.ndarray, scale: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean over series of MAE / seasonal-naive scale; series with no scale or no valid cells are skipped."""
    m = _m(y, mask).reshape(len(y), -1)
    err = np.abs(y - point).reshape(len(y), -1)
    cnt = m.sum(1)
    ok = (cnt > 0) & np.isfinite(scale) & (scale > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean((err * m).sum(1)[ok] / cnt[ok] / scale[ok]))


def pinball(y: np.ndarray, q_pred: np.ndarray, quantiles: Sequence[float], mask: Optional[np.ndarray] = None) -> float:
    """Mean pinball loss per cell, averaged over quantiles (units of demand)."""
    q = np.asarray(quantiles)
    u = y[..., None] - q_pred
    loss = np.maximum(q * u, (q - 1) * u).mean(-1)
    m = _m(y, mask)
    return float(loss[m].mean())


def weighted_quantile_loss(y: np.ndarray, q_pred: np.ndarray, quantiles: Sequence[float], mask: Optional[np.ndarray] = None) -> float:
    """Scale-free wQL = mean_q 2 Σ ρ_q / Σ |y| (GluonTS definition)."""
    q = np.asarray(quantiles)
    u = y[..., None] - q_pred
    loss = np.maximum(q * u, (q - 1) * u)
    m = _m(y, mask)
    return float(np.mean(2 * loss[m].sum(0) / max(np.abs(y)[m].sum(), 1e-9)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: Optional[np.ndarray] = None, tol: float = 1e-6) -> float:
    """Share of cells inside [lo, hi], with a floating-point tolerance at the bounds. Without it, zero-inflated count
    data (y = 0 exactly) is counted as uncovered whenever a reconciliation shift leaves lo at, e.g., 3e-8 instead of 0."""
    m = _m(y, mask)
    return float(((y >= lo - tol) & (y <= hi + tol))[m].mean())


def evaluate(
    y: np.ndarray,
    q_pred: np.ndarray,
    quantiles: Sequence[float],
    scale: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    med = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    point = q_pred[..., med]
    return {
        "WAPE": wape(y, point, mask),
        "MASE": mase(y, point, scale, mask),
        "Pinball": pinball(y, q_pred, quantiles, mask),
        "wQL": weighted_quantile_loss(y, q_pred, quantiles, mask),
        "Cov80": coverage(y, q_pred[..., 0], q_pred[..., -1], mask),
    }
