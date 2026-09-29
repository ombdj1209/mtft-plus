"""Conformal quantile calibration (split-conformal, per quantile level, in log1p space).

For quantile level tau and calibration pairs (y, q_tau):
    delta_tau = Quantile_tau( log1p(y) - log1p(q_tau) )        (finite-sample corrected)
    q'_tau    = expm1( log1p(q_tau) + delta_tau )
so that the empirical P(y <= q'_tau) = tau on the calibration set. Working in log1p space makes one
shift valid across nodes of very different scale (e.g. all article-national totals). The median is
left untouched by default so point accuracy (WAPE / MASE) is unchanged.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class LogConformalCalibrator:
    def __init__(self, quantiles: Sequence[float], calibrate: Optional[Sequence[int]] = None) -> None:
        self.quantiles = list(quantiles)
        med = int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))
        self.calibrate = [i for i in range(len(self.quantiles)) if i != med] if calibrate is None else list(calibrate)
        self.delta: Dict[int, np.ndarray] = {}

    def fit(self, y: np.ndarray, q: np.ndarray, mask: Optional[np.ndarray] = None, horizon_axis: Optional[int] = None) -> "LogConformalCalibrator":
        """y (...,), q (..., Q). If horizon_axis is given, one shift per horizon step."""
        m = np.ones_like(y, bool) if mask is None else mask.astype(bool)
        ly = np.log1p(y)
        for i in self.calibrate:
            tau = self.quantiles[i]
            r = ly - np.log1p(q[..., i])
            if horizon_axis is None:
                self.delta[i] = np.asarray(self._cq(r[m], tau))
            else:
                r_h = np.moveaxis(r, horizon_axis, 0)
                m_h = np.moveaxis(m, horizon_axis, 0)
                self.delta[i] = np.array([self._cq(r_h[h][m_h[h]], tau) for h in range(r_h.shape[0])])
        self.horizon_axis = horizon_axis
        return self

    @staticmethod
    def _cq(r: np.ndarray, tau: float) -> float:
        """Finite-sample conformal quantile: the ceil((n+1)tau)-th order statistic (upper tail) or
        floor((n+1)tau)-th (lower tail), clipped to the sample range."""
        n = r.size
        if n == 0:
            return 0.0
        rs = np.sort(r)
        k = int(np.ceil((n + 1) * tau)) if tau >= 0.5 else int(np.floor((n + 1) * tau))
        return float(rs[min(max(k, 1), n) - 1])

    def transform(self, q: np.ndarray) -> np.ndarray:
        out = q.copy()
        for i, d in self.delta.items():
            if self.horizon_axis is not None and d.ndim == 1:
                shape = [1] * (q.ndim - 1)
                shape[self.horizon_axis] = -1
                d = d.reshape(shape)
            out[..., i] = np.expm1(np.log1p(q[..., i]) + d).clip(min=0.0)
        return np.sort(out, axis=-1)
