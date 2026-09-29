"""Hierarchical forecast reconciliation with MinT-shrink (Wickramasuriya, Athanasopoulos & Hyndman, 2019).

Node ordering: y = [y_agg ; b],  y_agg = A b,  S = [A ; I_m].
Projection form of MinT (never materialises S or the n x n matrix W):

    y_tilde = y_hat - W C^T (C W C^T)^{-1} C y_hat,     C = [I_{n_a}, -A]

with W = lambda * D + (1 - lambda) * Sigma_hat stored as diag + low-rank:
    W = diag(delta) + U^T U,   U = sqrt((1 - lambda) / T) * X D^{1/2},   X = R D_raw^{-1/2}
and W C^T applied as a factored operator  [delta_a z ; -delta_b (A^T z)] + U^T (U C^T z),
so memory is O(n_a^2 + T * n + nnz(A)): neither W, S nor the (n x n_a) matrix W C^T is ever
formed. A may be dense or a torch sparse (COO/CSR) matrix; the sparse path is what makes
589k-series hierarchies (VISUELLE 2.0) fit in memory.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Hierarchy:
    """Grouped retail hierarchy: national total, FC totals, article-national totals, article x FC bottoms.

    Bottom index i = article * n_fc + fc.
    """

    n_articles: int
    n_fc: int

    @property
    def n_bottom(self) -> int:
        return self.n_articles * self.n_fc

    @property
    def n_agg(self) -> int:
        return 1 + self.n_fc + self.n_articles

    @property
    def n_total(self) -> int:
        return self.n_agg + self.n_bottom

    def level_slices(self) -> List[Tuple[str, slice]]:
        f, a = self.n_fc, self.n_articles
        return [
            ("national", slice(0, 1)),
            ("fulfilment_centre", slice(1, 1 + f)),
            ("article_national", slice(1 + f, 1 + f + a)),
            ("article_x_fc", slice(self.n_agg, self.n_total)),
        ]

    def aggregation_matrix(self, dtype: torch.dtype = torch.float32, sparse: bool = False) -> torch.Tensor:
        """A (n_agg, n_bottom). ``sparse=True`` returns a coalesced COO tensor with 3 * n_bottom non-zeros."""
        a_idx = torch.arange(self.n_articles).repeat_interleave(self.n_fc)
        f_idx = torch.arange(self.n_fc).repeat(self.n_articles)
        cols = torch.arange(self.n_bottom)
        if sparse:
            rows = torch.cat([torch.zeros_like(cols), 1 + f_idx, 1 + self.n_fc + a_idx])
            idx = torch.stack([rows, cols.repeat(3)])
            vals = torch.ones(3 * self.n_bottom, dtype=dtype)
            return torch.sparse_coo_tensor(idx, vals, (self.n_agg, self.n_bottom), check_invariants=False).coalesce()
        A = torch.zeros(self.n_agg, self.n_bottom, dtype=dtype)
        A[0] = 1.0
        A[1 + f_idx, cols] = 1.0
        A[1 + self.n_fc + a_idx, cols] = 1.0
        return A

    def aggregate_upper(self, bottom: np.ndarray) -> np.ndarray:
        """bottom (m, ...) -> aggregate nodes only (n_agg, ...), without copying the bottom level."""
        b = bottom.reshape(self.n_articles, self.n_fc, *bottom.shape[1:])
        return np.concatenate([b.sum((0, 1))[None], b.sum(0), b.sum(1)], axis=0)

    def aggregate(self, bottom: np.ndarray) -> np.ndarray:
        """bottom (m, ...) -> full node array (n, ...), coherent by construction."""
        return np.concatenate([self.aggregate_upper(bottom), bottom], axis=0)


def shrinkage_intensity(X: torch.Tensor) -> float:
    """Schäfer-Strimmer shrinkage intensity towards a diagonal target, computed in O(T^2 n).

    X: (T, n) residuals scaled to unit mean square per column (zero columns allowed).
    Matches ``hts::shrink.estim``:
        lambda = sum_{i!=j} Var(r_ij) / sum_{i!=j} r_ij^2
    """
    T = X.shape[0]
    if T < 3:
        return 1.0
    X = X.double()
    X2 = X * X
    row = X2.sum(1)
    sum_w2_off = (row * row).sum() - (X2 * X2).sum()  # Σ_k Σ_{i≠j} x_ki² x_kj²
    gram = X @ X.T
    diag = X2.sum(0)
    s_off = (gram * gram).sum() - (diag * diag).sum()  # Σ_{i≠j} (X^T X)_ij²
    v_sum = (sum_w2_off - s_off / T) / (T * (T - 1))
    d_sum = s_off / T**2
    if d_sum <= 0:
        return 1.0
    return float(torch.clamp(v_sum / d_sum, 0.0, 1.0))


class MinTReconciler:
    """MinT with shrinkage covariance, optional per-node prior variances for nodes without history."""

    def __init__(self, A: torch.Tensor, var_floor: float = 1e-6) -> None:
        self.sparse = A.layout != torch.strided
        if self.sparse:
            A = A.to_sparse_coo().coalesce().double()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta.*")
                self.A, self.At = A.to_sparse_csr(), A.t().coalesce().to_sparse_csr()
            self._A_coo = A
        else:
            self.A = A.double()
            self.At = self.A.T
        self.n_agg, self.n_bottom = self.A.shape
        self.var_floor = var_floor
        self.lam: Optional[float] = None

    def _Ab(self, b: torch.Tensor) -> torch.Tensor:
        """A @ b for b (m, K)."""
        return self.A @ b.contiguous()

    def _Atz(self, z: torch.Tensor) -> torch.Tensor:
        """A^T @ z for z (n_a, K)."""
        return self.At @ z.contiguous()

    def _A_diag_At(self, d: torch.Tensor) -> torch.Tensor:
        """Dense (n_a, n_a) matrix A diag(d) A^T."""
        if not self.sparse:
            return (self.A * d) @ self.A.T
        A = self._A_coo
        idx = A.indices()
        Ad = torch.sparse_coo_tensor(idx, A.values() * d[idx[1]], A.shape, check_invariants=False).coalesce()
        return torch.sparse.mm(Ad, A.t().coalesce()).to_dense()

    def fit(self, residuals: torch.Tensor, prior_var: Optional[torch.Tensor] = None, lam: Optional[float] = None) -> "MinTReconciler":
        """residuals: (T, n) base-forecast errors ordered [agg ; bottom]. prior_var: (n,) with NaN = use data."""
        R = residuals.double()
        T, n = R.shape
        if n != self.n_agg + self.n_bottom:
            raise ValueError("residual width does not match hierarchy")
        raw = (R * R).mean(0)
        has_data = raw > self.var_floor
        if prior_var is not None:
            pv = prior_var.double()
            override = ~torch.isnan(pv)
            has_data = has_data & ~override
        X = torch.where(has_data, R / raw.clamp_min(self.var_floor).sqrt(), torch.zeros_like(R))
        D = torch.where(has_data, raw, torch.full_like(raw, self.var_floor))
        if prior_var is not None:
            D = torch.where(override, pv.clamp_min(self.var_floor), D)
        self.lam = shrinkage_intensity(X[:, has_data]) if lam is None else float(lam)
        c_diag = (X * X).mean(0)  # 1 for nodes with data, 0 otherwise
        delta = self.lam * D + (1.0 - self.lam) * D * (1.0 - c_diag)
        U = X * D.sqrt() * np.sqrt((1.0 - self.lam) / T)

        na = self.n_agg
        UC = U[:, :na] - self._Ab(U[:, na:].T).T  # (T, n_a) = U C^T
        CWC = torch.diag(delta[:na]) + self._A_diag_At(delta[na:]) + UC.T @ UC
        CWC = 0.5 * (CWC + CWC.T)
        jitter = 1e-10 * CWC.diagonal().mean()
        self._chol = torch.linalg.cholesky(CWC + jitter * torch.eye(na, dtype=CWC.dtype))
        self._delta, self._U, self._UC = delta, U, UC
        return self

    def _WC(self, z: torch.Tensor) -> torch.Tensor:
        """W C^T z for z (n_a, K), without forming the (n, n_a) matrix."""
        na = self.n_agg
        low = self._U.T @ (self._UC @ z)  # (n, K)
        top = self._delta[:na, None] * z
        bot = -self._delta[na:, None] * self._Atz(z)
        return torch.cat([top, bot], 0) + low

    def reconcile(self, y_hat: torch.Tensor, nonnegative: bool = True) -> torch.Tensor:
        """y_hat: (n, K) base forecasts -> coherent (n, K)."""
        if self.lam is None:
            raise RuntimeError("call fit() first")
        y = y_hat.double()
        na = self.n_agg
        Cy = y[:na] - self._Ab(y[na:])
        y_t = y - self._WC(torch.cholesky_solve(Cy, self._chol))
        if nonnegative:
            b = y_t[na:].clamp_min(0.0)
            y_t = torch.cat([self._Ab(b), b], 0)
        return y_t.to(y_hat.dtype)

    def bottom_up(self, bottom: torch.Tensor) -> torch.Tensor:
        b = bottom.double()
        return torch.cat([self._Ab(b), b], 0).to(bottom.dtype)

    def reconcile_quantiles(self, q_hat: torch.Tensor, point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """q_hat: (n, K, Q) base quantiles, point: (n, K) base *mean* forecasts.

        MinT is an expectation-level projection, so the means are reconciled (sums of bottom
        medians are biased low for right-skewed demand). Each node's quantiles are shifted by
        its own mean adjustment, preserving level-specific predictive spread; then clipped
        at zero and sorted. Returns (quantiles, coherent means)."""
        pt = self.reconcile(point)
        out = (q_hat + (pt - point).unsqueeze(-1)).clamp_min(0.0)
        return torch.sort(out, dim=-1).values, pt


def swanson_mean(q: np.ndarray, quantiles) -> np.ndarray:
    """Swanson's rule E[Y] ~ 0.3 q10 + 0.4 q50 + 0.3 q90 (mean from a 3-point quantile forecast)."""
    qs = [round(float(x), 6) for x in quantiles]
    try:
        i10, i50, i90 = qs.index(0.1), qs.index(0.5), qs.index(0.9)
    except ValueError as e:
        raise ValueError("swanson_mean needs the 0.1, 0.5 and 0.9 quantiles") from e
    return 0.3 * q[..., i10] + 0.4 * q[..., i50] + 0.3 * q[..., i90]


class SeasonalRidgeForecaster:
    """Direct multi-horizon ridge per series in log1p space (base forecasts for aggregate nodes).

    Features for target step t = o + h: intercept, same-phase anchor y[t - s*(h//s + 1)] (s = season),
    trailing ``level_window``-step mean, phase dummies (s - 1 of them), known covariates at t.
    Quantiles from per-series in-sample log-residual quantiles.

    Daily data: season=7 (phase = day of week), level_window=28, min_history=35 (the v1-v3 setting).
    Weekly data: season=1 gives a last-value anchor y[o-1] and no dummies; season=52 needs > 1 year of history.
    Origins must satisfy o >= max(season, level_window).
    """

    def __init__(
        self, horizon: int, quantiles, alpha: float = 1.0, min_history: int = 35, chunk: int = 256,
        season: int = 7, level_window: int = 28,
    ) -> None:
        self.H, self.quantiles, self.alpha, self.min_history, self.chunk = horizon, list(quantiles), alpha, min_history, chunk
        self.season, self.level_window = int(season), int(level_window)

    def _design(self, Ylog: np.ndarray, cov: np.ndarray, phase: np.ndarray, origins: np.ndarray) -> np.ndarray:
        s, _ = Ylog.shape
        S, W = self.season, self.level_window
        if origins.min() < max(S, W):
            raise ValueError(f"origins must be >= max(season, level_window) = {max(S, W)}")
        h = np.arange(self.H)
        t = origins[:, None] + h[None]  # (O, H)
        anchor = Ylog[:, t - S * (h // S + 1)[None]]  # (s, O, H)
        csum = np.concatenate([np.zeros((s, 1)), np.cumsum(Ylog, 1)], 1)
        level = (csum[:, origins] - csum[:, origins - W]) / float(W)  # (s, O)
        level = np.broadcast_to(level[:, :, None], anchor.shape)
        parts = [np.ones(anchor.shape + (1,)), anchor[..., None], level[..., None]]
        if S > 1:
            ph = np.eye(S)[phase[t]][..., 1:]  # (O, H, S-1)
            parts.append(np.broadcast_to(ph[None], (s,) + ph.shape))
        parts.append(cov[:, t])  # (s, O, H, p)
        return np.concatenate(parts, -1)

    def fit(self, Y: np.ndarray, cov: np.ndarray, phase: np.ndarray, launch: np.ndarray, train_origins: np.ndarray) -> "SeasonalRidgeForecaster":
        Ylog = np.log1p(Y)
        self.beta, self.resid_q = [], []
        for s0 in range(0, Y.shape[0], self.chunk):
            sl = slice(s0, s0 + self.chunk)
            X = self._design(Ylog[sl], cov[sl], phase, train_origins)  # (s, O, H, p)
            y = Ylog[sl][:, train_origins[:, None] + np.arange(self.H)[None]]
            w = (train_origins[None, :] - launch[sl, None] >= self.min_history).astype(float)[:, :, None]  # (s, O, 1)
            Xw = X * w[..., None]
            p = X.shape[-1]
            XtX = np.einsum("sohp,sohq->spq", Xw, X) + self.alpha * np.eye(p)[None]
            Xty = np.einsum("sohp,soh->sp", Xw, y)
            beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
            res = y - np.einsum("sohp,sp->soh", X, beta)
            res = np.where(w > 0, res, np.nan).reshape(res.shape[0], -1)
            rq = np.nanquantile(res, self.quantiles, axis=1).T if np.isfinite(res).any() else np.zeros((res.shape[0], len(self.quantiles)))
            self.beta.append(beta)
            self.resid_q.append(np.nan_to_num(rq))
        self.beta = np.concatenate(self.beta)
        self.resid_q = np.concatenate(self.resid_q)
        return self

    def predict(self, Y: np.ndarray, cov: np.ndarray, phase: np.ndarray, launch: np.ndarray, origins: np.ndarray):
        """Returns quantiles (s, O, H, Q) in original units and a history-ok mask (s, O)."""
        Ylog = np.log1p(Y)
        outs = []
        for s0 in range(0, Y.shape[0], self.chunk):
            sl = slice(s0, s0 + self.chunk)
            X = self._design(Ylog[sl], cov[sl], phase, origins)
            mu = np.einsum("sohp,sp->soh", X, self.beta[sl])
            outs.append(np.expm1(mu[..., None] + self.resid_q[sl][:, None, None, :]).clip(min=0.0))
        ok = origins[None, :] - launch[:, None] >= self.min_history
        return np.concatenate(outs), ok
