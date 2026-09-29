"""Probabilistic multi-horizon losses.

Pinball loss for quantile level q and residual u = y - y_hat:

    rho_q(u) = max(q * u, (q - 1) * u)

The multi-horizon objective averages over valid (series, horizon) cells and sums over
quantile levels, optionally weighting quantile levels (e.g. up-weighting q10 for
perishable SKUs where over-forecasting creates waste) and individual samples.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


def newsvendor_quantile(underage_cost: float, overage_cost: float) -> float:
    """Critical ratio q* = c_u / (c_u + c_o) of the single-period newsvendor problem.

    c_u: margin lost per unit of unmet demand (stockout); c_o: cost per unsold unit
    (write-off for perishables). Short shelf life => large c_o => lower service quantile.
    """
    if underage_cost < 0 or overage_cost < 0 or underage_cost + overage_cost == 0:
        raise ValueError("costs must be non-negative and not both zero")
    return underage_cost / (underage_cost + overage_cost)


class QuantileLoss(nn.Module):
    """Masked, weighted pinball loss over a set of quantiles.

    Args:
        quantiles: quantile levels in (0, 1), sorted ascending.
        quantile_weights: optional per-quantile weights (same length as ``quantiles``).
        crossing_penalty: weight of a hinge penalty on quantile crossings
            (only needed when the output head is not monotone by construction).
    Shapes:
        y_pred: (..., Q)   y_true: (...)   mask / sample_weight: broadcastable to (...)
    """

    def __init__(
        self,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        quantile_weights: Optional[Sequence[float]] = None,
        crossing_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        q = torch.as_tensor(list(quantiles), dtype=torch.float32)
        if q.ndim != 1 or (q <= 0).any() or (q >= 1).any():
            raise ValueError("quantiles must be a 1-D sequence in (0, 1)")
        if not torch.all(q[1:] > q[:-1]):
            raise ValueError("quantiles must be strictly increasing")
        w = torch.ones_like(q) if quantile_weights is None else torch.as_tensor(list(quantile_weights), dtype=torch.float32)
        if w.shape != q.shape:
            raise ValueError("quantile_weights must match quantiles")
        self.register_buffer("quantiles", q, persistent=False)
        self.register_buffer("quantile_weights", w, persistent=False)
        self.crossing_penalty = float(crossing_penalty)

    def pinball(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Element-wise pinball loss, shape (..., Q)."""
        u = y_true.unsqueeze(-1) - y_pred
        q = self.quantiles.to(y_pred.dtype)
        return torch.maximum(q * u, (q - 1.0) * u)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y_pred.shape[:-1] != y_true.shape or y_pred.shape[-1] != self.quantiles.numel():
            raise ValueError(f"shape mismatch: y_pred {tuple(y_pred.shape)} vs y_true {tuple(y_true.shape)}")
        loss = (self.pinball(y_pred, y_true) * self.quantile_weights.to(y_pred.dtype)).sum(-1)
        if self.crossing_penalty > 0 and y_pred.shape[-1] > 1:
            loss = loss + self.crossing_penalty * torch.relu(y_pred[..., :-1] - y_pred[..., 1:]).sum(-1)
        weight = torch.ones_like(loss)
        if mask is not None:
            weight = weight * mask.to(loss.dtype)
        if sample_weight is not None:
            weight = weight * sample_weight.to(loss.dtype)
        denom = weight.sum().clamp_min(1.0)
        return (loss * weight).sum() / denom
