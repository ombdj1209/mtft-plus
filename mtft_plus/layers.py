"""Temporal Fusion Transformer building blocks (Lim et al., 2021) plus a monotone quantile head."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class GatedLinearUnit(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(self.fc(self.dropout(x)), dim=-1)


class GateAddNorm(nn.Module):
    """LayerNorm(skip + GLU(x))."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(d_in, d_out, dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(self.glu(x) + skip)


class GatedResidualNetwork(nn.Module):
    """GRN(a, c) = LayerNorm(a + GLU(W1 ELU(W2 a + W3 c)))."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, d_context: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.context = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.gate_norm = GateAddNorm(d_hidden, d_out, dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.skip(x)
        h = self.fc1(x)
        if context is not None and self.context is not None:
            c = self.context(context)
            while c.dim() < h.dim():
                c = c.unsqueeze(-2)
            h = h + c
        h = self.fc2(F.elu(h))
        return self.gate_norm(h, residual)


class RealEmbedding(nn.Module):
    """Independent linear embedding per scalar variable: (..., n) -> (..., n, d)."""

    def __init__(self, n_vars: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_vars, d_model) / math.sqrt(d_model))
        self.bias = nn.Parameter(torch.zeros(n_vars, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class CategoricalEmbedding(nn.Module):
    """One embedding table per categorical variable: (..., n) long -> (..., n, d)."""

    def __init__(self, cardinalities: Sequence[int], d_model: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(int(c), d_model) for c in cardinalities])

    def __len__(self) -> int:
        return len(self.embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([emb(x[..., i]) for i, emb in enumerate(self.embeddings)], dim=-2)


class GroupedLinear(nn.Module):
    """n independent linear maps applied in one einsum: (..., n, d_in) -> (..., n, d_out)."""

    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n, d_out))
        nn.init.uniform_(self.weight, -1 / math.sqrt(d_in), 1 / math.sqrt(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ni,nio->...no", x, self.weight) + self.bias


class GroupedGRN(nn.Module):
    """n per-variable GRNs (d -> d) evaluated jointly; mathematically identical to n separate GRNs."""

    def __init__(self, n: int, d: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = GroupedLinear(n, d, d)
        self.fc2 = GroupedLinear(n, d, d)
        self.glu = GroupedLinear(n, d, 2 * d)
        self.dropout = nn.Dropout(dropout)
        self.ln_weight = nn.Parameter(torch.ones(n, d))
        self.ln_bias = nn.Parameter(torch.zeros(n, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(F.elu(self.fc1(x)))
        h = F.glu(self.glu(self.dropout(h)), dim=-1)
        y = F.layer_norm(x + h, (x.shape[-1],))
        return y * self.ln_weight + self.ln_bias


class VariableSelectionNetwork(nn.Module):
    """Instance-wise soft variable selection. Input (..., n_vars, d) -> (..., d), weights (..., n_vars)."""

    def __init__(self, n_vars: int, d_model: int, d_context: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        if n_vars < 1:
            raise ValueError("VariableSelectionNetwork needs at least one variable")
        self.n_vars = n_vars
        self.flat_grn = GatedResidualNetwork(n_vars * d_model, d_model, n_vars, d_context, dropout) if n_vars > 1 else None
        self.var_grns = GroupedGRN(n_vars, d_model, dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        processed = self.var_grns(x)
        if self.flat_grn is None:
            weights = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
        else:
            weights = torch.softmax(self.flat_grn(x.flatten(-2), context), dim=-1)
        return (processed * weights.unsqueeze(-1)).sum(-2), weights


class InterpretableMultiHeadAttention(nn.Module):
    """TFT attention: per-head Q/K projections, a value projection shared across heads, heads averaged."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.d_k)
        self.out_proj = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, tq, _ = q.shape
        tk = k.shape[1]
        qh = self.q_proj(q).view(b, tq, self.n_heads, self.d_k).transpose(1, 2)
        kh = self.k_proj(k).view(b, tk, self.n_heads, self.d_k).transpose(1, 2)
        vh = self.v_proj(v).unsqueeze(1)  # shared across heads
        scores = qh @ kh.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ vh).mean(dim=1)
        return self.out_proj(out), attn


class MonotoneQuantileHead(nn.Module):
    """Non-crossing quantiles: median plus cumulative softplus offsets on either side."""

    def __init__(self, d_model: int, quantiles: Sequence[float]) -> None:
        super().__init__()
        qs = list(quantiles)
        if sorted(qs) != qs:
            raise ValueError("quantiles must be sorted")
        self.median_idx = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))
        self.proj = nn.Linear(d_model, len(qs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.proj(x)
        k = self.median_idx
        median = raw[..., k : k + 1]
        parts = []
        if k > 0:
            lower = F.softplus(raw[..., :k])
            parts.append(median - torch.flip(torch.cumsum(torch.flip(lower, [-1]), -1), [-1]))
        parts.append(median)
        if k < raw.shape[-1] - 1:
            parts.append(median + torch.cumsum(F.softplus(raw[..., k + 1 :]), -1))
        return torch.cat(parts, dim=-1)
