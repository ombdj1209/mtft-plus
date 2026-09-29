"""Lifecycle-aligned (LA) cross-series information for short-life-cycle retail (MTFT+ v4).

Everything is aligned by *product age* (steps since a series' own launch) instead of calendar time, and is strictly
causal: a value y(b, g, t) is usable at forecast origin o only if t <= o - 1 and the cell is live (launched and observed).

* LA siblings: for target series (a, f) and step t with target age k = t - launch(a, f), the same product's sales in
  another shop g at the same age, y(a, g, launch(a, g) + k). Up to M shops, the earliest-launched before the origin.
* LA analogues: the K most similar products b (cosine similarity of retrieval keys, b != a) that launched before the
  origin; each contributes its mean log1p per-shop sales at age k over the shops whose age-k week is <= o - 1.
* LA-pool: the four pooled per-step channels given to *every* model that uses LA information (the information control):
  mean available sibling value, sibling availability, similarity-softmax-weighted analogue value, analogue coverage.
* LA tokens: the individual sibling and analogue values, for the lifecycle attention of MTFT+ v4.

Precomputation: for every product b, age k and origin o, S[o, b, k] = sum over shops of log1p(y) of age-k weeks known
before o, and N[o, b, k] the number of such shops, built from event times (week + 1) and a cumulative sum over o.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

AGE_MIN, AGE_MAX = -16, 40  # clip range of the age index given to age embeddings
POOL_CHANNELS = ["la_sib_mean", "la_sib_avail", "la_ana_mean", "la_ana_cov"]


@dataclass
class LifecycleFeatures:
    launch_loc: np.ndarray  # (A, F) series launch step (T = never stocked)
    first_launch: np.ndarray  # (A,) first launch of each product in any shop
    shop_order: np.ndarray  # (A, F) shops sorted by launch step (stable, ties by shop index)
    W: int  # life-cycle window in steps (ages 0 .. W-1)
    S: np.ndarray  # (T + 1, A, W) float32, known-before-origin sum of log1p sales at age k
    N: np.ndarray  # (T + 1, A, W) float32, number of shops contributing to S
    M: int = 32
    K: int = 16
    temperature: float = 0.05
    neigh: Optional[np.ndarray] = None  # (A, R) retrieval neighbours, most similar first
    sim: Optional[np.ndarray] = None  # (A, R) cosine similarities
    emb: Optional[np.ndarray] = None  # (A, D) product embeddings given to the analogue tokens
    source: str = ""


def build_lifecycle(d, W: Optional[int] = None, M: int = 32, K: int = 16, temperature: float = 0.05) -> LifecycleFeatures:
    """Causal age-curve sums for panel ``d`` (a SyntheticRetailData, real or synthetic)."""
    A, F, T = d.y.shape
    L = np.asarray(d.series_launch(), dtype=np.int64)
    live = d.live_mask()
    if W is None:
        W = int(min(max(live.sum(2).max(), 1), 52)) if d.live is not None else int(min(T, 52))
    S = np.zeros((T + 1, A, W), dtype=np.float32)
    N = np.zeros((T + 1, A, W), dtype=np.float32)
    a_idx, g_idx = np.nonzero(L < T)
    ages = np.arange(W)
    for s0 in range(0, len(a_idx), 50_000):
        a, g = a_idx[s0 : s0 + 50_000], g_idx[s0 : s0 + 50_000]
        t = L[a, g][:, None] + ages[None]  # (P, W)
        ok = t < T
        tc = np.minimum(t, T - 1)
        ok &= live[a[:, None], g[:, None], tc]
        val = np.log1p(d.y[a[:, None], g[:, None], tc]).astype(np.float32)
        pa, pk = np.nonzero(ok)
        e = t[pa, pk] + 1  # first origin at which the value is known
        np.add.at(S, (e, a[pa], pk), val[pa, pk])
        np.add.at(N, (e, a[pa], pk), 1.0)
    np.cumsum(S, axis=0, out=S)
    np.cumsum(N, axis=0, out=N)
    order = np.argsort(L, axis=1, kind="stable")
    return LifecycleFeatures(L, L.min(1), order, W, S, N, M, K, temperature)


def with_retrieval(lf: LifecycleFeatures, keys: np.ndarray, emb: np.ndarray, R: int = 64, source: str = "") -> LifecycleFeatures:
    """Attach a retrieval index (top-R cosine neighbours on ``keys``, excluding the article itself) and token embeddings."""
    z = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
    A = len(z)
    R = min(R, A - 1)
    neigh = np.zeros((A, R), dtype=np.int64)
    sim = np.zeros((A, R), dtype=np.float32)
    for s0 in range(0, A, 2048):
        s = z[s0 : s0 + 2048] @ z.T
        s[np.arange(len(s)), np.arange(s0, s0 + len(s))] = -np.inf
        top = np.argpartition(-s, R, axis=1)[:, :R]
        st = np.take_along_axis(s, top, 1)
        o = np.argsort(-st, axis=1, kind="stable")
        neigh[s0 : s0 + len(s)] = np.take_along_axis(top, o, 1)
        sim[s0 : s0 + len(s)] = np.take_along_axis(st, o, 1)
    return replace(lf, neigh=neigh, sim=sim, emb=np.asarray(emb, dtype=np.float32), source=source)


def lifecycle_batch(lf: LifecycleFeatures, d, a: np.ndarray, f: np.ndarray, o: np.ndarray, L_enc: int, H: int, tokens: bool) -> Dict[str, np.ndarray]:
    """Per-window LA information for windows (a, f, o) over the L_enc + H steps t = o - L_enc .. o + H - 1."""
    A, F, T = d.y.shape
    B = len(a)
    W, M, K = lf.W, lf.M, lf.K
    tt = o[:, None] + np.arange(-L_enc, H)[None]  # (B, S)
    k = tt - lf.launch_loc[a, f][:, None]  # target age at each step
    age_ok = (k >= 0) & (k < W)
    out: Dict[str, np.ndarray] = {"age_idx": (np.clip(k, AGE_MIN, AGE_MAX) - AGE_MIN).astype(np.int64)}

    # ---------------------------------------------------------------- siblings: same product, other shops, same age
    order = lf.shop_order[a]  # (B, F)
    lg_all = lf.launch_loc[a[:, None], order]
    valid = (order != f[:, None]) & (lg_all <= (o - 1)[:, None])
    pos = np.cumsum(valid, 1) - 1
    take = valid & (pos < M)
    shop = np.full((B, M), F, dtype=np.int64)  # F = padding index
    bi, ci = np.nonzero(take)
    shop[bi, pos[bi, ci]] = order[bi, ci]
    slot = shop < F
    g = np.minimum(shop, F - 1)
    lg = np.where(slot, lf.launch_loc[a[:, None], g], T)  # (B, M)
    tg = lg[:, :, None] + k[:, None, :]  # (B, M, S) sibling calendar week at the target's age
    s_av = slot[:, :, None] & age_ok[:, None, :] & (tg <= (o - 1)[:, None, None]) & (tg >= 0) & (tg < T)
    tgc = np.clip(tg, 0, T - 1)
    s_av &= d.live_cells(a[:, None, None], g[:, :, None], tgc)
    s_val = np.where(s_av, np.log1p(d.y[a[:, None, None], g[:, :, None], tgc]), 0.0).astype(np.float32)
    n_s = s_av.sum(1)
    out["la_sib_mean"] = (s_val.sum(1) / np.maximum(n_s, 1)).astype(np.float32)
    out["la_sib_avail"] = (np.log1p(n_s) / np.log1p(M)).astype(np.float32)

    # ---------------------------------------------------------------- analogues: similar products launched earlier
    nb_all = lf.neigh[a]  # (B, R)
    ok = lf.first_launch[nb_all] <= (o - 1)[:, None]
    pos = np.cumsum(ok, 1) - 1
    take = ok & (pos < K)
    nb = np.full((B, K), A, dtype=np.int64)  # A = padding index
    sm = np.zeros((B, K), dtype=np.float32)
    bi, ci = np.nonzero(take)
    nb[bi, pos[bi, ci]] = nb_all[bi, ci]
    sm[bi, pos[bi, ci]] = lf.sim[a[bi], ci]
    aslot = nb < A
    bc = np.minimum(nb, A - 1)
    kc = np.clip(k, 0, W - 1)
    oc = np.clip(o, 0, T)
    Ssum = lf.S[oc[:, None, None], bc[:, :, None], kc[:, None, :]]  # (B, K, S)
    Ncnt = lf.N[oc[:, None, None], bc[:, :, None], kc[:, None, :]]
    a_av = aslot[:, :, None] & age_ok[:, None, :] & (Ncnt > 0)
    a_val = np.where(a_av, Ssum / np.maximum(Ncnt, 1), 0.0).astype(np.float32)
    logits = np.where(aslot, sm / lf.temperature, -np.inf)
    logits = logits - np.where(aslot.any(1, keepdims=True), logits.max(1, keepdims=True), 0.0)
    w = np.where(aslot, np.exp(logits), 0.0)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    wa = w[:, :, None] * a_av
    cov = wa.sum(1)
    out["la_ana_mean"] = ((wa * a_val).sum(1) / np.maximum(cov, 1e-6)).astype(np.float32)
    out["la_ana_cov"] = cov.astype(np.float32)

    if tokens:
        lag = np.where(slot, (lf.launch_loc[a, f][:, None] - lg) / float(W), 0.0)
        out.update({
            "la_sib_val": s_val, "la_sib_mask": s_av, "la_sib_shop": shop, "la_sib_lag": lag.astype(np.float32),
            "la_ana_val": a_val, "la_ana_mask": a_av, "la_ana_support": np.log1p(Ncnt * a_av).astype(np.float32),
            "la_ana_sim": sm, "la_ana_emb": np.where(aslot[:, :, None], lf.emb[bc], 0.0).astype(np.float32),
            "la_self_emb": lf.emb[a].astype(np.float32), "la_self_shop": f.astype(np.int64),
        })
    return out
