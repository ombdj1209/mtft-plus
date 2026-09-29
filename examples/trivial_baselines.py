"""Trivial reference forecasts on the calendar benchmark (pre-registration amendment (d) 1c).

    python examples/trivial_baselines.py --panel panel_v2.pkl --out results_visuelle2

  * all-zeros;
  * last value: the last observed own week before the origin (0 if the series has no own history);
  * LA-sibling mean: expm1 of the pooled lifecycle-aligned sibling value at each horizon step (the same product's
    sales in earlier-launched shops at the same age), falling back to the last value where no sibling is available.
Point forecasts are used for all three quantiles. Same live test windows, metrics and reconciliation as every model.
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from benchmark import QUANTILES, SeedContext, slug
from mtft_plus.lifecycle import lifecycle_batch


def point_forecasts(ctx, lf, idx: np.ndarray, kind: str) -> np.ndarray:
    d, L, H = ctx.data, ctx.L, ctx.H
    a, f, o = idx[:, 0], idx[:, 1], idx[:, 2]
    if kind == "zeros":
        return np.zeros((len(idx), H), np.float32)
    enc_t = o[:, None] + np.arange(-L, 0)[None]
    own = np.where(d.live_cells(a[:, None], f[:, None], np.clip(enc_t, 0, None)) & (enc_t >= 0), d.y[a[:, None], f[:, None], np.clip(enc_t, 0, None)], np.nan)
    has = np.isfinite(own).any(1)
    last_pos = L - 1 - np.argmax(np.isfinite(own[:, ::-1]), 1)
    last = np.where(has, own[np.arange(len(idx)), last_pos], 0.0)
    last_f = np.repeat(last[:, None], H, 1).astype(np.float32)
    if kind == "last value":
        return last_f
    if kind == "seasonal":
        m = ctx.season
        lag = L - m + (np.arange(H) % m)  # position in the encoder of the same season step in the last observed season
        return np.nan_to_num(own[:, lag], nan=0.0).astype(np.float32)
    out = np.empty((len(idx), H), np.float32)
    for s in range(0, len(idx), 8192):
        sl = slice(s, s + 8192)
        la = lifecycle_batch(lf, d, a[sl], f[sl], o[sl], L, H, tokens=False)
        sib = np.expm1(la["la_sib_mean"][:, L:])
        avail = la["la_sib_avail"][:, L:] > 0
        out[sl] = np.where(avail, sib, last_f[sl])
    return out


def main() -> None:
    import chronos2_baseline as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel_v2.pkl")
    ap.add_argument("--out", default="results_visuelle2")
    ap.add_argument("--seasonal", action="store_true", help="also the seasonal naive forecast (M5 pre-registration)")
    args = ap.parse_args()
    ctx = SeedContext(0, types.SimpleNamespace(panel=args.panel, k=10, fcs=0, articles=0, days=0))
    lf = ctx.lifecycle()["aligned"]
    part = Path(args.out) / "partials"
    part.mkdir(parents=True, exist_ok=True)
    kinds = [("zeros", "All-zeros"), ("last value", "Last value"), ("la sibling", "LA-sibling mean")] + ([("seasonal", "Seasonal naive")] if args.seasonal else [])
    for kind, name in kinds:
        path = part / f"0__{slug(name)}.json"
        if path.exists():
            continue
        t0 = time.time()
        qs = {}
        for split, idx, origins, mask in (("val", ctx.val_all_idx, ctx.val_o, ctx.mask_val_b), ("test", ctx.test_all_idx, ctx.test_o, ctx.mask_test_b)):
            p = point_forecasts(ctx, lf, idx, kind)
            qs[split] = C.scatter(ctx, np.repeat(p[..., None], len(QUANTILES), -1), idx, origins, mask)
        C.calendar_result(ctx, name, {"zero_shot": True, "trivial": kind}, 0, qs, t0, args, path)


if __name__ == "__main__":
    main()
