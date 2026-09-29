"""Why does MinT-shrink fail on VISUELLE 2.0? (pre-registration amendment (d) 3; analysis only, nothing is tuned)

    python benchmark.py --panel panel_v2.pkl --seeds 1 --models "MTFT-Picnic + siblings + LA-pool" --save-val-forecasts --out <dir>
    python examples/mint_diagnostics.py --panel panel_v2.pkl --run <dir> --name mtft-picnic-siblings-la-pool --seed 1

Rebuilds exactly the reconciliation of benchmark.reconcile from the stored base forecasts and reports:
the residual sample size, the shrinkage intensity, the condition number of C W C^T, the prior variances of
no-history nodes, the size and sign of the MinT adjustments per level, and the share of bottom forecasts clamped at 0.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from benchmark import QUANTILES, SeedContext
from mtft_plus.reconciliation import MinTReconciler, swanson_mean


def load(path: Path, shape) -> np.ndarray:
    z = np.load(path)
    q = np.zeros(shape, np.float32)
    q[z["series"], z["origin"], z["step"]] = z["q"]
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel_v2.pkl")
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", default="mtft-picnic-siblings-la-pool")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    ctx = SeedContext(args.seed, types.SimpleNamespace(panel=args.panel, k=10, fcs=0, articles=0, days=0))
    h, A, F, H = ctx.hier, ctx.A, ctx.F, ctx.H
    nv, nt, Q = len(ctx.val_o), len(ctx.test_o), len(QUANTILES)
    fdir = Path(args.run) / "forecasts"
    q_test = load(fdir / f"{args.seed}__{args.name}.npz", (A * F, nt, H, Q))
    q_val = load(fdir / f"{args.seed}__{args.name}__val.npz", (A * F, nv, H, Q))
    mean = lambda q: swanson_mean(q, QUANTILES)  # noqa: E731
    base_val = np.concatenate([np.where(ctx.agg_val_ok[:, :, None, None], ctx.agg_val_q, h.aggregate(q_val)[: h.n_agg]), q_val], 0)
    base_test = np.concatenate([np.where(ctx.agg_test_ok[:, :, None, None], ctx.agg_test_q, h.aggregate(q_test)[: h.n_agg]), q_test], 0)
    pv, pt = mean(base_val), mean(base_test)
    R = torch.as_tensor((ctx.y_val_all - pv).reshape(h.n_total, -1).T)
    level = ctx.y_val_all.reshape(h.n_total, -1).mean(1)
    var = (R.numpy() ** 2).mean(0)
    okv = (level > 0.5) & (var > 0)
    kappa = float(np.median(var[okv] / level[okv]))
    live_val_n, live_test_n = ctx.mask_val_all.any((1, 2)), ctx.mask_test_all.any((1, 2))
    no_hist = np.concatenate([~ctx.agg_test_ok.all(1), ctx.cold_b]) | (live_test_n & ~live_val_n)
    prior = np.full(h.n_total, np.nan)
    prior[no_hist] = kappa * np.maximum(pt[no_hist].mean((1, 2)), 1.0)
    K = nt * H
    levels = h.level_slices()
    print(f"nodes {h.n_total:,} (aggregates {h.n_agg:,}) | residual samples T = {R.shape[0]} (4 validation origins x 4 steps)")
    live_both = live_val_n & live_test_n
    for lvl, sl in levels:
        print(f"  {lvl:18s} live in test {int(live_test_n[sl].sum()):7,d} | of these live in validation {int(live_both[sl].sum()):7,d} | no-history prior {int(no_hist[sl].sum()):7,d}")
    print(f"kappa (variance / level of well-observed nodes) {kappa:.3f}")

    y_true = ctx.y_test_all.reshape(h.n_total, K)
    m = ctx.mask_test_all.reshape(h.n_total, K)
    base_pt = pt.reshape(h.n_total, K)
    bu = h.aggregate(base_pt[h.n_agg :])
    for tag, lam in (("MinT-shrink (pre-specified)", None), ("MinT-WLS (diagonal)", 1.0)):
        rec = MinTReconciler(ctx.Afull).fit(R, prior_var=torch.as_tensor(prior), lam=lam)
        L = rec._chol
        ev = torch.linalg.eigvalsh(L @ L.T)
        print(f"\n=== {tag}: lambda {rec.lam:.3f} | cond(C W C^T) = {float(ev[-1] / ev[0]):.3e} (eig min {float(ev[0]):.3e}, max {float(ev[-1]):.3e})")
        d_raw = rec._delta.numpy()
        print(f"    diagonal of W: median over nodes live in validation {np.median(d_raw[live_both]):.3f}; median prior (no-history) {np.median(prior[no_hist]):.3f}")
        raw = rec.reconcile(torch.as_tensor(base_pt), nonnegative=False).numpy()
        clamped = rec.reconcile(torch.as_tensor(base_pt), nonnegative=True).numpy()
        b = raw[h.n_agg :]
        live_b = m[h.n_agg :]
        neg = (b < 0) & live_b
        print(f"    bottom forecasts < 0 before clamping (live cells): {neg.mean() / max(live_b.mean(), 1e-9):.1%} | "
              f"negative mass / positive mass {(-b[b < 0].sum()) / max(b[b > 0].sum(), 1e-9):.3f}")
        print(f"    {'level':18s} {'actual':>10s} {'base':>10s} {'bottom-up':>10s} {'MinT raw':>10s} {'MinT clamped':>12s} | WAPE base / BU / MinT")
        for lvl, sl in levels:
            mm = m[sl]
            s = lambda x: float((x[sl] * mm).sum() / max(mm.sum(), 1))  # noqa: E731
            w = lambda x: float(np.abs(y_true[sl] - x[sl])[mm].sum() / max(y_true[sl][mm].sum(), 1e-9))  # noqa: E731
            print(f"    {lvl:18s} {s(y_true):10.2f} {s(base_pt):10.2f} {s(bu):10.2f} {s(raw):10.2f} {s(clamped):12.2f} | {w(base_pt):.3f} / {w(bu):.3f} / {w(clamped):.3f}")
        adj = raw - base_pt
        for lvl, sl in levels:
            mm = m[sl]
            print(f"    adjustment {lvl:18s} mean {float(adj[sl][mm].mean()):+9.3f} | mean |adj| / mean base {float(np.abs(adj[sl][mm]).mean() / max(np.abs(base_pt[sl][mm]).mean(), 1e-9)):.3f}")


if __name__ == "__main__":
    main()
