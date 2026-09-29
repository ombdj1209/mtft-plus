"""Diagnose Chronos-2 zero-shot on the calendar benchmark (pre-registration amendment (d) 1a).

    python examples/chronos_diagnostics.py --panel panel_v2.pkl

Scores every live test cell of the calendar benchmark (same windows as all models) under:
  * context handling: NaN-padded before launch (as used), trimmed to the live history, zero-filled;
  * point forecast: median (q50, as used) vs Chronos-2's mean;
  * time alignment: q50 scored against the actuals of steps h-1 and h+1 instead of h;
  * reference forecasts: all-zeros and the last observed value.
Also reports the ratio of the summed forecast to the summed actuals (output-scale check).
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from benchmark import QUANTILES, SeedContext


def wape(y, p, m):
    return float(np.abs(y - p)[m].sum() / y[m].sum())


def main() -> None:
    import chronos2_baseline as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel_v2.pkl")
    ap.add_argument("--model", default=".hf_models/chronos-2")
    args = ap.parse_args()
    ctx = SeedContext(0, types.SimpleNamespace(panel=args.panel, k=10, fcs=0, articles=0, days=0))
    d, H = ctx.data, ctx.H
    idx = ctx.test_all_idx
    a, f, o = idx[:, 0], idx[:, 1], idx[:, 2]
    tt = o[:, None] + np.arange(H)[None]
    y = d.y[a[:, None], f[:, None], tt].astype(np.float64)
    m = d.live[a[:, None], f[:, None], tt]
    pipe = C.load_pipeline(args.model)
    base = C.chronos_inputs(d, idx, cov=False)

    def run(inputs):
        qs, means = pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=list(QUANTILES), batch_size=256)
        return np.clip(torch.stack([x[0] for x in qs]).numpy(), 0, None), np.clip(torch.stack([x[0] for x in means]).numpy(), 0, None)

    variants = {"NaN-padded (as used)": base}
    trimmed, zero = [], []
    for x in base:
        t = x["target"]
        live = t[np.isfinite(t)]
        trimmed.append({"target": live if len(live) else np.zeros(1, np.float32)})
        zero.append({"target": np.nan_to_num(t, nan=0.0)})
    variants["trimmed to live history"] = trimmed
    variants["zero-filled"] = zero
    print(f"live test cells: {int(m.sum()):,} | windows: {len(idx):,} | mean actual per live cell {y[m].mean():.3f}")
    print(f"all-zeros forecast: WAPE {wape(y, np.zeros_like(y), m):.4f}")
    last = np.nan_to_num(np.array([x['target'][np.isfinite(x['target'])][-1] if np.isfinite(x['target']).any() else 0.0 for x in base]))
    print(f"last observed value: WAPE {wape(y, np.repeat(last[:, None], H, 1), m):.4f}")
    for name, inp in variants.items():
        q, mean = run(inp)
        med = q[..., 1]
        print(f"[{name}] q50 WAPE {wape(y, med, m):.4f} | mean-forecast WAPE {wape(y, mean, m):.4f} | "
              f"sum(q50)/sum(y) {med[m].sum() / y[m].sum():.3f} | sum(mean)/sum(y) {mean[m].sum() / y[m].sum():.3f}")
        if name.startswith("NaN"):
            # alignment: forecast for step h scored against actual h-1 / h+1 (cells where both are live)
            for shift in (-1, 1):
                if shift < 0:
                    yy, pp, mm = y[:, :-1], med[:, 1:], m[:, :-1] & m[:, 1:]
                else:
                    yy, pp, mm = y[:, 1:], med[:, :-1], m[:, 1:] & m[:, :-1]
                same = wape(y[:, 1:] if shift < 0 else y[:, :-1], med[:, 1:] if shift < 0 else med[:, :-1], mm)
                print(f"    alignment check, shift {shift:+d}: WAPE {wape(yy, pp, mm):.4f} vs unshifted on the same cells {same:.4f}")
            # by own-history length
            n_obs = np.array([np.isfinite(x["target"]).sum() for x in base])
            no_hist = np.array([not np.isfinite(x["target"][:-1]).any() and x["target"][-1] == 0 for x in base])
            for lo, hi, lab in ((0, 0, "no own history (launch in horizon)"), (1, 3, "1-3 weeks"), (4, 7, "4-7 weeks"), (8, 12, "8-12 weeks")):
                s = no_hist if lo == 0 else (~no_hist & (n_obs >= lo) & (n_obs <= hi))
                if s.any():
                    print(f"    {lab:36s} windows {s.sum():6d} | share of live units {y[s][m[s]].sum() / y[m].sum():.3f} | "
                          f"q50 WAPE {wape(y[s], med[s], m[s]):.4f} | zeros WAPE 1.0000 | q50/actual {med[s][m[s]].sum() / y[s][m[s]].sum():.3f}")


if __name__ == "__main__":
    main()
