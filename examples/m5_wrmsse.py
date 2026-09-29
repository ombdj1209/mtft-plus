"""RMSSE / WRMSSE for the M5 runs (secondary metric, M5 pre-registration amendment (b)).

    python examples/m5_wrmsse.py --panel panel_m5.pkl --run results_m5 [--check]

WRMSSE follows the M5 accuracy competition (Makridakis et al., 2022):
  * 12 aggregation levels (total, state, store, category, department, state x category, state x department,
    store x category, store x department, item, item x state, item x store), each weighted 1/12;
  * within a level, series weight = its share of dollar sales (units x sell price) over the 28 days before the test
    origin;
  * RMSSE = sqrt(mean squared forecast error / mean squared one-step difference of the series' own history), with
    the history running from its first non-zero sale up to the test origin.
Forecasts are aggregated bottom-up. Two point forecasts are scored: the Swanson mean of the 0.1/0.5/0.9 quantiles
(primary for this squared-error metric) and the median (the point forecast our WAPE uses). Cells not predicted (not
live) are 0. A series whose scale is 0 or undefined (fewer than two history points) is excluded from its level and
counted. By construction WRMSSE gives zero weight to series with no sales in the last 28 days, so series launching
inside the test horizon (cold starts) do not enter it; the bottom-level RMSSE of young series is reported separately.

``--check`` reproduces the M5 benchmark forecasts Naive (last value) and seasonal Naive (same weekday last week) from
the panel, to compare with the published evaluation-set scores (Naive 1.752, sNaive 0.847).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mtft_plus.reconciliation import swanson_mean

QUANTILES = (0.1, 0.5, 0.9)


def hierarchy(src: Path, items: np.ndarray, stores: list):
    """(name, sparse (n_level, A*F) aggregation matrix) for the 12 M5 levels, rows a-major (a * F + f)."""
    meta = pd.read_csv(src / "sales_train_evaluation.csv", usecols=["item_id", "dept_id", "cat_id"]).drop_duplicates("item_id").set_index("item_id").loc[items]
    A, F = len(items), len(stores)
    a = np.repeat(np.arange(A), F)
    f = np.tile(np.arange(F), A)
    store = np.asarray(stores)[f]
    state = np.array([s.split("_")[0] for s in stores])[f]
    cat, dept = meta.cat_id.to_numpy()[a], meta.dept_id.to_numpy()[a]
    item = np.asarray(items)[a]
    keys = {
        "total": np.zeros(A * F, dtype=object), "state": state, "store": store, "category": cat, "department": dept,
        "state x category": state + "|" + cat, "state x department": state + "|" + dept, "store x category": store + "|" + cat,
        "store x department": store + "|" + dept, "item": item, "item x state": item + "|" + state, "item x store": item + "|" + store,
    }
    out = []
    for name, k in keys.items():
        codes, _ = pd.factorize(k)
        out.append((name, sp.csr_matrix((np.ones(A * F), (codes, np.arange(A * F))), shape=(codes.max() + 1, A * F))))
    return out


def dollar_weights(src: Path, d, items: np.ndarray, stores: list, origin: int) -> np.ndarray:
    """(A*F,) dollar sales over days origin-28 .. origin-1."""
    cal = pd.read_csv(src / "calendar.csv").iloc[: d.y.shape[2]]
    p = pd.read_csv(src / "sell_prices.csv")
    wk = cal.wm_yr_wk.to_numpy()[origin - 28 : origin]
    a_pos, f_pos = {x: i for i, x in enumerate(items)}, {x: i for i, x in enumerate(stores)}
    p = p[p.wm_yr_wk.isin(set(wk))]
    A, F = len(items), len(stores)
    price = np.zeros((A, F, 28))
    for j, w in enumerate(wk):
        pw = p[p.wm_yr_wk == w]
        price[pw.item_id.map(a_pos).to_numpy(), pw.store_id.map(f_pos).to_numpy(), j] = pw.sell_price.to_numpy()
    return (d.y[:, :, origin - 28 : origin] * price).sum(2).reshape(A * F)


def scales(Y: np.ndarray, origin: int) -> np.ndarray:
    """Mean squared one-step difference of each row's history from its first non-zero value to origin - 1."""
    h = Y[:, :origin]
    nz = h > 0
    first = np.where(nz.any(1), nz.argmax(1), origin)
    dif = np.diff(h, axis=1) ** 2
    valid = np.arange(1, origin)[None] > first[:, None]  # difference t - (t-1) with t-1 >= first
    n = valid.sum(1)
    s = np.where(valid, dif, 0.0).sum(1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def wrmsse(levels, Yb: np.ndarray, Pb: np.ndarray, w_b: np.ndarray, origin: int, H: int) -> dict:
    out, total = {}, 0.0
    for name, M in levels:
        Y = np.asarray(M @ Yb)
        P = np.asarray(M @ Pb)
        w = np.asarray(M @ w_b).ravel()
        s = scales(Y, origin)
        err = ((Y[:, origin : origin + H] - P) ** 2).mean(1)
        ok = np.isfinite(s) & (s > 0)
        r = np.sqrt(err[ok] / s[ok])
        ww = w[ok] / max(w[ok].sum(), 1e-12)
        out[name] = {"WRMSSE": float((ww * r).sum()), "excluded": int((~ok).sum()), "excluded_weight_share": float(w[~ok].sum() / max(w.sum(), 1e-12))}
        total += out[name]["WRMSSE"] / len(levels)
    out["WRMSSE"] = total
    return out


def load_forecast(path: Path, n: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    assert (z["origin"] == 0).all(), "M5 splits have a single test origin"
    mean, med = np.zeros((n, H)), np.zeros((n, H))
    mean[z["series"], z["step"]] = swanson_mean(z["q"], QUANTILES)
    med[z["series"], z["step"]] = z["q"][:, QUANTILES.index(0.5)]
    return mean, med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--src", default="data/m5_raw")
    ap.add_argument("--run", default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)
    with open(args.panel, "rb") as fh:
        d = pickle.load(fh)
    items, stores = np.asarray(d.attributes["article_ids"]), list(d.fc_names)
    A, F, T = d.y.shape
    H, origin = d.cfg.horizon, d.split["test_start"]
    Yb = d.y.reshape(A * F, T).astype(np.float64)
    levels = hierarchy(src, items, stores)
    w_b = dollar_weights(src, d, items, stores, origin)
    young = np.asarray(d.attributes["young"]).reshape(A * F)
    s_b = scales(Yb, origin)
    if args.check:
        naive = np.repeat(Yb[:, origin - 1 : origin], H, 1)
        snaive = Yb[:, origin - 7 + (np.arange(H) % 7)]
        for name, P in (("Naive", naive), ("sNaive", snaive)):
            r = wrmsse(levels, Yb, P, w_b, origin, H)
            print(f"{name:7s} WRMSSE {r['WRMSSE']:.4f} | " + " ".join(f"{k}:{v['WRMSSE']:.3f}" for k, v in r.items() if k != "WRMSSE"))
    if not args.run:
        return
    rows = []
    for path in sorted((Path(args.run) / "forecasts").glob("*.npz")):
        if path.name.endswith("__val.npz"):
            continue
        seed, name = path.stem.split("__", 1)
        mean, med = load_forecast(path, A * F, H)
        row = {"seed": int(seed), "model": name}
        for tag, P in (("mean", mean), ("median", med)):
            r = wrmsse(levels, Yb, P, w_b, origin, H)
            row[f"WRMSSE_{tag}"] = r["WRMSSE"]
            row[f"levels_{tag}"] = r
            ok = np.isfinite(s_b) & (s_b > 0)
            rm = np.sqrt(((Yb[:, origin : origin + H] - P) ** 2).mean(1) / np.where(ok, s_b, 1.0))
            row[f"RMSSE_bottom_{tag}"] = float(rm[ok].mean())
            row[f"RMSSE_young_{tag}"] = float(rm[ok & young].mean()) if (ok & young).any() else float("nan")
        row["n_young_scored"] = int((np.isfinite(s_b) & (s_b > 0) & young).sum())
        rows.append(row)
        print(f"seed {seed} {name:45s} WRMSSE mean {row['WRMSSE_mean']:.4f} median {row['WRMSSE_median']:.4f} | bottom RMSSE {row['RMSSE_bottom_mean']:.4f} | young RMSSE {row['RMSSE_young_mean']:.4f}", flush=True)
    (Path(args.run) / "wrmsse.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
