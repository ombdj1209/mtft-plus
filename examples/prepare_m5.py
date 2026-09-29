"""Build the M5 (Walmart) panel for benchmark.py (pre-registration: results/m5_preregistration.md).

    python examples/prepare_m5.py --src data/m5_raw --out panel_m5.pkl

Daily grid, days 1-1969 (training file days 1-1941 + the official evaluation horizon 1942-1969). A series is launched
at the first day of its first price week (availability, known to the retailer; the first *sale* would leak launch-day
demand) and is live from then on. Validation = days 1914-1941, test = days 1942-1969 (one origin each, H = 28, L = 56).
Covariates: discount = 1 - price / running max price of the item x store (price plan, known ahead), promo = discount
> 5 %, holiday = any calendar event. No images / text: embeddings are zero with modality masks off. LA analogues are
retrieved on static keys [department one-hot, category one-hot, standardised log mean price].
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from mtft_plus.data import SyntheticConfig, SyntheticRetailData
from mtft_plus.panel import _z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/m5_raw")
    ap.add_argument("--out", default="panel_m5.pkl")
    ap.add_argument("--encoder-length", type=int, default=56)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--train-stride", type=int, default=7)
    ap.add_argument("--cold-start-steps", type=int, default=28)
    ap.add_argument("--end-day", type=int, default=0, help="truncate the grid after this many days (launch-rich split: 1176)")
    args = ap.parse_args()
    src = Path(args.src)
    tr = pd.read_csv(src / "sales_train_evaluation.csv")
    te = pd.read_csv(src / "sales_test_evaluation.csv")
    assert (tr.item_id.values == te.item_id.values).all() and (tr.store_id.values == te.store_id.values).all()
    cal = pd.read_csv(src / "calendar.csv", parse_dates=["date"])
    days = [c for c in tr.columns if c.startswith("d_")] + [c for c in te.columns if c.startswith("d_")]
    Yflat = np.concatenate([tr[[c for c in tr.columns if c.startswith("d_")]].to_numpy(np.float32),
                            te[[c for c in te.columns if c.startswith("d_")]].to_numpy(np.float32)], 1)
    if args.end_day:
        Yflat = Yflat[:, : args.end_day]
    T = Yflat.shape[1]
    cal = cal.iloc[:T].reset_index(drop=True)
    items = sorted(tr.item_id.unique())
    stores = sorted(tr.store_id.unique())
    a_pos, f_pos = {a: i for i, a in enumerate(items)}, {s: i for i, s in enumerate(stores)}
    A, F = len(items), len(stores)
    ai, fi = tr.item_id.map(a_pos).to_numpy(), tr.store_id.map(f_pos).to_numpy()
    y = np.zeros((A, F, T), np.float32)
    y[ai, fi] = np.clip(Yflat, 0, None)

    # prices -> launch (first price week) and discount (known price plan)
    p = pd.read_csv(src / "sell_prices.csv")
    cal["di"] = np.arange(T)
    wk_first = cal.groupby("wm_yr_wk").di.min()
    week_of_day = cal.wm_yr_wk.to_numpy()
    p["a"], p["f"] = p.item_id.map(a_pos), p.store_id.map(f_pos)
    p = p.dropna(subset=["a", "f"]).astype({"a": int, "f": int})
    p["d0"] = p.wm_yr_wk.map(wk_first)
    p = p.dropna(subset=["d0"])
    launch_loc = np.full((A, F), T, np.int64)
    first = p.groupby(["a", "f"]).d0.min()
    launch_loc[first.index.get_level_values(0), first.index.get_level_values(1)] = first.to_numpy().astype(np.int64)
    p = p.sort_values(["a", "f", "wm_yr_wk"])
    p["runmax"] = p.groupby(["a", "f"]).sell_price.cummax()
    p["disc"] = 1.0 - p.sell_price / p.runmax
    weeks = np.unique(week_of_day)
    w_idx = {w: i for i, w in enumerate(weeks)}
    D = np.zeros((A, F, len(weeks)), np.float32)
    D[p.a.to_numpy(), p.f.to_numpy(), p.wm_yr_wk.map(w_idx).to_numpy()] = p.disc.to_numpy(np.float32)
    discount = D[:, :, np.array([w_idx[w] for w in week_of_day])]
    promo = (discount > 0.05).astype(np.float32)
    live = np.arange(T)[None, None, :] >= launch_loc[:, :, None]
    y *= live

    # calendar: events -> holiday (all stores)
    ev = (cal.event_name_1.notna() | cal.event_name_2.notna()).to_numpy(np.float32)
    holiday = np.repeat(ev[None], F, 0)
    pre_holiday = np.concatenate([holiday[:, 1:], np.zeros((F, 1), np.float32)], 1) * (1 - holiday)

    meta = tr.drop_duplicates("item_id").set_index("item_id").loc[items]
    dept_codes, dept_levels = pd.factorize(meta.dept_id)
    cat_codes, _ = pd.factorize(meta.cat_id)
    mean_price = p.groupby("a").sell_price.mean().reindex(range(A)).fillna(p.sell_price.mean()).to_numpy()
    keys = np.concatenate([np.eye(dept_codes.max() + 1)[dept_codes], np.eye(cat_codes.max() + 1)[cat_codes], _z(np.log(mean_price))[:, None]], 1).astype(np.float32)
    state = np.array([s.split("_")[0] for s in stores])
    fc_country = np.array([["CA", "TX", "WI"].index(s) for s in state], np.int64)
    static_real = np.stack([_z(np.log(mean_price)), np.zeros(A, np.float32)], 1).astype(np.float32)

    L, H = args.encoder_length, args.horizon
    cfg = SyntheticConfig(n_fc=F, n_articles=A, n_days=T, embed_dim=8, encoder_length=L, horizon=H, n_test_origins=1, n_val_origins=1,
                          freq="D", season=7, level_window=28, min_history=35, steps_per_year=365.0)
    cfg.train_origin_stride = args.train_stride
    cfg.lifecycle_window = L
    test_start, val_start = T - H, T - 2 * H
    launch = launch_loc.min(1)
    cold = (launch >= test_start - args.cold_start_steps) & (launch < T)
    new_in_shop = (launch_loc >= test_start - args.cold_start_steps) & (launch_loc < T) & ~cold[:, None]
    young = (test_start - launch_loc < L) & (launch_loc < test_start)
    zeros = np.zeros((A, 8), np.float32)
    dates = pd.DatetimeIndex(cal.date)
    d = SyntheticRetailData(
        cfg=cfg, dates=dates, y=y, promo=promo, discount=discount.astype(np.float32), temp_z=np.zeros((F, T), np.float32),
        precip_z=np.zeros((F, T), np.float32), holiday=holiday.astype(np.float32), pre_holiday=pre_holiday.astype(np.float32),
        dow=dates.dayofweek.values.astype(np.int64), doy=dates.dayofyear.values.astype(np.int64), launch=launch.astype(np.int64),
        category=dept_codes.astype(np.int64), fc_country=fc_country, static_real=static_real, shelf_life=np.full(A, 365.0),
        attributes={"category_levels": np.asarray(dept_levels), "article_ids": np.asarray(items), "mm_mask": np.zeros((A, 2), bool),
                    "new_in_shop": new_in_shop, "young": young, "retrieval_keys": keys},
        emb_aligned_image=zeros, emb_aligned_text=zeros, emb_uni_image=zeros, emb_uni_text=zeros, cold_start=cold,
        split={"train_end": val_start, "val_start": val_start, "test_start": test_start, "cold_start_steps": args.cold_start_steps},
        fc_names=stores, launch_loc=launch_loc, live=live,
    )
    with open(args.out, "wb") as fh:
        pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    live_test = live[:, :, test_start:].any(2)
    print(f"M5 panel: {A} items x {F} stores x {T} days | zero share {np.mean(y[live] == 0):.3f} | series live in test {int(live_test.sum()):,}")
    print(f"launch (first price week): after day 0 {np.mean(launch_loc > 0):.3f}; cold-start items {int(cold.sum())}; new-in-shop series {int(new_in_shop.sum())}; young series (<{L} d at test) {int(young.sum())}")
    print(f"promo share {promo[live].mean():.3f} | event days {int(ev.sum())} | val {dates[val_start].date()}..{dates[test_start - 1].date()} | test {dates[test_start].date()}..{dates[-1].date()}")


if __name__ == "__main__":
    main()
