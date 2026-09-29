"""Convert VISUELLE 2.0 (Skenderi et al., CVPRW 2022) into the long-format schema of examples/prepare_panel.py.

    python examples/convert_visuelle2.py --src visuelle2 --out data/visuelle2_converted

Source files used (inspected, not assumed):
    sales.csv                  one row per (external_code, retail): release_date (always a Monday), restock and
                               units sold in the 12 weeks after release (columns "0".."11"); raw units, some
                               negative (returns)
    price_discount_series.csv  same keys: discount depth per life-cycle week "0".."11" and a max-normalised price
    shop_weather_pairs.pt      {retail: weather locality}
    vis2_weather_data.csv      daily weather per locality (dates d/m/Y)
    images/<season>/<code>.png
Not used here: customer_data.csv (loyalty-card transactions only, ~85 % of units, 65 % of cells match
sales.csv), restocks.csv, vis2_gtrends_data.csv, stfore_*.csv (official life-cycle split; see
examples/visuelle2_official.py).

Output (in --out):
    sales.csv      article_id, location_id, date (Monday of the week), units, discount, promo
    series.csv     article_id, location_id, launch_date, end_date   (observation window: 12 weeks from release)
    catalog.csv    article_id, category, color, fabric, season, text, image_path (absolute), price, launch_date
    locations.csv  location_id, country (IT), weather_locality
    weather.csv    location_id, date, temperature, precipitation    (daily; prepare_panel averages per week)
    holidays.csv   country, date                                    (Italian national holidays)

Weeks after ``--end`` are dropped. The default is the last release week (2019-12-30): later weeks would
contain no new launches, so aggregate totals there would be an artefact of the dataset end, not of demand.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WEEKS = [str(i) for i in range(12)]
EASTER_MONDAY = ["2016-03-28", "2017-04-17", "2018-04-02", "2019-04-22", "2020-04-13"]
IT_FIXED = ["01-01", "01-06", "04-25", "05-01", "06-02", "08-15", "11-01", "12-08", "12-25", "12-26"]


def italian_holidays(years) -> pd.DataFrame:
    days = [f"{y}-{md}" for y in years for md in IT_FIXED] + EASTER_MONDAY
    return pd.DataFrame({"country": "IT", "date": sorted(days)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="visuelle2")
    ap.add_argument("--out", default="data/visuelle2_converted")
    ap.add_argument("--end", default="2019-12-30", help="last grid week kept (Monday)")
    args = ap.parse_args()
    src, out = Path(args.src).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(src / "sales.csv", index_col=0, parse_dates=["release_date"])
    assert (s.release_date.dt.dayofweek == 0).all(), "release dates are expected to be Mondays"
    assert not s.duplicated(["external_code", "retail"]).any()
    p = pd.read_csv(src / "price_discount_series.csv")
    s = s.merge(p.rename(columns={w: f"d{w}" for w in WEEKS}), on=["external_code", "retail"], how="left", validate="1:1")
    end = pd.Timestamp(args.end)

    # ---------------------------------------------------------------- long sales (one row per series-week)
    units = s[WEEKS].to_numpy(float)
    disc = s[[f"d{w}" for w in WEEKS]].to_numpy(float)
    n_nan_disc = int(np.isnan(disc).sum())
    disc = np.nan_to_num(disc, nan=0.0)
    dates = s.release_date.to_numpy()[:, None] + (np.arange(12) * np.timedelta64(7, "D"))[None]
    sales = pd.DataFrame({
        "article_id": np.repeat(s.external_code.to_numpy(), 12),
        "location_id": np.repeat(s.retail.to_numpy(), 12),
        "date": dates.ravel(),
        "units": units.ravel(),
        "discount": disc.ravel(),
    })
    sales["promo"] = (sales.discount > 0).astype(float)
    n_all = len(sales)
    sales = sales[sales.date <= end]
    sales.to_csv(out / "sales.csv", index=False)

    series = pd.DataFrame({
        "article_id": s.external_code, "location_id": s.retail,
        "launch_date": s.release_date, "end_date": s.release_date + pd.Timedelta(weeks=11),
    })
    series = series[series.launch_date <= end]
    series.to_csv(out / "series.csv", index=False)

    # ---------------------------------------------------------------- catalogue (attributes constant per product)
    g = s.groupby("external_code")
    for c in ("category", "color", "fabric", "season", "image_path"):
        assert (g[c].nunique() == 1).all(), f"{c} varies within a product"
    cat = g.agg(category=("category", "first"), color=("color", "first"), fabric=("fabric", "first"),
                season=("season", "first"), image_path=("image_path", "first"), price=("price", "median"),
                launch_date=("release_date", "min")).reset_index().rename(columns={"external_code": "article_id"})
    cat["text"] = cat.color + " " + cat.fabric + " " + cat.category  # e.g. "grey acrylic long sleeve"
    cat["image_path"] = [str(src / "images" / pth) for pth in cat.image_path]
    missing = sum(not Path(pth).exists() for pth in cat.image_path)
    cat = cat[cat.launch_date <= end]
    cat[["article_id", "category", "color", "fabric", "season", "text", "image_path", "price", "launch_date"]].to_csv(out / "catalog.csv", index=False)

    # ---------------------------------------------------------------- shops, weather, holidays
    import torch

    pairs = torch.load(src / "shop_weather_pairs.pt", weights_only=False)
    shops = sorted(s.retail.unique())
    locs = pd.DataFrame({"location_id": shops, "country": "IT", "weather_locality": [pairs.get(int(r), -1) for r in shops]})
    locs.to_csv(out / "locations.csv", index=False)
    w = pd.read_csv(src / "vis2_weather_data.csv", encoding="utf-8")
    tcol = next(c for c in w.columns if c.startswith("avg temp"))
    rcol = next(c for c in w.columns if c.startswith("rain"))
    w["date"] = pd.to_datetime(w["date"], format="%d/%m/%Y")
    wl = w[["locality", "date", tcol, rcol]].rename(columns={tcol: "temperature", rcol: "precipitation"})
    weather = locs[["location_id", "weather_locality"]].merge(wl, left_on="weather_locality", right_on="locality")
    weather = weather[weather.date <= end + pd.Timedelta(days=6)]
    weather[["location_id", "date", "temperature", "precipitation"]].to_csv(out / "weather.csv", index=False)
    italian_holidays(range(2016, 2021)).to_csv(out / "holidays.csv", index=False)

    # ---------------------------------------------------------------- sanity checks
    kept = sales.units.to_numpy()
    first = s.groupby("external_code").release_date.transform("min")
    print(f"source: {len(s):,} product x shop series | {s.external_code.nunique():,} products | {len(shops)} shops "
          f"| releases {s.release_date.min().date()} .. {s.release_date.max().date()}")
    print(f"sales rows: {len(sales):,} of {n_all:,} kept (<= {end.date()}) | weeks {sales.date.min().date()} .. {sales.date.max().date()} "
          f"({(sales.date.max() - sales.date.min()).days // 7 + 1} weekly steps)")
    print(f"units: {kept.sum():,.0f} total | zero share {np.mean(kept == 0):.3f} | negative cells {int((kept < 0).sum()):,} "
          f"(sum {kept[kept < 0].sum():,.0f}; clipped to 0 per cell by load_panel)")
    print(f"launches: {cat.shape[0]:,} products, {len(series):,} product-shop launches; "
          f"{(s.release_date > first).mean():.1%} of series launch after the product's first shop "
          f"(median spread {((g.release_date.max() - g.release_date.min()).dt.days / 7).median():.0f} weeks)")
    print(f"discount: {n_nan_disc:,} NaN cells set to 0 | promo share {sales.promo.mean():.3f} | "
          f"shops per product median {g.retail.nunique().median():.0f}")
    print(f"weather: {weather.location_id.nunique()} shops mapped to {weather.weather_locality.nunique()} localities, {len(weather):,} rows "
          f"| shops without weather rows {len(shops) - weather.location_id.nunique()} (missing: set to the mean after z-scoring) "
          f"| images missing {missing}")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
