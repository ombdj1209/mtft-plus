"""Load a real retail panel (long-format DataFrames) into the structure the benchmark consumes.

Required
    sales:    article_id, location_id, date, units            [+ promo (0/1), discount (0..1)]
    catalog:  article_id, category                            [+ price, shelf_life_days, launch_date]
    locations: location_id, country
Optional
    weather:  location_id, date, temperature, precipitation
    holidays: country, date
    series:   article_id, location_id, launch_date [, end_date]   per-(article, location) launches and
              observation windows (inclusive end). Series not listed are never stocked. Cells outside
              [launch, end] are *not observed*: they are excluded from targets, metrics and sibling/analogue
              means, and their sales are set to zero. Article launch = its first series launch.
    embeddings: EmbeddingBank from MultimodalFeatureExtractor (aligned SigLIP space)
    unimodal_image / unimodal_text: (A, D) arrays for the Picnic-style baseline (e.g. ResNet / DistilBERT);
        if omitted the baseline compresses the aligned embeddings instead (state this in any write-up).

``freq`` sets the panel grid ("D" daily, "W-MON" weekly, ...); horizons and windows are in grid steps.
Every date is snapped to the start of the grid period containing it (sales are summed per period), and
launch dates are snapped the same way. Missing (article, location, date) cells are treated as zero
sales. Launch = catalog launch_date if given, else first positive sale. Articles launched within
``cold_start_days`` steps before the test period are held out of training as cold-start items.

Frequency-dependent settings (season, level_window, min_history, steps_per_year) default from
``freq`` via ``freq_defaults`` and can be overridden.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import COUNTRIES, SyntheticConfig, SyntheticRetailData

# daily values are the v1-v3 settings; weekly ones translate them (28 d -> 4 w, 35 d -> 5 w). Weekly
# season=1 (last-value anchor / non-seasonal MASE) because most fashion series live < 1 year.
_FREQ_DEFAULTS = {
    "D": dict(season=7, level_window=28, min_history=35, steps_per_year=365.0),
    "W": dict(season=1, level_window=4, min_history=5, steps_per_year=52.0),
}


def freq_defaults(freq: str) -> Dict[str, float]:
    base = pd.tseries.frequencies.to_offset(freq).name.split("-")[0]
    if base not in _FREQ_DEFAULTS:
        raise ValueError(f"no default seasonal settings for freq={freq!r}; pass season/level_window/min_history/steps_per_year")
    return dict(_FREQ_DEFAULTS[base])


def _snap(values: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Index of the grid period containing each date (-1 before the grid, len(dates) after it)."""
    idx = dates.searchsorted(values, side="right") - 1
    end = dates[-1] + dates.freq
    return np.where(values >= end, len(dates), idx)


def _z(x: np.ndarray) -> np.ndarray:
    """z-score ignoring NaN; missing values become 0 (the mean)."""
    z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


def load_panel(
    sales: pd.DataFrame,
    catalog: pd.DataFrame,
    locations: pd.DataFrame,
    weather: Optional[pd.DataFrame] = None,
    holidays: Optional[pd.DataFrame] = None,
    embeddings=None,
    unimodal_image: Optional[np.ndarray] = None,
    unimodal_text: Optional[np.ndarray] = None,
    encoder_length: int = 56,
    horizon: int = 14,
    n_val_origins: int = 4,
    n_test_origins: int = 4,
    cold_start_days: int = 21,
    freq: str = "D",
    season: Optional[int] = None,
    level_window: Optional[int] = None,
    min_history: Optional[int] = None,
    steps_per_year: Optional[float] = None,
    series: Optional[pd.DataFrame] = None,
) -> SyntheticRetailData:
    fs = freq_defaults(freq) if None in (season, level_window, min_history, steps_per_year) else {}
    season = int(season if season is not None else fs["season"])
    level_window = int(level_window if level_window is not None else fs["level_window"])
    min_history = int(min_history if min_history is not None else fs["min_history"])
    steps_per_year = float(steps_per_year if steps_per_year is not None else fs["steps_per_year"])
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"]).dt.normalize()
    off = pd.tseries.frequencies.to_offset(freq)
    dates = pd.date_range(off.rollback(sales["date"].min()), sales["date"].max(), freq=freq)
    sales["date"] = dates[np.clip(_snap(sales["date"], dates), 0, len(dates) - 1)]  # period starts
    arts: List[str] = list(catalog["article_id"].astype(str))
    locs: List[str] = list(locations["location_id"].astype(str))
    a_pos = {a: i for i, a in enumerate(arts)}
    l_pos = {l: i for i, l in enumerate(locs)}
    A, F, T = len(arts), len(locs), len(dates)
    if T < encoder_length + (n_val_origins + n_test_origins + 2) * horizon:
        raise ValueError("history too short for the requested encoder/validation/test layout")

    ai = sales["article_id"].astype(str).map(a_pos)
    li = sales["location_id"].astype(str).map(l_pos)
    ti = pd.Series(dates.get_indexer(sales["date"]), index=sales.index)
    ok = ai.notna() & li.notna() & (ti >= 0)
    ai, li, ti = ai[ok].astype(int).values, li[ok].astype(int).values, ti[ok].values

    def cube(col: str, accumulate: bool) -> np.ndarray:
        out = np.zeros((A, F, T), dtype=np.float32)
        if col in sales:
            vals = sales.loc[ok, col].astype(float).fillna(0.0).values
            np.add.at(out, (ai, li, ti), vals)  # duplicate rows (several orders per day, days in a week) summed
            if not accumulate:  # rates (promo flag, discount depth): mean over the rows in the cell
                cnt = np.zeros((A, F, T), dtype=np.float32)
                np.add.at(cnt, (ai, li, ti), 1.0)
                out = np.divide(out, cnt, out=out, where=cnt > 0)
        return out

    y = np.clip(cube("units", True), 0, None)
    promo = cube("promo", False)
    discount = cube("discount", False)

    cat_codes, cat_levels = pd.factorize(catalog["category"].astype(str))
    mm_mask = np.ones((A, 2), dtype=bool)
    if embeddings is not None:
        order = embeddings.indices(arts).numpy()
        img = embeddings.image.float().numpy()[order]
        txt = embeddings.text.float().numpy()[order]
        mm_mask = embeddings.mask.bool().numpy()[order]  # [has_image, has_text]
    else:
        img = txt = np.zeros((A, 8), dtype=np.float32)

    launch_loc = live = None
    if series is not None:
        sa = series["article_id"].astype(str).map(a_pos)
        sl = series["location_id"].astype(str).map(l_pos)
        m = sa.notna() & sl.notna()
        sa, sl = sa[m].astype(int).values, sl[m].astype(int).values
        start = np.clip(_snap(pd.to_datetime(series.loc[m, "launch_date"]).dt.normalize(), dates), 0, T)
        if "end_date" in series:
            end = np.clip(_snap(pd.to_datetime(series.loc[m, "end_date"]).dt.normalize(), dates), -1, T - 1)
        else:
            end = np.full(len(sa), T - 1)
        launch_loc = np.full((A, F), T, dtype=np.int64)
        np.minimum.at(launch_loc, (sa, sl), start)
        live = np.zeros((A, F, T), dtype=bool)
        tt = np.arange(T)
        for s0 in range(0, len(sa), 20000):  # (rows, T) chunks
            sel = slice(s0, s0 + 20000)
            win = (tt[None] >= start[sel, None]) & (tt[None] <= end[sel, None])
            np.logical_or.at(live, (sa[sel], sl[sel]), win)
        y *= live
        launch = launch_loc.min(1)
    elif "launch_date" in catalog:
        ld = pd.to_datetime(catalog["launch_date"]).dt.normalize()
        launch = np.clip(_snap(ld.fillna(dates[0]), dates), 0, T).astype(np.int64)
    else:
        any_sale = (y.sum(1) > 0)
        launch = np.where(any_sale.any(1), any_sale.argmax(1), T).astype(np.int64)

    country = locations["country"].astype(str).values
    fc_country = np.array([COUNTRIES.index(c) if c in COUNTRIES else len(COUNTRIES) - 1 for c in country], dtype=np.int64)

    temp = np.zeros((F, T), dtype=np.float32)
    precip = np.zeros((F, T), dtype=np.float32)
    if weather is not None:  # mean over the days in each grid period; missing periods/shops -> NaN -> 0 after z-scoring
        temp[:] = np.nan
        precip[:] = np.nan
        w = weather.copy()
        w["date"] = pd.to_datetime(w["date"]).dt.normalize()
        w["_l"] = w["location_id"].astype(str).map(l_pos)
        w["_t"] = _snap(w["date"], dates)
        w = w[w["_l"].notna() & (w["_t"] >= 0) & (w["_t"] < T)]
        g = w.groupby([w["_l"].astype(int), "_t"])[["temperature", "precipitation"]].mean()
        li_w, ti_w = g.index.get_level_values(0).values, g.index.get_level_values(1).values
        temp[li_w, ti_w] = g["temperature"].values
        precip[li_w, ti_w] = g["precipitation"].values
    holiday = np.zeros((F, T), dtype=np.float32)
    if holidays is not None:  # a period is a holiday period if it contains a holiday
        hd = holidays.copy()
        hd["date"] = pd.to_datetime(hd["date"]).dt.normalize()
        for f in range(F):
            idx = _snap(hd.loc[hd["country"].astype(str) == country[f], "date"], dates)
            holiday[f, idx[(idx >= 0) & (idx < T)]] = 1.0
    pre_holiday = np.concatenate([holiday[:, 1:], np.zeros((F, 1), np.float32)], 1) * (1 - holiday)

    price = catalog["price"].astype(float).values if "price" in catalog else np.ones(A)
    shelf = catalog["shelf_life_days"].astype(float).values if "shelf_life_days" in catalog else np.full(A, 30.0)
    static_real = np.stack([_z(np.log(np.maximum(price, 1e-3))), _z(np.log(np.maximum(shelf, 1.0)))], 1)

    cfg = SyntheticConfig(
        n_fc=F, n_articles=A, n_days=T, embed_dim=img.shape[1], encoder_length=encoder_length, horizon=horizon,
        n_test_origins=n_test_origins, n_val_origins=n_val_origins, freq=freq, season=season,
        level_window=level_window, min_history=min_history, steps_per_year=steps_per_year,
    )
    test_start = T - n_test_origins * horizon
    val_start = test_start - n_val_origins * horizon
    cold = (launch >= test_start - cold_start_days) & (launch < T)
    attributes = {"category_levels": np.asarray(cat_levels), "article_ids": np.asarray(arts), "mm_mask": mm_mask}
    if launch_loc is not None:  # series new to their location, of articles already sold elsewhere
        attributes["new_in_shop"] = (launch_loc >= test_start - cold_start_days) & (launch_loc < T) & ~cold[:, None]
    return SyntheticRetailData(
        cfg=cfg,
        dates=dates,
        y=y,
        promo=promo,
        discount=discount,
        temp_z=_z(temp),
        precip_z=_z(precip),
        holiday=holiday,
        pre_holiday=pre_holiday.astype(np.float32),
        dow=dates.dayofweek.values.astype(np.int64),
        doy=dates.dayofyear.values.astype(np.int64),
        launch=launch.astype(np.int64),
        category=cat_codes.astype(np.int64),
        fc_country=fc_country,
        static_real=static_real,
        shelf_life=shelf,
        attributes=attributes,
        emb_aligned_image=img.astype(np.float32),
        emb_aligned_text=txt.astype(np.float32),
        emb_uni_image=(unimodal_image if unimodal_image is not None else img).astype(np.float32),
        emb_uni_text=(unimodal_text if unimodal_text is not None else txt).astype(np.float32),
        cold_start=cold,
        split={"train_end": val_start, "val_start": val_start, "test_start": test_start, "cold_start_steps": cold_start_days},
        fc_names=locs,
        launch_loc=launch_loc,
        live=live,
    )


def panel_to_frames(d: SyntheticRetailData):
    """Inverse of ``load_panel`` for testing and for exporting the synthetic benchmark."""
    A, F, T = d.y.shape
    a, f, t = np.nonzero((d.y > 0) | (d.promo > 0))
    sales = pd.DataFrame(
        {
            "article_id": [f"A{i:04d}" for i in a],
            "location_id": [d.fc_names[i] for i in f],
            "date": d.dates[t],
            "units": d.y[a, f, t],
            "promo": d.promo[a, f, t],
            "discount": d.discount[a, f, t],
        }
    )
    catalog = pd.DataFrame(
        {
            "article_id": [f"A{i:04d}" for i in range(A)],
            "category": d.category.astype(str),
            "price": np.exp(d.static_real[:, 0]),
            "shelf_life_days": d.shelf_life,
            "launch_date": d.dates[np.minimum(d.launch, T - 1)],
        }
    )
    locations = pd.DataFrame({"location_id": d.fc_names, "country": [COUNTRIES[c] for c in d.fc_country]})
    return sales, catalog, locations
