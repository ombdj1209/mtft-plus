"""Synthetic multi-FC grocery demand with latent product semantics, plus a vectorised window dataset.

Demand for article a in fulfilment centre f on day t (negative-binomial, mean mu):
    log mu = base_a + fc_f + fc_org_f * organic_a + weekly_{c,dow} + seasonal_a(doy)
             + temp_sens_a * temp_{f,t} + elasticity_a * discount_{a,f,t} + post_promo_dip
             + holiday_{f,t} + trend_c + AR(1) noise,   mu *= launch ramp.
Product semantics (category, organic, visual freshness, premium packaging, brand, seasonal look)
drive base level, promo elasticity and seasonality. They are observable only through
simulated embeddings:
  * aligned  (SigLIP-like): image and text evidence share one concept dictionary;
  * unimodal (ResNet / DistilBERT-like): independent dictionaries per modality.
Both carry identical information content and identical nuisance variance, so any gap between
fusion strategies is attributable to alignment / dimensionality / fusion, not to leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FC_CATALOG = [("NL-AMS", "NL", 10.5), ("NL-UTR", "NL", 10.8), ("DE-DUS", "DE", 10.9), ("DE-KOL", "DE", 11.1), ("FR-PAR", "FR", 12.6)]
COUNTRIES = ["NL", "DE", "FR"]
HOLIDAYS = {
    "NL": [(1, 1), (4, 27), (5, 5), (12, 25), (12, 26), (12, 31)],
    "DE": [(1, 1), (5, 1), (10, 3), (12, 24), (12, 25), (12, 26)],
    "FR": [(1, 1), (5, 1), (7, 14), (8, 15), (11, 1), (12, 25)],
}
# name, shelf life (days), base log level, temperature sensitivity, yearly amplitude, peak day-of-year, organic rate, fresh
CATEGORIES = [
    ("fruit", 6, 2.4, 0.10, 0.15, 200, 0.35, True),
    ("vegetables", 7, 2.5, 0.00, 0.10, 240, 0.35, True),
    ("dairy", 12, 2.6, 0.00, 0.05, 30, 0.25, True),
    ("bakery", 3, 2.3, -0.05, 0.05, 350, 0.20, True),
    ("meat_fish", 5, 2.0, 0.05, 0.10, 190, 0.15, True),
    ("ready_meals", 5, 1.9, -0.10, 0.10, 20, 0.10, True),
    ("frozen", 180, 1.6, 0.05, 0.05, 10, 0.05, False),
    ("ice_cream", 365, 1.2, 0.45, 0.50, 200, 0.05, False),
    ("beverages", 365, 2.1, 0.25, 0.25, 195, 0.05, False),
    ("bbq", 30, 0.9, 0.40, 0.70, 185, 0.10, False),
    ("soups", 365, 1.3, -0.30, 0.40, 15, 0.10, False),
    ("pantry", 365, 1.8, 0.00, 0.05, 350, 0.10, False),
]
WEEKLY_BASE = np.array([0.02, -0.06, -0.03, 0.04, 0.12, 0.16, -0.25])  # Mon..Sun
KNOWN_REALS = ["promo", "discount", "temp_z", "precip_z", "holiday", "pre_holiday", "doy_sin", "doy_cos", "launched", "age"]
STATIC_REALS = ["log_price_z", "log_shelf_life_z"]
STATIC_CATS = ["fc", "category", "country"]


@dataclass
class SyntheticConfig:
    n_fc: int = 5
    n_articles: int = 1000
    n_days: int = 540
    embed_dim: int = 768
    encoder_length: int = 56
    horizon: int = 14
    n_test_origins: int = 4
    n_val_origins: int = 4
    cold_start_frac: float = 0.1
    late_launch_frac: float = 0.2
    nb_dispersion: float = 6.0
    nuisance_dims: int = 24
    nuisance_scale: float = 0.25
    noise_scale: float = 0.35
    start_date: str = "2024-01-01"
    seed: int = 7
    # grid-frequency settings (not read by generate_synthetic_retail; daily values reproduce v1-v3).
    # Old pickled configs fall back to these class-level defaults.
    freq: str = "D"
    season: int = 7  # seasonal-naive / ridge anchor period in steps
    level_window: int = 28  # steps in the recent-level statics (analogues, siblings) and the ridge level
    min_history: int = 35  # steps since launch before an aggregate node gets a ridge base forecast
    steps_per_year: float = 365.0  # normalises the product-age feature: log1p(age) / log1p(steps_per_year)


@dataclass
class SyntheticRetailData:
    cfg: SyntheticConfig
    dates: pd.DatetimeIndex
    y: np.ndarray  # (A, F, T) float32 units
    promo: np.ndarray  # (A, F, T) {0,1}
    discount: np.ndarray  # (A, F, T)
    temp_z: np.ndarray  # (F, T)
    precip_z: np.ndarray  # (F, T)
    holiday: np.ndarray  # (F, T)
    pre_holiday: np.ndarray  # (F, T)
    dow: np.ndarray  # (T,)
    doy: np.ndarray  # (T,)
    launch: np.ndarray  # (A,)
    category: np.ndarray  # (A,)
    fc_country: np.ndarray  # (F,)
    static_real: np.ndarray  # (A, 2)
    shelf_life: np.ndarray  # (A,)
    attributes: Dict[str, np.ndarray]
    emb_aligned_image: np.ndarray  # (A, D)
    emb_aligned_text: np.ndarray
    emb_uni_image: np.ndarray
    emb_uni_text: np.ndarray
    cold_start: np.ndarray  # (A,) bool
    split: Dict[str, int] = field(default_factory=dict)
    fc_names: list = field(default_factory=list)
    # Real panels only (None on the synthetic benchmark, where every location launches an article together and
    # every launched cell is observed):
    launch_loc: Optional[np.ndarray] = None  # (A, F) per-series launch step (T = never stocked there)
    live: Optional[np.ndarray] = None  # (A, F, T) bool: launched *and* sales observed (e.g. VISUELLE: 12 weeks)

    @property
    def n_articles(self) -> int:
        return self.y.shape[0]

    @property
    def n_fc(self) -> int:
        return self.y.shape[1]

    def series_launch(self) -> np.ndarray:
        """(A, F) launch step of every (article, location) series."""
        if self.launch_loc is not None:
            return self.launch_loc
        return np.broadcast_to(self.launch[:, None], (self.n_articles, self.n_fc))

    def live_cells(self, a, f, t) -> np.ndarray:
        """Broadcasting lookup of the live mask; falls back to t >= launch[a] when no mask is stored."""
        if self.live is not None:
            return self.live[a, f, t]
        return np.asarray(t) >= self.launch[a]

    def live_mask(self) -> np.ndarray:
        """(A, F, T) bool live mask (materialised)."""
        if self.live is not None:
            return self.live
        return np.broadcast_to(np.arange(self.y.shape[2])[None, None] >= self.launch[:, None, None], self.y.shape)

    def val_origins(self) -> np.ndarray:
        H = self.cfg.horizon
        return self.split["val_start"] + H * np.arange(self.cfg.n_val_origins)

    def test_origins(self) -> np.ndarray:
        H = self.cfg.horizon
        return self.split["test_start"] + H * np.arange(self.cfg.n_test_origins)


def _unit(rng: np.random.Generator, k: int, d: int) -> np.ndarray:
    v = rng.standard_normal((k, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def generate_synthetic_retail(cfg: SyntheticConfig = SyntheticConfig()) -> SyntheticRetailData:
    rng = np.random.default_rng(cfg.seed)
    A, F, T, D, H = cfg.n_articles, cfg.n_fc, cfg.n_days, cfg.embed_dim, cfg.horizon
    dates = pd.date_range(cfg.start_date, periods=T, freq="D")
    dow = dates.dayofweek.values.astype(np.int64)
    doy = dates.dayofyear.values.astype(np.int64)
    test_start = T - cfg.n_test_origins * H
    val_start = test_start - cfg.n_val_origins * H
    split = {"train_end": val_start, "val_start": val_start, "test_start": test_start}

    fcs = [FC_CATALOG[i % len(FC_CATALOG)] for i in range(F)]
    fc_names = [f"{n}" if i < len(FC_CATALOG) else f"{n}-{i}" for i, (n, _, _) in enumerate(fcs)]
    fc_country = np.array([COUNTRIES.index(c) for _, c, _ in fcs])

    # ---------------------------------------------------------------- weather & calendar
    season = np.sin(2 * np.pi * (doy - 110) / 365.25)
    country_noise = np.zeros((len(COUNTRIES), T))
    for t in range(1, T):
        country_noise[:, t] = 0.8 * country_noise[:, t - 1] + rng.normal(0, 1.6, len(COUNTRIES))
    temp = np.stack([mt + 8.5 * season + country_noise[fc_country[f]] + rng.normal(0, 0.8, T) for f, (_, _, mt) in enumerate(fcs)])
    precip = rng.gamma(0.6, 2.5, (F, T)) * (1.2 - 0.4 * season)[None]
    holiday = np.zeros((F, T))
    for f in range(F):
        md = set(HOLIDAYS[COUNTRIES[fc_country[f]]])
        holiday[f] = [(d.month, d.day) in md for d in dates]
    pre_holiday = np.concatenate([holiday[:, 1:], np.zeros((F, 1))], axis=1) * (1 - holiday)

    # ---------------------------------------------------------------- product semantics
    n_cat = len(CATEGORIES)
    cat_w = rng.dirichlet(np.full(n_cat, 4.0))
    category = rng.choice(n_cat, A, p=cat_w)
    cat_arr = lambda j: np.array([c[j] for c in CATEGORIES], dtype=float)  # noqa: E731
    shelf_life = cat_arr(1)[category] * rng.uniform(0.8, 1.2, A)
    fresh_cat = cat_arr(7)[category] > 0
    organic = (rng.random(A) < cat_arr(6)[category]).astype(float)
    freshness = np.clip(0.5 + 0.22 * organic * fresh_cat + rng.normal(0, 0.15, A), 0, 1)
    premium = rng.standard_normal(A)
    brand = rng.standard_normal(A)
    seasonal_look = rng.standard_normal(A)
    log_price = 1.0 + 0.25 * premium + 0.15 * brand + 0.20 * organic + rng.normal(0, 0.2, A)
    price_rel = log_price - np.array([log_price[category == c].mean() if (category == c).any() else 0 for c in range(n_cat)])[category]
    fz, oz, pz, bz, sz = _z(freshness), _z(organic), _z(premium), _z(brand), _z(seasonal_look)

    base = (
        cat_arr(2)[category]
        + 0.35 * brand
        - 0.6 * price_rel
        + 0.35 * np.clip(oz * fz, -2, 3) * fresh_cat  # cross-modal interaction: organic (text) x freshness (image)
        + rng.normal(0, 0.3, A)  # idiosyncratic, only learnable from history
    )
    elasticity = np.log1p(np.exp(1.2 + 0.7 * premium + 0.4 * brand))  # packaging/brand drive promo response
    yearly_amp = cat_arr(4)[category] * (1 + 0.5 * np.tanh(seasonal_look))
    temp_sens = cat_arr(3)[category] * (1 + 0.4 * np.tanh(seasonal_look))
    peak = cat_arr(5)[category]

    # ---------------------------------------------------------------- launches
    launch = np.zeros(A, dtype=np.int64)
    perm = rng.permutation(A)
    n_cold = int(round(cfg.cold_start_frac * A))
    n_late = int(round(cfg.late_launch_frac * A))
    cold_idx, late_idx = perm[:n_cold], perm[n_cold : n_cold + n_late]
    launch[cold_idx] = rng.integers(test_start - 21, test_start + 1, n_cold)
    launch[late_idx] = rng.integers(cfg.encoder_length + 14, val_start - 60, n_late)
    cold_start = np.zeros(A, dtype=bool)
    cold_start[cold_idx] = True

    # ---------------------------------------------------------------- promotions
    promo = np.zeros((A, F, T), dtype=np.float32)
    discount = np.zeros((A, F, T), dtype=np.float32)
    post = np.zeros((A, F, T), dtype=np.float32)
    rate = 1 / 55.0
    for a in range(A):
        n_ev = rng.poisson(rate * T)
        for _ in range(n_ev):
            s = int(rng.integers(0, T - 3))
            dur = int(rng.integers(3, 8))
            depth = float(rng.choice([0.15, 0.25, 0.35]))
            part = rng.random(F) < 0.85
            e = min(T, s + dur)
            promo[a, part, s:e] = 1.0
            discount[a, part, s:e] = depth
            post[a, part, e : min(T, e + 3)] = 1.0
    post *= 1 - promo

    # ---------------------------------------------------------------- demand
    fc_eff = np.log(rng.uniform(0.7, 1.4, F))
    fc_org = np.array([0.25 if COUNTRIES[c] == "NL" else (0.1 if COUNTRIES[c] == "DE" else -0.05) for c in fc_country])
    cat_week = WEEKLY_BASE[None] * (1 + rng.normal(0, 0.3, (n_cat, 1))) + rng.normal(0, 0.03, (n_cat, 7))
    trend = rng.normal(0, 0.3, n_cat)[:, None] * (np.arange(T) / 365.0)[None]
    log_mu = (
        base[:, None, None]
        + fc_eff[None, :, None]
        + fc_org[None, :, None] * organic[:, None, None]
        + cat_week[category][:, dow][:, None, :]
        + (yearly_amp[:, None] * np.cos(2 * np.pi * (doy[None] - peak[:, None]) / 365.25))[:, None, :]
        + temp_sens[:, None, None] * ((temp - 12.0) / 8.0)[None]
        + 2.0 * elasticity[:, None, None] * discount
        - 0.25 * post
        + (0.35 * pre_holiday - 0.45 * holiday)[None]
        + trend[category][:, None, :]
    )
    ar = np.zeros((A, F))
    noise = np.empty((A, F, T))
    for t in range(T):
        ar = 0.6 * ar + rng.normal(0, 0.12, (A, F))
        noise[..., t] = ar
    log_mu += noise
    age = np.arange(T)[None, :] - launch[:, None]
    ramp = np.where(age >= 0, 1 - np.exp(-(age + 1) / 7.0), 0.0)
    mu = np.exp(log_mu) * ramp[:, None, :]
    r = cfg.nb_dispersion
    y = rng.poisson(rng.gamma(r, np.maximum(mu, 1e-12) / r)).astype(np.float32)
    y[mu <= 0] = 0.0

    # ---------------------------------------------------------------- embeddings
    def encoder(concepts: Dict[str, np.ndarray], evidence: Dict[str, np.ndarray], nuis_basis: np.ndarray, gap: np.ndarray) -> np.ndarray:
        e = np.zeros((A, D))
        for k, coef in evidence.items():
            direction = concepts[k]
            e += coef[:, None] * direction if direction.ndim == 1 else coef @ direction
        e += (rng.normal(0, cfg.nuisance_scale, (A, cfg.nuisance_dims)) @ nuis_basis) + gap[None]
        e += rng.normal(0, cfg.noise_scale / np.sqrt(D), (A, D))
        return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)

    cat_oh = np.eye(n_cat)[category]
    vis_ev = {"cat": cat_oh, "fresh": 0.5 * fz, "premium": 0.5 * pz, "brand": 0.5 * bz, "season": 0.4 * sz}
    txt_ev = {"cat": cat_oh, "fresh": 0.5 * oz, "premium": 0.3 * pz, "brand": 0.5 * bz, "season": 0.4 * sz}

    def dictionary() -> Dict[str, np.ndarray]:
        return {"cat": _unit(rng, n_cat, D), **{k: _unit(rng, 1, D)[0] for k in ("fresh", "premium", "brand", "season")}}

    shared = dictionary()
    nuis_i, nuis_t = _unit(rng, cfg.nuisance_dims, D), _unit(rng, cfg.nuisance_dims, D)
    gap_i, gap_t = 0.3 * _unit(rng, 1, D)[0], 0.3 * _unit(rng, 1, D)[0]  # SigLIP-style modality gap
    emb_aligned_image = encoder(shared, vis_ev, nuis_i, gap_i)
    emb_aligned_text = encoder(shared, txt_ev, nuis_t, gap_t)
    emb_uni_image = encoder(dictionary(), vis_ev, nuis_i, gap_i)
    emb_uni_text = encoder(dictionary(), txt_ev, nuis_t, gap_t)

    static_real = np.stack([_z(log_price), _z(np.log(shelf_life))], 1).astype(np.float32)
    return SyntheticRetailData(
        cfg=cfg,
        dates=dates,
        y=y,
        promo=promo,
        discount=discount,
        temp_z=((temp - temp.mean()) / temp.std()).astype(np.float32),
        precip_z=((precip - precip.mean()) / precip.std()).astype(np.float32),
        holiday=holiday.astype(np.float32),
        pre_holiday=pre_holiday.astype(np.float32),
        dow=dow,
        doy=doy,
        launch=launch,
        category=category.astype(np.int64),
        fc_country=fc_country.astype(np.int64),
        static_real=static_real,
        shelf_life=shelf_life,
        attributes={"organic": organic, "freshness": freshness, "premium": premium, "brand": brand, "seasonal_look": seasonal_look, "elasticity": elasticity},
        emb_aligned_image=emb_aligned_image,
        emb_aligned_text=emb_aligned_text,
        emb_uni_image=emb_uni_image,
        emb_uni_text=emb_uni_text,
        cold_start=cold_start,
        split=split,
        fc_names=fc_names,
    )


def pca_compress(train_x: np.ndarray, all_x: np.ndarray, k: int) -> np.ndarray:
    """Picnic-style fixed k-dim compression (PCA fitted on training articles), z-scored."""
    mu = train_x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(train_x - mu, full_matrices=False)
    z_train = (train_x - mu) @ vt[:k].T
    z = (all_x - mu) @ vt[:k].T
    return ((z - z_train.mean(0)) / (z_train.std(0) + 1e-8)).astype(np.float32)


@dataclass
class Analogues:
    """Top-k analogue products per article. idx (A, k) into the article axis, weights (A, k) summing to 1."""

    idx: np.ndarray
    weights: np.ndarray
    source: str = ""


def build_analogues(keys: np.ndarray, pool: np.ndarray, k: int = 10, temperature: float = 0.05) -> Analogues:
    """Cosine top-k over ``keys`` (A, D) restricted to ``pool`` articles (bool, A). An article never retrieves itself,
    so training articles see the same leave-one-out setting as cold-start articles at test time."""
    z = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
    sim = z @ z[pool].T  # (A, P)
    pool_ids = np.where(pool)[0]
    sim[pool_ids[None, :] == np.arange(len(z))[:, None]] = -np.inf
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    s_top = np.take_along_axis(sim, top, 1)
    w = np.exp((s_top - s_top.max(1, keepdims=True)) / temperature)
    return Analogues(pool_ids[top], (w / w.sum(1, keepdims=True)).astype(np.float32), "embedding")


def build_category_analogues(category: np.ndarray, pool: np.ndarray, k: int = 10, seed: int = 0) -> Analogues:
    """Control: k random same-category pool articles with uniform weights (no multimodal information)."""
    rng = np.random.default_rng(seed)
    A = len(category)
    idx = np.zeros((A, k), dtype=np.int64)
    for a in range(A):
        cand = np.where(pool & (category == category[a]) & (np.arange(A) != a))[0]
        if len(cand) == 0:
            cand = np.where(pool & (np.arange(A) != a))[0]
        idx[a] = rng.choice(cand, k, replace=len(cand) < k)
    return Analogues(idx, np.full((A, k), 1.0 / k, dtype=np.float32), "category")


def build_index(data: SyntheticRetailData, origins: np.ndarray, articles: Optional[np.ndarray] = None, require_target: bool = True) -> np.ndarray:
    """All (article, fc, origin) triples in (article, fc, origin) lexicographic order; optionally only windows
    whose horizon contains at least one live (launched and observed) cell."""
    A, F, H = data.n_articles, data.n_fc, data.cfg.horizon
    arts = np.arange(A) if articles is None else np.asarray(articles)
    origins = np.asarray(origins)
    if not require_target:
        a, f, o = np.meshgrid(arts, np.arange(F), origins, indexing="ij")
        return np.stack([a.ravel(), f.ravel(), o.ravel()], 1).astype(np.int64)
    if data.live is None:
        keep = np.broadcast_to((data.launch[arts][:, None] < origins[None] + H)[:, None, :], (len(arts), F, len(origins)))
    else:  # built per origin: never materialises all A * F * O triples
        live = data.live[arts]
        keep = np.stack([live[:, :, o : o + H].any(2) for o in origins], -1)
    i, f, k = np.nonzero(keep)
    return np.stack([arts[i], f, origins[k]], 1).astype(np.int64)


class DemandWindowDataset(Dataset):
    """Vectorised window sampler. ``__getitems__`` builds a whole batch with numpy fancy indexing.

    mm: dict with ``image``/``text`` (A, D), ``mask`` (A, 2) and optional ``scalar`` (A, k).
    Use with ``collate_fn=identity_collate``.
    """

    def __init__(
        self,
        data: SyntheticRetailData,
        index: np.ndarray,
        mm: Dict[str, np.ndarray],
        analogues: Optional[Analogues] = None,
        siblings: bool = False,
        lifecycle=None,
        la_pool: bool = False,
        la_tokens: bool = False,
    ) -> None:
        """``lifecycle`` (mtft_plus.lifecycle.LifecycleFeatures): lifecycle-aligned information. ``la_pool`` appends the
        four pooled LA channels to the known inputs; ``la_tokens`` adds the individual sibling / analogue tokens."""
        self.d = data
        self.analogues = analogues
        self.siblings = siblings
        self.lifecycle, self.la_pool, self.la_tokens = lifecycle, la_pool, la_tokens
        if (la_pool or la_tokens) and lifecycle is None:
            raise ValueError("la_pool / la_tokens need lifecycle features")
        self.index = index
        self.L, self.H = data.cfg.encoder_length, data.cfg.horizon
        self.level_window = int(data.cfg.level_window)
        self.age_norm = float(np.log1p(data.cfg.steps_per_year))
        self.offsets = np.arange(-self.L, self.H)
        self.mm = {k: torch.as_tensor(v) for k, v in mm.items()}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.__getitems__([i])

    def __getitems__(self, ids) -> Dict[str, torch.Tensor]:
        d, L = self.d, self.L
        ix = self.index[np.asarray(ids)]
        a, f, o = ix[:, 0], ix[:, 1], ix[:, 2]
        tt = o[:, None] + self.offsets[None]
        A_, F_ = a[:, None], f[:, None]
        y = d.y[A_, F_, tt]
        age = tt - d.series_launch()[a, f][:, None]
        launched = (age >= 0).astype(np.float32)
        target_live = d.live_cells(A_, F_, tt[:, L:])
        doy = d.doy[tt]
        known = np.stack(
            [
                d.promo[A_, F_, tt],
                d.discount[A_, F_, tt] * 4.0,
                d.temp_z[F_, tt],
                d.precip_z[F_, tt],
                d.holiday[F_, tt],
                d.pre_holiday[F_, tt],
                np.sin(2 * np.pi * doy / 365.25),
                np.cos(2 * np.pi * doy / 365.25),
                launched,
                np.log1p(np.clip(age, 0, None)) / self.age_norm,
            ],
            -1,
        ).astype(np.float32)
        la = None
        if self.lifecycle is not None:
            from .lifecycle import POOL_CHANNELS, lifecycle_batch

            la = lifecycle_batch(self.lifecycle, d, a, f, o, L, self.H, tokens=self.la_tokens)
            if self.la_pool:
                known = np.concatenate([known, np.stack([la[c] for c in POOL_CHANNELS], -1)], -1).astype(np.float32)
        dow = d.dow[tt][..., None]
        ylog = np.log1p(y).astype(np.float32)
        enc_obs = ylog[:, :L, None]
        static_real = d.static_real[a]
        if self.analogues is not None:
            nb = self.analogues.idx[a]  # (B, k)
            ttl = tt[:, None, :L]
            ny = np.log1p(d.y[nb[:, :, None], f[:, None, None], ttl])  # (B, k, L): analogue demand, past only
            live = d.live_cells(nb[:, :, None], f[:, None, None], ttl).astype(np.float32)
            w = self.analogues.weights[a][:, :, None] * live
            wsum = w.sum(1)
            series = (w * ny).sum(1) / np.maximum(wsum, 1e-6)  # (B, L)
            level = series[:, -self.level_window :].mean(1, keepdims=True)
            enc_obs = np.concatenate([enc_obs, series[..., None], (wsum > 0)[..., None]], -1).astype(np.float32)
            static_real = np.concatenate([static_real, level], 1).astype(np.float32)
        if self.siblings:
            # cross-FC learning: mean log-demand of the same article in the *other* FCs where it is live, encoder
            # steps only. With simultaneous launches (synthetic benchmark) this is the plain mean over the F-1 others.
            a3, g3, t3 = a[:, None, None], np.arange(d.n_fc)[None, :, None], tt[:, None, :L]
            ylog_all = np.log1p(d.y[a3, g3, t3])  # (B, F, L)
            if d.live is None:
                sib = (ylog_all.sum(1) - ylog[:, :L]) / max(d.n_fc - 1, 1)
            else:
                lv = d.live[a3, g3, t3].astype(np.float32)
                own = lv[np.arange(len(a)), f]  # (B, L)
                n_other = lv.sum(1) - own
                sib = ((lv * ylog_all).sum(1) - own * ylog[:, :L]) / np.maximum(n_other, 1.0)
            enc_obs = np.concatenate([enc_obs, sib[..., None]], -1).astype(np.float32)
            static_real = np.concatenate([static_real, sib[:, -self.level_window :].mean(1, keepdims=True)], 1).astype(np.float32)
        a_t = torch.as_tensor(a)
        batch = {
            "static_cat": torch.as_tensor(np.stack([f, d.category[a], d.fc_country[f]], 1)),
            "static_real": torch.as_tensor(static_real),
            "enc_observed": torch.as_tensor(enc_obs),
            "enc_known_real": torch.as_tensor(known[:, :L]),
            "enc_known_cat": torch.as_tensor(dow[:, :L]),
            "dec_known_real": torch.as_tensor(known[:, L:]),
            "dec_known_cat": torch.as_tensor(dow[:, L:]),
            "target": torch.as_tensor(ylog[:, L:]),
            "target_mask": torch.as_tensor(np.asarray(target_live, dtype=bool)),
            "article_idx": a_t,
            "fc_idx": torch.as_tensor(f),
            "origin": torch.as_tensor(o),
            "mm_image": self.mm["image"][a_t],
            "mm_text": self.mm["text"][a_t],
            "mm_mask": self.mm["mask"][a_t],
        }
        if "scalar" in self.mm:
            batch["mm_scalar"] = self.mm["scalar"][a_t]
        if la is not None:
            batch["age_idx"] = torch.as_tensor(la["age_idx"])
            if self.la_tokens:
                for key in ("la_sib_val", "la_sib_mask", "la_sib_shop", "la_sib_lag", "la_ana_val", "la_ana_mask",
                            "la_ana_support", "la_ana_sim", "la_ana_emb", "la_self_emb", "la_self_shop"):
                    batch[key] = torch.as_tensor(la[key])
        return batch


def identity_collate(batch):
    return batch
