"""MTFT+ v2 benchmark: multi-seed, equal-compute comparison with controls and ablations.

    python benchmark.py                       # 3 data seeds x 5 main models (+2 ablations on the first seed)
    python benchmark.py --seeds 7 --quick     # smoke test
    python report.py                          # aggregate results/partials -> results/results.md

Every (seed, model) run writes results/partials/<seed>__<slug>.json, so interrupted runs resume.
The data generator is frozen from v1 (mtft_plus/data.py:generate_synthetic_retail) so the benchmark
cannot be tuned towards the proposed method.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler

from mtft_plus.calibration import LogConformalCalibrator
from mtft_plus.data import (
    KNOWN_REALS,
    STATIC_REALS,
    DemandWindowDataset,
    SyntheticConfig,
    build_analogues,
    build_category_analogues,
    build_index,
    generate_synthetic_retail,
    identity_collate,
    pca_compress,
)
from mtft_plus.lightning_module import MTFTLightningModule
from mtft_plus.losses import newsvendor_quantile
from mtft_plus.metrics import evaluate, seasonal_naive_scale
from mtft_plus.reconciliation import Hierarchy, MinTReconciler, SeasonalRidgeForecaster, swanson_mean

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*num_workers.*")
warnings.filterwarnings("ignore", ".*All-NaN slice.*")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

QUANTILES = (0.1, 0.5, 0.9)
REG = dict(mm_feature_dropout=0.2, mm_modality_dropout=0.2)
MAIN_MODELS = {
    "TFT": dict(fusion="none", mm="aligned", analogues=None),
    "MTFT-Picnic": dict(fusion="static_scalar", mm="picnic", analogues=None),
    "TFT + category analogues": dict(fusion="none", mm="aligned", analogues="category"),
    "MTFT-Picnic + RA": dict(fusion="static_scalar", mm="picnic", analogues="picnic"),
    "MTFT+ v2 (ours)": dict(fusion="cross_attention", mm="aligned", analogues="aligned", align_weight=0.05, **REG),
}
V3_MODELS = {
    # v3: learnings applied -- keep aligned embeddings + regularisation + retrieval, drop cross-attention
    # (ablation: no gain), add cross-FC sibling learning (Chronos-2-style cross-learning, as input channels).
    "MTFT+ v3 (ours)": dict(fusion="static_pooled", mm="aligned", analogues="aligned", siblings=True, align_weight=0.05, **REG),
    "MTFT-Picnic + siblings": dict(fusion="static_scalar", mm="picnic", analogues=None, siblings=True),
}
# MTFT+ v4 (results/v4_preregistration.md): lifecycle-aligned (LA) siblings and analogues. "la" = LA information:
# "pool" = the four pooled per-step channels (given to the control too), "attn" = lifecycle attention over the tokens.
V4_MODELS = {
    "MTFT-Picnic + siblings + LA-pool": dict(fusion="static_scalar", mm="picnic", analogues=None, siblings=True, la="pool", la_retrieval="aligned"),
    "MTFT+ v4 (ours)": dict(fusion="static_pooled", mm="aligned", analogues=None, siblings=True, la="attn", la_retrieval="aligned", age_emb=True, align_weight=0.05, **REG),
}
V41_MODELS = {  # exploratory (amendment (d) 5): history-aware gating of the lifecycle attention
    "MTFT+ v4.1 (exploratory)": dict(fusion="static_pooled", mm="aligned", analogues=None, siblings=True, la="attn", la_retrieval="aligned", age_emb=True, la_gate=True, align_weight=0.05, **REG),
}
V4_ABLATIONS = {
    "v4 w/o lifecycle attention": dict(fusion="static_pooled", mm="aligned", analogues=None, siblings=True, la="pool", la_retrieval="aligned", age_emb=True, align_weight=0.05, **REG),
    "v4 w/o analogue attention": dict(fusion="static_pooled", mm="aligned", analogues=None, siblings=True, la="attn", la_branches="siblings", la_retrieval="aligned", age_emb=True, align_weight=0.05, **REG),
    "v4 w/o sibling attention": dict(fusion="static_pooled", mm="aligned", analogues=None, siblings=True, la="attn", la_branches="analogues", la_retrieval="aligned", age_emb=True, align_weight=0.05, **REG),
    "v4 unimodal retrieval": dict(fusion="static_pooled", mm="aligned", analogues=None, siblings=True, la="attn", la_retrieval="picnic", age_emb=True, align_weight=0.05, **REG),
}
ALL_MODELS = {}  # filled below

ABLATIONS = {
    "ours w/o cross-attention": dict(fusion="static_pooled", mm="aligned", analogues="aligned", align_weight=0.05, **REG),
    "ours w/o retrieval (v1 + reg.)": dict(fusion="cross_attention", mm="aligned", analogues=None, align_weight=0.05, **REG),
}


ALL_MODELS.update(MAIN_MODELS, **V3_MODELS, **ABLATIONS, **V4_MODELS, **V4_ABLATIONS, **V41_MODELS)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class BestState(pl.Callback):
    def __init__(self) -> None:
        self.best, self.state, self.history = float("inf"), None, []

    def on_validation_epoch_end(self, trainer, module) -> None:
        v = float(trainer.callback_metrics.get("val_quantile_loss", float("inf")))
        self.history.append(v)
        if v < self.best:  # keep the copy on the CPU so GPU memory is not doubled
            self.best, self.state = v, {k: t.detach().cpu().clone() for k, t in module.state_dict().items()}


def resolve_device(accelerator: str) -> torch.device:
    if accelerator in ("gpu", "cuda") or (accelerator == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise SystemExit(f"--accelerator {accelerator} requested but torch.cuda.is_available() is False (torch {torch.__version__})")
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict(module: MTFTLightningModule, ds: DemandWindowDataset, device: torch.device, bs: int = 2048, attribution: bool = False):
    """Full-precision inference on ``device`` (mixed precision is used for training only)."""
    module.to(device).eval()
    module.model.return_mm_attribution = attribution
    preds, attrs, promos = [], [], []
    for s in range(0, len(ds), bs):
        batch = ds.__getitems__(list(range(s, min(s + bs, len(ds)))))
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = module.model(batch)
        preds.append(out["prediction"].float().cpu())
        if attribution and out["modality_attribution"] is not None:
            attrs.append(out["modality_attribution"][..., :3].float().cpu())
            promos.append(batch["dec_known_real"][..., KNOWN_REALS.index("promo")].cpu())
    module.model.return_mm_attribution = False
    q = torch.expm1(torch.cat(preds)).clamp_min(0).numpy()
    return q, (torch.cat(attrs).numpy() if attrs else None), (torch.cat(promos).numpy() if promos else None)


def product_features(d, warm: np.ndarray, k: int, seed: int):
    """Multimodal inputs and analogue sets. PCA (MTFT-Picnic's 10-d scalars) and analogue pools use warm articles only."""
    ones = np.asarray(d.attributes.get("mm_mask", np.ones((d.n_articles, 2), dtype=bool)), dtype=bool)  # [has_image, has_text]
    picnic_scalar = np.concatenate(
        [pca_compress(d.emb_uni_image[warm], d.emb_uni_image, min(10, d.emb_uni_image.shape[1])),
         pca_compress(d.emb_uni_text[warm], d.emb_uni_text, min(10, d.emb_uni_text.shape[1]))], 1
    )
    mm = {
        "aligned": {"image": d.emb_aligned_image, "text": d.emb_aligned_text, "mask": ones},
        "picnic": {"image": d.emb_uni_image, "text": d.emb_uni_text, "mask": ones, "scalar": picnic_scalar},
    }
    analogues = {
        "aligned": build_analogues(np.concatenate([d.emb_aligned_image, d.emb_aligned_text], 1), warm, k=k),
        "picnic": build_analogues(picnic_scalar, warm, k=k),
        "category": build_category_analogues(d.category, warm, k=k, seed=seed),
    }
    return mm, analogues


class SeedContext:
    """Everything that depends only on the data seed (shared by all models of that seed)."""

    def __init__(self, seed: int, args) -> None:
        t0 = time.time()
        self.seed = seed
        if args.panel:  # real data prepared by examples/prepare_panel.py; the seed then drives model init/sampling only
            with open(args.panel, "rb") as fh:
                d = self.data = pickle.load(fh)
            self.cfg = d.cfg
        else:
            self.cfg = SyntheticConfig(n_fc=args.fcs, n_articles=args.articles, n_days=args.days, seed=seed)
            d = self.data = generate_synthetic_retail(self.cfg)
        A, F, H, L = d.n_articles, d.n_fc, self.cfg.horizon, self.cfg.encoder_length
        self.A, self.F, self.H, self.L = A, F, H, L
        warm = ~d.cold_start
        self.warm_ids = np.where(warm)[0]
        self.mm, self.analogues = product_features(d, warm, args.k, seed)
        vs = d.split["val_start"]
        self.vs = vs
        self.val_o, self.test_o = d.val_origins(), d.test_origins()
        stride = int(getattr(self.cfg, "train_origin_stride", 1))  # 1 on VISUELLE / synthetic; 7 on M5 (pre-registered)
        self.train_idx = build_index(d, np.arange(L, vs - H + 1, stride), self.warm_ids)
        rng = np.random.default_rng(seed)
        vm = build_index(d, self.val_o, self.warm_ids)
        self.val_mon_idx = vm[rng.permutation(len(vm))[:4096]]
        # Real panels with a live mask: only windows with a live target cell are predicted; every other cell
        # (never stocked, not yet launched, or outside its observation window) is forecast as 0. Whether a cell is
        # live follows from the launch plan and the fixed observation window, both known in advance.
        self.sparse_pred = d.live is not None
        self.val_all_idx = build_index(d, self.val_o, require_target=self.sparse_pred)
        self.test_all_idx = build_index(d, self.test_o, require_target=self.sparse_pred)

        self.y_val = d.y[:, :, self.val_o[:, None] + np.arange(H)]
        self.y_test = d.y[:, :, self.test_o[:, None] + np.arange(H)]
        self.cold_b = np.repeat(d.cold_start, F)
        nis = d.attributes.get("new_in_shop")
        self.new_in_shop_b = None if nis is None else np.asarray(nis).reshape(A * F)
        yg = d.attributes.get("young")
        self.young_b = None if yg is None else np.asarray(yg).reshape(A * F)
        c = self.cfg  # frequency settings; daily defaults reproduce v1-v3
        self.season, self.level_window, self.min_history = int(c.season), int(c.level_window), int(c.min_history)
        Yb = d.y.reshape(A * F, -1)
        T = Yb.shape[1]
        live_b = d.live_mask().reshape(A * F, T)  # synthetic: t >= launch
        self.mask_val_b = live_b[:, self.val_o[:, None] + np.arange(H)]
        self.mask_test_b = live_b[:, self.test_o[:, None] + np.arange(H)]
        hv = live_b.copy()
        hv[:, vs:] = False
        # MASE scale: in-sample history before the validation start (v1-v3). Real panels with short life cycles
        # (VISUELLE: 12 weeks) would leave no test series with a scale, so there it uses history before the test start.
        self.scale_end = d.split["test_start"] if d.live is not None else vs
        hs = live_b.copy()
        hs[:, self.scale_end :] = False
        self.scale_b = seasonal_naive_scale(Yb, hs, season=self.season)
        del hs

        self.hier = h = Hierarchy(A, F)
        self.Afull = h.aggregation_matrix(torch.float64, sparse=True)  # dense A is O(n_agg * m): 26 GB at VISUELLE scale
        Yagg = h.aggregate_upper(Yb)
        live_agg = h.aggregate_upper(live_b) > 0  # an aggregate node is live while any of its series is
        del live_b
        launch_agg = np.concatenate([[0], np.zeros(F, dtype=np.int64), d.launch])
        # per-series pre-validation mean demand since launch (+1e-3): weights for the aggregate covariates
        live = hv.copy()
        cnt = live.sum(1)
        w_b = np.where(cnt > 0, np.where(live, Yb, 0).sum(1, dtype=np.float64) / np.maximum(cnt, 1), 0.0) + 1e-3
        del live
        # demand-weighted covariates of every aggregate node, built one covariate at a time (never (m, T, 6))
        W = w_b.reshape(A, F)
        wsum = np.concatenate([[W.sum()], W.sum(0), W.sum(1)])[:, None]  # (n_agg, 1)

        def agg_bottom(x):  # (A, F, T) -> (n_agg, T)
            return h.aggregate_upper((W[:, :, None] * x).reshape(A * F, T)) / wsum

        def agg_location(x):  # (F, T), shared by all articles -> (n_agg, T)
            return np.concatenate([(W.sum(0) @ x)[None], x * W.sum(0)[:, None], W @ x]) / wsum

        cov_agg = np.stack(
            [agg_bottom(d.discount * 4), agg_bottom(d.promo)]
            + [agg_location(x.astype(np.float64)) for x in (d.temp_z, d.precip_z, d.holiday, d.pre_holiday)],
            -1,
        )
        phase = d.dow if self.season == 7 else np.arange(T) % self.season
        stride = max(1, int(round(3 * float(c.steps_per_year) / 365.0)))  # every ~3 days (v1-v3), every step weekly
        ridge = SeasonalRidgeForecaster(H, QUANTILES, min_history=self.min_history, season=self.season, level_window=self.level_window)
        o0 = max(self.min_history, self.level_window, self.season)
        ridge.fit(Yagg, cov_agg, phase, launch_agg, np.arange(o0, vs - H + 1, stride))
        self.agg_val_q, self.agg_val_ok = ridge.predict(Yagg, cov_agg, phase, launch_agg, self.val_o)
        self.agg_test_q, self.agg_test_ok = ridge.predict(Yagg, cov_agg, phase, launch_agg, self.test_o)
        del cov_agg
        hva = live_agg.copy()
        hva[:, self.scale_end :] = False
        self.scale_all = np.concatenate([seasonal_naive_scale(Yagg, hva, season=self.season), self.scale_b])
        self.mask_val_all = np.concatenate([live_agg[:, self.val_o[:, None] + np.arange(H)], self.mask_val_b])
        self.mask_test_all = np.concatenate([live_agg[:, self.test_o[:, None] + np.arange(H)], self.mask_test_b])
        self.y_test_all = h.aggregate(self.y_test.reshape(A * F, len(self.test_o), H))
        self.y_val_all = h.aggregate(self.y_val.reshape(A * F, len(self.val_o), H))
        print(f"[seed {seed}] data+hierarchy+aggregate base ready in {time.time() - t0:.1f}s | train windows {len(self.train_idx):,}", flush=True)


def lifecycle_features(d, mm: Dict) -> Dict:
    """LA features with two retrieval indices: aligned SigLIP (image + text) and MTFT-Picnic's separate-encoder scalars."""
    from mtft_plus.lifecycle import build_lifecycle, with_retrieval

    base = build_lifecycle(d, W=getattr(d.cfg, "lifecycle_window", None))
    aligned = np.concatenate([d.emb_aligned_image, d.emb_aligned_text], 1)
    keys = d.attributes.get("retrieval_keys")  # M5: static keys (no image / text); both retrieval indices use them
    if keys is not None:
        return {"aligned": with_retrieval(base, keys, keys, source="static keys"), "picnic": with_retrieval(base, keys, keys, source="static keys")}
    return {
        "aligned": with_retrieval(base, aligned, aligned, source="aligned"),
        "picnic": with_retrieval(base, mm["picnic"]["scalar"], mm["picnic"]["scalar"], source="picnic"),
    }


def _ctx_lifecycle(self) -> Dict:
    if getattr(self, "_lifecycle", None) is None:
        t0 = time.time()
        self._lifecycle = lifecycle_features(self.data, self.mm)
        print(f"[seed {self.seed}] lifecycle-aligned features ready in {time.time() - t0:.1f}s", flush=True)
    return self._lifecycle


SeedContext.lifecycle = _ctx_lifecycle


def save_forecasts(path: Path, q: np.ndarray, mask: np.ndarray) -> None:
    """Live test cells only: row (series), origin and step indices plus the quantiles (float32)."""
    r, o, t = np.nonzero(mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, series=r.astype(np.int32), origin=o.astype(np.int16), step=t.astype(np.int16), q=q[r, o, t].astype(np.float32))


def per_article_national(ctx: SeedContext, bu_q: np.ndarray) -> Dict[str, list]:
    """Per-article sums at the article-national level of the bottom-up forecasts (product totals across shops)."""
    sl = ctx.hier.level_slices()[2][1]
    y, m, q = ctx.y_test_all[sl], ctx.mask_test_all[sl], bu_q[sl]
    med = QUANTILES.index(0.5)
    u = y[..., None] - q
    qa = np.asarray(QUANTILES)
    return {
        "an_abs_err": (np.abs(y - q[..., med]) * m).reshape(ctx.A, -1).sum(1).round(4).tolist(),
        "an_pinball": (np.maximum(qa * u, (qa - 1) * u).sum(-1) * m).reshape(ctx.A, -1).sum(1).round(4).tolist(),
        "an_abs_y": (np.abs(y) * m).reshape(ctx.A, -1).sum(1).round(4).tolist(),
    }


def per_article_sums(ctx: SeedContext, q: np.ndarray) -> Dict[str, list]:
    """Per-article sums for cluster bootstrap: |e| of the median, sum of pinball over q, |y|."""
    A, F = ctx.A, ctx.F
    y = ctx.y_test.reshape(A * F, -1)
    m = ctx.mask_test_b.reshape(A * F, -1)
    qq = q.reshape(A * F, -1, len(QUANTILES))
    med = QUANTILES.index(0.5)
    ae = np.abs(y - qq[..., med]) * m
    u = y[..., None] - qq
    qa = np.asarray(QUANTILES)
    pin = np.maximum(qa * u, (qa - 1) * u).sum(-1) * m
    return {
        "abs_err": ae.reshape(A, -1).sum(1).round(4).tolist(),
        "pinball": pin.reshape(A, -1).sum(1).round(4).tolist(),
        "abs_y": (np.abs(y) * m).reshape(A, -1).sum(1).round(4).tolist(),
        "cold": ctx.data.cold_start.tolist(),
    }


def lifecycle_for(lf_by_retrieval: Dict, spec: Dict):
    """(LifecycleFeatures or None, la_pool, la_tokens) for a model spec."""
    la = spec.get("la")
    if not la:
        return None, False, False
    return lf_by_retrieval[spec.get("la_retrieval", "aligned")], True, la == "attn"


def model_config(data, spec: Dict, args, has_analogues: bool, mm_scalar_dim: int, L: int, H: int, la_emb_dim: int = 0) -> Dict:
    """MTFTConfig kwargs for a model spec (shared by the calendar benchmark and the official VISUELLE protocol)."""
    sib = int(spec.get("siblings", False))
    la = spec.get("la")
    branches = spec.get("la_branches", "both")
    return dict(
        static_categorical_cardinalities=[data.n_fc, int(data.category.max()) + 1, 3],
        n_static_reals=len(STATIC_REALS) + int(has_analogues) + sib,
        n_observed_reals=1 + 2 * int(has_analogues) + sib,
        n_known_reals=len(KNOWN_REALS) + (4 if la else 0),
        lifecycle_attention=la == "attn",
        la_siblings=branches in ("both", "siblings"),
        la_analogues=branches in ("both", "analogues"),
        n_shops=data.n_fc,
        la_emb_dim=la_emb_dim,
        age_embedding=bool(spec.get("age_emb", False)),
        la_gate=bool(spec.get("la_gate", False)),
        known_categorical_cardinalities=[7],
        encoder_length=L,
        horizon=H,
        d_model=args.d_model,
        n_heads=args.heads,
        dropout=0.1,
        quantiles=QUANTILES,
        mm_embed_dim=data.cfg.embed_dim,
        mm_scalar_dim=mm_scalar_dim,
        mm_latents=8,
        fusion=spec["fusion"],
        align_weight=spec.get("align_weight", 0.0),
        mm_feature_dropout=spec.get("mm_feature_dropout", 0.0),
        mm_modality_dropout=spec.get("mm_modality_dropout", 0.0),
    )


def train_module(module: MTFTLightningModule, train_ds, val_ds, seed: int, args):
    """Equal-budget training: epochs x batches_per_epoch sampled batches, best validation epoch restored."""
    sampler = RandomSampler(train_ds, replacement=True, num_samples=args.batches_per_epoch * args.batch_size, generator=torch.Generator().manual_seed(seed))
    pin = args.device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=identity_collate, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=1024, collate_fn=identity_collate, pin_memory=pin)
    best = BestState()
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=args.devices, precision=args.precision,
        gradient_clip_val=1.0, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, callbacks=[best],
    )
    t0 = time.time()
    trainer.fit(module, train_dl, val_dl)
    module.load_state_dict(best.state)
    return best, time.time() - t0


def run_model(ctx: SeedContext, name: str, spec: Dict, args, out_dir: Path) -> Dict:
    pl.seed_everything(ctx.seed, verbose=False)
    A, F, H = ctx.A, ctx.F, ctx.H
    h = ctx.hier
    mm = ctx.mm[spec["mm"]]
    an = ctx.analogues[spec["analogues"]] if spec.get("analogues") else None
    lf, la_pool, la_tokens = lifecycle_for(ctx.lifecycle(), spec) if spec.get("la") else (None, False, False)
    cfg = model_config(ctx.data, spec, args, an is not None, ctx.mm["picnic"]["scalar"].shape[1], ctx.L, H, lf.emb.shape[1] if lf is not None else 0)
    module = MTFTLightningModule(cfg, learning_rate=args.lr)
    n_params = sum(p.numel() for p in module.parameters())
    ds = lambda idx: DemandWindowDataset(ctx.data, idx, mm, an, siblings=spec.get("siblings", False), lifecycle=lf, la_pool=la_pool, la_tokens=la_tokens)  # noqa: E731
    pin = args.device.type == "cuda"
    best, train_s = train_module(module, ds(ctx.train_idx), ds(ctx.val_mon_idx), ctx.seed, args)

    ours = spec["fusion"] == "cross_attention"
    ours_quota = name.startswith("MTFT+")
    q_val, _, _ = predict(module, ds(ctx.val_all_idx), args.device)
    q_test, attr, promo = predict(module, ds(ctx.test_all_idx), args.device, attribution=ours)
    module.cpu()
    if pin:
        torch.cuda.empty_cache()
    nv, nt = len(ctx.val_o), len(ctx.test_o)
    if ctx.sparse_pred:  # scatter the predicted windows into the full grid; non-live cells are forecast as 0
        def full(q, idx, origins, mask):
            out = np.zeros((A * F, len(origins), H, len(QUANTILES)), dtype=np.float32)
            out[idx[:, 0] * F + idx[:, 1], np.searchsorted(origins, idx[:, 2])] = q
            return out * mask[..., None]

        q_val = full(q_val, ctx.val_all_idx, ctx.val_o, ctx.mask_val_b)
        q_test = full(q_test, ctx.test_all_idx, ctx.test_o, ctx.mask_test_b)
    q_val = q_val.reshape(A * F, nv, H, -1)
    q_test = q_test.reshape(A * F, nt, H, -1)
    yt = ctx.y_test.reshape(A * F, nt, H)
    if args.panel:
        save_forecasts(out_dir / "forecasts" / f"{ctx.seed}__{slug(name)}.npz", q_test, ctx.mask_test_b)
        if getattr(args, "save_val_forecasts", False):
            save_forecasts(out_dir / "forecasts" / f"{ctx.seed}__{slug(name)}__val.npz", q_val, ctx.mask_val_b)

    bottom = []
    subsets = [("all", np.ones(A * F, bool)), ("warm", ~ctx.cold_b), ("cold-start", ctx.cold_b)]
    if ctx.new_in_shop_b is not None:
        subsets.append(("new-in-shop", ctx.new_in_shop_b))
    if getattr(ctx, "young_b", None) is not None:
        subsets.append(("young", ctx.young_b))
    for grp, sel in subsets:
        bottom.append({"subset": grp, **evaluate(yt[sel], q_test[sel], QUANTILES, ctx.scale_b[sel], ctx.mask_test_b[sel])})

    hier_rows, rec, coh, mint_cal_q, bu_q = reconcile(ctx, q_val, q_test)
    res = {
        "seed": ctx.seed,
        "model": name,
        "spec": spec,
        "params": n_params,
        "train_seconds": round(train_s, 1),
        "val_curve": best.history,
        "best_val": best.best,
        "bottom": bottom,
        "hierarchy": hier_rows,
        "mint_lambda": rec.lam,
        "coherence_error": coh,
        "per_article": {**per_article_sums(ctx, q_test), **per_article_national(ctx, bu_q)},
    }
    return finish_run(ctx, name, res, attr, promo, ours_quota, args, out_dir, mint_cal_q, bottom, best, train_s, rec, coh)


def reconcile(ctx: SeedContext, q_val: np.ndarray, q_test: np.ndarray):
    """Base forecasts for every node (ridge for aggregates with history, bottom-up otherwise), MinT-shrink on
    validation residuals, per-level conformal calibration; evaluation of base / bottom-up / MinT / MinT + conformal."""
    h, nt, H = ctx.hier, len(ctx.test_o), ctx.H
    mean = lambda q: swanson_mean(q, QUANTILES)  # noqa: E731
    base_val = np.concatenate([np.where(ctx.agg_val_ok[:, :, None, None], ctx.agg_val_q, h.aggregate(q_val)[: h.n_agg]), q_val], 0)
    base_test = np.concatenate([np.where(ctx.agg_test_ok[:, :, None, None], ctx.agg_test_q, h.aggregate(q_test)[: h.n_agg]), q_test], 0)
    base_val_pt, base_test_pt = mean(base_val), mean(base_test)
    R = torch.as_tensor((ctx.y_val_all - base_val_pt).reshape(h.n_total, -1).T)
    level = ctx.y_val_all.reshape(h.n_total, -1).mean(1)
    var = (R.numpy() ** 2).mean(0)
    okv = (level > 0.5) & (var > 0)
    kappa = float(np.median(var[okv] / level[okv]))
    # prior variance for nodes without validation evidence: cold-start series, aggregates without ridge history, and
    # (real panels) any node live in the test window but not in the validation window. On the synthetic benchmark the
    # third set adds nothing, so v1-v3 are unchanged.
    live_val_n = ctx.mask_val_all.any((1, 2))
    live_test_n = ctx.mask_test_all.any((1, 2))
    no_hist = np.concatenate([~ctx.agg_test_ok.all(1), ctx.cold_b]) | (live_test_n & ~live_val_n)
    prior = np.full(h.n_total, np.nan)
    prior[no_hist] = kappa * np.maximum(base_test_pt[no_hist].mean((1, 2)), 1.0)
    prior_t = torch.as_tensor(prior)
    rec = MinTReconciler(ctx.Afull).fit(R, prior_var=prior_t)  # MinT-shrink (pre-specified, v1-v3)
    rec_wls = MinTReconciler(ctx.Afull).fit(R, prior_var=prior_t, lam=1.0)  # MinT-WLS (diagonal), added for VISUELLE
    K = nt * H

    def mint(r: MinTReconciler, bq: np.ndarray, bpt: np.ndarray):
        qq, pt = r.reconcile_quantiles(torch.as_tensor(bq.reshape(h.n_total, K, -1)), torch.as_tensor(bpt.reshape(h.n_total, K)))
        return qq.numpy().reshape(bq.shape), pt.numpy().reshape(bpt.shape)

    base_test_cal = base_test.copy()
    for _, sl in h.level_slices():  # per-level split-conformal on validation origins (median untouched)
        cal = LogConformalCalibrator(QUANTILES).fit(ctx.y_val_all[sl], base_val[sl], ctx.mask_val_all[sl])
        base_test_cal[sl] = cal.transform(base_test[sl])
    mint_q, mint_pt = mint(rec, base_test, base_test_pt)
    mint_cal_q, _ = mint(rec, base_test_cal, base_test_pt)
    wls_q, _ = mint(rec_wls, base_test, base_test_pt)
    wls_cal_q, _ = mint(rec_wls, base_test_cal, base_test_pt)
    bu_pt = h.aggregate(mean(q_test))
    bu_q = np.sort(np.clip(base_test + (bu_pt - base_test_pt)[..., None], 0, None), -1)
    bu_q[h.n_agg :] = base_test[h.n_agg :]  # bottom level: bottom-up *is* the base forecast (no float-noise shift)
    coh = float(np.abs(mint_pt[: h.n_agg] - h.aggregate(mint_pt[h.n_agg :])[: h.n_agg]).max())
    hier_rows = []
    for method, qa in (("base", base_test), ("bottom-up", bu_q), ("MinT", mint_q), ("MinT + conformal", mint_cal_q),
                       ("MinT-WLS", wls_q), ("MinT-WLS + conformal", wls_cal_q)):
        for lvl, sl in h.level_slices():
            hier_rows.append({"method": method, "level": lvl, **evaluate(ctx.y_test_all[sl], qa[sl], QUANTILES, ctx.scale_all[sl], ctx.mask_test_all[sl])})
    hier_rows.append({"method": "_val_choice", "level": "national+fulfilment_centre", **validation_choice(ctx, base_val_pt, prior_t)})
    return hier_rows, rec, coh, mint_cal_q, bu_q


def validation_choice(ctx: SeedContext, base_val_pt: np.ndarray, prior: torch.Tensor) -> Dict:
    """Which MinT variant validation alone would pick: fit on the first half of the validation origins, score the
    reconciled means on the second half (WAPE summed over the national and fulfilment-centre levels)."""
    h, H = ctx.hier, ctx.H
    nv = len(ctx.val_o)
    resid = ctx.y_val_all - base_val_pt
    # halves of the validation origins; with a single validation origin (M5), halves of its horizon instead
    first = (slice(None), slice(0, nv // 2)) if nv > 1 else (slice(None), slice(None), slice(0, H // 2))
    second = (slice(None), slice(nv // 2, None)) if nv > 1 else (slice(None), slice(None), slice(H // 2, None))
    R1 = torch.as_tensor(resid[first].reshape(h.n_total, -1).T)
    y2 = ctx.y_val_all[second].reshape(h.n_total, -1)
    m2 = ctx.mask_val_all[second].reshape(h.n_total, -1)
    p2 = torch.as_tensor(base_val_pt[second].reshape(h.n_total, -1))
    sl = slice(0, 1 + ctx.F)
    out = {}
    for tag, lam in (("shrink", None), ("wls", 1.0)):
        pt = MinTReconciler(ctx.Afull).fit(R1, prior_var=prior, lam=lam).reconcile(p2).numpy()
        out[tag] = float(np.abs(y2[sl] - pt[sl])[m2[sl]].sum() / max(y2[sl][m2[sl]].sum(), 1e-9))
    out["choice"] = min(("shrink", "wls"), key=lambda k: out[k])
    return out


def finish_run(ctx, name, res, attr, promo, ours_quota, args, out_dir, mint_cal_q, bottom, best, train_s, rec, coh) -> Dict:
    A, H, h = ctx.A, ctx.H, ctx.hier
    if attr is not None:
        p = promo.reshape(-1) > 0.5
        a = attr.reshape(-1, 3)
        res["modality_attribution"] = {
            "promo": dict(zip(["image", "text", "fused"], a[p].mean(0).round(4).tolist())),
            "non_promo": dict(zip(["image", "text", "fused"], a[~p].mean(0).round(4).tolist())),
        }
    if ours_quota and args.write_quotas and not args.panel:  # shelf-life heuristic: synthetic demo only
        sl = h.level_slices()[2][1]
        q_nat = mint_cal_q[sl][:, 0]
        med = QUANTILES.index(0.5)
        service = np.array([newsvendor_quantile(1.0, 0.8 if s < 10 else 0.1) for s in ctx.data.shelf_life])
        hi = np.where(service[:, None] >= 0.5, q_nat[..., -1] - q_nat[..., med], q_nat[..., med] - q_nat[..., 0])
        order = np.maximum(q_nat[..., med] + (service[:, None] - 0.5) / 0.4 * hi, 0)
        rows = [
            (f"A{i:04d}", str(ctx.data.dates[ctx.test_o[0] + t].date()), round(float(ctx.data.shelf_life[i]), 1), *np.round(q_nat[i, t], 2), round(float(service[i]), 3), round(float(order[i, t]), 2))
            for i in range(A)
            for t in range(H)
        ]
        pd.DataFrame(rows, columns=["article", "date", "shelf_life_days", "q10", "q50", "q90", "service_level", "order_quantity"]).to_csv(
            out_dir / f"national_order_quotas_seed{ctx.seed}.csv", index=False
        )
    b = bottom[0]
    print(
        f"[seed {ctx.seed}] {name:32s} train {train_s:4.0f}s val {best.best:.4f} | WAPE {b['WAPE']:.4f} MASE {b['MASE']:.4f} wQL {b['wQL']:.4f} "
        f"| cold WAPE {bottom[2]['WAPE']:.4f} | MinT λ {rec.lam:.3f} coh {coh:.1e}",
        flush=True,
    )
    return res


def add_training_args(ap: argparse.ArgumentParser) -> None:
    """Training-budget, architecture and device flags shared with examples/visuelle2_official.py."""
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--k", type=int, default=10, help="analogues per article")
    ap.add_argument("--accelerator", default="auto", help="auto (GPU if available) | gpu | cpu; v1-v3 results used cpu")
    ap.add_argument("--devices", type=int, default=1)
    # 32-true by default: on an RTX 2060 (d_model 32, synthetic seed 7, full budget) 16-mixed matched fp32 accuracy
    # within noise but trained ~20 % slower, and at the --quick budget GradScaler's skipped steps cost ~0.05 WAPE.
    ap.add_argument("--precision", default="32-true", help="Lightning precision: 32-true | 16-mixed (Turing) | bf16-mixed (Ampere+)")


def setup_device(args) -> None:
    args.device = resolve_device(args.accelerator)
    args.accelerator = "gpu" if args.device.type == "cuda" else "cpu"
    print(f"device {args.device} ({torch.cuda.get_device_name(args.device) if args.device.type == 'cuda' else 'cpu'}) | precision {args.precision}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="7,11,23")
    ap.add_argument("--articles", type=int, default=1000)
    ap.add_argument("--fcs", type=int, default=5)
    ap.add_argument("--days", type=int, default=540)
    add_training_args(ap)
    ap.add_argument("--models", type=str, default="", help="comma-separated subset of model names (default: all)")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--no-quotas", dest="write_quotas", action="store_false", help="skip the (heuristic, demo-only) order-quota CSVs")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--panel", type=str, default="", help="pickled panel from examples/prepare_panel.py (real data)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save-val-forecasts", action="store_true", help="also save validation forecasts (live cells), for offline diagnostics")
    args = ap.parse_args()
    setup_device(args)
    if args.quick:
        args.articles, args.days, args.epochs, args.batches_per_epoch = 150, 420, 2, 20
    out_dir = Path(args.out)
    part_dir = out_dir / "partials"
    part_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    wanted = set(x.strip() for x in args.models.split(",") if x.strip())
    t0 = time.time()
    for si, seed in enumerate(seeds):
        plan = dict(MAIN_MODELS)
        plan.update(V3_MODELS)
        if si == 0 and not args.no_ablations:
            plan.update(ABLATIONS)
        if wanted:  # any registered model (incl. MTFT+ v4 and its ablations) can be selected by name, in the given order
            unknown = wanted - set(ALL_MODELS)
            if unknown:
                raise SystemExit(f"unknown model(s): {sorted(unknown)}")
            plan = {k: ALL_MODELS[k] for k in [x.strip() for x in args.models.split(",") if x.strip()]}
        todo = {k: v for k, v in plan.items() if not (part_dir / f"{seed}__{slug(k)}.json").exists()}
        if not todo:
            print(f"[seed {seed}] all models done (resume)", flush=True)
            continue
        ctx = SeedContext(seed, args)
        for name, spec in todo.items():
            res = run_model(ctx, name, spec, args, out_dir)
            res["args"] = {k: (str(v) if k == "device" else v) for k, v in vars(args).items()}
            if args.device.type == "cuda":
                res["gpu"] = torch.cuda.get_device_name(args.device)
            (part_dir / f"{seed}__{slug(name)}.json").write_text(json.dumps(res))
        del ctx
    print(f"done in {time.time() - t0:.0f}s - run `python report.py --out {args.out}`", flush=True)


if __name__ == "__main__":
    main()
