import numpy as np
import pytest
import torch

from mtft_plus import (
    ArticleRecord,
    Hierarchy,
    MinTReconciler,
    MTFTConfig,
    MultimodalFeatureExtractor,
    MultimodalTemporalFusionTransformer,
    QuantileLoss,
    shrinkage_intensity,
    siglip_alignment_loss,
)
from mtft_plus.layers import MonotoneQuantileHead

torch.manual_seed(0)


def test_quantile_loss_matches_definition():
    qs = (0.1, 0.5, 0.9)
    loss = QuantileLoss(qs)
    y = torch.tensor([[3.0, 0.0]])
    p = torch.tensor([[[1.0, 2.0, 4.0], [0.5, 0.5, 0.5]]])
    ref = 0.0
    for i in range(2):
        for j, q in enumerate(qs):
            u = y[0, i] - p[0, i, j]
            ref += max(q * u, (q - 1) * u)
    assert torch.isclose(loss(p, y), (ref / 2).clone().detach())
    mask = torch.tensor([[True, False]])
    ref0 = sum(max(q * (3 - v), (q - 1) * (3 - v)) for q, v in zip(qs, (1.0, 2.0, 4.0)))
    assert torch.isclose(loss(p, y, mask=mask), torch.tensor(float(ref0)))


def test_monotone_head_never_crosses():
    head = MonotoneQuantileHead(8, (0.05, 0.1, 0.5, 0.9, 0.95))
    out = head(torch.randn(64, 14, 8) * 10)
    assert (out[..., 1:] >= out[..., :-1]).all()


def _brute_lambda(X):
    T, n = X.shape
    W = torch.einsum("ki,kj->kij", X, X)
    wbar = W.mean(0)
    v = ((W - wbar) ** 2).sum(0) / (T * (T - 1))
    corr = wbar
    off = ~torch.eye(n, dtype=torch.bool)
    return float(torch.clamp(v[off].sum() / (corr[off] ** 2).sum(), 0, 1))


def test_shrinkage_intensity_matches_bruteforce():
    R = torch.randn(30, 12, dtype=torch.float64) @ torch.randn(12, 12, dtype=torch.float64)
    X = R / R.pow(2).mean(0).sqrt()
    assert abs(shrinkage_intensity(X) - _brute_lambda(X)) < 1e-10


def test_mint_coherent_and_matches_dense_formula():
    h = Hierarchy(n_articles=4, n_fc=3)
    A = h.aggregation_matrix(torch.float64)
    n = h.n_total
    b_true = torch.rand(h.n_bottom, 1, dtype=torch.float64) * 10
    R = torch.randn(40, n, dtype=torch.float64)
    y_hat = torch.cat([A @ b_true, b_true], 0) + torch.randn(n, 1, dtype=torch.float64)
    rec = MinTReconciler(A).fit(R)
    y_t = rec.reconcile(y_hat, nonnegative=False)
    assert torch.allclose(y_t[: h.n_agg], A @ y_t[h.n_agg :], atol=1e-8)
    S = torch.cat([A, torch.eye(h.n_bottom, dtype=torch.float64)], 0)
    Sig = R.T @ R / R.shape[0]
    D = torch.diag(Sig.diagonal())
    W = rec.lam * D + (1 - rec.lam) * Sig
    Wi = torch.linalg.inv(W)
    G = torch.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
    assert torch.allclose(y_t, S @ G @ y_hat, atol=1e-6)


def test_hierarchy_aggregate_matches_matrix():
    h = Hierarchy(n_articles=5, n_fc=2)
    b = np.random.rand(h.n_bottom, 3)
    full = h.aggregate(b)
    A = h.aggregation_matrix(torch.float64).numpy()
    assert np.allclose(full[: h.n_agg], A @ b)
    assert np.allclose(h.aggregate_upper(b), full[: h.n_agg])
    assert torch.equal(h.aggregation_matrix(torch.float64, sparse=True).to_dense(), torch.as_tensor(A))


def test_sparse_mint_matches_dense():
    h = Hierarchy(n_articles=7, n_fc=4)
    n = h.n_total
    R = torch.randn(25, n, dtype=torch.float64) @ torch.randn(n, n, dtype=torch.float64) * 0.1
    R[:, h.n_agg + 3] = 0.0  # a bottom series with no history ...
    prior = torch.full((n,), float("nan"), dtype=torch.float64)
    prior[h.n_agg + 3] = 2.5  # ... gets a prior variance
    y_hat = torch.rand(n, 6, dtype=torch.float64) * 10
    q_hat = torch.sort(torch.rand(n, 6, 3, dtype=torch.float64) * 10, -1).values
    dense = MinTReconciler(h.aggregation_matrix(torch.float64)).fit(R, prior_var=prior)
    sparse = MinTReconciler(h.aggregation_matrix(torch.float64, sparse=True)).fit(R, prior_var=prior)
    assert sparse.sparse and not dense.sparse and abs(sparse.lam - dense.lam) < 1e-12
    for nonneg in (False, True):
        assert torch.allclose(sparse.reconcile(y_hat, nonneg), dense.reconcile(y_hat, nonneg), atol=1e-9)
    for a, b in zip(sparse.reconcile_quantiles(q_hat, y_hat), dense.reconcile_quantiles(q_hat, y_hat)):
        assert torch.allclose(a, b, atol=1e-9)
    b = torch.rand(h.n_bottom, 2, dtype=torch.float64)
    assert torch.allclose(sparse.bottom_up(b), dense.bottom_up(b))


def test_ridge_and_mase_season_settings():
    from mtft_plus.metrics import seasonal_naive_scale
    from mtft_plus.reconciliation import SeasonalRidgeForecaster

    rng = np.random.default_rng(0)
    Y = rng.poisson(5.0, (3, 40)).astype(float)
    Ylog, cov, origins = np.log1p(Y), rng.normal(size=(3, 40, 2)), np.array([10, 20])
    weekly = SeasonalRidgeForecaster(4, (0.1, 0.5, 0.9), season=1, level_window=4)
    X = weekly._design(Ylog, cov, np.zeros(40, int), origins)
    assert X.shape[-1] == 3 + 2  # intercept, anchor, level, covariates; no phase dummies
    assert np.allclose(X[:, :, :, 1], Ylog[:, origins - 1][:, :, None])  # last-value anchor for every step
    assert np.allclose(X[:, :, 0, 2], np.stack([Ylog[:, o - 4 : o].mean(1) for o in origins], 1))
    daily = SeasonalRidgeForecaster(9, (0.1, 0.5, 0.9))
    Xd = daily._design(Ylog, cov, np.arange(40) % 7, np.array([30]))
    assert Xd.shape[-1] == 3 + 6 + 2
    h = np.arange(9)
    assert np.allclose(Xd[:, 0, :, 1], Ylog[:, 30 + h - 7 * (h // 7 + 1)])  # same-weekday anchor
    with pytest.raises(ValueError):
        daily._design(Ylog, cov, np.arange(40) % 7, np.array([20]))  # 20 < level window 28
    ok = np.ones_like(Y, bool)
    assert np.allclose(seasonal_naive_scale(Y, ok, season=1), np.abs(np.diff(Y, axis=1)).mean(1))


def test_load_panel_weekly_grid():
    import pandas as pd
    from mtft_plus.panel import load_panel

    days = pd.date_range("2024-01-03", periods=7 * 30, freq="D")  # starts on a Wednesday
    sales = pd.DataFrame({"article_id": "a", "location_id": "s1", "date": days, "units": 1.0, "discount": 0.2})
    sales = pd.concat([sales, sales.assign(article_id="b", location_id="s2", units=2.0, discount=0.0)])
    catalog = pd.DataFrame({"article_id": ["a", "b"], "category": ["x", "y"], "launch_date": ["2024-01-03", "2024-02-14"]})
    locations = pd.DataFrame({"location_id": ["s1", "s2"], "country": ["NL", "NL"]})
    weather = pd.DataFrame({"location_id": "s1", "date": days, "temperature": np.arange(len(days), dtype=float), "precipitation": 0.0})
    p = load_panel(sales, catalog, locations, weather=weather, encoder_length=8, horizon=2, freq="W-MON")
    assert p.dates[0] == pd.Timestamp("2024-01-01") and (p.dates.dayofweek == 0).all()
    assert p.cfg.season == 1 and p.cfg.level_window == 4 and p.cfg.min_history == 5 and p.cfg.freq == "W-MON"
    assert p.y[0, 0, 0] == 5 and p.y[0, 0, 1] == 7 and p.y.sum() == 7 * 30 * 3  # partial first week kept, weeks summed
    assert p.launch.tolist() == [0, 6]  # Wed 2024-02-14 lies in the week of Mon 2024-02-12
    assert np.isclose(p.discount[0, 0, 1], 0.2)  # rate columns are averaged, not summed
    weekly_mean = weather.groupby(days.to_period("W-SUN"))["temperature"].mean().values  # Monday-start weeks
    assert np.allclose(p.temp_z[0], (weekly_mean - weekly_mean.mean()) / (weekly_mean.std() + 1e-8), atol=1e-4)
    assert (p.temp_z[1] == 0).all()  # s2 has no weather rows: missing, not 0 degrees


def test_series_windows_live_mask_and_live_siblings():
    import pandas as pd
    from mtft_plus.data import DemandWindowDataset, build_index
    from mtft_plus.panel import load_panel

    wk = pd.date_range("2024-01-01", periods=30, freq="W-MON")
    # article a: shop s1 from week 2 for 4 weeks, s2 from week 5 for 4 weeks, never in s3; article b: s3 from week 0
    series = pd.DataFrame({"article_id": ["a", "a", "b"], "location_id": ["s1", "s2", "s3"],
                           "launch_date": [wk[2], wk[5], wk[0]], "end_date": [wk[5], wk[8], wk[29]]})
    rows = [("a", "s1", t, 2.0 + t) for t in range(2, 8)] + [("a", "s2", t, 10.0) for t in range(5, 9)] + [("b", "s3", t, 1.0) for t in range(30)]
    sales = pd.DataFrame([(a, l, wk[t], u) for a, l, t, u in rows], columns=["article_id", "location_id", "date", "units"])
    catalog = pd.DataFrame({"article_id": ["a", "b"], "category": ["x", "x"]})
    locations = pd.DataFrame({"location_id": ["s1", "s2", "s3"], "country": ["IT"] * 3})
    p = load_panel(sales, catalog, locations, encoder_length=4, horizon=2, n_val_origins=2, n_test_origins=2, freq="W-MON", series=series)
    assert p.launch_loc.tolist() == [[2, 5, 30], [30, 30, 0]] and p.launch.tolist() == [2, 0]
    assert p.live[0, 0].nonzero()[0].tolist() == [2, 3, 4, 5] and not p.live[0, 2].any()
    assert p.y[0, 0, 6] == 0 and p.y[0, 0, 5] == 7  # sales after the observation window are dropped
    idx = build_index(p, np.array([4, 6, 8]))
    assert {tuple(r) for r in idx} == {(0, 0, 4), (0, 1, 4), (0, 1, 6), (0, 1, 8), (1, 2, 4), (1, 2, 6), (1, 2, 8)}
    mm = {"image": p.emb_aligned_image, "text": p.emb_aligned_text, "mask": np.ones((2, 2), bool)}
    b = DemandWindowDataset(p, np.array([[0, 1, 8]]), mm, siblings=True).__getitems__([0])  # a @ s2, origin 8
    sib = b["enc_observed"][0, :, -1].numpy()  # encoder steps 4..7: only s1 is a live sibling, at 4 and 5
    assert np.allclose(sib, [np.log1p(6.0), np.log1p(7.0), 0.0, 0.0], atol=1e-6)
    assert b["target_mask"][0].tolist() == [True, False]  # step 8 live, step 9 after the window
    assert b["enc_known_real"][0, :, 8].tolist() == [0.0, 1.0, 1.0, 1.0]  # 'launched' is per series (s2 from 5)


def test_closed_form_ses_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.tsa.api")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from visuelle2_official import ses2

    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.poisson(3.0, 2).astype(float) + rng.random(2)
        ref = sm.SimpleExpSmoothing(x, initialization_method="estimated").fit(smoothing_level=0.3, optimized=True).forecast(1)[0]
        assert abs(ses2(x[0], x[1]) - ref) < 1e-4 * max(1.0, abs(ref))


def _staggered_panel():
    import pandas as pd
    from mtft_plus.panel import load_panel

    wk = pd.date_range("2024-01-01", periods=40, freq="W-MON")
    # product a: s1 from week 2, s2 from week 6, s3 from week 12 (4-week windows); product b (analogue): s1 from week 0
    launches = {("a", "s1"): 2, ("a", "s2"): 6, ("a", "s3"): 12, ("b", "s1"): 0, ("b", "s2"): 1}
    series = pd.DataFrame([(p, s, wk[t], wk[t + 3]) for (p, s), t in launches.items()], columns=["article_id", "location_id", "launch_date", "end_date"])
    rows = [(p, s, wk[t + k], 10.0 * (1 + k) + (5 if s == "s2" else 0)) for (p, s), t in launches.items() for k in range(4)]
    sales = pd.DataFrame(rows, columns=["article_id", "location_id", "date", "units"])
    sales = pd.concat([sales, pd.DataFrame([("a", "s1", wk[39], 0.0)], columns=sales.columns)])  # extend the grid
    catalog = pd.DataFrame({"article_id": ["a", "b"], "category": ["x", "x"]})
    locations = pd.DataFrame({"location_id": ["s1", "s2", "s3"], "country": ["IT"] * 3})
    return load_panel(sales, catalog, locations, encoder_length=2, horizon=4, n_val_origins=2, n_test_origins=2, freq="W-MON", series=series)


def test_lifecycle_alignment_is_exact_and_causal():
    from mtft_plus.lifecycle import build_lifecycle, lifecycle_batch, with_retrieval

    p = _staggered_panel()
    lf = with_retrieval(build_lifecycle(p), np.eye(2), np.eye(2))
    assert lf.W == 4
    # target a @ s3 (launch 12), origin 12, L=2, H=4: steps t = 10..15, target ages -2..3
    out = lifecycle_batch(lf, p, np.array([0]), np.array([2]), np.array([12]), 2, 4, tokens=True)
    slot_shops = out["la_sib_shop"][0][:2].tolist()
    assert slot_shops == [0, 1]  # earliest-launched first: s1 (2), s2 (6)
    # s1 at ages 0..3 = weeks 2..5 (known before 12): 10, 20, 30, 40; s2 at ages 0..3 = weeks 6..9: 15, 25, 35, 45
    assert np.allclose(np.expm1(out["la_sib_val"][0, 0, 2:]), [10, 20, 30, 40]) and not out["la_sib_mask"][0, 0, :2].any()
    assert np.allclose(np.expm1(out["la_sib_val"][0, 1, 2:]), [15, 25, 35, 45])
    assert np.allclose(out["la_sib_mean"][0, 2:], np.log1p([[10, 20, 30, 40], [15, 25, 35, 45]]).mean(0), atol=1e-5)
    # analogue b (launched weeks 0 / 1, windows end by week 4): mean log1p per-shop curve at each age
    ref = np.log1p(np.array([[10, 20, 30, 40], [15, 25, 35, 45]])).mean(0)
    assert np.allclose(out["la_ana_val"][0, 0, 2:], ref, atol=1e-5) and np.isclose(out["la_ana_cov"][0, 3], 1.0)
    # causality: at origin 8 only s1 ages 0..3 (weeks 2..5) and s2 ages 0..1 (weeks 6, 7) are known
    o8 = lifecycle_batch(lf, p, np.array([0]), np.array([2]), np.array([8]), 2, 4, tokens=True)
    # target a @ s3 launches at 12, so all of its steps at origin 8 have negative age: nothing is aligned
    assert not o8["la_sib_mask"].any()
    p.y[:, :, 8:] = 1e6  # corrupt everything from origin 8 on: features at origin 8 must not change
    lf2 = with_retrieval(build_lifecycle(p), np.eye(2), np.eye(2))
    q1 = lifecycle_batch(lf, p, np.array([0, 0]), np.array([1, 2]), np.array([8, 8]), 2, 4, tokens=True)
    q2 = lifecycle_batch(lf2, p, np.array([0, 0]), np.array([1, 2]), np.array([8, 8]), 2, 4, tokens=True)
    for key in ("la_sib_val", "la_sib_mask", "la_ana_val", "la_ana_mask", "la_sib_mean", "la_ana_mean"):
        assert np.array_equal(q1[key], q2[key]), key


def test_lifecycle_attention_model_forward_backward():
    from mtft_plus.data import DemandWindowDataset, build_index
    from mtft_plus.lifecycle import build_lifecycle, with_retrieval

    p = _staggered_panel()
    emb = np.random.default_rng(0).normal(size=(2, 6)).astype(np.float32)
    lf = with_retrieval(build_lifecycle(p), emb, emb)
    mm = {"image": np.zeros((2, 8), np.float32), "text": np.zeros((2, 8), np.float32), "mask": np.ones((2, 2), bool)}
    idx = build_index(p, np.arange(2, 30))
    ds = DemandWindowDataset(p, idx, mm, siblings=True, lifecycle=lf, la_pool=True, la_tokens=True)
    batch = ds.__getitems__(list(range(len(idx))))
    assert batch["enc_known_real"].shape[-1] == 10 + 4 and batch["la_sib_val"].shape == (len(idx), 32, 6)
    for branches, gate in (((True, True), False), ((True, False), False), ((False, True), False), ((True, True), True)):
        cfg = MTFTConfig(
            static_categorical_cardinalities=[3, 1, 3], n_static_reals=3, n_observed_reals=2, n_known_reals=14,
            known_categorical_cardinalities=[7], encoder_length=2, horizon=4, d_model=16, n_heads=4, fusion="static_pooled",
            mm_embed_dim=8, lifecycle_attention=True, la_siblings=branches[0], la_analogues=branches[1], n_shops=3,
            la_emb_dim=6, age_embedding=True, la_gate=gate,
        )
        model = MultimodalTemporalFusionTransformer(cfg)
        out = model(batch)
        assert torch.isfinite(out["prediction"]).all()  # includes windows with no available sibling / analogue token
        QuantileLoss()(out["prediction"], batch["target"], batch["target_mask"]).backward()
        assert all(g is None or torch.isfinite(g).all() for g in (q.grad for q in model.parameters()))


def test_chronos_finetune_grid_keeps_best_configuration_even_if_later():
    pytest.importorskip("chronos")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import chronos2_baseline as C
    from mtft_plus.data import build_index

    p = _staggered_panel()
    idx = build_index(p, np.arange(2, 30))

    class Model:
        def __init__(self, level):
            self.level = level

        def predict_quantiles(self, inputs, prediction_length, quantile_levels, **_):
            q = torch.full((1, prediction_length, len(quantile_levels)), self.level)
            return [q for _ in inputs], None

    class Pipe:
        def fit(self, inputs, prediction_length, learning_rate, **_):
            return Model({1e-5: 1000.0, 1e-6: 10.0}[learning_rate])  # the second (later) config is far better

    ft, info = C.finetune(Pipe(), p, idx, idx, False, None, 2, 4, [(1e-5, 10), (1e-6, 10)], 0, Path("."), lambda *_: None)
    assert info["chosen"]["lr"] == 1e-6 and ft.level == 10.0
    assert info["grid"][0]["val_wQL"] > info["grid"][1]["val_wQL"]


def test_chronos_group_inputs_put_targets_first_and_mask_non_live():
    pytest.importorskip("chronos")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import chronos2_baseline as C

    p = _staggered_panel()
    # product a at origin 8: target shop s2 (launched 6); s1 (launched 2, window 2-5) is live in the context 6..7? no ->
    # context weeks 6-7 only: s1 ended at 5, so the only variate is the target itself.
    inp, grp = C.group_inputs(p, np.array([[0, 1, 8]]), 2, 4)
    assert len(inp) == 1 and inp[0]["target"].shape == (1, 2) and grp[0].tolist() == [0]
    # origin 6, context weeks 4-5: target s2 has no history yet (launch 6), s1 is live (weeks 4, 5)
    inp, grp = C.group_inputs(p, np.array([[0, 1, 6]]), 2, 4, fit=True)
    t = inp[0]["target"]
    assert t.shape == (2, 6)
    assert np.isnan(t[0, 0]) and t[0, 1] == 0.0  # target without history: one forced zero, rest NaN
    assert np.allclose(t[1, :2], [30.0, 40.0])  # s1 at weeks 4, 5
    assert np.allclose(t[0, 2:], [15.0, 25.0, 35.0, 45.0]) and np.isnan(t[1, 2:]).all()  # s2 future; s1 ended


def test_level_window_setting_drives_sibling_level():
    from dataclasses import replace

    from mtft_plus.data import DemandWindowDataset, SyntheticConfig, build_index, generate_synthetic_retail

    d = generate_synthetic_retail(SyntheticConfig(n_articles=6, n_days=260, seed=4))
    d.cfg = replace(d.cfg, level_window=5)
    mm = {"image": d.emb_aligned_image, "text": d.emb_aligned_text, "mask": np.ones((6, 2), bool)}
    idx = build_index(d, np.array([150]), require_target=False)
    b = DemandWindowDataset(d, idx, mm, siblings=True).__getitems__(list(range(len(idx))))
    sib = b["enc_observed"][:, :, -1].numpy()
    assert np.allclose(b["static_real"][:, -1].numpy(), sib[:, -5:].mean(1), atol=1e-6)


def _batch(B=6, L=10, H=4, D=32, k=6):
    return {
        "static_cat": torch.stack([torch.randint(0, 3, (B,)), torch.randint(0, 5, (B,))], 1),
        "static_real": torch.randn(B, 2),
        "enc_observed": torch.randn(B, L, 1),
        "enc_known_real": torch.randn(B, L, 3),
        "enc_known_cat": torch.randint(0, 7, (B, L, 1)),
        "dec_known_real": torch.randn(B, H, 3),
        "dec_known_cat": torch.randint(0, 7, (B, H, 1)),
        "mm_image": torch.randn(B, D),
        "mm_text": torch.randn(B, D),
        "mm_mask": torch.tensor([[True, True], [True, False], [False, True], [True, True], [False, False], [True, True]])[:B],
        "mm_scalar": torch.randn(B, k),
        "target": torch.rand(B, H),
        "target_mask": torch.ones(B, H, dtype=torch.bool),
    }


@pytest.mark.parametrize("fusion", ["none", "static_scalar", "static_pooled", "cross_attention"])
def test_model_forward_backward(fusion):
    cfg = MTFTConfig(
        static_categorical_cardinalities=[3, 5], n_static_reals=2, n_observed_reals=1, n_known_reals=3,
        known_categorical_cardinalities=[7], encoder_length=10, horizon=4, d_model=16, n_heads=4,
        fusion=fusion, mm_embed_dim=32, mm_scalar_dim=6, mm_latents=4,
    )
    model = MultimodalTemporalFusionTransformer(cfg)
    model.return_mm_attribution = True
    batch = _batch()
    out = model(batch)
    assert out["prediction"].shape == (6, 4, 3)
    assert torch.isfinite(out["prediction"]).all()
    loss = QuantileLoss()(out["prediction"], batch["target"], batch["target_mask"])
    if out["align"]:
        tok = model.mm_tokenizer
        loss = loss + siglip_alignment_loss(out["align"]["image_latent"], out["align"]["text_latent"], tok.logit_scale, tok.logit_bias, torch.tensor([0, 0, 1, 2, 3, 4]), out["align"]["pair_mask"])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    if fusion == "cross_attention":
        model.eval()  # attention weights are returned post-dropout in train mode
        with torch.no_grad():
            attr = model(batch)["modality_attribution"]
        assert attr.shape == (6, 4, 3)
        assert torch.allclose(attr.sum(-1), torch.ones(6, 4), atol=1e-5)


def test_decoder_is_causal():
    cfg = MTFTConfig(
        static_categorical_cardinalities=[3, 5], n_static_reals=2, n_observed_reals=1, n_known_reals=3,
        known_categorical_cardinalities=[7], encoder_length=10, horizon=4, d_model=16, n_heads=4,
        fusion="cross_attention", mm_embed_dim=32, dropout=0.0,
    )
    model = MultimodalTemporalFusionTransformer(cfg).eval()
    b1 = _batch()
    b2 = {k: v.clone() for k, v in b1.items()}
    b2["dec_known_real"][:, 2:] += 5.0  # perturb future known inputs after step 1
    with torch.no_grad():
        p1, p2 = model(b1)["prediction"], model(b2)["prediction"]
    assert torch.allclose(p1[:, :2], p2[:, :2], atol=1e-5)
    assert not torch.allclose(p1[:, 2:], p2[:, 2:])


def test_feature_extractor_with_tiny_siglip(tmp_path):
    transformers = pytest.importorskip("transformers")
    from PIL import Image

    cfg = transformers.SiglipConfig(
        text_config=dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=8),
    )
    model = transformers.SiglipModel(cfg)

    class FakeProcessor:
        class tokenizer:
            pad_token_id = 1

        def __call__(self, text=None, images=None, max_length=16, **_):
            if text is not None:
                ids = torch.ones(len(text), max_length, dtype=torch.long)
                for i, t in enumerate(text):
                    toks = [2 + (ord(c) % 90) for c in t][:max_length]
                    ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long) if toks else ids[i, :0]
                return {"input_ids": ids}
            arr = torch.stack([torch.from_numpy(np.asarray(im.resize((32, 32)), dtype=np.float32) / 255.0).permute(2, 0, 1) for im in images])
            return {"pixel_values": arr}

    recs = [
        ArticleRecord("a1", "biologische bananen", Image.new("RGB", (40, 40), (250, 220, 40))),
        ArticleRecord("a2", "volle melk 1L", None),
        ArticleRecord("a3", None, Image.new("RGB", (40, 40), (10, 200, 10))),
    ]
    for tokens in (False, True):
        ext = MultimodalFeatureExtractor(cache_dir=tmp_path / str(tokens), model=model, processor=FakeProcessor(), device="cpu", max_text_length=16, return_tokens=tokens)
        bank = ext.encode(recs)
        assert bank.image.shape == (3, 32) and bank.text.shape == (3, 32)
        assert bank.mask.tolist() == [[True, True], [False, True], [True, False]]
        assert torch.allclose(bank.image[0].float().norm(), torch.tensor(1.0), atol=1e-2)
        assert bank.image[1].abs().sum() == 0 and bank.text[2].abs().sum() == 0
        again = ext.encode(recs)  # cache hit
        assert torch.equal(again.image, bank.image)
        if tokens:
            assert bank.image_tokens.shape[:2] == (3, 16) and bank.text_tokens.shape[:2] == (3, 16)


def test_panel_round_trip_and_analogues():
    from mtft_plus.data import SyntheticConfig, build_analogues, generate_synthetic_retail
    from mtft_plus.panel import load_panel, panel_to_frames

    d = generate_synthetic_retail(SyntheticConfig(n_articles=30, n_days=260, seed=3))
    sales, catalog, locations = panel_to_frames(d)
    p = load_panel(sales, catalog, locations)
    assert p.y.shape == d.y.shape and np.allclose(p.y, d.y)
    assert np.allclose(p.promo, d.promo) and np.allclose(p.discount, d.discount)
    assert (p.launch == np.minimum(d.launch, d.y.shape[2] - 1)).all()
    an = build_analogues(np.concatenate([d.emb_aligned_image, d.emb_aligned_text], 1), ~d.cold_start, k=5)
    assert not (an.idx == np.arange(30)[:, None]).any()
    assert (~d.cold_start)[an.idx].all() and np.allclose(an.weights.sum(1), 1)


def test_conformal_calibrator_reaches_nominal_coverage():
    from mtft_plus.calibration import LogConformalCalibrator

    rng = np.random.default_rng(0)
    y = rng.gamma(4.0, 5.0, (6000,))
    q = np.stack([np.full_like(y, 17.0), np.full_like(y, 19.0), np.full_like(y, 21.0)], -1)
    cal = LogConformalCalibrator((0.1, 0.5, 0.9)).fit(y[:3000], q[:3000])
    qc = cal.transform(q[3000:])
    cov = ((y[3000:] >= qc[:, 0]) & (y[3000:] <= qc[:, 2])).mean()
    assert abs(cov - 0.8) < 0.03 and np.allclose(qc[:, 1], 19.0)


def test_analogue_features_use_only_past():
    from mtft_plus.data import DemandWindowDataset, SyntheticConfig, build_analogues, build_index, generate_synthetic_retail

    d = generate_synthetic_retail(SyntheticConfig(n_articles=20, n_days=260, seed=5))
    an = build_analogues(np.concatenate([d.emb_aligned_image, d.emb_aligned_text], 1), ~d.cold_start, k=4)
    mm = {"image": d.emb_aligned_image, "text": d.emb_aligned_text, "mask": np.ones((20, 2), bool)}
    idx = build_index(d, np.array([150]), require_target=False)
    b1 = DemandWindowDataset(d, idx, mm, an).__getitems__(list(range(len(idx))))
    d.y[:, :, 150:] = 1e6  # corrupt everything from the origin onwards
    b2 = DemandWindowDataset(d, idx, mm, an).__getitems__(list(range(len(idx))))
    assert torch.equal(b1["enc_observed"], b2["enc_observed"]) and torch.equal(b1["static_real"], b2["static_real"])


def test_sibling_channel_is_mean_of_other_fcs_and_past_only():
    from mtft_plus.data import DemandWindowDataset, SyntheticConfig, build_index, generate_synthetic_retail

    d = generate_synthetic_retail(SyntheticConfig(n_articles=12, n_days=260, seed=9))
    mm = {"image": d.emb_aligned_image, "text": d.emb_aligned_text, "mask": np.ones((12, 2), bool)}
    idx = build_index(d, np.array([150]), require_target=False)
    b = DemandWindowDataset(d, idx, mm, siblings=True).__getitems__(list(range(len(idx))))
    a, f = idx[0, 0], idx[0, 1]
    L = d.cfg.encoder_length
    others = [g for g in range(d.n_fc) if g != f]
    ref = np.log1p(d.y[a, others, 150 - L : 150]).mean(0)
    assert np.allclose(b["enc_observed"][0, :, -1].numpy(), ref, atol=1e-5)
    d.y[:, :, 150:] = 1e6
    b2 = DemandWindowDataset(d, idx, mm, siblings=True).__getitems__(list(range(len(idx))))
    assert torch.equal(b["enc_observed"], b2["enc_observed"]) and torch.equal(b["static_real"], b2["static_real"])
