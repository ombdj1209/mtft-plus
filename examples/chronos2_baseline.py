"""Chronos-2 zero-shot baseline (Ansari et al., 2025; amazon/chronos-2) on the same test windows as our models.

    # calendar benchmark (writes partials that report.py aggregates next to the trained models)
    python examples/chronos2_baseline.py calendar --panel panel_v2.pkl --out results_visuelle2
    # official VISUELLE 2.0 protocol (writes partials that examples/visuelle2_official.py --report aggregates)
    python examples/chronos2_baseline.py official --panel panel_v2_full.pkl --src visuelle2 --out results_visuelle2_official

Inputs are built by the same DemandWindowDataset as the trained models, so Chronos-2 sees the same windows:
  * target: the series' own units over the encoder window; steps before the series launch are NaN (missing);
  * "+ covariates": past covariate = mean log1p demand of the product in its other live shops (the sibling channel),
    known covariate = discount depth over context and horizon;
  * "+ covariates + cross-learning": additionally cross_learning=True within groups of the same product and origin,
    Chronos-2's group attention (the closest zero-shot analogue of sibling learning).
No fine-tuning, no tuning on test data. Quantiles 0.1 / 0.5 / 0.9. Zero-shot, so the seed is irrelevant (recorded as 0).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from benchmark import QUANTILES, SeedContext, per_article_national, per_article_sums, reconcile, slug
from mtft_plus.data import KNOWN_REALS, DemandWindowDataset
from mtft_plus.metrics import evaluate

VARIANTS = {"Chronos-2": dict(cov=False, cross=False), "Chronos-2 + covariates": dict(cov=True, cross=False),
            "Chronos-2 + covariates + cross-learning": dict(cov=True, cross=True)}
I_DISC, I_LAUNCHED = KNOWN_REALS.index("discount"), KNOWN_REALS.index("launched")


def load_pipeline(path: str):
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(path, device_map="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)


LA_KNOWN = ["la_sib_mean", "la_sib_avail", "la_ana_mean", "la_ana_cov"]


def chronos_inputs(d, idx: np.ndarray, cov: bool, bs: int = 4096, lifecycle=None, fit: bool = False):
    """DemandWindowDataset windows -> Chronos-2 input dicts.

    Covariates (``cov``): discount (known future), the calendar sibling mean (past only) and, with ``lifecycle``, the
    four LA-pool channels (known future, computed causally at the origin). ``fit=True`` returns one context+horizon
    series per window for ``Chronos2Pipeline.fit`` (horizon target NaN where not live; past-only covariates NaN over
    the horizon); otherwise prediction inputs (context target + past covariates + future known covariates)."""
    mm = {"image": np.zeros((d.n_articles, 1), np.float32), "text": np.zeros((d.n_articles, 1), np.float32), "mask": np.ones((d.n_articles, 2), bool)}
    use_la = cov and lifecycle is not None
    ds = DemandWindowDataset(d, idx, mm, None, siblings=True, lifecycle=lifecycle if use_la else None, la_pool=use_la)
    n_k = len(KNOWN_REALS)
    out = []
    for s in range(0, len(idx), bs):
        b = ds.__getitems__(list(range(s, min(s + bs, len(idx)))))
        own = np.expm1(b["enc_observed"][..., 0].numpy()).astype(np.float32)
        own[b["enc_known_real"][..., I_LAUNCHED].numpy() < 0.5] = np.nan  # before this series' launch: missing
        sib = b["enc_observed"][..., -1].numpy()
        kp, kf = b["enc_known_real"].numpy(), b["dec_known_real"].numpy()
        known = {"discount": I_DISC, **({c: n_k + j for j, c in enumerate(LA_KNOWN)} if use_la else {})}
        fut = np.where(b["target_mask"].numpy(), np.expm1(b["target"].numpy()), np.nan).astype(np.float32)
        H = fut.shape[1]
        for i in range(len(own)):
            ctx = own[i].copy()
            if np.isnan(ctx).all():
                ctx[-1] = 0.0  # Chronos-2 needs one observed value; a launch week has zero past sales
            if fit:
                x = {"target": np.concatenate([ctx, fut[i]])}
                if cov:
                    x["past_covariates"] = {k: np.concatenate([kp[i, :, j], kf[i, :, j]]).astype(np.float32) for k, j in known.items()}
                    x["past_covariates"]["siblings"] = np.concatenate([sib[i], np.full(H, np.nan, np.float32)])
            else:
                x = {"target": ctx}
                if cov:
                    x["past_covariates"] = {"siblings": sib[i], **{k: kp[i, :, j].astype(np.float32) for k, j in known.items()}}
                    x["future_covariates"] = {k: kf[i, :, j].astype(np.float32) for k, j in known.items()}
            out.append(x)
    return out


def finetune(pipe, d_fit, tr_idx, va_idx, cov: bool, lifecycle, L: int, H: int, grid, seed: int, workdir: Path, log):
    """Full fine-tuning on the training windows; checkpoint selection on validation loss inside ``fit`` (every 100 steps),
    and the (learning rate, steps) configuration chosen by validation wQL of our own evaluation. Returns (pipeline, info)."""
    from chronos.chronos2.preprocess import from_list_of_dicts

    known = (["discount"] + (LA_KNOWN if lifecycle is not None else [])) if cov else None
    tr = from_list_of_dicts(chronos_inputs(d_fit, tr_idx, cov, lifecycle=lifecycle, fit=True), H, known_covariates_names=known)
    va_fit = from_list_of_dicts(chronos_inputs(d_fit, va_idx, cov, lifecycle=lifecycle, fit=True), H, known_covariates_names=known)
    va_pred = chronos_inputs(d_fit, va_idx, cov, lifecycle=lifecycle)
    ds = DemandWindowDataset(d_fit, va_idx, {"image": np.zeros((d_fit.n_articles, 1), np.float32), "text": np.zeros((d_fit.n_articles, 1), np.float32), "mask": np.ones((d_fit.n_articles, 2), bool)})
    b = ds.__getitems__(list(range(len(va_idx))))
    y_va, m_va = np.expm1(b["target"].numpy()), b["target_mask"].numpy()
    best, info = None, []
    for lr, steps in grid:
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        out_dir = workdir / f"lr{lr}_s{steps}"
        ft = pipe.fit(tr, prediction_length=H, validation_inputs=va_fit, learning_rate=lr, num_steps=steps, batch_size=256,
                      min_past=L, output_dir=str(out_dir), remove_printer_callback=True, seed=seed)
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)  # checkpoints (~0.5 GB each) are not needed: the best model is in memory
        q = chronos_predict(ft, va_pred, H)
        u = y_va[..., None] - q
        qa = np.asarray(QUANTILES)
        val_wql = float(np.mean(2 * (np.maximum(qa * u, (qa - 1) * u)[m_va]).sum(0) / max(y_va[m_va].sum(), 1e-9)))
        info.append({"lr": lr, "steps": steps, "val_wQL": val_wql, "fit_seconds": round(time.time() - t0, 1)})
        log(f"    fine-tune cov={cov} lr={lr} steps={steps}: val wQL {val_wql:.4f} ({time.time() - t0:.0f}s)")
        if best is None or val_wql < best[0]:
            best = (val_wql, ft, lr, steps)  # rebinding releases the previously best pipeline
        else:
            del ft
        torch.cuda.empty_cache()
    return best[1], {"grid": info, "chosen": {"lr": best[2], "steps": best[3], "val_wQL": best[0]}}


def group_inputs(d, idx: np.ndarray, L: int, H: int, fit: bool = False, max_var: int = 64, pad_to: int | None = None):
    """Group attention over all shops of a product (pre-registration amendment (d) 1b-ii).

    Windows are grouped by (article, origin). Each group is one multivariate target whose variates are the windows'
    shops first, then the product's other shops with a live cell in the context window (most recent launch first),
    at most ``max_var`` variates. Values outside a series' live window are NaN (missing). ``fit=True`` appends the
    horizon (NaN where not live), for ``Chronos2Pipeline.fit``. Returns (inputs, groups): groups[i] holds the positions
    in ``idx`` of the windows whose forecasts are rows 0..len(groups[i])-1 of input i."""
    A, F, T = d.y.shape
    launch = d.series_launch()
    keys = idx[:, 0].astype(np.int64) * (T + 1) + idx[:, 2]
    order = np.argsort(keys, kind="stable")
    bounds = np.flatnonzero(np.diff(keys[order])) + 1
    shops_all = np.arange(F)
    inputs, groups = [], []
    for g in np.split(order, bounds):
        a, o = int(idx[g[0], 0]), int(idx[g[0], 2])
        for c0 in range(0, len(g), max_var):
            gw = g[c0 : c0 + max_var]
            tshops = idx[gw, 1]
            t_ctx = np.arange(o - L, o)
            ok_t = t_ctx >= 0
            tc = np.clip(t_ctx, 0, T - 1)
            live_ctx = d.live_cells(a, shops_all[:, None], tc[None, :]) & ok_t[None, :]
            others = np.setdiff1d(np.flatnonzero(live_ctx.any(1)), tshops)
            others = others[np.argsort(-launch[a, others], kind="stable")][: max(max_var - len(tshops), 0)]
            shops = np.concatenate([tshops, others]).astype(np.int64)
            ctx = np.where(live_ctx[shops], d.y[a, shops[:, None], tc[None, :]], np.nan).astype(np.float32)
            empty = ~np.isfinite(ctx).any(1)
            ctx[empty, -1] = 0.0  # a variate needs one observed value; a launch week has zero past sales
            if fit:
                t_f = np.arange(o, o + H)
                tf = np.clip(t_f, 0, T - 1)
                live_f = d.live_cells(a, shops[:, None], tf[None, :]) & (t_f < T)[None, :]
                fut = np.where(live_f, d.y[a, shops[:, None], tf[None, :]], np.nan).astype(np.float32)
                ctx = np.concatenate([ctx, fut], 1)
            if pad_to is not None and len(ctx) < pad_to:
                # the fine-tuning API needs equal variate counts: pad with fully missing series (no loss, no signal)
                pad = np.full((pad_to - len(ctx), ctx.shape[1]), np.nan, np.float32)
                pad[:, L - 1] = 0.0
                ctx = np.concatenate([ctx, pad], 0)
            inputs.append({"target": ctx})
            groups.append(gw)
    return inputs, groups


def group_predict(pipe, inputs, groups, n: int, H: int, bs: int = 256) -> np.ndarray:
    """One predict call per variate count (the pipeline requires equal n_targets within a call; no padding is used)."""
    q = np.zeros((n, H, len(QUANTILES)), np.float32)
    nv = np.array([x["target"].shape[0] for x in inputs])
    for v in np.unique(nv):
        pos = np.flatnonzero(nv == v)
        for s in range(0, len(pos), 512):
            chunk = pos[s : s + 512]
            qs, _ = pipe.predict_quantiles([inputs[i] for i in chunk], prediction_length=H, quantile_levels=list(QUANTILES), batch_size=bs)
            for x, i in zip(qs, chunk):
                q[groups[i]] = x[: len(groups[i])].numpy()
    return np.clip(q, 0, None)


def finetune_group(pipe, d_fit, tr_idx, va_idx, L: int, H: int, grid, seed: int, workdir: Path, log):
    """Fine-tuning with multivariate (group) targets; same grid rule and validation-only selection as ``finetune``."""
    import shutil

    from chronos.chronos2.preprocess import from_list_of_dicts

    V = FT_GROUP_VARIATES
    tr_in, _ = group_inputs(d_fit, tr_idx, L, H, fit=True, max_var=V, pad_to=V)
    va_fit_in, _ = group_inputs(d_fit, va_idx, L, H, fit=True, max_var=V, pad_to=V)
    tr, va_fit = from_list_of_dicts(tr_in, H), from_list_of_dicts(va_fit_in, H)
    va_in, va_groups = group_inputs(d_fit, va_idx, L, H, max_var=V)
    mm = {"image": np.zeros((d_fit.n_articles, 1), np.float32), "text": np.zeros((d_fit.n_articles, 1), np.float32), "mask": np.ones((d_fit.n_articles, 2), bool)}
    b = DemandWindowDataset(d_fit, va_idx, mm).__getitems__(list(range(len(va_idx))))
    y_va, m_va = np.expm1(b["target"].numpy()), b["target_mask"].numpy()
    qa = np.asarray(QUANTILES)
    best, info = None, []
    for lr, steps in grid:
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        out_dir = workdir / f"lr{lr}_s{steps}"
        ft = pipe.fit(tr, prediction_length=H, validation_inputs=va_fit, learning_rate=lr, num_steps=steps, batch_size=256,
                      min_past=L, output_dir=str(out_dir), remove_printer_callback=True, seed=seed)
        shutil.rmtree(out_dir, ignore_errors=True)
        q = group_predict(ft, va_in, va_groups, len(va_idx), H)
        u = y_va[..., None] - q
        val_wql = float(np.mean(2 * (np.maximum(qa * u, (qa - 1) * u)[m_va]).sum(0) / max(y_va[m_va].sum(), 1e-9)))
        info.append({"lr": lr, "steps": steps, "val_wQL": val_wql, "fit_seconds": round(time.time() - t0, 1)})
        log(f"    fine-tune group lr={lr} steps={steps}: val wQL {val_wql:.4f} ({time.time() - t0:.0f}s)")
        if best is None or val_wql < best[0]:
            best = (val_wql, ft, lr, steps)
        else:
            del ft
        torch.cuda.empty_cache()
    return best[1], {"grid": info, "chosen": {"lr": best[2], "steps": best[3], "val_wQL": best[0]}}


def chronos_predict(pipe, inputs, H: int, groups: np.ndarray | None = None, bs: int = 256) -> np.ndarray:
    """(N, H, 3) quantiles; with ``groups``, cross-learning runs once per group (inputs in one group share attention)."""
    q = np.zeros((len(inputs), H, len(QUANTILES)), np.float32)
    if groups is None:
        for s in range(0, len(inputs), bs * 8):
            qs, _ = pipe.predict_quantiles(inputs[s : s + bs * 8], prediction_length=H, quantile_levels=list(QUANTILES), batch_size=bs)
            q[s : s + len(qs)] = torch.stack([x[0] for x in qs]).numpy()
        return np.clip(q, 0, None)
    order = np.argsort(groups, kind="stable")
    bounds = np.flatnonzero(np.diff(groups[order])) + 1
    for g in np.split(order, bounds):
        for s in range(0, len(g), 100):  # the Chronos-2 report uses groups of about 100 for cross-learning
            part = g[s : s + 100]
            qs, _ = pipe.predict_quantiles([inputs[i] for i in part], prediction_length=H, quantile_levels=list(QUANTILES),
                                           batch_size=len(part), cross_learning=True)
            q[part] = torch.stack([x[0] for x in qs]).numpy()
    return np.clip(q, 0, None)


# ---------------------------------------------------------------------------------------------------- calendar
def long_inputs(d, idx: np.ndarray, C: int):
    """Target-only prediction inputs with a context of ``C`` steps (longer than the models' encoder window): the series'
    own units at t = o - C .. o - 1, NaN where t < 0 or the cell is not live (before launch)."""
    a, f, o = idx[:, 0], idx[:, 1], idx[:, 2]
    t = o[:, None] + np.arange(-C, 0)[None]
    tc = np.clip(t, 0, None)
    ok = (t >= 0) & d.live_cells(a[:, None], f[:, None], tc)
    own = np.where(ok, d.y[a[:, None], f[:, None], tc], np.nan).astype(np.float32)
    out = []
    for i in range(len(idx)):
        x = own[i]
        first = np.argmax(np.isfinite(x)) if np.isfinite(x).any() else C - 1
        x = x[first:].copy()  # leading missing steps carry no information; trimming keeps batches small
        if np.isnan(x).all():
            x[-1] = 0.0
        out.append({"target": x})
    return out


def run_calendar(args) -> None:
    import types

    pipe = load_pipeline(args.model)
    ctx = SeedContext(0, types.SimpleNamespace(panel=args.panel, k=10, fcs=0, articles=0, days=0))
    d, A, F, H = ctx.data, ctx.A, ctx.F, ctx.H
    part_dir = Path(args.out) / "partials"
    part_dir.mkdir(parents=True, exist_ok=True)
    for C in [int(c) for c in args.long_context.split(",") if c]:
        name = f"Chronos-2 (context {C})"
        path = part_dir / f"0__{slug(name)}.json"
        if path.exists():
            continue
        t0 = time.time()
        qs = {}
        for split, idx, origins, mask in (("val", ctx.val_all_idx, ctx.val_o, ctx.mask_val_b), ("test", ctx.test_all_idx, ctx.test_o, ctx.mask_test_b)):
            qs[split] = scatter(ctx, chronos_predict(pipe, long_inputs(d, idx, C), H), idx, origins, mask)
        calendar_result(ctx, name, {"zero_shot": True, "context": C}, 0, qs, t0, args, path)
    for name, v in VARIANTS.items():
        path = part_dir / f"0__{slug(name)}.json"
        if path.exists():
            continue
        t0 = time.time()
        qs = {}
        for split, idx, origins, mask in (("val", ctx.val_all_idx, ctx.val_o, ctx.mask_val_b), ("test", ctx.test_all_idx, ctx.test_o, ctx.mask_test_b)):
            groups = idx[:, 0] * 10_000 + np.searchsorted(origins, idx[:, 2]) if v["cross"] else None
            qs[split] = scatter(ctx, chronos_predict(pipe, chronos_inputs(d, idx, v["cov"]), H, groups), idx, origins, mask)
        calendar_result(ctx, name, {"zero_shot": True, **v}, 0, qs, t0, args, path)


def scatter(ctx, q: np.ndarray, idx: np.ndarray, origins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    A, F, H = ctx.A, ctx.F, ctx.H
    full = np.zeros((A * F, len(origins), H, len(QUANTILES)), np.float32)
    full[idx[:, 0] * F + idx[:, 1], np.searchsorted(origins, idx[:, 2])] = q
    return full * mask[..., None]


def calendar_result(ctx, name: str, spec: dict, seed: int, qs: dict, t0: float, args, path: Path, extra: dict | None = None) -> None:
    from benchmark import save_forecasts

    A, F, H = ctx.A, ctx.F, ctx.H
    yt = ctx.y_test.reshape(A * F, len(ctx.test_o), H)
    subsets = [("all", np.ones(A * F, bool)), ("warm", ~ctx.cold_b), ("cold-start", ctx.cold_b)]
    if ctx.new_in_shop_b is not None:
        subsets.append(("new-in-shop", ctx.new_in_shop_b))
    if getattr(ctx, "young_b", None) is not None:
        subsets.append(("young", ctx.young_b))
    bottom = [{"subset": g, **evaluate(yt[s], qs["test"][s], QUANTILES, ctx.scale_b[s], ctx.mask_test_b[s])} for g, s in subsets]
    hier_rows, rec, coh, _, bu_q = reconcile(ctx, qs["val"], qs["test"])
    res = {"seed": seed, "model": name, "spec": spec, "params": 0, "train_seconds": 0.0, "val_curve": [],
           "best_val": float("nan"), "bottom": bottom, "hierarchy": hier_rows, "mint_lambda": rec.lam, "coherence_error": coh,
           "per_article": {**per_article_sums(ctx, qs["test"]), **per_article_national(ctx, bu_q)}, "inference_seconds": round(time.time() - t0, 1),
           "args": {k: str(v) for k, v in vars(args).items()}, **(extra or {})}
    path.write_text(json.dumps(res))
    save_forecasts(path.parent.parent / "forecasts" / path.name.replace(".json", ".npz"), qs["test"], ctx.mask_test_b)
    b = bottom[0]
    print(f"[calendar] seed {seed} {name:42s} {time.time() - t0:5.0f}s | WAPE {b['WAPE']:.4f} MASE {b['MASE']:.4f} wQL {b['wQL']:.4f} | cold WAPE {bottom[2]['WAPE']:.4f}", flush=True)


FT_VARIANTS = {"Chronos-2 fine-tuned": False, "Chronos-2 fine-tuned + covariates": True}


def parse_grid(s: str):
    return [(float(a), int(b)) for a, b in (x.split(":") for x in s.split(","))]


def run_finetune_calendar(args) -> None:
    import types

    pipe = load_pipeline(args.model)
    ctx = SeedContext(0, types.SimpleNamespace(panel=args.panel, k=10, fcs=0, articles=0, days=0))
    lf = ctx.lifecycle()["aligned"]
    part_dir = Path(args.out) / "partials"
    part_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")]
    for name, cov in FT_VARIANTS.items():
        chosen = None
        for seed in seeds:
            path = part_dir / f"{seed}__{slug(name)}.json"
            if path.exists():
                chosen = json.loads(path.read_text(encoding="utf-8"))["finetune"]["chosen"]
                continue
            t0 = time.time()
            rng = np.random.default_rng(seed)
            tr = ctx.train_idx[rng.permutation(len(ctx.train_idx))[: args.n_train]]
            grid = parse_grid(args.grid) if chosen is None else [(chosen["lr"], chosen["steps"])]
            ft, info = finetune(pipe, ctx.data, tr, ctx.val_mon_idx, cov, lf, ctx.L, ctx.H, grid, seed, Path(args.workdir) / f"cal_{slug(name)}_{seed}", print)
            chosen = chosen or info["chosen"]
            qs = {}
            for split, idx, origins, mask in (("val", ctx.val_all_idx, ctx.val_o, ctx.mask_val_b), ("test", ctx.test_all_idx, ctx.test_o, ctx.mask_test_b)):
                qs[split] = scatter(ctx, chronos_predict(ft, chronos_inputs(ctx.data, idx, cov, lifecycle=lf), ctx.H), idx, origins, mask)
            spec = {"zero_shot": False, "finetuned": True, "cov": cov, "grid_seed": seeds[0]}
            calendar_result(ctx, name, spec, seed, qs, t0, args, path, {"finetune": {**info, "chosen": chosen}, "train_seconds": round(time.time() - t0, 1)})
            del ft
            torch.cuda.empty_cache()


def run_finetune_official(args) -> None:
    import visuelle2_official as V
    from benchmark import product_features

    pipe = load_pipeline(args.model)
    proto = V.Protocol(args.panel, Path(args.src))
    mm_all, _ = product_features(proto.d_full, proto.warm, 10, 1)
    lfs = proto.lifecycle(mm_all)
    part_dir = Path(args.out) / "partials"
    part_dir.mkdir(parents=True, exist_ok=True)
    te = proto.pairs[proto.pairs.role == "test"]
    new = te.new_product.to_numpy()
    seeds = [int(x) for x in args.seeds.split(",")]
    for task_name in [t.strip() for t in args.tasks.split(",")]:
        task = V.TASKS[task_name]
        d_fit = proto.with_windows(proto.d_fit, task["L"], task["H"])
        d_test = proto.with_windows(proto.d_full, task["L"], task["H"])
        tr_all, va_all, te_idx = proto.index("train", task), proto.index("val", task), proto.index("test", task)
        gt = np.stack(te[task["gt"]].to_numpy())
        for name, cov in FT_VARIANTS.items():
            chosen = None
            for seed in seeds:
                path = part_dir / f"{task_name}__{seed}__{slug(name)}.json"
                if path.exists():
                    chosen = json.loads(path.read_text(encoding="utf-8"))["finetune"]["chosen"]
                    continue
                t0 = time.time()
                rng = np.random.default_rng(seed)
                tr = tr_all[rng.permutation(len(tr_all))[: args.n_train]]
                va = va_all[rng.permutation(len(va_all))[:8192]]
                grid = parse_grid(args.grid) if chosen is None else [(chosen["lr"], chosen["steps"])]
                ft, info = finetune(pipe, d_fit, tr, va, cov, lfs["fit"]["aligned"], task["L"], task["H"], grid, seed,
                                    Path(args.workdir) / f"{task_name}_{slug(name)}_{seed}", print)
                chosen = chosen or info["chosen"]
                q = chronos_predict(ft, chronos_inputs(d_test, te_idx, cov, lifecycle=lfs["full"]["aligned"]), task["H"])
                official_result(V, te, new, gt, task, task_name, name, seed, q, t0, path,
                                {"spec": {"zero_shot": False, "finetuned": True, "cov": cov, "grid_seed": seeds[0]},
                                 "finetune": {**info, "chosen": chosen}, "train_seconds": round(time.time() - t0, 1)})
                del ft
                torch.cuda.empty_cache()


def official_result(V, te, new, gt, task, task_name, name, seed, q, t0, path, extra) -> None:
    med = q[..., 1].reshape(len(te), task["n_origins"] * task["H"])
    y = gt[:, task["first"] : task["first"] + med.shape[1]]
    res = {"task": task_name, "model": name, "seed": seed, "train_seconds": 0.0,
           "all": V.metrics(y, med), "new_product": V.metrics(y[new], med[new]), "seen_product": V.metrics(y[~new], med[~new]),
           "coverage80": float(((y >= q[..., 0].reshape(med.shape)) & (y <= q[..., 2].reshape(med.shape))).mean()),
           "per_pair": {"abs_err": np.abs(y - med).sum(1).round(4).tolist(), "y": y.sum(1).round(4).tolist(),
                        "product": te.external_code.tolist(), "new_product": new.tolist()},
           "inference_seconds": round(time.time() - t0, 1), **extra}
    path.write_text(json.dumps(res))
    fdir = path.parent.parent / "forecasts"
    fdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fdir / path.name.replace(".json", ".npz"), q=q.reshape(len(te), -1, len(QUANTILES)).astype(np.float32))
    print(f"[{task_name}] seed {seed} {name:42s} {time.time() - t0:5.0f}s | WAPE {res['all']['WAPE']:6.2f} MAE {res['all']['MAE']:.3f} "
          f"| new-product {res['new_product']['WAPE']:6.2f}", flush=True)


GROUP_ZS = {"Chronos-2 + cross-learning": "cross", "Chronos-2 group (multivariate)": "group"}
GROUP_FT = "Chronos-2 fine-tuned group"
FT_GROUP_VARIATES = 16  # fixed group size for fine-tuning (library constraint); predictions use the same cap


def run_group_calendar(args) -> None:
    """Amendment (d) 1b on the calendar benchmark: zero-shot (i) cross-learning and (ii) multivariate group, target only;
    fine-tuned (ii) with the amendment (c) grid on the first seed and the chosen setting on every seed."""
    import types

    pipe = load_pipeline(args.model)
    ctx = SeedContext(0, types.SimpleNamespace(panel=args.panel, k=10, fcs=0, articles=0, days=0))
    d, H, L = ctx.data, ctx.H, ctx.L
    part = Path(args.out) / "partials"
    part.mkdir(parents=True, exist_ok=True)
    splits = (("val", ctx.val_all_idx, ctx.val_o, ctx.mask_val_b), ("test", ctx.test_all_idx, ctx.test_o, ctx.mask_test_b))
    for name, kind in GROUP_ZS.items():
        path = part / f"0__{slug(name)}.json"
        if path.exists():
            continue
        t0 = time.time()
        qs = {}
        for split, idx, origins, mask in splits:
            if kind == "cross":
                q = chronos_predict(pipe, chronos_inputs(d, idx, False), H, idx[:, 0] * 10_000 + np.searchsorted(origins, idx[:, 2]))
            else:
                inp, grp = group_inputs(d, idx, L, H)
                q = group_predict(pipe, inp, grp, len(idx), H)
            qs[split] = scatter(ctx, q, idx, origins, mask)
        calendar_result(ctx, name, {"zero_shot": True, "group": kind}, 0, qs, t0, args, path)
    chosen = None
    for seed in [int(x) for x in args.seeds.split(",")]:
        path = part / f"{seed}__{slug(GROUP_FT)}.json"
        if path.exists():
            chosen = json.loads(path.read_text(encoding="utf-8"))["finetune"]["chosen"]
            continue
        t0 = time.time()
        rng = np.random.default_rng(seed)
        tr = ctx.train_idx[rng.permutation(len(ctx.train_idx))[: args.n_train_groups]]
        grid = parse_grid(args.grid) if chosen is None else [(chosen["lr"], chosen["steps"])]
        ft, info = finetune_group(pipe, d, tr, ctx.val_mon_idx, L, H, grid, seed, Path(args.workdir) / f"cal_group_{seed}", print)
        chosen = chosen or info["chosen"]
        qs = {}
        for split, idx, origins, mask in splits:
            inp, grp = group_inputs(d, idx, L, H, max_var=FT_GROUP_VARIATES)
            qs[split] = scatter(ctx, group_predict(ft, inp, grp, len(idx), H), idx, origins, mask)
        calendar_result(ctx, GROUP_FT, {"zero_shot": False, "finetuned": True, "group": "group", "grid_seed": 1}, seed, qs, t0, args, path,
                        {"finetune": {**info, "chosen": chosen}, "train_seconds": round(time.time() - t0, 1)})
        del ft
        torch.cuda.empty_cache()


def run_group_official(args) -> None:
    import visuelle2_official as V

    pipe = load_pipeline(args.model)
    proto = V.Protocol(args.panel, Path(args.src))
    part = Path(args.out) / "partials"
    part.mkdir(parents=True, exist_ok=True)
    te = proto.pairs[proto.pairs.role == "test"]
    new = te.new_product.to_numpy()
    for task_name in [t.strip() for t in args.tasks.split(",")]:
        task = V.TASKS[task_name]
        L, H = task["L"], task["H"]
        d_fit, d_test = proto.with_windows(proto.d_fit, L, H), proto.with_windows(proto.d_full, L, H)
        te_idx = proto.index("test", task)
        gt = np.stack(te[task["gt"]].to_numpy())
        for name, kind in GROUP_ZS.items():
            path = part / f"{task_name}__0__{slug(name)}.json"
            if path.exists():
                continue
            t0 = time.time()
            if kind == "cross":
                q = chronos_predict(pipe, chronos_inputs(d_test, te_idx, False), H, te_idx[:, 0] * 10_000 + (te_idx[:, 2] - te_idx[:, 2].min()))
            else:
                inp, grp = group_inputs(d_test, te_idx, L, H)
                q = group_predict(pipe, inp, grp, len(te_idx), H)
            official_result(V, te, new, gt, task, task_name, name, 0, q, t0, path, {"spec": {"zero_shot": True, "group": kind}})
        tr_all, va_all = proto.index("train", task), proto.index("val", task)
        chosen = None
        for seed in [int(x) for x in args.seeds.split(",")]:
            path = part / f"{task_name}__{seed}__{slug(GROUP_FT)}.json"
            if path.exists():
                chosen = json.loads(path.read_text(encoding="utf-8"))["finetune"]["chosen"]
                continue
            t0 = time.time()
            rng = np.random.default_rng(seed)
            tr = tr_all[rng.permutation(len(tr_all))[: args.n_train_groups]]
            va = va_all[rng.permutation(len(va_all))[:8192]]
            grid = parse_grid(args.grid) if chosen is None else [(chosen["lr"], chosen["steps"])]
            ft, info = finetune_group(pipe, d_fit, tr, va, L, H, grid, seed, Path(args.workdir) / f"{task_name}_group_{seed}", print)
            chosen = chosen or info["chosen"]
            inp, grp = group_inputs(d_test, te_idx, L, H, max_var=FT_GROUP_VARIATES)
            q = group_predict(ft, inp, grp, len(te_idx), H)
            official_result(V, te, new, gt, task, task_name, GROUP_FT, seed, q, t0, path,
                            {"spec": {"zero_shot": False, "finetuned": True, "group": "group", "grid_seed": 1},
                             "finetune": {**info, "chosen": chosen}, "train_seconds": round(time.time() - t0, 1)})
            del ft
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------- official
def run_official(args) -> None:
    import visuelle2_official as V

    pipe = load_pipeline(args.model)
    proto = V.Protocol(args.panel, Path(args.src))
    part_dir = Path(args.out) / "partials"
    part_dir.mkdir(parents=True, exist_ok=True)
    te = proto.pairs[proto.pairs.role == "test"]
    new = te.new_product.to_numpy()
    for task_name, task in V.TASKS.items():
        d = proto.with_windows(proto.d_full, task["L"], task["H"])
        idx = proto.index("test", task)
        gt = np.stack(te[task["gt"]].to_numpy())
        for name, v in VARIANTS.items():
            path = part_dir / f"{task_name}__0__{slug(name)}.json"
            if path.exists():
                continue
            t0 = time.time()
            groups = idx[:, 0] * 10_000 + (idx[:, 2] - idx[:, 2].min()) if v["cross"] else None  # same product, same origin week
            q = chronos_predict(pipe, chronos_inputs(d, idx, v["cov"]), task["H"], groups)
            official_result(V, te, new, gt, task, task_name, name, 0, q, t0, path, {"spec": {"zero_shot": True, **v}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol", choices=["calendar", "official", "finetune-calendar", "finetune-official", "group-calendar", "group-official"])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--src", default="visuelle2")
    ap.add_argument("--model", default=".hf_models/chronos-2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5", help="fine-tuning seeds; the (lr, steps) grid runs on the first")
    ap.add_argument("--tasks", default="2-1,2-10,demand")
    ap.add_argument("--grid", default="1e-5:3000,1e-6:3000", help="lr:steps pairs searched on validation (pre-registration amendment c)")
    ap.add_argument("--n-train", type=int, default=200_000, help="training windows sampled per seed")
    ap.add_argument("--workdir", default=".chronos_ft")
    ap.add_argument("--long-context", default="", help="calendar: extra zero-shot runs with these context lengths (M5: 512)")
    ap.add_argument("--n-train-groups", type=int, default=50_000, help="training windows sampled per seed for group fine-tuning (each becomes a multivariate group)")
    args = ap.parse_args()
    {"calendar": run_calendar, "official": run_official, "finetune-calendar": run_finetune_calendar,
     "finetune-official": run_finetune_official, "group-calendar": run_group_calendar, "group-official": run_group_official}[args.protocol](args)


if __name__ == "__main__":
    main()
