"""The official VISUELLE 2.0 short-observation protocol (Skenderi et al., CVPRW 2022; github.com/HumaticsLAB/visuelle2.0-code).

    python examples/visuelle2_official.py --panel panel_v2_full.pkl --src visuelle2 --seeds 1,2,3,4,5 --out results_visuelle2_official
    python examples/visuelle2_official.py --report --out results_visuelle2_official

Protocol, reproduced from the official code:
  * Split: stfore_train.csv / stfore_test.csv (test = the 10 % most recent item-shop pairs). Values are sales / 53
    (stfore_sales_norm_scalar.npy); we multiply back, so MAE is in units.
  * Cleaning (SO-fore tasks only): if a pair's 12-week sales exceed its restock total, weeks from the first one where
    cumulative sales exceed the restock are set to 0 (dataset.py:frame_series). The demand task uses raw sales.
  * SO-fore 2-1: forecast week k from weeks k-2, k-1, for k = 3..12 (10 one-step forecasts per pair).
    SO-fore 2-10: forecast weeks 3..12 from weeks 1-2. Demand: forecast weeks 1..12 with no own sales history.
  * WAPE = 100 * sum|y - yhat| / sum y and MAE = mean |y - yhat| over all test cells.

Our models see exactly the protocol's own-history window (encoder length 2). They also see calendar covariates,
discounts, weather, and, for the sibling variants, the same product's sales in *other* shops over those two
calendar weeks, which happened before the forecast origin. Validation for early stopping is the 10 % most recent
*training* pairs (in the released code the validation loader is built from the test split). Test pairs are removed from the
panel that training and validation see, including as sibling or analogue inputs.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightning.pytorch as pl

from benchmark import ALL_MODELS, QUANTILES, add_training_args, lifecycle_features, lifecycle_for, model_config, predict, product_features, setup_device, slug, train_module
from mtft_plus.data import DemandWindowDataset
from mtft_plus.lightning_module import MTFTLightningModule

W = [str(i) for i in range(12)]
TASKS = {  # own-history window L, horizon H, first origin (weeks after release), origins per pair, gt kind
    "2-1": dict(L=2, H=1, first=2, n_origins=10, gt="clean"),
    "2-10": dict(L=2, H=10, first=2, n_origins=1, gt="clean"),
    "demand": dict(L=2, H=12, first=0, n_origins=1, gt="raw"),
}
DEFAULT_MODELS = ["TFT", "MTFT-Picnic", "MTFT-Picnic + siblings", "MTFT+ v3 (ours)"]
# Published results (Skenderi et al. 2022, Tables 1-2), WAPE % / MAE. In the released training code the validation loader is built from the test split.
PUBLISHED = {
    "2-10": [("Naive", 118.176, 1.31), ("SES", 111.265, 1.23), ("kNN", 91.13, 0.98), ("kNN + image", 97.97, 1.06),
             ("CrossAttnRNN", 35.13, 0.39), ("CrossAttnRNN w/ image", 32.25, 0.36)],
    "2-1": [("Naive", 101.922, 1.13), ("SES", 97.85, 1.08), ("kNN", 87.11, 0.94), ("kNN + image", 88.97, 0.96),
            ("CrossAttnRNN", 23.20, 0.26), ("CrossAttnRNN w/ images", 23.70, 0.26)],
    "demand": [("CrossAttnRNN w/ image", 83.33, 0.97)],
}


# ---------------------------------------------------------------------------------------------------- official data
def official_pairs(src: Path) -> pd.DataFrame:
    """One row per official item-shop pair: split, release date, raw and restock-cleaned 12-week sales (units)."""
    ns = float(np.load(src / "stfore_sales_norm_scalar.npy"))
    parts = []
    for split, f in (("train", "stfore_train.csv"), ("test", "stfore_test.csv")):
        df = pd.read_csv(src / f, parse_dates=["release_date"])
        raw = df[W].to_numpy(float) * ns
        restock = df["restock"].to_numpy(float) * ns
        clean = raw.copy()
        over = raw.sum(1) > restock  # dataset.py:frame_series
        cs = np.cumsum(raw, 1)
        clean[over] = np.where(cs[over] > restock[over, None], 0.0, raw[over])
        parts.append(pd.DataFrame({"external_code": df.external_code.astype(str), "retail": df.retail.astype(str),
                                   "release_date": df.release_date, "split": split,
                                   "raw": list(raw), "clean": list(clean)}))
    return pd.concat(parts, ignore_index=True)


def ses2(x1: np.ndarray, x2: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """SES forecast from two observations with fixed alpha and least-squares initial level, in closed form.

    One-step fits: yhat1 = l0, l1 = a x1 + (1-a) l0, yhat2 = l1. Minimising (x1-l0)^2 + (x2-l1)^2 over l0 gives
    l0 = (x1 + b (x2 - a x1)) / (1 + b^2) with b = 1-a; the forecast (flat for all horizons) is l2 = a x2 + b l1.
    Matches SimpleExpSmoothing(x, initialization_method="estimated").fit(smoothing_level=a) (tests/verify below);
    the official Oracle.py used the pre-0.12 statsmodels API, whose default also estimated the initial level."""
    b = 1.0 - alpha
    l0 = (x1 + b * (x2 - alpha * x1)) / (1 + b * b)
    l1 = alpha * x1 + b * l0
    return alpha * x2 + b * l1


def baselines(pairs: pd.DataFrame) -> dict:
    """Naive and SES exactly as models/Oracle.py (SES: smoothing_level=0.3, initial level optimised)."""
    te = np.stack(pairs.loc[pairs.split == "test", "clean"].to_numpy())
    y = te[:, 2:]
    out = {}
    naive21 = te[:, 1:11]
    naive210 = np.repeat(te[:, 1:2], 10, 1)
    ses21 = ses2(te[:, 0:10], te[:, 1:11])  # window k: weeks k-2, k-1 -> week k
    ses210 = np.repeat(ses2(te[:, 0:1], te[:, 1:2]), 10, 1)
    try:  # cross-check the closed form against statsmodels on a sample
        from statsmodels.tsa.api import SimpleExpSmoothing

        rng = np.random.default_rng(0)
        for i in rng.choice(len(te), 50, replace=False):
            x = te[i, 3:5]
            if x.std() > 0:
                sm = SimpleExpSmoothing(x, initialization_method="estimated").fit(smoothing_level=0.3, optimized=True).forecast(1)[0]
                assert abs(sm - ses21[i, 3]) < 1e-3 * max(1.0, abs(sm)), (sm, ses21[i, 3])
    except ImportError:
        pass
    for task, name, p in (("2-1", "Naive", naive21), ("2-1", "SES", ses21), ("2-10", "Naive", naive210), ("2-10", "SES", ses210)):
        out.setdefault(task, {})[name] = metrics(y, p)
    raw = np.stack(pairs.loc[pairs.split == "test", "raw"].to_numpy())
    tr = np.stack(pairs.loc[pairs.split == "train", "raw"].to_numpy())
    out["demand"] = {"train mean curve": metrics(raw, np.repeat(tr.mean(0, keepdims=True), len(raw), 0)),
                     "train median curve": metrics(raw, np.repeat(np.median(tr, 0, keepdims=True), len(raw), 0))}
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"WAPE": float(100 * np.abs(y - p).sum() / y.sum()), "MAE": float(np.abs(y - p).mean())}


# ---------------------------------------------------------------------------------------------------- panel prep
class Protocol:
    def __init__(self, panel_path: str, src: Path, val_frac: float = 0.1) -> None:
        with open(panel_path, "rb") as fh:
            d = pickle.load(fh)
        if d.live is None or "article_ids" not in d.attributes:
            raise SystemExit("panel needs series windows and article ids (examples/prepare_panel.py on convert_visuelle2.py output)")
        pairs = official_pairs(src)
        a_pos = {a: i for i, a in enumerate(map(str, d.attributes["article_ids"]))}
        f_pos = {f: i for i, f in enumerate(map(str, d.fc_names))}
        pairs["a"] = pairs.external_code.map(a_pos)
        pairs["f"] = pairs.retail.map(f_pos)
        if pairs[["a", "f"]].isna().any().any():
            raise SystemExit("official pairs missing from the panel; convert with --end at or after the last sales week (2020-03-16)")
        pairs[["a", "f"]] = pairs[["a", "f"]].astype(int)
        pairs["r"] = d.launch_loc[pairs.a, pairs.f]
        assert (d.dates[pairs.r] == pairs.release_date).all(), "panel launch weeks do not match official release dates"
        T = d.y.shape[2]
        if (pairs.r + 12 > T).any():
            raise SystemExit("panel ends before the last test week; convert with --end 2020-03-16")
        # restock-cleaned sales (negatives clipped) for every official pair, as the official SO-fore models are trained on
        a, f, r = pairs.a.to_numpy(), pairs.f.to_numpy(), pairs.r.to_numpy()
        tt = r[:, None] + np.arange(12)[None]
        d.y[a[:, None], f[:, None], tt] = np.clip(np.stack(pairs.clean.to_numpy()), 0, None).astype(np.float32)
        # validation = most recent val_frac of training pairs (mirrors how the official test set was drawn)
        trn = pairs[pairs.split == "train"].sort_values(["release_date", "external_code", "retail"], kind="stable")
        n_val = int(round(val_frac * len(trn)))
        pairs["role"] = pairs.split
        pairs.loc[trn.index[-n_val:], "role"] = "val"
        self.pairs, self.d_full = pairs.reset_index(drop=True), d
        # panel seen by training/validation: test pairs removed entirely (no targets, no sibling/analogue inputs)
        te = self.pairs[self.pairs.role == "test"]
        d_fit = copy.copy(d)
        d_fit.y, d_fit.live = d.y.copy(), d.live.copy()
        d_fit.y[te.a.to_numpy(), te.f.to_numpy()] = 0.0
        d_fit.live[te.a.to_numpy(), te.f.to_numpy()] = False
        d_fit.launch_loc = d.launch_loc.copy()  # test pairs are not launched as far as training can tell
        d_fit.launch_loc[te.a.to_numpy(), te.f.to_numpy()] = d.y.shape[2]
        self.d_fit = d_fit
        self._lifecycle = None
        seen = np.zeros(d.n_articles, bool)
        seen[self.pairs.loc[self.pairs.role != "test", "a"].unique()] = True
        self.warm = seen  # PCA / analogue pool: products with at least one training pair
        te_codes = set(te.external_code)
        tr_codes = set(self.pairs.loc[self.pairs.role != "test", "external_code"])
        self.pairs["new_product"] = ~self.pairs.external_code.isin(tr_codes)
        print(f"official pairs: train {int((self.pairs.role == 'train').sum()):,} | val {n_val:,} | test {len(te):,} "
              f"({len(te_codes):,} products, {len(te_codes - tr_codes)} never seen in training)", flush=True)

    def lifecycle(self, mm_all: dict) -> dict:
        """LA features from the training panel (test pairs removed) and from the full panel (test windows); both use
        the same retrieval indices. Seed-independent, so built once."""
        if self._lifecycle is None:
            t0 = time.time()
            self._lifecycle = {"fit": lifecycle_features(self.d_fit, mm_all), "full": lifecycle_features(self.d_full, mm_all)}
            print(f"lifecycle-aligned features ready in {time.time() - t0:.1f}s", flush=True)
        return self._lifecycle

    @staticmethod
    def with_windows(d, L: int, H: int):
        d = copy.copy(d)
        d.cfg = dataclasses.replace(d.cfg, encoder_length=L, horizon=H)
        return d

    def index(self, role: str, task: dict) -> np.ndarray:
        p = self.pairs[self.pairs.role == role]
        o = p.r.to_numpy()[:, None] + task["first"] + np.arange(task["n_origins"])[None]
        idx = np.stack([np.repeat(p.a.to_numpy(), task["n_origins"]), np.repeat(p.f.to_numpy(), task["n_origins"]), o.ravel()], 1)
        if role != "test":
            idx = idx[idx[:, 2] - task["L"] >= 0]  # the grid starts at the first release: no negative encoder steps
        return idx.astype(np.int64)


def run(proto: Protocol, task_name: str, model: str, seed: int, args, mm_all, an_all) -> dict:
    task = TASKS[task_name]
    spec = ALL_MODELS[model]
    pl.seed_everything(seed, verbose=False)
    d_fit = proto.with_windows(proto.d_fit, task["L"], task["H"])
    d_test = proto.with_windows(proto.d_full, task["L"], task["H"])
    mm = mm_all[spec["mm"]]
    an = an_all[spec["analogues"]] if spec.get("analogues") else None
    sib = spec.get("siblings", False)
    lf_fit, la_pool, la_tok = lifecycle_for(proto.lifecycle(mm_all)["fit"], spec)
    lf_full, _, _ = lifecycle_for(proto.lifecycle(mm_all)["full"], spec)
    la = dict(lifecycle=lf_fit, la_pool=la_pool, la_tokens=la_tok)
    la_test = dict(lifecycle=lf_full, la_pool=la_pool, la_tokens=la_tok)
    cfg = model_config(d_fit, spec, args, an is not None, mm_all["picnic"]["scalar"].shape[1], task["L"], task["H"], lf_fit.emb.shape[1] if lf_fit is not None else 0)
    module = MTFTLightningModule(cfg, learning_rate=args.lr)
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = proto.index("train", task), proto.index("val", task)
    va_idx = va_idx[rng.permutation(len(va_idx))[:8192]]
    best, train_s = train_module(module, DemandWindowDataset(d_fit, tr_idx, mm, an, siblings=sib, **la), DemandWindowDataset(d_fit, va_idx, mm, an, siblings=sib, **la), seed, args)
    te_idx = proto.index("test", task)
    q, _, _ = predict(module, DemandWindowDataset(d_test, te_idx, mm, an, siblings=sib, **la_test), args.device)
    module.cpu()
    te = proto.pairs[proto.pairs.role == "test"]
    n = len(te)
    med = q[..., QUANTILES.index(0.5)].reshape(n, task["n_origins"] * task["H"])  # (pairs, cells) in week order
    gt_all = np.stack(te[task["gt"]].to_numpy())
    y = gt_all[:, task["first"] : task["first"] + med.shape[1]]
    new = te.new_product.to_numpy()
    fdir = Path(args.out) / "forecasts"
    fdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fdir / f"{task_name}__{seed}__{slug(model)}.npz", q=q.reshape(n, -1, len(QUANTILES)).astype(np.float32),
                        product=te.external_code.to_numpy().astype(str), retail=te.retail.to_numpy().astype(str))
    res = {
        "task": task_name, "model": model, "seed": seed, "spec": spec, "train_seconds": round(train_s, 1),
        "best_val": best.best, "val_curve": best.history, "n_train_windows": int(len(tr_idx)),
        "all": metrics(y, med), "new_product": metrics(y[new], med[new]), "seen_product": metrics(y[~new], med[~new]),
        "coverage80": float(((y >= q[..., 0].reshape(med.shape)) & (y <= q[..., -1].reshape(med.shape))).mean()),
        "per_pair": {"abs_err": np.abs(y - med).sum(1).round(4).tolist(), "y": y.sum(1).round(4).tolist(),
                     "product": te.external_code.tolist(), "new_product": new.tolist()},
    }
    m = res["all"]
    print(f"[{task_name} seed {seed}] {model:26s} train {train_s:4.0f}s val {best.best:.4f} | WAPE {m['WAPE']:6.2f} MAE {m['MAE']:.3f} "
          f"| new-product WAPE {res['new_product']['WAPE']:6.2f} | seen {res['seen_product']['WAPE']:6.2f}", flush=True)
    return res


# ---------------------------------------------------------------------------------------------------- report
CONTROLS = [("MTFT-Picnic + siblings", "MTFT-Picnic"), ("MTFT-Picnic", "TFT"), ("MTFT+ v3 (ours)", "MTFT-Picnic + siblings")]


def paired_product_bootstrap(ra: dict, rb: dict, n_boot: int = 2000, seed: int = 0, only_new: bool = False):
    """WAPE(a) - WAPE(b), pooled over seeds; test products resampled with replacement (paired, stratified by seed)."""
    rng = np.random.default_rng(seed)
    per_seed = []
    zero_shot = set(rb) == {0} and rb[0].get("spec", {}).get("zero_shot")  # deterministic comparator: pair with every seed
    for s in (sorted(ra) if zero_shot else sorted(set(ra) & set(rb))):
        pa, pb = ra[s]["per_pair"], rb[0 if zero_shot else s]["per_pair"]
        df = pd.DataFrame({"p": pa["product"], "ea": pa["abs_err"], "eb": pb["abs_err"], "y": pa["y"], "new": pa["new_product"]})
        if only_new:
            df = df[df.new]
        df = df.drop(columns="new").groupby("p").sum()
        per_seed.append(df.to_numpy())
    stat = lambda parts: 100 * sum(x[:, 0].sum() - x[:, 1].sum() for x in parts) / sum(x[:, 2].sum() for x in parts)  # noqa: E731
    point = stat(per_seed)
    boots = [stat([x[rng.integers(0, len(x), len(x))] for x in per_seed]) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, lo, hi


def cov80(rs) -> str:
    """Mean 80 % interval coverage, or '–' for point-forecast runs (the re-run published baselines)."""
    v = [r["coverage80"] for r in rs if "coverage80" in r]
    return f"{np.mean(v):.2f}" if len(v) == len(rs) else "–"


def report(out: Path, ours: str = "MTFT+ v3 (ours)") -> str:
    runs = {}
    for p in sorted((out / "partials").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs.setdefault(r["task"], {}).setdefault(r["model"], {})[r["seed"]] = r
    base = json.loads((out / "baselines.json").read_text(encoding="utf-8")) if (out / "baselines.json").exists() else {}
    ms = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else f"{np.mean(v):.2f}"  # noqa: E731
    md = "# VISUELLE 2.0, official short-observation protocol\n\nWAPE in %, MAE in units, mean ± sd over seeds. Validation = most recent 10 % of training pairs.\n"
    for task in TASKS:
        if task not in runs:
            continue
        md += f"\n## {task}\n\n| Model | seeds | WAPE | MAE | new-product WAPE | seen-product WAPE | Cov80 |\n|---|---|---|---|---|---|---|\n"
        for name, m in base.get(task, {}).items():
            md += f"| {name} (our re-run) | – | {m['WAPE']:.2f} | {m['MAE']:.3f} | | | |\n"
        order = [m for m in DEFAULT_MODELS if m in runs[task]] + [m for m in runs[task] if m not in DEFAULT_MODELS]
        for model in order:
            rs = list(runs[task][model].values())
            g = lambda k, sub="all": [r[sub][k] for r in rs]  # noqa: E731
            md += f"| {model} | {len(rs)} | {ms(g('WAPE'))} | {ms(g('MAE'))} | {ms(g('WAPE', 'new_product'))} | {ms(g('WAPE', 'seen_product'))} | {cov80(rs)} |\n"
        if ours in runs[task]:
            md += f"\nPaired product-cluster bootstrap, {ours} − comparator (WAPE points, 95 % CI), and per-seed wins:\n\n| Comparator | ΔWAPE | wins |\n|---|---|---|\n"
            for model in order:
                if model == ours:
                    continue
                pt, lo, hi = paired_product_bootstrap(runs[task][ours], runs[task][model])
                rb = runs[task][model]
                zs = set(rb) == {0} and rb[0].get("spec", {}).get("zero_shot")
                common = sorted(runs[task][ours]) if zs else sorted(set(runs[task][ours]) & set(rb))
                wins = sum(runs[task][ours][s]["all"]["WAPE"] < rb[0 if zs else s]["all"]["WAPE"] for s in common)
                sig = "**" if hi < 0 or lo > 0 else ""
                md += f"| {model} | {sig}{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig} | {wins}/{len(common)} |\n"
        ctrl = [(a, b) for a, b in CONTROLS if a in runs[task] and b in runs[task]]
        if ctrl:
            md += "\nControls, A − B (WAPE points, 95 % CI; all test pairs / new products only), per-seed wins of A:\n\n| A | B | ΔWAPE all | ΔWAPE new products | wins |\n|---|---|---|---|---|\n"
            for a, b in ctrl:
                ra, rb = runs[task][a], runs[task][b]
                cells = []
                for only_new in (False, True):
                    pt, lo, hi = paired_product_bootstrap(ra, rb, only_new=only_new)
                    sig = "**" if hi < 0 or lo > 0 else ""
                    cells.append(f"{sig}{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig}")
                common = sorted(set(ra) & set(rb))
                wins = sum(ra[s]["all"]["WAPE"] < rb[s]["all"]["WAPE"] for s in common)
                md += f"| {a} | {b} | {cells[0]} | {cells[1]} | {wins}/{len(common)} |\n"
        md += "\nPublished (Skenderi et al. 2022; in the released training code the validation loader is built from the test split):\n\n| Method | WAPE | MAE |\n|---|---|---|\n"
        md += "".join(f"| {n} | {w} | {m} |\n" for n, w, m in PUBLISHED[task])
    (out / "results.md").write_text(md, encoding="utf-8")
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel_v2_full.pkl")
    ap.add_argument("--src", default="visuelle2")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--tasks", default="2-1,2-10,demand")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--out", default="results_visuelle2_official")
    ap.add_argument("--report", action="store_true", help="only aggregate existing partials")
    ap.add_argument("--ours", default="MTFT+ v3 (ours)", help="--report: the model compared against every other row")
    add_training_args(ap)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "partials").mkdir(parents=True, exist_ok=True)
    if args.report:
        sys.stdout.buffer.write(report(out, args.ours).encode("utf-8"))
        return
    setup_device(args)
    t0 = time.time()
    proto = Protocol(args.panel, Path(args.src))
    if not (out / "baselines.json").exists():
        b = baselines(proto.pairs)
        (out / "baselines.json").write_text(json.dumps(b, indent=1))
        print("baselines:", json.dumps(b), flush=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for seed in [int(s) for s in args.seeds.split(",")]:
        mm_all, an_all = product_features(proto.d_full, proto.warm, args.k, seed)
        for task in [t.strip() for t in args.tasks.split(",")]:
            for model in models:
                path = out / "partials" / f"{task}__{seed}__{slug(model)}.json"
                if path.exists():
                    continue
                res = run(proto, task, model, seed, args, mm_all, an_all)
                res["args"] = {k: (str(v) if k == "device" else v) for k, v in vars(args).items()}
                path.write_text(json.dumps(res))
    sys.stdout.buffer.write(report(out).encode("utf-8"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
