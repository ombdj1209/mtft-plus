"""Aggregate results/partials/*.json into results/results.md and results/summary.json.

Uncertainty: mean ± sd over data seeds, and a paired article-cluster bootstrap (stratified by seed)
for the difference in WAPE and wQL between the proposed model and each comparator. A difference is
called significant only if its 95 % interval excludes zero.
"""
from __future__ import annotations

import argparse
import sys
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OURS = "MTFT+ v3 (ours)"
ORDER = ["TFT", "MTFT-Picnic", "TFT + category analogues", "MTFT-Picnic + RA", "MTFT-Picnic + siblings", "MTFT+ v2 (ours)", OURS,
         "ours w/o cross-attention", "ours w/o retrieval (v1 + reg.)"]
N_Q = 3


def ms(vals, fmt="{:.4f}"):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
    if v.size == 0:
        return "n/a"
    return fmt.format(v.mean()) + (f" ± {v.std(ddof=1):.4f}" if v.size > 1 else "")


def bootstrap_diff(runs_a, runs_b, subset: str, metric: str, n_boot: int = 2000, seed: int = 0):
    """Paired over articles within each seed; returns (point diff a-b, lo, hi) of the pooled metric."""
    rng = np.random.default_rng(seed)
    per_seed = []
    zero_shot = set(runs_b) == {0} and runs_b[0].get("spec", {}).get("zero_shot")  # deterministic: pair with every seed
    for s in (sorted(runs_a) if zero_shot else sorted(set(runs_a) & set(runs_b))):
        pa, pb = runs_a[s]["per_article"], runs_b[0 if zero_shot else s]["per_article"]
        cold = np.asarray(pa["cold"])
        sel = {"all": np.ones_like(cold, bool), "cold-start": cold, "warm": ~cold}[subset]
        # "WAPE" / "wQL": article x shop cells; "WAPE_an" / "wQL_an": article-national totals of the bottom-up forecasts
        pre = "an_" if metric.endswith("_an") else ""
        if pre and (f"{pre}abs_err" not in pa or f"{pre}abs_err" not in pb):
            return None
        key = pre + ("abs_err" if metric.startswith("WAPE") else "pinball")
        per_seed.append((np.asarray(pa[key])[sel], np.asarray(pb[key])[sel], np.asarray(pa[pre + "abs_y"])[sel]))
    if not per_seed:
        return None

    def stat(parts):
        ea = sum(p[0].sum() for p in parts)
        eb = sum(p[1].sum() for p in parts)
        y = sum(p[2].sum() for p in parts)
        f = 1.0 if metric.startswith("WAPE") else 2.0 / N_Q
        return f * (ea - eb) / y

    point = stat(per_seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        parts = []
        for ea, eb, y in per_seed:
            idx = rng.integers(0, len(y), len(y))
            parts.append((ea[idx], eb[idx], y[idx]))
        boots[i] = stat(parts)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, lo, hi


def seed_wins(runs_a, runs_b) -> str:
    """'k/n': seeds on which A has the lower overall WAPE; a zero-shot B (seed 0 only) is compared with every seed of A."""
    zs = set(runs_b) == {0} and runs_b[0].get("spec", {}).get("zero_shot")
    common = sorted(runs_a) if zs else sorted(set(runs_a) & set(runs_b))
    w = lambda r: next(b for b in r["bottom"] if b["subset"] == "all")["WAPE"]  # noqa: E731
    return f"{sum(w(runs_a[s]) < w(runs_b[0 if zs else s]) for s in common)}/{len(common)}"


def main() -> None:
    global OURS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--ours", default=OURS, help="model the bootstrap compares against every other model")
    ap.add_argument("--readme", default="", help="also splice the tables into this README between RESULTS markers")
    ap.add_argument("--extra-pairs", default="", help='controls to test, e.g. "MTFT-Picnic + siblings|MTFT-Picnic,..." (A|B)')
    args = ap.parse_args()
    OURS = args.ours
    out = Path(args.out)
    runs = defaultdict(dict)
    for p in sorted((out / "partials").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs[r["model"]][r["seed"]] = r
    models = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]
    main_models = [m for m in models if len(runs[m]) > 1 or m in ORDER[:5]]
    seeds = sorted({s for m in runs for s in runs[m]})

    md = f"# MTFT+ v3 results\n\nData seeds: {seeds}. Mean ± sd across seeds (sd omitted for single-seed rows).\n\n"
    nis = any(b["subset"] == "new-in-shop" for m in models for r in runs[m].values() for b in r["bottom"])
    md += "## Article × FC level (test)\n\n| Model | seeds | val loss | WAPE | MASE | wQL | Cov80 | cold-start WAPE | cold-start wQL |" + (" new-in-shop WAPE |" if nis else "")
    md += "\n|---|---|---|---|---|---|---|---|---|" + ("---|" if nis else "") + "\n"
    for m in models:
        rs = list(runs[m].values())
        g = lambda sub, k: [next(b for b in r["bottom"] if b["subset"] == sub)[k] for r in rs]  # noqa: E731
        md += f"| {m} | {len(rs)} | {ms([r['best_val'] for r in rs])} | {ms(g('all', 'WAPE'))} | {ms(g('all', 'MASE'))} | {ms(g('all', 'wQL'))} | {ms(g('all', 'Cov80'), '{:.3f}')} | {ms(g('cold-start', 'WAPE'))} | {ms(g('cold-start', 'wQL'))} |"
        md += (f" {ms(g('new-in-shop', 'WAPE'))} |" if nis else "") + "\n"

    md += "\n## Paired article-cluster bootstrap: ours − comparator (negative = ours better), 95 % CI\n\n| Comparator | seeds | ΔWAPE all | ΔwQL all | ΔWAPE cold-start | ΔwQL cold-start |\n|---|---|---|---|---|---|\n"
    boot = {}
    if OURS in runs:
        for m in models:
            if m == OURS:
                continue
            cells = []
            for sub in ("all", "cold-start"):
                for met in ("WAPE", "wQL"):
                    r = bootstrap_diff(runs[OURS], runs[m], sub, met)
                    boot[(m, sub, met)] = r
                    if r is None:
                        cells.append("n/a")
                    else:
                        sig = "**" if (r[2] < 0 or r[1] > 0) else ""
                        cells.append(f"{sig}{r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]{sig}")
            zs = set(runs[m]) == {0} and runs[m][0].get("spec", {}).get("zero_shot")
            n = f"{len(runs[OURS])} (zero-shot)" if zs else len(set(runs[OURS]) & set(runs[m]))
            md += f"| {m} | {n} | " + " | ".join([cells[0], cells[1], cells[2], cells[3]]) + f" | {seed_wins(runs[OURS], runs[m])} |\n"
        md = md.replace("| ΔWAPE cold-start | ΔwQL cold-start |\n|---|---|---|---|---|---|\n", "| ΔWAPE cold-start | ΔwQL cold-start | per-seed WAPE wins |\n|---|---|---|---|---|---|---|\n", 1)
        md += "\nBold = 95 % interval excludes zero. Wins = seeds on which the first model has the lower WAPE (the bootstrap resamples articles, not training runs).\n"
    pairs = [tuple(x.split("|")) for x in args.extra_pairs.split(",") if "|" in x]
    pairs = [(a.strip(), b.strip()) for a, b in pairs if a.strip() in runs and b.strip() in runs]
    if pairs:
        md += "\n## Controls: A − B (negative = A better), 95 % CI\n\n| A | B | ΔWAPE all | ΔwQL all | ΔWAPE cold-start | per-seed WAPE wins (A) |\n|---|---|---|---|---|---|\n"
        for a, b in pairs:
            cells = []
            for sub, met in (("all", "WAPE"), ("all", "wQL"), ("cold-start", "WAPE")):
                r = bootstrap_diff(runs[a], runs[b], sub, met)
                sig = "**" if r is not None and (r[2] < 0 or r[1] > 0) else ""
                cells.append("n/a" if r is None else f"{sig}{r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]{sig}")
            md += f"| {a} | {b} | " + " | ".join(cells) + f" | {seed_wins(runs[a], runs[b])} |\n"
    an_pairs = [(OURS, m) for m in models if m != OURS] + pairs
    rows = []
    for a, b in an_pairs:
        if a in runs and b in runs:
            cells = []
            for sub, met in (("all", "WAPE_an"), ("all", "wQL_an"), ("cold-start", "WAPE_an")):
                r = bootstrap_diff(runs[a], runs[b], sub, met)
                sig = "**" if r is not None and (r[2] < 0 or r[1] > 0) else ""
                cells.append("n/a" if r is None else f"{sig}{r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]{sig}")
            if any(c != "n/a" for c in cells):
                rows.append(f"| {a} | {b} | " + " | ".join(cells) + " |\n")
    if rows:
        md += "\n## Product level (article-national totals of the bottom-up forecasts): A − B, 95 % CI\n\n| A | B | ΔWAPE | ΔwQL | ΔWAPE cold-start |\n|---|---|---|---|---|\n" + "".join(rows)

    md += "\n## Hierarchy: WAPE / wQL / Cov80 by level (MinT + conformal unless stated)\n\n| Model | Method | national | fulfilment centre | article-national | article × FC |\n|---|---|---|---|---|---|\n"
    levels = ["national", "fulfilment_centre", "article_national", "article_x_fc"]
    has_wls = any(h["method"] == "MinT-WLS" for m in models for r in runs[m].values() for h in r["hierarchy"])
    for m in models:
        methods = ["bottom-up", "base", "MinT", "MinT + conformal"] if m == OURS else ["MinT + conformal"]
        if has_wls:
            methods += ["MinT-WLS", "MinT-WLS + conformal"] if m == OURS else ["MinT-WLS + conformal"]
        for meth in methods:
            cells = []
            for lvl in levels:
                rows = [next(h for h in r["hierarchy"] if h["method"] == meth and h["level"] == lvl) for r in runs[m].values()]
                cells.append(f"{np.mean([x['WAPE'] for x in rows]):.4f} / {np.mean([x['wQL'] for x in rows]):.4f} / {np.mean([x['Cov80'] for x in rows]):.2f}")
            md += f"| {m} | {meth} | " + " | ".join(cells) + " |\n"
    lam = [r["mint_lambda"] for m in models for r in runs[m].values()]
    coh = max(r["coherence_error"] for m in models for r in runs[m].values())
    md += f"\nMinT λ̂ range {min(lam):.3f}–{max(lam):.3f}; max coherence error {coh:.1e}.\n"
    choices = [h for m in models for r in runs[m].values() for h in r["hierarchy"] if h["method"] == "_val_choice"]
    if choices:
        c = [x["choice"] for x in choices]
        md += (f"\nValidation-only choice between MinT-shrink and MinT-WLS (fit on the first half of the validation origins, national + FC WAPE "
               f"on the second half): shrink {c.count('shrink')} / WLS {c.count('wls')} of {len(c)} runs "
               f"(mean val WAPE shrink {np.mean([x['shrink'] for x in choices]):.3f}, WLS {np.mean([x['wls'] for x in choices]):.3f}).\n")

    if OURS in runs:
        md += "\n## Modality attribution (ours): image / text / fused\n\n| seed | promo days | non-promo days |\n|---|---|---|\n"
        for s, r in sorted(runs[OURS].items()):
            a = r.get("modality_attribution")
            if a:
                f = lambda d: f"{d['image']:.3f} / {d['text']:.3f} / {d['fused']:.3f}"  # noqa: E731
                md += f"| {s} | {f(a['promo'])} | {f(a['non_promo'])} |\n"

    (out / "results.md").write_text(md, encoding="utf-8")
    if args.readme:
        rp = Path(args.readme)
        text = rp.read_text(encoding="utf-8")
        b, e = "<!--RESULTS:BEGIN-->", "<!--RESULTS:END-->"
        if b in text and e in text:
            body = md.split("\n", 2)[2].replace("\n## ", "\n### ")
            rp.write_text(text[: text.index(b) + len(b)] + "\n" + body + "\n" + text[text.index(e) :], encoding="utf-8")
    summary = {
        m: {str(s): {"bottom": r["bottom"], "hierarchy": r["hierarchy"], "best_val": r["best_val"], "params": r["params"]} for s, r in runs[m].items()}
        for m in models
    }
    summary["_bootstrap_ours_minus"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in boot.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))


if __name__ == "__main__":
    main()
