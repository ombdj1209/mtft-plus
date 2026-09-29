"""Turn real retail CSVs + product images into a panel for `benchmark.py --panel`.

    python examples/prepare_panel.py --data-dir my_data --out panel.pkl                   # encode with SigLIP
    python examples/prepare_panel.py --data-dir my_data --embeddings bank.pt --out panel.pkl

Expected files in --data-dir (see mtft_plus/panel.py for column details):
    sales.csv      article_id, location_id, date, units [, promo, discount]
    catalog.csv    article_id, category [, price, shelf_life_days, launch_date, text, image_path]
    locations.csv  location_id, country
    weather.csv    (optional) location_id, date, temperature, precipitation
    holidays.csv   (optional) country, date
    series.csv     (optional) article_id, location_id, launch_date [, end_date]: per-location launches and
                   observation windows (see mtft_plus/panel.py)
Optional unimodal baseline features (for an MTFT-style baseline with separate encoders):
    unimodal_image.npy / unimodal_text.npy   (A, D) arrays aligned with catalog.csv row order
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without installing

import numpy as np
import pandas as pd

from mtft_plus.features import ArticleRecord, EmbeddingBank, MultimodalFeatureExtractor
from mtft_plus.panel import load_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embeddings", default="", help="precomputed EmbeddingBank (.pt); skips encoding")
    ap.add_argument("--model-name", default="google/siglip-base-patch16-224")
    ap.add_argument("--backend", default="siglip", choices=["siglip", "open_clip"])
    ap.add_argument("--cache-dir", default=".mm_cache")
    ap.add_argument("--freq", default="D")
    # window defaults depend on --freq: daily 56/14/21, weekly 12/4/3 (placeholders; set from the dataset protocol)
    ap.add_argument("--encoder-length", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--val-origins", type=int, default=4)
    ap.add_argument("--test-origins", type=int, default=4)
    ap.add_argument("--cold-start-steps", type=int, default=None)
    # seasonal settings; default from --freq (mtft_plus.panel.freq_defaults)
    ap.add_argument("--season", type=int, default=None, help="seasonal-naive / ridge anchor period in steps (daily 7, weekly 1)")
    ap.add_argument("--level-window", type=int, default=None)
    ap.add_argument("--min-history", type=int, default=None)
    ap.add_argument("--max-articles", type=int, default=0, help="random subset of articles (seed 0) for smoke runs")
    args = ap.parse_args()
    weekly = args.freq.upper().startswith("W")
    win = dict(encoder_length=12, horizon=4, cold=3) if weekly else dict(encoder_length=56, horizon=14, cold=21)
    args.encoder_length = args.encoder_length or win["encoder_length"]
    args.horizon = args.horizon or win["horizon"]
    args.cold_start_steps = args.cold_start_steps if args.cold_start_steps is not None else win["cold"]

    d = Path(args.data_dir)
    read = lambda n: pd.read_csv(d / n) if (d / n).exists() else None  # noqa: E731
    sales, catalog, locations = read("sales.csv"), read("catalog.csv"), read("locations.csv")
    if sales is None or catalog is None or locations is None:
        raise SystemExit("sales.csv, catalog.csv and locations.csv are required")
    catalog["article_id"] = catalog["article_id"].astype(str)
    series = read("series.csv")
    if args.max_articles and args.max_articles < len(catalog):
        catalog = catalog.sample(n=args.max_articles, random_state=0).sort_index()
        keep = set(catalog["article_id"])
        sales = sales[sales["article_id"].astype(str).isin(keep)]
        series = series[series["article_id"].astype(str).isin(keep)] if series is not None else None
        print(f"subset: {len(catalog)} articles")

    if args.embeddings:
        bank = EmbeddingBank.load(args.embeddings)
    else:
        recs = []
        for r in catalog.itertuples(index=False):
            img = getattr(r, "image_path", None)
            img = str(d / img) if isinstance(img, str) and img and (d / img).exists() else None
            txt = getattr(r, "text", None)
            recs.append(ArticleRecord(str(r.article_id), txt if isinstance(txt, str) else None, img))
        bank = MultimodalFeatureExtractor(args.model_name, backend=args.backend, cache_dir=args.cache_dir).encode(recs)
        bank.save(Path(args.out).with_suffix(".embeddings.pt"))

    rows = catalog.index.to_numpy()  # original catalog.csv rows (a subset keeps its row numbers)
    uni_i = np.load(d / "unimodal_image.npy")[rows] if (d / "unimodal_image.npy").exists() else None
    uni_t = np.load(d / "unimodal_text.npy")[rows] if (d / "unimodal_text.npy").exists() else None
    panel = load_panel(
        sales, catalog, locations, read("weather.csv"), read("holidays.csv"), bank, uni_i, uni_t,
        encoder_length=args.encoder_length, horizon=args.horizon, n_val_origins=args.val_origins,
        n_test_origins=args.test_origins, cold_start_days=args.cold_start_steps, freq=args.freq,
        season=args.season, level_window=args.level_window, min_history=args.min_history, series=series,
    )
    with open(args.out, "wb") as fh:
        pickle.dump(panel, fh, protocol=pickle.HIGHEST_PROTOCOL)
    c = panel.cfg
    print(f"panel: {panel.n_articles} articles x {panel.n_fc} locations x {panel.y.shape[2]} steps ({c.freq}) | cold-start {panel.cold_start.sum()} | embed dim {c.embed_dim}")
    print(f"windows: encoder {c.encoder_length}, horizon {c.horizon}, val/test origins {c.n_val_origins}/{c.n_test_origins} | season {c.season}, level window {c.level_window}, min history {c.min_history}")
    if uni_i is None:
        print("note: no unimodal features supplied -> the MTFT-style baseline compresses the aligned embeddings instead")


if __name__ == "__main__":
    main()
