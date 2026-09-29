"""Pre-registration amendment (h): move the early-stopped published-baseline results to a secondary row.

    python examples/published_secondary.py --out results_visuelle2_official

Every partial in <out>/partials written by examples/visuelle2_published.py without the full epoch budget
(spec.full_budget missing or false) is moved to <out>/published_early_stopped/, with " [early stopping, secondary]"
appended to its model name. The full-budget runs then write the primary rows. Nothing is deleted or re-scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUFFIX = " [early stopping, secondary]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_visuelle2_official")
    args = ap.parse_args()
    src = Path(args.out) / "partials"
    dst = Path(args.out) / "published_early_stopped"
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in sorted(src.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        spec = r.get("spec", {})
        if not spec.get("published_code") or spec.get("full_budget"):
            continue
        r["model"] = r["model"] + SUFFIX if not r["model"].endswith(SUFFIX) else r["model"]
        spec["early_stopping"] = True
        (dst / p.name).write_text(json.dumps(r), encoding="utf-8")
        p.unlink()
        moved += 1
        print(f"secondary: {p.name}  WAPE {r['all']['WAPE']:.2f}  epochs run {r.get('epochs_run')} of {r.get('epochs')}")
    print(f"moved {moved} early-stopped published-baseline results to {dst}")


if __name__ == "__main__":
    main()
