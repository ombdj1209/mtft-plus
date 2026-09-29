"""v4.1 adoption rule (pre-registration amendment (d) 5): validation loss only, seed 1.

    python examples/v41_decision.py --cal <v41 calendar out> --off <v41 official out>

Reads ONLY the ``best_val`` field of the seed-1 partials of v4.1 and v4 (calendar + 3 official tasks) and prints the
adoption decision: adopt if v4.1's validation loss is lower in at least 3 of the 4 settings. Test metrics are not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def best_val(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["best_val"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True)
    ap.add_argument("--off", required=True)
    args = ap.parse_args()
    rows = [("calendar", Path(args.cal) / "partials/1__mtft-v4-1-exploratory.json", Path("results_visuelle2/partials/1__mtft-v4-ours.json"))]
    rows += [(t, Path(args.off) / f"partials/{t}__1__mtft-v4-1-exploratory.json", Path(f"results_visuelle2_official/partials/{t}__1__mtft-v4-ours.json"))
             for t in ("2-1", "2-10", "demand")]
    wins = 0
    for name, p41, p4 in rows:
        a, b = best_val(p41), best_val(p4)
        wins += a < b
        print(f"{name:9s} validation loss  v4.1 {a:.4f}  v4 {b:.4f}  -> {'v4.1 lower' if a < b else 'v4 lower or equal'}")
    print(f"v4.1 lower in {wins}/4 settings -> {'ADOPT v4.1' if wins >= 3 else 'DO NOT ADOPT v4.1'} (rule: >= 3/4)")


if __name__ == "__main__":
    main()
