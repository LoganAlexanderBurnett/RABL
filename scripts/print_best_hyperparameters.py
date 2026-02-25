"""
Usage:
  python print_best_hyperparameters.py grid_search_results.json
  python print_best_hyperparameters.py grid_search_results.json -n 25

Assumes JSON structure:
{
  "results": [
    {
      "lookback": ...,
      "learning_rate": ...,
      "batch_size": ...,
      "n_lstm": ...,
      "hidden_lstm": ...,
      "hidden_fc": ...,
      "best_val_loss": ...
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

HP_KEYS = ["trial_number", "lookback", "learning_rate", "batch_size", "n_lstm", "hidden_lstm", "hidden_fc"]


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", type=Path, help="Path to grid search results JSON")
    ap.add_argument(
        "-n",
        "--n-rows",
        type=int,
        default=10,
        help="Number of trials to print (default: 10)",
    )
    args = ap.parse_args()

    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    raw_results: List[Dict[str, Any]] = data["results"]
    results = [
        {
            "trial_number": index + 1,
            **result,
        }
        for index, result in enumerate(raw_results)
    ]

    results_sorted = sorted(results, key=lambda r: r["best_val_loss"])
    n = max(1, args.n_rows)
    top = results_sorted[:n]

    headers = HP_KEYS + ["best_val_loss"]
    widths = {h: len(h) for h in headers}
    for r in top:
        for h in headers:
            widths[h] = max(widths[h], len(fmt(r.get(h))))

    sep = "  "
    print(sep.join(f"{h:{widths[h]}}" for h in headers))
    print(sep.join("-" * widths[h] for h in headers))
    for r in top:
        print(sep.join(f"{fmt(r.get(h)):{widths[h]}}" for h in headers))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
