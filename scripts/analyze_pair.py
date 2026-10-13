"""Analyze one stock pair for mean-reversion.

Usage:
    python scripts/analyze_pair.py KO PEP
    python scripts/analyze_pair.py V MA --start 2018-01-01 --window 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs.data import load_prices  # noqa: E402
from pairs.plots import plot_pair  # noqa: E402
from pairs.stats import analyze_pair, spread, zscore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("y", help="First ticker (held long when spread is low)")
    parser.add_argument("x", help="Second ticker (hedge leg)")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--window", type=int, default=60, help="Rolling z-score window in days")
    parser.add_argument("--entry", type=float, default=2.0, help="Z-score entry threshold")
    args = parser.parse_args()

    y, x = args.y.upper(), args.x.upper()
    prices = load_prices([y, x], start=args.start, end=args.end)
    stats = analyze_pair(prices, y, x)

    s = spread(prices[y], prices[x], stats.hedge_ratio, stats.intercept)
    z = zscore(s, window=args.window)

    print(f"\n{y} / {x}  ({prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days)")
    print("-" * 52)
    print(f"Correlation            {stats.correlation:>10.3f}")
    print(f"Hedge ratio (beta)     {stats.hedge_ratio:>10.3f}   shares of {x} per share of {y}")
    print(f"Cointegration p-value  {stats.coint_pvalue:>10.4f}   (Engle-Granger, < 0.05 is good)")
    print(f"Spread ADF p-value     {stats.adf_pvalue:>10.4f}")
    print(f"Half-life              {stats.half_life:>10.1f}   days")
    print(f"Current z-score        {z.iloc[-1]:>10.2f}")
    print("-" * 52)
    verdict = "COINTEGRATED" if stats.is_cointegrated else "NOT cointegrated"
    print(f"Verdict: {verdict} at the 5% level")
    if not stats.is_cointegrated and stats.correlation > 0.8:
        print("Note: high correlation but no cointegration -- the gap can drift indefinitely.")

    out = ROOT / "docs" / "images" / f"{y}_{x}.png"
    plot_pair(prices, y, x, z, entry=args.entry, out_path=out)
    print(f"\nChart saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
