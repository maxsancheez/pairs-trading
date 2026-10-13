"""Download and cache daily price data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return adjusted close prices, one column per ticker.

    Results are cached as CSV so repeated runs don't hit the network.
    Rows with a missing price for any ticker are dropped so the series align.
    """
    key = f"{'_'.join(sorted(tickers))}_{start}_{end or 'latest'}.csv"
    cache_file = CACHE_DIR / key

    if use_cache and cache_file.exists():
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
        if raw.empty:
            raise ValueError(f"No data returned for {tickers}")
        prices = raw["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_file)

    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"No data for: {sorted(missing)}")

    return prices[tickers].dropna()
