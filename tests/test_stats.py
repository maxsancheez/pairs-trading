"""Tests on synthetic data where we know the right answer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairs.stats import analyze_pair, half_life, hedge_ratio, zscore  # noqa: E402

RNG = np.random.default_rng(42)
N = 1000


def random_walk(n=N, start=100.0):
    return start + np.cumsum(RNG.normal(0, 1, n))


def ou_process(n=N, theta=0.1, sigma=1.0):
    """Mean-reverting noise with known half-life ln(2)/theta."""
    out = np.zeros(n)
    for t in range(1, n):
        out[t] = out[t - 1] - theta * out[t - 1] + RNG.normal(0, sigma)
    return out


def test_hedge_ratio_recovers_beta():
    x = pd.Series(random_walk())
    y = 1.5 * x + 10 + RNG.normal(0, 0.5, N)
    beta, intercept = hedge_ratio(y, x)
    assert abs(beta - 1.5) < 0.05


def test_cointegrated_pair_detected():
    x = random_walk()
    y = 0.8 * x + ou_process()
    prices = pd.DataFrame({"Y": y, "X": x})
    assert analyze_pair(prices, "Y", "X").is_cointegrated


def test_independent_walks_not_cointegrated():
    prices = pd.DataFrame({"Y": random_walk(), "X": random_walk()})
    assert not analyze_pair(prices, "Y", "X").is_cointegrated


def test_half_life_close_to_theory():
    theta = 0.1
    s = pd.Series(ou_process(n=5000, theta=theta))
    expected = np.log(2) / theta  # ~6.9 days
    assert abs(half_life(s) - expected) < 2


def test_half_life_infinite_for_trending_series():
    trending = pd.Series(np.arange(500, dtype=float) ** 1.5)
    assert half_life(trending) == float("inf")


def test_rolling_zscore_has_no_lookahead():
    s = pd.Series(RNG.normal(0, 1, 200))
    z_full = zscore(s, window=20)
    z_truncated = zscore(s.iloc[:100], window=20)
    pd.testing.assert_series_equal(z_full.iloc[:100], z_truncated)
