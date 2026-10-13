"""Statistical tools for pairs analysis.

The core question: is the spread between two stocks mean-reverting?
Correlation alone doesn't answer that -- two stocks can be highly correlated
while the gap between them drifts forever. Cointegration tests for exactly
the property we need: a linear combination of the two prices that is
stationary (keeps returning to its average).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class PairStats:
    y: str
    x: str
    correlation: float
    hedge_ratio: float
    intercept: float
    coint_pvalue: float
    adf_pvalue: float
    half_life: float

    @property
    def is_cointegrated(self) -> bool:
        return self.coint_pvalue < 0.05


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS regression y = intercept + beta * x. Returns (beta, intercept).

    beta tells you how many shares of x to short per share of y held long
    so the position is balanced.
    """
    model = sm.OLS(y, sm.add_constant(x)).fit()
    return float(model.params.iloc[1]), float(model.params.iloc[0])


def spread(y: pd.Series, x: pd.Series, beta: float, intercept: float = 0.0) -> pd.Series:
    """The residual y - (intercept + beta * x). This is what we trade."""
    return y - (intercept + beta * x)


def zscore(series: pd.Series, window: int | None = None) -> pd.Series:
    """Standardize a series. With a window, uses a rolling mean/std so each
    point only depends on past data (no look-ahead bias)."""
    if window is None:
        return (series - series.mean()) / series.std()
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def half_life(spread_series: pd.Series) -> float:
    """Estimated days for the spread to close half its gap to the mean.

    Fits the discrete Ornstein-Uhlenbeck model
        delta_spread_t = lambda * spread_{t-1} + noise
    and returns -ln(2) / lambda. A negative lambda means the spread pulls
    back toward the mean; if lambda >= 0 there's no mean reversion and we
    return infinity.
    """
    lagged = spread_series.shift(1).dropna()
    delta = spread_series.diff().dropna()
    lam = sm.OLS(delta, sm.add_constant(lagged)).fit().params.iloc[1]
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


def analyze_pair(prices: pd.DataFrame, y: str, x: str) -> PairStats:
    """Run the full set of tests on one pair."""
    py, px = prices[y], prices[x]
    beta, intercept = hedge_ratio(py, px)
    s = spread(py, px, beta, intercept)
    _, coint_p, _ = coint(py, px)
    adf_p = adfuller(s, autolag="AIC")[1]
    return PairStats(
        y=y,
        x=x,
        correlation=float(py.corr(px)),
        hedge_ratio=beta,
        intercept=intercept,
        coint_pvalue=float(coint_p),
        adf_pvalue=float(adf_p),
        half_life=half_life(s),
    )
