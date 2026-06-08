"""
signals/momentum.py

Momentum signal implementations for the STOXX 600 project.

Main signal: 12-1 month time-series momentum (Jegadeesh & Titman 1993)
             with inverse-volatility scaling following Moskowitz et al. (2012).

Also includes cross-sectional momentum (rank-based) which I test in
chapter 4 as a robustness check against the time-series version.

References
----------
Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling
    losers: Implications for stock market efficiency. JoF 48(1), 65-91.

Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum.
    JFE 104(2), 228-250.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import importlib as _il; BaseSignal = _il.import_module("signals.02_base").BaseSignal


class MomentumSignal(BaseSignal):
    """
    12-1 month time-series momentum with vol-scaling.

    Computes trailing return from t-lookback to t-skip (skipping the most
    recent month to avoid the well-documented short-term reversal effect).
    Sign of this return gives the direction, then scaled by inverse realised vol
    so that higher-vol stocks don't dominate the portfolio.

    I test lookbacks of 6, 9, 12 months -- 12 months tends to give the best
    Sharpe on STOXX data but 9 months is more stable out-of-sample.

    Parameters
    ----------
    lookback   : total lookback in trading days (default 252 ~ 12 months)
    skip       : recent days to exclude (default 21 ~ 1 month)
    vol_window : window for realised vol estimate (default 63 ~ 3 months)
    vol_floor  : min annualised vol to avoid insane leverage on low-vol stocks
    """

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        vol_window: int = 63,
        vol_floor: float = 0.05,
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.vol_window = vol_window
        self.vol_floor = vol_floor

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signal weights.

        Returns weights with same shape as prices. NaN for rows before
        we have enough history for the lookback window.
        """
        log_ret = np.log(prices / prices.shift(1))

        # trailing return from lookback to skip -- the "formation period"
        trailing = np.log(prices.shift(self.skip) / prices.shift(self.lookback))

        # direction: +1 for winners, -1 for losers
        direction = np.sign(trailing)

        # vol scaling -- annualise by sqrt(252)
        ann_vol = log_ret.rolling(self.vol_window).std() * np.sqrt(252)
        ann_vol = ann_vol.clip(lower=self.vol_floor)
        inv_vol = 1.0 / ann_vol

        raw_w = direction * inv_vol

        # normalise so gross exposure = 1 each day
        weights = self._normalise(raw_w)
        return weights

    def _normalise(self, weights: pd.DataFrame) -> pd.DataFrame:
        """demean and scale to unit gross exposure"""
        demeaned = weights.subtract(weights.mean(axis=1), axis=0)
        gross = demeaned.abs().sum(axis=1).replace(0, np.nan)
        return demeaned.divide(gross, axis=0)


class CrossSectionalMomentum(BaseSignal):
    """
    Cross-sectional (relative) momentum.

    Ranks all assets each period by trailing return, longs the top quintile
    and shorts the bottom quintile with equal weights. Classic Fama-French
    implementation.

    In the analysis I compare this against MomentumSignal -- the time-series
    version tends to have lower turnover (and therefore lower net TC drag)
    because the direction only flips when the sign changes, not just when
    rank ordering shuffles.

    Parameters
    ----------
    lookback  : trailing return window (days)
    skip      : recent days to skip (days)
    long_pct  : fraction of assets to long (default 0.2 = top quintile)
    short_pct : fraction of assets to short (default 0.2 = bottom quintile)
    """

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.long_pct = long_pct
        self.short_pct = short_pct

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-sectional momentum weights.

        This loops over dates which is slow for large panels -- TODO: vectorise
        this properly when I have time, it's fine for the ~600 stock universe.
        """
        trailing = prices.shift(self.skip) / prices.shift(self.lookback) - 1

        n = prices.shape[1]
        n_long = max(1, int(n * self.long_pct))
        n_short = max(1, int(n * self.short_pct))

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for date, row in trailing.iterrows():
            valid = row.dropna()
            if len(valid) < n_long + n_short:
                continue
            ranked = valid.sort_values()
            short_assets = ranked.iloc[:n_short].index
            long_assets = ranked.iloc[-n_long:].index
            weights.loc[date, long_assets] = 1.0 / n_long
            weights.loc[date, short_assets] = -1.0 / n_short

        return weights
