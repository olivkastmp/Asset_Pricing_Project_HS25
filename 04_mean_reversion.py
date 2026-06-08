"""
signals/mean_reversion.py

Mean reversion signals -- included for comparison against momentum,
not the main focus of the project.

The idea is that if momentum crashes (e.g. 2009, 2020) a mean-reversion
signal should in theory provide some hedge, but in practise the correlation
is not that stable. I ran some regime-conditional analysis but the evidence
was pretty weak on the STOXX sample so it didn't make it into the final paper.

De Bondt & Thaler (1985) is the classic reference for long-term reversal,
but the z-score version here is much shorter term (20-day window) which
is more like the short-term reversal anomaly from Lehmann (1990).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import importlib as _il; BaseSignal = _il.import_module("signals.02_base").BaseSignal


class MeanReversionSignal(BaseSignal):
    """
    Z-score mean reversion.

    Computes the rolling z-score of log returns. Assets with high positive
    z-scores are shorted (expected to revert down), negative z-scores longed.
    Only takes a position when |z| exceeds a threshold to avoid trading noise.

    This generated way too much turnover at daily frequency on the STOXX data
    (~150% per day) which completely killed net returns. Worked better at weekly
    rebalancing but still underperformed momentum signficantly.

    Parameters
    ----------
    window       : rolling window for mean/std (default 20 days)
    z_threshold  : minimum |z| to take a position (default 1.0)
    vol_scale    : scale by inverse vol like in the momentum signal
    vol_window   : window for vol estimate
    """

    def __init__(
        self,
        window: int = 20,
        z_threshold: float = 1.0,
        vol_scale: bool = True,
        vol_window: int = 63,
    ) -> None:
        self.window = window
        self.z_threshold = z_threshold
        self.vol_scale = vol_scale
        self.vol_window = vol_window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(prices / prices.shift(1))

        roll_mean = log_ret.rolling(self.window).mean()
        roll_std = log_ret.rolling(self.window).std().replace(0, np.nan)
        z = (log_ret - roll_mean) / roll_std

        # fade the z-score (mean revert), zero out if below threshold
        raw = -z.where(z.abs() >= self.z_threshold, other=0.0)

        if self.vol_scale:
            ann_vol = log_ret.rolling(self.vol_window).std() * np.sqrt(252)
            inv_vol = 1.0 / ann_vol.clip(lower=0.05)
            raw = raw * inv_vol

        gross = raw.abs().sum(axis=1).replace(0, np.nan)
        return raw.divide(gross, axis=0)


class RSIMeanReversion(BaseSignal):
    """
    RSI-based reversion signal.

    I added this as a quick experiment after reading about RSI-based strategies
    in Connors & Alvarez (2009). Not in the paper -- just something I tried.
    Works OK on single stocks but didn't add much at portfolio level.

    Long oversold (RSI < 30), short overbought (RSI > 70).
    Uses EWM smoothing which is slighly different from the classic Wilder (1978)
    definition but computationally more convienient.

    Parameters
    ----------
    period               : RSI lookback (default 14)
    oversold_threshold   : RSI below this = long (default 30)
    overbought_threshold : RSI above this = short (default 70)
    """

    def __init__(
        self,
        period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
    ) -> None:
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def _calc_rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        """compute RSI for all assets at once"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        rsi = self._calc_rsi(prices)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights[rsi < self.oversold_threshold] = 1.0
        weights[rsi > self.overbought_threshold] = -1.0

        gross = weights.abs().sum(axis=1).replace(0, np.nan)
        return weights.divide(gross, axis=0).fillna(0.0)
