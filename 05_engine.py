"""
backtest/engine.py

Vectorised backtesting engine.

Wrote this about halfway through the semester after getting frustrated with looping
over dates in my initial implementation (was taking ~40 seconds per run
on the full STOXX panel, now takes < 1 second).

The engine takes a price panel and a weight dataframe, simulates a daily
rebalanced portfolio with transaction costs, and returns a BacktestResult
with the equity curve, turnover, and some other useful series.

Main simplifications / assumptions:
- Close-to-close returns, no intraday
- TC applied on *changes* in weight (one-way), not on gross weight
- No short-selling constraints (but set long_only=True if you want that)
- No slippage model -- TC in bps is a rough proxy
- Weights are L1 normalised to unit gross exposure before applying

I use 5bps one-way as a rough estimate for large-cap
European equities. Some papers use 10bps which seems conservative for
STOXX 50 but maybe reasonable for the smaller STOXX 600 members.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """holds the output of a backtest run"""

    equity_curve: pd.Series
    """cumulative portfolio value starting at 1.0"""

    portfolio_returns: pd.Series
    """daily net returns (after transaction costs)"""

    gross_returns: pd.Series
    """daily returns before TC"""

    turnover: pd.Series
    """daily one-way turnover = sum(|delta_w|)"""

    weights: pd.DataFrame
    """realised weights used in simulation"""

    transaction_costs: pd.Series
    """daily TC drag in return terms"""

    params: dict = field(default_factory=dict)
    """backtest params, kept for reference"""


class Backtest:
    """
    Vectorised portfolio backtester.

    Parameters
    ----------
    prices               : price dataframe (dates x tickers)
    weights              : signal weights (same shape)
    transaction_cost_bps : one-way TC in basis points (default 5)
    long_only            : clip negatives to zero if True
    rebalance_freq       : 'D' daily (default), 'W' weekly, 'M' monthly
    initial_capital      : just cosmetic, doesn't affect returns
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        transaction_cost_bps: float = 5.0,
        long_only: bool = False,
        rebalance_freq: str = "D",
        initial_capital: float = 1.0,
    ) -> None:
        self.prices = prices
        self.weights = weights
        self.tc = transaction_cost_bps / 10_000  # convert bps to decimal
        self.long_only = long_only
        self.rebalance_freq = rebalance_freq
        self.initial_capital = initial_capital

    def run(self) -> BacktestResult:
        """Run the backtest and return results."""
        prices, weights = self._align()
        weights = self._prep_weights(weights)

        simple_ret = prices.pct_change()

        if self.rebalance_freq != "D":
            weights = self._apply_rebalance_freq(weights)

        # portfolio return on day t = sum over assets of (w_{t-1} * r_t)
        # (use yesterday's weights because we rebalance at close of t-1)
        gross_ret = (weights.shift(1) * simple_ret).sum(axis=1)

        # turnover = sum of absolute weight changes on each day
        turnover = weights.diff().abs().sum(axis=1)

        # cost drag
        tc_drag = turnover * self.tc

        net_ret = gross_ret - tc_drag
        equity = (1 + net_ret).cumprod() * self.initial_capital

        return BacktestResult(
            equity_curve=equity,
            portfolio_returns=net_ret,
            gross_returns=gross_ret,
            turnover=turnover,
            weights=weights,
            transaction_costs=tc_drag,
            params={
                "tc_bps": self.tc * 10_000,
                "long_only": self.long_only,
                "rebalance_freq": self.rebalance_freq,
                "n_assets": prices.shape[1],
                "start": str(prices.index[0].date()),
                "end": str(prices.index[-1].date()),
            },
        )

    def _align(self):
        """align prices and weights to common dates and tickers"""
        idx = self.prices.index.intersection(self.weights.index)
        cols = self.prices.columns.intersection(self.weights.columns)
        return self.prices.loc[idx, cols], self.weights.loc[idx, cols]

    def _prep_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        """clean up weights: handle long-only, remove inf/nan, normalise"""
        w = weights.copy()

        if self.long_only:
            w = w.clip(lower=0)

        # replace any inf values that sometimes come out of signal calcs
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # L1 normalise -- unit gross exposure
        gross = w.abs().sum(axis=1).replace(0, np.nan)
        return w.divide(gross, axis=0).fillna(0.0)

    def _apply_rebalance_freq(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        For non-daily rebalancing: set weights to NaN on non-rebalance days,
        then forward fill so the portfolio holds positions between rebalances.

        Note: this slightly overstates TC because weights drift between
        rebalances but we only count the difference on rebalance days.
        Good enough for the project but worth fixing for a real system.
        """
        rebalance_dates = weights.resample(self.rebalance_freq).last().index
        is_rebalance = weights.index.isin(rebalance_dates)
        out = weights.copy()
        out[~is_rebalance] = np.nan
        return out.ffill()
