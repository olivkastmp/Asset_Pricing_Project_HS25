"""
utils/performance.py

Performance analytics and tearsheet plotting.

Standard performance metrics used in the analysis. Most are standard
hedge fund metrics -- Sharpe, Sortino, max drawdown etc. The Calmar ratio
is maybe less common in academic papers but I included it because it captures
the risk of the momentum crash scenario (2009) better than vol-based metrics.

Note: all annualisation uses 252 trading days. Some papers use 260 or 261,
shouldn't make a meaningfull difference.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from backtest import BacktestResult


ANN = 252  # trading days in a year


# -------------------------------------------------------------------
# individual metric functions -- these are also used directly in the
# course notebooks to compute stats on subperiods etc.
# -------------------------------------------------------------------

def annualised_return(returns: pd.Series) -> float:
    """compound annualised return"""
    total = (1 + returns).prod()
    n = len(returns)
    return float(total ** (ANN / n) - 1)


def annualised_vol(returns: pd.Series) -> float:
    """annualised std of daily returns"""
    return float(returns.std() * np.sqrt(ANN))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Annualised Sharpe ratio.

    rf is the annualised risk-free rate. For most of the sample period
    (2010-2023) the ECB policy rate was near zero so I use rf=0 for
    simplicity. Could use EONIA/ESTER for a more accurate calculation.
    """
    excess = returns - rf / ANN
    vol = excess.std()
    if vol == 0:
        return np.nan
    return float(excess.mean() / vol * np.sqrt(ANN))


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Sortino ratio -- like Sharpe but only penalises downside deviation.

    More relevant for momentum strategies because the return distribution
    has negative skew (momentum crashes), so Sortino captures the crash risk
    better than symetric vol.
    """
    excess = returns - rf / ANN
    downside = excess[excess < 0]
    down_std = downside.std() * np.sqrt(ANN)
    if down_std == 0:
        return np.nan
    return float(excess.mean() * ANN / down_std)


def max_drawdown(equity: pd.Series) -> float:
    """maximum peak-to-trough drawdown (returned as a negative number)"""
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    return float(dd.min())


def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    """annualised return divided by absolute max drawdown"""
    ann_ret = annualised_return(returns)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return np.nan
    return float(ann_ret / mdd)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """full drawdown time series (always <= 0)"""
    return equity / equity.cummax() - 1


def hit_rate(returns: pd.Series) -> float:
    """fraction of days with positive return"""
    return float((returns > 0).mean())


def information_ratio(port_ret: pd.Series, bm_ret: pd.Series) -> float:
    """
    Information ratio: annualised active return / tracking error.

    I use the STOXX Europe 600 total return index as the benchmark.
    """
    active = port_ret - bm_ret
    te = active.std() * np.sqrt(ANN)
    if te == 0:
        return np.nan
    return float(active.mean() * ANN / te)


# -------------------------------------------------------------------
# PerformanceReport -- wrapper that produces the full tearsheet
# -------------------------------------------------------------------

class PerformanceReport:
    """
    Takes a BacktestResult and produces the performance table and plots.

    Parameters
    ----------
    result    : BacktestResult from Backtest.run()
    benchmark : benchmark return series (e.g. STOXX 600 daily rets)
    rf        : annualised risk-free rate (default 0)
    name      : strategy name for plot titles
    """

    def __init__(
        self,
        result: BacktestResult,
        benchmark: Optional[pd.Series] = None,
        rf: float = 0.0,
        name: str = "Strategy",
    ) -> None:
        self.result = result
        self.benchmark = benchmark
        self.rf = rf
        self.name = name

    def metrics(self) -> dict:
        """compute all metrics and return as dict"""
        r = self.result.portfolio_returns
        eq = self.result.equity_curve

        m = {
            "Ann. Return": annualised_return(r),
            "Ann. Volatility": annualised_vol(r),
            "Sharpe Ratio": sharpe_ratio(r, self.rf),
            "Sortino Ratio": sortino_ratio(r, self.rf),
            "Max Drawdown": max_drawdown(eq),
            "Calmar Ratio": calmar_ratio(r, eq),
            "Hit Rate": hit_rate(r),
            "Avg Daily TC (bps)": self.result.transaction_costs.mean() * 10_000,
            "Avg Daily Turnover": self.result.turnover.mean(),
        }

        if self.benchmark is not None:
            bm = self.benchmark.reindex(r.index).fillna(0)
            m["Info. Ratio vs Benchmark"] = information_ratio(r, bm)
            # beta to benchmark
            cov_mat = np.cov(r.fillna(0), bm.fillna(0))
            bm_var = bm.var()
            m["Beta"] = float(cov_mat[0, 1] / bm_var) if bm_var > 0 else np.nan

        return m

    def summary(self) -> pd.Series:
        """print a formatted performance summary table"""
        m = self.metrics()

        print(f"\n{'='*45}")
        print(f"  {self.name} -- Performance Summary")
        print(f"{'='*45}")
        for k, v in m.items():
            if any(x in k for x in ["Return", "Drawdown", "Vol", "Rate", "Turnover"]):
                print(f"  {k:<30} {v*100:>8.2f}%")
            else:
                print(f"  {k:<30} {v:>8.4f}")
        print(f"{'='*45}\n")

        return pd.Series(m)

    def plot(self, figsize: tuple = (13, 10), save_path: Optional[str] = None) -> None:
        """
        4-panel tearsheet: equity curve, drawdown, rolling Sharpe, turnover.

        The rolling Sharpe window is 63 days (~3 months). I briefly tried
        126 days but it smooths over the momentum crashes too much.
        """
        r = self.result.portfolio_returns
        eq = self.result.equity_curve
        dd = drawdown_series(eq)

        # rolling 63d Sharpe
        roll_sh = r.rolling(63).apply(
            lambda x: x.mean() / x.std() * np.sqrt(ANN) if x.std() > 0 else np.nan,
            raw=True,
        )

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 1, hspace=0.45)

        # equity curve
        ax1 = fig.add_subplot(gs[0])
        eq.plot(ax=ax1, color="#1a5276", linewidth=1.5, label=self.name)
        if self.benchmark is not None:
            bm_eq = (1 + self.benchmark.reindex(r.index).fillna(0)).cumprod()
            bm_eq.plot(ax=ax1, color="#aab7b8", linewidth=1.0,
                       linestyle="--", label="STOXX 600")
            ax1.legend(fontsize=9)
        ax1.set_title("Cumulative Return", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Portfolio value (x)")

        # drawdown
        ax2 = fig.add_subplot(gs[1])
        dd.plot(ax=ax2, color="#922b21", linewidth=1.0)
        ax2.fill_between(dd.index, dd, 0, alpha=0.25, color="#e74c3c")
        ax2.set_title("Drawdown", fontsize=10, fontweight="bold")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

        # rolling Sharpe
        ax3 = fig.add_subplot(gs[2])
        roll_sh.plot(ax=ax3, color="#1e8449", linewidth=1.1)
        ax3.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax3.set_title("Rolling 63-day Sharpe (annualised)", fontsize=10, fontweight="bold")

        # turnover
        ax4 = fig.add_subplot(gs[3])
        self.result.turnover.rolling(21).mean().plot(ax=ax4, color="#6c3483", linewidth=1.1)
        ax4.set_title("21-day Avg Turnover", fontsize=10, fontweight="bold")
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

        fig.suptitle(f"{self.name} — Tearsheet", fontsize=12, fontweight="bold", y=1.01)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        else:
            plt.tight_layout()
            plt.show()
