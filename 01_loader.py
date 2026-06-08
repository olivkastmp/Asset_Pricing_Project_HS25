"""
data/loader.py

Utilities for loading and cleaning price data.
Written for a quant finance course project (STOXX 600 momentum study) but
should work for any wide-format price CSV.

Used Refinitiv Eikon exports which come out in long format with a 'Ticker'
and 'Close' column. Added wide-format support later when I started testing
on synthetic data.

TODO: add proper handling for corporate actions / splits -- currently
      relying on Eikon's adjusted prices which should be fine but worth checking
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_prices(
    source: str | Path,
    price_col: str = "Close",
    date_col: str = "Date",
    fill_method: str = "ffill",
    min_history: int = 252,
) -> pd.DataFrame:
    """
    Load a price matrix from CSV.

    Handles both wide format (date index, one col per ticker) and
    long format (Date, Ticker, Close columns) -- the latter is what
    Refinitiv Eikon exports by default.

    Parameters
    ----------
    source       : path to the CSV file
    price_col    : which price column to use, default 'Close'
    date_col     : name of the date column
    fill_method  : 'ffill' to carry prices forward over non-trading days,
                   or None to leave NaNs
    min_history  : drop any asset with fewer than this many observations
                   (avoids look-ahead from short-listed stocks in the STOXX)

    Returns
    -------
    pd.DataFrame  wide format: rows = dates, columns = tickers
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"cant find price file: {path}")

    raw = pd.read_csv(path, parse_dates=[date_col])
    raw = raw.sort_values(date_col).set_index(date_col)

    # detect format
    if "Ticker" in raw.columns:
        # long format (Eikon default)
        prices = raw.pivot(columns="Ticker", values=price_col)
    else:
        # assume wide already
        prices = raw.select_dtypes(include=[np.number])

    prices = prices.astype(float)

    if fill_method:
        prices = getattr(prices, fill_method)()

    # drop thin assets -- important for STOXX survivorship handling
    counts = prices.notna().sum()
    thin = counts[counts < min_history].index.tolist()
    if thin:
        warnings.warn(
            f"dropping {len(thin)} assets with less than {min_history} obs. "
            f"First few: {thin[:5]}"
        )
        prices = prices.drop(columns=thin)

    return prices


def download_prices(
    tickers: list[str],
    start: str = "2005-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download prices via yfinance (mostly used for quick tests, real data
    came from Eikon).

    Note: yfinance tickers for European stocks need the exchange suffix
    e.g. 'AIR.PA' for Airbus, 'VOW3.DE' for VW etc.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.ffill().dropna(how="all")
    return prices


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
    periods: int = 1,
) -> pd.DataFrame:
    """
    Compute returns from prices.

    I use log returns throughout the project (easier to aggregate across time),
    but simple returns are available too and are better for cross-sectional
    comparisons within a single period.

    Parameters
    ----------
    method  : 'log' or 'simple'
    periods : lag -- use 1 for daily, 5 for weekly etc.
    """
    if method == "log":
        rets = np.log(prices / prices.shift(periods))
    elif method == "simple":
        rets = prices.pct_change(periods)
    else:
        raise ValueError(f"unknown method: '{method}' -- use 'log' or 'simple'")

    return rets


def align_panel(
    prices: pd.DataFrame,
    min_assets: int = 5,
    dropna_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Remove rows where too many assets have missing data.
    Useful for the start/end of the STOXX sample where constituents change a lot.

    Parameters
    ----------
    min_assets       : minimum non-NaN assets per row
    dropna_threshold : drop rows where more than this fraction are NaN
    """
    frac_missing = prices.isna().mean(axis=1)
    enough_assets = prices.notna().sum(axis=1) >= min_assets
    not_too_sparse = frac_missing <= dropna_threshold
    return prices.loc[enough_assets & not_too_sparse]


def generate_sample_prices(
    n_assets: int = 20,
    n_days: int = 1500,
    seed: int = 42,
    start: str = "2010-01-01",
) -> pd.DataFrame:
    """
    Synthetic price panel for testing the pipeline without real data.

    Simple GBM with a common market factor and idiosyncratic noise.
    The tickers are named STOXX_XX to vaguely resemble the actual dataset.

    n_days=1500 ~ 6 years of daily data, covers most of the sample period.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"STOXX_{i:02d}" for i in range(n_assets)]
    dates = pd.bdate_range(start=start, periods=n_days)

    # per-asset drift and vol -- loosely calibrated to STOXX 600 historicals
    mu = rng.uniform(-0.0002, 0.0006, n_assets)
    sigma = rng.uniform(0.010, 0.025, n_assets)

    # common market factor (represents EuroStoxx 50 exposure)
    mkt = rng.normal(0.0002, 0.012, (n_days, 1))
    betas = rng.uniform(0.5, 1.5, (1, n_assets))
    idio = rng.normal(0, 1, (n_days, n_assets)) * sigma

    daily_rets = mu + betas * mkt + idio
    log_px = np.cumsum(daily_rets, axis=0)
    prices = 100 * np.exp(log_px)

    return pd.DataFrame(prices, index=dates, columns=tickers)
