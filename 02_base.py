"""
signals/base.py

Abstract base class for signals. All signals need to implement compute().

Added this after I had 3 different signal classes with slightly different
interfaces and kept confusing myself -- a base class forces consistancy.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BaseSignal(ABC):
    """
    Base class for all signal generators.

    All subclasses must implement compute(prices) which takes a wide-format
    price dataframe and returns a dataframe of weights (same shape).

    Positive weight = long, negative = short.
    Weights don't need to be normalised here -- the backtest engine handles that.
    """

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signal weights from price history.

        Parameters
        ----------
        prices : pd.DataFrame
            rows = dates, columns = asset tickers

        Returns
        -------
        pd.DataFrame
            portfolio weights, same index and columns as prices
        """
        ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"
