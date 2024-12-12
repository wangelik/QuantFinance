# import table
from typing import Union

import numpy as np
import pandas as pd


def returns(
    prices: Union[pd.Series, np.array, list]
) -> Union[pd.Series, np.array, list]:
    """Utility function to compute returns of pricing data"""

    # pandas series data
    if isinstance(prices, pd.Series):
        return prices / prices.shift(1) - 1
    # numpy array
    elif isinstance(prices, np.ndarray):
        ans = prices / np.roll(prices, 1) - 1
        ans[0] = np.nan
        return ans
    # list
    elif isinstance(prices, list):
        ans = pd.Series(prices)
        ans = ans / ans.shift(1) - 1
        return list(ans)
    # other types
    else:
        raise TypeError("Unsupported input type")


def logreturns(
    prices: Union[pd.Series, np.array, list]
) -> Union[pd.Series, np.array, list]:
    """Utility function to compute log returns of pricing data

    Note: Taylor series of log(1+x) ~ x
    """

    # pandas series data
    if isinstance(prices, pd.Series):
        return np.log(prices / prices.shift(1))
    # numpy array
    elif isinstance(prices, np.ndarray):
        ans = np.log(prices / np.roll(prices, 1))
        ans[0] = np.nan
        return ans
    # list
    elif isinstance(prices, list):
        ans = pd.Series(prices)
        ans = np.log(ans / ans.shift(1))
        return list(ans)
    # other types
    else:
        raise TypeError("Unsupported input type")
