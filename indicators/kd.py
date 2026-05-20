import numpy as np
import pandas as pd


def calculate_kd(df_ohlcv, k_period: int = 14, d_period: int = 3):
    """
    Calculate KD indicator (%K, %D) for OHLCV data.

    Args:
        df_ohlcv (pd.DataFrame): Must contain 'High', 'Low', 'Close' columns.
        k_period (int): Lookback period for %K.
        d_period (int): Smoothing period for %D (SMA of %K).

    Returns:
        pd.DataFrame: Original DataFrame with added '%K' and '%D' columns.
    """
    df = df_ohlcv.copy()

    if not {'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")

    low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['High'].rolling(window=k_period, min_periods=1).max()
    denom = (high_max - low_min)
    df['%K'] = 100 * ((df['Close'] - low_min) / denom.replace(0, np.nan))
    df['%K'] = df['%K'].fillna(0)
    df['%D'] = df['%K'].rolling(window=d_period, min_periods=1).mean()
    df['%D'] = df['%D'].fillna(0)

    return df
