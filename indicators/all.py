from .kd import calculate_kd
from .rsi import calculate_rsi
from .ma import calculate_ma
from .macd import calculate_macd

import pandas as pd


def merge_indicators(dfs):
    # Helper to merge multiple indicator DataFrames on their index
    res = dfs[0]
    for df in dfs[1:]:
        res = res.merge(df, left_index=True, right_index=True, how='left')
    return res


def calculate_all_indicators(df_ohlcv, kd_params=None, rsi_params=None, ma_periods=None, macd_params=None):
    """
    Convenience entry that calculates all indicators and returns a single DataFrame
    containing KD, RSI, MA (multi periods), and MACD outputs.

    Args:
        df_ohlcv (pd.DataFrame): DataFrame with at least High, Low, Close columns.
        kd_params (dict): {'k_period': int, 'd_period': int}
        rsi_params (dict): {'rsi_period': int}
        ma_periods (list[int]): [5, 20, ...]
        macd_params (dict): {'fast': int, 'slow': int, 'signal': int}

    Returns:
        pd.DataFrame: DataFrame with all indicators merged.
    """
    kd_res = calculate_kd(df_ohlcv, **(kd_params or {})) if kd_params is not None else df_ohlcv
    rsi_res = calculate_rsi(df_ohlcv, **(rsi_params or {})) if rsi_params is not None else df_ohlcv
    ma_res = calculate_ma(df_ohlcv, periods=ma_periods or []) if ma_periods else df_ohlcv
    macd_res = calculate_macd(df_ohlcv, **(macd_params or {})) if macd_params is not None else df_ohlcv

    # Merge all results; since each function returns the df with new columns, we can start from df_ohlcv
    # and progressively merge the additional columns by aligning on index
    merged = df_ohlcv.copy()
    for df in [kd_res, rsi_res, ma_res, macd_res]:
        if df is not df_ohlcv:
            merged = merged.merge(df.drop(columns=[c for c in df.columns if c in merged.columns], errors='ignore'), left_index=True, right_index=True, how='left')
    return merged
