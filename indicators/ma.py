import pandas as pd


def calculate_ma(df_ohlcv, periods=[5, 20]):
    """
    Calculate moving averages for given periods.

    Args:
        df_ohlcv (pd.DataFrame): Must contain 'Close' column.
        periods (list): List of integers representing MA periods.

    Returns:
        pd.DataFrame: Original DataFrame with MA_X columns for each period.
    """
    df = df_ohlcv.copy()
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    for p in periods:
        df[f'MA_{p}'] = df['Close'].rolling(window=p).mean().fillna(0)

    return df
