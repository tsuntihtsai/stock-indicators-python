import numpy as np
import pandas as pd


def calculate_rsi(df_ohlcv, rsi_period: int = 14):
    """
    Calculate RSI indicator for OHLCV data.

    Args:
        df_ohlcv (pd.DataFrame): Must contain 'Close' column.
        rsi_period (int): RSI period.

    Returns:
        pd.DataFrame: Original DataFrame with added 'RSI' column.
    """
    df = df_ohlcv.copy()

    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(0)

    return df
