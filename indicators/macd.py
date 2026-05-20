import pandas as pd


def calculate_macd(df_ohlcv, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicators for OHLCV data.

    Returns:
        pd.DataFrame: Original DataFrame with MACD_Line, Signal_Line, MACD_Hist.
    """
    df = df_ohlcv.copy()
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['Signal_Line'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']
    return df
