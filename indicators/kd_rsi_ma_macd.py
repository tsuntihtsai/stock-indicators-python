import pandas as pd
import numpy as np


def calculate_kd_rsi_ma_macd(df_ohlcv, k_period: int = 14, d_period: int = 3, rsi_period: int = 14,
                              ma_short: int = 5, ma_long: int = 20,
                              macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
    """
    Calculate KD (%K, %D), RSI, 5-/20-day Moving Averages, MACD indicators,
    and an additional Typical Price (Typical_Price) for OHLCV data.
   
    Typical Price = (High + Low + Close) / 3

    and append them as columns to the input DataFrame.

    Args:
        df_ohlcv (pd.DataFrame): Must contain at least 'High', 'Low', 'Close' columns.
        k_period (int): Lookback for KD %K (default 14).
        d_period (int): Smoothing period for KD %D (default 3).
        rsi_period (int): RSI period (default 14).
        ma_short (int): Short-term Moving Average period (default 5).
        ma_long (int): Long-term Moving Average period (default 20).
        macd_fast (int): Fast EMA period for MACD (default 12).
        macd_slow (int): Slow EMA period for MACD (default 26).
        macd_signal (int): Signal line EMA period for MACD (default 9).

    Returns:
        pd.DataFrame: Original DataFrame with new columns:
            '%K'        - KD %K
            '%D'        - KD %D
            'RSI'        - RSI values
            'MA_5'/'MA_20' - Moving Averages (depending on provided ma_short/ma_long)
            'MACD_Line'  - MACD DIF line
            'Signal_Line'- MACD Signal line
            'MACD_Hist'  - MACD histogram (MACD_Line - Signal_Line)
            'Typical_Price' - (High + Low + Close) / 3
    """
    df = df_ohlcv.copy()

    if not {'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")

    # --- KD Indicator (Stochastic Oscillator) ---
    low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['High'].rolling(window=k_period, min_periods=1).max()
    denom = (high_max - low_min)

    df['%K'] = 100 * ((df['Close'] - low_min) / denom.replace(0, np.nan))
    df['%K'] = df['%K'].fillna(0)
    df['%D'] = df['%K'].rolling(window=d_period, min_periods=1).mean()
    df['%D'] = df['%D'].fillna(0)

    # --- RSI Indicator ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Use EMA smoothing for RSI (commonly used approach)
    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(0)

    # --- Moving Averages (MA) ---
    df[f'MA_{ma_short}'] = df['Close'].rolling(window=ma_short).mean().fillna(0)
    df[f'MA_{ma_long}'] = df['Close'].rolling(window=ma_long).mean().fillna(0)

    # --- MACD Indicator ---
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['Signal_Line'] = df['MACD_Line'].ewm(span=macd_signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']

    # Typical Price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Fill initial NaN defaults for added columns
    for col in ['%K', '%D', 'RSI', f'MA_{ma_short}', f'MA_{ma_long}', 'MACD_Line', 'Signal_Line', 'MACD_Hist', 'Typical_Price']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
