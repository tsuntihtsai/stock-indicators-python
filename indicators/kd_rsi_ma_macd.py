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

    注意（修正說明）：暖機期（例如 MA_20 前19天、KD/RSI 初期）本來就沒有
    足夠資料算出可靠數值，這裡刻意保留 NaN，不再用 fillna(0) 填成假的0值。
    填0會讓均線/RSI/KD在圖表上出現不存在的「摔到0再彈回來」的假尖刺，
    誤導判讀。畫圖或做進一步計算時，請用 pandas 的 NaN-aware方法
    （例如 dropna()、fillna(method='ffill') 等）自行處理，
    不要在指標計算階段就填0。
    """
    df = df_ohlcv.copy()
    if not {'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")

    # --- KD Indicator (Stochastic Oscillator) ---
    low_min = df['Low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['High'].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_max - low_min)
    df['%K'] = 100 * ((df['Close'] - low_min) / denom.replace(0, np.nan))
    df['%D'] = df['%K'].rolling(window=d_period, min_periods=d_period).mean()

    # --- RSI Indicator ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    # Use EMA smoothing for RSI (commonly used approach)
    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    # RSI 在資料筆數 < rsi_period 時仍不夠可靠，同樣保留 NaN
    if len(df) >= rsi_period:
        df.iloc[:rsi_period - 1, df.columns.get_loc('RSI')] = np.nan
    else:
        df['RSI'] = np.nan

    # --- Moving Averages (MA) ---
    df[f'MA_{ma_short}'] = df['Close'].rolling(window=ma_short, min_periods=ma_short).mean()
    df[f'MA_{ma_long}'] = df['Close'].rolling(window=ma_long, min_periods=ma_long).mean()

    # --- MACD Indicator ---
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['Signal_Line'] = df['MACD_Line'].ewm(span=macd_signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']
    # MACD 在資料筆數不足 macd_slow 天時同樣不可靠，保留 NaN
    if len(df) < macd_slow:
        df[['MACD_Line', 'Signal_Line', 'MACD_Hist']] = np.nan

    # Typical Price（沒有暖機期問題，任何時候都能算，不需要 NaN 處理）
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3

    return df
