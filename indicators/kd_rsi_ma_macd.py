import pandas as pd
import numpy as np


def calculate_kd_rsi_ma_macd(df_ohlcv, k_period: int = 14, d_period: int = 3, rsi_period: int = 14,
                              ma_short: int = 5, ma_long: int = 20,
                              macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                              williams_period: int = 14, adx_period: int = 14):
    """
    Calculate KD (%K, %D), RSI, 5-/20-day Moving Averages, MACD indicators,
    Williams %R, 乖離率(BIAS), ADX趨勢強度指標,
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
        williams_period (int): Lookback for Williams %R (default 14)。
            注意：若跟 k_period 用相同期間，數學上 Williams %R = %K - 100，
            資訊上等同KD、只是換個刻度，不是真正獨立的訊號。
            預設保留跟k_period一致的14，但可依需求各自調整，讓兩者能提供差異化參考。
        adx_period (int): Lookback for ADX/+DI/-DI (default 14)。
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
            'Williams_R' - Williams %R（-100~0，<-80超賣，>-20超買）
            'BIAS_MA5'/'BIAS_MA20' - 乖離率(%)，股價偏離該均線的百分比
            'Plus_DI'/'Minus_DI'/'ADX' - 趨勢方向與強度指標

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

    # --- Williams %R ---
    low_min_w = df['Low'].rolling(window=williams_period, min_periods=williams_period).min()
    high_max_w = df['High'].rolling(window=williams_period, min_periods=williams_period).max()
    denom_w = (high_max_w - low_min_w)
    df['Williams_R'] = -100 * ((high_max_w - df['Close']) / denom_w.replace(0, np.nan))

    # --- 乖離率 BIAS（股價偏離均線的百分比，正值代表股價在均線之上）---
    df['BIAS_MA5'] = (df['Close'] - df[f'MA_{ma_short}']) / df[f'MA_{ma_short}'] * 100
    df['BIAS_MA20'] = (df['Close'] - df[f'MA_{ma_long}']) / df[f'MA_{ma_long}'] * 100

    # --- ADX（趨勢強度）+ Plus_DI/Minus_DI（趨勢方向）---
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    prev_close_adx = df['Close'].shift(1)
    tr_adx = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close_adx).abs(),
        (df['Low'] - prev_close_adx).abs()
    ], axis=1).max(axis=1)

    # 用跟RSI一致的Wilder平滑法（com=period-1 相當於 alpha=1/period）
    atr_wilder = tr_adx.ewm(com=adx_period - 1, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(com=adx_period - 1, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(com=adx_period - 1, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr_wilder.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / atr_wilder.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=adx_period - 1, adjust=False).mean()

    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['ADX'] = adx
    # ADX需要雙重平滑（DM/TR平滑 -> DX -> 再平滑成ADX），暖機期比單純EMA更長，
    # 資料量不足2倍period時，數值還很不穩定，保留NaN較誠實
    if len(df) < adx_period * 2:
        df[['Plus_DI', 'Minus_DI', 'ADX']] = np.nan

    # Typical Price（沒有暖機期問題，任何時候都能算，不需要 NaN 處理）
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3

    return df
