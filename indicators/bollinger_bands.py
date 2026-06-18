import pandas as pd

def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算布林通道 (Bollinger Bands)
    """
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_up'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_low'] = df['bb_mid'] - (df['bb_std'] * 2)
    return df
