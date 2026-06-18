import pandas as pd

def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    # 這裡開頭必須是「剛好 4 個空格」，不能用 Tab
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_up'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_low'] = df['bb_mid'] - (df['bb_std'] * 2)
    return df
