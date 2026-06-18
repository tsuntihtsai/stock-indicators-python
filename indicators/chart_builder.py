import pandas as pd

def calculate_volume_ma(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算成交量 5日與 20日均線
    """
    if 'volume' in df.columns:
        df['v_ma5'] = df['volume'].rolling(window=5).mean()
        df['v_ma20'] = df['volume'].rolling(window=20).mean()
    return df
