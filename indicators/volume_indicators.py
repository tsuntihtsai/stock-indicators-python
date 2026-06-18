import pandas as pd

def calculate_volume_ma(df: pd.DataFrame) -> pd.DataFrame:
    # 這裡開头也是 4 個空格
    if 'volume' in df.columns:
        # 這裡開頭則是「剛好 8 個空格」
        df['v_ma5'] = df['volume'].rolling(window=5).mean()
        df['v_ma20'] = df['volume'].rolling(window=20).mean()
    return df
