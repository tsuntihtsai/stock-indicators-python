import pandas as pd
 
 
def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_up'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_low'] = df['bb_mid'] - (df['bb_std'] * 2)
    # 布林通道寬度（相對於中軌的百分比），用來判斷目前波動率是收縮還是擴張，
    # 數字越小代表通道越窄、波動越低，常是變盤（大漲或大跌）前兆。
    # 實際「是否處於收縮狀態」的判斷（跟歷史比較）在main.py的determine_band_squeeze()處理，
    # 這裡只負責算出寬度本身。
    df['bb_width'] = (df['bb_up'] - df['bb_low']) / df['bb_mid'] * 100
    return df
