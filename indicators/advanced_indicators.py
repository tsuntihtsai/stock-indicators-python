import pandas as pd
import numpy as np

def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算高階量化指標：ATR, Fibonacci 關鍵位, Donchian Channel, Stochastic RSI
    """
    df = df.sort_index()
    
    # --- 1. ATR (真實發展波幅 - 預設 14 日) ---
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # --- 2. Donchian Channel (唐奇安通道 - 預設 20 日) ---
    df['donchian_up'] = df['high'].rolling(window=20).max()
    df['donchian_low'] = df['low'].rolling(window=20).min()
    
    # --- 3. Fibonacci 回撤關鍵支撐位 (滾動 60 日波段抓極值) ---
    # 自動找出過去 60 天的最高與最低點，拿來算目前的黃金分割率
    roll_high = df['high'].rolling(window=60).max()
    roll_low = df['low'].rolling(window=60).min()
    diff = roll_high - roll_low
    
    df['fib_382'] = roll_high - diff * 0.382
    df['fib_500'] = roll_high - diff * 0.500
    df['fib_618'] = roll_high - diff * 0.618
    
    # --- 4. Stochastic RSI (隨機相對強弱指標 - 預設 14, 3, 3) ---
    # 先確保有 rsi 欄位，如果沒有就就地簡易算一個
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
    rsi_min = df['rsi'].rolling(window=14).min()
    rsi_max = df['rsi'].rolling(window=14).max()
    
    stoch_rsi = ((df['rsi'] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)) * 100
    df['stoch_rsi_k'] = stoch_rsi.rolling(window=3).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
    
    # 排毒處理
    if len(df) > 20:
        for col in ['atr', 'donchian_up', 'donchian_low', 'fib_382', 'fib_500', 'fib_618', 'stoch_rsi_k', 'stoch_rsi_d']:
            df.iloc[:20, df.columns.get_loc(col)] = np.nan
            
    return df
