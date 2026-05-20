import pandas as pd
from stock_indicators_python.indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd


def test_basic_run():
    # Create a tiny OHLCV DataFrame
    data = {
        'High': [101, 102, 103, 104, 105, 106, 107, 108],
        'Low':  [99, 100, 101, 102, 103, 104, 105, 106],
        'Close':[100, 101, 102, 103, 104, 105, 106, 107],
    }
    df = pd.DataFrame(data)
    df_out = calculate_kd_rsi_ma_macd(df, k_period=14, d_period=3, rsi_period=14,
                                   ma_short=5, ma_long=20,
                                   macd_fast=12, macd_slow=26, macd_signal=9)
    # Check expected columns exist
    for col in ['%K', '%D', 'RSI', 'MA_5', 'MA_20', 'MACD_Line', 'Signal_Line', 'MACD_Hist', 'Typical_Price']:
        assert col in df_out.columns
    assert not df_out.empty
