import os
import sys
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 確保能正確匯入專案模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd

def create_sample_data():
    """產生 100 天的模擬股票 OHLCV 資料（實際使用時可替換成從 API 或 CSV 取得的資料）"""
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=100)
    close_prices = 150 + np.cumsum(np.random.randn(100) * 2.5)
    high_prices = close_prices + np.abs(np.random.randn(100) * 2)
    low_prices = close_prices - np.abs(np.random.randn(100) * 2)
    
    return pd.DataFrame({
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices
    }, index=dates)

def main():
    # 1. 取得並計算指標
    df = create_sample_data()
    df_out = calculate_kd_rsi_ma_macd(df)
    
    # 2. 開始繪製多合一趨勢圖 (使用 subplots 進行佈局)
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    # Panel 1: 收盤價與 5日/20日 均線
    axes[0].plot(df_out.index, df_out['Close'], label='Close Price', color='black', linewidth=1.5)
    axes[0].plot(df_out.index, df_out['MA_5'], label='MA 5', color='blue', linestyle='--', linewidth=1)
    axes[0].plot(df_out.index, df_out['MA_20'], label='MA 20', color='orange', linestyle='--', linewidth=1)
    axes[0].set_title('Stock Price & Moving Averages')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: KD 指標
    axes[1].plot(df_out.index, df_out['%K'], label='%K', color='dodgerblue', linewidth=1)
    axes[1].plot(df_out.index, df_out['%D'], label='%D', color='darkorange', linewidth=1)
    axes[1].axhline(80, color='red', linestyle=':', alpha=0.5, label='Overbought (80)')
    axes[1].axhline(20, color='green', linestyle=':', alpha=0.5, label='Oversold (20)')
    axes[1].set_title('KD Indicator')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: RSI 指標
    axes[2].plot(df_out.index, df_out['RSI'], label='RSI', color='purple', linewidth=1)
    axes[2].axhline(70, color='red', linestyle=':', alpha=0.5, label='Overbought (70)')
    axes[2].axhline(30, color='green', linestyle=':', alpha=0.5, label='Oversold (30)')
    axes[2].set_title('RSI Indicator')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    # Panel 4: MACD 指標 (含柱狀圖)
    axes[3].plot(df_out.index, df_out['MACD_Line'], label='MACD DIF', color='blue', linewidth=1)
    axes[3].plot(df_out.index, df_out['Signal_Line'], label='Signal Line', color='orange', linewidth=1)
    # 根據正負值設定柱狀圖顏色
    hist_colors = ['green' if x >= 0 else 'red' for x in df_out['MACD_Hist']]
    axes[3].bar(df_out.index, df_out['MACD_Hist'], label='MACD Hist', color=hist_colors, alpha=0.6, width=0.8)
    axes[3].set_title('MACD Indicator')
    axes[3].legend(loc='upper left')
    axes[3].grid(True, alpha=0.3)
    
    # 調整排版並儲存圖檔
    plt.tight_layout()
    output_filename = 'stock_indicators_chart.png'
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)
    print(f"成功產出數據圖：{output_filename}")

if __name__ == "__main__":
    main()
