import io
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_ultimate_chart(df_out):
    """
    純繪圖函式：接收算好指標的 DataFrame，畫出五合一看盤圖
    """
    # ==================== 繪圖前：前 20 天未完成指標的 NaN 排毒處理 ====================
    ma5_col = 'MA_5' if 'MA_5' in df_out.columns else 'ma_5'
    ma20_col = 'MA_20' if 'MA_20' in df_out.columns else 'ma_20'
    k_col = '%K' if '%K' in df_out.columns else '%k'
    d_col = '%D' if '%D' in df_out.columns else '%d'
    rsi_col = 'RSI' if 'RSI' in df_out.columns else 'rsi'
    
    if len(df_out) > 20:
        nan_cols = [ma5_col, ma20_col, 'bb_mid', 'bb_up', 'bb_low', 'v_ma20', k_col, d_col, rsi_col]
        for col in nan_cols:
            if col in df_out.columns:
                df_out.iloc[:20, df_out.columns.get_loc(col)] = np.nan
        if 'v_ma5' in df_out.columns:
            df_out.iloc[:5, df_out.columns.get_loc('v_ma5')] = np.nan
    # =========================================================================

    # 1. 建立 5 個子圖
    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True, 
                             gridspec_kw={'height_ratios': [2.8, 1.2, 1, 1, 1]})
    
    # 自動調整主圖 Y 軸邊距
    y_min = df_out['bb_low'].dropna().min() * 0.98 if 'bb_low' in df_out.columns and not df_out['bb_low'].dropna().empty else df_out['close'].min() * 0.95
    y_max = df_out['bb_up'].dropna().max() * 1.02 if 'bb_up' in df_out.columns and not df_out['bb_up'].dropna().empty else df_out['close'].max() * 1.05
    axes[0].set_ylim(y_min, y_max)
    
    # 2. 繪製標準台股紅綠 K 線蠟燭棒
    candle_width = 0.7
    for idx, row in df_out.iterrows():
        o, c = float(row['open']), float(row['close'])
        h = float(row['max'] if 'max' in row else row['high'])
        l = float(row['min'] if 'min' in row else row['low'])
        
        color = 'red' if c > o else ('green' if c < o else 'red')
        bottom_y = o if c > o else c
        height = abs(c - o)
        
        if height < (y_max - y_min) * 0.003:
            height = (y_max - y_min) * 0.003
            
        axes[0].vlines(idx, l, h, color=color, linewidth=1.2, zorder=2)
        rect_left = idx - timedelta(hours=int(24 * candle_width / 2))
        rect = Rectangle((rect_left, bottom_y), timedelta(hours=int(24 * candle_width)), height, 
                         facecolor=color, edgecolor=color, fill=True, zorder=3)
        axes[0].add_patch(rect)

     # 🔴 疊加繪製唐奇安通道 (不要畫得太粗，用 alpha=0.4 淡淡的襯托在背景即可)
    if 'donchian_up' in df_out.columns:
            axes[0].plot(df_out.index, df_out['donchian_up'], color='dodgerblue', linewidth=0.7, linestyle=':', alpha=0.5, label='Donchian Up')
            axes[0].plot(df_out.index, df_out['donchian_low'], color='dodgerblue', linewidth=0.7, linestyle=':', alpha=0.5, label='Donchian Low')
        
    # 3. 主圖軌道與均線繪製 (布林通道 + MA)
    axes[0].plot(df_out.index, df_out[ma5_col], label='MA 5', color='blue', linewidth=0.8)
    axes[0].plot(df_out.index, df_out[ma20_col], label='MA 20 (BB Mid)', color='orange', linewidth=1.0)
    
    if 'bb_up' in df_out.columns:
        axes[0].plot(df_out.index, df_out['bb_up'], label='BB Upper', color='#b0b0b0', linewidth=0.8)
        axes[0].plot(df_out.index, df_out['bb_low'], label='BB Lower', color='#b0b0b0', linewidth=0.8)
        axes[0].fill_between(df_out.index, df_out['bb_up'], df_out['bb_low'], color='#f5f5f5', alpha=0.3, zorder=1)
    
    axes[0].set_title('Stock Price, MA & Bollinger Bands', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.15)
    
    # 4. 副圖 1：成交量區塊 (今日收 >= 開 畫紅柱，反之綠柱)
    if 'volume' in df_out.columns:
        v_colors = ['red' if float(r['close']) >= float(r['open']) else 'green' for i, r in df_out.iterrows()]
        axes[1].bar(df_out.index, df_out['volume'], color=v_colors, alpha=0.7, width=0.7, label='Volume')
        if 'v_ma5' in df_out.columns:
            axes[1].plot(df_out.index, df_out['v_ma5'], label='V_MA 5', color='blue', linewidth=0.8)
            axes[1].plot(df_out.index, df_out['v_ma20'], label='V_MA 20', color='orange', linewidth=0.8)
        axes[1].set_title('Volume & Volume MA', fontsize=10, fontweight='bold')
        axes[1].legend(loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.15)
        axes[1].set_ylim(0, df_out['volume'].max() * 1.1)
    
    # 副圖 2：KD 指標
    axes[2].plot(df_out.index, df_out[k_col], label='%K', color='dodgerblue', linewidth=1.2)
    axes[2].plot(df_out.index, df_out[d_col], label='%D', color='darkorange', linewidth=1.2)
    axes[2].axhline(80, color='red', linestyle=':', alpha=0.4)
    axes[2].axhline(20, color='green', linestyle=':', alpha=0.4)
    axes[2].set_title('KD Indicator', fontsize=10, fontweight='bold')
    axes[2].legend(loc='upper left', fontsize=8)
    axes[2].grid(True, alpha=0.15)
    
    # 副圖 3：RSI 指標
    axes[3].plot(df_out.index, df_out[rsi_col], label='RSI', color='purple', linewidth=1.2)
    axes[3].axhline(70, color='red', linestyle=':', alpha=0.4)
    axes[3].axhline(30, color='green', linestyle=':', alpha=0.4)
    axes[3].set_title('RSI Indicator', fontsize=10, fontweight='bold')
    axes[3].legend(loc='upper left', fontsize=8)
    axes[3].grid(True, alpha=0.15)
    
    # 副圖 4：MACD 指標
    macd_line = 'MACD_Line' if 'MACD_Line' in df_out.columns else 'macd_line'
    sig_line = 'Signal_Line' if 'Signal_Line' in df_out.columns else 'signal_line'
    hist_col = 'MACD_Hist' if 'MACD_Hist' in df_out.columns else 'macd_hist'
    
    axes[4].plot(df_out.index, df_out[macd_line], label='MACD Line', color='blue', linewidth=1.2)
    axes[4].plot(df_out.index, df_out[sig_line], label='Signal Line', color='orange', linewidth=1.2)
    macd_colors = ['red' if x >= 0 else 'green' for x in df_out[hist_col]]
    axes[4].bar(df_out.index, df_out[hist_col], color=macd_colors, alpha=0.7, width=0.6)
    axes[4].axhline(0, color='gray', linestyle='-', alpha=0.2)
    axes[4].set_title('MACD Indicator', fontsize=10, fontweight='bold')
    axes[4].legend(loc='upper left', fontsize=8)
    axes[4].grid(True, alpha=0.15)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=145)
    buf.seek(0)
    plt.close(fig)
    return buf
