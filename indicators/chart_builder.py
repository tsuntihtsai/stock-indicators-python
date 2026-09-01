import io
import numpy as np
from datetime import timedelta
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

# ==================== 中文字型設定 ====================
# 修正說明：matplotlib 預設字型（DejaVu Sans）不支援中文字，
# 如果股票名稱（例如「台積電」）被放進圖表標題，字會顯示成
# 一格一格的空白方塊（missing glyph）。這裡在載入時自動偵測
# 系統上可用的中文字型並套用，找不到就靜默略過（回到原本行為），
# 不會讓程式報錯。
_CJK_FONT_CANDIDATES = [
    'Noto Sans CJK TC', 'Noto Sans CJK JP', 'Noto Sans CJK SC',
    'Microsoft JhengHei', 'PingFang TC', 'PingFang SC', 'SimHei', 'Heiti TC'
]
_available_font_names = {f.name for f in fm.fontManager.ttflist}
for _font_name in _CJK_FONT_CANDIDATES:
    if _font_name in _available_font_names:
        matplotlib.rcParams['font.sans-serif'] = [_font_name] + matplotlib.rcParams.get('font.sans-serif', [])
        break
matplotlib.rcParams['axes.unicode_minus'] = False  # 避免套用中文字型後負號顯示異常
# =======================================================

def draw_ultimate_chart(df_out, stock_label: str = ""):
    """
    終極整合繪圖函式：自動補算高階指標，並精確排列圖層畫出

    stock_label: 選填，股票名稱或代號，會顯示在主圖標題上，
    方便AI視覺辨識這張圖對應的是哪支股票（也方便使用者肉眼確認）。
    """
    # 確保資料按時間排序
    df_out = df_out.sort_index()

    # ====== 🔴 強制就地補算「唐奇安通道」與高階指標，保證 100% 欄位存在 ======
    if 'donchian_up' not in df_out.columns:
        df_out['donchian_up'] = df_out['high'].rolling(window=20).max()
        df_out['donchian_low'] = df_out['low'].rolling(window=20).min()

    if 'atr' not in df_out.columns:
        high = df_out['high']
        low = df_out['low']
        close_prev = df_out['close'].shift(1)
        tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
        df_out['atr'] = tr.rolling(window=14).mean()
    # ======================================================================

    # ==================== 前 N 天未完成指標的 NaN 排毒處理 ====================
    ma5_col = 'MA_5' if 'MA_5' in df_out.columns else 'ma_5'
    ma20_col = 'MA_20' if 'MA_20' in df_out.columns else 'ma_20'
    k_col = '%K' if '%K' in df_out.columns else '%k'
    d_col = '%D' if '%D' in df_out.columns else '%d'
    rsi_col = 'RSI' if 'RSI' in df_out.columns else 'rsi'

    # 修正重點：原本這裡是「資料超過20天就無條件把前20天設NaN」。
    # 問題是：如果資料只有 22~30 天左右（main.py 允許送進來的最小值附近），
    # 砍掉前20天之後，圖上只剩 2~10 天可畫，KD/RSI等子圖會變成
    # 一大片空白裡飄著一小段孤立線條，看起來完全不像正常走勢圖。
    #
    # 改法：動態計算要清空幾天，確保清空之後至少留下 MIN_VISIBLE_DAYS 天
    # 可以畫，而不是死板地固定砍20天。資料量夠多時（例如100天），
    # 行為跟原本完全一樣（一樣砍20天暖機期）；資料量偏少時，
    # 少砍一點，讓圖至少看得出走勢，犧牲一點點暖機期的準確度。
    MIN_VISIBLE_DAYS = 20
    warmup_cutoff = min(20, max(len(df_out) - MIN_VISIBLE_DAYS, 0))

    if warmup_cutoff > 0:
        nan_cols = [ma5_col, ma20_col, 'bb_mid', 'bb_up', 'bb_low', 'v_ma20', k_col, d_col, rsi_col, 'donchian_up', 'donchian_low']
        for col in nan_cols:
            if col in df_out.columns:
                df_out.iloc[:warmup_cutoff, df_out.columns.get_loc(col)] = np.nan
        v_ma5_cutoff = min(5, warmup_cutoff)
        if 'v_ma5' in df_out.columns and v_ma5_cutoff > 0:
            df_out.iloc[:v_ma5_cutoff, df_out.columns.get_loc('v_ma5')] = np.nan
    # =========================================================================

    # 1. 建立 5 個子圖
    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True,
                             gridspec_kw={'height_ratios': [2.8, 1.2, 1, 1, 1]})

    # 自動調整主圖 Y 軸邊距
    # 修正重點：原本只用 bb_low/bb_up 算 y 軸範圍，資料量少時 bb 只有寥寥幾個
    # 有效值，可能比實際股價範圍窄很多，導致大部分K棒被裁切掉、畫面不完整。
    # 改成同時考慮實際 K 棒的 high/low，取「布林通道」和「K棒本身」兩者中
    # 更寬的範圍，確保所有K棒都完整顯示。
    price_low = df_out['low'].min()
    price_high = df_out['high'].max()
    bb_low_min = df_out['bb_low'].dropna().min() if 'bb_low' in df_out.columns and not df_out['bb_low'].dropna().empty else price_low
    bb_up_max = df_out['bb_up'].dropna().max() if 'bb_up' in df_out.columns and not df_out['bb_up'].dropna().empty else price_high
    y_min = min(price_low, bb_low_min) * 0.98
    y_max = max(price_high, bb_up_max) * 1.02
    axes[0].set_ylim(y_min, y_max)

    # 2. 🟢 讓布林通道背景灰色區塊先畫 (zorder=1)，才不會蓋住後面的線
    if 'bb_up' in df_out.columns:
        axes[0].plot(df_out.index, df_out['bb_up'], label='BB Upper', color='#b0b0b0', linewidth=0.8, zorder=1)
        axes[0].plot(df_out.index, df_out['bb_low'], label='BB Lower', color='#b0b0b0', linewidth=0.8, zorder=1)
        axes[0].fill_between(df_out.index, df_out['bb_up'], df_out['bb_low'], color='#f5f5f5', alpha=0.3, zorder=1)

    # 3. 繪製標準台股紅綠 K 線蠟燭棒
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

    # 4. 繪製移動平均線
    axes[0].plot(df_out.index, df_out[ma5_col], label='MA 5', color='blue', linewidth=0.8, zorder=4)
    axes[0].plot(df_out.index, df_out[ma20_col], label='MA 20 (BB Mid)', color='orange', linewidth=1.0, zorder=4)

    # 5. 🔴 疊加繪製唐奇安通道 (最高層級 zorder=5，並改用顯眼的深天藍色虛線)
    if 'donchian_up' in df_out.columns:
        axes[0].plot(df_out.index, df_out['donchian_up'], color='#00a2ff', linewidth=1.0, linestyle='--', alpha=0.8, label='Donchian Up', zorder=5)
        axes[0].plot(df_out.index, df_out['donchian_low'], color='#00a2ff', linewidth=1.0, linestyle='--', alpha=0.8, label='Donchian Low', zorder=5)

    title_text = 'Stock Price, MA & Bollinger Bands'
    if stock_label:
        title_text = f'{stock_label} — {title_text}'
    axes[0].set_title(title_text, fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.15)

    # 6. 副圖 1：成交量區塊
    if 'volume' in df_out.columns:
        v_colors = ['red' if float(r['close']) >= float(r['open']) else 'green' for i, r in df_out.iterrows()]
        axes[1].bar(df_out.index, df_out['volume'], color=v_colors, alpha=0.7, width=0.7, label='Volume')
        if 'v_ma5' in df_out.columns:
            axes[1].plot(df_out.index, df_out['v_ma5'], label='V_MA 5', color='blue', linewidth=0.8)
            axes[1].plot(df_out.index, df_out['v_ma20'], label='V_MA 20', color='orange', linewidth=0.8)
        axes[1].set_title('Volume & Volume MA', fontsize=10, fontweight='bold')
        axes[1].legend(loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.15)
        v_max = df_out['volume'].max()
        axes[1].set_ylim(0, v_max * 1.1 if v_max > 0 else 1)

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
    plt.savefig(buf, format='png', dpi=90)
    buf.seek(0)
    plt.close(fig)
    return buf
