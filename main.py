import io
import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import matplotlib.pyplot as plt

# 確保能正確引入 indicators 資料夾中的模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd

app = FastAPI(title="Stock Indicators API for n8n")

# 定義 n8n 傳入的資料格式
class OHLCVRow(BaseModel):
    Date: str
    High: float
    Low: float
    Close: float

class IndicatorRequest(BaseModel):
    data: list[OHLCVRow]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Stock Indicators API is running!"}

from datetime import timedelta  # 🔴 修正 1：務必確保有引入 timedelta

@app.post("/analyze")
def analyze_stock(payload: IndicatorRequest):
    if not payload.data:
        raise HTTPException(status_code=400, detail="Data list cannot be empty")
        
    try:
        # 1. 將 n8n 傳來的 JSON 陣列轉換成 Pandas DataFrame
        df = pd.DataFrame([row.model_dump() for row in payload.data])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True) # 確保時間依序排列
        
        # 🔴 修正 2：如果 n8n 傳來的是小寫欄位名，自動與後續的畫 K 線邏輯完美相容
        df.columns = df.columns.str.lower()
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'max': 'High', 
            'low': 'Low', 'min': 'Low', 'close': 'Close'
        })
        
        # 2. 呼叫技術指標計算
        df_out = calculate_kd_rsi_ma_macd(df)
        
        # ==================== 🔴 核心修復：防止均線拉扯，同時保留足夠天數 ====================
        import numpy as np
        if len(df_out) > 20:
            df_out.iloc[:20, df_out.columns.get_loc('MA_5')] = np.nan
            df_out.iloc[:20, df_out.columns.get_loc('MA_20')] = np.nan
        # =========================================================================
        
        # 3. 繪製技術指標多合一數據圖
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, 
                                 gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        # 🔴 修正 3：提前計算 Y 軸上下邊距，避免後續計算十字線高度時噴錯
        y_min, y_max = df_out['Close'].min() * 0.95, df_out['Close'].max() * 1.05
        axes[0].set_ylim(y_min, y_max)
        
        # --- 🔴 修正 4：分析師等級——飽滿 K 線蠟燭棒繪製 (使用 Rectangle) ---
        from matplotlib.patches import Rectangle
        
        # 設定蠟燭實體的寬度 (以天數為單位，0.6 筆 K 棒寬度是最舒服、不擁擠的黃金比例)
        candle_width = 0.6  
        
        for idx, row in df_out.iterrows():
            o = row['Open'] if 'Open' in row else row['Close']
            h = row['High'] if 'High' in row else row['Close']
            l = row['Low'] if 'Low' in row else row['Close']
            c = row['Close']
            
            # 1. 判定漲跌顏色
            if c >= o:
                color = 'red'       # 漲：紅色
                fill = True         # 實心
                bottom_y = o
                height = c - o
            else:
                color = 'green'     # 跌：綠色
                fill = True         # 跌棒同樣填滿，看盤最清晰
                bottom_y = c
                height = o - c
                
            # 2. 畫上下影線 (一條貫穿 High 到 Low 的細線)
            axes[0].vlines(idx, l, h, color=color, linewidth=1.2, zorder=2)
            
            # 3. 畫蠟燭實體方塊 (Rectangle)
            # 因為 idx 是置中的時間點，矩形起點要往左移半個寬度，方塊才會完美對齊影線中心
            rect_left = idx - timedelta(hours=int(24 * candle_width / 2))
            
            # 如果漲跌幅為 0 (十字線)，給予一個極小的最低高度，避免方塊完全消失
            if height == 0:
                height = (y_max - y_min) * 0.002
                
            rect = Rectangle(
                (rect_left, bottom_y), 
                timedelta(hours=int(24 * candle_width)), 
                height, 
                facecolor=color, 
                edgecolor=color,
                fill=fill,
                zorder=3
            )
            axes[0].add_patch(rect)
        # --- 🔴 蠟燭棒繪製結束 ---

        # 均線改為「細實線」，更具質感，不再用喧賓奪主的粗虛線
        axes[0].plot(df_out.index, df_out['MA_5'], label='MA 5', color='blue', linewidth=1.0)
        axes[0].plot(df_out.index, df_out['MA_20'], label='MA 20', color='orange', linewidth=1.0)
        axes[0].set_title('Stock Price & Moving Averages (Professional Candlestick)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.2, linestyle='-')
        
        # 副圖 1：KD 指標 (優化格線與粗細)
        axes[1].plot(df_out.index, df_out['%K'], label='%K', color='dodgerblue', linewidth=1.2)
        axes[1].plot(df_out.index, df_out['%D'], label='%D', color='darkorange', linewidth=1.2)
        axes[1].axhline(80, color='red', linestyle=':', alpha=0.4)
        axes[1].axhline(20, color='green', linestyle=':', alpha=0.4)
        axes[1].set_title('KD Indicator')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.2)
        
        # 副圖 2：RSI 指標
        axes[2].plot(df_out.index, df_out['RSI'], label='RSI', color='purple', linewidth=1.2)
        axes[2].axhline(70, color='red', linestyle=':', alpha=0.4)
        axes[2].axhline(30, color='green', linestyle=':', alpha=0.4)
        axes[2].set_title('RSI Indicator')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.2)
        
        # 副圖 3：MACD 指標 (台股經典配色：紅正綠負)
        axes[3].plot(df_out.index, df_out['MACD_Line'], label='MACD Line', color='blue', linewidth=1.2)
        axes[3].plot(df_out.index, df_out['Signal_Line'], label='Signal Line', color='orange', linewidth=1.2)
        
        # MACD 柱狀圖改為紅（正值）綠（負值）
        macd_colors = ['red' if x >= 0 else 'green' for x in df_out['MACD_Hist']]
        axes[3].bar(df_out.index, df_out['MACD_Hist'], color=macd_colors, alpha=0.7, width=0.6)
        axes[3].axhline(0, color='gray', linestyle='-', alpha=0.2)
        axes[3].set_title('MACD Indicator')
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        # 將圖片儲存到記憶體緩衝區
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=140)
        buf.seek(0)
        plt.close(fig)
        
        # 4. 直接將二進位圖檔串流回傳給 n8n
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")
