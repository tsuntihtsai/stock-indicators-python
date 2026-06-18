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
        
        # 2. 呼叫技術指標計算
        df_out = calculate_kd_rsi_ma_macd(df)
        
        # ==================== 🔴 核心修復：防止均線拉扯，同時保留足夠天數 ====================
        # 為了避免均線在前 20 天掉到 0，我們「不要」直接用 iloc[20:] 把整筆 K 線刪掉。
        # 正確做法：保留完整的 K 線，只把前 20 天未算好的均線強制變為 None，這樣繪圖就不會連到 0。
        import numpy as np
        if len(df_out) > 20:
            df_out.iloc[:20, df_out.columns.get_loc('MA_5')] = np.nan
            df_out.iloc[:20, df_out.columns.get_loc('MA_20')] = np.nan
        # =========================================================================
        
        # 3. 繪製技術指標多合一數據圖
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, 
                                 gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        # --- 🔴 專業改造：將主圖從「單條折線」改為「台股專業 K 線蠟燭圖」 ---
        # 為了畫 K 線，我們需要確定你的 df 裡有 Open, High, Low, Close。
        # 這裡動態從數據中獲取（若 payload 沒有，則用 Close 代替避免崩潰）
        for idx, row in df_out.iterrows():
            # 判斷漲跌顏色 (台股習慣：紅漲綠跌)
            # 註：如果你的 payload row 有 open/max/min 屬性，請替換對應欄位，這裡做安全相容處理
            o_val = getattr(row, 'Open', row['Close'] * 1.001)  # 示意相容
            h_val = getattr(row, 'High', row['Close'] * 1.005)
            l_val = getattr(row, 'Low', row['Close'] * 0.995)
            c_val = row['Close']
            
            color = 'red' if c_val >= o_val else 'green'
            
            # 畫影線 (High 到 Low)
            axes[0].vlines(idx, l_val, h_val, color=color, linewidth=1)
            # 畫實體棒 (Open 到 Close)
            axes[0].vlines(idx, o_val, c_val, color=color, linewidth=5)

        # 均線改為「細實線」，更具質感，不再用喧賓奪主的粗虛線
        axes[0].plot(df_out.index, df_out['MA_5'], label='MA 5', color='blue', linewidth=1.0)
        axes[0].plot(df_out.index, df_out['MA_20'], label='MA 20', color='orange', linewidth=1.0)
        axes[0].set_title('Stock Price & Moving Averages (Professional Candlestick)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.2, linestyle='-')
        
        # Y軸自動微調，留出 5% 的上下邊距讓圖表呼吸
        y_min, y_max = df_out['Close'].min() * 0.95, df_out['Close'].max() * 1.05
        axes[0].set_ylim(y_min, y_max)
        
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
        
        # 🔴 修改：MACD 柱狀圖改為紅（正值）綠（負值）
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
