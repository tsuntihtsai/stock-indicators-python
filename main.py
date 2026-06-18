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
        
        # 2. 呼叫您現有的技術指標計算函式
        
        df_out = calculate_kd_rsi_ma_macd(df)
        
        # ==================== 🔴 核心修復：徹底消滅掉到 0 的線 ====================
        # 因為計算 20MA 需要 20 天的資料，前 20 筆是指標未完成的無效資料（常常會是 0）
        # 我們直接用 .iloc[20:] 把前 20 筆剔除掉，這樣畫圖就不會從 0 開始飆上去了！
        df_out = df_out.iloc[20:]
        # =========================================================================
        
        # 3. 繪製技術指標多合一數據圖 (接下來維持你原本的繪圖程式碼...)
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, 
                                 gridspec_kw={'height_ratios': [2, 1, 1, 1]})

        # 主圖：股價與均線
        axes[0].plot(df_out.index, df_out['Close'], label='Close Price', color='black', linewidth=1.5)
        axes[0].plot(df_out.index, df_out['MA_5'], label='MA 5', color='blue', linestyle='--')
        axes[0].plot(df_out.index, df_out['MA_20'], label='MA 20', color='orange', linestyle='--')
        axes[0].set_title('Stock Price & Moving Averages')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # 副圖 1：KD 指標
        axes[1].plot(df_out.index, df_out['%K'], label='%K', color='dodgerblue')
        axes[1].plot(df_out.index, df_out['%D'], label='%D', color='darkorange')
        axes[1].axhline(80, color='red', linestyle=':', alpha=0.5)
        axes[1].axhline(20, color='green', linestyle=':', alpha=0.5)
        axes[1].set_title('KD Indicator')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # 副圖 2：RSI 指標
        axes[2].plot(df_out.index, df_out['RSI'], label='RSI', color='purple')
        axes[2].axhline(70, color='red', linestyle=':', alpha=0.5)
        axes[2].axhline(30, color='green', linestyle=':', alpha=0.5)
        axes[2].set_title('RSI Indicator')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # 副圖 3：MACD 指標
        axes[3].plot(df_out.index, df_out['MACD_Line'], label='MACD Line', color='blue')
        axes[3].plot(df_out.index, df_out['Signal_Line'], label='Signal Line', color='orange')
        hist_colors = ['green' if x >= 0 else 'red' for x in df_out['MACD_Hist']]
        axes[3].bar(df_out.index, df_out['MACD_Hist'], color=hist_colors, alpha=0.6, width=0.8)
        axes[3].set_title('MACD Indicator')
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 將圖片儲存到記憶體緩衝區 (BytesIO)，不佔用伺服器硬碟空間
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=140)
        buf.seek(0)
        plt.close(fig)
        
        # 4. 直接將二進位圖檔串流回傳給 n8n
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")
