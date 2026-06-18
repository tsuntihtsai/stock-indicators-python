import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 確保引入模組路徑正確
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd  # 原本的舊指標函式
from indicators.bollinger_bands import calculate_bollinger_bands
from indicators.volume_indicators import calculate_volume_ma
from indicators.chart_builder import draw_ultimate_chart

app = FastAPI(title="Stock Indicators API for n8n")

# 精準對接 n8n 傳來的首字大寫 JSON 欄位
class OHLCVFM(BaseModel):
    date: str = Field(alias="Date")
    open: float = Field(alias="Open")
    high: float = Field(alias="High")
    low: float = Field(alias="Low")
    close: float = Field(alias="Close")
    volume: float = Field(default=0.0, alias="Trading_Volume")

class IndicatorRequestFM(BaseModel):
    data: list[OHLCVFM]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Stock Indicators API is running!"}

@app.post("/analyze")
def analyze_stock(payload: IndicatorRequestFM):
    if not payload.data:
        raise HTTPException(status_code=400, detail="Data list cannot be empty")
        
    try:
        # 1. 將資料轉為 DataFrame (Pydantic 預設轉成小寫欄位名)
        df = pd.DataFrame([row.model_dump() for row in payload.data])
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 🔴 修正關鍵：為了讓你原本不動的舊函式不崩潰，先將欄位更名為首字大寫
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        # 2. 🟢 執行你原本的技術指標計算 (它現在能開心地看到 'High', 'Low', 'Close' 了！)
        df_out = calculate_kd_rsi_ma_macd(df)
        
        # 🔴 修正關鍵：為了配合新拆分出去的布林、量能指標與畫圖模組，我們統一將所有欄位轉回小寫
        df_out.columns = df_out.columns.str.lower()
        
        # 3. 🔴 依序疊加跑新檔案裡的指標計算 (吃小寫欄位)
        df_out = calculate_bollinger_bands(df_out)
        df_out = calculate_volume_ma(df_out)
        
        # 4. 呼叫外部的純繪圖模組 (吃小寫欄位)
        chart_buffer = draw_ultimate_chart(df_out)
        
        return StreamingResponse(chart_buffer, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
