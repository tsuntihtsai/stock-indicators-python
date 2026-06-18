import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 確保引入模組路徑正確
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd
from indicators.bollinger_bands import calculate_bollinger_bands
from indicators.volume_indicators import calculate_volume_ma
from indicators.chart_builder import draw_ultimate_chart

app = FastAPI(title="Stock Indicators API for n8n")

# --- 🔴 核心修正：將模型欄位精準改為首字大寫，對齊 n8n 輸出 ---
class OHLCVFM(BaseModel):
    date: str = Field(alias="Date")          # 👈 精準對齊 "Date"
    open: float = Field(alias="Open")        # 👈 精準對齊 "Open"
    high: float = Field(alias="High")        # 👈 精準對齊 "High"
    low: float = Field(alias="Low")          # 👈 精準對齊 "Low"
    close: float = Field(alias="Close")      # 👈 精準對齊 "Close"
    volume: float = Field(default=0.0, alias="Trading_Volume") # 👈 精準對齊你的成交量欄位名！

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
        # 1. 資料打包進 DataFrame
        # 這裡 model_dump() 預設會轉成欄位名稱(全小寫: date, open, high, low, close, volume)
        df = pd.DataFrame([row.model_dump() for row in payload.data])
        
        # 2. 轉換日期索引與排序
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 3. 執行指標計算接力賽
        df_out = calculate_kd_rsi_ma_macd(df)       # 計算原有指標
        df_out = calculate_bollinger_bands(df_out)  # 新檔案 1：布林
        df_out = calculate_volume_ma(df_out)       # 新檔案 2：量均線
        
        # 4. 呼叫外部的純繪圖模組
        chart_buffer = draw_ultimate_chart(df_out)
        
        return StreamingResponse(chart_buffer, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
