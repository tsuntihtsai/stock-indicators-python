import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 確保引入模組路徑正確
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd  # 🟢 原本的程式，完全不動
from indicators.bollinger_bands import calculate_bollinger_bands  # 🔴 新指標 1
from indicators.volume_indicators import calculate_volume_ma      # 🔴 新指標 2
from indicators.chart_builder import draw_ultimate_chart

app = FastAPI(title="Stock Indicators API for n8n")

class OHLCVFM(BaseModel):
    date: str = Field(alias="date")
    open: float = Field(alias="open")
    high: float = Field(alias="max")
    low: float = Field(alias="min")
    close: float = Field(alias="close")

class IndicatorRequestFM(BaseModel):
    msg: str | None = None
    status: int | None = None
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
        df = pd.DataFrame([row.model_dump() for row in payload.data])
        df.columns = df.columns.str.lower()
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 2. 🟢 執行你原本不動的技術指標計算
        df_out = calculate_kd_rsi_ma_macd(df)
        
        # 3. 🔴 依序疊加跑新檔案裡的指標計算
        df_out = calculate_bollinger_bands(df_out)
        df_out = calculate_volume_ma(df_out)
        
        # 4. 呼叫外部的純繪圖模組
        chart_buffer = draw_ultimate_chart(df_out)
        
        return StreamingResponse(chart_buffer, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
