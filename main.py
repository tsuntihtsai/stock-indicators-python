import os
import sys
import io
import base64
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd
from indicators.bollinger_bands import calculate_bollinger_bands
from indicators.volume_indicators import calculate_volume_ma
from indicators.chart_builder import draw_ultimate_chart

app = FastAPI(title="Stock Indicators Ultimate API")

class OHLCVFM(BaseModel):
    date: str = Field(alias="Date")
    open: float = Field(alias="Open")
    high: float = Field(alias="High")
    low: float = Field(alias="Low")
    close: float = Field(alias="Close")
    volume: float = Field(default=0.0, alias="Trading_Volume")

class IndicatorRequestFM(BaseModel):
    data: list[OHLCVFM]

@app.post("/analyze")
def analyze_stock_v3(payload: IndicatorRequestFM):
    # 🔴 防禦機制 1：檢查資料是否為空
    if not payload.data:
        raise HTTPException(status_code=400, detail="Data list cannot be empty")
    
    # 🔴 防禦機制 2：檢查資料天數是否足夠計算 20日均線與布林通道
    if len(payload.data) < 20:
        raise HTTPException(
            status_code=400, 
            detail=f"資料天數不足！計算布林與技術指標至少需要 20 天的歷史數據，目前只有 {len(payload.data)} 天。請調整 n8n 撈取的時間區間。"
        )
        
    try:
        # 1. 建立 DataFrame
        df = pd.DataFrame([row.model_dump() for row in payload.data])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 2. 迎合舊函式的大寫更名
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        # 3. 執行接力計算
        df_out = calculate_kd_rsi_ma_macd(df)
        df_out.columns = df_out.columns.str.lower()
        df_out = calculate_bollinger_bands(df_out)
        df_out = calculate_volume_ma(df_out)
        
        # 4. 產出圖片 buffer
        chart_buffer = draw_ultimate_chart(df_out)
        
        # 5. 將二進位圖檔轉為 Base64 字串
        image_base64 = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')
        
        # 6. 將包含高階指標的 DataFrame 轉回 JSON 字典格式
        df_json = df_out.reset_index()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        
        # 🔴 為了防止 JSON 體積過大，我們只回傳「最新那一天（當天）」的指標數據給 AI Agent
        # 如果你希望回傳整段，可以改回 df_json.to_dict(orient='records')
        latest_metrics = df_json.tail(1).to_dict(orient='records')[0]
        
        return {
            "status": "success",
            "image_data": f"data:image/png;base64,{image_base64}",
            "metrics": latest_metrics  # 👑 只回傳最新一天的精準指標，對 AI 來說完全足夠，且速度極快！
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
