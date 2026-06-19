import os
import sys
import io
import base64
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 確保引入模組路徑正確
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
    data: List[OHLCVFM]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Stock Indicators API is running!"}

@app.post("/analyze")
def analyze_stock_v3(payload: IndicatorRequestFM):
    if not payload.data:
        raise HTTPException(status_code=400, detail="Data list cannot be empty")
    
    if len(payload.data) < 20:
        raise HTTPException(status_code=400, detail="Data length must be at least 20 days")
        
    try:
        # 1. 建立 DataFrame
        df = pd.DataFrame([row.model_dump() for row in payload.data])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 2. 變更欄位名稱以符合原有指標函式
        df.rename(columns={
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'volume': 'Volume'
        }, inplace=True)
        
        # 3. 執行指標計算
        df_out = calculate_kd_rsi_ma_macd(df)
        df_out.columns = df_out.columns.str.lower()
        df_out = calculate_bollinger_bands(df_out)
        df_out = calculate_volume_ma(df_out)
        
        # 4. 強制就地補算高階數據
        df_out['donchian_up'] = df_out['high'].rolling(window=20).max()
        df_out['donchian_low'] = df_out['low'].rolling(window=20).min()
        
        high_s = df_out['high']
        low_s = df_out['low']
        close_p = df_out['close'].shift(1)
        tr = pd.concat([high_s - low_s, (high_s - close_p).abs(), (low_s - close_p).abs()], axis=1).max(axis=1)
        df_out['atr'] = tr.rolling(window=14).mean()
        
        # 5. 繪製圖表
        chart_buffer = draw_ultimate_chart(df_out)
        
        # 6. 圖檔轉 Base64 字串
        image_base64 = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')
        
        # 7. 轉回 JSON 格式並擷取最新一筆
        df_json = df_out.reset_index()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        latest_metrics = df_json.tail(1).to_dict(orient='records')[0]
        
        return {
            "status": "success",
            "image_data": f"data:image/png;base64,{image_base64}",
            "metrics": latest_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
