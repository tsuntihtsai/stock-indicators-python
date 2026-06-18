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
    if not payload.data:
        raise HTTPException(status_code=400, detail="Data list cannot be empty")
        
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
        
        # 🔴 核心升級：將二進位圖檔轉為 Base64 字串，準備塞進 JSON 中
        image_base64 = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')
        
        # 5. 🔴 將包含高階指標的 DataFrame 轉回 JSON 字典格式 (重設索引讓日期變成欄位)
        df_json = df_out.reset_index()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        metrics_list = df_json.to_dict(orient='records')
        
        # 6. 👑 雙棲回傳：同時把「新指標 JSON」與「圖片文字」打包成一個大 JSON 回傳
        return {
            "status": "success",
            "image_data": f"data:image/png;base64,{image_base64}", # 圖片就在這
            "metrics": metrics_list                               # 新指標都在這！
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")
