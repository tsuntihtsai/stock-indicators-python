import import os
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
    
    if len(payload.data) < 20:
        raise HTTPException(status_code=400, detail="資料天數不足 20 天，無法計算技術指標。")
        
    try:
        # 1. 建立 DataFrame
        df = pd.DataFrame([row.model_dump() for row in payload.data])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 2. 迎合舊函式的大寫更名
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        # 3. 執行指標接力計算
        df_out = calculate_kd_rsi_ma_macd(df)
        df_out.columns = df_out.columns.str.lower()
        df_out = calculate_bollinger_bands(df_out)
        df_out = calculate_volume_ma(df_out)
        
        # 🔴 核心修復：在 main.py 裡「百分之百確保」高階指標被計算並存入 df_out 中
        # A. 唐奇安通道
        df_out['donchian_up'] = df_out['high'].rolling(window=20).max()
        df_out['donchian_low'] = df_out['low'].rolling(window=20).min()
        
        # B. ATR 真實發展波幅 (14日)
        high_series = df_out['high']
        low_series = df_out['low']
        close_prev = df_out['close'].shift(1)
        tr = pd.concat([high_series - low_series, (high_series - close_prev).abs(), (low_series - close_prev).abs()], axis=1).max(axis=1)
        df_out['atr'] = tr.rolling(window=14).mean()
        
        # 4. 產出圖片 buffer (此時 df_out 已包含所有小寫高階欄位)
        chart_buffer = draw_ultimate_chart(df_out)
        
        # 5. 將二進位圖檔轉為 Base64 字串
        image_base64 = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')
        
        # 6. 將包含所有技術指標的 DataFrame 轉回 JSON 字典格式
        df_json = df_out.reset_index()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        
        # 只提取最新那一天（當天）的全部數據回傳給 n8n
        latest_metrics = df_json.tail(1).to_dict(orient='records')[0]
        
        return {
            "status": "success",
            "image_data": f"data:image/png;base64,{image_base64}",
            "metrics": latest_metrics  # 👑 這裡面現在百分之百包含小寫的 atr 與 donchian_up 了！
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
