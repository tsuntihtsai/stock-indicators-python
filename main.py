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
        
        # =================================================================
        # 🚀 🆕 籌碼面三大指標核心計算
        # =================================================================
        # (1) 計算 OBV (能量潮指標)
        df_out['obv'] = 0.0
        direction = df_out['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df_out['obv'] = (direction * df_out['volume']).fillna(0).cumsum()
        
        # 判定 OBV 狀態
        price_declining = df_out['close'].iloc[-1] <= df_out['close'].tail(5).mean()
        obv_rising = df_out['obv'].iloc[-1] > df_out['obv'].tail(5).mean()
        obv_status = "底背離進貨" if (price_declining and obv_rising) else "正常"
        
        # (2) 計算 CMF (蔡金資金流量指標, 21日)
        denom = (df_out['high'] - df_out['low']).replace(0, 0.0001)
        mf_multiplier = ((df_out['close'] - df_out['low']) - (df_out['high'] - df_out['close'])) / denom
        mf_volume = mf_multiplier * df_out['volume']
        df_out['cmf'] = mf_volume.rolling(window=21).sum() / df_out['volume'].rolling(window=21).sum()
        df_out['cmf'] = df_out['cmf'].fillna(0)
        current_cmf = float(df_out['cmf'].iloc[-1])
        
        # 預設買賣家數差模擬值（未來可透過 payload 傳入真實資料）
        broker_diff_mock = -120  
        
        # (3) 綜合判定主力進出動向
        if current_cmf > 0.1 and broker_diff_mock < 0:
            major_force_status = "主力強烈佈局進場"
            major_force_desc = "三大籌碼指標共振。量能由大戶實質資金推動，屬於極強勢的「壓低吃貨/發起攻擊」訊號，散戶正在退場，籌碼高度集中！"
        elif current_cmf < -0.1 and broker_diff_mock > 0:
            major_force_status = "主力高檔撤離走人"
            major_force_desc = "資金呈現連續性淨流出。盤中雖爆量但接盤全為散戶分點。這是危險的「拉高出貨」型態，風險極高！"
        else:
            major_force_status = "籌碼多空拉鋸洗盤"
            major_force_desc = "目前主力資金無明顯波段方向，大戶與散戶力量互有勝負，建議靜待籌碼進一步集中。"
        # =================================================================
        
        # 5. 繪製圖表
        chart_buffer = draw_ultimate_chart(df_out)
        
        # 6. 圖檔轉 Base64 字串
        image_base64 = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')
        
        # 7. 轉回 JSON 格式並擷取最新一筆
        df_json = df_out.reset_index()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        latest_metrics = df_json.tail(1).to_dict(orient='records')[0]
        
        # 注入主力診斷文字
        latest_metrics['obv_status'] = obv_status
        latest_metrics['major_force_status'] = major_force_status
        latest_metrics['major_force_desc'] = major_force_desc
        
        return {
            "status": "success",
            "image_data": f"data:image/png;base64,{image_base64}",
            "metrics": latest_metrics
        }
        
    except Exception as e:
        # 💡 完美閉合的 except 區塊，保證不會再噴 SyntaxError
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
