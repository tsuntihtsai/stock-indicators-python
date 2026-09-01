import os
import sys
import io
import base64
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 確保引入模組路徑正確
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd
from indicators.bollinger_bands import calculate_bollinger_bands
from indicators.volume_indicators import calculate_volume_ma
from indicators.chart_builder import draw_ultimate_chart

app = FastAPI(title="Stock Indicators Ultimate API")

# 若前端會直接從瀏覽器呼叫這支 API，需要開 CORS，否則瀏覽器會擋掉請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 正式上線建議改成指定的前端網域，而不是 "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


class OHLCVFM(BaseModel):
    date: str = Field(alias="Date")
    open: float = Field(alias="Open")
    high: float = Field(alias="High")
    low: float = Field(alias="Low")
    close: float = Field(alias="Close")
    volume: float = Field(default=0.0, alias="Trading_Volume")

    # 選填：真實的「買賣家數差」或籌碼資料。
    # 沒有提供時，主力判斷只會用 CMF 資金流量指標，不會再用假資料湊數。
    broker_diff: Optional[float] = Field(default=None, alias="Broker_Diff")


class IndicatorRequestFM(BaseModel):
    data: List[OHLCVFM]


@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Stock Indicators API is running!"}


def determine_trend_bias(latest: pd.Series, weights: Optional[dict] = None) -> dict:
    """
    用「5日/20日均線交叉」、「MACD柱狀圖動能」、「KD(%K/%D)黃金/死亡交叉」
    三個因子做加權評分，判斷目前趨勢偏多/偏空/中性。
    這仍然是粗略的規則型分類，不是預測，也不是買賣訊號。

    weights: 可覆寫各因子權重，例如 {"kd_cross": 0.5} 代表想降低KD的影響力。
             未提供的因子會用預設值 1.0。
    threshold: 總分 >= threshold 判偏多，<= -threshold 判偏空，中間算中性。
               預設 1.5，代表至少要有「均線+MACD」或「均線+KD」等兩個因子同向，
               才會判定出明確方向，避免單一指標就下結論。

    欄位對應 calculate_kd_rsi_ma_macd() 輸出（轉小寫後）：
    ma_5、ma_20、macd_hist、rsi、%k、%d
    """
    default_weights = {"ma_cross": 1.0, "macd_hist": 1.0, "kd_cross": 1.0}
    w = {**default_weights, **(weights or {})}
    threshold = 1.5

    ma_short = latest.get('ma_5')
    ma_long = latest.get('ma_20')
    macd_hist = latest.get('macd_hist')
    rsi = latest.get('rsi')
    k_val = latest.get('%k')
    d_val = latest.get('%d')

    reasons = []
    score = 0.0

    if pd.notna(ma_short) and pd.notna(ma_long):
        if ma_short > ma_long:
            score += w["ma_cross"]
            reasons.append(f"5日均線在20日均線之上（+{w['ma_cross']}，短線偏多）")
        elif ma_short < ma_long:
            score -= w["ma_cross"]
            reasons.append(f"5日均線在20日均線之下（-{w['ma_cross']}，短線偏空）")

    if pd.notna(macd_hist):
        if macd_hist > 0:
            score += w["macd_hist"]
            reasons.append(f"MACD柱狀圖為正（+{w['macd_hist']}，動能偏多）")
        elif macd_hist < 0:
            score -= w["macd_hist"]
            reasons.append(f"MACD柱狀圖為負（-{w['macd_hist']}，動能偏空）")

    if pd.notna(k_val) and pd.notna(d_val):
        if k_val > d_val:
            score += w["kd_cross"]
            reasons.append(f"%K在%D之上（+{w['kd_cross']}，KD黃金交叉狀態）")
        elif k_val < d_val:
            score -= w["kd_cross"]
            reasons.append(f"%K在%D之下（-{w['kd_cross']}，KD死亡交叉狀態）")

    overbought_oversold = None
    if pd.notna(rsi):
        if rsi >= 70:
            overbought_oversold = "RSI過熱（>=70），此時追價進場風險較高"
        elif rsi <= 30:
            overbought_oversold = "RSI過冷（<=30），可能處於超賣區間"

    if score >= threshold:
        trend_bias = "偏多"
    elif score <= -threshold:
        trend_bias = "偏空"
    else:
        trend_bias = "中性/不明確"

    return {
        "trend_bias": trend_bias,
        "trend_score": round(score, 2),
        "trend_reasons": reasons,
        "overbought_oversold": overbought_oversold
    }


def calculate_trade_levels(df_out: pd.DataFrame, trend_info: Optional[dict] = None) -> dict:
    """
    根據 ATR / Donchian 通道 / 布林通道，計算一組風險報酬型的
    建議進場價、停損價、目標價。trend_info（來自 determine_trend_bias）
    只會用來調整 trade_note 的文字提醒，不會改變數字算法本身。

    這是「規則型試算」，不是預測漲跌，也不是投資建議：
    - entry_price：以現價為基準
    - stop_loss：現價 - 1.5倍ATR，和近10日低點取較高者（避免停損設太遠）
    - target_price_1：以 2倍風險報酬比（2R）反推
    - target_price_2：近期壓力位（Donchian上緣 / 布林上軌，取較保守者）

    布林通道欄位對應 calculate_bollinger_bands() 實際輸出：
    bb_mid（中軌）、bb_up（上軌）、bb_low（下軌）。
    這裡用 .get() 做保護，抓不到就退回用 target_price_1。
    """
    latest = df_out.iloc[-1]
    current_price = float(latest['close'])
    atr_val = latest.get('atr')
    atr = float(atr_val) if pd.notna(atr_val) else None

    if atr is None or atr <= 0:
        return {
            "entry_price": round(current_price, 2),
            "stop_loss": None,
            "target_price_1": None,
            "target_price_2": None,
            "risk_reward_ratio": None,
            "trade_note": "ATR 資料不足（需至少14天資料才能計算），無法給出風險報酬建議"
        }

    entry_price = current_price

    atr_stop = entry_price - 1.5 * atr
    recent_low = float(df_out['low'].tail(10).min())
    stop_loss = max(atr_stop, recent_low)

    risk = entry_price - stop_loss
    target_price_1 = entry_price + 2 * risk if risk > 0 else None

    donchian_up = latest.get('donchian_up')
    donchian_up = float(donchian_up) if pd.notna(donchian_up) else None

    # 布林上軌欄位名稱對應 calculate_bollinger_bands() 實際輸出的 'bb_up'
    bb_upper = latest.get('bb_up')
    bb_upper = float(bb_upper) if bb_upper is not None and pd.notna(bb_upper) else None

    resistance_candidates = [
        v for v in [donchian_up, bb_upper] if v is not None and v > entry_price
    ]
    target_price_2 = min(resistance_candidates) if resistance_candidates else target_price_1

    risk_reward_ratio = None
    if target_price_1 is not None and risk > 0:
        risk_reward_ratio = round((target_price_1 - entry_price) / risk, 2)

    note_parts = [
        "此為量化規則試算（進場=現價、停損=1.5倍ATR或近期低點、目標=2倍風險或近期壓力位），僅供參考，非投資建議"
    ]
    if trend_info:
        if trend_info.get("trend_bias") == "偏空":
            note_parts.append("目前趨勢判斷偏空，若考慮做多進場需格外謹慎")
        if trend_info.get("overbought_oversold"):
            note_parts.append(trend_info["overbought_oversold"])
    note_parts.append("實際下單請自行評估風險")

    return {
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target_price_1": round(target_price_1, 2) if target_price_1 is not None else None,
        "target_price_2": round(target_price_2, 2) if target_price_2 is not None else None,
        "risk_reward_ratio": risk_reward_ratio,
        "trade_note": "；".join(note_parts)
    }


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

        # 記下最新一筆使用者提供的真實 broker_diff（若有）
        latest_broker_diff = df['broker_diff'].iloc[-1] if 'broker_diff' in df.columns else None

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
        # ⚠️ 注意：上面這行把所有欄位轉小寫了，包含 open/high/low/close/volume。
        # 如果 calculate_bollinger_bands / calculate_volume_ma 內部是用大寫欄位名
        # （例如 'Close'、'Volume'）去抓資料，這裡會 KeyError，
        # 請確認這兩個函式內部抓的欄位名稱大小寫，和這裡輸出的一致。
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
        # 🚀 籌碼面指標核心計算
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

        # (3) 主力進出動向判定
        # 改動：不再用寫死的 broker_diff_mock = -120（那會讓其中一個分支永遠不可能觸發）。
        # 若使用者有透過 payload 傳入真實 broker_diff，就用真實值；
        # 沒有的話，只依 CMF 判斷，並在文字中誠實標示「籌碼資料不足」。
        has_real_broker_data = latest_broker_diff is not None and pd.notna(latest_broker_diff)
        broker_diff_value = float(latest_broker_diff) if has_real_broker_data else None

        if has_real_broker_data:
            if current_cmf > 0.1 and broker_diff_value < 0:
                major_force_status = "主力強烈佈局進場"
                major_force_desc = "資金流量與買賣家數差同步顯示大戶進場，籌碼有集中跡象，但仍需搭配成交量與後續走勢確認。"
            elif current_cmf < -0.1 and broker_diff_value > 0:
                major_force_status = "主力高檔撤離走人"
                major_force_desc = "資金呈現淨流出，且買賣家數差顯示散戶承接為主，需留意籌碼鬆動風險。"
            else:
                major_force_status = "籌碼多空拉鋸洗盤"
                major_force_desc = "目前資金流量與買賣家數差未同步指向單一方向，建議靜待籌碼進一步集中。"
        else:
            # 沒有真實籌碼資料時，只用 CMF 判斷，且明確標示資料不足
            if current_cmf > 0.1:
                major_force_status = "資金流入（僅供參考）"
                major_force_desc = "CMF資金流量指標偏多，但缺乏買賣家數差等籌碼資料佐證，判斷僅供參考。"
            elif current_cmf < -0.1:
                major_force_status = "資金流出（僅供參考）"
                major_force_desc = "CMF資金流量指標偏空，但缺乏買賣家數差等籌碼資料佐證，判斷僅供參考。"
            else:
                major_force_status = "資金流向不明"
                major_force_desc = "CMF資金流量指標無明顯方向，且缺乏買賣家數差等籌碼資料。"
        # =================================================================

        # (4) 趨勢偏多/偏空判斷（均線交叉 + MACD動能 + RSI過熱過冷）
        trend_info = determine_trend_bias(df_out.iloc[-1])

        # (5) 進場價 / 停損價 / 目標價（文字提醒會參考趨勢判斷）
        trade_levels = calculate_trade_levels(df_out, trend_info=trend_info)

        # 5. 繪製圖表
        chart_buffer = draw_ultimate_chart(df_out)

        # 6. 圖檔轉 Base64 字串
        image_base64 = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')

        # 7. 轉回 JSON 格式並擷取最新一筆
        df_json = df_out.reset_index()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        latest_metrics = df_json.tail(1).to_dict(orient='records')[0]

        # 注入主力診斷文字與交易價位建議
        latest_metrics['obv_status'] = obv_status
        latest_metrics['major_force_status'] = major_force_status
        latest_metrics['major_force_desc'] = major_force_desc
        latest_metrics['has_real_broker_data'] = has_real_broker_data
        latest_metrics.update(trend_info)
        latest_metrics.update(trade_levels)

        return {
            "status": "success",
            "image_data": f"data:image/png;base64,{image_base64}",
            "metrics": latest_metrics,
            "disclaimer": "本分析為技術指標與規則型試算結果，非投資建議，投資人應自行判斷並承擔風險。"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
