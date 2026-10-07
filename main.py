import os
import sys
import io
import base64
from typing import List, Optional
import numpy as np
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


def determine_major_force(current_cmf: float, obv_status: str, payload: "IndicatorRequestFM",
                           latest_broker_diff) -> dict:
    """
    綜合判斷主力籌碼動向。優先使用真實法人/融資融券資料，
    沒有的話才退回用價量推估的 CMF/OBV，且用詞會明確標示「僅供參考」，
    避免像之前那樣，明明沒有真實籌碼資料佐證，卻講出「主力強烈佈局」
    這種聽起來很篤定、但其實是CMF單一指標硬套出來的結論。
    """
    reasons = []
    score = 0.0

    foreign = payload.foreign_net_buy
    trust = payload.trust_net_buy
    dealer = payload.dealer_net_buy
    margin_chg = payload.margin_balance_change
    short_chg = payload.short_balance_change

    institutional_parts = [v for v in [foreign, trust, dealer] if v is not None]
    institutional_total = sum(institutional_parts) if institutional_parts else None

    has_legacy_broker_data = latest_broker_diff is not None and pd.notna(latest_broker_diff)
    has_real_chip_data = (institutional_total is not None) or (margin_chg is not None) \
        or (short_chg is not None) or has_legacy_broker_data

    if institutional_total is not None:
        if institutional_total > 0:
            score += 1.5
            reasons.append(f"三大法人合計買超約 {institutional_total:,.0f} 股")
        elif institutional_total < 0:
            score -= 1.5
            reasons.append(f"三大法人合計賣超約 {abs(institutional_total):,.0f} 股")
    elif has_legacy_broker_data:
        # 向下相容：沒有三大法人細項資料時，退回用舊的 broker_diff 判斷
        broker_diff_value = float(latest_broker_diff)
        if broker_diff_value < 0:
            score += 1
            reasons.append("買賣家數差顯示大戶券商分點買超")
        elif broker_diff_value > 0:
            score -= 1
            reasons.append("買賣家數差顯示散戶券商分點買超為主")

    if margin_chg is not None:
        if margin_chg > 0:
            reasons.append(f"融資餘額增加約 {margin_chg:,.0f} 股（散戶做多槓桿增加，籌碼較不穩定）")
        elif margin_chg < 0:
            score += 0.5
            reasons.append(f"融資餘額減少約 {abs(margin_chg):,.0f} 股（散戶去槓桿，籌碼趨於乾淨）")

    if short_chg is not None:
        if short_chg > 0:
            reasons.append(f"融券餘額增加約 {short_chg:,.0f} 股（空方力道增加，但也隱含軋空潛力）")
        elif short_chg < 0:
            reasons.append(f"融券餘額減少約 {abs(short_chg):,.0f} 股（空方回補）")

    if current_cmf > 0.1:
        score += 1
        reasons.append("CMF資金流量指標為正")
    elif current_cmf < -0.1:
        score -= 1
        reasons.append("CMF資金流量指標為負")

    if obv_status == "底背離進貨":
        score += 1
        reasons.append("OBV出現底背離進貨訊號")

    streak_days = payload.institutional_streak_days
    if streak_days is not None and streak_days != 0:
        has_real_chip_data = True  # 有連續天數資料，代表使用者確實有在追蹤真實籌碼歷史
        if streak_days > 0:
            # 連續買超天數越多，加分越多，但設上限避免無限累加蓋過其他指標
            streak_bonus = min(streak_days * 0.3, 1.5)
            score += streak_bonus
            reasons.append(f"三大法人連續買超 {streak_days} 天")
        else:
            streak_bonus = min(abs(streak_days) * 0.3, 1.5)
            score -= streak_bonus
            reasons.append(f"三大法人連續賣超 {abs(streak_days)} 天")

    # 近5日/10日/20日三大法人合計買賣超：業界標準區間，同時比對短中期趨勢是否一致
    window_defs = [("5日", payload.institutional_net_5d), ("10日", payload.institutional_net_10d),
                   ("20日", payload.institutional_net_20d)]
    window_signs = []
    for label, value in window_defs:
        if value is None:
            continue
        has_real_chip_data = True
        if value > 0:
            reasons.append(f"近{label}三大法人合計買超約 {value:,.0f} 股")
            window_signs.append(1)
        elif value < 0:
            reasons.append(f"近{label}三大法人合計賣超約 {abs(value):,.0f} 股")
            window_signs.append(-1)
        else:
            window_signs.append(0)

    has_mixed_window_signal = False
    if window_signs:
        if all(s > 0 for s in window_signs):
            score += 1.5
            reasons.append("短中期籌碼方向一致偏多")
        elif all(s < 0 for s in window_signs):
            score -= 1.5
            reasons.append("短中期籌碼方向一致偏空")
        else:
            # 各區間方向不一致（例如5日轉負但20日仍為正），代表多空拉鋸，不宜下定論。
            # 這裡不只是不加分不扣分，還要強制最終結論走向「拉鋸」，
            # 避免其他因子(今日買賣超、連續天數)的分數蓋過這個明確的矛盾訊號，
            # 造成文字說「拉鋸」但結論卻寫「偏多/偏空」的自相矛盾。
            has_mixed_window_signal = True
            reasons.append("短中期籌碼方向不一致，判斷為多空拉鋸")

    if has_real_chip_data:
        if has_mixed_window_signal:
            # 短中期方向明確衝突時，不管其他因子分數多高，都判定為拉鋸，
            # 避免文字說「方向不一致」但結論卻寫「偏多/偏空」的自相矛盾
            status = "籌碼多空拉鋸洗盤"
        elif score >= 1.5:
            status = "主力偏多佈局"
        elif score <= -1.5:
            status = "主力偏空撤離"
        else:
            status = "籌碼多空拉鋸洗盤"
    else:
        # 沒有任何真實籌碼資料，只能用價量推估的CMF/OBV，語氣必須保守、明確標示參考性質
        if score >= 1:
            status = "資金流入（僅供參考，缺乏法人籌碼資料佐證）"
        elif score <= -1:
            status = "資金流出（僅供參考，缺乏法人籌碼資料佐證）"
        else:
            status = "資金流向不明"

    desc = "；".join(reasons) if reasons else "目前資料不足以判斷主力動向"
    if not has_real_chip_data:
        desc += "。本次判斷僅根據價量推估的CMF/OBV，並無三大法人買賣超、融資融券等真實籌碼資料佐證，僅供參考，不代表主力真實動向。"

    # 就算有真實籌碼資料，如果歷史天數還很少（例如剛開始存資料），
    # 單日或短短幾天的數字容易被一次性大單、ETF調整成分股等雜訊干擾，信心度較低。
    # 這裡明確標示出來，避免報告用過於篤定的語氣呈現一個其實還不穩定的判斷。
    LOW_CONFIDENCE_DAYS_THRESHOLD = 5
    history_days = payload.institutional_history_days
    is_low_confidence = has_real_chip_data and history_days is not None and history_days < LOW_CONFIDENCE_DAYS_THRESHOLD
    if is_low_confidence:
        desc += f"（注意：目前僅累積 {history_days} 天籌碼歷史，資料仍在累積中，單日或短期數字容易受一次性大單干擾，信心度較低，建議累積至少{LOW_CONFIDENCE_DAYS_THRESHOLD}個交易日以上再視為穩定趨勢判斷）"

    return {
        "major_force_status": status,
        "major_force_desc": desc,
        "major_force_score": round(score, 2),
        "has_real_chip_data": has_real_chip_data,
        "is_low_confidence_chip_data": is_low_confidence,
    }


def determine_signal_divergence(trend_bias: str, major_force_status: str) -> dict:
    """
    trend_bias（技術面）與 major_force_status（籌碼面）是完全獨立計算的兩套分數，
    suggest_entry_strategy() 只依據 trend_bias 決定策略，並未納入籌碼面方向。
    當兩者方向相反時，策略建議只代表技術面單一視角，這裡明確標註出來，
    避免使用者誤以為策略建議已經綜合考慮籌碼面。
    """
    bullish_trend = trend_bias == "偏多"
    bearish_trend = trend_bias == "偏空"
    bullish_chip = ("偏多" in major_force_status) or ("佈局" in major_force_status)
    bearish_chip = ("偏空" in major_force_status) or ("撤離" in major_force_status)

    if bullish_trend and bearish_chip:
        return {
            "signal_alignment": "背離（技術偏多／籌碼偏空）",
            "signal_alignment_note": (
                "⚠️ 訊號分歧提醒：技術面指標顯示偏多，但籌碼面（三大法人）同步顯示主力偏空撤離。"
                "上方的策略建議僅依據技術面計算，並未納入籌碼面方向。"
                "籌碼面轉向常領先技術面反應，此類分歧格局追價風險較高，"
                "建議降低部位規模、縮小停損距離，或等待籌碼面轉為同向後再考慮進場。"
            )
        }
    elif bearish_trend and bullish_chip:
        return {
            "signal_alignment": "背離（技術偏空／籌碼偏多）",
            "signal_alignment_note": (
                "⚠️ 訊號分歧提醒：技術面指標顯示偏空，但籌碼面（三大法人）同步顯示主力偏多佈局。"
                "可能代表法人正在逢低承接、但短線技術指標尚未反映，"
                "上方策略建議僅依據技術面計算，不宜單純依此判斷做空。"
            )
        }
    else:
        return {"signal_alignment": "一致或中性", "signal_alignment_note": None}
        

class IndicatorRequestFM(BaseModel):
    data: List[OHLCVFM]
    stock_symbol: Optional[str] = Field(default=None, description="股票代號，例如 2330")
    stock_name: Optional[str] = Field(default=None, description="股票名稱，例如 台積電")

    # 真實籌碼面資料（選填）。可從證交所/櫃買中心公開資料免費取得：
    # - 三大法人買賣超：https://www.twse.com.tw/zh/trading/foreign/bfi82u.html (TWSE OpenAPI也有對應端點)
    # - 融資融券餘額：https://www.twse.com.tw/zh/trading/margin/mi-margin.html
    # 單位皆為「股數」，正值代表買超/增加，負值代表賣超/減少。
    foreign_net_buy: Optional[float] = Field(default=None, description="外資當日買賣超股數")
    trust_net_buy: Optional[float] = Field(default=None, description="投信當日買賣超股數")
    dealer_net_buy: Optional[float] = Field(default=None, description="自營商當日買賣超股數")
    margin_balance_change: Optional[float] = Field(default=None, description="融資餘額當日增減股數")
    short_balance_change: Optional[float] = Field(default=None, description="融券餘額當日增減股數")

    # 連續買賣超天數（選填）。正值代表連續買超天數，負值代表連續賣超天數，0代表持平或無資料。
    # 由於三大法人買賣超的官方API通常只提供「最新一天」的快照、沒有回溯查詢功能，
    # 這個天數需要使用者自行每天存檔累積歷史後計算出來，再傳進來。
    institutional_streak_days: Optional[int] = Field(default=None, description="三大法人合計連續買超(正)/賣超(負)天數")

    # 目前累積了幾天的籌碼歷史資料（選填）。
    # 單日籌碼數據容易受一次性大單、ETF調整成分股等雜訊干擾，信心度較低；
    # 這個欄位讓 main.py 可以判斷歷史資料夠不夠多，資料太少時會在報告裡明確標示「僅供初步參考」。
    institutional_history_days: Optional[int] = Field(default=None, description="目前累積的籌碼歷史天數")

    # 近5日/10日/20日三大法人合計買賣超（選填，單位：股）。
    # 這是台股籌碼分析業界慣用的標準區間（跟Yahoo股市、玩股網等平台的「法人進出」頁面一致），
    # 用來同時比對短期(5日)、中短期(10日)、中期(20日)的買賣超方向是否一致：
    # 三個區間同方向 → 趨勢較明確；方向不一致（例如5日轉負但20日仍為正）→ 判斷為多空拉鋸，
    # 而不是只看單日或隨便一個區間就下定論。
    institutional_net_5d: Optional[float] = Field(default=None, description="近5個交易日三大法人合計買賣超股數")
    institutional_net_10d: Optional[float] = Field(default=None, description="近10個交易日三大法人合計買賣超股數")
    institutional_net_20d: Optional[float] = Field(default=None, description="近20個交易日三大法人合計買賣超股數")


@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Stock Indicators API is running!",
        # 版本標記：每次重大修改main.py後更新這個字串，
        # 部署完成後直接瀏覽器打開這支API的根目錄網址（例如
        # https://tsuntih-stock.zeabur.app/），
        # 看這裡的版本字串有沒有變成最新的，比每次都跑完整/analyze測試快很多，
        # 也能立刻判斷「到底是main.py沒改對，還是部署沒生效」。
        "version": "2026-09-22-divergence-capitulation-rolloff-only"
    }


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


def detect_candlestick_patterns(df_out: pd.DataFrame) -> dict:
    """
    用規則型邏輯辨識最新一根K棒（必要時搭配前一根）的常見型態，
    避免完全交給AI「用眼睛看圖」猜測型態名稱（容易誤判或講不出具體根據）。
    這是簡化版規則，判斷依據是實體與影線長度的相對比例，不是嚴謹的量化回測工具。
    """
    latest = df_out.iloc[-1]
    o, h, l, c = float(latest['open']), float(latest['high']), float(latest['low']), float(latest['close'])
    body = abs(c - o)
    candle_range = (h - l) if (h - l) > 0 else 0.0001
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    body_ratio = body / candle_range

    patterns = []

    if body_ratio < 0.1:
        patterns.append("十字線（Doji）：實體極小，多空拉鋸激烈，方向尚未明朗")

    if upper_shadow > body * 2 and upper_shadow > lower_shadow:
        if c > o:
            patterns.append("長上影線：盤中一度走高但被賣壓打回，上檔壓力沉重")
        else:
            patterns.append("流星型態：高檔賣壓明顯，若出現在相對高點需留意反轉")

    if lower_shadow > body * 2 and lower_shadow > upper_shadow:
        if c > o:
            patterns.append("鎚子線：盤中一度殺低但拉回收復，若出現在相對低點具止跌意義")
        else:
            patterns.append("長下影線：低點出現買盤承接，留意是否止穩")

    if body_ratio > 0.85:
        if c > o:
            patterns.append("長紅棒（近似光頭光腳）：買盤全場強勢主導")
        else:
            patterns.append("長黑棒（近似光頭光腳）：賣盤全場強勢主導")

    if len(df_out) >= 2:
        prev = df_out.iloc[-2]
        po, pc = float(prev['open']), float(prev['close'])
        if c > o and po > pc and c > po and o < pc:
            patterns.append("看漲吞噬：今日紅K完全吞沒昨日黑K實體，具短線反轉訊號")
        elif c < o and pc > po and o > pc and c < po:
            patterns.append("看跌吞噬：今日黑K完全吞沒昨日紅K實體，具短線反轉訊號")

    if not patterns:
        patterns.append("無明顯特殊型態，屬一般漲跌K棒，型態上無額外訊號")

    return {"candlestick_patterns": patterns}


def analyze_volume_price_relation(df_out: pd.DataFrame) -> dict:
    """
    判斷最新一天的「量價關係」：價漲量增/價漲量縮/價跌量增/價跌量縮，
    這是技術分析裡驗證「這根K棒的漲跌有沒有量能支撐」的標準做法，
    比單純講「量能穩定」這種模糊詞更有判斷依據。
    """
    if len(df_out) < 2:
        return {"volume_price_label": "資料不足", "volume_price_desc": "資料筆數不足，無法比較前一日", "volume_ratio_vs_ma5": None}

    latest = df_out.iloc[-1]
    prev_close = float(df_out['close'].iloc[-2])
    price_change = float(latest['close']) - prev_close

    volume = latest.get('volume')
    v_ma5 = latest.get('v_ma5')
    volume_ratio = None
    if volume is not None and v_ma5 is not None and pd.notna(v_ma5) and v_ma5 > 0:
        volume_ratio = float(volume) / float(v_ma5)

    if volume_ratio is None:
        return {"volume_price_label": "資料不足", "volume_price_desc": "缺乏5日均量資料，無法判斷量能是否放大", "volume_ratio_vs_ma5": None}

    price_up = price_change > 0
    volume_expanding = volume_ratio > 1.1
    volume_shrinking = volume_ratio < 0.9

    if price_up and volume_expanding:
        label, desc = "價漲量增", "價量同步走揚，換手積極，短線動能有量能支撐"
    elif price_up and volume_shrinking:
        label, desc = "價漲量縮", "價格上漲但量能未同步放大，追價意願不足，若是突破訊號則真實性需保留觀察"
    elif (not price_up) and volume_expanding:
        label, desc = "價跌量增", "下跌伴隨放量，賣壓沉重，需留意是否有進一步破底風險"
    elif (not price_up) and volume_shrinking:
        label, desc = "價跌量縮", "下跌但量能萎縮，賣壓趨緩，可能進入惜售整理格局"
    else:
        label, desc = "價量普通", "價格與量能變化都不明顯，暫無特殊量價訊號"

    return {"volume_price_label": label, "volume_price_desc": desc, "volume_ratio_vs_ma5": round(volume_ratio, 2)}


def determine_band_squeeze(df_out: pd.DataFrame) -> dict:
    """
    判斷目前布林通道寬度(bb_width)是否處於「收縮(squeeze)」狀態——
    也就是近期波動率相對過去而言明顯偏低。這種情況常常是變盤（大漲或大跌）前兆，
    但收縮本身「不代表方向」，只代表接下來波動可能放大，須留意帶量突破的方向。
    """
    if 'bb_width' not in df_out.columns:
        return {"bb_width": None, "bb_squeeze": None, "bb_squeeze_note": None}

    current_width = df_out['bb_width'].iloc[-1]
    if pd.isna(current_width):
        return {"bb_width": None, "bb_squeeze": None, "bb_squeeze_note": "資料不足，無法判斷布林通道寬度"}

    lookback = min(120, len(df_out))
    history = df_out['bb_width'].tail(lookback).dropna()

    if len(history) < 20:
        return {
            "bb_width": round(float(current_width), 2),
            "bb_squeeze": None,
            "bb_squeeze_note": "歷史資料不足20筆，無法判斷目前是否處於收縮狀態"
        }

    percentile_20 = float(np.percentile(history, 20))
    is_squeeze = bool(current_width <= percentile_20)

    if is_squeeze:
        note = (
            f"目前布林通道寬度({round(float(current_width), 2)}%)處於近{lookback}日相對低點"
            f"（低於20百分位{round(percentile_20, 2)}%），波動率明顯收縮，是變盤前兆，"
            "但收縮本身不代表方向，須留意帶量突破的實際方向再判斷"
        )
    else:
        note = f"目前布林通道寬度({round(float(current_width), 2)}%)未處於收縮狀態，波動率屬正常範圍"

    return {"bb_width": round(float(current_width), 2), "bb_squeeze": is_squeeze, "bb_squeeze_note": note}


def detect_divergence(df_out: pd.DataFrame, lookback: int = 40, split: int = 15) -> dict:
    """
    簡化版背離偵測：把最近 lookback 天分成「近期」跟「先前」兩段，
    分別找出這兩段各自的最低點(判斷底背離)/最高點(判斷頂背離)，
    比較價格與指標(RSI/MACD柱狀值)在這兩個低點/高點上是否方向不一致(背離)。

    底背離：股價創新低，但指標未同步破底 → 市場常視為較扎實的反轉預警訊號
    頂背離：股價創新高，但指標未同步創高 → 動能可能已經開始減弱

    這是簡化版判斷（用區間內最低/最高點簡化代替嚴謹的波段高低點演算法），
    能抓出多數常見的背離情境，但不是100%嚴謹的技術分析工具，僅供參考。
    """
    if len(df_out) < lookback:
        return {"divergence_signal": None, "divergence_note": "資料不足，無法判斷是否存在背離"}

    recent = df_out.tail(split)
    prior = df_out.iloc[-lookback:-split]

    if prior.empty or recent.empty:
        return {"divergence_signal": None, "divergence_note": "資料不足，無法判斷是否存在背離"}

    # --- 底背離 (Bullish Divergence)：價格創新低，但指標未創新低 ---
    recent_low_idx = recent['close'].idxmin()
    prior_low_idx = prior['close'].idxmin()
    recent_low_price = recent.loc[recent_low_idx, 'close']
    prior_low_price = prior.loc[prior_low_idx, 'close']
    recent_low_rsi = recent.loc[recent_low_idx, 'rsi'] if 'rsi' in recent.columns else None
    prior_low_rsi = prior.loc[prior_low_idx, 'rsi'] if 'rsi' in prior.columns else None
    recent_low_macd = recent.loc[recent_low_idx, 'macd_hist'] if 'macd_hist' in recent.columns else None
    prior_low_macd = prior.loc[prior_low_idx, 'macd_hist'] if 'macd_hist' in prior.columns else None

    bullish_rsi_div = (pd.notna(recent_low_rsi) and pd.notna(prior_low_rsi)
                       and recent_low_price < prior_low_price and recent_low_rsi > prior_low_rsi)
    bullish_macd_div = (pd.notna(recent_low_macd) and pd.notna(prior_low_macd)
                        and recent_low_price < prior_low_price and recent_low_macd > prior_low_macd)

    # --- 頂背離 (Bearish Divergence)：價格創新高，但指標未創新高 ---
    recent_high_idx = recent['close'].idxmax()
    prior_high_idx = prior['close'].idxmax()
    recent_high_price = recent.loc[recent_high_idx, 'close']
    prior_high_price = prior.loc[prior_high_idx, 'close']
    recent_high_rsi = recent.loc[recent_high_idx, 'rsi'] if 'rsi' in recent.columns else None
    prior_high_rsi = prior.loc[prior_high_idx, 'rsi'] if 'rsi' in prior.columns else None
    recent_high_macd = recent.loc[recent_high_idx, 'macd_hist'] if 'macd_hist' in recent.columns else None
    prior_high_macd = prior.loc[prior_high_idx, 'macd_hist'] if 'macd_hist' in prior.columns else None

    bearish_rsi_div = (pd.notna(recent_high_rsi) and pd.notna(prior_high_rsi)
                        and recent_high_price > prior_high_price and recent_high_rsi < prior_high_rsi)
    bearish_macd_div = (pd.notna(recent_high_macd) and pd.notna(prior_high_macd)
                         and recent_high_price > prior_high_price and recent_high_macd < prior_high_macd)

    signals = []
    if bullish_rsi_div:
        signals.append("RSI底背離")
    if bullish_macd_div:
        signals.append("MACD底背離")
    if bearish_rsi_div:
        signals.append("RSI頂背離")
    if bearish_macd_div:
        signals.append("MACD頂背離")

    if not signals:
        return {"divergence_signal": "無明顯背離",
                "divergence_note": "近期價格與指標走勢方向一致，未偵測到底背離或頂背離"}

    is_bullish = any("底背離" in s for s in signals)
    is_bearish = any("頂背離" in s for s in signals)

    if is_bullish and not is_bearish:
        note = (f"偵測到{'/'.join(signals)}：股價創近期新低，但動能指標並未同步破底，"
                "這是相對扎實的反轉預警訊號，但仍須配合成交量與後續K線確認，不代表立即反轉")
    elif is_bearish and not is_bullish:
        note = (f"偵測到{'/'.join(signals)}：股價創近期新高，但動能指標並未同步創高，"
                "動能可能已經開始減弱，需留意漲勢是否後繼無力")
    else:
        note = f"同時偵測到底背離與頂背離訊號（{'/'.join(signals)}），訊號較為混雜，建議謹慎判讀"

    return {"divergence_signal": "；".join(signals), "divergence_note": note}


def detect_capitulation_signal(df_out: pd.DataFrame, candlestick_patterns: list) -> dict:
    """
    窒息量（量縮到極致）或爆量長下影線，都是市場實務上常見的落底訊號：
    - 窒息量：成交量萎縮到近期相對低點，代表想賣的人跟想買的人都退場觀望，
      是一種量能死寂的狀態，短線止跌機率較高
    - 爆量長下影線：當天成交量異常放大，且K線出現長下影線（盤中重挫後又拉回），
      代表買盤積極承接，是較強的單日反轉訊號
    """
    if 'volume' not in df_out.columns:
        return {"capitulation_signal": None, "capitulation_note": "缺乏成交量資料，無法判斷"}

    volume = df_out['volume']
    latest_volume = volume.iloc[-1]
    lookback = min(60, len(df_out))
    history = volume.tail(lookback)

    if len(history.dropna()) < 20:
        return {"capitulation_signal": None, "capitulation_note": "資料不足，無法判斷窒息量或爆量狀態"}

    # 目前量能落在近期歷史的百分位（越低代表量越萎縮，越高代表量越爆量）
    volume_percentile = float((history <= latest_volume).mean() * 100)
    has_long_lower_shadow = any(("鎚子線" in p or "長下影線" in p) for p in candlestick_patterns)

    signals = []
    if volume_percentile <= 15:
        signals.append("窒息量")
    if has_long_lower_shadow and volume_percentile >= 80:
        signals.append("爆量長下影線")

    if not signals:
        return {
            "capitulation_signal": "無明顯訊號",
            "capitulation_note": f"目前量能約在近{lookback}日的{round(volume_percentile, 1)}百分位，未達窒息量或爆量長下影線的判斷門檻"
        }

    notes = []
    if "窒息量" in signals:
        notes.append(f"目前成交量處於近{lookback}日相對極低水準（約{round(volume_percentile, 1)}百分位），賣壓與買氣同步低迷，短線落底機率提高")
    if "爆量長下影線" in signals:
        notes.append(f"當天成交量異常放大（約{round(volume_percentile, 1)}百分位）且出現長下影線，顯示盤中重挫後有買盤積極承接，屬於較強的單日反轉訊號")

    return {"capitulation_signal": "；".join(signals), "capitulation_note": "；".join(notes)}


def analyze_volume_ma_rolloff(df_out: pd.DataFrame) -> dict:
    """
    量能均線「扣抵」判斷：均線是移動平均，隨著時間推進，最舊的一筆資料會被「扣掉」，
    若扣掉的是相對高的量，代表就算之後量能持平，均線也會自然往下彎；
    扣掉的是相對低的量，代表均線容易被墊高、往上彎。
    這能提前判斷量能結構接下來要轉強還是轉弱，不用等到真的發生才知道。
    """
    if 'volume' not in df_out.columns:
        return {}

    volume = df_out['volume']
    result = {}
    for period, label in [(5, 'v_ma5'), (20, 'v_ma20')]:
        if len(volume) < period + 1:
            result[f'{label}_rolloff_note'] = f"資料不足，無法判斷{label}扣抵"
            continue
        rolloff_value = float(volume.iloc[-(period + 1)])  # 下一筆將被扣掉的量(即將滾出視窗的那一筆)
        recent_avg = float(volume.tail(min(5, len(volume))).mean())  # 近期量能水準參考

        if recent_avg <= 0:
            result[f'{label}_rolloff_note'] = f"近期量能資料異常，無法判斷{label}扣抵"
            continue

        if rolloff_value > recent_avg * 1.15:
            note = f"{label}下一筆將扣抵較高的量能（{rolloff_value:,.0f}），若接下來量能持平，{label}將自然向下彎，量能結構可能轉弱"
        elif rolloff_value < recent_avg * 0.85:
            note = f"{label}下一筆將扣抵較低的量能（{rolloff_value:,.0f}），若接下來量能持平，{label}將自然向上彎，量能結構可能轉強"
        else:
            note = f"{label}即將扣抵的量能（{rolloff_value:,.0f}）與近期水準相近，短期均線走向主要仍取決於接下來的實際量能變化"
        result[f'{label}_rolloff_note'] = note
    return result

def detect_bullish_reversal(df_out: pd.DataFrame,
                            lookback: int = 60,
                            min_base_bars: int = 5,
                            min_drop_pct: float = 10.0,
                            max_base_rebound_pct: float = 25.0,
                            min_gain_pct: float = 2.0,
                            min_body_ratio: float = 0.6,
                            surge_volume_ratio: float = 1.5,
                            base_volume_ratio: float = 1.5) -> dict:
    """
    底部反轉多頭確認訊號（規則型簡化版，以最新一根日K為「確認日」）。

    必要條件：
    1. 空頭下跌後在低檔打底（前高到低點跌幅>=min_drop_pct，低點後盤整>=min_base_bars根）
    2. 確認日收盤突破打底區間的轉折高點（頸線）
    3. 確認日為實體長紅，漲幅>=min_gain_pct，且成交量>=前5日均量的surge_volume_ratio倍
    4. 均線條件：完整版（收盤在5/10/20日線之上、多頭排列、三線向上）
       或較弱版（收盤在20日線之上且20日線上揚）

    加分項（不影響是否成立，只提高可信度）：底部放量、KD向上、MACD動能改善。

    注意：「打底」「轉折高點」本來就帶有主觀成分，這裡用固定規則簡化，
    實際抓到的型態可能跟人眼判讀有出入，建議用歷史案例回測後再調參數。
    """
    no_data = {
        "bullish_reversal_signal": None,
        "bullish_reversal_note": "資料不足，無法判斷底部反轉訊號",
        "bullish_reversal_bonus": [],
        "bullish_reversal_unmet": [],
        "bullish_reversal_details": {},
    }
    if len(df_out) < 30:
        return no_data

    df = df_out.copy()
    for p in (5, 10, 20):
        if f'ma_{p}' not in df.columns:
            df[f'ma_{p}'] = df['close'].rolling(p).mean()

    w = df.tail(min(lookback, len(df)))
    last = len(w) - 1
    if last < min_base_bars + 3:
        return no_data

    close = w['close'].to_numpy(dtype=float)
    open_ = w['open'].to_numpy(dtype=float)
    high = w['high'].to_numpy(dtype=float)
    low = w['low'].to_numpy(dtype=float)
    vol = w['volume'].to_numpy(dtype=float)
    if np.isnan(close).any() or np.isnan(vol).any():
        return no_data

    def _valid(*vals):
        return all(v is not None and not np.isnan(v) for v in vals)

    # ---------- 1. 打底判斷 ----------
    low_pos = int(np.argmin(close[:last]))           # 最低收盤（不含今天）
    low_close = float(close[low_pos])
    base_bars = last - low_pos - 1                   # 低點後到昨天的盤整根數

    prior_high = float(np.max(close[:low_pos])) if low_pos > 0 else None
    drop_pct = (prior_high - low_close) / prior_high * 100 if prior_high else None
    drop_ok = bool(drop_pct is not None and drop_pct >= min_drop_pct)
    base_bars_ok = bool(base_bars >= min_base_bars)

    neckline = float(np.max(close[low_pos + 1:last])) if base_bars >= 1 else None
    rebound_pct = (neckline - low_close) / low_close * 100 if neckline else None
    rebound_ok = bool(rebound_pct is not None and rebound_pct <= max_base_rebound_pct)

    # ---------- 2. 突破頸線 ----------
    c = float(close[last])
    o = float(open_[last])
    h = float(high[last])
    l = float(low[last])
    prev_c = float(close[last - 1])
    breakout_ok = bool(neckline is not None and c > neckline)

    # ---------- 3. 確認日K棒 ----------
    body_ratio = (c - o) / max(h - l, 1e-4)
    gain_pct = (c - prev_c) / prev_c * 100
    red_ok = bool(c > o and body_ratio >= min_body_ratio)
    gain_ok = bool(gain_pct >= min_gain_pct)

    avg_prior5 = float(np.mean(vol[last - 5:last]))
    vol_ratio = float(vol[last]) / avg_prior5 if avg_prior5 > 0 else None
    volume_ok = bool(vol_ratio is not None and vol_ratio >= surge_volume_ratio)

    # ---------- 4. 均線條件 ----------
    ma5 = w['ma_5'].to_numpy(dtype=float)
    ma10 = w['ma_10'].to_numpy(dtype=float)
    ma20 = w['ma_20'].to_numpy(dtype=float)
    strong_ma = weak_ma = False
    if _valid(ma5[last], ma5[last - 1], ma10[last], ma10[last - 1], ma20[last], ma20[last - 1]):
        above_all = c > ma5[last] and c > ma10[last] and c > ma20[last]
        aligned = ma5[last] > ma10[last] > ma20[last]
        rising = (ma5[last] > ma5[last - 1] and ma10[last] > ma10[last - 1]
                  and ma20[last] > ma20[last - 1])
        strong_ma = bool(above_all and aligned and rising)
        weak_ma = bool(c > ma20[last] and ma20[last] > ma20[last - 1])

    # ---------- 加分項 ----------
    bonus = []
    base_zone = vol[max(0, low_pos - 2):last]
    window_avg_vol = float(np.mean(vol[:last]))
    base_vol_ratio = float(np.max(base_zone)) / window_avg_vol if window_avg_vol > 0 else None
    if base_vol_ratio is not None and base_vol_ratio >= base_volume_ratio:
        bonus.append(f"底部明顯放量（區間最大量為均量{base_vol_ratio:.1f}倍）")

    def _last_two(col):
        if col not in w.columns:
            return None, None
        s = w[col].to_numpy(dtype=float)
        return float(s[last]), float(s[last - 1])

    k, k_prev = _last_two('%k')
    d, d_prev = _last_two('%d')
    if _valid(k, k_prev, d, d_prev) and k > k_prev:
        if k > d and d >= d_prev:
            bonus.append("KD多頭排列向上")
        else:
            bonus.append("KD的K值向上走高")

    hist, hist_prev = _last_two('macd_hist')
    if _valid(hist, hist_prev) and hist > hist_prev:
        if hist > 0 and hist_prev > 0:
            bonus.append("MACD紅柱延長")
        elif hist > 0:
            bonus.append("MACD綠翻紅")
        else:
            bonus.append("MACD綠柱縮短")

    # ---------- 綜合判定 ----------
    core_checks = [
        (f"空頭下跌段不足（前高到低點跌幅需>={min_drop_pct}%）", drop_ok),
        (f"低點後盤整不足{min_base_bars}根K棒", base_bars_ok),
        (f"打底期間反彈幅度過大（超過{max_base_rebound_pct}%，較像已經上漲而非打底）", rebound_ok),
        ("收盤尚未突破打底區間轉折高點", breakout_ok),
        (f"非實體長紅（需紅K且實體佔比>={int(min_body_ratio * 100)}%）", red_ok),
        (f"漲幅未達{min_gain_pct}%", gain_ok),
        (f"成交量未達前5日均量{surge_volume_ratio}倍", volume_ok),
    ]
    unmet = [label for label, ok in core_checks if not ok]
    core_ok = len(unmet) == 0

    signal = None
    if core_ok and strong_ma:
        signal = "底部反轉多頭確認（完整）"
    elif core_ok and weak_ma:
        signal = "底部反轉多頭確認（均線條件較弱）"
    elif core_ok:
        unmet.append("均線條件未達（收盤需在20日線之上且20日線上揚）")

    details = {
        "bottom_close": round(low_close, 2),
        "base_bars": int(base_bars),
        "neckline": round(neckline, 2) if neckline is not None else None,
        "gain_pct": round(gain_pct, 2),
        "body_ratio": round(body_ratio, 2),
        "volume_ratio_vs_prior5": round(vol_ratio, 2) if vol_ratio is not None else None,
    }

    if signal:
        ma_desc = ("收盤站上5/10/20日線，三線多頭排列且向上" if strong_ma
                   else "收盤站上20日線且20日線上揚，但5/10/20日線尚未完整多頭排列")
        note = (f"{signal}：低點{details['bottom_close']}後盤整{base_bars}根K棒，"
                f"今日收盤{c}突破轉折高點{details['neckline']}，漲幅{gain_pct:.2f}%，"
                f"實體佔比{body_ratio:.0%}，成交量為前5日均量{vol_ratio:.2f}倍。{ma_desc}。"
                + (f"加分項：{'、'.join(bonus)}。" if bonus else "加分項：無。")
                + "此為規則型技術訊號，仍需自行搭配停損與籌碼面判斷。")
    else:
        note = "目前未同時符合底部反轉多頭確認條件"

    return {
        "bullish_reversal_signal": signal,
        "bullish_reversal_note": note,
        "bullish_reversal_bonus": bonus,
        "bullish_reversal_unmet": unmet,
        "bullish_reversal_details": details,
    }



def determine_breakout_risk_warning(latest: pd.Series) -> Optional[str]:
    """
    當RSI過熱、KD死亡交叉同時出現時，代表指標已經偏向超買、動能出現轉弱跡象。
    這種情況下如果策略是「站上前高/壓力位再進場」的突破追價邏輯，
    假突破（跌破前高後又拉回、俗稱被巴）的失敗率會比正常情況更高，
    這個提醒不能省略，否則報告會給人「指標超買中還敢建議追突破」的錯誤印象。
    """
    rsi = latest.get('rsi')
    k_val = latest.get('%k')
    d_val = latest.get('%d')

    triggers = []
    if pd.notna(rsi) and rsi >= 70:
        triggers.append("RSI已達過熱區(>=70)")
    if pd.notna(k_val) and pd.notna(d_val) and k_val < d_val:
        triggers.append("KD呈死亡交叉")

    if len(triggers) >= 1:
        return ("、".join(triggers) + "。此時若採取「突破前高/壓力位再進場」的策略，"
                "追高的假突破風險高於平常，建議等待量能同步放大確認、或指標回檔整理後再評估，"
                "不宜見高點被觸及就直接視為進場訊號。")
    return None


def suggest_entry_strategy(latest: pd.Series, trend_info: dict, breakout_warning: Optional[str],
                            stop_loss: Optional[float] = None, atr: Optional[float] = None) -> dict:
    """
    給出一個「真正有分析含量」的建議進場價/策略，取代單純把現價當成進場價的舊做法。

    核心邏輯：不是所有情況都建議「現在進場」，而是依照目前位置分成三種策略：
    1. 突破進場：趨勢偏多、但股價還沒站上關鍵壓力位 → 建議「站上OO再進場」，現在不追價
    2. 拉回進場：指標過熱/死叉，或趨勢中性不明確 → 建議「拉回到OO支撐不破再進場」，不追高
    3. 現價可進場：趨勢偏多，且已經站穩主要壓力之上、也沒有過熱警訊 → 現價才真正具備進場條件

    stop_loss / atr 用來檢查算出來的支撐候選價位是否跟停損價過於接近。
    如果兩者距離小於 0.5倍ATR（甚至相等），代表這個支撐候選本身就是最近的支撐關卡，
    等到那個價位進場時幾乎沒有回檔容錯空間，會在建議文字裡明確警示。
    """
    close = float(latest['close'])
    bb_mid = latest.get('bb_mid')
    bb_up = latest.get('bb_up')
    bb_low = latest.get('bb_low')
    donchian_up = latest.get('donchian_up')
    donchian_low = latest.get('donchian_low')
    ma5 = latest.get('ma_5')
    ma20 = latest.get('ma_20')
    trend_bias = trend_info.get('trend_bias')

    def _clean(v):
        return float(v) if v is not None and pd.notna(v) else None

    bb_mid, bb_up, bb_low, donchian_up, donchian_low, ma5, ma20 = map(
        _clean, [bb_mid, bb_up, bb_low, donchian_up, donchian_low, ma5, ma20]
    )

    min_buffer = 0.5 * atr if (atr is not None and atr > 0) else 0.0

    def _too_close_to_stop(price):
        if price is None or stop_loss is None:
            return False
        return abs(price - stop_loss) < min_buffer

    def _build_pullback_result(pullback_ref, entry_type_label, base_note):
        note = base_note
        if _too_close_to_stop(pullback_ref):
            note += (
                f"（⚠️ 注意：此進場參考價 {round(pullback_ref, 2)} 已非常接近建議停損價 "
                f"{round(stop_loss, 2)}，兩者相距不到0.5倍ATR，實際進場後容錯空間極小，"
                "建議搭配更寬鬆的停損設定，或改以更保守的支撐位分批進場，避免一有正常波動就被洗出場)"
            )
        return {
            "suggested_entry_type": entry_type_label,
            "suggested_entry_price": round(pullback_ref, 2) if pullback_ref is not None else None,
            "suggested_entry_note": note
        }

    if breakout_warning:
        support_candidates = [v for v in [ma5, ma20, bb_mid, bb_low, donchian_low] if v is not None and v < close]
        pullback_ref = max(support_candidates) if support_candidates else None
        base_note = (
            f"指標已偏向過熱/死叉，不建議現在追高，建議等股價拉回至約 "
            f"{round(pullback_ref, 2) if pullback_ref is not None else '支撐區'} 附近且不破，再考慮進場"
        )
        return _build_pullback_result(pullback_ref, "拉回進場", base_note)

    if trend_bias == "偏多":
        resistance_candidates = [v for v in [bb_up, donchian_up] if v is not None and v > close]
        if resistance_candidates:
            trigger = min(resistance_candidates)
            note = f"若股價帶量站穩 {round(trigger, 2)} 之上，可視為偏多訊號進場；目前尚未站上此關卡，不建議現在追價"
            if _too_close_to_stop(trigger):
                note += (
                    f"（⚠️ 注意：此突破觸發價 {round(trigger, 2)} 與建議停損價 {round(stop_loss, 2)} 相距過近，"
                    "實際站上後的合理停損可能需要另外評估，不宜直接沿用下方停損數字)"
                )
            return {
                "suggested_entry_type": "突破進場",
                "suggested_entry_price": round(trigger, 2),
                "suggested_entry_note": note
            }
        return {
            "suggested_entry_type": "現價可進場",
            "suggested_entry_price": round(close, 2),
            "suggested_entry_note": "股價已站穩主要壓力關卡之上，趨勢偏多且無即時過熱疑慮，現價具備進場條件（仍請自行搭配停損執行）"
        }

    if trend_bias == "偏空":
        return {
            "suggested_entry_type": "觀望",
            "suggested_entry_price": None,
            "suggested_entry_note": "目前趨勢偏空，不建議進場做多，請等待止跌訊號出現後再評估"
        }

    support_candidates = [v for v in [ma20, bb_mid, bb_low, donchian_low] if v is not None and v < close]
    pullback_ref = max(support_candidates) if support_candidates else None
    base_note = (
        f"目前趨勢不明確，建議等股價拉回至約 "
        f"{round(pullback_ref, 2) if pullback_ref is not None else '支撐區'} "
        "或出現更明確的轉折訊號後，再考慮進場"
    )
    return _build_pullback_result(pullback_ref, "觀望/等待轉折", base_note)
                                


def calculate_trade_levels(df_out: pd.DataFrame, trend_info: Optional[dict] = None) -> dict:
    """
    根據 ATR / Donchian 通道 / 布林通道，計算一組風險報酬型的
    參考價位（entry_price / stop_loss / target_price_1 / target_price_2）。
    trend_info（來自 determine_trend_bias）只會用來調整 trade_note 的文字提醒，
    不會改變數字算法本身。

    這是「規則型試算」，不是預測漲跌，也不是投資建議，也不是「現在進場」的訊號：
    - entry_price：現價，純粹是風控試算的基準點，不代表「建議現在進場」。
      是否真的要進場，要看 trend_bias / major_force_status / 圖表視覺是否共同支持，
      這幾個判斷是分開計算的，entry_price 本身不包含任何「該不該進場」的資訊。
    - stop_loss：現價 - 1.5倍ATR，和近10日低點取較高者（避免停損設太遠）
    - target_price_1：以 2倍風險報酬比（2R）反推，risk_reward_ratio 對應的正是這個目標價
    - target_price_2：近期壓力位（Donchian上緣 / 布林上軌，取較保守者），
      這是「價格圖表上的壓力關卡」，不是用風險報酬比反推出來的目標，
      所以另外提供 risk_reward_ratio_2，避免像之前那樣，
      報告同時列出兩個目標價、卻只講一個風報比，讓人誤以為兩個目標價的風報比是一樣的。

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
            "entry_price_note": "此為現價，僅供風控試算基準，不代表建議現在進場",
            "stop_loss": None,
            "target_price_1": None,
            "target_price_2": None,
            "risk_reward_ratio": None,
            "risk_reward_ratio_2": None,
            "breakout_risk_warning": None,
            "trade_note": "ATR 資料不足（需至少14天資料才能計算），無法給出風險報酬建議"
        }

    entry_price = current_price

    atr_stop = entry_price - 1.5 * atr
    recent_low = float(df_out['low'].tail(10).min())
    stop_loss = max(atr_stop, recent_low)

    risk = entry_price - stop_loss

    # 🐛 修正邊界案例：如果近10日最低點剛好非常接近（甚至等於）現價，
    # 常見於「今天正好創近10日新低、且收在最低點附近」的長黑棒（光頭光腳型態），
    # 會導致 risk 趨近於0，讓 target_price_1／risk_reward_ratio 整組算不出來（回傳None，
    # 報告上顯示成「無」）。這裡加一個最低風險距離（至少0.5倍ATR）的保底機制，
    # 風險距離太小時改用純ATR停損，確保風控數字不會整組消失。
    MIN_RISK = 0.5 * atr
    if risk < MIN_RISK:
        stop_loss = entry_price - max(1.5 * atr, MIN_RISK)
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

    # target_price_2 是「壓力關卡價位」，不是用固定風報比反推出來的，
    # 所以要另外算一個對應target_price_2的風報比，跟target_price_1的風報比分開標示，
    # 避免報告誤把「風險報酬比2」套用到target_price_2上。
    risk_reward_ratio_2 = None
    if target_price_2 is not None and risk > 0:
        risk_reward_ratio_2 = round((target_price_2 - entry_price) / risk, 2)

    breakout_risk_warning = determine_breakout_risk_warning(latest)

    # 🐛 修正：target_price_1/2 的計算邏輯，不管trend_bias是什麼，永遠是「假設現在做多」
    # 情境下往上推算的目標價。這在趨勢偏空時會造成嚴重矛盾——報告一邊說「偏空、建議觀望/
    # 拉回進場」，一邊卻列出一組現價往上漲25~30%的「目標價」，兩者互相打架，容易誤導。
    # 這裡加一個明確的disclaimer欄位，讓Prompt在趨勢偏空時，把這組數字明確標示成
    # 「假設性試算」而不是「目前建議的操作目標」。
    target_price_disclaimer = None
    if trend_info and trend_info.get("trend_bias") == "偏空":
        target_price_disclaimer = (
            "目前趨勢判斷為偏空，下面的目標價①/②是「假設現在做多」情境下的技術性風控試算，"
            "並非目前建議追價進場的目標，偏空格局下不建議依此目標價做多操作。"
        )

    note_parts = [
        "此為量化規則試算（entry_price=現價、停損=1.5倍ATR或近期低點、目標①=2倍風險反推、目標②=近期壓力位），"
        "僅供風控試算參考，非投資建議，entry_price不代表建議現在進場"
    ]
    if trend_info:
        if trend_info.get("trend_bias") == "偏空":
            note_parts.append("目前趨勢判斷偏空，若考慮做多進場需格外謹慎")
        if trend_info.get("overbought_oversold"):
            note_parts.append(trend_info["overbought_oversold"])
    if breakout_risk_warning:
        note_parts.append(breakout_risk_warning)
    note_parts.append("實際下單請自行評估風險")

    return {
        "entry_price": round(entry_price, 2),
        "entry_price_note": "此為現價，僅供風控試算基準，不代表建議現在進場",
        "stop_loss": round(stop_loss, 2),
        "target_price_1": round(target_price_1, 2) if target_price_1 is not None else None,
        "target_price_2": round(target_price_2, 2) if target_price_2 is not None else None,
        "risk_reward_ratio": risk_reward_ratio,
        "risk_reward_ratio_2": risk_reward_ratio_2,
        "breakout_risk_warning": breakout_risk_warning,
        "target_price_disclaimer": target_price_disclaimer,
        "trade_note": "；".join(note_parts)
    }


@app.post("/analyze")
def analyze_stock_v3(payload: IndicatorRequestFM):
    if not payload.data:
        raise HTTPException(status_code=400, detail="Data list cannot be empty")

    if len(payload.data) < 45:
        # 提高門檻的原因：chart_builder.py 會把前20天的 MA/KD/RSI/布林/唐奇安
        # 全部設為 NaN（避免顯示暖機期不準確的數值）。如果資料筆數太接近20天，
        # 扣掉這前20天之後，圖上幾乎沒有資料可畫，會變成一張看起來很奇怪、
        # 大片空白的圖（這也是你之前看到的「鳥圖」的真正原因）。
        # 45天可以確保扣掉20天暖機期後，還有至少25天足夠畫出有意義的走勢圖。
        raise HTTPException(status_code=400, detail="Data length must be at least 45 days for a meaningful chart")

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

        # (3) 主力進出動向判定（改用 determine_major_force，整合三大法人/融資融券真實資料）
        chip_info = determine_major_force(current_cmf, obv_status, payload, latest_broker_diff)
        # =================================================================

        # (4) 趨勢偏多/偏空判斷（均線交叉 + MACD動能 + RSI過熱過冷）
        trend_info = determine_trend_bias(df_out.iloc[-1])

        # (5) 進場價 / 停損價 / 目標價（文字提醒會參考趨勢判斷）
        trade_levels = calculate_trade_levels(df_out, trend_info=trend_info)

        # (6) K線型態辨識 + 量價關係分析（規則型判斷，補強圖表視覺解讀的具體依據）
        candlestick_info = detect_candlestick_patterns(df_out)
        volume_price_info = analyze_volume_price_relation(df_out)
        band_squeeze_info = determine_band_squeeze(df_out)

        # (6.5) 法人常看的進階指標：背離偵測、窒息量/爆量長下影線、量能扣抵
        # （融資融券組合訊號、成交密集區估算兩項已依需求移除，不再計算）
        divergence_info = detect_divergence(df_out)
        capitulation_info = detect_capitulation_signal(df_out, candlestick_info.get('candlestick_patterns', []))
        volume_rolloff_info = analyze_volume_ma_rolloff(df_out)

        bullish_reversal_info = detect_bullish_reversal(df_out)

        # (7) 建議進場策略：根據趨勢位置給出「突破進場/拉回進場/現價可進場/觀望」的具體建議，
        # 取代單純把現價當成建議進場價的舊做法
        entry_strategy = suggest_entry_strategy(
                                                df_out.iloc[-1],
                                                trend_info,
                                                trade_levels.get('breakout_risk_warning'),
                                                stop_loss=trade_levels.get('stop_loss'),
                                                atr=float(df_out.iloc[-1].get('atr')) if pd.notna(df_out.iloc[-1].get('atr')) else None
        )
        
        # 🐛 修正：不管是「拉回進場」還是「突破進場」，只要 suggested_entry_type 不是
        # 「現價可進場」，代表建議的實際進場價（suggested_entry_price）跟 stop_loss/
        # target_price_1/2（永遠以「現價」為基準計算）用的是不同的假設進場點，
        # 兩組數字彼此對不上——之前只在「趨勢偏空」時加過類似提醒，但這個問題其實
        # 不限於偏空，中性/不明確、甚至拉回進場的情境下都會出現，這裡把範圍擴大到
        # 只要 suggested_entry_type != 現價可進場 就一律加上提醒，涵蓋所有情境。
        entry_basis_disclaimer = None
        suggested_type = entry_strategy.get("suggested_entry_type")
        if suggested_type and suggested_type != "現價可進場":
            suggested_price = entry_strategy.get("suggested_entry_price")
            price_desc = f"{suggested_price}" if suggested_price is not None else "建議價位"
            entry_basis_disclaimer = (
                f"注意：下方停損價／目標價①②是以「現價」{trade_levels.get('entry_price')}為基準計算的技術性試算，"
                f"但建議進場策略是「{suggested_type}」（參考價約{price_desc}），兩者假設的進場點不同。"
                f"若實際依建議等到{price_desc}附近才進場，屆時的停損／目標價應以那個實際進場價重新計算，"
                "不等於下方列出的數字，僅供風控邏輯示範參考。"
            )
        entry_strategy["entry_basis_disclaimer"] = entry_basis_disclaimer

        # 目標價②（近期壓力關卡）跟「突破進場」的觸發價，常常抓的是同一個技術關卡（例如唐奇安上軌），
        # 若兩者相同或target_price_2更低，代表依建議等站穩此關卡才進場時，目標價②可能已經達成或被超過，
        # 此目標對突破進場策略幾乎無獲利空間，需明確提醒。
        if (suggested_type == "突破進場"
                and entry_strategy.get("suggested_entry_price") is not None
                and trade_levels.get("target_price_2") is not None
                and trade_levels["target_price_2"] <= entry_strategy["suggested_entry_price"]):
            extra_note = (
                f"⚠️ 目標價②（{trade_levels['target_price_2']}）與建議突破進場觸發價"
                f"（{entry_strategy['suggested_entry_price']}）相同或更低，"
                "代表若依「站穩此關卡才進場」的策略執行，實際進場時可能已達成或超過目標價②，"
                "此目標對突破進場策略幾乎無獲利空間，建議以目標價①作為主要參考，"
                "或等待更上方的壓力關卡出現後再重新評估目標價。"
            )
            existing = trade_levels.get("target_price_disclaimer")
            trade_levels["target_price_disclaimer"] = (existing + "；" + extra_note) if existing else extra_note
                    
        signal_divergence_info = determine_signal_divergence(trend_info.get("trend_bias"),chip_info.get("major_force_status"))     
        # 5. 繪製圖表
        stock_label_parts = [p for p in [payload.stock_symbol, payload.stock_name] if p]
        stock_label = " ".join(stock_label_parts)
        chart_buffer = draw_ultimate_chart(df_out, stock_label=stock_label)

        # 6. 圖檔轉 Base64 字串
        image_base64 = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')

        # 7. 轉回 JSON 格式並擷取最新一筆
        df_json = df_out.reset_index()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        latest_metrics = df_json.tail(1).to_dict(orient='records')[0]

        # 注入主力診斷文字與交易價位建議
        latest_metrics['obv_status'] = obv_status
        latest_metrics.update(chip_info)
        # 🐛 修正：determine_major_force() 內部有讀取這些欄位去算分數跟desc文字，
        # 但算完之後這些「原始數字」從來沒有被放進回傳的metrics裡，
        # 導致 n8n Prompt 模板寫 {{ $json.metrics.foreign_net_buy }} 之類的引用永遠是 undefined
        # （雖然 major_force_desc 的文字裡有帶到這些數字，但沒有獨立的欄位可以讓Prompt直接抓）。
        # 這裡把payload收到的原始籌碼欄位，原封不動也放進metrics輸出，供Prompt直接引用。
        latest_metrics['foreign_net_buy'] = payload.foreign_net_buy
        latest_metrics['trust_net_buy'] = payload.trust_net_buy
        latest_metrics['dealer_net_buy'] = payload.dealer_net_buy
        latest_metrics['institutional_streak_days'] = payload.institutional_streak_days
        latest_metrics['institutional_history_days'] = payload.institutional_history_days
        latest_metrics['institutional_net_5d'] = payload.institutional_net_5d
        latest_metrics['institutional_net_10d'] = payload.institutional_net_10d
        latest_metrics['institutional_net_20d'] = payload.institutional_net_20d
        latest_metrics['stock_symbol'] = payload.stock_symbol or ""
        latest_metrics['stock_name'] = payload.stock_name or ""
        latest_metrics.update(trend_info)
        latest_metrics.update(trade_levels)
        latest_metrics.update(candlestick_info)
        latest_metrics.update(volume_price_info)
        latest_metrics.update(band_squeeze_info)
        latest_metrics.update(divergence_info)
        latest_metrics.update(capitulation_info)
        latest_metrics.update(volume_rolloff_info)
        latest_metrics.update(bullish_reversal_info)
        latest_metrics.update(entry_strategy)
        latest_metrics.update(signal_divergence_info)

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
