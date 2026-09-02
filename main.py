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
        "version": "2026-09-01-candlestick-volume-price-riskfix"
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
