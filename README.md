# stock_indicators_python
KD 指標、RSI 指標、5日與20日移動平均線、以及 MACD 指標的 Python 實作範例，針對 OHLCV 資料（High、Low、Close）計算並輸出多個技術指標欄位。

## 專案簡介
本專案提供可直接導入的 Python 模組，用於計算多種短期股市技術分析指標。方便在分析、回測、或交易策略開發中使用。

## 安裝與依賴
- 請先安裝 Python 3.8+。
- 安裝必要依賴：
  - pandas
  - numpy

可使用 pip 安裝：

```
pip install pandas numpy
```

如果要直接從專案中安裝（後續若新增 packaging 設定也會更方便）：

```
pip install -e .
```

## 使用說明
1) 匯入並呼叫函式
```python
from stock_indicators_python.indicators.kd_rsi_ma_macd import calculate_kd_rsi_ma_macd
import pandas as pd

# 假設 df 已包含 High、Low、Close 欄位，並以時間順序排序
# df = pd.read_csv('your_ohlcv.csv')
df_out = calculate_kd_rsi_ma_macd(
    df_ohlcv=df,
    k_period=14, d_period=3, rsi_period=14,
    ma_short=5, ma_long=20,
    macd_fast=12, macd_slow=26, macd_signal=9
)
print(df_out.tail())
```

2) 輸出欄位說明
- KD 指標：%K、%D
- RSI 指標：RSI
- 移動平均：MA_5、MA_20
- MACD 指標：MACD_Line、Signal_Line、MACD_Hist
- 額外輸出：Typical_Price = (High + Low + Close) / 3

## 指標意義簡介（快速瞭解）
- KD：衡量收盤價相對於過去區間內的波動範圍，常用於判斷超買/超賣。
- RSI：相對強弱指標，反映價格變動的速度與變化的幅度。
- MA（5/20）：短期與中期均線的差異，能提供趨勢方向與支撐/壓力位的線索。
- MACD：利用快慢二條 EMA 的差距，搭配訊號線，可用於辨識動能與轉折。
- Typical Price：典型價格，是高低收三者的平均值，常用於平滑或作為特定策略的輸入。

## 測試與驗證
本專案提供單元測試，確保指標函式在不同輸入下能返回預期欄位。

- 測試框架：pytest
- 測試檔案位置：stock_indicators_python/tests/test_indicators.py

執行測試：

```
pytest -q
```

## GitHub 上傳（上傳步驟範例）
以下為手動推送至 GitHub 的常見流程，假設你已經在 GitHub 建好一個新的倉庫，例如：`stock-indicators-python`。

1) 初始化本地倉庫
```
cd stock_indicators_python
git init
```

2) 新增檔案並提交
```
git add .
git commit -m "feat: add KD/RSI/MA/MACD indicators with Typical Price" 
```

3) 連接遠端儲存庫（以 https 為例）
```
git remote add origin https://github.com/your-username/stock-indicators-python.git
```

4) 推送至 main 分支
```
git branch -M main
git push -u origin main
```

如果你偏好使用 SSH，將 remote 設定為 `git@github.com:your-username/stock-indicators-python.git`。

## 自動化測試與 CI（建議）
建議使用 GitHub Actions 進行 CI 檢查，確保在 PR/Push 後自動跑測試。

建立工作流程檔案 `.github/workflows/python-ci.yml`，內容如下：

```
name: Python package

on:
  push:
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest -q
```

## 貢獻指南
歡迎貢獻！若要貢獻，請先建立分支，提交 Pull Request，並附上單元測試及說明。

## 授權
MIT License
