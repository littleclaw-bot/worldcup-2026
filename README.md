# worldcup-2026 — 世足預測模型

**🌐 Live: <https://wc2026-predict.streamlit.app/>**

2026 世界盃（美加墨，48 隊新賽制）的 Dixon-Coles 預測模型 + 賽程 Monte Carlo
模擬 + Elo 實力評分 + 對賭市場校準 + 球星導覽。純紙上驗證，不真下注。

## 架構

```text
data/
  results.csv      # martj42/international_results：1872 至今所有國際賽（訓練 + Elo）
  cup.txt          # openfootball 分組/賽程/開球時間
  cup_finals.txt   # 淘汰賽 bracket（slot 落位規則）
  odds.csv         # 手動維護的市場賠率（decimal odds）
  history/         # 每日快照：odds_history / elo_history / calibration_history
wc/
  data.py          # 分組表、R32 bracket、資料載入、開球時間解析
  model.py         # Dixon-Coles：加權 Poisson GLM + rho 低比分修正
  simulate.py      # 小組賽 + 8 最佳第三 backtracking 落位 + 淘汰賽 MC
  bracket.py       # 最可能淘汰賽對戰樹 + 夢幻對決相遇機率
  elo.py           # World Football Elo（1872 全史自算）
  odds.py          # implied prob、edge 偵測、Brier/log loss 計分
  live.py          # ESPN 即時比分 overlay（補 martj42 delay）
  teamguide.py     # 球星導覽資料（10 隊 48 人：背號/球隊/年薪/身價/年齡）
app.py             # Streamlit 儀表板（10 分頁）
run_predict.py     # daily driver（含每日快照）
snapshot_history.py# 歷史快照（可 --backfill 以「某日視角」回算）
calendar_scores.py # 列已踢完場次 + 比分（給行事曆標題更新用）
backtest_wc.py     # 過去三屆 OOS 回測
```

## 模型

- 訓練：2018 起所有國際賽（~8100 場），半衰期 2.5 年時間衰減，友誼賽 ×0.6
- `log λ = μ + home_adv·非中立 + att[隊] − def[對手]`，sklearn PoissonRegressor
- Dixon-Coles ρ 修正低比分相關性；淘汰賽平局 → 延長賽（λ/3）→ PK 50/50
- 主場優勢只給地主國（美/加/墨在自己國家的場次）
- **Elo**（`wc/elo.py`）：另一套獨立實力評分，跟 Poisson 模型互相印證

## 資料源（兩層）

| 來源 | 角色 | 即時性 |
|---|---|---|
| **martj42/international_results** | 歷史全史（訓練 + Elo + 基準比分），GitHub raw CSV | 慢幾小時 |
| **ESPN scoreboard API** | 當屆世足即時 FT 比分，補 martj42 還沒更新的 | FT 即時 |
| **openfootball/worldcup** | 分組、賽程、開球時間、淘汰賽 slot | 靜態 |

ESPN overlay（`wc/live.py`）抓 `site.api.espn.com/.../soccer/fifa.world/scoreboard`
的 FT 結果，patch 進 martj42 還沒補的場次。失敗自動 fallback 純 martj42。

## 儀表板（10 分頁）

賽事預測 / 賽程表（台灣開球時間） / 冠軍機率 / 🗺️對戰樹 / 💪Elo實力榜 /
分組形勢 / 📊模型vs市場 / 📈歷史走勢（冠軍率+Elo+校準三線） / 單場下鑽 /
🌟球星導覽。

```bash
# 本機啟動（雙擊也可）
start_dashboard.bat            # → http://localhost:8510
```

## Daily routine

```bash
python run_predict.py --refresh   # 更新比分、預測、append 當日快照
```

「跑今天的世足預測」= refresh + 補當天賠率到 odds.csv + push + 順手更新行事曆比分。

## 回測結論（誠實聲明）

- 三屆（2014/18/22）小組賽 144 場 OOS：Brier **0.590**（亂猜 0.667、
  bookmaker 0.56–0.60）→ **模型與市場同級，但無 edge**
- 運彩抽水 15–20% → 期望值必負 → **不下注，純紙上練 calibration**
- 模型只看歷史比分，不知傷兵/輪換/陣容；跟市場分歧大時通常是市場對
- 小樣本（世足 104 場），校準結論統計檢力有限

## 部署

GitHub `littleclaw-bot/worldcup-2026` → push main 自動重新部署到 Streamlit
Community Cloud。`data/odds.csv` 與 `data/history/` 已 commit，雲端讀得到。
