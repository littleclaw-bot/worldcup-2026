# worldcup-2026 — 世足預測模型

2026 世界盃（美加墨，48 隊新賽制）的 Dixon-Coles 預測模型 + 賽程 Monte Carlo 模擬 + 對賭市場校準追蹤。純紙上驗證，不真下注。

## 架構

```
data/
  results.csv      # martj42/international_results：1872 至今所有國際賽 +
                   # 2026 世足 72 場小組賽（未踢=NA，上游持續更新比分）
  cup.txt          # openfootball 分組/賽程（參考用）
  cup_finals.txt   # 淘汰賽 bracket（參考用）
  odds.csv         # 手動維護的市場賠率（decimal odds，每場一列）
wc/
  data.py          # 分組表、R32 bracket（含第三名分配限制）、資料載入
  model.py         # Dixon-Coles：加權 Poisson GLM + rho 低比分修正
  simulate.py      # 小組賽 + 8 最佳第三名 bracket 匹配 + 淘汰賽 MC
  odds.py          # implied prob、edge 偵測、Brier/log loss 計分
run_predict.py     # daily driver
out/               # 輸出（match_predictions.csv / tournament_probs.csv / scoreboard.csv）
```

## 模型

- 訓練資料：2018 起所有國際賽（~8100 場），半衰期 2.5 年指數時間衰減，友誼賽權重 ×0.6
- `log λ = μ + home_adv·非中立 + att[隊] − def[對手]`，sklearn PoissonRegressor（L2 α=5e-4）
- Dixon-Coles ρ 修正低比分相關性（profile likelihood 二段估計）
- 淘汰賽平局 → 延長賽（λ/3 Poisson）→ PK 50/50
- 主場優勢只給地主國（美/加/墨在自己國家的場次）

## 賽制處理

48 隊 12 組，每組前二 + 8 個最佳第三晉級 32 強。8 個第三名的 bracket
落位有官方限制（每個 slot 只接受特定 5 組的第三名），用 backtracking
完美匹配解；無解時退化為任意分配。

## Daily routine

```bash
# 每天跑一次（results.csv 上游會更新比分）
C:/ProgramData/anaconda3/envs/python310/python.exe run_predict.py --refresh
```

輸出：近 3 天賽事勝平負機率、model vs market edge（≥5%）、
累積 Brier/log loss（模型 vs 市場誰準）、冠軍機率榜。

### 賠率維護

`data/odds.csv` 目前手動填（web 搜當天賠率，美式賠率換算 decimal：
`+250 → 3.50`、`-200 → 1.50`）。要自動化可去 [the-odds-api.com](https://the-odds-api.com)
申請免費 key（500 credits/月），sport key = `soccer_fifa_world_cup`。

## 已知限制（誠實聲明)

- 模型只看歷史比分，不知道傷兵、輪換、陣容——跟市場分歧大時（如開賽首週
  動輒 10%+ edge）**通常是市場對**，靠 Brier 追蹤驗證而不是直接信 edge
- 小樣本：世足 104 場，校準結論的統計檢力有限
- 第三名 bracket 匹配在多解時取字典序第一解，與 FIFA 實際分配可能不同
  （對機率影響極小）
