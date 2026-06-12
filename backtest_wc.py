"""Backtest: train as-of each past World Cup's opening day, score OOS predictions.

Group-stage matches only (knockout results in results.csv include extra time,
which muddies 90-minute 1X2 scoring).

Usage:
    python backtest_wc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wc import data, model as dcmodel

# (label, train_asof = day before opening, group stage date range)
WORLD_CUPS = [
    ("WC2014 巴西", "2014-06-11", "2014-06-12", "2014-06-26"),
    ("WC2018 俄羅斯", "2018-06-13", "2018-06-14", "2018-06-28"),
    ("WC2022 卡達", "2022-11-19", "2022-11-20", "2022-12-02"),
]


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - y) ** 2).sum())


def main() -> None:
    df = data.load_results()
    rows = []
    details = []

    for label, asof, d0, d1 in WORLD_CUPS:
        matches = df[
            (df["tournament"] == "FIFA World Cup")
            & (df["date"] >= d0) & (df["date"] <= d1)
            & df["home_score"].notna()
        ]
        train = data.training_matches(
            df, start=str(pd.Timestamp(asof) - pd.DateOffset(years=8))[:10],
            asof=pd.Timestamp(asof),
        )
        m = dcmodel.fit(train)
        known = set(m.att)

        b_model, ll_model, hits, b_unif = [], [], [], []
        n_skipped = 0
        for r in matches.itertuples():
            if r.home_team not in known or r.away_team not in known:
                n_skipped += 1
                continue
            p = m.outcome_probs(r.home_team, r.away_team, not r.neutral)
            pv = np.array([p["home"], p["draw"], p["away"]])
            out = ("home" if r.home_score > r.away_score
                   else "away" if r.home_score < r.away_score else "draw")
            y = np.array([out == k for k in ("home", "draw", "away")], float)
            b_model.append(brier(pv, y))
            ll_model.append(-np.log(max(pv[y.astype(bool)][0], 1e-12)))
            hits.append(["home", "draw", "away"][int(pv.argmax())] == out)
            b_unif.append(brier(np.full(3, 1 / 3), y))
            details.append({
                "wc": label, "date": r.date.date(),
                "match": f"{r.home_team} vs {r.away_team}",
                "score": f"{int(r.home_score)}-{int(r.away_score)}",
                "p_home": round(p["home"], 3), "p_draw": round(p["draw"], 3),
                "p_away": round(p["away"], 3), "outcome": out,
                "brier": round(b_model[-1], 4),
            })

        rows.append({
            "世足": label, "場數": len(b_model), "略過": n_skipped,
            "訓練場數": len(train),
            "Brier": np.mean(b_model), "LogLoss": np.mean(ll_model),
            "命中率": np.mean(hits), "Brier_亂猜": np.mean(b_unif),
        })

    out = pd.DataFrame(rows)
    print("=== 過去三屆小組賽 OOS 回測（Brier 越低越好，亂猜=0.667）===")
    print(out.round(4).to_string(index=False))
    tot = pd.DataFrame(details)
    print(f"\n合計 {len(tot)} 場  Brier={tot['brier'].mean():.4f}  "
          f"命中率={(np.mean([r['命中率'] * r['場數'] for r in rows]) / np.mean([r['場數'] for r in rows])):.1%}")

    # 參考基準：歷屆研究中博彩公司收盤賠率的小組賽 Brier 約 0.56-0.60
    Path("out").mkdir(exist_ok=True)
    tot.to_csv("out/backtest_past_wc.csv", index=False)
    print("\n明細已存 out/backtest_past_wc.csv")
    print("\n最離譜的 5 場（模型最有信心卻錯最大）：")
    print(tot.nlargest(5, "brier").to_string(index=False))


if __name__ == "__main__":
    main()
