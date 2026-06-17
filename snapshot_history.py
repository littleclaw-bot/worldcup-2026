"""每日預測快照,累積歷史走勢（冠軍/晉級機率 + 模型 vs 市場校準）.

以「某日視角」重算:只用該日(含)之前的資料 fit 模型,並把該日之後的
比賽視為未踢,再跑模擬。可回填開賽至今的每日曲線。

輸出（append,以日期 dedup,不覆蓋歷史）：
    data/history/odds_history.csv        date,team,R32,R16,QF,SF,F,champion
    data/history/calibration_history.csv date,n,model_brier,mkt_brier,
                                         model_logloss,mkt_logloss

用法：
    python snapshot_history.py              # 只快照今天
    python snapshot_history.py --backfill   # 回填 2026-06-11 至今每一天
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wc import data, model as dcmodel, odds as oddsmod, simulate

HIST_DIR = data.DATA_DIR / "history"
ODDS_HIST = HIST_DIR / "odds_history.csv"
CALIB_HIST = HIST_DIR / "calibration_history.csv"
TOURNAMENT_START = "2026-06-11"


def _fixtures_asof(fixtures: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """把 asof 之後的比賽分數清成 NaN（視為未踢）."""
    f = fixtures.copy()
    mask = f["date"] > asof
    f.loc[mask, ["home_score", "away_score"]] = np.nan
    return f


def snapshot(df: pd.DataFrame, fixtures: pd.DataFrame, asof: pd.Timestamp,
             odds: pd.DataFrame, n_sims: int = 8000):
    """回傳 (champion_rows: DataFrame, calib_row: dict|None) 以 asof 視角."""
    train = data.training_matches(df, asof=asof)
    m = dcmodel.fit(train)
    f_asof = _fixtures_asof(fixtures, asof)

    tab = simulate.simulate_tournament(m, f_asof, n_sims=n_sims)
    tab = tab.drop(columns=["group"]).copy()
    tab.insert(0, "date", asof.date().isoformat())

    preds = pd.DataFrame([
        {"date": r.date, "home_team": r.home_team, "away_team": r.away_team,
         **{f"p_{k}": v for k, v in
            m.outcome_probs(r.home_team, r.away_team, not r.neutral).items()}}
        for r in fixtures.itertuples()
    ])
    sb = oddsmod.scoreboard(preds, odds, f_asof)
    calib = None
    if len(sb):
        calib = {
            "date": asof.date().isoformat(), "n": len(sb),
            "model_brier": round(sb["model_brier"].mean(), 4),
            "model_logloss": round(sb["model_logloss"].mean(), 4),
            "mkt_brier": round(sb["mkt_brier"].mean(), 4)
            if "mkt_brier" in sb and sb["mkt_brier"].notna().any() else np.nan,
            "mkt_logloss": round(sb["mkt_logloss"].mean(), 4)
            if "mkt_logloss" in sb and sb["mkt_logloss"].notna().any() else np.nan,
        }
    return tab, calib


def _append_dedup(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    """以 keys 去重後寫回（新資料覆蓋同 key 舊資料）."""
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=keys, keep="last")
    else:
        combined = new
    combined.to_csv(path, index=False)


def run(asofs: list[pd.Timestamp], n_sims: int = 8000) -> None:
    df = data.load_results()
    fixtures = data.wc_fixtures(df)
    odds = oddsmod.load_odds()
    for asof in asofs:
        tab, calib = snapshot(df, fixtures, asof, odds, n_sims=n_sims)
        _append_dedup(ODDS_HIST, tab, keys=["date", "team"])
        if calib is not None:
            _append_dedup(CALIB_HIST, pd.DataFrame([calib]), keys=["date"])
        top = tab.nlargest(3, "champion")[["team", "champion"]]
        leaders = ", ".join(f"{r.team} {r.champion:.1%}" for r in top.itertuples())
        print(f"{asof.date()}  快照完成（前三：{leaders}）"
              f"{'  + 校準' if calib else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="回填開賽至今每一天")
    ap.add_argument("--sims", type=int, default=8000)
    args = ap.parse_args()

    today = pd.Timestamp.now().normalize()
    if args.backfill:
        asofs = list(pd.date_range(TOURNAMENT_START, today, freq="D"))
    else:
        asofs = [today]
    run(asofs, n_sims=args.sims)
    print(f"\n歷史已更新：{ODDS_HIST}")


if __name__ == "__main__":
    main()
