"""Daily driver: refresh data, fit model, predict upcoming, simulate tournament.

Usage:
    python run_predict.py                 # use local data
    python run_predict.py --refresh       # re-download results.csv first
    python run_predict.py --sims 20000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wc import data, model as dcmodel, odds as oddsmod, simulate

OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--days-ahead", type=int, default=3)
    args = ap.parse_args()

    if args.refresh:
        print("refreshing results.csv ...")
        data.refresh_results()

    df = data.load_results()
    fixtures = data.wc_fixtures(df)
    train = data.training_matches(df)
    print(f"training on {len(train)} matches "
          f"({train['date'].min().date()} .. {train['date'].max().date()})")

    m = dcmodel.fit(train)
    print(f"home_adv={m.home_adv:.3f}  rho={m.rho:.4f}  mu={m.mu:.3f}")

    missing = [t for t in data.ALL_TEAMS if t not in m.att]
    if missing:
        raise SystemExit(f"teams with no training data: {missing}")

    # --- match predictions for the whole group stage (pending matches) ---
    pend = fixtures[fixtures["home_score"].isna()]
    preds = []
    for r in pend.itertuples():
        p = m.outcome_probs(r.home_team, r.away_team, not r.neutral)
        preds.append({
            "date": r.date, "group": r.group,
            "home_team": r.home_team, "away_team": r.away_team,
            "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
        })
    preds = pd.DataFrame(preds)

    # also score predictions retroactively on already-played WC matches
    played = fixtures[fixtures["home_score"].notna()]
    backpreds = []
    for r in played.itertuples():
        p = m.outcome_probs(r.home_team, r.away_team, not r.neutral)
        backpreds.append({
            "date": r.date, "group": r.group,
            "home_team": r.home_team, "away_team": r.away_team,
            "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
        })
    backpreds = pd.DataFrame(backpreds)

    OUT_DIR.mkdir(exist_ok=True)
    all_preds = pd.concat([backpreds, preds], ignore_index=True)
    all_preds.to_csv(OUT_DIR / "match_predictions.csv", index=False)

    horizon = preds[preds["date"] <= preds["date"].min()
                    + pd.Timedelta(days=args.days_ahead)]
    print(f"\n=== upcoming matches (next {args.days_ahead} days) ===")
    for r in horizon.itertuples():
        print(f"{r.date.date()} [{r.group}] {r.home_team} vs {r.away_team}: "
              f"H {r.p_home:.0%} / D {r.p_draw:.0%} / A {r.p_away:.0%}")

    # --- market comparison ---
    odds = oddsmod.load_odds()
    if len(odds):
        ed = oddsmod.edges(preds, odds)
        if len(ed):
            print("\n=== model vs market edges (>=5%) ===")
            print(ed.to_string(index=False))
        sb = oddsmod.scoreboard(all_preds, odds, fixtures)
        if len(sb):
            cols = [c for c in ["model_brier", "mkt_brier",
                                "model_logloss", "mkt_logloss"] if c in sb]
            print("\n=== calibration so far (lower = better) ===")
            print(sb[cols].mean().round(4).to_string())
            sb.to_csv(OUT_DIR / "scoreboard.csv", index=False)
    else:
        sb = oddsmod.scoreboard(all_preds, odds, fixtures)
        if len(sb):
            print(f"\nmodel Brier on {len(sb)} played matches: "
                  f"{sb['model_brier'].mean():.4f} "
                  f"(logloss {sb['model_logloss'].mean():.4f})")
        print("(no data/odds.csv yet — market comparison skipped)")

    # --- tournament simulation ---
    print(f"\nsimulating tournament x{args.sims} ...")
    tab = simulate.simulate_tournament(m, fixtures, n_sims=args.sims)
    tab.to_csv(OUT_DIR / "tournament_probs.csv", index=False)
    top = tab.head(15).copy()
    for c in simulate.ROUNDS:
        top[c] = (top[c] * 100).round(1)
    print("\n=== title race (top 15, % chance to reach each round) ===")
    print(top.to_string(index=False))

    # --- 累積今日歷史快照（給「歷史走勢」分頁）---
    import snapshot_history
    snapshot_history.run([pd.Timestamp.now().normalize()], n_sims=args.sims)


if __name__ == "__main__":
    main()
