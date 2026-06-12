"""Market odds: implied probabilities, model-vs-market comparison, scoring.

data/odds.csv columns:
    date,home_team,away_team,odds_home,odds_draw,odds_away,source
Decimal (European) odds. One row per match (use closing or latest odds).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR

ODDS_CSV = DATA_DIR / "odds.csv"


def load_odds() -> pd.DataFrame:
    if not ODDS_CSV.exists():
        return pd.DataFrame(
            columns=["date", "home_team", "away_team",
                     "odds_home", "odds_draw", "odds_away", "source"]
        )
    df = pd.read_csv(ODDS_CSV, parse_dates=["date"])
    raw = 1.0 / df[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    margin = raw.sum(axis=1)
    imp = raw / margin[:, None]  # proportional margin removal
    df[["mkt_home", "mkt_draw", "mkt_away"]] = imp
    df["overround"] = margin - 1.0
    return df


def _outcome(hs: float, as_: float) -> str:
    return "home" if hs > as_ else ("away" if hs < as_ else "draw")


def scoreboard(model_preds: pd.DataFrame, odds: pd.DataFrame,
               results: pd.DataFrame) -> pd.DataFrame:
    """Per finished match: model vs market Brier / log loss.

    model_preds: date, home_team, away_team, p_home, p_draw, p_away
    results: rows with home_score/away_score filled.
    """
    fin = results[results["home_score"].notna()].copy()
    fin["outcome"] = [
        _outcome(r.home_score, r.away_score) for r in fin.itertuples()
    ]
    m = model_preds.merge(
        fin[["date", "home_team", "away_team", "home_score", "away_score", "outcome"]],
        on=["home_team", "away_team"], suffixes=("", "_r"),
    )
    if odds is not None and len(odds):
        m = m.merge(
            odds[["home_team", "away_team", "mkt_home", "mkt_draw", "mkt_away"]],
            on=["home_team", "away_team"], how="left",
        )
    rows = []
    for r in m.itertuples():
        y = np.array([r.outcome == k for k in ("home", "draw", "away")], float)
        p_model = np.array([r.p_home, r.p_draw, r.p_away])
        row = {
            "date": r.date, "match": f"{r.home_team} vs {r.away_team}",
            "score": f"{int(r.home_score)}-{int(r.away_score)}",
            "outcome": r.outcome,
            "model_brier": float(((p_model - y) ** 2).sum()),
            "model_logloss": float(-np.log(max(p_model[y.astype(bool)][0], 1e-12))),
        }
        if hasattr(r, "mkt_home") and pd.notna(getattr(r, "mkt_home", np.nan)):
            p_mkt = np.array([r.mkt_home, r.mkt_draw, r.mkt_away])
            row["mkt_brier"] = float(((p_mkt - y) ** 2).sum())
            row["mkt_logloss"] = float(-np.log(max(p_mkt[y.astype(bool)][0], 1e-12)))
        rows.append(row)
    return pd.DataFrame(rows)


def edges(model_preds: pd.DataFrame, odds: pd.DataFrame,
          min_edge: float = 0.05) -> pd.DataFrame:
    """Where model prob exceeds market implied prob by >= min_edge (paper only)."""
    if not len(odds):
        return pd.DataFrame()
    m = model_preds.merge(
        odds, on=["home_team", "away_team"], suffixes=("", "_o")
    )
    rows = []
    for r in m.itertuples():
        for side, p, q, o in [
            ("home", r.p_home, r.mkt_home, r.odds_home),
            ("draw", r.p_draw, r.mkt_draw, r.odds_draw),
            ("away", r.p_away, r.mkt_away, r.odds_away),
        ]:
            edge = p - q
            if edge >= min_edge:
                rows.append({
                    "match": f"{r.home_team} vs {r.away_team}",
                    "side": side, "model_p": round(p, 3),
                    "market_p": round(q, 3), "odds": o,
                    "edge": round(edge, 3),
                    "ev_per_unit": round(p * o - 1.0, 3),
                })
    return pd.DataFrame(rows).sort_values("edge", ascending=False) if rows else pd.DataFrame()
