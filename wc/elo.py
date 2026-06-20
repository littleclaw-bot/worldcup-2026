"""World Football Elo 評分,從 results.csv 全史比賽自行計算.

公式（World Football Elo Ratings 標準）：
    R' = R + K · G · (W − We)
    We = 1 / (10^(−dr/400) + 1)，dr = (R_home + 主場優勢) − R_away
    K  = 賽事重要性權重；G = 進球差倍率；W = 勝1/和0.5/負0
所有隊 1500 起跳,按時間順序跑完全部比賽,得到當前實力分。
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

BASE = 1500.0
HOME_ADV = 100.0


def _k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "qualif" in t:
        return 40.0
    if any(s in t for s in ("euro", "copa", "african", "asian cup", "gold cup")):
        return 50.0
    if "nations league" in t or "confederations" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _g_mult(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(results: pd.DataFrame) -> dict[str, float]:
    """回傳 {team: 當前 Elo 分}，依時間順序跑完所有已賽比賽."""
    df = results[results["home_score"].notna() & results["away_score"].notna()]
    df = df.sort_values("date")
    rating: dict[str, float] = defaultdict(lambda: BASE)

    for r in df.itertuples():
        rh, ra = rating[r.home_team], rating[r.away_team]
        ha = 0.0 if r.neutral else HOME_ADV
        we = 1.0 / (10 ** (-((rh + ha) - ra) / 400.0) + 1.0)
        if r.home_score > r.away_score:
            w = 1.0
        elif r.home_score < r.away_score:
            w = 0.0
        else:
            w = 0.5
        delta = _k_factor(r.tournament) * _g_mult(
            int(r.home_score) - int(r.away_score)
        ) * (w - we)
        rating[r.home_team] = rh + delta
        rating[r.away_team] = ra - delta

    return dict(rating)


def win_probs(rating: dict[str, float], home: str, away: str,
              home_field: bool = False) -> dict[str, float]:
    """純 Elo 的勝/和/負機率（和局用經驗近似拆分）."""
    rh = rating.get(home, BASE) + (HOME_ADV if home_field else 0.0)
    ra = rating.get(away, BASE)
    p_home_or_draw = 1.0 / (10 ** (-(rh - ra) / 400.0) + 1.0)
    # 以分差估和局率（分差越小越易和）：經驗式
    draw = 0.27 * (1 - abs(rh - ra) / 1000.0)
    draw = max(0.10, min(0.30, draw))
    home = p_home_or_draw - draw / 2
    away = 1 - p_home_or_draw - draw / 2
    return {"home": max(0.0, home), "draw": draw, "away": max(0.0, away)}
