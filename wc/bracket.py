"""預測淘汰賽對戰樹 + 任兩隊相遇（夢幻對決）機率.

兩個產物都來自同一次蒙地卡羅：
  • predicted bracket — 取「最可能的 32 強對戰」當起點,之後每場用模型
    解析勝率(含延長賽/PK,與 simulate.sample_winner 同一套規則)讓較強者
    晉級,得到一條內部一致的「最可能淘汰賽路徑」。
  • meeting probs — 統計每對球隊在淘汰賽相遇的次數(單屆最多遇一次,
    所以各輪次數相加 = 兩隊「會碰頭」的機率)。
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .data import GROUPS, KNOCKOUT_ROUNDS, TEAM_TO_GROUP
from .model import DCModel, score_grid_from_lambdas
from .simulate import MatchSampler, _assign_thirds, _standings

ROUND_ORDER = ["R32", "R16", "QF", "SF", "F"]
ROUND_ZH = {"R32": "32強", "R16": "16強", "QF": "8強", "SF": "4強", "F": "決賽"}

# match_no -> 該場屬於哪一輪
MATCH_ROUND = {no: rn for rn, ms in KNOCKOUT_ROUNDS for no, *_ in ms}


def ko_win_prob(model: DCModel, home: str, away: str, home_field: bool) -> float:
    """P(home 晉級):正規賽 + 延長賽 + PK,對應 simulate.MatchSampler.sample_winner。

    延長賽用 lambda/3(30 分鐘≈正規賽 1/3),PK 視為 50/50。
    """
    reg = model.score_grid(home, away, home_field)
    p_home = float(np.tril(reg, -1).sum())
    p_draw = float(np.trace(reg))
    lam_h, lam_a = model.lambdas(home, away, home_field)
    et = score_grid_from_lambdas(lam_h / 3.0, lam_a / 3.0, 0.0)
    p_home_et = float(np.tril(et, -1).sum()) + float(np.trace(et)) * 0.5
    return p_home + p_draw * p_home_et


def simulate_bracket(
    model: DCModel, fixtures: pd.DataFrame, n_sims: int = 10000, seed: int = 42
):
    """跑完整賽事 MC,回傳 (r32_home, r32_away, meet)。

    r32_home/r32_away: {match_no: Counter(team -> 佔據該位次數)}（32 強雙方）
    meet: {round_name: Counter(frozenset({a, b}) -> 相遇次數)}
    """
    rng = np.random.default_rng(seed)
    sampler = MatchSampler(model, rng)

    played = fixtures[fixtures["home_score"].notna()]
    pending = fixtures[fixtures["home_score"].isna()]
    played_rows = [
        (r.home_team, r.away_team, int(r.home_score), int(r.away_score))
        for r in played.itertuples()
    ]
    pending_rows = [
        (r.home_team, r.away_team, not r.neutral) for r in pending.itertuples()
    ]
    third_slots = [
        (no, as_slot.split(":")[1])
        for name, rnd in KNOCKOUT_ROUNDS if name == "R32"
        for no, _, as_slot, _ in rnd if as_slot.startswith("3:")
    ]

    r32_matches = KNOCKOUT_ROUNDS[0][1]
    r32_home = {no: Counter() for no, *_ in r32_matches}
    r32_away = {no: Counter() for no, *_ in r32_matches}
    meet = {rn: Counter() for rn in ROUND_ORDER}

    for _ in range(n_sims):
        results = list(played_rows)
        results += [
            (h, a, *sampler.sample(h, a, hf)) for h, a, hf in pending_rows
        ]
        by_group = {g: [] for g in GROUPS}
        for h, a, hs, as_ in results:
            by_group[TEAM_TO_GROUP[h]].append((h, a, hs, as_))

        slot_team: dict[str, str] = {}
        thirds = []
        for g, teams in GROUPS.items():
            rows = _standings(teams, by_group[g], rng)
            slot_team[f"1{g}"] = rows[0][0]
            slot_team[f"2{g}"] = rows[1][0]
            thirds.append((g,) + rows[2])
        thirds.sort(key=lambda r: (r[2], r[3], r[4], r[5]), reverse=True)
        qual_thirds = {g: team for g, team, *_ in thirds[:8]}
        assignment = _assign_thirds(list(qual_thirds), third_slots)

        winners: dict[str, str] = {}
        for rname, matches in KNOCKOUT_ROUNDS:
            for no, hs, as_slot, venue in matches:
                home = winners[hs] if hs.startswith("W") else slot_team[hs]
                if as_slot.startswith("3:"):
                    away = qual_thirds[assignment[no]]
                elif as_slot.startswith("W"):
                    away = winners[as_slot]
                else:
                    away = slot_team[as_slot]
                if rname == "R32":
                    r32_home[no][home] += 1
                    r32_away[no][away] += 1
                meet[rname][frozenset((home, away))] += 1
                hf = home == venue
                af = away == venue
                if af and not hf:
                    home, away = away, home
                winners[f"W{no}"] = sampler.sample_winner(home, away, hf and not af)
    return r32_home, r32_away, meet


def predicted_bracket(
    model: DCModel, fixtures: pd.DataFrame, n_sims: int = 10000, seed: int = 42
):
    """回傳 (bracket, champion, meet)。

    bracket: {match_no: {round, home, away, p_home, winner, win_p, venue}}
    champion: 預測冠軍隊名
    meet: 同 simulate_bracket（給夢幻對決用）
    """
    r32_home, r32_away, meet = simulate_bracket(model, fixtures, n_sims, seed)

    teams_at: dict[str, str] = {}  # "W{no}" -> 晉級者
    bracket: dict[int, dict] = {}
    for rname, matches in KNOCKOUT_ROUNDS:
        for no, hs, as_slot, venue in matches:
            if rname == "R32":
                home = r32_home[no].most_common(1)[0][0]
                away = r32_away[no].most_common(1)[0][0]
            else:
                home = teams_at[hs]
                away = teams_at[as_slot]

            hf = home == venue
            af = away == venue
            if af and not hf:
                p_home = 1.0 - ko_win_prob(model, away, home, True)
            else:
                p_home = ko_win_prob(model, home, away, hf)

            winner = home if p_home >= 0.5 else away
            bracket[no] = {
                "round": rname, "home": home, "away": away,
                "p_home": p_home, "winner": winner,
                "win_p": p_home if p_home >= 0.5 else 1.0 - p_home,
                "venue": venue,
            }
            teams_at[f"W{no}"] = winner

    return bracket, teams_at["W104"], meet


def meeting_breakdown(meet: dict, n_sims: int, a: str, b: str):
    """回傳 (總相遇機率, {round_name: 機率})。"""
    pair = frozenset((a, b))
    by_round = {rn: meet[rn].get(pair, 0) / n_sims for rn in ROUND_ORDER}
    return sum(by_round.values()), by_round


def likely_finals(meet: dict, n_sims: int, k: int = 8):
    """最可能的決賽對戰 Top k：[(teamA, teamB, prob), ...]（依機率排序）。"""
    out = []
    for pair, c in meet["F"].most_common(k):
        a, b = sorted(pair)
        out.append((a, b, c / n_sims))
    return out
