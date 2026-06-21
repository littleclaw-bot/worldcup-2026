"""ESPN 即時比分 overlay：補 martj42 還沒更新的當屆世足 FT 結果.

ESPN 非官方 scoreboard API（乾淨 JSON、無 key、FT 即時）:
    https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=YYYYMMDD
只有當屆/指定日,無歷史 → 不取代 martj42,只補它 delay 的那幾小時。
任何失敗都回空 dict,讓上層 fallback 回純 martj42（絕不掛掉 app）。
"""
from __future__ import annotations

import json
from urllib.request import urlopen

import pandas as pd

ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
)
# ESPN 隊名 → martj42/我們的隊名（只有這 4 個不一致,其餘 44 隊相同）
NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
}


def _norm(name: str) -> str:
    return NAME_MAP.get(name, name)


def fetch_wc_live_results(days_back: int = 3) -> dict[tuple[str, str], tuple[int, int]]:
    """回 {(home, away): (home_score, away_score)},只收已完賽(FT)場次.

    查 now 起往前 days_back 天（UTC）——martj42 沒補的都是近幾天的。
    """
    out: dict[tuple[str, str], tuple[int, int]] = {}
    try:
        now = pd.Timestamp.utcnow()
    except Exception:
        return out
    for i in range(days_back + 1):
        ymd = (now - pd.Timedelta(days=i)).strftime("%Y%m%d")
        try:
            raw = urlopen(f"{ESPN_URL}?dates={ymd}", timeout=15).read()
            data = json.loads(raw)
        except Exception:
            continue
        for e in data.get("events", []):
            try:
                comp = e["competitions"][0]
                if comp.get("status", e.get("status", {})).get(
                    "type", {}
                ).get("name") != "STATUS_FULL_TIME":
                    continue
                sides = {x["homeAway"]: x for x in comp["competitors"]}
                h, a = sides["home"], sides["away"]
                out[(_norm(h["team"]["displayName"]),
                     _norm(a["team"]["displayName"]))] = (
                    int(h["score"]), int(a["score"])
                )
            except (KeyError, IndexError, ValueError, TypeError):
                continue
    return out


def apply_live_scores(
    df: pd.DataFrame, live: dict[tuple[str, str], tuple[int, int]]
) -> pd.DataFrame:
    """把 live FT 比分補進 df 裡『還沒有比分』的世足場次（martj42 尚未更新者）.

    以隊伍集合比對（容忍 ESPN 與 martj42 主客順序不同的中立場）。
    """
    if not live:
        return df
    df = df.copy()
    mask = (
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= "2026-06-01")
        & df["home_score"].isna()
    )
    for idx in df[mask].index:
        h, a = df.at[idx, "home_team"], df.at[idx, "away_team"]
        if (h, a) in live:
            hs, as_ = live[(h, a)]
        elif (a, h) in live:
            as_, hs = live[(a, h)]
        else:
            continue
        df.at[idx, "home_score"] = hs
        df.at[idx, "away_score"] = as_
    return df
