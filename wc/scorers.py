"""ESPN 射手榜 overlay：逐場撈當屆世足的進球紀錄，組出個人金靴戰況.

martj42 的 results.csv 只有每場比分、沒有進球者，所以個人射手榜得另接資料源。
ESPN 非官方 summary API 的 keyEvents 含每顆進球的描述文字（進球者寫在 text，
athletesInvolved 是空的），逐場解析即可累計每位球員的進球數（含 PK，排除烏龍球）。

    scoreboard: .../soccer/fifa.world/scoreboard?dates=YYYYMMDD   → 當日 event id
    summary:    .../soccer/fifa.world/summary?event=ID            → keyEvents

任何失敗都回空 DataFrame，讓上層自己處理（絕不掛掉 app）。非官方 endpoint
將來可能變/壞 → 全程 try/except。
"""
from __future__ import annotations

import json
import re
from urllib.request import urlopen

import pandas as pd

SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
)
SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"
)
SEASON_START = "2026-06-11"  # 2026 世足開幕

# ESPN 進球文字裡的隊名 → 我們/martj42 的隊名（只有這 7 個不一致）
NAME_MAP = {
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}

# "Goal! Argentina 2, Austria 0. Lionel Messi (Argentina) left footed shot ..."
# → 抓 ". 進球者 (隊伍)"。隊名/人名都不含 '(' 或句點，故樣式穩定。
_GOAL_RE = re.compile(r"\.\s+([^().]+?)\s+\(([^)]+)\)")


def _norm(team: str) -> str:
    return NAME_MAP.get(team, team)


def _event_ids(days: list[str]) -> list[str]:
    """掃這幾天的 scoreboard，回所有 event id（去重）."""
    ids: list[str] = []
    for ymd in days:
        try:
            raw = urlopen(f"{SCOREBOARD}?dates={ymd}", timeout=15).read()
            data = json.loads(raw)
        except Exception:
            continue
        for e in data.get("events", []):
            eid = e.get("id")
            if eid:
                ids.append(eid)
    return list(dict.fromkeys(ids))


def fetch_wc_scorers() -> pd.DataFrame:
    """回 DataFrame[player, team, goals, penalties]，依進球數遞減.

    掃開幕日到今天（UTC）的所有場次。失敗或無資料回空 DataFrame
    （欄位齊全，上層可安全 .empty 判斷）。
    """
    cols = ["player", "team", "goals", "penalties"]
    try:
        now = pd.Timestamp.utcnow()
        start = pd.Timestamp(SEASON_START, tz="UTC")
    except Exception:
        return pd.DataFrame(columns=cols)
    if now < start:
        return pd.DataFrame(columns=cols)

    n_days = (now.normalize() - start.normalize()).days + 1
    days = [(start + pd.Timedelta(days=i)).strftime("%Y%m%d") for i in range(n_days)]

    goals: dict[str, int] = {}
    pens: dict[str, int] = {}
    team_of: dict[str, str] = {}

    for eid in _event_ids(days):
        try:
            raw = urlopen(f"{SUMMARY}?event={eid}", timeout=15).read()
            summ = json.loads(raw)
        except Exception:
            continue
        for e in summ.get("keyEvents", []):
            try:
                if (e.get("type") or {}).get("text") != "Goal":
                    continue
                txt = e.get("text", "")
                if "own goal" in txt.lower():  # 烏龍球不算進球者
                    continue
                m = _GOAL_RE.search(txt)
                if not m:
                    continue
                player = m.group(1).strip()
                team = _norm(m.group(2).strip())
                goals[player] = goals.get(player, 0) + 1
                team_of[player] = team
                if "penalt" in txt.lower():
                    pens[player] = pens.get(player, 0) + 1
            except (KeyError, AttributeError, TypeError):
                continue

    if not goals:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(
        [
            {
                "player": p,
                "team": team_of[p],
                "goals": g,
                "penalties": pens.get(p, 0),
            }
            for p, g in goals.items()
        ]
    )
    return df.sort_values(["goals", "player"], ascending=[False, True]).reset_index(
        drop=True
    )
