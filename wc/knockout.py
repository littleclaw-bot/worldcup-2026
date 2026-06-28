"""ESPN 淘汰賽賽程：抓 R32→決賽的真實對戰（隊伍/台灣時間/比分/球場）.

小組賽的對戰是固定的（openfootball 排好），但淘汰賽要等小組名次出爐才知道誰碰誰。
與其自己重算 FIFA 同分規則（H2H 等規則繁瑣易錯），直接抓 ESPN——它的賽程
已把淘汰賽對戰排好（含尚未確定的「Round of 32 Winner」佔位），且帶 season.slug
標明回合。任何失敗回空 DataFrame，上層 fallback（絕不掛掉 app）。

    scoreboard?dates=YYYYMMDD → events[].season.slug = round-of-32 / round-of-16 /
    quarterfinals / semifinals / third-place / final
"""
from __future__ import annotations

import json
from urllib.request import urlopen

import pandas as pd

from .live import ESPN_URL, _norm  # 共用 scoreboard endpoint 與隊名映射

# season.slug → (顯示順序, 中文回合名)
ROUND_INFO = {
    "round-of-32": (0, "🏆 32 強"),
    "round-of-16": (1, "16 強"),
    "quarterfinals": (2, "8 強"),
    "semifinals": (3, "4 強"),
    "third-place": (4, "季軍戰"),
    "final": (5, "🏅 決賽"),
}
KO_START = "2026-06-28"  # R32 開打日
KO_END = "2026-07-20"    # 決賽（7/19）後一天，含時差緩衝

# ESPN 球場所在國 → 我們的隊名（判主辦國主場優勢用，只有這三個主辦國）
HOST_COUNTRY = {"USA": "United States", "Mexico": "Mexico", "Canada": "Canada"}


def fetch_knockout_fixtures() -> pd.DataFrame:
    """回 DataFrame，依開球時間排序.

    欄位：round_slug, round_order, round_zh, kickoff_tw(tz-aware),
    home, away, home_score, away_score, status, venue。
    home/away 尚未確定時為 ESPN 佔位字串（如 "Round of 32 1 Winner"）。
    失敗或無賽事回空 DataFrame（欄位齊全）。
    """
    cols = ["round_slug", "round_order", "round_zh", "kickoff_tw",
            "home", "away", "home_score", "away_score", "status",
            "venue", "venue_country"]
    rows: list[dict] = []
    try:
        days = pd.date_range(KO_START, KO_END, freq="D")
    except Exception:
        return pd.DataFrame(columns=cols)

    for ymd in days.strftime("%Y%m%d"):
        try:
            raw = urlopen(f"{ESPN_URL}?dates={ymd}", timeout=15).read()
            data = json.loads(raw)
        except Exception:
            continue
        for e in data.get("events", []):
            try:
                slug = (e.get("season") or {}).get("slug", "")
                if slug not in ROUND_INFO:
                    continue  # 略過小組賽
                comp = e["competitions"][0]
                sides = {x["homeAway"]: x for x in comp["competitors"]}
                h, a = sides["home"], sides["away"]
                status = comp.get("status", {}).get("type", {}).get("name", "")
                ft = status == "STATUS_FULL_TIME"
                ven = comp.get("venue", {}) or {}
                order, zh = ROUND_INFO[slug]
                rows.append({
                    "round_slug": slug,
                    "round_order": order,
                    "round_zh": zh,
                    "kickoff_tw": pd.to_datetime(e["date"], utc=True).tz_convert(
                        "Asia/Taipei"
                    ),
                    "home": _norm(h["team"]["displayName"]),
                    "away": _norm(a["team"]["displayName"]),
                    "home_score": int(h["score"]) if ft and h.get("score") not in
                    (None, "") else None,
                    "away_score": int(a["score"]) if ft and a.get("score") not in
                    (None, "") else None,
                    "status": status,
                    "venue": ven.get("fullName", ""),
                    "venue_country": HOST_COUNTRY.get(
                        (ven.get("address", {}) or {}).get("country", ""), ""
                    ),
                })
            except (KeyError, IndexError, ValueError, TypeError):
                continue

    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    return df.sort_values(["round_order", "kickoff_tw"]).reset_index(drop=True)
