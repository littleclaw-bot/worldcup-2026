"""列出追蹤球隊已踢完的小組賽 + 比分 + 行事曆標題（給 Claude 更新行事曆用）.

行事曆標題慣例：
    未踢   "⚽[組] 主隊 vs 客隊"
    已踢   "⚽[組] 主隊 X:Y 客隊"

用法：
    python calendar_scores.py            # 用本地 results.csv
    python calendar_scores.py --refresh  # 先抓最新比分
輸出 JSON：每場 {group, old_title, new_title, finished}，
Claude 據此 search_events 找到事件、update_event 改標題。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wc import data

# 追蹤的 9 隊（與行事曆事件對應）
TRACKED = {
    "Argentina", "Spain", "Brazil", "England", "Japan",
    "South Korea", "Portugal", "France", "Germany",
}
ZH = {
    "Argentina": "阿根廷", "Spain": "西班牙", "Brazil": "巴西",
    "England": "英格蘭", "Japan": "日本", "South Korea": "南韓",
    "Portugal": "葡萄牙", "France": "法國", "Germany": "德國",
    "Mexico": "墨西哥", "South Africa": "南非", "Czech Republic": "捷克",
    "Canada": "加拿大", "Bosnia and Herzegovina": "波赫", "Qatar": "卡達",
    "Switzerland": "瑞士", "Morocco": "摩洛哥", "Haiti": "海地",
    "Scotland": "蘇格蘭", "United States": "美國", "Paraguay": "巴拉圭",
    "Australia": "澳洲", "Turkey": "土耳其", "Curaçao": "古拉索",
    "Ivory Coast": "象牙海岸", "Ecuador": "厄瓜多", "Netherlands": "荷蘭",
    "Sweden": "瑞典", "Tunisia": "突尼西亞", "Belgium": "比利時",
    "Egypt": "埃及", "Iran": "伊朗", "New Zealand": "紐西蘭",
    "Cape Verde": "維德角", "Saudi Arabia": "沙烏地", "Uruguay": "烏拉圭",
    "Senegal": "塞內加爾", "Iraq": "伊拉克", "Norway": "挪威",
    "Algeria": "阿爾及利亞", "Austria": "奧地利", "Jordan": "約旦",
    "DR Congo": "民主剛果", "Uzbekistan": "烏茲別克", "Colombia": "哥倫比亞",
    "Croatia": "克羅埃西亞", "Ghana": "迦納", "Panama": "巴拿馬",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh:
        data.refresh_results()

    f = data.wc_fixtures(data.load_results())
    rows = []
    for r in f.itertuples():
        if r.home_team not in TRACKED and r.away_team not in TRACKED:
            continue
        g, h, a = r.group, ZH[r.home_team], ZH[r.away_team]
        old = f"⚽[{g}] {h} vs {a}"
        if pd.notna(r.home_score):
            new = f"⚽[{g}] {h} {int(r.home_score)}:{int(r.away_score)} {a}"
            rows.append({"group": g, "old_title": old, "new_title": new,
                         "finished": True})
        else:
            rows.append({"group": g, "old_title": old, "new_title": old,
                         "finished": False})

    done = [r for r in rows if r["finished"]]
    print(f"# 追蹤 {len(rows)} 場，已踢完 {len(done)} 場\n")
    for r in done:
        print(r["new_title"])
    Path("out").mkdir(exist_ok=True)
    Path("out/calendar_scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n# 完整清單已存 out/calendar_scores.json")


if __name__ == "__main__":
    main()
