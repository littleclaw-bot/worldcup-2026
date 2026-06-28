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
    "Congo DR": "DR Congo",
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


# 球員中文譯名：日韓用實際漢字，其餘採台灣慣用譯名（找不到的就回原文英文）。
# 名單隨進球者增加，新進球的陌生球員會先顯示英文，再陸續補。
PLAYER_ZH = {
    "Lionel Messi": "梅西",
    "Erling Haaland": "哈蘭德",
    "Kylian Mbappé": "姆巴佩",
    "Ousmane Dembélé": "登貝萊",
    "Brian Brobbey": "布羅貝",
    "Deniz Undav": "溫達夫",
    "Elijah Just": "賈斯特",
    "Ismael Saibari": "賽巴里",
    "Ismaïla Sarr": "薩爾",
    "Johan Manzambi": "曼贊比",
    "Jonathan David": "大衛",
    "Matheus Cunha": "庫尼亞",
    "Vinícius Júnior": "維尼修斯",
    "Anthony Elanga": "埃蘭加",
    "Cody Gakpo": "加克波",
    "Crysencio Summerville": "桑默維爾",
    "Cyle Larin": "拉林",
    "Daniel Muñoz": "穆尼奧斯",
    "Ermin Mahmic": "馬赫米奇",
    "Folarin Balogun": "巴洛根",
    "Julián Quiñones": "基尼奧內斯",
    "Mikel Oyarzabal": "奧亞薩瓦爾",
    "Nicolas Pépé": "佩佩",
    "Pape Gueye": "蓋耶",
    "Ramin Rezaeian": "雷扎伊安",
    "Rubén Vargas": "巴爾加斯",
    "Yasin Ayari": "阿亞里",
    "Abdulelah Al Amri": "阿姆里",
    "Achraf Hakimi": "哈基米",
    "Agustín Canobbio": "卡諾比奧",
    "Alexander Isak": "伊薩克",
    "Alexis Saelemaekers": "薩勒馬克斯",
    "Ali Olwan": "奧爾萬",
    "Amad Diallo": "阿馬德·迪亞洛",
    "Amine Gouiri": "古伊里",
    "Ante Budimir": "布迪米爾",
    "Arda Güler": "居萊爾",
    "Auston Trusty": "特拉斯蒂",
    "Ayase Ueda": "上田綺世",
    "Baris Alper Yilmaz": "耶爾馬茲",
    "Bradley Barcola": "巴爾科拉",
    "Caleb Yirenkyi": "伊倫基",
    "Connor Metcalfe": "梅特卡夫",
    "Cristiano Ronaldo": "C羅",
    "Daichi Kamada": "鎌田大地",
    "Daizen Maeda": "前田大然",
    "Emam Ashour": "阿舒爾",
    "Felix Nmecha": "恩梅查",
    "Franck Kessie": "凱西",
    "Gessime Yassine": "亞辛",
    "Giovanni Reyna": "雷納",
    "Gonzalo Plata": "普拉塔",
    "Habib Diarra": "迪亞拉",
    "Hassan Al Haydos": "海多斯",
    "Hwang In-Beom": "黃仁範",
    "Hélio Varela": "瓦雷拉",
    "Ibrahim Mbaye": "姆巴耶",
    "Iliman Ndiaye": "恩迪亞耶",
    "Jamal Musiala": "穆夏拉",
    "John McGinn": "麥金",
    "Jude Bellingham": "貝林漢",
    "Junya Ito": "伊東純也",
    "Kaan Ayhan": "艾汗",
    "Kai Havertz": "哈弗茨",
    "Keito Nakamura": "中村敬斗",
    "Kerim Alajbegovic": "阿拉伊貝戈維奇",
    "Kevin De Bruyne": "德布勞內",
    "Lamine Yamal": "亞馬爾",
    "Leandro Trossard": "特羅薩德",
    "Leroy Sané": "薩內",
    "Livano Comenencia": "科梅嫩西亞",
    "Luis Díaz": "路易斯·迪亞斯",
    "Luis Romo": "羅莫",
    "Marcus Pedersen": "佩德森",
    "Marcus Rashford": "拉什福德",
    "Martin Baturina": "巴圖里納",
    "Mateo Chávez": "查韋斯",
    "Mattias Svanberg": "斯萬貝里",
    "Matías Galarza": "加拉薩",
    "Mauricio": "毛里西奧",
    "Maxi Araújo": "阿勞霍",
    "Michal Sadílek": "薩迪萊克",
    "Mohamed Salah": "薩拉赫",
    "Nestory Irankunda": "伊蘭昆達",
    "Nilson Angulo": "安古洛",
    "Nizar Al Rashdan": "拉什丹",
    "Oh Hyeon-Gyu": "吳賢奎",
    "Petar Musa": "穆薩",
    "Petar Sucic": "蘇契奇",
    "Promise David": "普羅米斯·大衛",
    "Rafael Leão": "拉斐爾·萊昂",
    "Romano Schmid": "施密德",
    "Sebastian Berhalter": "貝哈爾特",
    "Soufiane Rahimi": "拉希米",
    "Thapelo Maseko": "馬塞科",
    "Thelo Aasgaard": "奧斯加德",
    "Viktor Gyökeres": "約克雷斯",
    "Wilson Isidor": "伊西多",
    "Álex Baena": "巴埃納",
    "Álvaro Fidalgo": "菲達爾戈",
    "Harry Kane": "凱恩",
    "Abbosbek Fayzullaev": "法伊祖拉耶夫",
    "Alex Freeman": "弗里曼",
    "Aymen Hussein": "侯賽因",
    "Breel Embolo": "恩博洛",
    "Derrick Luckassen": "盧卡森",
    "Désiré Doué": "杜埃",
    "Eldor Shomurodov": "紹穆羅多夫",
    "Finn Surman": "蘇爾曼",
    "Granit Xhaka": "扎卡",
    "Hazem Mastouri": "馬斯圖里",
    "Jaminton Campaz": "坎帕斯",
    "Jan Paul van Hecke": "範赫克",
    "Jovo Lukic": "盧基奇",
    "João Neves": "內維斯",
    "Kevin Pina": "皮納",
    "Ladislav Krejcí": "克雷伊奇",
    "Leo Østigård": "厄斯蒂高",
    "Mahmoud Saber": "薩貝爾",
    "Marko Arnautovic": "阿瑙托維奇",
    "Mohammad Mohebbi": "莫赫比",
    "Mostafa Zico": "齊科",
    "Nadhir Benbouali": "本布阿利",
    "Nathan Saliba": "薩利巴",
    "Nathaniel Brown": "布朗",
    "Nico Schlotterbeck": "施洛特貝克",
    "Nikola Vlasic": "弗拉西奇",
    "Nuno Mendes": "努諾·門德斯",
    "Omar Rekik": "雷基克",
    "Raúl Jiménez": "希門尼斯",
    "Romelu Lukaku": "盧卡庫",
    "Teboho Mokoena": "莫科埃納",
    "Trezeguet": "特雷澤蓋",
    "Virgil van Dijk": "范戴克",
    "Yoane Wissa": "維薩",
}


def player_zh(name: str) -> str:
    """球員中文譯名；查不到回原文英文."""
    return PLAYER_ZH.get(name, name)


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
                # ESPN 把進球拆成多種 type（Goal / Goal - Header /
                # Penalty - Scored / 自由球…），但進球播報文字一律以
                # "Goal!" 開頭——用它判定才不會漏頭槌、點球等。
                txt = e.get("text", "")
                if not txt.startswith("Goal!"):
                    continue
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
