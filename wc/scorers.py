"""ESPN 個人數據 overlay：逐場撈當屆世足的進球紀錄，組出金靴＋助攻戰況.

martj42 的 results.csv 只有每場比分、沒有進球者，所以個人榜得另接資料源。
ESPN 非官方 summary API 的 keyEvents 含每顆進球的描述文字與 participants：

    text:         "Goal! Egypt 1, Australia 0. Emam Ashour (Egypt) header ...
                   Assisted by Karim Hafez with a cross."
    participants: [進球者, 助攻者]（無助攻時只有 1 個）

進球者/助攻者一律取 participants（結構化、拼字正確），text 只用來判定「這是進球」、
「有沒有助攻」、以及抓隊名。text 的名字拼法會漏重音、連字號（Trézéguet 寫成
Trezeguet），故僅在 participants 缺漏時當備援。

    scoreboard: .../soccer/fifa.world/scoreboard?dates=YYYYMMDD   → 當日 event id
    summary:    .../soccer/fifa.world/summary?event=ID            → keyEvents

進球含 12 碼 PK、排除烏龍球；PK 大戰的球 ESPN 不給 "Goal!" 文字，天然不會被算進來
（與 FIFA 官方統計一致）。助攻只認 text 裡明寫 "Assisted by" 的那顆。

任何失敗都回空 DataFrame，讓上層自己處理（絕不掛掉 app）。非官方 endpoint
將來可能變/壞 → 全程 try/except。
"""
from __future__ import annotations

import json
import re
import unicodedata
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


def _fold(s: str) -> str:
    """去重音/標點 + 小寫，用來容錯比對球員/隊名.

    ESPN 同一個人在 text 與 participants/roster 裡拼法會不一致（'Trézéguet' vs
    'Trezeguet'、'Al-Amri' vs 'Al Amri'），折疊後才對得上。
    """
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )
    s = re.sub(r"[-'`.]", " ", s)
    return re.sub(r"\s+", " ", s).lower().strip()


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
    "Marcus Holmgren Pedersen": "佩德森",
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
    # 以下為助攻榜帶進來的球員（本屆有助攻、未必有進球）
    "Michael Olise": "奧利斯",
    "Brahim Díaz": "布拉希姆·迪亞斯",
    "Bruno Guimarães": "布魯諾·吉馬良斯",
    "Martin Ødegaard": "奧德高",
    "Roberto Alvarado": "阿爾瓦拉多",
    "Florian Wirtz": "維爾茨",
    "Bukayo Saka": "薩卡",
    "Andreas Schjelderup": "謝爾德魯普",
    "Anthony Gordon": "戈登",
    "Julio Enciso": "恩西索",
    "Joshua Kimmich": "基米希",
    "Ryan Gravenberch": "格拉文貝赫",
    "Hannibal Mejbri": "梅布里",
    "Chris Wood": "伍德",
    "Denzel Dumfries": "鄧弗里斯",
    "Patrick Berg": "貝里",
    "Hans Vanaken": "范納肯",
    "Nicolas Raskin": "拉斯金",
    "Houssem Aouar": "奧亞爾",
    "Marc Cucurella": "庫庫雷利亞",
    "Alexis Mac Allister": "麥卡利斯特",
    "Rodrigo De Paul": "德保羅",
    "Lautaro Martínez": "勞塔羅·馬丁內斯",
    "Lisandro Martínez": "利桑德羅·馬丁內斯",
    "Julián Álvarez": "胡利安·阿爾瓦雷斯",
    "Adrien Rabiot": "拉比奧",
    "Aurélien Tchouaméni": "楚阿梅尼",
    "Bruno Fernandes": "布魯諾·費南德斯",
    "João Cancelo": "坎塞洛",
    "Pedro Neto": "佩德羅·內托",
    "Aymeric Laporte": "拉波爾特",
    "Dani Olmo": "奧爾莫",
    "Ferran Torres": "費蘭·托雷斯",
    "Marcos Llorente": "略倫特",
    "Nico González": "尼科·岡薩雷斯",
    "Declan Rice": "賴斯",
    "Elliot Anderson": "安德森",
    "Christian Pulisic": "普利西奇",
    "Malik Tillman": "蒂爾曼",
    "David Alaba": "阿拉巴",
    "Konrad Laimer": "萊默爾",
    "Xaver Schlager": "施拉格",
    "Michael Gregoritsch": "格雷戈里奇",
    "Nadiem Amiri": "阿米里",
    "Gabriel Magalhães": "加布里埃爾",
    "Lucas Paquetá": "帕奎塔",
    "Memphis Depay": "德佩",
    "Tijjani Reijnders": "雷因德斯",
    "Ibrahim Sangaré": "桑加雷",
    "Luka Modric": "莫德里奇",
    "Ivan Perisic": "佩里西奇",
    "Mateo Kovacic": "科瓦契奇",
    "Riyad Mahrez": "馬赫雷斯",
    "Sadio Mané": "馬內",
    "Nicolas Jackson": "傑克森",
    "Lamine Camara": "卡馬拉",
    "Moussa Niakhaté": "尼亞卡特",
    "Abdoulaye Seck": "塞克",
    "Luis Suárez": "蘇亞雷斯",
    "Juan Fernando Quintero": "金特羅",
    "Cucho Hernández": "庫喬·埃爾南德斯",
    "Takefusa Kubo": "久保建英",
    "Ritsu Doan": "堂安律",
    "Kou Itakura": "板倉滉",
    "Kaishu Sano": "佐野海舟",
    "Koki Ogawa": "小川航基",
    "Lee Kang-In": "李康仁",
    "Timothy Castagne": "卡斯塔涅",
    "Thomas Meunier": "穆尼耶",
    "Charles De Ketelaere": "德凱特拉雷",
    "Chancel Mbemba": "姆本巴",
    "Wilfried Singo": "辛戈",
    "Ernest Nuamah": "努阿馬",
    "Stephen Eustáquio": "尤斯塔基奧",
    "Sidny Lopes Cabral": "卡布拉爾",
    "Karim Hafez": "哈菲茲",
    "Orkun Kökçü": "科克曲",
    "Ricardo Rodríguez": "里卡多·羅德里格斯",
    "Lucas Bergvall": "貝里瓦爾",
    "Sead Kolasinac": "科拉希納茨",
    "Vladimír Coufal": "庫法爾",
    "Gonzalo Montiel": "蒙鐵爾",
    "Facundo Medina": "梅迪納",
    "Jorge Sánchez": "桑切斯",
    "Érik Lira": "利拉",
    "José Manuel López": "洛佩斯",
    "Kevin Rodríguez": "凱文·羅德里格斯",
    "Pedro Vite": "維特",
    "Yan Diomande": "迪奧曼德",
    "Chemsdine Talbi": "塔爾比",
    "Chadi Riad": "里亞德",
    "Arthur Masuaku": "馬蘇阿庫",
    "Meschack Elia": "埃利亞",
    "Tim Payne": "佩恩",
    "Brandon Thomas-Asante": "托馬斯-阿桑特",
    "Tshepang Moremi": "莫雷米",
    "Ryan Mendes": "門德斯",
    "Yannick Semedo": "塞梅多",
    "Edmílson Junior": "埃德米爾森",
    "Haissem Hassan": "哈桑",
    "Mohamed Hany": "哈尼",
    "Ehsan Haddad": "哈達德",
    "Akmal Mozgovoy": "莫茲戈沃伊",
    "Alexandr Sojka": "索伊卡",
    "Dennis Hadzikadunic": "哈吉卡杜尼奇",
    "Ivan Basic": "巴西奇",
    "Jean-Kévin Duverne": "杜維爾內",
    "Josip Stanisic": "斯坦尼西奇",
    "Paul Okon-Engstler": "奧康-恩斯特勒",
    "Gustavo Puerta": "普爾塔",
    "David Møller Wolfe": "沃爾夫",
    "Rayan": "拉揚",
    # 補齊本屆其餘進球/助攻者
    "Neymar": "內馬爾",
    "Casemiro": "卡塞米羅",
    "Gabriel Martinelli": "馬丁內利",
    "Enzo Fernández": "恩佐·費南德斯",
    "Cristian Romero": "羅梅羅",
    "Giovani Lo Celso": "洛塞爾索",
    "Mikel Merino": "梅里諾",
    "Fabián Ruiz": "法比安·魯伊斯",
    "Pedro Porro": "波羅",
    "Gonçalo Ramos": "拉莫斯",
    "Youri Tielemans": "蒂勒曼斯",
    "Marcel Sabitzer": "薩比策",
    "Sasa Kalajdzic": "卡拉伊季奇",
    "Dan Ndoye": "恩多耶",
    "Antonio Nusa": "努薩",
    "Azzedine Ounahi": "奧納希",
    "Issa Diop": "迪奧普",
    "Jhon Arias": "阿里亞斯",
    "Brian Cipenga": "西彭加",
    "Fiston Mayele": "馬耶萊",
    "Deroy Duarte": "杜阿爾特",
    "Rafik Belghali": "貝爾加利",
    "Mousa Al-Tamari": "塔馬里",
    "Amir Al-Ammari": "阿馬里",
    "Noor Al-Rawabdeh": "拉瓦布德",
    "Marawan Attia": "阿蒂亞",
    "Yasser Ibrahim": "易卜拉欣",
}


_PLAYER_ZH_FOLDED = {_fold(k): v for k, v in PLAYER_ZH.items()}


def player_zh(name: str) -> str:
    """球員中文譯名；查不到回原文英文.

    先精確比對，再用 _fold 容錯（重音/連字號拼法不同也查得到）。
    """
    if name in PLAYER_ZH:
        return PLAYER_ZH[name]
    return _PLAYER_ZH_FOLDED.get(_fold(name), name)


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
    """回 DataFrame[player, team, goals, penalties, assists, jersey]，依進球數遞減.

    收錄所有「有進球或有助攻」的球員——只助攻沒進球的人 goals=0 也會在表裡，
    上層要純射手榜就自己 filter goals > 0。

    掃開幕日到今天（UTC）的所有場次。失敗或無資料回空 DataFrame
    （欄位齊全，上層可安全 .empty 判斷）。背號取自同一份 summary 的
    rosters（不另發請求），對不到名字就留空字串。
    """
    cols = ["player", "team", "goals", "penalties", "assists", "jersey"]
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
    assists: dict[str, int] = {}
    team_of: dict[str, str] = {}
    # 背號：主索引以 (隊伍, 球員名) 去重音為 key，避免不同隊同名球員撞號；
    # 後備索引只用球員名（解隊名格式不一致，如 Bosnia 的對戰文字 vs roster）。
    jersey_team: dict[tuple[str, str], str] = {}
    jersey_name: dict[str, str] = {}

    for eid in _event_ids(days):
        try:
            raw = urlopen(f"{SUMMARY}?event={eid}", timeout=15).read()
            summ = json.loads(raw)
        except Exception:
            continue
        for r in summ.get("rosters", []):
            try:
                team = _norm((r.get("team") or {}).get("displayName", ""))
                for p in r.get("roster", []):
                    name = (p.get("athlete") or {}).get("displayName")
                    num = p.get("jersey")
                    if name and num:
                        jersey_team[(_fold(team), _fold(name))] = str(num)
                        jersey_name[_fold(name)] = str(num)
            except (KeyError, AttributeError, TypeError):
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
                team = _norm(m.group(2).strip())
                parts = e.get("participants") or []
                names = [
                    (p.get("athlete") or {}).get("displayName") for p in parts
                ]
                # participants[0]=進球者、[1]=助攻者（沒助攻時只有一個）。
                player = names[0] if names and names[0] else m.group(1).strip()
                goals[player] = goals.get(player, 0) + 1
                team_of[player] = team
                if "penalt" in txt.lower():
                    pens[player] = pens.get(player, 0) + 1
                # 助攻以播報文字為準（PK、單刀等無人助攻的球不會有這句）。
                if "Assisted by" in txt and len(names) > 1 and names[1]:
                    helper = names[1]
                    assists[helper] = assists.get(helper, 0) + 1
                    team_of.setdefault(helper, team)  # 助攻者必為同隊隊友
            except (KeyError, AttributeError, TypeError, IndexError):
                continue

    if not goals and not assists:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(
        [
            {
                "player": p,
                "team": team_of[p],
                "goals": goals.get(p, 0),
                "penalties": pens.get(p, 0),
                "assists": assists.get(p, 0),
                "jersey": jersey_team.get(
                    (_fold(team_of[p]), _fold(p)), jersey_name.get(_fold(p), "")
                ),
            }
            for p in dict.fromkeys([*goals, *assists])  # 進球者優先，助攻者接後
        ]
    )
    return df.sort_values(["goals", "player"], ascending=[False, True]).reset_index(
        drop=True
    )
