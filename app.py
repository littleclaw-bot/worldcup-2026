"""WC2026 預測儀表板.

啟動：
    streamlit run app.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wc import (
    data, model as dcmodel, odds as oddsmod, simulate, elo as elomod,
    bracket as bracketmod, live as livemod, scorers as scorersmod,
)
from wc.teamguide import TEAM_GUIDE, STORYLINES, MARKET_VALUE, BIRTH

st.set_page_config(page_title="WC2026 預測", page_icon="⚽", layout="wide")

ZH = {
    "Mexico": "墨西哥", "South Africa": "南非", "South Korea": "南韓",
    "Czech Republic": "捷克", "Canada": "加拿大",
    "Bosnia and Herzegovina": "波赫", "Qatar": "卡達", "Switzerland": "瑞士",
    "Brazil": "巴西", "Morocco": "摩洛哥", "Haiti": "海地", "Scotland": "蘇格蘭",
    "United States": "美國", "Paraguay": "巴拉圭", "Australia": "澳洲",
    "Turkey": "土耳其", "Germany": "德國", "Curaçao": "古拉索",
    "Ivory Coast": "象牙海岸", "Ecuador": "厄瓜多", "Netherlands": "荷蘭",
    "Japan": "日本", "Sweden": "瑞典", "Tunisia": "突尼西亞",
    "Belgium": "比利時", "Egypt": "埃及", "Iran": "伊朗",
    "New Zealand": "紐西蘭", "Spain": "西班牙", "Cape Verde": "維德角",
    "Saudi Arabia": "沙烏地", "Uruguay": "烏拉圭", "France": "法國",
    "Senegal": "塞內加爾", "Iraq": "伊拉克", "Norway": "挪威",
    "Argentina": "阿根廷", "Algeria": "阿爾及利亞", "Austria": "奧地利",
    "Jordan": "約旦", "Portugal": "葡萄牙", "DR Congo": "民主剛果",
    "Uzbekistan": "烏茲別克", "Colombia": "哥倫比亞", "England": "英格蘭",
    "Croatia": "克羅埃西亞", "Ghana": "迦納", "Panama": "巴拿馬",
}


# FIFA 官方三碼（電視轉播用的縮寫）
FIFA = {
    "Mexico": "MEX", "South Africa": "RSA", "South Korea": "KOR",
    "Czech Republic": "CZE", "Canada": "CAN",
    "Bosnia and Herzegovina": "BIH", "Qatar": "QAT", "Switzerland": "SUI",
    "Brazil": "BRA", "Morocco": "MAR", "Haiti": "HAI", "Scotland": "SCO",
    "United States": "USA", "Paraguay": "PAR", "Australia": "AUS",
    "Turkey": "TUR", "Germany": "GER", "Curaçao": "CUW",
    "Ivory Coast": "CIV", "Ecuador": "ECU", "Netherlands": "NED",
    "Japan": "JPN", "Sweden": "SWE", "Tunisia": "TUN", "Belgium": "BEL",
    "Egypt": "EGY", "Iran": "IRN", "New Zealand": "NZL", "Spain": "ESP",
    "Cape Verde": "CPV", "Saudi Arabia": "KSA", "Uruguay": "URU",
    "France": "FRA", "Senegal": "SEN", "Iraq": "IRQ", "Norway": "NOR",
    "Argentina": "ARG", "Algeria": "ALG", "Austria": "AUT", "Jordan": "JOR",
    "Portugal": "POR", "DR Congo": "COD", "Uzbekistan": "UZB",
    "Colombia": "COL", "England": "ENG", "Croatia": "CRO", "Ghana": "GHA",
    "Panama": "PAN",
}


EUR_TWD = 35  # 近似匯率,會浮動


def twd_from_salary(salary: str) -> str:
    """從薪資字串解析第一個 €XXM,換算約略台幣（億/萬）."""
    m = re.search(r"€\s*([\d.]+)\s*M", salary)
    if not m:
        return ""
    twd_m = float(m.group(1)) * EUR_TWD  # 百萬台幣
    if twd_m >= 100:  # ≥ 1 億
        return f"約台幣 {twd_m / 100:.1f} 億"
    return f"約台幣 {twd_m * 100:.0f} 萬"


def age_str(name: str) -> str:
    """依當下日期動態算現齡,回傳『 · 🎂 XX 歲（YYYY）』."""
    bd = BIRTH.get(name)
    if not bd:
        return ""
    y, m, d = map(int, bd.split("-"))
    now = pd.Timestamp.now()
    age = now.year - y - ((now.month, now.day) < (m, d))
    return f" · 🎂 {age} 歲（{y}）"


def tname(t: str) -> str:
    """中文 + 英文 + FIFA 三碼，給表格/標題用."""
    if t not in ZH:
        return t
    return f"{ZH[t]} {t} ({FIFA[t]})"


def zh(t: str) -> str:
    """中文短名，給空間小的地方用."""
    return ZH.get(t, t)


ISO2 = {
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr",
    "Czech Republic": "cz", "Canada": "ca", "Bosnia and Herzegovina": "ba",
    "Qatar": "qa", "Switzerland": "ch", "Brazil": "br", "Morocco": "ma",
    "Haiti": "ht", "Scotland": "gb-sct", "United States": "us",
    "Paraguay": "py", "Australia": "au", "Turkey": "tr", "Germany": "de",
    "Curaçao": "cw", "Ivory Coast": "ci", "Ecuador": "ec",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Spain": "es", "Cape Verde": "cv", "Saudi Arabia": "sa", "Uruguay": "uy",
    "France": "fr", "Senegal": "sn", "Iraq": "iq", "Norway": "no",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "DR Congo": "cd", "Uzbekistan": "uz", "Colombia": "co",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}


def flag_url(t: str) -> str | None:
    """國旗小圖（Windows 不渲染旗子 emoji，用真圖）.

    用 flag-icons 的統一 4:3 版本：各國旗原始長寬比不一
    （瑞士正方形、卡達 28:11），st.ImageColumn 又會按欄寬縮放，
    原比例圖檔在表格裡大小必亂；4:3 統一規格每面旗完全等大。
    """
    if t not in ISO2:
        return None
    return (
        "https://cdn.jsdelivr.net/gh/lipis/flag-icons@7.2.3"
        f"/flags/4x3/{ISO2[t]}.svg"
    )


def localize(s: str) -> str:
    """把字串中的英文隊名換成 旗+中文（長名優先避免部分覆蓋）."""
    for t in sorted(ZH, key=len, reverse=True):
        s = s.replace(t, zh(t))
    return s


def bracket_dot(br: dict, champion: str) -> str:
    """把預測對戰樹轉成 graphviz DOT（左→右,冠軍之路金色高亮）.

    用 HTML 表格讓「隊名靠左、勝率靠右」分兩欄對齊（欄寬固定,中文長短不一
    也不會參差）;fontname 指定 CJK 字型讓瀏覽器渲染得出中文。
    """
    def row(team: str, p: float, win: bool) -> str:
        color = "#137333" if win else "#9aa0a6"
        b0, b1 = ("<B>", "</B>") if win else ("", "")
        return (
            f'<TR><TD ALIGN="LEFT" WIDTH="78">'
            f'<FONT COLOR="{color}">{b0}{zh(team)}{b1}</FONT></TD>'
            f'<TD ALIGN="RIGHT" WIDTH="46">'
            f'<FONT COLOR="{color}">{b0}{p * 100:.0f}%{b1}</FONT></TD></TR>'
        )

    lines = [
        "digraph B {",
        "  rankdir=LR; bgcolor=transparent; ranksep=0.45; nodesep=0.12;",
        '  node [shape=box, style="rounded,filled",'
        ' fontname="Microsoft JhengHei,PingFang TC,sans-serif",'
        ' fontsize=14, color="#d0d0d0", fillcolor="white", margin="0.14,0.08"];',
        '  edge [color="#c4c4c4", arrowsize=0.5];',
    ]
    for no, b in br.items():
        home_win = b["winner"] == b["home"]
        label = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
            + row(b["home"], b["p_home"], home_win)
            + row(b["away"], 1 - b["p_home"], not home_win)
            + "</TABLE>>"
        )
        attrs = [f"label={label}"]
        if b["winner"] == champion:
            attrs += ['color="#e0a800"', "penwidth=2"]
        if no == 104:
            attrs.append('fillcolor="#fff3cd"')
        lines.append(f'  m{no} [{", ".join(attrs)}];')
    for _rn, matches in data.KNOCKOUT_ROUNDS:
        for mno, hs, as_slot, _venue in matches:
            for s in (hs, as_slot):
                if s.startswith("W"):
                    lines.append(f"  m{s[1:]} -> m{mno};")
    lines.append("}")
    return "\n".join(lines)


def _mtimes() -> tuple:
    paths = [data.RESULTS_CSV, oddsmod.ODDS_CSV]
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)


@st.cache_data(ttl=300, show_spinner="抓即時比分 (ESPN) ...")
def get_live(mtimes: tuple) -> tuple:
    """ESPN FT 比分,回 hashable live_key（每 5 分鐘刷新一次）.

    回傳排序後的 ((home, away, hs, as), ...) tuple,可直接當 cache key,
    也能還原成 dict 給 apply_live_scores。失敗回空 tuple → fallback 純 martj42。
    """
    sc = livemod.fetch_wc_live_results()
    return tuple(sorted((h, a, hs, as_) for (h, a), (hs, as_) in sc.items()))


def _live_dict(live_key: tuple) -> dict:
    return {(h, a): (hs, as_) for h, a, hs, as_ in live_key}


@st.cache_data(show_spinner=False)
def get_martj42_missing(mtimes: tuple) -> set:
    """原始 martj42 裡『還沒比分』的世足對戰（隊伍集合）,給即時狀態比對用."""
    raw = data.load_results()
    wc = raw[(raw["tournament"] == "FIFA World Cup") & (raw["date"] >= "2026-06-01")]
    return {
        frozenset((r.home_team, r.away_team))
        for r in wc.itertuples() if pd.isna(r.home_score)
    }


@st.cache_resource(show_spinner="fit 模型中 ...")
def get_model(mtimes: tuple, live_key: tuple):
    df = data.load_results()
    df = livemod.apply_live_scores(df, _live_dict(live_key))  # ESPN 即時補
    train = data.training_matches(df)
    m = dcmodel.fit(train)
    return m, df, len(train)


@st.cache_data(show_spinner="Monte Carlo 模擬中 ...")
def get_sim(mtimes: tuple, n_sims: int, live_key: tuple) -> pd.DataFrame:
    m, df, _ = get_model(mtimes, live_key)
    return simulate.simulate_tournament(m, data.wc_fixtures(df), n_sims=n_sims)


@st.cache_data(show_spinner="計算 Elo ...")
def get_elo(mtimes: tuple, live_key: tuple) -> dict:
    _, df, _ = get_model(mtimes, live_key)
    return elomod.compute_elo(df)


@st.cache_data(show_spinner="推算對戰樹 ...")
def get_bracket(mtimes: tuple, n_sims: int, live_key: tuple):
    m, df, _ = get_model(mtimes, live_key)
    return bracketmod.predicted_bracket(m, data.wc_fixtures(df), n_sims=n_sims)


@st.cache_data(ttl=300, show_spinner="抓射手榜 (ESPN) ...")
def get_scorers() -> pd.DataFrame:
    """ESPN 逐場進球紀錄組出的個人射手榜（cache 5 分鐘）。失敗回空 DataFrame."""
    return scorersmod.fetch_wc_scorers()


# ---------------- sidebar ----------------
st.sidebar.title("⚽ WC2026 預測")

data.ensure_data()  # 雲端首次啟動沒有 data/，自動補齊

# 過期自動更新：每個 session 只檢查一次（開頁時），之後純互動（篩選、
# 換分頁）不再觸發下載，避免無謂地讓模型 cache 失效重 fit。
STALE_HOURS = 6
if not st.session_state.get("_freshness_checked"):
    st.session_state["_freshness_checked"] = True
    _age_h = (pd.Timestamp.now().timestamp()
              - data.RESULTS_CSV.stat().st_mtime) / 3600
    if _age_h > STALE_HOURS:
        try:
            with st.spinner(f"資料已 {_age_h:.0f} 小時未更新，自動抓最新比分 ..."):
                data.refresh_results()
            st.toast("✅ 已自動更新比分資料")
        except Exception as e:
            st.sidebar.warning(f"自動更新失敗（{e}），先用本地資料")

if st.sidebar.button("🔄 立刻抓最新比分"):
    with st.spinner("下載 results.csv ..."):
        data.refresh_results()
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

n_sims = st.sidebar.select_slider("模擬次數", [2000, 5000, 10000, 20000], value=10000)

mt = _mtimes()
live_key = get_live(mt)  # ESPN 即時 FT 比分（cache 5 分鐘）
model, results_df, n_train = get_model(mt, live_key)
fixtures = data.wc_fixtures(results_df)
odds = oddsmod.load_odds()

# 算「ESPN 補了幾場 martj42 還沒有的」→ 給側欄即時狀態
_missing = get_martj42_missing(mt)
_live_only = sum(1 for h, a, _, _ in live_key if frozenset((h, a)) in _missing)

_fetched = (
    pd.Timestamp(data.RESULTS_CSV.stat().st_mtime, unit="s", tz="UTC")
    .tz_convert("Asia/Taipei")
)
st.sidebar.caption(
    f"訓練 {n_train} 場（2018–今）\n\n"
    f"home_adv={model.home_adv:.3f}, rho={model.rho:.4f}\n\n"
    f"資料最後日期：{results_df['date'].max().date()}\n\n"
    f"資料抓取時間：{_fetched:%Y-%m-%d %H:%M}"
)
if _live_only:
    st.sidebar.success(f"🔴 ESPN 即時補了 {_live_only} 場 martj42 還沒更新的比分")
elif live_key:
    st.sidebar.caption(f"🟢 ESPN 即時比分已同步（martj42 也已跟上）")

played = fixtures[fixtures["home_score"].notna()]
pending = fixtures[fixtures["home_score"].isna()]

(tab_match, tab_sched, tab_title, tab_bracket, tab_elo, tab_groups, tab_market,
 tab_history, tab_drill, tab_stars, tab_scorers) = st.tabs(
    ["📅 賽事預測", "🗓️ 賽程表", "🏆 冠軍機率", "🗺️ 對戰樹", "💪 Elo 實力榜",
     "📋 分組形勢", "📊 模型 vs 市場", "📈 歷史走勢", "🔍 單場下鑽", "🌟 球星導覽",
     "🥇 射手榜"]
)

# ---------------- 賽事預測 ----------------
with tab_match:
    days = st.slider("顯示未來幾天", 1, 16, 4)
    horizon = pending[
        pending["date"] <= pending["date"].min() + pd.Timedelta(days=days)
    ]
    odds_key = (
        odds.set_index(["home_team", "away_team"]) if len(odds) else None
    )
    for d, day_matches in horizon.groupby(horizon["date"].dt.date):
        st.subheader(str(d))
        for r in day_matches.itertuples():
            p = model.outcome_probs(r.home_team, r.away_team, not r.neutral)
            c1, c2 = st.columns([2.2, 3])
            with c1:
                hf = "🏟️" if not r.neutral else ""
                st.markdown(
                    f"**[{r.group}] {tname(r.home_team)} vs "
                    f"{tname(r.away_team)}** {hf}"
                )
                mkt_txt = ""
                if odds_key is not None and (r.home_team, r.away_team) in odds_key.index:
                    o = odds_key.loc[(r.home_team, r.away_team)]
                    mkt_txt = (
                        f"市場：{o['mkt_home']:.0%} / {o['mkt_draw']:.0%} / "
                        f"{o['mkt_away']:.0%}（{o['source']}）"
                    )
                st.caption(
                    f"模型：主 {p['home']:.0%} / 和 {p['draw']:.0%} / "
                    f"客 {p['away']:.0%}　{mkt_txt}"
                )
            with c2:
                fig = go.Figure()
                for name, val, color in [
                    (zh(r.home_team), p["home"], "#2563eb"),
                    ("和局", p["draw"], "#9ca3af"),
                    (zh(r.away_team), p["away"], "#dc2626"),
                ]:
                    fig.add_trace(go.Bar(
                        x=[val], y=[""], name=name, orientation="h",
                        marker_color=color, text=f"{name} {val:.0%}",
                        textposition="inside",
                    ))
                fig.update_layout(
                    barmode="stack", height=60, showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False, range=[0, 1]),
                    yaxis=dict(visible=False),
                )
                st.plotly_chart(
                    fig, width="stretch",
                    key=f"bar{r.Index}", config={"displayModeBar": False},
                )

    if len(played):
        st.divider()
        st.subheader("✅ 已賽結果 vs 模型賽前機率")
        rows = []
        for r in played.itertuples():
            p = model.outcome_probs(r.home_team, r.away_team, not r.neutral)
            out = ("home" if r.home_score > r.away_score
                   else "away" if r.home_score < r.away_score else "draw")
            rows.append({
                "日期": r.date.date(), "組": r.group,
                "比賽": f"{tname(r.home_team)} {int(r.home_score)}-"
                        f"{int(r.away_score)} {tname(r.away_team)}",
                "主勝": f"{p['home']:.0%}", "和": f"{p['draw']:.0%}",
                "客勝": f"{p['away']:.0%}",
                "模型給對的機率": f"{p[out]:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------------- 賽程表 ----------------
# 淘汰賽：(場次, 日期, 主slot, 客slot, 城市, 輪次)
KO_SCHEDULE = [
    (73, "06-28", "2A", "2B", "洛杉磯", "32強"),
    (74, "06-29", "1E", "3:ABCDF", "波士頓", "32強"),
    (75, "06-29", "1F", "2C", "蒙特雷 🇲🇽", "32強"),
    (76, "06-29", "1C", "2F", "休士頓", "32強"),
    (77, "06-30", "1I", "3:CDFGH", "紐約/紐澤西", "32強"),
    (78, "06-30", "2E", "2I", "達拉斯", "32強"),
    (79, "06-30", "1A", "3:CEFHI", "墨西哥城 🇲🇽", "32強"),
    (80, "07-01", "1L", "3:EHIJK", "亞特蘭大", "32強"),
    (81, "07-01", "1D", "3:BEFIJ", "舊金山灣區", "32強"),
    (82, "07-01", "1G", "3:AEHIJ", "西雅圖", "32強"),
    (83, "07-02", "2K", "2L", "多倫多 🇨🇦", "32強"),
    (84, "07-02", "1H", "2J", "洛杉磯", "32強"),
    (85, "07-02", "1B", "3:EFGIJ", "溫哥華 🇨🇦", "32強"),
    (86, "07-03", "1J", "2H", "邁阿密", "32強"),
    (87, "07-03", "1K", "3:DEIJL", "堪薩斯城", "32強"),
    (88, "07-03", "2D", "2G", "達拉斯", "32強"),
    (89, "07-04", "W74", "W77", "費城", "16強"),
    (90, "07-04", "W73", "W75", "休士頓", "16強"),
    (91, "07-05", "W76", "W78", "紐約/紐澤西", "16強"),
    (92, "07-05", "W79", "W80", "墨西哥城 🇲🇽", "16強"),
    (93, "07-06", "W83", "W84", "達拉斯", "16強"),
    (94, "07-06", "W81", "W82", "西雅圖", "16強"),
    (95, "07-07", "W86", "W88", "亞特蘭大", "16強"),
    (96, "07-07", "W85", "W87", "溫哥華 🇨🇦", "16強"),
    (97, "07-09", "W89", "W90", "波士頓", "8強"),
    (98, "07-10", "W93", "W94", "洛杉磯", "8強"),
    (99, "07-11", "W91", "W92", "邁阿密", "8強"),
    (100, "07-11", "W95", "W96", "堪薩斯城", "8強"),
    (101, "07-14", "W97", "W98", "達拉斯", "4強"),
    (102, "07-15", "W99", "W100", "亞特蘭大", "4強"),
    (103, "07-18", "L101", "L102", "邁阿密", "季軍戰"),
    (104, "07-19", "W101", "W102", "紐約/紐澤西", "決賽"),
]

CITY_ZH = {
    "Mexico City": "墨西哥城 🇲🇽", "Zapopan": "瓜達拉哈拉 🇲🇽",
    "Guadalupe": "蒙特雷 🇲🇽", "Atlanta": "亞特蘭大", "Foxborough": "波士頓",
    "Arlington": "達拉斯", "Houston": "休士頓", "Kansas City": "堪薩斯城",
    "Inglewood": "洛杉磯", "Miami Gardens": "邁阿密",
    "East Rutherford": "紐約/紐澤西", "Philadelphia": "費城",
    "Santa Clara": "舊金山灣區", "Seattle": "西雅圖",
    "Toronto": "多倫多 🇨🇦", "Vancouver": "溫哥華 🇨🇦",
}


def slot_zh(s: str) -> str:
    if s.startswith("W"):
        return f"第{s[1:]}場勝者"
    if s.startswith("L"):
        return f"第{s[1:]}場敗者"
    if s.startswith("3:"):
        return f"最佳第三（{'/'.join(s[2:])} 組）"
    return f"{s[1]}組第{s[0]}名"


WEEK_ZH = "一二三四五六日"


def _tw_fmt(ts) -> str:
    if ts is None:
        return "—"
    return f"{ts:%m-%d %H:%M}（{WEEK_ZH[ts.dayofweek]}）"


with tab_sched:
    fc1, fc2 = st.columns(2)
    with fc1:
        group_filter = st.selectbox(
            "篩選組別", ["全部"] + [f"{g} 組" for g in data.GROUPS]
        )
    # 隊伍下拉跟著組別連動：選了組就只列該組 4 隊
    team_pool = (
        data.GROUPS[group_filter[0]] if group_filter != "全部"
        else data.ALL_TEAMS
    )
    with fc2:
        team_filter = st.selectbox(
            "篩選隊伍", ["全部"] + [tname(t) for t in team_pool]
        )
    only_pending = st.checkbox("只看未踢", value=False)

    g_times, ko_times = data.kickoff_times()
    rows = []
    for r in fixtures.itertuples():
        score = (
            f"{int(r.home_score)} : {int(r.away_score)}"
            if pd.notna(r.home_score) else "—"
        )
        tw = (g_times.get((r.home_team, r.away_team))
              or g_times.get((r.away_team, r.home_team)))
        rows.append({
            "當地日期": str(r.date.date()), "台灣時間": _tw_fmt(tw),
            "輪次": f"小組 {r.group}",
            "主旗": flag_url(r.home_team), "主隊": tname(r.home_team),
            "比分": score,
            "客旗": flag_url(r.away_team), "客隊": tname(r.away_team),
            "城市": CITY_ZH.get(r.city, r.city),
            "_teams": {r.home_team, r.away_team}, "_group": r.group,
            "_pending": pd.isna(r.home_score),
            "_sort": tw if tw is not None else pd.Timestamp(r.date),
        })
    for no, md, hs, as_, city, rnd in KO_SCHEDULE:
        tw = ko_times.get(no)
        rows.append({
            "當地日期": f"2026-{md}", "台灣時間": _tw_fmt(tw),
            "輪次": f"{rnd}（{no}）",
            "主旗": None, "主隊": slot_zh(hs), "比分": "—",
            "客旗": None, "客隊": slot_zh(as_),
            "城市": city, "_teams": set(), "_group": None, "_pending": True,
            "_sort": tw if tw is not None else pd.Timestamp(f"2026-{md}"),
        })

    sched = pd.DataFrame(rows).sort_values("_sort")
    if group_filter != "全部":
        sched = sched[sched["_group"] == group_filter[0]]
    if team_filter != "全部":
        eng = data.ALL_TEAMS[[tname(t) for t in data.ALL_TEAMS].index(team_filter)]
        sched = sched[sched["_teams"].map(lambda s: eng in s)]
    if only_pending:
        sched = sched[sched["_pending"]]
    today_md = f"{pd.Timestamp.now():%m-%d}"
    st.caption(
        f"共 {len(sched)} 場（台灣今天 {today_md}）。"
        "台灣時間=開球時刻；淘汰賽對戰隊伍待小組賽底定。"
    )

    def _hl_today(row):
        # 半透明黃：dark/light 模式下文字都保持可讀；以台灣日期判定今天
        style = (
            "background-color: rgba(250, 204, 21, 0.22)"
            if row["台灣時間"].startswith(today_md) else ""
        )
        return [style] * len(row)

    show_cols = ["台灣時間", "當地日期", "輪次", "主旗", "主隊", "比分",
                 "客旗", "客隊", "城市"]
    st.dataframe(
        sched[show_cols].style.apply(_hl_today, axis=1),
        hide_index=True, width="stretch", height=700,
        column_config={
            "主旗": st.column_config.ImageColumn("", width=36),
            "客旗": st.column_config.ImageColumn("", width=36),
        },
    )

# ---------------- 冠軍機率 ----------------
with tab_title:
    tab = get_sim(mt, n_sims, live_key)
    top_n = st.slider("顯示前幾名", 5, 48, 16)
    show = tab.head(top_n).iloc[::-1]
    fig = px.bar(
        show, x="champion", y="team", orientation="h",
        text=show["champion"].map(lambda v: f"{v:.1%}"),
    )
    fig.update_traces(marker_color="#16a34a", textposition="outside")
    fig.update_layout(
        height=28 * top_n + 80, xaxis_tickformat=".0%",
        xaxis_title="奪冠機率", yaxis_title="",
        yaxis=dict(ticktext=[tname(t) for t in show["team"]],
                   tickvals=show["team"]),
        margin=dict(l=0, r=40, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("各輪晉級機率")
    full = tab.copy()
    full.insert(0, "旗", full["team"].map(flag_url))
    full["team"] = full["team"].map(tname)
    full = full.rename(columns={
        "team": "隊伍", "group": "組", "R32": "32強", "R16": "16強",
        "QF": "8強", "SF": "4強", "F": "決賽", "champion": "冠軍",
    })
    st.dataframe(
        full.style.format(
            {c: "{:.1%}" for c in ["32強", "16強", "8強", "4強", "決賽", "冠軍"]}
        ).background_gradient(
            subset=["32強", "16強", "8強", "4強", "決賽", "冠軍"], cmap="Greens"
        ),
        hide_index=True, width="stretch", height=600,
        column_config={"旗": st.column_config.ImageColumn("", width=36)},
    )

# ---------------- 對戰樹 ----------------
with tab_bracket:
    st.caption(
        "把模型推成一條「最可能的淘汰賽路徑」：32 強對戰取蒙地卡羅裡最常出現的"
        "組合,之後每場由模型算勝率(含延長賽/PK)讓較強者晉級。這是**最可能的單一劇本**,"
        "不代表必然——每場都有冷門空間,真正的機率看『🏆 冠軍機率』分頁。"
    )
    br, champion, meet = get_bracket(mt, n_sims, live_key)
    final = br[104]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"### 🏆 預測冠軍：{tname(champion)}")
    with c2:
        st.markdown(
            f"#### 預測決賽\n{zh(final['home'])} vs {zh(final['away'])} "
            f"→ **{zh(final['winner'])}** {final['win_p'] * 100:.0f}%"
        )

    st.graphviz_chart(bracket_dot(br, champion), width="content")
    st.caption(
        "🟩 綠字＝該場預測晉級者及其勝率　|　🟨 金框＝冠軍的晉級之路"
    )

    with st.expander("👑 冠軍晉級之路（逐輪對手與勝率）"):
        path = sorted(
            (b for b in br.values() if b["winner"] == champion),
            key=lambda b: bracketmod.ROUND_ORDER.index(b["round"]),
        )
        rows = []
        for b in path:
            opp = b["away"] if b["winner"] == b["home"] else b["home"]
            rows.append({
                "輪次": bracketmod.ROUND_ZH[b["round"]],
                "對手": tname(opp), "勝率": b["win_p"],
            })
        st.dataframe(
            pd.DataFrame(rows).style.format({"勝率": "{:.0%}"}).background_gradient(
                subset=["勝率"], cmap="Greens", vmin=0.4, vmax=1.0),
            hide_index=True, width="stretch",
        )

    st.divider()
    st.subheader("✨ 夢幻對決機率")
    st.caption("任選兩隊,算牠們在淘汰賽『碰頭』的機率(單屆最多遇一次,各輪相加即總機率)。")
    opts = data.ALL_TEAMS
    dc1, dc2 = st.columns(2)
    with dc1:
        ta = st.selectbox("隊伍 A", opts, index=opts.index("Argentina"),
                          format_func=tname, key="dm_a")
    with dc2:
        tb = st.selectbox("隊伍 B", opts, index=opts.index("Brazil"),
                          format_func=tname, key="dm_b")
    if ta == tb:
        st.info("請選兩支不同的球隊。")
    else:
        tot, byr = bracketmod.meeting_breakdown(meet, n_sims, ta, tb)
        st.metric(f"{zh(ta)} ⚔️ {zh(tb)}　淘汰賽相遇機率", f"{tot * 100:.1f}%")
        bdf = pd.DataFrame(
            [{"輪次": bracketmod.ROUND_ZH[rn], "機率": byr[rn]}
             for rn in bracketmod.ROUND_ORDER]
        )
        fig = px.bar(bdf, x="輪次", y="機率", text="機率")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside",
                          marker_color="#2a9d8f")
        fig.update_layout(yaxis_tickformat=".0%", height=260,
                          margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### 🥇 最可能的決賽對戰 Top 8")
    fdf = pd.DataFrame(
        [{"決賽對戰": f"{tname(a)}  vs  {tname(b)}", "機率": p}
         for a, b, p in bracketmod.likely_finals(meet, n_sims, 8)]
    )
    st.dataframe(
        fdf.style.format({"機率": "{:.1%}"}).background_gradient(
            subset=["機率"], cmap="Greens"),
        hide_index=True, width="stretch",
    )

# ---------------- Elo 實力榜 ----------------
with tab_elo:
    st.caption(
        "Elo 用一個數字代表實力（西洋棋評分系統,搬到足球）。從 1872 年至今"
        "所有國際賽結果自行計算:贏球加分、輸球扣分,爆冷贏加更多、大勝加更多、"
        "世界盃權重 > 友誼賽。與左邊『冠軍機率』(Poisson 模擬)是兩套獨立方法,可互相印證。"
    )
    elo = get_elo(mt, live_key)
    champ = get_sim(mt, n_sims, live_key).set_index("team")["champion"].to_dict()
    rows = [
        {"team": t, "group": data.TEAM_TO_GROUP[t],
         "elo": round(elo.get(t, 1500)), "champ": champ.get(t, 0)}
        for t in data.ALL_TEAMS
    ]
    edf = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
    edf.insert(0, "排名", edf.index + 1)

    top_n = st.slider("顯示前幾名", 10, 48, 20, key="elo_top")
    show = edf.head(top_n).iloc[::-1]
    fig = px.bar(
        show, x="elo", y="team", orientation="h",
        text="elo", color="elo", color_continuous_scale="Tealgrn",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=26 * top_n + 80, coloraxis_showscale=False,
        xaxis_title="Elo 實力分", yaxis_title="",
        xaxis_range=[show["elo"].min() - 60, show["elo"].max() + 60],
        yaxis=dict(ticktext=[tname(t) for t in show["team"]], tickvals=show["team"]),
        margin=dict(l=0, r=30, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    disp = edf.copy()
    disp.insert(1, "旗", disp["team"].map(flag_url))
    disp["team"] = disp["team"].map(tname)
    disp = disp.rename(columns={"team": "隊伍", "group": "組",
                                "elo": "Elo", "champ": "模型冠軍率"})
    st.dataframe(
        disp.style.format({"模型冠軍率": "{:.1%}"}).background_gradient(
            subset=["Elo"], cmap="Greens"),
        hide_index=True, width="stretch", height=560,
        column_config={"旗": st.column_config.ImageColumn("", width=36)},
    )
    st.caption("💡 Elo 排名與模型冠軍率高度一致 → 兩套獨立方法互相印證;"
               "若某隊兩者分歧大,值得深究(可能 Elo 沒反映近期陣容變化)。")

# ---------------- 分組形勢 ----------------
with tab_groups:
    tab = get_sim(mt, n_sims, live_key)
    adv = tab.set_index("team")["R32"]
    cols = st.columns(2)
    for i, (g, teams) in enumerate(data.GROUPS.items()):
        gp = played[played["group"] == g]
        pts = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        pl = {t: 0 for t in teams}
        for r in gp.itertuples():
            hs, as_ = int(r.home_score), int(r.away_score)
            pl[r.home_team] += 1; pl[r.away_team] += 1
            gf[r.home_team] += hs; gf[r.away_team] += as_
            gd[r.home_team] += hs - as_; gd[r.away_team] += as_ - hs
            if hs > as_:
                pts[r.home_team] += 3
            elif hs < as_:
                pts[r.away_team] += 3
            else:
                pts[r.home_team] += 1; pts[r.away_team] += 1
        rows = sorted(
            [(t, pl[t], pts[t], gd[t], gf[t], adv[t]) for t in teams],
            key=lambda x: (x[2], x[3], x[4], x[5]), reverse=True,
        )
        df_g = pd.DataFrame(
            rows, columns=["隊伍", "賽", "積分", "球差", "進球", "晉級32強"]
        )
        df_g.insert(0, "旗", df_g["隊伍"].map(flag_url))
        df_g["隊伍"] = df_g["隊伍"].map(tname)
        with cols[i % 2]:
            st.markdown(f"#### {g} 組")
            st.dataframe(
                df_g.style.format({"晉級32強": "{:.0%}"}).background_gradient(
                    subset=["晉級32強"], cmap="Greens", vmin=0, vmax=1
                ),
                hide_index=True, width="stretch",
                column_config={"旗": st.column_config.ImageColumn("", width=36)},
            )

# ---------------- 模型 vs 市場 ----------------
with tab_market:
    if not len(odds):
        st.info("data/odds.csv 還沒有賠率資料")
    else:
        preds = []
        for r in fixtures.itertuples():
            p = model.outcome_probs(r.home_team, r.away_team, not r.neutral)
            preds.append({
                "date": r.date, "home_team": r.home_team,
                "away_team": r.away_team,
                "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
            })
        preds = pd.DataFrame(preds)

        st.subheader("💡 Edge（模型機率 − 市場隱含機率 ≥ 5%）")
        st.caption("⚠️ 紙上驗證用。開賽初期 edge 偏大通常代表模型過度自信（不看陣容傷兵），不是市場錯。")
        ed = oddsmod.edges(preds, odds)
        if len(ed):
            ed2 = ed.rename(columns={
                "match": "比賽", "side": "方向", "model_p": "模型",
                "market_p": "市場", "odds": "賠率", "edge": "Edge",
                "ev_per_unit": "每注期望值",
            })
            ed2["比賽"] = ed2["比賽"].map(localize)
            ed2["方向"] = ed2["方向"].map(
                {"home": "主勝", "draw": "和局", "away": "客勝"}
            )
            st.dataframe(
                ed2.style.format(
                    {"模型": "{:.1%}", "市場": "{:.1%}", "Edge": "{:+.1%}",
                     "每注期望值": "{:+.2f}", "賠率": "{:.2f}"}
                ).background_gradient(subset=["Edge"], cmap="RdYlGn"),
                hide_index=True, width="stretch",
            )
        else:
            st.write("目前沒有 ≥5% 的分歧。")

        st.subheader("🎯 校準記分板（誰比較準）")
        sb = oddsmod.scoreboard(preds, odds, fixtures)
        if len(sb):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("模型 Brier", f"{sb['model_brier'].mean():.4f}")
            c3.metric("模型 LogLoss", f"{sb['model_logloss'].mean():.4f}")
            if "mkt_brier" in sb and sb["mkt_brier"].notna().any():
                c2.metric("市場 Brier", f"{sb['mkt_brier'].mean():.4f}")
                c4.metric("市場 LogLoss", f"{sb['mkt_logloss'].mean():.4f}")
            sb = sb.copy()
            sb["match"] = sb["match"].map(localize)
            st.dataframe(sb, hide_index=True, width="stretch")
        else:
            st.write("還沒有可計分的已賽場次。")

# ---------------- 歷史走勢 ----------------
with tab_history:
    odds_hist_path = data.DATA_DIR / "history" / "odds_history.csv"
    calib_hist_path = data.DATA_DIR / "history" / "calibration_history.csv"
    st.caption("每日預測快照累積的走勢。每天更新會 append 一筆,看機率隨賽事演變。")

    if not odds_hist_path.exists():
        st.info("還沒有歷史快照。跑 `python snapshot_history.py --backfill` 產生。")
    else:
        oh = pd.read_csv(odds_hist_path, parse_dates=["date"])
        round_map = {"冠軍": "champion", "決賽": "F", "4 強": "SF",
                     "8 強": "QF", "16 強": "R16", "32 強": "R32"}
        rsel = st.radio("看哪一輪的機率", list(round_map), horizontal=True)
        rcol = round_map[rsel]

        latest = oh[oh["date"] == oh["date"].max()]
        default = latest.nlargest(6, rcol)["team"].tolist()
        all_teams = (oh.groupby("team")[rcol].max()
                     .sort_values(ascending=False).index.tolist())
        picked = st.multiselect(
            "選球隊", options=all_teams, default=default,
            format_func=lambda t: zh(t),
        )
        if picked:
            sub = oh[oh["team"].isin(picked)].copy()
            sub["隊伍"] = sub["team"].map(zh)
            fig = px.line(
                sub, x="date", y=rcol, color="隊伍", markers=True,
                labels={"date": "日期", rcol: f"{rsel}機率"},
            )
            fig.update_layout(height=440, yaxis_tickformat=".0%",
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
            st.caption("註:過去日期為「以該日視角」回算(只用當天前資料 + 之後比賽當未踢)。")

            elo_hist_path = data.DATA_DIR / "history" / "elo_history.csv"
            if elo_hist_path.exists():
                st.subheader("💪 Elo 實力走勢（同上選的球隊）")
                eh = pd.read_csv(elo_hist_path, parse_dates=["date"])
                esub = eh[eh["team"].isin(picked)].copy()
                esub["隊伍"] = esub["team"].map(zh)
                efig = px.line(esub, x="date", y="elo", color="隊伍",
                               markers=True, labels={"date": "日期", "elo": "Elo 分"})
                efig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(efig, width="stretch")
                st.caption("Elo 隨比賽即時調整:贏球加分、爆冷加更多。看誰的實力曲線在世足期間爬升/下滑。")
        else:
            st.write("選至少一支球隊。")

        if calib_hist_path.exists():
            st.divider()
            st.subheader("🎯 模型 vs 市場 校準走勢（Brier,越低越準）")
            ch = pd.read_csv(calib_hist_path, parse_dates=["date"])
            plot = ch.melt(
                id_vars=["date"], value_vars=["model_brier", "mkt_brier"],
                var_name="來源", value_name="Brier",
            ).dropna(subset=["Brier"])
            plot["來源"] = plot["來源"].map(
                {"model_brier": "模型", "mkt_brier": "市場"}
            )
            if len(plot):
                fig2 = px.line(plot, x="date", y="Brier", color="來源",
                               markers=True, labels={"date": "日期"})
                fig2.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig2, width="stretch")
                st.caption("有賠率的已賽場次才計分,初期樣本少、會抖動。")

# ---------------- 單場下鑽 ----------------
with tab_drill:
    opts = [
        f"{r.date.date()} [{r.group}] {zh(r.home_team)} vs {zh(r.away_team)}"
        for r in pending.itertuples()
    ]
    sel = st.selectbox("選一場", opts)
    r = list(pending.itertuples())[opts.index(sel)]
    grid = model.score_grid(r.home_team, r.away_team, not r.neutral)
    p = model.outcome_probs(r.home_team, r.away_team, not r.neutral)
    lam_h, lam_a = model.lambdas(r.home_team, r.away_team, not r.neutral)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{zh(r.home_team)} 勝", f"{p['home']:.0%}")
    c2.metric("和局", f"{p['draw']:.0%}")
    c3.metric(f"{zh(r.away_team)} 勝", f"{p['away']:.0%}")
    c4.metric("預期進球（主）", f"{lam_h:.2f}")
    c5.metric("預期進球（客）", f"{lam_a:.2f}")

    k = 6
    fig = px.imshow(
        grid[:k, :k] * 100,
        x=[str(i) for i in range(k)], y=[str(i) for i in range(k)],
        color_continuous_scale="Blues", text_auto=".1f", aspect="auto",
        labels=dict(x=f"{zh(r.away_team)} 進球", y=f"{zh(r.home_team)} 進球",
                    color="機率 %"),
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    flat = [(i, j, grid[i, j]) for i in range(k) for j in range(k)]
    flat.sort(key=lambda x: x[2], reverse=True)
    st.caption("最可能比分：" + "、".join(
        f"{i}-{j}（{v:.1%}）" for i, j, v in flat[:5]
    ))

# ---------------- 球星導覽 ----------------
with tab_stars:
    st.caption("10 支重點球隊的核心球員（含背號、現役球隊、年薪、身價）。"
               "💰年薪=俱樂部稅前固定年薪估計;📈身價=Transfermarkt 市值概值"
               "（看「現在值多少」最準,隨表現/轉會窗大幅變動）。標「約」者為概估,"
               f"台幣以 €1≈NT${EUR_TWD} 概算。💖 = 球迷/媒體公認人氣顏值。")
    for s in STORYLINES:
        st.markdown(f"- {s}")
    st.divider()

    champ = get_sim(mt, n_sims, live_key).set_index("team")["champion"].to_dict()
    tiers = ["冠軍熱門", "歐洲傳統強權", "亞洲雙雄"]
    for tier in tiers:
        st.subheader(tier)
        teams = [t for t, g in TEAM_GUIDE.items() if g["tier"] == tier]
        teams.sort(key=lambda t: champ.get(t, 0), reverse=True)
        for t in teams:
            g = TEAM_GUIDE[t]
            grp = data.TEAM_TO_GROUP[t]
            title = (f"{tname(t)}　·　{grp} 組　·　冠軍 {champ.get(t, 0):.1%}")
            with st.expander(title):
                c1, c2 = st.columns([1, 6])
                with c1:
                    fu = flag_url(t)
                    if fu:
                        st.image(fu, width=72)
                with c2:
                    st.markdown(f"**{g['blurb']}**")
                    if g["injury"]:
                        st.markdown(f"🩹 {g['injury']}")
                st.markdown("")
                for n, nm, pos, club, salary, note in g["players"]:
                    club_txt = f" · {club}" if club and club != "—" else ""
                    st.markdown(
                        f"<span style='font-size:1.5em;font-weight:800;"
                        f"color:#16a34a'>{n}</span>　"
                        f"<span style='font-size:1.15em;font-weight:700'>{nm}</span>"
                        f"　<small style='color:#888'>{pos}{club_txt}"
                        f"{age_str(nm)}</small>",
                        unsafe_allow_html=True,
                    )
                    twd = twd_from_salary(salary)
                    twd_txt = f"　<span style='color:#888'>{twd}</span>" if twd else ""
                    mv = MARKET_VALUE.get(nm, "")
                    mv_twd = twd_from_salary(mv)
                    mv_txt = (
                        f"　📈 身價 {mv}　<span style='color:#888'>{mv_twd}</span>"
                        if mv else ""
                    )
                    st.markdown(
                        f"<small>💰 年薪 {salary}{twd_txt}{mv_txt}</small>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(note)
                    st.markdown("")

# ---------------- 射手榜 ----------------
with tab_scorers:
    st.subheader("🥇 個人射手榜（金靴戰況）")
    st.caption(
        "ESPN 逐場進球紀錄即時累計（含 12 碼 PK，排除烏龍球）。martj42 只有比分、"
        "沒有進球者，故個人榜另接 ESPN——非官方 API，壞了這頁會空、不影響其他分頁。"
    )
    sc = get_scorers()
    if sc.empty:
        st.info("目前抓不到射手資料（賽事未開始或 ESPN 暫時無回應）。")
    else:
        # 標準並列名次（5,4,4,4 → 1,2,2,2,5...）
        sc = sc.copy()
        sc["名次"] = sc["goals"].rank(method="min", ascending=False).astype(int)

        top = sc.iloc[0]
        leaders = sc[sc["goals"] == top["goals"]]
        names = "、".join(
            f"{scorersmod.player_zh(p)}（{zh(t)}）"
            for p, t in zip(leaders["player"], leaders["team"])
        )
        st.success(f"🏆 目前金靴領先：**{names}** — {int(top['goals'])} 球")

        show = pd.DataFrame({
            "名次": sc["名次"],
            "旗": sc["team"].map(flag_url),
            "球員": sc["player"].map(scorersmod.player_zh),
            "原文": sc["player"],
            "隊伍": sc["team"].map(zh),
            "進球": sc["goals"],
            "其中 PK": sc["penalties"],
        })
        st.dataframe(
            show,
            hide_index=True, width="stretch", height=600,
            column_config={
                "旗": st.column_config.ImageColumn("", width=36),
                "進球": st.column_config.ProgressColumn(
                    "進球", format="%d",
                    min_value=0, max_value=int(sc["goals"].max()),
                ),
            },
        )
        st.caption(
            f"共 {len(sc)} 位球員進球，總計 {int(sc['goals'].sum())} 顆"
            f"（其中 PK {int(sc['penalties'].sum())} 顆）。"
        )
