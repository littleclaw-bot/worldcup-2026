"""WC2026 預測儀表板.

啟動：
    C:/ProgramData/anaconda3/envs/python310/python.exe -m streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wc import data, model as dcmodel, odds as oddsmod, simulate

st.set_page_config(page_title="WC2026 預測", page_icon="⚽", layout="wide")

FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}


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


def tname(t: str) -> str:
    """旗 + 中文 + 英文，給表格/標題用."""
    return f"{FLAGS.get(t, '')} {ZH.get(t, t)} {t}"


def zh(t: str) -> str:
    """旗 + 中文，給空間小的地方用."""
    return f"{FLAGS.get(t, '')} {ZH.get(t, t)}"


def localize(s: str) -> str:
    """把字串中的英文隊名換成 旗+中文（長名優先避免部分覆蓋）."""
    for t in sorted(ZH, key=len, reverse=True):
        s = s.replace(t, zh(t))
    return s


def _mtimes() -> tuple:
    paths = [data.RESULTS_CSV, oddsmod.ODDS_CSV]
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)


@st.cache_resource(show_spinner="fit 模型中 ...")
def get_model(mtimes: tuple):
    df = data.load_results()
    train = data.training_matches(df)
    m = dcmodel.fit(train)
    return m, df, len(train)


@st.cache_data(show_spinner="Monte Carlo 模擬中 ...")
def get_sim(mtimes: tuple, n_sims: int) -> pd.DataFrame:
    m, df, _ = get_model(mtimes)
    return simulate.simulate_tournament(m, data.wc_fixtures(df), n_sims=n_sims)


# ---------------- sidebar ----------------
st.sidebar.title("⚽ WC2026 預測")
if st.sidebar.button("🔄 抓最新比分（重新下載）"):
    with st.spinner("下載 results.csv ..."):
        data.refresh_results()
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

n_sims = st.sidebar.select_slider("模擬次數", [2000, 5000, 10000, 20000], value=10000)

mt = _mtimes()
model, results_df, n_train = get_model(mt)
fixtures = data.wc_fixtures(results_df)
odds = oddsmod.load_odds()

st.sidebar.caption(
    f"訓練 {n_train} 場（2018–今）\n\n"
    f"home_adv={model.home_adv:.3f}, rho={model.rho:.4f}\n\n"
    f"資料最後日期：{results_df['date'].max().date()}"
)

played = fixtures[fixtures["home_score"].notna()]
pending = fixtures[fixtures["home_score"].isna()]

tab_match, tab_title, tab_groups, tab_market, tab_drill = st.tabs(
    ["📅 賽事預測", "🏆 冠軍機率", "📋 分組形勢", "📊 模型 vs 市場", "🔍 單場下鑽"]
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

# ---------------- 冠軍機率 ----------------
with tab_title:
    tab = get_sim(mt, n_sims)
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
    )

# ---------------- 分組形勢 ----------------
with tab_groups:
    tab = get_sim(mt, n_sims)
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
        df_g["隊伍"] = df_g["隊伍"].map(tname)
        with cols[i % 2]:
            st.markdown(f"#### {g} 組")
            st.dataframe(
                df_g.style.format({"晉級32強": "{:.0%}"}).background_gradient(
                    subset=["晉級32強"], cmap="Greens", vmin=0, vmax=1
                ),
                hide_index=True, width="stretch",
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
