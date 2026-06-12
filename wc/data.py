"""Data layer: historical results, WC2026 groups / fixtures / bracket.

Team names follow martj42/international_results conventions
(South Korea, Czech Republic, United States, Ivory Coast, Turkey, ...).
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

TEAM_TO_GROUP: dict[str, str] = {
    t: g for g, teams in GROUPS.items() for t in teams
}
ALL_TEAMS: list[str] = sorted(TEAM_TO_GROUP)

# Round of 32: (match_no, home_slot, away_slot, venue_country)
# Slots: "1A" group winner, "2A" runner-up, "3:ABCDF" best third from one of
# those groups, "W74" winner of match 74.
R32 = [
    (73, "2A", "2B", "United States"),
    (74, "1E", "3:ABCDF", "United States"),
    (75, "1F", "2C", "Mexico"),
    (76, "1C", "2F", "United States"),
    (77, "1I", "3:CDFGH", "United States"),
    (78, "2E", "2I", "United States"),
    (79, "1A", "3:CEFHI", "Mexico"),
    (80, "1L", "3:EHIJK", "United States"),
    (81, "1D", "3:BEFIJ", "United States"),
    (82, "1G", "3:AEHIJ", "United States"),
    (83, "2K", "2L", "Canada"),
    (84, "1H", "2J", "United States"),
    (85, "1B", "3:EFGIJ", "Canada"),
    (86, "1J", "2H", "United States"),
    (87, "1K", "3:DEIJL", "United States"),
    (88, "2D", "2G", "United States"),
]
R16 = [
    (89, "W74", "W77", "United States"),
    (90, "W73", "W75", "United States"),
    (91, "W76", "W78", "United States"),
    (92, "W79", "W80", "Mexico"),
    (93, "W83", "W84", "United States"),
    (94, "W81", "W82", "United States"),
    (95, "W86", "W88", "United States"),
    (96, "W85", "W87", "Canada"),
]
QF = [
    (97, "W89", "W90", "United States"),
    (98, "W93", "W94", "United States"),
    (99, "W91", "W92", "United States"),
    (100, "W95", "W96", "United States"),
]
SF = [
    (101, "W97", "W98", "United States"),
    (102, "W99", "W100", "United States"),
]
FINAL = [(104, "W101", "W102", "United States")]
KNOCKOUT_ROUNDS = [("R32", R32), ("R16", R16), ("QF", QF), ("SF", SF), ("F", FINAL)]


_NAME_FIX = {"USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina"}
_MONTHS = {"Jun": 6, "June": 6, "Jul": 7, "July": 7}
_DATE_RE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(June?|July?)\s+(\d{1,2})\b"
)
_GROUP_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s+UTC([+-]\d+)\s+(.*?)\s+@")
_KO_TIME_RE = re.compile(r"^\s*\((\d+)\)\s+(\d{1,2}):(\d{2})\s+UTC([+-]\d+)")
_VS_SPLIT_RE = re.compile(r"\s+(?:v|\d+-\d+(?:\s*\(\d+-\d+\))?)\s+")


def kickoff_times() -> tuple[dict[tuple[str, str], pd.Timestamp], dict[int, pd.Timestamp]]:
    """Parse cup.txt / cup_finals.txt kickoff times -> Taiwan time (UTC+8).

    Returns ({(home, away): tw_ts} for group stage, {match_no: tw_ts} for KO).
    """
    def to_tw(month: int, day: int, hh: int, mm: int, offset: int) -> pd.Timestamp:
        local = pd.Timestamp(2026, month, day, hh, mm)
        return local + pd.Timedelta(hours=8 - offset)

    group_map: dict[tuple[str, str], pd.Timestamp] = {}
    month = day = None
    for line in (DATA_DIR / "cup.txt").read_text(encoding="utf-8").splitlines():
        if m := _DATE_RE.match(line.strip()):
            month, day = _MONTHS[m.group(1)], int(m.group(2))
            continue
        if month is None or not (m := _GROUP_TIME_RE.match(line)):
            continue
        teams = _VS_SPLIT_RE.split(m.group(4).strip())
        if len(teams) != 2:
            continue
        home, away = (_NAME_FIX.get(t.strip(), t.strip()) for t in teams)
        group_map[(home, away)] = to_tw(
            month, day, int(m.group(1)), int(m.group(2)), int(m.group(3))
        )

    ko_map: dict[int, pd.Timestamp] = {}
    month = day = None
    for line in (DATA_DIR / "cup_finals.txt").read_text(encoding="utf-8").splitlines():
        if m := _DATE_RE.match(line.strip()):
            month, day = _MONTHS[m.group(1)], int(m.group(2))
            continue
        if month is None or not (m := _KO_TIME_RE.match(line)):
            continue
        ko_map[int(m.group(1))] = to_tw(
            month, day, int(m.group(2)), int(m.group(3)), int(m.group(4))
        )
    return group_map, ko_map


def refresh_results() -> None:
    """Re-download results.csv (continuously updated upstream with WC scores)."""
    raw = urlopen(RESULTS_URL, timeout=60).read()
    pd.read_csv(io.BytesIO(raw))  # validate before overwriting
    RESULTS_CSV.write_bytes(raw)


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(bool)
    return df


def training_matches(
    df: pd.DataFrame,
    start: str = "2018-01-01",
    asof: pd.Timestamp | None = None,
    half_life_days: float = 913.0,
    friendly_weight: float = 0.6,
) -> pd.DataFrame:
    """Played matches with exponential time-decay weights (as of `asof`)."""
    if asof is None:
        asof = pd.Timestamp.now().normalize()
    m = df[
        (df["date"] >= start)
        & (df["date"] <= asof)
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()
    age_days = (asof - m["date"]).dt.days.clip(lower=0)
    m["weight"] = np.exp(-np.log(2.0) * age_days / half_life_days)
    m.loc[m["tournament"] == "Friendly", "weight"] *= friendly_weight
    return m


def wc_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """The 72 group-stage rows of WC2026, with group labels."""
    f = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-06-01")].copy()
    f["group"] = f["home_team"].map(TEAM_TO_GROUP)
    missing = f[f["group"].isna()]
    if len(missing):
        raise ValueError(f"unmapped WC team names: {missing['home_team'].unique()}")
    if sorted(set(f["home_team"]) | set(f["away_team"])) != ALL_TEAMS:
        raise ValueError("fixture teams do not match GROUPS table")
    return f.sort_values("date").reset_index(drop=True)
