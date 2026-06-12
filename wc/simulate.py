"""Monte Carlo simulation of the 48-team WC2026 format.

Group stage: 12 groups of 4; top 2 + 8 best thirds advance.
Thirds are assigned to bracket slots by constraint matching (each slot
admits thirds from 5 specific groups, per the official bracket).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GROUPS, KNOCKOUT_ROUNDS, TEAM_TO_GROUP
from .model import DCModel, MAX_GOALS, score_grid_from_lambdas

ROUNDS = ["R32", "R16", "QF", "SF", "F", "champion"]


class MatchSampler:
    """Caches score grids per (home, away, home_field) and samples scores."""

    def __init__(self, model: DCModel, rng: np.random.Generator):
        self.model = model
        self.rng = rng
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _grids(self, home: str, away: str, home_field: bool):
        key = (home, away, home_field)
        if key not in self._cache:
            reg = self.model.score_grid(home, away, home_field).ravel()
            lam_h, lam_a = self.model.lambdas(home, away, home_field)
            et = score_grid_from_lambdas(lam_h / 3.0, lam_a / 3.0, 0.0).ravel()
            self._cache[key] = (reg, et)
        return self._cache[key]

    def sample(self, home: str, away: str, home_field: bool) -> tuple[int, int]:
        reg, _ = self._grids(home, away, home_field)
        f = self.rng.choice(reg.size, p=reg)
        return f // (MAX_GOALS + 1), f % (MAX_GOALS + 1)

    def sample_winner(self, home: str, away: str, home_field: bool) -> str:
        h, a = self.sample(home, away, home_field)
        if h != a:
            return home if h > a else away
        _, et = self._grids(home, away, home_field)
        f = self.rng.choice(et.size, p=et)
        eh, ea = f // (MAX_GOALS + 1), f % (MAX_GOALS + 1)
        if eh != ea:
            return home if eh > ea else away
        return home if self.rng.random() < 0.5 else away  # penalties


def _standings(teams: list[str], results: list[tuple[str, str, int, int]],
               rng: np.random.Generator) -> list[tuple]:
    """Returns rows (team, pts, gd, gf) sorted best-first."""
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for h, a, hs, as_ in results:
        gf[h] += hs; ga[h] += as_
        gf[a] += as_; ga[a] += hs
        if hs > as_:
            pts[h] += 3
        elif hs < as_:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    rows = [(t, pts[t], gf[t] - ga[t], gf[t], rng.random()) for t in teams]
    rows.sort(key=lambda r: (r[1], r[2], r[3], r[4]), reverse=True)
    return rows


def _assign_thirds(qualified_groups: list[str], slots: list[tuple[int, str]]):
    """Backtracking perfect matching: third-place group -> R32 slot.

    slots: [(match_no, allowed_groups_str), ...]; returns {match_no: group}.
    """
    assignment: dict[int, str] = {}

    def bt(i: int, used: set[str]) -> bool:
        if i == len(slots):
            return True
        match_no, allowed = slots[i]
        for g in qualified_groups:
            if g in allowed and g not in used:
                assignment[match_no] = g
                if bt(i + 1, used | {g}):
                    return True
                del assignment[match_no]
        return False

    if not bt(0, set()):
        # No perfect matching (FIFA would resolve ad hoc); relax constraints.
        free = [g for g in qualified_groups if g not in assignment.values()]
        for match_no, _ in slots:
            if match_no not in assignment:
                assignment[match_no] = free.pop()
    return assignment


def simulate_tournament(
    model: DCModel,
    fixtures: pd.DataFrame,
    n_sims: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Returns per-team probabilities of reaching each round."""
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

    counts = {t: {r: 0 for r in ROUNDS} for t in TEAM_TO_GROUP}

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
                counts[home][rname] += 1
                counts[away][rname] += 1
                hf = home == venue  # host nations only (team name == country)
                af = away == venue
                if af and not hf:
                    home, away = away, home
                winners[f"W{no}"] = sampler.sample_winner(home, away, hf and not af)
        counts[winners["W104"]]["champion"] += 1

    out = pd.DataFrame(
        [
            {"team": t, "group": TEAM_TO_GROUP[t],
             **{r: counts[t][r] / n_sims for r in ROUNDS}}
            for t in TEAM_TO_GROUP
        ]
    ).sort_values("champion", ascending=False).reset_index(drop=True)
    return out
