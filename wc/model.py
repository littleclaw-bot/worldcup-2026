"""Dixon-Coles style model.

Stage 1: weighted Poisson GLM
    log(lambda) = mu + home * is_home_nonneutral + att[team] - def[opponent]
Stage 2: Dixon-Coles low-score correction rho fitted by profile likelihood
         given stage-1 lambdas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MAX_GOALS = 10


@dataclass
class DCModel:
    teams: list[str]
    att: dict[str, float]
    deff: dict[str, float]
    mu: float
    home_adv: float
    rho: float

    def lambdas(self, home: str, away: str, home_field: bool) -> tuple[float, float]:
        lam_h = np.exp(
            self.mu
            + (self.home_adv if home_field else 0.0)
            + self.att[home]
            - self.deff[away]
        )
        lam_a = np.exp(self.mu + self.att[away] - self.deff[home])
        return float(lam_h), float(lam_a)

    def score_grid(self, home: str, away: str, home_field: bool) -> np.ndarray:
        """(MAX_GOALS+1, MAX_GOALS+1) matrix of P(home=i, away=j)."""
        lam_h, lam_a = self.lambdas(home, away, home_field)
        return score_grid_from_lambdas(lam_h, lam_a, self.rho)

    def outcome_probs(self, home: str, away: str, home_field: bool) -> dict[str, float]:
        g = self.score_grid(home, away, home_field)
        return {
            "home": float(np.tril(g, -1).sum()),
            "draw": float(np.trace(g)),
            "away": float(np.triu(g, 1).sum()),
        }


def _tau(hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray,
         rho: float) -> np.ndarray:
    t = np.ones_like(lam_h, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    t = np.where(m00, 1 - lam_h * lam_a * rho, t)
    t = np.where(m01, 1 + lam_h * rho, t)
    t = np.where(m10, 1 + lam_a * rho, t)
    t = np.where(m11, 1 - rho, t)
    return t


def score_grid_from_lambdas(lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    k = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(k, lam_h)
    pa = poisson.pmf(k, lam_a)
    grid = np.outer(ph, pa)
    hh, aa = np.meshgrid(k, k, indexing="ij")
    grid *= np.clip(_tau(hh, aa, lam_h, lam_a, rho), 1e-10, None)
    return grid / grid.sum()


def fit(matches: pd.DataFrame, alpha: float = 5e-4) -> DCModel:
    """matches: home_team, away_team, home_score, away_score, neutral, weight."""
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    n_m = len(matches)

    hi = matches["home_team"].map(idx).to_numpy()
    ai = matches["away_team"].map(idx).to_numpy()
    home_field = (~matches["neutral"]).to_numpy().astype(float)

    # two rows per match: home-side goals then away-side goals
    # columns: [home_flag, att x n_t, def x n_t]
    rows, cols, vals = [], [], []
    for r in range(n_m):
        rows += [r, r, r]
        cols += [0, 1 + hi[r], 1 + n_t + ai[r]]
        vals += [home_field[r], 1.0, -1.0]
        s = n_m + r
        rows += [s, s]
        cols += [1 + ai[r], 1 + n_t + hi[r]]
        vals += [1.0, -1.0]
    X = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(2 * n_m, 1 + 2 * n_t)
    )
    y = np.concatenate(
        [matches["home_score"].to_numpy(float), matches["away_score"].to_numpy(float)]
    )
    w = np.concatenate([matches["weight"].to_numpy(float)] * 2)

    reg = PoissonRegressor(alpha=alpha, max_iter=2000, tol=1e-8)
    reg.fit(X, y, sample_weight=w)

    coef = reg.coef_
    att = coef[1 : 1 + n_t].copy()
    deff = coef[1 + n_t :].copy()
    # identifiability: shift means into mu
    mu = float(reg.intercept_ + att.mean() - deff.mean())
    att -= att.mean()
    deff -= deff.mean()
    model = DCModel(
        teams=teams,
        att=dict(zip(teams, att)),
        deff=dict(zip(teams, deff)),
        mu=mu,
        home_adv=float(coef[0]),
        rho=0.0,
    )
    model.rho = _fit_rho(model, matches, hi, ai, home_field)
    return model


def _fit_rho(model: DCModel, matches: pd.DataFrame, hi, ai, home_field) -> float:
    teams = model.teams
    att = np.array([model.att[t] for t in teams])
    deff = np.array([model.deff[t] for t in teams])
    lam_h = np.exp(model.mu + model.home_adv * home_field + att[hi] - deff[ai])
    lam_a = np.exp(model.mu + att[ai] - deff[hi])
    hg = matches["home_score"].to_numpy(float)
    ag = matches["away_score"].to_numpy(float)
    w = matches["weight"].to_numpy(float)

    def nll(rho: float) -> float:
        t = np.clip(_tau(hg, ag, lam_h, lam_a, rho), 1e-10, None)
        return -float((w * np.log(t)).sum())

    res = minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)
