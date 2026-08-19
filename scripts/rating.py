"""Bradley-Terry team ratings with recency + league-tier weighting.

P(i beats j on a given game) = sigmoid(r_i - r_j + h * radiant_indicator)

Fit by weighted MLE with an L2 prior pulling ratings toward 0, so teams with
thin match histories stay near average instead of blowing up. Ratings are in
logit units; multiply by 400/ln(10) for an Elo-like scale.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIER_WEIGHT = {
    "premium": 1.0,
    "professional": 0.75,
    "excluded": 0.15,
    None: 0.15,
}


def load_matches(alias=None):
    with open(os.path.join(ROOT, "data", "raw", "pro_matches.json")) as f:
        rows = json.load(f)
    alias = alias or {}
    for r in rows:
        r["radiant_team_id"] = alias.get(r["radiant_team_id"], r["radiant_team_id"])
        r["dire_team_id"] = alias.get(r["dire_team_id"], r["dire_team_id"])
    rows = [r for r in rows if r["radiant_team_id"] != r["dire_team_id"]]
    rows.sort(key=lambda r: r["start_time"])
    return rows


def build_design(rows, now, half_life_days=150.0, min_games=5):
    """Return (i, j, y, w, index) arrays. i=radiant, j=dire, y=1 if radiant won."""
    counts = {}
    for r in rows:
        counts[r["radiant_team_id"]] = counts.get(r["radiant_team_id"], 0) + 1
        counts[r["dire_team_id"]] = counts.get(r["dire_team_id"], 0) + 1
    keep = {t for t, c in counts.items() if c >= min_games}
    rows = [r for r in rows if r["radiant_team_id"] in keep and r["dire_team_id"] in keep]

    teams = sorted({r["radiant_team_id"] for r in rows} | {r["dire_team_id"] for r in rows})
    idx = {t: k for k, t in enumerate(teams)}

    lam = np.log(2) / (half_life_days * 86400.0)
    i = np.array([idx[r["radiant_team_id"]] for r in rows])
    j = np.array([idx[r["dire_team_id"]] for r in rows])
    y = np.array([1.0 if r["radiant_win"] else 0.0 for r in rows])
    t = np.array([r["start_time"] for r in rows], dtype=float)
    tier = np.array([TIER_WEIGHT.get(r.get("tier"), 0.15) for r in rows])
    w = np.exp(-lam * (now - t)) * tier
    return i, j, y, w, idx, np.array([r["start_time"] for r in rows])


def fit(i, j, y, w, n_teams, l2=2.0, iters=3000, lr=0.05):
    """Adam on the L2-penalised weighted log-likelihood. Returns (ratings, home)."""
    theta = np.zeros(n_teams + 1)  # last slot is the radiant-side advantage
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for step in range(1, iters + 1):
        r, h = theta[:-1], theta[-1]
        d = r[i] - r[j] + h
        p = 1.0 / (1.0 + np.exp(-d))
        g = w * (y - p)

        grad = np.zeros_like(theta)
        np.add.at(grad, i, g)
        np.add.at(grad, j, -g)
        grad[:-1] -= l2 * r
        grad[-1] = g.sum()

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mh = m / (1 - b1**step)
        vh = v / (1 - b2**step)
        theta += lr * mh / (np.sqrt(vh) + eps)
        theta[:-1] -= theta[:-1].mean()  # ratings are only identified up to a shift

    return theta[:-1], theta[-1]


def logloss(r, h, i, j, y, w=None):
    d = r[i] - r[j] + h
    p = np.clip(1.0 / (1.0 + np.exp(-d)), 1e-9, 1 - 1e-9)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    if w is None:
        return -ll.mean(), ((p > 0.5) == (y > 0.5)).mean()
    return -(w * ll).sum() / w.sum(), ((p > 0.5) == (y > 0.5)).mean()


def bo_win_prob(p, wins_needed):
    """P(win a race-to-`wins_needed` series) given per-game prob p."""
    from math import comb

    total = 0.0
    for losses in range(wins_needed):
        total += comb(wins_needed - 1 + losses, losses) * p**wins_needed * (1 - p) ** losses
    return total
