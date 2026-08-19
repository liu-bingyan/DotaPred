"""Margin-of-victory features and a margin-based rating fit.

A win/loss rating learns 1 bit per game. A margin rating learns how *hard* the
win was, which is a far lower-variance signal about team strength -- the same
reason point-differential ratings beat win-loss ratings in every other sport.

Margins are deliberately bounded (tanh / capped counts). Dota snowballs, so an
unbounded gold lead mostly measures how long the loser refused to call GG, not
how much better the winner was.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIER_WEIGHT = {"premium": 1.0, "professional": 0.75, "excluded": 0.15, None: 0.15}


def popcount(x):
    return bin(int(x)).count("1") if x is not None else 0


def load_rich(alias=None):
    with open(os.path.join(ROOT, "data", "raw", "pro_matches_rich.json")) as f:
        rows = json.load(f)
    alias = alias or {}
    for r in rows:
        r["radiant_team_id"] = alias.get(r["radiant_team_id"], r["radiant_team_id"])
        r["dire_team_id"] = alias.get(r["dire_team_id"], r["dire_team_id"])
    rows = [r for r in rows if r["radiant_team_id"] != r["dire_team_id"]]
    rows.sort(key=lambda r: r["start_time"])
    return rows


def margins(rows):
    """Per-game margin components, all from the radiant point of view."""
    n = len(rows)
    out = {k: np.zeros(n) for k in
           ("gdpm", "xpdpm", "kdpm", "tower", "rax", "tanh_gold", "dur", "comeback")}
    win = np.zeros(n)
    for k, r in enumerate(rows):
        mins = max(r["duration"], 1) / 60.0
        g = r.get("final_gold_adv") or 0
        x = r.get("final_xp_adv") or 0
        out["gdpm"][k] = g / mins
        out["xpdpm"][k] = x / mins
        out["kdpm"][k] = ((r.get("radiant_score") or 0) - (r.get("dire_score") or 0)) / mins
        out["tower"][k] = popcount(r.get("tower_status_radiant")) - popcount(r.get("tower_status_dire"))
        out["rax"][k] = popcount(r.get("barracks_status_radiant")) - popcount(r.get("barracks_status_dire"))
        out["tanh_gold"][k] = np.tanh(g / 15000.0)
        out["dur"][k] = mins
        # how big a deficit did the winner overturn? (0 if they led throughout)
        mx, mn = r.get("max_gold_adv") or 0, r.get("min_gold_adv") or 0
        out["comeback"][k] = (-mn if g > 0 else mx) / 10000.0
        win[k] = 1.0 if r["radiant_win"] else 0.0
    return out, win


def composite(m, win):
    """One bounded scalar per game. Blend of dominance signals, sign-anchored to
    the actual result so a 'lost but had better stats' game can't outrank a win."""
    z = (
        0.40 * np.tanh(m["gdpm"] / 400.0)
        + 0.20 * np.tanh(m["xpdpm"] / 500.0)
        + 0.15 * np.clip(m["rax"] / 6.0, -1, 1)
        + 0.15 * np.clip(m["tower"] / 11.0, -1, 1)
        + 0.10 * np.tanh(m["kdpm"] / 0.6)
    )
    sign = np.where(win > 0.5, 1.0, -1.0)
    # keep the sign of the result; magnitude says how convincing it was
    return sign * np.clip(np.abs(z), 0.05, 1.0) * np.where(np.sign(z) == sign, 1.0, 0.5)


def build_design(rows, now, half_life_days=150.0, min_games=20):
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
    t = np.array([r["start_time"] for r in rows], dtype=float)
    tier = np.array([TIER_WEIGHT.get(r.get("tier"), 0.15) for r in rows])
    w = np.exp(-lam * (now - t)) * tier
    m, win = margins(rows)
    return i, j, w, idx, m, win, rows


def fit_margin(i, j, y, w, n_teams, l2=1.0, iters=3000, lr=0.05):
    """Weighted ridge on  y ~ r_i - r_j + h  (y is a continuous margin)."""
    theta = np.zeros(n_teams + 1)
    mom = np.zeros_like(theta)
    vel = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, iters + 1):
        r, h = theta[:-1], theta[-1]
        resid = y - (r[i] - r[j] + h)
        g = w * resid
        grad = np.zeros_like(theta)
        np.add.at(grad, i, g)
        np.add.at(grad, j, -g)
        grad[:-1] -= l2 * r
        grad[-1] = g.sum()
        mom = b1 * mom + (1 - b1) * grad
        vel = b2 * vel + (1 - b2) * grad * grad
        theta += lr * (mom / (1 - b1**step)) / (np.sqrt(vel / (1 - b2**step)) + eps)
        theta[:-1] -= theta[:-1].mean()
    return theta[:-1], theta[-1]


def fit_logistic(X, y, w=None, l2=1e-3, iters=2000, lr=0.05):
    """Small logistic regression to map rating differences -> win probability."""
    X = np.column_stack([X, np.ones(len(y))])
    w = np.ones(len(y)) if w is None else w
    beta = np.zeros(X.shape[1])
    mom = np.zeros_like(beta)
    vel = np.zeros_like(beta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, iters + 1):
        p = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (w * (y - p)) - l2 * beta
        mom = b1 * mom + (1 - b1) * grad
        vel = b2 * vel + (1 - b2) * grad * grad
        beta += lr * (mom / (1 - b1**step)) / (np.sqrt(vel / (1 - b2**step)) + eps)
    return beta


def predict_logistic(beta, X):
    X = np.column_stack([X, np.ones(X.shape[0])])
    return 1 / (1 + np.exp(-X @ beta))
