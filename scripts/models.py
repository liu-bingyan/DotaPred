"""Model variants beyond a single team-level scalar rating.

  team    one rating per org, fit on gold-diff-per-minute      (current best)
  player  one rating per *player*; team strength is the sum of its five.
          Rosters move between orgs and orgs rename -- player ratings track
          the thing that actually carries skill, and pool a player's evidence
          across every org they have played for.
  phase   three ratings per team (laning / mid / late) fit on the gold curve
          at minute 10, 10->25 and 25->end. A single scalar cannot express
          "strong early, folds late"; three can.
  blend   logistic stack over whichever rating differences are supplied.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ridge_ratings(rows_idx, w, y, n_ent, l2=1.0, iters=2500, lr=0.05):
    """Weighted ridge on  y ~ sum(+1 for home entities) - sum(-1 for away) + h.

    rows_idx is a list of (plus_ids, minus_ids) per observation, so this covers
    both the 1-vs-1 team case and the 5-vs-5 player case.
    """
    n_obs = len(rows_idx)
    plus_flat, plus_row, minus_flat, minus_row = [], [], [], []
    for k, (pl, mi) in enumerate(rows_idx):
        plus_flat.extend(pl)
        plus_row.extend([k] * len(pl))
        minus_flat.extend(mi)
        minus_row.extend([k] * len(mi))
    plus_flat = np.array(plus_flat)
    plus_row = np.array(plus_row)
    minus_flat = np.array(minus_flat)
    minus_row = np.array(minus_row)

    theta = np.zeros(n_ent + 1)
    mom = np.zeros_like(theta)
    vel = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, iters + 1):
        r, h = theta[:-1], theta[-1]
        pred = np.zeros(n_obs)
        np.add.at(pred, plus_row, r[plus_flat])
        np.add.at(pred, minus_row, -r[minus_flat])
        pred += h
        g = w * (y - pred)
        grad = np.zeros_like(theta)
        np.add.at(grad, plus_flat, g[plus_row])
        np.add.at(grad, minus_flat, -g[minus_row])
        grad[:-1] -= l2 * r
        grad[-1] = g.sum()
        mom = b1 * mom + (1 - b1) * grad
        vel = b2 * vel + (1 - b2) * grad * grad
        theta += lr * (mom / (1 - b1**step)) / (np.sqrt(vel / (1 - b2**step)) + eps)
        theta[:-1] -= theta[:-1].mean()
    return theta[:-1], theta[-1]


def load_lineups():
    path = os.path.join(ROOT, "data", "raw", "lineups.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for r in raw:
        out.setdefault(r["match_id"], ([], []))[0 if r["player_slot"] < 128 else 1].append(
            r["account_id"]
        )
    return {k: v for k, v in out.items() if len(v[0]) == 5 and len(v[1]) == 5}


def load_phase():
    path = os.path.join(ROOT, "data", "raw", "phase_curves.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["match_id"]: r for r in json.load(f)}


def phase_targets(rows, phase):
    """Per-game laning / mid / late gold rates, radiant perspective."""
    n = len(rows)
    early = np.zeros(n)
    mid = np.zeros(n)
    late = np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    for k, r in enumerate(rows):
        p = phase.get(r["match_id"])
        if not p or p.get("g10") is None:
            continue
        g10 = p["g10"] or 0
        g25 = p.get("g25")
        dur_min = max(r["duration"], 1) / 60.0
        final = r.get("final_gold_adv") or 0
        early[k] = g10 / 10.0
        if g25 is not None:
            mid[k] = (g25 - g10) / 15.0
            if dur_min > 26:
                late[k] = (final - g25) / max(dur_min - 25.0, 1.0)
            else:
                late[k] = (final - g10) / max(dur_min - 10.0, 1.0)
        else:
            mid[k] = (final - g10) / max(dur_min - 10.0, 1.0)
            late[k] = mid[k]
        ok[k] = True
    return early, mid, late, ok
