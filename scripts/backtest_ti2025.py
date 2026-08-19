"""Backtest the whole pipeline against TI2025 -- the objective metric itself.

Every accuracy number so far measures "who wins a game", which is not what we
are scored on. We are scored on placing 16 teams into 6 fixed-size buckets.
This runs the exact production pipeline using only data available before the
2025 group stage started, then counts how many buckets it would have got right
and how many points that would have scored.

It is a single event, so it cannot establish much on its own. Its value is as a
sanity check: does reality land where the simulation says it should?
"""

import collections
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin  # noqa: E402
import models  # noqa: E402
import rating  # noqa: E402
from experiment import player_feature, player_fit  # noqa: E402
from optimize_groups import BUCKETS, CAPACITY, GROUP_POINTS  # noqa: E402
from simulate import simulate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TI25_LEAGUE = 18324
GROUP_END = int(dt.datetime(2025, 9, 8, tzinfo=dt.timezone.utc).timestamp())
GROUP_START = int(dt.datetime(2025, 9, 4, tzinfo=dt.timezone.utc).timestamp())
SWISS_END = int(dt.datetime(2025, 9, 7, tzinfo=dt.timezone.utc).timestamp())


def series_results(games):
    """Collapse games into (teamA, teamB, winner) per series."""
    S = collections.defaultdict(list)
    for g in games:
        S[g.get("series_id") or -g["match_id"]].append(g)
    out = []
    for sid, gs in S.items():
        pairs = {frozenset((g["radiant_team_id"], g["dire_team_id"])) for g in gs}
        if len(pairs) != 1 or len(next(iter(pairs))) != 2:
            continue
        a, b = tuple(next(iter(pairs)))
        wa = sum(1 for g in gs if (g["radiant_team_id"] == a) == g["radiant_win"])
        wb = len(gs) - wa
        if wa != wb:
            out.append((a, b, a if wa > wb else b))
    return out


def actual_buckets(all_rows):
    """Reconstruct each team's prediction bucket.

    OpenDota is missing one of the five elimination series, so who won the
    elimination round is inferred from the playoff field instead: the eight
    playoff teams are the three direct qualifiers plus the five elimination
    winners. That needs no elimination-day data at all.
    """
    swiss = [r for r in all_rows if r["leagueid"] == TI25_LEAGUE
             and GROUP_START <= r["start_time"] < SWISS_END]
    playoff = [r for r in all_rows if r["leagueid"] == TI25_LEAGUE
               and r["start_time"] >= GROUP_END]

    rec = collections.defaultdict(lambda: [0, 0])
    for a, b, w in series_results(swiss):
        loser = b if w == a else a
        rec[w][0] += 1
        rec[loser][1] += 1

    in_playoff = set()
    for r in playoff:
        in_playoff.add(r["radiant_team_id"])
        in_playoff.add(r["dire_team_id"])

    out = {}
    for t, (win, loss) in rec.items():
        if win == 4 and loss == 0:
            out[t] = 0
        elif win == 4 and loss == 1:
            out[t] = 1
        elif win == 1 and loss == 4:
            out[t] = 4
        elif win == 0 and loss == 4:
            out[t] = 5
        else:  # 3-2 or 2-3 -> played the elimination round
            out[t] = 2 if t in in_playoff else 3
    return out, rec


def main():
    rows_all = margin.load_rich()
    lineups = models.load_lineups()
    truth, rec = actual_buckets(rows_all)

    print("TI2025 actual outcome")
    counts = collections.Counter(truth.values())
    print(f"  reconstructed buckets: " +
          ", ".join(f"{BUCKETS[b]}={counts.get(b, 0)}" for b in range(6)) +
          f"   (format requires {CAPACITY})")
    if [counts.get(b, 0) for b in range(6)] != CAPACITY:
        print("  !! reconstruction does not match the forced capacities -- check below")
    for t, b in sorted(truth.items(), key=lambda kv: kv[1]):
        print(f"    {t:>9}  {BUCKETS[b]:<7} (swiss {rec[t][0]}-{rec[t][1]})")

    # ---- model, using only pre-tournament data
    train = [r for r in rows_all
             if r["start_time"] < GROUP_START and r.get("tier") in {"premium", "professional"}]
    i, j, w, idx, m, win, tr = margin.build_design(train, GROUP_START, 150.0, 20)
    gd = m["gdpm"] / 400.0
    r_team, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=0.3)
    r_pl, pidx = player_fit(tr, w, lineups, gd, l2=30.0)
    X = np.column_stack([r_team[i] - r_team[j], player_feature(tr, r_pl, pidx, lineups)])
    mu = np.nanmean(X, axis=0)
    beta = margin.fit_logistic(np.where(np.isnan(X), mu, X), win, w)

    field = [t for t in truth if t in idx]
    unrated = [t for t in truth if t not in idx]
    print(f"\n{len(field)}/16 teams rated by the pre-tournament model")

    # per-team lineup as of the tournament: take it from their group-stage games
    pl_rating = {}
    for t in field:
        vals = []
        for r in rows_all:
            if r["leagueid"] != TI25_LEAGUE or r["start_time"] >= GROUP_END:
                continue
            lu = lineups.get(r["match_id"])
            if not lu:
                continue
            side = lu[0] if r["radiant_team_id"] == t else (lu[1] if r["dire_team_id"] == t else None)
            if side:
                vals = [r_pl[pidx[a]] for a in side if a in pidx]
                if len(vals) >= 4:
                    break
        pl_rating[t] = float(np.mean(vals)) if len(vals) >= 4 else float(mu[1])

    strength = {t: beta[0] * r_team[idx[t]] + beta[1] * pl_rating[t] for t in field}
    order = sorted(field, key=lambda t: -strength[t])

    M = {a: {b: (0.5 if a == b else float(1 / (1 + np.exp(-(strength[a] - strength[b])))))
             for b in order} for a in order}
    P = np.array([[rating.bo_win_prob(M[a][b], 2) if a != b else 0.5 for b in order]
                  for a in order])

    if len(field) != 16:
        print(f"    unrated: {unrated} -- backfilled at the bottom of the order")
        order = order + unrated
    slate = np.array([0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5])
    correct = sum(1 for k, t in enumerate(order) if truth.get(t) == slate[k])
    pts = GROUP_POINTS[correct]

    print(f"\n{'rank':>4} {'team_id':>9} {'predicted':<8} {'actual':<8} {'hit':>4}")
    for k, t in enumerate(order):
        hit = "OK" if truth.get(t) == slate[k] else ""
        print(f"{k + 1:>4} {t:>9} {BUCKETS[slate[k]]:<8} {BUCKETS[truth[t]]:<8} {hit:>4}")

    print(f"\n>>> would have got {correct}/16 buckets right = {pts:.0f} points")

    rng = np.random.default_rng(1)
    sims = simulate(P, 20000, rng, upset_sd=0.30)
    sl = slate[:P.shape[0]]
    sim_correct = (sims == sl[None, :]).sum(axis=1)
    pct = (sim_correct < correct).mean()
    print(f"    simulation said: E[correct]={sim_correct.mean():.2f}, "
          f"E[points]={GROUP_POINTS[sim_correct].mean():.0f}")
    print(f"    the realised {correct} sits at the {pct:.0%} percentile of the "
          f"model's own predictive distribution")
    print(f"    random-slate expectation would be {GROUP_POINTS[3].mean():.0f}-ish "
          f"(E[correct]=3.75)")


if __name__ == "__main__":
    main()
