"""Group-stage fantasy roster, with game quality conditioned on the result.

Two corrections over the first attempt:

  1. Only the group stage matters for this decision. The playoff roster is
     picked separately, after the group stage, from teams that already
     qualified -- so "team misses the playoffs" is not a risk this decision
     carries at all.

  2. Game stats are not independent of the result. Measured on this field:
     core production is 28% higher in wins, mid 41% higher, support 1% lower
     -- i.e. supports ward and stack the same whether they win or lose, while
     cores and mids collapse in losses (deaths alone swing 60%+).
     Since scoring takes the *best* series, and won series are made of won
     games, teams that win more are strongly favoured for core and mid, and
     barely at all for support.

Series counts are forced by the final record:
     4-0 -> 4 series (all won)      4-1 -> 5 (4 won)
     elim-W -> 6 (~3.5 won)         elim-L -> 6 (~2.5 won)
     1-4 -> 5 (1 won)               0-4 -> 4 (none won)
"""

import collections
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
from fantasy_model import role_game_scores  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSITION = {"core": (2, 0, 1), "mid": (1, 1, 1), "support": (0, 2, 1)}
# bucket -> (series played, series won)
RECORD = {0: (4, 4), 1: (5, 4), 2: (6, 3.5), 3: (6, 2.5), 4: (5, 1), 5: (4, 0)}


def won(r):
    return (r["player_slot"] < 128) == bool(r["radiant_win"])


def simulate(win_vals, loss_vals, bucket, n, rng):
    """n independent group stages for one slot, returning the best-series score."""
    n_series, exp_wins = RECORD[bucket]
    lo, hi = int(np.floor(exp_wins)), int(np.ceil(exp_wins))
    frac = exp_wins - lo
    best = np.zeros(n)
    n_wins = lo + (rng.random(n) < frac).astype(int)

    def draw(pool, size):
        return pool[rng.integers(0, len(pool), size)] if len(pool) else np.zeros(size)

    for k in range(n_series):
        # is this the k-th series a win for the team?
        is_win = k < n_wins
        # a won series contributes two winning games; a lost series is
        # 0-2 or 1-2, so its best two games are two losses or a win and a loss
        g1 = np.where(is_win, draw(win_vals, n), draw(loss_vals, n))
        second_is_win = is_win | (rng.random(n) < 0.45)
        g2 = np.where(second_is_win, draw(win_vals, n), draw(loss_vals, n))
        best = np.maximum(best, g1 + g2)
    return best


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    _, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)
    slots = role_game_scores(by_player, roles, teams)

    # tag each stored game with the result
    result = {}
    for a, games in by_player.items():
        for g in games:
            result[g["match_id"]] = result.get(g["match_id"]) or won(g)

    B = np.load(os.path.join(ROOT, "data", "sim_buckets_boot.npy"))
    names = json.load(open(os.path.join(ROOT, "data", "sim_teams_boot.json")))
    tcol = {n: k for k, n in enumerate(names)}
    rng = np.random.default_rng(31337)
    Bs = B[rng.integers(0, B.shape[0], n_sims)]

    keys = sorted(slots)
    V = np.zeros((n_sims, len(keys)))
    picks = {}
    for si, (team, role) in enumerate(keys):
        rows = slots[(team, role)]
        byc = {}
        for c, ss in FS.COLOR.items():
            m = {s: float(np.nanmean([r["pts"][s] for r in rows])) for s in ss}
            byc[c] = sorted((s for s, v in m.items() if np.isfinite(v)),
                            key=lambda s: -m[s])
        nr, nb, ng = COMPOSITION[role]
        stat = byc["red"][:nr] + byc["blue"][:nb] + byc["green"][:ng]
        picks[(team, role)] = stat
        vals, isw = [], []
        for r in rows:
            vals.append(sum(r["pts"][s] for s in stat))
            isw.append(bool(result.get(r["match_id"], True)))
        vals = np.array(vals)
        isw = np.array(isw)
        wv, lv = vals[isw], vals[~isw]
        if len(wv) < 5 or len(lv) < 5:
            wv = lv = vals
        b = Bs[:, tcol[team]]
        for bucket in range(6):
            mask = b == bucket
            k = int(mask.sum())
            if k:
                V[mask, si] = simulate(wv, lv, bucket, k, rng)

    io = {k: i for i, k in enumerate(keys)}
    for role in ("core", "mid", "support"):
        sub = [k for k in keys if k[1] == role]
        sub.sort(key=lambda k: -V[:, io[k]].mean())
        print(f"=== {role}")
        print(f"{'#':>3} {'team':<18}{'E[小组赛]':>11}{'p25':>9}{'sd':>8}  统计项")
        for n, k in enumerate(sub, 1):
            v = V[:, io[k]]
            print(f"{n:>3} {k[0]:<18}{v.mean():>11,.0f}{np.percentile(v, 25):>9,.0f}"
                  f"{v.std():>8,.0f}  {'+'.join(picks[k])}")
        print()

    tn = sorted({t for t, _ in keys})
    best = []
    for c, m, s in itertools.product(tn, repeat=3):
        if (c, "core") in io and (m, "mid") in io and (s, "support") in io:
            tot = V[:, io[(c, "core")]] + V[:, io[(m, "mid")]] + V[:, io[(s, "support")]]
            best.append(((c, m, s), tot.mean(), np.percentile(tot, 25), tot.std()))
    print("=== 最优小组赛阵容")
    print(f"{'#':>3} {'core':<17}{'mid':<17}{'support':<17}{'E':>10}{'p25':>9}{'sd':>8}")
    for i, r in enumerate(sorted(best, key=lambda r: -r[1])[:10], 1):
        (c, m, s), mu, p25, sd = r
        print(f"{i:>3} {c:<17}{m:<17}{s:<17}{mu:>10,.0f}{p25:>9,.0f}{sd:>8,.0f}")


if __name__ == "__main__":
    main()
