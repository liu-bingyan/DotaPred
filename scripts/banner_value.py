"""Value a banner as max2(sum), not sum(EV) -- and enumerate the stat choices.

The scoring rule ranks *games*, not stats: a game's score is the sum over the
three emblems, the top two games of a series are added, and the best series of
the period counts. So the object to optimise is

    E[ max over series ( sum of the top two games of  Σ_s w_s x_s ) ]

which does not decompose per stat. Ranking stats by their own mean is only
correct when every candidate has the same variance and the same covariance with
the rest of the banner; ranking them by their own max2 is strictly wrong, since
Σ max2(x_s) >= max2(Σ x_s) with equality only if all three peak in the same
game -- exactly the combinations we should be avoiding.

The joint distribution never has to be modelled: one historical game is already
one joint draw of all 18 stats, so resampling *whole games* carries the real
correlation structure, including its non-linear parts.

Weights are a parameter throughout. Step one runs them all at 1.0 (no quality,
no trait, no title), which makes same-colour emblems interchangeable and the
choice a set rather than an assignment. Once emblems carry different
multipliers the same machinery ranks ordered assignments instead.

Series structure, measured on 2,144 completed pro Bo3s: 61.8% end 2-0, 38.2%
go the distance. A won series is two wins plus, when it goes three, one loss;
a lost series is the mirror image. Games are drawn from the slot's own winning
and losing games separately, because production is strongly result-conditioned
(core +28%, mid +41%, support -1% in wins).
"""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
from fantasy_model import role_game_scores  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATS = FS.COLOR["red"] + FS.COLOR["blue"] + FS.COLOR["green"]
IDX = {s: i for i, s in enumerate(STATS)}

# emblems per colour, fixed by role, read off the client
NEEDS = {"core": {"red": 2, "green": 1},
         "mid": {"red": 1, "blue": 1, "green": 1},
         "support": {"blue": 2, "green": 1}}
# bucket -> (series played, expected series won)
RECORD = {0: (4, 4), 1: (5, 4), 2: (6, 3.5), 3: (6, 2.5), 4: (5, 1), 5: (4, 0)}
P_THREE = 0.382


def combos(role):
    """Every stat set the role's banner can hold, as tuples of stat names."""
    pools = {c: [s for s in FS.COLOR[c] if s not in FS.UNAVAILABLE]
             for c in ("red", "blue", "green")}
    per_colour = [list(itertools.combinations(pools[c], k))
                  for c, k in NEEDS[role].items()]
    return [tuple(s for group in parts for s in group)
            for parts in itertools.product(*per_colour)]


def game_matrix(rows):
    """(n_games, 18) points, and the win mask. NaN stats become 0."""
    X = np.array([[r["pts"][s] for s in STATS] for r in rows], dtype=float)
    return np.nan_to_num(X), np.array([r["win"] for r in rows], dtype=bool)


def period_values(Vw, Vl, bucket_col, n_sims, rng):
    """E[best series] per combo, given per-game combo values split by result.

    Vw / Vl are (n_games, n_combos). bucket_col is the team's simulated bucket
    in each of n_sims group stages. All combos see identical game draws, so
    they are compared pairwise and the Monte-Carlo noise mostly cancels.
    """
    n_comb = Vw.shape[1]
    out = np.zeros((n_sims, n_comb))

    def draw(pool, k):
        return pool[rng.integers(0, len(pool), k)]

    for bucket in range(6):
        mask = bucket_col == bucket
        k = int(mask.sum())
        if not k:
            continue
        n_series, exp_wins = RECORD[bucket]
        lo = int(np.floor(exp_wins))
        n_wins = lo + (rng.random(k) < exp_wins - lo).astype(int)

        best = np.zeros((k, n_comb))
        for s in range(n_series):
            is_win = (s < n_wins)[:, None]
            same = np.where(is_win, draw(Vw, k) if len(Vw) else draw(Vl, k),
                            draw(Vl, k) if len(Vl) else draw(Vw, k))
            same2 = np.where(is_win, draw(Vw, k) if len(Vw) else draw(Vl, k),
                             draw(Vl, k) if len(Vl) else draw(Vw, k))
            other = np.where(is_win, draw(Vl, k) if len(Vl) else draw(Vw, k),
                             draw(Vw, k) if len(Vw) else draw(Vl, k))
            three = (rng.random(k) < P_THREE)[:, None]
            # top two of {same, same2, other} when the series goes three
            top2 = np.where(three,
                            same + same2 + other
                            - np.minimum(np.minimum(same, same2), other),
                            same + same2)
            best = np.maximum(best, top2)
        out[mask] = best
    return out


def evaluate(rows, bucket_col, stat_sets, weights=None, n_sims=8000, seed=0):
    """-> (n_sims, n_sets) period scores. weights: dict stat -> multiplier."""
    X, win = game_matrix(rows)
    M = np.zeros((len(STATS), len(stat_sets)))
    for j, ss in enumerate(stat_sets):
        for s in ss:
            M[IDX[s], j] = 1.0 if weights is None else weights.get(s, 1.0)
    V = X @ M
    rng = np.random.default_rng(seed)
    return period_values(V[win], V[~win], bucket_col, n_sims, rng)


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    _, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)
    slots = role_game_scores(by_player, roles, teams)

    B = np.load(os.path.join(ROOT, "data", "sim_buckets_boot.npy"))
    names = json.load(open(os.path.join(ROOT, "data", "sim_teams_boot.json")))
    tcol = {n: k for k, n in enumerate(names)}
    rng = np.random.default_rng(31337)
    Bs = B[rng.integers(0, B.shape[0], n_sims)]

    result, agree = {}, {"core": [0, 0], "mid": [0, 0], "support": [0, 0]}
    for (team, role), rows in sorted(slots.items(), key=lambda kv: kv[0][::-1]):
        sets = combos(role)
        P = evaluate(rows, Bs[:, tcol[team]], sets, n_sims=n_sims, seed=7)
        mu = P.mean(axis=0)
        order = np.argsort(-mu)

        # what ranking by each stat's own mean would have picked
        X, _ = game_matrix(rows)
        means = dict(zip(STATS, X.mean(axis=0)))
        ev_pick = []
        for c, k in NEEDS[role].items():
            pool = [s for s in FS.COLOR[c] if s not in FS.UNAVAILABLE]
            ev_pick += sorted(pool, key=lambda s: -means[s])[:k]
        ev_j = {frozenset(s): j for j, s in enumerate(sets)}[frozenset(ev_pick)]
        rank_of_ev = int(np.where(order == ev_j)[0][0]) + 1
        agree[role][0] += rank_of_ev == 1
        agree[role][1] += 1

        best_j = int(order[0])
        result[f"{team}|{role}"] = {
            "best": list(sets[best_j]), "best_mean": float(mu[best_j]),
            "best_p25": float(np.percentile(P[:, best_j], 25)),
            "best_sd": float(P[:, best_j].std()),
            "ev_pick": ev_pick, "ev_mean": float(mu[ev_j]),
            "ev_rank": rank_of_ev, "n_combos": len(sets),
            "gap_pct": float(100 * (mu[best_j] - mu[ev_j]) / mu[ev_j]),
            "top5": [{"stats": list(sets[j]), "mean": float(mu[j])}
                     for j in order[:5]],
        }

    for role in ("core", "mid", "support"):
        print(f"\n=== {role}   （{len(combos(role))} 种组合 / 队）")
        print(f"{'队伍':<16}{'max2(sum) 最优组合':<40}{'E[小组赛]':>10}"
              f"{'p25':>9}{'EV法排名':>9}{'差距':>8}")
        sub = sorted((k for k in result if k.endswith("|" + role)),
                     key=lambda k: -result[k]["best_mean"])
        for k in sub:
            r = result[k]
            print(f"{k.split('|')[0]:<14}{'+'.join(r['best']):<40}"
                  f"{r['best_mean']:>10,.0f}{r['best_p25']:>9,.0f}"
                  f"{r['ev_rank']:>9}{r['gap_pct']:>7.2f}%")
        a, n = agree[role]
        print(f"  EV 法选出的组合恰好最优：{a}/{n} 支队")

    json.dump(result, open(os.path.join(ROOT, "data", "banner_value.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n-> data/banner_value.json")


if __name__ == "__main__":
    main()
