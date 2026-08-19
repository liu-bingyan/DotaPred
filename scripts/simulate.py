"""Monte-Carlo simulator for the TI2026 group stage.

Format (confirmed against the client's node_group definitions):
  16 teams, 5 rounds of Bo3 Swiss, 4 wins advances / 4 losses eliminates.
  The bracket arithmetic is forced -- after round 5 the field is always exactly
      1x(4-0)  2x(4-1)  5x(3-2)  5x(2-3)  2x(1-4)  1x(0-4)
  4-0 and 4-1 go straight to the playoffs; 3-2 and 2-3 play a single Bo3
  elimination round (5 matches, winners advance); 1-4 and 0-4 are out.

Prediction buckets, in the order the client shows them:
  0 = 4-0            (1 slot)
  1 = 4-1            (2 slots)
  2 = elim winner    (5 slots)
  3 = elim loser     (5 slots)
  4 = 1-4            (2 slots)
  5 = 0-4            (1 slot)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rating  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUCKETS = ["4-0", "4-1", "elim-W", "elim-L", "1-4", "0-4"]
CAPACITY = [1, 2, 5, 5, 2, 1]


def load_bo3_matrix():
    wm = json.load(open(os.path.join(ROOT, "data", "win_matrix.json")))
    names = list(wm["single_game"].keys())
    n = len(names)
    P = np.zeros((n, n))
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            if a != b:
                P[a, b] = rating.bo_win_prob(wm["single_game"][na][nb], 2)
    return names, P


def simulate(P, n_sims, rng, upset_sd=0.0):
    """Return (n_sims, 16) array of bucket indices per team.

    `upset_sd` adds a per-simulation Gaussian shock to each team's rating so the
    simulation reflects model uncertainty (form swings, patch reads, jetlag),
    not just game-to-game variance. Without it the sim is overconfident.
    """
    n = P.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        logit_P = np.log(P / (1 - P))
    out = np.empty((n_sims, n), dtype=np.int8)

    for s in range(n_sims):
        if upset_sd > 0:
            shock = rng.normal(0, upset_sd, n)
            lp = logit_P + shock[:, None] - shock[None, :]
            Ps = 1 / (1 + np.exp(-lp))
        else:
            Ps = P

        wins = np.zeros(n, dtype=int)
        losses = np.zeros(n, dtype=int)
        played = [set() for _ in range(n)]

        for _ in range(5):
            active = [t for t in range(n) if wins[t] < 4 and losses[t] < 4]
            groups = {}
            for t in active:
                groups.setdefault((wins[t], losses[t]), []).append(t)
            for key, members in groups.items():
                pool = list(members)
                rng.shuffle(pool)
                # greedy rematch-avoiding pairing within the record group
                pairs, left = [], pool[:]
                while len(left) > 1:
                    a = left.pop(0)
                    pick = next((k for k, b in enumerate(left) if b not in played[a]), 0)
                    b = left.pop(pick)
                    pairs.append((a, b))
                for a, b in pairs:
                    played[a].add(b)
                    played[b].add(a)
                    if rng.random() < Ps[a, b]:
                        wins[a] += 1
                        losses[b] += 1
                    else:
                        wins[b] += 1
                        losses[a] += 1

        bucket = np.empty(n, dtype=np.int8)
        hi = [t for t in range(n) if wins[t] == 3 and losses[t] == 2]
        lo = [t for t in range(n) if wins[t] == 2 and losses[t] == 3]
        for t in range(n):
            if wins[t] == 4 and losses[t] == 0:
                bucket[t] = 0
            elif wins[t] == 4 and losses[t] == 1:
                bucket[t] = 1
            elif wins[t] == 1 and losses[t] == 4:
                bucket[t] = 4
            elif wins[t] == 0 and losses[t] == 4:
                bucket[t] = 5

        rng.shuffle(hi)
        rng.shuffle(lo)
        for a, b in zip(hi, lo):
            if rng.random() < Ps[a, b]:
                bucket[a], bucket[b] = 2, 3
            else:
                bucket[a], bucket[b] = 3, 2

        out[s] = bucket

    return out


def check_capacities(buckets):
    """The format forces exact bucket sizes -- assert the sim respects them."""
    for b, cap in enumerate(CAPACITY):
        counts = (buckets == b).sum(axis=1)
        assert (counts == cap).all(), f"bucket {BUCKETS[b]}: got {set(counts.tolist())}"


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    upset_sd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30

    names, P = load_bo3_matrix()
    rng = np.random.default_rng(20260813)
    print(f"simulating {n_sims} group stages (upset_sd={upset_sd})...")
    buckets = simulate(P, n_sims, rng, upset_sd=upset_sd)
    check_capacities(buckets)

    np.save(os.path.join(ROOT, "data", "sim_buckets.npy"), buckets)
    json.dump(names, open(os.path.join(ROOT, "data", "sim_teams.json"), "w"))

    print(f"\n{'team':18s} " + "".join(f"{b:>9}" for b in BUCKETS) + "   P(playoff)")
    for t, name in enumerate(names):
        marg = [(buckets[:, t] == b).mean() for b in range(6)]
        po = marg[0] + marg[1] + marg[2]
        print(f"{name:18s} " + "".join(f"{m:>9.3f}" for m in marg) + f"{po:>13.3f}")


if __name__ == "__main__":
    main()
