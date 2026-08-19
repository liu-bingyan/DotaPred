"""Group-stage simulation that integrates over parameter uncertainty properly.

simulate.py takes the point-estimate ratings and adds a hand-picked Gaussian
shock (upset_sd) of the *same size to every team*. The bootstrap shows that is
wrong in shape: estimation noise ranges from 0.14 for teams with a long clean
record to 0.41 for Team Resilience, which has barely any. A uniform shock
understates uncertainty exactly where the slate is most fragile.

Here each simulated tournament instead draws a whole rating vector from the
bootstrap replicates, so teams carry their own uncertainty and the correlations
between their ratings are preserved.

Usage: simulate_boot.py [n_sims] [extra_sd]
  extra_sd adds an optional uniform shock on top, for genuine future-facing
  uncertainty the bootstrap cannot see (roster moves, patch adaptation, form).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rating  # noqa: E402
from simulate import BUCKETS, check_capacities  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def simulate_from_strengths(S, n_sims, rng, extra_sd=0.0):
    n = S.shape[1]
    out = np.empty((n_sims, n), dtype=np.int8)
    picks = rng.integers(0, S.shape[0], n_sims)

    for s in range(n_sims):
        st = S[picks[s]].copy()
        if extra_sd > 0:
            st = st + rng.normal(0, extra_sd, n)
        d = st[:, None] - st[None, :]
        pg = 1.0 / (1.0 + np.exp(-d))
        # per-game -> Bo3
        Ps = pg**2 * (3 - 2 * pg)

        wins = np.zeros(n, dtype=int)
        losses = np.zeros(n, dtype=int)
        played = [set() for _ in range(n)]
        for _ in range(5):
            active = [t for t in range(n) if wins[t] < 4 and losses[t] < 4]
            groups = {}
            for t in active:
                groups.setdefault((wins[t], losses[t]), []).append(t)
            for _key, members in groups.items():
                pool = list(members)
                rng.shuffle(pool)
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


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    extra_sd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    S = np.load(os.path.join(ROOT, "data", "bootstrap_strengths.npy"))
    names = json.load(open(os.path.join(ROOT, "data", "bootstrap_teams.json")))
    print(f"{S.shape[0]} bootstrap replicates x {S.shape[1]} teams; "
          f"{n_sims} simulations, extra_sd={extra_sd}")
    print("per-team bootstrap sd: " +
          ", ".join(f"{n.split()[-1][:6]}={S[:, k].std():.2f}"
                    for k, n in enumerate(names)))

    rng = np.random.default_rng(20260813)
    buckets = simulate_from_strengths(S, n_sims, rng, extra_sd)
    check_capacities(buckets)
    np.save(os.path.join(ROOT, "data", "sim_buckets_boot.npy"), buckets)
    json.dump(names, open(os.path.join(ROOT, "data", "sim_teams_boot.json"), "w"))

    order = np.argsort(-S.mean(axis=0))
    print(f"\n{'team':18s}" + "".join(f"{b:>9}" for b in BUCKETS) + "   P(playoff)")
    for t in order:
        marg = [(buckets[:, t] == b).mean() for b in range(6)]
        print(f"{names[t]:18s}" + "".join(f"{m:>9.3f}" for m in marg)
              + f"{marg[0] + marg[1] + marg[2]:>13.3f}")


if __name__ == "__main__":
    main()
