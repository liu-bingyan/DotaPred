"""Before/after: constant upset shock vs bootstrap-integrated uncertainty.

Optimises a slate under each uncertainty model, then cross-evaluates -- the
slate chosen under A scored under B, and vice versa. If the two slates score
the same under both worlds, the modelling choice did not matter and we should
say so rather than claim an improvement.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_groups import (BUCKETS, CAPACITY, GROUP_POINTS, Objective,  # noqa: E402
                             anneal, greedy_marginal)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def describe(names, assign):
    return {BUCKETS[b]: [names[t] for t in range(len(names)) if assign[t] == b]
            for b in range(6)}


def score(obj, assign):
    c = obj.counts(assign)
    pts = GROUP_POINTS[c]
    return c.mean(), pts.mean(), (c >= 10).mean(), np.percentile(pts, 95)


def main():
    # each simulation wrote its own column order -- reindex both onto one list
    specs = [("constant shock (upset_sd=0.30)", "sim_buckets.npy", "sim_teams.json"),
             ("bootstrap-integrated", "sim_buckets_boot.npy", "sim_teams_boot.json")]
    loaded = []
    for label, bf, tf in specs:
        bp, tp = os.path.join(ROOT, "data", bf), os.path.join(ROOT, "data", tf)
        if os.path.exists(bp) and os.path.exists(tp):
            loaded.append((label, np.load(bp), json.load(open(tp))))
    assert loaded, "no simulations found"
    names = loaded[0][2]
    assert all(sorted(n) == sorted(names) for _, _, n in loaded), "team sets differ"
    worlds = {}
    for label, B_, n_ in loaded:
        perm = [n_.index(t) for t in names]      # columns -> canonical order
        worlds[label] = Objective(B_[:, perm], "ev")

    rng = np.random.default_rng(11)
    slates = {}
    for label, obj in worlds.items():
        a, _ = anneal(obj, rng, iters=40_000, restarts=4)
        g, _ = greedy_marginal(obj, rng)
        # keep whichever of the two is better under its own world
        slates[label] = a if obj.value(obj.counts(a)) >= obj.value(obj.counts(g)) else g

    for label, assign in slates.items():
        print(f"\n=== slate optimised under: {label}")
        for b, cap in enumerate(CAPACITY):
            picks = [names[t] for t in range(len(names)) if assign[t] == b]
            print(f"  {BUCKETS[b]:>7} ({cap}): " + ", ".join(picks))

    print(f"\n{'slate \\ evaluated under':<34}" +
          "".join(f"{w[:26]:>28}" for w in worlds))
    for slabel, assign in slates.items():
        cells = ""
        for wlabel, obj in worlds.items():
            c, pts, p10, p95 = score(obj, assign)
            cells += f"{f'{pts:.0f} pts / {c:.2f} corr':>28}"
        print(f"{slabel[:32]:<34}{cells}")

    print(f"\n{'':34}" + "".join(f"{'P(10+ correct)':>28}" for _ in worlds))
    for slabel, assign in slates.items():
        cells = "".join(f"{score(obj, assign)[2]:>28.4f}" for obj in worlds.values())
        print(f"{slabel[:32]:<34}{cells}")

    keys = list(slates)
    if len(keys) == 2:
        a, b = slates[keys[0]], slates[keys[1]]
        diff = [(names[t], BUCKETS[a[t]], BUCKETS[b[t]])
                for t in range(len(names)) if a[t] != b[t]]
        print("\ndifferences between the two slates:")
        if not diff:
            print("  none -- the uncertainty model does not change the decision")
        for n, x, y in diff:
            print(f"  {n:<18} {x:>7}  ->  {y:>7}")


if __name__ == "__main__":
    main()
