"""Pick the group-stage prediction slate.

The scoring table is strongly convex in the number of correct picks, so the
slate that maximises E[correct] is NOT the slate that maximises E[points].
Convexity rewards *correlated* bets: you want to be very right in one world
rather than a bit right in every world.

This searches the assignment space (16 teams into 1/2/5/5/2/1 slots, ~3.6e8
arrangements) by simulated annealing over pair swaps, against Monte-Carlo
tournament outcomes.

Objectives:
  ev      maximise E[points]
  tail    maximise P(points >= target) -- the right objective if the goal is a
          Top-100 leaderboard finish rather than a good average
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUCKETS = ["4-0", "4-1", "elim-W", "elim-L", "1-4", "0-4"]
CAPACITY = [1, 2, 5, 5, 2, 1]

# DOTA_Predictions26_RewardInfoGroups_*_Reward, indexed by number correct
GROUP_POINTS = np.array(
    [0, 30, 60, 120, 360, 720, 1200, 1800, 2520, 3360, 4320, 5400, 6600,
     7920, 9360, 10920, 12000], dtype=float
)


class Objective:
    def __init__(self, buckets, kind="ev", target=None):
        self.B = buckets
        self.n_sims, self.n_teams = buckets.shape
        self.kind = kind
        self.target = target
        # EQ[t][b] -> boolean vector over sims: did team t land in bucket b?
        self.EQ = [[(buckets[:, t] == b) for b in range(6)] for t in range(self.n_teams)]

    def counts(self, assign):
        c = np.zeros(self.n_sims, dtype=np.int16)
        for t, b in enumerate(assign):
            c += self.EQ[t][b]
        return c

    def value(self, counts):
        pts = GROUP_POINTS[counts]
        if self.kind == "ev":
            return pts.mean()
        return (pts >= self.target).mean()

    def delta_counts(self, counts, u, v, bu, bv):
        return (counts
                + self.EQ[u][bv] + self.EQ[v][bu]
                - self.EQ[u][bu] - self.EQ[v][bv])


def random_assignment(rng, n_teams=16):
    slots = []
    for b, cap in enumerate(CAPACITY):
        slots += [b] * cap
    slots = np.array(slots)
    rng.shuffle(slots)
    return slots


def anneal(obj, rng, iters=60_000, restarts=6, t0=None):
    best_assign, best_val = None, -np.inf
    for _ in range(restarts):
        assign = random_assignment(rng, obj.n_teams)
        counts = obj.counts(assign)
        val = obj.value(counts)
        temp0 = t0 if t0 is not None else max(abs(val) * 0.5, 1e-6)

        for k in range(iters):
            temp = temp0 * (1 - k / iters) ** 2 + 1e-9
            u, v = rng.integers(0, obj.n_teams, 2)
            if assign[u] == assign[v]:
                continue
            bu, bv = assign[u], assign[v]
            new_counts = obj.delta_counts(counts, u, v, bu, bv)
            new_val = obj.value(new_counts)
            if new_val >= val or rng.random() < np.exp((new_val - val) / temp):
                assign = assign.copy()
                assign[u], assign[v] = bv, bu
                counts, val = new_counts, new_val
            if val > best_val:
                best_val, best_assign = val, assign.copy()
    return best_assign, best_val


def greedy_marginal(obj, rng):
    """Baseline: maximise E[correct]. Solved as an assignment problem by
    annealing on the linear objective (exact enough at this size)."""
    marg = np.zeros((obj.n_teams, 6))
    for t in range(obj.n_teams):
        for b in range(6):
            marg[t, b] = obj.EQ[t][b].mean()

    best, best_v = None, -np.inf
    for _ in range(400):
        assign = random_assignment(rng, obj.n_teams)
        improved = True
        while improved:
            improved = False
            for u in range(obj.n_teams):
                for v in range(u + 1, obj.n_teams):
                    bu, bv = assign[u], assign[v]
                    if bu == bv:
                        continue
                    cur = marg[u, bu] + marg[v, bv]
                    alt = marg[u, bv] + marg[v, bu]
                    if alt > cur + 1e-12:
                        assign[u], assign[v] = bv, bu
                        improved = True
        v = sum(marg[t, assign[t]] for t in range(obj.n_teams))
        if v > best_v:
            best_v, best = v, assign.copy()
    return best, best_v


def report(label, names, assign, obj):
    counts = obj.counts(assign)
    pts = GROUP_POINTS[counts]
    print(f"\n=== {label}")
    for b, cap in enumerate(CAPACITY):
        picks = [names[t] for t in range(len(names)) if assign[t] == b]
        print(f"  {BUCKETS[b]:>7} ({cap}): " + ", ".join(picks))
    print(f"  E[correct] = {counts.mean():.2f}   E[points] = {pts.mean():.0f}   "
          f"median = {np.median(pts):.0f}")
    qs = [50, 75, 90, 95, 99]
    print("  points percentiles: " +
          "  ".join(f"p{q}={np.percentile(pts, q):.0f}" for q in qs))
    print(f"  P(>=4320 pts, i.e. 10+ correct) = {(counts >= 10).mean():.4f}")
    return pts


def main():
    buckets = np.load(os.path.join(ROOT, "data", "sim_buckets.npy"))
    names = json.load(open(os.path.join(ROOT, "data", "sim_teams.json")))
    rng = np.random.default_rng(7)
    print(f"{buckets.shape[0]} simulated group stages\n")

    obj_ev = Objective(buckets, "ev")

    g_assign, g_val = greedy_marginal(obj_ev, rng)
    report("BASELINE: maximise E[correct] (the intuitive slate)", names, g_assign, obj_ev)

    a_assign, a_val = anneal(obj_ev, rng)
    report("OPTIMAL: maximise E[points]", names, a_assign, obj_ev)

    obj_tail = Objective(buckets, "tail", target=4320.0)  # 10+ correct
    t_assign, t_val = anneal(obj_tail, rng, t0=0.02)
    report("AGGRESSIVE: maximise P(10+ correct) -- leaderboard play", names, t_assign, obj_ev)

    out = {
        "teams": names,
        "buckets": BUCKETS,
        "slates": {
            "max_expected_correct": [int(x) for x in g_assign],
            "max_expected_points": [int(x) for x in a_assign],
            "max_tail": [int(x) for x in t_assign],
        },
    }
    json.dump(out, open(os.path.join(ROOT, "data", "group_slates.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
