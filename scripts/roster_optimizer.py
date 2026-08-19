"""Pick the three teams, accounting for correlation between the slots.

Slot values cannot be added up independently. If one team fills two slots and
that team misses the playoffs, *both* slots lose the entire main-event period
at once. Concentration raises the mean and fattens the downside; diversifying
costs mean but truncates it. Which is right depends on the objective:

  max E[total]      -> concentrate on the best team
  max P(total >= T) -> depends where T sits relative to the distribution

The user's objective is the second kind: they need the 90th percentile overall
to get the Terrain Token, and nothing above it is worth anything extra.

Correlation is preserved by scoring every slot inside the *same* simulated
tournament, so team outcomes move together exactly as they will in reality.
"""

import collections
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
from fantasy_model import REACHES_PLAYOFF, SERIES_IN_GROUP, role_game_scores  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSITION = {"core": (2, 0, 1), "mid": (1, 1, 1), "support": (0, 2, 1)}


def period_samples(vals, n_series, n, rng):
    """Vectorised: n independent periods of `n_series` Bo3s, best series each."""
    if n_series <= 0:
        return np.zeros(n)
    # 3 games per series, but only the top 2 count, so drawing 3 and taking the
    # top 2 covers both the 2-0 and 2-1 cases closely enough
    draws = vals[rng.integers(0, len(vals), size=(n, n_series, 3))]
    top2 = np.sort(draws, axis=2)[:, :, -2:].sum(axis=2)
    return top2.max(axis=1)


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    _, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)
    slots = role_game_scores(by_player, roles, teams)

    buckets = np.load(os.path.join(ROOT, "data", "sim_buckets_boot.npy"))
    sim_names = json.load(open(os.path.join(ROOT, "data", "sim_teams_boot.json")))
    tcol = {n: k for k, n in enumerate(sim_names)}
    rng = np.random.default_rng(2026)

    draw_idx = rng.integers(0, buckets.shape[0], n_sims)
    B = buckets[draw_idx]           # (n_sims, 16) bucket per team, correlated

    slot_keys = sorted(slots)
    V = np.zeros((n_sims, len(slot_keys)))
    chosen_stats = {}

    for si, (team, role) in enumerate(slot_keys):
        rows = slots[(team, role)]
        by_colour = {}
        for col, ss in FS.COLOR.items():
            means = {s: float(np.nanmean([r["pts"][s] for r in rows])) for s in ss}
            by_colour[col] = sorted((s for s, v in means.items() if np.isfinite(v)),
                                    key=lambda s: -means[s])
        nr, nb, ng = COMPOSITION[role]
        stat_set = by_colour["red"][:nr] + by_colour["blue"][:nb] + by_colour["green"][:ng]
        chosen_stats[(team, role)] = stat_set
        vals = np.array([sum(r["pts"][s] for s in stat_set) for r in rows])

        b = B[:, tcol[team]]
        for bucket in range(6):
            mask = b == bucket
            k = int(mask.sum())
            if not k:
                continue
            g = period_samples(vals, SERIES_IN_GROUP[bucket], k, rng)
            if bucket in REACHES_PLAYOFF:
                ns = 1 + rng.binomial(4, 0.45, size=k)
                m = np.zeros(k)
                for u in np.unique(ns):
                    mm = ns == u
                    m[mm] = period_samples(vals, int(u), int(mm.sum()), rng)
            else:
                m = np.zeros(k)
            V[mask, si] = g + m

    idx_of = {k: i for i, k in enumerate(slot_keys)}
    team_names = sorted({t for t, _ in slot_keys})

    rosters = []
    for c, m, s in itertools.product(team_names, repeat=3):
        if (c, "core") not in idx_of or (m, "mid") not in idx_of \
                or (s, "support") not in idx_of:
            continue
        tot = V[:, idx_of[(c, "core")]] + V[:, idx_of[(m, "mid")]] + V[:, idx_of[(s, "support")]]
        rosters.append(((c, m, s), tot.mean(), np.percentile(tot, 10),
                        np.percentile(tot, 25), tot.std(), len({c, m, s})))

    print(f"{len(rosters)} rosters over {n_sims} correlated tournament draws\n")

    def show(title, key, n=12):
        print(f"=== {title}")
        print(f"{'#':>3} {'core':<16}{'mid':<16}{'support':<16}"
              f"{'E[总分]':>10}{'p10':>9}{'p25':>9}{'sd':>8}{'队数':>5}")
        for i, r in enumerate(sorted(rosters, key=key)[:n], 1):
            (c, m, s), mu, p10, p25, sd, nt = r
            print(f"{i:>3} {c:<16}{m:<16}{s:<16}{mu:>10,.0f}{p10:>9,.0f}"
                  f"{p25:>9,.0f}{sd:>8,.0f}{nt:>5}")
        print()

    show("max E[total] -- the EV answer", lambda r: -r[1])
    show("max p10 -- the 'never collapse' answer", lambda r: -r[2])
    show("max p25 -- the threshold answer", lambda r: -r[3])

    # how much does concentrating actually cost or gain?
    print("=== concentration vs diversification")
    for nt in (1, 2, 3):
        sub = [r for r in rosters if r[5] == nt]
        best_mu = max(sub, key=lambda r: r[1])
        best_p25 = max(sub, key=lambda r: r[3])
        print(f"  {nt} 支不同队伍: 最佳 E={best_mu[1]:,.0f} ({'/'.join(best_mu[0])})")
        print(f"{'':17}最佳 p25={best_p25[3]:,.0f} ({'/'.join(best_p25[0])})")

    json.dump({"stats": {f"{k[0]}|{k[1]}": v for k, v in chosen_stats.items()}},
              open(os.path.join(ROOT, "data", "roster_stats.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
