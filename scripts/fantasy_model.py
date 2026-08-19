"""Expected fantasy score per team-role slot.

Three things have to be combined:

  1. Per-game production. Each player's own games, used as an empirical
     distribution -- not a mean. The scoring rule takes the best two games of a
     series and then the best series of the period, so it is an order statistic
     and the shape of the tail matters more than the average.

  2. How many series the team plays. In the group stage this is *determined* by
     the final bucket:
        4-0 -> 4 series      4-1 -> 5      3-2 / 2-3 -> 5 swiss + 1 elimination
        1-4 -> 5             0-4 -> 4
     In the main event it depends on how far they go, and a team that misses
     the playoffs scores **zero** for that whole period.

  3. Role pairing. Core is the safelane+offlane pair, support the two supports,
     mid a single player. Per game the role's score is the average over its
     players, so pairs are sampled from games they actually played together.

Emblem quality and trait multipliers are deliberately left out: they multiply
every slot alike, so they cannot reorder the teams. This ranks slots; the
banner build is a separate problem.
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# group-stage series count, by prediction bucket index
SERIES_IN_GROUP = {0: 4, 1: 5, 2: 6, 3: 6, 4: 5, 5: 4}
# only buckets 0,1,2 reach the playoffs
REACHES_PLAYOFF = {0, 1, 2}


def role_game_scores(by_player, roles, teams):
    """Per-game role scores, per stat, keyed by (team, role).

    For paired roles the two players' scores are averaged within the same game,
    which is what the scoring rule does. Games where only one of the pair
    appears fall back to that player alone.
    """
    stats = FS.COLOR["red"] + FS.COLOR["blue"] + FS.COLOR["green"]
    out = {}
    for team, r in roles.items():
        if not r:
            continue
        for role in ("core", "mid", "support"):
            accs = r[role]
            # index each player's games by match
            per_match = collections.defaultdict(dict)
            for a in accs:
                for g in by_player.get(a, []):
                    per_match[g["match_id"]][a] = g
            rows = []
            for mid, d in per_match.items():
                pts = [FS.to_points(FS.game_stats(g)) for g in d.values()]
                avg = {s: float(np.nanmean([p[s] for p in pts])) for s in stats}
                any_g = next(iter(d.values()))
                rows.append({"match_id": mid,
                             "series_id": any_g.get("series_id") or -mid,
                             "start": any_g["start_time"],
                             "pts": avg,
                             "win": (any_g["player_slot"] < 128) == bool(
                                 any_g["radiant_win"]),
                             "n_players": len(d)})
            if len(rows) >= 12:
                out[(team, role)] = rows
    return out


def period_score(rows, stat_set, n_series, rng, sum_top2=True):
    """Simulate one period: n_series series, each 2-3 games, best series wins.

    Games are drawn with replacement from the slot's history. Series length is
    drawn 2 or 3 to match the Bo3 format.
    """
    if n_series <= 0 or not rows:
        return 0.0
    vals = np.array([sum(r["pts"][s] for s in stat_set if np.isfinite(r["pts"][s]))
                     for r in rows])
    best = 0.0
    for _ in range(n_series):
        n_games = 2 if rng.random() < 0.55 else 3
        draw = vals[rng.integers(0, len(vals), n_games)]
        top2 = np.sort(draw)[-2:]
        s = top2.sum() if sum_top2 else top2.mean()
        best = max(best, s)
    return best


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    rows_all, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)
    slots = role_game_scores(by_player, roles, teams)
    print(f"{len(slots)} team-role slots with enough history\n")

    buckets = np.load(os.path.join(ROOT, "data", "sim_buckets_boot.npy"))
    sim_names = json.load(open(os.path.join(ROOT, "data", "sim_teams_boot.json")))
    tcol = {n: k for k, n in enumerate(sim_names)}

    # An "ideal" banner: the best stats available in each colour. Reported for
    # several sizes because the real emblem count per role is not yet known.
    rng = np.random.default_rng(4)
    # confirmed from the client: colour layout is fixed per role
    COMPOSITION = {"core": (2, 0, 1), "mid": (1, 1, 1), "support": (0, 2, 1)}

    results = collections.defaultdict(dict)
    for (team, role), rows in slots.items():
        # rank stats by mean value for this slot
        by_colour = {}
        for col, ss in FS.COLOR.items():
            means = {s: float(np.nanmean([r["pts"][s] for r in rows])) for s in ss}
            means = {s: v for s, v in means.items() if np.isfinite(v)}
            by_colour[col] = sorted(means, key=lambda s: -means[s])

        col_idx = tcol.get(team)
        nr, nb, ng = COMPOSITION[role]
        for label in ["actual banner"]:
            stat_set = (by_colour["red"][:nr] + by_colour["blue"][:nb]
                        + by_colour["green"][:ng])
            tot = np.zeros(n_sims)
            for k in range(n_sims):
                b = buckets[rng.integers(0, buckets.shape[0]), col_idx]
                g = period_score(rows, stat_set, SERIES_IN_GROUP[int(b)], rng)
                # main event: zero unless they qualified; depth ~ 1-5 series
                if int(b) in REACHES_PLAYOFF:
                    ns = 1 + rng.binomial(4, 0.45)
                    m = period_score(rows, stat_set, ns, rng)
                else:
                    m = 0.0
                tot[k] = g + m
            results[(team, role)][label] = (tot.mean(), np.percentile(tot, 10),
                                            tot.std(), stat_set)

    for label in ["actual banner"]:
        print(f"=== {label}")
        order = sorted(results, key=lambda k: -results[k][label][0])
        print(f"{'#':>3} {'team':<18}{'role':<9}{'E[两期总分]':>13}{'p10':>10}{'sd':>10}")
        for n, key in enumerate(order, 1):
            mu, p10, sd, ss = results[key][label]
            print(f"{n:>3} {key[0]:<18}{key[1]:<9}{mu:>13,.0f}{p10:>10,.0f}{sd:>10,.0f}  {'+'.join(ss)}")
        print()

    json.dump({f"{k[0]}|{k[1]}": {lbl: list(v[:3]) + [v[3]] for lbl, v in d.items()}
               for k, d in results.items()},
              open(os.path.join(ROOT, "data", "fantasy_slot_values.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
