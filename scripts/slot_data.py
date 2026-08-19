"""Build fantasy slot estimates that trade bias against variance honestly.

Two sources of evidence for a slot like "LGD support = Thiolicor + KingJungles":

  CLEAN  games the pair actually played together for this org. Unbiased, but
         several slots have fewer than 100 of these because rosters moved --
         LGD's supports have only played together since 2026-01, Spirit's since
         2026-05, and OG's two never have.

  SYNTH  games each player played apart. Plentiful, but from a different team
         context, so biased by an unknown amount.

Dropping SYNTH leaves slots with 26-60 games and useless variance. Using it raw
imports the bias. So the synthetic estimate gets a weight measured from the
data rather than assumed: slots rich in CLEAN games let us compare the two
estimators directly and see how far off SYNTH runs.

Synthetic pairs are matched on result and game length before averaging. Drawing
the two players' games independently would destroy the within-game correlation,
and since scoring takes a maximum over series, understating variance
understates the score.
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSITION = {"core": (2, 0, 1), "mid": (1, 1, 1), "support": (0, 2, 1)}
DUR_EDGES = [0, 33, 38, 43, 48, 999]


def won(g):
    return (g["player_slot"] < 128) == bool(g["radiant_win"])


def team_of(g):
    return g["radiant_team_id"] if g["player_slot"] < 128 else g["dire_team_id"]


def context(g):
    """Bucket a game by result and length -- the two things that drive both
    players' fantasy output in the same game."""
    d = g["duration"] / 60.0
    return (won(g), int(np.digitize(d, DUR_EDGES)))


def collect(team, role, accs, by_player, ok_ids):
    """Split a slot's evidence into CLEAN (pair, on this org) and per-player pools."""
    per_match = collections.defaultdict(dict)
    for a in accs:
        for g in by_player.get(a, []):
            per_match[g["match_id"]][a] = g

    clean, used = [], set()
    for mid, d in per_match.items():
        if len(d) == len(accs):
            g0 = next(iter(d.values()))
            if team_of(g0) in ok_ids:
                clean.append(d)
                used.add(mid)

    pools = {a: collections.defaultdict(list) for a in accs}
    for a in accs:
        for g in by_player.get(a, []):
            if g["match_id"] in used:
                continue
            pools[a][context(g)].append(g)
    return clean, pools


def score_games(d, stat_set):
    """Average the role's players within one game, as the scoring rule does."""
    vals = []
    for g in d.values():
        p = FS.to_points(FS.game_stats(g))
        vals.append(sum(p[s] for s in stat_set if np.isfinite(p[s])))
    return float(np.mean(vals))


def synth_samples(pools, accs, stat_set, n, rng):
    """Synthetic pair-games: draw each player from the same context bucket."""
    ctxs = [c for c in pools[accs[0]] if all(pools[a].get(c) for a in accs)]
    if not ctxs:
        return np.array([])
    wts = np.array([min(len(pools[a][c]) for a in accs) for c in ctxs], dtype=float)
    wts /= wts.sum()
    out = np.empty(n)
    pick = rng.choice(len(ctxs), size=n, p=wts)
    for k in range(n):
        c = ctxs[pick[k]]
        vals = []
        for a in accs:
            g = pools[a][c][rng.integers(0, len(pools[a][c]))]
            p = FS.to_points(FS.game_stats(g))
            vals.append(sum(p[s] for s in stat_set if np.isfinite(p[s])))
        out[k] = float(np.mean(vals))
    return out


def best_stats(games_scored):
    """Pick the highest-value stat per emblem colour for a slot."""
    return games_scored


def build(min_date="2025-06-01", seed=0):
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    _, by_player = FS.load(min_date=min_date)
    roles = FS.assign_roles(by_player, teams)
    rng = np.random.default_rng(seed)

    out = {}
    for team, r in roles.items():
        if not r:
            continue
        canon = teams[team]["team_id"]
        ok = {canon} | {k for k, v in aliases.items() if v == canon}
        for role in ("core", "mid", "support"):
            accs = r[role]
            clean, pools = collect(team, role, accs, by_player, ok)
            if not clean:
                out[(team, role)] = None
                continue
            # choose stats from the clean games only -- picking them on the
            # synthetic pool would bake its bias into the choice
            byc = {}
            for col, ss in FS.COLOR.items():
                m = {}
                for s in ss:
                    vals = [np.mean([FS.to_points(FS.game_stats(g))[s]
                                     for g in d.values()]) for d in clean]
                    v = float(np.nanmean(vals))
                    if np.isfinite(v):
                        m[s] = v
                byc[col] = sorted(m, key=lambda s: -m[s])
            nr, nb, ng = COMPOSITION[role]
            stat = byc["red"][:nr] + byc["blue"][:nb] + byc["green"][:ng]

            cv = np.array([score_games(d, stat) for d in clean])
            sv = synth_samples(pools, accs, stat, 4000, rng)
            out[(team, role)] = {
                "stats": stat, "clean": cv, "synth": sv,
                "n_clean": len(cv),
                "n_solo": int(np.mean([sum(len(v) for v in pools[a].values())
                                       for a in accs])),
            }
    return out


if __name__ == "__main__":
    data = build()
    print(f"{'slot':<28}{'n_clean':>8}{'n_solo':>8}{'clean均值':>10}{'合成均值':>10}"
          f"{'偏差':>9}{'clean se':>10}")
    rows = []
    for k, v in sorted(data.items()):
        if not v or v["n_clean"] < 5 or not len(v["synth"]):
            print(f"{k[0] + '/' + k[1]:<28}{'--- 数据不足 ---':>20}")
            continue
        c, s = v["clean"], v["synth"]
        se = c.std(ddof=1) / np.sqrt(len(c))
        bias = s.mean() - c.mean()
        rows.append((k, c.mean(), s.mean(), bias, se, v["n_clean"]))
        print(f"{k[0] + '/' + k[1]:<28}{v['n_clean']:>8}{v['n_solo']:>8}"
              f"{c.mean():>10,.0f}{s.mean():>10,.0f}{bias:>+9,.0f}{se:>10,.0f}")

    rich = [r for r in rows if r[5] >= 150]
    if rich:
        b = np.array([r[3] for r in rich])
        cm = np.array([r[1] for r in rich])
        sm = np.array([r[2] for r in rich])
        print(f"\n以 n_clean>=150 的 {len(rich)} 个槽位为标尺：")
        print(f"  合成估计的平均偏差 {b.mean():+,.0f} ({b.mean() / cm.mean():+.1%})，"
              f"标准差 {b.std():,.0f}")
        print(f"  合成 vs 真实 的相关系数 {np.corrcoef(cm, sm)[0, 1]:.3f}")
