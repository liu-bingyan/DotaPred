"""How much of the TI16 ranking is real, and how much is sampling noise?

Poisson bootstrap clustered by series: each series gets a multiplier drawn from
Poisson(1) and the whole model is refit. Clustering matters -- the 2-3 games of
a Bo3 are one observation of a matchup, not three.

Reports, per team, the distribution over final rank, and how often each slot of
the recommended slate survives a resample. Slots that flip on half the
replicates are coin flips being dressed up as predictions.

Caveat: this measures estimation noise in the match record only. It says
nothing about roster moves, patch adaptation or form between now and August.
Real uncertainty is strictly larger than what comes out of here.
"""

import collections
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
import margin  # noqa: E402
import models  # noqa: E402
from experiment import player_feature  # noqa: E402
from optimize_groups import BUCKETS, CAPACITY  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = {"premium", "professional"}
HL, L2T, L2P = 150.0, 0.3, 30.0
MIN_PLAYER_GAMES = 30
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ITERS = 1200


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    teams = FS.apply_roster_overrides(
        json.load(open(os.path.join(ROOT, "data", "teams.json"))))
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    lineups = models.load_lineups()
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())

    i, j, w, idx, m, win, tr = margin.build_design(rows, now, HL, 20)
    gd = m["gdpm"] / 400.0

    # cluster id per training game
    cl = np.array([r.get("series_id") or -r["match_id"] for r in tr])
    uniq, cl_pos = np.unique(cl, return_inverse=True)
    print(f"{len(tr)} games in {len(uniq)} series; {B} bootstrap replicates")

    # player design, built once
    counts = collections.Counter()
    for r in tr:
        lu = lineups.get(r["match_id"])
        if lu:
            for a in lu[0] + lu[1]:
                counts[a] += 1
    pidx = {a: k for k, a in enumerate(sorted(a for a, c in counts.items()
                                              if c >= MIN_PLAYER_GAMES))}
    prows, pkeep = [], []
    for k, r in enumerate(tr):
        lu = lineups.get(r["match_id"])
        if not lu:
            continue
        a = [pidx[x] for x in lu[0] if x in pidx]
        b = [pidx[x] for x in lu[1] if x in pidx]
        if len(a) >= 4 and len(b) >= 4:
            prows.append((a, b))
            pkeep.append(k)
    pkeep = np.array(pkeep)

    tids = {n: v["team_id"] for n, v in teams.items() if v["team_id"] in idx}
    accs = {n: [p["account_id"] for p in teams[n]["players"][:5]] for n in tids}
    names = list(tids)

    rank_counts = {n: np.zeros(16) for n in names}
    strengths = {n: [] for n in names}
    slates = collections.Counter()
    rng = np.random.default_rng(20260813)

    for b in range(B):
        mult = rng.poisson(1.0, size=len(uniq))[cl_pos]
        wb = w * mult
        if wb.sum() <= 0:
            continue
        r_team, _ = margin.fit_margin(i, j, gd, wb, len(idx), l2=L2T, iters=ITERS)
        r_pl, _ = models.ridge_ratings(prows, wb[pkeep], gd[pkeep], len(pidx),
                                       l2=L2P, iters=ITERS)

        X = np.column_stack([r_team[i] - r_team[j],
                             player_feature(tr, r_pl, pidx, lineups)])
        mu = np.nanmean(X, axis=0)
        beta = margin.fit_logistic(np.where(np.isnan(X), mu, X), win, wb, iters=800)

        s = {}
        for n in names:
            known = [r_pl[pidx[a]] for a in accs[n] if a in pidx]
            pr = np.mean(known) if len(known) >= 4 else mu[1]
            s[n] = beta[0] * r_team[idx[tids[n]]] + beta[1] * pr
        order = sorted(names, key=lambda n: -s[n])
        for k, n in enumerate(order):
            rank_counts[n][k] += 1
            strengths[n].append(s[n])
        slates[tuple(order)] += 1
        if (b + 1) % 25 == 0:
            print(f"  {b + 1}/{B}", flush=True)

    # dump the replicate strength vectors so the simulator can integrate over
    # them instead of adding a made-up constant shock to every team alike
    S = np.array([[strengths[n][b] for n in names] for b in range(len(strengths[names[0]]))])
    np.save(os.path.join(ROOT, "data", "bootstrap_strengths.npy"), S)
    json.dump(names, open(os.path.join(ROOT, "data", "bootstrap_teams.json"), "w"))

    tot = sum(rank_counts[names[0]])
    print(f"\nrank distribution over {int(tot)} replicates (%)")
    hdr = "".join(f"{k + 1:>4}" for k in range(16))
    print(f"{'team':<18}{hdr}   {'P(top3)':>8}{'P(bot3)':>8}")
    mean_rank = {n: float(np.dot(rank_counts[n], np.arange(1, 17)) / tot) for n in names}
    for n in sorted(names, key=lambda x: mean_rank[x]):
        cells = "".join(f"{100 * rank_counts[n][k] / tot:>4.0f}" if rank_counts[n][k] else
                        f"{'.':>4}" for k in range(16))
        p3 = 100 * rank_counts[n][:3].sum() / tot
        pb = 100 * rank_counts[n][13:].sum() / tot
        print(f"{n:<18}{cells}   {p3:>7.0f}%{pb:>7.0f}%")

    print(f"\n{'team':<18}{'mean rank':>10}{'strength':>10}{'boot sd':>9}")
    for n in sorted(names, key=lambda x: mean_rank[x]):
        arr = np.array(strengths[n])
        print(f"{n:<18}{mean_rank[n]:>10.2f}{arr.mean():>10.3f}{arr.std():>9.3f}")

    # slot stability of the recommended slate
    slate_map = []
    for b_, cap in enumerate(CAPACITY):
        slate_map += [b_] * cap
    modal = max(slates.items(), key=lambda kv: kv[1])[0]
    print(f"\nmost common ordering appeared {slates[modal]}/{int(tot)} times "
          f"({100 * slates[modal] / tot:.0f}%) -- {len(slates)} distinct orderings seen")

    print(f"\nslot stability of the recommended slate")
    print(f"{'slot':<9}{'pick':<18}{'P(this team lands in this slot across replicates)':>10}")
    base_order = sorted(names, key=lambda x: mean_rank[x])
    for k, n in enumerate(base_order):
        slot = slate_map[k]
        lo = sum(CAPACITY[:slot])
        hi = lo + CAPACITY[slot]
        p = 100 * rank_counts[n][lo:hi].sum() / tot
        print(f"{BUCKETS[slot]:<9}{n:<18}{p:>9.0f}%")


if __name__ == "__main__":
    main()
