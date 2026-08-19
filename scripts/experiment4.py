"""Round 4: weight a team's history by roster continuity, not just by age.

The time decay currently punishes games for being *old*. What actually
invalidates a game is the roster having *changed*. A six-month-old game played
by the same five players is fully informative; last month's game played by a
lineup that has since lost three members is not.

For each team at each cutoff we take its most recent lineup as "current", then
weight every historical game of that team by how many of those five played in
it. Tested against the plain time-decay baseline on the SELECT splits.
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin  # noqa: E402
import models  # noqa: E402
from experiment import player_feature, player_fit, stack_eval, team_design  # noqa: E402
from validate import SELECT, TOP, next_month, ts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def current_rosters(train_rows, lineups):
    """Most recent lineup each team fielded before the cutoff."""
    latest = {}
    for r in sorted(train_rows, key=lambda x: x["start_time"]):
        lu = lineups.get(r["match_id"])
        if not lu:
            continue
        latest[r["radiant_team_id"]] = set(lu[0])
        latest[r["dire_team_id"]] = set(lu[1])
    return latest


def overlap_weights(train_rows, lineups, rosters):
    """Per-game roster overlap for each side, in 0..5."""
    n = len(train_rows)
    ov_r = np.full(n, np.nan)
    ov_d = np.full(n, np.nan)
    for k, r in enumerate(train_rows):
        lu = lineups.get(r["match_id"])
        if not lu:
            continue
        cr = rosters.get(r["radiant_team_id"])
        cd = rosters.get(r["dire_team_id"])
        if cr is not None:
            ov_r[k] = len(cr & set(lu[0]))
        if cd is not None:
            ov_d[k] = len(cd & set(lu[1]))
    return ov_r, ov_d


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    lineups = models.load_lineups()

    res = collections.defaultdict(lambda: [[], []])
    ov_hist = []

    for (y, mo) in SELECT:
        cutoff, hi = ts(y, mo), ts(*next_month(y, mo))
        train = [r for r in rows if r["start_time"] < cutoff]
        test = [r for r in rows if cutoff <= r["start_time"] < hi]
        i, j, w, idx, m, win, tr = team_design(train, cutoff)
        gd = m["gdpm"] / 400.0

        te = [k for k in test if k["radiant_team_id"] in idx and k["dire_team_id"] in idx]
        if len(te) < 30:
            continue
        ti = np.array([idx[k["radiant_team_id"]] for k in te])
        tj = np.array([idx[k["dire_team_id"]] for k in te])
        y_te = np.array([1.0 if k["radiant_win"] else 0.0 for k in te])

        rosters = current_rosters(tr, lineups)
        ov_r, ov_d = overlap_weights(tr, lineups, rosters)
        ov_min = np.fmin(np.nan_to_num(ov_r, nan=5.0), np.nan_to_num(ov_d, nan=5.0))
        ov_hist.append(ov_min)

        r_pl, pidx = player_fit(tr, w, lineups, gd, l2=30.0)
        pf_tr = player_feature(tr, r_pl, pidx, lineups)
        pf_te = player_feature(te, r_pl, pidx, lineups)

        # baseline: time decay only
        r0, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=0.3)
        ll, ok = stack_eval([r0[i] - r0[j], pf_tr], win, w, [r0[ti] - r0[tj], pf_te], y_te)
        res["time decay only (baseline)"][0].append(ll)
        res["time decay only (baseline)"][1].append(ok)

        # roster-continuity weighting, several strengths
        for gamma, label in [(1.0, "overlap^1"), (2.0, "overlap^2"), (4.0, "overlap^4")]:
            cont = (ov_min / 5.0) ** gamma
            w2 = w * np.clip(cont, 0.02, 1.0)
            r2, _ = margin.fit_margin(i, j, gd, w2, len(idx), l2=0.3)
            ll, ok = stack_eval([r2[i] - r2[j], pf_tr], win, w2,
                                [r2[ti] - r2[tj], pf_te], y_te)
            res[f"roster {label}"][0].append(ll)
            res[f"roster {label}"][1].append(ok)

        # hard cut: only games with >=4 of the current five
        w3 = w * (ov_min >= 4)
        if w3.sum() > 0:
            r3, _ = margin.fit_margin(i, j, gd, w3, len(idx), l2=0.3)
            ll, ok = stack_eval([r3[i] - r3[j], pf_tr], win, w3,
                                [r3[ti] - r3[tj], pf_te], y_te)
            res["roster >=4/5 hard cut"][0].append(ll)
            res["roster >=4/5 hard cut"][1].append(ok)

        # roster continuity AND a shorter half-life, in case they interact
        i4, j4, w4, idx4, m4, win4, tr4 = margin.build_design(train, cutoff, 90.0, 20)
        rost4 = current_rosters(tr4, lineups)
        o_r, o_d = overlap_weights(tr4, lineups, rost4)
        o4 = np.fmin(np.nan_to_num(o_r, nan=5.0), np.nan_to_num(o_d, nan=5.0))
        w4b = w4 * np.clip((o4 / 5.0) ** 2, 0.02, 1.0)
        r4, _ = margin.fit_margin(i4, j4, m4["gdpm"] / 400.0, w4b, len(idx4), l2=0.3)
        conv = [idx4.get(t) for t in sorted(idx, key=lambda z: idx[z])]
        if all(c is not None for c in conv):
            cv = np.array(conv)
            ll, ok = stack_eval([r4[cv[i]] - r4[cv[j]], pf_tr], win4 if False else win, w4b,
                                [r4[cv[ti]] - r4[cv[tj]], pf_te], y_te)
            res["roster^2 + 90d half-life"][0].append(ll)
            res["roster^2 + 90d half-life"][1].append(ok)

    allov = np.concatenate(ov_hist)
    print("how much roster churn is actually in the training data?")
    for k in range(6):
        print(f"   games where the weaker side had {k}/5 of its current five: "
              f"{(allov == k).mean():6.1%}")
    print()

    base = np.concatenate(res["time decay only (baseline)"][0])
    print(f"{'variant':<30}{'logloss':>10}{'acc':>9}{'gain':>9}{'t':>7}")
    for name, (lls, oks) in sorted(res.items(), key=lambda kv: np.concatenate(kv[1][0]).mean()):
        L = np.concatenate(lls)
        A = np.concatenate(oks)
        d = base - L
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0.0
        print(f"{name:<30}{L.mean():>10.4f}{A.mean():>9.4f}{d.mean():>+9.4f}{t:>7.2f}")


if __name__ == "__main__":
    main()
