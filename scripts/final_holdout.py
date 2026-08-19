"""One run on the sealed holdout. Nothing below was tuned on these splits.

Config was frozen on the SELECT splits (2025-04..2025-10):
  target        gold-diff per minute
  half-life     150 days
  team ridge    0.3
  player ridge  30.0, players with >=30 games
  stack         logistic on [team rating diff, mean player rating diff]

Reported: single-game and Bo3-series accuracy, against the win/loss baseline,
with series-clustered standard errors.
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin  # noqa: E402
import models  # noqa: E402
import rating  # noqa: E402
from experiment import player_feature, player_fit, stack_eval, team_design  # noqa: E402
from experiment3 import collect_series  # noqa: E402
from validate import HOLDOUT, TOP, clustered_se, next_month, ts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    lineups = models.load_lineups()

    res = collections.defaultdict(lambda: {"ll": [], "ok": [], "cl": []})
    ser = collections.defaultdict(lambda: {"ok": [], "n": 0})

    for (y, mo) in HOLDOUT:
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
        cl = np.array([k.get("series_id") or -k["match_id"] for k in te])

        r_bin, h_bin = rating.fit(i, j, win, w, len(idx), l2=0.1)
        r_team, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=0.3)
        r_pl, pidx = player_fit(tr, w, lineups, gd, l2=30.0)

        cand = {
            "win/loss baseline": ([r_bin[i] - r_bin[j]], [r_bin[ti] - r_bin[tj]]),
            "gold margin (team)": ([r_team[i] - r_team[j]], [r_team[ti] - r_team[tj]]),
        }
        if r_pl is not None:
            cand["gold margin (team+player)"] = (
                [r_team[i] - r_team[j], player_feature(tr, r_pl, pidx, lineups)],
                [r_team[ti] - r_team[tj], player_feature(te, r_pl, pidx, lineups)],
            )

        betas = {}
        for name, (ftr, fte) in cand.items():
            ll, ok = stack_eval(ftr, win, w, fte, y_te)
            res[name]["ll"].append(ll)
            res[name]["ok"].append(ok)
            res[name]["cl"].append(cl)
            X = np.column_stack(ftr)
            mu = np.nanmean(X, axis=0)
            betas[name] = (margin.fit_logistic(np.where(np.isnan(X), mu, X), win, w), mu)

        # series-level accuracy for the final model. A series has one lineup per
        # side, so the player feature is well defined -- take it from any game.
        name = "gold margin (team+player)" if r_pl is not None else "gold margin (team)"
        b, mu = betas[name]
        lineup_by_pair = {}
        for k in test:
            sid = k.get("series_id")
            if sid and sid not in lineup_by_pair:
                lineup_by_pair[sid] = k
        for a, bb, a_won, nglen, sid in collect_series(test, lambda t: t in idx,
                                                       with_id=True):
            g = lineup_by_pair.get(sid)
            pf = np.nan
            if g is not None and r_pl is not None:
                lu = lineups.get(g["match_id"])
                if lu:
                    xa = [r_pl[pidx[x]] for x in lu[0] if x in pidx]
                    xb = [r_pl[pidx[x]] for x in lu[1] if x in pidx]
                    if len(xa) >= 4 and len(xb) >= 4:
                        pf = np.mean(xa) - np.mean(xb)
                        if g["radiant_team_id"] != a:
                            pf = -pf
            feats = [r_team[idx[a]] - r_team[idx[bb]]]
            if len(b) == 3:
                feats.append(pf)
            X = np.array([feats])
            X = np.where(np.isnan(X), mu, X)
            p = float(margin.predict_logistic(b, X)[0])
            ps = rating.bo_win_prob(p, 2 if nglen <= 3 else 3)
            ser[name]["ok"].append((ps > 0.5) == a_won)
            ser[name]["n"] += 1

    print("SEALED HOLDOUT  (2025-11 .. 2026-06, monthly non-overlapping windows)\n")
    print(f"{'model':<28}{'logloss':>10}{'acc':>9}{'n':>7}")
    for name, d in res.items():
        L = np.concatenate(d["ll"])
        A = np.concatenate(d["ok"])
        print(f"{name:<28}{L.mean():>10.4f}{A.mean():>9.4f}{len(L):>7}")

    base = np.concatenate(res["win/loss baseline"]["ll"])
    cl = np.concatenate(res["win/loss baseline"]["cl"])
    print()
    for name in res:
        if name == "win/loss baseline":
            continue
        d = base - np.concatenate(res[name]["ll"])
        mean, se, g = clustered_se(d, cl)
        print(f"{name} vs baseline: logloss gain {mean:+.5f}  "
              f"clustered se {se:.5f}  t={mean / se:+.2f}  ({g} series)")

    for name, d in ser.items():
        if d["n"]:
            print(f"\nBo3 series accuracy [{name}]: {np.mean(d['ok']):.4f}  (n={d['n']})")


if __name__ == "__main__":
    main()
