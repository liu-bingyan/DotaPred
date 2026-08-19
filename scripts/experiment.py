"""Compare model variants on the SELECT splits only.

Everything here is development. The sealed HOLDOUT splits are not touched;
whatever wins here gets exactly one holdout run in validate.py.

Deliberately kept shallow. The earlier hand-tuned 5-component composite lost to
a single feature, which is the signal that this data supports simple models --
so each variant adds at most a couple of parameters over the baseline.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin  # noqa: E402
import models  # noqa: E402
from validate import SELECT, TOP, next_month, ts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HL, L2T, L2P = 150.0, 0.3, 8.0
MIN_PLAYER_GAMES = 30


def team_design(train, cutoff):
    return margin.build_design(train, cutoff, HL, min_games=20)


def player_fit(train_rows, w, lineups, target, l2=L2P):
    counts = {}
    usable = []
    for k, r in enumerate(train_rows):
        lu = lineups.get(r["match_id"])
        if not lu:
            continue
        for a in lu[0] + lu[1]:
            counts[a] = counts.get(a, 0) + 1
        usable.append(k)
    keep = {a for a, c in counts.items() if c >= MIN_PLAYER_GAMES}
    idx = {a: k for k, a in enumerate(sorted(keep))}

    rows_idx, yy, ww = [], [], []
    for k in usable:
        lu = lineups[train_rows[k]["match_id"]]
        pl = [idx[a] for a in lu[0] if a in idx]
        mi = [idx[a] for a in lu[1] if a in idx]
        if len(pl) < 4 or len(mi) < 4:
            continue
        rows_idx.append((pl, mi))
        yy.append(target[k])
        ww.append(w[k])
    if len(rows_idx) < 500:
        return None, None
    r, h = models.ridge_ratings(rows_idx, np.array(ww), np.array(yy), len(idx), l2=l2)
    return r, idx


def player_feature(rows, r, idx, lineups):
    """Mean player-rating difference; nan when the lineup is unknown."""
    out = np.full(len(rows), np.nan)
    for k, m in enumerate(rows):
        lu = lineups.get(m["match_id"])
        if not lu:
            continue
        a = [r[idx[x]] for x in lu[0] if x in idx]
        b = [r[idx[x]] for x in lu[1] if x in idx]
        if len(a) >= 4 and len(b) >= 4:
            out[k] = np.mean(a) - np.mean(b)
    return out


def stack_eval(feats_tr, y_tr, w_tr, feats_te, y_te):
    """Fit a logistic stack on train, score on test. nan features -> column mean."""
    Xtr = np.column_stack(feats_tr)
    Xte = np.column_stack(feats_te)
    mu = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr)
    Xte = np.where(np.isnan(Xte), mu, Xte)
    beta = margin.fit_logistic(Xtr, y_tr, w_tr)
    p = np.clip(margin.predict_logistic(beta, Xte), 1e-9, 1 - 1e-9)
    ll = -(y_te * np.log(p) + (1 - y_te) * np.log(1 - p))
    return ll, ((p > 0.5) == (y_te > 0.5)).astype(float)


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    lineups = models.load_lineups()
    phase = models.load_phase()
    print(f"{len(rows)} games | {len(lineups)} lineups | {len(phase)} phase curves\n")

    variants = ["team", "player", "team+player", "team+phase", "team+player+phase"]
    acc = {v: [] for v in variants}
    llv = {v: [] for v in variants}

    for (y, mo) in SELECT:
        cutoff, hi = ts(y, mo), ts(*next_month(y, mo))
        train = [r for r in rows if r["start_time"] < cutoff]
        test = [r for r in rows if cutoff <= r["start_time"] < hi]

        i, j, w, idx, m, win, tr = team_design(train, cutoff)
        gd = m["gdpm"] / 400.0
        r_team, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=L2T)

        e, mid, late, ok = models.phase_targets(tr, phase)
        r_e, _ = margin.fit_margin(i, j, e / 100.0, w, len(idx), l2=L2T)
        r_m, _ = margin.fit_margin(i, j, mid / 200.0, w, len(idx), l2=L2T)
        r_l, _ = margin.fit_margin(i, j, late / 400.0, w, len(idx), l2=L2T)

        r_pl, pidx = player_fit(tr, w, lineups, gd)

        te = [k for k in test if k["radiant_team_id"] in idx and k["dire_team_id"] in idx]
        if len(te) < 30:
            continue
        ti = np.array([idx[k["radiant_team_id"]] for k in te])
        tj = np.array([idx[k["dire_team_id"]] for k in te])
        y_te = np.array([1.0 if k["radiant_win"] else 0.0 for k in te])

        f_tr = {"team": (r_team[i] - r_team[j]),
                "e": (r_e[i] - r_e[j]), "m": (r_m[i] - r_m[j]), "l": (r_l[i] - r_l[j])}
        f_te = {"team": (r_team[ti] - r_team[tj]),
                "e": (r_e[ti] - r_e[tj]), "m": (r_m[ti] - r_m[tj]), "l": (r_l[ti] - r_l[tj])}
        if r_pl is not None:
            f_tr["player"] = player_feature(tr, r_pl, pidx, lineups)
            f_te["player"] = player_feature(te, r_pl, pidx, lineups)

        sets = {
            "team": ["team"],
            "player": ["player"],
            "team+player": ["team", "player"],
            "team+phase": ["team", "e", "m", "l"],
            "team+player+phase": ["team", "player", "e", "m", "l"],
        }
        for name, cols in sets.items():
            if any(c not in f_tr for c in cols):
                continue
            ll, ok_ = stack_eval([f_tr[c] for c in cols], win, w,
                                 [f_te[c] for c in cols], y_te)
            llv[name].append(ll)
            acc[name].append(ok_)

    print(f"{'variant':<20}{'logloss':>10}{'acc':>9}{'n':>8}{'vs team':>10}")
    base_ll = np.concatenate(llv["team"]).mean()
    for v in variants:
        if not llv[v]:
            print(f"{v:<20}{'--':>10}")
            continue
        L = np.concatenate(llv[v])
        A = np.concatenate(acc[v])
        print(f"{v:<20}{L.mean():>10.4f}{A.mean():>9.4f}{len(L):>8}"
              f"{base_ll - L.mean():>+10.4f}")


if __name__ == "__main__":
    main()
