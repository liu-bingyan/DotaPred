"""Round 2: tune the player model, try a joint fit, try a multi-timescale blend.

Round 1 said: phase splits hurt, player ratings help a little on top of team
ratings. So this round only explores directions that keep the model small.

  A. player ridge strength / minimum games sweep
  B. joint fit -- team effect and player effects estimated together with
     separate penalties, instead of two models stacked afterwards
  C. multi-timescale -- the same rating at 60/150/400-day half-lives, stacked.
     Form and class are different signals; one half-life has to compromise.
"""

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


def joint_fit(train_rows, i, j, w, y, n_teams, lineups, l2_team=0.3, l2_pl=8.0,
              min_pg=30, iters=2500, lr=0.05):
    """margin ~ (team_i - team_j) + mean(players_i) - mean(players_j)

    One optimisation, two penalties. The team term absorbs org-level effects
    (coaching, prep) that don't belong to any individual player.
    """
    counts = {}
    for r in train_rows:
        lu = lineups.get(r["match_id"])
        if lu:
            for a in lu[0] + lu[1]:
                counts[a] = counts.get(a, 0) + 1
    pidx = {a: k for k, a in enumerate(sorted(a for a, c in counts.items() if c >= min_pg))}
    npl = len(pidx)

    rad, dire, keep = [], [], []
    for k, r in enumerate(train_rows):
        lu = lineups.get(r["match_id"])
        if not lu:
            continue
        a = [pidx[x] for x in lu[0] if x in pidx]
        b = [pidx[x] for x in lu[1] if x in pidx]
        if len(a) < 4 or len(b) < 4:
            continue
        rad.append(a)
        dire.append(b)
        keep.append(k)
    keep = np.array(keep)
    if len(keep) < 500:
        return None
    ii, jj, ww, yy = i[keep], j[keep], w[keep], y[keep]
    n = len(keep)

    rflat = np.array([p for row in rad for p in row])
    rrow = np.array([k for k, row in enumerate(rad) for _ in row])
    rscale = np.array([1.0 / len(row) for row in rad])
    dflat = np.array([p for row in dire for p in row])
    drow = np.array([k for k, row in enumerate(dire) for _ in row])
    dscale = np.array([1.0 / len(row) for row in dire])

    theta = np.zeros(n_teams + npl + 1)
    mom = np.zeros_like(theta)
    vel = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, iters + 1):
        t = theta[:n_teams]
        p = theta[n_teams:-1]
        h = theta[-1]
        pr = np.zeros(n)
        np.add.at(pr, rrow, p[rflat])
        pr *= rscale
        pd = np.zeros(n)
        np.add.at(pd, drow, p[dflat])
        pd *= dscale
        pred = t[ii] - t[jj] + pr - pd + h
        g = ww * (yy - pred)
        grad = np.zeros_like(theta)
        np.add.at(grad, ii, g)
        np.add.at(grad, jj, -g)
        np.add.at(grad, n_teams + rflat, (g * rscale)[rrow])
        np.add.at(grad, n_teams + dflat, -(g * dscale)[drow])
        grad[:n_teams] -= l2_team * t
        grad[n_teams:-1] -= l2_pl * p
        grad[-1] = g.sum()
        mom = b1 * mom + (1 - b1) * grad
        vel = b2 * vel + (1 - b2) * grad * grad
        theta += lr * (mom / (1 - b1**step)) / (np.sqrt(vel / (1 - b2**step)) + eps)
        theta[:n_teams] -= theta[:n_teams].mean()
    return theta[:n_teams], theta[n_teams:-1], pidx, theta[-1]


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    lineups = models.load_lineups()

    results = {}

    def record(name, ll, ok):
        results.setdefault(name, [[], []])
        results[name][0].append(ll)
        results[name][1].append(ok)

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

        r_team, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=0.3)
        base_tr, base_te = r_team[i] - r_team[j], r_team[ti] - r_team[tj]
        ll, ok = stack_eval([base_tr], win, w, [base_te], y_te)
        record("team (baseline)", ll, ok)

        # A. player ridge sweep
        for l2p in [2.0, 8.0, 30.0]:
            for mpg in [15, 30, 60]:
                r_pl, pidx = player_fit(tr, w, lineups, gd, l2=l2p)
                if r_pl is None:
                    continue
                f_tr = player_feature(tr, r_pl, pidx, lineups)
                f_te = player_feature(te, r_pl, pidx, lineups)
                ll, ok = stack_eval([base_tr, f_tr], win, w, [base_te, f_te], y_te)
                record(f"team+player l2={l2p} mpg={mpg}", ll, ok)
                break  # min-games is baked into player_fit; sweep l2 only

        # B. joint fit
        for l2p in [4.0, 15.0]:
            jf = joint_fit(tr, i, j, w, gd, len(idx), lineups, l2_pl=l2p)
            if jf is None:
                continue
            t_r, p_r, pidx, _ = jf

            def jfeat(rws, ii_, jj_):
                out = np.zeros(len(rws))
                for k, mm in enumerate(rws):
                    lu = lineups.get(mm["match_id"])
                    pa = pb = 0.0
                    if lu:
                        a = [p_r[pidx[x]] for x in lu[0] if x in pidx]
                        b = [p_r[pidx[x]] for x in lu[1] if x in pidx]
                        if len(a) >= 4 and len(b) >= 4:
                            pa, pb = np.mean(a), np.mean(b)
                    out[k] = t_r[ii_[k]] - t_r[jj_[k]] + pa - pb
                return out

            ll, ok = stack_eval([jfeat(tr, i, j)], win, w, [jfeat(te, ti, tj)], y_te)
            record(f"joint l2_pl={l2p}", ll, ok)

        # C. multi-timescale
        extra_tr, extra_te = [base_tr], [base_te]
        for hl in [60.0, 400.0]:
            i2, j2, w2, idx2, m2, win2, tr2 = margin.build_design(train, cutoff, hl, 20)
            r2, _ = margin.fit_margin(i2, j2, m2["gdpm"] / 400.0, w2, len(idx2), l2=0.3)
            conv = {t: idx2.get(t) for t in idx}
            f_tr = np.array([r2[conv[k["radiant_team_id"]]] - r2[conv[k["dire_team_id"]]]
                             if conv.get(k["radiant_team_id"]) is not None
                             and conv.get(k["dire_team_id"]) is not None else np.nan
                             for k in tr])
            f_te = np.array([r2[conv[k["radiant_team_id"]]] - r2[conv[k["dire_team_id"]]]
                             if conv.get(k["radiant_team_id"]) is not None
                             and conv.get(k["dire_team_id"]) is not None else np.nan
                             for k in te])
            extra_tr.append(f_tr)
            extra_te.append(f_te)
        ll, ok = stack_eval(extra_tr, win, w, extra_te, y_te)
        record("multi-timescale (60/150/400)", ll, ok)

        # C2. multi-timescale + player
        r_pl, pidx = player_fit(tr, w, lineups, gd, l2=8.0)
        if r_pl is not None:
            ll, ok = stack_eval(extra_tr + [player_feature(tr, r_pl, pidx, lineups)],
                                win, w,
                                extra_te + [player_feature(te, r_pl, pidx, lineups)], y_te)
            record("multi-timescale + player", ll, ok)

    base = np.concatenate(results["team (baseline)"][0])
    print(f"{'variant':<32}{'logloss':>10}{'acc':>9}{'gain':>9}{'t':>7}")
    for name, (lls, oks) in sorted(results.items(), key=lambda kv: np.concatenate(kv[1][0]).mean()):
        L = np.concatenate(lls)
        A = np.concatenate(oks)
        d = base - L
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0
        print(f"{name:<32}{L.mean():>10.4f}{A.mean():>9.4f}{d.mean():>+9.4f}{t:>7.2f}")


if __name__ == "__main__":
    main()
