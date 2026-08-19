"""Round 3: two things that matter for the actual task rather than for
single-game accuracy.

  A. Series mapping. The simulator currently converts a per-game probability to
     a Bo3 probability with p^2(3-2p), which assumes the games are independent.
     They are not -- same opponent, same prep, same day, live draft adaptation.
     If games are positively correlated the true series probability is less
     extreme than the binomial formula says, and the simulator is overconfident
     about the whole tournament. Fit the mapping empirically instead.

  B. Patch weighting. TI is played on a fresh patch. Games from the current
     patch may deserve more weight than the exponential time decay alone gives
     them.
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


def collect_series(rows, idx_ok, with_id=False):
    """Group games into clean series: same two teams, decisive."""
    S = collections.defaultdict(list)
    for k in rows:
        sid = k.get("series_id")
        if not sid:
            continue
        S[sid].append(k)
    out = []
    for sid, gs in S.items():
        pairs = {frozenset((g["radiant_team_id"], g["dire_team_id"])) for g in gs}
        if len(pairs) != 1:
            continue
        pr = next(iter(pairs))
        if len(pr) != 2:
            continue
        a, b = tuple(pr)
        if not (idx_ok(a) and idx_ok(b)):
            continue
        wa = sum(1 for g in gs if (g["radiant_team_id"] == a) == g["radiant_win"])
        wb = len(gs) - wa
        if wa == wb:
            continue
        out.append((a, b, wa > wb, len(gs), sid) if with_id
                   else (a, b, wa > wb, len(gs)))
    return out


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    lineups = models.load_lineups()

    ser_logit, ser_y, ser_len = [], [], []
    patch_res = collections.defaultdict(lambda: [[], []])

    for (y, mo) in SELECT:
        cutoff, hi = ts(y, mo), ts(*next_month(y, mo))
        train = [r for r in rows if r["start_time"] < cutoff]
        test = [r for r in rows if cutoff <= r["start_time"] < hi]
        i, j, w, idx, m, win, tr = team_design(train, cutoff)
        gd = m["gdpm"] / 400.0
        r_team, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=0.3)
        r_pl, pidx = player_fit(tr, w, lineups, gd, l2=30.0)

        f_tr = [r_team[i] - r_team[j], player_feature(tr, r_pl, pidx, lineups)]
        Xtr = np.column_stack(f_tr)
        mu = np.nanmean(Xtr, axis=0)
        beta = margin.fit_logistic(np.where(np.isnan(Xtr), mu, Xtr), win, w)

        # --- A: series mapping, using out-of-sample games only
        def game_p(a, b):
            t = r_team[idx[a]] - r_team[idx[b]]
            pl = 0.0
            X = np.array([[t, pl]])
            X = np.where(np.isnan(X), mu, X)
            return float(margin.predict_logistic(beta, X)[0])

        for a, b, a_won, nglen in collect_series(test, lambda t: t in idx):
            p = game_p(a, b)
            p = min(max(p, 1e-4), 1 - 1e-4)
            ser_logit.append(np.log(p / (1 - p)))
            ser_y.append(1.0 if a_won else 0.0)
            ser_len.append(nglen)

        # --- B: patch weighting (proxy: extra weight on the most recent 45 days)
        for boost, label in [(1.0, "none"), (2.0, "2x recent-45d"), (4.0, "4x recent-45d")]:
            w2 = w.copy()
            recent = np.array([r["start_time"] > cutoff - 45 * 86400 for r in tr])
            w2[recent] *= boost
            r2, _ = margin.fit_margin(i, j, gd, w2, len(idx), l2=0.3)
            te = [k for k in test if k["radiant_team_id"] in idx and k["dire_team_id"] in idx]
            if len(te) < 30:
                continue
            ti = np.array([idx[k["radiant_team_id"]] for k in te])
            tj = np.array([idx[k["dire_team_id"]] for k in te])
            y_te = np.array([1.0 if k["radiant_win"] else 0.0 for k in te])
            ll, ok = stack_eval([r2[i] - r2[j]], win, w2, [r2[ti] - r2[tj]], y_te)
            patch_res[label][0].append(ll)
            patch_res[label][1].append(ok)

    print("=== B. patch / recency boost")
    print(f"{'weighting':<18}{'logloss':>10}{'acc':>9}")
    for k, (lls, oks) in patch_res.items():
        print(f"{k:<18}{np.concatenate(lls).mean():>10.4f}{np.concatenate(oks).mean():>9.4f}")

    print("\n=== A. game probability -> series probability")
    L = np.array(ser_logit)
    Y = np.array(ser_y)
    N = np.array(ser_len)
    print(f"{len(L)} decisive series")

    # theoretical Bo3 curve implies a slope > 1 on the logit scale
    beta_s = margin.fit_logistic(L[:, None], Y)
    print(f"  empirical fit : logit(P_series) = {beta_s[0]:.3f} * logit(p_game) "
          f"{beta_s[1]:+.3f}")

    p = 1 / (1 + np.exp(-L))
    theo = p**2 * (3 - 2 * p)
    theo = np.clip(theo, 1e-6, 1 - 1e-6)
    emp = np.clip(margin.predict_logistic(beta_s, L[:, None]), 1e-6, 1 - 1e-6)
    for name, q in [("independent-games p^2(3-2p)", theo), ("empirical mapping", emp)]:
        ll = -(Y * np.log(q) + (1 - Y) * np.log(1 - q)).mean()
        print(f"  {name:<30} logloss={ll:.4f}  acc={((q > .5) == (Y > .5)).mean():.4f}")

    # implied slope of the theoretical curve, for comparison
    lo, hi_ = 0.35, 0.65
    tl = np.log((lo**2 * (3 - 2 * lo)) / (1 - lo**2 * (3 - 2 * lo)))
    th = np.log((hi_**2 * (3 - 2 * hi_)) / (1 - hi_**2 * (3 - 2 * hi_)))
    gl = np.log(lo / (1 - lo))
    gh = np.log(hi_ / (1 - hi_))
    print(f"  (independence implies a slope of ~{(th - tl) / (gh - gl):.2f} near even matchups)")


if __name__ == "__main__":
    main()
