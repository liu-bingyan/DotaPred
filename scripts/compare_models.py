"""Does margin-of-victory actually beat win/loss? Same walk-forward protocol.

Models compared, all fit only on data before each cutoff and scored on the
next 60 days of premium/professional games:

  binary      Bradley-Terry on win/loss                     (the current model)
  gdpm        rating fit on gold-diff per minute
  tanh_gold   rating fit on tanh(final gold adv / 15k)
  rax         rating fit on barracks differential
  composite   rating fit on the blended dominance score
  combined    logistic on [binary rating diff, composite rating diff]

Reports per-split accuracy and log-loss plus BOTH the unweighted and the
n-weighted aggregate, because the splits differ in size by 6x.
"""

import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin  # noqa: E402
import rating  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = {"premium", "professional"}
CUTOFFS = [
    int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp())
    for y, m in [(2025, 9), (2025, 12), (2026, 3), (2026, 6)]
]


def metrics(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (
        -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
        (((p > 0.5) == (y > 0.5)).mean()),
        ((p - y) ** 2).mean(),
    )


def run_split(rows, cutoff, half_life=150, l2_bt=0.3, l2_m=1.0):
    train = [r for r in rows if r["start_time"] < cutoff]
    test = [r for r in rows if cutoff <= r["start_time"] < cutoff + 60 * 86400]
    if len(train) < 1000 or len(test) < 100:
        return None

    i, j, w, idx, m, win, tr = margin.build_design(train, cutoff, half_life)

    # --- ratings from each target
    ratings = {}
    r_bt, h_bt = rating.fit(i, j, win, w, len(idx), l2=l2_bt)
    ratings["binary"] = (r_bt, h_bt)
    targets = {
        "gdpm": m["gdpm"] / 400.0,
        "tanh_gold": m["tanh_gold"],
        "rax": m["rax"] / 6.0,
        "composite": margin.composite(m, win),
    }
    for name, y in targets.items():
        ratings[name] = margin.fit_margin(i, j, y, w, len(idx), l2=l2_m)

    # --- held-out design
    ti, tj, ty = [], [], []
    for r in test:
        a, b = r["radiant_team_id"], r["dire_team_id"]
        if a in idx and b in idx:
            ti.append(idx[a])
            tj.append(idx[b])
            ty.append(1.0 if r["radiant_win"] else 0.0)
    if len(ti) < 100:
        return None
    ti, tj, ty = np.array(ti), np.array(tj), np.array(ty)

    out = {"n": len(ti)}
    # each rating scale is arbitrary -> calibrate diff->P(win) on the TRAIN set only
    for name, (r, h) in ratings.items():
        d_tr = (r[i] - r[j])[:, None]
        beta = margin.fit_logistic(d_tr, win, w)
        p = margin.predict_logistic(beta, (r[ti] - r[tj])[:, None])
        out[name] = metrics(p, ty)

    r_c, _ = ratings["composite"]
    X_tr = np.column_stack([(r_bt[i] - r_bt[j]), (r_c[i] - r_c[j])])
    beta = margin.fit_logistic(X_tr, win, w)
    X_te = np.column_stack([(r_bt[ti] - r_bt[tj]), (r_c[ti] - r_c[tj])])
    out["combined"] = metrics(margin.predict_logistic(beta, X_te), ty)
    return out


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    print(f"{len(rows)} top-tier games\n")

    models = ["binary", "gdpm", "tanh_gold", "rax", "composite", "combined"]
    splits = [run_split(rows, c) for c in CUTOFFS]
    splits = [s for s in splits if s]

    print(f"{'model':<12}" + "".join(f"{dt.datetime.fromtimestamp(c, dt.timezone.utc):%y-%m}"
                                     .rjust(9) for c in CUTOFFS))
    print(f"{'n':<12}" + "".join(f"{s['n']:>9}" for s in splits))
    print("-- accuracy --")
    for mdl in models:
        cells = "".join(f"{s[mdl][1]:>9.4f}" for s in splits)
        ns = np.array([s["n"] for s in splits])
        acc = np.array([s[mdl][1] for s in splits])
        print(f"{mdl:<12}{cells}   unw={acc.mean():.4f}  wtd={(ns * acc).sum() / ns.sum():.4f}")
    print("-- log-loss --")
    for mdl in models:
        cells = "".join(f"{s[mdl][0]:>9.4f}" for s in splits)
        ns = np.array([s["n"] for s in splits])
        ll = np.array([s[mdl][0] for s in splits])
        print(f"{mdl:<12}{cells}   unw={ll.mean():.4f}  wtd={(ns * ll).sum() / ns.sum():.4f}")


if __name__ == "__main__":
    main()
