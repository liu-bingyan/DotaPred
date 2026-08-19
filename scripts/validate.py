"""Honest walk-forward validation with a sealed holdout.

Fixes three sources of optimism in the first pass:

  1. Hyperparameters (and the choice of target variable) were picked by looking
     at the same splits the result was reported on. Here the timeline is cut in
     two: SELECT splits are used for every decision, HOLDOUT splits are scored
     once at the end and nothing is tuned on them.
  2. Test windows overlapped and there were only 4 of them. Here windows are
     monthly and non-overlapping, so every game is tested exactly once, and
     there are 12 of them.
  3. The alias map was built from the full player history, so validating in
     2025 used 2026 rebrand knowledge. `--no-alias` re-runs without it to
     measure how much that was worth.

Paired standard errors are clustered by series, because the 2-3 games of a Bo3
are not independent observations.
"""

import argparse
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

SELECT = [(2025, m) for m in range(4, 11)]           # 2025-04 .. 2025-10
HOLDOUT = [(2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 7)]  # sealed


def ts(y, m):
    return int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp())


def next_month(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def one_split(rows, y, m, target, half_life, l2, min_games):
    cutoff = ts(y, m)
    hi = ts(*next_month(y, m))
    train = [r for r in rows if r["start_time"] < cutoff]
    test = [r for r in rows if cutoff <= r["start_time"] < hi]
    if len(train) < 2000 or len(test) < 50:
        return None

    i, j, w, idx, mm, win, _ = margin.build_design(train, cutoff, half_life, min_games)
    if target == "binary":
        r, _ = rating.fit(i, j, win, w, len(idx), l2=l2)
    else:
        r, _ = margin.fit_margin(i, j, mm["gdpm"] / 400.0, w, len(idx), l2=l2)
    beta = margin.fit_logistic((r[i] - r[j])[:, None], win, w)

    ti, tj, ty, cl = [], [], [], []
    for k in test:
        a, b = k["radiant_team_id"], k["dire_team_id"]
        if a in idx and b in idx:
            ti.append(idx[a])
            tj.append(idx[b])
            ty.append(1.0 if k["radiant_win"] else 0.0)
            cl.append(k.get("series_id") or -k["match_id"])
    if len(ti) < 30:
        return None
    ti, tj, ty = np.array(ti), np.array(tj), np.array(ty)
    p = np.clip(margin.predict_logistic(beta, (r[ti] - r[tj])[:, None]), 1e-9, 1 - 1e-9)
    ll = -(ty * np.log(p) + (1 - ty) * np.log(1 - p))
    return {"ll": ll, "correct": ((p > 0.5) == (ty > 0.5)).astype(float),
            "cluster": np.array(cl), "n": len(ty), "label": f"{y}-{m:02d}"}


def aggregate(per_split):
    ll = np.concatenate([s["ll"] for s in per_split])
    ok = np.concatenate([s["correct"] for s in per_split])
    return ll.mean(), ok.mean(), len(ll)


def clustered_se(diff, cluster):
    """SE of a mean, allowing arbitrary correlation inside a cluster."""
    n = len(diff)
    uniq = {}
    for d, c in zip(diff, cluster):
        uniq.setdefault(c, []).append(d)
    sums = np.array([np.sum(v) for v in uniq.values()])
    g = len(sums)
    mean = diff.mean()
    sizes = np.array([len(v) for v in uniq.values()])
    resid = sums - mean * sizes
    var = (resid**2).sum() / (n**2) * (g / max(g - 1, 1))
    return mean, np.sqrt(var), g


def run(rows, splits, target, half_life, l2, min_games):
    out = []
    for y, m in splits:
        s = one_split(rows, y, m, target, half_life, l2, min_games)
        if s:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-alias", action="store_true")
    ap.add_argument("--stage", choices=["select", "holdout"], default="select")
    args = ap.parse_args()

    aliases = {}
    if not args.no_alias:
        aliases = {int(k): v for k, v in
                   json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    print(f"{len(rows)} top-tier games   alias={'off' if args.no_alias else 'on'}   "
          f"stage={args.stage}\n")

    splits = SELECT if args.stage == "select" else HOLDOUT

    if args.stage == "select":
        print("HYPERPARAMETER SELECTION (these splits may be reused freely)")
        print(f"{'target':>8}{'hl':>6}{'l2':>6}{'min_g':>7}{'logloss':>10}{'acc':>8}{'n':>7}")
        best = None
        for target in ["binary", "gdpm"]:
            for hl in [90, 150, 240]:
                for l2 in ([0.1, 0.3, 1.0] if target == "binary" else [0.3, 1.0, 3.0]):
                    ps = run(rows, splits, target, hl, l2, 20)
                    if not ps:
                        continue
                    ll, acc, n = aggregate(ps)
                    print(f"{target:>8}{hl:>6}{l2:>6}{20:>7}{ll:>10.4f}{acc:>8.4f}{n:>7}")
                    if best is None or ll < best[0]:
                        best = (ll, target, hl, l2)
        print(f"\nselected: target={best[1]} half_life={best[2]} l2={best[3]} "
              f"(select-stage logloss {best[0]:.4f})")
        json.dump({"target": best[1], "half_life": best[2], "l2": best[3], "min_games": 20},
                  open(os.path.join(ROOT, "data", "val_config.json"), "w"), indent=2)
        return

    cfg = json.load(open(os.path.join(ROOT, "data", "val_config.json")))
    print(f"SEALED HOLDOUT -- config chosen on earlier splits only: {cfg}\n")

    res = {}
    for target in ["binary", "gdpm"]:
        ps = run(rows, splits, target, cfg["half_life"], cfg["l2"], cfg["min_games"])
        res[target] = ps
        print(f"-- {target}")
        for s in ps:
            print(f"   {s['label']}  n={s['n']:>4}  logloss={s['ll'].mean():.4f}  "
                  f"acc={s['correct'].mean():.4f}")
        ll, acc, n = aggregate(ps)
        print(f"   POOLED       n={n:>4}  logloss={ll:.4f}  acc={acc:.4f}\n")

    d = np.concatenate([a["ll"] for a in res["binary"]]) - \
        np.concatenate([a["ll"] for a in res["gdpm"]])
    cl = np.concatenate([a["cluster"] for a in res["binary"]])
    mean, se, g = clustered_se(d, cl)
    print(f"paired logloss advantage of gdpm over binary:")
    print(f"  mean = {mean:+.5f}   clustered se = {se:.5f}   t = {mean / se:+.2f}   "
          f"({len(d)} games in {g} series)")
    naive = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  (naive unclustered se would have been {naive:.5f}, t = {mean / naive:+.2f})")


if __name__ == "__main__":
    main()
