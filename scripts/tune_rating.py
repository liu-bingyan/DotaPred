"""Walk-forward validation of the Bradley-Terry rating model.

Splits on time: fit on everything before a cutoff, score the games after it.
Reports weighted log-loss / accuracy / Brier on held-out games, plus a
calibration table, so we know whether the win probabilities can be trusted
before we feed them into a tournament simulator.
"""

import datetime as dt
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rating  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


TOP_TIERS = {"premium", "professional"}


def evaluate(rows, cutoff, half_life, l2, min_games=5, test_top_tier_only=True,
             horizon_days=60):
    """Fit on everything before `cutoff`; score the next `horizon_days` of games.

    The held-out set is restricted to premium/professional games by default:
    those are the only ones that resemble what we actually have to predict at
    TI, and the amateur tier is mostly roster-soup noise that drowns the signal.
    """
    train = [r for r in rows if r["start_time"] < cutoff]
    hi = cutoff + horizon_days * 86400
    test = [r for r in rows if cutoff <= r["start_time"] < hi]
    if test_top_tier_only:
        test = [r for r in test if r.get("tier") in TOP_TIERS]
    if not train or not test:
        return None

    i, j, y, w, idx, _ = rating.build_design(
        train, now=cutoff, half_life_days=half_life, min_games=min_games
    )
    r, h = rating.fit(i, j, y, w, len(idx), l2=l2)

    # keep only test games where both teams were rated
    ti, tj, ty = [], [], []
    for m in test:
        a, b = m["radiant_team_id"], m["dire_team_id"]
        if a in idx and b in idx:
            ti.append(idx[a])
            tj.append(idx[b])
            ty.append(1.0 if m["radiant_win"] else 0.0)
    if len(ti) < 100:
        return None
    ti, tj, ty = np.array(ti), np.array(tj), np.array(ty)
    d = r[ti] - r[tj] + h
    p = np.clip(1 / (1 + np.exp(-d)), 1e-9, 1 - 1e-9)
    ll = -(ty * np.log(p) + (1 - ty) * np.log(1 - p)).mean()
    acc = ((p > 0.5) == (ty > 0.5)).mean()
    brier = ((p - ty) ** 2).mean()
    return {"n": len(ti), "logloss": ll, "acc": acc, "brier": brier, "p": p, "y": ty}


def main():
    rows = rating.load_matches()
    print(f"{len(rows)} games loaded\n")

    # three walk-forward cutoffs through the last year
    cutoffs = [
        int(dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc).timestamp()),
        int(dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc).timestamp()),
        int(dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc).timestamp()),
    ]

    grid = list(itertools.product([90, 150, 240, 400], [0.5, 2.0, 6.0, 15.0]))
    print(f"{'half_life':>10} {'l2':>6} {'logloss':>9} {'acc':>7} {'brier':>7}")
    results = []
    for hl, l2 in grid:
        scores = [evaluate(rows, c, hl, l2) for c in cutoffs]
        scores = [s for s in scores if s]
        if not scores:
            continue
        ll = np.mean([s["logloss"] for s in scores])
        acc = np.mean([s["acc"] for s in scores])
        br = np.mean([s["brier"] for s in scores])
        results.append((ll, hl, l2, acc, br))
        print(f"{hl:>10} {l2:>6} {ll:>9.4f} {acc:>7.4f} {br:>7.4f}")

    results.sort()
    ll, hl, l2, acc, br = results[0]
    print(f"\nbest: half_life={hl}d  l2={l2}  logloss={ll:.4f}  acc={acc:.4f}")

    # calibration on the most recent split with the winning config
    s = evaluate(rows, cutoffs[-1], hl, l2)
    print(f"\ncalibration on games after {dt.datetime.utcfromtimestamp(cutoffs[-1]):%Y-%m-%d} "
          f"(n={s['n']})")
    edges = np.array([0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0])
    print(f"{'bucket':>12} {'n':>6} {'pred':>7} {'actual':>7}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s["p"] >= lo) & (s["p"] < hi)
        if mask.sum() > 20:
            print(f"{lo:.2f}-{hi:.2f} {mask.sum():>10} "
                  f"{s['p'][mask].mean():>7.3f} {s['y'][mask].mean():>7.3f}")

    with open(os.path.join(ROOT, "data", "rating_config.json"), "w") as f:
        json.dump({"half_life_days": hl, "l2": l2, "val_logloss": ll, "val_acc": acc}, f, indent=2)


if __name__ == "__main__":
    main()
