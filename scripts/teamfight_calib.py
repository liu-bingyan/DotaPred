"""How much does Teamfight Participation move when you change its definition?

The stat wins the green emblem in all 48 team-role slots, and the banner model
flips to Roshan or Tormentor once its true value drops to about 80% of what we
feed it. We use OpenDota's number; Valve computes its own server-side. Neither
publishes the rule.

Valve's rule is not recoverable. What is recoverable is the *sensitivity*:
rebuild fights from raw kill/death/assist positions under a grid of thresholds
and see how far the resulting participation spreads. A tight spread means the
input is safe to trust at face value; a wide one means the green emblem is
genuinely undecided and the choice has to be made robust instead.

Fights are built by single-linkage clustering of deaths in time (and optionally
space); a player counts as present if they killed, died or assisted inside the
fight window. That last test is coarser than OpenDota's, which also credits
players who only dealt damage, so the *level* here runs low by construction --
the informative quantity is the spread across thresholds, not the level.
"""

import collections
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    fights = json.load(open(os.path.join(ROOT, "data", "raw",
                                         "stratz_fights.json")))
    od = json.load(open(os.path.join(ROOT, "data", "raw", "od_tfp.json")))
    tfp = {(r["match_id"], r["account_id"]): r["teamfight_participation"]
           for r in od if r.get("teamfight_participation") is not None}
    return [m for m in fights.values() if m], tfp


def cluster(deaths, dt, dr, min_deaths):
    """Single-linkage over deaths -> list of (t0, t1, cx, cy). deaths sorted."""
    out, cur = [], []
    for e in deaths:
        if cur and (e[0] - cur[-1][0] <= dt and
                    (dr is None or
                     np.hypot(e[1] - cur[-1][1], e[2] - cur[-1][2]) <= dr)):
            cur.append(e)
        else:
            if len(cur) >= min_deaths:
                out.append(cur)
            cur = [e]
    if len(cur) >= min_deaths:
        out.append(cur)
    return [(c[0][0], c[-1][0],
             float(np.mean([x[1] for x in c])),
             float(np.mean([x[2] for x in c]))) for c in out]


def participation(match, dt, dr, min_deaths, pad, radius,
                  keys=("killEvents", "deathEvents", "assistEvents")):
    deaths = []
    for p in match["players"]:
        for e in (p["stats"] or {}).get("deathEvents") or []:
            if e.get("positionX") is None:
                continue
            deaths.append((e["time"], e["positionX"], e["positionY"]))
    deaths.sort()
    fights = cluster(deaths, dt, dr, min_deaths)
    if not fights:
        return {}
    out = {}
    for p in match["players"]:
        st = p["stats"] or {}
        ev = []
        for key in keys:
            for e in st.get(key) or []:
                if e.get("positionX") is not None:
                    ev.append((e["time"], e["positionX"], e["positionY"]))
        n = 0
        for t0, t1, cx, cy in fights:
            for t, x, y in ev:
                if t0 - pad <= t <= t1 + pad and (
                        radius is None or np.hypot(x - cx, y - cy) <= radius):
                    n += 1
                    break
        out[p["steamAccountId"]] = n / len(fights)
    return out


def main():
    matches, tfp = load()
    print(f"{len(matches)} 场有事件流，OpenDota 参与率 {len(tfp)} 条\n")

    KEYS = {"击杀/死亡/助攻": ("killEvents", "deathEvents", "assistEvents"),
            "击杀/死亡": ("killEvents", "deathEvents"),
            "仅死亡": ("deathEvents",)}
    grid = [(dt, dr, md, pad, rad, kn)
            for kn in KEYS
            for dt in (10, 20, 30)
            for dr in (None, 40)
            for md in (2, 3)
            for pad, rad in ((10, None), (15, 60))]
    print(f"{'在场判据':>14}{'时间窗':>6}{'空间':>7}{'最少死亡':>9}{'容差':>6}{'半径':>6}"
          f"{'均值':>9}{'与OD相关':>10}{'占OD比':>9}")
    rows = []
    for dt, dr, md, pad, rad, kn in grid:
        mine, theirs = [], []
        for m in matches:
            pp = participation(m, dt, dr, md, pad, rad, KEYS[kn])
            for acc, v in pp.items():
                t = tfp.get((m["id"], acc))
                if t is None:
                    continue
                mine.append(v)
                theirs.append(t)
        if len(mine) < 100:
            continue
        a, b = np.array(mine), np.array(theirs)
        r = float(np.corrcoef(a, b)[0, 1])
        rows.append(a.mean())
        print(f"{kn:>13}{dt:>6}{str(dr):>7}{md:>9}{pad:>6}{str(rad):>6}"
              f"{a.mean():>9.3f}{r:>10.2f}{a.mean()/b.mean():>9.2f}")
    if rows:
        lo, hi = min(rows), max(rows)
        print(f"\n定义敏感区间：{lo:.3f} ~ {hi:.3f}   "
              f"极差 / 中位 = {(hi-lo)/np.median(rows):.1%}")
        print(f"OpenDota 的均值：{np.mean(theirs):.3f}")


if __name__ == "__main__":
    main()
