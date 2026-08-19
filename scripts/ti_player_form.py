"""TI2026 八强 40 名选手的当前状态：小组赛表现相对自己赛前基线的偏离。

构造与 player_form.py 一致，但数据源换成 data/raw/player_matches.json —— premium
级在 2025-12 之后几乎没有比赛，TI2026 选手的赛前基线只能从 professional 级取。

    python3 scripts/ti_player_form.py
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
from player_form import STATS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TI = 19719
TI_START = 1786924800  # 2026-08-13


def build(rows):
    by = collections.defaultdict(list)
    for r in rows:
        side = 0 if r["player_slot"] < 128 else 1
        by[(r["match_id"], side)].append(r)
    recs = []
    for (mid, side), team in by.items():
        if len(team) != 5:
            continue
        dur = max(team[0]["duration"], 1) / 60.0
        for pos, r in enumerate(sorted(team, key=lambda x: -(x.get("net_worth") or 0))):
            d = dict(account_id=r["account_id"], match_id=mid, pos=pos,
                     start_time=r["start_time"], leagueid=r["leagueid"])
            for k, sign in STATS:
                v = r.get(k) or 0
                d[k] = sign * (v if k in ("gold_per_min", "xp_per_min") else v / dur)
            recs.append(d)
    for pos in range(5):
        sel = [r for r in recs if r["pos"] == pos]
        for k, _ in STATS:
            v = np.array([r[k] for r in sel], dtype=float)
            mu, sd = np.nanmean(v), np.nanstd(v) or 1.0
            for r, x in zip(sel, v):
                r["z_" + k] = (x - mu) / sd
    for r in recs:
        r["perf"] = float(np.mean([r["z_" + k] for k, _ in STATS]))
    return recs


def main():
    teams = FS.apply_roster_overrides(
        json.load(open(os.path.join(ROOT, "data", "teams.json"))))
    br = json.load(open(os.path.join(ROOT, "data", "ti2026_bracket.json")))
    p8 = sorted({b[k] for b in br["playoff"] for k in ("team_1", "team_2") if b[k]},
                key=lambda t: -json.load(
                    open(os.path.join(ROOT, "data", "playoff_probs_hl45.json")))
                ["strength"][t])
    name_of = {}
    for t in p8:
        for p in teams[t]["players"][:5]:
            name_of[p["account_id"]] = (t, p["name"])

    recs = build(json.load(open(os.path.join(ROOT, "data", "raw", "player_matches.json"))))
    by_p = collections.defaultdict(list)
    for r in recs:
        by_p[r["account_id"]].append(r)

    strength = json.load(open(os.path.join(ROOT, "data", "playoff_probs_hl45.json")))["strength"]
    print(f"{'队伍':16s} {'选手':12s} {'基线场次':>6} {'基线':>7} {'TI':>7} {'状态偏离':>8}")
    team_form = collections.defaultdict(list)
    for t in p8:
        for p in teams[t]["players"][:5]:
            acc = p["account_id"]
            base = [r["perf"] for r in by_p.get(acc, [])
                    if TI_START - 365 * 86400 <= r["start_time"] < TI_START]
            ti = [r["perf"] for r in by_p.get(acc, []) if r["leagueid"] == TI]
            if len(base) < 10 or len(ti) < 4:
                print(f"{t:16s} {p['name']:12s} {len(base):>6} {'—':>7} {'—':>7} {'数据不足':>8}")
                continue
            d = float(np.mean(ti) - np.mean(base))
            team_form[t].append(d)
            bar = ("+" * min(int(abs(d) * 10), 12)) if d > 0 else ("-" * min(int(abs(d) * 10), 12))
            print(f"{t:16s} {p['name']:12s} {len(base):>6} {np.mean(base):>+7.2f} "
                  f"{np.mean(ti):>+7.2f} {d:>+8.2f}  {bar}")
        print()

    print(f"{'队伍':16s} {'强度':>7} {'队均状态偏离':>10}  (五人均值)")
    for t in sorted(team_form, key=lambda x: -np.mean(team_form[x])):
        print(f"{t:16s} {strength[t]:>7.2f} {np.mean(team_form[t]):>+10.2f}")


if __name__ == "__main__":
    main()
