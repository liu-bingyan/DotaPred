"""Per-game emblem value for every team-role slot, with no multipliers applied.

The client fixes each banner at three emblems with a colour layout that depends
on the role and cannot be changed:

    core     red  green red
    mid      red  blue  green
    support  blue green blue

So the stat choice is made *within a colour, within a slot*: pick 2 of the 6
red stats for a core, 2 of the 6 blue for a support, 1 of the 6 green for
everyone. This script scores that choice at its base value -- quality, trait
and title multipliers are all left out, because they multiply an emblem's stat
value and so cannot reorder the stats within one emblem.

A slot's score in a game is the average over the players filling it (two for
core and support, one for mid), which is what the scoring rule does.

Caveat worth keeping in mind downstream: the period score is the top two games
of a series, selected on the *sum* over the three emblems. Ranking by marginal
mean is therefore the right first cut but not the final answer -- a stat that
spikes in the same games as the rest of the banner is worth more than its mean
suggests. That interaction belongs to the banner optimiser, not here.

Writes data/emblem_stats.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
from fantasy_model import role_game_scores  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# colours actually present on each role's banner, and how many emblems of each
NEEDS = {"core": {"red": 2, "green": 1},
         "mid": {"red": 1, "blue": 1, "green": 1},
         "support": {"blue": 2, "green": 1}}


def describe(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)), "p75": float(np.percentile(v, 75)),
            "max": float(v.max())}


def main():
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    _, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)
    slots = role_game_scores(by_player, roles, teams)

    out = {}
    for (team, role), rows in slots.items():
        for stat in FS.COLOR["red"] + FS.COLOR["blue"] + FS.COLOR["green"]:
            d = describe([r["pts"][stat] for r in rows])
            out.setdefault(f"{team}|{role}", {})[stat] = d

    order = sorted(slots, key=lambda k: (k[1], k[0]))
    for role in ("core", "mid", "support"):
        for colour, k in NEEDS[role].items():
            stats = FS.COLOR[colour]
            print(f"\n=== {role} / {colour}  （战旗上 {k} 枚）  单位：分/局，均值 ± 标准差")
            print(f"{'队伍':<17}" + "".join(f"{s:>17}" for s in stats))
            best = {}
            for team, r in order:
                if r != role:
                    continue
                row = out[f"{team}|{role}"]
                cells = ""
                vals = []
                for s in stats:
                    d = row.get(s)
                    if d is None:
                        cells += f"{'--':>17}"
                        vals.append(-np.inf)
                    else:
                        cells += f"{d['mean']:>9,.0f}±{d['sd']:<7,.0f}"
                        vals.append(d["mean"])
                print(f"{team:<15}" + cells)
                pick = [stats[i] for i in np.argsort(vals)[::-1][:k]]
                best[team] = pick
            tally = {}
            for p in best.values():
                for s in p:
                    tally[s] = tally.get(s, 0) + 1
            print("  各队最优选择的统计：" +
                  "  ".join(f"{s}×{c}" for s, c in
                            sorted(tally.items(), key=lambda kv: -kv[1])))

    json.dump(out, open(os.path.join(ROOT, "data", "emblem_stats.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n{len(out)} 个槽位 -> data/emblem_stats.json")


if __name__ == "__main__":
    main()
