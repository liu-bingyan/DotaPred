"""Production ratings v3 = team rating + player rating, both on gold margin.

Sealed-holdout performance (2025-11..2026-06, nothing tuned on it):
  single game   62.03%  logloss 0.6559   (win/loss baseline: 60.31% / 0.6644)
  Bo3 series    65.26%

The player term matters most for exactly the teams we care about: five of the
sixteen are playing under an org id with almost no history, and the player
ratings carry those rosters' real track record across the rename.
"""

import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
import margin  # noqa: E402
import models  # noqa: E402
import rating  # noqa: E402
from experiment import player_feature, player_fit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = {"premium", "professional"}
HL, L2T, L2P = 150.0, 0.3, 30.0


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    teams = FS.apply_roster_overrides(
        json.load(open(os.path.join(ROOT, "data", "teams.json"))))
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    lineups = models.load_lineups()
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())

    i, j, w, idx, m, win, tr = margin.build_design(rows, now, HL, 20)
    gd = m["gdpm"] / 400.0
    r_team, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=L2T)
    r_pl, pidx = player_fit(tr, w, lineups, gd, l2=L2P)

    f_tr = np.column_stack([r_team[i] - r_team[j],
                            player_feature(tr, r_pl, pidx, lineups)])
    mu = np.nanmean(f_tr, axis=0)
    beta = margin.fit_logistic(np.where(np.isnan(f_tr), mu, f_tr), win, w)
    print(f"fit on {len(i)} games / {len(idx)} teams / {len(pidx)} players")
    print(f"P(win) = sigmoid({beta[0]:.3f}*team_diff + {beta[1]:.3f}*player_diff "
          f"{beta[2]:+.3f})\n")

    out = {}
    for name, info in teams.items():
        tid = info["team_id"]
        if tid not in idx:
            continue
        accs = [p["account_id"] for p in info["players"][:5]]
        known = [r_pl[pidx[a]] for a in accs if a in pidx]
        out[name] = {
            "team_id": tid,
            "team_rating": float(r_team[idx[tid]]),
            "player_rating": float(np.mean(known)) if len(known) >= 4 else float("nan"),
            "n_rated_players": len(known),
            "players": {p["name"]: (float(r_pl[pidx[p["account_id"]]])
                                    if p["account_id"] in pidx else None)
                        for p in info["players"][:5]},
        }

    # combined strength on the logit scale used for predictions
    med_pl = np.nanmedian([v["player_rating"] for v in out.values()])
    for v in out.values():
        pr = v["player_rating"]
        if not np.isfinite(pr):
            pr = med_pl
            v["player_rating"] = float(pr)
        v["strength"] = float(beta[0] * v["team_rating"] + beta[1] * pr)

    order = sorted(out.items(), key=lambda kv: -kv[1]["strength"])
    print(f"{'#':>3} {'team':18s} {'strength':>9} {'team_r':>8} {'player_r':>9} {'rated':>6}")
    for n, (name, v) in enumerate(order, 1):
        print(f"{n:>3} {name:18s} {v['strength']:>9.3f} {v['team_rating']:>8.3f} "
              f"{v['player_rating']:>9.3f} {v['n_rated_players']:>6}")

    names = [k for k, _ in order]
    M = {a: {} for a in names}
    for a in names:
        for b in names:
            if a == b:
                M[a][b] = 0.5
            else:
                z = out[a]["strength"] - out[b]["strength"] + beta[2] - beta[2]
                M[a][b] = float(1 / (1 + np.exp(-z)))

    print("\nBo3 win probability (row beats column):")
    print(f"{'':18s}" + "".join(f"{b[:6]:>7}" for b in names))
    for a in names:
        print(f"{a:18s}" + "".join(
            f"{rating.bo_win_prob(M[a][b], 2):>7.2f}" if a != b else f"{'--':>7}"
            for b in names))

    json.dump(out, open(os.path.join(ROOT, "data", "ratings_v3.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump({"model": "team+player gold-margin", "single_game": M},
              open(os.path.join(ROOT, "data", "win_matrix.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
