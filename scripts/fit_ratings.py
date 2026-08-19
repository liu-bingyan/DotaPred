"""Fit the final team ratings for the TI2026 field and dump the win matrix.

Writes:
  data/ratings.json       -- rating per TI team (+ its Elo-scale equivalent)
  data/win_matrix.json    -- P(row team wins a single game vs column team)
"""

import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rating  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HALF_LIFE = 150.0
L2 = 0.3
MIN_GAMES = 20
ELO = 400 / np.log(10)


def main():
    aliases = {int(k): v for k, v in json.load(
        open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))

    rows = rating.load_matches(alias=aliases)
    rows = [r for r in rows if r.get("tier") in {"premium", "professional"}]
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())

    i, j, y, w, idx, _ = rating.build_design(
        rows, now=now, half_life_days=HALF_LIFE, min_games=MIN_GAMES
    )
    r, h = rating.fit(i, j, y, w, len(idx), l2=L2)
    print(f"fit on {len(i)} games, {len(idx)} teams, radiant advantage = {h:+.3f} logit "
          f"({1 / (1 + np.exp(-h)):.3f} win rate on even matchups)\n")

    # effective sample size behind each TI team's rating
    eff = {}
    for k in range(len(i)):
        eff[i[k]] = eff.get(i[k], 0.0) + w[k]
        eff[j[k]] = eff.get(j[k], 0.0) + w[k]

    out = {}
    for name, info in teams.items():
        tid = info["team_id"]
        if tid not in idx:
            print(f"  !! {name} ({tid}) not rated -- too few games")
            continue
        k = idx[tid]
        out[name] = {
            "team_id": tid,
            "rating": float(r[k]),
            "elo": float(1500 + r[k] * ELO),
            "eff_games": float(eff.get(k, 0.0)),
        }

    order = sorted(out.items(), key=lambda kv: -kv[1]["rating"])
    print(f"{'#':>3} {'team':18s} {'rating':>8} {'elo':>7} {'eff.games':>10}")
    for n, (name, v) in enumerate(order, 1):
        print(f"{n:>3} {name:18s} {v['rating']:>8.3f} {v['elo']:>7.0f} {v['eff_games']:>10.1f}")

    # single-game win matrix, side-neutral (average of playing radiant and dire)
    names = [k for k, _ in order]
    M = {}
    for a in names:
        M[a] = {}
        for b in names:
            if a == b:
                M[a][b] = 0.5
                continue
            d = out[a]["rating"] - out[b]["rating"]
            pa = 1 / (1 + np.exp(-(d + h)))
            pb = 1 / (1 + np.exp(-(d - h)))
            M[a][b] = float((pa + pb) / 2)

    print("\nBo3 win probability (row beats column):")
    hdr = "".join(f"{b[:6]:>7}" for b in names)
    print(f"{'':18s}{hdr}")
    for a in names:
        cells = "".join(
            f"{rating.bo_win_prob(M[a][b], 2):>7.2f}" if a != b else f"{'--':>7}"
            for b in names
        )
        print(f"{a:18s}{cells}")

    json.dump(out, open(os.path.join(ROOT, "data", "ratings.json"), "w"), indent=2)
    json.dump({"radiant_adv": float(h), "single_game": M},
              open(os.path.join(ROOT, "data", "win_matrix.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
