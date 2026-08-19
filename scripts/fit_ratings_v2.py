"""Production ratings v2: fit on gold-difference-per-minute, not win/loss.

Validated in compare_models.py -- against the win/loss baseline, on the same
walk-forward splits, the margin target gives paired per-game log-loss
improvement of +0.0092 (t = +5.15, n = 3373). Small but unambiguous.

Rating units here are margin units, not logits, so a fitted logistic maps
rating differences to win probability.
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
HALF_LIFE, L2 = 150.0, 1.0


def main():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())

    i, j, w, idx, m, win, _ = margin.build_design(rows, now, HALF_LIFE)
    r, h = margin.fit_margin(i, j, m["gdpm"] / 400.0, w, len(idx), l2=L2)

    # calibrate rating difference -> P(win a single game)
    beta = margin.fit_logistic((r[i] - r[j])[:, None], win, w)
    print(f"fit on {len(i)} games / {len(idx)} teams")
    print(f"calibration: P(win) = sigmoid({beta[0]:.3f} * rating_diff + {beta[1]:+.3f})")

    eff = {}
    for k in range(len(i)):
        eff[i[k]] = eff.get(i[k], 0.0) + w[k]
        eff[j[k]] = eff.get(j[k], 0.0) + w[k]

    out = {}
    for name, info in teams.items():
        t = info["team_id"]
        if t in idx:
            out[name] = {"team_id": t, "rating": float(r[idx[t]]),
                         "eff_games": float(eff.get(idx[t], 0.0))}

    order = sorted(out.items(), key=lambda kv: -kv[1]["rating"])
    print(f"\n{'#':>3} {'team':18s} {'rating':>8} {'eff.games':>10}")
    for n, (name, v) in enumerate(order, 1):
        print(f"{n:>3} {name:18s} {v['rating']:>8.3f} {v['eff_games']:>10.1f}")

    names = [k for k, _ in order]
    M = {}
    for a in names:
        M[a] = {}
        for b in names:
            if a == b:
                M[a][b] = 0.5
            else:
                d = np.array([[out[a]["rating"] - out[b]["rating"]]])
                M[a][b] = float(margin.predict_logistic(beta, d)[0])

    print("\nBo3 win probability (row beats column):")
    print(f"{'':18s}" + "".join(f"{b[:6]:>7}" for b in names))
    for a in names:
        cells = "".join(f"{rating.bo_win_prob(M[a][b], 2):>7.2f}" if a != b else f"{'--':>7}"
                        for b in names)
        print(f"{a:18s}{cells}")

    json.dump(out, open(os.path.join(ROOT, "data", "ratings_v2.json"), "w"), indent=2)
    json.dump({"model": "gdpm-margin", "calibration": beta.tolist(), "single_game": M},
              open(os.path.join(ROOT, "data", "win_matrix.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
