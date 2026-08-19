"""队伍评级半衰期的头对头：只在「跨断点预测淘汰赛」这一个任务上比。

背景：两个测试集给了相反的答案。
  weighting_backtest 的测试集 B（赛事后半程，158 系列赛）  -> HL=30 最好
  break_effect --hl-sweep（断点之后，534 系列赛，仅队伍级）-> HL=45 最好
两者的差距都在噪声内，但产出的填法差 3 票，所以必须判一次。

这里用**完整的生产模型**（队伍 + 选手堆叠，选手半衰期固定 150 天，即回测选中的
那一行配置），测试集只取断点之后的系列赛 —— TI 淘汰赛的真实处境。选手评级只依赖
选手半衰期和截止时间，所以每个赛事只拟合一次，四个队伍半衰期共用。

    python3 scripts/postbreak_headtohead.py --premium
"""

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import break_effect as BE  # noqa: E402
import margin  # noqa: E402
import models  # noqa: E402
from experiment import MIN_PLAYER_GAMES, player_feature  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER_HL, L2T, L2P = 150.0, 0.3, 30.0


def fit_stack(train, cutoff, hl, lineups, player_cache):
    """返回 (r_team, idx, r_pl, pidx, beta, mu)。选手部分按 cutoff 缓存复用。"""
    i, j, w, idx, m, win, tr = margin.build_design(train, cutoff, hl, 20)
    gd = m["gdpm"] / 400.0
    r_team, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=L2T)

    if cutoff not in player_cache:
        _, _, wp, _, mp, _, trp = margin.build_design(train, cutoff, PLAYER_HL, 20)
        counts = collections.Counter()
        for r in trp:
            lu = lineups.get(r["match_id"])
            if lu:
                for a in lu[0] + lu[1]:
                    counts[a] += 1
        pidx = {a: k for k, a in enumerate(sorted(a for a, c in counts.items()
                                                  if c >= MIN_PLAYER_GAMES))}
        prows, pkeep = [], []
        for k, r in enumerate(trp):
            lu = lineups.get(r["match_id"])
            if not lu:
                continue
            a = [pidx[x] for x in lu[0] if x in pidx]
            b = [pidx[x] for x in lu[1] if x in pidx]
            if len(a) >= 4 and len(b) >= 4:
                prows.append((a, b))
                pkeep.append(k)
        if len(prows) < 500:
            player_cache[cutoff] = (None, None)
        else:
            r_pl, _ = models.ridge_ratings(prows, wp[np.array(pkeep)],
                                           (mp["gdpm"] / 400.0)[np.array(pkeep)],
                                           len(pidx), l2=L2P)
            player_cache[cutoff] = (r_pl, pidx)
    r_pl, pidx = player_cache[cutoff]

    cols = [r_team[i] - r_team[j]]
    if r_pl is not None:
        cols.append(player_feature(tr, r_pl, pidx, lineups))
    X = np.column_stack(cols)
    mu = np.nanmean(X, axis=0)
    beta = margin.fit_logistic(np.where(np.isnan(X), mu, X), win, w)
    return r_team, idx, r_pl, pidx, beta, mu


def series_prob(a, b, r_team, idx, r_pl, pidx, beta, mu, roster, need):
    if a not in idx or b not in idx:
        return None
    x = [r_team[idx[a]] - r_team[idx[b]]]
    if r_pl is not None:
        pa = [r_pl[pidx[p]] for p in roster.get(a, []) if p in pidx]
        pb = [r_pl[pidx[p]] for p in roster.get(b, []) if p in pidx]
        x.append(np.mean(pa) - np.mean(pb) if len(pa) >= 4 and len(pb) >= 4 else mu[1])
    z = float(np.dot(beta[:len(x)], x))
    p = 1 / (1 + math.exp(-z))
    return sum(math.comb(need - 1 + k, k) * p**need * (1 - p) ** k for k in range(need))


def rosters_at(train, lineups, cutoff, window_days=120):
    """断点前若干天里，每支队最常出场的五人。评级要按谁真的在打来取。"""
    seen = collections.defaultdict(collections.Counter)
    lo = cutoff - window_days * 86400
    for r in train:
        if r["start_time"] < lo:
            continue
        lu = lineups.get(r["match_id"])
        if not lu:
            continue
        for tid, side in ((r["radiant_team_id"], 0), (r["dire_team_id"], 1)):
            for a in lu[side]:
                seen[tid][a] += 1
    return {t: [a for a, _ in c.most_common(5)] for t, c in seen.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premium", action="store_true")
    ap.add_argument("--min-games", type=int, default=40)
    ap.add_argument("--hls", default="30,45,60,150")
    args = ap.parse_args()

    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    keep = {"premium"} if args.premium else BE.TOP
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in keep]
    lineups = models.load_lineups()
    events = BE.find_events(rows, args.min_games, 40.0, 0.8, 30, 15)

    tasks = []
    for e in events:
        post = [r for r in e["rows"] if r["start_time"] > e["break_at"]]
        ser = BE.series_of(post)
        if len(ser) < 5:
            continue
        train = [r for r in rows if r["start_time"] <= e["break_at"]]
        tasks.append((e, train, ser))
    print(f"{len(tasks)} 个赛事，{sum(len(t[2]) for t in tasks)} 个断点后系列赛\n")

    hls = [float(x) for x in args.hls.split(",")]
    per_ev = {h: [] for h in hls}
    lls = {h: [] for h in hls}
    accs = {h: [] for h in hls}
    for e, train, ser in tasks:
        cache = {}
        roster = rosters_at(train, lineups, e["break_at"])
        for h in hls:
            rt, idx, rp, pidx, beta, mu = fit_stack(train, e["break_at"], h,
                                                    lineups, cache)
            ev = []
            for g, a, a_won in ser:
                b = (g[0]["dire_team_id"] if g[0]["radiant_team_id"] == a
                     else g[0]["radiant_team_id"])
                ps = series_prob(a, b, rt, idx, rp, pidx, beta, mu, roster,
                                 3 if len(g) > 3 else 2)
                if ps is None:
                    continue
                ps = min(max(ps, 1e-6), 1 - 1e-6)
                ev.append(-(math.log(ps) if a_won else math.log(1 - ps)))
                accs[h].append(float((ps > 0.5) == a_won))
            if ev:
                per_ev[h].append(sum(ev))
                lls[h].extend(ev)
        print(".", end="", flush=True)
    print()

    ref = hls[-1]
    print(f"\n{'队伍半衰期':14s} {'系列LL':>9} {'系列准':>8} "
          f"{'Δ vs ' + f'{ref:g}天':>12} {'t':>7}")
    for h in hls:
        d = t = 0.0
        if h != ref:
            n = min(len(lls[h]), len(lls[ref]))
            dv = np.array(lls[ref][:n]) - np.array(lls[h][:n])
            g = np.array(per_ev[ref]) - np.array(per_ev[h])
            se = math.sqrt(((g - g.mean()) ** 2).sum() * len(g)
                           / max(len(g) - 1, 1)) / n
            d, t = dv.mean(), (dv.mean() / se if se > 0 else 0.0)
        print(f"{f'HL={h:g}天':14s} {np.mean(lls[h]):>9.4f} {np.mean(accs[h]):>7.1%} "
              f"{d:>+12.4f} {t:>+7.2f}")

    # 30 vs 45 的直接配对
    if 30.0 in per_ev and 45.0 in per_ev:
        g = np.array(per_ev[45.0]) - np.array(per_ev[30.0])
        n = len(lls[30.0])
        se = math.sqrt(((g - g.mean()) ** 2).sum() * len(g) / max(len(g) - 1, 1)) / n
        d = (np.array(lls[45.0]) - np.array(lls[30.0])).mean()
        print(f"\nHL=45 相对 HL=30：Δlogloss {-d:+.4f}  t {-d / se:+.2f}"
              f"  （正值 = 45 更好）")


if __name__ == "__main__":
    main()
