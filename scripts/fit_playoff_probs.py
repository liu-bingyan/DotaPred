"""按指定的时间权重口径重新拟合评级，输出主赛事 8 队的单局胜率矩阵。

和 fit_ratings_v3.py 是同一个模型，区别只在半衰期等超参可以从命令行给，而且
输出限定在打进主赛事的 8 队。生产评级 data/ratings_v3.json 保持 HL=150 不动，
这里另写一份文件，免得把已经写进 docs 的产物覆盖掉。

带 --boot 时会做按系列赛聚类的 Poisson bootstrap，把每个复制样本的胜率矩阵一并
写出。凸的计分函数下这件事很重要：参数不确定性不是把每场都推向 50%，而是让
「Nigma 其实很强」这种世界线在 14 个节点上同时成立 —— 相关性正是凸目标要吃的东西。

    python3 scripts/fit_playoff_probs.py --hl 30 --boot 200
"""

import argparse
import collections
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
import margin  # noqa: E402
import models  # noqa: E402
from experiment import MIN_PLAYER_GAMES, player_feature  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = {"premium", "professional"}


def playoff_teams():
    br = json.load(open(os.path.join(ROOT, "data", "ti2026_bracket.json")))
    seats = set()
    for b in br["playoff"]:
        for k in ("team_1", "team_2"):
            if b[k]:
                seats.add(b[k])
    return sorted(seats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hl", type=float, default=30.0, help="队伍评级半衰期（天）")
    ap.add_argument("--player-hl", type=float, default=None,
                    help="选手评级半衰期，默认与 --hl 相同")
    ap.add_argument("--l2t", type=float, default=0.3)
    ap.add_argument("--l2p", type=float, default=30.0)
    ap.add_argument("--boot", type=int, default=0, help="bootstrap 复制样本数")
    ap.add_argument("--league", type=int, default=None,
                    help="只用这个 league_id 的比赛（如 19719 = 只用 TI 本届）")
    ap.add_argument("--min-games", type=int, default=20,
                    help="进入评级所需的最少场次；限定单赛事时必须调低")
    ap.add_argument("--min-player-games", type=int, default=MIN_PLAYER_GAMES)
    ap.add_argument("--no-player", action="store_true",
                    help="只用队伍级评级。单赛事下每队阵容固定，选手项与队伍项共线")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "playoff_probs.json"))
    args = ap.parse_args()
    php = args.player_hl if args.player_hl is not None else args.hl

    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    teams = FS.apply_roster_overrides(
        json.load(open(os.path.join(ROOT, "data", "teams.json"))))
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in TOP]
    if args.league:
        rows = [r for r in rows if r["leagueid"] == args.league]
    lineups = models.load_lineups()
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())

    names = playoff_teams()
    name_of = {}
    for n, v in teams.items():
        name_of[v["team_id"]] = n
    # data/teams.json 的写法和 bracket 里的可能差一个大小写
    canon = {n.lower(): n for n in teams}
    names = [canon.get(n.lower(), n) for n in names]
    missing = [n for n in names if n not in teams]
    if missing:
        raise SystemExit(f"teams.json 里没有: {missing}")

    i, j, w, idx, m, win, tr = margin.build_design(rows, now, args.hl, args.min_games)
    gd = m["gdpm"] / 400.0
    accs = {n: [p["account_id"] for p in teams[n]["players"][:5]] for n in names}
    print(f"训练集 {len(tr)} 场 / {len(idx)} 支队"
          + (f"（只用 league {args.league}）" if args.league else ""))
    missing_t = [n for n in names if teams[n]["team_id"] not in idx]
    if missing_t:
        raise SystemExit(f"这些队没进评级（场次不足 {args.min_games}）: {missing_t}")

    # 选手项用自己的半衰期，训练行与队伍项对齐
    wp = (w if php == args.hl
          else margin.build_design(rows, now, php, args.min_games)[2])

    counts = collections.Counter()
    for r in tr:
        lu = lineups.get(r["match_id"])
        if lu:
            for a in lu[0] + lu[1]:
                counts[a] += 1
    pidx = {a: k for k, a in enumerate(sorted(a for a, c in counts.items()
                                              if c >= args.min_player_games))}
    prows, pkeep = [], []
    for k, r in enumerate(tr):
        lu = lineups.get(r["match_id"])
        if not lu:
            continue
        a = [pidx[x] for x in lu[0] if x in pidx]
        b = [pidx[x] for x in lu[1] if x in pidx]
        if len(a) >= 4 and len(b) >= 4:
            prows.append((a, b))
            pkeep.append(k)
    pkeep = np.array(pkeep)

    use_player = not args.no_player and len(prows) >= 50

    def one_fit(weights, wpl, iters=3000):
        r_team, _ = margin.fit_margin(i, j, gd, weights, len(idx), l2=args.l2t,
                                      iters=iters)
        cols = [r_team[i] - r_team[j]]
        r_pl = None
        if use_player:
            r_pl, _ = models.ridge_ratings(prows, wpl[pkeep], gd[pkeep], len(pidx),
                                           l2=args.l2p, iters=iters)
            cols.append(player_feature(tr, r_pl, pidx, lineups))
        X = np.column_stack(cols)
        mu = np.nanmean(X, axis=0)
        beta = margin.fit_logistic(np.where(np.isnan(X), mu, X), win, weights,
                                   iters=min(iters, 2000))
        s = {}
        for n in names:
            v = beta[0] * r_team[idx[teams[n]["team_id"]]]
            if use_player:
                known = [r_pl[pidx[a]] for a in accs[n] if a in pidx]
                pr = np.mean(known) if len(known) >= 4 else mu[1]
                v += beta[1] * pr
            s[n] = float(v)
        return s, beta

    strength, beta = one_fit(w, wp)
    print(f"HL={args.hl:g} 选手HL={php:g} l2t={args.l2t} l2p={args.l2p}")
    if use_player:
        print(f"P(单局胜) = sigmoid({beta[0]:.3f}*队伍差 + {beta[1]:.3f}*选手差 "
              f"{beta[2]:+.3f})\n")
    else:
        print(f"P(单局胜) = sigmoid({beta[0]:.3f}*队伍差 {beta[1]:+.3f})  [无选手项]\n")
    for k, n in enumerate(sorted(names, key=lambda x: -strength[x]), 1):
        print(f"  {k} {n:18s} {strength[n]:>7.3f}")

    def matrix(s):
        return {a: {b: (0.5 if a == b else float(1 / (1 + np.exp(-(s[a] - s[b])))))
                    for b in names} for a in names}

    out = {"model": ("team gold-margin" if not use_player else "team+player gold-margin")
                    + (f", league {args.league} only" if args.league
                       else f", HL={args.hl:g}"),
           "half_life": args.hl, "player_half_life": php, "league": args.league,
           "n_train_games": len(tr),
           "strength": strength, "single_game": matrix(strength)}

    if args.boot:
        cl = np.array([r.get("series_id") or -r["match_id"] for r in tr])
        _, cl_pos = np.unique(cl, return_inverse=True)
        rng = np.random.default_rng(20260820)
        reps = []
        for b in range(args.boot):
            mult = rng.poisson(1.0, size=cl_pos.max() + 1)[cl_pos]
            if (w * mult).sum() <= 0:
                continue
            s, _ = one_fit(w * mult, wp * mult, iters=1200)
            reps.append([s[n] for n in names])
            if (b + 1) % 25 == 0:
                print(f"  bootstrap {b + 1}/{args.boot}", flush=True)
        S = np.array(reps)
        out["bootstrap_teams"] = names
        # 跟着 --out 走，否则两次不同口径的拟合会互相覆盖
        boot_path = os.path.splitext(args.out)[0] + "_boot.npy"
        out["bootstrap_file"] = os.path.relpath(boot_path, ROOT)
        np.save(boot_path, S)
        print(f"\n{'team':18s}{'strength':>10}{'boot sd':>9}")
        for n in sorted(names, key=lambda x: -strength[x]):
            k = names.index(n)
            print(f"{n:18s}{strength[n]:>10.3f}{S[:, k].std():>9.3f}")

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
