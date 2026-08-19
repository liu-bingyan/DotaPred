"""正赛期的槽位估值：按**对阵树**算，不再给所有队一律 6 个系列赛。

一期得分 = 该定位「最好的那个系列赛」的两局之和（docs/06-banner.md §1.3）。
所以系列赛数 k 是取最大值的次数 —— 它不是次要参数，它是这一期的主变量。

`banner_decide` 的 `structure="fixed", n_series=6` 给八支队一律按 6 个算。6 是双败树的
**理论上限**（首轮就输、再从败者组一路打到总决赛），八队里没有一支的期望接近它：
Team Vision 4.36、Team Liquid 3.46、Nigma Galaxy 只有 **2.58**（59% 概率只打 2 个）。
一刀切等于系统性地补贴弱队 —— 恰好补贴在把弱队排到第一的方向上。

`bracket_optimize` 已经把 2^14 = 16,384 种完整对阵树和精确概率算出来了
（`docs/09-playoff.md` §6），所以这里的一切都是**精确枚举**，没有蒙特卡洛误差：

* 每支队打几个系列赛、赢了几个 —— 直接数每种结果里它出现在哪些节点上；
* 哪几个是 Bo5 —— `ti2026_bracket.json` 里只有总决赛（node 21）是 bo5，其余 13 个 bo3。

比 `structure="fixed"` 多修的第三件事：**胜负不再按历史逐局胜率随机抽**。对阵树给出
每支队赢了几个系列赛，而赢一个 Bo3 就是「2 胜 + 至多 1 负」—— 这是硬约束，
原来的做法是每个系列赛独立抛硬币，会同时高估弱队的上限和低估强队的下限。

    python scripts/playoff_value.py                     # 三个定位 × 八支队
    python scripts/playoff_value.py --hl 150            # 换评级口径复核
    python scripts/playoff_value.py --flat 6            # 复现旧的一刀切口径
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banner_craft as BC  # noqa: E402
import banner_decide as BD  # noqa: E402
import bracket_optimize as BO  # noqa: E402
from banner_value import game_matrix, STATS, IDX, P_THREE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = ("core", "mid", "support")

# Bo3 走三局的经验概率是 P_THREE=0.382 ⇒ 2p(1-p)=0.382 ⇒ 隐含的单局胜率 p≈0.743。
# 用同一个 p 推 Bo5 的局数分布，保持两处假设一致（系列赛长度对双方对称）。
_P = 0.5 + (0.25 - P_THREE / 2) ** 0.5
P_GAMES = {
    3: {2: 1 - P_THREE, 3: P_THREE},
    5: {3: _P ** 3 + (1 - _P) ** 3,
        4: 3 * _P ** 3 * (1 - _P) + 3 * (1 - _P) ** 3 * _P,
        5: 6 * _P ** 2 * (1 - _P) ** 2},
}


def bracket_dist(hl="30", boot=True):
    """每支队的 (Bo3胜, Bo3负, Bo5胜, Bo5负) 精确分布。

    -> {队名: [(pattern, prob), ...]}，pattern 是那个四元组。
    """
    probs = os.path.join(ROOT, "data", f"playoff_probs_hl{hl}.json")
    nodes, _single, teams, tidx = BO.load(probs)
    win, lose, _pair = BO.enumerate_brackets(nodes, tidx)
    n_out, n_node = win.shape
    bo5 = np.array([nodes[nid]["bo"] == 5 for nid in BO.ORDER])

    S = json.load(open(probs))
    strength = [S["strength"][t] for t in teams]
    p = BO.outcome_probs(win, lose, nodes, strength)
    p /= p.sum()
    if boot:
        B = np.load(os.path.join(ROOT, "data", f"playoff_probs_hl{hl}_boot.npy"))
        acc = np.zeros(n_out)
        for b in range(B.shape[0]):
            q = BO.outcome_probs(win, lose, nodes, B[b])
            acc += q / q.sum()
        p = acc / B.shape[0]

    out = {}
    for t, name in enumerate(teams):
        w3 = ((win == t) & ~bo5).sum(1)
        l3 = ((lose == t) & ~bo5).sum(1)
        w5 = ((win == t) & bo5).sum(1)
        l5 = ((lose == t) & bo5).sum(1)
        key = (w3 * 1000 + l3 * 100 + w5 * 10 + l5)
        agg = {}
        for k, pr in zip(key, p):
            agg[int(k)] = agg.get(int(k), 0.0) + float(pr)
        out[name] = sorted(
            ((k // 1000, k // 100 % 10, k // 10 % 10, k % 10), pr)
            for k, pr in agg.items() if pr > 0)
    return out, teams


def _series(Vw, Vl, k, bo, won, rng):
    """k 条并行的系列赛，各取「最好两局之和」。won=True 表示这支队赢下该系列赛。

    赢一个 Bo{bo} 就是拿满 need=(bo+1)/2 胜、其余是负；输就反过来。局数从
    P_GAMES 抽，胜负局分别从 Vw / Vl 池里有放回地抽。
    """
    need = (bo + 1) // 2
    pool_w = Vw if len(Vw) else Vl
    pool_l = Vl if len(Vl) else Vw
    ns, ps = zip(*sorted(P_GAMES[bo].items()))
    n_games = np.array(ns)[np.searchsorted(np.cumsum(ps), rng.random(k))]

    nc = pool_w.shape[1]
    best = np.zeros((k, nc))
    second = np.zeros((k, nc))
    for g in range(bo):
        live = g < n_games
        if not live.any():
            break
        # 赢下系列赛 = 拿满 need 胜、其余是负；输掉 = 只赢 n_games-need 局。
        # 局与局之间是同分布独立抽的，所以把赢的那几局排在前面不损失一般性。
        n_win_games = np.full(k, need) if won else n_games - need
        from_win = g < n_win_games
        v = np.where(from_win[:, None],
                     pool_w[rng.integers(0, len(pool_w), k)],
                     pool_l[rng.integers(0, len(pool_l), k)])
        v = np.where(live[:, None], v, -np.inf)
        hi = np.maximum(best, v)
        second = np.maximum(second, np.minimum(best, v))
        best = hi
    return best + second


def period_value_bracket(Vw, Vl, dist, n_sims, rng):
    """E[一期得分] —— 对 (Bo3胜,Bo3负,Bo5胜,Bo5负) 的精确分布加权。"""
    nc = Vw.shape[1] if len(Vw) else Vl.shape[1]
    total = np.zeros(nc)
    for (w3, l3, w5, l5), pr in dist:
        if pr <= 0:
            continue
        best = np.full((n_sims, nc), -np.inf)
        for bo, won, cnt in ((3, True, w3), (3, False, l3),
                             (5, True, w5), (5, False, l5)):
            for _ in range(cnt):
                best = np.maximum(best, _series(Vw, Vl, n_sims, bo, won, rng))
        total += pr * np.where(np.isfinite(best), best, 0.0).mean(axis=0)
    return total


def slot_value(rows, banners, dist, n_sims, seed):
    X, win = game_matrix(rows)
    M = np.zeros((len(STATS), len(banners)))
    for j, b in enumerate(banners):
        for (s, _, _), wi in zip(b, BC.multipliers(b)):
            M[IDX[s], j] += wi
    V = X @ M
    rng = np.random.default_rng(seed)
    return period_value_bracket(V[win], V[~win], dist, n_sims, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hl", default="30", choices=("30", "150"))
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-boot", action="store_true",
                    help="只用点估计强度，不对参数不确定性积分")
    ap.add_argument("--flat", type=int, default=None,
                    help="复现旧口径：所有队一律 N 个 Bo3、赢一半")
    args = ap.parse_args()

    log = BD.load_log()
    st, banners, held = BD.current(log)
    dist, teams = bracket_dist(args.hl, boot=not args.no_boot)
    if args.flat:
        n = args.flat
        dist = {t: [((n - n // 2, n // 2, 0, 0), 1.0)] for t in teams}

    ev = BD.Evaluator({r: held[r] for r in ROLES}, 10, args.seed,
                      structure="fixed", n_series=6)
    slots = ev.slots

    print(f"评级口径 HL={args.hl}"
          + ("（点估计）" if args.no_boot else "（已对 bootstrap 积分）")
          + (f"   ⚠ --flat {args.flat}：复现一刀切口径" if args.flat else ""))
    print(f"Bo3 局数 {P_GAMES[3]}   Bo5 局数 "
          + "{" + ", ".join(f"{k}: {v:.3f}" for k, v in P_GAMES[5].items()) + "}")
    print()
    for r in ROLES:
        print(f"  {BD.ROLE_CN[r]}  {BD.emb(banners[r])}")
    print()

    res = {}
    for r in ROLES:
        print(f"=== {BD.ROLE_CN[r]}  {BD.emb_short(banners[r])} ===")
        print(f"{'队伍':<16}{'正赛期分':>10}{'E[系列赛]':>11}"
              f"{'P(只打2个)':>11}   局数")
        rows = []
        for t in teams:
            if (t, r) not in slots:
                continue
            d = dist[t]
            ek = sum(pr * sum(x) for x, pr in d)
            p2 = sum(pr for x, pr in d if sum(x) == 2)
            v = float(slot_value(slots[(t, r)], [banners[r]], d,
                                 args.sims, args.seed)[0])
            rows.append((t, v, ek, p2, len(slots[(t, r)])))
        for t, v, ek, p2, n in sorted(rows, key=lambda x: -x[1]):
            cur = "  ← 现在" if held[r] == t else ""
            print(f"{t:<16}{v:>10,.0f}{ek:>11.2f}{p2:>11.0%}   {n:>3} 局{cur}")
        res[r] = rows
        best = max(rows, key=lambda x: x[1])
        cur = next(x for x in rows if x[0] == held[r])
        print(f"  最优 {best[0]}   {best[1] - cur[1]:+,.0f}\n")

    now = sum(next(x[1] for x in res[r] if x[0] == held[r]) for r in ROLES)
    top = {r: max(res[r], key=lambda x: x[1]) for r in ROLES}
    print(f"现状      {now:>9,.0f}   "
          + "  ".join(f"{BD.ROLE_CN[r]}={held[r]}" for r in ROLES))
    print(f"最优      {sum(t[1] for t in top.values()):>9,.0f}   "
          + "  ".join(f"{BD.ROLE_CN[r]}={top[r][0]}" for r in ROLES)
          + f"   ({sum(t[1] for t in top.values()) - now:+,.0f})")


if __name__ == "__main__":
    main()
