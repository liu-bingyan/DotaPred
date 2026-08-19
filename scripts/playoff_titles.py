"""正赛期的称号：按**对阵树**的系列赛分布挑前缀 + 后缀。

称号和阵容必须一起定，而且顺序是**先阵容后称号** —— 前缀是选手级条件
（「使用蓝色英雄时 +11%」），命中率取决于挂的那支队这个位置真正在打的英雄，
所以阵容一变，前缀的排序就可能变（`docs/07-titles.md` §6）。

结构口径同 `playoff_value.py`：不用 `data/sim_buckets_boot.npy`（那是 2026-08-09
生成的**小组赛**名次桶，见 `docs/08-roster.md`），改成从 `bracket_optimize` 的
16,384 棵对阵树里数出来的、每支队自己的 (系列赛数, 系列赛胜场) 精确分布。

    python scripts/playoff_titles.py
    python scripts/playoff_titles.py --roster core=Team Vision,mid=...   # 试别的阵容
    python scripts/playoff_titles.py --flat 6                            # 旧的一刀切口径

⚠ 近似：`title_value.period_value` 只建了 Bo3。总决赛那一个 Bo5 节点在这里按 Bo3 处理
（14 个节点里只有 1 个，而且只在双方都进总决赛时才发生），对称号之间的**相对**排序
影响可以忽略 —— 它对所有称号候选是同一个偏置。
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banner_craft as BC  # noqa: E402
import banner_decide as BD  # noqa: E402
import fantasy_stats as FS  # noqa: E402
import playoff_value as PV  # noqa: E402
import roster_pick as RP  # noqa: E402
import title_value as TV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = ("core", "mid", "support")


class BracketDraws:
    """和 TV.Draws 同接口，但分组来自对阵树而不是小组赛名次桶。

    TV.Draws 按 6 个名次桶分组，每组的 (系列赛数, 期望胜场) 查 TV.RECORD。
    这里改成按这支队在对阵树里的 (系列赛数, 系列赛胜场) 模式分组 —— 模式和它的
    概率都是 16,384 棵树上数出来的精确值，不是模拟。
    """

    def __init__(self, patterns, n_sims, seed=11):
        rng = np.random.default_rng(seed)
        # 按 (总系列赛数, 总胜场) 归并；Bo5 在这里当 Bo3 处理（见模块 docstring）
        agg = {}
        for (w3, l3, w5, l5), pr in patterns:
            agg[(w3 + l3 + w5 + l5, w3 + w5)] = \
                agg.get((w3 + l3 + w5 + l5, w3 + w5), 0.0) + pr
        keys = sorted(agg)
        p = np.array([agg[k] for k in keys], dtype=float)
        p /= p.sum()
        who = rng.choice(len(keys), n_sims, p=p)

        self.bucket_col = who          # 只为接口兼容，period_value 只用它的长度
        self.groups = []
        for j, (n_series, n_wins) in enumerate(keys):
            mask = who == j
            k = int(mask.sum())
            if not k or not n_series:
                continue
            self.groups.append({
                "mask": mask, "k": k, "n_series": n_series,
                "n_wins": np.full(k, n_wins),
                "u_idx": rng.random((k, n_series, 3)),
                "u_three": rng.random((k, n_series)),
                "u_cond": rng.random((k, n_series, 3)),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hl", default="30", choices=("30", "150"))
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--roster", default=None,
                    help="core=X,mid=Y,support=Z；默认读日志里的当前阵容")
    ap.add_argument("--have", default="blue,clutch", help="手上的 前缀,后缀")
    ap.add_argument("--flat", type=int, default=None,
                    help="复现旧口径：所有队一律 N 个系列赛、赢一半")
    args = ap.parse_args()

    log = BD.load_log()
    st, banners, held = BD.current(log)
    roster = dict(held)
    if args.roster:
        roster.update(dict(kv.split("=") for kv in args.roster.split(",")))
    have = tuple(args.have.split(","))

    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    _, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)

    # 定稿的战旗倍率就是每个统计项的权重
    W = {r: {s: w for (s, _, _), w in zip(banners[r], BC.multipliers(banners[r]))}
         for r in ROLES}
    slots = TV.slot_games(by_player, roles, lambda t, r: W[r])

    dist, _teams = PV.bracket_dist(args.hl, boot=True)
    if args.flat:
        n = args.flat
        dist = {t: [((n - n // 2, n // 2, 0, 0), 1.0)] for t in dist}

    print(f"评级口径 HL={args.hl}   {args.sims:,} 次模拟"
          + (f"   ⚠ --flat {args.flat}" if args.flat else ""))
    print("阵容  " + "   ".join(
        f"{BD.ROLE_CN[r]}={roster[r]}（E[系列赛] "
        f"{sum(pr * sum(x) for x, pr in dist[roster[r]]):.2f}）" for r in ROLES))
    print()

    cube = {}
    for r in ROLES:
        key = (roster[r], r)
        if key not in slots:
            raise SystemExit(f"没有 {key} 的逐局数据")
        cube[r] = RP.eval_slot(RP.slot_arrays(slots[key]),
                               BracketDraws(dist[roster[r]], args.sims))

    def total(pre, suf):
        return sum(cube[r][(pre, suf)] for r in ROLES)

    pres = sorted(TV.PREFIX_BONUS, key=lambda p: -TV.PREFIX_BONUS[p])
    sufs = list(TV.SUFFIXES)
    base, now = total(None, None), total(*have)
    print(f"裸分（不带称号）            {base:>9,.0f}")
    print(f"手上的 {TV.PREFIX_ZH[have[0]]}+"
          f"{TV.SUFFIXES[have[1]][2].split()[0]:<8}     {now:>9,.0f}"
          f"   （称号值 {now - base:+,.0f}）\n")

    print("前缀（后缀固定为手上的那个）        逐槽位增益")
    for p in pres:
        v = total(p, have[1])
        per = "  ".join(
            f"{BD.ROLE_CN[r]} {100 * (cube[r][(p, have[1])] / cube[r][(None, have[1])] - 1):+.2f}%"
            for r in ROLES)
        print(f"  {TV.PREFIX_ZH[p]:<8} +{TV.PREFIX_BONUS[p]:.0%}"
              f"{v:>10,.0f}{v - now:>+9,.0f}   {per}")

    print("\n后缀（前缀固定为手上的那个）")
    for v, s in sorted(((total(have[0], s), s) for s in sufs), reverse=True):
        b, _k, lbl = TV.SUFFIXES[s]
        print(f"  {lbl:<34} +{b:>3.0%}{v:>10,.0f}{v - now:>+9,.0f}")

    best = max(itertools.product(pres, sufs), key=lambda ps: total(*ps))
    print(f"\n联合最优  {TV.PREFIX_ZH[best[0]]}+"
          f"{TV.SUFFIXES[best[1]][2].split()[0]}   {total(*best):>9,.0f}"
          f"   （比手上的 {total(*best) - now:+,.0f}）")


if __name__ == "__main__":
    main()
