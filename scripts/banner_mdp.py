"""代币的真实边际价值：把「操作随机到达」和「补救不限于同一操作」一起算进去。

`banner_decide.py --lookahead` 给的 W_k 有两个已知偏差，方向相反：

  1. **上偏**：它假设同一个操作能再出现 k 次。真实是每次刷新从 70 种里抽 3 个，
     特定操作出现的概率约 3/70，而且等它出现本身要花代币。
  2. **下偏**：它只许用同一个操作补救。真实是摇坏之后，当次牌面里**任何**能打到
     那一枚的操作都能修。

本模块把两条都放进一个 MDP，所以不再需要「两个偏差方向相反、量级不明」这种说法：

    V(s, n) = E_抽3 [ max( max_{op ∈ 抽中} V(op(s), n-1),  V(s, n-1) ) ]

第二项是「只刷新选项」——它和点一个操作同样花 1 枚、同样给新牌面，所以是
每回合的保底动作，也是「阈值为 0」那条结论在多步下的正确形式。

**范围**：状态只取一面战旗的**阶数**（特性和统计项固定）。5 枚 ⇒ 3125 个状态。
品质类操作按位置可分解，作用在阶数上就是沿某几个轴的线性算子，所以整张值表
可以用张量收缩精确推进，不用采样。特性/统计项类操作（34/70）在这个子空间里
不改变状态，等价于「只刷新」，会被自动当成保底动作。

⇒ 得到的是**品质维度上**代币的边际价值，以及「现在点这个操作」相对于
「留着代币」的真实差额。特性和统计项那两维的价值不在里面，所以这是个下界。

    python scripts/banner_mdp.py --role mid --tokens 27
    python scripts/banner_mdp.py --role mid --op roll_quality/last_ruby
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banner_craft as BC  # noqa: E402
import banner_decide as BD  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIERS = [1, 2, 3, 4, 5]
POOL = 70          # 徽标类操作的枚举空间，见 docs/05-fantasy.md §1.5
DRAW = 3


def kernels(qdist):
    """-> (升阶核, 重选核)，都是 5x5 行随机矩阵，行列都按 TIERS 索引。"""
    inc = np.zeros((5, 5))
    for i, t in enumerate(TIERS):
        inc[i, TIERS.index(min(5, t + 1))] = 1.0
    rr = np.zeros((5, 5))
    for i, t in enumerate(TIERS):
        opts = [(u, qdist[u]) for u in TIERS if u != t]
        z = sum(p for _, p in opts)
        for u, p in opts:
            rr[i, TIERS.index(u)] = p / z
    return inc, rr


def apply_axes(V, K, axes):
    """沿 axes 各作用一次核 K：E[V(s')]。

    K[i, j] = P(阶 i -> 阶 j)，要的是 out[..i..] = Σ_j K[i,j] V[..j..]，
    所以必须和 K 的**第二个**轴收缩。和第一个轴收缩等于用了 K 的转置 ——
    那个矩阵列和为 1、行和不为 1，概率不守恒，值会指数发散。
    """
    out = V
    for ax in axes:
        out = np.moveaxis(np.tensordot(out, K, axes=([ax], [1])), -1, ax)
    return out


def op_operators(n_emb, colors, qdist):
    """-> {操作名: 作用在值表上的函数}，只含会改变阶数的那些操作。

    随机类目标 = 各单枚结果的均值；「自选一枚」= 各单枚结果的逐点最大
    （玩家先看牌面再挑徽标）。
    """
    inc, rr = kernels(qdist)
    pos_all = list(range(n_emb))
    by_color = {c: [i for i in pos_all if colors[i] == c]
                for c in ("red", "blue", "green")}
    named = {"all": pos_all}
    for c, cn in (("red", "ruby"), ("blue", "sapphire"), ("green", "emerald")):
        if by_color[c]:
            named[cn] = by_color[c]
    single, randomised, choose = {}, {}, []
    for pre, pick in (("first", lambda L: [L[0]]), ("last", lambda L: [L[-1]])):
        if pos_all:
            single[f"{pre}_all"] = pick(pos_all)
        for c, cn in (("red", "ruby"), ("blue", "sapphire"), ("green", "emerald")):
            if by_color[c]:
                single[f"{pre}_{cn}"] = pick(by_color[c])
    randomised["random_all"] = pos_all
    for c, cn in (("red", "ruby"), ("blue", "sapphire"), ("green", "emerald")):
        if by_color[c]:
            randomised[f"random_{cn}"] = by_color[c]
    choose = pos_all

    ops = {}
    for verb, K in (("increase_quality", inc), ("roll_quality", rr)):
        for t, P in list(named.items()) + list(single.items()):
            ops[f"{verb}/{t}"] = (lambda V, K=K, P=P: apply_axes(V, K, P))
        for t, P in randomised.items():
            ops[f"{verb}/{t}"] = (lambda V, K=K, P=P: np.mean(
                [apply_axes(V, K, [p]) for p in P], axis=0))
        ops[f"{verb}/choose_one"] = (lambda V, K=K, P=choose: np.max(
            [apply_axes(V, K, [p]) for p in P], axis=0))

    ops["increase_one_quality"] = lambda V: np.mean(
        [apply_axes(V, inc, [p]) for p in pos_all], axis=0)

    picks = [(d, u) for d in pos_all
             for u in itertools.combinations([p for p in pos_all if p != d], 2)]

    def i2d1(V):
        acc = 0.0
        for d, u in picks:
            x = apply_axes(V, DOWN, [d])
            x = apply_axes(x, UP, list(u))
            acc = acc + x
        return acc / len(picks)

    ops["increase_two_decrease_one"] = i2d1
    return ops


def directed_kernel(qdist, up):
    K = np.zeros((5, 5))
    for i, t in enumerate(TIERS):
        side = [u for u in TIERS if (u > t if up else u < t)]
        if not side:
            K[i, i] = 1.0
            continue
        z = sum(qdist[u] for u in side)
        for u in side:
            K[i, TIERS.index(u)] = qdist[u] / z
    return K


def draw_weights(m):
    """从 m 种里不放回抽 DRAW 个，值降序排列时第 i 名成为最大值的概率。"""
    tot = math.comb(m, DRAW)
    return np.array([math.comb(m - i - 1, DRAW - 1) / tot for i in range(m)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="mid", choices=BD.ROLES)
    ap.add_argument("--tokens", type=int, default=None, help="默认取快照里的代币数")
    ap.add_argument("--qdist", default="温和递减", choices=list(BC.QDIST))
    ap.add_argument("--sims", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--op", default=None,
                    help="额外报告「现在点这个操作」相对于「留着代币」的真实差额")
    args = ap.parse_args()

    global UP, DOWN
    qdist = BC.QDIST[args.qdist]
    UP, DOWN = directed_kernel(qdist, True), directed_kernel(qdist, False)

    BD.MISSING = set()
    log = BD.load_log()
    st, banners, tm = BD.current(log)
    role = args.role
    banner = banners[role]
    n_emb = len(banner)
    n_tok = args.tokens if args.tokens is not None else st.get("tokens_before", 0)
    colors = [BD.COLOR_OF[s] for s, _, _ in banner]
    ev = BD.Evaluator(tm, args.sims, args.seed, data="replay",
                      structure="fixed" if st.get("period") == "ti" else "group")

    # 全部阶数组合的裸值
    grid = list(itertools.product(TIERS, repeat=n_emb))
    states = [tuple((s, t, tr) for (s, _, tr), t in zip(banner, g)) for g in grid]
    print(f"{BD.ROLE_CN[role]}（{tm[role]}）  {BD.emb(banner)}")
    print(f"枚举 {len(states)} 个阶数组合的裸值 …", flush=True)
    v = ev(role, states).reshape((5,) * n_emb)
    cur = tuple(TIERS.index(t) for _, t, _ in banner)
    print(f"当前 {v[cur]:,.0f}   该空间内最好 {v.max():,.0f}（阶数 "
          f"{[TIERS[i] for i in np.unravel_index(v.argmax(), v.shape)]}）"
          f"   最差 {v.min():,.0f}\n")

    ops = op_operators(n_emb, colors, qdist)
    n_tier_ops = len(ops)
    W = draw_weights(POOL)
    print(f"品质类操作 {n_tier_ops} 种会改变阶数，其余 {POOL-n_tier_ops} 种在这个"
          f"子空间里等价于「只刷新」\n")

    V = v.copy()
    hist = [float(v[cur])]
    for n in range(1, n_tok + 1):
        cand = np.stack([f(V) for f in ops.values()] +
                        [V] * (POOL - n_tier_ops))          # 每种操作的 V(op(s), n-1)
        cand = np.sort(cand, axis=0)[::-1]                  # 降序
        best_drawn = np.tensordot(W, cand, axes=([0], [0]))  # E[抽中的最好一个]
        V = np.maximum(best_drawn, V)                        # 「只刷新」是保底
        hist.append(float(V[cur]))
    print(f"{'代币 n':<8}" + "".join(f"{n:>9}" for n in (0, 1, 2, 3, 5, 10, 20, n_tok)))
    print(f"{'V(s,n)':<8}" + "".join(f"{hist[min(n,len(hist)-1)]:>9,.0f}"
                                     for n in (0, 1, 2, 3, 5, 10, 20, n_tok)))
    print(f"{'比裸值':<8}" + "".join(f"{hist[min(n,len(hist)-1)]-hist[0]:>+9,.0f}"
                                    for n in (0, 1, 2, 3, 5, 10, 20, n_tok)))
    marg = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
    print(f"\n代币的边际价值：第 1 枚 {marg[0]:+,.0f}，第 2 枚 {marg[1]:+,.0f}，"
          f"第 5 枚 {marg[4]:+,.0f}，第 10 枚 {marg[9]:+,.0f}，"
          f"第 {n_tok} 枚 {marg[-1]:+,.0f}")

    if args.op:
        verb, _, tgt = args.op.partition("/")
        key = args.op if args.op in ops else f"{verb}/{tgt}"
        if key not in ops:
            raise SystemExit(f"{args.op} 不是品质类操作（本模型只覆盖阶数维度）")
        Vn1 = v.copy()
        for _ in range(max(0, n_tok - 1)):
            cand = np.stack([f(Vn1) for f in ops.values()] +
                            [Vn1] * (POOL - n_tier_ops))
            cand = np.sort(cand, axis=0)[::-1]
            Vn1 = np.maximum(np.tensordot(W, cand, axes=([0], [0])), Vn1)
        take = float(ops[key](Vn1)[cur])
        skip = float(Vn1[cur])
        one = float(ops[key](v)[cur] - v[cur])
        print(f"\n手上 {n_tok} 枚时点「{BD.op_cn(verb, tgt)}」：")
        print(f"  一步 E[Δ]（把操作后当终局）        {one:>+10,.0f}")
        print(f"  点它   V(op(s), {n_tok-1})              {take:>10,.0f}")
        print(f"  不点   V(s, {n_tok-1})（只刷新）        {skip:>10,.0f}")
        print(f"  真实差额                          {take-skip:>+10,.0f}")


if __name__ == "__main__":
    main()


# ------------------------------------------------ 任意维度的 MDP（特性 / 统计项）
def dim_mdp(ev, role, banner, dim, n_tokens, qdist, color=None, model="reroll",
            max_states=400, verbs=None):
    """在**某一个维度**上跑和上面同构的 MDP，用真实到达率。

    dim="trait" 或 "stat"：状态空间是只改变该维度所能到达的全部战旗。
    70 种操作里，作用在别的维度上的那些在这里当作恒等（等价于「只刷新」）——
    所以这是**条件价值**：假定这枚代币只用在这一维上。阶数那一维在 main() 里
    单独算，两维大致可分，加总是个近似。

    返回每个代币数下的 V(s,n)，以及每个操作在 n 枚时「点」与「不点」的差额。
    """
    import collections
    start = banner
    # 只让某一种颜色的那几枚可变：否则统计项维度会是各色排列的乘积，
    # 中单是 6P2 x 6 x 6P2 = 5400 个状态，既算不动也不是我们要问的问题。
    allowed = set(range(len(banner))) if color is None else {
        i for i, (st_, _, _) in enumerate(banner) if BD.COLOR_OF[st_] == color}
    seen, dq = {start}, collections.deque([start])
    # dim 只是默认动词的简写；传 verbs 可以同时放开多个维度 —— 那才是**联合**价值。
    # 只放开一个维度得到的是偏导数：换统计项的价值取决于它坐在第几阶，
    # 提阶的价值取决于那一枚是什么统计项，两者互为条件。
    if verbs is None:
        verbs = ["roll_trait"] if dim == "trait" else ["roll_stat"]
    targets = []
    for t in BD.TARGET_CN:
        if t == "none":
            continue
        if t == "choose_one":
            targets.append(t)
            continue
        hs = BD.hit_sets(banner, t)
        if hs and all(set(pos) <= allowed for pos, _ in hs):
            targets.append(t)
    while dq:
        s0 = dq.popleft()
        for verb in verbs:
            for t in targets:
                o = BD.outcomes(s0, verb, t, qdist, model)
                if o is None:
                    continue
                if o == "CHOOSE":
                    o = [x for p in sorted(allowed)
                         for x in BD.apply_verb(s0, (p,), verb, qdist)]
                for ns, _ in o:
                    if ns not in seen:
                        if len(seen) > max_states:
                            raise SystemExit(f"状态空间超过 {max_states}")
                        seen.add(ns)
                        dq.append(ns)
    states = sorted(seen)
    idx = {s0: i for i, s0 in enumerate(states)}
    v = ev(role, states)

    # 每个操作的转移矩阵（稀疏成 (状态, [(下标, 概率)])）
    trans = {}
    for verb in verbs:
        for t in targets:
            rows = []
            ok = True
            for s0 in states:
                o = BD.outcomes(s0, verb, t, qdist, model)
                if o is None:
                    ok = False
                    break
                if o == "CHOOSE":     # 玩家挑徽标 -> 逐状态取最好，稍后处理
                    rows.append([[x for x in BD.apply_verb(s0, (p,), verb, qdist)]
                                 for p in sorted(allowed)])
                else:
                    rows.append(o)
            if ok:
                trans[f"{verb}/{t}"] = rows
    n_act = len(trans)

    W = draw_weights(POOL)
    V = v.copy()
    hist = [float(v[idx[start]])]
    per_op = {}
    for n in range(1, n_tokens + 1):
        cols = []
        for name, rows in trans.items():
            col = np.empty(len(states))
            for i, r in enumerate(rows):
                if r and isinstance(r[0], list):        # choose_one
                    col[i] = max(sum(p * V[idx[ns]] for ns, p in alt)
                                 for alt in r)
                else:
                    col[i] = sum(p * V[idx[ns]] for ns, p in r)
            cols.append(col)
            if n == n_tokens:
                per_op[name] = col
        cand = np.stack(cols + [V] * (POOL - n_act))
        V = np.maximum(np.tensordot(W, np.sort(cand, axis=0)[::-1],
                                    axes=([0], [0])), V)
        hist.append(float(V[idx[start]]))
    return {"states": len(states), "hist": hist, "idx": idx[start],
            "per_op": {k: float(c[idx[start]]) for k, c in per_op.items()},
            "V_prev": float(V[idx[start]])}
