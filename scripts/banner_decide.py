"""客户端给出三个操作选项时，该点哪一个、点在哪面战旗上。

    python scripts/banner_decide.py roll_stat/emerald increase_two_decrease_one \
                                    roll_quality/random_ruby

**状态是一条追加序列，不是一个当前值。** 唯一的文件是 data/roll_options_log.json：
`banner_states` 按 tokens_before 递减排成时间轴，末项就是现在；`draws` 是每次刷新
看到的三个选项。两条序列合起来才是这轮活动的完整记录 —— 覆盖写会毁掉两样东西：
(a) 「预测 +1,488、实际摇到什么」这种事后校准，(b) 抄错时没法回溯到哪一步错的。
所以 --apply 是 append，任何情况下都不改历史条目。

每个操作对三面战旗都算一遍：操作只作用于**当前打开的那面**，而换战旗免费
（docs/05-fantasy.md §1.5），所以决策变量是 (操作, 战旗) 的组合，不是操作本身。
灰掉的组合（战旗上没有该颜色的徽标）自动排除。

打分口径同 docs/06-banner.md：max2(sum) 蒙特卡洛，不是 EV 相加。所有候选状态共用
同一批随机局号，组合之间是配对比较；种子固定 ⇒ 同样的输入永远给同样的结论。

典型一轮：
    banner_decide.py A B C --record          # 看选项 → 结论，并把这次刷新记下来
    banner_decide.py --apply core:towers/5/benevolent,teamfight/1/benevolent,gpm/3/fractal \
                     --shown 290,170,190 --tokens 10 --via increase_two_decrease_one \
                     --chosen increase_two_decrease_one
    banner_decide.py --history               # 回看整条序列和每一步的预测/实际
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
from fantasy_model import role_game_scores  # noqa: E402
from banner_value import game_matrix, period_values, STATS, IDX, P_THREE  # noqa: E402
import banner_craft as BC  # noqa: E402
import import_replay_stats as IR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "data", "roll_options_log.json")
ROLES = ("core", "mid", "support")

COLOR_OF = {s: c for c, ss in FS.COLOR.items() for s in ss}
CN = {"kills": "击杀", "deaths": "死亡", "cs": "正反补", "gpm": "GPM",
      "madstone": "狂石", "towers": "推塔",
      "wards": "假眼", "stacks": "堆野怪", "runes": "拾神符",
      "watchers": "观察者", "smokes": "诡计之雾", "lotus": "莲花",
      "roshan": "肉山", "teamfight": "团战", "stuns": "眩晕",
      "tormentor": "痛苦魔方", "firstblood": "第一滴血", "courier": "杀信使"}
ROLE_CN = {"core": "核心", "mid": "中单", "support": "辅助"}
VERB_CN = {"increase_quality": "提升品质", "roll_quality": "重选品质",
           "roll_trait": "重选特性", "roll_stat": "重选统计项",
           "increase_one_quality": "随机提升一项品质",
           "increase_two_decrease_one": "随机提升两项、降低一项"}
TARGET_CN = {"choose_one": "自选一枚", "all": "全部", "none": "",
             "ruby": "红色全部", "sapphire": "蓝色全部", "emerald": "绿色全部",
             "first_all": "第一枚", "last_all": "最后一枚", "random_all": "随机一枚"}
for _c, _cn in (("ruby", "红"), ("sapphire", "蓝"), ("emerald", "绿")):
    TARGET_CN[f"first_{_c}"] = f"第一枚{_cn}"
    TARGET_CN[f"last_{_c}"] = f"最后一枚{_cn}"
    TARGET_CN[f"random_{_c}"] = f"随机一枚{_cn}"
TARGET_COLOR = {"ruby": "red", "sapphire": "blue", "emerald": "green"}

TARGETED = {"increase_quality", "roll_quality", "roll_trait", "roll_stat"}
UNTARGETED = {"increase_one_quality", "increase_two_decrease_one"}


def emb(banner):
    return "  ".join(f"{CN[s]}/{t}阶/{BC.TRAIT_CN[tr]}" for s, t, tr in banner)


def emb_short(banner):
    return "".join(f"{CN[s]}{t} " for s, t, _ in banner).strip()


# ---------------------------------------------------------------- 目标 → 位置
def hit_sets(banner, target):
    """-> [(positions, prob)]；空表示该操作在这面战旗上是灰的。

    位置退化（战旗只有 3 枚）：中单红蓝绿各一枚，整色/第一枚/最后一枚/随机
    在中单上全部退化成同一枚；核心红绿红、辅助蓝绿蓝，绿色类目标恒精确指向②。
    """
    n = len(banner)
    if target == "all":
        return [(tuple(range(n)), 1.0)]
    if target in TARGET_COLOR:
        pos = tuple(i for i, b in enumerate(banner)
                    if COLOR_OF[b[0]] == TARGET_COLOR[target])
        return [(pos, 1.0)] if pos else []
    for pre in ("first", "last", "random"):
        if not target.startswith(pre + "_"):
            continue
        rest = target[len(pre) + 1:]
        pos = [i for i, b in enumerate(banner)
               if rest == "all" or COLOR_OF[b[0]] == TARGET_COLOR[rest]]
        if not pos:
            return []
        if pre == "first":
            return [((pos[0],), 1.0)]
        if pre == "last":
            return [((pos[-1],), 1.0)]
        return [((i,), 1.0 / len(pos)) for i in pos]
    raise SystemExit(f"未知目标 {target}")


# ---------------------------------------------------------------- 动词 → 结果
def apply_verb(banner, pos, verb, qdist):
    """把 verb 作用在 pos 这几枚上 -> [(banner, prob)]。"""
    if verb == "increase_quality":
        nb = list(banner)
        for p in pos:
            s, t, tr = nb[p]
            nb[p] = (s, min(5, t + 1), tr)
        return [(tuple(nb), 1.0)]

    if verb == "roll_quality":
        choices = []
        for p in pos:                       # 重选保证换一个阶
            opts = [(t, q) for t, q in qdist.items() if t != banner[p][1]]
            z = sum(q for _, q in opts)
            choices.append([(t, q / z) for t, q in opts])
        out = []
        for combo in itertools.product(*choices):
            nb, pr = list(banner), 1.0
            for p, (t, q) in zip(pos, combo):
                nb[p] = (banner[p][0], t, banner[p][2])
                pr *= q
            out.append((tuple(nb), pr))
        return out

    if verb == "roll_trait":
        choices = [[tr for tr in BC.TRAITS if tr != banner[p][2]] for p in pos]
        out = []
        for combo in itertools.product(*choices):
            nb = list(banner)
            for p, tr in zip(pos, combo):
                nb[p] = (banner[p][0], banner[p][1], tr)
            out.append(tuple(nb))
        return [(b, 1.0 / len(out)) for b in out]

    if verb == "roll_stat":
        # 保证换成不同的一项，且同一战旗上不重复统计项
        keep = {banner[i][0] for i in range(len(banner)) if i not in pos}
        choices = [[s for s in FS.COLOR[COLOR_OF[banner[p][0]]]
                    if s != banner[p][0] and s not in keep] for p in pos]
        out = []
        for combo in itertools.product(*choices):
            if len(set(combo)) != len(combo):
                continue
            nb = list(banner)
            for p, s in zip(pos, combo):
                nb[p] = (s, banner[p][1], banner[p][2])
            out.append(tuple(nb))
        return [(b, 1.0 / len(out)) for b in out]

    raise SystemExit(f"未知动词 {verb}")


def directed_tiers(cur, up, qdist):
    """新阶数严格高于/低于 cur，权重取 qdist 在该侧的条件分布。
    没有可去的阶（升到顶 / 降到底）时返回原阶，即这一枚白费。"""
    side = [t for t in qdist if (t > cur if up else t < cur)]
    if not side:
        return [(cur, 1.0)]
    z = sum(qdist[t] for t in side)
    return [(t, qdist[t] / z) for t in side]


def untargeted(banner, verb, qdist, model):
    n = len(banner)
    if verb == "increase_one_quality":
        # 2026-08-17 实测：客户端把「随机提升一项品质」的**候选**用黄框标出来，
        # 框住的恰好是所有未满阶的徽标 —— 5 阶的一个都没框。所以它是在
        # 「未满阶」里均匀抽，不是在 5 枚里均匀抽然后撞到满阶就空放。
        elig = [p for p in range(n) if banner[p][1] < 5]
        if not elig:
            return []
        out = []
        for p in elig:
            for t, q in directed_tiers(banner[p][1], True, qdist):
                out.append((tuple((s, t, tr) if i == p else (s, tt, tr)
                                  for i, (s, tt, tr) in enumerate(banner)),
                            q / len(elig)))
        return out

    if verb == "increase_two_decrease_one":
        # 2026-08-08 实测：核心 推塔5→1、团战2→4、GPM2→4。一次操作跨了 4 阶和 2 阶，
        # 所以它不是「走一格」，而是**带方向的重选品质** —— 两枚重选到更高的阶、
        # 一枚重选到更低的阶。step / indep 两个模型留着只是为了复算旧结论。
        out = []
        if model == "reroll":
            # 5 枚之后「升两降一」不再是分割：降 1 枚、升 2 枚、剩下 n-3 枚不动。
            # 分配数 = n * C(n-1, 2)，3 枚时退化回原来的 3 种。
            # 同 increase_one_quality（2026-08-17 的黄框实测）：只在**可动**的徽标里抽 ——
            # 要降的那枚必须 >1 阶，要升的两枚必须 <5 阶。以前是在全部 n*C(n-1,2) 种分配上
            # 均匀抽、抽到不可能的方向就整支概率蒸发（总质量 <1，残量被当成空放），
            # 那等于系统性地低估了这个操作。
            picks = [(d, u) for d in range(n) if banner[d][1] > 1
                     for u in itertools.combinations(
                         [p for p in range(n) if p != d and banner[p][1] < 5], 2)]
            if not picks:
                return []
            for down, ups in picks:
                for td, qd in directed_tiers(banner[down][1], False, qdist):
                    for t0, q0 in directed_tiers(banner[ups[0]][1], True, qdist):
                        for t1, q1 in directed_tiers(banner[ups[1]][1], True, qdist):
                            nt = {down: td, ups[0]: t0, ups[1]: t1}
                            out.append((tuple((s, nt.get(i, tt), tr)
                                              for i, (s, tt, tr) in enumerate(banner)),
                                        qd * q0 * q1 / len(picks)))
        elif model == "step":    # 旧模型：升降各 ±1 阶，已被 2026-08-08 的观测证伪
            for down in range(n):
                out.append((tuple(
                    (s, min(5, t + 1) if i != down else max(1, t - 1), tr)
                    for i, (s, t, tr) in enumerate(banner)), 1 / n))
        else:                    # indep：±1 阶且升降目标独立抽，可能撞在同一枚上
            combos = [(u, d) for u in itertools.combinations_with_replacement(range(n), 2)
                      for d in range(n)]
            for ups, down in combos:
                d = [0] * n
                for u in ups:
                    d[u] += 1
                d[down] -= 1
                out.append((tuple(
                    (s, min(5, max(1, t + d[i])), tr)
                    for i, (s, t, tr) in enumerate(banner)), 1 / len(combos)))
        merged = {}
        for b, p in out:
            merged[b] = merged.get(b, 0.0) + p
        return list(merged.items())
    raise SystemExit(verb)


def outcomes(banner, verb, target, qdist, model):
    """-> [(banner, prob)]；None = 灰掉；"CHOOSE" = 由玩家指定徽标，另行处理。"""
    if verb in UNTARGETED:
        return untargeted(banner, verb, qdist, model)
    if target == "choose_one":
        return "CHOOSE"
    hits = hit_sets(banner, target)
    if not hits:
        return None
    merged = {}
    for pos, p in hits:
        for nb, q in apply_verb(banner, pos, verb, qdist):
            merged[nb] = merged.get(nb, 0.0) + p * q
    return list(merged.items())


# ---------------------------------------------------------------- 评估
def period_values_fixed(Vw, Vl, n_series, p_win, n_sims, rng):
    """正赛期：固定 n_series 个系列赛，每个按 p_win 独立定胜负，取最好的那个。

    小组赛期的名次桶（4-0 打 4 场、3-2 打 6 场）是瑞士轮的产物，正赛是淘汰赛，
    场次取决于走多远，且没进淘汰赛整期 0 分（`fantasy_model.py` 的假设，尚未证实）。
    「归零」对一面战旗的所有候选操作是同一个常数因子，不改变操作排序，所以这里
    只保留「打几个系列赛」这一个参数，把它显式化而不是假装知道。
    """
    n_comb = Vw.shape[1] if len(Vw) else Vl.shape[1]
    best = np.zeros((n_sims, n_comb))

    def draw(pool, k):
        return pool[rng.integers(0, len(pool), k)]

    for _ in range(n_series):
        is_win = (rng.random(n_sims) < p_win)[:, None]
        pw, pl = (Vw, Vl) if len(Vw) else (Vl, Vl)
        if not len(Vl):
            pl = Vw
        g1 = np.where(is_win, draw(pw, n_sims), draw(pl, n_sims))
        g2 = np.where(is_win, draw(pw, n_sims), draw(pl, n_sims))
        g3 = np.where(is_win, draw(pl, n_sims), draw(pw, n_sims))
        three = (rng.random(n_sims) < P_THREE)[:, None]
        top2 = np.where(three, g1 + g2 + g3 - np.minimum(np.minimum(g1, g2), g3),
                        g1 + g2)
        best = np.maximum(best, top2)
    return best


def values_many(rows, bcol, banners, n_sims, seed, chunk=400):
    """所有候选战旗一次算完，共用同一批随机数。

    5 枚徽标之后枚举空间上千，逐个调 evaluate() 会跑几分钟。这里直接把每个候选
    的权重摊成 M 的一列，V = X @ M 一次算出所有候选的逐局分，再跑一次 period_values。
    period_values 的抽样只依赖 (局数, 名次分布, n_sims)，与列数无关 ⇒ 分批不影响
    配对性，同一个 seed 下第 1 批和第 5 批看到的是同一批局号。
    """
    X, win = game_matrix(rows)
    out = []
    for i in range(0, len(banners), chunk):
        part = banners[i:i + chunk]
        M = np.zeros((len(STATS), len(part)))
        for j, b in enumerate(part):
            for (s, _, _), wi in zip(b, BC.multipliers(b)):
                M[IDX[s], j] += wi
        V = X @ M
        rng = np.random.default_rng(seed)
        if isinstance(bcol, tuple):          # ("fixed", n_series, p_win)
            _, n_series, p_win = bcol
            P = period_values_fixed(V[win], V[~win], n_series, p_win, n_sims, rng)
        else:
            P = period_values(V[win], V[~win], bcol, n_sims, rng)
        out.append(P.mean(axis=0))
    return np.concatenate(out)


class Evaluator:
    """一个槽位一套逐局分布 + 一套赛制结构。

    data="replay"（默认）走 Valve 录像解析的量 —— 它是唯一有莲花和占领观察者的来源，
    而且 13 项与 OpenDota 逐行相同、狂石和痛苦魔方我们原来算错三倍
    （见 docs/06-banner.md §4.5）。data="opendota" 保留下来复算旧结论。

    structure="group" 用小组赛的名次桶；"fixed" 用固定场次 + 该槽位自己的胜率，
    对应正赛期。胜率取该槽位的历史逐局胜率，只用来决定从赢局池还是输局池抽。
    """

    def __init__(self, teams_of, n_sims, seed, data="replay", leagues=None,
                 structure="group", n_series=6):
        self.n_sims, self.seed = n_sims, seed
        teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
        _, by_od = FS.load()
        roles = FS.assign_roles(by_od, teams)
        if data == "replay":
            _, by_rep = IR.load(leagues)
            slots = IR.slot_rows(by_rep, roles, teams)
        else:
            slots = role_game_scores(by_od, roles, teams)
        self.slots = slots
        if structure == "group":
            B = np.load(os.path.join(ROOT, "data", "sim_buckets_boot.npy"))
            names = json.load(open(os.path.join(ROOT, "data",
                                                "sim_teams_boot.json")))
            tcol = {n: k for k, n in enumerate(names)}
            rng = np.random.default_rng(31337)
            Bs = B[rng.integers(0, B.shape[0], n_sims)]
        self.rows = {}
        for role, t in teams_of.items():
            if (t, role) not in slots:
                raise SystemExit(f"没有 {t} 的{ROLE_CN[role]}逐局数据"
                                 f"（data={data}）")
            rws = slots[(t, role)]
            if structure == "group":
                self.rows[role] = (rws, Bs[:, tcol[t]])
            else:
                p_win = float(np.mean([r["win"] for r in rws]))
                self.rows[role] = (rws, ("fixed", n_series, p_win))
        self.cache = {}

    def n(self, role):
        return len(self.rows[role][0])

    def __call__(self, role, banners):
        todo = [b for b in banners if (role, b) not in self.cache]
        if todo:
            rows, bcol = self.rows[role]
            v = values_many(rows, bcol, todo, self.n_sims, self.seed)
            for b, x in zip(todo, v):
                self.cache[(role, b)] = float(x)
        return np.array([self.cache[(role, b)] for b in banners])


MISSING = set(FS.UNAVAILABLE)      # 由 main() 按数据源覆写


def scoreable(banner):
    """没有逐局数据的统计项，落到它们身上的结果算不了分。
    录像数据源下 MISSING 是空集（莲花和观察者都有），OpenDota 源下是那两项。"""
    return not any(s in MISSING for s, _, _ in banner)


def assess(ev, role, banner, outs):
    keep = [(b, p) for b, p in outs if scoreable(b)]
    if not keep:
        return None
    dropped = 1.0 - sum(p for _, p in keep)
    z = sum(p for _, p in keep)
    bs = [b for b, _ in keep]
    pr = np.array([p / z for _, p in keep])
    v = ev(role, [banner] + bs)
    base, alt = v[0], v[1:]
    order = np.argsort(-alt)
    return {"base": base, "delta": float((alt * pr).sum()) - base,
            "p_better": float(pr[alt > base].sum()),
            "p_worse": float(pr[alt < base].sum()), "dropped": dropped,
            "rows": [(bs[k], float(alt[k]), float(pr[k])) for k in order]}


def lookahead(ev, role, banner, verb, target, qdist, model, k,
              max_states=3000):
    """有 k 次机会反复用同一个操作时，这面战旗值多少。

    一步期望把操作后的状态当终局，只要代币还在这就是错的：摇坏了可以再摇。
    正确的量是带停时的动态规划

        V_k(s) = max( v(s), Σ_s' P(s'|s) · V_{k-1}(s') )      V_0(s) = v(s)

    max 里的第一项是「停在这里、把剩下的代币拿去别处」，第二项是「再摇一次」。
    这个 max 就是多次机会的全部价值来源 —— 下行可以被后续尝试截断，上行留着。

    状态空间是该操作能到达的闭包（品质类操作 = 被瞄准那几枚的阶数组合），
    对单枚/双枚目标很小；瞄准全部徽标的操作会爆炸，超过 max_states 就放弃。
    返回 (V_k − v(s), 各步的 V, 到达更好状态的概率)。
    """
    import collections
    seen, dq, trans = {banner}, collections.deque([banner]), {}
    while dq:
        s0 = dq.popleft()
        o = outcomes(s0, verb, target, qdist, model)
        if o is None or o == "CHOOSE":
            return None
        trans[s0] = o
        for ns, _ in o:
            if ns not in seen:
                if len(seen) >= max_states:
                    return None
                seen.add(ns)
                dq.append(ns)
    states = [s0 for s0 in seen if scoreable(s0)]
    if banner not in states:
        return None
    idx = {s0: i for i, s0 in enumerate(states)}
    v = ev(role, states)
    V = v.copy()
    hist = []
    for _ in range(k):
        EV = np.array([sum(p * V[idx[ns]] for ns, p in trans[s0]
                           if ns in idx) for s0 in states])
        V = np.maximum(v, EV)
        hist.append(float(V[idx[banner]]))
    # W_k = 强制走第一步、之后还有 k-1 次自由机会。W_1 就是一步期望。
    # W_k - W_1 = 「摇坏了还能再摇」这个补救权本身值多少。
    V = v.copy()
    W = []
    for _ in range(k):
        W.append(float(sum(p * V[idx[ns]] for ns, p in trans[banner]
                           if ns in idx)))
        EV = np.array([sum(p * V[idx[ns]] for ns, p in trans[s0]
                           if ns in idx) for s0 in states])
        V = np.maximum(v, EV)
    return {"base": float(v[idx[banner]]), "V": hist, "W": W,
            "n_states": len(states), "gain": hist[-1] - float(v[idx[banner]])}


def best_choose_one(ev, role, banner, verb, qdist):
    """『自选一枚』：先挑徽标再摇，所以按每枚的 E[Δ] 取最大。"""
    best = None
    for p in range(len(banner)):
        a = assess(ev, role, banner, apply_verb(banner, (p,), verb, qdist))
        if a and (best is None or a["delta"] > best[1]["delta"]):
            best = (p, a)
    return best


# ---------------------------------------------------------------- 序列 I/O
def parse_op(tok):
    verb, _, target = tok.partition("/")
    target = target or "none"
    if verb not in TARGETED | UNTARGETED:
        raise SystemExit(f"未知动词 {verb}；可选 {sorted(TARGETED | UNTARGETED)}")
    if verb in UNTARGETED:
        target = "none"
    elif target == "none":
        raise SystemExit(f"{verb} 需要一个目标，如 {verb}/emerald")
    return verb, target


def op_cn(verb, target):
    t = TARGET_CN.get(target, target)
    return f"{VERB_CN[verb]}·{t}" if t else VERB_CN[verb]


def load_log():
    return json.load(open(LOG_PATH))


def save_log(log):
    json.dump(log, open(LOG_PATH, "w"), indent=1, ensure_ascii=False)


def teams_of(log, upto=None):
    """队伍是免费可换的，所以从序列里往回找最近一条写了 teams 的快照。"""
    for st in reversed(log["banner_states"][:upto]):
        if "teams" in st:
            return dict(st["teams"])
    raise SystemExit("序列里没有任何一条记了 teams，补一条再跑")


def state_at(st):
    return {r: tuple(tuple(x) for x in st[r]) for r in ROLES if r in st}


def current(log):
    """序列末项 = 现在。历史条目一律只读。"""
    if not log["banner_states"]:
        raise SystemExit("banner_states 是空的")
    st = log["banner_states"][-1]
    banners = state_at(st)
    tm = teams_of(log)
    for r, b in banners.items():
        got = [round(100 * x) for x in BC.multipliers(b)]
        want = st.get("shown", {}).get(r)
        if want and got != list(want):
            raise SystemExit(
                f"❌ 末项的{ROLE_CN[r]}对不上：算出 {got}，记的 shown 是 {want}。"
                f"\n   历史不要改，补一条更正的快照。")
    return st, banners, tm


def parse_banner_arg(s):
    """core:towers/5/benevolent,teamfight/1/benevolent,gpm/3/fractal"""
    role, _, rest = s.partition(":")
    if role not in ROLES or not rest:
        raise SystemExit("格式：core:stat/tier/trait,stat/tier/trait,stat/tier/trait")
    out = []
    for part in rest.split(","):
        stat, tier, trait = part.strip().split("/")
        if stat not in COLOR_OF:
            raise SystemExit(f"未知统计项 {stat}")
        if trait not in BC.TRAITS + ["incorruptible", "base"]:
            raise SystemExit(f"未知特性 {trait}")
        out.append((stat, int(tier), trait))
    if len(out) not in (3, 5):
        raise SystemExit("一面战旗 3 枚（小组赛期）或 5 枚（国际邀请赛期）徽标")
    return role, tuple(out)


def do_apply(args, log):
    """追加一条新快照。历史条目不动。"""
    st, banners, tm = current(log)
    role, new = parse_banner_arg(args.apply)
    got = [round(100 * x) for x in BC.multipliers(new)]
    if args.shown:
        want = [int(x) for x in args.shown.split(",")]
        if got != want:
            raise SystemExit(f"❌ 按 {emb(new)} 算出 {got}，你抄的是 {want}。"
                             f"有一处抄错了，别写进序列。")
    else:
        print(f"⚠ 没给 --shown，无法对账。算出的倍率是 {got}，自己核一眼。")

    via = None
    if args.via == "reroll_options":     # 只刷新选项：战旗不动，只掉 1 枚代币
        via = {"op": "reroll_options", "applied_to": None}
        if new != banners[role]:
            raise SystemExit("--via reroll_options 时战旗状态不该变")
    elif args.via:
        verb, target = parse_op(args.via)
        outs = outcomes(banners[role], verb, target,
                        BC.QDIST[args.qdist], args.i2d1)
        reach = set() if outs in (None, "CHOOSE") else {b for b, _ in outs}
        if outs == "CHOOSE":
            for p in range(3):
                reach |= {b for b, _ in
                          apply_verb(banners[role], (p,), verb, BC.QDIST[args.qdist])}
        via = {"op": f"{verb}/{target}", "applied_to": role,
               "reachable": new in reach}
        if not reach:
            print(f"⚠ {op_cn(verb, target)} 在{ROLE_CN[role]}上是灰的，对不上。")
        elif new in reach:
            print(f"✓ 新状态在 {op_cn(verb, target)} 的可达结果里，机制模型没被证伪。")
        else:
            print(f"⚠ 新状态**不在** {op_cn(verb, target)} 的可达结果里 —— "
                  f"要么抄错了，要么我们对这个操作的理解是错的。已如实记进序列。")
    if args.predicted is not None:
        via = (via or {})
        via["predicted_delta"] = args.predicted

    banners[role] = new
    if args.team:
        if args.team not in json.load(open(os.path.join(ROOT, "data",
                                                        "teams.json"))):
            raise SystemExit(f"teams.json 里没有 {args.team}")
        print(f"   队伍 {tm[role]} -> {args.team}")
        tm[role] = args.team
    snap = {"tokens_before": args.tokens if args.tokens is not None
            else (st.get("tokens_before", 0) - 1),
            # period 决定赛制结构（小组赛的名次桶 vs 正赛的固定场次），
            # 不继承下来的话每次 --apply 之后打分口径都会悄悄变回上一期
            **({"period": st["period"]} if "period" in st else {}),
            **{r: [list(x) for x in b] for r, b in banners.items()},
            "teams": tm,
            "shown": {r: [round(100 * x) for x in BC.multipliers(b)]
                      for r, b in banners.items()}}
    if args.as_of:
        snap["as_of"] = args.as_of
    if via:
        snap["via"] = via
    if args.chosen:
        snap["chosen"] = args.chosen
    log["banner_states"].append(snap)
    save_log(log)
    print(f"已追加第 {len(log['banner_states'])} 条快照："
          f"{ROLE_CN[role]} {emb(new)}   倍率 {got}   代币 {snap['tokens_before']}")


def do_record(log, ops, chosen, state):
    log["draws"].append({
        "tokens_before": state.get("tokens_before"),
        "banner_open": state.get("open"),
        "censored": False,
        "options": [{"verb": v, "target": t, "cn": op_cn(v, t)} for v, t in ops],
        "chosen": chosen,
    })
    save_log(log)
    print(f"\n已记入 draws（第 {len(log['draws'])} 条）")


def do_history(log):
    tm = teams_of(log)
    print(f"队伍：" + "  ".join(f"{ROLE_CN[r]}={tm[r]}" for r in ROLES))
    print(f"\n{'#':<3}{'代币':>5}  {'核心':<34}{'中单':<34}{'辅助':<34}来源")
    for i, st in enumerate(log["banner_states"], 1):
        b = state_at(st)
        via = st.get("via", {})
        tag = via.get("op", "")
        if tag == "reroll_options":
            tag = "只刷新选项"
        elif tag:
            v, _, t = tag.partition("/")
            tag = op_cn(v, t) + f"→{ROLE_CN[via.get('applied_to','?')]}"
            if via.get("reachable") is False:
                tag += " ⚠不可达"
            if "predicted_delta" in via:
                tag += f"  预测{via['predicted_delta']:+,.0f}"
        print(f"{i:<3}{st.get('tokens_before','?'):>5}  "
              + "".join(f"{emb_short(b[r]) if r in b else '—':<34}" for r in ROLES)
              + tag)
    n = len(log["draws"])
    cens = sum(1 for d in log["draws"] if d.get("censored"))
    print(f"\ndraws {n} 条（其中 censored {cens} 条不能用于估计选项池分布）")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ops", nargs="*", help="三个选项，如 roll_stat/emerald")
    ap.add_argument("--qdist", default="温和递减", choices=list(BC.QDIST),
                    help="重选品质时的阶数分布；重选品质类操作会对三种都报一遍")
    ap.add_argument("--i2d1", default="reroll", choices=("reroll", "step", "indep"),
                    help="『升两降一』的机制模型。reroll=带方向的重选品质（2026-08-08 "
                         "实测支持，默认）；step/indep=旧的 ±1 阶模型，已被证伪，留作复算")
    ap.add_argument("--data", default="replay", choices=("replay", "opendota"),
                    help="逐局数据源。replay=Valve 录像解析（默认，唯一有莲花/观察者）")
    ap.add_argument("--leagues", default=None,
                    help="只用这些 leagueid（逗号分隔）。19719=TI，19785=石油杯")
    ap.add_argument("--structure", default=None, choices=("group", "fixed"),
                    help="赛制结构。默认按快照的 period 自动选：小组赛期=group，TI 期=fixed")
    ap.add_argument("--series", type=int, default=6, help="fixed 结构下打几个系列赛")
    ap.add_argument("--sims", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--detail", action="store_true", help="展开每个结果")
    ap.add_argument("--history", action="store_true", help="打印整条序列")
    ap.add_argument("--scan", action="store_true",
                    help="扫描全部 70 种操作 × 三面战旗，列出所有 E[Δ] 为正的组合。"
                         "这是代币的购物清单：看到清单上的选项就点，看不到就刷新")
    ap.add_argument("--top", type=int, default=40, help="--scan 最多列几行")
    ap.add_argument("--lookahead", type=int, default=0, metavar="K",
                    help="给出的每个操作，额外算「有 K 次机会反复用它」的带停时价值")
    ap.add_argument("--record", action="store_true", help="把这次刷新追加进 draws")
    ap.add_argument("--chosen", default=None, help="最后点了什么（配合 --record/--apply）")
    ap.add_argument("--apply", default=None,
                    help="操作后追加新快照：core:towers/5/benevolent,...")
    ap.add_argument("--shown", default=None, help="配合 --apply 对账：270,150,150")
    ap.add_argument("--tokens", type=int, default=None, help="操作后剩余代币")
    ap.add_argument("--via", default=None, help="这条快照是哪个操作产生的，会做可达性检验")
    ap.add_argument("--team", default=None,
                    help="同时改这个槽位的队伍（换队免费，但会改变打分基准）")
    ap.add_argument("--predicted", type=float, default=None,
                    help="操作前脚本给的 E[Δ]，存进序列以便事后校准")
    ap.add_argument("--as-of", dest="as_of", default=None)
    args = ap.parse_args()

    log = load_log()
    if args.history:
        return do_history(log)
    if args.apply:
        return do_apply(args, log)
    if not args.ops and not args.scan:
        ap.error("给我三个选项，或者用 --apply / --history / --scan")

    st, banners, tm = current(log)
    ops = [parse_op(t) for t in args.ops]
    structure = args.structure or ("fixed" if st.get("period") == "ti" else "group")
    lg = [int(x) for x in args.leagues.split(",")] if args.leagues else None
    global MISSING
    MISSING = set() if args.data == "replay" else set(FS.UNAVAILABLE)
    ev = Evaluator(tm, args.sims, args.seed, data=args.data, leagues=lg,
                   structure=structure, n_series=args.series)
    roles = [r for r in ROLES if r in banners]

    print(f"序列第 {len(log['banner_states'])} 条快照   代币 {st.get('tokens_before','?')}"
          f"   当前打开：{ROLE_CN.get(st.get('open'), '未记')}"
          f"   品质分布：{args.qdist}   升两降一：{args.i2d1}")
    print(f"数据源 {args.data}"
          + (f"（leagues {args.leagues}）" if args.leagues else "")
          + f"   赛制结构 {structure}"
          + (f"（{args.series} 个系列赛）" if structure == "fixed" else "")
          + "   槽位局数 " + " ".join(
              f"{ROLE_CN[r]}={ev.n(r)}" for r in ROLES if r in banners))
    print("\n当前战旗")
    for r in roles:
        b = banners[r]
        m = [round(100 * x) for x in BC.multipliers(b)]
        print(f"  {ROLE_CN[r]:<3}{tm[r]:<14}{emb(b):<50}{str(m):<20}"
              f"期望 {ev(r, [b])[0]:>9,.0f}")

    if args.scan:
        jobs = [(v, t) for v in sorted(TARGETED) for t in TARGET_CN if t != "none"]
        jobs += [(v, "none") for v in sorted(UNTARGETED)]
        rows = []
        for verb, target in jobs:
            for r in roles:
                o = outcomes(banners[r], verb, target, BC.QDIST[args.qdist],
                             args.i2d1)
                if o is None:
                    continue
                if o == "CHOOSE":
                    g = best_choose_one(ev, r, banners[r], verb,
                                        BC.QDIST[args.qdist])
                    if not g:
                        continue
                    pick, a = g
                    lbl = f"{op_cn(verb, target)}[第{pick+1}枚]"
                else:
                    a = assess(ev, r, banners[r], o)
                    if a is None:
                        continue
                    lbl = op_cn(verb, target)
                rows.append((a["delta"], lbl, r, a))
        rows.sort(key=lambda x: -x[0])
        pos = [x for x in rows if x[0] > 0]
        print(f"\n全操作空间扫描：{len(rows)} 个可用 (操作, 战旗) 组合，"
              f"{len(pos)} 个为正（{100*len(pos)/len(rows):.0f}%）\n")
        print(f"  {'操作':<30}{'战旗':<6}{'E[Δ]':>9}{'相对':>7}{'变好':>6}{'变差':>6}")
        for d, l, r, a in pos[:args.top]:
            print(f"  {l:<30}{ROLE_CN[r]:<6}{d:>+9,.0f}"
                  f"{100*d/a['base']:>6.1f}%{a['p_better']:>6.0%}{a['p_worse']:>6.0%}")
        if len(pos) > args.top:
            print(f"  …… 还有 {len(pos)-args.top} 个为正的（--top 调）")
        print("  最差三个：" + "   ".join(
            f"{l}→{ROLE_CN[r]} {d:,.0f}" for d, l, r, a in rows[-3:]))
        if args.record:
            do_record(log, [], args.chosen, st)
        return

    results = []
    print(f"\n{'':<2}{'操作':<28}{'战旗':<6}{'E[Δ]':>9}{'相对':>8}"
          f"{'变好':>6}{'变差':>6}   最好 / 最坏")
    for oi, (verb, target) in enumerate(ops, 1):
        label = op_cn(verb, target)
        for r in roles:
            banner = banners[r]
            outs = outcomes(banner, verb, target, BC.QDIST[args.qdist], args.i2d1)
            pick = None
            if outs is None:
                print(f"{oi:<2}{label:<28}{ROLE_CN[r]:<6}"
                      f"{'—— 灰（战旗上没有该颜色）':>26}")
                continue
            if outs == "CHOOSE":
                got = best_choose_one(ev, r, banner, verb, BC.QDIST[args.qdist])
                if not got:
                    continue
                pick, a = got
            else:
                a = assess(ev, r, banner, outs)
            if a is None:
                continue
            hi, lo = a["rows"][0], a["rows"][-1]
            note = f"   [选第{pick+1}枚]" if pick is not None else ""
            print(f"{oi:<2}{label:<28}{ROLE_CN[r]:<6}{a['delta']:>+9,.0f}"
                  f"{100*a['delta']/a['base']:>7.1f}%{a['p_better']:>6.0%}"
                  f"{a['p_worse']:>6.0%}   {emb_short(hi[0])} {hi[1]-a['base']:+,.0f}"
                  f"  /  {emb_short(lo[0])} {lo[1]-a['base']:+,.0f}{note}")
            if a["dropped"] > 1e-9:
                print(f"{'':<36}⚠ {a['dropped']:.0%} 的结果落在无逐局数据的统计项上"
                      f"（{'/'.join(CN[s] for s in sorted(FS.UNAVAILABLE))}），剔除后归一")
            results.append((a["delta"], oi, label, r, a, pick))

    if args.lookahead:
        print(f"\n多次机会的价值（带停时的动态规划，K={args.lookahead}）")
        print("  V_k = 可以不点（有停时）   W_k = 强制点第一次、之后 k-1 次自由")
        print(f"  {'操作':<26}{'战旗':<5}"
              + "".join(f"{'W'+str(i+1):>9}" for i in range(args.lookahead))
              + "".join(f"{'V'+str(i+1):>8}" for i in range(args.lookahead))
              + f"{'状态':>6}")
        for oi, (verb, target) in enumerate(ops, 1):
            for r in roles:
                la = lookahead(ev, r, banners[r], verb, target,
                               BC.QDIST[args.qdist], args.i2d1, args.lookahead)
                if la is None:
                    continue
                o = outcomes(banners[r], verb, target, BC.QDIST[args.qdist],
                             args.i2d1)
                a = assess(ev, r, banners[r], o) if isinstance(o, list) else None
                one = a["delta"] if a else float("nan")
                print(f"  {op_cn(verb, target):<26}{ROLE_CN[r]:<5}"
                      + "".join(f"{x-la['base']:>+9,.0f}" for x in la["W"])
                      + "".join(f"{x-la['base']:>+8,.0f}" for x in la["V"])
                      + f"{la['n_states']:>6}")

    if not results:
        print("\n三个选项没有一个能用。花 1 枚刷新选项。")
        return

    results.sort(key=lambda x: -x[0])
    top = results[0]
    print("\n" + "=" * 82)
    if top[0] <= 0:
        print(f"结论：最优组合也只有 {top[0]:+,.0f}（{top[2]} → {ROLE_CN[top[3]]}），"
              f"全部为负。**花 1 枚刷新选项**，不要点。")
    else:
        a, pick = top[4], top[5]
        print(f"结论：点第 {top[1]} 项「{top[2]}」，"
              f"**先切到{ROLE_CN[top[3]]}战旗**（{tm[top[3]]}）"
              + (f"，选第 {pick+1} 枚徽标" if pick is not None else ""))
        print(f"      E[Δ] = {a['delta']:+,.0f}（{100*a['delta']/a['base']:+.1f}%），"
              f"变好 {a['p_better']:.0%} / 变差 {a['p_worse']:.0%}")
        if len(results) > 1:
            s = results[1]
            print(f"      次优：{s[2]} → {ROLE_CN[s[3]]}  {s[0]:+,.0f}"
                  f"（差 {top[0]-s[0]:,.0f}）")
        if a["p_worse"] > 0.5:
            print("      ⚠ 变差概率过半，正的只是均值；不想吃方差就刷新选项。")
        print(f"      点完记得：--apply {top[3]}:... --shown ... --tokens ... "
              f"--via {[o for o in args.ops][top[1]-1]} --predicted {a['delta']:.0f}")

    if args.detail:
        for delta, oi, label, r, a, pick in results:
            print(f"\n--- {oi} {label} → {ROLE_CN[r]}   当前 {a['base']:,.0f}")
            for b, v, p in a["rows"]:
                print(f"    {v:>10,.0f} ({v-a['base']:>+8,.0f})  p={p:.3f}  {emb(b)}")

    if any(v == "roll_quality" for v, _ in ops):
        print("\n品质分布敏感性（重选品质类操作的 E[Δ]）")
        for dn, dist in BC.QDIST.items():
            line = []
            for oi, (verb, target) in enumerate(ops, 1):
                if verb != "roll_quality":
                    continue
                for r in roles:
                    outs = outcomes(banners[r], verb, target, dist, args.i2d1)
                    if outs in (None, "CHOOSE"):
                        continue
                    a = assess(ev, r, banners[r], outs)
                    if a:
                        line.append(f"{oi}→{ROLE_CN[r]} {a['delta']:+,.0f}")
            print(f"  {dn:<6}{'   '.join(line)}")

    if args.record:
        do_record(log, ops, args.chosen, st)


if __name__ == "__main__":
    main()
