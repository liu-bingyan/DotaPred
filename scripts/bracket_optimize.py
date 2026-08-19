"""主赛事 14 个系列赛的最优填法。精确枚举，不用模拟。

填法空间和结果空间是同一个集合：每个节点二选一，一共 2^14 = 16,384 种完整对阵树。
所以既能穷举所有候选填法，也能对每种真实结果算出精确概率，E[得分] 是一个
16384 x 16384 的精确求和 —— 没有蒙特卡洛误差。

计分口径有歧义，本地化只写「pick the team you think is going to win each series」：

  loose  只看该节点的实际胜者是不是你选的队。你选的队从另一条路走到这个节点并
         赢了，也算对。押强队更值，因为它从败者组绕回来仍能兑现。
  strict 还要求该节点的两个参赛队也和你预测的一致，即整条路径都对。

两种都算，并检查最优解是否相同。

目标函数是凸的（边际收益递增：第 1 个对只值 120，第 14 个值 +1,080），所以
最大化 E[得分] 不等于最大化 E[猜对数] —— 应该押相关性高的世界线。两个都报。
"""

import argparse
import itertools
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE = [0, 120, 360, 720, 1200, 1800, 2520, 3360, 4320, 5400, 6600, 7920,
         9360, 10920, 12000]
# 解算顺序：每个节点的上游必须排在它前面
ORDER = [14, 15, 16, 17, 18, 19, 22, 23, 20, 24, 25, 26, 27, 21]


def bo_prob(p, need):
    return sum(math.comb(need - 1 + k, k) * p**need * (1 - p) ** k for k in range(need))


def load(prob_path):
    br = json.load(open(os.path.join(ROOT, "data", "ti2026_bracket.json")))
    nodes = {b["node_id"]: b for b in br["playoff"]}
    probs = json.load(open(prob_path))
    single = probs["single_game"]
    teams = sorted({nodes[n]["team_1"] for n in (14, 15, 16, 17)} |
                   {nodes[n]["team_2"] for n in (14, 15, 16, 17)})
    tidx = {t: k for k, t in enumerate(teams)}
    return nodes, single, teams, tidx


def enumerate_brackets(nodes, tidx):
    """遍历 2^14 种完整对阵树。结构与概率无关，所以只枚举一次。

    返回 win[N,14]  每个节点的胜者（队伍下标）
         lose[N,14] 每个节点的败者
         pair[N,14] 每个节点的对阵（无序两队编码成 8*min+max）
    """
    n_comb = 1 << len(ORDER)
    win = np.zeros((n_comb, len(ORDER)), dtype=np.uint8)
    lose = np.zeros((n_comb, len(ORDER)), dtype=np.uint8)
    pair = np.zeros((n_comb, len(ORDER)), dtype=np.uint8)
    for c, bits in enumerate(itertools.product((0, 1), repeat=len(ORDER))):
        st = {}
        for k, nid in enumerate(ORDER):
            b = nodes[nid]
            t1 = b["team_1"] or (st[b["from_1"][1]][0] if b["from_1"][0] == "W"
                                 else st[b["from_1"][1]][1])
            t2 = b["team_2"] or (st[b["from_2"][1]][0] if b["from_2"][0] == "W"
                                 else st[b["from_2"][1]][1])
            w, loser = (t2, t1) if bits[k] else (t1, t2)
            st[nid] = (w, loser)
            a, bb = tidx[t1], tidx[t2]
            win[c, k] = tidx[w]
            lose[c, k] = tidx[loser]
            pair[c, k] = min(a, bb) * 8 + max(a, bb)
    return win, lose, pair


def outcome_probs(win, lose, nodes, strength):
    """给定一组队伍强度，算每种完整结果的概率（各系列赛条件独立）。"""
    s = np.asarray(strength)
    single = 1.0 / (1.0 + np.exp(-(s[:, None] - s[None, :])))
    mats = {}
    for need in (2, 3):
        mats[need] = np.vectorize(lambda p, n=need: bo_prob(p, n))(single)
    need = np.array([3 if nodes[nid]["bo"] == 5 else 2 for nid in ORDER])
    logp = np.zeros(win.shape)
    for k, nd in enumerate(need):
        logp[:, k] = np.log(np.clip(mats[nd][win[:, k], lose[:, k]], 1e-12, 1))
    return np.exp(logp.sum(1))


def evaluate(win, pair, prob, strict, chunk=512):
    """每个候选填法的 E[得分] 与 E[猜对数]。候选和结果是同一批对阵树。"""
    lut = np.array(SCORE, dtype=np.float64)
    n = win.shape[0]
    exp_score = np.empty(n)
    exp_hits = np.empty(n)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        agree = win[lo:hi, None, :] == win[None, :, :]
        if strict:
            agree &= pair[lo:hi, None, :] == pair[None, :, :]
        k = agree.sum(2)                      # [chunk, n_outcomes]
        exp_score[lo:hi] = lut[k] @ prob
        exp_hits[lo:hi] = k @ prob
    return exp_score, exp_hits


def describe(nodes, teams, win_row, tidx):
    out = []
    st = {}
    for k, nid in enumerate(ORDER):
        b = nodes[nid]
        t1 = b["team_1"] or (st[b["from_1"][1]][0] if b["from_1"][0] == "W"
                             else st[b["from_1"][1]][1])
        t2 = b["team_2"] or (st[b["from_2"][1]][0] if b["from_2"][0] == "W"
                             else st[b["from_2"][1]][1])
        w = teams[win_row[k]]
        st[nid] = (w, t1 if w == t2 else t2)
        out.append((b["label"], t1, t2, w))
    return out


def dist_for(win, pair, prob, cand, strict):
    agree = win[cand] == win
    if strict:
        agree &= pair[cand] == pair
    k = agree.sum(1)
    d = np.bincount(k, weights=prob, minlength=15)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=os.path.join(ROOT, "data", "win_matrix.json"))
    ap.add_argument("--boot", default=None,
                    help="bootstrap 强度矩阵 .npy，对参数不确定性积分")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "bracket_choice.json"))
    args = ap.parse_args()

    nodes, single, teams, tidx = load(args.probs)
    print(f"概率来源 {os.path.relpath(args.probs, ROOT)}；8 队 {teams}\n")
    win, lose, pair = enumerate_brackets(nodes, tidx)

    meta = json.load(open(args.probs))
    if "strength" in meta:
        s0 = np.array([meta["strength"][t] for t in teams])
    else:   # 老格式只有胜率矩阵，从对第一支队的胜率反解相对强度
        base = teams[0]
        s0 = np.array([0.0 if t == base else
                       math.log(single[t][base] / (1 - single[t][base])) for t in teams])
    if args.boot:
        S = np.load(args.boot)
        order = meta.get("bootstrap_teams") or teams
        cols = [order.index(t) for t in teams]
        prob = np.mean([outcome_probs(win, lose, nodes, S[b, cols])
                        for b in range(S.shape[0])], axis=0)
        print(f"对 {S.shape[0]} 个 bootstrap 复制样本积分")
    else:
        prob = outcome_probs(win, lose, nodes, s0)
    print(f"枚举 {len(prob)} 种完整对阵树，概率和 = {prob.sum():.6f}\n")

    es_by = {}
    result = {}
    for strict in (False, True):
        tag = "strict 路径必须一致" if strict else "loose 只看该场胜者"
        es, eh = evaluate(win, pair, prob, strict)
        es_by[strict] = (es, eh)
        best = int(np.argmax(es))
        best_hits = int(np.argmax(eh))
        d = dist_for(win, pair, prob, best, strict)
        print(f"=== 口径：{tag} ===")
        print(f"最优填法 E[得分] = {es[best]:.0f}   E[猜对] = {eh[best]:.2f}   "
              f"P(>=8) = {d[8:].sum():.1%}")
        for label, t1, t2, w in describe(nodes, teams, win[best], tidx):
            print(f"   {label:6s} {t1:16s} vs {t2:16s} -> {w}")
        if best_hits != best:
            print(f"   （最大化 E[猜对数] 会给出另一套填法，E[得分] 只有 "
                  f"{es[best_hits]:.0f}，凸性确实在起作用）")
        else:
            print("   （最大化 E[猜对数] 给出同一套填法）")
        print(f"   猜对数分布: " + "  ".join(f"{i}:{d[i]:.1%}"
                                          for i in range(15) if d[i] > 0.005))
        order = np.argsort(-es)[:args.top]
        print(f"   前 {args.top} 名填法的 E[得分]: " +
              " ".join(f"{es[i]:.0f}" for i in order))
        result[("strict" if strict else "loose")] = {
            "expected_score": float(es[best]), "expected_hits": float(eh[best]),
            "picks": [{"label": l, "team_1": a, "team_2": b, "pick": w}
                      for l, a, b, w in describe(nodes, teams, win[best], tidx)],
            "hit_distribution": [float(x) for x in d],
        }
        print()

    same = ([p["pick"] for p in result["loose"]["picks"]]
            == [p["pick"] for p in result["strict"]["picks"]])
    print(f"两种口径的最优填法{'一致' if same else '不一致'}。")
    result["identical"] = bool(same)

    # 计分口径未知时的稳健选择：假设两种规则各一半可能，最大化两者的平均分。
    # 同时报告每套填法在另一种口径下要亏多少 —— 如果亏得少，这个歧义就不重要。
    esl, esr = es_by[False][0], es_by[True][0]
    bl, bs = int(np.argmax(esl)), int(np.argmax(esr))
    mix = 0.5 * esl + 0.5 * esr
    bm = int(np.argmax(mix))
    print(f"\n{'填法':22s}{'loose 下':>10}{'strict 下':>11}{'各半平均':>10}")
    for name, c in (("loose 最优", bl), ("strict 最优", bs), ("各半混合最优", bm)):
        print(f"{name:22s}{esl[c]:>10.0f}{esr[c]:>11.0f}{mix[c]:>10.0f}")
    print(f"选 loose 最优、结果规则是 strict：亏 {esr[bs] - esr[bl]:.0f} 分 "
          f"({(esr[bs] - esr[bl]) / esr[bs]:.1%})")
    print(f"选 strict 最优、结果规则是 loose：亏 {esl[bl] - esl[bs]:.0f} 分 "
          f"({(esl[bl] - esl[bs]) / esl[bl]:.1%})")
    if bm not in (bl, bs):
        print("混合最优是第三套填法：")
    for label, t1, t2, w in describe(nodes, teams, win[bm], tidx):
        print(f"   {label:6s} {t1:16s} vs {t2:16s} -> {w}")
    result["mixed"] = {
        "expected_loose": float(esl[bm]), "expected_strict": float(esr[bm]),
        "picks": [{"label": l, "team_1": a, "team_2": b, "pick": w}
                  for l, a, b, w in describe(nodes, teams, win[bm], tidx)]}
    json.dump(result, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
