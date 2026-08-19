"""对模型不确定性积分之后的最优填法。

半衰期 45 / 60 / 150 三个配置在「跨断点预测淘汰赛」这个任务上互相分不出优劣
（两两 logloss 差 ≤0.0025，t ≤0.24），但它们给出的填法在 3 个节点上不一致。
拿三个点估计手工取平均是糊的：正确做法是把三个配置的 bootstrap 复制样本合并成
一个模型池 —— 池里既有参数不确定性（同一配置内的重抽样），也有配置不确定性
（跨半衰期），然后在这个池上求 E[得分] 的 argmax。

也报告每个节点上「换一支队会更好」的复制样本占比。50% 附近的节点说明那一票是
硬币，不管最终填谁都不影响期望分 —— 这比只报一个 argmax 诚实得多。

    python3 scripts/bracket_model_average.py \\
        data/playoff_probs_hl45.json data/playoff_probs_hl60.json \\
        data/playoff_probs_hl150.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bracket_optimize as BO  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 已提交到客户端的填法
SUBMITTED = {
    "UB-A": "Iron Wing", "UB-B": "Team Vision", "UB-C": "Team Yandex",
    "UB-D": "Team Falcons", "UB-E": "Team Vision", "UB-F": "Team Falcons",
    "LB1-1": "BoomBoys", "LB1-2": "Team Liquid", "UB-G": "Team Vision",
    "LB2-1": "Team Yandex", "LB2-2": "Iron Wing", "LB3": "Iron Wing",
    "LB-F": "Team Falcons", "GF": "Team Vision",
}


def load_pool(paths, teams):
    """把每个配置的 bootstrap 复制样本读成 [n_rep, 8] 的强度矩阵，拼在一起。"""
    mats, tags = [], []
    for p in paths:
        meta = json.load(open(p))
        boot = os.path.splitext(p)[0] + "_boot.npy"
        order = meta.get("bootstrap_teams") or teams
        cols = [order.index(t) for t in teams]
        if os.path.exists(boot):
            S = np.load(boot)[:, cols]
        else:
            S = np.array([[meta["strength"][t] for t in teams]])
        mats.append(S)
        tags += [f"HL={meta['half_life']:g}"] * len(S)
        print(f"  {os.path.basename(p):32s} 队伍HL={meta['half_life']:<5g} "
              f"选手HL={meta['player_half_life']:<5g} 复制样本 {len(S)}")
    return np.vstack(mats), tags


def score_of(cands, out_win, out_pair, prob, win, pair, strict=False):
    """给定若干候选填法的下标，算它们在这套结果概率下的 E[得分]。

    BO.evaluate 是拿同一批对阵树同时当候选和结果，这里只需要少数几个候选对全部
    结果打分，所以单写一个。"""
    lut = np.array(BO.SCORE, dtype=np.float64)
    out = np.empty(len(cands))
    for n, c in enumerate(cands):
        agree = win[c] == out_win
        if strict:
            agree &= pair[c] == out_pair
        out[n] = lut[agree.sum(1)] @ prob
    return out


def descendants(nodes):
    """节点 k 的下游 = 所有输入直接或间接依赖 k 的节点。翻掉 k 之后只有它们会跟着变。"""
    child = {nid: [] for nid in BO.ORDER}
    for nid in BO.ORDER:
        for spec in (nodes[nid]["from_1"], nodes[nid]["from_2"]):
            if spec:
                child[spec[1]].append(nid)
    out = {}
    for nid in BO.ORDER:
        seen, stack = set(), [nid]
        while stack:
            cur = stack.pop()
            for nxt in child[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out[nid] = seen
    return out


def index_of(win, teams, labels, picks):
    for c in range(win.shape[0]):
        if {labels[k]: teams[win[c, k]] for k in range(len(labels))} == picks:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probs", nargs="+")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "bracket_final.json"))
    args = ap.parse_args()

    nodes, _single, teams, tidx = BO.load(args.probs[0])
    labels = [nodes[n]["label"] for n in BO.ORDER]
    win, lose, pair = BO.enumerate_brackets(nodes, tidx)

    print("模型池：")
    S, tags = load_pool(args.probs, teams)
    print(f"合计 {len(S)} 个概率模型\n")

    prob = np.mean([BO.outcome_probs(win, lose, nodes, S[b]) for b in range(len(S))],
                   axis=0)
    es, eh = BO.evaluate(win, pair, prob, strict=False)
    best = int(np.argmax(es))
    sub = index_of(win, teams, labels, SUBMITTED)

    print(f"池上最优填法  E[得分] {es[best]:.0f}  E[猜对] {eh[best]:.2f}")
    if sub is not None:
        rank = int((es > es[sub]).sum()) + 1
        print(f"已提交的填法  E[得分] {es[sub]:.0f}  E[猜对] {eh[sub]:.2f}  "
              f"（16,384 套里排第 {rank}，落后最优 {es[best] - es[sub]:.0f} 分 = "
              f"{(es[best] - es[sub]) / es[best]:.2%}）")
    order = np.argsort(-es)[:args.top]
    print(f"前 {args.top} 名 E[得分]: " + " ".join(f"{es[i]:.0f}" for i in order))

    desc = descendants(nodes)
    pos = {nodes[nid]["label"]: k for k, nid in enumerate(BO.ORDER)}
    print(f"\n{'节点':7s} {'池上最优':16s} {'已提交':16s} {'翻这一票的代价':>14}")
    for k, nid in enumerate(BO.ORDER):
        lab = labels[k]
        alt = None
        if sub is not None:
            # 翻掉这一票，下游允许跟着重排（结构上必须），其余节点保持不变
            free = {pos[nodes[d]["label"]] for d in desc[nid]} | {k}
            fixed = [c for c in range(len(labels)) if c not in free]
            same = np.all(win[:, fixed] == win[sub, fixed], axis=1)
            cand = np.where(same & (win[:, k] != win[sub, k]))[0]
            alt = es[cand].max() - es[sub] if len(cand) else None
        cost = f"{alt:>+14.0f}" if alt is not None else f"{'—':>14}"
        star = "  <<<" if sub is not None and teams[win[best, k]] != SUBMITTED[lab] else ""
        print(f"{lab:7s} {teams[win[best, k]]:16s} "
              f"{(SUBMITTED[lab] if sub is not None else '-'):16s} {cost}{star}")

    # 池里有多少比例的模型认为「池上最优」比「已提交」好
    if sub is not None and best != sub:
        print("\n模型池投票（已提交 vs 池上最优）：")
        votes = {}
        for b in range(len(S)):
            p1 = BO.outcome_probs(win, lose, nodes, S[b])
            e1 = score_of([sub, best], win, pair, p1, win, pair)
            v = votes.setdefault(tags[b], [0, 0])
            v[0] += int(e1[1] > e1[0])
            v[1] += 1
        tot = [0, 0]
        for t, (a, n) in sorted(votes.items()):
            print(f"  {t:9s} 偏向池上最优 {a}/{n} = {a / n:>4.0%}")
            tot[0] += a
            tot[1] += n
        print(f"  {'合计':9s} 偏向池上最优 {tot[0]}/{tot[1]} = {tot[0] / tot[1]:>4.0%}"
              f"   （接近 50% = 这一票是硬币）")

    json.dump({"pool_size": int(len(S)),
               "best": {labels[k]: teams[win[best, k]] for k in range(len(labels))},
               "best_score": float(es[best]),
               "submitted_score": float(es[sub]) if sub is not None else None},
              open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
