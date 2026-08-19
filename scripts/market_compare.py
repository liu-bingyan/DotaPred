"""把 Polymarket 的 TI2026 盘口反解成队伍强度，看市场会填出什么样的对阵树。

市场是一个独立的、有真金白银背书的概率来源。它和我们的模型在哪里分歧、分歧多大，
既是对模型的外部校验，也是「该不该改填法」的另一条证据。

反解方式：8 支队的强度里固定一个为 0，其余 7 个自由，最小化

    Σ (首轮 Bo3 预测 − 盘口)²  +  Σ (冠军概率预测 − 盘口归一化)²

冠军概率由 bracket_optimize 的 16,384 种结果树精确算出，所以这个拟合是自洽的：
反解出来的强度放回同一套枚举里，会复现市场的边际概率。

没有 scipy，用带随机重启的坐标下降，7 维足够了。

    python3 scripts/market_compare.py
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bracket_optimize as BO  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2026-08-19 抓取。Match Winner 是 Bo3 系列赛胜者盘口。
PM_R1 = {("Iron Wing", "Team Spirit"): 0.535,
         ("Team Vision", "BoomBoys"): 0.735,
         ("Team Liquid", "Team Yandex"): 0.545,
         ("Team Falcons", "Nigma Galaxy"): 0.665}
PM_CHAMP = {"Team Vision": 0.410, "Team Spirit": 0.135, "Team Liquid": 0.125,
            "Team Falcons": 0.115, "Team Yandex": 0.083, "Iron Wing": 0.075,
            "BoomBoys": 0.070, "Nigma Galaxy": 0.061}
SUBMITTED = {"UB-A": "Iron Wing", "UB-B": "Team Vision", "UB-C": "Team Yandex",
             "UB-D": "Team Falcons", "UB-E": "Team Vision", "UB-F": "Team Falcons",
             "LB1-1": "BoomBoys", "LB1-2": "Team Liquid", "UB-G": "Team Vision",
             "LB2-1": "Team Yandex", "LB2-2": "Iron Wing", "LB3": "Iron Wing",
             "LB-F": "Team Falcons", "GF": "Team Vision"}


def bo3(p):
    return p * p * (3 - 2 * p)


def main():
    nodes, _, teams, tidx = BO.load(os.path.join(ROOT, "data", "playoff_probs_hl45.json"))
    labels = [nodes[n]["label"] for n in BO.ORDER]
    win, lose, pair = BO.enumerate_brackets(nodes, tidx)
    gf = labels.index("GF")
    tot = sum(PM_CHAMP.values())
    champ_target = np.array([PM_CHAMP[t] / tot for t in teams])
    r1 = [(tidx[a], tidx[b], p) for (a, b), p in PM_R1.items()]

    def loss(s):
        e = 0.0
        for i, j, p in r1:
            e += (bo3(1 / (1 + math.exp(-(s[i] - s[j])))) - p) ** 2
        pr = BO.outcome_probs(win, lose, nodes, s)
        ch = np.array([pr[win[:, gf] == k].sum() for k in range(8)])
        return e + float(((ch - champ_target) ** 2).sum())

    rng = np.random.default_rng(7)
    best_s, best_l = None, 1e9
    for restart in range(4):
        s = rng.normal(0, 0.5, 8)
        s[0] = 0.0
        cur = loss(s)
        step = 0.6
        for _ in range(90):
            improved = False
            for k in range(1, 8):
                for d in (step, -step):
                    t = s.copy()
                    t[k] += d
                    lv = loss(t)
                    if lv < cur - 1e-9:
                        s, cur, improved = t, lv, True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        if cur < best_l:
            best_s, best_l = s.copy(), cur
    s = best_s - best_s.mean()
    print(f"反解完成，残差平方和 {best_l:.5f}\n")

    ours = json.load(open(os.path.join(ROOT, "data", "playoff_probs_hl45.json")))["strength"]
    om = np.mean([ours[t] for t in teams])
    print(f"{'队伍':16s}{'市场隐含强度':>12}{'我们的强度':>11}{'差':>8}")
    for k, t in sorted(enumerate(teams), key=lambda kv: -s[kv[0]]):
        print(f"{t:16s}{s[k]:>12.2f}{ours[t] - om:>11.2f}{s[k] - (ours[t] - om):>+8.2f}")

    pr = BO.outcome_probs(win, lose, nodes, s)
    print(f"\n拟合复现检查  {'队伍':14s}{'市场':>8}{'反解后':>8}")
    ch = np.array([pr[win[:, gf] == k].sum() for k in range(8)])
    for k, t in sorted(enumerate(teams), key=lambda kv: -ch[kv[0]]):
        print(f"{'':14s}{t:14s}{champ_target[k]:>8.3f}{ch[k]:>8.3f}")

    es, eh = BO.evaluate(win, pair, pr, strict=False)
    b = int(np.argmax(es))
    got = {labels[i]: teams[win[b, i]] for i in range(14)}
    sub = [c for c in range(win.shape[0])
           if {labels[i]: teams[win[c, i]] for i in range(14)} == SUBMITTED][0]
    diff = [l for l in labels if got[l] != SUBMITTED[l]]
    print(f"\n按市场概率的最优填法：E[得分] {es[b]:.0f}   E[猜对] {eh[b]:.2f}")
    print(f"我们已提交的填法在市场概率下：E[得分] {es[sub]:.0f}"
          f"（排第 {int((es > es[sub]).sum()) + 1}，落后 {es[b] - es[sub]:.0f} 分）")
    print(f"差异 {len(diff)} 票：")
    for l in labels:
        mark = "  <<<" if got[l] != SUBMITTED[l] else ""
        print(f"   {l:6s} 已提交 {SUBMITTED[l]:14s} 市场 {got[l]:14s}{mark}")


if __name__ == "__main__":
    main()
