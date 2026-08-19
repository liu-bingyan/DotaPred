"""可靠性分析：我们的概率到底准不准，以及和 Polymarket 比谁的锐度是真的。

方法借自 ajhildebrand/WinLossSequencesSpring24 的 calibration_plot/ —— 把预测概率
分桶，看每桶的实际胜率。那个 repo 用的是 Elo vs moneyline，这里是我们的评级 vs
Polymarket。两处扩展：

1. **校准斜率对真实结果拟合**，不是对市场拟合。logit(结果) = a + b·logit(预测)，
   b < 1 = 过度自信（概率该往 0.5 收），b > 1 = 过度保守。之前那个「我们 vs 市场
   斜率 1.87」只能说明两者相对更极端，判不了谁对。

2. **Brier 的 Murphy 分解** = 不可靠度 − 分辨度 + 不确定度。这能回答一个 logloss
   回答不了的问题：我们相对市场的优势是来自校准更准，还是来自区分能力更强。
   分辨度高 = 真的能把强弱分开；不可靠度低 = 报出来的数字可信。

    python3 scripts/calibration.py
"""

import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import break_effect as BE  # noqa: E402
import margin  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def brier_decomp(p, y, nbins=10):
    """Murphy 分解：BS = 不可靠度 − 分辨度 + 不确定度。"""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    bs = float(np.mean((p - y) ** 2))
    base = y.mean()
    unc = base * (1 - base)
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, nbins - 1)
    rel = res = 0.0
    for b in range(nbins):
        m = idx == b
        if not m.any():
            continue
        w = m.sum() / len(p)
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return bs, rel, res, unc


def calib_slope(p, y):
    """logit(结果) ~ a + b·logit(预测)。b<1 = 过度自信。返回 (b, SE)。"""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    y = np.asarray(y, float)
    beta = np.zeros(2)
    mom = np.zeros(2)
    vel = np.zeros(2)
    X = np.column_stack([x, np.ones(len(x))])
    for step in range(1, 4001):
        q = 1 / (1 + np.exp(-X @ beta))
        g = X.T @ (y - q)
        mom, vel = 0.9 * mom + 0.1 * g, 0.999 * vel + 0.001 * g * g
        beta += 0.05 * (mom / (1 - 0.9**step)) / (np.sqrt(vel / (1 - 0.999**step)) + 1e-8)
    q = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.pinv((X * (q * (1 - q))[:, None]).T @ X)
    return beta[0], math.sqrt(max(cov[0, 0], 0))


def reliability(name, p, y, edges):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    print(f"  {name}")
    print(f"    {'预测区间':14s}{'场次':>5}{'平均预测':>9}{'实际胜率':>9}{'偏差':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        print(f"    {f'{lo:.2f}-{hi:.2f}':14s}{int(m.sum()):>5}{p[m].mean():>9.3f}"
              f"{y[m].mean():>9.3f}{y[m].mean() - p[m].mean():>+8.3f}")


def symmetrise(p, y):
    """把每场按两个方向各算一次，可靠性图才在 0.5 两侧对称。"""
    return np.r_[p, 1 - np.asarray(p)], np.r_[y, 1 - np.asarray(y)]


def report(tag, series, edges):
    print(f"\n=== {tag} ===")
    for name, p, y in series:
        bs, rel, res, unc = brier_decomp(p, y)
        b, se = calib_slope(p, y)
        ll = -np.mean(np.asarray(y) * np.log(np.clip(p, 1e-9, 1))
                      + (1 - np.asarray(y)) * np.log(np.clip(1 - np.asarray(p), 1e-9, 1)))
        print(f"\n{name}: n={len(p)}  logloss {ll:.4f}  Brier {bs:.4f}")
        print(f"    不可靠度 {rel:.4f}（越小越好） 分辨度 {res:.4f}（越大越好）"
              f" 不确定度 {unc:.4f}")
        print(f"    校准斜率 b = {b:.2f} (SE {se:.2f})  "
              f"{'过度自信' if b < 1 - se else ('过度保守' if b > 1 + se else '与 1 无显著差异')}")
        ps, ys = symmetrise(p, y)
        reliability(name, ps, ys, edges)


def load_market():
    f = os.path.join(ROOT, "data", "raw", "polymarket_ti2026.json")
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    # ---------- A: TI2026 小组赛，我们 vs 市场 ----------
    mk = load_market()
    if mk:
        import market_backtest as MB
        aliases = {int(k): v for k, v in
                   json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
        rows = [r for r in margin.load_rich(aliases)
                if r.get("tier") in {"premium", "professional"}]
        import models
        lineups = models.load_lineups()
        teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
        tid = {n: v["team_id"] for n, v in teams.items()}
        roster = {n: [p["account_id"] for p in v["players"][:5]] for n, v in teams.items()}
        ours, mkt, ys = [], [], []
        for s in sorted(mk, key=lambda x: x["start"]):
            a, b = MB.NAME.get(s["team_a"]), MB.NAME.get(s["team_b"])
            if a not in tid or b not in tid:
                continue
            rt, idx, rp, pidx, beta, mu = MB.fit_at(rows, lineups, s["start"])
            if tid[a] not in idx or tid[b] not in idx:
                continue
            x = [rt[idx[tid[a]]] - rt[idx[tid[b]]]]
            pa = [rp[pidx[q]] for q in roster[a] if q in pidx]
            pb = [rp[pidx[q]] for q in roster[b] if q in pidx]
            x.append(np.mean(pa) - np.mean(pb) if len(pa) >= 4 and len(pb) >= 4 else mu[1])
            p1 = 1 / (1 + math.exp(-float(np.dot(beta[:2], x))))
            ours.append(p1 * p1 * (3 - 2 * p1))
            mkt.append(s["price_a"])
            ys.append(float(s["a_won"]))
        report("A  TI2026 小组赛 41 个系列赛：我们 vs Polymarket",
               [("我们的模型", ours, ys), ("Polymarket 赛前价", mkt, ys)],
               [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])

    # ---------- B: 大样本，只有我们（历史断点后系列赛）----------
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") == "premium"]
    events = BE.find_events(rows, 40, 40.0, 0.8, 30, 15)
    P, Y = [], []
    for e in events:
        post = [r for r in e["rows"] if r["start_time"] > e["break_at"]]
        ser = BE.series_of(post)
        if len(ser) < 5:
            continue
        train = [r for r in rows if r["start_time"] <= e["break_at"]]
        rt, idx = BE.fit_rating(train, e["break_at"], 45.0, BE.PRIOR_L2, 20)
        if rt is None:
            continue
        for g, a, a_won in ser:
            b = (g[0]["dire_team_id"] if g[0]["radiant_team_id"] == a
                 else g[0]["radiant_team_id"])
            if a not in idx or b not in idx:
                continue
            p1 = 1 / (1 + math.exp(-0.74 * (rt[idx[a]] - rt[idx[b]])))
            need = 3 if len(g) > 3 else 2
            P.append(sum(math.comb(need - 1 + k, k) * p1**need * (1 - p1) ** k
                         for k in range(need)))
            Y.append(float(a_won))
    report(f"B  37 个 premium 赛事的 {len(P)} 个断点后系列赛（只有队伍级评级）",
           [("我们的模型", P, Y)], [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])


if __name__ == "__main__":
    main()
