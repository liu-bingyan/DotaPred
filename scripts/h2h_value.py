"""直接交手记录，在评级之外还有没有增量预测力？

我们的填法有 3 票和 TI 小组赛的直接交手结果相反（Liquid 赢过 Yandex 和 Iron Wing，
Iron Wing 赢过 Falcons）。评级模型拟合的是经济差，它把这些 2-1 读成「势均力敌」。
问题是：该不该因为交手结果去覆盖评级？

这取决于「A 赢过 B」这件事本身有没有超出评级的信息。用 37 个 premium 赛事的断点后
系列赛测：在评级差之外，加入交手记录差，看系数是否显著、logloss 有没有改善。

  x_rating   断点前拟合的评级差（HL=45，队伍级）
  x_h2h_all  断点前 540 天内的净交手战绩，收缩 (wa-wb)/(wa+wb+2)
  x_h2h_ev   只算本赛事内部的净交手战绩，同样收缩

SE 按赛事聚类。

    python3 scripts/h2h_value.py
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def h2h_table(rows, lo, hi, leagueid=None):
    """[lo,hi) 区间内每对队伍的系列赛净胜负。"""
    ser = collections.defaultdict(list)
    for r in rows:
        if not (lo <= r["start_time"] < hi):
            continue
        if leagueid is not None and r["leagueid"] != leagueid:
            continue
        ser[r["series_id"] or -r["match_id"]].append(r)
    tab = collections.defaultdict(lambda: [0, 0])
    for g in ser.values():
        a = g[0]["radiant_team_id"]
        b = g[0]["dire_team_id"]
        if a == b:
            continue
        wa = sum((r["radiant_win"] if r["radiant_team_id"] == a else not r["radiant_win"])
                 for r in g)
        if wa * 2 == len(g):
            continue
        key = (min(a, b), max(a, b))
        tab[key][0 if (wa * 2 > len(g)) == (a == key[0]) else 1] += 1
    return tab


def feat(tab, a, b, k=2.0):
    key = (min(a, b), max(a, b))
    if key not in tab:
        return 0.0, 0
    wa, wb = tab[key] if a == key[0] else tab[key][::-1]
    return (wa - wb) / (wa + wb + k), wa + wb


def logistic(X, y, l2=1e-3, iters=4000, lr=0.05):
    X = np.column_stack([X, np.ones(len(y))])
    beta = np.zeros(X.shape[1])
    mom = np.zeros_like(beta)
    vel = np.zeros_like(beta)
    for step in range(1, iters + 1):
        p = 1 / (1 + np.exp(-X @ beta))
        g = X.T @ (y - p) - l2 * beta
        mom, vel = 0.9 * mom + 0.1 * g, 0.999 * vel + 0.001 * g * g
        beta += lr * (mom / (1 - 0.9**step)) / (np.sqrt(vel / (1 - 0.999**step)) + 1e-8)
    return beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hl", type=float, default=45.0)
    ap.add_argument("--window", type=float, default=540.0, help="交手记录回看天数")
    args = ap.parse_args()

    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") == "premium"]
    events = BE.find_events(rows, 40, 40.0, 0.8, 30, 15)

    X, y, grp, npair = [], [], [], []
    for e in events:
        brk = e["break_at"]
        post = [r for r in e["rows"] if r["start_time"] > brk]
        ser = BE.series_of(post)
        if len(ser) < 5:
            continue
        train = [r for r in rows if r["start_time"] <= brk]
        r_team, idx = BE.fit_rating(train, brk, args.hl, BE.PRIOR_L2, 20)
        if r_team is None:
            continue
        tab_all = h2h_table(rows, brk - args.window * 86400, brk)
        tab_ev = h2h_table(rows, e["rows"][0]["start_time"], brk, e["leagueid"])
        for g, a, a_won in ser:
            b = (g[0]["dire_team_id"] if g[0]["radiant_team_id"] == a
                 else g[0]["radiant_team_id"])
            if a not in idx or b not in idx:
                continue
            f_all, n_all = feat(tab_all, a, b)
            f_ev, _ = feat(tab_ev, a, b)
            X.append([r_team[idx[a]] - r_team[idx[b]], f_all, f_ev])
            y.append(float(a_won))
            grp.append(e["leagueid"])
            npair.append(n_all)
    X = np.array(X)
    y = np.array(y)
    grp = np.array(grp)
    npair = np.array(npair)
    print(f"{len(y)} 个断点后系列赛，{len(set(grp))} 个赛事；"
          f"其中 {int((npair > 0).sum())} 个在 {args.window:g} 天内有过交手记录\n")

    def report(name, cols, labels):
        b = logistic(X[:, cols], y)
        se = BE.cluster_se(X[:, cols], y, b, grp)
        p = 1 / (1 + np.exp(-(X[:, cols] @ b[:-1] + b[-1])))
        ll = -np.mean(y * np.log(np.clip(p, 1e-9, 1)) +
                      (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        print(f"--- {name}  (样内 logloss {ll:.4f}) ---")
        for k, lab in enumerate(labels):
            print(f"  {lab:26s} {b[k]:>+7.3f}  (SE {se[k]:.3f}, t {b[k] / se[k]:>+5.2f})")
        return ll

    l0 = report("只用评级差", [0], ["x_rating"])
    l1 = report("+ 540天内的交手记录", [0, 1], ["x_rating", "x_h2h_all"])
    l2 = report("+ 本赛事内的交手记录", [0, 2], ["x_rating", "x_h2h_event"])
    report("三个都放", [0, 1, 2], ["x_rating", "x_h2h_all", "x_h2h_event"])
    print(f"\nlogloss 改善：加 540天交手 {l0 - l1:+.4f}   加 本赛事交手 {l0 - l2:+.4f}")

    # 只看那些确实交手过的对阵，交手赢的一方后来赢了多少
    sel = npair > 0
    agree = ((X[sel, 1] > 0) == (y[sel] > 0.5))
    rating_agree = ((X[sel, 0] > 0) == (y[sel] > 0.5))
    print(f"\n在 {int(sel.sum())} 个「双方近期交手过」的系列赛里：")
    print(f"  按交手记录押（谁赢过押谁）  命中 {agree.mean():.1%}")
    print(f"  按评级押                    命中 {rating_agree.mean():.1%}")
    dis = sel & ((X[:, 0] > 0) != (X[:, 1] > 0)) & (X[:, 1] != 0)
    if dis.sum() > 10:
        print(f"  两者矛盾的 {int(dis.sum())} 个系列赛里，评级正确 "
              f"{(((X[dis, 0] > 0) == (y[dis] > 0.5)).mean()):.1%}")


if __name__ == "__main__":
    main()
