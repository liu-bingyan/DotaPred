"""选手状态变量：能不能从小组赛的个人表现里提取出「状态」，用它帮助预测淘汰赛？

这是赛程断点问题的另一种建模方式。§7 的 break_effect 用的是队伍级经济差，这里换成
选手级的个人产出。

关键的构造问题：单个赛事内一支队的五个人每场都是同一批，所以任何直接反映胜负/经济
的选手指标都会和队伍级状态共线。为了得到真正的**选手**信号，把每场的个人表现对
「这局队伍打成什么样」做残差化 —— 剩下的才是「在同样的团队局势下，这个人比自己平常
好还是差」。

  perf        每场每人的表现分 = 若干每分钟指标在同位置内的 z 分均值
  perf_resid  上面对 (队伍经济差, 时长, 位置) 回归后的残差
  baseline    赛前 365 天该选手的 perf 均值
  form        本赛事均值 − baseline

三个问题按顺序回答：
  1. form 有没有持续性（断点前的 form 能不能预测断点后的 form）
  2. 有没有断点效应（跨断点的持续性 vs 不跨断点的持续性）
  3. 有没有增量预测力（在队伍评级 + 队伍级状态之外，选手 form 差还剩多少）

位置用「队内净资产排名」定，比 OpenDota 的 lane_role 可靠。

    python3 scripts/player_form.py
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
PS = os.path.join(ROOT, "data", "raw", "premium_player_stats.json")

# 参与表现分的每分钟指标，以及方向（死亡是负向）
STATS = [("gold_per_min", 1), ("xp_per_min", 1), ("last_hits", 1),
         ("hero_damage", 1), ("tower_damage", 1), ("kills", 1), ("assists", 1),
         ("deaths", -1), ("obs_placed", 1), ("stuns", 1)]


def load_perf():
    """每条 (account_id, match_id) 记录算一个表现分，并按位置 z 标准化。"""
    rows = json.load(open(PS))
    by_match = collections.defaultdict(list)
    for r in rows:
        by_match[r["match_id"]].append(r)

    recs = []
    for mid, g in by_match.items():
        if len(g) != 10:
            continue
        dur = max(g[0]["duration"], 1) / 60.0
        for side in (0, 1):
            team = [r for r in g if (r["player_slot"] < 128) == (side == 0)]
            if len(team) != 5:
                continue
            # 队内净资产排名 -> 位置 0..4
            order = sorted(team, key=lambda r: -(r.get("net_worth") or 0))
            for pos, r in enumerate(order):
                vals = {}
                for k, sign in STATS:
                    v = r.get(k)
                    vals[k] = sign * ((v or 0) / dur if k not in ("gold_per_min", "xp_per_min")
                                      else sign * (v or 0) / 1.0)
                recs.append(dict(account_id=r["account_id"], match_id=mid, pos=pos,
                                 side=side, start_time=r["start_time"],
                                 leagueid=r["leagueid"], duration=r["duration"],
                                 radiant_win=r["radiant_win"],
                                 radiant_team_id=r["radiant_team_id"],
                                 dire_team_id=r["dire_team_id"], **vals))
    # 按位置做 z 标准化
    for pos in range(5):
        sel = [r for r in recs if r["pos"] == pos]
        for k, _ in STATS:
            v = np.array([r[k] for r in sel], dtype=float)
            mu, sd = np.nanmean(v), np.nanstd(v) or 1.0
            for r, x in zip(sel, v):
                r["z_" + k] = (x - mu) / sd
    for r in recs:
        r["perf"] = float(np.mean([r["z_" + k] for k, _ in STATS]))
    return recs


def residualise(recs, gdpm):
    """把表现分对 (队伍经济差, 时长) 在每个位置内做线性回归，取残差。"""
    for pos in range(5):
        sel = [r for r in recs if r["pos"] == pos and r["match_id"] in gdpm]
        if len(sel) < 100:
            for r in [x for x in recs if x["pos"] == pos]:
                r["resid"] = r["perf"]
            continue
        X = np.array([[gdpm[r["match_id"]] * (1 if r["side"] == 0 else -1),
                       r["duration"] / 60.0, 1.0] for r in sel])
        y = np.array([r["perf"] for r in sel])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        for r, e in zip(sel, y - X @ beta):
            r["resid"] = float(e)
        for r in [x for x in recs if x["pos"] == pos and "resid" not in x]:
            r["resid"] = r["perf"]
    return recs


def mean_over(recs_by_player, acc, lo, hi, key, leagueid=None):
    v = [r[key] for r in recs_by_player.get(acc, [])
         if lo <= r["start_time"] < hi and (leagueid is None or r["leagueid"] == leagueid)]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-days", type=float, default=365.0)
    ap.add_argument("--key", default="resid", choices=["perf", "resid"])
    ap.add_argument("--min-games", type=int, default=4, help="计算 form 所需最少场次")
    args = ap.parse_args()

    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rich = [r for r in margin.load_rich(aliases) if r.get("tier") == "premium"]
    gdpm = {r["match_id"]: (r.get("final_gold_adv") or 0) / (max(r["duration"], 1) / 60.0)
            for r in rich}

    print("读取选手逐场数据…", flush=True)
    recs = residualise(load_perf(), gdpm)
    by_player = collections.defaultdict(list)
    for r in recs:
        by_player[r["account_id"]].append(r)
    print(f"  {len(recs)} 条选手-比赛记录，{len(by_player)} 名选手\n")

    events = BE.find_events(rich, 40, 40.0, 0.8, 30, 15)
    K = args.key

    # ---------- 问题 1/2：form 的持续性，以及跨不跨断点有没有差别 ----------
    pre_post, pre_mid = [], []
    for e in events:
        rs = e["rows"]
        ev_start = rs[0]["start_time"]
        brk = e["break_at"]
        pre = [r for r in rs if r["start_time"] <= brk]
        cut1 = pre[int(len(pre) * 0.7)]["start_time"]
        lo = ev_start - args.baseline_days * 86400
        players = {r["account_id"] for r in recs if r["leagueid"] == e["leagueid"]}
        for acc in players:
            base, nb = mean_over(by_player, acc, lo, ev_start, K)
            if base is None or nb < 10:
                continue
            f1, n1 = mean_over(by_player, acc, ev_start, cut1, K, e["leagueid"])
            f2, n2 = mean_over(by_player, acc, cut1, brk + 1, K, e["leagueid"])
            f3, n3 = mean_over(by_player, acc, brk + 1, 1 << 40, K, e["leagueid"])
            if f1 is None or n1 < args.min_games:
                continue
            if f2 is not None and n2 >= 2:
                pre_mid.append((f1 - base, f2 - base))
            if f3 is not None and n3 >= 2:
                pre_post.append((f1 - base, f3 - base))

    def corr(pairs, name):
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        r = float(np.corrcoef(a, b)[0, 1])
        se = math.sqrt(max(1 - r * r, 1e-9) / max(len(a) - 2, 1))
        print(f"  {name:34s} n={len(a):5d}  r={r:+.3f}  (SE {se:.3f}, t {r / se:+.2f})")
        return r

    print("问题 1/2：选手 form 的持续性（form = 本段均值 − 赛前 365 天基线）")
    r_mid = corr(pre_mid, "小组赛前70% → 小组赛后30%（无断点）")
    r_post = corr(pre_post, "小组赛前70% → 淘汰赛（跨断点）")
    print(f"  跨断点的衰减：{r_post - r_mid:+.3f}\n")

    # ---------- 问题 3：增量预测力 ----------
    X, y, grp = [], [], []
    for e in events:
        rs = e["rows"]
        ev_start = rs[0]["start_time"]
        brk = e["break_at"]
        pre = [r for r in rs if r["start_time"] <= brk]
        post = [r for r in rs if r["start_time"] > brk]
        ser = BE.series_of(post)
        if len(ser) < 5:
            continue
        train = [r for r in rich if r["start_time"] <= brk]
        r_team, idx = BE.fit_rating(train, brk, 45.0, BE.PRIOR_L2, 20)
        r_frm, i_frm = BE.fit_rating(pre, brk, 1e6, 1.0, 2)
        if r_team is None or r_frm is None:
            continue

        # 每队的 form = 本赛事出场最多的五人的 form 均值
        appear = collections.defaultdict(collections.Counter)
        for r in recs:
            if r["leagueid"] != e["leagueid"] or r["start_time"] > brk:
                continue
            tid = r["radiant_team_id"] if r["side"] == 0 else r["dire_team_id"]
            appear[tid][r["account_id"]] += 1
        lo = ev_start - args.baseline_days * 86400
        tform = {}
        for tid, cnt in appear.items():
            vals = []
            for acc, _ in cnt.most_common(5):
                base, nb = mean_over(by_player, acc, lo, ev_start, K)
                f1, n1 = mean_over(by_player, acc, ev_start, brk + 1, K, e["leagueid"])
                if base is not None and nb >= 10 and f1 is not None and n1 >= args.min_games:
                    vals.append(f1 - base)
            if len(vals) >= 4:
                tform[tid] = float(np.mean(vals))

        for g, a, a_won in ser:
            b = (g[0]["dire_team_id"] if g[0]["radiant_team_id"] == a
                 else g[0]["radiant_team_id"])
            if a not in idx or b not in idx or a not in tform or b not in tform:
                continue
            if a not in i_frm or b not in i_frm:
                continue
            X.append([r_team[idx[a]] - r_team[idx[b]],
                      r_frm[i_frm[a]] - r_frm[i_frm[b]],
                      tform[a] - tform[b]])
            y.append(float(a_won))
            grp.append(e["leagueid"])
    X = np.array(X)
    y = np.array(y)
    grp = np.array(grp)
    sd_raw = float(np.std(X[:, 2])) or 1.0
    X[:, 2] /= sd_raw
    print(f"问题 3：增量预测力（{len(y)} 个断点后系列赛，{len(set(grp))} 个赛事）")
    print(f"  x_form_player 标准化前的 SD = {sd_raw:.4f}"
          f"（历史赛事上两队 form 差的离散度）")

    def report(name, cols, labels):
        b = BE.logistic(X[:, cols], y)
        se = BE.cluster_se(X[:, cols], y, b, grp)
        p = 1 / (1 + np.exp(-(X[:, cols] @ b[:-1] + b[-1])))
        ll = -np.mean(y * np.log(np.clip(p, 1e-9, 1)) +
                      (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        print(f"  --- {name}  (样内 logloss {ll:.4f}) ---")
        for k, lab in enumerate(labels):
            print(f"    {lab:24s} {b[k]:>+7.3f}  (SE {se[k]:.3f}, t {b[k] / se[k]:>+5.2f})")
        return ll

    l0 = report("评级差", [0], ["x_rating"])
    l1 = report("+ 队伍级本赛事状态", [0, 1], ["x_rating", "x_form_team"])
    l2 = report("+ 选手级 form", [0, 1, 2],
                ["x_rating", "x_form_team", "x_form_player"])
    print(f"\n  logloss 改善：加队伍级状态 {l0 - l1:+.4f}   再加选手级 form {l1 - l2:+.4f}")

    # 把标准化系数换算成「强度 = 基础强度 + k × 队均 form」里的 k
    b = BE.logistic(X[:, [0, 1, 2]], y)
    se = BE.cluster_se(X[:, [0, 1, 2]], y, b, grp)
    k_hat, k_se = b[2] / sd_raw, se[2] / sd_raw
    print(f"\n  换算成强度调整系数 k（strength += k × 队均 form 偏离）:")
    print(f"    k = {k_hat:+.2f}  (SE {k_se:.2f}, t {b[2] / se[2]:+.2f})")
    print(f"    95% 区间 [{k_hat - 1.96 * k_se:+.2f}, {k_hat + 1.96 * k_se:+.2f}]")
    from math import erf, sqrt
    for thr, what in ((1.0, "改 1 票 (LB2-1, 值 6 分)"), (2.0, "改 3 票")):
        pr = 0.5 * (1 - erf((thr - k_hat) / (k_se * sqrt(2))))
        print(f"    P(k >= {thr:g}) = {pr:.1%}   -> {what}")


if __name__ == "__main__":
    main()
