"""赛程断点效应：小组赛打出来的状态，隔几天之后还算不算数？

坊间说法是「小组赛到淘汰赛中间那几天会有重大变动」。这个脚本用历史赛事检验它。

难点在于不能直接比「小组赛内部的预测准确率」和「淘汰赛的预测准确率」——淘汰赛只
剩强队、对阵更胶着，准确率天然更低，跟状态衰不衰减没关系。所以设计成同一个模型、
两个测试集：

    在每个赛事的小组赛前 70% 上拟合一个「本赛事状态」评级 F
      测试集 A = 该赛事小组赛的后 30%      （距离拟合截止 0–2 天，中间没有断点）
      测试集 B = 该赛事断点之后的淘汰赛     （距离拟合截止 = 断点长度 + 若干天）

F 在两个测试集上是同一套参数、同样的噪声水平，唯一的差别是隔了多久、以及中间有没
有那个断点。再把「赛前长期评级」x_prior 作为协变量放进同一个 logistic，控制住对阵
胶着程度，看的是 **x_form 的增量系数**：

    logit p = a0 + a1·x_prior + a2·x_form + a3·(x_form × 断点后)

a3 < 0 就说明状态跨过断点确实衰减了。再换成 a3·(x_form × 断点天数) 看衰减是否随
间隔变长而加剧。

两个特征都在赛事内部标准化，否则不同赛事的样本量差异会让系数不可比。

    python3 scripts/break_effect.py
    python3 scripts/break_effect.py --min-gap 2      # 只看断点 >=2 天的赛事
"""

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = {"premium", "professional"}
PRIOR_HL, PRIOR_L2 = 150.0, 0.3
FORM_L2 = 1.0


def find_events(rows, min_games, max_span_days, min_gap_days, min_pre, min_post):
    """挑出「短周期锦标赛 + 一个明显断点」的赛事。长期联赛会被 span 条件筛掉。"""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["leagueid"]].append(r)
    out = []
    for lid, rs in by.items():
        if len(rs) < min_games:
            continue
        rs.sort(key=lambda r: r["start_time"])
        ts = np.array([r["start_time"] for r in rs], dtype=float)
        span = (ts[-1] - ts[0]) / 86400
        if span > max_span_days:
            continue
        gaps = np.diff(ts)
        k = int(np.argmax(gaps))
        gap = gaps[k] / 86400
        if gap < min_gap_days or k + 1 < min_pre or len(rs) - k - 1 < min_post:
            continue
        out.append({
            "leagueid": lid, "name": rs[0].get("league_name"), "rows": rs,
            "gap_days": gap, "break_at": float(ts[k]),
            "n_pre": k + 1, "n_post": len(rs) - k - 1, "span": span,
        })
    return sorted(out, key=lambda e: e["gap_days"])


def fit_rating(rows, now, hl, l2, min_games):
    if len(rows) < 10:
        return None, None
    i, j, w, idx, m, _, tr = margin.build_design(rows, now, hl, min_games)
    if len(i) < 10 or len(idx) < 4:
        return None, None
    r, _ = margin.fit_margin(i, j, m["gdpm"] / 400.0, w, len(idx), l2=l2)
    return r, idx


def diff_feature(test_rows, r, idx):
    out = np.full(len(test_rows), np.nan)
    if r is None:
        return out
    for k, m in enumerate(test_rows):
        a, b = idx.get(m["radiant_team_id"]), idx.get(m["dire_team_id"])
        if a is not None and b is not None:
            out[k] = r[a] - r[b]
    return out


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


def cluster_se(X, y, beta, groups):
    """按赛事聚类的稳健标准误（sandwich）。"""
    Xa = np.column_stack([X, np.ones(len(y))])
    p = 1 / (1 + np.exp(-Xa @ beta))
    bread = np.linalg.pinv((Xa * (p * (1 - p))[:, None]).T @ Xa)
    meat = np.zeros_like(bread)
    for g in set(groups):
        s = groups == g
        u = Xa[s].T @ (y[s] - p[s])
        meat += np.outer(u, u)
    cov = bread @ meat @ bread
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def build(events, rows_all, form_frac):
    """每个赛事产出两个测试集的行 + 特征。"""
    recs = []
    for e in events:
        rs = e["rows"]
        ev_start = rs[0]["start_time"]
        pre = [r for r in rs if r["start_time"] <= e["break_at"]]
        post = [r for r in rs if r["start_time"] > e["break_at"]]

        cut1 = pre[int(len(pre) * form_frac)]["start_time"]
        form_train = [r for r in pre if r["start_time"] < cut1]
        test_a = [r for r in pre if r["start_time"] >= cut1]
        if len(form_train) < 20 or len(test_a) < 8:
            continue

        prior_rows = [r for r in rows_all if r["start_time"] < ev_start]
        r_pri, i_pri = fit_rating(prior_rows, ev_start, PRIOR_HL, PRIOR_L2, 20)
        r_frm, i_frm = fit_rating(form_train, cut1, 1e6, FORM_L2, 2)
        if r_frm is None:
            continue

        for tag, test in (("A 断点前", test_a), ("B 断点后", post)):
            xp = diff_feature(test, r_pri, i_pri)
            xf = diff_feature(test, r_frm, i_frm)
            y = np.array([1.0 if r["radiant_win"] else 0.0 for r in test])
            elapsed = np.array([(r["start_time"] - cut1) / 86400 for r in test])
            ok = ~np.isnan(xf)
            xp = np.where(np.isnan(xp), 0.0, xp)
            if ok.sum() < 5:
                continue
            # 赛事内标准化，让不同赛事的系数可比
            sp = np.std(xp[ok]) or 1.0
            sf = np.std(xf[ok]) or 1.0
            recs.append(dict(event=e, tag=tag, post=(tag[0] == "B"),
                             x_prior=xp[ok] / sp, x_form=xf[ok] / sf,
                             y=y[ok], elapsed=elapsed[ok], n=int(ok.sum())))
    return recs


def series_of(rs):
    by = {}
    for r in rs:
        by.setdefault(r["series_id"] or -r["match_id"], []).append(r)
    out = []
    for g in by.values():
        a = g[0]["radiant_team_id"]
        w = sum((r["radiant_win"] if r["radiant_team_id"] == a else not r["radiant_win"])
                for r in g)
        if w * 2 != len(g):
            out.append((g, a, w * 2 > len(g)))
    return out


def hl_sweep(events, rows_all, hls):
    """对每个赛事：用断点之前的全部数据（含本赛事小组赛）拟合，预测断点之后的系列赛。

    这正是 TI 淘汰赛的处境。只用队伍级评级 —— 加上选手项后每个口径要多跑十几秒，
    而半衰期这一维的结论在带不带选手项时是一样的（见 09-playoff.md §3）。
    """
    tasks = []
    for e in events:
        pre = [r for r in e["rows"] if r["start_time"] <= e["break_at"]]
        post = [r for r in e["rows"] if r["start_time"] > e["break_at"]]
        ser = series_of(post)
        if len(pre) < 30 or len(ser) < 5:
            continue
        train = [r for r in rows_all if r["start_time"] <= e["break_at"]]
        tasks.append((e, train, ser))
    print(f"HL 扫描：{len(tasks)} 个赛事，"
          f"{sum(len(t[2]) for t in tasks)} 个断点后系列赛\n")

    print(f"{'半衰期':14s} {'系列LL':>9} {'系列准':>8} {'Δ vs 150天':>12} {'t':>7}")
    base = None
    for hl in hls:
        per_ev, lls, accs = [], [], []
        for e, train, ser in tasks:
            r, idx = fit_rating(train, e["break_at"], hl, PRIOR_L2, 20)
            if r is None:
                continue
            ev_ll = []
            for g, a, a_won in ser:
                b = g[0]["dire_team_id"] if g[0]["radiant_team_id"] == a \
                    else g[0]["radiant_team_id"]
                if a not in idx or b not in idx:
                    continue
                # 评级差 -> 单局胜率，斜率用赛事外的经验值 0.74（与生产模型一致）
                p = 1 / (1 + math.exp(-0.74 * (r[idx[a]] - r[idx[b]])))
                need = 3 if len(g) > 3 else 2
                ps = sum(math.comb(need - 1 + k, k) * p**need * (1 - p) ** k
                         for k in range(need))
                ps = min(max(ps, 1e-6), 1 - 1e-6)
                ev_ll.append(-(math.log(ps) if a_won else math.log(1 - ps)))
                accs.append(float((ps > 0.5) == a_won))
            if ev_ll:
                per_ev.append(sum(ev_ll))
                lls.extend(ev_ll)
        ll = float(np.mean(lls))
        if base is None:
            base_lls, base_per = list(lls), list(per_ev)
        d = t = 0.0
        if base is not None:
            dv = np.array(base_lls) - np.array(lls)
            gsum = np.array(base_per) - np.array(per_ev)
            se = math.sqrt(((gsum - gsum.mean()) ** 2).sum() * len(gsum)
                           / max(len(gsum) - 1, 1)) / len(dv)
            d, t = dv.mean(), (dv.mean() / se if se > 0 else 0.0)
        tag = f"HL={hl:g}天" if hl < 1e5 else "全历史等权"
        print(f"{tag:14s} {ll:>9.4f} {np.mean(accs):>7.1%} {d:>+12.4f} {t:>+7.2f}")
        if base is None:
            base = ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gap", type=float, default=0.8, help="断点最短天数")
    ap.add_argument("--min-games", type=int, default=50)
    ap.add_argument("--max-span", type=float, default=40.0, help="赛事总跨度上限（天）")
    ap.add_argument("--form-frac", type=float, default=0.7,
                    help="小组赛前多少比例用于拟合状态评级")
    ap.add_argument("--premium", action="store_true",
                    help="只用 premium 级赛事（大型 LAN），排除线上联赛和预选赛")
    ap.add_argument("--list", type=int, default=0, help="列出前 N 个赛事")
    ap.add_argument("--hl-sweep", action="store_true",
                    help="换个问法：专门预测断点之后的比赛时，最优半衰期是多少")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    keep = {"premium"} if args.premium else TOP
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in keep]
    events = find_events(rows, args.min_games, args.max_span, args.min_gap, 30, 15)
    print(f"符合条件的赛事 {len(events)} 个（≥{args.min_games} 场，跨度 ≤{args.max_span:g} 天，"
          f"断点 ≥{args.min_gap:g} 天）\n")

    if args.hl_sweep:
        hl_sweep(events, rows, [150, 7, 14, 21, 30, 45, 60, 100, 250, 400, 1e6])
        return

    recs = build(events, rows, args.form_frac)
    used = {r["event"]["leagueid"] for r in recs}
    shown = [e for e in events if e["leagueid"] in used]
    print(f"实际可用 {len(shown)} 个赛事")
    pick = [e for e in shown if "International" in str(e["name"])]
    if args.list:
        pick = shown[:args.list]
    if pick:
        print(f"{'赛事':38s} {'断点':>6} {'A局':>5} {'B局':>5}")
        for e in pick:
            na = sum(r["n"] for r in recs if r["event"] is e and not r["post"])
            nb = sum(r["n"] for r in recs if r["event"] is e and r["post"])
            print(f"{str(e['name'])[:38]:38s} {e['gap_days']:>5.1f}天 {na:>5} {nb:>5}")

    X = np.column_stack([
        np.concatenate([r["x_prior"] for r in recs]),
        np.concatenate([r["x_form"] for r in recs]),
        np.concatenate([r["x_form"] * r["post"] for r in recs]),
    ])
    y = np.concatenate([r["y"] for r in recs])
    grp = np.concatenate([np.full(r["n"], r["event"]["leagueid"]) for r in recs])
    post = np.concatenate([np.full(r["n"], r["post"]) for r in recs])
    gap = np.concatenate([np.full(r["n"], r["event"]["gap_days"]) for r in recs])

    print(f"\n合计 {len(y)} 局（断点前 {int((~post).sum())}，断点后 {int(post.sum())}），"
          f"{len(set(grp))} 个赛事\n")

    def report(name, cols, labels):
        b = logistic(cols, y)
        se = cluster_se(cols, y, b, grp)
        print(f"--- {name} ---")
        for k, lab in enumerate(labels):
            t = b[k] / se[k] if se[k] > 0 else 0
            print(f"  {lab:26s} {b[k]:>+7.3f}  (SE {se[k]:.3f}, t {t:>+5.2f})")
        return b

    report("基准：赛前评级 + 本赛事状态",
           X[:, :2], ["x_prior 赛前长期评级", "x_form 本赛事状态"])
    report("加断点交互项", X,
           ["x_prior 赛前长期评级", "x_form 本赛事状态", "x_form × 断点后"])
    Xg = np.column_stack([X[:, :2], X[:, 1] * post * gap])
    report("交互项换成断点天数", Xg,
           ["x_prior 赛前长期评级", "x_form 本赛事状态", "x_form × 断点天数"])

    # 分组描述：两个测试集各自单独拟合，直接看 x_form 的斜率
    print("\n--- 两个测试集分别拟合（同一个状态评级，只是隔的天数不同）---")
    print(f"{'测试集':12s} {'局数':>5} {'x_prior':>9} {'x_form':>9} "
          f"{'加x_form的logloss增益':>20}")
    for tag, sel in (("A 断点前", ~post), ("B 断点后", post)):
        b2 = logistic(X[sel][:, :2], y[sel])
        b1 = logistic(X[sel][:, :1], y[sel])
        ll2 = -np.mean(np.log(np.clip(np.where(
            y[sel] > 0.5, 1 / (1 + np.exp(-(X[sel][:, :2] @ b2[:2] + b2[2]))),
            1 - 1 / (1 + np.exp(-(X[sel][:, :2] @ b2[:2] + b2[2])))), 1e-9, 1)))
        ll1 = -np.mean(np.log(np.clip(np.where(
            y[sel] > 0.5, 1 / (1 + np.exp(-(X[sel][:, :1] @ b1[:1] + b1[1]))),
            1 - 1 / (1 + np.exp(-(X[sel][:, :1] @ b1[:1] + b1[1])))), 1e-9, 1)))
        print(f"{tag:12s} {int(sel.sum()):>5} {b2[0]:>+9.3f} {b2[1]:>+9.3f} "
              f"{ll1 - ll2:>+20.4f}")

    # 按断点长度分桶，看 x_form 的斜率是否随间隔变长而下降
    print("\n--- 断点后的比赛，按断点长度分桶 ---")
    print(f"{'断点':12s} {'赛事':>5} {'局数':>6} {'x_prior':>9} {'x_form':>9}")
    buckets = [(0, 1.5), (1.5, 3), (3, 5), (5, 100)]
    for lo, hi in buckets:
        sel = post & (gap >= lo) & (gap < hi)
        if sel.sum() < 60:
            continue
        b = logistic(X[sel][:, :2], y[sel])
        print(f"{f'{lo:g}-{hi:g}天':12s} {len(set(grp[sel])):>5} {int(sel.sum()):>6} "
              f"{b[0]:>+9.3f} {b[1]:>+9.3f}")
    sel = ~post
    b = logistic(X[sel][:, :2], y[sel])
    print(f"{'(断点前对照)':12s} {len(set(grp[sel])):>5} {int(sel.sum()):>6} "
          f"{b[0]:>+9.3f} {b[1]:>+9.3f}")

    if args.json:
        json.dump({"n_games": int(len(y)), "n_events": len(set(grp))},
                  open(args.json, "w"))


if __name__ == "__main__":
    main()
