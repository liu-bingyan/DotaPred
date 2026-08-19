"""时间权重口径的样本外回测：预测 TI2026 小组赛，以及六个近期 LAN 的后半程。

要回答的问题是「预测 TI 淘汰赛该用哪些比赛、怎么加权」。三个候选口径其实是同一族
里的三个点：

    w(Δt) = exp(-ln2 · Δt / HL) × tier_weight

  只用本赛事      -> event_only，或 HL 极小
  最近一段        -> HL ≈ 20–45 天，或硬窗口
  全历史时间衰减   -> HL = 150 天（现生产模型）… ∞（等权）

所以扫 HL 就同时评了三个版本，并且能把 HL 当参数拟合。另外单列了两种不属于这一族
的口径：硬窗口（窗口内等权、窗口外全丢），以及长短两个时间尺度同时进 logistic
堆叠（让模型自己决定近期该占多少）。

两个测试集，训练集一律严格早于测试局：

  A  TI2026 小组赛 —— 109 场 / 44 个系列赛。最贴近的目标人群，但赛前没有任何本赛事
     数据，评不了「只用本赛事」这一版。
  B  六个近期赛事的后半程 —— 以每个赛事自己的时间中位数切开，训练集包含该赛事前半
     程。这才是淘汰赛预测的真实处境（手上已有本赛事数据），也是唯一能回答「本赛事
     数据值多少」的实验。

样本量小（A 只有 44 个系列赛），所以所有对比都给按赛事聚类的配对检验，别只看点估计。

用法：
    python3 scripts/weighting_backtest.py
    python3 scripts/weighting_backtest.py --json out.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin  # noqa: E402
import models  # noqa: E402
from experiment import player_feature, player_fit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = {"premium", "professional"}
TI = 19719

EVENTS = [
    (19543, "PGL Wallachia S8"),
    (19696, "DreamLeague S29"),
    (19101, "BLAST SLAM VII"),
    (19785, "石油杯 EWC 2026"),
    (20009, "1win Essence II"),
    (TI, "TI2026 小组赛后半"),
]


def bo_prob(p, need):
    return sum(math.comb(need - 1 + k, k) * p**need * (1 - p) ** k for k in range(need))


def load():
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    return [r for r in margin.load_rich(aliases) if r.get("tier") in TOP], models.load_lineups()


class Spec:
    """一个口径 = 若干个队伍评级分量（每个有自己的 HL/窗口/赛事限制）+ 可选选手项。"""

    def __init__(self, label, components, player_hl=150.0, l2t=0.3, l2p=30.0,
                 min_games=20):
        self.label = label
        self.components = components   # [(hl, window_days|None, event_only)]
        self.player_hl = player_hl     # None -> 不用选手项
        self.l2t, self.l2p = l2t, l2p
        self.min_games = min_games


def _design(rows, cutoff, hl, window, event_only, leagueid, min_games):
    lo = -1 if window is None else cutoff - window * 86400
    train = [r for r in rows
             if lo <= r["start_time"] < cutoff
             and (not event_only or r["leagueid"] == leagueid)]
    return margin.build_design(train, cutoff, hl, min_games)


def fit_at(rows, lineups, cutoff, spec, leagueid):
    """在 cutoff 拟合，返回 predict(test_rows) -> (p_radiant, 可用掩码)。"""
    parts = []          # (idx, r_team)
    base_i = base_j = base_w = base_tr = None
    for hl, window, ev_only in spec.components:
        mg = 4 if ev_only else spec.min_games
        i, j, w, idx, m, _, tr = _design(rows, cutoff, hl, window, ev_only,
                                         leagueid, mg)
        r_team, _ = margin.fit_margin(i, j, m["gdpm"] / 400.0, w, len(idx), l2=spec.l2t)
        parts.append((idx, r_team))
        if base_tr is None or len(tr) > len(base_tr):
            base_i, base_j, base_w, base_tr = i, j, w, tr

    # logistic 的训练行用样本最多的那个分量，其余分量在这些行上取特征
    win = np.array([1.0 if r["radiant_win"] else 0.0 for r in base_tr])
    gd = np.array([(r.get("final_gold_adv") or 0) / (max(r["duration"], 1) / 60.0)
                   for r in base_tr]) / 400.0

    def team_cols(rws):
        cols, ok = [], np.ones(len(rws), dtype=bool)
        for idx, r_team in parts:
            c = np.full(len(rws), np.nan)
            for k, r in enumerate(rws):
                a, b = idx.get(r["radiant_team_id"]), idx.get(r["dire_team_id"])
                if a is not None and b is not None:
                    c[k] = r_team[a] - r_team[b]
            cols.append(c)
        # 第一个分量（主分量）必须有评级才算这场
        ok &= ~np.isnan(cols[0])
        return cols, ok

    cols_tr, _ = team_cols(base_tr)
    r_pl = pidx = None
    if spec.player_hl is not None:
        _, _, wp, _, mp, _, trp = _design(rows, cutoff, spec.player_hl, None, False,
                                          leagueid, spec.min_games)
        r_pl, pidx = player_fit(trp, wp, lineups, mp["gdpm"] / 400.0, l2=spec.l2p)
        if r_pl is not None:
            cols_tr.append(player_feature(base_tr, r_pl, pidx, lineups))

    X = np.column_stack(cols_tr)
    mu = np.nanmean(X, axis=0)
    beta = margin.fit_logistic(np.where(np.isnan(X), mu, X), win, base_w)
    del gd

    def predict(test_rows):
        cols, ok = team_cols(test_rows)
        if r_pl is not None:
            cols.append(player_feature(test_rows, r_pl, pidx, lineups))
        Xt = np.column_stack(cols)
        return margin.predict_logistic(beta, np.where(np.isnan(Xt), mu, Xt)), ok

    return predict


def series_of(test_rows):
    by = {}
    for r in test_rows:
        by.setdefault(r["series_id"] or -r["match_id"], []).append(r)
    out = []
    for sid, rs in by.items():
        a = rs[0]["radiant_team_id"]
        wins = sum((r["radiant_win"] if r["radiant_team_id"] == a else not r["radiant_win"])
                   for r in rs)
        if wins * 2 == len(rs):
            continue
        out.append((sid, rs, a, wins * 2 > len(rs)))
    return out


def score(predict, test_rows):
    """返回逐系列赛的 logloss 向量（用于配对检验）和汇总指标。"""
    p, ok = predict(test_rows)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = np.array([1.0 if r["radiant_win"] else 0.0 for r in test_rows])
    g_ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    pos = {r["match_id"]: k for k, r in enumerate(test_rows)}
    s_ll, s_acc, s_key = [], [], []
    for sid, rs, a, a_won in series_of(test_rows):
        ks = [pos[r["match_id"]] for r in rs]
        if not all(ok[k] for k in ks):
            continue
        pa = np.mean([p[k] if rs[t]["radiant_team_id"] == a else 1 - p[k]
                      for t, k in enumerate(ks)])
        need = 3 if (rs[0].get("series_type") == 2 or len(rs) > 3) else 2
        ps = float(np.clip(bo_prob(pa, need), 1e-6, 1 - 1e-6))
        s_ll.append(-(math.log(ps) if a_won else math.log(1 - ps)))
        s_acc.append(float((ps > 0.5) == a_won))
        s_key.append(sid)
    return dict(game_ll=g_ll[ok], game_acc=(p[ok] > 0.5) == (y[ok] > 0.5),
                series_ll=np.array(s_ll), series_acc=np.array(s_acc),
                series_key=s_key)


def run(rows, lineups, spec, targets):
    per = []
    for cutoff, test_rows, _lab, lid in targets:
        per.append(score(fit_at(rows, lineups, cutoff, spec, lid), test_rows))
    cat = {k: (np.concatenate([p[k] for p in per]) if k != "series_key"
               else [x for p in per for x in p[k]]) for k in per[0]}
    return cat, per


def paired(a, b, per_a, per_b, key="series_ll"):
    """a 相对 b 的改善，只在两者都评得了的系列赛上配对，SE 按赛事聚类。

    正 t 值 = a 更好（logloss 更低）。两个口径可用的系列赛集合不同（评级最少场次
    的门槛不一样），不对齐就会拿不同的题目比分数。"""
    ia = {k: n for n, k in enumerate(a["series_key"])}
    ib = {k: n for n, k in enumerate(b["series_key"])}
    common = [k for k in a["series_key"] if k in ib]
    if not common:
        return 0.0, 0.0
    d = np.array([b[key][ib[k]] - a[key][ia[k]] for k in common])
    ev = {}
    off = 0
    for pa in per_a:
        for k in pa["series_key"]:
            ev[k] = off
        off += 1
    g = {}
    for k, val in zip(common, d):
        g.setdefault(ev.get(k, -1), []).append(val)
    sums = np.array([sum(v) for v in g.values()])
    n = len(d)
    if len(sums) < 2:
        se = d.std(ddof=1) / math.sqrt(n)
    else:
        se = math.sqrt(((sums - sums.mean()) ** 2).sum() * len(sums)
                       / (len(sums) - 1)) / n
    return d.mean(), (d.mean() / se if se > 0 else 0.0)


def sweep(rows, lineups, target_a, target_b):
    """按测试集 B 选参（那才是「手上已有本赛事数据」的处境），A 只作旁证。"""
    base = Spec("ref", [(150.0, None, False)])
    ref_b = run(rows, lineups, base, target_b)
    ref_a = run(rows, lineups, base, target_a)
    grid = []
    for hl in (21, 30, 40, 50, 60, 80):
        for l2t in (0.1, 0.3, 1.0):
            for php in (30.0, 60.0, 150.0, None):
                grid.append(Spec(f"HL={hl:g} l2t={l2t} 选手HL={php or '-'}",
                                 [(float(hl), None, False)], player_hl=php, l2t=l2t))
    out = []
    for spec in grid:
        cb, pb = run(rows, lineups, spec, target_b)
        ca, pa = run(rows, lineups, spec, target_a)
        db, tb = paired(cb, ref_b[0], pb, ref_b[1])
        da, ta = paired(ca, ref_a[0], pa, ref_a[1])
        out.append((cb["series_ll"].mean(), spec.label, cb, ca, db, tb, da, ta))
        print(".", end="", flush=True)
    print("\n按测试集 B 的系列赛 logloss 排序（基线 HL=150 = "
          f"{ref_b[0]['series_ll'].mean():.4f} / A {ref_a[0]['series_ll'].mean():.4f}）")
    print(f"{'口径':34s} {'B系列LL':>8} {'B系列准':>7} {'Δ':>8} {'t':>6} | "
          f"{'A系列LL':>8} {'A系列准':>7} {'Δ':>8} {'t':>6}")
    for _, lab, cb, ca, db, tb, da, ta in sorted(out)[:18]:
        print(f"{lab:34s} {cb['series_ll'].mean():>8.4f} {cb['series_acc'].mean():>6.1%} "
              f"{db:>+8.4f} {tb:>+6.2f} | {ca['series_ll'].mean():>8.4f} "
              f"{ca['series_acc'].mean():>6.1%} {da:>+8.4f} {ta:>+6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="半衰期 x 正则化 x 选手项半衰期 的联合细网格")
    args = ap.parse_args()

    rows, lineups = load()
    ti_rows = [r for r in rows if r["leagueid"] == TI]
    ti_start = min(r["start_time"] for r in ti_rows)
    target_a = [(ti_start, ti_rows, "TI2026 小组赛", TI)]

    target_b = []
    for lid, name in EVENTS:
        ev = sorted([r for r in rows if r["leagueid"] == lid], key=lambda r: r["start_time"])
        cut = ev[len(ev) // 2]["start_time"]
        sids = {r["series_id"] for r in ev if r["start_time"] < cut}
        test = [r for r in ev if r["start_time"] >= cut and r["series_id"] not in sids]
        target_b.append((cut, test, name, lid))

    if args.sweep:
        sweep(rows, lineups, target_a, target_b)
        return

    HLS = [7, 14, 21, 30, 45, 60, 100, 150, 250, 400, 1e6]
    specs = [Spec(f"指数衰减 HL={hl:g}天" if hl < 1e5 else "全历史等权",
                  [(hl, None, False)]) for hl in HLS]
    specs += [
        Spec("硬窗口 最近45天等权", [(1e6, 45, False)], min_games=6),
        Spec("硬窗口 最近90天等权", [(1e6, 90, False)], min_games=6),
        Spec("只用本赛事（V1 字面版）", [(1e6, None, True)]),
        Spec("堆叠 HL=150 + 本赛事", [(150.0, None, False), (1e6, None, True)]),
        Spec("堆叠 HL=150 + HL=30", [(150.0, None, False), (30.0, None, False)]),
        Spec("堆叠 HL=150 + HL=30 + 本赛事",
             [(150.0, None, False), (30.0, None, False), (1e6, None, True)]),
        Spec("HL=30 无选手项", [(30.0, None, False)], player_hl=None),
        Spec("HL=150 无选手项（旧基线）", [(150.0, None, False)], player_hl=None),
    ]

    out = {}
    for tag, targets, ref_label in (
            ("A  TI2026 小组赛（赛前无本赛事数据）", target_a, "指数衰减 HL=150天"),
            ("B  六赛事后半程（含本赛事前半）", target_b, "指数衰减 HL=150天")):
        for c, t, n, _l in targets:
            pass
        print(f"=== 测试集 {tag} ===")
        n_s = sum(len(series_of(t[1])) for t in targets)
        n_g = sum(len(t[1]) for t in targets)
        print(f"    {n_g} 场 / {n_s} 系列赛"
              + ("" if len(targets) == 1 else
                 "  [" + " ".join(f"{t[2]}:{len(series_of(t[1]))}" for t in targets) + "]"))
        res = {}
        for spec in specs:
            if any(c[2] for c in spec.components) and targets is target_a:
                continue          # 赛前没有本赛事数据，含「本赛事」分量的口径评不了
            res[spec.label] = run(rows, lineups, spec, targets)
        ref = res.get(ref_label)
        print(f"{'口径':30s} {'局LL':>8} {'局准':>7} {'系列LL':>8} {'系列准':>7} {'ΔLL vs 150天':>13} {'t':>6}")
        for lab, (cat, per) in res.items():
            if ref is not None and lab != ref_label:
                d, t = paired(cat, ref[0], per, ref[1], "series_ll")
            else:
                d, t = 0.0, 0.0
            print(f"{lab:30s} {cat['game_ll'].mean():>8.4f} {cat['game_acc'].mean():>6.1%} "
                  f"{cat['series_ll'].mean():>8.4f} {cat['series_acc'].mean():>6.1%} "
                  f"{d:>+13.4f} {t:>+6.2f}")
            out.setdefault(tag, {})[lab] = dict(
                game_ll=float(cat["game_ll"].mean()), game_acc=float(cat["game_acc"].mean()),
                series_ll=float(cat["series_ll"].mean()),
                series_acc=float(cat["series_acc"].mean()),
                d_vs_ref=float(d), t_vs_ref=float(t), n_series=int(len(cat["series_ll"])))
        print()

    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
