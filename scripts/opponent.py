"""对手强度：把「打谁打出来的分」校正到 TI 的对手水平。

一支队在小组赛前刷了一堆二线队，幻想分会虚高 —— 但**方向不是想当然的**。
逐局回归（槽位固定效应，只用同一个槽位内部的变化识别斜率）给出的是：

    核心   对手强度 +1 单位 → 单局 −7.8%   （t = −7.5）
    中单   对手强度 +1 单位 → 单局 +2.5%   （t = +2.3）
    辅助   对手强度 +1 单位 → 单局 +5.9%   （t = +5.5）

核心确实打不动强队（少人头、少推塔、GPM 低）；中单和辅助反过来 —— 强队之间的局
更长、团战更多、烟雾更多，而中单/辅助的战旗吃的正是正反补、团战、诡计之雾。
所以「TI 更难打 → 分更低」只对核心成立。

评分用的是本项目自己的 margin 评分（`scripts/margin.py`，README 里那句
"opponent rating is the measure that actually works"），对全部 458 支有 ≥20 场
的队伍拟合，不只是 TI 这 16 支。结果缓存在 `data/team_ratings_all.json`。

    python scripts/opponent.py          # 拟合 + 逐槽位的对手强度与校正量
    python scripts/opponent.py --refit  # 强制重新拟合
"""

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
import margin  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "team_ratings_all.json")
ROLES = ("core", "mid", "support")
HALF_LIFE, L2, MIN_GAMES = 150.0, 0.3, 20
CLIP = 0.25          # 单局校正幅度的上限，防止个别极端对手把一局放飞


def team_ratings(refit=False):
    """team_id -> margin 评分。与 fit_ratings_v3 同一套超参，但不丢掉低级别联赛
    （README：过滤掉反而更差，t = −3.1，它们把评分图连起来了）。"""
    if os.path.exists(CACHE) and not refit:
        return {int(k): v for k, v in json.load(open(CACHE)).items()}
    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = margin.load_rich(aliases)
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    i, j, w, idx, m, _win, _rows = margin.build_design(rows, now, HALF_LIFE,
                                                       MIN_GAMES)
    r, _h = margin.fit_margin(i, j, m["gdpm"] / 400.0, w, len(idx), l2=L2)
    out = {int(t): float(r[k]) for t, k in idx.items()}
    json.dump(out, open(CACHE, "w"), indent=0)
    return out


def field_mean(teams, R):
    """TI 现场的平均对手强度 = 16 支队评分的均值（自己也在里面，差别 <0.05）。"""
    vals = [R[teams[t]["team_id"]] for t in teams if teams[t]["team_id"] in R]
    return float(np.mean(vals))


def slopes(slots, R, control_duration=False):
    """每个定位一条斜率：单局得分 ~ 对手强度，槽位固定效应。

    `control_duration=True` 时把时长一起放进回归 —— 这是**另一个问题的答案**：
    对手强一单位，比赛平均长 3 分钟，而中单/辅助的战旗吃时长。总效应（默认）是
    「到了 TI 会打出多少分」，控制时长后的偏效应是「同样长的一局里打得好不好」。
    预测该用总效应：TI 的局确实会更长。两者差多少见 `main()` 的对照。

    -> {role: (相对斜率 每单位的百分比, t 值, 样本量, 平均单局分)}
    """
    out = {}
    for role in ROLES:
        cols, Y, means = [], [], []
        for (_t, r), rows in slots.items():
            if r != role:
                continue
            trip = [(R[row["opp"]], row["scores"].mean(), row["duration"] / 60)
                    for row in rows if row["opp"] in R]
            if len(trip) < 20:
                continue
            a = np.array(trip, dtype=float)
            means.append(a[:, 1].mean())
            a -= a.mean(axis=0)          # 槽位固定效应 = 组内去均值
            cols.append(a)
        A = np.concatenate(cols)
        Y = A[:, 1]
        X = A[:, [0, 2]] if control_duration else A[:, [0]]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        xtx_inv = np.linalg.inv(X.T @ X)
        s2 = (resid @ resid) / (len(Y) - X.shape[1])
        se = float(np.sqrt(s2 * xtx_inv[0, 0]))
        b = float(beta[0])
        base = float(np.mean(means))
        out[role] = (b / base, b / se, len(Y), base)
    return out


def factors(rows, role, R, mu, sl):
    """逐局的校正系数：把这一局的对手换成 TI 平均水平会多/少打出多少。"""
    rel = sl[role][0]
    f = np.ones(len(rows))
    for k, row in enumerate(rows):
        o = R.get(row["opp"])
        if o is not None:
            f[k] = 1.0 + np.clip(rel * (mu - o), -CLIP, CLIP)
    return f


def adjust(rows, role, R, mu, sl):
    """返回校正过分数的新 rows（浅拷贝，只换 scores）。"""
    f = factors(rows, role, R, mu, sl)
    return [dict(row, scores=row["scores"] * f[k]) for k, row in enumerate(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()

    import banner_craft as BC
    import banner_decide as BD
    import title_value as TV

    R = team_ratings(args.refit)
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    _st, banners, _held = BD.current(BD.load_log())
    W = {r: {s: w for (s, _, _), w in zip(banners[r], BC.multipliers(banners[r]))}
         for r in ROLES}
    _, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)
    slots = TV.slot_games(by_player, roles, lambda t, r: W[r])

    mu = field_mean(teams, R)
    sl = slopes(slots, R)
    sl_d = slopes(slots, R, control_duration=True)
    print(f"全池 {len(R)} 支队有评分；TI 这 16 支的平均 {mu:+.3f}，"
          f"全池中位数 {np.median(list(R.values())):+.3f}")
    print(f"\n{'定位':<8}{'总效应/单位':>13}{'t':>7}{'控时长后':>11}{'t':>7}"
          f"{'逐局样本':>10}{'平均单局':>10}")
    for role in ROLES:
        rel, t, n, base = sl[role]
        rd, td, _, _ = sl_d[role]
        print(f"{BD.ROLE_CN[role]:<8}{100 * rel:>+12.1f}%{t:>7.1f}"
              f"{100 * rd:>+10.1f}%{td:>7.1f}{n:>10,}{base:>10,.0f}")
    print("  总效应含「强队之间的局更长」这条通道（对手 +1 单位 → 长约 3 分钟），"
          "预测该用它；控住时长是另一个问题。")

    print(f"\n{'队伍':<18}{'对手均值':>10}   " + "".join(
        f"{BD.ROLE_CN[r] + ' 校正':>12}" for r in ROLES))
    rowsout = []
    for team in sorted(teams):
        f, opp = {}, []
        for role in ROLES:
            rows = slots.get((team, role))
            if not rows:
                continue
            o = [R[row["opp"]] for row in rows if row["opp"] in R]
            opp += o
            f[role] = float(np.mean(factors(rows, role, R, mu, sl)))
        if f:
            rowsout.append((float(np.mean(opp)), team, f))
    for o, team, f in sorted(rowsout):
        cells = "".join(f"{100 * (f.get(r, 1) - 1):>+11.1f}%" for r in ROLES)
        print(f"{team:<18}{o:>+10.3f}   {cells}")
    print(f"\n-> {os.path.relpath(CACHE, ROOT)}")


if __name__ == "__main__":
    main()
