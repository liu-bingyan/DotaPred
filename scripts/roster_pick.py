"""战旗定稿之后：这三面旗该挂哪三支队、配哪个称号。

代币花完了，统计项和倍率再也动不了，但**换队伍和换称号都是免费的**
（`Tutorial2026_ViewTablet_Body`: "You may freely change the chosen team"）。
所以现在的问题反过来了 —— 不是「给这支队选什么词条」，而是
**「给这套已经固定的权重，挑产出最高的队 + 称号」**。

队伍和称号必须一起挑：前缀是**选手级**条件（「使用蓝色英雄时 +11%」），
命中率取决于这支队这个位置真正在打的英雄。Liquid 中单的蓝色命中率是 Yandex 中单
的两倍多，蔚蓝的能把两者的差距再拉开一截 —— 裸分排序不等于含称号排序。
所以这里把 8 前缀 × 9 后缀 × 16 队 全部展开，取联合最优。

一期总分 = 三个定位各自「最好的那个系列赛」之和，每个定位独立取最大
（docs/06-banner.md §1.3）。称号是**全局**的（三个定位共用一个前缀、一个后缀），
所以外层枚举 72 个称号组合，内层三个定位各挑各的队，不用跑 16³。

打分口径与 title_value.py 一致：max2(sum) 蒙特卡洛、整局重抽保留联合分布、
按胜负分池、名次分布来自 data/sim_buckets_boot.npy，并且逐局逐人把称号乘子乘进去
（前缀只作用于打了该类英雄的那名选手，权重是他在这一局的得分占比）。
所有队、所有称号共用同一批预抽随机数，是配对比较。

战旗状态读 data/roll_options_log.json 的 banner_states 末项；手上的称号读同一条的
`title` 字段（没有就用 --have 指定，默认 docs/07-titles.md 记的 猩红的/残酷之人）。

    python scripts/roster_pick.py                  # 排名 + 换队/换称号建议
    python scripts/roster_pick.py --sims 20000
    python scripts/roster_pick.py --have blue,cruel
    python scripts/roster_pick.py --no-title       # 旧口径：不含称号的裸分
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
import banner_craft as BC  # noqa: E402
import banner_decide as BD  # noqa: E402
import opponent as OPP  # noqa: E402
import title_value as TV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROLES = ("core", "mid", "support")
# 「关键之人」的两种读法只在 2-0 的系列赛上不同（docs/07-titles.md §5.3）。
# 结论走保守的读法 A，读法 B 的差额单独报一行。
OPTIMISTIC = "clutch_played"
DEFAULT_HAVE = ("red", "cruel")


def slot_arrays(rows):
    """把一个槽位摊成数组：胜负掩码、每个前缀的逐局得分、每个后缀的逐局命中概率。

    自助抽样只要在这些数组上重抽下标就行，不用回到 rows 重算一遍分。
    """
    win = np.array([r["win"] for r in rows])
    zero = np.zeros(len(rows))

    vals = {None: np.array([r["scores"].mean() for r in rows])}
    for pre, b in TV.PREFIX_BONUS.items():
        vals[pre] = np.array([
            float(np.mean([s * (1 + b * (pre in t))
                           for s, t in zip(r["scores"], r["tags"])]))
            for r in rows])

    conds = {None: (zero, 0.0, "iid")}
    for suf, (b, kind, _lbl) in TV.SUFFIXES.items():
        p = TV.game_cond_p(rows, kind)
        if p is None:
            p = np.full(len(rows), TV.CONST_P.get(suf, 0.0))
        conds[suf] = (p, b, kind)
    return win, vals, conds


def eval_one(arr, draws, title, pick=None):
    """一个 (前缀, 后缀) 组合的 E[一期得分]。pick=(赢的局下标, 输的局下标)
    时按这组下标重抽（自助抽样），不传就是原样本。"""
    win, vals, conds = arr
    v, (p, b, kind) = vals[title[0]], conds[title[1]]
    vw, pw, vl, pl = v[win], p[win], v[~win], p[~win]
    if pick is not None:
        iw, il = pick
        vw, pw, vl, pl = vw[iw], pw[iw], vl[il], pl[il]
    return float(TV.period_value(vw, pw, vl, pl, b, kind, draws).mean())


def eval_slot(arr, draws):
    """一个槽位在 (前缀, 后缀) 每个组合下的 E[一期得分]。None 表示不带。"""
    _win, vals, conds = arr
    return {(pre, suf): eval_one(arr, draws, (pre, suf))
            for pre in vals for suf in conds}


def boot_dist(arr, draws, title, n_boot, rng):
    """对**比赛样本**做分层自助抽样（胜局、负局各自重抽，保持胜率不变）。

    比赛的抽样噪声就是「这支队是不是恰好走运」的度量：如果一个槽位的高分只靠
    两三局爆发撑着，重抽掉它们分就塌了，自助分布会很宽。
    """
    win = arr[0]
    nw, nl = int(win.sum()), int((~win).sum())
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = (rng.integers(0, nw, nw) if nw else np.zeros(0, int),
                rng.integers(0, nl, nl) if nl else np.zeros(0, int))
        out[b] = eval_one(arr, draws, title, pick)
    return out


def hit_rates(rows):
    """这个槽位每个前缀的命中率 = 逐局「命中的选手占比」的平均。"""
    return {pre: float(np.mean([np.mean([pre in t for t in r["tags"]])
                                for r in rows]))
            for pre in TV.PREFIX_BONUS}


def pname(pre):
    return "裸分" if pre is None else TV.PREFIX_ZH[pre]


def sname(suf):
    return "裸分" if suf is None else TV.SUFFIXES[suf][2].split()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=12000)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--have", default=None,
                    help="手上的称号，如 red,cruel（默认读 log，再默认 猩红的/残酷之人）")
    ap.add_argument("--no-title", action="store_true", help="旧口径：只看裸分")
    ap.add_argument("--raw", action="store_true",
                    help="不做对手强度校正（默认做，见 scripts/opponent.py）")
    ap.add_argument("--slope", choices=("total", "dur"), default="total",
                    help="对手强度斜率：total=总效应（默认），dur=控住时长后的偏效应")
    ap.add_argument("--boot", type=int, default=200,
                    help="对比赛样本做几次自助抽样，0 = 不做")
    ap.add_argument("--boot-sims", type=int, default=3000)
    args = ap.parse_args()

    log = BD.load_log()
    st, banners, held = BD.current(log)
    have = tuple((args.have or st.get("title") or ",".join(DEFAULT_HAVE)).split(","))
    if args.no_title:
        have = (None, None)

    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    _, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)

    # 定稿的战旗倍率就是每个统计项的权重
    W = {r: {s: w for (s, _, _), w in zip(banners[r], BC.multipliers(banners[r]))}
         for r in ROLES}
    slots = TV.slot_games(by_player, roles, lambda t, r: W[r])

    B = np.load(os.path.join(ROOT, "data", "sim_buckets_boot.npy"))
    names = json.load(open(os.path.join(ROOT, "data", "sim_teams_boot.json")))
    tcol = {n: k for k, n in enumerate(names)}
    rng = np.random.default_rng(31337)
    Bs = B[rng.integers(0, B.shape[0], args.sims)]

    print("定稿的三面战旗（倍率已与客户端对账）")
    for r in ROLES:
        b = banners[r]
        m = [round(100 * x) for x in BC.multipliers(b)]
        print(f"  {BD.ROLE_CN[r]:<4}" + "   ".join(
            f"{BD.CN[s]} {mm}%" for (s, _, _), mm in zip(b, m)))

    # ---- 阵容改动与样本不足的槽位，明说出来，不要让它们静悄悄地消失
    ov = json.load(open(os.path.join(ROOT, "data", "roster_overrides.json")))
    for tname, spec in ov.items():
        if not isinstance(spec, dict) or tname not in teams:
            continue
        print("\n阵容改动（data/roster_overrides.json，{}）：{} {} → {}".format(
            spec.get("as_of", ""), tname,
            "／".join(f"{d['name']}（{d.get('reason', '')}）"
                      for d in spec.get("drop", [])),
            "／".join(a["name"] for a in spec.get("add", []))))
    short = []
    for team in sorted(teams):
        r = roles.get(team)
        if not r:
            short.append((team, "五人凑不齐"))
            continue
        for role in ROLES:
            if (team, role) not in slots:
                n = len({g["match_id"] for a in r[role]
                         for g in by_player.get(a, [])})
                short.append((f"{team} {BD.ROLE_CN[role]}", f"只有 {n} 场，样本不足"))
    for what, why in short:
        print(f"  ⚠ {what}：{why} —— 不进排名")

    # ---- 对手强度校正：把「打谁打出来的分」折算到 TI 的对手水平
    slots0 = slots
    if args.raw:
        adj = None
    else:
        R = OPP.team_ratings()
        mu = OPP.field_mean(teams, R)
        sl = OPP.slopes(slots, R, control_duration=args.slope == "dur")
        slots = {(t, r): OPP.adjust(rows, r, R, mu, sl)
                 for (t, r), rows in slots.items()}
        adj = (R, mu, sl)
        print(f"\n对手强度校正（TI 全场平均 {mu:+.3f}，全池 {len(R)} 队）：" + "  ".join(
            f"{BD.ROLE_CN[r]} {100 * sl[r][0]:+.1f}%/单位 (t={sl[r][1]:+.1f})"
            for r in ROLES))

    # match_id -> 参赛的两个 team_id。选手的历史里混着他在**老东家**打的局，
    # 新组的队（LGD 只有 58 场队伍比赛）会因此被严重高估或低估，所以每支队都
    # 同时报「全样本」和「只算在本队打的局」，两个读法一致才敢当结论。
    m2t = {}
    for a, gs in by_player.items():
        for g in gs:
            m2t.setdefault(g["match_id"], set()).update(
                [g.get("radiant_team_id"), g.get("dire_team_id")])

    # ---- 逐槽位把 (前缀 × 后缀) 的立方体算出来，全样本与仅本队各一份
    cube, hits, n_own, arrays, opp_mean = {}, {}, {}, {}, {}
    for (team, role), rows in sorted(slots.items()):
        draws = TV.Draws(Bs[:, tcol[team]], args.sims)
        tid = teams[team]["team_id"]
        own = [x for x in rows if tid in m2t.get(x["match_id"], set())]
        n_own[(team, role)] = (len(own), len(rows))
        hits[(team, role)] = hit_rates(rows)
        if adj:
            o = [adj[0][x["opp"]] for x in rows if x["opp"] in adj[0]]
            opp_mean[(team, role)] = float(np.mean(o)) if o else float("nan")
        a_full = slot_arrays(rows)
        a_org = slot_arrays(own) if len(own) >= 50 else None
        arrays[(team, role)] = (a_full, a_org)
        cube[(team, role)] = {
            "full": eval_slot(a_full, draws),
            "org": eval_slot(a_org, draws) if a_org else None,
        }

    def val(team, role, title, reading="lo"):
        """title=(prefix, suffix)。lo = 两个读法里较差的那个。"""
        c = cube[(team, role)]
        full = c["full"][title]
        org = c["org"][title] if c["org"] else None
        if reading == "full":
            return full
        if reading == "org":
            return org
        return full if org is None else min(full, org)

    def best_team(role, title):
        cand = [t for (t, r) in cube if r == role]
        return max(cand, key=lambda t: val(t, role, title))

    # ---- 联合最优：外层 72 个称号组合，内层三个定位各挑各的队
    prefixes = [None] if args.no_title else list(TV.PREFIX_BONUS)
    suffixes = [None] if args.no_title else list(TV.SUFFIXES)
    combos = []
    for pre in prefixes:
        for suf in suffixes:
            picks = {r: best_team(r, (pre, suf)) for r in ROLES}
            tot = sum(val(picks[r], r, (pre, suf)) for r in ROLES)
            combos.append((tot, (pre, suf), picks))
    combos.sort(key=lambda x: -x[0])
    conservative = [c for c in combos if c[1][1] != OPTIMISTIC]
    best_tot, best_title, best_picks = conservative[0]

    # ---- 自助抽样：这支队的领先是真的，还是几局爆发撑出来的
    boot, p_best = {}, {}
    if args.boot:
        brng = np.random.default_rng(101)
        for (team, role), (a_full, a_org) in sorted(arrays.items()):
            d = TV.Draws(Bs[:args.boot_sims, tcol[team]], args.boot_sims)
            bf = boot_dist(a_full, d, best_title, args.boot, brng)
            if a_org is not None:
                bf = np.minimum(bf, boot_dist(a_org, d, best_title,
                                              args.boot, brng))
            boot[(team, role)] = bf
        for role in ROLES:
            cand = sorted(t for (t, r) in cube if r == role)
            M = np.array([boot[(t, role)] for t in cand])       # (队, 抽样)
            wins = np.bincount(M.argmax(axis=0), minlength=len(cand))
            for k, t in enumerate(cand):
                p_best[(t, role)] = wins[k] / args.boot

    # ---- 每个定位的队伍排名（在建议的称号下）
    for r in ROLES:
        stat_set = tuple(x[0] for x in banners[r])
        head = ("" if args.no_title else
                f"   （含称号：{pname(best_title[0])}+{sname(best_title[1])}）")
        print(f"\n=== {BD.ROLE_CN[r]}   " + "+".join(BD.CN[s] for s in stat_set)
              + head)
        print(f"  {'队伍':<18}{'全样本':>9}{'仅本队':>9}{'较差者':>9}"
              + (f"{'自助p25':>9}{'P(最优)':>9}" if args.boot else "")
              + ("" if args.no_title else f"{'前缀命中':>9}")
              + (f"{'对手均值':>10}" if adj else "")
              + f"{'本队局数':>12}")
        rows_r = sorted((t for (t, rr) in cube if rr == r),
                        key=lambda t: -val(t, r, best_title))
        for i, team in enumerate(rows_r[:args.top], 1):
            full = val(team, r, best_title, "full")
            org = val(team, r, best_title, "org")
            lo = val(team, r, best_title)
            o = f"{org:>9,.0f}" if org is not None else f"{'样本不足':>7}"
            tag = "  ← 现在" if team == held[r] else ""
            cells = ""
            if args.boot:
                cells += (f"{np.percentile(boot[team, r], 25):>9,.0f}"
                          f"{p_best[team, r]:>9.0%}")
            if not args.no_title:
                cells += f"{hits[team, r][best_title[0]]:>9.1%}"
            if adj:
                cells += f"{opp_mean[team, r]:>+10.3f}"
            print(f"{i:>3} {team:<18}{full:>9,.0f}{o}{lo:>9,.0f}{cells}"
                  f"{n_own[team, r][0]:>7}/{n_own[team, r][1]:<4}{tag}")
        if rows_r[0] != held[r]:
            d = val(rows_r[0], r, best_title) - val(held[r], r, best_title)
            print(f"  → 换成 {rows_r[0]}：{d:+,.0f}"
                  + (f"；自助抽样里 {rows_r[0]} 最优的比例 "
                     f"{p_best[rows_r[0], r]:.0%}，{held[r]} 是 "
                     f"{p_best[held[r], r]:.0%}" if args.boot else ""))

    # ---- 对手校正的三种读法会不会给出不同的队
    robust = dict(best_picks)
    if adj:
        R, mu, sl_total = adj
        modes = [("不校正", None), ("总效应", sl_total),
                 ("控时长", OPP.slopes(slots0, R, control_duration=True))]
        print(f"\n对手强度三种读法（建议称号下，各定位前 4 名）")
        print(f"  {'队伍':<20}" + "".join(f"{lbl:>10}" for lbl, _ in modes)
              + f"{'最差读法':>12}")
        for r in ROLES:
            cand = sorted((t for (t, rr) in cube if rr == r),
                          key=lambda t: -val(t, r, best_title))[:4]
            worst = {}
            for team in cand:
                d = TV.Draws(Bs[:, tcol[team]], args.sims)
                tid = teams[team]["team_id"]
                vs = []
                for _lbl, s in modes:
                    rows = (slots0[(team, r)] if s is None else
                            OPP.adjust(slots0[(team, r)], r, R, mu, s))
                    own = [x for x in rows
                           if tid in m2t.get(x["match_id"], set())]
                    v = eval_one(slot_arrays(rows), d, best_title)
                    if len(own) >= 50:
                        v = min(v, eval_one(slot_arrays(own), d, best_title))
                    vs.append(v)
                worst[team] = vs
            robust[r] = max(cand, key=lambda t: min(worst[t]))
            print(f"{BD.ROLE_CN[r]}")
            for team in cand:
                mark = "  ← 最差读法最优" if team == robust[r] else ""
                print(f"  {team:<20}" + "".join(f"{v:>10,.0f}" for v in worst[team])
                      + f"{min(worst[team]):>12,.0f}{mark}")

    if args.no_title:
        print("\n" + "=" * 78)
        cur = sum(val(held[r], r, (None, None)) for r in ROLES)
        print(f"当前阵容 {cur:>10,.0f}   "
              + "  ".join(f"{BD.ROLE_CN[r]}={held[r]}" for r in ROLES))
        print(f"建议阵容 {best_tot:>10,.0f}   "
              + "  ".join(f"{BD.ROLE_CN[r]}={best_picks[r]}" for r in ROLES))
        return

    # ---- 前缀会不会改变选队？这是「选队要考虑英雄」的直接检验
    print(f"\n{'前缀':<14}{'加成':>5}   " + "".join(
        f"{BD.ROLE_CN[r] + ' 最优队':<22}" for r in ROLES) + "合计")
    for pre in sorted(TV.PREFIX_BONUS, key=lambda p: -TV.PREFIX_BONUS[p]):
        title = (pre, best_title[1])
        picks = {r: best_team(r, title) for r in ROLES}
        tot = sum(val(picks[r], r, title) for r in ROLES)
        cells = "".join(
            f"{picks[r] + ' ' + format(hits[picks[r], r][pre], '.0%'):<22}"
            for r in ROLES)
        print(f"{TV.PREFIX_ZH[pre]:<14}{TV.PREFIX_BONUS[pre]:>5.0%}   "
              f"{cells}{tot:>10,.0f}")

    # ---- 后缀（队伍固定在建议阵容上）
    print(f"\n{'后缀':<34}{'加成':>5}{'合计':>10}{'相对裸后缀':>12}")
    zero_suf = sum(val(best_picks[r], r, (best_title[0], None)) for r in ROLES)
    srows = []
    for suf, (b, _kind, lbl) in TV.SUFFIXES.items():
        tot = sum(val(best_picks[r], r, (best_title[0], suf)) for r in ROLES)
        srows.append((tot, suf, lbl, b))
    for tot, suf, lbl, b in sorted(srows, reverse=True):
        mark = "  ★读法B" if suf == OPTIMISTIC else ""
        print(f"{lbl:<34}{b:>5.0%}{tot:>10,.0f}{tot - zero_suf:>+12,.0f}{mark}")

    # ---- 结论
    print("\n" + "=" * 78)
    cur_tot = sum(val(held[r], r, have) for r in ROLES)
    naked = sum(val(held[r], r, (None, None)) for r in ROLES)
    print(f"{'现状':<10}{cur_tot:>10,.0f}   {pname(have[0])}+{sname(have[1])}   "
          + "  ".join(f"{BD.ROLE_CN[r]}={held[r]}" for r in ROLES))
    print(f"{'（裸分）':<9}{naked:>10,.0f}   —— 称号在现状下值 {cur_tot - naked:+,.0f}")
    keep_team = max(((sum(val(held[r], r, t) for r in ROLES), t)
                     for _, t, _ in conservative), key=lambda x: x[0])
    print(f"{'只换称号':<10}{keep_team[0]:>10,.0f}   "
          f"{pname(keep_team[1][0])}+{sname(keep_team[1][1])}   队伍不动   "
          f"（{keep_team[0] - cur_tot:+,.0f}）")
    keep_title = {r: best_team(r, have) for r in ROLES}
    kt = sum(val(keep_title[r], r, have) for r in ROLES)
    print(f"{'只换队伍':<10}{kt:>10,.0f}   称号不动   "
          + "  ".join(f"{BD.ROLE_CN[r]}={keep_title[r]}" for r in ROLES)
          + f"   （{kt - cur_tot:+,.0f}）")
    # 「不看英雄选队」会损失多少 —— 先按裸分挑队，再在这套队上挂最优称号
    naive = {r: best_team(r, (None, None)) for r in ROLES}
    nv = sum(val(naive[r], r, best_title) for r in ROLES)
    print(f"{'按裸分选队':<9}{nv:>10,.0f}   "
          f"{pname(best_title[0])}+{sname(best_title[1])}   "
          + "  ".join(f"{BD.ROLE_CN[r]}={naive[r]}" for r in ROLES)
          + f"   （比联合最优少 {best_tot - nv:,.0f}）")
    print(f"{'联合最优':<10}{best_tot:>10,.0f}   "
          f"{pname(best_title[0])}+{sname(best_title[1])}   "
          + "  ".join(f"{BD.ROLE_CN[r]}={best_picks[r]}" for r in ROLES)
          + f"   （{best_tot - cur_tot:+,.0f}）")
    if adj and robust != best_picks:
        rv = sum(val(robust[r], r, best_title) for r in ROLES)
        print(f"{'稳健(读法)':<8}{rv:>10,.0f}   "
              f"{pname(best_title[0])}+{sname(best_title[1])}   "
              + "  ".join(f"{BD.ROLE_CN[r]}={robust[r]}" for r in ROLES)
              + f"   （三种对手校正读法里最差的那个也最优；点估计少 "
                f"{best_tot - rv:,.0f}）")
    if args.boot:
        # 稳健读法：按自助分布的 25% 分位挑队（点估计高但靠几局爆发撑着的会掉下去）
        safe = {r: max((t for (t, rr) in cube if rr == r),
                       key=lambda t: np.percentile(boot[t, r], 25))
                for r in ROLES}
        sv = sum(val(safe[r], r, best_title) for r in ROLES)
        same = all(safe[r] == best_picks[r] for r in ROLES)
        print(f"{'稳健(p25)':<8}{sv:>10,.0f}   "
              f"{pname(best_title[0])}+{sname(best_title[1])}   "
              + "  ".join(f"{BD.ROLE_CN[r]}={safe[r]}" for r in ROLES)
              + ("   —— 与联合最优同一套队" if same
                 else f"   （点估计比联合最优少 {best_tot - sv:,.0f}）"))
    opt = combos[0]
    if opt[1][1] == OPTIMISTIC:
        print(f"{'（读法B）':<9}{opt[0]:>10,.0f}   "
              f"{pname(opt[1][0])}+{sname(opt[1][1])}   "
              + "  ".join(f"{BD.ROLE_CN[r]}={opt[2][r]}" for r in ROLES)
              + "   —— 关键之人按「实际打的最后一局」判定时")

    out = {f"{t}|{r}": {
        "n_own": n_own[t, r][0], "n_all": n_own[t, r][1],
        "hit": hits[t, r],
        "opp_mean": opp_mean.get((t, r)),
        "boot": (None if not args.boot else
                 {"title": "|".join(map(str, best_title)),
                  "mean": float(boot[t, r].mean()),
                  "sd": float(boot[t, r].std()),
                  "p25": float(np.percentile(boot[t, r], 25)),
                  "p_best": p_best[t, r]}),
        "full": {f"{p}|{s}": v for (p, s), v in c["full"].items()},
        "org": (None if c["org"] is None
                else {f"{p}|{s}": v for (p, s), v in c["org"].items()}),
    } for (t, r), c in cube.items()}
    path = os.path.join(ROOT, "data", "roster_title.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"\n-> data/roster_title.json（{len(out)} 个槽位 × 9 前缀 × 10 后缀）")


if __name__ == "__main__":
    main()
