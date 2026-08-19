"""称号层估值：前缀（英雄标签）× 后缀（比赛事件），放进 max2(sum) 里算。

两条不能用 `命中率 × 加成` 排序的理由：

  1. 计分取一个系列赛里最好的两局、再取一期里最好的系列赛。命中的那一局被抬高，
     因而**更容易被选进 top2** —— 罕见但大的加成比常见但小的更值钱。
  2. 前缀是**选手级**条件：核心/辅助是两人平均，一人命中只得部分加成，
     而且权重是那名选手在这一局里的得分占比，不是二分之一。

所以这里逐局、逐人地把乘子乘进去，再跑与 `banner_value.py` 同构的模拟。
所有选项共用同一批随机数（局号、系列赛长度、事件是否发生都预先抽好），
选项之间是配对比较。

前缀命中判定来自 `data/hero_adjectives.json`（客户端 npc_heroes.txt 的
Adjectives 块，见 `scripts/hero_adjectives.py`）—— 是判定表本身，不是估计。

后缀有五个能逐局确定（时长 <25 分、时长秒数末位为 8、该局输、第一滴血 <1 分、
第一滴血 >6 分），「泉水阵亡」按时长条件概率抽（STRATZ 400 场拟合），
「关键之人」是系列赛内的位置条件（两种读法都算了），其余按常数概率抽。

输出 data/title_value.json。
"""

import functools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

P_THREE = 0.382
RECORD = {0: (4, 4), 1: (5, 4), 2: (6, 3.5), 3: (6, 2.5), 4: (5, 1), 5: (4, 0)}

# 手上的 8 个前缀（+ 全池里另外两个已能算的），加成来自客户端
PREFIX_BONUS = {"blue": 0.11, "purple": 0.10, "heroic": 0.09, "golden": 0.08,
                "elemental": 0.08, "otherworldly": 0.07, "red": 0.06,
                "green": 0.06}
PREFIX_ZH = {"red": "猩红的", "blue": "蔚蓝的", "green": "碧绿的", "purple": "紫气的",
             "golden": "金光的", "elemental": "精通元素的",
             "otherworldly": "异界的", "heroic": "盖世英雄",
             "hairy": "（全池）有胡子或毛茸茸", "horned": "（全池）有角或翅膀"}

# 后缀：(加成, 判定方式)。判定条件以 fantasy_crafting.vdata_c 里的 m_vecStats
# 为准 —— 有两条与帮助页面的文案对不上，见 scripts/fantasy_crafting.py。
#   dur_lt   时长 < 25 分钟          逐局确定
#   sec8     时长秒数末位为 8        逐局确定
#   loss     该局输                  逐局确定
#   fb_early 第一滴血 < 1 分钟       逐局确定（未解析的局用总体比例）
#   fb_late  第一滴血 > 6 分钟       同上
#   fountain 有人死在己方泉水        按时长条件概率
#   const    常数概率
#   clutch   系列赛可能的最后一局    位置条件
#   clutch2  系列赛实际打的最后一局  位置条件
SUFFIXES = {
    "decisive":  (0.24, "dur_lt",   "果敢之人  时长<25分"),
    "tormented": (0.23, "const",    "受难之人  有人死于痛苦魔方"),
    "patient":   (0.23, "fb_late",  "隐忍之人  第一滴血>6分(文案写10分)"),
    "lucky":     (0.21, "sec8",     "幸运之人  比赛时间以8结尾"),
    "clutch":    (0.16, "clutch",   "关键之人  系列赛「可能的」最后一局"),
    # 同一条后缀的另一种读法，见文档：统计名是 league_last_match_in_series
    "clutch_played": (0.16, "clutch2", "关键之人  系列赛实际打的最后一局"),
    "cruel":     (0.13, "fountain", "残酷之人  有人死在己方泉水"),
    "flayed":    (0.09, "fb_early", "剥皮双子  第一滴血<1分(文案写号角前)"),
    "underdog":  (0.06, "loss",     "败者      该选手失利的比赛"),
}
CONST_P = {"tormented": 0.039}


# ---------------------------------------------------------------- 数据

def match_meta():
    """match_id -> 第一滴血时刻。OpenDota 把未解析的局记成 0，与真正的 0 秒混在
    一起（顶级职业局里 15.6%），这些局按 None 处理、回落到总体比例。"""
    path = os.path.join(ROOT, "data", "raw", "pro_matches_rich.json")
    meta = {}
    for m in json.load(open(path)):
        fb = m.get("first_blood_time")
        meta[m["match_id"]] = None if not fb else fb
    return meta


def slot_games(by_player, roles, weights_of):
    """(team, role) -> 逐局记录，含每名选手的得分与英雄，用于逐人乘加成。

    `weights_of(team, role)` 返回 {统计项: 倍率}，返回 None 表示跳过这个槽位。
    倍率全 1 就是裸的统计项之和（`banner_value.py` 第一步的口径），
    传战旗的实际倍率就是定稿后的口径（`roster_pick.py` 用的）。
    """
    H = json.load(open(os.path.join(ROOT, "data", "hero_adjectives.json")))["heroes"]
    FB = match_meta()
    out = {}
    for team, r in roles.items():
        if not r:
            continue
        for role in ("core", "mid", "support"):
            w = weights_of(team, role)
            if not w:
                continue
            per_match = {}
            for a in r[role]:
                for g in by_player.get(a, []):
                    per_match.setdefault(g["match_id"], []).append(g)
            rows = []
            for mid, gs in per_match.items():
                scores, tags = [], []
                for g in gs:
                    p = FS.to_points(FS.game_stats(g))
                    scores.append(float(np.nansum([p[s] * wi
                                                   for s, wi in w.items()])))
                    tags.append(set(H[str(g["hero_id"])]["prefix"]))
                g0 = gs[0]
                radiant = g0["player_slot"] < 128
                rows.append({
                    "match_id": mid,
                    "scores": np.array(scores), "tags": tags,
                    "win": radiant == bool(g0["radiant_win"]),
                    "duration": g0["duration"],
                    "fb": FB.get(mid),
                    # 对手是「这名选手当时的对面」，不是「本队的对面」—— 老东家的局
                    # 也照样有对手，用 player_slot 定边才不会把它们丢掉
                    "opp": g0.get("dire_team_id") if radiant
                    else g0.get("radiant_team_id"),
                })
            if len(rows) >= 12:
                out[(team, role)] = rows
    return out


@functools.lru_cache(maxsize=1)
def fountain_curve():
    """P(有人死在己方泉水 | 时长) —— STRATZ 400 场，按时长分箱后线性内插。"""
    path = os.path.join(ROOT, "data", "raw", "stratz_deaths.json")
    if not os.path.exists(path):
        return lambda d: 0.308
    raw = json.load(open(path))
    FOUNT = {True: (74.0, 78.0), False: (184.0, 180.0)}   # 天辉 / 夜魇泉水
    R2 = 8.0 ** 2
    pts = []
    for m in raw.values():
        if not m:
            continue
        hit = False
        for p in m.get("players") or []:
            fx, fy = FOUNT[bool(p["isRadiant"])]
            for e in ((p.get("stats") or {}).get("deathEvents") or []):
                x, y = e.get("positionX"), e.get("positionY")
                if x is None:
                    continue
                if (x - fx) ** 2 + (y - fy) ** 2 <= R2:
                    hit = True
                    break
            if hit:
                break
        pts.append((m["durationSeconds"], hit))
    pts.sort()
    edges = [0, 1800, 2100, 2400, 2700, 3000, 10 ** 9]
    mids, rate = [], []
    for lo, hi in zip(edges, edges[1:]):
        sub = [h for d, h in pts if lo <= d < hi]
        if len(sub) >= 15:
            mids.append(min(hi, 3600) if lo else 1500)
            rate.append(sum(sub) / len(sub))
    mids = np.array(mids, float)
    rate = np.array(rate, float)
    return lambda d: float(np.interp(d, mids, rate))


def game_cond_p(rows, kind):
    """每一局该后缀命中的概率（确定性条件返回 0/1）。"""
    if kind == "dur_lt":
        return np.array([1.0 if r["duration"] < 1500 else 0.0 for r in rows])
    if kind == "sec8":
        return np.array([1.0 if r["duration"] % 10 == 8 else 0.0 for r in rows])
    if kind == "loss":
        return np.array([0.0 if r["win"] else 1.0 for r in rows])
    if kind == "fountain":
        f = fountain_curve()
        return np.array([f(r["duration"]) for r in rows])
    if kind in ("fb_early", "fb_late"):
        known = [r["fb"] for r in rows if r["fb"]]
        test = (lambda t: t < 60) if kind == "fb_early" else (lambda t: t > 360)
        rate = np.mean([test(t) for t in known]) if known else 0.0
        return np.array([rate if not r["fb"] else float(test(r["fb"]))
                         for r in rows])
    return None


# ---------------------------------------------------------------- 模拟

class Draws:
    """预先抽好的随机数：所有称号选项看到完全相同的抽样。"""

    def __init__(self, bucket_col, n_sims, seed=11):
        rng = np.random.default_rng(seed)
        self.bucket_col = bucket_col
        self.groups = []
        for bucket in range(6):
            mask = bucket_col == bucket
            k = int(mask.sum())
            if not k:
                continue
            n_series, exp_wins = RECORD[bucket]
            lo = int(np.floor(exp_wins))
            n_wins = lo + (rng.random(k) < exp_wins - lo).astype(int)
            self.groups.append({
                "mask": mask, "k": k, "n_series": n_series, "n_wins": n_wins,
                "u_idx": rng.random((k, n_series, 3)),
                "u_three": rng.random((k, n_series)),
                "u_cond": rng.random((k, n_series, 3)),
            })


def period_value(vals_w, p_w, vals_l, p_l, bonus, kind, draws):
    """E[一期内最好的系列赛]，逐局施加称号乘子。"""
    n = len(draws.bucket_col)
    out = np.zeros(n)
    if len(vals_w) == 0:
        vals_w, p_w = vals_l, p_l
    if len(vals_l) == 0:
        vals_l, p_l = vals_w, p_w

    def pick(u, is_win):
        """u:(k,) -> 该局得分（已乘上称号乘子，clutch 除外）"""
        iw = (u * len(vals_w)).astype(int)
        il = (u * len(vals_l)).astype(int)
        v = np.where(is_win, vals_w[iw], vals_l[il])
        p = np.where(is_win, p_w[iw], p_l[il]) if p_w is not None else None
        return v, p

    for g in draws.groups:
        k, ns = g["k"], g["n_series"]
        best = np.zeros(k)
        for s in range(ns):
            is_win = s < g["n_wins"]
            three = g["u_three"][:, s] < P_THREE
            gv = []
            for j in range(3):
                # 前两局与系列赛结果同向，第三局（j=2）相反
                w = is_win if j < 2 else ~is_win
                v, p = pick(g["u_idx"][:, s, j], w)
                if kind == "clutch":
                    # 第三局才是「可能的最后一局」，且只有打满才存在
                    mult = 1.0 + bonus * (three & (j == 1))
                elif kind == "clutch2":
                    # 系列赛实际打的最后一局：2-0 时是第二局，打满时是第三局。
                    # 两种读法只在 2-0 的系列赛上不同。
                    mult = 1.0 + bonus * (j == 1)
                else:
                    mult = 1.0 + bonus * (g["u_cond"][:, s, j] < p)
                gv.append(v * mult)
            g1, g2, g3 = gv
            top2 = np.where(three, g1 + g2 + g3 - np.minimum(np.minimum(g1, g2), g3),
                            g1 + g2)
            best = np.maximum(best, top2)
        out[g["mask"]] = best
    return out


# ---------------------------------------------------------------- 主流程

def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    banners = json.load(open(os.path.join(ROOT, "data", "banner_value.json")))
    _, by_player = FS.load()
    roles = FS.assign_roles(by_player, teams)
    slots = slot_games(by_player, roles, lambda t, r: {
        s: 1.0 for s in banners.get(f"{t}|{r}", {}).get("best", [])})

    B = np.load(os.path.join(ROOT, "data", "sim_buckets_boot.npy"))
    names = json.load(open(os.path.join(ROOT, "data", "sim_teams_boot.json")))
    tcol = {n: k for k, n in enumerate(names)}
    rng = np.random.default_rng(31337)
    Bs = B[rng.integers(0, B.shape[0], n_sims)]

    result = {}
    for (team, role), rows in sorted(slots.items()):
        draws = Draws(Bs[:, tcol[team]], n_sims)
        win = np.array([r["win"] for r in rows])
        base = np.array([r["scores"].mean() for r in rows])
        zero = np.zeros(len(rows))

        ref = period_value(base[win], zero[win], base[~win], zero[~win],
                           0.0, "iid", draws).mean()
        entry = {"base": float(ref), "prefix": {}, "suffix": {}}

        for pre, b in PREFIX_BONUS.items():
            # 逐人乘：加成只作用于打了该类英雄的那名选手
            v = np.array([float(np.mean([s * (1 + b * (pre in t))
                                         for s, t in zip(r["scores"], r["tags"])]))
                          for r in rows])
            got = period_value(v[win], zero[win], v[~win], zero[~win],
                               0.0, "iid", draws).mean()
            hit = float(np.mean([np.mean([pre in t for t in r["tags"]])
                                 for r in rows]))
            entry["prefix"][pre] = {"uplift": 100 * (got / ref - 1),
                                    "hit": hit, "naive": 100 * b * hit}

        for suf, (b, kind, _lbl) in SUFFIXES.items():
            p = game_cond_p(rows, kind)
            if p is None:
                p = np.full(len(rows), CONST_P.get(suf, 0.0))
            got = period_value(base[win], p[win], base[~win], p[~win],
                               b, kind, draws).mean()
            if kind == "clutch":
                hit = P_THREE / (2 + P_THREE)
            elif kind == "clutch2":
                hit = 1.0 / (2 + P_THREE)
            else:
                hit = float(p.mean())
            entry["suffix"][suf] = {"uplift": 100 * (got / ref - 1),
                                    "hit": hit, "naive": 100 * b * hit}
        result[f"{team}|{role}"] = entry

    json.dump(result, open(os.path.join(ROOT, "data", "title_value.json"), "w"),
              indent=1, ensure_ascii=False)

    # ---- 报告
    best_slot = {}
    for role in ("core", "mid", "support"):
        ks = [k for k in result if k.endswith("|" + role)]
        best_slot[role] = max(ks, key=lambda k: result[k]["base"])

    print(f"最优阵容（沿用 banner_value 的槽位）：")
    for role, k in best_slot.items():
        print(f"  {role:<8}{k.split('|')[0]:<18}裸分 {result[k]['base']:>9,.0f}")
    tot = sum(result[k]["base"] for k in best_slot.values())
    print(f"  合计 {tot:,.0f}\n")

    print(f"{'前缀':<22}{'命中率':>8}{'朴素 b×p':>10}"
          f"{'实际增益(核心/中单/辅助)':>26}{'合计加分':>10}")
    rowsp = []
    for pre in sorted(PREFIX_BONUS, key=lambda p: -PREFIX_BONUS[p]):
        ups = [result[best_slot[r]]["prefix"][pre] for r in
               ("core", "mid", "support")]
        gain = sum(result[best_slot[r]]["base"] * ups[i]["uplift"] / 100
                   for i, r in enumerate(("core", "mid", "support")))
        hit = np.mean([u["hit"] for u in ups])
        rowsp.append((gain, pre, hit, ups))
    for gain, pre, hit, ups in sorted(rowsp, reverse=True):
        nm = f"{PREFIX_ZH[pre]} +{PREFIX_BONUS[pre]:.0%}"
        per = " / ".join("%.2f%%" % u["uplift"] for u in ups)
        naive = np.mean([u["naive"] for u in ups])
        print(f"{nm:<22}{hit:>8.1%}{naive:>9.2f}%{per:>26}{gain:>10,.0f}")

    print(f"\n{'后缀':<34}{'命中率':>8}{'朴素 b×p':>10}"
          f"{'实际增益(核心/中单/辅助)':>26}{'合计加分':>10}")
    rowss = []
    for suf, (b, kind, lbl) in SUFFIXES.items():
        ups = [result[best_slot[r]]["suffix"][suf] for r in
               ("core", "mid", "support")]
        gain = sum(result[best_slot[r]]["base"] * ups[i]["uplift"] / 100
                   for i, r in enumerate(("core", "mid", "support")))
        rowss.append((gain, suf, lbl, b, ups))
    for gain, suf, lbl, b, ups in sorted(rowss, reverse=True):
        nm = f"{lbl} +{b:.0%}"
        per = " / ".join("%.2f%%" % u["uplift"] for u in ups)
        hit = np.mean([u["hit"] for u in ups])
        naive = np.mean([u["naive"] for u in ups])
        print(f"{nm:<34}{hit:>8.1%}{naive:>9.2f}%{per:>26}{gain:>10,.0f}")
    print("\n-> data/title_value.json")


if __name__ == "__main__":
    main()
