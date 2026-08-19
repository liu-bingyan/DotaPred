"""Polymarket vs 我们的模型，在 TI2026 小组赛的同一批系列赛上正面对比。

主赛事首轮我们和市场在 Liquid vs Yandex 上分歧 15.5 个百分点，方向相反。谁更准这件
事不用猜 —— 同一个赛事的小组赛已经打完，Polymarket 上每个系列赛都有盘口，取赛前
最后一个成交价，和模型的赛前预测放在一起，用同一批结果评分。

模型这边是逐场重拟合（只用该系列赛开打之前的比赛），和市场看到的信息集对齐。

    python3 scripts/market_backtest.py
"""

import calendar
import collections
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin  # noqa: E402
import models  # noqa: E402
from experiment import MIN_PLAYER_GAMES, player_feature  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "raw", "polymarket_ti2026.json")
TI = 19719
HL, PLAYER_HL, L2T, L2P = float(os.environ.get("MB_HL", 45.0)), 150.0, 0.3, 30.0

NAME = {"TEAM VISION": "Team Vision", "BoomBoys": "BoomBoys", "Team Falcons": "Team Falcons",
        "Iron Wing": "Iron Wing", "Team Liquid": "Team Liquid", "Team Yandex": "Team Yandex",
        "Nigma Galaxy": "Nigma Galaxy", "Team Spirit": "Team Spirit",
        "Aurora": "Aurora Gaming", "LGD Gaming": "LGD Gaming",
        "Xtreme Gaming": "Xtreme Gaming", "Vici Gaming": "Vici Gaming",
        "GamerLegion": "GamerLegion", "Team Resilience": "Team Resilience",
        "HULIGANI": "Huligani", "OG": "OG"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "DotaPred/0.1"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {a + 1}: {e}", flush=True)
            time.sleep(3 * (a + 1))
    return None


def collect():
    """把 TI2026 的系列赛盘口抓下来并缓存（赛前最后成交价 + 结算结果）。"""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    slugs = set()
    for q in ["The International", "dota2", "Dota 2"] + list(NAME):
        d = get("https://gamma-api.polymarket.com/public-search?q="
                + urllib.parse.quote(q) + "&limit_per_type=60")
        for e in (d or {}).get("events", []):
            t = str(e.get("title") or "")
            if "Dota 2:" in t and "The International" in t and "Playoffs" not in t:
                slugs.add(e["slug"])
        time.sleep(0.4)
    print(f"找到 {len(slugs)} 个 TI2026 小组赛/淘汰赛盘口")

    out = []
    for s in sorted(slugs):
        d = get(f"https://gamma-api.polymarket.com/events?slug={s}")
        if not d:
            continue
        e = d[0]
        for m in e.get("markets", []):
            if "Match Winner" not in str(m.get("groupItemTitle") or ""):
                continue
            if not m.get("closed"):
                continue
            outs = json.loads(m.get("outcomes") or "[]")
            prs = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            ids = json.loads(m.get("clobTokenIds") or "[]")
            gst = m.get("gameStartTime") or e.get("startDate")
            if len(outs) != 2 or len(ids) != 2 or not gst:
                continue
            # gameStartTime 是 UTC。time.mktime 会当本地时间解析，必须用 timegm。
            ts = calendar.timegm(time.strptime(str(gst)[:19].replace("T", " "),
                                               "%Y-%m-%d %H:%M:%S"))
            h = get(f"https://clob.polymarket.com/prices-history?market={ids[0]}"
                    f"&startTs={ts - 36 * 3600}&endTs={ts}&fidelity=10")
            pre = [p for p in (h or {}).get("history", []) if p["t"] < ts]
            if not pre:
                continue
            lag = (ts - pre[-1]["t"]) / 60.0
            out.append(dict(slug=s, title=e["title"], team_a=outs[0], team_b=outs[1],
                            price_a=float(pre[-1]["p"]), a_won=prs[0] > 0.5,
                            start=ts, n_points=len(pre), lag_min=lag))
            print(f"  {outs[0][:14]:16s} vs {outs[1][:14]:16s} 赛前价 {pre[-1]['p']:.3f} "
                  f"(开赛前 {lag:.0f} 分钟) -> {'A胜' if prs[0] > 0.5 else 'B胜'}", flush=True)
            time.sleep(0.3)
    json.dump(out, open(CACHE, "w"), ensure_ascii=False)
    return out


def fit_at(rows, lineups, cutoff):
    i, j, w, idx, m, win, tr = margin.build_design(rows, cutoff, HL, 20)
    gd = m["gdpm"] / 400.0
    r_team, _ = margin.fit_margin(i, j, gd, w, len(idx), l2=L2T)
    _, _, wp, _, mp, _, trp = margin.build_design(rows, cutoff, PLAYER_HL, 20)
    counts = collections.Counter()
    for r in trp:
        lu = lineups.get(r["match_id"])
        if lu:
            for a in lu[0] + lu[1]:
                counts[a] += 1
    pidx = {a: k for k, a in enumerate(sorted(a for a, c in counts.items()
                                              if c >= MIN_PLAYER_GAMES))}
    prows, pkeep = [], []
    for k, r in enumerate(trp):
        lu = lineups.get(r["match_id"])
        if not lu:
            continue
        a = [pidx[x] for x in lu[0] if x in pidx]
        b = [pidx[x] for x in lu[1] if x in pidx]
        if len(a) >= 4 and len(b) >= 4:
            prows.append((a, b))
            pkeep.append(k)
    r_pl, _ = models.ridge_ratings(prows, wp[np.array(pkeep)],
                                   (mp["gdpm"] / 400.0)[np.array(pkeep)], len(pidx), l2=L2P)
    X = np.column_stack([r_team[i] - r_team[j], player_feature(tr, r_pl, pidx, lineups)])
    mu = np.nanmean(X, axis=0)
    beta = margin.fit_logistic(np.where(np.isnan(X), mu, X), win, w)
    return r_team, idx, r_pl, pidx, beta, mu


def main():
    mk = collect()
    print(f"\n可用盘口 {len(mk)} 个系列赛\n")

    aliases = {int(k): v for k, v in
               json.load(open(os.path.join(ROOT, "data", "aliases.json"))).items()}
    rows = [r for r in margin.load_rich(aliases) if r.get("tier") in
            {"premium", "professional"}]
    lineups = models.load_lineups()
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    tid = {n: v["team_id"] for n, v in teams.items()}
    roster = {n: [p["account_id"] for p in v["players"][:5]] for n, v in teams.items()}

    res = []
    for k, s in enumerate(sorted(mk, key=lambda x: x["start"])):
        a, b = NAME.get(s["team_a"]), NAME.get(s["team_b"])
        if a not in tid or b not in tid:
            continue
        rt, idx, rp, pidx, beta, mu = fit_at(rows, lineups, s["start"])
        if tid[a] not in idx or tid[b] not in idx:
            continue
        x = [rt[idx[tid[a]]] - rt[idx[tid[b]]]]
        pa = [rp[pidx[p]] for p in roster[a] if p in pidx]
        pb = [rp[pidx[p]] for p in roster[b] if p in pidx]
        x.append(np.mean(pa) - np.mean(pb) if len(pa) >= 4 and len(pb) >= 4 else mu[1])
        p1 = 1 / (1 + math.exp(-float(np.dot(beta[:2], x))))
        ours = p1 * p1 * (3 - 2 * p1)
        res.append((s, ours, s["price_a"], s["a_won"]))
        print(f"  {a[:13]:15s} vs {b[:13]:15s} 我们 {ours:.3f} 市场 {s['price_a']:.3f} "
              f"-> {'A胜' if s['a_won'] else 'B胜'}")

    def ll(p, y):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -(math.log(p) if y else math.log(1 - p))

    n = len(res)
    o_ll = np.array([ll(r[1], r[3]) for r in res])
    m_ll = np.array([ll(r[2], r[3]) for r in res])
    o_ac = np.array([(r[1] > 0.5) == r[3] for r in res], float)
    m_ac = np.array([(r[2] > 0.5) == r[3] for r in res], float)
    blend = np.array([ll(0.5 * r[1] + 0.5 * r[2], r[3]) for r in res])
    d = m_ll - o_ll
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"\n{n} 个系列赛")
    print(f"{'':10s}{'logloss':>9}{'准确率':>8}")
    print(f"{'我们':10s}{o_ll.mean():>9.4f}{o_ac.mean():>8.1%}")
    print(f"{'市场':10s}{m_ll.mean():>9.4f}{m_ac.mean():>8.1%}")
    print(f"{'五五混合':10s}{blend.mean():>9.4f}")
    print(f"\n我们相对市场 Δlogloss {d.mean():+.4f}  (SE {se:.4f}, t {d.mean() / se:+.2f})"
          f"   正值 = 我们更好")
    dis = [r for r in res if (r[1] > 0.5) != (r[2] > 0.5)]
    if dis:
        w = sum(1 for r in dis if (r[1] > 0.5) == r[3])
        print(f"\n两者选边相反的 {len(dis)} 场：我们对 {w}，市场对 {len(dis) - w}")
        for r in dis:
            print(f"   {NAME.get(r[0]['team_a']):14s} vs {NAME.get(r[0]['team_b']):14s}"
                  f" 我们 {r[1]:.2f} 市场 {r[2]:.2f} -> "
                  f"{'我们对' if (r[1] > 0.5) == r[3] else '市场对'}")


if __name__ == "__main__":
    main()
