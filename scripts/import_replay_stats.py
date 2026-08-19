"""Valve 录像解析出来的逐局 fantasy 统计，来自 ti-fantasy.site 的公开数据集。

为什么要它：**莲花采集和占领观察者这两项，OpenDota 和 STRATZ 都没有**
（2026-08-13 查过 STRATZ 的整份 schema，只有一个 outposts 字段，不是这两项）。
唯一的来源是解析 Valve 录像。TI 期辅助战旗第⑤枚就是莲花 250%，不补上这一项，
那面旗根本算不了分。

顺带补的第二件事更重要：**团战参与在这份数据里是 Valve 自己的字段**
（`CDOTA_PlayerResource m_flT...`），而我们一直用的是 OpenDota 自己把击杀聚类成
团战再算的估计值。`fetch_stratz_fights.py` 的注释把它列为「整个战旗模型里风险最高的
未校准输入 —— 48 个槽位的绿色徽标全靠它」。现在可以直接对账。

来源：https://github.com/TinyKiecoo/Calculator-for-DOTA2-TI-Fantasy
       data/{leagueid}/full.json，字段出处逐项写在 meta.fieldProvenance 里。
这是第三方解析结果，所以 `validate_replay_stats.py` 会把它和我们自己的 OpenDota
数字在重叠比赛上逐项对账；没对上的项不要直接用。

    python scripts/import_replay_stats.py                # 默认拉 TI + 石油杯
    python scripts/import_replay_stats.py --leagues 19719
"""

import argparse
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "replay_stats.json")
BASE = ("https://raw.githubusercontent.com/TinyKiecoo/"
        "Calculator-for-DOTA2-TI-Fantasy/HEAD/data/%d/full.json")

# 他们的字段名 -> 我们的统计项键（fantasy_stats.COLOR 里的那 18 个）
FIELD = {
    "kills": "kills", "deaths": "deaths", "creep_score": "cs", "gpm": "gpm",
    "madstones_collected": "madstone", "towers_destroyed": "towers",
    "observer_wards_placed": "wards", "camps_stacked": "stacks",
    "runes_picked_up": "runes", "watchers_captured": "watchers",
    "smokes_used": "smokes", "lotuses_collected": "lotus",
    "roshans_killed": "roshan", "teamfight_participation": "teamfight",
    "stun_seconds": "stuns", "tormentors_killed": "tormentor",
    "first_blood": "firstblood", "couriers_killed": "courier",
}

LEAGUES = {19719: "The International 2026", 19785: "Esports World Cup 2026",
           20009: "1win Essence II"}


def fetch(leagueid):
    req = urllib.request.Request(BASE % leagueid,
                                 headers={"User-Agent": "DotaPred/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def rows_from(doc):
    """-> 每个 (比赛, 选手) 一行，18 项原始量 + 归属信息。"""
    out = []
    for m in doc["matches"]:
        for p in m["players"]:
            st = p["stats"]
            radiant = p["playerSlot"] < 128
            row = {"account_id": p["accountId"],
                   "match_id": m["matchId"],
                   "series_id": m.get("seriesId") or m["matchId"],
                   "series_type": m.get("seriesType"),
                   "start_time": m["startTime"],
                   "duration": m.get("duration"),
                   "leagueid": doc["meta"]["leagueId"],
                   "team_id": p.get("teamId"),
                   "opponent_team_id": (p.get("opponent") or {}).get("teamId"),
                   "radiant": radiant,
                   "win": bool(p.get("won")),
                   "role_theirs": p.get("role"),
                   "hero_id": p.get("heroId")}
            for src, dst in FIELD.items():
                row[dst] = st.get(src)
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default="19719,19785",
                    help="逗号分隔的 leagueid，默认 TI + 石油杯")
    args = ap.parse_args()
    ids = [int(x) for x in args.leagues.split(",") if x.strip()]

    allrows, meta = [], {}
    for lid in ids:
        doc = fetch(lid)
        rs = rows_from(doc)
        allrows += rs
        cov = doc["meta"]["coverage"]
        meta[str(lid)] = {"name": doc["meta"].get("leagueName", LEAGUES.get(lid)),
                          "generatedAt": doc["meta"].get("generatedAt"),
                          "matches": cov.get("matches"),
                          "parsedMatches": cov.get("parsedMatches"),
                          "playerGameRows": cov.get("playerGameRows"),
                          "completeSeries": cov.get("completeSeries"),
                          "incompleteSeries": cov.get("incompleteSeries"),
                          "eventStage": doc["meta"].get("eventStage"),
                          "provenance": doc["meta"].get("fieldProvenance")}
        print(f"{lid} {meta[str(lid)]['name']:<32}"
              f"{cov.get('parsedMatches')}/{cov.get('matches')} 场已解析  "
              f"{len(rs)} 行  生成于 {meta[str(lid)]['generatedAt']}")

    json.dump({"source": BASE % 0, "meta": meta, "rows": allrows},
              open(OUT, "w"))
    print(f"\nwrote {len(allrows)} rows -> {OUT}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- 供下游使用
def load(leagueids=None):
    """-> (rows, by_player)，by_player[account_id] = [每局一条]。"""
    d = json.load(open(OUT))
    rows = d["rows"]
    if leagueids:
        keep = set(leagueids)
        rows = [r for r in rows if r["leagueid"] in keep]
    by = {}
    for r in rows:
        by.setdefault(r["account_id"], []).append(r)
    return d["meta"], by


def slot_rows(by_player, roles, teams):
    """和 fantasy_model.role_game_scores 同形状的输出，但底层是 Valve 录像的量。

    双人槽位在同一局内先对两名选手取平均，与计分规则一致；只有一人上场时退化。
    """
    import collections
    import fantasy_stats as FS
    stats = FS.COLOR["red"] + FS.COLOR["blue"] + FS.COLOR["green"]
    out = {}
    for team, r in roles.items():
        if not r:
            continue
        for role in ("core", "mid", "support"):
            per_match = collections.defaultdict(dict)
            for a in r[role]:
                for g in by_player.get(a, []):
                    per_match[g["match_id"]][a] = g
            rws = []
            for mid, d in per_match.items():
                pts = [FS.to_points({s: g.get(s) for s in stats})
                       for g in d.values()]
                avg = {s: float(sum(p[s] for p in pts) / len(pts)) for s in stats}
                any_g = next(iter(d.values()))
                rws.append({"match_id": mid, "series_id": any_g["series_id"],
                            "start": any_g["start_time"], "pts": avg,
                            "win": bool(any_g["win"]), "n_players": len(d),
                            "leagueid": any_g["leagueid"]})
            if rws:
                out[(team, role)] = rws
    return out
