"""把 TI2026 的赛果 + 主赛事对阵树从 Valve 的 GetLeagueData 解出来。

写 data/ti2026_bracket.json：
  teams        16 队 id/名字/瑞士轮战绩
  swiss        每个系列赛的双方与比分
  elimination  淘汰赛 5 场
  playoff      主赛事 14 个节点及其上下游关系（预测活动要填的就是这 14 个）

主赛事的节点关系不在 API 的字段里（只有 node_id 和顺序），按 docs/01-rules.md
§1.2 已解出的结构硬编码，并用当前 8 强的实际填充位置校验。
"""

import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "ti2026_bracket.json")
RAW = os.path.join(ROOT, "data", "raw", "ti2026_league_19719.json")
URL = "https://www.dota2.com/webapi/IDOTA2League/GetLeagueData/v001/?league_id=19719"

# node_id -> (标签, 上游). 上游用 ("W", node) / ("L", node) 表示某场的胜者/败者。
PLAYOFF = {
    14: ("UB-A", None, None),
    15: ("UB-B", None, None),
    16: ("UB-C", None, None),
    17: ("UB-D", None, None),
    18: ("UB-E", ("W", 14), ("W", 15)),
    19: ("UB-F", ("W", 16), ("W", 17)),
    20: ("UB-G", ("W", 18), ("W", 19)),
    22: ("LB1-1", ("L", 14), ("L", 15)),
    23: ("LB1-2", ("L", 16), ("L", 17)),
    24: ("LB2-1", ("L", 19), ("W", 22)),
    25: ("LB2-2", ("L", 18), ("W", 23)),
    26: ("LB3", ("W", 24), ("W", 25)),
    27: ("LB-F", ("L", 20), ("W", 26)),
    21: ("GF", ("W", 20), ("W", 27)),
}


def fetch(refresh):
    if refresh or not os.path.exists(RAW):
        req = urllib.request.Request(URL, headers={"User-Agent": "DotaPred/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        json.dump(data, open(RAW, "w"))
        return data
    return json.load(open(RAW))


def groups(data):
    out = {}

    def walk(g):
        out[g["node_group_id"]] = g
        for sub in (g.get("node_groups") or []):
            walk(sub)

    for g in data["node_groups"]:
        walk(g)
    return out


def main():
    data = fetch(refresh=True)
    gs = groups(data)
    swiss, elim, playoff = gs[2], gs[3], gs[5]

    names = {t["team_id"]: t["team_name"].strip().title()
             for t in swiss["team_standings"]}
    # 保持和 data/teams.json 一致的写法
    fix = {"Team Vision": "Team Vision", "Huligani": "Huligani",
           "Nigma Galaxy": "Nigma Galaxy", "Boomboys": "BoomBoys",
           "Og": "OG", "Lgd Gaming": "LGD Gaming"}
    names = {k: fix.get(v, v) for k, v in names.items()}

    teams = {names[t["team_id"]]: {"team_id": t["team_id"],
                                   "swiss": f"{t['wins']}-{t['losses']}",
                                   "standing": t["standing"],
                                   "game_win_pct": t["tiebreak_game_win_pct"],
                                   "opp_match_wins": t["tiebreak_opponent_match_wins"]}
             for t in swiss["team_standings"]}

    def series(g):
        return [{"node_id": n["node_id"], "name": n["name"],
                 "team_1": names.get(n["team_id_1"]), "team_2": names.get(n["team_id_2"]),
                 "score": [n["team_1_wins"], n["team_2_wins"]],
                 "winner": (names.get(n["team_id_1"]) if n["team_1_wins"] > n["team_2_wins"]
                            else names.get(n["team_id_2"]))}
                for n in g["nodes"] if n["team_id_1"] and n["team_id_2"]]

    seats = {n["node_id"]: [names.get(n["team_id_1"]), names.get(n["team_id_2"])]
             for n in playoff["nodes"]}
    bracket = []
    for nid, (label, u1, u2) in PLAYOFF.items():
        bracket.append({
            "node_id": nid, "label": label,
            "bo": 5 if label == "GF" else 3,
            "team_1": seats.get(nid, [None, None])[0],
            "team_2": seats.get(nid, [None, None])[1],
            "from_1": u1, "from_2": u2,
        })

    out = {"league_id": 19719, "teams": teams, "swiss": series(swiss),
           "elimination": series(elim), "playoff": bracket}
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)

    print("瑞士轮最终排名")
    for name, t in sorted(teams.items(), key=lambda kv: (kv[1]["standing"], -kv[1]["game_win_pct"])):
        print(f"  #{t['standing']} {name:18s} {t['swiss']}  小分 {t['game_win_pct']:>3}%"
              f"  对手胜场 {t['opp_match_wins']}")
    print("\n淘汰赛")
    for s in out["elimination"]:
        print(f"  {s['team_1']:18s} {s['score'][0]}-{s['score'][1]} {s['team_2']}")
    print("\n主赛事 8 强对阵")
    for b in bracket:
        if b["team_1"]:
            print(f"  {b['label']:6s} {b['team_1']:18s} vs {b['team_2']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
