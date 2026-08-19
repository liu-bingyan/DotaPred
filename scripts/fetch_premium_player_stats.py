"""拉 premium 级赛事的逐场选手数据，用来建「选手状态」变量。

data/raw/player_matches.json 只覆盖 TI2026 那 84 名选手，做不了历史验证。这里按
赛事级别拉：premium 一共 8,091 场（2022-01 起），约 81k 行，够为每名选手建一个
赛前基线 + 赛中偏离。

Writes data/raw/premium_player_stats.json
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "premium_player_stats.json")

COLS = """
 pm.account_id, pm.match_id, pm.hero_id, pm.player_slot, pm.kills, pm.deaths,
 pm.assists, pm.last_hits, pm.denies, pm.gold_per_min, pm.xp_per_min,
 pm.net_worth, pm.obs_placed, pm.sen_placed, pm.camps_stacked,
 pm.rune_pickups, pm.teamfight_participation, pm.towers_killed,
 pm.roshans_killed, pm.stuns, pm.lane_role, pm.hero_damage, pm.tower_damage,
 pm.hero_healing,
 m.start_time, m.duration, m.leagueid, m.radiant_win,
 m.radiant_team_id, m.dire_team_id, m.series_id
"""


def query(sql):
    url = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaPred/0.1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["rows"]
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(6 * (attempt + 1))
    raise RuntimeError("query failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2022-01-01")
    ap.add_argument("--tier", default="premium")
    args = ap.parse_args()
    start = dt.datetime.fromisoformat(args.since).replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)

    rows = []
    cur = start
    while cur < now:
        nxt = min(cur + dt.timedelta(days=120), now)
        got = query(f"""
        select {COLS}
        from player_matches pm
        join matches m using(match_id)
        join leagues l on l.leagueid = m.leagueid
        where m.start_time >= {int(cur.timestamp())}
          and m.start_time < {int(nxt.timestamp())}
          and l.tier = '{args.tier}'
          and m.radiant_team_id is not null and m.dire_team_id is not null
        """)
        rows.extend(got)
        print(f"{cur:%Y-%m-%d} .. {nxt:%Y-%m-%d}  {len(got):6d}  (累计 {len(rows)})",
              flush=True)
        cur = nxt
        time.sleep(1.5)

    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"\nwrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
