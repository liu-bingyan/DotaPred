"""Pull per-game stats for every TI2026 player.

These are exactly the columns Fantasy scores on (kills/deaths/CS/GPM/towers/
wards/stacks/runes/roshan/stuns/first blood), plus the `killed` and `item_uses`
json blobs which is where courier kills, Tormentors and smokes hide.

Writes data/raw/player_matches.json
"""

import argparse
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "player_matches.json")
START = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)

COLS = """
 pm.account_id, pm.match_id, pm.hero_id, pm.player_slot, pm.kills, pm.deaths,
 pm.assists, pm.last_hits, pm.denies, pm.gold_per_min, pm.xp_per_min, pm.level,
 pm.net_worth, pm.obs_placed, pm.sen_placed, pm.creeps_stacked, pm.camps_stacked,
 pm.rune_pickups, pm.firstblood_claimed, pm.teamfight_participation,
 pm.towers_killed, pm.roshans_killed, pm.stuns, pm.lane, pm.lane_role,
 pm.is_roaming, pm.hero_damage, pm.tower_damage, pm.hero_healing,
 pm.killed, pm.item_uses,
 m.start_time, m.duration, m.leagueid, m.radiant_win, m.radiant_team_id,
 m.dire_team_id, m.series_id, m.series_type, l.tier
"""


def query(sql):
    url = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaPred/0.1"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["rows"]
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("query failed")


def month_bounds(start, end):
    cur = start
    while cur < end:
        nxt = (cur.replace(day=28) + dt.timedelta(days=6)).replace(day=1)
        yield int(cur.timestamp()), int(min(nxt, end).timestamp()), cur
        cur = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", default=None,
                    help="只拉这几个 account_id（逗号分隔），默认拉 accounts.txt 全部")
    ap.add_argument("--merge", action="store_true",
                    help="把结果并回已有文件（先删掉这些 account_id 的旧行）")
    args = ap.parse_args()

    accounts = (args.accounts or
                open(os.path.join(ROOT, "data", "accounts.txt")).read().strip())
    ids = {int(x) for x in accounts.replace("\n", ",").split(",") if x.strip()}
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for lo, hi, label in month_bounds(START, now):
        sql = f"""
        select {COLS}
        from player_matches pm
        join matches m using(match_id)
        left join leagues l on l.leagueid = m.leagueid
        where pm.account_id in ({accounts})
          and m.start_time >= {lo} and m.start_time < {hi}
          and m.radiant_team_id is not null and m.dire_team_id is not null
        """
        got = query(sql)
        rows.extend(got)
        print(f"{label:%Y-%m}  {len(got):5d} rows   (total {len(rows)})", flush=True)
        time.sleep(1.5)

    if args.merge and os.path.exists(OUT):
        old = json.load(open(OUT))
        kept = [r for r in old if r["account_id"] not in ids]
        print(f"merge: 旧文件 {len(old)} 行，去掉这些选手的 {len(old) - len(kept)} 行，"
              f"加入 {len(rows)} 行")
        rows = kept + rows

    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"\nwrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
