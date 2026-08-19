"""Re-pull pro matches with margin-of-victory detail, not just win/loss.

A binary result compresses a game to 1 bit and throws away the difference
between a 20-minute 0-tower stomp and a 60-minute base race. These columns let
us score *how* a team won:

  final/max/min gold+xp advantage   -> dominance, and whether a lead was blown
  tower_status / barracks_status    -> how much of the base was still standing
  radiant_score / dire_score        -> kill differential
  duration, first_blood_time        -> pace
  series_id / series_type           -> which games belong to the same Bo3

Writes data/raw/pro_matches_rich.json
"""

import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "pro_matches_rich.json")
START = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)

COLS = """
  m.match_id, m.start_time, m.leagueid, m.radiant_team_id, m.dire_team_id,
  m.radiant_win, m.duration, m.radiant_score, m.dire_score, m.first_blood_time,
  m.tower_status_radiant, m.tower_status_dire,
  m.barracks_status_radiant, m.barracks_status_dire,
  m.radiant_gold_adv[array_length(m.radiant_gold_adv,1)] as final_gold_adv,
  m.radiant_xp_adv[array_length(m.radiant_xp_adv,1)]     as final_xp_adv,
  (select max(v) from unnest(m.radiant_gold_adv) v) as max_gold_adv,
  (select min(v) from unnest(m.radiant_gold_adv) v) as min_gold_adv,
  m.series_id, m.series_type, l.tier, l.name as league_name
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
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for lo, hi, label in month_bounds(START, now):
        sql = f"""
        select {COLS}
        from matches m left join leagues l on l.leagueid = m.leagueid
        where m.start_time >= {lo} and m.start_time < {hi}
          and m.radiant_team_id is not null and m.dire_team_id is not null
          and m.radiant_win is not null
        """
        got = query(sql)
        rows.extend(got)
        parsed = sum(1 for r in got if r.get("final_gold_adv") is not None)
        print(f"{label:%Y-%m}  {len(got):5d} matches  {parsed:5d} with gold curves "
              f"({parsed / max(len(got), 1):.0%})", flush=True)
        time.sleep(1.5)

    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"\nwrote {len(rows)} matches -> {OUT}")


if __name__ == "__main__":
    main()
