"""Pull all pro (team-vs-team) matches since 2025-01-01 via the OpenDota
explorer SQL endpoint, month by month so no single query times out.

Writes data/raw/pro_matches.json  (list of dicts)
"""

import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "pro_matches.json")
START = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)


def query(sql):
    url = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaPred/0.1"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
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
        select m.match_id, m.start_time, m.leagueid, m.radiant_team_id,
               m.dire_team_id, m.radiant_win, m.duration, l.tier, l.name as league_name
        from matches m left join leagues l on l.leagueid = m.leagueid
        where m.start_time >= {lo} and m.start_time < {hi}
          and m.radiant_team_id is not null and m.dire_team_id is not null
          and m.radiant_win is not null
        """
        got = query(sql)
        rows.extend(got)
        print(f"{label:%Y-%m}  {len(got):5d} matches   (total {len(rows)})")
        time.sleep(1.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"\nwrote {len(rows)} matches -> {OUT}")


if __name__ == "__main__":
    main()
